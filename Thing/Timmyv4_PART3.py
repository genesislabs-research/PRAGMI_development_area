# RM numbers reference "The code classes explained" in README.md

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from Timmyv4_PART1 import (
    SSMLIFSubstrate,
    SubstrateState,
    SpikeMixer,
    LemmaStratumConfig,
    PhonologicalConfig,
    BrocasConfig,
    ArcuateConfig,
    ProductionConfig,
    OrthographyConfig,
    CascadeConfig,
    PHONEME_INVENTORY,
    PHONEME_START_IDX,
    PHONEME_END_IDX,
    PHONEME_SYL_BOUNDARY_IDX,
    PHONEME_MORPH_BOUNDARY_IDX,
    PHONEME_PAD_IDX,
)


# RM301
class LemmaStatus(IntEnum):
    UNALLOCATED = 0
    SCAFFOLD = 1
    PROVISIONAL = 2
    CONFIRMED = 3


# RM302
@dataclass
class LemmaStratumState:
    activations: Tensor
    velocities: Tensor
    last_spikes: Tensor
    activation_window: List[Tensor]
    tick_count: int = 0
    last_top_idx: int = -1


# RM303
@dataclass
class LemmaStratumOutput:
    a_l: Tensor
    spikes: Tensor
    c_cas: Tensor
    selected_lemma: int
    doubt: Tensor
    attention_flipped: bool
    flip_cost: float
    acquired_row: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


_SCAFFOLD_COORDS: List[Tuple[int, List[float]]] = [
    (0, [+0.50, 0.00, 0.00, 0.00, 0.00]),
    (1, [-0.50, 0.00, 0.00, 0.00, 0.00]),
    (2, [-0.50, +0.20, 0.00, 0.00, 0.00]),
    (3, [-0.50, -0.20, 0.00, 0.00, 0.00]),
    (4, [0.00, 0.00, +0.50, 0.00, 0.00]),
    (5, [0.00, 0.00, -0.50, 0.00, 0.00]),
    (6, [0.00, 0.00, 0.00, -0.50, 0.00]),
    (7, [0.00, 0.00, 0.00, +0.50, 0.00]),
    (8, [0.00, 0.00, 0.00, 0.00, +0.35]),
    (9, [0.00, 0.00, 0.00, 0.00, -0.35]),
]

_SCAFFOLD_ROLE_AXES: List[Tuple[int, List[int]]] = [
    (0, [1]),
    (1, [1]),
    (2, [1, 2]),
    (3, [1, 2]),
    (4, [3]),
    (5, [3]),
    (6, [4]),
    (7, [4]),
    (8, [5]),
    (9, [5]),
]


# RM304
class LemmaStratum(nn.Module):
    def __init__(self, cfg: Optional[LemmaStratumConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or LemmaStratumConfig()
        gen = torch.Generator().manual_seed(self.cfg.rng_seed)

        N = self.cfg.population_size
        M = self.cfg.manifold_dim
        R = self.cfg.role_dim
        C = self.cfg.concept_dim

        self.substrate = SSMLIFSubstrate(
            num_neurons=N,
            n_dim=32,
            dt=self.cfg.dt_ms,
            v_threshold_base=1.0,
            theta_decay=0.95,
            theta_increment=0.5,
            surrogate_alpha=2.0,  # NOT biological. Training artifact.
        )

        coords = torch.zeros(N, M, dtype=torch.float32)
        role_bindings = torch.zeros(N, R, dtype=torch.float32)
        status = torch.zeros(N, dtype=torch.long)
        for row_idx, vec in _SCAFFOLD_COORDS:
            coords[row_idx, :5] = torch.tensor(vec, dtype=torch.float32)
            status[row_idx] = int(LemmaStatus.SCAFFOLD)
        for row_idx, channels in _SCAFFOLD_ROLE_AXES:
            role_bindings[row_idx, 0] = 1.0
            for ch in channels:
                role_bindings[row_idx, ch] = 1.0

        self.coordinates = nn.Parameter(coords, requires_grad=False)
        self.register_buffer("scaffold_coords_snapshot", coords.clone())
        self.register_buffer("role_bindings", role_bindings)
        self.register_buffer("status", status)
        self.register_buffer(
            "scaffold_mask",
            (status == int(LemmaStatus.SCAFFOLD)).to(torch.float32),
        )

        w_ll = torch.randn(N, N, generator=gen) * (1.0 / math.sqrt(N))
        w_ll = w_ll * 0.1
        self.W_LL = nn.Parameter(w_ll, requires_grad=False)

        w_ctol = torch.randn(C, N, generator=gen) * (1.0 / math.sqrt(C))
        w_ctol = w_ctol * 0.1
        self.W_CtoL = nn.Parameter(w_ctol, requires_grad=False)

        w_re = torch.randn(64, N, generator=gen) * (1.0 / math.sqrt(64))
        w_re = w_re * 0.1
        self.W_reentry_to_L = nn.Parameter(w_re, requires_grad=False)

        # NOT biological. Engineering convenience for new-row coord init.
        x_proj = torch.randn(C, M, generator=gen) * (1.0 / math.sqrt(C))
        self.register_buffer("W_x_to_manifold", x_proj * 0.1)

        self.register_buffer(
            "velocities", torch.zeros(N, M, dtype=torch.float32),
        )

        self._reentry_buffer: Deque[Tensor] = deque(
            maxlen=self.cfg.reentry_delay_ticks,
        )

        self.register_buffer(
            "activations", torch.zeros(1, N, dtype=torch.float32),
        )
        self._activation_decay: float = 0.9

        self._activation_window: Deque[Tensor] = deque(
            maxlen=self.cfg.doubt_window_ticks,
        )

        self.register_buffer(
            "ctol_pre_trace", torch.zeros(C, dtype=torch.float32),
        )
        self.register_buffer(
            "ctol_post_trace", torch.zeros(N, dtype=torch.float32),
        )
        self.register_buffer(
            "ll_pre_trace", torch.zeros(N, dtype=torch.float32),
        )
        self.register_buffer(
            "ll_post_trace", torch.zeros(N, dtype=torch.float32),
        )

        self.register_buffer(
            "tick_count", torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "last_top_idx", torch.tensor(-1, dtype=torch.long),
        )

        self._substrate_state: Optional[SubstrateState] = None

        self.register_buffer(
            "next_unalloc",
            torch.tensor(self.cfg.scaffold_size, dtype=torch.long),
        )

    def _ensure_state(self, batch: int, device: torch.device) -> None:
        N = self.cfg.population_size
        needs_init = (
            self._substrate_state is None
            or self._substrate_state.h.shape[0] != batch
            or self._substrate_state.h.device != device
        )
        if needs_init:
            self._substrate_state = self.substrate.init_state(
                batch_size=batch, device=device,
            )
        if (
            self.activations.shape[0] != batch
            or self.activations.device != device
        ):
            self.activations = torch.zeros(
                batch, N, dtype=torch.float32, device=device,
            )
            self._reentry_buffer.clear()
            self._activation_window.clear()

    def _lc_high_ne_phase(self) -> bool:
        if self.cfg.ablate_lc_flipflop:
            return False
        t = int(self.tick_count.item())
        period = max(self.cfg.lc_period_ticks, 1)
        phase = (2.0 * math.pi * (t % period)) / period
        return math.sin(phase) > 0.0

    def _three_factor_admits(
        self, omega: Optional[Tensor], local_depol: float,
    ) -> bool:
        if omega is None:
            return False
        if self.cfg.ablate_three_factor:
            dopamine = float(omega[..., 0].mean().item())
            return dopamine > self.cfg.dopamine_burst_threshold
        dopamine = float(omega[..., 0].mean().item())
        ach = float(omega[..., 2].mean().item())
        if dopamine <= self.cfg.dopamine_burst_threshold:
            return False
        if ach >= self.cfg.cholinergic_pause_threshold:
            return False
        if local_depol <= self.cfg.local_depol_threshold:
            return False
        return True

    def _scaffold_reset(self) -> None:
        with torch.no_grad():
            mask = self.scaffold_mask.bool()
            self.coordinates.data[mask] = (
                self.scaffold_coords_snapshot[mask]
            )
            self.velocities[mask] = 0.0

    def _compute_doubt(
        self, a_l: Tensor, omega: Optional[Tensor],
    ) -> Tensor:
        device = a_l.device
        confirmed_mask = (
            self.status == int(LemmaStatus.CONFIRMED)
        ).to(device).to(a_l.dtype)
        if confirmed_mask.sum() < 1.0:
            peak_low = torch.ones(a_l.shape[0], device=device)
        else:
            confirmed_a = a_l * confirmed_mask.unsqueeze(0)
            peak = confirmed_a.max(dim=-1).values
            peak_low = (1.0 - peak).clamp(0.0, 1.0)
        if len(self._activation_window) >= 2:
            stacked = torch.stack(list(self._activation_window), dim=0)
            if stacked.shape[0] >= 2:
                a = stacked[:-1]
                b = stacked[1:]
                a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
                b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
                cos = (a_n * b_n).sum(dim=-1)
                tangling = (1.0 - cos.mean(dim=0)).clamp(0.0, 1.0)
            else:
                tangling = torch.zeros(a_l.shape[0], device=device)
        else:
            tangling = torch.zeros(a_l.shape[0], device=device)
        if tangling.dim() == 0:
            tangling = tangling.unsqueeze(0).expand(a_l.shape[0])
        if tangling.shape[0] != a_l.shape[0]:
            tangling = tangling.expand(a_l.shape[0])
        doubt = 0.5 * peak_low + 0.5 * tangling
        if omega is not None:
            doubt = doubt + 0.2 * omega[..., 4].to(device)
        return doubt.clamp(0.0, 1.0)

    def _role_gated_hazard(self, a_l: Tensor, g_t: Tensor) -> Tensor:
        if self.cfg.ablate_role_gate:
            return a_l
        alignment = self.role_bindings @ g_t.t()
        alignment = alignment.t()
        gated = a_l * alignment.clamp(min=0.0)
        denom = gated.sum(dim=-1, keepdim=True) + 1e-8
        hazard = gated / denom
        return hazard

    def _rstdp_update_ctol(
        self, pre: Tensor, post: Tensor, modulation: float,
    ) -> None:
        device = self.W_CtoL.device
        pre_d = pre.detach().mean(dim=0).to(device).float()
        post_d = post.detach().mean(dim=0).to(device).float()
        old_pre = self.ctol_pre_trace.clone()
        old_post = self.ctol_post_trace.clone()
        decay = math.exp(-1.0 / self.cfg.stdp_tau_pre_ms)
        self.ctol_pre_trace.mul_(decay).add_(pre_d)
        self.ctol_post_trace.mul_(decay).add_(post_d)
        ltp = torch.outer(old_pre, post_d)
        ltd = torch.outer(pre_d, old_post)
        delta = self.cfg.stdp_lr * modulation * (
            ltp - self.cfg.depression_ratio * ltd
        )
        with torch.no_grad():
            self.W_CtoL.data.add_(delta)
            self.W_CtoL.data.clamp_(
                self.cfg.weight_min, self.cfg.weight_max,
            )

    def _rstdp_update_ll(
        self, pre: Tensor, post: Tensor, modulation: float,
    ) -> None:
        device = self.W_LL.device
        pre_d = pre.detach().mean(dim=0).to(device).float()
        post_d = post.detach().mean(dim=0).to(device).float()
        old_pre = self.ll_pre_trace.clone()
        old_post = self.ll_post_trace.clone()
        decay = math.exp(-1.0 / self.cfg.stdp_tau_pre_ms)
        self.ll_pre_trace.mul_(decay).add_(pre_d)
        self.ll_post_trace.mul_(decay).add_(post_d)
        ltp = torch.outer(post_d, old_pre)
        ltd = torch.outer(old_post, pre_d)
        delta = self.cfg.stdp_lr * modulation * (
            ltp - self.cfg.depression_ratio * ltd
        )
        with torch.no_grad():
            self.W_LL.data.add_(delta)
            self.W_LL.data.clamp_(
                self.cfg.weight_min, self.cfg.weight_max,
            )

    def _spring_step(self, dt_factor: float = 1.0) -> None:
        N = self.cfg.population_size
        M = self.cfg.manifold_dim
        device = self.coordinates.device
        with torch.no_grad():
            w = self.W_LL.abs().clamp(min=1e-8)
            k = self.cfg.alpha_stiffness * (w ** self.cfg.beta_exponent)
            neighbor_mean = k @ self.coordinates.data / (
                k.sum(dim=-1, keepdim=True) + 1e-8
            )
            force = neighbor_mean - self.coordinates.data
            confirmed_scalar = (
                self.status == int(LemmaStatus.CONFIRMED)
            ).to(device).float()
            mass = (
                self.cfg.mass_baseline
                * (1.0 + self.cfg.mass_confirmation_scale * confirmed_scalar)
            ).unsqueeze(-1)
            damping = self.cfg.gamma_damping * self.velocities
            accel = (force - damping) / (mass + 1e-8)
            self.velocities.add_(accel * dt_factor)
            self.coordinates.data.add_(self.velocities * dt_factor)
            if self.cfg.disorder_axis2_collapse > 0.0:
                for row in (2, 3):
                    self.coordinates.data[row, 0] += (
                        self.cfg.disorder_axis2_collapse * 0.01
                    )
            if not self.cfg.ablate_scaffold_lock:
                self._scaffold_reset()

    def tick(
        self,
        c_lex: Tensor,
        g_t: Tensor,
        x_t: Tensor,
        tau_proj: int = 0,
        r_reentry: Optional[Tensor] = None,
        omega: Optional[Tensor] = None,
        teach_label: Optional[int] = None,
    ) -> LemmaStratumOutput:
        if c_lex.dim() != 2 or c_lex.shape[-1] != self.cfg.concept_dim:
            raise ValueError(
                f"c_lex must be (B, {self.cfg.concept_dim}); got "
                f"{tuple(c_lex.shape)}"
            )
        B = c_lex.shape[0]
        device = c_lex.device
        N = self.cfg.population_size

        self._ensure_state(B, device)

        if (
            not self.cfg.ablate_reentry
            and len(self._reentry_buffer) >= self.cfg.reentry_delay_ticks
        ):
            r_delayed = self._reentry_buffer[0].to(device)
        else:
            r_delayed = torch.zeros(B, 64, device=device)

        concept_in = c_lex @ self.W_CtoL
        reentry_in = r_delayed @ self.W_reentry_to_L
        recurrent_in = self.activations @ self.W_LL.t()

        lc_factor: Any = 1.0
        if self._lc_high_ne_phase():
            top_idx = int(self.activations.mean(dim=0).argmax().item())
            top_coord = self.coordinates.data[top_idx]
            dists = torch.norm(
                self.coordinates.data - top_coord.unsqueeze(0), dim=-1,
            )
            normed = dists / (dists.max() + 1e-8)
            lc_factor = (1.0 - 0.5 * normed).unsqueeze(0)

        substrate_input = concept_in + reentry_in + 0.5 * recurrent_in
        if isinstance(lc_factor, Tensor):
            substrate_input = substrate_input * lc_factor

        spikes, self._substrate_state = self.substrate(
            substrate_input, self._substrate_state,
        )

        with torch.no_grad():
            self.activations = (
                self._activation_decay * self.activations
                + spikes.detach()
            )
            self.activations.clamp_(0.0, 10.0)

        self._activation_window.append(
            self.activations.detach().mean(dim=0).cpu().clone()
        )

        a_l = self.activations

        hazard = self._role_gated_hazard(a_l, g_t)
        top_idx = int(hazard.mean(dim=0).argmax().item())
        prev_top = int(self.last_top_idx.item())
        attention_flipped = False
        flip_cost = 0.0
        if (
            not self.cfg.ablate_attention_flip
            and prev_top >= 0
            and top_idx != prev_top
        ):
            attention_flipped = True
            old = float(hazard.mean(dim=0)[prev_top].item()) + 1e-8
            new = float(hazard.mean(dim=0)[top_idx].item()) + 1e-8
            delta_I = math.log(old) - math.log(new)
            flip_cost = self.cfg.flip_weight * delta_I
        self.last_top_idx.fill_(top_idx)

        doubt = self._compute_doubt(a_l, omega)

        acquired_row = -1
        if teach_label is not None and 0 <= teach_label < N:
            local_depol = float(
                spikes.detach().mean(dim=0)[teach_label].item()
            )
        else:
            local_depol = float(spikes.detach().mean().item())

        admit_acq = (
            teach_label is not None
            and (
                tau_proj == 0
                or self.cfg.ablate_tau_proj_guard
            )
            and (
                self.cfg.ablate_three_factor
                or self._three_factor_admits(omega, local_depol)
            )
            and float(doubt.mean().item())
            >= self.cfg.acquisition_doubt_threshold
        )
        if admit_acq:
            acquired_row = self._allocate(
                g_t=g_t.detach().mean(dim=0),
                x_t=x_t.detach().mean(dim=0),
                lemma_hint=teach_label,
            )

        if (
            self.cfg.ablate_three_factor
            or self._three_factor_admits(omega, local_depol)
        ) and tau_proj == 0:
            mod = 1.0
            if omega is not None:
                mod = 1.0 + float(omega[..., 0].mean().item())
            self._rstdp_update_ll(spikes, spikes, mod)
            self._rstdp_update_ctol(c_lex, spikes, mod)

        period_ticks = int(
            self.cfg.rhythm3_period_ms / self.cfg.dt_ms
        )
        if (
            period_ticks > 0
            and int(self.tick_count.item()) % period_ticks == 0
            and int(self.tick_count.item()) > 0
        ):
            self._spring_step(dt_factor=0.01)

        if r_reentry is not None and not self.cfg.ablate_reentry:
            re = r_reentry.detach()
            if re.dim() == 1:
                re = re.unsqueeze(0).expand(B, -1)
            if re.shape[-1] != 64:
                pad = torch.zeros(
                    B, 64, device=re.device, dtype=re.dtype,
                )
                if re.shape[-1] < 64:
                    pad[:, :re.shape[-1]] = re
                    re = pad
                else:
                    re = re[:, :64]
            self._reentry_buffer.append(re.clone())

        c_cas = a_l @ self.W_CtoL.t()

        self.tick_count.add_(1)

        return LemmaStratumOutput(
            a_l=a_l,
            spikes=spikes,
            c_cas=c_cas,
            selected_lemma=top_idx,
            doubt=doubt,
            attention_flipped=attention_flipped,
            flip_cost=flip_cost,
            acquired_row=acquired_row,
            diagnostics={
                "spike_rate": float(spikes.detach().mean().item()),
                "n_confirmed": int(
                    (self.status == int(LemmaStatus.CONFIRMED)).sum().item()
                ),
                "n_provisional": int(
                    (self.status == int(LemmaStatus.PROVISIONAL))
                    .sum().item()
                ),
                "n_unalloc": int(
                    (self.status == int(LemmaStatus.UNALLOCATED))
                    .sum().item()
                ),
            },
        )

    def _allocate(
        self,
        g_t: Tensor,
        x_t: Tensor,
        lemma_hint: Optional[int] = None,
    ) -> int:
        N = self.cfg.population_size
        device = self.coordinates.device
        target = -1
        if lemma_hint is not None:
            cur_status = int(self.status[lemma_hint].item())
            if cur_status == int(LemmaStatus.UNALLOCATED):
                target = int(lemma_hint)
            elif cur_status in (
                int(LemmaStatus.PROVISIONAL),
                int(LemmaStatus.CONFIRMED),
            ):
                return int(lemma_hint)
        if target < 0:
            unalloc_mask = (
                self.status == int(LemmaStatus.UNALLOCATED)
            )
            if unalloc_mask.sum().item() == 0:
                return -1
            indices = torch.nonzero(unalloc_mask, as_tuple=False).flatten()
            target = int(indices[0].item())
        with torch.no_grad():
            self.status[target] = int(LemmaStatus.PROVISIONAL)
            new_coord = x_t.to(device) @ self.W_x_to_manifold
            self.coordinates.data[target] = new_coord
            self.role_bindings[target] = g_t.to(device)
            self.velocities[target] = 0.0
            self.next_unalloc.fill_(target + 1)
        return target

    @property
    def maturity(self) -> float:
        n_confirmed = int(
            (self.status == int(LemmaStatus.CONFIRMED)).sum().item()
        )
        return n_confirmed / max(self.cfg.maturity_saturation_count, 1)

    def sleep_consolidate(self) -> Dict[str, Any]:
        if self.cfg.ablate_sleep:
            return {"promoted": 0, "reclaimed": 0, "skipped": True}
        promoted = 0
        reclaimed = 0
        device = self.coordinates.device
        N = self.cfg.population_size
        gen = torch.Generator(device="cpu").manual_seed(
            int(self.tick_count.item()) + self.cfg.rng_seed,
        )
        with torch.no_grad():
            for step in range(self.cfg.sleep_consolidation_steps):
                noise = torch.poisson(
                    torch.full((1, N), 0.05), generator=gen,
                ).to(device)
                if self._substrate_state is None:
                    self._substrate_state = self.substrate.init_state(
                        batch_size=1, device=device,
                    )
                spikes, self._substrate_state = self.substrate(
                    noise, self._substrate_state,
                )
                self.activations = (
                    self._activation_decay * self.activations + spikes
                )
            for row in range(N):
                if int(self.status[row].item()) != int(
                    LemmaStatus.PROVISIONAL
                ):
                    continue
                weight_sum = float(self.W_LL[row].abs().sum().item())
                if weight_sum > self.cfg.confirmation_threshold:
                    self.status[row] = int(LemmaStatus.CONFIRMED)
                    promoted += 1
                elif weight_sum < self.cfg.reclaim_threshold:
                    self.status[row] = int(LemmaStatus.UNALLOCATED)
                    self.coordinates.data[row] = 0.0
                    self.role_bindings[row] = 0.0
                    reclaimed += 1
        return {"promoted": promoted, "reclaimed": reclaimed, "skipped": False}

    def get_state(self) -> Dict[str, Any]:
        return {
            "coordinates": self.coordinates.detach().cpu().clone(),
            "role_bindings": self.role_bindings.detach().cpu().clone(),
            "status": self.status.detach().cpu().clone(),
            "W_LL": self.W_LL.detach().cpu().clone(),
            "W_CtoL": self.W_CtoL.detach().cpu().clone(),
            "W_reentry_to_L": (
                self.W_reentry_to_L.detach().cpu().clone()
            ),
            "velocities": self.velocities.detach().cpu().clone(),
            "activations": self.activations.detach().cpu().clone(),
            "tick_count": int(self.tick_count.item()),
            "last_top_idx": int(self.last_top_idx.item()),
            "next_unalloc": int(self.next_unalloc.item()),
            "reentry_buffer": [
                t.detach().cpu().clone() for t in self._reentry_buffer
            ],
            "activation_window": [
                t.clone() for t in self._activation_window
            ],
            "ctol_pre_trace": self.ctol_pre_trace.detach().cpu().clone(),
            "ctol_post_trace": (
                self.ctol_post_trace.detach().cpu().clone()
            ),
            "ll_pre_trace": self.ll_pre_trace.detach().cpu().clone(),
            "ll_post_trace": self.ll_post_trace.detach().cpu().clone(),
            "substrate_state": (
                self.substrate.state_dict_full(self._substrate_state)
                if self._substrate_state is not None else None
            ),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        device = self.coordinates.device
        with torch.no_grad():
            self.coordinates.data.copy_(
                state["coordinates"].to(device),
            )
            self.role_bindings.copy_(
                state["role_bindings"].to(device),
            )
            self.status.copy_(state["status"].to(device))
            self.W_LL.data.copy_(state["W_LL"].to(device))
            self.W_CtoL.data.copy_(state["W_CtoL"].to(device))
            self.W_reentry_to_L.data.copy_(
                state["W_reentry_to_L"].to(device),
            )
            self.velocities.copy_(state["velocities"].to(device))
        self.activations = state["activations"].to(device)
        self.tick_count.fill_(int(state["tick_count"]))
        self.last_top_idx.fill_(int(state["last_top_idx"]))
        self.next_unalloc.fill_(int(state["next_unalloc"]))
        self._reentry_buffer.clear()
        for t in state.get("reentry_buffer", []):
            self._reentry_buffer.append(t.to(device))
        self._activation_window.clear()
        for t in state.get("activation_window", []):
            self._activation_window.append(t.clone())
        self.ctol_pre_trace.copy_(state["ctol_pre_trace"].to(device))
        self.ctol_post_trace.copy_(state["ctol_post_trace"].to(device))
        self.ll_pre_trace.copy_(state["ll_pre_trace"].to(device))
        self.ll_post_trace.copy_(state["ll_post_trace"].to(device))
        sub = state.get("substrate_state")
        if sub is not None:
            self._substrate_state = self.substrate.load_state_full(
                sub, device=device,
            )


# RM305
class PhonologicalCodeStore(nn.Module):
    def __init__(
        self,
        cfg: Optional[PhonologicalConfig] = None,
        population_size: int = 4106,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PhonologicalConfig()
        self.population_size = population_size
        N = population_size
        D = self.cfg.code_dim
        V = self.cfg.phoneme_inventory_size

        self.population = SSMLIFSubstrate(
            num_neurons=N,
            n_dim=32,
            dt=1.0,
            v_threshold_base=1.0,
            theta_decay=0.95,
            theta_increment=0.5,
            surrogate_alpha=2.0,  # NOT biological. Training artifact.
        )

        self.decoder = SSMLIFSubstrate(
            num_neurons=self.cfg.decoder_population_size,
            n_dim=self.cfg.decoder_n_hidden,
            dt=1.0,
            v_threshold_base=1.0,
            theta_decay=0.95,
            theta_increment=0.5,
            surrogate_alpha=self.cfg.decoder_surrogate_alpha,  # NOT biological. Training artifact.
        )

        # W_LtoP tied; ZERO-init so a row is non-zero only after add_lemma.
        self.W_LtoP = nn.Parameter(
            torch.zeros(N, D), requires_grad=False,
        )

        gen = torch.Generator().manual_seed(13579)
        w_dec = torch.randn(
            self.cfg.decoder_population_size, V, generator=gen,
        ) * (1.0 / math.sqrt(self.cfg.decoder_population_size))
        self.W_decoder_readout = nn.Parameter(w_dec, requires_grad=False)

        phon_emb = torch.randn(V, D, generator=gen) * 0.1
        self.register_buffer("phoneme_embedding", phon_emb)

        self.register_buffer(
            "morphemes",
            torch.full((N, self.cfg.max_morphemes), -1, dtype=torch.int16),
        )

        self._percept_buffer: Deque[Tensor] = deque(
            maxlen=self.cfg.percept_delay_ticks,
        )
        self.register_buffer(
            "phoneme_evidence",
            torch.zeros(1, D, dtype=torch.float32),
        )
        self._percept_decay: float = math.exp(
            -1.0 / self.cfg.tau_decay_ticks
        )

        self._population_state: Optional[SubstrateState] = None

        self.register_buffer(
            "code_allocated", torch.zeros(N, dtype=torch.bool),
        )

        # Per-lemma RNG for code init.
        # Seed 11 matches all other manual_seed calls in this file (lines 99,
        # 594, 737, 952, 1252, 1362, 1567). The prior expression evaluated to
        # 11 in both branches of the Python and/or idiom, making the
        # conditional dead. Collapsed to the constant it always produced.
        self._rng = torch.Generator().manual_seed(11)

    def add_lemma(self, lemma_idx: int) -> None:
        if not (0 <= lemma_idx < self.population_size):
            raise ValueError(f"lemma_idx out of range: {lemma_idx}")
        if self.cfg.ablate_wernickes_aphasia:
            return
        with torch.no_grad():
            code = torch.randn(
                self.cfg.code_dim, generator=self._rng,
            ) * 0.1
            self.W_LtoP.data[lemma_idx] = code
            self.code_allocated[lemma_idx] = True

    def set_morphemes(
        self, lemma_idx: int, morpheme_ids: List[int],
    ) -> None:
        with torch.no_grad():
            n = min(len(morpheme_ids), self.cfg.max_morphemes)
            for k in range(self.cfg.max_morphemes):
                self.morphemes[lemma_idx, k] = (
                    int(morpheme_ids[k]) if k < n else -1
                )

    def _get_phi(self, lemma_idx: int) -> Tensor:
        # Compositional: if morphemes are set, sum their codes.
        m = self.morphemes[lemma_idx]
        valid = [int(v.item()) for v in m if int(v.item()) >= 0]
        if valid:
            phi = torch.zeros(1, self.cfg.code_dim, device=self.W_LtoP.device)
            for mid in valid:
                phi = phi + self.W_LtoP[mid].unsqueeze(0)
            return phi
        one_hot = torch.zeros(
            1, self.population_size, device=self.W_LtoP.device,
        )
        one_hot[0, lemma_idx] = 1.0
        return one_hot @ self.W_LtoP

    def produce_code(self, lemma_idx: int) -> Tensor:
        if self.cfg.ablate_wernickes_aphasia:
            return torch.zeros(1, self.cfg.code_dim)
        return self._get_phi(lemma_idx)

    def produce_segments(self, lemma_idx: int) -> List[int]:
        if self.cfg.ablate_wernickes_aphasia:
            return [0]
        phi = self.produce_code(lemma_idx)
        device = phi.device
        state = self.decoder.init_state(1, device=device)
        segments: List[int] = []
        dec_in_dim = self.cfg.decoder_population_size
        prev_seg_emb = torch.zeros(1, self.cfg.code_dim, device=device)
        for seg_i in range(self.cfg.decoder_max_segments):
            spike_acc = torch.zeros(1, dec_in_dim, device=device)
            for tick in range(self.cfg.decoder_ticks_per_segment):
                code_proj = (
                    phi[:, :dec_in_dim] if phi.shape[-1] >= dec_in_dim
                    else torch.cat([
                        phi,
                        torch.zeros(
                            1, dec_in_dim - phi.shape[-1], device=device,
                        ),
                    ], dim=-1)
                )
                prev_proj = (
                    prev_seg_emb[:, :dec_in_dim]
                    if prev_seg_emb.shape[-1] >= dec_in_dim
                    else torch.cat([
                        prev_seg_emb,
                        torch.zeros(
                            1, dec_in_dim - prev_seg_emb.shape[-1],
                            device=device,
                        ),
                    ], dim=-1)
                )
                dec_input = code_proj + 0.3 * prev_proj
                spikes, state = self.decoder(dec_input, state)
                spike_acc = spike_acc + spikes
            rate = spike_acc / self.cfg.decoder_ticks_per_segment
            logits = rate @ self.W_decoder_readout
            phoneme_idx = int(logits.argmax(dim=-1).item())
            segments.append(phoneme_idx)
            if phoneme_idx == PHONEME_END_IDX:
                break
            prev_seg_emb = self.phoneme_embedding[phoneme_idx].unsqueeze(0)
        return segments

    def receive_phoneme(self, phoneme_idx: int) -> None:
        if not (0 <= phoneme_idx < self.cfg.phoneme_inventory_size):
            raise ValueError(f"phoneme_idx out of range: {phoneme_idx}")
        if self.cfg.ablate_wernickes_aphasia:
            return
        emb = self.phoneme_embedding[phoneme_idx].unsqueeze(0)
        self._percept_buffer.append(emb.clone())

    def perception_tick(self) -> Tensor:
        if self.cfg.ablate_wernickes_aphasia:
            return torch.zeros(1, self.population_size)
        device = self.phoneme_evidence.device
        if len(self._percept_buffer) >= self.cfg.percept_delay_ticks:
            arriving = self._percept_buffer[0].to(device)
        else:
            arriving = torch.zeros(1, self.cfg.code_dim, device=device)
        with torch.no_grad():
            self.phoneme_evidence.mul_(self._percept_decay).add_(arriving)
        a_l_percept = self.phoneme_evidence @ self.W_LtoP.t()
        return a_l_percept

    def get_state(self) -> Dict[str, Any]:
        return {
            "W_LtoP": self.W_LtoP.detach().cpu().clone(),
            "W_decoder_readout": (
                self.W_decoder_readout.detach().cpu().clone()
            ),
            "morphemes": self.morphemes.detach().cpu().clone(),
            "code_allocated": (
                self.code_allocated.detach().cpu().clone()
            ),
            "phoneme_evidence": (
                self.phoneme_evidence.detach().cpu().clone()
            ),
            "percept_buffer": [
                t.detach().cpu().clone() for t in self._percept_buffer
            ],
            "population_state": (
                self.population.state_dict_full(self._population_state)
                if self._population_state is not None else None
            ),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        device = self.W_LtoP.device
        with torch.no_grad():
            self.W_LtoP.data.copy_(state["W_LtoP"].to(device))
            self.W_decoder_readout.data.copy_(
                state["W_decoder_readout"].to(device),
            )
            self.morphemes.copy_(state["morphemes"].to(device))
            self.code_allocated.copy_(
                state["code_allocated"].to(device),
            )
            self.phoneme_evidence.copy_(
                state["phoneme_evidence"].to(device),
            )
        self._percept_buffer.clear()
        for t in state.get("percept_buffer", []):
            self._percept_buffer.append(t.to(device))
        sub = state.get("population_state")
        if sub is not None:
            self._population_state = self.population.load_state_full(
                sub, device=device,
            )


_VOWEL_PHONEMES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 39, 41}
_ILLEGAL_ONSETS = {
    (24, 26),   # /k n/
    (28, 26),   # /p n/
    (32, 26),   # /t n/
    (28, 14),   # /p f/
}


def _is_vowel(phoneme_idx: int) -> bool:
    return phoneme_idx in _VOWEL_PHONEMES


def _onset_legal(onset_phonemes: List[int]) -> bool:
    if len(onset_phonemes) <= 1:
        return True
    if len(onset_phonemes) >= 2:
        pair = (onset_phonemes[0], onset_phonemes[1])
        if pair in _ILLEGAL_ONSETS:
            return False
    return True


# RM306
class BrocasModule(nn.Module):
    def __init__(self, cfg: Optional[BrocasConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or BrocasConfig()
        gen = torch.Generator().manual_seed(24680)

        self.B44 = SSMLIFSubstrate(
            num_neurons=self.cfg.B44_size,
            n_dim=self.cfg.n_hidden,
            dt=1.0,
            v_threshold_base=1.0,
            theta_decay=self.cfg.B44_rho_theta,
            theta_increment=self.cfg.B44_Delta_theta,
            surrogate_alpha=self.cfg.B44_alpha,  # NOT biological. Training artifact.
        )

        self.B45 = SSMLIFSubstrate(
            num_neurons=self.cfg.B45_size,
            n_dim=self.cfg.n_hidden,
            dt=1.0,
            v_threshold_base=1.0,
            theta_decay=self.cfg.B45_rho_theta,
            theta_increment=self.cfg.B45_Delta_theta,
            surrogate_alpha=self.cfg.B45_alpha,  # NOT biological. Training artifact.
        )

        V = self.cfg.phoneme_inventory_size
        w_seg = torch.randn(V, self.cfg.B44_size, generator=gen) * 0.3
        self.W_seg_to_B44 = nn.Parameter(w_seg, requires_grad=False)

        mixer_out_dim = 256
        self.mixer_out_dim = mixer_out_dim
        self.B44_mixer = SpikeMixer(self.cfg.B44_size, mixer_out_dim)
        self.B45_mixer = SpikeMixer(self.cfg.B45_size, mixer_out_dim)

        w_b4to5 = torch.randn(
            mixer_out_dim, self.cfg.B45_size, generator=gen,
        ) * (1.0 / math.sqrt(mixer_out_dim))
        self.W_B44_to_B45 = nn.Parameter(w_b4to5, requires_grad=False)

        self.completion_detector = nn.Linear(
            2 * mixer_out_dim, 1, bias=True,
        )
        with torch.no_grad():
            self.completion_detector.weight.mul_(0.1)
            self.completion_detector.bias.fill_(-0.5)
        for p in self.completion_detector.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "B44_receptor_scaling",
            torch.tensor([
                self.cfg.B44_receptor_DA,
                self.cfg.B44_receptor_NE,
                self.cfg.B44_receptor_ACh,
                self.cfg.B44_receptor_5HT,
            ]),
        )
        self.register_buffer(
            "B45_receptor_scaling",
            torch.tensor([
                self.cfg.B45_receptor_DA,
                self.cfg.B45_receptor_NE,
                self.cfg.B45_receptor_ACh,
                self.cfg.B45_receptor_5HT,
            ]),
        )

        self.register_buffer(
            "b_buffer",
            torch.zeros(1, self.cfg.max_buffer_slots, dtype=torch.float32),
        )
        w_kbuf = torch.randn(V, self.cfg.max_buffer_slots, generator=gen) * 0.1
        self.W_k_buf = nn.Parameter(w_kbuf, requires_grad=False)

        frame_t = torch.randn(self.cfg.num_frame_templates, 64, generator=gen) * 0.2
        self.register_buffer("frame_templates", frame_t)

        self._emission_ring: Deque[List[int]] = deque(
            maxlen=self.cfg.emission_ring_buffer_size,
        )
        self._current_segments: List[int] = []
        self._ticks_in_segment: int = 0
        self._rolled_back_this_segment: bool = False

        self._B44_state: Optional[SubstrateState] = None
        self._B45_state: Optional[SubstrateState] = None

    def _ensure_states(self, batch: int, device: torch.device) -> None:
        if (
            self._B44_state is None
            or self._B44_state.h.shape[0] != batch
            or self._B44_state.h.device != device
        ):
            self._B44_state = self.B44.init_state(batch, device=device)
        if (
            self._B45_state is None
            or self._B45_state.h.shape[0] != batch
            or self._B45_state.h.device != device
        ):
            self._B45_state = self.B45.init_state(batch, device=device)
        if self.b_buffer.shape[0] != batch:
            self.b_buffer = torch.zeros(
                batch, self.cfg.max_buffer_slots, device=device,
            )

    def reset_emission(self) -> None:
        self._emission_ring.clear()
        self._current_segments.clear()
        self._ticks_in_segment = 0
        self._rolled_back_this_segment = False

    def tick_segment(
        self,
        phoneme_idx: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Optional[List[int]], bool]:
        self._ensure_states(1, device)
        V = self.cfg.phoneme_inventory_size
        seg_one_hot = torch.zeros(1, V, device=device)
        if 0 <= phoneme_idx < V:
            seg_one_hot[0, phoneme_idx] = 1.0

        rolled_back = False
        if (
            not self.cfg.ablate_BA45
            and _is_vowel(phoneme_idx)
            and len(self._emission_ring) > 0
            and not self._rolled_back_this_segment
            and len(self._current_segments) == 0
        ):
            prev_sigma = list(self._emission_ring[-1])
            if (
                len(prev_sigma) >= 2
                and not _is_vowel(prev_sigma[-1])
                and not _is_vowel(prev_sigma[-2])
            ):
                transferred = prev_sigma.pop()
                ring_list = list(self._emission_ring)
                ring_list[-1] = prev_sigma
                self._emission_ring.clear()
                for s in ring_list:
                    self._emission_ring.append(s)
                self._current_segments = [transferred]
                self._rolled_back_this_segment = True
                rolled_back = True

        if self.cfg.ablate_BA44:
            b44_input = torch.zeros(1, self.cfg.B44_size, device=device)
        else:
            b44_input = seg_one_hot @ self.W_seg_to_B44

        b44_spike_acc = torch.zeros(1, self.cfg.B44_size, device=device)
        b45_spike_acc = torch.zeros(1, self.cfg.B45_size, device=device)
        for t in range(self.cfg.ticks_per_segment):
            b44_spikes, self._B44_state = self.B44(
                b44_input, self._B44_state,
            )
            b44_spike_acc = b44_spike_acc + b44_spikes
            b44_mix = self.B44_mixer(b44_spikes)
            if self.cfg.ablate_BA45:
                b45_input = torch.zeros(
                    1, self.cfg.B45_size, device=device,
                )
            else:
                b45_input = b44_mix @ self.W_B44_to_B45
            b45_spikes, self._B45_state = self.B45(
                b45_input, self._B45_state,
            )
            b45_spike_acc = b45_spike_acc + b45_spikes

        with torch.no_grad():
            phon_contrib = seg_one_hot @ self.W_k_buf
            self.b_buffer.mul_(0.95).add_(phon_contrib)

        self._current_segments.append(phoneme_idx)

        b44_mix_final = self.B44_mixer(
            b44_spike_acc / self.cfg.ticks_per_segment
        )
        b45_mix_final = self.B45_mixer(
            b45_spike_acc / self.cfg.ticks_per_segment
        )
        combined = torch.cat([b44_mix_final, b45_mix_final], dim=-1)
        completion_logit = self.completion_detector(combined)
        completion_prob = torch.sigmoid(completion_logit).item()

        emitted: Optional[List[int]] = None
        if (
            completion_prob > self.cfg.completion_threshold
            and self._is_segment_legal()
        ):
            emitted = list(self._current_segments)
            self._emission_ring.append(emitted)
            self._current_segments = []
            self._rolled_back_this_segment = False
        else:
            if (
                len(self._current_segments) >= 2
                and _is_vowel(self._current_segments[-1])
                and self._is_segment_legal()
                and len(self._current_segments) <= 3
            ):
                emitted = list(self._current_segments)
                self._emission_ring.append(emitted)
                self._current_segments = []
                self._rolled_back_this_segment = False
            elif (
                len(self._current_segments) == 3
                and _is_vowel(self._current_segments[1])
                and self._is_segment_legal()
            ):
                emitted = list(self._current_segments)
                self._emission_ring.append(emitted)
                self._current_segments = []
                self._rolled_back_this_segment = False

        return emitted, rolled_back

    def _is_segment_legal(self) -> bool:
        if not self._current_segments:
            return False
        onset: List[int] = []
        for p in self._current_segments:
            if _is_vowel(p):
                break
            onset.append(p)
        return _onset_legal(onset)

    def force_emit(self) -> Optional[List[int]]:
        if not self._current_segments:
            return None
        emitted = list(self._current_segments)
        self._emission_ring.append(emitted)
        self._current_segments = []
        self._rolled_back_this_segment = False
        return emitted

    def reverse_cluster(self, sigmas: List[List[int]]) -> List[int]:
        out: List[int] = []
        for sigma in sigmas:
            out.extend(sigma)
        return out

    def consume_buffer_slot(self, slot: int = 0) -> None:
        if not (0 <= slot < self.cfg.max_buffer_slots):
            raise ValueError(f"slot out of range: {slot}")
        with torch.no_grad():
            self.b_buffer[:, slot] = 0.0

    def get_emissions(self) -> List[List[int]]:
        return list(self._emission_ring)

    def compute_e_monitor(
        self,
        w_monitor: Tensor,
        w_intended: Tensor,
    ) -> Tensor:
        """Compute the self-monitoring error signal from the production cascade.

        BIOLOGICAL GROUNDING: PRAGMI Brocas Corpus Section 23, Equation 23.3.
        Internal self-monitoring (Levelt 1983) compares the phonological word
        currently being produced by Broca's syllabification against the
        intended utterance held in the lemma stratum. The error signal e_monitor
        re-enters the DMN at PCC via the Section 25 closure
        (Brocas Corpus Section 25, Equations 25.1-25.8), traversing
        MPFC and IPL with Davey (2016) modulatory connection strengths, and
        returns through the synaptic re-entry register at the 160 ms ignition
        latency to modulate lemma competition.

        v4 implementation note: w_monitor is supplied by the caller as c_cas
        (the cascade output in concept_dim=64 space), which is the closest
        available per-tick proxy for Broca's current linguistic content. The
        v5 refactor will replace this with the syllabification output projected
        from the b_buffer into P's code space as a continuous per-tick signal.

        Args:
            w_monitor: current phonological monitor signal, shape (B, code_dim).
                In v4 this is c_cas from the lemma stratum cascade output.
            w_intended: intended phonological code, shape (B, code_dim).
                This is phi = W_LtoP[selected_lemma] from PhonologicalCodeStore.

        Returns:
            e_monitor: squared L2 error, shape (B,). One scalar per batch item.
                Dimensionless monitoring error. NOT a biological quantity per se;
                the biological signal is the prediction error carried by the
                auditory self-monitoring pathway.
        """
        diff = w_monitor - w_intended
        return (diff * diff).sum(dim=-1)

    def get_state(self) -> Dict[str, Any]:
        return {
            "W_seg_to_B44": self.W_seg_to_B44.detach().cpu().clone(),
            "W_B44_to_B45": self.W_B44_to_B45.detach().cpu().clone(),
            "W_k_buf": self.W_k_buf.detach().cpu().clone(),
            "b_buffer": self.b_buffer.detach().cpu().clone(),
            "emission_ring": [list(s) for s in self._emission_ring],
            "current_segments": list(self._current_segments),
            "rolled_back_this_segment": self._rolled_back_this_segment,
            "B44_state": (
                self.B44.state_dict_full(self._B44_state)
                if self._B44_state is not None else None
            ),
            "B45_state": (
                self.B45.state_dict_full(self._B45_state)
                if self._B45_state is not None else None
            ),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        device = self.W_seg_to_B44.device
        with torch.no_grad():
            self.W_seg_to_B44.data.copy_(state["W_seg_to_B44"].to(device))
            self.W_B44_to_B45.data.copy_(state["W_B44_to_B45"].to(device))
            self.W_k_buf.data.copy_(state["W_k_buf"].to(device))
            self.b_buffer.copy_(state["b_buffer"].to(device))
        self._emission_ring.clear()
        for s in state.get("emission_ring", []):
            self._emission_ring.append(list(s))
        self._current_segments = list(state.get("current_segments", []))
        self._rolled_back_this_segment = bool(
            state.get("rolled_back_this_segment", False)
        )
        b44 = state.get("B44_state")
        if b44 is not None:
            self._B44_state = self.B44.load_state_full(b44, device=device)
        b45 = state.get("B45_state")
        if b45 is not None:
            self._B45_state = self.B45.load_state_full(b45, device=device)


# RM307
class ArcuateTransport(nn.Module):
    def __init__(
        self,
        cfg: Optional[ArcuateConfig] = None,
        phoneme_dim: int = 32,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ArcuateConfig()
        self.phoneme_dim = phoneme_dim
        gen = torch.Generator().manual_seed(31415)
        eye = torch.eye(phoneme_dim)
        noise = torch.randn(
            phoneme_dim, phoneme_dim, generator=gen,
        ) * self.cfg.near_identity_perturbation
        w = eye + noise
        # FROZEN. NOT plastic.
        self.W_arc = nn.Parameter(w, requires_grad=False)
        self._delay_ticks: int = max(1, int(math.ceil(self.cfg.tau_arc_ms)))
        self._buffer: Deque[Tensor] = deque(maxlen=self._delay_ticks)

    def forward(self, seg_emb: Tensor) -> Tensor:
        if self.cfg.ablate_arcuate:
            return torch.zeros_like(seg_emb)
        projected = seg_emb @ self.W_arc
        self._buffer.append(projected.detach().clone())
        if len(self._buffer) >= self._delay_ticks:
            return self._buffer[0]
        return torch.zeros_like(seg_emb)

    def reset(self) -> None:
        self._buffer.clear()

    def get_state(self) -> Dict[str, Any]:
        return {
            "W_arc": self.W_arc.detach().cpu().clone(),
            "buffer": [t.detach().cpu().clone() for t in self._buffer],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        device = self.W_arc.device
        with torch.no_grad():
            self.W_arc.data.copy_(state["W_arc"].to(device))
        self._buffer.clear()
        for t in state.get("buffer", []):
            self._buffer.append(t.to(device))


# RM308
class ProductionLoop(nn.Module):
    def __init__(
        self,
        lemma_stratum: LemmaStratum,
        phonological_store: PhonologicalCodeStore,
        arcuate: ArcuateTransport,
        brocas: BrocasModule,
        cfg: Optional[ProductionConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ProductionConfig()
        self.lemma_stratum = lemma_stratum
        self.phonological_store = phonological_store
        self.arcuate = arcuate
        self.brocas = brocas

    def produce(
        self,
        lemma_idx: int,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        phi = self.phonological_store.produce_code(lemma_idx)
        segments = self.phonological_store.produce_segments(lemma_idx)
        self.brocas.reset_emission()
        self.arcuate.reset()
        emissions: List[List[int]] = []
        rollbacks: int = 0
        for seg in segments:
            if seg == PHONEME_END_IDX:
                break
            seg_emb = torch.zeros(1, self.arcuate.phoneme_dim, device=device)
            if seg < self.arcuate.phoneme_dim:
                seg_emb[0, seg] = 1.0
            else:
                seg_emb[0, seg % self.arcuate.phoneme_dim] = 1.0
            _ = self.arcuate(seg_emb)
            emitted, rolled = self.brocas.tick_segment(seg, device=device)
            if rolled:
                rollbacks += 1
            if emitted is not None:
                emissions.append(emitted)
        last = self.brocas.force_emit()
        if last is not None:
            emissions.append(last)
        return {
            "phi": phi,
            "segments": segments,
            "sigmas": emissions,
            "rollbacks": rollbacks,
        }

    def get_state(self) -> Dict[str, Any]:
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        pass


# RM309
class OrthographyModule(nn.Module):
    def __init__(
        self,
        cfg: Optional[OrthographyConfig] = None,
        lemma_population_size: int = 4106,
        brocas_B44_size: int = 1024,
    ) -> None:
        super().__init__()
        self.cfg = cfg or OrthographyConfig()
        self.lemma_population_size = lemma_population_size
        self.brocas_B44_size = brocas_B44_size
        N_o = self.cfg.population_size
        gen = torch.Generator().manual_seed(57721)

        self.substrate = SSMLIFSubstrate(
            num_neurons=N_o,
            n_dim=self.cfg.n_hidden,
            dt=self.cfg.dt,
            v_threshold_base=self.cfg.v_threshold_base,
            theta_decay=self.cfg.theta_decay,
            theta_increment=self.cfg.theta_increment,
            surrogate_alpha=self.cfg.surrogate_alpha,  # NOT biological. Training artifact.
        )

        w_otol = torch.randn(
            N_o, lemma_population_size, generator=gen,
        ) * (1.0 / math.sqrt(N_o))
        w_otol = w_otol * 0.1
        self.W_OtoL = nn.Parameter(w_otol, requires_grad=False)

        w_otob44 = torch.randn(
            N_o, brocas_B44_size, generator=gen,
        ) * (1.0 / math.sqrt(N_o))
        w_otob44 = w_otob44 * 0.1
        self.W_OtoB44 = nn.Parameter(w_otob44, requires_grad=False)

        self.route_winner_threshold: float = 0.5
        self.spelling_emit_threshold: float = 0.3

        self._substrate_state: Optional[SubstrateState] = None

        self.register_buffer(
            "otol_pre_trace",
            torch.zeros(N_o, dtype=torch.float32),
        )
        self.register_buffer(
            "otol_post_trace",
            torch.zeros(lemma_population_size, dtype=torch.float32),
        )

    def encode_string(self, word: str) -> Tensor:
        if len(word) > self.cfg.n_positions:
            raise ValueError(
                f"Word length {len(word)} exceeds n_positions "
                f"{self.cfg.n_positions}. Words longer than "
                f"{self.cfg.n_positions} letters refused at this stage."
            )
        activation = torch.zeros(1, self.cfg.population_size)
        for pos, ch in enumerate(word.lower()):
            if not ('a' <= ch <= 'z'):
                continue
            letter_idx = ord(ch) - ord('a')
            row = letter_idx * self.cfg.n_positions + pos
            activation[0, row] = 1.0
        return activation

    def read(
        self,
        word_activation: Tensor,
        direct_only: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        if (
            self._substrate_state is None
            or self._substrate_state.h.shape[0] != word_activation.shape[0]
            or self._substrate_state.h.device != device
        ):
            self._substrate_state = self.substrate.init_state(
                batch_size=word_activation.shape[0], device=device,
            )
        substrate_input = word_activation * 5.0
        spike_accumulator = torch.zeros(
            word_activation.shape[0],
            self.cfg.population_size,
            device=device,
        )
        for t in range(self.cfg.encode_ticks):
            spikes, self._substrate_state = self.substrate(
                substrate_input, self._substrate_state,
            )
            spike_accumulator = spike_accumulator + spikes
        o_activation = spike_accumulator / self.cfg.encode_ticks

        if self.cfg.ablate_lexical_route:
            direct_lemma = torch.zeros(
                word_activation.shape[0],
                self.lemma_population_size,
                device=device,
            )
        else:
            direct_lemma = o_activation @ self.W_OtoL

        if self.cfg.ablate_sublexical_route:
            indirect_b44 = torch.zeros(
                word_activation.shape[0],
                self.brocas_B44_size,
                device=device,
            )
        else:
            indirect_b44 = o_activation @ self.W_OtoB44

        direct_peak = float(direct_lemma.max().item())
        winner = "direct"
        if direct_peak < self.route_winner_threshold:
            winner = "indirect"
        return {
            "o_activation": o_activation,
            "direct_lemma_activation": direct_lemma,
            "indirect_b44_drive": indirect_b44,
            "winner": winner,
            "direct_peak": direct_peak,
        }

    def write(
        self,
        lemma_activation: Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> str:
        o_act = lemma_activation @ self.W_OtoL.t()
        o_view = o_act.view(
            o_act.shape[0],
            self.cfg.n_letters,
            self.cfg.n_positions,
        )
        chars: List[str] = []
        for pos in range(self.cfg.n_positions):
            slice_pos = o_view[0, :, pos]
            max_val = float(slice_pos.max().item())
            if max_val < self.spelling_emit_threshold:
                break
            letter_idx = int(slice_pos.argmax().item())
            chars.append(chr(ord('a') + letter_idx))
        return ''.join(chars)

    def update_lexical(
        self,
        o_activation: Tensor,
        lemma_activation: Tensor,
        modulation: float = 1.0,
    ) -> None:
        if self.cfg.ablate_lexical_route:
            return
        with torch.no_grad():
            o_d = o_activation.detach().mean(dim=0).float()
            l_d = lemma_activation.detach().mean(dim=0).float()
            decay = 0.95
            old_pre = self.otol_pre_trace.clone()
            self.otol_pre_trace.mul_(decay).add_(o_d)
            self.otol_post_trace.mul_(decay).add_(l_d)
            ltp = torch.outer(old_pre, l_d)
            delta = 1e-3 * modulation * ltp
            self.W_OtoL.data.add_(delta)
            self.W_OtoL.data.clamp_(-1.0, 1.0)

    def get_state(self) -> Dict[str, Any]:
        return {
            "W_OtoL": self.W_OtoL.detach().cpu().clone(),
            "W_OtoB44": self.W_OtoB44.detach().cpu().clone(),
            "otol_pre_trace": self.otol_pre_trace.detach().cpu().clone(),
            "otol_post_trace": (
                self.otol_post_trace.detach().cpu().clone()
            ),
            "substrate_state": (
                self.substrate.state_dict_full(self._substrate_state)
                if self._substrate_state is not None else None
            ),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        device = self.W_OtoL.device
        with torch.no_grad():
            self.W_OtoL.data.copy_(state["W_OtoL"].to(device))
            self.W_OtoB44.data.copy_(state["W_OtoB44"].to(device))
            self.otol_pre_trace.copy_(state["otol_pre_trace"].to(device))
            self.otol_post_trace.copy_(state["otol_post_trace"].to(device))
        sub = state.get("substrate_state")
        if sub is not None:
            self._substrate_state = self.substrate.load_state_full(
                sub, device=device,
            )


@dataclass
class CascadeOutput:
    spikes: Tensor
    x_act: Tensor
    parse_posteriors: Tensor
    forecast_production: Tensor
    forecast_comprehension: Tensor
    attention_flipped: bool
    flip_cost: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# RM310
class CascadeComposer(nn.Module):
    def __init__(
        self,
        cfg: Optional[CascadeConfig] = None,
        lemma_population_size: int = 4106,
        brocas_B45_size: int = 1024,
    ) -> None:
        super().__init__()
        self.cfg = cfg or CascadeConfig()
        self.lemma_population_size = lemma_population_size
        self.brocas_B45_size = brocas_B45_size
        N_x = self.cfg.population_size
        gen = torch.Generator().manual_seed(86420)

        self.substrate = SSMLIFSubstrate(
            num_neurons=N_x,
            n_dim=self.cfg.n_hidden,
            dt=self.cfg.dt,
            v_threshold_base=self.cfg.v_threshold_base,
            theta_decay=self.cfg.theta_decay,
            theta_increment=self.cfg.theta_increment,
            surrogate_alpha=self.cfg.surrogate_alpha,  # NOT biological. Training artifact.
        )

        w_xtol = torch.randn(
            N_x, lemma_population_size, generator=gen,
        ) * (1.0 / math.sqrt(N_x))
        w_xtol = w_xtol * 0.1
        self.W_XtoL = nn.Parameter(w_xtol, requires_grad=False)

        w_xtob45 = torch.randn(
            N_x, brocas_B45_size, generator=gen,
        ) * (1.0 / math.sqrt(N_x))
        w_xtob45 = w_xtob45 * 0.1
        self.W_X_to_B45 = nn.Parameter(w_xtob45, requires_grad=False)

        self.parse_separator = nn.Linear(
            N_x, self.cfg.n_hypotheses, bias=False,
        )
        with torch.no_grad():
            self.parse_separator.weight.mul_(0.05)
        for p in self.parse_separator.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "frame_templates",
            torch.randn(16, 64, generator=gen) * 0.2,
        )

        self._composer_delay_ms: float = 100.0
        self._composer_delay_ticks: int = int(
            math.ceil(self._composer_delay_ms / self.cfg.dt)
        )
        self._composer_buffer: Deque[Tensor] = deque(
            maxlen=self._composer_delay_ticks,
        )
        self._eta_X: float = 0.10
        self._flip_weight: float = 2.0

        self.register_buffer(
            "parse_posteriors",
            torch.ones(1, self.cfg.n_hypotheses) / self.cfg.n_hypotheses,
        )
        self.register_buffer(
            "last_top_parse",
            torch.tensor(-1, dtype=torch.long),
        )

        self._substrate_state: Optional[SubstrateState] = None

        self.register_buffer(
            "activations",
            torch.zeros(1, N_x, dtype=torch.float32),
        )
        self._activation_decay: float = 0.9

    def _ensure_state(self, batch: int, device: torch.device) -> None:
        N_x = self.cfg.population_size
        if (
            self._substrate_state is None
            or self._substrate_state.h.shape[0] != batch
            or self._substrate_state.h.device != device
        ):
            self._substrate_state = self.substrate.init_state(
                batch_size=batch, device=device,
            )
        if (
            self.activations.shape[0] != batch
            or self.activations.device != device
        ):
            self.activations = torch.zeros(
                batch, N_x, dtype=torch.float32, device=device,
            )
            self._composer_buffer.clear()
        if self.parse_posteriors.shape[0] != batch:
            self.parse_posteriors = (
                torch.ones(batch, self.cfg.n_hypotheses, device=device)
                / self.cfg.n_hypotheses
            )

    def tick(
        self,
        lemma_activation: Tensor,
        brocas_b45_activation: Optional[Tensor] = None,
    ) -> CascadeOutput:
        B = lemma_activation.shape[0]
        device = lemma_activation.device
        N_x = self.cfg.population_size

        self._ensure_state(B, device)

        l_input = lemma_activation @ self.W_XtoL.t()
        if brocas_b45_activation is not None:
            b45_proj = brocas_b45_activation @ self.W_X_to_B45.t()
            if not self.cfg.ablate_top_down_bias:
                substrate_input = l_input + self.cfg.top_down_gain * b45_proj
            else:
                substrate_input = l_input
        else:
            substrate_input = l_input

        spikes, self._substrate_state = self.substrate(
            substrate_input, self._substrate_state,
        )

        with torch.no_grad():
            self.activations = (
                self._activation_decay * self.activations
                + spikes.detach()
            )
            self.activations.clamp_(0.0, 10.0)

        parse_logits = self.parse_separator(self.activations)
        evidence = torch.softmax(parse_logits, dim=-1)
        with torch.no_grad():
            new_post = self.parse_posteriors * evidence
            new_post = new_post / (
                new_post.sum(dim=-1, keepdim=True) + 1e-8
            )
            self.parse_posteriors.copy_(new_post)

        top_idx = int(self.parse_posteriors.mean(dim=0).argmax().item())
        prev_top = int(self.last_top_parse.item())
        flipped = False
        flip_cost = 0.0
        if (
            not self.cfg.ablate_top_down_bias
            and prev_top >= 0
            and top_idx != prev_top
        ):
            flipped = True
            old = float(
                self.parse_posteriors.mean(dim=0)[prev_top].item()
            ) + 1e-8
            new = float(
                self.parse_posteriors.mean(dim=0)[top_idx].item()
            ) + 1e-8
            delta_I = math.log(old) - math.log(new)
            flip_cost = self._flip_weight * delta_I
        self.last_top_parse.fill_(top_idx)

        forecast_prod = self._rollout_forecast(
            depth=self.cfg.depth_production, device=device,
        )
        forecast_comp = self._rollout_forecast(
            depth=self.cfg.depth_comprehension, device=device,
        )

        x_back = self.activations.detach() @ self.W_XtoL
        self._composer_buffer.append(x_back.clone())

        return CascadeOutput(
            spikes=spikes,
            x_act=self.activations,
            parse_posteriors=self.parse_posteriors,
            forecast_production=forecast_prod,
            forecast_comprehension=forecast_comp,
            attention_flipped=flipped,
            flip_cost=flip_cost,
            diagnostics={
                "top_parse": top_idx,
                "x_spike_rate": float(spikes.detach().mean().item()),
            },
        )

    def _rollout_forecast(
        self,
        depth: int,
        device: torch.device,
    ) -> Tensor:
        rolled_state = SubstrateState(
            h=self._substrate_state.h.detach().clone(),
            v=self._substrate_state.v.detach().clone(),
            theta=self._substrate_state.theta.detach().clone(),
            last_spike=self._substrate_state.last_spike.detach().clone(),
        )
        batch = self._substrate_state.h.shape[0]
        N_x = self.cfg.population_size
        accumulator = torch.zeros(batch, N_x, device=device)
        for d in range(depth):
            zero_input = torch.zeros(batch, N_x, device=device)
            spikes_d, rolled_state = self.substrate(
                zero_input, rolled_state,
            )
            accumulator = accumulator + spikes_d
        return accumulator / max(depth, 1)

    def pop_composition_feedback(self) -> Optional[Tensor]:
        if len(self._composer_buffer) >= self._composer_delay_ticks:
            return self._eta_X * self._composer_buffer[0]
        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            "W_XtoL": self.W_XtoL.detach().cpu().clone(),
            "W_X_to_B45": self.W_X_to_B45.detach().cpu().clone(),
            "parse_separator_weight": (
                self.parse_separator.weight.detach().cpu().clone()
            ),
            "parse_posteriors": (
                self.parse_posteriors.detach().cpu().clone()
            ),
            "last_top_parse": int(self.last_top_parse.item()),
            "activations": self.activations.detach().cpu().clone(),
            "composer_buffer": [
                t.detach().cpu().clone() for t in self._composer_buffer
            ],
            "substrate_state": (
                self.substrate.state_dict_full(self._substrate_state)
                if self._substrate_state is not None else None
            ),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        device = self.W_XtoL.device
        with torch.no_grad():
            self.W_XtoL.data.copy_(state["W_XtoL"].to(device))
            self.W_X_to_B45.data.copy_(state["W_X_to_B45"].to(device))
            self.parse_separator.weight.data.copy_(
                state["parse_separator_weight"].to(device),
            )
            self.parse_posteriors.copy_(
                state["parse_posteriors"].to(device),
            )
        self.last_top_parse.fill_(int(state["last_top_parse"]))
        self.activations = state["activations"].to(device)
        self._composer_buffer.clear()
        for t in state.get("composer_buffer", []):
            self._composer_buffer.append(t.to(device))
        sub = state.get("substrate_state")
        if sub is not None:
            self._substrate_state = self.substrate.load_state_full(
                sub, device=device,
            )
