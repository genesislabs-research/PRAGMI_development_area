# RM numbers reference "The code classes explained" in README.md

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from Timmyv4_PART1 import SSMLIFSubstrate, SubstrateState


# =========================================================================
# Loop-state format version + migration registry
# =========================================================================

LOOP_STATE_FORMAT_VERSION: int = 1


def _migrate_loop_state(
    state: Dict[str, Any],
    from_version: int,
    to_version: int,
) -> Dict[str, Any]:
    if from_version == to_version:
        return state
    fn = _LOOP_STATE_MIGRATIONS.get((from_version, to_version))
    if fn is None:
        v = from_version
        while v < to_version:
            step = _LOOP_STATE_MIGRATIONS.get((v, v + 1))
            if step is None:
                raise ValueError(
                    f"no migration path from loop-state v{v} to "
                    f"v{v + 1}; saved=v{from_version} target="
                    f"v{to_version}"
                )
            state = step(state)
            v += 1
        return state
    return fn(state)


_LOOP_STATE_MIGRATIONS: Dict[
    Tuple[int, int], Any
] = {}


# =========================================================================
# Section 1: Operating condition and Phi projection
# =========================================================================

# RM202
@dataclass
class OperatingCondition:
    da: float = 0.0
    ne: float = 0.0
    ach: float = 0.5
    five_ht: float = 0.0
    novelty: float = 0.0
    arousal: float = 0.0
    valence: float = 0.0
    maturity: float = 0.0

    def to_tensor(self, device: torch.device | str = "cpu") -> Tensor:
        return torch.tensor(
            [self.da, self.ne, self.ach, self.five_ht,
             self.novelty, self.arousal, self.valence, self.maturity],
            dtype=torch.float32,
            device=device,
        )

    @classmethod
    def from_tensor(cls, t: Tensor) -> "OperatingCondition":
        v = t.detach().cpu().tolist()
        if len(v) != 8:
            raise ValueError(
                f"OperatingCondition tensor must have 8 elements, "
                f"got {len(v)}"
            )
        return cls(
            da=v[0], ne=v[1], ach=v[2], five_ht=v[3],
            novelty=v[4], arousal=v[5], valence=v[6], maturity=v[7],
        )


# RM203
class Phi(nn.Module):
    def __init__(
        self,
        encoding_dim: int,
        reserved_subspace_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoding_dim = encoding_dim
        if reserved_subspace_size is None:
            reserved_subspace_size = max(8, encoding_dim // 8)
        if reserved_subspace_size < 8:
            raise ValueError(
                "reserved_subspace_size must be at least 8 to hold the "
                "eight-component operating condition"
            )
        self.reserved_subspace_size = reserved_subspace_size
        block = reserved_subspace_size // 8
        proj = torch.zeros(8, encoding_dim, dtype=torch.float32)
        for i in range(8):
            start = i * block
            end = start + block
            proj[i, start:end] = 1.0 / math.sqrt(block)
        # NOT a Parameter; this is fixed at construction.
        self.register_buffer("projection", proj)

    def forward(self, omega: Tensor) -> Tensor:
        return omega @ self.projection


# =========================================================================
# Section 2: R-STDP plasticity
# =========================================================================

# RM204
@dataclass
class RSTDPConfig:
    lr: float = 0.005
    trace_tau: float = 0.92
    depression_ratio: float = 0.55
    weight_min: float = -1.0
    weight_max: float = 1.0
    modulation_floor: float = 0.05


# RM205
class RSTDPRule(nn.Module):
    def __init__(
        self,
        pre_dim: int,
        post_dim: int,
        cfg: RSTDPConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("pre_trace", torch.zeros(pre_dim))
        self.register_buffer("post_trace", torch.zeros(post_dim))

    @torch.no_grad()
    def update(
        self,
        weight: Tensor,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        modulation: float,
    ) -> None:
        m_signed = float(modulation)
        m = max(abs(m_signed), self.cfg.modulation_floor)
        if m_signed < 0:
            m = -m
        device = weight.device
        pre = pre_spikes.detach().to(device).float()
        post = post_spikes.detach().to(device).float()
        if pre.dim() > 1:
            pre = pre.mean(dim=0)
        if post.dim() > 1:
            post = post.mean(dim=0)
        if pre.numel() != weight.shape[1]:
            raise ValueError(
                f"R-STDP shape mismatch: weight={tuple(weight.shape)} "
                f"pre={tuple(pre.shape)}"
            )
        if post.numel() != weight.shape[0]:
            raise ValueError(
                f"R-STDP shape mismatch: weight={tuple(weight.shape)} "
                f"post={tuple(post.shape)}"
            )
        if self.pre_trace.device != device:
            self.pre_trace = self.pre_trace.to(device)
            self.post_trace = self.post_trace.to(device)
        old_pre = self.pre_trace.clone()
        old_post = self.post_trace.clone()
        self.pre_trace.mul_(self.cfg.trace_tau).add_(pre)
        self.post_trace.mul_(self.cfg.trace_tau).add_(post)
        ltp = torch.outer(post, old_pre)
        ltd = torch.outer(old_post, pre)
        delta = self.cfg.lr * m * (
            ltp - self.cfg.depression_ratio * ltd
        )
        weight.add_(delta)
        weight.clamp_(min=self.cfg.weight_min, max=self.cfg.weight_max)

    def reset(self) -> None:
        self.pre_trace.zero_()
        self.post_trace.zero_()

    def serialize(self) -> Dict[str, Tensor]:
        return {
            "pre_trace": self.pre_trace.detach().cpu().clone(),
            "post_trace": self.post_trace.detach().cpu().clone(),
        }

    def restore(self, state: Dict[str, Tensor]) -> None:
        self.pre_trace.copy_(
            state["pre_trace"].to(self.pre_trace.device),
        )
        self.post_trace.copy_(
            state["post_trace"].to(self.post_trace.device),
        )


# =========================================================================
# Section 3: Episode store
# =========================================================================

# RM206
@dataclass
class EpisodeRecord:
    tick: int
    pcc_spikes: Tensor
    mpfc_spikes: Tensor
    prior_pcc_trace: Tensor
    predicted_pcc_trace: Tensor
    operating_condition: Tensor
    novelty: float
    prediction_error: float
    source_sha256: str
    structural_hash: str
    tau_proj: int = 0
    is_real: bool = True

    def to_state(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "pcc_spikes": self.pcc_spikes.detach().cpu().clone(),
            "mpfc_spikes": self.mpfc_spikes.detach().cpu().clone(),
            "prior_pcc_trace": (
                self.prior_pcc_trace.detach().cpu().clone()
            ),
            "predicted_pcc_trace": (
                self.predicted_pcc_trace.detach().cpu().clone()
            ),
            "operating_condition": (
                self.operating_condition.detach().cpu().clone()
            ),
            "novelty": float(self.novelty),
            "prediction_error": float(self.prediction_error),
            "source_sha256": str(self.source_sha256),
            "structural_hash": str(self.structural_hash),
            "tau_proj": int(self.tau_proj),
            "is_real": bool(self.is_real),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "EpisodeRecord":
        return cls(
            tick=int(state["tick"]),
            pcc_spikes=state["pcc_spikes"],
            mpfc_spikes=state["mpfc_spikes"],
            prior_pcc_trace=state["prior_pcc_trace"],
            predicted_pcc_trace=state["predicted_pcc_trace"],
            operating_condition=state["operating_condition"],
            novelty=float(state["novelty"]),
            prediction_error=float(state["prediction_error"]),
            source_sha256=str(state["source_sha256"]),
            structural_hash=str(state["structural_hash"]),
            tau_proj=int(state.get("tau_proj", 0)),
            is_real=bool(state.get("is_real", True)),
        )


# =========================================================================
# Section 4: Write gate and retrieval
# =========================================================================

class WriteGateError(RuntimeError):
    pass


# RM207
class WriteGate:
    @staticmethod
    def write(
        episodes: List[EpisodeRecord],
        record: EpisodeRecord,
        max_episodes: int,
        eviction: str = "lowest_novelty",
    ) -> None:
        if record.tau_proj != 0:
            raise WriteGateError(
                f"WriteGate refuses tau_proj={record.tau_proj}; only "
                f"tau_proj=0 is admissible to the causal record."
            )
        if eviction not in {"lowest_novelty", "fifo"}:
            raise ValueError(
                f"eviction must be 'lowest_novelty' or 'fifo', "
                f"got {eviction!r}"
            )
        if len(episodes) >= max_episodes:
            if eviction == "fifo":
                episodes.pop(0)
            else:  # lowest_novelty
                min_idx = 0
                min_key = (episodes[0].novelty, episodes[0].tick)
                for i in range(1, len(episodes)):
                    key = (episodes[i].novelty, episodes[i].tick)
                    if key < min_key:
                        min_key = key
                        min_idx = i
                episodes.pop(min_idx)
        episodes.append(record)


# RM208
class RetrievalDraw:
    @staticmethod
    def draw_past(
        episodes: List[EpisodeRecord],
        query_omega: Tensor,
        n: int = 1,
    ) -> List[EpisodeRecord]:
        if not episodes:
            return []
        q = query_omega.detach().to(torch.float32).flatten()
        if q.numel() != 8:
            raise ValueError(
                f"query_omega must have 8 elements, got {q.numel()}"
            )
        q_norm = q / (q.norm() + 1e-8)
        scored: List[Tuple[float, EpisodeRecord]] = []
        for ep in episodes:
            o = ep.operating_condition.detach().to(torch.float32).flatten()
            o_norm = o / (o.norm() + 1e-8)
            sim = float((q_norm * o_norm).sum().item())
            scored.append((sim, ep))
        scored.sort(key=lambda pair: -pair[0])
        return [ep for _, ep in scored[:n]]

    @staticmethod
    def draw_future(
        episodes: List[EpisodeRecord],
        query_omega: Tensor,
        n: int = 1,
    ) -> List[EpisodeRecord]:
        past = RetrievalDraw.draw_past(episodes, query_omega, n)
        forward = []
        for ep in past:
            ep_copy = EpisodeRecord(
                tick=ep.tick,
                pcc_spikes=ep.pcc_spikes.clone(),
                mpfc_spikes=ep.mpfc_spikes.clone(),
                prior_pcc_trace=ep.prior_pcc_trace.clone(),
                predicted_pcc_trace=ep.predicted_pcc_trace.clone(),
                operating_condition=ep.operating_condition.clone(),
                novelty=ep.novelty,
                prediction_error=ep.prediction_error,
                source_sha256=ep.source_sha256,
                structural_hash=ep.structural_hash,
                tau_proj=1,
            )
            forward.append(ep_copy)
        return forward


# =========================================================================
# Section 4b: Synaptic re-entry register (Equations 29 and 30)
# =========================================================================

# RM209
@dataclass
class SynapticReentryConfig:
    n_states: int = 20
    weight_min: float = -1.0
    weight_max: float = 1.0
    epsilon_quant: float = 0.05
    timing_window_pre_ms: float = -20.0
    timing_window_post_ms: float = 5.0
    update_floor: float = 1e-3  # NOT a biological quantity. Engineering floor.


# RM210
class SynapticReentryRegister(nn.Module):
    def __init__(
        self,
        cfg: SynapticReentryConfig,
        n_channels: int,
    ) -> None:
        super().__init__()
        if cfg.n_states < 2:
            raise ValueError(
                f"n_states must be at least 2, got {cfg.n_states}"
            )
        if cfg.weight_min >= cfg.weight_max:
            raise ValueError(
                f"weight_min must be less than weight_max"
            )
        if cfg.epsilon_quant <= 0.0:
            raise ValueError(
                f"epsilon_quant must be positive, "
                f"got {cfg.epsilon_quant}"
            )
        self.cfg = cfg
        self.n_channels = n_channels
        ladder = torch.linspace(
            cfg.weight_min, cfg.weight_max, cfg.n_states + 1,
            dtype=torch.float32,
        )
        self.register_buffer("ladder", ladder)
        mid = cfg.n_states // 2
        self.register_buffer(
            "state_index",
            torch.full((n_channels,), mid, dtype=torch.long),
        )
        self.register_buffer(
            "quantized_error",
            torch.zeros(n_channels, dtype=torch.long),
        )

    @torch.no_grad()
    def update(self, r_reentry_per_channel: Tensor) -> None:
        x = r_reentry_per_channel.detach().to(self.ladder.device).float()
        if x.dim() > 1:
            x = x.flatten()
        if x.numel() != self.n_channels:
            raise ValueError(
                f"r_reentry_per_channel must have {self.n_channels} "
                f"elements, got {x.numel()}"
            )
        k = torch.floor(
            x.abs() / self.cfg.epsilon_quant,
        ).clamp(min=0).long()
        self.quantized_error.copy_(k)
        target_value = x.clamp(self.cfg.weight_min, self.cfg.weight_max)
        diffs = (
            self.ladder.unsqueeze(0) - target_value.unsqueeze(1)
        ).abs()
        target_idx = diffs.argmin(dim=1)
        qualifying = x.abs() >= self.cfg.update_floor
        delta = torch.sign(
            (target_idx - self.state_index).float(),
        ).long()
        delta = delta * qualifying.long()
        new_idx = (self.state_index + delta).clamp(
            min=0, max=self.cfg.n_states,
        )
        self.state_index.copy_(new_idx)

    def read(self) -> Tensor:
        return self.ladder[self.state_index]

    def read_quantized_error(self) -> Tensor:
        k = self.quantized_error.float()
        k_capped = k.clamp(max=30.0)
        return 1.0 - torch.pow(2.0, -k_capped)

    def reset(self) -> None:
        mid = self.cfg.n_states // 2
        self.state_index.fill_(mid)
        self.quantized_error.zero_()

    def serialize(self) -> Dict[str, Tensor]:
        return {
            "state_index": self.state_index.detach().cpu().clone(),
            "quantized_error": (
                self.quantized_error.detach().cpu().clone()
            ),
        }

    def restore(self, state: Dict[str, Tensor]) -> None:
        self.state_index.copy_(
            state["state_index"].to(self.state_index.device),
        )
        self.quantized_error.copy_(
            state["quantized_error"].to(self.quantized_error.device),
        )


# =========================================================================
# Section 5: Configuration
# =========================================================================

# RM201
@dataclass
class IamCoreConfig:
    # ------ Population dimensions ------
    coordinate_dim: int = 64
    proto_self_dim: int = 64
    pcc_dim: int = 128
    mpfc_dim: int = 96
    ipl_dim: int = 96
    state_dim: int = 16

    # ------ Davey 2016 Model 27 connectivity ------
    pcc_to_mpfc_endogenous: float = 0.20
    pcc_to_mpfc_modulatory: float = 0.62
    pcc_to_ipl_endogenous: float = 0.33
    pcc_to_ipl_modulatory: float = 0.56
    ipl_to_mpfc_endogenous: float = 0.11
    mpfc_to_pcc_endogenous: float = -0.18
    mpfc_to_pcc_modulatory: float = -0.53
    self_reference_gain: float = 1.0

    proto_to_pcc_init: float = 0.15
    coord_to_pcc_init: float = 0.20

    # ------ Tonic baseline drives ------
    # NOT biological; engineering scalars retuned per dt by an external
    # calibration step. The defaults below assume dt_ms=1.0 with the
    # SSMLIFSubstrate's default threshold of 1.0.
    pcc_baseline_drive: float = 3.0
    mpfc_baseline_drive: float = 3.5
    ipl_baseline_drive: float = 3.5
    proto_baseline_drive: float = 2.5

    # ------ Persistent traces ------
    pcc_trace_tau: float = 0.92
    pcc_predicted_trace_tau: float = 0.90
    condition_trace_tau: float = 0.95
    reentry_trace_tau: float = 200.0  # ticks; Eq. 27 slow self-feed

    # ------ Mixing weights for autobio readout (Eq. 27) ------
    zeta_fast: float = 1.0
    zeta_slow: float = 0.3

    # ------ proto_self placement on PCC drive ------
    proto_self_placement: str = "inside_tanh"

    # ------ Episode store ------
    novelty_threshold: float = 0.18
    store_cooldown_ticks: int = 3
    max_episodes: int = 4096
    autobio_readback_n: int = 3
    autobio_readback_scale: float = 0.4
    eviction_policy: str = "lowest_novelty"

    # ------ Stability gate on episodic writes ------
    enable_stability_gate: bool = True
    stability_threshold: float = 0.10
    stability_min_consecutive: int = 3
    stability_max_wait_ticks: int = 50

    # ------ Conduction delays ------
    ignition_latency_ms: float = 160.0
    inhibitory_return_latency_ms: float = 1.0

    # ------ Theta gate ------
    f_theta_hz: float = 6.0
    enable_theta_gate: bool = True
    theta_gated_rstdp: bool = False

    # ------ R-STDP plasticity ------
    rstdp_lr: float = 0.005
    rstdp_trace_tau: float = 0.92
    rstdp_depression_ratio: float = 0.55
    weight_min: float = -1.0
    weight_max: float = 1.0
    rstdp_modulation_floor: float = 0.05

    # ------ Surrogate alpha ------
    # NOT biological. Training artifact only. Controls gradient flow
    # during backpropagation through the spiking nonlinearity.
    pcc_surrogate_alpha: float = 2.0
    proto_surrogate_alpha: float = 2.0
    mpfc_surrogate_alpha: float = 1.0
    ipl_surrogate_alpha: float = 1.0

    # ------ Substrate parameters shared across populations ------
    v_threshold_base: float = 1.0
    theta_decay: float = 0.95
    theta_increment: float = 0.5

    # ------ Maturity ------
    maturity_mode: str = "omega_variance"
    maturity_horizon_ticks: int = 100_000  # only used when 'tick_count'
    maturity_window: int = 200             # only used when 'omega_variance'
    maturity_var_low: float = 0.005
    maturity_var_high: float = 0.05

    # ------ Phi encoding subspace ------
    encoding_dim: int = 64
    reserved_subspace_size: int = 64

    # ------ Arousal derivation source ------
    arousal_source: str = "r_reentry_norm"

    # ------ Synaptic re-entry register (Equations 29 and 30) ------
    enable_synaptic_reentry_register: bool = True
    synaptic_reentry_n_states: int = 20
    synaptic_reentry_epsilon_quant: float = 0.05

    # ------ Diagnostics ------
    # 0: empty (production). 1: minimal. 2: full (default).
    diagnostics_level: int = 2

    # ------ Numerical ------
    dt_ms: float = 1.0
    rng_seed: int = 1729

    # ------ Upstream kernel hook ------
    m_ca3_provider: Optional[Any] = None

    def ignition_latency_ticks(self) -> int:
        if self.dt_ms <= 0.0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")
        return int(math.ceil(self.ignition_latency_ms / self.dt_ms))

    def inhibitory_return_ticks(self) -> int:
        if self.dt_ms <= 0.0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")
        return max(
            1, int(math.ceil(self.inhibitory_return_latency_ms / self.dt_ms))
        )

    def validate(self) -> None:
        if self.proto_self_placement not in {
            "inside_tanh", "outside_tanh", "modulator",
        }:
            raise ValueError(
                f"proto_self_placement must be one of "
                f"'inside_tanh', 'outside_tanh', 'modulator'; "
                f"got {self.proto_self_placement!r}"
            )
        if self.eviction_policy not in {"lowest_novelty", "fifo"}:
            raise ValueError(
                f"eviction_policy must be 'lowest_novelty' or 'fifo'; "
                f"got {self.eviction_policy!r}"
            )
        if self.maturity_mode not in {"tick_count", "omega_variance"}:
            raise ValueError(
                f"maturity_mode must be 'tick_count' or "
                f"'omega_variance'; got {self.maturity_mode!r}"
            )
        if self.diagnostics_level not in {0, 1, 2}:
            raise ValueError(
                f"diagnostics_level must be 0, 1, or 2; "
                f"got {self.diagnostics_level}"
            )
        if self.maturity_var_low >= self.maturity_var_high:
            raise ValueError(
                f"maturity_var_low must be less than maturity_var_high"
            )
        if self.arousal_source not in {
            "r_reentry_norm", "prediction_error", "max",
        }:
            raise ValueError(
                f"arousal_source must be one of 'r_reentry_norm', "
                f"'prediction_error', 'max'; got {self.arousal_source!r}"
            )


# =========================================================================
# Section 6: Loop output
# =========================================================================

# RM211
@dataclass
class IamLoopOutput:
    pcc_spikes: Tensor
    mpfc_spikes: Tensor
    ipl_spikes: Tensor
    proto_self_spikes: Tensor
    r_reentry: Tensor
    omega: Optional[Tensor]
    omega_binding: Optional[Tensor]
    novelty: Tensor
    prediction_error: Tensor
    doubt: Tensor
    stored_episode: bool
    theta_admitted: bool
    diagnostics: Dict[str, float] = field(default_factory=dict)


# =========================================================================
# Section 7: The I-am core
# =========================================================================

# RM212
class IamCore(nn.Module):
    def __init__(self, cfg: Optional[IamCoreConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or IamCoreConfig()
        self.cfg.validate()

        gen = torch.Generator().manual_seed(self.cfg.rng_seed)

        torch.manual_seed(self.cfg.rng_seed + 100)
        self.proto_self = SSMLIFSubstrate(
            num_neurons=self.cfg.proto_self_dim,
            n_dim=self.cfg.state_dim,
            dt=self.cfg.dt_ms,
            v_threshold_base=self.cfg.v_threshold_base,
            theta_decay=self.cfg.theta_decay,
            theta_increment=self.cfg.theta_increment,
            surrogate_alpha=self.cfg.proto_surrogate_alpha,  # NOT biological. Training artifact.
        )

        torch.manual_seed(self.cfg.rng_seed + 200)
        self.pcc = SSMLIFSubstrate(
            num_neurons=self.cfg.pcc_dim,
            n_dim=self.cfg.state_dim,
            dt=self.cfg.dt_ms,
            v_threshold_base=self.cfg.v_threshold_base,
            theta_decay=self.cfg.theta_decay,
            theta_increment=self.cfg.theta_increment,
            surrogate_alpha=self.cfg.pcc_surrogate_alpha,  # NOT biological. Training artifact.
        )

        torch.manual_seed(self.cfg.rng_seed + 300)
        self.mpfc = SSMLIFSubstrate(
            num_neurons=self.cfg.mpfc_dim,
            n_dim=self.cfg.state_dim,
            dt=self.cfg.dt_ms,
            v_threshold_base=self.cfg.v_threshold_base,
            theta_decay=self.cfg.theta_decay,
            theta_increment=self.cfg.theta_increment,
            surrogate_alpha=self.cfg.mpfc_surrogate_alpha,  # NOT biological. Training artifact.
        )

        torch.manual_seed(self.cfg.rng_seed + 400)
        self.ipl = SSMLIFSubstrate(
            num_neurons=self.cfg.ipl_dim,
            n_dim=self.cfg.state_dim,
            dt=self.cfg.dt_ms,
            v_threshold_base=self.cfg.v_threshold_base,
            theta_decay=self.cfg.theta_decay,
            theta_increment=self.cfg.theta_increment,
            surrogate_alpha=self.cfg.ipl_surrogate_alpha,  # NOT biological. Training artifact.
        )

        self.W_coord_to_pcc = self._init_dense(
            self.cfg.pcc_dim, self.cfg.coordinate_dim,
            mean=self.cfg.coord_to_pcc_init, std=0.04, gen=gen,
        )
        self.W_proto_to_pcc = self._init_dense(
            self.cfg.pcc_dim, self.cfg.proto_self_dim,
            mean=self.cfg.proto_to_pcc_init, std=0.04, gen=gen,
        )
        pcc_mpfc_scale = (
            self.cfg.pcc_to_mpfc_endogenous
            + self.cfg.self_reference_gain
            * self.cfg.pcc_to_mpfc_modulatory
        )
        self.W_pcc_to_mpfc = self._init_dense(
            self.cfg.mpfc_dim, self.cfg.pcc_dim,
            mean=pcc_mpfc_scale, std=0.05, gen=gen,
        )
        pcc_ipl_scale = (
            self.cfg.pcc_to_ipl_endogenous
            + self.cfg.self_reference_gain
            * self.cfg.pcc_to_ipl_modulatory
        )
        self.W_pcc_to_ipl = self._init_dense(
            self.cfg.ipl_dim, self.cfg.pcc_dim,
            mean=pcc_ipl_scale, std=0.05, gen=gen,
        )
        self.W_ipl_to_mpfc = self._init_dense(
            self.cfg.mpfc_dim, self.cfg.ipl_dim,
            mean=self.cfg.ipl_to_mpfc_endogenous, std=0.04, gen=gen,
        )
        mpfc_pcc_scale = (
            self.cfg.mpfc_to_pcc_endogenous
            + self.cfg.self_reference_gain
            * self.cfg.mpfc_to_pcc_modulatory
        )
        self.W_mpfc_to_pcc = self._init_dense(
            self.cfg.pcc_dim, self.cfg.mpfc_dim,
            mean=mpfc_pcc_scale, std=0.04, gen=gen,
        )

        rstdp_cfg = RSTDPConfig(
            lr=self.cfg.rstdp_lr,
            trace_tau=self.cfg.rstdp_trace_tau,
            depression_ratio=self.cfg.rstdp_depression_ratio,
            weight_min=self.cfg.weight_min,
            weight_max=self.cfg.weight_max,
            modulation_floor=self.cfg.rstdp_modulation_floor,
        )
        self.rstdp_coord_to_pcc = RSTDPRule(
            self.cfg.coordinate_dim, self.cfg.pcc_dim, rstdp_cfg,
        )
        self.rstdp_proto_to_pcc = RSTDPRule(
            self.cfg.proto_self_dim, self.cfg.pcc_dim, rstdp_cfg,
        )
        self.rstdp_pcc_to_mpfc = RSTDPRule(
            self.cfg.pcc_dim, self.cfg.mpfc_dim, rstdp_cfg,
        )
        self.rstdp_pcc_to_ipl = RSTDPRule(
            self.cfg.pcc_dim, self.cfg.ipl_dim, rstdp_cfg,
        )
        self.rstdp_ipl_to_mpfc = RSTDPRule(
            self.cfg.ipl_dim, self.cfg.mpfc_dim, rstdp_cfg,
        )
        self.rstdp_mpfc_to_pcc = RSTDPRule(
            self.cfg.mpfc_dim, self.cfg.pcc_dim, rstdp_cfg,
        )

        self.register_buffer(
            "pcc_trace", torch.zeros(self.cfg.pcc_dim),
        )
        self.register_buffer(
            "pcc_predicted_trace", torch.zeros(self.cfg.pcc_dim),
        )
        self.register_buffer(
            "condition_trace", torch.zeros(8, dtype=torch.float32),
        )
        self.register_buffer(
            "reentry_trace",
            torch.zeros(1, self.cfg.pcc_dim),
        )
        self._reentry_decay = math.exp(
            -1.0 / max(self.cfg.reentry_trace_tau, 1.0),
        )

        ipl_reentry_proj = torch.randn(
            self.cfg.pcc_dim, self.cfg.ipl_dim, generator=gen,
        ) * (1.0 / math.sqrt(self.cfg.ipl_dim))
        self.register_buffer("W_ipl_to_reentry", ipl_reentry_proj)

        self._omega_delay_ticks = self.cfg.ignition_latency_ticks()
        self._omega_delay_buffer: Deque[Tensor] = deque(
            maxlen=self._omega_delay_ticks,
        )

        self._inhibitory_delay_ticks = self.cfg.inhibitory_return_ticks()
        self._mpfc_delay_buffer: Deque[Tensor] = deque(
            maxlen=self._inhibitory_delay_ticks,
        )

        self.synaptic_reentry: Optional[SynapticReentryRegister] = None
        if self.cfg.enable_synaptic_reentry_register:
            srcfg = SynapticReentryConfig(
                n_states=self.cfg.synaptic_reentry_n_states,
                weight_min=self.cfg.weight_min,
                weight_max=self.cfg.weight_max,
                epsilon_quant=self.cfg.synaptic_reentry_epsilon_quant,
            )
            self.synaptic_reentry = SynapticReentryRegister(
                cfg=srcfg, n_channels=self.cfg.pcc_dim,
            )

        self.register_buffer(
            "tick", torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "last_store_tick",
            torch.tensor(-(10 ** 9), dtype=torch.long),
        )
        self.register_buffer(
            "doubt", torch.zeros(1, dtype=torch.float32),
        )
        self.register_buffer(
            "previous_pcc_trace",
            torch.zeros(self.cfg.pcc_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "consecutive_stable",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "ticks_since_admit_eligible",
            torch.tensor(0, dtype=torch.long),
        )
        self._theta_phase: float = 0.0

        self._proto_state: Optional[SubstrateState] = None
        self._pcc_state: Optional[SubstrateState] = None
        self._mpfc_state: Optional[SubstrateState] = None
        self._ipl_state: Optional[SubstrateState] = None

        self._episodes: List[EpisodeRecord] = []

        self.phi = Phi(
            encoding_dim=self.cfg.encoding_dim,
            reserved_subspace_size=self.cfg.reserved_subspace_size,
        )

        self._tick_count_for_maturity: int = 0
        self._omega_window: Deque[Tensor] = deque(
            maxlen=self.cfg.maturity_window,
        )

    def _init_dense(
        self,
        post: int,
        pre: int,
        mean: float,
        std: float,
        gen: torch.Generator,
    ) -> nn.Parameter:
        w = torch.randn(post, pre, generator=gen) * std + mean
        return nn.Parameter(w, requires_grad=False)

    def _ensure_substrate_states(
        self, batch: int, device: torch.device,
    ) -> None:
        def _needs_init(s: Optional[SubstrateState]) -> bool:
            if s is None:
                return True
            if s.h.shape[0] != batch:
                return True
            if s.h.device != device:
                return True
            return False

        reinit = False
        if _needs_init(self._proto_state):
            self._proto_state = self.proto_self.init_state(
                batch_size=batch, device=device,
            )
            reinit = True
        if _needs_init(self._pcc_state):
            self._pcc_state = self.pcc.init_state(
                batch_size=batch, device=device,
            )
            reinit = True
        if _needs_init(self._mpfc_state):
            self._mpfc_state = self.mpfc.init_state(
                batch_size=batch, device=device,
            )
            reinit = True
        if _needs_init(self._ipl_state):
            self._ipl_state = self.ipl.init_state(
                batch_size=batch, device=device,
            )
            reinit = True
        if self.reentry_trace.shape[0] != batch:
            self.reentry_trace = torch.zeros(
                batch, self.cfg.pcc_dim, device=device,
            )
            reinit = True
        if reinit:
            self._mpfc_delay_buffer.clear()
            self._omega_delay_buffer.clear()
            self._omega_window.clear()
            if self.synaptic_reentry is not None:
                self.synaptic_reentry.reset()

    def forward(
        self,
        coords: Tensor,
        interoceptive: Optional[Tensor] = None,
        external_neuromod: Optional[Tensor] = None,
        theta_phase: Optional[float] = None,
    ) -> IamLoopOutput:
        if (
            coords.dim() != 2
            or coords.shape[-1] != self.cfg.coordinate_dim
        ):
            raise ValueError(
                f"coords must be (B, {self.cfg.coordinate_dim}); "
                f"got {tuple(coords.shape)}"
            )
        batch = coords.shape[0]
        device = coords.device

        self._ensure_substrate_states(batch, device)

        if interoceptive is None:
            interoceptive = torch.zeros(
                batch, self.cfg.proto_self_dim, device=device,
            )

        prior_pcc_trace = (
            self.pcc_trace.to(device).detach().clone()
        )
        prior_predicted = (
            self.pcc_predicted_trace.to(device).detach().clone()
        )

        # ---- Proto-self forward ----
        proto_current = (
            torch.sigmoid(interoceptive)
            + self.cfg.proto_baseline_drive
        )
        proto_spikes, self._proto_state = self.proto_self(
            proto_current, self._proto_state,
        )

        # ---- Autobiographical readback (Eq. 27 fast term) ----
        autobio_drive = self._compute_autobio_readback(
            batch=batch, device=device,
        )

        # ---- Slow self-feed (Eq. 27 integral term) ----
        slow_self_feed = (
            self.cfg.zeta_slow * self.reentry_trace.to(device)
        )

        # ---- Pop the delayed MPFC inhibition ----
        if (
            len(self._mpfc_delay_buffer)
            >= self._inhibitory_delay_ticks
        ):
            mpfc_inhibition_at_pcc = self._mpfc_delay_buffer[0]
        else:
            mpfc_inhibition_at_pcc = torch.zeros(
                batch, self.cfg.pcc_dim, device=device,
            )

        # ---- PCC forward ----
        coord_current = coords @ self.W_coord_to_pcc.t()
        proto_pcc_current = proto_spikes @ self.W_proto_to_pcc.t()
        cascade_in_tanh = coord_current
        if self.cfg.proto_self_placement == "inside_tanh":
            cascade_in_tanh = cascade_in_tanh + proto_pcc_current
            cascade_after_tanh = torch.tanh(cascade_in_tanh)
        elif self.cfg.proto_self_placement == "outside_tanh":
            outside_tanh = proto_pcc_current
            cascade_after_tanh = torch.tanh(cascade_in_tanh) + outside_tanh
        else:  # 'modulator'
            gate = torch.sigmoid(proto_pcc_current)
            cascade_after_tanh = gate * torch.tanh(cascade_in_tanh)

        pcc_input = (
            cascade_after_tanh
            + self.cfg.zeta_fast * autobio_drive
            + slow_self_feed
            + mpfc_inhibition_at_pcc
            + self.cfg.pcc_baseline_drive
        )
        pcc_spikes, self._pcc_state = self.pcc(
            pcc_input, self._pcc_state,
        )

        # ---- IPL forward (Eq. 24) ----
        ipl_input = (
            pcc_spikes @ self.W_pcc_to_ipl.t()
            + self.cfg.ipl_baseline_drive
        )
        ipl_spikes, self._ipl_state = self.ipl(
            ipl_input, self._ipl_state,
        )

        # ---- MPFC forward (Eq. 22) ----
        mpfc_input = (
            pcc_spikes @ self.W_pcc_to_mpfc.t()
            + ipl_spikes @ self.W_ipl_to_mpfc.t()
            + self.cfg.mpfc_baseline_drive
        )
        mpfc_spikes, self._mpfc_state = self.mpfc(
            mpfc_input, self._mpfc_state,
        )

        # ---- Push new MPFC inhibition into the inhibitory return deque ----
        new_mpfc_drive = (mpfc_spikes @ self.W_mpfc_to_pcc.t()).detach()
        self._mpfc_delay_buffer.append(new_mpfc_drive)

        # ---- Regulated re-entry r_reentry (Eq. 25, three-term form) ----
        r_pcc_inh_in_pcc = -new_mpfc_drive
        ipl_contribution = ipl_spikes @ self.W_ipl_to_reentry.t()
        r_reentry = pcc_spikes + r_pcc_inh_in_pcc + ipl_contribution

        # ---- Update reentry_trace (Eq. 27 integral) ----
        self.reentry_trace = (
            self._reentry_decay * self.reentry_trace.to(device)
            + (1.0 - self._reentry_decay) * r_reentry.detach()
        )

        # ---- Push r_reentry onto the omega-path delay buffer (Eq. 26) ----
        r_reentry_per_batch = r_reentry.detach()
        r_reentry_mean = r_reentry_per_batch.mean(dim=0)
        self._omega_delay_buffer.append(r_reentry_per_batch.clone())

        # ---- Update the synaptic re-entry register (Eq. 29 and 30) ----
        if self.synaptic_reentry is not None:
            self.synaptic_reentry.update(r_reentry_mean)

        # ---- Per-batch novelty and prediction error ----
        pcc_per_batch = pcc_spikes.detach()
        novelty = self._compute_novelty(
            pcc_per_batch, prior_pcc_trace, prior_predicted,
        )
        prediction_error = torch.mean(
            torch.abs(
                pcc_per_batch - prior_predicted.to(device).unsqueeze(0)
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        mean_pcc = pcc_per_batch.mean(dim=0)

        # ---- Doubt: per-batch and population-mean ----
        with torch.no_grad():
            floor = 0.2 * (1.0 - self.cfg.novelty_threshold)
            doubt_per_batch = (
                0.4 * novelty.detach().to(device)
                + 0.4 * prediction_error.detach().to(device)
                + floor
            ).clamp(0.0, 1.0)
            self.doubt.fill_(float(doubt_per_batch.mean().item()))

        # ---- Theta phase: external override or internal advance ----
        dt_s = self.cfg.dt_ms / 1000.0
        if theta_phase is not None:
            self._theta_phase = float(theta_phase) % (2.0 * math.pi)
        else:
            self._theta_phase = (
                self._theta_phase
                + 2.0 * math.pi * self.cfg.f_theta_hz * dt_s
            ) % (2.0 * math.pi)
        if self.cfg.enable_theta_gate:
            theta_admitted = math.sin(self._theta_phase) > 0.0
        else:
            theta_admitted = True

        # ---- Read r_reentry at the omega-path delay tail ----
        batch = int(pcc_per_batch.shape[0])
        if (
            len(self._omega_delay_buffer)
            >= self._omega_delay_ticks
        ):
            buffered = self._omega_delay_buffer[0].to(device)
            if buffered.dim() == 1:
                r_reentry_delayed = (
                    buffered.unsqueeze(0).expand(batch, -1).contiguous()
                )
            elif int(buffered.shape[0]) != batch:
                broadcast_src = buffered.mean(dim=0)
                r_reentry_delayed = (
                    broadcast_src.unsqueeze(0)
                    .expand(batch, -1)
                    .contiguous()
                )
            else:
                r_reentry_delayed = buffered
        else:
            r_reentry_delayed = torch.zeros(
                batch, self.cfg.pcc_dim, device=device,
            )
        if self.synaptic_reentry is not None:
            register_read = self.synaptic_reentry.read().to(device)
            r_reentry_for_omega = (
                register_read.unsqueeze(0).expand(batch, -1).contiguous()
            )
        else:
            r_reentry_for_omega = r_reentry_delayed

        # ---- Compute omega and Phi(omega) on theta-admitted ticks ----
        omega: Optional[Tensor] = None
        omega_binding: Optional[Tensor] = None
        if theta_admitted:
            omega = self._compute_omega(
                r_reentry=r_reentry_for_omega,
                r_reentry_unquantized=r_reentry_delayed,
                novelty=novelty,
                prediction_error=prediction_error,
                external_neuromod=external_neuromod,
                device=device,
            )
            omega_binding = self.phi(omega)
            omega_pop_mean = omega.detach().mean(dim=0)
            self.condition_trace.copy_(
                self.cfg.condition_trace_tau
                * self.condition_trace.to(device)
                + (1.0 - self.cfg.condition_trace_tau)
                * omega_pop_mean,
            )
            self._omega_window.append(omega_pop_mean.cpu().clone())

        # ---- R-STDP on all six matrices ----
        if omega is not None:
            mod_omega = omega.detach().mean(dim=0)
        else:
            mod_omega = self.condition_trace.to(device)
        dopamine = float(mod_omega[0].item())
        norepinephrine = float(mod_omega[1].item())
        nov_scalar = float(mod_omega[4].item())
        arousal_scalar = float(mod_omega[5].item())
        mod_input = 1.0 + dopamine + nov_scalar
        mod_pcc_outward = 1.0 + dopamine + nov_scalar
        mod_to_mpfc = 1.0 + dopamine + arousal_scalar
        mod_inhibitory = 1.0 + norepinephrine + nov_scalar
        if self.cfg.theta_gated_rstdp and not theta_admitted:
            sign = -self.cfg.rstdp_depression_ratio
            mod_input *= sign
            mod_pcc_outward *= sign
            mod_to_mpfc *= sign
            mod_inhibitory *= sign

        coord_pre = coords.detach().mean(dim=0).clamp(0.0, 1.0)
        self.rstdp_coord_to_pcc.update(
            self.W_coord_to_pcc.data,
            coord_pre, mean_pcc, mod_input,
        )
        proto_pre = proto_spikes.detach().mean(dim=0)
        self.rstdp_proto_to_pcc.update(
            self.W_proto_to_pcc.data,
            proto_pre, mean_pcc, mod_input,
        )
        ipl_post = ipl_spikes.detach().mean(dim=0)
        self.rstdp_pcc_to_ipl.update(
            self.W_pcc_to_ipl.data,
            mean_pcc, ipl_post, mod_pcc_outward,
        )
        mpfc_post = mpfc_spikes.detach().mean(dim=0)
        self.rstdp_pcc_to_mpfc.update(
            self.W_pcc_to_mpfc.data,
            mean_pcc, mpfc_post, mod_to_mpfc,
        )
        self.rstdp_ipl_to_mpfc.update(
            self.W_ipl_to_mpfc.data,
            ipl_post, mpfc_post, mod_to_mpfc,
        )
        self.rstdp_mpfc_to_pcc.update(
            self.W_mpfc_to_pcc.data,
            mpfc_post, mean_pcc, mod_inhibitory,
        )

        # ---- Stability gate state update ----
        self._update_stability_gate(mean_pcc)

        # ---- Episodic write gate: one entity, one decision per tick ----
        write_eligible_mask = self._can_store_mask(novelty)
        all_eligible = bool(write_eligible_mask.all().item())
        stored = False
        if (
            theta_admitted
            and omega is not None
            and all_eligible
        ):
            self.ticks_since_admit_eligible.add_(1)
        else:
            self.ticks_since_admit_eligible.fill_(0)
        if (
            theta_admitted
            and omega is not None
            and all_eligible
            and self._stability_admits_write()
        ):
            mean_omega = omega.detach().mean(dim=0)
            mean_mpfc_post = mpfc_post.detach().mean(dim=0)
            mean_coords = coords.detach().mean(dim=0)
            self._store_episode(
                pcc_spikes=mean_pcc,
                mpfc_spikes=mean_mpfc_post,
                prior_pcc_trace=prior_pcc_trace,
                predicted_pcc_trace=prior_predicted,
                operating_condition=mean_omega,
                novelty=float(novelty.detach().mean().item()),
                prediction_error=float(
                    prediction_error.detach().mean().item()
                ),
                source=mean_coords,
                is_real=True,
            )
            stored = True
            self.ticks_since_admit_eligible.fill_(0)

        # ---- Update persistent traces and tick counter ----
        self._update_traces(
            mean_pcc=mean_pcc,
            mean_mpfc_inhibition=new_mpfc_drive.mean(dim=0).detach(),
        )
        self.tick.add_(1)
        self._tick_count_for_maturity += 1

        # ---- Diagnostics by level ----
        diagnostics = self._build_diagnostics(
            theta_admitted=theta_admitted,
            mean_pcc=mean_pcc,
            mpfc_post=mpfc_post,
            ipl_post=ipl_post,
            proto_pre=proto_pre,
            novelty=novelty,
            prediction_error=prediction_error,
            stored=stored,
            batch=batch,
            omega=omega,
        )

        return IamLoopOutput(
            pcc_spikes=pcc_spikes,
            mpfc_spikes=mpfc_spikes,
            ipl_spikes=ipl_spikes,
            proto_self_spikes=proto_spikes,
            r_reentry=r_reentry,
            omega=omega,
            omega_binding=omega_binding,
            novelty=novelty.detach().to(device),
            prediction_error=prediction_error.detach().to(device),
            doubt=doubt_per_batch.detach(),
            stored_episode=stored,
            theta_admitted=theta_admitted,
            diagnostics=diagnostics,
        )

    def _build_diagnostics(
        self,
        theta_admitted: bool,
        mean_pcc: Tensor,
        mpfc_post: Tensor,
        ipl_post: Tensor,
        proto_pre: Tensor,
        novelty: Tensor,
        prediction_error: Tensor,
        stored: bool,
        n_written: int = 0,
        batch: int = 1,
        omega: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        if self.cfg.diagnostics_level == 0:
            return {}
        novelty_mean = float(novelty.detach().mean().item())
        d: Dict[str, float] = {
            "tick": float(self.tick.item()),
            "theta_admitted": float(1.0 if theta_admitted else 0.0),
            "pcc_spike_rate": float(mean_pcc.mean().item()),
            "novelty": novelty_mean,
            "n_episodes": float(len(self._episodes)),
            "stored_episode": float(1.0 if stored else 0.0),
        }
        if self.cfg.diagnostics_level == 1:
            return d
        d.update({
            "theta_phase": self._theta_phase,
            "mpfc_spike_rate": float(mpfc_post.mean().item()),
            "ipl_spike_rate": float(ipl_post.mean().item()),
            "proto_self_spike_rate": float(proto_pre.mean().item()),
            "prediction_error": float(
                prediction_error.detach().mean().item()
            ),
            "omega_delay_filled": float(
                len(self._omega_delay_buffer)
                / max(self._omega_delay_ticks, 1)
            ),
            "inhibitory_delay_filled": float(
                len(self._mpfc_delay_buffer)
                / max(self._inhibitory_delay_ticks, 1)
            ),
            "W_pcc_to_mpfc_norm": float(
                self.W_pcc_to_mpfc.norm().item()
            ),
            "W_mpfc_to_pcc_norm": float(
                self.W_mpfc_to_pcc.norm().item()
            ),
            "maturity": float(self._maturity()),
        })
        d["batch"] = float(batch)
        d["n_written"] = float(n_written)
        if omega is not None:
            d["omega_mean"] = float(omega.detach().mean().item())
            d["omega_std"] = float(omega.detach().std().item())
        else:
            d["omega_mean"] = 0.0
            d["omega_std"] = 0.0
        return d

    def _compute_autobio_readback(
        self, batch: int, device: torch.device,
    ) -> Tensor:
        if self.cfg.m_ca3_provider is not None:
            try:
                m_ca3 = self.cfg.m_ca3_provider()
            except Exception:
                m_ca3 = None
            if m_ca3 is not None:
                m_ca3 = m_ca3.detach().to(device).float()
                if m_ca3.dim() == 1:
                    if m_ca3.numel() != self.cfg.pcc_dim:
                        raise ValueError(
                            f"m_ca3_provider returned shape "
                            f"{tuple(m_ca3.shape)}; expected "
                            f"({self.cfg.pcc_dim},) or "
                            f"(B, {self.cfg.pcc_dim})"
                        )
                    m_ca3 = m_ca3.unsqueeze(0).expand(batch, -1)
                elif m_ca3.dim() == 2:
                    if m_ca3.shape[-1] != self.cfg.pcc_dim:
                        raise ValueError(
                            f"m_ca3_provider returned shape "
                            f"{tuple(m_ca3.shape)}; expected last "
                            f"dim {self.cfg.pcc_dim}"
                        )
                else:
                    raise ValueError(
                        f"m_ca3_provider must return 1d or 2d, got "
                        f"{m_ca3.dim()}d"
                    )
                return (
                    self.cfg.autobio_readback_scale * m_ca3
                ).contiguous()
        if not self._episodes:
            return torch.zeros(
                batch, self.cfg.pcc_dim, device=device,
            )
        query = self.condition_trace.to(device).detach()
        drawn = RetrievalDraw.draw_past(
            self._episodes,
            query_omega=query,
            n=self.cfg.autobio_readback_n,
        )
        if not drawn:
            return torch.zeros(
                batch, self.cfg.pcc_dim, device=device,
            )
        stacked = torch.stack(
            [ep.pcc_spikes.to(device) for ep in drawn], dim=0,
        )
        mean_pcc = stacked.mean(dim=0)
        return (
            self.cfg.autobio_readback_scale * mean_pcc
        ).unsqueeze(0).expand(batch, -1).contiguous()

    def _compute_omega(
        self,
        r_reentry: Tensor,
        r_reentry_unquantized: Tensor,
        novelty: Tensor,
        prediction_error: Tensor,
        external_neuromod: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        batch = int(r_reentry.shape[0])
        if external_neuromod is None:
            da = torch.zeros(batch, device=device)
            ne = torch.zeros(batch, device=device)
            ach = torch.full((batch,), 0.5, device=device)
            five_ht = torch.zeros(batch, device=device)
        else:
            nm = external_neuromod.detach().to(device).float()
            if nm.dim() != 2 or nm.shape[-1] != 4:
                raise ValueError(
                    f"external_neuromod must be (B, 4), got "
                    f"{tuple(nm.shape)}"
                )
            if int(nm.shape[0]) != batch:
                raise ValueError(
                    f"external_neuromod batch dim {nm.shape[0]} does "
                    f"not match r_reentry batch dim {batch}"
                )
            da = nm[:, 0].clamp(-1.0, 1.0)
            ne = nm[:, 1].clamp(0.0, 1.0)
            ach = nm[:, 2].clamp(0.0, 1.0)
            five_ht = nm[:, 3].clamp(-1.0, 1.0)

        r_norm_per_batch = r_reentry.detach().norm(dim=-1)
        arousal_dynamics = torch.tanh(
            r_norm_per_batch / math.sqrt(self.cfg.pcc_dim)
        )
        if self.cfg.arousal_source == "r_reentry_norm":
            arousal = arousal_dynamics
        elif self.cfg.arousal_source == "prediction_error":
            arousal = prediction_error.detach().to(device).float()
        else:  # 'max'
            pe_per_batch = prediction_error.detach().to(device).float()
            arousal = torch.maximum(arousal_dynamics, pe_per_batch)
        valence = torch.tanh(
            r_reentry_unquantized.detach().mean(dim=-1)
        )
        maturity_scalar = self._maturity()
        maturity = torch.full(
            (batch,), maturity_scalar, device=device,
        )

        omega = torch.stack(
            [
                da,
                ne,
                ach,
                five_ht,
                novelty.to(device).clamp(0.0, 1.0),
                arousal.to(device).clamp(0.0, 1.0),
                valence.to(device),
                maturity.clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        return omega

    def _compute_novelty(
        self,
        pcc_per_batch: Tensor,
        prior_trace: Tensor,
        prior_predicted: Tensor,
    ) -> Tensor:
        device = pcc_per_batch.device
        prior_t = prior_trace.to(device).unsqueeze(0)
        prior_p = prior_predicted.to(device).unsqueeze(0)
        trace_mismatch = torch.mean(
            torch.abs(pcc_per_batch - prior_t), dim=-1,
        )
        pred_mismatch = torch.mean(
            torch.abs(pcc_per_batch - prior_p), dim=-1,
        )
        return (
            0.5 * trace_mismatch + 0.5 * pred_mismatch
        ).clamp(0.0, 1.0)

    def _can_store_mask(self, novelty: Tensor) -> Tensor:
        cooldown_ok = int(
            self.tick.item() - self.last_store_tick.item()
        ) >= self.cfg.store_cooldown_ticks
        if not cooldown_ok:
            return torch.zeros_like(novelty, dtype=torch.bool)
        return novelty > self.cfg.novelty_threshold

    def _update_stability_gate(self, mean_pcc: Tensor) -> None:
        device = mean_pcc.device
        prev = self.previous_pcc_trace.to(device)
        delta = float(torch.norm(mean_pcc - prev).item())
        if delta < self.cfg.stability_threshold:
            self.consecutive_stable.add_(1)
        else:
            self.consecutive_stable.fill_(0)
        with torch.no_grad():
            self.previous_pcc_trace.copy_(mean_pcc.detach())

    def _stability_admits_write(self) -> bool:
        if not self.cfg.enable_stability_gate:
            return True
        stable = (
            int(self.consecutive_stable.item())
            >= self.cfg.stability_min_consecutive
        )
        timed_out = (
            int(self.ticks_since_admit_eligible.item())
            >= self.cfg.stability_max_wait_ticks
        )
        return stable or timed_out

    def _store_episode(
        self,
        pcc_spikes: Tensor,
        mpfc_spikes: Tensor,
        prior_pcc_trace: Tensor,
        predicted_pcc_trace: Tensor,
        operating_condition: Tensor,
        novelty: float,
        prediction_error: float,
        source: Tensor,
        is_real: bool = True,
    ) -> None:
        source_bytes = (
            source.detach().cpu().contiguous().numpy().tobytes()
        )
        record = EpisodeRecord(
            tick=int(self.tick.item()),
            pcc_spikes=pcc_spikes.detach().cpu().clone(),
            mpfc_spikes=mpfc_spikes.detach().cpu().clone(),
            prior_pcc_trace=prior_pcc_trace.detach().cpu().clone(),
            predicted_pcc_trace=(
                predicted_pcc_trace.detach().cpu().clone()
            ),
            operating_condition=(
                operating_condition.detach().cpu().clone()
            ),
            novelty=float(novelty),
            prediction_error=float(prediction_error),
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            structural_hash=self.structural_hash(),
            tau_proj=0,
            is_real=bool(is_real),
        )
        WriteGate.write(
            self._episodes, record, self.cfg.max_episodes,
            eviction=self.cfg.eviction_policy,
        )
        self.last_store_tick.copy_(self.tick)

    def _update_traces(
        self,
        mean_pcc: Tensor,
        mean_mpfc_inhibition: Tensor,
    ) -> None:
        device = self.pcc_trace.device
        pcc = mean_pcc.detach().to(device)
        self.pcc_trace.mul_(self.cfg.pcc_trace_tau).add_(
            (1.0 - self.cfg.pcc_trace_tau) * pcc
        )
        carry_mean = mean_mpfc_inhibition.to(device)
        predicted = (
            pcc - 0.3 * carry_mean.clamp(min=0.0)
        ).clamp(0.0, 1.0)
        self.pcc_predicted_trace.mul_(
            self.cfg.pcc_predicted_trace_tau
        ).add_(
            (1.0 - self.cfg.pcc_predicted_trace_tau) * predicted
        )

    def _maturity(self) -> float:
        if self.cfg.maturity_mode == "tick_count":
            if self.cfg.maturity_horizon_ticks <= 0:
                return 1.0
            m = (
                self._tick_count_for_maturity
                / self.cfg.maturity_horizon_ticks
            )
            return min(1.0, max(0.0, m))
        if len(self._omega_window) < 2:
            return 0.0
        stacked = torch.stack(list(self._omega_window), dim=0)
        variance = stacked.var(dim=0, unbiased=False)
        var_trace = float(variance.sum().item())
        lo = self.cfg.maturity_var_low
        hi = self.cfg.maturity_var_high
        if var_trace <= lo:
            return 1.0
        if var_trace >= hi:
            return 0.0
        return float((hi - var_trace) / (hi - lo))

    def reset(self, level: str = "turn") -> None:
        if level not in {"turn", "episode"}:
            raise ValueError(
                f"level must be 'turn' or 'episode', got {level!r}"
            )
        self._proto_state = None
        self._pcc_state = None
        self._mpfc_state = None
        self._ipl_state = None
        self._mpfc_delay_buffer.clear()
        self._omega_delay_buffer.clear()
        self._omega_window.clear()
        if level == "episode":
            self.pcc_trace.zero_()
            self.pcc_predicted_trace.zero_()
            self.condition_trace.zero_()
            with torch.no_grad():
                self.reentry_trace.zero_()
            for r in [
                self.rstdp_coord_to_pcc, self.rstdp_proto_to_pcc,
                self.rstdp_pcc_to_mpfc, self.rstdp_pcc_to_ipl,
                self.rstdp_ipl_to_mpfc, self.rstdp_mpfc_to_pcc,
            ]:
                r.reset()
            if self.synaptic_reentry is not None:
                self.synaptic_reentry.reset()
            self.last_store_tick.fill_(-(10 ** 9))
            self._theta_phase = 0.0

    def clear_episodes(self) -> None:
        self._episodes.clear()

    def episode_count(self) -> int:
        return len(self._episodes)

    def latest_episode(self) -> Optional[EpisodeRecord]:
        return self._episodes[-1] if self._episodes else None

    def current_doubt(self) -> float:
        return float(self.doubt.item())

    def is_doubtful(self, threshold: float = 0.7) -> bool:
        return self.current_doubt() >= threshold

    def get_loop_state(self) -> Dict[str, Any]:
        return {
            "format_version": LOOP_STATE_FORMAT_VERSION,
            "config": self.cfg.__dict__.copy(),
            "tick": int(self.tick.item()),
            "last_store_tick": int(self.last_store_tick.item()),
            "theta_phase": float(self._theta_phase),
            "tick_count_for_maturity": int(
                self._tick_count_for_maturity
            ),
            "doubt": float(self.doubt.item()),
            "previous_pcc_trace": (
                self.previous_pcc_trace.detach().cpu().clone()
            ),
            "consecutive_stable": int(self.consecutive_stable.item()),
            "ticks_since_admit_eligible": int(
                self.ticks_since_admit_eligible.item()
            ),
            "pcc_trace": self.pcc_trace.detach().cpu().clone(),
            "pcc_predicted_trace": (
                self.pcc_predicted_trace.detach().cpu().clone()
            ),
            "condition_trace": (
                self.condition_trace.detach().cpu().clone()
            ),
            "reentry_trace": (
                self.reentry_trace.detach().cpu().clone()
            ),
            "W_coord_to_pcc": (
                self.W_coord_to_pcc.detach().cpu().clone()
            ),
            "W_proto_to_pcc": (
                self.W_proto_to_pcc.detach().cpu().clone()
            ),
            "W_pcc_to_mpfc": (
                self.W_pcc_to_mpfc.detach().cpu().clone()
            ),
            "W_pcc_to_ipl": (
                self.W_pcc_to_ipl.detach().cpu().clone()
            ),
            "W_ipl_to_mpfc": (
                self.W_ipl_to_mpfc.detach().cpu().clone()
            ),
            "W_mpfc_to_pcc": (
                self.W_mpfc_to_pcc.detach().cpu().clone()
            ),
            "W_ipl_to_reentry": (
                self.W_ipl_to_reentry.detach().cpu().clone()
            ),
            "proto_substrate_state": (
                self.proto_self.state_dict_full(self._proto_state)
                if self._proto_state is not None else None
            ),
            "pcc_substrate_state": (
                self.pcc.state_dict_full(self._pcc_state)
                if self._pcc_state is not None else None
            ),
            "mpfc_substrate_state": (
                self.mpfc.state_dict_full(self._mpfc_state)
                if self._mpfc_state is not None else None
            ),
            "ipl_substrate_state": (
                self.ipl.state_dict_full(self._ipl_state)
                if self._ipl_state is not None else None
            ),
            "rstdp_coord_to_pcc": (
                self.rstdp_coord_to_pcc.serialize()
            ),
            "rstdp_proto_to_pcc": (
                self.rstdp_proto_to_pcc.serialize()
            ),
            "rstdp_pcc_to_mpfc": (
                self.rstdp_pcc_to_mpfc.serialize()
            ),
            "rstdp_pcc_to_ipl": (
                self.rstdp_pcc_to_ipl.serialize()
            ),
            "rstdp_ipl_to_mpfc": (
                self.rstdp_ipl_to_mpfc.serialize()
            ),
            "rstdp_mpfc_to_pcc": (
                self.rstdp_mpfc_to_pcc.serialize()
            ),
            "mpfc_delay_buffer": [
                t.detach().cpu().clone()
                for t in self._mpfc_delay_buffer
            ],
            "omega_delay_buffer": [
                t.detach().cpu().clone()
                for t in self._omega_delay_buffer
            ],
            "omega_window": [
                t.detach().cpu().clone()
                for t in self._omega_window
            ],
            "inhibitory_delay_ticks": int(
                self._inhibitory_delay_ticks
            ),
            "omega_delay_ticks": int(self._omega_delay_ticks),
            "synaptic_reentry": (
                self.synaptic_reentry.serialize()
                if self.synaptic_reentry is not None else None
            ),
            "episodes": [ep.to_state() for ep in self._episodes],
            "structural_hash": self.structural_hash(),
        }

    def verify_structural_hash(
        self, saved_state: Dict[str, Any],
    ) -> bool:
        saved_hash = saved_state.get("structural_hash", "")
        if not saved_hash:
            return False
        return saved_hash == self.structural_hash()

    def load_loop_state(
        self,
        state: Dict[str, Any],
        enforce_structural_hash: bool = True,
    ) -> None:
        saved_version = int(state.get(
            "format_version", LOOP_STATE_FORMAT_VERSION,
        ))
        if saved_version > LOOP_STATE_FORMAT_VERSION:
            raise ValueError(
                f"saved loop state format v{saved_version} is newer "
                f"than this code understands "
                f"(v{LOOP_STATE_FORMAT_VERSION}); upgrade required"
            )
        if saved_version < LOOP_STATE_FORMAT_VERSION:
            state = _migrate_loop_state(
                state, saved_version, LOOP_STATE_FORMAT_VERSION,
            )

        device = self.pcc_trace.device
        with torch.no_grad():
            self.W_coord_to_pcc.copy_(
                state["W_coord_to_pcc"].to(device),
            )
            self.W_proto_to_pcc.copy_(
                state["W_proto_to_pcc"].to(device),
            )
            self.W_pcc_to_mpfc.copy_(
                state["W_pcc_to_mpfc"].to(device),
            )
            self.W_pcc_to_ipl.copy_(
                state["W_pcc_to_ipl"].to(device),
            )
            self.W_ipl_to_mpfc.copy_(
                state["W_ipl_to_mpfc"].to(device),
            )
            self.W_mpfc_to_pcc.copy_(
                state["W_mpfc_to_pcc"].to(device),
            )
            if "W_ipl_to_reentry" in state:
                self.W_ipl_to_reentry.copy_(
                    state["W_ipl_to_reentry"].to(device),
                )

        if enforce_structural_hash:
            saved_hash = state.get("structural_hash", "")
            live_hash = self.structural_hash()
            if saved_hash and saved_hash != live_hash:
                raise ValueError(
                    f"Structural-hash mismatch on load: "
                    f"saved={saved_hash}, live={live_hash}. The "
                    f"substrate that just woke up is not bitwise "
                    f"identical to the substrate that was saved."
                )

        self.tick.fill_(int(state.get("tick", 0)))
        self.last_store_tick.fill_(
            int(state.get("last_store_tick", -(10 ** 9))),
        )
        self._theta_phase = float(state.get("theta_phase", 0.0))
        self._tick_count_for_maturity = int(
            state.get("tick_count_for_maturity", 0),
        )
        if "doubt" in state:
            self.doubt.fill_(float(state["doubt"]))
        if "previous_pcc_trace" in state:
            self.previous_pcc_trace.copy_(
                state["previous_pcc_trace"].to(device),
            )
        if "consecutive_stable" in state:
            self.consecutive_stable.fill_(
                int(state["consecutive_stable"]),
            )
        if "ticks_since_admit_eligible" in state:
            self.ticks_since_admit_eligible.fill_(
                int(state["ticks_since_admit_eligible"]),
            )
        self.pcc_trace.copy_(state["pcc_trace"].to(device))
        self.pcc_predicted_trace.copy_(
            state["pcc_predicted_trace"].to(device),
        )
        self.condition_trace.copy_(
            state["condition_trace"].to(device),
        )
        self.reentry_trace = state["reentry_trace"].to(device)

        proto_sub = state.get("proto_substrate_state")
        if proto_sub is not None:
            self._proto_state = self.proto_self.load_state_full(
                proto_sub, device=device,
            )
        pcc_sub = state.get("pcc_substrate_state")
        if pcc_sub is not None:
            self._pcc_state = self.pcc.load_state_full(
                pcc_sub, device=device,
            )
        mpfc_sub = state.get("mpfc_substrate_state")
        if mpfc_sub is not None:
            self._mpfc_state = self.mpfc.load_state_full(
                mpfc_sub, device=device,
            )
        ipl_sub = state.get("ipl_substrate_state")
        if ipl_sub is not None:
            self._ipl_state = self.ipl.load_state_full(
                ipl_sub, device=device,
            )

        self.rstdp_coord_to_pcc.restore(state["rstdp_coord_to_pcc"])
        self.rstdp_proto_to_pcc.restore(state["rstdp_proto_to_pcc"])
        self.rstdp_pcc_to_mpfc.restore(state["rstdp_pcc_to_mpfc"])
        self.rstdp_pcc_to_ipl.restore(state["rstdp_pcc_to_ipl"])
        self.rstdp_ipl_to_mpfc.restore(state["rstdp_ipl_to_mpfc"])
        self.rstdp_mpfc_to_pcc.restore(state["rstdp_mpfc_to_pcc"])

        self._mpfc_delay_buffer.clear()
        for t in state.get("mpfc_delay_buffer", []):
            self._mpfc_delay_buffer.append(t.to(device))
        self._omega_delay_buffer.clear()
        for t in state.get("omega_delay_buffer", []):
            self._omega_delay_buffer.append(t.to(device))

        self._omega_window.clear()
        for t in state.get("omega_window", []):
            self._omega_window.append(t.detach().cpu().clone())

        sr_state = state.get("synaptic_reentry")
        if sr_state is not None and self.synaptic_reentry is not None:
            self.synaptic_reentry.restore(sr_state)

        self._episodes = [
            EpisodeRecord.from_state(item)
            for item in state.get("episodes", [])
        ]

    def structural_hash(self) -> str:
        h = hashlib.sha256()
        for w in [
            self.W_coord_to_pcc, self.W_proto_to_pcc,
            self.W_pcc_to_mpfc, self.W_pcc_to_ipl,
            self.W_ipl_to_mpfc, self.W_mpfc_to_pcc,
            self.W_ipl_to_reentry,
        ]:
            h.update(
                w.detach().cpu().contiguous().numpy().tobytes()
            )
        for pop in [
            self.proto_self, self.pcc, self.mpfc, self.ipl,
        ]:
            for name in [
                "a_matrix", "b_vector", "c_vector",
                "ltc_gate_weight", "ltc_gate_bias",
            ]:
                t = getattr(pop, name)
                h.update(
                    t.detach().cpu().contiguous().numpy().tobytes()
                )
        return h.hexdigest()
