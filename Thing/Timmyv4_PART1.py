# RM numbers reference "The code classes explained" in README.md

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from math import ceil

import torch
from torch import Tensor, nn

_LOG = logging.getLogger(__name__)


# RM105
def hippo_legs_init(
    n: int, dtype: torch.dtype = torch.float32
) -> tuple[Tensor, Tensor]:
    indices = torch.arange(n, dtype=dtype)
    rows = indices.unsqueeze(1)
    cols = indices.unsqueeze(0)
    factor = torch.sqrt((2.0 * rows + 1.0) * (2.0 * cols + 1.0))
    a_continuous = torch.where(rows > cols, -factor, torch.zeros_like(factor))
    a_continuous = a_continuous + torch.diag(-(indices + 1.0))
    b_continuous = torch.sqrt(2.0 * indices + 1.0).to(dtype)
    return a_continuous, b_continuous


# RM105
def discretize_zoh(
    a_continuous: Tensor, b_continuous: Tensor, dt: float
) -> tuple[Tensor, Tensor]:
    n = a_continuous.shape[0]
    a_d = torch.linalg.matrix_exp(a_continuous * dt)
    eye = torch.eye(n, dtype=a_continuous.dtype, device=a_continuous.device)
    b_d = torch.linalg.solve(
        a_continuous, (a_d - eye) @ b_continuous.unsqueeze(-1)
    ).squeeze(-1)
    return a_d, b_d


# RM101
@dataclass
class SubstrateState:
    h: Tensor           # (batch, num_neurons, n_dim)
    v: Tensor           # (batch, num_neurons)
    theta: Tensor       # (batch, num_neurons)
    last_spike: Tensor  # (batch, num_neurons)

    def detach(self) -> SubstrateState:
        return SubstrateState(
            h=self.h.detach(),
            v=self.v.detach(),
            theta=self.theta.detach(),
            last_spike=self.last_spike.detach(),
        )

    def to(self, device: torch.device | str) -> SubstrateState:
        return SubstrateState(
            h=self.h.to(device),
            v=self.v.to(device),
            theta=self.theta.to(device),
            last_spike=self.last_spike.to(device),
        )


# RM102
class HeavisideWithSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_theta: Tensor, alpha: float) -> Tensor:
        ctx.save_for_backward(v_minus_theta)
        ctx.alpha = alpha
        return (v_minus_theta >= 0).to(v_minus_theta.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        (v_minus_theta,) = ctx.saved_tensors
        alpha = ctx.alpha
        denom = 1.0 + alpha * v_minus_theta.abs()
        grad_input = grad_output * alpha / (denom * denom)
        return grad_input, None


# RM102
def spike_fn(v_minus_theta: Tensor, alpha: float) -> Tensor:
    return HeavisideWithSurrogate.apply(v_minus_theta, alpha)


# RM103
class SpikeMixer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, spikes: Tensor) -> Tensor:
        return torch.nn.functional.gelu(self.linear(spikes))


# RM104
class SSMLIFSubstrate(nn.Module):
    def __init__(
        self,
        num_neurons: int,
        n_dim: int,
        dt: float = 1.0,
        v_rest: float = 0.0,
        v_threshold_base: float = 1.0,
        theta_decay: float = 0.95,
        theta_increment: float = 0.5,
        surrogate_alpha: float = 2.0,  # NOT biological. Training artifact.
        share_state_matrix: bool = True,
    ) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.n_dim = n_dim
        self.dt = dt
        self.v_rest = v_rest
        self.v_threshold_base = v_threshold_base
        self.theta_decay = theta_decay
        self.theta_increment = theta_increment
        self.surrogate_alpha = surrogate_alpha
        self.share_state_matrix = share_state_matrix

        a_continuous, b_continuous = hippo_legs_init(n_dim)
        a_d, b_d = discretize_zoh(a_continuous, b_continuous, dt)

        # A is a frozen developmental prior. Does not train.
        if share_state_matrix:
            self.register_buffer("a_matrix", a_d.clone())
        else:
            self.register_buffer(
                "a_matrix",
                a_d.clone().unsqueeze(0).expand(num_neurons, -1, -1).contiguous(),
            )

        if share_state_matrix:
            self.b_vector = nn.Parameter(b_d.clone())
        else:
            self.b_vector = nn.Parameter(
                b_d.clone().unsqueeze(0).expand(num_neurons, -1).contiguous()
            )

        c_init = torch.empty(num_neurons, n_dim)
        nn.init.normal_(c_init, mean=0.0, std=1.0 / math.sqrt(n_dim))
        self.c_vector = nn.Parameter(c_init)

        ltc_init = torch.empty(num_neurons, 2)
        nn.init.normal_(ltc_init, mean=0.0, std=0.1)
        self.ltc_gate_weight = nn.Parameter(ltc_init)
        self.ltc_gate_bias = nn.Parameter(torch.zeros(num_neurons))

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> SubstrateState:
        device = device if device is not None else self.a_matrix.device
        return SubstrateState(
            h=torch.zeros(batch_size, self.num_neurons, self.n_dim, device=device),
            v=torch.full((batch_size, self.num_neurons), self.v_rest, device=device),
            theta=torch.zeros(batch_size, self.num_neurons, device=device),
            last_spike=torch.zeros(batch_size, self.num_neurons, device=device),
        )

    def forward(
        self, x: Tensor, state: SubstrateState
    ) -> tuple[Tensor, SubstrateState]:
        if x.dim() != 2 or x.shape[1] != self.num_neurons:
            raise ValueError(
                f"x must be (batch, {self.num_neurons}); got {tuple(x.shape)}"
            )

        h_prev, v_prev, theta_prev = state.h, state.v, state.theta

        # h[t] = A h[t-1] + B x[t]
        if self.share_state_matrix:
            h_new = torch.einsum("bnd,ed->bne", h_prev, self.a_matrix)
            h_new = h_new + x.unsqueeze(-1) * self.b_vector
        else:
            h_new = torch.einsum("bnd,ned->bne", h_prev, self.a_matrix)
            h_new = h_new + x.unsqueeze(-1) * self.b_vector.unsqueeze(0)

        # drive[t] = c . h[t]
        membrane_drive = torch.einsum("bnd,nd->bn", h_new, self.c_vector)

        # gate[t] = sigmoid(w_x * x + w_v * v_prev + b)
        gate = torch.sigmoid(
            x * self.ltc_gate_weight[:, 0]
            + v_prev * self.ltc_gate_weight[:, 1]
            + self.ltc_gate_bias
        )

        # v[t] = (1 - gate) * v_prev + gate * (drive + v_rest)
        v_new = (1.0 - gate) * v_prev + gate * (membrane_drive + self.v_rest)

        effective_threshold = self.v_threshold_base + theta_prev
        spikes = spike_fn(v_new - effective_threshold, self.surrogate_alpha)

        # subtractive reset
        v_after_reset = v_new - spikes * effective_threshold

        # theta[t] = decay * theta[t-1] + increment * s[t]
        theta_new = self.theta_decay * theta_prev + self.theta_increment * spikes

        return spikes, SubstrateState(
            h=h_new, v=v_after_reset, theta=theta_new, last_spike=spikes
        )

    def state_dict_full(self, state: SubstrateState) -> dict[str, Tensor]:
        return {
            "h": state.h.detach().cpu(),
            "v": state.v.detach().cpu(),
            "theta": state.theta.detach().cpu(),
            "last_spike": state.last_spike.detach().cpu(),
        }

    def load_state_full(
        self,
        state_dict: dict[str, Tensor],
        device: torch.device | str | None = None,
    ) -> SubstrateState:
        device = device if device is not None else self.a_matrix.device
        return SubstrateState(
            h=state_dict["h"].to(device),
            v=state_dict["v"].to(device),
            theta=state_dict["theta"].to(device),
            last_spike=state_dict["last_spike"].to(device),
        )


# RM110
@dataclass
class LemmaStratumConfig:
    manifold_dim: int = 32
    role_dim: int = 16
    concept_dim: int = 64
    iam_pcc_dim: int = 128
    scaffold_size: int = 10
    reservoir_size: int = 4096
    dt_ms: float = 1.0
    rhythm3_period_ms: float = 250.0
    reentry_delay_ms: float = 160.0
    sleep_consolidation_steps: int = 500
    alpha_stiffness: float = 1.0
    beta_exponent: float = 1.5
    gamma_damping: float = 0.4
    mass_baseline: float = 1.0
    mass_confirmation_scale: float = 5.0
    stdp_lr: float = 1e-3
    stdp_tau_pre_ms: float = 20.0
    stdp_tau_post_ms: float = 20.0
    weight_min: float = -1.0
    weight_max: float = 1.0
    depression_ratio: float = 0.5
    hazard_dt: float = 1.0
    flip_weight: float = 2.0
    interference_kappa: float = 0.5
    band_beta2_hz: float = 22.0
    band_low_gamma_hz: float = 35.0
    band_mid_gamma_hz: float = 57.0
    eta_reentry_ratio: float = 0.05
    cholinergic_pause_threshold: float = 0.3
    dopamine_burst_threshold: float = 0.5
    local_depol_threshold: float = 0.4
    lc_period_seconds: float = 70.0
    lc_ca_exponent: int = 5
    doubt_window_ticks: int = 20
    doubt_threshold: float = 0.4
    maturity_saturation_count: int = 500
    provisional_decay_per_tick: float = 1e-3
    provisional_activation_floor: float = 1e-3
    confirmation_threshold: float = 1.0
    reclaim_threshold: float = 0.05
    acquisition_doubt_threshold: float = 0.6
    ablate_role_gate: bool = False
    ablate_three_factor: bool = False
    ablate_lc_flipflop: bool = False
    ablate_tau_proj_guard: bool = False
    ablate_reentry: bool = False
    ablate_sleep: bool = False
    ablate_scaffold_lock: bool = False
    ablate_attention_flip: bool = False
    disorder_axis2_collapse: float = 0.0
    rng_seed: int = 42

    @property
    def population_size(self) -> int:
        return self.scaffold_size + self.reservoir_size

    @property
    def reentry_delay_ticks(self) -> int:
        return ceil(self.reentry_delay_ms / self.dt_ms)

    @property
    def lc_period_ticks(self) -> int:
        return int(self.lc_period_seconds * 1000.0 / self.dt_ms)


# RM111
@dataclass
class PhonologicalConfig:
    code_dim: int = 64
    phoneme_dim: int = 32
    phoneme_inventory_size: int = 49
    max_morphemes: int = 4
    percept_delay_ms: float = 160.0
    tau_decay_ticks: int = 50
    decoder_population_size: int = 64
    decoder_n_hidden: int = 16
    decoder_surrogate_alpha: float = 0.8  # NOT biological. Training artifact.
    decoder_ticks_per_segment: int = 25
    decoder_max_segments: int = 20
    rstdp_lr_phon: float = 1e-3
    weight_min_phon: float = -1.0
    weight_max_phon: float = 1.0
    ablate_wernickes_aphasia: bool = False

    @property
    def percept_delay_ticks(self) -> int:
        return ceil(self.percept_delay_ms / 1.0)


PHONEME_INVENTORY: tuple = (
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L",
    "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V",
    "W", "Y", "Z", "ZH",
    "AX", "AXR", "IX", "DX", "TS",
    "<start>", "<end>", "<syl>", "<morph>", "<pad>",
)

assert len(PHONEME_INVENTORY) == 49

PHONEME_START_IDX: int = 44
PHONEME_END_IDX: int = 45
PHONEME_SYL_BOUNDARY_IDX: int = 46
PHONEME_MORPH_BOUNDARY_IDX: int = 47
PHONEME_PAD_IDX: int = 48


# RM112
@dataclass
class BrocasConfig:
    B44_size: int = 1024
    B45_size: int = 1024
    n_hidden: int = 16
    B44_alpha: float = 0.8          # NOT biological. Training artifact.
    B44_rho_theta: float = 0.92
    B44_Delta_theta: float = 0.6
    B45_alpha: float = 2.0          # NOT biological. Training artifact.
    B45_rho_theta: float = 0.95
    B45_Delta_theta: float = 0.5
    B44_receptor_DA: float = 1.0
    B44_receptor_NE: float = 1.0
    B44_receptor_ACh: float = 1.0
    B44_receptor_5HT: float = 1.0
    B44_receptor_GABA_A: float = 1.3
    B45_receptor_DA: float = 1.4
    B45_receptor_NE: float = 1.0
    B45_receptor_ACh: float = 1.0
    B45_receptor_5HT: float = 1.3
    B45_receptor_GABA_A: float = 1.0
    num_frame_templates: int = 16
    max_buffer_slots: int = 7
    ticks_per_segment: int = 25
    completion_threshold: float = 0.5
    emission_ring_buffer_size: int = 3
    phoneme_inventory_size: int = 49
    ablate_BA44: bool = False
    ablate_BA45: bool = False


# RM113
@dataclass
class ArcuateConfig:
    tau_arc_ms: float = 7.5
    near_identity_perturbation: float = 0.01
    ablate_arcuate: bool = False


# RM114
@dataclass
class ProductionConfig:
    cascade_to_lemma_select_ticks: int = 200
    max_segments_per_word: int = 20


# RM115
@dataclass
class OrthographyConfig:
    n_letters: int = 26
    n_positions: int = 16
    n_hidden: int = 16
    dt: float = 1.0
    v_threshold_base: float = 1.0
    theta_decay: float = 0.92
    theta_increment: float = 0.6
    surrogate_alpha: float = 0.8    # NOT biological. Training artifact.
    encode_ticks: int = 200
    ablate_lexical_route: bool = False
    ablate_sublexical_route: bool = False

    @property
    def population_size(self) -> int:
        return self.n_letters * self.n_positions


# RM116
@dataclass
class CascadeConfig:
    population_size: int = 2048
    n_hidden: int = 32
    dt: float = 1.0
    v_threshold_base: float = 1.0
    theta_decay: float = 0.95
    theta_increment: float = 0.5
    surrogate_alpha: float = 2.0    # NOT biological. Training artifact.
    n_hypotheses: int = 4
    depth_production: int = 5
    depth_comprehension: int = 8
    top_down_gain: float = 0.1
    ablate_top_down_bias: bool = False
    ablate_grammaticality: bool = False
