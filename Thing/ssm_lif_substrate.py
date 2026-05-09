"""
SSM-LIF Hybrid Substrate.

Per-neuron state-space model with HiPPO-LegS initialization, deterministic
Heaviside threshold, adaptive threshold trace, liquid time constant, and
SpikeMixer population aggregation. Deployment runs as a causal iterative
forward; the convolutional reformulation and the surrogate dynamic network
used to parallelize training live in the training pipeline and are not
present in this file.

The hidden state is n-dimensional per neuron. The state-transition matrix A
is initialized with HiPPO-LegS coefficients in normal-plus-low-rank form so
the substrate carries history as a polynomial projection of the input
trajectory rather than as a decaying scalar trace. Spikes are caused by
threshold crossings of the integrated membrane potential. There is no
sampling at the substrate level.

Serialization writes and restores the full per-neuron state, including the
hidden vector h, the membrane potential v, the adaptive threshold trace
theta, and the liquid time constant gate state. The whole point of the
substrate is that what is restored is a causal record of what the substrate
integrated, so partial state save would defeat the architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# HiPPO-LegS initialization
# ---------------------------------------------------------------------------


def hippo_legs_nplr(n: int, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor, Tensor]:
    """Return the HiPPO-LegS state matrix in normal-plus-low-rank form.

    The dense HiPPO-LegS matrix A_full is decomposed as
        A_full = A_normal - p p^T
    where A_normal is the skew-symmetric (normal) component and p is the
    low-rank correction. The triple (A_normal, p, B_legs) is what the
    substrate needs to initialize its per-neuron recurrence; the full A is
    not materialized at runtime because the substrate does not require it
    for the iterative forward.

    The closed form for HiPPO-LegS comes from Gu et al. The (i, j) entry of
    A_full is
        -sqrt((2 i + 1)(2 j + 1))   for i > j,
        -(i + 1)                    for i = j,
        0                           for i < j,
    and the input projection B_legs has entries sqrt(2 i + 1).
    """
    indices = torch.arange(n, dtype=dtype)
    rows = indices.unsqueeze(1)
    cols = indices.unsqueeze(0)

    factor = torch.sqrt((2.0 * rows + 1.0) * (2.0 * cols + 1.0))
    a_full = torch.where(rows > cols, -factor, torch.zeros_like(factor))
    a_full = a_full + torch.diag(-(indices + 1.0))

    # Low-rank correction: p_i = sqrt(i + 1/2). Then A_full + p p^T is the
    # skew-symmetric normal part up to a scaling that the literature absorbs
    # into the low-rank term. We keep the convention A_full = A_normal - p p^T
    # so that the substrate computes A_normal h - p (p^T h) at each step.
    p = torch.sqrt(indices + 0.5).to(dtype)
    a_normal = a_full + torch.outer(p, p)

    b_legs = torch.sqrt(2.0 * indices + 1.0).to(dtype)
    return a_normal, p, b_legs


def discretize_zoh(a_continuous: Tensor, b_continuous: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    """Zero-order-hold discretization of the continuous-time SSM.

    Returns (A_d, B_d) such that
        h[t] = A_d h[t-1] + B_d x[t].
    The continuous form is h_dot = A h + B x. ZOH gives
        A_d = expm(A dt),
        B_d = A^{-1} (A_d - I) B.
    A_d uses the matrix exponential. B_d is computed via solve rather than
    explicit inverse for numerical stability when A is near-singular.
    """
    n = a_continuous.shape[0]
    a_d = torch.linalg.matrix_exp(a_continuous * dt)
    eye = torch.eye(n, dtype=a_continuous.dtype, device=a_continuous.device)
    b_d = torch.linalg.solve(a_continuous, (a_d - eye) @ b_continuous.unsqueeze(-1)).squeeze(-1)
    return a_d, b_d


# ---------------------------------------------------------------------------
# Substrate state container
# ---------------------------------------------------------------------------


@dataclass
class SubstrateState:
    """Full per-neuron state. Serialization writes this in its entirety.

    h: hidden state, shape (batch, num_neurons, n_dim).
    v: membrane potential, shape (batch, num_neurons).
    theta: adaptive threshold trace, shape (batch, num_neurons).
    last_spike: most recent spike output, shape (batch, num_neurons).
    """

    h: Tensor
    v: Tensor
    theta: Tensor
    last_spike: Tensor

    def detach(self) -> "SubstrateState":
        return SubstrateState(
            h=self.h.detach(),
            v=self.v.detach(),
            theta=self.theta.detach(),
            last_spike=self.last_spike.detach(),
        )

    def to(self, device: torch.device | str) -> "SubstrateState":
        return SubstrateState(
            h=self.h.to(device),
            v=self.v.to(device),
            theta=self.theta.to(device),
            last_spike=self.last_spike.to(device),
        )


# ---------------------------------------------------------------------------
# Surrogate gradient for the deterministic spike
# ---------------------------------------------------------------------------


class HeavisideWithSurrogate(torch.autograd.Function):
    """Deterministic Heaviside on the forward pass, fast-sigmoid surrogate
    on the backward pass.

    The forward output is exactly 0 or 1, never a sample. The surrogate is
    used only to provide a gradient signal during training; it never alters
    what the substrate emits. The temperature alpha is a per-layer
    hyperparameter and is supplied at the call site.
    """

    @staticmethod
    def forward(ctx, v_minus_theta: Tensor, alpha: float) -> Tensor:
        ctx.save_for_backward(v_minus_theta)
        ctx.alpha = alpha
        return (v_minus_theta >= 0).to(v_minus_theta.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        (v_minus_theta,) = ctx.saved_tensors
        alpha = ctx.alpha
        # Fast-sigmoid surrogate derivative: alpha / (1 + alpha |x|)^2.
        denom = 1.0 + alpha * v_minus_theta.abs()
        grad_input = grad_output * alpha / (denom * denom)
        return grad_input, None


def spike_fn(v_minus_theta: Tensor, alpha: float) -> Tensor:
    return HeavisideWithSurrogate.apply(v_minus_theta, alpha)


# ---------------------------------------------------------------------------
# SpikeMixer
# ---------------------------------------------------------------------------


class SpikeMixer(nn.Module):
    """Population-level mixing applied to inputs before per-neuron recurrence.

    The mixer is a single linear projection followed by GELU. Inter-neuron
    communication happens here; the per-neuron SSM recurrence below treats
    each neuron as independent given its mixed input.
    """

    def __init__(self, num_neurons: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Linear(num_neurons, num_neurons, bias=bias)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.weight(x))


# ---------------------------------------------------------------------------
# The substrate
# ---------------------------------------------------------------------------


class SSMLIFSubstrate(nn.Module):
    """SSM-LIF Hybrid neuron population.

    Each of num_neurons neurons carries its own n-dimensional hidden state
    initialized with HiPPO-LegS coefficients. The state matrix A is shared
    across neurons by default (parameter sharing keeps memory bounded for
    large populations); the input projection B and output projection C are
    per-neuron. Spikes are deterministic Heaviside threshold crossings.
    The membrane time constant is liquid: it is computed at each step as a
    sigmoid of the current input and the current membrane potential.

    Args:
        num_neurons: population size.
        n_dim: dimension of each neuron's hidden state. The HiPPO-LegS basis
            of order n_dim gives the neuron a polynomial approximation of
            its input history up to that order.
        dt: discretization step in the same time units as the Lou-window
            buffer used by Section 11. The default of 1.0 ms matches the
            substrate timestep convention used elsewhere in the corpus.
        v_rest: membrane resting potential.
        v_threshold_base: baseline threshold before adaptation.
        theta_decay: decay constant for the adaptive threshold trace.
        theta_increment: per-spike increment to the adaptive threshold.
        surrogate_alpha: per-layer surrogate gradient temperature. Layers
            doing rate-coded computation should use 2.0 to 4.0; layers
            doing spike-timing computation should use 0.5 to 1.0.
        share_state_matrix: if True, A is shared across neurons. If False,
            each neuron carries its own A (more capacity, more memory).
    """

    def __init__(
        self,
        num_neurons: int,
        n_dim: int,
        dt: float = 1.0,
        v_rest: float = 0.0,
        v_threshold_base: float = 1.0,
        theta_decay: float = 0.95,
        theta_increment: float = 0.5,
        surrogate_alpha: float = 2.0,
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

        a_normal, p, b_legs = hippo_legs_nplr(n_dim)
        a_full = a_normal - torch.outer(p, p)
        b_continuous = b_legs

        a_d, b_d = discretize_zoh(a_full, b_continuous, dt)

        if share_state_matrix:
            self.a_matrix = nn.Parameter(a_d.clone())
            self.b_vector = nn.Parameter(b_d.clone())
        else:
            self.a_matrix = nn.Parameter(a_d.clone().unsqueeze(0).expand(num_neurons, -1, -1).contiguous())
            self.b_vector = nn.Parameter(b_d.clone().unsqueeze(0).expand(num_neurons, -1).contiguous())

        # Per-neuron output projection from hidden state to membrane current.
        c_init = torch.empty(num_neurons, n_dim)
        nn.init.normal_(c_init, mean=0.0, std=1.0 / math.sqrt(n_dim))
        self.c_vector = nn.Parameter(c_init)

        # Liquid time constant gate. Two scalar projections per neuron map
        # (input, membrane) onto a sigmoid that becomes the integration
        # rate at the current step.
        ltc_init = torch.empty(num_neurons, 2)
        nn.init.normal_(ltc_init, mean=0.0, std=0.1)
        self.ltc_gate_weight = nn.Parameter(ltc_init)
        self.ltc_gate_bias = nn.Parameter(torch.zeros(num_neurons))

    def init_state(self, batch_size: int, device: torch.device | str | None = None) -> SubstrateState:
        device = device if device is not None else self.a_matrix.device
        return SubstrateState(
            h=torch.zeros(batch_size, self.num_neurons, self.n_dim, device=device),
            v=torch.full((batch_size, self.num_neurons), self.v_rest, device=device),
            theta=torch.zeros(batch_size, self.num_neurons, device=device),
            last_spike=torch.zeros(batch_size, self.num_neurons, device=device),
        )

    def forward(self, x: Tensor, state: SubstrateState) -> tuple[Tensor, SubstrateState]:
        """Causal single-step forward.

        Args:
            x: input current at the current step, shape (batch, num_neurons).
                The caller is responsible for any pre-mixing through a
                SpikeMixer or other projection; this method assumes x is
                already the per-neuron drive.
            state: current SubstrateState.

        Returns:
            spikes: deterministic spike output, shape (batch, num_neurons),
                values in {0, 1}.
            new_state: updated SubstrateState.
        """
        h_prev = state.h
        v_prev = state.v
        theta_prev = state.theta

        # SSM hidden-state update: h[t] = A h[t-1] + B x[t].
        if self.share_state_matrix:
            # h_prev: (batch, num_neurons, n_dim). a_matrix: (n_dim, n_dim).
            h_new = torch.einsum("bnd,ed->bne", h_prev, self.a_matrix)
            h_new = h_new + x.unsqueeze(-1) * self.b_vector
        else:
            h_new = torch.einsum("bnd,ned->bne", h_prev, self.a_matrix)
            h_new = h_new + x.unsqueeze(-1) * self.b_vector.unsqueeze(0)

        # Output projection from hidden state to membrane current.
        membrane_drive = torch.einsum("bnd,nd->bn", h_new, self.c_vector)

        # Liquid time constant: per-neuron gate as a function of input and
        # current membrane potential. The gate is in (0, 1); 1 means full
        # update from the new drive, 0 means full retention of v_prev.
        gate_input = x * self.ltc_gate_weight[:, 0] + v_prev * self.ltc_gate_weight[:, 1]
        gate_input = gate_input + self.ltc_gate_bias
        gate = torch.sigmoid(gate_input)

        v_new = (1.0 - gate) * v_prev + gate * (membrane_drive + self.v_rest)

        # Adaptive threshold: theta decays at theta_decay per step and is
        # incremented by theta_increment on each spike. The effective
        # threshold for the spike decision is v_threshold_base + theta_prev.
        effective_threshold = self.v_threshold_base + theta_prev
        spikes = spike_fn(v_new - effective_threshold, self.surrogate_alpha)

        # Membrane reset: subtract effective threshold on spike. This is the
        # subtractive-reset variant; it preserves residual current that
        # exceeded threshold, which is closer to biology than hard reset.
        v_after_reset = v_new - spikes * effective_threshold

        # Threshold trace update.
        theta_new = self.theta_decay * theta_prev + self.theta_increment * spikes

        new_state = SubstrateState(
            h=h_new,
            v=v_after_reset,
            theta=theta_new,
            last_spike=spikes,
        )
        return spikes, new_state

    # -- serialization ----------------------------------------------------

    def state_dict_full(self, state: SubstrateState) -> dict[str, Tensor]:
        """Return a flat dict suitable for torch.save.

        The full per-neuron state is serialized. Restoring from this dict
        reconstructs the substrate's causal record up to the saved step.
        """
        return {
            "h": state.h.detach().cpu(),
            "v": state.v.detach().cpu(),
            "theta": state.theta.detach().cpu(),
            "last_spike": state.last_spike.detach().cpu(),
        }

    def load_state_full(self, state_dict: dict[str, Tensor], device: torch.device | str | None = None) -> SubstrateState:
        device = device if device is not None else self.a_matrix.device
        return SubstrateState(
            h=state_dict["h"].to(device),
            v=state_dict["v"].to(device),
            theta=state_dict["theta"].to(device),
            last_spike=state_dict["last_spike"].to(device),
        )


# ---------------------------------------------------------------------------
# Composite: mixer plus substrate
# ---------------------------------------------------------------------------


class MixedSubstrate(nn.Module):
    """SpikeMixer applied to inputs, then the SSM-LIF substrate.

    This is the unit that downstream modules consume. The mixer handles
    inter-neuron communication; the substrate handles per-neuron history
    and spike generation.
    """

    def __init__(
        self,
        num_neurons: int,
        n_dim: int,
        dt: float = 1.0,
        surrogate_alpha: float = 2.0,
        share_state_matrix: bool = True,
    ) -> None:
        super().__init__()
        self.mixer = SpikeMixer(num_neurons)
        self.substrate = SSMLIFSubstrate(
            num_neurons=num_neurons,
            n_dim=n_dim,
            dt=dt,
            surrogate_alpha=surrogate_alpha,
            share_state_matrix=share_state_matrix,
        )

    def init_state(self, batch_size: int, device: torch.device | str | None = None) -> SubstrateState:
        return self.substrate.init_state(batch_size, device)

    def forward(self, x: Tensor, state: SubstrateState) -> tuple[Tensor, SubstrateState]:
        mixed = self.mixer(x)
        return self.substrate(mixed, state)
