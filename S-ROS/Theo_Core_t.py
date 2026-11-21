"""
Theo_Core_t.py
THEO - Deterministic Recurrent Execution Core for S-ROS
(AM 4-10-2026) ==> Gets to ~0.05–0.02 MSE on very simple sequences single counter, basic IF/THEN but on anything with nested loops, GOSUB stack it's not even close maybe a BASIC program state encoder could fix this


BIOLOGICAL GROUNDING
This file implements the frozen deterministic core as a recurrent function approximator.
It models crystallized procedural skills (BASIC-style loops, GOSUB, conditionals) as a
stable dynamical system using Discrete-Time Spiking Recurrent Neural Network (DTSRNN)
dynamics with LIF neurons. SCAM serves only as a rare-state seed/fallback. The main
executor learns and then freezes a general transition function f(sensor_spikes + coords)
→ next_action + next_coords. This satisfies the requirement: deterministic, generalizable,
not a lookup table. Noise resilience and long state-holding come from Sun et al. (2025).

Primary grounding papers:
Sun Z et al. (2025). "General and stable emulation of finite state machines with spiking neural networks." Neuromorphic Computing and Engineering, 5(1):014016. DOI: 10.1088/2634-4386/adc32d
Mink JW (1996). "The basal ganglia: focused selection and inhibition of competing motor programs." Progress in Neurobiology, 50(4):381-425. DOI: 10.1016/S0301-0082(96)00042-1
Wolpert DM, Miall RC, Kawato M (1998). "Internal models in the cerebellum." Trends in Cognitive Sciences, 2(9):338-347. DOI: 10.1016/S1364-6613(98)01221-6
Tse et al. (2007). "Schemas and Memory Consolidation." Science, 316(5821):76-82. DOI: 10.1126/science.1135935
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

from theo_scam import TheoSCAM  # the one canonical S-CAM


@dataclass
class TheoExecutionState:
    action_spikes: torch.Tensor
    predicted_coords: torch.Tensor
    confidence: float
    yield_to_plastic_core: bool
    mismatch_detected: bool
    requires_hardware_reset: bool
    source: str = "executor"      # "scam" (frozen engram hit) or "executor" (plastic path)
    engram_index: int = -1        # S-CAM slot on a hit, else -1


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid derivative backward, so the executor is trainable."""
    @staticmethod
    def forward(ctx, v_minus_thr):
        ctx.save_for_backward(v_minus_thr)
        return (v_minus_thr >= 0).to(v_minus_thr.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out / (1.0 + 10.0 * x.abs()) ** 2


def spike_fn(v_minus_thr: torch.Tensor) -> torch.Tensor:
    return _SurrogateSpike.apply(v_minus_thr)


class TheoRecurrentExecutor(nn.Module):
    """
    Recurrent LIF function approximator: (sensor spikes, coords) -> (action logits, next coords).

    It is plastic until freeze() is called. Spikes use a surrogate gradient, so
    it can actually be trained; the readout is taken from the mean firing rate
    of the last layer over num_steps.
    """
    def __init__(self, spike_dim: int = 128, coord_dim: int = 64,
                 hidden_dim: int = 256, num_layers: int = 2,
                 beta: float = 0.7, num_steps: int = 25):
        super().__init__()
        self.spike_dim = spike_dim
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.input_dim = spike_dim + coord_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = self.input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_d, hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, spike_dim + coord_dim)

    def lif_step(self, v: torch.Tensor, current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.beta * v + (1 - self.beta) * current
        spikes = spike_fn(v - 1.0)
        v = v * (1 - spikes.detach())  # reset
        return v, spikes

    def forward(self, sensor_spikes: torch.Tensor, current_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([sensor_spikes, current_coords], dim=-1)

        v = [torch.zeros(x.shape[0], self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        spikes = [torch.zeros_like(v[0]) for _ in range(self.num_layers)]
        rate = torch.zeros_like(v[0])

        for _ in range(self.num_steps):
            for i in range(self.num_layers):
                current = self.layers[i](x if i == 0 else spikes[i - 1])
                v[i], spikes[i] = self.lif_step(v[i], current)
            rate = rate + spikes[-1]
        rate = rate / self.num_steps

        out = self.output_proj(rate)
        action_logits = out[:, :self.spike_dim]
        next_coords = out[:, self.spike_dim:]
        return action_logits, next_coords

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.parameters())


class TheoCore(nn.Module):
    """
    THEO - Deterministic Recurrent Execution Core

    Inference order (README, Stage Zero): S-CAM exact match first, trainable
    executor second. A matching engram always wins, and its stored action is
    emitted unmodified, so executor drift cannot change a crystallized skill.
    """
    def __init__(self, spike_dim: int = 128, coord_dim: int = 64,
                 confidence_threshold: float = 0.99, max_engrams: int = 1024,
                 hidden_dim: int = 256, num_layers: int = 2,
                 beta: float = 0.7, num_steps: int = 25):
        super().__init__()
        self.spike_dim = spike_dim
        self.coord_dim = coord_dim
        self.confidence_threshold = confidence_threshold

        self.scam = TheoSCAM(spike_dim, max_engrams=max_engrams)
        self.executor = TheoRecurrentExecutor(spike_dim, coord_dim, hidden_dim, num_layers, beta, num_steps)

        # Basal ganglia: tonic inhibition of every action channel, released by
        # the striatal selector. Units are logits, so the gate is commensurate
        # with the executor's output and is learnable.
        self.tonic_inhibition = nn.Parameter(torch.ones(spike_dim) * 2.0)
        self.striatal_selector = nn.Linear(spike_dim, spike_dim, bias=True)

        self.cerebellar_predictor = nn.Sequential(
            nn.Linear(spike_dim + coord_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, coord_dim)
        )

    def executor_logits(self, sensor_spikes: torch.Tensor, current_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plastic path only (differentiable). Train against this."""
        action_logits, exec_coords = self.executor(sensor_spikes, current_coords)
        gate_signal = torch.sigmoid(self.striatal_selector(sensor_spikes))
        gated_logits = action_logits - self.tonic_inhibition * (1.0 - gate_signal)
        return gated_logits, exec_coords

    def forward(self, sensor_spikes: torch.Tensor, current_coords: torch.Tensor) -> TheoExecutionState:
        if sensor_spikes.shape[0] != 1:
            raise ValueError("TheoCore.forward executes one state at a time; use scam.lookup / executor_logits for batches")

        # 1. Frozen engram: exact S-CAM match wins outright.
        engram_action, score, idx = self.scam.retrieve_engram(sensor_spikes)
        if idx >= 0:
            selected_action = engram_action
            confidence = score
            source = "scam"
        else:
            # 2. Plastic executor through the basal ganglia gate.
            gated_logits, _ = self.executor_logits(sensor_spikes, current_coords)
            selected_action = spike_fn(gated_logits)
            probs = torch.sigmoid(gated_logits)
            # How far every channel is from the decision boundary (1.0 = fully decided).
            confidence = float((2.0 * (probs - 0.5).abs()).min().item())
            source = "executor"

        # Cerebellar forward model
        prediction_input = torch.cat([selected_action, current_coords], dim=-1)
        predicted_coords = self.cerebellar_predictor(prediction_input)

        return TheoExecutionState(
            action_spikes=selected_action,
            predicted_coords=predicted_coords,
            confidence=confidence,
            yield_to_plastic_core=(source == "executor" and confidence < self.confidence_threshold),
            mismatch_detected=False,
            requires_hardware_reset=False,
            source=source,
            engram_index=idx,
        )

    def evaluate_execution_error(self, state: TheoExecutionState, actual_resulting_coords: torch.Tensor, error_threshold: float = 0.5) -> TheoExecutionState:
        if state.yield_to_plastic_core:
            return state
        mismatch_error = F.mse_loss(state.predicted_coords, actual_resulting_coords).item()
        if mismatch_error > error_threshold:
            state.mismatch_detected = True
            state.yield_to_plastic_core = True
            state.requires_hardware_reset = True
        return state

    def get_yield_target(self) -> str:
        return "plastic_hippocampal_core"

    @torch.no_grad()
    def crystallize_from_plastic(self, sensor_pattern: torch.Tensor, action_pattern: torch.Tensor,
                                 overwrite: bool = False) -> int:
        """
        Called once Neo's gate has passed for this (sensor, action) pair.
        Writes a frozen engram and returns its slot. The executor stays plastic
        so later skills can still be learned; call executor.freeze() to stop that.
        """
        return self.scam.crystallize_skill(sensor_pattern, action_pattern, overwrite=overwrite)


# ====================== SELF-TEST ======================
def run_self_test() -> None:
    torch.manual_seed(0)
    D = 128
    theo = TheoCore()
    coords = torch.randn(1, 64) * 0.1

    def onehot(i):
        v = torch.zeros(1, D); v[0, i] = 1.0
        return v

    # 1. empty S-CAM: executor path, correct shapes
    st = theo(onehot(42), coords)
    assert st.source == "executor" and st.engram_index == -1
    assert st.action_spikes.shape == (1, D) and st.predicted_coords.shape == (1, 64)
    print("PASS 1: empty S-CAM falls through to the executor with correct shapes")

    # 2. the executor is trainable (surrogate gradient reaches the first layer)
    logits, _ = theo.executor_logits(onehot(42), coords)
    F.binary_cross_entropy_with_logits(logits, onehot(7)).backward()
    g = theo.executor.layers[0].weight.grad
    assert g is not None and g.abs().sum() > 0, "no gradient reaches the executor"
    theo.zero_grad()
    print("PASS 2: gradient reaches the first executor layer")

    # 3. crystallized engrams win and are emitted unmodified
    for k in range(39):
        theo.crystallize_from_plastic(onehot(k), onehot(k + 50))
    for k in range(39):
        st = theo(onehot(k), coords)
        assert st.source == "scam" and st.confidence == 1.0 and not st.yield_to_plastic_core
        assert torch.equal(st.action_spikes, onehot(k + 50)), f"engram {k} not emitted exactly"
    print("PASS 3: 39/39 S-CAM hits emit the stored action exactly")

    # 4. interference: train the executor hard on other data, engrams must not move
    before = [theo(onehot(k), coords).action_spikes.clone() for k in range(39)]
    w_before = theo.executor.layers[0].weight.clone()
    h_before = theo.scam.content_hash()
    opt = torch.optim.SGD(theo.parameters(), lr=0.5)
    for step in range(30):
        x = torch.zeros(8, D); x[torch.arange(8), torch.randint(60, D, (8,))] = 1.0
        y = torch.zeros(8, D); y[torch.arange(8), torch.randint(0, D, (8,))] = 1.0
        opt.zero_grad()
        lg, _ = theo.executor_logits(x, torch.randn(8, 64))
        F.binary_cross_entropy_with_logits(lg, y).backward()
        opt.step()
    assert not torch.equal(w_before, theo.executor.layers[0].weight), "executor did not actually train"
    assert theo.scam.content_hash() == h_before, "S-CAM content changed during training"
    for k in range(39):
        assert torch.equal(theo(onehot(k), coords).action_spikes, before[k]), f"engram {k} drifted"
    print("PASS 4: executor weights changed, all 39 engram readouts bit-identical")

    # 5. freeze really freezes
    theo.executor.freeze()
    assert theo.executor.is_frozen
    print("PASS 5: executor.freeze() leaves no trainable executor parameter")

    print("\nAll TheoCore self-tests passed.")


if __name__ == "__main__":
    run_self_test()
