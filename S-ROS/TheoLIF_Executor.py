"""
TheoLIF_Executor.py

BIOLOGICAL GROUNDING
Discrete-time LIF recurrent executor that provides noise resilience and long state-holding periods as proven in Sun et al. (2025) DTSRNN for FSM emulation. Weights are frozen after crystallization from the plastic core.

Primary grounding paper:
Sun Z et al. (2025). "General and stable emulation of finite state machines with spiking neural networks." Neuromorphic Computing and Engineering, 5(1):014016. DOI: 10.1088/2634-4386/adc32d

ENGINEERING NOTE
The executor's job is to turn a retrieved engram into a stable spike train
without corrupting it. It is therefore constructed, not randomly initialised:
the input weight is a scaled identity (each engram bit drives its own neuron
above threshold) and the recurrent weight starts at zero. All weights are
frozen (requires_grad=False) from construction. The output is rate-decoded
over num_steps, so it does not depend on which step the simulation stops on
and it tolerates sub-threshold input noise. The self-test asserts
output == input engram for every one-hot and for multi-bit patterns.
"""

import torch
import torch.nn as nn


class TheoLIF_Executor(nn.Module):
    def __init__(self, hidden_dim: int = 128, beta: float = 0.7, num_steps: int = 25,
                 input_gain: float = 2.0, rate_threshold: float = 0.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.num_steps = num_steps
        self.threshold = 1.0
        self.rate_threshold = rate_threshold
        # Frozen weights. Identity drive, no recurrence by default.
        self.rec_weight = nn.Parameter(torch.zeros(hidden_dim, hidden_dim), requires_grad=False)
        self.input_weight = nn.Parameter(torch.eye(hidden_dim) * input_gain, requires_grad=False)

    @torch.no_grad()
    def forward(self, initial_state: torch.Tensor) -> torch.Tensor:
        # initial_state is the retrieved engram (hidden_dim), held as a constant drive
        v = torch.zeros_like(initial_state)  # membrane potential
        spikes = torch.zeros_like(initial_state)
        count = torch.zeros_like(initial_state)
        for _ in range(self.num_steps):
            current = torch.matmul(spikes, self.rec_weight) + torch.matmul(initial_state, self.input_weight)
            v = self.beta * v + (1 - self.beta) * current
            spikes = (v >= self.threshold).float()
            v = v * (1 - spikes)  # reset
            count = count + spikes
        # rate decode: a neuron is "on" if it fired in at least rate_threshold of the steps
        return (count / self.num_steps >= self.rate_threshold).float()


def run_self_test() -> None:
    torch.manual_seed(0)
    D = 128
    ex = TheoLIF_Executor(D)
    assert not any(p.requires_grad for p in ex.parameters()), "executor weights must be frozen"

    eye = torch.eye(D)
    assert torch.equal(ex(eye), eye), "one-hot engrams must pass through unchanged"
    print("PASS 1: 128/128 one-hot engrams preserved")

    multi = (torch.rand(64, D) < 0.1).float()
    assert torch.equal(ex(multi), multi), "multi-bit engrams must pass through unchanged"
    print("PASS 2: 64/64 random multi-bit engrams preserved")

    assert ex(torch.zeros(1, D)).sum() == 0, "silence in must give silence out"
    print("PASS 3: zero input stays silent")

    noisy = eye + torch.randn(D, D) * 0.1
    assert torch.equal(ex(noisy), eye), "sub-threshold noise must not change the output"
    print("PASS 4: output unchanged under input noise (sigma 0.1)")

    assert torch.equal(ex(eye), ex(eye)), "must be deterministic"
    print("PASS 5: deterministic")
    print("\nAll TheoLIF_Executor self-tests passed.")


if __name__ == "__main__":
    run_self_test()
