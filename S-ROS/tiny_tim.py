"""
tiny_Tim.py
Consolidated PRAGMI Smoke-Test Harness (April 6, 2026)

Features:
- Full Hippocampal Core (DG, CA3, CA1, Subiculum, EC, Astro)
- Physiological Epistemic Doubt Reflex (Weighted Substrate Signal)
- TinyTimmyEnsemble with Usage-Driven Structural Expansion (+64 neurons)
- Comprehensive Self-Check Suite & Save/Load Verification
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Configuration
# =========================================================================

@dataclass
class CognitiveKernelConfig:
    coordinate_dim: int = 64
    dentate_gyrus_dim: int = 512
    ca3_dim: int = 512
    ca1_dim: int = 256
    subiculum_dim: int = 128
    dg_sparsity: float = 0.04
    ca3_max_episodes: int = 64
    ca3_settle_steps: int = 5
    ca3_recurrent_gain: float = 0.7
    ca1_novelty_threshold: float = 0.3
    ec_buffer_tau: float = 0.5
    astro_glutamate_tau: float = 0.95
    astro_calcium_gain: float = 4.0
    astro_calcium_target: float = 0.5
    astro_eta_min: float = 0.1
    astro_eta_max: float = 3.0
    anatomical_layout: bool = True
    ca3_distance_sigma: float = 2.0

# =========================================================================
# Subsystems
# =========================================================================

class NeuronPositionRegistry(nn.Module):
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        total_neurons = (
            cfg.coordinate_dim + cfg.dentate_gyrus_dim + 
            cfg.ca3_dim + cfg.ca1_dim + cfg.subiculum_dim
        )
        if cfg.anatomical_layout:
            positions = self._anatomical_positions(cfg)
        else:
            positions = torch.rand(total_neurons, 3) * 10.0
        self.register_buffer("positions", positions)
        self._boundaries = self._compute_boundaries(cfg)

    @staticmethod
    def _compute_boundaries(cfg: CognitiveKernelConfig) -> Dict[str, Tuple[int, int]]:
        offset = 0
        boundaries = {}
        for name, size in [
            ("ec", cfg.coordinate_dim), ("dg", cfg.dentate_gyrus_dim),
            ("ca3", cfg.ca3_dim), ("ca1", cfg.ca1_dim), ("subiculum", cfg.subiculum_dim),
        ]:
            boundaries[name] = (offset, offset + size)
            offset += size
        return boundaries

    @staticmethod
    def _anatomical_positions(cfg: CognitiveKernelConfig) -> torch.Tensor:
        positions = []
        gen = torch.Generator().manual_seed(0)
        # Layers: EC -> DG -> CA3 -> CA1 -> Subiculum
        for i, (n, y) in enumerate([
            (cfg.coordinate_dim, 0.0), (cfg.dentate_gyrus_dim, 2.0),
            (cfg.ca3_dim, 4.0), (cfg.ca1_dim, 6.0), (cfg.subiculum_dim, 7.5)
        ]):
            p = torch.zeros(n, 3)
            p[:, 0] = torch.linspace(-3.0, 3.0, n) if i != 2 else 3.0 * torch.cos(torch.linspace(0, math.pi, n))
            p[:, 1] = y + torch.randn(n, generator=gen) * 0.2
            p[:, 2] = torch.randn(n, generator=gen) * 0.5
            positions.append(p)
        return torch.cat(positions, dim=0)

    def get_population_positions(self, name: str) -> torch.Tensor:
        start, end = self._boundaries[name]
        return self.positions[start:end]

    def pairwise_distances(self, name: str) -> torch.Tensor:
        pos = self.get_population_positions(name)
        return torch.sqrt(((pos.unsqueeze(0) - pos.unsqueeze(1))**2).sum(-1) + 1e-8)

class DentateGyrus(nn.Module):
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.perforant_path = nn.Linear(cfg.coordinate_dim, cfg.dentate_gyrus_dim)
        self.norm = nn.LayerNorm(cfg.dentate_gyrus_dim)
        self.k = max(1, int(cfg.dentate_gyrus_dim * cfg.dg_sparsity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.norm(F.relu(self.perforant_path(x)))
        v, i = proj.topk(self.k, dim=-1)
        return torch.zeros_like(proj).scatter_(1, i, v)

class CA3AttractorNetwork(nn.Module):
    def __init__(self, cfg: CognitiveKernelConfig, pos: NeuronPositionRegistry) -> None:
        super().__init__()
        self.cfg = cfg
        self.mossy_fiber = nn.Linear(cfg.dentate_gyrus_dim, cfg.ca3_dim, bias=False)
        self.register_buffer("W", torch.zeros(cfg.ca3_dim, cfg.ca3_dim))
        mask = torch.exp(-pos.pairwise_distances("ca3") / cfg.ca3_distance_sigma)
        self.register_buffer("mask", mask)
        self._stored_patterns: List[torch.Tensor] = []
        self.schaffer = nn.Linear(cfg.ca3_dim, cfg.ca1_dim)

    def store_episode(self, p: torch.Tensor):
        p = p.detach().clone()
        p /= (p.norm() + 1e-8)
        if len(self._stored_patterns) >= self.cfg.ca3_max_episodes:
            self._stored_patterns.pop(0)
        self._stored_patterns.append(p)
        patterns = torch.stack(self._stored_patterns)
        W = patterns.T @ patterns
        W.fill_diagonal_(0)
        self.W.copy_(W * self.mask)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cue = self.mossy_fiber(x)
        state = cue
        # Snapshot: store_episode() rewrites W in place after this settle,
        # which would otherwise break loss.backward().
        W = self.W.clone()
        for _ in range(self.cfg.ca3_settle_steps):
            recurrent = F.linear(state, W)
            state = torch.tanh(self.cfg.ca3_recurrent_gain * recurrent + (1-self.cfg.ca3_recurrent_gain)*cue)
        return state, self.schaffer(state)

class CA1Comparator(nn.Module):
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.ta = nn.Linear(cfg.coordinate_dim, cfg.ca1_dim)
        self.cs = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, False)
        self.cd = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, False)
        self.og = nn.Linear(cfg.ca1_dim, cfg.ca1_dim)

    def forward(self, s: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.ta(e)
        nov = (1.0 - F.cosine_similarity(self.cs(s), self.cd(d), dim=-1)).clamp(0, 1)
        gate = nov.unsqueeze(-1)
        out = torch.tanh(self.og(gate * d + (1-gate) * s))
        return out, nov

class Subiculum(nn.Module):
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.c2s = nn.Linear(cfg.ca1_dim, cfg.subiculum_dim)
        self.s2e = nn.Linear(cfg.subiculum_dim, cfg.coordinate_dim)
        self.gate = nn.Linear(cfg.subiculum_dim, cfg.subiculum_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.relu(self.c2s(x))
        return self.s2e(s * torch.sigmoid(self.gate(s)))

class AstrocyticRegulator(nn.Module):
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("glu", torch.tensor(0.0))
        self.register_buffer("ca", torch.tensor(cfg.astro_calcium_target))

    def update(self, x: torch.Tensor) -> float:
        act = x.detach().abs().mean().item()
        tau = self.cfg.astro_glutamate_tau
        self.glu.copy_(tau * self.glu + (1-tau) * act)
        drv = torch.sigmoid(torch.tensor(self.cfg.astro_calcium_gain * (self.glu.item() - self.cfg.astro_calcium_target))).item()
        self.ca.copy_(tau * self.ca + (1-tau) * drv)
        eta = 1.0 + self.ca.item() - self.cfg.astro_calcium_target
        return max(self.cfg.astro_eta_min, min(self.cfg.astro_eta_max, eta))

class EntorhinalCortex(nn.Module):
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("buf", torch.zeros(cfg.coordinate_dim))
        self.norm = nn.LayerNorm(cfg.coordinate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.buf.copy_(self.cfg.ec_buffer_tau * self.buf + (1-self.cfg.ec_buffer_tau) * x.detach().mean(0))
        return self.norm(x + 0.1 * self.buf.unsqueeze(0))

# =========================================================================
# Main Kernel
# =========================================================================

class CognitiveKernel(nn.Module):
    def __init__(self, cfg: Optional[CognitiveKernelConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CognitiveKernelConfig()

        self.neuron_positions = NeuronPositionRegistry(self.cfg)
        self.entorhinal_cortex = EntorhinalCortex(self.cfg)
        self.dentate_gyrus = DentateGyrus(self.cfg)
        self.ca3 = CA3AttractorNetwork(self.cfg, self.neuron_positions)
        self.ca1 = CA1Comparator(self.cfg)
        self.subiculum = Subiculum(self.cfg)
        self.astrocyte = AstrocyticRegulator(self.cfg)

        self.register_buffer("working_memory", torch.zeros(self.cfg.ca1_dim))
        self.register_buffer("doubt_current", torch.tensor(0.0))

    def forward(self, coords: torch.Tensor, store_if_novel: bool = True):
        ec_output = self.entorhinal_cortex(coords)
        dg_sparse = self.dentate_gyrus(ec_output)
        ca3_state, schaffer_to_ca1 = self.ca3(dg_sparse)
        ca1_output, novelty = self.ca1(schaffer_to_ca1, ec_output)

        with torch.no_grad():
            self.working_memory.copy_(ca1_output.detach().mean(dim=0))

        reconstructed = self.subiculum(ca1_output)

        # === Physiological Epistemic Doubt Reflex ===
        with torch.no_grad():
            recon_error = (1.0 - F.cosine_similarity(reconstructed, coords, dim=-1).mean()).clamp_(0.0, 1.0)
            doubt = (
                0.40 * novelty.mean() +
                0.40 * recon_error +
                0.20 * (1.0 - self.cfg.ca1_novelty_threshold)
            ).clamp_(0.0, 1.0)
            self.doubt_current.copy_(doubt)

        diagnostics = {
            "dg_sparsity": (dg_sparse == 0).float().mean().item(),
            "mean_novelty": novelty.mean().item(),
            "doubt_current": float(self.doubt_current.item()),
            "num_stored_episodes": len(self.ca3._stored_patterns),
            "astro_eta": self.astrocyte.update(ca3_state),
        }

        if store_if_novel and diagnostics["mean_novelty"] > self.cfg.ca1_novelty_threshold:
            mean_dg = dg_sparse.detach().mean(dim=0)
            self.ca3.store_episode(self.ca3.mossy_fiber(mean_dg))
            diagnostics["stored_this_step"] = True
        else:
            diagnostics["stored_this_step"] = False

        return reconstructed, novelty, diagnostics

    def get_doubt_current(self) -> float:
        return float(self.doubt_current.item())

    def is_doubtful(self, threshold: float = 0.70) -> bool:
        return self.doubt_current.item() > threshold

    def sleep_consolidation(self) -> Dict[str, float]:
        self.entorhinal_cortex.buf.zero_()
        self.working_memory.zero_()
        return {"episodes_in_ca3": len(self.ca3._stored_patterns)}

    def get_memory_state(self) -> Dict[str, object]:
        return {
            "working_memory": self.working_memory.clone(),
            "doubt_current": self.doubt_current.clone(),
            "ca3_W": self.ca3.W.clone(),
            "ca3_patterns": [p.clone() for p in self.ca3._stored_patterns],
        }

    def load_memory_state(self, state: Dict[str, object]) -> None:
        with torch.no_grad():
            if "working_memory" in state:
                self.working_memory.copy_(state["working_memory"])
            if "doubt_current" in state:
                self.doubt_current.copy_(state["doubt_current"])
            if "ca3_W" in state and state["ca3_W"] is not None:
                self.ca3.W.copy_(state["ca3_W"])
            if "ca3_patterns" in state:
                self.ca3._stored_patterns = [p.clone() for p in state["ca3_patterns"]]

# =========================================================================
# Tiny Ensemble Wrapper
# =========================================================================

class DummyColumn(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

    def expand_for_new_d_model(self, new_d: int):
        self.d_model = new_d

class TinyTimmyEnsemble:
    def __init__(self, kernel: CognitiveKernel, num_initial_specialists: int = 3):
        self.kernel = kernel
        self.prime = None
        self.specialists: List[DummyColumn] = []
        self.specialist_usage: List[float] = []
        self.current_d_model: int = 256

        for _ in range(num_initial_specialists):
            self.specialists.append(None)
            self.specialist_usage.append(0.0)

    def set_prime(self, prime_column: DummyColumn):
        self.prime = prime_column
        for i in range(len(self.specialists)):
            self.specialists[i] = copy.deepcopy(prime_column)

    def forward(self, coords: torch.Tensor):
        idx = torch.randint(0, len(self.specialists), (1,)).item()
        self.specialist_usage[idx] += 1.0
        return self.kernel(coords)

    def sleep_consolidation(self):
        if not self.specialists or any(s is None for s in self.specialists):
            return self.kernel.sleep_consolidation()

        usage = torch.tensor(self.specialist_usage)
        busiest_idx = int(usage.argmax().item())
        busiest = self.specialists[busiest_idx]

        old_d = busiest.d_model
        new_d = old_d + 64
        busiest.expand_for_new_d_model(new_d)
        self.current_d_model = new_d

        print(f"[Sleep] Expanded busiest specialist #{busiest_idx} from {old_d} → {new_d}")
        self.specialist_usage = [0.0] * len(self.specialist_usage)
        return self.kernel.sleep_consolidation()

# =========================================================================
# Self-Check Suite
# =========================================================================

def run_comprehensive_self_check():
    print("=" * 90)
    print("PRAGMI TINY CORE SELF-CHECK + SAVE/LOAD VERIFICATION")
    print("=" * 90)

    cfg = CognitiveKernelConfig()
    kernel = CognitiveKernel(cfg)
    ensemble = TinyTimmyEnsemble(kernel, num_initial_specialists=3)
    ensemble.set_prime(DummyColumn(d_model=256))

    # 1. Forward Pass & Doubt Check
    coords = torch.randn(4, 64)
    _, _, diag = kernel(coords)
    print(f"[OK] Forward pass | Initial Doubt: {diag['doubt_current']:.3f}")

    # 2. Save/Load Round-Trip
    state = kernel.get_memory_state()
    new_kernel = CognitiveKernel(cfg)
    new_kernel.load_memory_state(state)
    assert abs(kernel.get_doubt_current() - new_kernel.get_doubt_current()) < 1e-6
    print("[OK] Save/Load state verification (Doubt Current sync'd)")

    # 3. Dynamic Expansion Test
    ensemble.specialist_usage[1] = 100.0  # Force index 1 to be busiest
    ensemble.sleep_consolidation()
    assert ensemble.specialists[1].d_model == 320
    print(f"[OK] Growth system verified (256 → {ensemble.specialists[1].d_model})")

    # 4. Novelty Accumulation
    for _ in range(5): kernel(torch.randn(4, 64), store_if_novel=True)
    assert len(kernel.ca3._stored_patterns) > 0
    print(f"[OK] CA3 episodic storage: {len(kernel.ca3._stored_patterns)} patterns active")

    print("=" * 90)
    print("SELF-CHECK PASSED: tiny_Tim is nominal.")
    print("=" * 90)

if __name__ == "__main__":
    run_comprehensive_self_check()

    # Visual Day-Cycle Test
    print("\nStarting visual Day-Cycle simulation...")
    kernel = CognitiveKernel()
    ensemble = TinyTimmyEnsemble(kernel, num_initial_specialists=3)
    ensemble.set_prime(DummyColumn(d_model=256))

    for day in range(2):
        print(f"\n--- Day {day+1} ---")
        for step in range(3):
            # Simulate slight data drift
            data = torch.randn(4, 64) + (day * 0.5)
            _, _, diag = ensemble.forward(data)
            print(f"  Step {step+1} | Doubt: {diag['doubt_current']:.3f} | Astro Eta: {diag['astro_eta']:.3f}")
        ensemble.sleep_consolidation()
  
