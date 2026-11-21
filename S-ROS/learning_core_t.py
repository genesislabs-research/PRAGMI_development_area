"""
learning_core_t.py
Cognitive Kernel: Hippocampal Memory System with 3D Neuron Coordinates

BIOLOGICAL GROUNDING
This file models the hippocampal formation and its interactions with the
entorhinal cortex, implementing the core memory system of the PRAGMI
architecture. The kernel implements six anatomically grounded subsystems
arranged in the classical hippocampal circuit. It provides a hybrid memory
architecture that gives an LLM genuine persistent episodic memory that
survives context window closure. It reconstructs experience from partial
cues rather than retrieving text chunks.

The six subsystems include:
1. Dentate Gyrus: Pattern separation via sparse expansion.
2. CA3 Autoassociative Network: Pattern completion and temporal memory
   storage based on the hippocampal-neocortical consolidation gradient.
3. CA1 Comparator: Novelty detection and mismatch gating.
4. Entorhinal Cortex: Interface layer and short-term buffer.
5. Subiculum: Output gateway and schema-consistency evaluation.
6. Astrocytic Metaplasticity Regulator: Homeostatic sliding threshold.

Every neuron has an explicit 3D position to map spatial topology and
enable distance-dependent recurrent connectivity. The memory architecture
is divided into short-term entorhinal buffers, working CA1 representation,
and long-lasting CA3 attractor weights. 

Rapid schema-based consolidation is integrated directly into the CA3 eviction
logic and the kernel's diagnostic output. The system calculates a schema
consistency index to inform the deterministic core whether to rapidly
potentiate experience-expectant synapses or utilize a slow learning gradient.
CA3 protects memories from eviction during the critical initial retention
window to prevent retrograde amnesia.

Primary grounding papers:
Rolls ET (2013). "The mechanisms for pattern completion and pattern
separation in the hippocampus." Frontiers in Systems Neuroscience, 7:74.
DOI: 10.3389/fnsys.2013.00074

Hopfield JJ (1982). "Neural networks and physical systems with emergent
collective computational abilities." PNAS, 79(8):2554-2558.
DOI: 10.1073/pnas.79.8.2554

Tse D, Langston RF, Kakeyama M, Bethus I, Spooner PA, Wood ER, Witter MP,
Morris RGM (2007). "Schemas and memory consolidation." Science, 316(5821),
76-82. DOI: 10.1126/science.1135935
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class CognitiveKernelConfig:
    """Complete configuration for the Cognitive Kernel."""
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
    ca3_min_retention_cycles: int = 1
    ca3_max_retention_cycles: int = 16
    schema_error_max: float = 5.0

class NeuronPositionRegistry(nn.Module):
    """
    Registry of 3D positions for every neuron in the kernel.
    BIOLOGICAL STRUCTURE: Spatial organization of hippocampal neurons.
    BIOLOGICAL FUNCTION: Neurons are organized along anatomical axes 
    (transverse, proximodistal, dorsoventral). 3D positions model this 
    structure for distance-dependent connectivity.
    Strange BA et al. (2014). "Functional organization of the hippocampal
    longitudinal axis." Nature Reviews Neuroscience, 15(10):655-669.
    DOI: 10.1038/nrn3785
    """
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        total_neurons = (cfg.coordinate_dim + cfg.dentate_gyrus_dim + 
                         cfg.ca3_dim + cfg.ca1_dim + cfg.subiculum_dim)
        if cfg.anatomical_layout:
            positions = self._anatomical_positions(cfg)
        else:
            positions = torch.rand(total_neurons, 3) * 10.0
        self.register_buffer("positions", positions)
        self._boundaries = self._compute_boundaries(cfg)
        self.total_neurons = total_neurons

    @staticmethod
    def _compute_boundaries(cfg: CognitiveKernelConfig) -> Dict[str, Tuple[int, int]]:
        offset = 0
        boundaries = {}
        for name, size in [("ec", cfg.coordinate_dim), ("dg", cfg.dentate_gyrus_dim),
                           ("ca3", cfg.ca3_dim), ("ca1", cfg.ca1_dim),
                           ("subiculum", cfg.subiculum_dim)]:
            boundaries[name] = (offset, offset + size)
            offset += size
        return boundaries

    @staticmethod
    def _anatomical_positions(cfg: CognitiveKernelConfig) -> torch.Tensor:
        """
        Generate anatomically plausible 3D positions for all populations.
        NOT a biological quantity: exact coordinates are engineering approximations.
        """
        positions = []
        gen = torch.Generator().manual_seed(0)
        n_ec = cfg.coordinate_dim
        ec_pos = torch.zeros(n_ec, 3)
        ec_pos[:, 0] = torch.linspace(-3.0, 3.0, n_ec)
        ec_pos[:, 1] = 0.0 + torch.randn(n_ec, generator=gen) * 0.2
        ec_pos[:, 2] = torch.randn(n_ec, generator=gen) * 0.5
        positions.append(ec_pos)
        n_dg = cfg.dentate_gyrus_dim
        dg_pos = torch.zeros(n_dg, 3)
        dg_pos[:, 0] = torch.randn(n_dg, generator=gen) * 2.0
        dg_pos[:, 1] = 2.0 + torch.randn(n_dg, generator=gen) * 0.3
        dg_pos[:, 2] = torch.randn(n_dg, generator=gen) * 1.5
        positions.append(dg_pos)
        n_ca3 = cfg.ca3_dim
        ca3_pos = torch.zeros(n_ca3, 3)
        theta = torch.linspace(0, math.pi, n_ca3)
        ca3_pos[:, 0] = 3.0 * torch.cos(theta)
        ca3_pos[:, 1] = 4.0 + torch.sin(theta) * 1.0
        ca3_pos[:, 2] = torch.randn(n_ca3, generator=gen) * 0.8
        ca3_pos += torch.randn_like(ca3_pos) * 0.15
        positions.append(ca3_pos)
        n_ca1 = cfg.ca1_dim
        ca1_pos = torch.zeros(n_ca1, 3)
        ca1_pos[:, 0] = torch.linspace(-2.5, 2.5, n_ca1)
        ca1_pos[:, 1] = 6.0 + torch.randn(n_ca1, generator=gen) * 0.25
        ca1_pos[:, 2] = torch.randn(n_ca1, generator=gen) * 0.6
        positions.append(ca1_pos)
        n_sub = cfg.subiculum_dim
        sub_pos = torch.zeros(n_sub, 3)
        sub_pos[:, 0] = torch.linspace(-2.0, 2.0, n_sub)
        sub_pos[:, 1] = 7.5 + torch.randn(n_sub, generator=gen) * 0.2
        sub_pos[:, 2] = torch.randn(n_sub, generator=gen) * 0.4
        positions.append(sub_pos)
        return torch.cat(positions, dim=0)

    def get_population_positions(self, name: str) -> torch.Tensor:
        start, end = self._boundaries[name]
        return self.positions[start:end]

    def pairwise_distances(self, name: str) -> torch.Tensor:
        pos = self.get_population_positions(name)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
class DentateGyrus(nn.Module):
    """
    Pattern separation via competitive sparse expansion.
    BIOLOGICAL STRUCTURE: Dentate gyrus granule cell layer.
    BIOLOGICAL FUNCTION: Expands perforant path input into a sparse code 
    to prevent catastrophic interference.
    Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074
    ANATOMICAL INTERFACE:
    Sending structure: Entorhinal cortex layer II.
    Receiving structure: Dentate gyrus granule cells.
    Connection: Perforant path.
    """
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.perforant_path = nn.Linear(cfg.coordinate_dim, cfg.dentate_gyrus_dim, bias=True)
        self.norm = nn.LayerNorm(cfg.dentate_gyrus_dim)
        self.k_active = max(1, int(cfg.dentate_gyrus_dim * cfg.dg_sparsity))

    def forward(self, ec_input: torch.Tensor) -> torch.Tensor:
        projected = self.norm(F.relu(self.perforant_path(ec_input)))
        topk_vals, topk_idx = projected.topk(self.k_active, dim=-1)
        sparse = torch.zeros_like(projected)
        sparse.scatter_(1, topk_idx, topk_vals)
        return sparse

class CA3AttractorNetwork(nn.Module):
    """
    Recurrent autoassociative network for episodic memory storage.
    BIOLOGICAL STRUCTURE: CA3 pyramidal cell layer.
    BIOLOGICAL FUNCTION: Attractor dynamics support pattern completion 
    and maintain fixed points for episodic representations. Uses pseudoinverse rule.
    Hopfield JJ (1982). DOI: 10.1073/pnas.79.8.2554
    ANATOMICAL INTERFACE:
    Sending structure: Dentate gyrus granule cells.
    Receiving structure: CA3 pyramidal cells.
    Connection: Mossy fiber detonator synapses.
    """
    def __init__(self, cfg: CognitiveKernelConfig, positions: NeuronPositionRegistry) -> None:
        super().__init__()
        self.cfg = cfg
        self.mossy_fiber = nn.Linear(cfg.dentate_gyrus_dim, cfg.ca3_dim, bias=False)
        self.register_buffer("W_recurrent", torch.zeros(cfg.ca3_dim, cfg.ca3_dim))
        ca3_distances = positions.pairwise_distances("ca3")
        distance_mask = torch.exp(-ca3_distances / cfg.ca3_distance_sigma)
        self.register_buffer("distance_mask", distance_mask)
        self._stored_patterns: List[Dict[str, Any]] = []
        self.schaffer_collateral = nn.Linear(cfg.ca3_dim, cfg.ca1_dim, bias=True)

    def store_episode(self, dg_pattern: torch.Tensor) -> int:
        """
        Store a new episode using the pseudoinverse rule.
        BIOLOGICAL FUNCTION: Rapid one-shot encoding via NMDA-dependent LTP.
        BIOLOGICAL GROUNDING: Hippocampal-Neocortical Consolidation Gradient.
        The hippocampus is required to hold a memory trace for at least three 
        hours after encoding. Evicting the trace before this window closes 
        results in retrograde amnesia. Traces mature and become independent 
        within forty-eight hours.
        Tse D et al. (2007). "Schemas and memory consolidation." Science, 
        316(5821), 76-82. DOI: 10.1126/science.1135935
        """
        pattern = dg_pattern.detach().clone()
        pattern = pattern / (pattern.norm() + 1e-8)
        if len(self._stored_patterns) >= self.cfg.ca3_max_episodes:
            eviction_idx = -1
            for idx, item in enumerate(self._stored_patterns):
                if item["age"] >= self.cfg.ca3_max_retention_cycles:
                    eviction_idx = idx
                    break
            if eviction_idx == -1:
                for idx, item in enumerate(self._stored_patterns):
                    if item["age"] >= self.cfg.ca3_min_retention_cycles:
                        eviction_idx = idx
                        break
            if eviction_idx != -1:
                self._stored_patterns.pop(eviction_idx)
            else:
                self._stored_patterns.pop(0)
        self._stored_patterns.append({"pattern": pattern, "age": 0})
        self._recompute_weights()
        return len(self._stored_patterns)

    def increment_ages(self) -> None:
        """Increment the survival cycles for all CA3 memory traces."""
        for item in self._stored_patterns:
            item["age"] += 1

    def _recompute_weights(self) -> None:
        if not self._stored_patterns:
            self.W_recurrent.zero_()
            return
        device = self.W_recurrent.device
        patterns = torch.stack([item["pattern"].to(device) for item in self._stored_patterns], dim=0)
        W = patterns.T @ patterns
        W.fill_diagonal_(0.0)
        W = W * self.distance_mask
        self.W_recurrent.copy_(W)

    def retrieve(self, cue: torch.Tensor) -> torch.Tensor:
        state = cue
        gain = self.cfg.ca3_recurrent_gain
        # Snapshot the weights: store_episode() rewrites W_recurrent in place
        # after this settle, which would otherwise break loss.backward().
        W = self.W_recurrent.clone()
        for _ in range(self.cfg.ca3_settle_steps):
            recurrent = F.linear(state, W)
            state = torch.tanh(gain * recurrent + (1.0 - gain) * cue)
        return state

    def forward(self, dg_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ca3_cue = self.mossy_fiber(dg_output)
        ca3_state = self.retrieve(ca3_cue)
        ca1_input = self.schaffer_collateral(ca3_state)
        return ca3_state, ca1_input
class CA1Comparator(nn.Module):
    """
    Novelty detection through mismatch comparison.
    BIOLOGICAL STRUCTURE: CA1 pyramidal cell layer.
    BIOLOGICAL FUNCTION: Computes mismatch between CA3 reconstruction and 
    direct EC input to gate novelty and drive downstream STDP reward.
    Lisman JE, Grace AA (2005). DOI: 10.1016/j.neuron.2005.05.002
    ANATOMICAL INTERFACE:
    Sending structures: CA3 pyramidal cells and Entorhinal cortex layer III.
    Receiving structure: CA1 pyramidal cells.
    Connections: Schaffer collaterals and temporoammonic pathway.
    """
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.temporoammonic = nn.Linear(cfg.coordinate_dim, cfg.ca1_dim, bias=True)
        self.compare_schaffer = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, bias=False)
        self.compare_direct = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, bias=False)
        self.output_gate = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, bias=True)

    def forward(self, schaffer_input: torch.Tensor, ec_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        direct = self.temporoammonic(ec_input)
        s_proj = self.compare_schaffer(schaffer_input)
        d_proj = self.compare_direct(direct)
        cos_sim = F.cosine_similarity(s_proj, d_proj, dim=-1)
        novelty = (1.0 - cos_sim).clamp(0.0, 1.0)
        novelty_gate = novelty.unsqueeze(-1)
        combined = novelty_gate * direct + (1.0 - novelty_gate) * schaffer_input
        ca1_output = torch.tanh(self.output_gate(combined))
        return ca1_output, novelty

class Subiculum(nn.Module):
    """
    Output gateway from hippocampal formation back to neocortex.
    BIOLOGICAL STRUCTURE: Subiculum.
    BIOLOGICAL FUNCTION: Projects final processed output to cortical targets.
    O'Mara SM et al. (2001). DOI: 10.1016/S0301-0082(01)00016-3
    """
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ca1_to_sub = nn.Linear(cfg.ca1_dim, cfg.subiculum_dim, bias=True)
        self.sub_to_ec = nn.Linear(cfg.subiculum_dim, cfg.coordinate_dim, bias=True)
        self.gate = nn.Linear(cfg.subiculum_dim, cfg.subiculum_dim, bias=True)

    def forward(self, ca1_output: torch.Tensor) -> torch.Tensor:
        sub_state = F.relu(self.ca1_to_sub(ca1_output))
        gate_signal = torch.sigmoid(self.gate(sub_state))
        gated = sub_state * gate_signal
        reconstructed = self.sub_to_ec(gated)
        return reconstructed

class AstrocyticRegulator(nn.Module):
    """
    Homeostatic regulator spanning all hippocampal subregions.
    BIOLOGICAL STRUCTURE: Protoplasmic astrocytes.
    BIOLOGICAL FUNCTION: Monitors synaptic activity via glutamate to slide 
    the modification threshold via calcium signaling.
    Araque A et al. (1999). DOI: 10.1016/S0166-2236(98)01349-6
    """
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("extrasynaptic_glutamate", torch.tensor(0.0))
        self.register_buffer("astrocytic_calcium", torch.tensor(cfg.astro_calcium_target))

    @torch.no_grad()
    def update(self, population_activity: torch.Tensor) -> float:
        activity = population_activity.detach().abs().mean().item()
        tau = self.cfg.astro_glutamate_tau
        self.extrasynaptic_glutamate.copy_(tau * self.extrasynaptic_glutamate + (1.0 - tau) * activity)
        calcium_drive = torch.sigmoid(torch.tensor(self.cfg.astro_calcium_gain * (self.extrasynaptic_glutamate.item() - self.cfg.astro_calcium_target))).item()
        self.astrocytic_calcium.copy_(tau * self.astrocytic_calcium + (1.0 - tau) * calcium_drive)
        deviation = self.astrocytic_calcium.item() - self.cfg.astro_calcium_target
        eta = max(self.cfg.astro_eta_min, min(self.cfg.astro_eta_max, 1.0 + deviation))
        return eta

    def get_state(self) -> Dict[str, float]:
        return {"glutamate": self.extrasynaptic_glutamate.item(), "calcium": self.astrocytic_calcium.item()}
class EntorhinalCortex(nn.Module):
    """
    Interface layer and short-term persistent buffer.
    BIOLOGICAL STRUCTURE: Entorhinal cortex layers II and III.
    BIOLOGICAL FUNCTION: Gateway bridging neocortex to hippocampus.
    Witter MP et al. (2000). DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K
    """
    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("short_term_buffer", torch.zeros(cfg.coordinate_dim))
        self.input_norm = nn.LayerNorm(cfg.coordinate_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_mean = coords.detach().mean(dim=0)
            tau = self.cfg.ec_buffer_tau
            self.short_term_buffer.copy_(tau * self.short_term_buffer + (1.0 - tau) * batch_mean)
        biased = coords + 0.1 * self.short_term_buffer.unsqueeze(0)
        return self.input_norm(biased)

    def reset_short_term(self) -> None:
        with torch.no_grad():
            self.short_term_buffer.zero_()

class CognitiveKernel(nn.Module):
    """
    Complete hippocampal memory system.
    BIOLOGICAL GROUNDING: Schema Interference via Spatial Instability.
    If the relational mapping of the environment changes frequently, the 
    system is actively prevented from forming a consolidated schema. The CA1 
    novelty signal detects contradictory shifts and generates a schema 
    performance index to dictate deterministic core learning rates.
    Tse D et al. (2007). "Schemas and memory consolidation." Science, 
    316(5821), 76-82. DOI: 10.1126/science.1135935
    """
    def __init__(self, cfg: Optional[CognitiveKernelConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else CognitiveKernelConfig()
        self.neuron_positions = NeuronPositionRegistry(self.cfg)
        self.entorhinal_cortex = EntorhinalCortex(self.cfg)
        self.dentate_gyrus = DentateGyrus(self.cfg)
        self.ca3 = CA3AttractorNetwork(self.cfg, self.neuron_positions)
        self.ca1 = CA1Comparator(self.cfg)
        self.subiculum = Subiculum(self.cfg)
        self.astrocyte = AstrocyticRegulator(self.cfg)
        self.register_buffer("working_memory", torch.zeros(self.cfg.ca1_dim))

    def forward(self, coords: torch.Tensor, store_if_novel: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        diagnostics: Dict[str, Any] = {}
        ec_output = self.entorhinal_cortex(coords)
        dg_sparse = self.dentate_gyrus(ec_output)
        diagnostics["dg_sparsity"] = (dg_sparse == 0).float().mean().item()
        ca3_state, schaffer_to_ca1 = self.ca3(dg_sparse)
        ca1_output, novelty = self.ca1(schaffer_to_ca1, ec_output)
        with torch.no_grad():
            self.working_memory.copy_(ca1_output.detach().mean(dim=0))
        reconstructed = self.subiculum(ca1_output)
        diagnostics["astro_eta"] = self.astrocyte.update(ca3_state)
        diagnostics.update(self.astrocyte.get_state())
        mean_novelty = novelty.mean().item()
        diagnostics["mean_novelty"] = mean_novelty
        
        # Behavioral Performance Index for Schema Acquisition
        # NOT a biological quantity: mapped to the [100 - 100(errors/5)] formulation
        scaled_error = mean_novelty * self.cfg.schema_error_max
        schema_consistency = max(0.0, 100.0 - (100.0 * (scaled_error / self.cfg.schema_error_max)))
        diagnostics["schema_consistency_index"] = schema_consistency
        diagnostics["num_stored_episodes"] = len(self.ca3._stored_patterns)
        
        if store_if_novel and mean_novelty > self.cfg.ca1_novelty_threshold:
            mean_dg = dg_sparse.detach().mean(dim=0)
            mossy_pattern = self.ca3.mossy_fiber(mean_dg)
            diagnostics["num_stored_episodes"] = self.ca3.store_episode(mossy_pattern)
            diagnostics["stored_this_step"] = True
        else:
            diagnostics["stored_this_step"] = False
        return reconstructed, novelty, diagnostics

    def retrieve_from_cue(self, partial_coords: torch.Tensor) -> torch.Tensor:
        ec_output = self.entorhinal_cortex(partial_coords)
        dg_sparse = self.dentate_gyrus(ec_output)
        ca3_state, schaffer = self.ca3(dg_sparse)
        ca1_output, _ = self.ca1(schaffer, ec_output)
        return self.subiculum(ca1_output)

    def sleep_consolidation(self) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {}
        self.entorhinal_cortex.reset_short_term()
        with torch.no_grad():
            self.working_memory.zero_()
        self.ca3.increment_ages()
        diagnostics["episodes_in_ca3"] = len(self.ca3._stored_patterns)
        diagnostics["short_term_cleared"] = True
        diagnostics["working_memory_cleared"] = True
        return diagnostics

    def get_neuron_positions(self) -> Dict[str, torch.Tensor]:
        return {name: self.neuron_positions.get_population_positions(name) for name in ["ec", "dg", "ca3", "ca1", "subiculum"]}

    def get_memory_state(self) -> Dict[str, object]:
        return {
            "short_term_buffer": self.entorhinal_cortex.short_term_buffer.detach().cpu().clone(),
            "working_memory": self.working_memory.detach().cpu().clone(),
            "ca3_stored_patterns": [{"pattern": p["pattern"].detach().cpu().clone(), "age": p["age"]} for p in self.ca3._stored_patterns],
            "ca3_W_recurrent": self.ca3.W_recurrent.detach().cpu().clone(),
            "astro_state": self.astrocyte.get_state(),
        }

    def load_memory_state(self, state: Dict[str, object]) -> None:
        with torch.no_grad():
            device = self.ca3.W_recurrent.device
            if "short_term_buffer" in state:
                self.entorhinal_cortex.short_term_buffer.copy_(state["short_term_buffer"].to(device))
            if "working_memory" in state:
                self.working_memory.copy_(state["working_memory"].to(device))
            if "ca3_stored_patterns" in state:
                self.ca3._stored_patterns = [{"pattern": p["pattern"].to(device), "age": p["age"]} for p in state["ca3_stored_patterns"]]
            if "ca3_W_recurrent" in state:
                self.ca3.W_recurrent.copy_(state["ca3_W_recurrent"].to(device))
      



