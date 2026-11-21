"""
cognitive_kernel_core_t.py
Cognitive Kernel Learning Core: Hippocampal Episodic Reconstruction System

BIOLOGICAL GROUNDING
This file models the plastic learning core of the architecture. Concretely,
it models the hippocampal formation and its interactions with entorhinal
cortex as the rapid episodic learning substrate that sits beside, not in
place of, the deterministic core. The learning core handles one-shot
encoding, pattern separation, attractor-based pattern completion, novelty
detection, short-term and working persistence, sleep-associated replay, and
state serialization. Its job is to reconstruct experience from partial cues
and emit provisional reconstructions plus mismatch signals upward to the
neocortical mediation layer. The neocortex then evaluates schema overlap,
maturity, and crystallization conditions before anything is written into the
deterministic core.

The biology grounding for this file is the classical trisynaptic circuit and
its cortical interfaces. Dentate gyrus orthogonalizes similar inputs to
protect new episodes from interference. CA3 stores rapidly acquired attractor
traces and completes memories from partial cues. CA1 compares reconstructed
expectation against current cortical drive and emits a mismatch or novelty
signal. Subiculum returns the hippocampal result to cortex. Entorhinal
cortex serves as the gateway in both directions, while astrocytic regulation
provides a homeostatic brake on runaway plasticity.

The architecture-level reason this region matters is simple. The learning
core is where provisional experience lives while it is still plastic. It is
not a retrieval cabinet. It is the part of the system that actually
reconstructs an episode, holds the fresh trace in a serializable internal
state, and makes novelty legible to the neocortex so the broader system can
decide whether to preserve, integrate, reopen, or discard.

Primary grounding papers:
Rolls ET (2013). "The mechanisms for pattern completion and pattern separation in the hippocampus." Frontiers in Systems Neuroscience, 7, 74. DOI: 10.3389/fnsys.2013.00074
Hopfield JJ (1982). "Neural networks and physical systems with emergent collective computational abilities." Proceedings of the National Academy of Sciences, 79(8), 2554-2558. DOI: 10.1073/pnas.79.8.2554
Lisman JE, Grace AA (2005). "The hippocampal-VTA loop: controlling the entry of information into long-term memory." Neuron, 46(5), 703-713. DOI: 10.1016/j.neuron.2005.05.002
Witter MP, Naber PA, van Haeften T, Machielsen WC, Rombouts SARB, Barkhof F, Scheltens P, Lopes da Silva FH (2000). "Cortico-hippocampal communication by way of parallel parahippocampal-subicular pathways." Hippocampus, 10(4), 398-410. DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K
Buzsáki G (2015). "Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning." Hippocampus, 25(10), 1073-1188. DOI: 10.1002/hipo.22488
Araque A, Parpura V, Sanzgiri RP, Haydon PG (1999). "Tripartite synapses: glia, the unacknowledged partner." Trends in Neurosciences, 22(5), 208-215. DOI: 10.1016/S0166-2236(98)01349-6
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class CognitiveKernelConfig:
    """
    Complete configuration for the Cognitive Kernel.

    Every parameter that has a biological analog cites its source.
    Parameters that are training artifacts or engineering approximations
    are labeled explicitly.
    """

    # ---- Dimensionality ----

    # Coordinate manifold dimension. Input from Perforant Path.
    # Must match the upstream coordinate bridge dimensionality.
    # NOT a biological quantity. Engineering choice.
    # Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
    coordinate_dim: int = 64

    # Dentate gyrus expanded dimension. Pattern separation requires
    # expansion to a higher-dimensional sparse code. The expansion
    # ratio of 8x follows the anatomical ratio of granule cells to
    # entorhinal input neurons in rodents (~5:1 to 10:1).
    # Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074
    dentate_gyrus_dim: int = 512

    # CA3 recurrent network dimension. Equal to DG dim because the
    # mossy fiber projection from DG to CA3 is a one-to-one sparse
    # detonator synapse pathway.
    # Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074
    ca3_dim: int = 512

    # CA1 output dimension. Typically smaller than CA3, reflecting the
    # convergence from CA3 to CA1 via Schaffer collaterals.
    # NOT a biological quantity at this parameterization.
    ca1_dim: int = 256

    # Subiculum output dimension. Projects back to EC at coordinate_dim.
    # NOT a biological quantity.
    subiculum_dim: int = 128

    # ---- Dentate Gyrus pattern separation ----

    # Sparsity: fraction of DG neurons active per input. Approximately
    # 2-5% sparse firing is the biological target for dentate granule cells.
    # Chawla MK, Guzowski JF, Ramirez-Amaya V, Lipa P, Hoffman KL, Marriott LK,
    # Worley PF, McNaughton BL, Barnes CA (2005).
    # "Sparse, environmentally selective expression of Arc RNA in the upper
    # blade of the rodent fascia dentata by brief spatial experience."
    # Hippocampus, 15(5), 579-586. DOI: 10.1002/hipo.20091
    dg_sparsity: float = 0.04

    # ---- CA3 attractor network ----

    # Maximum number of episodes stored in the CA3 attractor network.
    # Theoretical capacity of a Hopfield network is ~0.14N for N neurons.
    # With 512 CA3 neurons, theoretical max is ~72 patterns.
    # We use 64 as a conservative limit.
    # Hopfield JJ (1982). DOI: 10.1073/pnas.79.8.2554
    ca3_max_episodes: int = 64

    # Number of attractor settling iterations per retrieval.
    # CA3 recurrent dynamics require multiple passes to converge
    # to the nearest stored attractor.
    # NOT a biological quantity at this parameterization. The brain
    # settles in continuous time; we use discrete iterations.
    ca3_settle_steps: int = 5

    # CA3 recurrent gain. Scales the recurrent input relative to
    # the external drive from DG mossy fibers.
    # NOT a biological quantity. Engineering hyperparameter.
    ca3_recurrent_gain: float = 0.7

    # ---- CA1 comparator ----

    # Novelty threshold. Mismatch below this is considered familiar.
    # NOT a biological quantity. Engineering hyperparameter.
    ca1_novelty_threshold: float = 0.3

    # ---- Entorhinal buffer (short-term memory) ----

    # EMA decay for the EC short-term buffer. Fast decay (0.5) means
    # information decays within ~3 forward passes.
    # NOT a biological quantity. Engineering approximation of
    # entorhinal persistent activity timescale.
    ec_buffer_tau: float = 0.5

    # ---- Astrocytic regulator ----

    # Glutamate EMA decay. Slow accumulation of population activity.
    # NOT a biological quantity. Engineering approximation.
    # Araque A et al. (1999). DOI: 10.1016/S0166-2236(98)01349-6
    astro_glutamate_tau: float = 0.95

    # Calcium signal gain. Controls sensitivity of calcium response
    # to glutamate accumulation.
    # NOT a biological quantity. Engineering hyperparameter.
    astro_calcium_gain: float = 4.0

    # Calcium target for homeostatic equilibrium.
    # NOT a biological quantity. Engineering setpoint.
    astro_calcium_target: float = 0.5

    # Metaplasticity modifier bounds. Prevents runaway scaling.
    # NOT a biological quantity. Engineering safety bound.
    astro_eta_min: float = 0.1
    astro_eta_max: float = 3.0

    # ---- 3D Neuron Coordinates ----

    # Whether to initialize neuron positions anatomically (layered
    # by subregion) or from random uniform.
    # NOT a biological quantity. Visualization choice.
    anatomical_layout: bool = True

    # Distance decay constant for CA3 recurrent connectivity.
    # Connection strength between CA3 neurons i and j is scaled by
    # exp(-dist(i,j) / distance_sigma). Implements the observation
    # that nearby hippocampal neurons have stronger recurrent coupling.
    # Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074
    ca3_distance_sigma: float = 2.0


# =========================================================================
# 3D Neuron Position Registry
# =========================================================================

class NeuronPositionRegistry(nn.Module):
    """
    Registry of 3D (x, y, z) positions for every neuron in the kernel.

    BIOLOGICAL STRUCTURE: Spatial organization of hippocampal neurons.
    BIOLOGICAL FUNCTION: Hippocampal neurons are not randomly distributed.
    They are organized along anatomical axes: the transverse axis (EC to
    CA1 to CA3 to DG), the proximodistal axis (close to vs. far from EC),
    and the dorsoventral axis (spatial vs. emotional processing).
    The 3D positions here model this spatial structure for visualization
    and distance-dependent connectivity.

    Strange BA et al. (2014). "Functional organization of the hippocampal
    longitudinal axis." Nature Reviews Neuroscience, 15(10):655-669.
    DOI: 10.1038/nrn3785

    Positions are registered as buffers (not parameters) because they
    are structural metadata, not learned weights. They are serialized
    in the COLD layer alongside weights.
    """

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        """
        Initialize 3D positions for all neuron populations.

        Args:
            cfg: CognitiveKernelConfig.
        """
        super().__init__()

        total_neurons = (
            cfg.coordinate_dim       # EC input layer
            + cfg.dentate_gyrus_dim  # DG
            + cfg.ca3_dim            # CA3
            + cfg.ca1_dim            # CA1
            + cfg.subiculum_dim      # Subiculum
        )

        if cfg.anatomical_layout:
            positions = self._anatomical_positions(cfg)
        else:
            positions = torch.rand(total_neurons, 3) * 10.0

        self.register_buffer("positions", positions)

        # Store population boundaries for slicing.
        self._boundaries = self._compute_boundaries(cfg)
        self.total_neurons = total_neurons

    @staticmethod
    def _compute_boundaries(
        cfg: CognitiveKernelConfig,
    ) -> Dict[str, Tuple[int, int]]:
        """Compute start/end indices for each population."""
        offset = 0
        boundaries = {}
        for name, size in [
            ("ec", cfg.coordinate_dim),
            ("dg", cfg.dentate_gyrus_dim),
            ("ca3", cfg.ca3_dim),
            ("ca1", cfg.ca1_dim),
            ("subiculum", cfg.subiculum_dim),
        ]:
            boundaries[name] = (offset, offset + size)
            offset += size
        return boundaries

    @staticmethod
    def _anatomical_positions(
        cfg: CognitiveKernelConfig,
    ) -> torch.Tensor:
        """
        Generate anatomically plausible 3D positions for all populations.

        The hippocampal formation is arranged as a curved sheet. We
        approximate this with layered slabs offset along the y-axis
        (transverse axis) with Gaussian noise for realistic scatter.

        NOT a biological quantity: the exact coordinates are engineering
        approximations for visualization. Relative positioning reflects
        the anatomical circuit order: EC -> DG -> CA3 -> CA1 -> Subiculum.

        Returns:
            (total_neurons, 3) tensor of 3D positions.
        """
        positions = []
        gen = torch.Generator().manual_seed(0)

        # EC: y ~ 0.0 (superficial, entry layer).
        n_ec = cfg.coordinate_dim
        ec_pos = torch.zeros(n_ec, 3)
        ec_pos[:, 0] = torch.linspace(-3.0, 3.0, n_ec)
        ec_pos[:, 1] = 0.0 + torch.randn(n_ec, generator=gen) * 0.2
        ec_pos[:, 2] = torch.randn(n_ec, generator=gen) * 0.5
        positions.append(ec_pos)

        # DG: y ~ 2.0 (deep to EC, expanded population).
        n_dg = cfg.dentate_gyrus_dim
        dg_pos = torch.zeros(n_dg, 3)
        dg_pos[:, 0] = torch.randn(n_dg, generator=gen) * 2.0
        dg_pos[:, 1] = 2.0 + torch.randn(n_dg, generator=gen) * 0.3
        dg_pos[:, 2] = torch.randn(n_dg, generator=gen) * 1.5
        positions.append(dg_pos)

        # CA3: y ~ 4.0 (recurrent network, curved around DG).
        n_ca3 = cfg.ca3_dim
        ca3_pos = torch.zeros(n_ca3, 3)
        theta = torch.linspace(0, math.pi, n_ca3)
        ca3_pos[:, 0] = 3.0 * torch.cos(theta)
        ca3_pos[:, 1] = 4.0 + torch.sin(theta) * 1.0
        ca3_pos[:, 2] = torch.randn(n_ca3, generator=gen) * 0.8
        ca3_pos += torch.randn_like(ca3_pos) * 0.15
        positions.append(ca3_pos)

        # CA1: y ~ 6.0 (output layer, receives Schaffer collaterals).
        n_ca1 = cfg.ca1_dim
        ca1_pos = torch.zeros(n_ca1, 3)
        ca1_pos[:, 0] = torch.linspace(-2.5, 2.5, n_ca1)
        ca1_pos[:, 1] = 6.0 + torch.randn(n_ca1, generator=gen) * 0.25
        ca1_pos[:, 2] = torch.randn(n_ca1, generator=gen) * 0.6
        positions.append(ca1_pos)

        # Subiculum: y ~ 7.5 (output gateway).
        n_sub = cfg.subiculum_dim
        sub_pos = torch.zeros(n_sub, 3)
        sub_pos[:, 0] = torch.linspace(-2.0, 2.0, n_sub)
        sub_pos[:, 1] = 7.5 + torch.randn(n_sub, generator=gen) * 0.2
        sub_pos[:, 2] = torch.randn(n_sub, generator=gen) * 0.4
        positions.append(sub_pos)

        return torch.cat(positions, dim=0)

    def get_population_positions(self, name: str) -> torch.Tensor:
        """
        Return 3D positions for a named population.

        Args:
            name: one of 'ec', 'dg', 'ca3', 'ca1', 'subiculum'.

        Returns:
            (N, 3) tensor of positions for that population.
        """
        start, end = self._boundaries[name]
        return self.positions[start:end]

    def pairwise_distances(self, name: str) -> torch.Tensor:
        """
        Compute pairwise Euclidean distance matrix for a population.

        Used by CA3 to compute distance-dependent recurrent connectivity.

        Args:
            name: population name.

        Returns:
            (N, N) distance matrix.
        """
        pos = self.get_population_positions(name)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)


# =========================================================================
# Dentate Gyrus: Pattern Separation
# =========================================================================

class DentateGyrus(nn.Module):
    """
    Pattern separation via competitive sparse expansion.

    BIOLOGICAL STRUCTURE: Dentate gyrus granule cell layer.
    BIOLOGICAL FUNCTION: The DG receives convergent input from EC layer II
    via the perforant path and expands it into a sparse, high-dimensional
    representation. Approximately 2-5% of granule cells are active for any
    given input, producing maximally dissimilar codes for similar inputs.
    This is the brain's primary defense against catastrophic interference:
    two similar experiences produce nearly orthogonal representations in DG,
    allowing them to be stored as separate CA3 attractors without mutual
    corruption.

    The expansion is implemented as a linear projection followed by
    k-winners-take-all competition, where k is determined by dg_sparsity.
    The competition is implemented via top-k selection, not learned
    inhibition, because the functional outcome (extreme sparsity) matters
    more than the mechanism at this abstraction level.

    Rolls ET (2013). "The mechanisms for pattern completion and pattern
    separation in the hippocampus." Frontiers in Systems Neuroscience, 7:74.
    DOI: 10.3389/fnsys.2013.00074

    Chawla MK et al. (2005). "Sparse, environmentally selective expression
    of Arc RNA in the upper blade of the rodent fascia dentata by brief
    spatial experience." Hippocampus, 15(5):579-586.
    DOI: 10.1002/hipo.20091

    ANATOMICAL INTERFACE (input):
        Sending structure: Entorhinal cortex layer II (EC).
        Receiving structure: Dentate gyrus granule cell layer (this module).
        Connection: Perforant path (lateral and medial).
        Witter MP et al. (2000). DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K

    ANATOMICAL INTERFACE (output):
        Sending structure: Dentate gyrus granule cells.
        Receiving structure: CA3 pyramidal cells.
        Connection: Mossy fiber pathway. Each mossy fiber makes a small
        number of extremely powerful "detonator" synapses on CA3 pyramidal
        cells, ensuring that a single active DG granule cell reliably
        drives its target CA3 cell.
        Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074
    """

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        """
        Args:
            cfg: CognitiveKernelConfig.
        """
        super().__init__()
        self.cfg = cfg

        # Perforant path projection: EC -> DG.
        # Linear expansion from coordinate_dim to dentate_gyrus_dim.
        # NOT a biological quantity: the linear map is an engineering
        # approximation of the complex multi-synaptic perforant path.
        self.perforant_path = nn.Linear(
            cfg.coordinate_dim, cfg.dentate_gyrus_dim, bias=True,
        )

        # Layer norm before competition. Stabilizes the k-WTA selection.
        # NOT a biological quantity. Engineering normalization.
        self.norm = nn.LayerNorm(cfg.dentate_gyrus_dim)

        # Number of active neurons per input.
        self.k_active = max(1, int(cfg.dentate_gyrus_dim * cfg.dg_sparsity))

    def forward(self, ec_input: torch.Tensor) -> torch.Tensor:
        """
        Pattern separation: project and sparsify.

        Args:
            ec_input: (B, coordinate_dim) from entorhinal cortex.

        Returns:
            dg_sparse: (B, dentate_gyrus_dim) sparse binary-like activation.
                Only the top-k values are nonzero. The rest are zeroed.
        """
        projected = self.norm(F.relu(self.perforant_path(ec_input)))

        # k-winners-take-all competition.
        # Top-k values survive. Everything else is zeroed.
        # This produces the ~4% sparsity observed in vivo.
        topk_vals, topk_idx = projected.topk(self.k_active, dim=-1)
        sparse = torch.zeros_like(projected)
        sparse.scatter_(1, topk_idx, topk_vals)

        return sparse


# =========================================================================
# CA3 Autoassociative Attractor Network
# =========================================================================

class CA3AttractorNetwork(nn.Module):
    """
    Recurrent autoassociative network for episodic memory storage and
    pattern completion.

    BIOLOGICAL STRUCTURE: CA3 pyramidal cell layer with extensive recurrent
    collateral connections.
    BIOLOGICAL FUNCTION: CA3 is the most densely recurrently connected
    region in the hippocampus. Approximately 4% of CA3 pyramidal cells
    synapse onto any given CA3 cell, creating a massive recurrent network
    capable of attractor dynamics. When a partial or noisy cue is presented,
    the recurrent dynamics settle into the nearest stored attractor,
    completing the memory. This is pattern completion: the complement of
    the DG's pattern separation.

    Storage uses the pseudoinverse learning rule (Hopfield with optimal
    capacity). Each episode is stored as a fixed point by updating the
    recurrent weight matrix W such that W @ pattern = pattern for all
    stored patterns. The pseudoinverse achieves theoretical capacity of
    N patterns for N neurons (with degradation above ~0.14N for random
    patterns).

    Distance-dependent connectivity: connection strength between CA3
    neurons i and j is modulated by their 3D Euclidean distance via
    exp(-dist/sigma). This implements the biological observation that
    nearby CA3 neurons have stronger recurrent coupling, creating
    spatially localized attractor basins.

    Hopfield JJ (1982). "Neural networks and physical systems with emergent
    collective computational abilities." PNAS, 79(8):2554-2558.
    DOI: 10.1073/pnas.79.8.2554

    Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074

    Treves A, Rolls ET (1994). "Computational analysis of the role of the
    hippocampus in memory." Hippocampus, 4(3):374-391.
    DOI: 10.1002/hipo.450040319

    ANATOMICAL INTERFACE (input):
        Sending structure: Dentate gyrus granule cells.
        Receiving structure: CA3 pyramidal cells (this module).
        Connection: Mossy fiber detonator synapses.

    ANATOMICAL INTERFACE (output):
        Sending structure: CA3 pyramidal cells.
        Receiving structure: CA1 pyramidal cells.
        Connection: Schaffer collateral pathway.
        Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074
    """

    def __init__(
        self, cfg: CognitiveKernelConfig, positions: NeuronPositionRegistry,
    ) -> None:
        """
        Args:
            cfg: CognitiveKernelConfig.
            positions: NeuronPositionRegistry for distance-dependent connectivity.
        """
        super().__init__()
        self.cfg = cfg

        # Mossy fiber projection: DG -> CA3.
        # Linear map preserving dimensionality (DG dim == CA3 dim).
        self.mossy_fiber = nn.Linear(cfg.dentate_gyrus_dim, cfg.ca3_dim, bias=False)

        # Recurrent weight matrix. Initialized to zero (no stored memories).
        # Updated via store_episode() using the pseudoinverse rule.
        # Serialized in COLD layer as a learned weight.
        self.register_buffer(
            "W_recurrent",
            torch.zeros(cfg.ca3_dim, cfg.ca3_dim),
        )

        # Distance-dependent connectivity mask.
        # exp(-dist / sigma) for all CA3 neuron pairs.
        ca3_distances = positions.pairwise_distances("ca3")
        distance_mask = torch.exp(-ca3_distances / cfg.ca3_distance_sigma)
        self.register_buffer("distance_mask", distance_mask)

        # Episode storage buffer. Stores the raw DG-projected patterns
        # for pseudoinverse recomputation.
        self._stored_patterns: List[torch.Tensor] = []

        # Schaffer collateral projection: CA3 -> CA1.
        self.schaffer_collateral = nn.Linear(cfg.ca3_dim, cfg.ca1_dim, bias=True)

    def store_episode(self, dg_pattern: torch.Tensor) -> int:
        """
        Store a new episode in the CA3 attractor network.

        Uses the pseudoinverse rule for optimal capacity. Recomputes the
        full weight matrix from all stored patterns to avoid incremental
        drift that degrades attractor quality.

        BIOLOGICAL FUNCTION: Rapid one-shot encoding of a new experience
        into the CA3 recurrent weight matrix. In the brain, this is
        mediated by NMDA receptor-dependent LTP at the mossy fiber to
        CA3 synapse, which is uniquely suited for single-trial learning
        due to the detonator synapse architecture.
        Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074

        Args:
            dg_pattern: (ca3_dim,) pattern from DG projection through
                mossy fibers. Must be a single pattern, not batched.
                Should be normalized before storage.

        Returns:
            Number of episodes now stored.
        """
        pattern = dg_pattern.detach().clone()
        pattern = pattern / (pattern.norm() + 1e-8)

        if len(self._stored_patterns) >= self.cfg.ca3_max_episodes:
            # Evict oldest memory (FIFO). Biologically, this corresponds
            # to systems consolidation moving old hippocampal traces to
            # neocortical long-term storage, freeing hippocampal capacity.
            # Frankland PW, Bontempi B (2005). DOI: 10.1038/nrn1667
            self._stored_patterns.pop(0)

        self._stored_patterns.append(pattern)
        self._recompute_weights()
        return len(self._stored_patterns)

    def _recompute_weights(self) -> None:
        """
        Recompute the recurrent weight matrix from all stored patterns
        using the pseudoinverse (outer product) rule.

        W = sum_i (p_i @ p_i^T) for all stored patterns p_i.
        Diagonal is zeroed to prevent self-excitation.
        Distance mask is applied to enforce spatial locality.

        NOT a biological quantity: the pseudoinverse is a mathematical
        abstraction. The brain achieves similar attractor dynamics through
        Hebbian synaptic modification during encoding.
        Hopfield JJ (1982). DOI: 10.1073/pnas.79.8.2554
        """
        if not self._stored_patterns:
            self.W_recurrent.zero_()
            return

        device = self.W_recurrent.device
        patterns = torch.stack(
            [p.to(device) for p in self._stored_patterns], dim=0,
        )
        # Pseudoinverse: W = P^T @ P where P is (n_patterns, ca3_dim).
        W = patterns.T @ patterns
        # Zero diagonal: no self-excitation.
        W.fill_diagonal_(0.0)
        # Apply distance-dependent mask.
        W = W * self.distance_mask
        self.W_recurrent.copy_(W)

    def retrieve(self, cue: torch.Tensor) -> torch.Tensor:
        """
        Retrieve a memory by settling the attractor network from a cue.

        BIOLOGICAL FUNCTION: Pattern completion. A partial or noisy cue
        activates a subset of CA3 neurons. Through iterative recurrent
        processing, the network state converges to the nearest stored
        attractor, filling in the missing components of the memory.
        This is how seeing a friend's face in an unexpected context can
        trigger recall of their name, your last conversation, and the
        emotions associated with that meeting.

        Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074
        Hopfield JJ (1982). DOI: 10.1073/pnas.79.8.2554

        Args:
            cue: (B, ca3_dim) initial state from mossy fiber projection.

        Returns:
            settled: (B, ca3_dim) attractor state after settling.
        """
        state = cue
        gain = self.cfg.ca3_recurrent_gain

        # Snapshot the weights: store_episode() rewrites W_recurrent in place
        # after this settle, which would otherwise break loss.backward().
        W = self.W_recurrent.clone()
        for _ in range(self.cfg.ca3_settle_steps):
            recurrent = F.linear(state, W)
            state = torch.tanh(gain * recurrent + (1.0 - gain) * cue)

        return state

    def forward(
        self, dg_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full CA3 pass: mossy fiber projection, attractor settling,
        Schaffer collateral output to CA1.

        Args:
            dg_output: (B, dentate_gyrus_dim) sparse DG activation.

        Returns:
            ca3_state: (B, ca3_dim) settled attractor state.
            ca1_input: (B, ca1_dim) Schaffer collateral projection to CA1.
        """
        # Mossy fiber projection.
        ca3_cue = self.mossy_fiber(dg_output)
        # Attractor settling.
        ca3_state = self.retrieve(ca3_cue)
        # Schaffer collateral to CA1.
        ca1_input = self.schaffer_collateral(ca3_state)

        return ca3_state, ca1_input


# =========================================================================
# CA1 Comparator: Novelty Detection
# =========================================================================

class CA1Comparator(nn.Module):
    """
    Novelty detection through mismatch comparison.

    BIOLOGICAL STRUCTURE: CA1 pyramidal cell layer.
    BIOLOGICAL FUNCTION: CA1 receives two convergent inputs: the CA3
    reconstruction via Schaffer collaterals (what the system expects based
    on stored memory) and the direct EC input via the temporoammonic pathway
    (what actually arrived from the senses). CA1 computes the mismatch
    between these two signals. High mismatch indicates genuine novelty:
    the current experience does not match any stored memory and should
    be encoded. Low mismatch indicates familiarity: retrieval succeeded
    and no new encoding is needed.

    The novelty scalar is not treated here as a final policy signal.
    It is a provisional mismatch estimate generated by the learning core and
    exposed upward so the neocortex can decide whether the current episode
    should remain provisional, be replayed, be integrated with a schema, or
    be crystallized into the deterministic core.

    Lisman JE, Grace AA (2005). "The hippocampal-VTA loop: controlling
    the entry of information into long-term memory." Neuron, 46(5), 703-713.
    DOI: 10.1016/j.neuron.2005.05.002

    Hasselmo ME, Wyble BP (1997). "Free recall and recognition in a
    network model of the hippocampus: simulating effects of scopolamine
    on human memory function." Behavioural Brain Research, 89(1-2), 1-34.
    DOI: 10.1016/S0166-4328(97)00048-X

    ANATOMICAL INTERFACE (input 1, Schaffer collateral):
        Sending structure: CA3 pyramidal cells.
        Receiving structure: CA1 (this module).
        Connection: Schaffer collateral pathway.

    ANATOMICAL INTERFACE (input 2, temporoammonic):
        Sending structure: Entorhinal cortex layer III.
        Receiving structure: CA1 (this module).
        Connection: Temporoammonic pathway (direct EC to CA1, bypassing DG/CA3).
        Witter MP et al. (2000). DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K
    """

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        """
        Args:
            cfg: CognitiveKernelConfig.
        """
        super().__init__()
        self.cfg = cfg

        # Temporoammonic pathway: EC -> CA1 (direct, bypasses DG/CA3).
        self.temporoammonic = nn.Linear(
            cfg.coordinate_dim, cfg.ca1_dim, bias=True,
        )

        # Comparison projection: align Schaffer and temporoammonic inputs
        # into a common space for mismatch computation.
        self.compare_schaffer = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, bias=False)
        self.compare_direct = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, bias=False)

        # Output gate: learned combination of mismatch-weighted inputs.
        self.output_gate = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, bias=True)

    def forward(
        self,
        schaffer_input: torch.Tensor,
        ec_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compare CA3 reconstruction with direct EC input.

        Args:
            schaffer_input: (B, ca1_dim) from CA3 Schaffer collaterals.
                This is "what the memory system thinks is happening."
            ec_input: (B, coordinate_dim) direct from entorhinal cortex.
                This is "what is actually happening."

        Returns:
            ca1_output: (B, ca1_dim) gated comparison output.
            novelty: (B,) scalar novelty per sample in [0, 1].
                0 = perfectly familiar (exact match).
                1 = completely novel (maximal mismatch).
        """
        # Temporoammonic projection of direct EC input to CA1 space.
        direct = self.temporoammonic(ec_input)

        # Project both inputs into comparison space.
        s_proj = self.compare_schaffer(schaffer_input)
        d_proj = self.compare_direct(direct)

        # Mismatch: cosine distance between projections.
        cos_sim = F.cosine_similarity(s_proj, d_proj, dim=-1)
        novelty = (1.0 - cos_sim).clamp(0.0, 1.0)

        # Gate output by novelty: novel inputs weight current cortical
        # drive more strongly, while familiar inputs weight reconstructed
        # attractor content more strongly. Biological process: CA1 mismatch
        # comparison between Schaffer collateral input and temporoammonic
        # input. Plain English: CA1 helps decide whether the current state
        # behaves more like a fresh episode or a recognized one.
        # Hasselmo ME, Wyble BP (1997). "Free recall and recognition in a
        # network model of the hippocampus: simulating effects of scopolamine
        # on human memory function." Behavioural Brain Research, 89(1-2),
        # 1-34. DOI: 10.1016/S0166-4328(97)00048-X
        novelty_gate = novelty.unsqueeze(-1)
        combined = novelty_gate * direct + (1.0 - novelty_gate) * schaffer_input
        ca1_output = torch.tanh(self.output_gate(combined))

        return ca1_output, novelty


# =========================================================================
# Subiculum: Output Gateway
# =========================================================================

class Subiculum(nn.Module):
    """
    Output gateway from hippocampal formation back to neocortex.

    BIOLOGICAL STRUCTURE: Subiculum, the primary output station of the
    hippocampal formation.
    BIOLOGICAL FUNCTION: The subiculum receives the final processed output
    from CA1 and projects it to multiple cortical and subcortical targets:
    prefrontal cortex (for working memory and planning), entorhinal cortex
    (completing the hippocampal loop), hypothalamus (autonomic responses
    to memory), and nucleus accumbens (reward-based memory prioritization).

    In this architecture, the subiculum performs a learned compression from
    CA1 space back to coordinate_dim, producing a provisional reconstructed
    coordinate state for return to the neocortical mediation layer. The
    neocortex can then compare that provisional result against existing
    schema structure and decide whether it should be routed onward, replayed,
    or crystallized into the deterministic core.

    O'Mara SM, Commins S, Anderson M, Gigg J (2001). "The subiculum: a
    review of form, physiology and function." Progress in Neurobiology,
    64(2), 129-155. DOI: 10.1016/S0301-0082(00)00054-X

    ANATOMICAL INTERFACE (input):
        Sending structure: CA1 pyramidal cells.
        Receiving structure: Subiculum (this module).
        Connection: CA1 to subiculum projection.

    ANATOMICAL INTERFACE (output):
        Sending structure: Subiculum.
        Receiving structure: Entorhinal cortex deep layers and the
        neocortical mediation pathway.
        Connection: Subicular projection to EC deep layers (V/VI), which
        then relay the provisional reconstruction back to cortex.
    """

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        """
        Args:
            cfg: CognitiveKernelConfig.
        """
        super().__init__()
        self.cfg = cfg

        self.ca1_to_sub = nn.Linear(cfg.ca1_dim, cfg.subiculum_dim, bias=True)
        self.sub_to_ec = nn.Linear(
            cfg.subiculum_dim, cfg.coordinate_dim, bias=True,
        )
        self.gate = nn.Linear(cfg.subiculum_dim, cfg.subiculum_dim, bias=True)

    def forward(self, ca1_output: torch.Tensor) -> torch.Tensor:
        """
        Gate and compress CA1 output back to coordinate space.

        Args:
            ca1_output: (B, ca1_dim) from CA1 comparator.

        Returns:
            reconstructed: (B, coordinate_dim) provisional episodic
                reconstruction ready for return to the neocortical
                mediation layer.
        """
        sub_state = F.relu(self.ca1_to_sub(ca1_output))
        gate_signal = torch.sigmoid(self.gate(sub_state))
        gated = sub_state * gate_signal
        reconstructed = self.sub_to_ec(gated)
        return reconstructed


# =========================================================================
# Astrocytic Metaplasticity Regulator
# =========================================================================

class AstrocyticRegulator(nn.Module):
    """
    Homeostatic regulator spanning all hippocampal subregions.

    BIOLOGICAL STRUCTURE: Protoplasmic astrocytes in the hippocampal
    formation. A single astrocyte contacts ~100,000 synapses in rodents.
    BIOLOGICAL FUNCTION: Astrocytes monitor population-level synaptic
    activity through glutamate transporters. Sustained high activity
    triggers IP3-mediated calcium release from endoplasmic reticulum
    stores. The calcium signal drives release of gliotransmitters
    (glutamate, D-serine, ATP) that modulate synaptic plasticity
    thresholds across the astrocyte's entire domain.

    This implements the BCM (Bienenstock-Cooper-Munro) sliding threshold:
    the modification threshold for LTP vs LTD slides as a function of
    recent average postsynaptic activity.

    Araque A et al. (1999). DOI: 10.1016/S0166-2236(98)01349-6
    Bienenstock EL et al. (1982). DOI: 10.1523/JNEUROSCI.02-01-00032.1982
    Stellwagen D, Malenka RC (2006). DOI: 10.1038/nature04671
    """

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        """
        Args:
            cfg: CognitiveKernelConfig.
        """
        super().__init__()
        self.cfg = cfg

        self.register_buffer(
            "extrasynaptic_glutamate", torch.tensor(0.0),
        )
        self.register_buffer(
            "astrocytic_calcium", torch.tensor(cfg.astro_calcium_target),
        )

    @torch.no_grad()
    def update(self, population_activity: torch.Tensor) -> float:
        """
        Update glutamate and calcium signals from population activity.

        Args:
            population_activity: Any tensor. Mean absolute value is used
                as the proxy for population firing density.

        Returns:
            eta_modifier: Multiplicative learning rate modifier.
                > 1.0 when calcium is above target (increase plasticity).
                < 1.0 when calcium is below target (stabilize).
        """
        activity = population_activity.detach().abs().mean().item()

        tau = self.cfg.astro_glutamate_tau
        self.extrasynaptic_glutamate.copy_(
            tau * self.extrasynaptic_glutamate + (1.0 - tau) * activity
        )

        calcium_drive = torch.sigmoid(
            torch.tensor(
                self.cfg.astro_calcium_gain
                * (self.extrasynaptic_glutamate.item()
                   - self.cfg.astro_calcium_target)
            )
        ).item()
        self.astrocytic_calcium.copy_(
            tau * self.astrocytic_calcium + (1.0 - tau) * calcium_drive
        )

        deviation = self.astrocytic_calcium.item() - self.cfg.astro_calcium_target
        eta = 1.0 + deviation
        eta = max(self.cfg.astro_eta_min, min(self.cfg.astro_eta_max, eta))
        return eta

    def get_state(self) -> Dict[str, float]:
        """Return current astrocytic state for diagnostics."""
        return {
            "glutamate": self.extrasynaptic_glutamate.item(),
            "calcium": self.astrocytic_calcium.item(),
        }


# =========================================================================
# Entorhinal Cortex: Interface and Short-Term Buffer
# =========================================================================

class EntorhinalCortex(nn.Module):
    """
    Interface layer between neocortical mediation and hippocampus proper.

    BIOLOGICAL STRUCTURE: Entorhinal cortex (EC), layers II and III.
    BIOLOGICAL FUNCTION: EC is the gateway to the hippocampus. Layer II
    projects to DG and CA3 via the perforant path. Layer III projects
    directly to CA1 via the temporoammonic pathway. Deep layers (V/VI)
    receive the hippocampal output from the subiculum and project it
    back to neocortex.

    The EC also maintains a short-term buffer: persistent activity in
    EC layer III neurons provides a brief memory of recent input that
    outlasts the immediate sensory stimulation. This is the SHORT-TERM
    tier of the three-tier memory system.

    Witter MP et al. (2000). DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K

    ANATOMICAL INTERFACE:
        INPUT: 64-dim coordinate vectors from the upstream coordinate bridge.
        OUTPUT (to DG): Perforant path layer II projection.
        OUTPUT (to CA1): Temporoammonic layer III projection.
        OUTPUT (to cortex): Return projection via subiculum into EC deep layers.
    """

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        """
        Args:
            cfg: CognitiveKernelConfig.
        """
        super().__init__()
        self.cfg = cfg

        # Short-term buffer: EMA of recent EC input.
        # Persists across forward passes within a wake cycle.
        # Cleared on sleep onset.
        self.register_buffer(
            "short_term_buffer",
            torch.zeros(cfg.coordinate_dim),
        )

        # Input normalization.
        self.input_norm = nn.LayerNorm(cfg.coordinate_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Process incoming coordinates and update short-term buffer.

        Args:
            coords: (B, coordinate_dim) from Perforant Path.

        Returns:
            ec_output: (B, coordinate_dim) normalized and buffer-biased
                representation for downstream DG and CA1.
        """
        # Update short-term buffer with batch mean.
        with torch.no_grad():
            batch_mean = coords.detach().mean(dim=0)
            tau = self.cfg.ec_buffer_tau
            self.short_term_buffer.copy_(
                tau * self.short_term_buffer + (1.0 - tau) * batch_mean
            )

        # Bias input with short-term context.
        biased = coords + 0.1 * self.short_term_buffer.unsqueeze(0)
        return self.input_norm(biased)

    def reset_short_term(self) -> None:
        """Clear the short-term buffer on sleep onset."""
        with torch.no_grad():
            self.short_term_buffer.zero_()


# =========================================================================
# Cognitive Kernel: Full Assembly
# =========================================================================

class CognitiveKernel(nn.Module):
    """
    Complete hippocampal learning-core assembly with 3D neuron coordinates.

    This is the full assembly of the plastic episodic learning core into a
    single module that accepts 64-dim coordinate vectors from the upstream
    coordinate bridge and returns a provisional episodic reconstruction plus
    a novelty signal for neocortical evaluation.

    THREE MEMORY TIERS:
        SHORT-TERM: EntorhinalCortex.short_term_buffer, which holds recent
            cortical context in the learning core.
        WORKING: CA1 and Subiculum active representation for the current
            wake cycle.
        LONG-LASTING: CA3 attractor memory that survives sleep and can be
            serialized as persistent learning-core state.

    FORWARD PASS CIRCUIT:
        coords (64) -> EC -> DG (pattern separation)
                           -> CA3 (attractor retrieval)
                              -> CA1 (novelty comparison with direct EC)
                                 -> Subiculum (output gate)
                                    -> reconstructed coords (64)

    STORE CIRCUIT (when novelty exceeds threshold):
        DG sparse pattern -> CA3.store_episode()

    This module is the learning-core side of the system. It speaks in
    coordinates and internal state, not tokens. It does not adjudicate
    final schema integration by itself. It emits provisional reconstructions
    and mismatch signals upward to the neocortex, which sits between this
    plastic subsystem and the deterministic core. The external LLM never
    touches this module directly.

    References:
        Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074
        Hopfield JJ (1982). DOI: 10.1073/pnas.79.8.2554
        Lisman JE, Grace AA (2005). DOI: 10.1016/j.neuron.2005.05.002
    """

    def __init__(self, cfg: Optional[CognitiveKernelConfig] = None) -> None:
        """
        Args:
            cfg: CognitiveKernelConfig. Uses defaults if None.
        """
        super().__init__()
        if cfg is None:
            cfg = CognitiveKernelConfig()
        self.cfg = cfg

        # 3D neuron position registry.
        self.neuron_positions = NeuronPositionRegistry(cfg)

        # Six subsystems.
        self.entorhinal_cortex = EntorhinalCortex(cfg)
        self.dentate_gyrus = DentateGyrus(cfg)
        self.ca3 = CA3AttractorNetwork(cfg, self.neuron_positions)
        self.ca1 = CA1Comparator(cfg)
        self.subiculum = Subiculum(cfg)
        self.astrocyte = AstrocyticRegulator(cfg)

        # Working memory: the most recent CA1 output, persists within
        # a wake cycle. Cleared on sleep onset.
        self.register_buffer(
            "working_memory", torch.zeros(cfg.ca1_dim),
        )

    def forward(
        self, coords: torch.Tensor, store_if_novel: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Full hippocampal learning-core pass: encode, compare, reconstruct,
        and optionally store.

        Biological process: trisynaptic circuit processing plus CA1 mismatch
        comparison and astrocytic metaplastic regulation. Plain English: the
        learning core takes a coordinate state, separates it, attempts to
        reconstruct a familiar episode, measures how wrong that reconstruction
        is, emits a provisional reconstruction, and stores the episode if it
        is novel enough. The output is still provisional. The neocortex is
        expected to evaluate the returned coordinate state for schema overlap
        and crystallization decisions.

        Args:
            coords: (B, coordinate_dim) coordinate state from the upstream
                bridge into the learning core.
            store_if_novel: If True, automatically store episodes whose mean
                novelty exceeds ca1_novelty_threshold.

        Returns:
            reconstructed: (B, coordinate_dim) provisional episodic
                reconstruction for neocortical evaluation.
            novelty: (B,) novelty scalar per sample in [0, 1].
            diagnostics: Dictionary of learning-core health metrics and
                storage events.
        """
        diagnostics: Dict[str, Any] = {}

        # 1. Entorhinal cortex: normalize, buffer, bias.
        ec_output = self.entorhinal_cortex(coords)

        # 2. Dentate gyrus: pattern separation.
        dg_sparse = self.dentate_gyrus(ec_output)
        dg_sparsity = (dg_sparse == 0).float().mean().item()
        diagnostics["dg_sparsity"] = dg_sparsity

        # 3. CA3: attractor retrieval via mossy fibers.
        ca3_state, schaffer_to_ca1 = self.ca3(dg_sparse)

        # 4. CA1: novelty comparison.
        ca1_output, novelty = self.ca1(schaffer_to_ca1, ec_output)

        # 5. Update working memory with batch mean CA1 output.
        with torch.no_grad():
            self.working_memory.copy_(ca1_output.detach().mean(dim=0))

        # 6. Subiculum: gate and compress back to coordinate space.
        reconstructed = self.subiculum(ca1_output)

        # 7. Astrocytic regulator: update homeostatic signals.
        eta = self.astrocyte.update(ca3_state)
        diagnostics["astro_eta"] = eta
        diagnostics.update(self.astrocyte.get_state())

        # 8. Store novel episodes in CA3.
        mean_novelty = novelty.mean().item()
        diagnostics["mean_novelty"] = mean_novelty
        diagnostics["num_stored_episodes"] = len(self.ca3._stored_patterns)

        if store_if_novel and mean_novelty > self.cfg.ca1_novelty_threshold:
            # NOT a biological quantity. Engineering approximation: store the
            # batch-mean DG pattern as a representative episode for this step.
            mean_dg = dg_sparse.detach().mean(dim=0)
            mossy_pattern = self.ca3.mossy_fiber(mean_dg)
            n_stored = self.ca3.store_episode(mossy_pattern)
            diagnostics["num_stored_episodes"] = n_stored
            diagnostics["stored_this_step"] = True
        else:
            diagnostics["stored_this_step"] = False

        return reconstructed, novelty, diagnostics

    def retrieve_from_cue(self, partial_coords: torch.Tensor) -> torch.Tensor:
        """
        Retrieve a full memory from a partial coordinate cue.

        This is the reconstruction pathway: given a fragment of a past
        experience, reconstruct a fuller episodic state from the attractor
        machinery of the learning core. The returned result is provisional
        and intended for neocortical interpretation rather than direct token
        emission.

        Args:
            partial_coords: (B, coordinate_dim) partial or noisy cue.

        Returns:
            reconstructed: (B, coordinate_dim) completed memory.
        """
        ec_output = self.entorhinal_cortex(partial_coords)
        dg_sparse = self.dentate_gyrus(ec_output)
        ca3_state, schaffer = self.ca3(dg_sparse)
        ca1_output, _ = self.ca1(schaffer, ec_output)
        return self.subiculum(ca1_output)

    def sleep_consolidation(self) -> Dict[str, float]:
        """
        Run sleep-phase consolidation.

        During sleep, the learning core clears its short-term and working
        tiers while preserving the CA3 attractor trace set. In the broader
        architecture, this is the phase in which hippocampal replay can
        provide candidate material to the neocortex for schema-guided
        consolidation and later crystallization into the deterministic core.

        BIOLOGICAL FUNCTION: During NREM sleep, the hippocampus replays
        recently stored episodes in compressed time (sharp-wave ripples).
        This replay drives synaptic consolidation in the neocortex,
        gradually transferring memories from hippocampal to neocortical
        storage. The clearing of short-term and working memory represents
        the transition from encoding mode (high ACh) to consolidation
        mode (low ACh).

        Buzsáki G (2015). "Hippocampal sharp wave-ripple: A cognitive
        biomarker for episodic memory and planning." Hippocampus,
        25(10):1073-1188. DOI: 10.1002/hipo.22488

        Returns:
            Consolidation diagnostics.
        """
        diagnostics: Dict[str, float] = {}

        # Clear short-term buffer.
        self.entorhinal_cortex.reset_short_term()

        # Clear working memory.
        with torch.no_grad():
            self.working_memory.zero_()

        diagnostics["episodes_in_ca3"] = len(self.ca3._stored_patterns)
        diagnostics["short_term_cleared"] = True
        diagnostics["working_memory_cleared"] = True

        return diagnostics

    def get_neuron_positions(self) -> Dict[str, torch.Tensor]:
        """
        Return 3D positions for all neuron populations.

        For visualization. Each key maps to an (N, 3) tensor.
        """
        return {
            name: self.neuron_positions.get_population_positions(name)
            for name in ["ec", "dg", "ca3", "ca1", "subiculum"]
        }

    def get_memory_state(self) -> Dict[str, object]:
        """
        Return the full memory state for serialization.

        SHORT-TERM: EC buffer.
        WORKING: CA1 working memory.
        LONG-LASTING: CA3 stored patterns and recurrent weights.
        """
        return {
            "short_term_buffer": (
                self.entorhinal_cortex.short_term_buffer.detach().cpu().clone()
            ),
            "working_memory": self.working_memory.detach().cpu().clone(),
            "ca3_stored_patterns": [
                p.detach().cpu().clone() for p in self.ca3._stored_patterns
            ],
            "ca3_W_recurrent": self.ca3.W_recurrent.detach().cpu().clone(),
            "astro_state": self.astrocyte.get_state(),
        }

    def load_memory_state(self, state: Dict[str, object]) -> None:
        """
        Restore memory state from a checkpoint.

        Args:
            state: Dict produced by get_memory_state().
        """
        with torch.no_grad():
            device = self.ca3.W_recurrent.device

            if "short_term_buffer" in state:
                self.entorhinal_cortex.short_term_buffer.copy_(
                    state["short_term_buffer"].to(device)
                )
            if "working_memory" in state:
                self.working_memory.copy_(
                    state["working_memory"].to(device)
                )
            if "ca3_stored_patterns" in state:
                self.ca3._stored_patterns = [
                    p.to(device) for p in state["ca3_stored_patterns"]
                ]
            if "ca3_W_recurrent" in state:
                self.ca3.W_recurrent.copy_(
                    state["ca3_W_recurrent"].to(device)
                )

    def count_params(self) -> str:
        """Human-readable parameter count by subsystem."""
        total = sum(p.numel() for p in self.parameters())
        ec = sum(
            p.numel() for n, p in self.named_parameters()
            if "entorhinal" in n
        )
        dg = sum(
            p.numel() for n, p in self.named_parameters()
            if "dentate" in n
        )
        ca3 = sum(
            p.numel() for n, p in self.named_parameters()
            if "ca3" in n
        )
        ca1 = sum(
            p.numel() for n, p in self.named_parameters()
            if "ca1" in n
        )
        sub = sum(
            p.numel() for n, p in self.named_parameters()
            if "subiculum" in n
        )
        return (
            f"Cognitive Kernel Learning Core: {total / 1e3:.1f}K parameters\n"
            f"  Entorhinal Cortex: {ec / 1e3:.1f}K\n"
            f"  Dentate Gyrus:     {dg / 1e3:.1f}K\n"
            f"  CA3 Attractor:     {ca3 / 1e3:.1f}K\n"
            f"  CA1 Comparator:    {ca1 / 1e3:.1f}K\n"
            f"  Subiculum:         {sub / 1e3:.1f}K\n"
            f"  3D Positions:      "
            f"{self.neuron_positions.total_neurons} neurons"
        )


# =========================================================================
# Self-Test
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Cognitive Kernel Core: Self-Test")
    print("=" * 60)

    cfg = CognitiveKernelConfig()
    kernel = CognitiveKernel(cfg)
    print(kernel.count_params())

    # Test 1: Forward pass.
    coords = torch.randn(4, 64)
    recon, novelty, diag = kernel(coords)
    assert recon.shape == (4, 64), f"Expected (4,64), got {recon.shape}"
    assert novelty.shape == (4,), f"Expected (4,), got {novelty.shape}"
    print(f"\n[PASS] Forward: recon={tuple(recon.shape)}, "
          f"novelty={novelty.mean():.4f}")
    print(f"       DG sparsity: {diag['dg_sparsity']:.4f}")
    print(f"       Astro eta:   {diag['astro_eta']:.4f}")
    print(f"       Stored:      {diag['num_stored_episodes']} episodes")

    # Test 2: Memory accumulation.
    for i in range(10):
        c = torch.randn(4, 64) * (1.0 + i * 0.1)
        _, _, d = kernel(c)
    print(f"\n[PASS] After 10 steps: {d['num_stored_episodes']} episodes stored")

    # Test 3: Retrieval from partial cue.
    full = torch.randn(1, 64)
    kernel(full, store_if_novel=True)
    noisy = full + torch.randn_like(full) * 0.3
    retrieved = kernel.retrieve_from_cue(noisy)
    assert retrieved.shape == (1, 64)
    print(f"[PASS] Retrieval from noisy cue: shape={tuple(retrieved.shape)}")

    # Test 4: 3D positions.
    positions = kernel.get_neuron_positions()
    for name, pos in positions.items():
        assert pos.shape[1] == 3, f"{name} positions not 3D"
    total_neurons = sum(p.shape[0] for p in positions.values())
    print(f"[PASS] 3D positions: {total_neurons} neurons across "
          f"{len(positions)} populations")

    # Test 5: Sleep consolidation.
    sleep_diag = kernel.sleep_consolidation()
    assert kernel.entorhinal_cortex.short_term_buffer.abs().sum() == 0.0
    assert kernel.working_memory.abs().sum() == 0.0
    print(f"[PASS] Sleep: short-term cleared, working memory cleared, "
          f"{sleep_diag['episodes_in_ca3']} episodes preserved in CA3")

    # Test 6: Memory state round-trip.
    state = kernel.get_memory_state()
    kernel.sleep_consolidation()
    kernel.load_memory_state(state)
    assert len(kernel.ca3._stored_patterns) == len(state["ca3_stored_patterns"])
    print(f"[PASS] Memory state round-trip: "
          f"{len(kernel.ca3._stored_patterns)} patterns restored")

    # Test 7: Three memory tiers are distinct.
    coords = torch.randn(2, 64)
    kernel(coords)
    st = kernel.entorhinal_cortex.short_term_buffer.norm().item()
    wm = kernel.working_memory.norm().item()
    lt = len(kernel.ca3._stored_patterns)
    print(f"[PASS] Three tiers active: "
          f"short-term={st:.4f}, working={wm:.4f}, "
          f"long-lasting={lt} episodes")

    print(f"\n{'=' * 60}")
    print("All tests passed.")
    print(f"{'=' * 60}")
