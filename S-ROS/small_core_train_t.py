"""
small_core_train.py

BIOLOGICAL GROUNDING
====================
This file is the integration training harness for the PRAGMI (Persistent
Reconstructive Architecture for Generative Memory and Imagination) system.
It does not model a single brain region. It models the organism-level
training cycle: the repeated loop of experience, encoding, consolidation,
and prediction refinement that underlies learning in biological nervous
system

The five subsystems wired together here correspond to the following
biological functions:

CognitiveKernel: Hippocampal formation (entorhinal cortex, dentate gyrus,
CA3 attractor network, CA1 comparator, subiculum, astrocytic regulator).
The episodic memory system that encodes novel experiences and reconstructs
them from partial cues. O'Keefe J, Nadel L (1978). "The Hippocampus as a
Cognitive Map." Oxford University Press.

WorldModelEnsemble: Neocortical generative model implementing predictive
coding. The brain's forward model of its own sensory and internal states.
Friston K (2010). DOI: 10.1038/nrn2787

NeuromodulatorBroadcast: Diffuse neuromodulatory systems (dopamine,
norepinephrine, acetylcholine, serotonin). Global chemical modulators that
reconfigure network operating mode in response to reward, uncertainty, and
arousal. Yu AJ, Dayan P (2005). DOI: 10.1016/j.neuron.2005.04.026

EpistemicSelector: Active foraging / epistemic action selection. Organisms
orient toward informative experiences rather than passively accepting
whatever arrives. Friston K et al. (2017). DOI: 10.1162/NECO_a_00912

CorticalBuffer: Prefrontal cortex delay-period persistent activity. Working
memory maintained through recurrent synaptic reverberation across
inter-pass intervals. Goldman-Rakic PS (1995). DOI: 10.1016/0896-6273(95)90304-6

PLACEHOLDER NOTE: TimmyModel (the spiking neural network translation layer)
is not yet available in this repository. In its place, this harness uses a
FixedCoordinateProjection (deterministic random orthonormal projection) and
a LightweightEncoder (small feedforward network). Both are explicitly labeled
as engineering stand-ins and must be replaced when TimmyModel is integrated.

The training loop implements the biological sleep-wake cycle: continuous
encoding during waking (every step), periodic consolidation during sleep
(every sleep_interval steps), and world model refinement throughout.

Primary papers grounding this file:

Friston K (2010). "The free-energy principle: a unified brain theory."
Nature Reviews Neuroscience, 11(2), 127-138. DOI: 10.1038/nrn2787

O'Reilly RC, Frank MJ (2006). "Making working memory work: a computational
model of learning in the prefrontal cortex and basal ganglia." Neural
Computation, 18(2), 283-328. DOI: 10.1162/089976606775093909

Diekelmann S, Born J (2010). "The memory function of sleep." Nature Reviews
Neuroscience, 11(2), 114-126. DOI: 10.1038/nrn2762
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Subsystem imports. Production versions take priority; teaching files are the
# fallback. The try/except pattern allows the harness to run in both
# development (teaching files only) and production (stripped files present)
# environments without modification.
# NOT a biological concept: import fallback is an engineering pattern.
#
# The neuromodulator, epistemic selector and cortical buffer teaching files
# live in ../Teaching, so that directory is appended to the path (after this
# one, so S-ROS copies win). The fallback catches any Exception, not just
# ImportError, because a corrupt production file raises SyntaxError.
# ---------------------------------------------------------------------------
_TEACHING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Teaching")
if os.path.isdir(_TEACHING_DIR) and _TEACHING_DIR not in sys.path:
    sys.path.append(_TEACHING_DIR)

try:
    from cognitive_kernel_core_c import CognitiveKernel, CognitiveKernelConfig
except Exception:
    from cognitive_kernel_core_t import CognitiveKernel, CognitiveKernelConfig  # type: ignore[no-redef]

try:
    from world_model_ensemble_p import WorldModelEnsemble, WorldModelConfig
except Exception:
    from world_model_ensemble_t import WorldModelEnsemble, WorldModelConfig  # type: ignore[no-redef]

try:
    from neuromodulator_broadcast_p import NeuromodulatorBroadcast, NeuromodulatorConfig
except Exception:
    from neuromodulator_broadcast_t import NeuromodulatorBroadcast, NeuromodulatorConfig  # type: ignore[no-redef]

# Note on import names: the spec references select_batch_by_epistemic_value
# and score_candidates. The actual epistemic_selector module exports
# select_batch and select_batch_deterministic. The harness uses the real
# exported names. When the production module is written, it should expose
# these same names.
try:
    from epistemic_selector_p import select_batch, select_batch_deterministic
except Exception:
    from epistemic_selector_t import select_batch, select_batch_deterministic  # type: ignore[no-redef]

try:
    from cortical_buffer_p import CorticalBuffer, CorticalBufferConfig
except Exception:
    from cortical_buffer_t import CorticalBuffer, CorticalBufferConfig  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Coordinate space dimension. Fixed at 64 throughout the PRAGMI architecture.
# This is the communication channel between the encoder stack and the
# CognitiveKernel. It corresponds to the dimensionality of the Perforant
# Path coordinate manifold.
# NOT a biological quantity: engineering parameter.
# Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
# ---------------------------------------------------------------------------
_COORDINATE_DIM: int = 64


# ===========================================================================
# FixedCoordinateProjection
# ===========================================================================

class FixedCoordinateProjection(nn.Module):
    """
    Deterministic fixed random orthonormal projection to 64-dim coordinate space.

    BIOLOGICAL CONTEXT: This is an engineering stand-in for the
    PerforantPathSymphonyBridge, which will eventually be implemented as
    TimmyModel (a spiking neural network language model). The real bridge
    learns to map between the external LLM's token embedding space and the
    CognitiveKernel's coordinate manifold. This class does not learn. It
    provides a fixed geometric transformation so the rest of the architecture
    can be trained and validated before TimmyModel is integrated.

    ANATOMICAL INTERFACE (input):
        Sending structure: LightweightEncoder (stand-in for TimmyModel).
        Receiving structure: FixedCoordinateProjection (this module).
        Connection: Simulated entorhinal input projection. The real connection
            is the perforant path from entorhinal cortex to the coordinate
            manifold. Steward O, Scoville SA (1976). DOI: 10.1002/cne.901690105
    ANATOMICAL INTERFACE (output):
        Sending structure: FixedCoordinateProjection.
        Receiving structure: CognitiveKernel (EntorhinalCortex input).
        Connection: Perforant path. Same anatomical connection as above.

    IMPLEMENTATION NOTE: The projection matrix is generated via QR decomposition
    from a seeded random matrix, yielding a matrix with orthonormal columns. This
    preserves inner products and distances in the output space, which is important
    for the CognitiveKernel's distance-based operations. The matrix is registered
    as a buffer (not a Parameter) so it does not receive gradient updates and is
    moved correctly by .to(device).

    NOT a biological quantity: the projection matrix, seed, and dimensionality
    are engineering choices made to stand in for TimmyModel pending that
    module's implementation.

    REPLACEMENT INSTRUCTIONS: When TimmyModel is integrated, replace this
    class entirely with the PerforantPathSymphonyBridge. The interface contract
    is: input (B, encoder_dim), output (B, 64). The output must be under
    torch.no_grad() or otherwise gradient-isolated from the CognitiveKernel's
    optimizer to prevent cross-contamination of training signals.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = _COORDINATE_DIM,
        seed: int = 42,
    ) -> None:
        """
        Initialize the fixed random orthonormal projection.

        Args:
            input_dim: Dimensionality of the input embedding space (from
                LightweightEncoder or, eventually, TimmyModel).
            output_dim: Dimensionality of the coordinate manifold. Fixed at 64
                throughout the architecture. NOT a biological quantity.
            seed: Random seed for QR decomposition. Fixed at 42 to ensure
                determinism across runs and machines. NOT a biological quantity.
        """
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        random_matrix = torch.randn(input_dim, output_dim, generator=rng)
        q, _ = torch.linalg.qr(random_matrix)
        # Register as buffer: not a Parameter, participates in state_dict
        # and device movement, but receives no gradient updates.
        # NOT a biological quantity: engineering registration choice.
        self.register_buffer("projection", q[:, :output_dim])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input embeddings to the 64-dim coordinate manifold.

        The @torch.no_grad() decorator ensures that this projection
        contributes zero gradient to any computation graph. This is an
        engineering constraint: the FixedCoordinateProjection is not a
        learnable component and must not pollute the CognitiveKernel's
        or encoder's gradient flows.

        ANATOMICAL INTERFACE:
            Sending: LightweightEncoder (stand-in for TimmyModel).
            Receiving: CognitiveKernel via EntorhinalCortex.
            Connection: Perforant path analog.

        Args:
            x: Input embeddings, shape (B, input_dim).

        Returns:
            Coordinate projections, shape (B, output_dim). Detached from
            any computation graph. requires_grad is False.
        """
        return x @ self.projection


# ===========================================================================
# LightweightEncoder
# ===========================================================================

class LightweightEncoder(nn.Module):
    """
    Minimal feedforward encoder converting token IDs to continuous embeddings.

    NOT a biological model. This is an engineering stand-in for TimmyModel
    (the spiking neural network language model from the PRAGMI architecture).
    TimmyModel is the biological analog of the cortical translation layer that
    maps between external token representations and internal coordinate-space
    representations. LightweightEncoder exists solely so the integration
    harness can train on real text data without requiring the full SNN stack.

    REPLACEMENT INSTRUCTIONS: When TimmyModel is integrated, remove this class
    entirely and replace it with TimmyModel in the train_small_core function.
    The interface contract is: input (B, seq_len) integer token IDs,
    output (B, d_model) float embeddings. TimmyModel's output feeds directly
    into FixedCoordinateProjection (and eventually the PerforantPathSymphonyBridge).

    NOT a biological quantity: every component of this class (embedding,
    pooling, linear projection) is a training artifact with no biological
    grounding. It is documented here as a stand-in, not as an architectural
    contribution.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        seq_len: int = 128,
    ) -> None:
        """
        Initialize the lightweight encoder.

        Args:
            vocab_size: Number of token IDs in the vocabulary. Default 50257
                matches GPT-2 tokenizer vocabulary size. NOT a biological
                quantity: engineering parameter.
            d_model: Output embedding dimensionality. Must be >= _COORDINATE_DIM
                (64) for the FixedCoordinateProjection to have enough input
                dimensions. Default 256. NOT a biological quantity.
            seq_len: Expected sequence length. Used only to size the adaptive
                average pooling layer. NOT a biological quantity.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Token embedding table. Maps integer token IDs to continuous vectors.
        # NOT a biological quantity: engineering stand-in for TimmyModel's
        # spiking encoding pathway.
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Adaptive average pooling to collapse the sequence dimension to 1.
        # Simplification: ignores sequential structure entirely. TimmyModel
        # preserves sequential structure through spike timing.
        # NOT a biological quantity: engineering simplification.
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Linear projection after pooling. Provides a learned nonlinear
        # transformation step before the fixed orthonormal projection.
        # NOT a biological quantity: engineering component.
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token IDs to continuous embeddings.

        Args:
            token_ids: Integer token IDs, shape (B, S) where S is seq_len.

        Returns:
            Continuous embeddings, shape (B, d_model).
        """
        # (B, S) -> (B, S, d_model)
        x = self.embedding(token_ids)
        # (B, S, d_model) -> (B, d_model, S) for AdaptiveAvgPool1d
        x = x.transpose(1, 2)
        # (B, d_model, S) -> (B, d_model, 1) -> (B, d_model)
        x = self.pool(x).squeeze(-1)
        # (B, d_model) -> (B, d_model)
        x = self.proj(x)
        return x


# ===========================================================================
# SimpleTextDataset
# ===========================================================================

class SimpleTextDataset(Dataset):
    """
    Minimal text dataset for the integration harness.

    NOT a biological concept. Engineering utility class that tokenizes
    a list of text strings using a simple whitespace/character tokenizer,
    pads or truncates to seq_len, and returns integer tensors. Avoids
    any dependency on external tokenizer packages (tiktoken, sentencepiece,
    HuggingFace tokenizers).

    The tokenization strategy is: split on whitespace, map each word to
    a hash index modulo vocab_size. This is a training artifact with no
    relationship to any linguistically meaningful tokenization scheme.
    It provides integer inputs so the LightweightEncoder can train; the
    actual token distribution does not matter for integration testing.
    NOT a biological quantity.
    """

    def __init__(
        self,
        texts: List[str],
        seq_len: int = 128,
        vocab_size: int = 50257,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            texts: List of raw text strings.
            seq_len: Token sequence length after padding/truncation.
            vocab_size: Vocabulary size for hash-based tokenization.
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.samples: List[torch.Tensor] = []
        for text in texts:
            tokens = self._tokenize(text)
            self.samples.append(tokens)

    def _tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize a text string to a fixed-length integer tensor.

        Uses whitespace splitting and hash-based word-to-index mapping.
        Pads with zeros if shorter than seq_len, truncates if longer.
        NOT a biological quantity: engineering tokenization stand-in.

        Args:
            text: Raw text string.

        Returns:
            Integer tensor of shape (seq_len,).
        """
        words = text.split()
        ids = [abs(hash(w)) % self.vocab_size for w in words]
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        else:
            ids = ids[: self.seq_len]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single tokenized sample."""
        return self.samples[idx]


# ===========================================================================
# SyntheticDataset
# ===========================================================================

class SyntheticDataset(Dataset):
    """
    Synthetic random integer dataset for fallback when real data is unavailable.

    NOT a biological concept. Engineering fallback that generates random
    integer tensors in [0, vocab_size). Used when both HuggingFace dataset
    and fallback dataset are unavailable (network failure or missing package).
    Training on synthetic data exercises the full integration pipeline without
    producing a meaningful model.
    NOT a biological quantity.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = 128,
        vocab_size: int = 50257,
    ) -> None:
        """
        Initialize the synthetic dataset.

        Args:
            n_samples: Number of synthetic samples to generate.
            seq_len: Sequence length per sample.
            vocab_size: Range for random integer generation.
        """
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))

    def __len__(self) -> int:
        """Return total number of synthetic samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single synthetic sample."""
        return self.data[idx]


# ===========================================================================
# cosine_lr
# ===========================================================================

def cosine_lr(
    step: int,
    total_steps: int,
    lr: float,
    warmup_steps: int = 200,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    NOT a biological quantity. Engineering learning rate schedule that linearly
    increases from 0 to lr over warmup_steps, then follows a cosine decay to
    approximately 0 by total_steps. This matches the schedule used in
    train_timmy.py for consistency across the PRAGMI training suite.

    Loshchilov I, Hutter F (2017). "SGDR: Stochastic gradient descent with
    warm restarts." ICLR 2017. DOI: {To be added later.}

    Args:
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        lr: Peak learning rate reached at end of warmup.
        warmup_steps: Number of linear warmup steps before cosine decay.

    Returns:
        Learning rate scalar for the current step.
    """
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ===========================================================================
# run_tests
# ===========================================================================

def run_tests() -> bool:
    """
    Integration test suite for all five PRAGMI subsystems.

    Runs on CPU, requires no data, and validates instantiation and wiring
    of all subsystems. The training loop does not start if this returns False.

    NOT a biological concept. Engineering validation suite ensuring all
    subsystem contracts are satisfied before training begins.

    Returns:
        True if all 18 tests pass, False otherwise.
    """
    print("=" * 70)
    print("PRAGMI Integration Test Suite")
    print("=" * 70)

    passed = 0
    total = 18

    # ------------------------------------------------------------------
    # Test 1: LightweightEncoder instantiation and forward pass shape
    # ------------------------------------------------------------------
    try:
        enc = LightweightEncoder(vocab_size=50257, d_model=256, seq_len=128)
        dummy_ids = torch.randint(0, 50257, (2, 128))
        enc_out = enc(dummy_ids)
        assert enc_out.shape == (2, 256), (
            f"Expected (2, 256), got {enc_out.shape}"
        )
        print(f"[PASS]  1/18  LightweightEncoder: input (2,128) -> output {tuple(enc_out.shape)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL]  1/18  LightweightEncoder: {e}")

    # ------------------------------------------------------------------
    # Test 2: FixedCoordinateProjection instantiation and output shape
    # ------------------------------------------------------------------
    try:
        proj = FixedCoordinateProjection(input_dim=256, output_dim=64, seed=42)
        dummy_emb = torch.randn(2, 256)
        coords = proj(dummy_emb)
        assert coords.shape == (2, 64), f"Expected (2, 64), got {coords.shape}"

        # Determinism: same seed must produce same output
        proj2 = FixedCoordinateProjection(input_dim=256, output_dim=64, seed=42)
        coords2 = proj2(dummy_emb)
        assert torch.allclose(coords, coords2), "Same seed did not produce same output"

        print(f"[PASS]  2/18  FixedCoordinateProjection: shape {tuple(coords.shape)}, determinism confirmed")
        passed += 1
    except Exception as e:
        print(f"[FAIL]  2/18  FixedCoordinateProjection: {e}")

    # ------------------------------------------------------------------
    # Test 3: FixedCoordinateProjection gradient isolation
    # ------------------------------------------------------------------
    try:
        proj3 = FixedCoordinateProjection(input_dim=256, output_dim=64, seed=42)
        inp = torch.randn(2, 256, requires_grad=True)
        coords3 = proj3(inp)
        assert not coords3.requires_grad, (
            "coords.requires_grad must be False after FixedCoordinateProjection forward"
        )
        print(f"[PASS]  3/18  FixedCoordinateProjection gradient isolation: requires_grad=False confirmed")
        passed += 1
    except Exception as e:
        print(f"[FAIL]  3/18  FixedCoordinateProjection gradient isolation: {e}")

    # ------------------------------------------------------------------
    # Test 4: CognitiveKernel instantiation and forward pass shapes
    # ------------------------------------------------------------------
    try:
        cfg_ck = CognitiveKernelConfig()
        kernel = CognitiveKernel(cfg_ck)
        dummy_coords = torch.randn(2, 64)
        recon, novelty, diag = kernel(dummy_coords)
        assert recon.shape == (2, 64), f"Expected recon (2, 64), got {recon.shape}"
        assert novelty.shape == (2,), f"Expected novelty (2,), got {novelty.shape}"
        assert "astro_eta" in diag, "Missing astro_eta in diagnostics"
        print(f"[PASS]  4/18  CognitiveKernel forward: recon={tuple(recon.shape)}, novelty={tuple(novelty.shape)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL]  4/18  CognitiveKernel forward: {e}")

    # ------------------------------------------------------------------
    # Test 5: CognitiveKernel episode accumulation
    # ------------------------------------------------------------------
    try:
        cfg_ck5 = CognitiveKernelConfig()
        kernel5 = CognitiveKernel(cfg_ck5)
        for i in range(10):
            c = torch.randn(2, 64) * (1.0 + i * 0.3)
            _, _, d5 = kernel5(c)
        episodes = len(kernel5.ca3._stored_patterns)
        assert episodes > 0, "No episodes stored after 10 forward passes"
        print(f"[PASS]  5/18  CognitiveKernel episode accumulation: {episodes} episodes stored after 10 steps")
        passed += 1
    except Exception as e:
        print(f"[FAIL]  5/18  CognitiveKernel episode accumulation: {e}")

    # ------------------------------------------------------------------
    # Test 6: CognitiveKernel sleep consolidation
    # ------------------------------------------------------------------
    try:
        cfg_ck6 = CognitiveKernelConfig()
        kernel6 = CognitiveKernel(cfg_ck6)
        for i in range(5):
            kernel6(torch.randn(2, 64) * (1.0 + i * 0.4))
        episodes_before = len(kernel6.ca3._stored_patterns)
        sleep_diag = kernel6.sleep_consolidation()
        assert kernel6.entorhinal_cortex.short_term_buffer.abs().sum().item() == 0.0, (
            "Short-term buffer not cleared after sleep"
        )
        assert kernel6.working_memory.abs().sum().item() == 0.0, (
            "Working memory not cleared after sleep"
        )
        episodes_after = len(kernel6.ca3._stored_patterns)
        assert episodes_after == episodes_before, (
            f"CA3 episodes changed across sleep: {episodes_before} -> {episodes_after}"
        )
        print(
            f"[PASS]  6/18  CognitiveKernel sleep: short-term cleared, "
            f"working memory cleared, {episodes_after} CA3 episodes preserved"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL]  6/18  CognitiveKernel sleep: {e}")

    # ------------------------------------------------------------------
    # Test 7: CognitiveKernel memory state serialization round-trip
    # ------------------------------------------------------------------
    try:
        cfg_ck7 = CognitiveKernelConfig()
        kernel7 = CognitiveKernel(cfg_ck7)
        for i in range(4):
            kernel7(torch.randn(2, 64) * (1.0 + i * 0.5))
        state = kernel7.get_memory_state()
        n_patterns_before = len(kernel7.ca3._stored_patterns)
        kernel7.sleep_consolidation()
        kernel7.load_memory_state(state)
        n_patterns_after = len(kernel7.ca3._stored_patterns)
        assert n_patterns_after == n_patterns_before, (
            f"Pattern count mismatch after round-trip: {n_patterns_before} -> {n_patterns_after}"
        )
        print(f"[PASS]  7/18  CognitiveKernel memory round-trip: {n_patterns_after} patterns restored")
        passed += 1
    except Exception as e:
        print(f"[FAIL]  7/18  CognitiveKernel memory round-trip: {e}")

    # ------------------------------------------------------------------
    # Test 8: WorldModelEnsemble forward pass shapes and positive variance
    # ------------------------------------------------------------------
    try:
        wm_cfg = WorldModelConfig()
        wm = WorldModelEnsemble(wm_cfg)
        wm_coords = torch.randn(2, 64)
        predictions, mean_pred, ensemble_var = wm(wm_coords)
        assert predictions.shape == (5, 2, 64), (
            f"Expected predictions (5, 2, 64), got {predictions.shape}"
        )
        assert mean_pred.shape == (2, 64), (
            f"Expected mean_pred (2, 64), got {mean_pred.shape}"
        )
        assert ensemble_var.item() > 0.0, "Ensemble variance must be positive"
        print(
            f"[PASS]  8/18  WorldModelEnsemble forward: preds={tuple(predictions.shape)}, "
            f"variance={ensemble_var.item():.6f}"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL]  8/18  WorldModelEnsemble forward: {e}")

    # ------------------------------------------------------------------
    # Test 9: WorldModelEnsemble learning convergence on a learnable map
    # ------------------------------------------------------------------
    try:
        # The target must be learnable. Independent random targets have an
        # MSE floor of ~1.0, which made this test a coin flip.
        torch.manual_seed(9)
        wm9_cfg = WorldModelConfig()
        wm9 = WorldModelEnsemble(wm9_cfg)
        true_map = torch.linalg.qr(torch.randn(64, 64))[0]
        losses_9: List[float] = []
        for _ in range(1000):  # the ensemble learns slowly by design (lr 1e-4)
            c_now = torch.randn(16, 64)
            c_next = c_now @ true_map
            loss_val = wm9.update(c_now, c_next)
            losses_9.append(loss_val)
        first_10 = sum(losses_9[:10]) / 10
        last_10 = sum(losses_9[-10:]) / 10
        assert last_10 < 0.7 * first_10, (
            f"WorldModel did not converge: first_10={first_10:.4f}, last_10={last_10:.4f}"
        )
        print(
            f"[PASS]  9/18  WorldModelEnsemble convergence: "
            f"first_10_mean={first_10:.4f} -> last_10_mean={last_10:.4f}"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL]  9/18  WorldModelEnsemble convergence: {e}")

    # ------------------------------------------------------------------
    # Test 10: WorldModelEnsemble HOT state round-trip
    # ------------------------------------------------------------------
    try:
        wm10_cfg = WorldModelConfig()
        wm10 = WorldModelEnsemble(wm10_cfg)
        for _ in range(5):
            wm10.update(torch.randn(2, 64), torch.randn(2, 64))
            wm10.evaluate_batch(torch.randn(2, 64))
        hot_state = wm10.get_hot_state()
        wm10b = WorldModelEnsemble(wm10_cfg)
        wm10b.load_hot_state(hot_state)
        assert len(wm10b._recent_errors) == len(wm10._recent_errors), (
            "Error history mismatch after HOT round-trip"
        )
        assert len(wm10b._recent_variances) == len(wm10._recent_variances), (
            "Variance history mismatch after HOT round-trip"
        )
        print(f"[PASS] 10/18  WorldModelEnsemble HOT round-trip: {len(wm10b._recent_errors)} errors, {len(wm10b._recent_variances)} variances restored")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 10/18  WorldModelEnsemble HOT round-trip: {e}")

    # ------------------------------------------------------------------
    # Test 11: NeuromodulatorBroadcast DA and NE updates move baselines
    # ------------------------------------------------------------------
    try:
        nm_cfg = NeuromodulatorConfig()
        nm = NeuromodulatorBroadcast(nm_cfg)
        da_before = nm.da.item()
        ne_before = nm.ne.item()
        nm.update_da(1.0)
        nm.update_ne(1.0)
        assert nm.da.item() != da_before, "DA baseline did not change after update_da(1.0)"
        assert nm.ne.item() != ne_before, "NE baseline did not change after update_ne(1.0)"
        print(
            f"[PASS] 11/18  NeuromodulatorBroadcast: "
            f"DA {da_before:.4f}->{nm.da.item():.4f}, "
            f"NE {ne_before:.4f}->{nm.ne.item():.4f}"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL] 11/18  NeuromodulatorBroadcast DA/NE updates: {e}")

    # ------------------------------------------------------------------
    # Test 12: NeuromodulatorBroadcast maturity across 15 sleep cycles
    # ------------------------------------------------------------------
    try:
        nm12_cfg = NeuromodulatorConfig()
        nm12 = NeuromodulatorBroadcast(nm12_cfg)
        maturity_vals: List[float] = []
        for i in range(15):
            m = nm12.compute_maturity(
                routing_entropy=0.5 + 0.01 * i,
                loss_value=1.0 - 0.05 * i,
                probe_response=0.5,
                mean_recent_variance=0.05,
            )
            maturity_vals.append(m)
        assert len(maturity_vals) == 15, f"Expected 15 maturity values, got {len(maturity_vals)}"
        assert all(0.0 <= v <= 1.0 for v in maturity_vals), "Maturity values out of [0, 1]"
        print(
            f"[PASS] 12/18  NeuromodulatorBroadcast maturity: "
            f"15 cycles, final maturity={maturity_vals[-1]:.4f}"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL] 12/18  NeuromodulatorBroadcast maturity: {e}")

    # ------------------------------------------------------------------
    # Test 13: NeuromodulatorBroadcast HOT state round-trip
    # ------------------------------------------------------------------
    try:
        nm13_cfg = NeuromodulatorConfig()
        nm13 = NeuromodulatorBroadcast(nm13_cfg)
        for i in range(5):
            nm13.compute_maturity(0.5, 0.5, 0.5, 0.05)
        hot13 = nm13.get_hot_state()
        nm13b = NeuromodulatorBroadcast(nm13_cfg)
        nm13b.load_hot_state(hot13)
        assert nm13b._sleep_cycles_elapsed == nm13._sleep_cycles_elapsed, (
            "Sleep cycle count mismatch after HOT round-trip"
        )
        assert len(nm13b._entropy_history) == len(nm13._entropy_history), (
            "Entropy history length mismatch after HOT round-trip"
        )
        print(
            f"[PASS] 13/18  NeuromodulatorBroadcast HOT round-trip: "
            f"{nm13b._sleep_cycles_elapsed} cycles restored"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL] 13/18  NeuromodulatorBroadcast HOT round-trip: {e}")

    # ------------------------------------------------------------------
    # Test 14: EpistemicSelector near-random at low maturity
    # ------------------------------------------------------------------
    try:
        wm14_cfg = WorldModelConfig()
        wm14 = WorldModelEnsemble(wm14_cfg)
        candidates = [torch.randn(2, 64) for _ in range(8)]
        # At maturity 0.1 (below MATURITY_RANDOM=0.3), selection should be random.
        # We verify this by running many trials and checking that multiple
        # different indices are selected (not always the same one).
        rng14 = torch.Generator().manual_seed(0)
        indices: set = set()
        for _ in range(30):
            idx, _ = select_batch(candidates, wm14, global_maturity=0.1, rng=rng14)
            indices.add(idx)
        assert len(indices) > 1, (
            f"At maturity=0.1 selection was not random: only index {indices} selected across 30 trials"
        )
        print(
            f"[PASS] 14/18  EpistemicSelector low-maturity random: "
            f"{len(indices)} distinct indices across 30 trials"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL] 14/18  EpistemicSelector low-maturity random: {e}")

    # ------------------------------------------------------------------
    # Test 15: EpistemicSelector score_candidates (select_batch_deterministic)
    # returns scores for all candidates
    # ------------------------------------------------------------------
    try:
        wm15_cfg = WorldModelConfig()
        wm15 = WorldModelEnsemble(wm15_cfg)
        candidates15 = [torch.randn(2, 64) for _ in range(8)]
        idx15, best15, variances15 = select_batch_deterministic(
            candidates15, wm15, global_maturity=0.8
        )
        assert len(variances15) == 8, f"Expected 8 scores, got {len(variances15)}"
        assert all(v >= 0.0 for v in variances15), "Negative variance scores"
        assert 0 <= idx15 < 8, f"Selected index {idx15} out of range"
        print(
            f"[PASS] 15/18  EpistemicSelector score_candidates: "
            f"8 scores, selected idx={idx15}, max_var={max(variances15):.6f}"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL] 15/18  EpistemicSelector score_candidates: {e}")

    # ------------------------------------------------------------------
    # Test 16: CorticalBuffer instantiation, forward pass, nonzero state
    # ------------------------------------------------------------------
    try:
        cb_cfg = CorticalBufferConfig(d_model=256, buffer_dim=32)
        cb = CorticalBuffer(cb_cfg)
        injection_before = cb.get_injection()
        assert injection_before.shape == (256,), (
            f"Expected injection shape (256,), got {injection_before.shape}"
        )
        # Update with a nonzero membrane potential
        dummy_vmem = torch.randn(4, 256)
        cb.update(dummy_vmem)
        state_norm = cb.state.norm().item()
        assert state_norm > 0.0, "Buffer state is still zero after update"
        print(
            f"[PASS] 16/18  CorticalBuffer: injection shape {tuple(injection_before.shape)}, "
            f"state norm={state_norm:.4f} after update"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL] 16/18  CorticalBuffer: {e}")

    # ------------------------------------------------------------------
    # Test 17: CorticalBuffer reset zeros the state
    # ------------------------------------------------------------------
    try:
        cb17_cfg = CorticalBufferConfig(d_model=256, buffer_dim=32)
        cb17 = CorticalBuffer(cb17_cfg)
        cb17.update(torch.randn(4, 256))
        assert cb17.state.norm().item() > 0.0, "State should be nonzero before reset"
        cb17.reset()
        assert cb17.state.abs().sum().item() == 0.0, "State not zeroed after reset()"
        print("[PASS] 17/18  CorticalBuffer reset: state zeroed correctly")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 17/18  CorticalBuffer reset: {e}")

    # ------------------------------------------------------------------
    # Test 18: Full pipeline test
    #   encoder -> projection -> kernel -> world model update ->
    #   neuromodulator update
    # ------------------------------------------------------------------
    try:
        # Instantiate all components
        enc18 = LightweightEncoder(vocab_size=50257, d_model=256, seq_len=128)
        proj18 = FixedCoordinateProjection(input_dim=256, output_dim=64, seed=42)
        kernel18 = CognitiveKernel(CognitiveKernelConfig())
        wm18 = WorldModelEnsemble(WorldModelConfig())
        nm18 = NeuromodulatorBroadcast(NeuromodulatorConfig())

        # Step 1: encode tokens
        ids18 = torch.randint(0, 50257, (4, 128))
        emb18 = enc18(ids18)
        assert emb18.shape == (4, 256), f"Encoder output shape error: {emb18.shape}"

        # Step 2: project to coordinate space (no gradient)
        coords18 = proj18(emb18)
        assert coords18.shape == (4, 64), f"Projection shape error: {coords18.shape}"
        assert not coords18.requires_grad, "Coordinates must be gradient-free"

        # Step 3: kernel forward
        recon18, novelty18, diag18 = kernel18(coords18)
        assert recon18.shape == (4, 64), f"Reconstruction shape error: {recon18.shape}"

        # Step 4: compute reconstruction loss (kernel loss)
        ec_target = diag18.get("ec_output", None)
        if ec_target is None:
            target18 = coords18.detach()
        else:
            target18 = ec_target if isinstance(ec_target, torch.Tensor) else coords18.detach()
        kernel_loss = F.mse_loss(recon18, target18.detach() if isinstance(target18, torch.Tensor) else target18)
        assert torch.isfinite(kernel_loss), f"kernel_loss is not finite: {kernel_loss}"

        # Step 5: world model forward
        ids2_18 = torch.randint(0, 50257, (4, 128))
        emb2_18 = enc18(ids2_18)
        coords2_18 = proj18(emb2_18)
        _, _, wm_var = wm18(coords18)
        wm_loss = wm18.update(coords18, coords2_18)
        assert math.isfinite(wm_loss), f"World model loss not finite: {wm_loss}"

        # Step 6: neuromodulator update
        eta18 = diag18.get("astro_eta", 1.0)
        norm_signal = float(torch.clamp(torch.tensor(wm_loss / (wm_loss + 1.0)), 0.0, 1.0))
        nm18.update_da(norm_signal)
        nm18.update_ne(float(torch.clamp(wm_var.detach(), 0.0, 1.0)))

        print(
            f"[PASS] 18/18  Full pipeline: "
            f"kernel_loss={kernel_loss.item():.4f}, "
            f"wm_loss={wm_loss:.4f}, "
            f"astro_eta={eta18:.4f}"
        )
        passed += 1
    except Exception as e:
        print(f"[FAIL] 18/18  Full pipeline: {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"Test results: {passed} / {total} tests passed")
    print("=" * 70)
    return passed == total


# ===========================================================================
# _build_dataloader
# ===========================================================================

def _build_dataloader(
    dataset_name: str,
    fallback_dataset: str,
    batch_size: int,
    seq_len: int,
) -> DataLoader:
    """
    Construct a DataLoader from the specified dataset with automatic fallback.

    Attempts to load dataset_name from HuggingFace datasets. If that fails
    (network error, missing package), falls back to fallback_dataset. If both
    fail, generates a SyntheticDataset and prints a warning.

    NOT a biological concept. Engineering utility function.

    Args:
        dataset_name: Primary HuggingFace dataset identifier.
        fallback_dataset: Fallback HuggingFace dataset identifier.
        batch_size: DataLoader batch size.
        seq_len: Token sequence length.

    Returns:
        DataLoader yielding (batch_size, seq_len) integer tensors.
    """
    def _try_hf(name: str) -> Optional[List[str]]:
        """Attempt to load a HuggingFace dataset and return its text list."""
        try:
            from datasets import load_dataset  # type: ignore[import]
            print(f"Loading dataset: {name}")
            ds = load_dataset(name, split="train", streaming=False)
            texts = []
            for item in ds:
                if "text" in item and item["text"]:
                    texts.append(item["text"])
                if len(texts) >= 50000:
                    break
            if texts:
                print(f"Loaded {len(texts)} samples from {name}")
                return texts
        except Exception as err:
            print(f"Could not load {name}: {err}")
        return None

    texts = _try_hf(dataset_name)
    if texts is None:
        texts = _try_hf(fallback_dataset)

    if texts is not None:
        dataset: Dataset = SimpleTextDataset(texts, seq_len=seq_len)
    else:
        print(
            "WARNING: Both primary and fallback datasets failed to load. "
            "Using synthetic random data. The model will not learn meaningful representations."
        )
        dataset = SyntheticDataset(n_samples=20000, seq_len=seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )


# ===========================================================================
# train_small_core
# ===========================================================================

def train_small_core(
    n_steps: int = 5000,
    batch_size: int = 32,
    seq_len: int = 128,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "checkpoints",
    log_interval: int = 50,
    sleep_interval: int = 500,
    eval_interval: int = 200,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    fallback_dataset: str = "roneneldan/TinyStories",
) -> None:
    """
    Main training loop for the PRAGMI small core integration harness.

    Wires together all five PRAGMI subsystems (CognitiveKernel,
    WorldModelEnsemble, NeuromodulatorBroadcast, EpistemicSelector,
    CorticalBuffer) into a single training loop operating on real text data.

    BIOLOGICAL GROUNDING: This loop approximates the biological wake-sleep
    cycle. During each step (wake phase), the encoder processes a token
    batch, the CognitiveKernel encodes the experience, the WorldModelEnsemble
    refines its generative model, and NeuromodulatorBroadcast updates global
    chemical state. Every sleep_interval steps, the kernel consolidates memory
    (hippocampal replay and synaptic homeostasis) and the neuromodulator
    records a sleep cycle for maturity computation.

    Diekelmann S, Born J (2010). DOI: 10.1038/nrn2762

    NOT a biological quantity: the specific hyperparameters (n_steps,
    batch_size, lr, etc.) are engineering choices without direct biological
    analogs. The sleep_interval is an engineering approximation of the
    biological sleep-wake cycle period.

    OPTIMIZER DESIGN: The kernel optimizer and encoder optimizer are SEPARATE.
    They do not share parameter groups. The WorldModelEnsemble owns its
    internal optimizer. The FixedCoordinateProjection contributes zero
    gradient (all forward passes under torch.no_grad()). W_recurrent and
    distance_mask in CA3 are registered buffers and do not appear in any
    optimizer.

    Args:
        n_steps: Total number of training steps.
        batch_size: Number of samples per training batch.
        seq_len: Token sequence length.
        lr: Peak learning rate for encoder and kernel optimizers.
        device: Compute device ('cpu' or 'cuda').
        checkpoint_dir: Directory for checkpoint files.
        log_interval: Steps between console log entries.
        sleep_interval: Steps between sleep consolidation events.
        eval_interval: Steps between EpistemicSelector diagnostic evaluations.
        dataset_name: Primary HuggingFace dataset identifier.
        fallback_dataset: Fallback HuggingFace dataset identifier.
    """
    print(f"\nTraining device: {device}")
    print(f"Steps: {n_steps}  Batch: {batch_size}  Seq: {seq_len}  LR: {lr}")
    print(f"Sleep interval: {sleep_interval}  Eval interval: {eval_interval}")
    print(f"Checkpoint dir: {checkpoint_dir}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Instantiate all subsystems
    # ------------------------------------------------------------------

    # LightweightEncoder: stand-in for TimmyModel.
    # NOT a biological quantity: engineering stand-in.
    encoder = LightweightEncoder(vocab_size=50257, d_model=256, seq_len=seq_len).to(device)

    # FixedCoordinateProjection: stand-in for PerforantPathSymphonyBridge.
    # Receives: encoder embeddings (B, 256).
    # Sends: coordinate-space representations (B, 64) to CognitiveKernel.
    # Anatomical connection: perforant path (entorhinal cortex -> hippocampus).
    # Steward O, Scoville SA (1976). DOI: 10.1002/cne.901690105
    coord_proj = FixedCoordinateProjection(input_dim=256, output_dim=64, seed=42).to(device)

    # CognitiveKernel: hippocampal formation.
    # Receives: 64-dim coordinate representations via perforant path analog.
    # Returns: reconstructed coordinates, novelty signal, diagnostics.
    kernel = CognitiveKernel(CognitiveKernelConfig()).to(device)

    # WorldModelEnsemble: neocortical generative model (predictive coding).
    # Receives: coordinate representations from FixedCoordinateProjection.
    # Returns: ensemble predictions, mean prediction, epistemic uncertainty.
    # Friston K (2010). DOI: 10.1038/nrn2787
    world_model = WorldModelEnsemble(WorldModelConfig()).to(device)

    # NeuromodulatorBroadcast: diffuse neuromodulatory systems.
    # DA, NE, ACh, 5-HT baselines and global maturity.
    # Yu AJ, Dayan P (2005). DOI: 10.1016/j.neuron.2005.04.026
    neuromod = NeuromodulatorBroadcast(NeuromodulatorConfig()).to(device)

    # CorticalBuffer: PFC delay-period working memory.
    # Instantiated and exercised in tests. In this harness it is idle during
    # training because it requires TimmyModel's membrane state (v_mem) to
    # be meaningful. Wired in when TimmyModel arrives.
    # Goldman-Rakic PS (1995). DOI: 10.1016/0896-6273(95)90304-6
    cortical_buffer = CorticalBuffer(CorticalBufferConfig(d_model=256, buffer_dim=32)).to(device)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    # Encoder optimizer. Separate from kernel optimizer.
    # Encoder parameters only: embedding, pool, proj.
    # NOT a biological quantity: Adam optimizer is a training artifact.
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    # Kernel optimizer. Separate from encoder optimizer.
    # Kernel parameters only: learnable Linear layers in EC, DG, CA3, CA1, Sub.
    # W_recurrent and distance_mask are registered buffers; they are excluded
    # automatically because buffer tensors do not have requires_grad=True.
    # Verify: filter to only parameters with requires_grad=True.
    kernel_params = [p for p in kernel.parameters() if p.requires_grad]
    kernel_optimizer = torch.optim.Adam(kernel_params, lr=lr)

    # WorldModelEnsemble owns its internal optimizer. Do not create an external
    # one. Call world_model.update() to train it.

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    loader = _build_dataloader(dataset_name, fallback_dataset, batch_size, seq_len)
    data_iter = iter(loader)

    def _next_batch() -> torch.Tensor:
        """Fetch the next batch, cycling the loader as needed."""
        nonlocal data_iter
        try:
            return next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter).to(device)

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------

    # Running statistics for loss normalization.
    # Used to compute a normalized reward signal for dopamine update.
    # NOT a biological quantity: engineering normalization.
    loss_ema: float = 1.0
    loss_var_ema: float = 1.0
    loss_ema_decay: float = 0.99

    # Store previous step coordinates for WorldModelEnsemble.update() target.
    # The world model predicts next_coords from current coords. We use the
    # previous step's coordinates as the target for the current step's update.
    # NOT a biological quantity: engineering delay-one training target.
    prev_coords: Optional[torch.Tensor] = None

    loss_history: List[float] = []
    t_start = time.time()

    print("Starting training loop...")
    print("-" * 70)

    for step in range(n_steps):
        # ----------------------------------------------------------------
        # Apply learning rate schedules to both optimizers
        # ----------------------------------------------------------------
        current_lr = cosine_lr(step, n_steps, lr, warmup_steps=200)
        for param_group in encoder_optimizer.param_groups:
            param_group["lr"] = current_lr
        for param_group in kernel_optimizer.param_groups:
            param_group["lr"] = current_lr

        # ----------------------------------------------------------------
        # Step 1: Get a batch and encode to embeddings
        # ----------------------------------------------------------------
        token_ids = _next_batch()  # (B, seq_len)
        embeddings = encoder(token_ids)  # (B, 256)

        # ----------------------------------------------------------------
        # Step 2: Project to coordinate space (no gradient through projection)
        # ----------------------------------------------------------------
        # ANATOMICAL INTERFACE:
        #   Sending: LightweightEncoder (stand-in for TimmyModel).
        #   Receiving: CognitiveKernel via EntorhinalCortex.
        #   Connection: Perforant path analog.
        #   Steward O, Scoville SA (1976). DOI: 10.1002/cne.901690105
        coords = coord_proj(embeddings)  # (B, 64), no gradient, requires_grad=False

        # ----------------------------------------------------------------
        # Step 3: CognitiveKernel forward
        # ----------------------------------------------------------------
        reconstruction, novelty, diagnostics = kernel(coords)
        # reconstruction: (B, 64) — subiculum output reconstructing EC input
        # novelty: (B,) — CA1 comparator novelty signal
        # diagnostics: dict with astro_eta, dg_sparsity, etc.

        # ----------------------------------------------------------------
        # Step 4: Kernel reconstruction loss and backprop
        # ----------------------------------------------------------------
        # Target: EC-normalized input (available in diagnostics as 'ec_output').
        # If 'ec_output' is not present (production kernel strips diagnostics),
        # fall back to raw coordinates.
        # NOT a biological quantity: MSE reconstruction loss is a training
        # artifact. The biological analog is prediction error minimization.
        # Friston K (2010). DOI: 10.1038/nrn2787
        ec_target = diagnostics.get("ec_output", None)
        if isinstance(ec_target, torch.Tensor):
            kernel_target = ec_target.detach()
        else:
            kernel_target = coords.detach()

        kernel_loss = F.mse_loss(reconstruction, kernel_target)

        kernel_optimizer.zero_grad()
        kernel_loss.backward()
        kernel_optimizer.step()

        # ----------------------------------------------------------------
        # Step 5: Encoder loss
        # Encoder receives a training signal: minimize MSE between its
        # projected output and the kernel's reconstruction (detached).
        # This gives the encoder a learning signal without coupling gradients
        # back through the kernel.
        # NOT a biological quantity: cross-module training signal is an
        # engineering artifact.
        # ----------------------------------------------------------------
        # Apply the frozen projection directly rather than through
        # coord_proj.forward(), which is @torch.no_grad() and would cut the
        # encoder off from its own loss. The projection is a buffer, so it
        # still receives no update; only the encoder does.
        enc_coords = embeddings @ coord_proj.projection
        encoder_target = reconstruction.detach()  # (B, 64), detached from kernel graph
        # encoder_target is in 64-dim space; enc_coords is also 64-dim.
        encoder_loss = F.mse_loss(enc_coords, encoder_target)

        encoder_optimizer.zero_grad()
        encoder_loss.backward()
        encoder_optimizer.step()

        # ----------------------------------------------------------------
        # Step 6: WorldModelEnsemble forward and update
        # ----------------------------------------------------------------
        _, _, wm_variance = world_model(coords.detach())
        ensemble_variance_scalar = wm_variance.detach().item()

        if prev_coords is not None:
            # Train world model: predict coords from prev_coords.
            # target is the current step's coordinates.
            # prev_coords and coords are both detached inside world_model.update().
            wm_loss_val = world_model.update(prev_coords, coords)
        else:
            wm_loss_val = float(kernel_loss.item())

        prev_coords = coords.detach().clone()

        # ----------------------------------------------------------------
        # Step 7: Loss normalization for neuromodulator signal
        # ----------------------------------------------------------------
        raw_loss = kernel_loss.item()
        loss_history.append(raw_loss)

        # Update running EMA and variance for normalization.
        # NOT a biological quantity: running statistics are a training artifact.
        delta = raw_loss - loss_ema
        loss_ema = loss_ema_decay * loss_ema + (1.0 - loss_ema_decay) * raw_loss
        loss_var_ema = loss_ema_decay * loss_var_ema + (1.0 - loss_ema_decay) * delta ** 2
        normalized_loss_signal = delta / (math.sqrt(loss_var_ema) + 1e-8)
        # Clamp to [0, 1] for DA update: positive signal = better than expected.
        da_signal = float(torch.clamp(torch.tensor(-normalized_loss_signal * 0.5 + 0.5), 0.0, 1.0))

        # ----------------------------------------------------------------
        # Step 8: NeuromodulatorBroadcast updates
        # ----------------------------------------------------------------
        # Dopamine encodes reward prediction error (loss improvement proxy).
        # Schultz W (2016). DOI: 10.1038/nrn.2015.26
        neuromod.update_da(da_signal)

        # Norepinephrine encodes unexpected uncertainty (world model variance).
        # Aston-Jones G, Cohen JD (2005). DOI: 10.1146/annurev.neuro.28.061604.135709
        ne_signal = float(torch.clamp(torch.tensor(float(ensemble_variance_scalar) * 10.0), 0.0, 1.0))
        neuromod.update_ne(ne_signal)

        # ----------------------------------------------------------------
        # Step 9: Astrocytic eta modulates kernel optimizer learning rate
        # ----------------------------------------------------------------
        # The astrocytic regulator returns eta in [astro_eta_min, astro_eta_max].
        # We multiply the current cosine LR by eta to get the effective LR.
        # High calcium (overactive network) -> eta < 1 (slow down).
        # Low calcium (underactive network) -> eta > 1 (speed up).
        # This is an engineering approximation of astrocytic synaptic scaling.
        # Perea G et al. (2009). DOI: 10.1038/nrn2722
        # NOT a biological quantity in this parameterization.
        eta = diagnostics.get("astro_eta", 1.0)
        effective_kernel_lr = current_lr * eta
        for param_group in kernel_optimizer.param_groups:
            param_group["lr"] = effective_kernel_lr

        # ----------------------------------------------------------------
        # Step 10: EpistemicSelector diagnostic (every eval_interval steps)
        # ----------------------------------------------------------------
        if (step + 1) % eval_interval == 0:
            with torch.no_grad():
                # Generate 8 candidate coordinate batches for scoring.
                # In a full implementation these would be real data candidates.
                # Here we generate random candidates to exercise the selector.
                # NOT a biological quantity: diagnostic-only evaluation.
                candidates = [torch.randn(batch_size, 64, device=device) for _ in range(8)]
                maturity_now = neuromod.global_maturity.item()
                sel_idx, sel_batch, sel_scores = select_batch_deterministic(
                    candidates, world_model, global_maturity=maturity_now
                )
                print(
                    f"  [Epistemic eval @ step {step+1}] "
                    f"maturity={maturity_now:.4f}, "
                    f"selected_idx={sel_idx}, "
                    f"max_var={max(sel_scores):.6f}, "
                    f"min_var={min(sel_scores):.6f}"
                )

        # ----------------------------------------------------------------
        # Step 11: Sleep consolidation (every sleep_interval steps)
        # ----------------------------------------------------------------
        if (step + 1) % sleep_interval == 0:
            sleep_diag = kernel.sleep_consolidation()
            neuromod.compute_maturity(
                routing_entropy=0.5,      # placeholder: no routing in this harness
                loss_value=raw_loss,
                probe_response=0.5,       # placeholder: no probe in this harness
                mean_recent_variance=world_model.mean_recent_variance(),
            )
            # record_sleep_cycle: NeuromodulatorBroadcast tracks elapsed cycles
            # internally via _sleep_cycles_elapsed, incremented in compute_maturity.
            print(
                f"\n  [Sleep @ step {step+1}] "
                f"episodes_in_CA3={sleep_diag['episodes_in_ca3']}, "
                f"maturity={neuromod.global_maturity.item():.4f}\n"
            )

        # ----------------------------------------------------------------
        # Step 12: Checkpoint (every 1000 steps)
        # ----------------------------------------------------------------
        if (step + 1) % 1000 == 0:
            ckpt_path = Path(checkpoint_dir) / f"ckpt_step_{step+1}.pt"
            checkpoint = {
                "step": step + 1,
                "encoder_state_dict": encoder.state_dict(),
                "kernel_memory_state": kernel.get_memory_state(),
                "world_model_hot_state": world_model.get_hot_state(),
                "neuromod_hot_state": neuromod.get_hot_state(),
                "cortical_buffer_hot_state": cortical_buffer.get_hot_state(),
                "loss_history": loss_history[-1000:],  # last 1000 losses
            }
            torch.save(checkpoint, ckpt_path)
            print(f"  [Checkpoint saved] {ckpt_path}")

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / elapsed
            print(
                f"step {step+1:5d}/{n_steps} | "
                f"kernel_loss={raw_loss:.4f} | "
                f"enc_loss={encoder_loss.item():.4f} | "
                f"wm_loss={wm_loss_val:.4f} | "
                f"DA={neuromod.da.item():.4f} | "
                f"NE={neuromod.ne.item():.4f} | "
                f"eta={eta:.3f} | "
                f"episodes={diagnostics.get('num_stored_episodes', 0)} | "
                f"{steps_per_sec:.1f} steps/s"
            )

    print("\nTraining complete.")
    final_ckpt = Path(checkpoint_dir) / "ckpt_final.pt"
    checkpoint = {
        "step": n_steps,
        "encoder_state_dict": encoder.state_dict(),
        "kernel_memory_state": kernel.get_memory_state(),
        "world_model_hot_state": world_model.get_hot_state(),
        "neuromod_hot_state": neuromod.get_hot_state(),
        "cortical_buffer_hot_state": cortical_buffer.get_hot_state(),
        "loss_history": loss_history,
    }
    torch.save(checkpoint, final_ckpt)
    print(f"Final checkpoint saved: {final_ckpt}")


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    """
    CLI entry point for the PRAGMI small core integration harness.

    Parses command-line arguments, runs the integration test suite, and
    starts training if all tests pass (unless --test-only is specified).

    NOT a biological concept. Engineering entry point.
    """
    parser = argparse.ArgumentParser(
        description="PRAGMI small core integration harness: tests + training loop."
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run the integration test suite and exit without training.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=5000,
        help="Total training steps (default: 5000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: auto-detect cuda or cpu).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="Primary HuggingFace dataset (default: HuggingFaceFW/fineweb-edu).",
    )
    parser.add_argument(
        "--fallback-dataset",
        type=str,
        default="roneneldan/TinyStories",
        help="Fallback HuggingFace dataset (default: roneneldan/TinyStories).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints).",
    )
    parser.add_argument(
        "--sleep-interval",
        type=int,
        default=500,
        help="Steps between sleep consolidation events (default: 500).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Steps between console log entries (default: 50).",
    )

    args = parser.parse_args()

    all_passed = run_tests()

    if args.test_only:
        sys.exit(0 if all_passed else 1)

    if not all_passed:
        print("\nERROR: Integration tests failed. Training will not start.")
        print("Fix the failing tests before proceeding.")
        sys.exit(1)

    print("\nAll tests passed. Starting training...\n")
    train_small_core(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        sleep_interval=args.sleep_interval,
        log_interval=args.log_interval,
        dataset_name=args.dataset,
        fallback_dataset=args.fallback_dataset,
    )


if __name__ == "__main__":
    main()
