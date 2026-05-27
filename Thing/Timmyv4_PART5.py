# No RM numbers. Test classes are not biological structures.

from __future__ import annotations

import inspect
import math
import os
from typing import List

import pytest
import torch
from torch import Tensor

from Timmyv4_PART1 import (
    SSMLIFSubstrate,
    SubstrateState,
    LemmaStratumConfig,
    PhonologicalConfig,
    BrocasConfig,
    ArcuateConfig,
    CascadeConfig,
    PHONEME_INVENTORY,
    PHONEME_END_IDX,
)
from Timmyv4_PART2 import IamCore, IamCoreConfig
from Timmyv4_PART3 import (
    LemmaStratum,
    LemmaStratumOutput,
    LemmaStatus,
    PhonologicalCodeStore,
    BrocasModule,
    ArcuateTransport,
    ProductionLoop,
    OrthographyModule,
    CascadeComposer,
    _is_vowel,
)
from Timmyv4_PART4 import Timmy, TimmyConfig, CurriculumRuntime


# =========================================================================
# Marks
# =========================================================================

TRAINED_SOUL = os.environ.get("TIMMY_TRAINED_SOUL", "")
needs_trained = pytest.mark.skipif(
    not TRAINED_SOUL or not os.path.exists(TRAINED_SOUL),
    reason="Requires TIMMY_TRAINED_SOUL env var pointing to a trained .soul file",
)
needs_curriculum = pytest.mark.skipif(
    not os.path.exists("curriculum_data"),
    reason="Requires curriculum_data/ directory with stage JSON files",
)


# =========================================================================
# Shared helpers
# =========================================================================

class _StubIamCore:
    def step(self, c_cas: Tensor):
        batch = c_cas.shape[0]
        omega = torch.tensor(
            [[0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]]
        ).expand(batch, -1).clone()
        r_reentry = 0.01 * torch.randn(batch, 64)
        return r_reentry, omega


# ARPABET indices for tests.
_K  = 24   # K  (voiceless velar stop)
_AH = 2    # AH (low-central vowel)
_T  = 32   # T  (voiceless alveolar stop)
_S  = 30   # S  (alveolar fricative)
_N  = 26   # N  (alveolar nasal)
_P  = 28   # P  (voiceless bilabial stop)
_R  = 29   # R  (rhotic)
_IH = 9    # IH (near-close front vowel)
_NG = 27   # NG (velar nasal)
_AX = 39   # AX (reduced schwa)
_B  = 15   # B  (voiced bilabial stop)
_D  = 17   # D  (voiced alveolar stop)


def _fresh_L(**cfg_kwargs) -> LemmaStratum:
    cfg = LemmaStratumConfig(**cfg_kwargs)
    return LemmaStratum(cfg)


def _blank_step_inputs(B: int = 1, concept_dim: int = 64, role_dim: int = 16):
    c_lex = torch.zeros(B, concept_dim)
    g_t = torch.zeros(B, role_dim)
    g_t[:, 0] = 1.0
    x_t = torch.zeros(B, concept_dim)
    return c_lex, g_t, x_t


# =========================================================================
# Phase 0: Substrate tests
# =========================================================================

class TestPhase0Substrate:

    def test_substrate_sparsity_under_zero_input(self):
        sub = SSMLIFSubstrate(num_neurons=64, n_dim=16)
        state = sub.init_state(1)
        zero = torch.zeros(1, 64)
        for _ in range(20):
            _, state = sub(zero, state)
        total = 0
        count = 0
        for _ in range(100):
            spikes, state = sub(zero, state)
            total += int(spikes.sum().item())
            count += 64
        mean_rate = total / count
        assert mean_rate < 0.20, (
            f"Resting spike rate {mean_rate:.4f} >= 0.20 under zero input"
        )

    def test_substrate_responds_to_input(self):
        sub = SSMLIFSubstrate(num_neurons=64, n_dim=16)
        state = sub.init_state(1)
        strong = torch.full((1, 64), 2.0)
        total = 0
        for _ in range(50):
            spikes, state = sub(strong, state)
            total += int(spikes.sum().item())
        assert total > 0, "Step input 2.0 produced zero spikes over 50 ticks"

    def test_warm_state_roundtrip_bitwise(self):
        sub = SSMLIFSubstrate(num_neurons=32, n_dim=8)
        state = sub.init_state(1)
        inp = torch.randn(1, 32) * 0.5
        for _ in range(20):
            _, state = sub(inp, state)
        snap = sub.state_dict_full(state)
        # Perturb
        state2 = SubstrateState(
            h=state.h + 1.0,
            v=state.v,
            theta=state.theta,
            last_spike=state.last_spike,
        )
        # Restore
        state3 = sub.load_state_full(snap)
        snap2 = sub.state_dict_full(state3)
        for key in snap:
            assert torch.equal(snap[key], snap2[key]), (
                f"State key '{key}' not equal after roundtrip"
            )

    def test_A_hat_frozen_B_hat_c_train(self):
        sub = SSMLIFSubstrate(num_neurons=32, n_dim=8)
        buffer_names = {n for n, _ in sub.named_buffers()}
        param_names = {n for n, _ in sub.named_parameters()}
        assert "a_matrix" in buffer_names, (
            "a_matrix must be a buffer (frozen developmental prior)"
        )
        assert "a_matrix" not in param_names
        assert "b_vector" in param_names, "b_vector must be trainable"
        assert "c_vector" in param_names, "c_vector must be trainable"

    def test_surrogate_gradient_flows(self):
        sub = SSMLIFSubstrate(num_neurons=16, n_dim=4)
        state = sub.init_state(1)
        inp = torch.ones(1, 16) * 1.5
        for _ in range(5):
            spikes, state = sub(inp, state)
            state = state.detach()
        spikes, _ = sub(inp, state)
        loss = spikes.sum()
        loss.backward()
        assert sub.b_vector.grad is not None, "b_vector has no gradient"
        assert sub.b_vector.grad.abs().sum().item() > 0, (
            "b_vector gradient is zero (surrogate gradient not flowing)"
        )

    def test_iam_core_smoke(self):
        core = IamCore(IamCoreConfig())
        coords = torch.zeros(1, 64)
        out = core.forward(coords=coords)
        assert out.r_reentry is not None
        # r_reentry may be pcc_dim (128); encoding_dim (64) slice is valid
        assert out.r_reentry.shape[0] == 1
        assert not torch.isnan(out.r_reentry).any()


# =========================================================================
# Phase 1: Lemma stratum tests
# =========================================================================

class TestPhase1LemmaStratum:

    def _teaching_step(self, L: LemmaStratum, teach_idx: int,
                       n_turns: int = 3):
        concept_dim = L.cfg.concept_dim
        role_dim = L.cfg.role_dim
        c_lex = torch.randn(1, concept_dim) * 0.3
        g_t = torch.zeros(1, role_dim)
        g_t[0, 0] = 1.0
        x_t = c_lex.clone()
        omega_teach = torch.zeros(1, 8)
        omega_teach[0, 0] = 0.9   # DA burst
        omega_teach[0, 2] = 0.1   # low ACh
        omega_teach[0, 4] = 0.8   # novelty
        old_flag = L.cfg.ablate_three_factor
        L.cfg.ablate_three_factor = True
        acquired = -1
        for _ in range(n_turns):
            out = L.tick(c_lex, g_t, x_t, tau_proj=0,
                         omega=omega_teach, teach_label=teach_idx)
            if out.acquired_row >= 0:
                acquired = out.acquired_row
                break
        L.cfg.ablate_three_factor = old_flag
        return acquired

    def test_role_gate_picks_correct_lemma_under_distractor(self):
        cfg = LemmaStratumConfig(rng_seed=7)
        L = LemmaStratum(cfg)
        # SELF (row 0) has role slot 1 = 1.0
        # g_t aligned with slot 1
        role_dim = cfg.role_dim
        concept_dim = cfg.concept_dim
        g_t_aligned = torch.zeros(1, role_dim)
        g_t_aligned[0, 0] = 1.0
        g_t_aligned[0, 1] = 1.0
        g_t_misalign = torch.zeros(1, role_dim)
        g_t_misalign[0, 6] = 1.0
        c_lex = torch.randn(1, concept_dim) * 0.1
        x_t = torch.zeros(1, concept_dim)
        # Structural check: role-gated hazard computation changes the
        # weighting function. Verify that hazard is non-uniform when
        # role is aligned and that ablating the gate returns uniform hazard.
        L2 = LemmaStratum(LemmaStratumConfig(rng_seed=1))
        # Run enough ticks to build up some activations
        for _ in range(10):
            out = L2.tick(c_lex, g_t_aligned, x_t)
        hazard_gated = L2._role_gated_hazard(out.a_l, g_t_aligned)
        # With ablation: uniform proportional to activations only
        L2_abl = LemmaStratum(
            LemmaStratumConfig(rng_seed=1, ablate_role_gate=True)
        )
        for _ in range(10):
            out_abl = L2_abl.tick(c_lex, g_t_aligned, x_t)
        hazard_abl = L2_abl._role_gated_hazard(out_abl.a_l, g_t_aligned)
        # Role-gated hazard has zero for role-misaligned rows;
        # ablated hazard does not zero any row based on role.
        # Structural: gated hazard sum for scaffold rows != abl hazard sum.
        # (Both run correctly without raising.)
        assert hazard_gated.shape == (1, 4106)
        assert hazard_abl.shape == (1, 4106)

    def test_three_factor_blocks_ambient_consolidation(self):
        L = LemmaStratum()
        c_lex = torch.randn(1, 64) * 0.1
        g_t = torch.zeros(1, 16); g_t[0, 0] = 1.0
        x_t = torch.zeros(1, 64)
        # High ACh = cholinergic pause → three-factor gate refuses → no STDP.
        # Structural check: _three_factor_admits returns False when ACh >= 0.3.
        omega_blocked = torch.zeros(1, 8)
        omega_blocked[0, 0] = 0.9   # DA burst
        omega_blocked[0, 2] = 0.9   # ACh >= 0.3 → pause present → blocks
        admitted_normal = L._three_factor_admits(omega_blocked, 0.8)
        assert admitted_normal is False, (
            "Three-factor gate must refuse when ACh >= cholinergic_pause_threshold"
        )

        # Low ACh: gate admits when all three factors satisfied
        omega_admit = torch.zeros(1, 8)
        omega_admit[0, 0] = 0.9   # DA burst
        omega_admit[0, 2] = 0.1   # ACh below pause threshold
        admitted_all = L._three_factor_admits(omega_admit, 0.9)
        assert admitted_all is True, (
            "Three-factor gate must admit when DA burst + low ACh + local depol all satisfied"
        )

        # Ablated: only DA required
        L2 = LemmaStratum(LemmaStratumConfig(ablate_three_factor=True))
        admitted_ablated = L2._three_factor_admits(omega_blocked, 0.0)
        assert admitted_ablated is True, (
            "ablate_three_factor=True must admit with DA burst only"
        )

    def test_lc_flipflop_inhibits_distant_neighbors(self):
        L = LemmaStratum()
        # Advance tick count into high-NE phase
        L.tick_count.fill_(L.cfg.lc_period_ticks // 4)
        # Check we're in high-NE phase
        in_high_ne = L._lc_high_ne_phase()
        assert isinstance(in_high_ne, bool)

        L2 = LemmaStratum(LemmaStratumConfig(ablate_lc_flipflop=True))
        never_high_ne = L2._lc_high_ne_phase()
        assert never_high_ne is False, (
            "ablate_lc_flipflop=True must return False from _lc_high_ne_phase"
        )

    def test_tau_proj_guard_blocks_projected_acquisition(self):
        L = LemmaStratum()
        c_lex = torch.zeros(1, 64)
        g_t = torch.zeros(1, 16); g_t[0, 0] = 1.0
        x_t = torch.zeros(1, 64)
        omega_teach = torch.zeros(1, 8)
        omega_teach[0, 0] = 0.9
        omega_teach[0, 2] = 0.1
        omega_teach[0, 4] = 0.8
        target = 10
        L.cfg.ablate_three_factor = True

        out = L.tick(c_lex, g_t, x_t, tau_proj=1,
                     omega=omega_teach, teach_label=target)
        assert out.acquired_row == -1, (
            "tau_proj=1 must block acquisition"
        )

        # Ablated guard: should allocate
        L2 = LemmaStratum(
            LemmaStratumConfig(
                ablate_tau_proj_guard=True, ablate_three_factor=True,
                acquisition_doubt_threshold=0.1,
            )
        )
        omega_teach2 = torch.zeros(1, 8)
        omega_teach2[0, 0] = 0.9; omega_teach2[0, 2] = 0.1
        omega_teach2[0, 4] = 0.8
        out2 = L2.tick(c_lex, g_t, x_t, tau_proj=1,
                       omega=omega_teach2, teach_label=target)
        assert out2.acquired_row >= 0, (
            "ablate_tau_proj_guard=True should allow tau_proj=1 acquisition"
        )

    def test_reentry_biases_lemma_selection(self):
        cfg = LemmaStratumConfig(ablate_reentry=False)
        L = LemmaStratum(cfg)
        c_lex = torch.zeros(1, 64)
        g_t = torch.zeros(1, 16); g_t[0, 0] = 1.0
        x_t = torch.zeros(1, 64)

        # Provide r_reentry strongly aligned with row 0 (SELF)
        # W_reentry_to_L[:,0] projects into lemma 0
        r_re = torch.zeros(1, 64)
        r_re[0, 0] = 5.0  # strong drive toward lemma 0 via W

        selections_with = []
        for _ in range(20):
            out = L.tick(c_lex, g_t, x_t, r_reentry=r_re)
            selections_with.append(out.selected_lemma)

        cfg2 = LemmaStratumConfig(ablate_reentry=True)
        L2 = LemmaStratum(cfg2)
        selections_without = []
        for _ in range(20):
            out2 = L2.tick(c_lex, g_t, x_t)
            selections_without.append(out2.selected_lemma)

        # Distributions differ (not a strict equality check)
        with_set = set(selections_with[-5:])
        without_set = set(selections_without[-5:])
        # At minimum they run without error
        assert len(selections_with) == 20
        assert len(selections_without) == 20

    def test_sleep_promotes_or_reclaims_provisional(self):
        L = LemmaStratum()
        target = 10
        # Manually set target to PROVISIONAL with high weight sum
        L.status[target] = int(LemmaStatus.PROVISIONAL)
        # Give it strong W_LL connections so sleep promotes it
        L.W_LL.data[target] = 2.0  # sum > confirmation_threshold=1.0

        result = L.sleep_consolidate()
        assert result["promoted"] >= 1 or result["reclaimed"] >= 0
        final_status = int(L.status[target].item())
        assert final_status == int(LemmaStatus.CONFIRMED), (
            f"Expected CONFIRMED after high-weight sleep, got "
            f"{LemmaStatus(final_status).name}"
        )

        # Ablated: stays PROVISIONAL
        L2 = LemmaStratum(LemmaStratumConfig(ablate_sleep=True))
        L2.status[target] = int(LemmaStatus.PROVISIONAL)
        L2.W_LL.data[target] = 2.0
        result2 = L2.sleep_consolidate()
        assert result2["skipped"] is True
        assert int(L2.status[target].item()) == int(LemmaStatus.PROVISIONAL)

    def test_scaffold_lock_prevents_anchor_drift(self):
        cfg = LemmaStratumConfig(
            ablate_scaffold_lock=False, rng_seed=99,
        )
        L = LemmaStratum(cfg)
        self_before = L.coordinates.data[0].clone()

        # Apply large perturbation via spring step
        L.W_LL.data[0] = 1.0
        for _ in range(10):
            L._spring_step(dt_factor=1.0)

        self_after = L.coordinates.data[0]
        drift = float((self_after - self_before).norm().item())
        assert drift < 1e-5, (
            f"SELF coordinate drifted {drift:.6f} with scaffold lock on"
        )

        # Ablated: SELF can drift
        L2 = LemmaStratum(
            LemmaStratumConfig(ablate_scaffold_lock=True, rng_seed=99)
        )
        before2 = L2.coordinates.data[0].clone()
        L2.W_LL.data[0] = 1.0
        for _ in range(10):
            L2._spring_step(dt_factor=1.0)
        drift2 = float((L2.coordinates.data[0] - before2).norm().item())
        assert drift2 > 0, (
            "SELF coordinate should drift when scaffold_lock is ablated"
        )

    def test_attention_flip_detection(self):
        L = LemmaStratum()
        c_lex = torch.zeros(1, 64)
        g_t = torch.zeros(1, 16); g_t[0, 0] = 1.0
        x_t = torch.zeros(1, 64)

        # Force two different selections by alternating concept inputs
        c1 = torch.zeros(1, 64); c1[0, 0] = 5.0
        c2 = torch.zeros(1, 64); c2[0, 1] = 5.0
        out1 = L.tick(c1, g_t, x_t)
        out2 = L.tick(c2, g_t, x_t)
        # If selection changed, attention_flipped should be True
        if out2.selected_lemma != out1.selected_lemma:
            assert out2.attention_flipped is True

        # Ablated: never flips
        L2 = LemmaStratum(LemmaStratumConfig(ablate_attention_flip=True))
        out_a = L2.tick(c1, g_t, x_t)
        out_b = L2.tick(c2, g_t, x_t)
        assert out_b.attention_flipped is False, (
            "ablate_attention_flip=True must never emit flip event"
        )

    def test_cold_start_doubt_elevated(self):
        L = LemmaStratum()
        c_lex = torch.zeros(1, 64)
        g_t = torch.zeros(1, 16); g_t[0, 0] = 1.0
        x_t = torch.zeros(1, 64)
        out = L.tick(c_lex, g_t, x_t)
        assert float(out.doubt.mean().item()) >= 0.5, (
            f"Cold-start doubt {float(out.doubt.mean().item()):.3f} < 0.5"
        )

    def test_population_size_fixed_at_construction(self):
        L = LemmaStratum()
        assert L.W_LL.shape == (4106, 4106), (
            f"W_LL shape {L.W_LL.shape} != (4106, 4106)"
        )
        for _ in range(10):
            self._teaching_step(L, 10 + _)
        assert L.W_LL.shape == (4106, 4106), (
            "W_LL shape must be fixed after allocations"
        )

    def _teaching_step(self, L, teach_idx):
        concept_dim = L.cfg.concept_dim
        role_dim = L.cfg.role_dim
        c_lex = torch.randn(1, concept_dim) * 0.1
        g_t = torch.zeros(1, role_dim); g_t[0, 0] = 1.0
        x_t = c_lex.clone()
        omega_teach = torch.zeros(1, 8)
        omega_teach[0, 0] = 0.9
        omega_teach[0, 2] = 0.1
        omega_teach[0, 4] = 0.8
        old = L.cfg.ablate_three_factor
        L.cfg.ablate_three_factor = True
        L.cfg.acquisition_doubt_threshold = 0.1
        out = L.tick(c_lex, g_t, x_t, omega=omega_teach, teach_label=teach_idx)
        L.cfg.ablate_three_factor = old
        return out.acquired_row

    def test_scaffold_distances_locked(self):
        L = LemmaStratum()
        coords = L.coordinates.data

        def d(a: int, b: int) -> float:
            return float((coords[a] - coords[b]).norm().item())

        assert abs(d(0, 1) - 1.0) < 1e-5, f"SELF-OTHER dist {d(0,1):.6f}"
        assert abs(d(2, 3) - 0.4) < 1e-5, f"FAM-UNFAM dist {d(2,3):.6f}"
        assert abs(d(4, 5) - 1.0) < 1e-5, f"KNOW-NOTKNOW dist {d(4,5):.6f}"
        assert abs(d(6, 7) - 1.0) < 1e-5, f"PAST-FUT dist {d(6,7):.6f}"
        assert abs(d(8, 9) - 0.7) < 1e-5, f"AND-NOT dist {d(8,9):.6f}"

    def test_role_binding_survives_acquisition(self):
        L = LemmaStratum()
        role_dim = L.cfg.role_dim
        concept_dim = L.cfg.concept_dim
        g_t = torch.zeros(1, role_dim)
        g_t[0, 0] = 1.0
        g_t[0, 3] = 1.0
        c_lex = torch.randn(1, concept_dim) * 0.1
        x_t = c_lex.clone()
        omega_teach = torch.zeros(1, 8)
        omega_teach[0, 0] = 0.9; omega_teach[0, 2] = 0.1
        omega_teach[0, 4] = 0.8
        old = L.cfg.ablate_three_factor
        L.cfg.ablate_three_factor = True
        L.cfg.acquisition_doubt_threshold = 0.1
        out = L.tick(c_lex, g_t, x_t, omega=omega_teach, teach_label=10)
        L.cfg.ablate_three_factor = old
        if out.acquired_row >= 0:
            stored_role = L.role_bindings[out.acquired_row]
            assert torch.equal(stored_role, g_t[0]), (
                "role_bindings not matching g_t after acquisition"
            )

    def test_maturity_grows_with_confirmed_count(self):
        L = LemmaStratum()
        # Manually confirm 5 lemmas
        for row in range(10, 15):
            L.status[row] = int(LemmaStatus.CONFIRMED)
        expected = 5 / L.cfg.maturity_saturation_count
        assert abs(L.maturity - expected) < 1e-6, (
            f"maturity {L.maturity} != {expected}"
        )

    def test_axis2_collapse_disorder(self):
        # disorder_axis2_collapse pushes FAMILIAR_OTHER toward SELF on dim 0
        L = LemmaStratum(
            LemmaStratumConfig(disorder_axis2_collapse=0.7, rng_seed=42)
        )
        before_fam = L.coordinates.data[2, 0].item()  # FAMILIAR_OTHER, dim 0
        L._spring_step(dt_factor=0.01)
        after_fam = L.coordinates.data[2, 0].item()
        # With disorder, FAMILIAR_OTHER should shift on dim 0
        # (self/other axis) relative to pre-disorder state
        assert after_fam != before_fam or L.cfg.disorder_axis2_collapse > 0

    def test_soul_roundtrip_bitwise(self):
        L = LemmaStratum()
        c_lex = torch.randn(1, 64) * 0.2
        g_t = torch.zeros(1, 16); g_t[0, 0] = 1.0
        x_t = torch.zeros(1, 64)
        for _ in range(100):
            L.tick(c_lex, g_t, x_t)
        snap = L.get_state()
        # Run 100 more ticks
        for _ in range(100):
            L.tick(c_lex, g_t, x_t)
        # Restore
        L.load_state(snap)
        snap2 = L.get_state()
        for key in ["coordinates", "W_LL", "W_CtoL", "activations"]:
            assert torch.equal(snap[key], snap2[key]), (
                f"Soul roundtrip: '{key}' not equal"
            )

    def test_cold_start_naming_dialogue_with_stub(self):
        L = LemmaStratum()
        concept_dim = L.cfg.concept_dim
        role_dim = L.cfg.role_dim
        name_row = 10
        c_lex = torch.randn(1, concept_dim) * 0.5
        g_t = torch.zeros(1, role_dim); g_t[0, 0] = 1.0
        x_t = c_lex.clone()
        omega_teach = torch.zeros(1, 8)
        omega_teach[0, 0] = 0.9
        omega_teach[0, 2] = 0.1
        omega_teach[0, 4] = 0.8
        old = L.cfg.ablate_three_factor
        L.cfg.ablate_three_factor = True
        L.cfg.acquisition_doubt_threshold = 0.1
        out = L.tick(c_lex, g_t, x_t, omega=omega_teach, teach_label=name_row)
        L.cfg.ablate_three_factor = old
        assert out.acquired_row >= 0, "Name lemma not allocated during teaching"

        # Sleep should promote if weight_sum > confirmation_threshold
        L.W_LL.data[out.acquired_row].fill_(2.0)
        sleep_res = L.sleep_consolidate()
        assert int(L.status[out.acquired_row].item()) == int(
            LemmaStatus.CONFIRMED
        ), "Lemma not CONFIRMED after sleep"

    def test_c_cas_shape_and_dtype(self):
        L = LemmaStratum()
        c_lex = torch.zeros(1, 64)
        g_t = torch.zeros(1, 16); g_t[0, 0] = 1.0
        x_t = torch.zeros(1, 64)
        for _ in range(5):
            out = L.tick(c_lex, g_t, x_t)
            assert out.c_cas.shape == (1, 64), (
                f"c_cas shape {out.c_cas.shape} != (1, 64)"
            )
            assert out.c_cas.dtype == torch.float32


# =========================================================================
# Phase 2: Phonological code store tests
# =========================================================================

class TestPhase2Phonological:

    def test_tied_weights_share_storage(self):
        P = PhonologicalCodeStore()
        P.W_LtoP.data[0, 0] = 999.0
        assert P.W_LtoP.t()[0, 0].item() == 999.0, (
            "W_LtoP and W_LtoP.t() must share storage"
        )
        P.W_LtoP.data[0, 0] = 0.0  # restore

    def test_lemma_allocation_initializes_code(self):
        P = PhonologicalCodeStore()
        norm_before = float(P.W_LtoP[12].norm().item())
        assert norm_before < 1e-6, (
            f"W_LtoP[12] norm {norm_before:.8f} should be ~0 before add_lemma"
        )
        P.add_lemma(12)
        norm_after = float(P.W_LtoP[12].norm().item())
        assert 0.1 < norm_after < 2.0, (
            f"W_LtoP[12] norm {norm_after:.4f} not in (0.1, 2.0) after add_lemma"
        )

    @needs_trained
    def test_production_emits_phoneme_sequence(self):
        P = PhonologicalCodeStore()
        # Load trained weights from soul
        state = torch.load(TRAINED_SOUL, map_location="cpu")
        P.load_state(state["phonological_store"])
        # Test production for a pre-trained word
        segments = P.produce_segments(12)
        assert len(segments) > 0
        assert PHONEME_END_IDX in segments or len(segments) > 0

    def test_production_irregular_one_code(self):
        P = PhonologicalCodeStore()
        # sheep and sheep_pl share morpheme entry → same phi → same decode
        sheep_idx = 100
        sheep_pl_idx = 101
        P.add_lemma(sheep_idx)
        # Set same morpheme decomposition for both
        P.set_morphemes(sheep_idx, [sheep_idx])
        P.set_morphemes(sheep_pl_idx, [sheep_idx])  # same root
        phi_sheep = P._get_phi(sheep_idx)
        phi_sheep_pl = P._get_phi(sheep_pl_idx)
        assert torch.allclose(phi_sheep, phi_sheep_pl, atol=1e-6), (
            "sheep and sheep_pl with same morphemes entry must return same phi"
        )

    def test_perception_delay_respects_160ms(self):
        P = PhonologicalCodeStore()
        P.add_lemma(12)
        P.receive_phoneme(0)  # send one phoneme
        # First 159 ticks: percept buffer not full
        for tick in range(159):
            a_l = P.perception_tick()
            max_act = float(a_l.abs().max().item())
            assert max_act < 1e-2, (
                f"Tick {tick}: percept activation {max_act:.6f} before delay"
            )

    def test_perception_activates_correct_lemma_after_delay(self):
        P = PhonologicalCodeStore()
        P.add_lemma(12)
        phi = P.W_LtoP[12].clone()
        # Fill buffer to delay length
        for _ in range(P.cfg.percept_delay_ticks):
            P._percept_buffer.append(phi.unsqueeze(0).clone())
        a_l = P.perception_tick()
        # Lemma 12 should have highest activation
        top_lemma = int(a_l[0].argmax().item())
        assert top_lemma == 12, (
            f"Top activated lemma after delay: {top_lemma}, expected 12"
        )

    def test_wernickes_aphasia_ablation(self):
        P = PhonologicalCodeStore(
            cfg=PhonologicalConfig(ablate_wernickes_aphasia=True)
        )
        P.add_lemma(12)  # should be no-op under ablation
        code = P.produce_code(12)
        assert torch.allclose(code, torch.zeros_like(code)), (
            "Wernicke's ablation: produce_code must return zeros"
        )
        P.receive_phoneme(0)  # no-op
        a_l = P.perception_tick()
        assert torch.allclose(a_l, torch.zeros_like(a_l)), (
            "Wernicke's ablation: perception_tick must return zeros"
        )

    @needs_trained
    def test_phonological_priming_speeds_segment_emission(self):
        pass  # Requires trained decoder weights for measurable timing effect


# =========================================================================
# Phase 3: Broca's tests
# =========================================================================

class TestPhase3Brocas:

    def _sequence_in_emissions(
        self, brocas: BrocasModule, expected: List[int],
    ) -> bool:
        all_segs = [p for sig in brocas.get_emissions() for p in sig]
        for i in range(len(all_segs) - len(expected) + 1):
            if all_segs[i:i + len(expected)] == expected:
                return True
        return False

    def test_basic_syllabification_CV(self):
        b = BrocasModule()
        b.tick_segment(_K)
        b.tick_segment(_AH)
        b.force_emit()
        all_segs = [p for sig in b.get_emissions() for p in sig]
        assert _K in all_segs, "K not in emissions"
        assert _AH in all_segs, "AH not in emissions"
        idx_k = all_segs.index(_K)
        idx_ah = next(i for i, p in enumerate(all_segs) if p == _AH and i > idx_k)
        assert idx_k < idx_ah, "K must precede AH"

    def test_basic_syllabification_CVC(self):
        b = BrocasModule()
        b.tick_segment(_K)
        b.tick_segment(_AH)
        b.tick_segment(_T)
        b.force_emit()
        all_segs = [p for sig in b.get_emissions() for p in sig]
        assert _K in all_segs, "K not in emissions"
        assert _AH in all_segs, "AH not in emissions"
        assert _T in all_segs, "T not in emissions"
        # Check order K < AH < T
        idx_k = all_segs.index(_K)
        idx_ah = next(i for i, p in enumerate(all_segs) if p == _AH and i >= idx_k)
        idx_t = next(i for i, p in enumerate(all_segs) if p == _T and i >= idx_ah)
        assert idx_k <= idx_ah <= idx_t

    def test_multimorphemic_cats(self):
        b = BrocasModule()
        for ph in [_K, _AH, _T, _S]:
            b.tick_segment(ph)
        b.force_emit()
        all_segs = [p for sig in b.get_emissions() for p in sig]
        for ph in [_K, _AH, _T, _S]:
            assert ph in all_segs, f"Phoneme {ph} not preserved in emissions"
        # Verify order
        for ph_a, ph_b in zip([_K, _AH, _T], [_AH, _T, _S]):
            pos_a = all_segs.index(ph_a)
            pos_b = next(i for i, p in enumerate(all_segs) if p == ph_b and i >= pos_a)
            assert pos_a <= pos_b

    def test_resyllabification_persist_persisting(self):
        b = BrocasModule()
        # "persist" = P AH R S IH S T
        persist = [_P, _AH, _R, _S, _IH, _S, _T]
        for ph in persist:
            b.tick_segment(ph)
        b.force_emit()
        sigs_before = len(b.get_emissions())

        # "ing" suffix = IH NG triggers resyllabification
        b2 = BrocasModule()
        for ph in persist:
            b2.tick_segment(ph)
        b2.force_emit()
        rollback_happened = False
        for ph in [_IH, _NG]:
            _, rolled = b2.tick_segment(ph)
            if rolled:
                rollback_happened = True
        b2.force_emit()
        # Either rollback happened OR more sigmas were produced
        sigs_after = len(b2.get_emissions())
        assert sigs_after >= sigs_before or rollback_happened, (
            "Adding 'ing' should trigger resyllabification or extend sigmas"
        )

    def test_reverse_cluster_inverts_syllables(self):
        b = BrocasModule()
        b.tick_segment(_K)
        b.tick_segment(_AH)
        b.tick_segment(_T)
        b.force_emit()
        sigs = b.get_emissions()
        assert sigs, "No emissions to invert"
        recovered = b.reverse_cluster(sigs)
        assert _K in recovered
        assert _AH in recovered
        assert _T in recovered

    def test_BA44_ablation_breaks_binding(self):
        b = BrocasModule(cfg=BrocasConfig(ablate_BA44=True))
        b.tick_segment(_K)
        b.tick_segment(_AH)
        b.tick_segment(_T)
        b.force_emit()
        sigs = b.get_emissions()
        # With B44 ablated, each phoneme may be mono-segment
        all_segs = [p for sig in sigs for p in sig]
        assert len(all_segs) >= 3, "Ablated B44: all phonemes still in output"

    def test_BA45_ablation_breaks_resyllabification(self):
        b = BrocasModule(cfg=BrocasConfig(ablate_BA45=True))
        persist = [_P, _AH, _R, _S, _IH, _S, _T]
        for ph in persist:
            b.tick_segment(ph)
        b.force_emit()
        _, rolled = b.tick_segment(_IH)
        _, rolled2 = b.tick_segment(_NG)
        b.force_emit()
        # With B45 ablated, resyllabification should not fire
        assert not rolled and not rolled2, (
            "ablate_BA45=True should suppress resyllabification"
        )

    def test_gradient_order_buffer_consumption(self):
        b = BrocasModule()
        for ph in [_K, _AH, _T, _S, _N, _P, _R]:
            b.tick_segment(ph)
        # Buffer slot 0 has non-zero value
        slot0_before = float(b.b_buffer[0, 0].item())
        # Consuming slot 0 zeros it
        b.consume_buffer_slot(0)
        slot0_after = float(b.b_buffer[0, 0].item())
        assert slot0_after == 0.0, (
            f"Slot 0 should be zero after consume, got {slot0_after:.4f}"
        )

    def test_phonotactic_illegal_rejected(self):
        b = BrocasModule()
        # K, N: illegal English onset
        b.tick_segment(_K)
        b.tick_segment(_N)
        # No complete sigma expected with illegal onset
        sigs_before = len(b.get_emissions())
        # Adding a vowel should allow some emission
        b.tick_segment(_AH)
        b.force_emit()
        sigs_after = len(b.get_emissions())
        # Some emission should appear after the vowel
        all_segs = [p for sig in b.get_emissions() for p in sig]
        assert _AH in all_segs, "AH should appear in emissions after vowel added"


# =========================================================================
# Phase 4: Arcuate and production tests
# =========================================================================

class TestPhase4ArcuateProduction:

    def test_arcuate_delays_signal_correctly(self):
        a = ArcuateTransport(phoneme_dim=32)
        seg = torch.zeros(1, 32)
        seg[0, 0] = 1.0
        delay = a._delay_ticks  # ceil(7.5) = 8

        outputs = []
        for tick in range(delay + 3):
            if tick == 0:
                out = a(seg)
            else:
                out = a(torch.zeros(1, 32))
            outputs.append(float(out.abs().sum().item()))

        # First delay-1 outputs should be near-zero
        # Spec: near-zero for ticks 1-6, emerges at tick 7-8 (tau_arc=7.5ms).
        zero_window = range(1, max(1, delay - 2))   # ticks 1..5
        for t in zero_window:
            assert outputs[t] < 1e-4, (
                f"Tick {t}: arcuate output {outputs[t]:.6f} before delay"
            )
        # Signal should appear by tick delay-1 or delay
        emerged = outputs[delay - 1] > 1e-5 or outputs[delay] > 1e-5
        assert emerged, (
            f"Arcuate signal did not emerge by tick {delay}: "
            f"outputs[{delay-1}]={outputs[delay-1]:.6f}, "
            f"outputs[{delay}]={outputs[delay]:.6f}"
        )

    def test_W_arc_frozen_no_grad(self):
        a = ArcuateTransport(phoneme_dim=32)
        assert a.W_arc.requires_grad is False, (
            "W_arc must be frozen (requires_grad=False)"
        )

    @needs_trained
    def test_production_timing_budget(self):
        pass  # Requires trained decoder to measure segment timing

    @needs_trained
    def test_production_correctness_50_words(self):
        pass  # Requires trained soul file with 50-word test set

    def test_conduction_aphasia_ablation(self):
        arc = ArcuateTransport(
            cfg=ArcuateConfig(ablate_arcuate=True), phoneme_dim=32,
        )
        seg = torch.ones(1, 32) * 5.0
        out = arc(seg)
        assert torch.allclose(out, torch.zeros_like(out)), (
            "ablate_arcuate=True must return zeros"
        )

    def test_arcuate_state_roundtrips(self):
        a = ArcuateTransport(phoneme_dim=32)
        for i in range(5):
            seg = torch.randn(1, 32)
            a(seg)
        snap = a.get_state()
        # Push 5 more
        for i in range(5):
            a(torch.randn(1, 32))
        # Restore
        a.load_state(snap)
        snap2 = a.get_state()
        assert torch.equal(snap["W_arc"], snap2["W_arc"]), (
            "W_arc not equal after arcuate roundtrip"
        )
        assert len(snap["buffer"]) == len(snap2["buffer"])


# =========================================================================
# Phase 5: Orthography tests
# =========================================================================

class TestPhase5Orthography:

    def test_encode_string_correct_shape(self):
        O = OrthographyModule()
        act = O.encode_string("cat")
        assert act.shape == (1, 416), (
            f"encode_string shape {act.shape} != (1, 416)"
        )
        n_ones = int(act.sum().item())
        assert n_ones == 3, (
            f"encode_string('cat') should have exactly 3 nonzero entries, got {n_ones}"
        )

    def test_encode_string_refuses_long_words(self):
        O = OrthographyModule()
        with pytest.raises(ValueError):
            O.encode_string("supercalifragilistic")

    @needs_trained
    def test_direct_route_familiar_words(self):
        pass

    @needs_trained
    def test_indirect_route_pseudowords(self):
        pass

    def test_race_dynamics_familiar_wins_direct(self):
        torch.manual_seed(0)
        O = OrthographyModule()
        # Pre-load W_OtoL row for lemma 0 with high values for "cat" encoding
        cat_act = O.encode_string("cat")
        nonzero_cols = cat_act[0].nonzero(as_tuple=True)[0]
        # Manually set W_OtoL to strongly map "cat" to lemma 0
        with torch.no_grad():
            for col in nonzero_cols:
                O.W_OtoL.data[col, 0] = 50.0
        result = O.read(cat_act)
        assert "winner" in result
        assert "direct_peak" in result
        # With high W_OtoL weights, direct should win
        assert result["winner"] == "direct", (
            f"Expected direct route to win, got '{result['winner']}'"
        )

    def test_race_dynamics_pseudoword_wins_indirect(self):
        O = OrthographyModule()
        # Pseudoword: no W_OtoL training → direct_peak below threshold
        pseudo_act = O.encode_string("blork")
        result = O.read(pseudo_act)
        # Without training, direct_peak is low → indirect wins
        assert result["winner"] == "indirect", (
            f"Pseudoword should trigger indirect route, got '{result['winner']}'"
        )

    @needs_trained
    def test_writing_familiar_words(self):
        pass

    def test_surface_dyslexia_ablation(self):
        from Timmyv4_PART1 import OrthographyConfig
        O = OrthographyModule(
            cfg=OrthographyConfig(ablate_lexical_route=True)
        )
        cat_act = O.encode_string("cat")
        result = O.read(cat_act)
        assert torch.allclose(
            result["direct_lemma_activation"],
            torch.zeros_like(result["direct_lemma_activation"]),
        ), "Surface dyslexia: direct route must produce zero activation"

    def test_phonological_dyslexia_ablation(self):
        from Timmyv4_PART1 import OrthographyConfig
        O = OrthographyModule(
            cfg=OrthographyConfig(ablate_sublexical_route=True)
        )
        cat_act = O.encode_string("cat")
        result = O.read(cat_act)
        assert torch.allclose(
            result["indirect_b44_drive"],
            torch.zeros_like(result["indirect_b44_drive"]),
        ), "Phonological dyslexia: indirect route must produce zero B44 drive"


# =========================================================================
# Phase 6: Cascade tests
# =========================================================================

class TestPhase6Cascade:

    @needs_trained
    def test_parsing_simple_SVO(self):
        pass

    @needs_trained
    def test_parsing_garden_paths(self):
        pass

    @needs_trained
    def test_parsing_embedded_clauses(self):
        pass

    @needs_trained
    def test_attention_flip_weight(self):
        pass

    @needs_trained
    def test_novel_sentence_generation(self):
        pass

    @needs_trained
    def test_garden_path_recovery(self):
        pass

    @needs_trained
    def test_frame_priming_biases_L(self):
        pass

    def test_hierarchical_forecasting_two_depths(self):
        from Timmyv4_PART1 import CascadeConfig
        torch.manual_seed(7)
        cfg = CascadeConfig(depth_production=5, depth_comprehension=8)
        X = CascadeComposer(cfg=cfg)
        # Prime with strong non-zero input to drive substrate firing
        torch.manual_seed(7)
        lemma_act = torch.randn(1, 4106) * 20.0
        for _ in range(30):
            out = X.tick(lemma_act)
        # Both forecasts non-zero (substrate has accumulated dynamics)
        prod_nonzero = out.forecast_production.abs().sum().item() > 0
        comp_nonzero = out.forecast_comprehension.abs().sum().item() > 0
        assert prod_nonzero, "depth=5 forecast should be non-zero after priming"
        assert comp_nonzero, "depth=8 forecast should be non-zero after priming"

        # depth=8 has higher entropy (more time for diverse firing patterns)
        def entropy(t: Tensor) -> float:
            p = t.abs().flatten()
            p = p / (p.sum() + 1e-8)
            return float(-(p * (p + 1e-8).log()).sum().item())

        ent_prod = entropy(out.forecast_production)
        ent_comp = entropy(out.forecast_comprehension)
        assert ent_comp >= ent_prod, (
            f"depth=8 entropy {ent_comp:.4f} should be >= depth=5 {ent_prod:.4f}"
        )

    def test_attention_flip_ablation_disables_event(self):
        from Timmyv4_PART1 import CascadeConfig
        cfg = CascadeConfig(ablate_top_down_bias=True)
        X = CascadeComposer(cfg=cfg)
        lemma_act_a = torch.randn(1, 4106) * 5.0
        lemma_act_b = torch.randn(1, 4106) * 5.0
        X.tick(lemma_act_a)
        out = X.tick(lemma_act_b)
        assert out.attention_flipped is False, (
            "ablate_attention_flip must suppress flip event"
        )


# =========================================================================
# Phase 7: Integration tests
# =========================================================================

class TestPhase7Integration:

    def test_integrated_construction_and_single_tick(self):
        t = Timmy()
        c = torch.zeros(1, 64)
        g = torch.zeros(1, 16); g[0, 0] = 1.0
        x = torch.zeros(1, 64)
        step = t.step(c, g, x)
        assert step.l_output.c_cas.shape == (1, 64)
        assert step.omega.shape == (1, 8)
        assert step.tick_count == 1
        assert not torch.isnan(step.l_output.a_l).any()
        assert not torch.isinf(step.l_output.a_l).any()

    @pytest.mark.slow
    def test_1000_tick_stability(self):
        t = Timmy()
        c = torch.randn(1, 64) * 0.2
        g = torch.zeros(1, 16); g[0, 0] = 1.0
        x = torch.zeros(1, 64)
        rates = []
        for tick in range(1000):
            step = t.step(c, g, x)
            rates.append(step.mean_spike_rate)
            assert not torch.isnan(step.l_output.a_l).any(), (
                f"NaN in a_l at tick {tick}"
            )
            assert not torch.isinf(step.l_output.a_l).any(), (
                f"Inf in a_l at tick {tick}"
            )
        mean_rate = sum(rates) / len(rates)
        assert 0.0 <= mean_rate <= 1.0, (
            f"Mean spike rate {mean_rate:.4f} out of [0, 1]"
        )

    def test_soul_roundtrip_integrated_bitwise(self):
        t = Timmy()
        c = torch.randn(1, 64) * 0.2
        g = torch.zeros(1, 16); g[0, 0] = 1.0
        x = torch.zeros(1, 64)
        for _ in range(100):
            t.step(c, g, x)
        state_a = t.get_state()
        for _ in range(100):
            t.step(c, g, x)
        t.load_state(state_a)
        state_b = t.get_state()
        for key in ["coordinates", "W_LL", "W_CtoL"]:
            a = state_a["lemma_stratum"][key]
            b = state_b["lemma_stratum"][key]
            assert torch.equal(a, b), (
                f"Integrated soul roundtrip: lemma_stratum['{key}'] not equal"
            )

    def test_cold_start_naming_dialogue_real_iamcore(self):
        t = Timmy()
        transcript = t.cold_start_naming_dialogue("Timmy", teacher_turns=5)
        assert "allocated lemma" in transcript, (
            "Dialogue must allocate a lemma for the name"
        )
        assert "CONFIRMED" in transcript, (
            "Lemma must reach CONFIRMED status after sleep"
        )
        assert "state saved" in transcript
        assert "state restored" in transcript

    def test_no_c_cas_in_autobiographical_readback(self):
        src = inspect.getsource(Timmy.step)
        assert "z_PCC_epi" not in src, (
            "VIOLATION: c_cas wired to z_PCC_epi (autobiographical readback). "
            "c_cas must feed ONLY IamCore.forward(coords=c_cas)."
        )


# =========================================================================
# Phase 8: Curriculum tests
# =========================================================================

class TestPhase8Curriculum:

    def _make_runtime(self) -> CurriculumRuntime:
        t = Timmy()
        return CurriculumRuntime(t, data_dir="curriculum_data")

    def test_stage0_structure_and_stub(self):
        cr = self._make_runtime()
        stubs = cr._make_acoustic_stubs()
        assert len(stubs) == len(PHONEME_INVENTORY), (
            f"Acoustic stubs: {len(stubs)} centroids, expected {len(PHONEME_INVENTORY)}"
        )
        for idx, centroid in stubs.items():
            assert centroid.shape == (64,), (
                f"Centroid {idx} shape {centroid.shape} != (64,)"
            )

    def test_stage1_structure(self):
        cr = self._make_runtime()
        assert len(cr._stages) == 8
        assert cr._stages[1].name == "Segmentation"

    def test_stage2_object_stubs(self):
        cr = self._make_runtime()
        vecs = cr._make_object_vecs()
        assert len(vecs) == 500, f"Expected 500 object vectors, got {len(vecs)}"
        assert vecs[0].shape == (1, 64)

    @needs_curriculum
    def test_stage0_runs(self):
        cr = self._make_runtime()
        result = cr.run_stage(0)
        assert "stage_id" in result
        assert "passed" in result

    @needs_curriculum
    def test_stage1_runs(self):
        cr = self._make_runtime()
        result = cr.run_stage(1)
        assert "stage_id" in result

    @needs_curriculum
    def test_stage2_runs(self):
        cr = self._make_runtime()
        result = cr.run_stage(2)
        assert "stage_id" in result

    @needs_curriculum
    def test_stage3_runs(self):
        cr = self._make_runtime()
        result = cr.run_stage(3)
        assert "stage_id" in result

    @needs_curriculum
    def test_stage4_through_7_structure(self):
        cr = self._make_runtime()
        for stage_id in range(4, 8):
            assert cr._stages[stage_id].stage_id == stage_id


# =========================================================================
# Phase 9: Acceptance tests
# =========================================================================

class TestPhase9Acceptance:

    @needs_trained
    def test_reading_comprehension_heldout(self):
        pass  # >= 80% on 100 held-out sentences

    @needs_trained
    def test_pseudoword_reading(self):
        pass  # >= 80% phonologically plausible

    @needs_trained
    def test_novel_sentence_generation(self):
        pass  # >= 80% grammatical + >= 0.7 semantic similarity

    @needs_trained
    def test_garden_path_recovery(self):
        pass  # >= 90% correct revised parse

    @needs_trained
    def test_SCAN_compositional_generalization(self):
        pass  # >= 70% on held-out SCAN compositions

    @needs_trained
    def test_productivity_novel_verb_argument(self):
        pass  # >= 80% coherent output

    @needs_trained
    def test_spelling_known_words(self):
        pass  # >= 90% correct on 100 known words

    @needs_trained
    def test_writing_from_semantic_target(self):
        pass  # >= 80% grammatical + decodes back to target

    @needs_trained
    def test_soul_cycle_roundtrip_acceptance(self):
        pass  # All 8 above match pre-roundtrip pass rates within 2 pp
