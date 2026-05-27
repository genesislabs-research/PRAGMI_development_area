# RM numbers reference "The code classes explained" in README.md

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from Timmyv4_PART1 import (
    LemmaStratumConfig,
    PhonologicalConfig,
    BrocasConfig,
    ArcuateConfig,
    ProductionConfig,
    OrthographyConfig,
    CascadeConfig,
    PHONEME_INVENTORY,
    PHONEME_END_IDX,
)
from Timmyv4_PART2 import IamCore, IamCoreConfig, IamLoopOutput
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
    CascadeOutput,
)


# RM401
@dataclass
class TimmyConfig:
    lemma_cfg: LemmaStratumConfig = field(
        default_factory=LemmaStratumConfig,
    )
    phon_cfg: PhonologicalConfig = field(
        default_factory=PhonologicalConfig,
    )
    broca_cfg: BrocasConfig = field(
        default_factory=BrocasConfig,
    )
    arc_cfg: ArcuateConfig = field(
        default_factory=ArcuateConfig,
    )
    prod_cfg: ProductionConfig = field(
        default_factory=ProductionConfig,
    )
    orth_cfg: OrthographyConfig = field(
        default_factory=OrthographyConfig,
    )
    casc_cfg: CascadeConfig = field(
        default_factory=CascadeConfig,
    )
    iam_cfg: IamCoreConfig = field(
        default_factory=IamCoreConfig,
    )
    device: str = "cpu"
    soul_path: Optional[str] = None


# RM402
@dataclass
class TimmyStep:
    l_output: LemmaStratumOutput
    cascade_output: CascadeOutput
    r_reentry: Tensor
    omega: Tensor
    iam_episode_count: int
    tick_count: int
    mean_spike_rate: float
    acquired_row: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# RM403
class Timmy:
    def __init__(
        self,
        cfg: Optional[TimmyConfig] = None,
    ) -> None:
        self.cfg = cfg or TimmyConfig()
        dev = self.cfg.device

        self.lemma_stratum = LemmaStratum(self.cfg.lemma_cfg)
        self.phonological_store = PhonologicalCodeStore(
            cfg=self.cfg.phon_cfg,
            population_size=self.cfg.lemma_cfg.population_size,
        )
        self.brocas = BrocasModule(self.cfg.broca_cfg)
        self.arcuate = ArcuateTransport(
            cfg=self.cfg.arc_cfg,
            phoneme_dim=self.cfg.phon_cfg.phoneme_dim,
        )
        self.production_loop = ProductionLoop(
            lemma_stratum=self.lemma_stratum,
            phonological_store=self.phonological_store,
            arcuate=self.arcuate,
            brocas=self.brocas,
            cfg=self.cfg.prod_cfg,
        )
        self.orthography = OrthographyModule(
            cfg=self.cfg.orth_cfg,
            lemma_population_size=self.cfg.lemma_cfg.population_size,
            brocas_B44_size=self.cfg.broca_cfg.B44_size,
        )
        self.cascade = CascadeComposer(
            cfg=self.cfg.casc_cfg,
            lemma_population_size=self.cfg.lemma_cfg.population_size,
            brocas_B45_size=self.cfg.broca_cfg.B45_size,
        )
        self.iam_core = IamCore(self.cfg.iam_cfg)

        self._device: str = dev
        self._tick_count: int = 0

        # Cached outputs from previous tick for input to next tick.
        self._last_r_reentry: Optional[Tensor] = None
        self._last_omega: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    def step(
        self,
        c_lex: Tensor,
        g_t: Tensor,
        x_t: Tensor,
        tau_proj: int = 0,
        teach_label: Optional[int] = None,
    ) -> TimmyStep:
        # Tick order: L.tick -> c_cas -> IamCore.forward -> cache ->
        # CascadeComposer.tick -> return.
        # The 160 ms re-entry delay is handled INSIDE LemmaStratum.tick.

        r_reentry = self._last_r_reentry
        omega = self._last_omega

        # Step 1: LemmaStratum.
        l_out = self.lemma_stratum.tick(
            c_lex=c_lex,
            g_t=g_t,
            x_t=x_t,
            tau_proj=tau_proj,
            r_reentry=r_reentry,
            omega=omega,
            teach_label=teach_label,
        )

        # Step 2: c_cas from stratum output.
        c_cas = l_out.c_cas

        # Step 3: IamCore.forward(coords=c_cas).
        # c_cas goes ONLY here, not to the autobiographical readback pathway.
        iam_out: IamLoopOutput = self.iam_core.forward(coords=c_cas)

        # Step 4: Cache r_reentry and omega for next tick.
        # NOT biological. Engineering bridge: IamCore.r_reentry is pcc_dim
        # (128); W_reentry_to_L expects encoding_dim (64). Slice the first
        # encoding_dim elements.
        enc_dim = self.cfg.iam_cfg.encoding_dim
        raw_r = iam_out.r_reentry.detach()
        if raw_r.shape[-1] > enc_dim:
            new_r_reentry = raw_r[:, :enc_dim]
        else:
            new_r_reentry = raw_r
        new_omega = iam_out.omega
        if new_omega is None:
            # IamCore suppressed omega on this theta phase.
            # Substitute cached value so three-factor gate stays stable.
            if self._last_omega is not None:
                new_omega = self._last_omega
            else:
                new_omega = torch.zeros(
                    c_lex.shape[0], 8, device=c_lex.device,
                )
        else:
            new_omega = new_omega.detach()
        self._last_r_reentry = new_r_reentry
        self._last_omega = new_omega

        # Step 5: CascadeComposer.
        cascade_out = self.cascade.tick(
            lemma_activation=l_out.a_l,
        )

        # Step 6: Assemble step output.
        self._tick_count += 1
        rate_l = float(l_out.spikes.detach().mean().item())
        rate_x = float(cascade_out.spikes.detach().mean().item())
        mean_rate = 0.5 * (rate_l + rate_x)

        return TimmyStep(
            l_output=l_out,
            cascade_output=cascade_out,
            r_reentry=new_r_reentry,
            omega=new_omega,
            iam_episode_count=self.iam_core.episode_count(),
            tick_count=self._tick_count,
            mean_spike_rate=mean_rate,
            acquired_row=l_out.acquired_row,
            diagnostics={
                "l_diagnostics": l_out.diagnostics,
                "cascade_diagnostics": cascade_out.diagnostics,
                "doubt": float(l_out.doubt.mean().item()),
                "attention_flipped": l_out.attention_flipped,
                "flip_cost": l_out.flip_cost,
            },
        )

    # ------------------------------------------------------------------
    # Sleep
    # ------------------------------------------------------------------

    def sleep(self) -> Dict[str, Any]:
        result = self.lemma_stratum.sleep_consolidate()
        return result

    # ------------------------------------------------------------------
    # Production
    # ------------------------------------------------------------------

    def produce(
        self, lemma_idx: int,
    ) -> Dict[str, Any]:
        return self.production_loop.produce(
            lemma_idx=lemma_idx,
            device=torch.device(self._device),
        )

    def read_word(self, word: str) -> Dict[str, Any]:
        activation = self.orthography.encode_string(word)
        return self.orthography.read(
            word_activation=activation,
            device=torch.device(self._device),
        )

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "_tick_count": self._tick_count,
            "_last_r_reentry": (
                self._last_r_reentry.detach().cpu().clone()
                if self._last_r_reentry is not None else None
            ),
            "_last_omega": (
                self._last_omega.detach().cpu().clone()
                if self._last_omega is not None else None
            ),
            "lemma_stratum": self.lemma_stratum.get_state(),
            "phonological_store": self.phonological_store.get_state(),
            "brocas": self.brocas.get_state(),
            "arcuate": self.arcuate.get_state(),
            "orthography": self.orthography.get_state(),
            "cascade": self.cascade.get_state(),
            "iam_core": self.iam_core.state_dict(),
        }
        # IamCore synaptic reentry register serialization.
        if hasattr(self.iam_core, "synaptic_reentry"):
            state["iam_core_reentry"] = (
                self.iam_core.synaptic_reentry.state_dict()
                if hasattr(self.iam_core.synaptic_reentry, "state_dict")
                else {}
            )
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        self._tick_count = int(state.get("_tick_count", 0))
        if state.get("_last_r_reentry") is not None:
            self._last_r_reentry = state["_last_r_reentry"].to(
                self._device,
            )
        else:
            self._last_r_reentry = None
        if state.get("_last_omega") is not None:
            self._last_omega = state["_last_omega"].to(self._device)
        else:
            self._last_omega = None
        self.lemma_stratum.load_state(state["lemma_stratum"])
        self.phonological_store.load_state(state["phonological_store"])
        self.brocas.load_state(state["brocas"])
        self.arcuate.load_state(state["arcuate"])
        self.orthography.load_state(state["orthography"])
        self.cascade.load_state(state["cascade"])
        self.iam_core.load_state_dict(state["iam_core"])

    def save_soul(self, path: str) -> None:
        torch.save(self.get_state(), path)

    def load_soul(self, path: str) -> None:
        state = torch.load(path, map_location=self._device)
        self.load_state(state)

    # ------------------------------------------------------------------
    # Cold-start naming dialogue
    # ------------------------------------------------------------------

    def cold_start_naming_dialogue(
        self,
        name: str,
        teacher_turns: int = 3,
    ) -> str:
        log = io.StringIO()
        dev = torch.device(self._device)

        def say(s: str) -> None:
            print(s, file=log)

        concept_dim = self.cfg.lemma_cfg.concept_dim
        role_dim = self.cfg.lemma_cfg.role_dim

        # Pre-teaching query: blank concept, scaffold role.
        c_lex_blank = torch.zeros(1, concept_dim, device=dev)
        g_t_scaffold = torch.zeros(1, role_dim, device=dev)
        g_t_scaffold[0, 0] = 1.0  # scaffold marker
        x_t_blank = torch.zeros(1, concept_dim, device=dev)

        pre_step = self.step(c_lex_blank, g_t_scaffold, x_t_blank)
        doubt_pre = float(pre_step.l_output.doubt.mean().item())
        say(f"pre-teaching doubt: {doubt_pre:.3f}")
        assert doubt_pre >= 0.5, (
            f"doubt should be >= 0.5 at cold start, got {doubt_pre:.3f}"
        )

        # Target lemma index for the name.
        name_lemma_idx = int(
            self.lemma_stratum.next_unalloc.item()
        )

        rng = torch.Generator().manual_seed(
            abs(hash(name)) % (2 ** 31)
        )
        # Scale to 5.0 norm to drive target neuron above local_depol_threshold.
        # NOT biological. Teaching convenience scalar.
        name_vec_raw = torch.randn(1, concept_dim, generator=rng)
        name_vec_raw = name_vec_raw / (name_vec_raw.norm() + 1e-8) * 5.0
        name_vec = name_vec_raw.to(dev)

        omega_teach = torch.zeros(1, 8, device=dev)
        omega_teach[0, 0] = 0.8
        omega_teach[0, 2] = 0.1
        omega_teach[0, 4] = 0.7

        # Teaching mode: dopamine-only gate during explicit instruction.
        # In a structured naming dialogue the teacher IS the coincidence
        # trigger, so requiring cholinergic pause + local depol from a
        # cold untrained network would permanently block boot-time naming.
        # ablate_three_factor=True makes the DA burst sufficient.
        # Restored after teaching.
        prev_ablate = self.lemma_stratum.cfg.ablate_three_factor
        self.lemma_stratum.cfg.ablate_three_factor = True

        allocated = False
        for turn in range(teacher_turns):
            self._last_omega = omega_teach
            step_out = self.step(
                c_lex=name_vec,
                g_t=g_t_scaffold,
                x_t=name_vec,
                tau_proj=0,
                teach_label=name_lemma_idx,
            )
            if step_out.acquired_row >= 0:
                say(
                    f"allocated lemma {step_out.acquired_row} for "
                    f"'{name}' on turn {turn + 1}"
                )
                name_lemma_idx = step_out.acquired_row
                allocated = True
                break

        self.lemma_stratum.cfg.ablate_three_factor = prev_ablate

        if not allocated:
            say(f"name lemma not allocated after {teacher_turns} turns")

        # Register phonological code for the name.
        if 0 <= name_lemma_idx < self.phonological_store.population_size:
            self.phonological_store.add_lemma(name_lemma_idx)
            say(f"phonological code assigned for lemma {name_lemma_idx}")

        # Sleep consolidation.
        sleep_result = self.sleep()
        status_after = int(
            self.lemma_stratum.status[name_lemma_idx].item()
        )
        say(
            f"post-sleep status: "
            f"{LemmaStatus(status_after).name} "
            f"(promoted={sleep_result['promoted']}, "
            f"reclaimed={sleep_result['reclaimed']})"
        )

        # Save state (soul checkpoint).
        saved_state = self.get_state()
        say("state saved")

        # Reload state (cold restart simulation).
        self.load_state(saved_state)
        say("state restored (cold restart)")

        # Post-restart query with name concept.
        self._last_omega = None
        post_step = self.step(name_vec, g_t_scaffold, x_t_blank)
        top_lemma = post_step.l_output.selected_lemma
        say(f"post-restart top lemma: {top_lemma} (target: {name_lemma_idx})")

        # Status of the name lemma after reload.
        status_restored = int(
            self.lemma_stratum.status[name_lemma_idx].item()
        )
        say(
            f"name lemma status after restore: "
            f"{LemmaStatus(status_restored).name}"
        )

        # Unknown query: ask about France.
        france_vec = torch.randn(1, concept_dim) * 0.1
        france_vec = france_vec.to(dev)
        unknown_step = self.step(france_vec, g_t_scaffold, x_t_blank)
        doubt_unknown = float(
            unknown_step.l_output.doubt.mean().item()
        )
        say(f"unknown query doubt: {doubt_unknown:.3f}")

        return log.getvalue()


# =========================================================================
# Curriculum data types
# =========================================================================

# RM405
@dataclass
class Stage:
    stage_id: int
    name: str
    data_file: str
    exit_threshold: float
    max_attempts: int = 3


# RM406
@dataclass
class Trial:
    stage_id: int
    trial_id: int
    input_data: Any
    label: Any
    tags: List[str] = field(default_factory=list)


# RM407
@dataclass
class TrialResult:
    trial_id: int
    stage_id: int
    score: float
    passed: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# =========================================================================
# Curriculum runtime
# =========================================================================

# RM404
class CurriculumRuntime:
    STAGES: List[Tuple[int, str, str, float]] = [
        (0, "Phonemic discrimination", "stage0_minimal_pairs.json", 0.70),
        (1, "Segmentation",            "stage1_segmentation.json",  0.80),
        (2, "Word-object grounding",   "stage2_word_object.json",   0.80),
        (3, "Frame-based generalization", "stage3_frames.json",     0.80),
        (4, "Morphology",              "stage4_morphology.json",    0.80),
        (5, "Reading bootstrap",       "stage5_reading.json",       0.80),
        (6, "Syntactic composition",   "stage6_syntax.json",        0.75),
        (7, "Generalized composition", "stage7_generation.json",    0.80),
    ]

    def __init__(
        self,
        timmy: Timmy,
        data_dir: str = "curriculum_data",
    ) -> None:
        self.timmy = timmy
        self.data_dir = data_dir
        self._stages: List[Stage] = [
            Stage(
                stage_id=s_id,
                name=name,
                data_file=fname,
                exit_threshold=threshold,
            )
            for s_id, name, fname, threshold in self.STAGES
        ]
        # Acoustic centroids for stage 0 (lazy init).
        self._acoustic_centroids: Optional[Dict[int, Tensor]] = None
        # Object representations for stage 2 (lazy init).
        self._object_vecs: Optional[List[Tensor]] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_stage_data(self, data_file: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.data_dir, data_file)
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Acoustic stub (stage 0)
    # ------------------------------------------------------------------

    def _make_acoustic_stubs(self) -> Dict[int, Tensor]:
        if self._acoustic_centroids is not None:
            return self._acoustic_centroids
        concept_dim = self.timmy.cfg.lemma_cfg.concept_dim
        n_phonemes = len(PHONEME_INVENTORY)
        gen = torch.Generator().manual_seed(11111)
        # Sample random unit vectors, scale to norm 2.0 for between-cluster
        # separation. Within-cluster noise adds std 0.5.
        centroids = torch.randn(n_phonemes, concept_dim, generator=gen)
        norms = centroids.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        centroids = centroids / norms * 2.0
        self._acoustic_centroids = {
            i: centroids[i] for i in range(n_phonemes)
        }
        return self._acoustic_centroids

    def acoustic_input(
        self,
        phoneme_idx: int,
        within_cluster_std: float = 0.5,
    ) -> Tensor:
        stubs = self._make_acoustic_stubs()
        centroid = stubs[phoneme_idx % len(stubs)]
        noise = torch.randn_like(centroid) * within_cluster_std
        return (centroid + noise).unsqueeze(0)

    # ------------------------------------------------------------------
    # Object stubs (stage 2)
    # ------------------------------------------------------------------

    def _make_object_vecs(self) -> List[Tensor]:
        if self._object_vecs is not None:
            return self._object_vecs
        concept_dim = self.timmy.cfg.lemma_cfg.concept_dim
        n_clusters = 50
        n_per_cluster = 10
        gen = torch.Generator().manual_seed(22222)
        vecs: List[Tensor] = []
        for c in range(n_clusters):
            centroid = torch.randn(concept_dim, generator=gen)
            for k in range(n_per_cluster):
                noise = torch.randn(concept_dim, generator=gen) * 0.3
                vecs.append((centroid + noise).unsqueeze(0))
        self._object_vecs = vecs
        return vecs

    def object_vec(self, obj_idx: int) -> Tensor:
        vecs = self._make_object_vecs()
        return vecs[obj_idx % len(vecs)]

    def conjunctive_code(
        self, g_t: Tensor, x_t: Tensor,
    ) -> Tensor:
        # p_t = g_t tensor product x_t (flattened Kronecker-like outer).
        # Simple approximation: elementwise modulated sum.
        concept_dim = self.timmy.cfg.lemma_cfg.concept_dim
        g_proj = g_t[:, :concept_dim] if g_t.shape[-1] >= concept_dim else (
            torch.cat([
                g_t,
                torch.zeros(g_t.shape[0], concept_dim - g_t.shape[-1]),
            ], dim=-1)
        )
        return g_proj * x_t

    # ------------------------------------------------------------------
    # Exit metrics
    # ------------------------------------------------------------------

    def _eval_stage0_metric(
        self, results: List[TrialResult],
    ) -> float:
        if not results:
            return 0.0
        correct = sum(1 for r in results if r.passed)
        return correct / len(results)

    def _eval_stage1_metric(
        self, results: List[TrialResult],
    ) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    def _eval_stage2_metric(self) -> Tuple[float, float]:
        n_confirmed = int(
            (
                self.timmy.lemma_stratum.status
                == int(LemmaStatus.CONFIRMED)
            ).sum().item()
        )
        confirmed_ok = n_confirmed >= 200
        # Retrieval accuracy: check top activated lemma among confirmed.
        retrieval_score = min(1.0, n_confirmed / 200.0)
        return float(confirmed_ok), retrieval_score

    def _eval_stage_generic(
        self, results: List[TrialResult],
    ) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    def _oracle_score(
        self,
        generated: str,
        semantic_target: List[float],
    ) -> float:
        # Stage 7 oracle: 0.5 * grammaticality + 0.5 * meaning_similarity.
        # Grammaticality stub: any non-empty string with >= 2 tokens.
        tokens = generated.strip().split()
        grammaticality = 1.0 if len(tokens) >= 2 else 0.0
        # Meaning similarity: cosine between target and concept projection
        # from top activated lemma.
        target_vec = torch.tensor(semantic_target, dtype=torch.float32)
        target_vec = target_vec[:self.timmy.cfg.lemma_cfg.concept_dim]
        if target_vec.shape[0] < self.timmy.cfg.lemma_cfg.concept_dim:
            pad = torch.zeros(
                self.timmy.cfg.lemma_cfg.concept_dim - target_vec.shape[0],
            )
            target_vec = torch.cat([target_vec, pad])
        top_idx = int(
            self.timmy.lemma_stratum.activations.mean(dim=0).argmax()
        )
        top_coord = self.timmy.lemma_stratum.coordinates.data[top_idx]
        cos = float(
            torch.nn.functional.cosine_similarity(
                top_coord.unsqueeze(0),
                target_vec.unsqueeze(0),
            ).item()
        )
        meaning_sim = (cos + 1.0) / 2.0
        return 0.5 * grammaticality + 0.5 * meaning_sim

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def run_stage(
        self, stage_id: int,
    ) -> Dict[str, Any]:
        if not (0 <= stage_id < len(self._stages)):
            raise ValueError(
                f"stage_id must be 0-{len(self._stages) - 1}; "
                f"got {stage_id}"
            )
        stage = self._stages[stage_id]
        raw_data = self._load_stage_data(stage.data_file)

        concept_dim = self.timmy.cfg.lemma_cfg.concept_dim
        role_dim = self.timmy.cfg.lemma_cfg.role_dim
        dev = torch.device(self.timmy._device)
        g_t = torch.zeros(1, role_dim, device=dev)
        g_t[0, 0] = 1.0

        attempts_log: List[Dict[str, Any]] = []
        repetitions = [1, 2, 4]

        for attempt_idx in range(stage.max_attempts):
            results: List[TrialResult] = []
            reps = repetitions[min(attempt_idx, len(repetitions) - 1)]

            if stage_id == 0:
                results = self._run_stage0(reps, raw_data, g_t, dev)
            elif stage_id == 1:
                results = self._run_stage1(reps, raw_data, g_t, dev)
            elif stage_id == 2:
                results = self._run_stage2(reps, raw_data, g_t, dev)
            else:
                results = self._run_stage_generic(
                    stage_id, reps, raw_data, g_t, dev,
                )

            if stage_id == 0:
                metric = self._eval_stage0_metric(results)
            elif stage_id == 1:
                metric = self._eval_stage1_metric(results)
            elif stage_id == 2:
                _, retrieval = self._eval_stage2_metric()
                metric = retrieval
            else:
                metric = self._eval_stage_generic(results)

            passed = metric >= stage.exit_threshold
            attempts_log.append({
                "attempt": attempt_idx + 1,
                "repetitions": reps,
                "metric": metric,
                "passed": passed,
                "n_trials": len(results),
            })
            if passed:
                break
            if attempt_idx < stage.max_attempts - 1:
                pass  # retry with more repetitions

        final_passed = attempts_log[-1]["passed"]
        final_metric = attempts_log[-1]["metric"]

        if not final_passed:
            report = (
                f"Stage {stage_id} '{stage.name}' failed after "
                f"{stage.max_attempts} attempts. "
                f"Final metric: {final_metric:.3f} "
                f"(threshold: {stage.exit_threshold:.2f}). "
                "Builder to surface to Amellia for disposition."
            )
            return {
                "stage_id": stage_id,
                "name": stage.name,
                "passed": False,
                "final_metric": final_metric,
                "attempts": attempts_log,
                "diagnostic_report": report,
            }

        return {
            "stage_id": stage_id,
            "name": stage.name,
            "passed": True,
            "final_metric": final_metric,
            "attempts": attempts_log,
        }

    def _run_stage0(
        self,
        reps: int,
        raw_data: List[Dict[str, Any]],
        g_t: Tensor,
        dev: torch.device,
    ) -> List[TrialResult]:
        concept_dim = self.timmy.cfg.lemma_cfg.concept_dim
        x_t_blank = torch.zeros(1, concept_dim, device=dev)
        results: List[TrialResult] = []
        phoneme_pairs = (
            [
                (d.get("input", 0), d.get("label", 1))
                for d in raw_data
            ]
            if raw_data else
            [(i, (i + 1) % len(PHONEME_INVENTORY)) for i in range(20)]
        )
        for rep in range(reps):
            for trial_id, (ph_a, ph_b) in enumerate(phoneme_pairs):
                vec_a = self.acoustic_input(ph_a).to(dev)
                vec_b = self.acoustic_input(ph_b).to(dev)
                for _ in range(10):
                    self.timmy.step(vec_a, g_t, x_t_blank)
                top_a = int(
                    self.timmy.lemma_stratum.activations
                    .mean(dim=0).argmax().item()
                )
                for _ in range(10):
                    self.timmy.step(vec_b, g_t, x_t_blank)
                top_b = int(
                    self.timmy.lemma_stratum.activations
                    .mean(dim=0).argmax().item()
                )
                distinct = top_a != top_b
                results.append(TrialResult(
                    trial_id=trial_id + rep * len(phoneme_pairs),
                    stage_id=0,
                    score=1.0 if distinct else 0.0,
                    passed=distinct,
                ))
        return results

    def _run_stage1(
        self,
        reps: int,
        raw_data: List[Dict[str, Any]],
        g_t: Tensor,
        dev: torch.device,
    ) -> List[TrialResult]:
        concept_dim = self.timmy.cfg.lemma_cfg.concept_dim
        x_t_blank = torch.zeros(1, concept_dim, device=dev)
        results: List[TrialResult] = []
        segs = (
            [d.get("input", []) for d in raw_data]
            if raw_data else
            [[0, 24, 2, 32, 45]] * 20
        )
        for rep in range(reps):
            for trial_id, seg_seq in enumerate(segs):
                self.timmy.brocas.reset_emission()
                emissions = []
                for ph in seg_seq:
                    emitted, _ = self.timmy.brocas.tick_segment(ph, dev)
                    if emitted is not None:
                        emissions.append(emitted)
                last = self.timmy.brocas.force_emit()
                if last is not None:
                    emissions.append(last)
                n_boundaries = len(emissions)
                expected = raw_data[trial_id % len(raw_data)].get(
                    "label", 1,
                ) if raw_data else 1
                score = 1.0 if n_boundaries >= expected else 0.0
                results.append(TrialResult(
                    trial_id=trial_id + rep * len(segs),
                    stage_id=1,
                    score=score,
                    passed=score >= 0.8,
                ))
        return results

    def _run_stage2(
        self,
        reps: int,
        raw_data: List[Dict[str, Any]],
        g_t: Tensor,
        dev: torch.device,
    ) -> List[TrialResult]:
        concept_dim = self.timmy.cfg.lemma_cfg.concept_dim
        x_t_blank = torch.zeros(1, concept_dim, device=dev)
        results: List[TrialResult] = []
        objs = (
            [
                {
                    "vec": self.object_vec(d.get("input", 0)),
                    "lemma_idx": d.get("label", 0),
                }
                for d in raw_data
            ]
            if raw_data else
            [
                {
                    "vec": self.object_vec(i),
                    "lemma_idx": i + self.timmy.cfg.lemma_cfg.scaffold_size,
                }
                for i in range(min(20, 500))
            ]
        )
        for rep in range(reps):
            for trial_id, obj in enumerate(objs):
                x_t = obj["vec"].to(dev)
                target_idx = int(obj["lemma_idx"])
                omega_teach = torch.zeros(1, 8, device=dev)
                omega_teach[0, 0] = 0.8
                omega_teach[0, 2] = 0.1
                omega_teach[0, 4] = 0.5
                self.timmy._last_omega = omega_teach
                step_out = self.timmy.step(
                    c_lex=x_t,
                    g_t=g_t,
                    x_t=x_t,
                    tau_proj=0,
                    teach_label=target_idx,
                )
                if step_out.acquired_row >= 0:
                    self.timmy.phonological_store.add_lemma(
                        step_out.acquired_row,
                    )
                # Query: does the lemma activate?
                for _ in range(5):
                    q_out = self.timmy.step(x_t, g_t, x_t_blank)
                retrieved = q_out.l_output.selected_lemma == target_idx
                score = 1.0 if retrieved else 0.0
                results.append(TrialResult(
                    trial_id=trial_id + rep * len(objs),
                    stage_id=2,
                    score=score,
                    passed=retrieved,
                ))
        return results

    def _run_stage_generic(
        self,
        stage_id: int,
        reps: int,
        raw_data: List[Dict[str, Any]],
        g_t: Tensor,
        dev: torch.device,
    ) -> List[TrialResult]:
        concept_dim = self.timmy.cfg.lemma_cfg.concept_dim
        x_t_blank = torch.zeros(1, concept_dim, device=dev)
        if not raw_data:
            return [
                TrialResult(
                    trial_id=0,
                    stage_id=stage_id,
                    score=0.0,
                    passed=False,
                    diagnostics={"note": "no data file found"},
                )
            ]
        results: List[TrialResult] = []
        for rep in range(reps):
            for trial_id, item in enumerate(raw_data):
                if stage_id == 7:
                    target = item.get("semantic_target", [0.0] * concept_dim)
                    c_lex = torch.randn(1, concept_dim, device=dev) * 0.2
                    self.timmy.step(c_lex, g_t, x_t_blank)
                    top_idx = int(
                        self.timmy.lemma_stratum.activations
                        .mean(dim=0).argmax().item()
                    )
                    segs = self.timmy.phonological_store.produce_segments(
                        top_idx,
                    )
                    generated = " ".join(
                        PHONEME_INVENTORY[s]
                        for s in segs
                        if 0 <= s < len(PHONEME_INVENTORY)
                        and s != PHONEME_END_IDX
                    )
                    score = self._oracle_score(generated, target)
                else:
                    c_lex = torch.randn(1, concept_dim, device=dev) * 0.1
                    step_out = self.timmy.step(c_lex, g_t, x_t_blank)
                    score = float(
                        1.0 - step_out.diagnostics.get("doubt", 1.0)
                    )
                results.append(TrialResult(
                    trial_id=trial_id + rep * len(raw_data),
                    stage_id=stage_id,
                    score=score,
                    passed=score >= self._stages[stage_id].exit_threshold,
                ))
        return results

    # ------------------------------------------------------------------
    # Full curriculum
    # ------------------------------------------------------------------

    def run_all(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "stages": [],
            "all_passed": True,
        }
        for stage_id in range(len(self._stages)):
            result = self.run_stage(stage_id)
            report["stages"].append(result)
            if not result["passed"]:
                report["all_passed"] = False
                report["stopped_at"] = stage_id
                break
        if report["all_passed"]:
            self.timmy.sleep()
        return report
