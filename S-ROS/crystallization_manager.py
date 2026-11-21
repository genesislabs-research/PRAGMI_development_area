"""
crystallization_manager.py
Neo: Generalization-Gated Crystallization Monitor for S-ROS

This module does not care about memorization. It cares about understanding.
A pathway crystallizes when the network produces correct output for held-out
instances of a rule class it has never been trained on. That is the test for
genuine learning. Everything else is filing.

BIOLOGICAL GROUNDING
Critical period closure: Turrigiano (2008) synaptic scaling, Abraham (2008)
metaplasticity, Bhatt et al. (2009) astrocytic calcium gating of structural
plasticity windows. Crystallization is the computational analog of critical
period closure: once a rule is learned (generalization confirmed), the pathway
locks and plasticity budget is freed for unconverged rules.

The crystallization criterion is generalization, not convergence on trained
examples. This is the difference between understanding and memorization.

WHAT NEO GUARANTEES (each item is asserted by a self-test at the bottom)

  1. Held-out data is held out. Registration refuses a rule whose test inputs
     also appear in its training inputs.
  2. Every rule is evaluated on its own clock. Evaluation cadence counts that
     rule's own training steps, so no rule is starved by round-robin order.
  3. Four gates, all over the same K consecutive evaluation windows:
       a. mean training loss in the window is below loss_threshold
       b. held-out accuracy is at or above generalization_target
       c. the rule's learning signal has stopped: its variance over the
          recent window is below variance_threshold AND its mean has decayed
          to stability_decay_ratio of the largest window mean ever seen for
          the rule. Variance alone is not enough: a steady, large gradient
          has zero variance and is not "stable".
       d. the answer is specific to the rule: shown held-out inputs from
          OTHER rules, the model must not keep giving this rule's answers.
          Without this a constant predictor passes any single-answer rule.
  4. Crystallizing does something. On crystallization Neo calls commit_fn,
     which is where the caller writes the skill into frozen storage
     (Theo's S-CAM, a weight snapshot, anything gradient cannot reach).
  5. Crystallized is a claim that keeps being checked. A crystallized rule is
     never reported as 100% by fiat. audit_retention() re-runs every
     crystallized rule's train and held-out pairs through the real inference
     path and records a regression if the rule has been forgotten.
  6. State survives a restart. to_dict()/from_dict() round-trip every history
     and every decision.

Neo observes only what the caller shows it through encode_fn / decode_fn /
compare_fn. It never reads the loss surface or the weights to decide
correctness.

Genesis Labs Research, 2026
Amellia Mendel, Lisa Adler
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class RuleClass:
    """One rule class with its training and held-out test instances."""
    rule_id: str
    description: str
    train_pairs: List[Dict]
    test_pairs: List[Dict]
    # per-step records
    train_loss_history: List[float] = field(default_factory=list)
    gradient_rms_history: List[float] = field(default_factory=list)
    parameter_displacement_history: List[float] = field(default_factory=list)
    # per-evaluation-window records
    train_window_mean_history: List[float] = field(default_factory=list)
    generalization_accuracy_history: List[float] = field(default_factory=list)
    test_loss_history: List[float] = field(default_factory=list)
    stability_history: List[bool] = field(default_factory=list)
    false_positive_rate_history: List[float] = field(default_factory=list)
    consecutive_generalization_passes: int = 0
    # bookkeeping
    steps_seen: int = 0
    losses_since_eval: List[float] = field(default_factory=list)
    peak_signal_window_mean: float = 0.0
    # crystallization
    is_crystallized: bool = False
    crystallized_at_step: Optional[int] = None
    crystallized_weights: Optional[Dict[str, torch.Tensor]] = None
    # retention (only meaningful after crystallization)
    retention_history: List[float] = field(default_factory=list)
    retention_failed: bool = False

    @property
    def weight_delta_history(self) -> List[float]:
        """Legacy name. Trainers passed gradient norms under this name."""
        return self.gradient_rms_history


_HISTORY_FIELDS = (
    "train_loss_history", "gradient_rms_history", "parameter_displacement_history",
    "train_window_mean_history", "generalization_accuracy_history", "test_loss_history",
    "stability_history", "false_positive_rate_history", "consecutive_generalization_passes", "steps_seen",
    "losses_since_eval", "peak_signal_window_mean", "is_crystallized",
    "crystallized_at_step", "retention_history", "retention_failed",
)


def _recent_variance(history: List[float], window: int) -> Optional[float]:
    if len(history) < max(window, 2):
        return None
    return torch.var(torch.tensor(history[-window:], dtype=torch.float64)).item()


def _pair_output(pair: Dict) -> Any:
    for key in ("output", "out", "action", "y", "label"):
        if key in pair:
            return pair[key]
    return None


def _pair_input(pair: Dict) -> Any:
    """The input side of a pair, whichever key convention the dataset uses."""
    for key in ("input", "in", "sensor", "x"):
        if key in pair:
            return pair[key]
    return None


class CrystallizationManager:
    """
    Neo: watches the plastic side and decides when a rule has been understood.

    Not a doctor. A thermometer. But the thermometer reads generalization,
    not memorization. See the module docstring for the exact guarantees.
    """

    def __init__(
        self,
        loss_threshold: float = 0.01,
        generalization_target: float = 0.90,
        variance_threshold: float = 1e-5,
        consecutive_windows_required: int = 5,
        window_size: int = 50,
        variance_check_quantity: str = "gradient",
        stability_decay_ratio: float = 0.25,
        max_false_positive_rate: float = 0.10,
        max_foils: int = 32,
        retention_target: float = 1.0,
        on_regression: str = "flag",
        commit_fn: Optional[Callable[[str, RuleClass], None]] = None,
        trajectory_dense_logging_steps: int = 0,
        trajectory_log_dir: Optional[str] = None,
    ):
        """
        Args:
            loss_threshold: gate (a). Mean training loss per evaluation window.
            generalization_target: gate (b). Held-out accuracy, 0..1.
            variance_threshold: gate (c). Variance of the learning signal.
            consecutive_windows_required: K. Gates (a) and (b) must hold for
                K evaluation windows in a row.
            window_size: number of recent per-step records gate (c) looks at.
            variance_check_quantity: which learning signal gate (c) reads:
                "gradient" (gradient RMS for this rule's steps),
                "displacement" (parameter displacement per step), or "both".
            stability_decay_ratio: gate (c) also requires the recent mean of
                the signal to be at most this fraction of the largest window
                mean the rule has ever shown. Set to None to disable.
            max_false_positive_rate: gate (d). Largest fraction of other
                rules' held-out inputs that may be answered with this rule's
                answers. Set to None to disable.
            max_foils: how many other-rule inputs gate (d) samples per check.
            retention_target: accuracy a crystallized rule must keep.
            on_regression: "flag" records the failure and keeps the rule
                crystallized; "decrystallize" returns it to the plastic pool.
            commit_fn: called as commit_fn(rule_id, rule_class) at the moment
                of crystallization. This is where the skill is written into
                frozen storage. If it raises, the rule does not crystallize.
        """
        if variance_check_quantity not in ("gradient", "displacement", "both"):
            raise ValueError("variance_check_quantity must be 'gradient', 'displacement' or 'both'")
        if on_regression not in ("flag", "decrystallize"):
            raise ValueError("on_regression must be 'flag' or 'decrystallize'")
        self.loss_threshold = loss_threshold
        self.generalization_target = generalization_target
        self.variance_threshold = variance_threshold
        self.consecutive_windows_required = consecutive_windows_required
        self.window_size = window_size
        self.variance_check_quantity = variance_check_quantity
        self.stability_decay_ratio = stability_decay_ratio
        self.max_false_positive_rate = max_false_positive_rate
        self.max_foils = max_foils
        self.retention_target = retention_target
        self.on_regression = on_regression
        self.commit_fn = commit_fn
        self.trajectory_dense_logging_steps = trajectory_dense_logging_steps
        self.trajectory_log_dir = trajectory_log_dir

        self.rule_classes: Dict[str, RuleClass] = {}
        self.crystallization_log: List[Dict] = []
        self.regression_log: List[Dict] = []
        self.global_step: int = 0
        self._trajectory_file = None
        self._trajectory_records = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_rule_class(
        self,
        rule_id: str,
        description: str,
        train_pairs: List[Dict],
        test_pairs: List[Dict],
    ) -> None:
        """Register a rule class with its training and held-out test data."""
        if len(test_pairs) == 0:
            raise ValueError(
                f"Rule class '{rule_id}' has no held-out test pairs. "
                "Cannot measure generalization without held-out data."
            )
        train_inputs = [_pair_input(p) for p in train_pairs]
        leaked = [
            _pair_input(p) for p in test_pairs
            if _pair_input(p) is not None and _pair_input(p) in train_inputs
        ]
        if leaked:
            raise ValueError(
                f"Rule class '{rule_id}': {len(leaked)} held-out input(s) also appear in "
                f"the training set (first: {leaked[0]!r}). Held-out data must be unseen."
            )
        self.rule_classes[rule_id] = RuleClass(
            rule_id=rule_id,
            description=description,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_train_loss(self, rule_id: str, loss: float) -> None:
        """Record training loss for a rule class after a training step."""
        rc = self.rule_classes[rule_id]
        if rc.is_crystallized:
            return
        rc.train_loss_history.append(float(loss))
        rc.losses_since_eval.append(float(loss))

    def record_gradient_rms_norms(self, rule_id: str, grad_rms: float) -> None:
        """Record the gradient RMS produced by this rule's training step."""
        rc = self.rule_classes[rule_id]
        if rc.is_crystallized:
            return
        rc.gradient_rms_history.append(float(grad_rms))

    def record_parameter_displacement(self, rule_id: str, displacement: float) -> None:
        """Record how far the parameters moved on this rule's training step."""
        rc = self.rule_classes[rule_id]
        if rc.is_crystallized:
            return
        rc.parameter_displacement_history.append(float(displacement))

    def record_weight_deltas(self, rule_id: str, delta_norm: float) -> None:
        """Legacy name: the trainers pass a gradient norm here."""
        self.record_gradient_rms_norms(rule_id, delta_norm)

    @staticmethod
    def _compute_gradient_rms_norm(model: torch.nn.Module) -> float:
        """RMS of all gradient elements currently held on the model."""
        total, count = 0.0, 0
        for p in model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().item())
                count += p.grad.numel()
        return (total / max(count, 1)) ** 0.5

    @staticmethod
    def snapshot_parameters(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in model.named_parameters()}

    @staticmethod
    def _compute_parameter_displacement_norm(
        model: torch.nn.Module, pre_snapshot: Dict[str, torch.Tensor]
    ) -> float:
        """L2 distance between the model's parameters now and a snapshot."""
        total = 0.0
        for n, p in model.named_parameters():
            total += float((p.detach() - pre_snapshot[n]).pow(2).sum().item())
        return total ** 0.5

    def gradient_rms_history_per_rule(self) -> Dict[str, List[float]]:
        return {rid: list(rc.gradient_rms_history) for rid, rc in self.rule_classes.items()}

    def parameter_displacement_history_per_rule(self) -> Dict[str, List[float]]:
        return {rid: list(rc.parameter_displacement_history) for rid, rc in self.rule_classes.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _accuracy(pairs: List[Dict], model, encode_fn, decode_fn, compare_fn) -> float:
        """Run pairs through the caller's real inference path. Restores train mode."""
        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        correct = 0
        try:
            with torch.no_grad():
                for pair in pairs:
                    predicted = decode_fn(model(encode_fn(pair)))
                    if compare_fn(predicted, pair):
                        correct += 1
        finally:
            if was_training and hasattr(model, "train"):
                model.train()
        return correct / len(pairs) if pairs else 0.0

    def evaluate_generalization(
        self,
        rule_id: str,
        model: torch.nn.Module,
        encode_fn,
        decode_fn,
        compare_fn,
    ) -> float:
        """
        Run the model on held-out test pairs for this rule class and close
        the current evaluation window. Returns held-out accuracy (0.0 to 1.0).

        encode_fn(pair) -> model input
        decode_fn(model_output) -> predicted output representation
        compare_fn(predicted, expected_pair) -> bool (correct or not)

        A crystallized rule is evaluated too. Its result goes to the
        retention history; it is never assumed to be 1.0.
        """
        rc = self.rule_classes[rule_id]
        accuracy = self._accuracy(rc.test_pairs, model, encode_fn, decode_fn, compare_fn)

        if rc.is_crystallized:
            rc.retention_history.append(accuracy)
            return accuracy

        rc.generalization_accuracy_history.append(accuracy)
        rc.test_loss_history.append(1.0 - accuracy)
        if accuracy >= self.generalization_target:
            rc.consecutive_generalization_passes += 1
        else:
            rc.consecutive_generalization_passes = 0

        if rc.losses_since_eval:
            rc.train_window_mean_history.append(sum(rc.losses_since_eval) / len(rc.losses_since_eval))
            rc.losses_since_eval = []
        else:
            # No training happened in this window, so gate (a) has no evidence.
            rc.train_window_mean_history.append(float("inf"))

        rc.stability_history.append(self._signal_stable(rc))
        rc.false_positive_rate_history.append(
            self._false_positive_rate(rc, model, encode_fn, decode_fn, compare_fn)
        )
        return accuracy

    def _false_positive_rate(self, rc: RuleClass, model, encode_fn, decode_fn, compare_fn) -> float:
        """
        Gate (d). Show the model held-out inputs that belong to OTHER rules and
        count how often it wrongly gives one of THIS rule's answers.

        A model that says "PRINT" to everything scores 100% on the PRINT rule's
        own held-out set. It also says "PRINT" to every GOTO line, and that is
        what this measures.
        """
        if self.max_false_positive_rate is None:
            return 0.0
        foils = [p for rid, other in self.rule_classes.items() if rid != rc.rule_id
                 for p in other.test_pairs]
        if not foils:
            return 0.0
        if len(foils) > self.max_foils:           # deterministic, evenly spread sample
            stride = len(foils) / self.max_foils
            foils = [foils[int(i * stride)] for i in range(self.max_foils)]
        # one representative per distinct answer this rule can give
        own: List[Dict] = []
        for p in rc.test_pairs:
            if not any(_pair_output(p) == _pair_output(q) for q in own):
                own.append(p)

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        wrong = 0
        try:
            with torch.no_grad():
                for foil in foils:
                    predicted = decode_fn(model(encode_fn(foil)))
                    if compare_fn(predicted, foil):
                        continue                  # correct for its own rule: not a false positive
                    if any(compare_fn(predicted, p) for p in own):
                        wrong += 1
        finally:
            if was_training and hasattr(model, "train"):
                model.train()
        return wrong / len(foils)

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    def _signal_stable(self, rc: RuleClass) -> bool:
        """Gate (c): has this rule's learning signal actually stopped?"""
        histories = []
        if self.variance_check_quantity in ("gradient", "both"):
            histories.append(rc.gradient_rms_history)
        if self.variance_check_quantity in ("displacement", "both"):
            histories.append(rc.parameter_displacement_history)

        stable = True
        for hist in histories:
            if len(hist) < self.window_size:
                return False
            recent = torch.tensor(hist[-self.window_size:], dtype=torch.float64)
            mean = float(recent.mean().item())
            rc.peak_signal_window_mean = max(rc.peak_signal_window_mean, mean)
            if float(recent.var().item()) >= self.variance_threshold:
                stable = False
            if self.stability_decay_ratio is not None:
                # "Stopped" means decayed relative to this rule's own peak,
                # which is scale-free. A signal that never fell is not stable,
                # however low its variance.
                if mean > self.stability_decay_ratio * rc.peak_signal_window_mean:
                    stable = False
        return stable

    def gate_report(self, rule_id: str) -> Dict[str, bool]:
        """Which of the four gates currently hold for this rule."""
        rc = self.rule_classes[rule_id]
        k = self.consecutive_windows_required
        train_converged = (
            len(rc.train_window_mean_history) >= k
            and all(l < self.loss_threshold for l in rc.train_window_mean_history[-k:])
        )
        generalization_met = (
            len(rc.generalization_accuracy_history) >= k
            and all(a >= self.generalization_target for a in rc.generalization_accuracy_history[-k:])
        )
        weights_stable = bool(rc.stability_history) and rc.stability_history[-1]
        specific = (
            self.max_false_positive_rate is None
            or (len(rc.false_positive_rate_history) >= k
                and all(f <= self.max_false_positive_rate for f in rc.false_positive_rate_history[-k:]))
        )
        return {
            "train_converged": train_converged,
            "generalization_met": generalization_met,
            "weights_stable": weights_stable,
            "specific": specific,
        }

    def check_crystallization(self, rule_id: str) -> bool:
        """
        Check whether a rule class meets all four crystallization criteria.
        Returns True if the rule should crystallize NOW.
        """
        if self.rule_classes[rule_id].is_crystallized:
            return False
        return all(self.gate_report(rule_id).values())

    # ------------------------------------------------------------------
    # Crystallization
    # ------------------------------------------------------------------
    def crystallize(
        self,
        rule_id: str,
        model: torch.nn.Module,
        step: int,
    ) -> Dict:
        """
        Commit this rule class to frozen storage and mark it crystallized.
        Returns the crystallization event record.

        commit_fn runs first. If the commit fails, nothing is marked.
        """
        rc = self.rule_classes[rule_id]
        if self.commit_fn is not None:
            self.commit_fn(rule_id, rc)

        rc.is_crystallized = True
        rc.crystallized_at_step = step
        rc.retention_failed = False
        rc.crystallized_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        w = self.window_size
        event = {
            "event": "crystallized",
            "rule_id": rule_id,
            "description": rc.description,
            "crystallized_at_step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_train_loss": rc.train_loss_history[-1] if rc.train_loss_history else None,
            "final_generalization_accuracy": (
                rc.generalization_accuracy_history[-1]
                if rc.generalization_accuracy_history else None
            ),
            "final_gradient_rms_variance": _recent_variance(rc.gradient_rms_history, w),
            "final_parameter_displacement_variance": _recent_variance(rc.parameter_displacement_history, w),
            "final_weight_delta_variance": _recent_variance(rc.gradient_rms_history, w),  # legacy name
            "final_false_positive_rate": (
                rc.false_positive_rate_history[-1] if rc.false_positive_rate_history else None
            ),
            "total_train_steps": len(rc.train_loss_history),
            "num_train_pairs": len(rc.train_pairs),
            "num_test_pairs": len(rc.test_pairs),
            "committed_to_frozen_store": self.commit_fn is not None,
        }
        self.crystallization_log.append(event)
        return event

    def audit_retention(self, model, encode_fn, decode_fn, compare_fn) -> Dict[str, float]:
        """
        Re-test every crystallized rule on its training AND held-out pairs
        through the real inference path. Returns {rule_id: accuracy}.

        A rule below retention_target is recorded in regression_log and
        either flagged (retention_failed=True) or returned to the plastic
        pool, according to on_regression.
        """
        results: Dict[str, float] = {}
        for rid, rc in self.rule_classes.items():
            if not rc.is_crystallized:
                continue
            acc = self._accuracy(rc.train_pairs + rc.test_pairs, model, encode_fn, decode_fn, compare_fn)
            rc.retention_history.append(acc)
            results[rid] = acc
            if acc < self.retention_target:
                self.regression_log.append({
                    "event": "regression",
                    "rule_id": rid,
                    "detected_at_step": self.global_step,
                    "crystallized_at_step": rc.crystallized_at_step,
                    "retention_accuracy": acc,
                    "action": self.on_regression,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                if self.on_regression == "decrystallize":
                    rc.is_crystallized = False
                    rc.crystallized_at_step = None
                    rc.crystallized_weights = None
                    rc.consecutive_generalization_passes = 0
                    rc.generalization_accuracy_history.append(0.0)
                    rc.train_window_mean_history.append(float("inf"))
                    rc.stability_history.append(False)
                    rc.false_positive_rate_history.append(1.0)
                else:
                    rc.retention_failed = True
            else:
                rc.retention_failed = False
        return results

    def step(
        self,
        rule_id: str,
        train_loss: float,
        delta_norm: float,
        model: torch.nn.Module,
        encode_fn,
        decode_fn,
        compare_fn,
        eval_interval: int = 10,
        displacement_norm: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        All-in-one step: record metrics, evaluate generalization on this
        rule's own cadence, check crystallization, and crystallize if ready.
        Returns the crystallization event if one occurred, None otherwise.
        """
        self.global_step += 1
        rc = self.rule_classes[rule_id]
        if rc.is_crystallized:
            return None

        rc.steps_seen += 1
        self.record_train_loss(rule_id, train_loss)
        self.record_gradient_rms_norms(rule_id, delta_norm)
        if displacement_norm is not None:
            self.record_parameter_displacement(rule_id, displacement_norm)

        if rc.steps_seen % eval_interval == 0:
            self.evaluate_generalization(rule_id, model, encode_fn, decode_fn, compare_fn)
            if self.check_crystallization(rule_id):
                return self.crystallize(rule_id, model, self.global_step)

        return None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def status(self) -> Dict:
        """Summary of all rule classes and their crystallization state."""
        summary = {}
        for rid, rc in self.rule_classes.items():
            summary[rid] = {
                "crystallized": rc.is_crystallized,
                "crystallized_at_step": rc.crystallized_at_step,
                "train_steps": len(rc.train_loss_history),
                "last_train_loss": rc.train_loss_history[-1] if rc.train_loss_history else None,
                "last_generalization": (
                    rc.generalization_accuracy_history[-1]
                    if rc.generalization_accuracy_history else None
                ),
                "consecutive_passes": rc.consecutive_generalization_passes,
                "gates": self.gate_report(rid),
                "last_retention": rc.retention_history[-1] if rc.retention_history else None,
                "retention_failed": rc.retention_failed,
            }
        return summary

    def save_log(self, path: str) -> None:
        """Write the crystallization and regression event logs to disk."""
        with open(path, "w") as f:
            json.dump(self.crystallization_log + self.regression_log, f, indent=2)

    def fraction_crystallized(self) -> float:
        """What fraction of registered rule classes have crystallized."""
        if not self.rule_classes:
            return 0.0
        n_frozen = sum(1 for rc in self.rule_classes.values() if rc.is_crystallized)
        return n_frozen / len(self.rule_classes)

    def fraction_retained(self) -> float:
        """Of the crystallized rules, the fraction not currently flagged as forgotten."""
        frozen = [rc for rc in self.rule_classes.values() if rc.is_crystallized]
        if not frozen:
            return 1.0
        return sum(1 for rc in frozen if not rc.retention_failed) / len(frozen)

    # ------------------------------------------------------------------
    # Trajectory logging
    # ------------------------------------------------------------------
    def open_trajectory_logger(self, stage: int, log_dir: Optional[str] = None,
                               filename: str = "trajectory.jsonl") -> str:
        log_dir = log_dir or self.trajectory_log_dir or "."
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, filename)
        self.close_trajectory_logger()
        self._trajectory_file = open(path, "a")
        self._trajectory_stage = stage
        return path

    def log_trajectory_record(self, rule_id: str, global_step: int, epoch: int,
                              gradient_rms_norm: float, parameter_displacement_norm: float,
                              train_loss: float, recording_cadence_hit: bool = True) -> None:
        if self._trajectory_file is None:
            return
        dense = self._trajectory_records < self.trajectory_dense_logging_steps
        if not (dense or recording_cadence_hit):
            return
        self._trajectory_file.write(json.dumps({
            "stage": getattr(self, "_trajectory_stage", None),
            "rule_id": rule_id,
            "global_step": global_step,
            "epoch": epoch,
            "gradient_rms_norm": gradient_rms_norm,
            "parameter_displacement_norm": parameter_displacement_norm,
            "train_loss": train_loss,
            "crystallized": bool(self.rule_classes[rule_id].is_crystallized)
            if rule_id in self.rule_classes else False,
        }) + "\n")
        self._trajectory_records += 1

    def close_trajectory_logger(self) -> None:
        if self._trajectory_file is not None:
            self._trajectory_file.close()
            self._trajectory_file = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """Full state. commit_fn is not serialized; pass it again to from_dict."""
        return {
            "format": "neo_manager_v2",
            "config": {
                "loss_threshold": self.loss_threshold,
                "generalization_target": self.generalization_target,
                "variance_threshold": self.variance_threshold,
                "consecutive_windows_required": self.consecutive_windows_required,
                "window_size": self.window_size,
                "variance_check_quantity": self.variance_check_quantity,
                "stability_decay_ratio": self.stability_decay_ratio,
                "max_false_positive_rate": self.max_false_positive_rate,
                "max_foils": self.max_foils,
                "retention_target": self.retention_target,
                "on_regression": self.on_regression,
                "trajectory_dense_logging_steps": self.trajectory_dense_logging_steps,
                "trajectory_log_dir": self.trajectory_log_dir,
            },
            "global_step": self.global_step,
            "crystallization_log": list(self.crystallization_log),
            "regression_log": list(self.regression_log),
            "rule_classes": {
                rid: {
                    "rule_id": rc.rule_id,
                    "description": rc.description,
                    "train_pairs": rc.train_pairs,
                    "test_pairs": rc.test_pairs,
                    "crystallized_weights": rc.crystallized_weights,
                    **{f: (list(getattr(rc, f)) if isinstance(getattr(rc, f), list) else getattr(rc, f))
                       for f in _HISTORY_FIELDS},
                }
                for rid, rc in self.rule_classes.items()
            },
        }

    @classmethod
    def from_dict(cls, state: Dict, commit_fn: Optional[Callable] = None) -> "CrystallizationManager":
        if state.get("format") != "neo_manager_v2":
            raise ValueError(f"Unknown manager state format: {state.get('format')!r}")
        cm = cls(commit_fn=commit_fn, **state["config"])
        cm.global_step = state["global_step"]
        cm.crystallization_log = list(state["crystallization_log"])
        cm.regression_log = list(state["regression_log"])
        for rid, d in state["rule_classes"].items():
            rc = RuleClass(rule_id=d["rule_id"], description=d["description"],
                           train_pairs=d["train_pairs"], test_pairs=d["test_pairs"])
            for f in _HISTORY_FIELDS:
                v = d[f]
                setattr(rc, f, list(v) if isinstance(v, list) else v)
            rc.crystallized_weights = d["crystallized_weights"]
            cm.rule_classes[rid] = rc
        return cm


# =========================================================================
# SELF-TESTS
#
# These run real models through the real gate. Nothing is injected into a
# history list to make a gate pass.
# =========================================================================

def _run_rule(cm, rid, model, opt, make_batch, encode_fn, decode_fn, compare_fn,
              steps, eval_interval=10):
    """Train one rule for `steps` steps, feeding Neo honestly. Returns the event or None."""
    for _ in range(steps):
        x, y = make_batch()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        grad = cm._compute_gradient_rms_norm(model)
        opt.step()
        event = cm.step(rid, loss.item(), grad, model, encode_fn, decode_fn, compare_fn,
                        eval_interval=eval_interval)
        if event is not None:
            return event
    return None


def _toy_setup(seed=0):
    """
    A rule that can be understood: class = which half of the input has the
    larger sum. Training and held-out inputs are disjoint random vectors, and
    cases too close to the boundary to call are left out of both.
    """
    g = torch.Generator().manual_seed(seed)

    def sample(n):
        xs, ys = [], []
        while len(xs) < n:
            x = torch.randn(16, generator=g)
            margin = float(x[:8].sum() - x[8:].sum())
            if abs(margin) > 1.0:
                xs.append(x)
                ys.append(int(margin > 0))
        return torch.stack(xs), torch.tensor(ys)

    xtr, ytr = sample(256)
    xte, yte = sample(64)
    train_pairs = [{"input": tuple(x.tolist()), "output": int(y)} for x, y in zip(xtr, ytr)]
    test_pairs = [{"input": tuple(x.tolist()), "output": int(y)} for x, y in zip(xte, yte)]
    encode = lambda p: torch.tensor(p["input"]).unsqueeze(0)
    decode = lambda out: int(out.argmax(-1).item())
    compare = lambda pred, p: pred == p["output"]

    def make_batch():
        idx = torch.randint(0, len(xtr), (32,), generator=g)
        return xtr[idx], ytr[idx]

    return train_pairs, test_pairs, encode, decode, compare, make_batch


def _toy_manager(**kw):
    cfg = dict(loss_threshold=0.25, generalization_target=0.90, variance_threshold=1e-4,
               consecutive_windows_required=3, window_size=20)
    cfg.update(kw)
    return CrystallizationManager(**cfg)


def test_register_and_status():
    cm = CrystallizationManager()
    cm.register_rule_class(
        rule_id="print_basic",
        description="PRINT with string literal",
        train_pairs=[{"in": 'PRINT "Hello"', "out": "Print 'Hello' to screen"}],
        test_pairs=[{"in": 'PRINT "World"', "out": "Print 'World' to screen"}],
    )
    s = cm.status()
    assert "print_basic" in s
    assert s["print_basic"]["crystallized"] is False
    print("PASS: test_register_and_status")


def test_no_test_pairs_raises():
    cm = CrystallizationManager()
    try:
        cm.register_rule_class("bad", "no test data", [{"in": "x", "out": "y"}], [])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: test_no_test_pairs_raises")


def test_leaked_test_pair_raises():
    """Held-out data that is also in the training set is not held out."""
    cm = CrystallizationManager()
    try:
        cm.register_rule_class(
            "leak", "leaky",
            train_pairs=[{"input": "PRINT 1", "output": "a"}, {"input": "PRINT 2", "output": "b"}],
            test_pairs=[{"input": "PRINT 2", "output": "b"}],
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "held-out" in str(e).lower()
    print("PASS: test_leaked_test_pair_raises")


def test_genuine_rule_crystallizes():
    """A model that can represent the rule learns it, generalizes, and crystallizes."""
    torch.manual_seed(0)
    train_pairs, test_pairs, enc, dec, cmp_, make_batch = _toy_setup()
    cm = _toy_manager()
    cm.register_rule_class("energy", "which half has the larger sum", train_pairs, test_pairs)
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    event = _run_rule(cm, "energy", model, opt, make_batch, enc, dec, cmp_, steps=3000)
    assert event is not None, f"genuine rule failed to crystallize: {cm.status()['energy']}"
    assert event["final_generalization_accuracy"] >= 0.90
    print(f"PASS: test_genuine_rule_crystallizes (step {event['crystallized_at_step']}, "
          f"held-out {event['final_generalization_accuracy']:.0%})")


def test_memorizer_never_crystallizes():
    """
    A lookup table drives training loss to zero and cannot answer a single
    unseen input above chance. Neo must refuse it, and the refusal must come
    from the generalization gate.
    """
    torch.manual_seed(0)
    train_pairs, test_pairs, _, dec, cmp_, _ = _toy_setup()
    all_inputs = {p["input"]: i for i, p in enumerate(train_pairs + test_pairs)}
    table = torch.nn.Embedding(len(all_inputs), 2)       # one free row per example: pure memory
    enc = lambda p: torch.tensor([all_inputs[p["input"]]])
    ytr = torch.tensor([p["output"] for p in train_pairs])

    def make_batch():
        idx = torch.randint(0, len(train_pairs), (32,))
        return idx, ytr[idx]

    cm = _toy_manager()
    cm.register_rule_class("energy", "memorized", train_pairs, test_pairs)
    opt = torch.optim.Adam(table.parameters(), lr=5e-2)
    event = _run_rule(cm, "energy", table, opt, make_batch, enc, dec, cmp_, steps=3000)
    gates = cm.gate_report("energy")
    assert event is None, "a lookup table was crystallized"
    assert gates["train_converged"], "memorizer should have converged on training data"
    assert not gates["generalization_met"], "generalization gate should be the one refusing"
    gen = cm.rule_classes["energy"].generalization_accuracy_history[-1]
    print(f"PASS: test_memorizer_never_crystallizes (train gate open, held-out {gen:.0%}, refused)")


def test_steady_gradient_is_not_stable():
    """Zero variance is not stillness. A constant, undecayed signal must fail gate (c)."""
    cm = _toy_manager()
    cm.register_rule_class("r", "steady", [{"input": 1, "output": 1}], [{"input": 2, "output": 1}])
    always_right = (lambda p: p, lambda o: 1, lambda pred, p: True)
    model = torch.nn.Linear(1, 1)
    for _ in range(200):
        ev = cm.step("r", 0.001, 0.5, lambda x: x, *always_right, eval_interval=10)
        assert ev is None, "crystallized on a constant non-decaying learning signal"
    gates = cm.gate_report("r")
    assert gates == {"train_converged": True, "generalization_met": True,
                     "weights_stable": False, "specific": True}, gates
    # the same rule does crystallize once the signal actually decays
    ev = None
    for i in range(200):
        ev = ev or cm.step("r", 0.001, 0.5 * (0.9 ** i), model, lambda p: torch.zeros(1, 1),
                           lambda o: 1, lambda pred, p: True, eval_interval=10)
    assert ev is not None, "did not crystallize after the signal decayed"
    print("PASS: test_steady_gradient_is_not_stable")


def test_constant_predictor_never_crystallizes():
    """
    Train on ONE class only. The model learns to answer that class for every
    input, scores 100% on that rule's own held-out set, and must be refused
    because it gives the same answer to the other rules' inputs.
    """
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    protos = torch.randn(4, 16, generator=g) * 3

    def pairs(c, n):
        return [{"input": tuple((protos[c] + torch.randn(16, generator=g) * 0.3).tolist()), "output": c}
                for _ in range(n)]

    cm = _toy_manager()
    data = {c: (pairs(c, 40), pairs(c, 10)) for c in range(4)}
    for c, (tr, te) in data.items():
        cm.register_rule_class(f"class{c}", "", tr, te)
    enc = lambda p: torch.tensor(p["input"]).unsqueeze(0)
    dec = lambda out: int(out.argmax(-1).item())
    cmp_ = lambda pred, p: pred == p["output"]
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x0 = torch.tensor([p["input"] for p in data[0][0]])

    def only_class0():
        idx = torch.randint(0, 40, (16,))
        return x0[idx], torch.zeros(16, dtype=torch.long)

    event = _run_rule(cm, "class0", model, opt, only_class0, enc, dec, cmp_, steps=1500)
    gates = cm.gate_report("class0")
    fpr = cm.rule_classes["class0"].false_positive_rate_history[-1]
    assert event is None, "a constant predictor was crystallized"
    assert gates["generalization_met"] and not gates["specific"], gates
    print(f"PASS: test_constant_predictor_never_crystallizes "
          f"(own held-out 100%, false-positive rate {fpr:.0%}, refused)")

    # the same model trained on all classes together does crystallize class0
    xs = torch.tensor([p["input"] for c in range(4) for p in data[c][0]])
    ys = torch.tensor([p["output"] for c in range(4) for p in data[c][0]])

    def mixed():
        idx = torch.randint(0, len(xs), (32,))
        return xs[idx], ys[idx]

    event = _run_rule(cm, "class0", model, opt, mixed, enc, dec, cmp_, steps=3000)
    assert event is not None, f"interleaved training should crystallize: {cm.status()['class0']}"
    print("PASS: ...and the same rule crystallizes once classes are trained together")


def test_every_rule_gets_evaluated():
    """Round-robin order must not starve any rule of evaluation."""
    cm = _toy_manager()
    n_rules = 15
    for i in range(n_rules):
        cm.register_rule_class(f"r{i}", "", [{"input": i, "output": 0}], [{"input": 100 + i, "output": 0}])
    fns = (lambda p: p, lambda o: 0, lambda pred, p: True)
    for _ in range(40):
        for i in range(n_rules):
            cm.step(f"r{i}", 1.0, 1.0, lambda x: x, *fns, eval_interval=n_rules)
    counts = [len(cm.rule_classes[f"r{i}"].generalization_accuracy_history) for i in range(n_rules)]
    assert min(counts) >= 2, f"some rules were never evaluated: {counts}"
    print(f"PASS: test_every_rule_gets_evaluated (evaluations per rule: {min(counts)}..{max(counts)})")


def test_retention_audit_detects_forgetting():
    """
    Crystallize rule A, then train the same weights on the opposite rule.
    Without a frozen store the audit must catch the forgetting. With a frozen
    store consulted first at inference, retention must stay at 100%.
    """
    torch.manual_seed(0)
    train_pairs, test_pairs, enc, dec, cmp_, make_batch = _toy_setup()
    frozen_store: Dict[Any, int] = {}

    def commit(rule_id, rc):
        for p in rc.train_pairs + rc.test_pairs:
            frozen_store[p["input"]] = p["output"]

    for use_store in (False, True):
        frozen_store.clear()
        torch.manual_seed(0)
        cm = _toy_manager(commit_fn=commit if use_store else None)
        cm.register_rule_class("energy", "", train_pairs, test_pairs)
        model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        assert _run_rule(cm, "energy", model, opt, make_batch, enc, dec, cmp_, steps=3000) is not None

        # interference: train the same weights on the inverted rule
        for _ in range(300):
            x, y = make_batch()
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(x), 1 - y).backward()
            opt.step()

        # inference path: frozen store first, plastic model second
        def infer_decode(pair_and_out):
            return pair_and_out

        def encode_with_store(p):
            return p

        class Path(torch.nn.Module):
            def forward(self, p):
                if p["input"] in frozen_store:
                    return frozen_store[p["input"]]
                return dec(model(enc(p)))

        retention = cm.audit_retention(Path(), encode_with_store, infer_decode, cmp_)["energy"]
        if use_store:
            assert retention == 1.0 and cm.fraction_retained() == 1.0 and not cm.regression_log
            protected = retention
        else:
            assert retention < 0.5, f"interference did not cause forgetting ({retention:.0%})"
            assert cm.rule_classes["energy"].retention_failed and len(cm.regression_log) == 1
            unprotected = retention
    print(f"PASS: test_retention_audit_detects_forgetting "
          f"(no frozen store: {unprotected:.0%} retained and flagged; frozen store: {protected:.0%})")


def test_decrystallize_on_regression():
    cm = _toy_manager(on_regression="decrystallize")
    cm.register_rule_class("r", "", [{"input": 1, "output": 1}], [{"input": 2, "output": 1}])
    cm.crystallize("r", torch.nn.Linear(1, 1), step=1)
    cm.audit_retention(lambda p: 0, lambda p: p, lambda o: o, lambda pred, p: pred == p["output"])
    assert not cm.rule_classes["r"].is_crystallized and cm.fraction_crystallized() == 0.0
    assert not cm.check_crystallization("r"), "must re-earn crystallization from scratch"
    print("PASS: test_decrystallize_on_regression")


def test_failed_commit_blocks_crystallization():
    def bad_commit(rule_id, rc):
        raise RuntimeError("frozen store is full")
    cm = _toy_manager(commit_fn=bad_commit)
    cm.register_rule_class("r", "", [{"input": 1, "output": 1}], [{"input": 2, "output": 1}])
    try:
        cm.crystallize("r", torch.nn.Linear(1, 1), step=1)
        assert False, "commit failure must propagate"
    except RuntimeError:
        pass
    assert not cm.rule_classes["r"].is_crystallized and not cm.crystallization_log
    print("PASS: test_failed_commit_blocks_crystallization")


def test_state_round_trip():
    """A manager restored from to_dict() makes the same decisions as the original."""
    torch.manual_seed(0)
    train_pairs, test_pairs, enc, dec, cmp_, make_batch = _toy_setup()
    cm = _toy_manager()
    cm.register_rule_class("energy", "", train_pairs, test_pairs)
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    assert _run_rule(cm, "energy", model, opt, make_batch, enc, dec, cmp_, steps=25) is None
    assert cm.rule_classes["energy"].generalization_accuracy_history, "need a mid-flight state"

    import copy
    restored = CrystallizationManager.from_dict(copy.deepcopy(cm.to_dict()))
    assert restored.status() == cm.status()
    assert restored.to_dict() == cm.to_dict()
    for _ in range(40):
        a = cm.step("energy", 0.01, 0.0, model, enc, dec, cmp_)
        b = restored.step("energy", 0.01, 0.0, model, enc, dec, cmp_)
        assert (a is None) == (b is None)
    assert restored.gate_report("energy") == cm.gate_report("energy")
    print("PASS: test_state_round_trip")


if __name__ == "__main__":
    test_register_and_status()
    test_no_test_pairs_raises()
    test_leaked_test_pair_raises()
    test_genuine_rule_crystallizes()
    test_memorizer_never_crystallizes()
    test_steady_gradient_is_not_stable()
    test_constant_predictor_never_crystallizes()
    test_every_rule_gets_evaluated()
    test_retention_audit_detects_forgetting()
    test_decrystallize_on_regression()
    test_failed_commit_blocks_crystallization()
    test_state_round_trip()
    print("\nAll crystallization_manager tests passed.")
