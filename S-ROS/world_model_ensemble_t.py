"""
world_model_ensemble_teaching.py

BIOLOGICAL GROUNDING
====================
This file models the predictive coding function of neocortical circuits, specifically
the generative model that the brain maintains over its own sensory and internal states.

In predictive processing theory, the brain continuously predicts the causes of its
sensory inputs and updates those predictions based on prediction error. Neocortical
pyramidal neurons in superficial layers (II/III) send predictions downward and to
lateral areas; deep layers (V/VI) send prediction errors upward. The brain does not
passively receive information but actively anticipates it and is surprised by
violations of its predictions.

The WorldModelEnsemble implements a simplified version of this generative model
operating in the 64-dimensional coordinate space of the Perforant Path manifold.
Five independent prediction heads each learn to predict the next coordinate-space
representation given the current one. Ensemble disagreement (variance across heads)
is the epistemic uncertainty signal: when the five heads disagree, the world model
is uncertain, which means the current region of the manifold is undersampled and
high in expected information gain.

This module is the only component of the CuriosityHead refactor that has learnable
parameters and its own optimizer. It is extracted from the monolithic CuriosityHead
class per the Task 3 spec from MARCHING_ORDERS_APRIL5.md. The MaturityComputer and
EpistemicSelector consume its outputs but do not own it.

Primary papers grounding this file:

Friston K, et al. (2017). "Active inference: a process theory." Neural Computation,
29(1), 1-49. DOI: 10.1162/NECO_a_00912

Friston K (2010). "The free-energy principle: a unified brain theory." Nature Reviews
Neuroscience, 11(2), 127-138. DOI: 10.1038/nrn2787

Rao RP, Ballard DH (1999). "Predictive coding in the visual cortex: a functional
interpretation of some extra-classical receptive-field effects." Nature Neuroscience,
2(1), 79-87. DOI: 10.1038/4580
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelConfig:
    """
    Configuration for the WorldModelEnsemble.

    Attributes:
        coordinate_dim: Dimensionality of the Perforant Path coordinate manifold.
            Fixed at 64 throughout the architecture. This is the communication
            channel between Timmy columns and the Cognitive Kernel.
            NOT a biological quantity: engineering parameter.
            Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026

        hidden_dim: Width of each prediction MLP's hidden layers.
            NOT a biological quantity: engineering hyperparameter.
            Initial value 128 chosen as 2x coordinate_dim.

        n_heads: Number of independent prediction MLPs in the ensemble.
            NOT a biological quantity: engineering hyperparameter.
            Five heads chosen as the minimum ensemble size that provides
            a reliable variance estimate without excessive compute cost.
            Lakshminarayanan B et al. (2017) showed five members sufficient
            for calibrated uncertainty in deep ensembles.
            Lakshminarayanan B, Pritzel A, Blundell C (2017). "Simple and
            scalable predictive uncertainty estimation using deep ensembles."
            NeurIPS. DOI: {To be added later.}

        learning_rate: Adam optimizer learning rate for ensemble parameters.
            NOT a biological quantity: training artifact.

        variance_ceiling: Maximum expected ensemble variance used to normalize
            the ensemble_agreement_component of maturity. Prevents a single
            high-variance event from dominating the maturity signal.
            NOT a biological quantity: engineering hyperparameter.
            [NEEDS INPUT] Initial value 0.1 is a placeholder pending empirical
            calibration from early training runs.

        error_window: Number of recent prediction errors to average when
            computing mean_prediction_error for maturity computation.
            NOT a biological quantity: engineering hyperparameter.
    """
    coordinate_dim: int = 64
    hidden_dim: int = 128
    n_heads: int = 5
    learning_rate: float = 1e-4
    variance_ceiling: float = 0.1
    error_window: int = 50


class PredictionHead(nn.Module):
    """
    Single MLP prediction head within the WorldModelEnsemble.

    BIOLOGICAL STRUCTURE: Superficial layer (II/III) pyramidal neurons in
    neocortex that generate predictions about incoming sensory states.
    BIOLOGICAL FUNCTION: These neurons send lateral and feedback projections
    that carry predictions of expected input to lower cortical areas and
    neighboring columns, implementing the generative model of predictive coding.

    Friston K (2010). DOI: 10.1038/nrn2787
    Rao RP, Ballard DH (1999). DOI: 10.1038/4580

    Each head is a three-layer MLP: input -> hidden -> hidden -> output.
    All five heads in the ensemble share the same architecture but have
    independent weights initialized differently, ensuring they explore
    different regions of the prediction function space.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        """
        Initialize a single prediction head.

        Args:
            cfg: WorldModelConfig.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.coordinate_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.coordinate_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Predict the next coordinate-space representation.

        Args:
            coords: Current coordinate representation, shape (B, coordinate_dim).

        Returns:
            Predicted next coordinate representation, shape (B, coordinate_dim).
        """
        return self.net(coords)


class WorldModelEnsemble(nn.Module):
    """
    Ensemble of five prediction heads implementing the system's generative model
    over the Perforant Path coordinate manifold.

    BIOLOGICAL STRUCTURE: Neocortical generative model; predictive coding network
    spanning superficial pyramidal layers across association cortex.
    BIOLOGICAL FUNCTION: The brain maintains a model of its own sensory and
    internal states. Predictions from this model are continuously compared with
    actual states; the mismatch (prediction error) drives both learning and
    attention allocation. High prediction error signals a region of experience
    that is novel or undermodeled, indicating where learning resources should
    be directed. This is the neural basis of curiosity and active exploration.

    Friston K, et al. (2017). DOI: 10.1162/NECO_a_00912
    Friston K (2010). DOI: 10.1038/nrn2787

    ANATOMICAL INTERFACE (input):
        Sending structure: PerforantPathSymphonyBridge (TimmyArray output).
        Receiving structure: WorldModelEnsemble (this module).
        Connection: Perforant path, entorhinal cortex to hippocampus and
            association cortex; here modeled as the 64-dim coordinate manifold.
        Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026

    ANATOMICAL INTERFACE (output, uncertainty signal):
        Sending structure: WorldModelEnsemble.
        Receiving structure: MaturityComputer (on NeuromodulatorBroadcast).
        Connection: No direct biological analog. Engineering interface:
            ensemble_variance is read by MaturityComputer once per sleep cycle
            to update the global maturity scalar.

    ANATOMICAL INTERFACE (output, batch selection):
        Sending structure: WorldModelEnsemble.
        Receiving structure: EpistemicSelector (on ActiveDataLoader).
        Connection: No direct biological analog. Engineering interface:
            EpistemicSelector queries this module under torch.no_grad()
            to rank candidate batches by expected information gain.

    This is the ONLY component of the CuriosityHead refactor with learnable
    parameters. The MaturityComputer and EpistemicSelector are stateless
    methods that consume this module's outputs.

    The optimizer lives inside this module intentionally. The world model
    has an independent learning dynamic from the main spiking network: it
    trains on detached coordinate targets and must not influence the gradients
    of the spiking substrate.
    Optimizer state is serialized in the HOT layer alongside STDP scalars
    and MoE EMAs.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        """
        Initialize the ensemble of prediction heads and optimizer.

        Args:
            cfg: WorldModelConfig.
        """
        super().__init__()
        self.cfg = cfg

        # Five independent prediction heads.
        # Each is initialized with different random weights by default
        # PyTorch initialization, ensuring ensemble diversity.
        # NOT a biological quantity: engineering ensemble design.
        self.heads = nn.ModuleList([
            PredictionHead(cfg) for _ in range(cfg.n_heads)
        ])

        # Adam optimizer for ensemble parameters only.
        # Lives inside the module because the world model has an independent
        # learning dynamic from the main spiking network.
        # NOT a biological quantity: training artifact.
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=cfg.learning_rate
        )

        # Rolling buffer of recent prediction errors for maturity computation.
        # Stored as a plain list, not a registered buffer, because it does not
        # need to participate in state_dict or device movement.
        # Serialized in HOT layer via get_hot_state()/load_hot_state().
        # NOT a biological quantity: engineering diagnostic buffer.
        self._recent_errors: list[float] = []

        # Rolling buffer of recent ensemble variances.
        # NOT a biological quantity: engineering diagnostic buffer.
        self._recent_variances: list[float] = []

    def forward(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run all five prediction heads and return predictions plus uncertainty metrics.

        BIOLOGICAL FUNCTION: The generative model predicts the next state from
        the current state. Ensemble disagreement is the epistemic uncertainty:
        regions of the manifold where heads disagree are undersampled and
        represent high expected information gain.
        Friston K, et al. (2017). DOI: 10.1162/NECO_a_00912

        Args:
            coords: Current coordinate representation from PerforantPathSymphonyBridge,
                shape (B, coordinate_dim).

        Returns:
            predictions: All head outputs stacked, shape (n_heads, B, coordinate_dim).
            mean_prediction: Mean across heads, shape (B, coordinate_dim).
            ensemble_variance: Scalar mean variance across heads and dimensions.
                This is the epistemic uncertainty signal consumed by
                MaturityComputer and EpistemicSelector.
        """
        predictions = torch.stack(
            [head(coords) for head in self.heads], dim=0
        )
        mean_prediction = predictions.mean(dim=0)
        ensemble_variance = predictions.var(dim=0).mean()
        return predictions, mean_prediction, ensemble_variance

    def update(
        self, coords: torch.Tensor, next_coords: torch.Tensor
    ) -> float:
        """
        Train the ensemble to predict next_coords from coords.

        BIOLOGICAL FUNCTION: Prediction error drives synaptic weight updates
        in the generative model, reducing future prediction error in the
        current region of experience. This is the Hebbian-like updating of
        the cortical generative model described by predictive coding theory.
        Rao RP, Ballard DH (1999). DOI: 10.1038/4580

        Gradients do NOT flow into the spiking substrate. next_coords is
        detached before use. This is a training artifact ensuring the world
        model's learning signal does not contaminate the main BPTT graph.
        NOT a biological quantity: gradient isolation is an engineering
        constraint imposed by the discrete-time BPTT training regime.
        Neftci EO et al. (2019). DOI: 10.1109/MSP.2019.2931595

        Args:
            coords: Current coordinate representation, shape (B, coordinate_dim).
                Detached inside this method before the forward call. The caller
                does not need to detach, but doing so is harmless.
            next_coords: Actual next coordinate representation,
                shape (B, coordinate_dim). Used as prediction target.
                Detached before use inside this method regardless of
                whether the caller detaches it.

        Returns:
            mean_loss: Mean MSE loss across all heads. Recorded in the
                rolling error buffer for maturity computation.
        """
        # Detach both inputs. coords must be detached so the world model's
        # backward pass does not accumulate gradients into the spiking
        # network's computation graph. This is a training artifact.
        # NOT a biological quantity.
        # Neftci EO et al. (2019). DOI: 10.1109/MSP.2019.2931595
        target = next_coords.detach()
        coords_detached = coords.detach()
        self.optimizer.zero_grad()
        predictions, _, _ = self.forward(coords_detached)
        losses = torch.stack([
            F.mse_loss(predictions[i], target)
            for i in range(self.cfg.n_heads)
        ])
        total_loss = losses.mean()
        total_loss.backward()
        self.optimizer.step()
        error_val = total_loss.item()
        self._recent_errors.append(error_val)
        if len(self._recent_errors) > self.cfg.error_window:
            self._recent_errors.pop(0)
        return error_val

    @torch.no_grad()
    def evaluate_batch(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Evaluate a candidate batch without updating the model.

        Called by EpistemicSelector to rank candidate batches by expected
        information gain. Runs under torch.no_grad() to prevent gradient
        accumulation during batch selection.

        BIOLOGICAL FUNCTION: The organism evaluates potential actions (data
        batches) by simulating their expected epistemic value before committing.
        This is the active inference action selection principle.
        Friston K, et al. (2017). DOI: 10.1162/NECO_a_00912

        Args:
            coords: Candidate batch coordinate representation,
                shape (B, coordinate_dim).

        Returns:
            mean_prediction: Mean prediction across heads,
                shape (B, coordinate_dim).
            ensemble_variance: Scalar epistemic uncertainty for this batch.
                Higher variance = higher expected information gain.
        """
        _, mean_prediction, ensemble_variance = self.forward(coords)
        var_val = ensemble_variance.item()
        self._recent_variances.append(var_val)
        if len(self._recent_variances) > self.cfg.error_window:
            self._recent_variances.pop(0)
        return mean_prediction, var_val

    def mean_recent_error(self) -> float:
        """
        Return the mean prediction error over the recent error window.

        Consumed by MaturityComputer once per sleep cycle as the
        loss_smoothness proxy signal for maturity computation.

        Returns:
            Mean of recent MSE losses, or 1.0 if no errors recorded yet
            (pessimistic default: assume high error before any training).
        """
        if not self._recent_errors:
            return 1.0
        return float(sum(self._recent_errors) / len(self._recent_errors))

    def mean_recent_variance(self) -> float:
        """
        Return the mean ensemble variance over the recent variance window.

        Consumed by MaturityComputer once per sleep cycle as the
        ensemble_agreement_component of maturity.

        Returns:
            Mean of recent ensemble variances, or 1.0 if no variances
            recorded yet (pessimistic default: assume high uncertainty).
        """
        if not self._recent_variances:
            return 1.0
        return float(sum(self._recent_variances) / len(self._recent_variances))

    def get_hot_state(self) -> dict:
        """
        Return optimizer state and rolling buffers for HOT layer serialization.

        The ensemble weights go in the COLD layer via state_dict(). The
        optimizer state and rolling buffers go in the HOT layer because they
        are runtime training state, not learned weights.

        Returns:
            Dict with 'optimizer_state', 'recent_errors', 'recent_variances'.
        """
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "recent_errors": list(self._recent_errors),
            "recent_variances": list(self._recent_variances),
        }

    def load_hot_state(self, hot: dict) -> None:
        """
        Restore optimizer state and rolling buffers from a HOT layer checkpoint.

        Args:
            hot: Dict produced by get_hot_state().
        """
        self.optimizer.load_state_dict(hot["optimizer_state"])
        self._recent_errors = list(hot.get("recent_errors", []))
        self._recent_variances = list(hot.get("recent_variances", []))
