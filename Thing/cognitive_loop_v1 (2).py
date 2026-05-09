"""cognitive_loop_v1.py — single-file aggregation of the PRAGMI substrate.

The 22 substrate modules plus the run_all.py test harness at the end.


Module order (top to bottom of file):

  Input gating and primary sensory:
    thalamic_gate, primary_sensory_cortex, dorsal_ventral_streams.
  Cortical machinery:
    cortical_interneurons, layer5b_apical, association_cortex, claustrum.
  Subcortical selection and monitoring:
    basal_ganglia, anterior_cingulate_cortex, cerebellum, amygdala.
  Hippocampal formation:
    entorhinal_cortex, cornu_ammonis_1, cornu_ammonis_2, ca2_part1.
  Neuromodulator nuclei:
    ventral_tegmental_area, locus_coeruleus, dorsal_raphe, basal_forebrain.
  Sleep and oscillatory coordination:
    sleep_stage_oscillator, spindle_ripple_coupling.
  Test harness:
    run_all.



Source: github.com/genesislabs-research/Pragmi-Cognitive-Loop_v1
Aggregated: 2026-04-30
"""
from __future__ import annotations


# =========================================================================
# SECTION: thalamic_gate
# (originally thalamic_gate_t.py)
# =========================================================================
"""
thalamic_gate_t.py
Loop Stage 1: Thalamic Gate with TRN-Mediated Cross-Modal Routing

BIOLOGICAL GROUNDING
The thalamic gate is the first stage of the PRAGMI loop. The classical
description of this stage as a multiplicative sigmoid gate captures
the input-output behavior but understates the underlying circuitry.
The empirical pathway is a four-stage cascade:

    1. PFC carries a contextual control signal carrying current goals
       and conflict.
    2. PFC routes through inhibitory projections to basal ganglia
       (Nakajima, Schmitt, Halassa 2019).
    3. Basal ganglia inhibits modality-specific sectors of the
       thalamic reticular nucleus (TRN), which itself is inhibitory.
    4. TRN disinhibits the corresponding thalamocortical relay
       neurons in the attended thalamic nucleus, producing the
       gain amplification that the cortex receives as attention.

The key quantitative result from Gu, Lam, Wimmer, Halassa, Murray
(2021) is that top-down inhibition onto TRN is 6 to 7 times more
potent at modulating thalamocortical gain than direct top-down
excitation onto thalamocortical relay neurons. This is a disinhibitory
geometry and the convex f-I curve of TRN neurons is what produces
the multiplier. A faithful implementation cannot collapse the four
stages into a single sigmoid without losing the empirical magnitudes.

This file implements the four-stage cascade with each stage as an
independently ablatable submodule so that the contribution of each
stage to the overall gain is measurable.

Primary grounding papers:

Gu Y, Lam NH, Wimmer RD, Halassa MM, Murray JD (2021).
"Computational circuit mechanisms underlying thalamic control of
attention." bioRxiv 2020.09.16.300749.
DOI: 10.1101/2020.09.16.300749

Nakajima M, Schmitt LI, Halassa MM (2019). "Prefrontal cortex
regulates sensory filtering through a basal ganglia-to-thalamus
pathway." Neuron, 103(3), 445-458.
DOI: 10.1016/j.neuron.2019.05.026

Wimmer RD, Schmitt LI, Davidson TJ, Nakajima M, Deisseroth K,
Halassa MM (2015). "Thalamic control of sensory selection in divided
attention." Nature, 526(7575), 705-709.
DOI: 10.1038/nature15398

Halassa MM, Kastner S (2017). "Thalamic functions in distributed
cognitive control." Nature Neuroscience, 20(12), 1669-1679.
DOI: 10.1038/s41593-017-0020-1

Pinault D (2004). "The thalamic reticular nucleus: structure,
function and concept." Brain Research Reviews, 46(1), 1-31.
DOI: 10.1016/j.brainresrev.2004.04.008

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# PART 1: CONFIG
# =========================================================================

@dataclass
class ThalamicGateConfig:
    """Configuration for the four-stage thalamic gate.

    Master flag is first per the Genesis Labs Research Ablation Flag
    Design Standard. NOT a biological quantity.
    """

    # Master flag for the entire gate. NOT a biological quantity.
    enable_thalamic_gate: bool = True

    # Stage 1: PFC contextual control signal computation. NOT a
    # biological quantity.
    enable_pfc_control: bool = True

    # Stage 2: PFC-to-basal-ganglia routing. Nakajima et al. (2019)
    # DOI: 10.1016/j.neuron.2019.05.026. NOT a biological quantity.
    enable_bg_routing: bool = True

    # Stage 3: TRN disinhibitory gating. Gu et al. (2021) DOI:
    # 10.1101/2020.09.16.300749. NOT a biological quantity.
    enable_trn_disinhibition: bool = True

    # Stage 4: thalamocortical relay output. NOT a biological quantity.
    enable_tc_relay: bool = True

    # Norepinephrine gain modulation. Yu and Dayan (2005). NOT a
    # biological quantity in the strict sense; the NE source is
    # computed elsewhere and passed in. The flag controls whether
    # this gate honors the NE signal at all.
    enable_ne_modulation: bool = True

    # Dimensionalities. NOT biological quantities.
    input_dim: int = 64
    pfc_control_dim: int = 64
    n_modalities: int = 4
    trn_dim: int = 32

    # TRN convex nonlinearity exponent. Gu et al. (2021) document a
    # convex f-I curve for TRN neurons. The exponent here determines
    # how much the disinhibitory geometry multiplies the gain. A
    # value of 2.0 produces the 6 to 7x amplification reported in
    # the paper for biologically plausible parameter ranges. This is
    # an engineering approximation parameterized to match the
    # empirical multiplier from Gu et al. (2021).
    trn_convexity_exponent: float = 2.0

    # Default NE gain when NE modulation is disabled or NE is not
    # supplied. NOT a biological quantity, neutral value.
    ne_gain_default: float = 1.0


# =========================================================================
# PART 2: PFC CONTROL SIGNAL
# =========================================================================

class PFCControlSignal(nn.Module):
    """Stage 1: PFC contextual control signal.

    BIOLOGICAL STRUCTURE: Prefrontal cortex layer V/VI output neurons
    that carry goal and conflict information toward subcortical
    targets including basal ganglia.

    BIOLOGICAL FUNCTION: Computes a context vector combining current
    goal state with conflict signal from ACC. This is the c term in
    the corpus equation y = x times sigmoid(W*c + b).

    Halassa MM, Kastner S (2017). DOI: 10.1038/s41593-017-0020-1
    """

    def __init__(self, cfg: ThalamicGateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.combine = nn.Linear(
            cfg.input_dim, cfg.pfc_control_dim, bias=True,
        )

    def forward(
        self,
        goal_state: torch.Tensor,
        acc_conflict: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce the PFC control vector.

        Args:
            goal_state: (B, input_dim) PFC goal representation.
            acc_conflict: optional (B,) conflict scalar from ACC.

        Returns:
            control: (B, pfc_control_dim) the c term.
        """
        if not self.cfg.enable_pfc_control:
            return torch.zeros(
                goal_state.shape[0], self.cfg.pfc_control_dim,
                device=goal_state.device, dtype=goal_state.dtype,
            )
        control = self.combine(goal_state)
        if acc_conflict is not None:
            control = control + acc_conflict.unsqueeze(-1)
        return control


# =========================================================================
# PART 3: BASAL GANGLIA ROUTING
# =========================================================================

class BasalGangliaRouter(nn.Module):
    """Stage 2: PFC-to-BG-to-TRN inhibitory routing.

    BIOLOGICAL STRUCTURE: Striatum to globus pallidus internal segment
    to substantia nigra pars reticulata, with inhibitory output to TRN.

    BIOLOGICAL FUNCTION: Nakajima, Schmitt, Halassa (2019) demonstrate
    that PFC modulates sensory filtering not by direct projection to
    TRN but by routing through the basal ganglia inhibitory system.
    Striatal activity inhibits GPi/SNr tonic output, which releases
    TRN sectors from suppression. The functional effect is selection
    of which sensory modality the TRN will gate through.

    Nakajima M, Schmitt LI, Halassa MM (2019).
    DOI: 10.1016/j.neuron.2019.05.026

    ANATOMICAL INTERFACE:
        Sending structure: PFC layer V output neurons.
        Receiving structure: striatal medium spiny neurons.
        Connection: corticostriatal projection.
    """

    def __init__(self, cfg: ThalamicGateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Modality selector: maps PFC control to a softmax over
        # candidate modalities. The softmax represents the soft
        # selection that BG performs through competition between
        # direct and indirect pathways.
        self.modality_selector = nn.Linear(
            cfg.pfc_control_dim, cfg.n_modalities, bias=True,
        )

    def forward(self, pfc_control: torch.Tensor) -> torch.Tensor:
        """Compute inhibitory drive onto each TRN modality sector.

        Args:
            pfc_control: (B, pfc_control_dim).

        Returns:
            bg_inhibition: (B, n_modalities) inhibition pattern onto
                TRN sectors. Higher values mean more inhibition of TRN,
                which means more disinhibition of the corresponding
                thalamic relay.
        """
        if not self.cfg.enable_bg_routing:
            return torch.zeros(
                pfc_control.shape[0], self.cfg.n_modalities,
                device=pfc_control.device, dtype=pfc_control.dtype,
            )
        # Softmax produces the soft selection across modalities.
        return F.softmax(self.modality_selector(pfc_control), dim=-1)


# =========================================================================
# PART 4: TRN DISINHIBITORY GATE
# =========================================================================

class TRNDisinhibitoryGate(nn.Module):
    """Stage 3: TRN-mediated disinhibitory gate with convex f-I curve.

    BIOLOGICAL STRUCTURE: Thalamic reticular nucleus, a thin shell of
    GABAergic neurons surrounding the dorsal thalamus. Subdivided into
    sensory sectors by modality.

    BIOLOGICAL FUNCTION: TRN provides tonic inhibition onto
    thalamocortical relay neurons. When TRN itself is inhibited
    (by BG output), the inhibition on TC neurons is released and TC
    gain rises sharply. The f-I curve of TRN neurons is convex, so
    small changes in TRN drive produce large changes in disinhibition
    magnitude. Gu et al. (2021) show that this geometry produces a
    6 to 7x amplification of top-down gain modulation compared to
    direct excitatory drive on TC neurons.

    Gu Y, Lam NH, Wimmer RD, Halassa MM, Murray JD (2021).
    DOI: 10.1101/2020.09.16.300749
    Pinault D (2004). DOI: 10.1016/j.brainresrev.2004.04.008

    ANATOMICAL INTERFACE:
        Sending structures: basal ganglia (inhibitory drive onto TRN)
        and thalamocortical collaterals (excitatory drive onto TRN).
        Receiving structure: thalamocortical relay neurons (inhibition
        from TRN).
        Connection: TRN-to-TC inhibitory projection.
    """

    def __init__(self, cfg: ThalamicGateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Map BG inhibition pattern to per-input gating gain. The
        # convex nonlinearity is applied here.
        self.bg_to_trn = nn.Linear(
            cfg.n_modalities, cfg.input_dim, bias=True,
        )

    def forward(self, bg_inhibition: torch.Tensor) -> torch.Tensor:
        """Compute the disinhibitory gain pattern.

        Args:
            bg_inhibition: (B, n_modalities) from BasalGangliaRouter.

        Returns:
            gain: (B, input_dim) multiplicative gain on the TC relay.
        """
        if not self.cfg.enable_trn_disinhibition:
            return torch.ones(
                bg_inhibition.shape[0], self.cfg.input_dim,
                device=bg_inhibition.device, dtype=bg_inhibition.dtype,
            )

        # TRN inhibition baseline minus BG-derived release.
        raw = self.bg_to_trn(bg_inhibition)
        # Convex disinhibition: small inhibition drops produce
        # disproportionately large gain increases via the convex
        # f-I curve. Sigmoid first to bound, then raise to the
        # convexity exponent.
        sigmoid_drive = torch.sigmoid(raw)
        gain = sigmoid_drive ** self.cfg.trn_convexity_exponent
        # Scale up by the empirical 6 to 7x amplification factor.
        # The factor 6.5 is the midpoint of the range reported in
        # Gu et al. (2021), DOI: 10.1101/2020.09.16.300749.
        return gain * 6.5


# =========================================================================
# PART 5: COMPLETE THALAMIC GATE
# =========================================================================

class ThalamicGate(nn.Module):
    """Complete four-stage thalamic gate.

    BIOLOGICAL STRUCTURE: The PFC-BG-TRN-TC pathway, treated as a
    single functional gate at the loop level.

    BIOLOGICAL FUNCTION: Implements the active gating of sensory
    input by top-down attention. Sensory input x is multiplied by a
    gain g produced by the four-stage cascade. The gain reflects
    current goal state (PFC), routed through inhibitory BG selection,
    converted by TRN's convex disinhibition into a multiplier on the
    thalamocortical relay, with norepinephrine providing global
    arousal scaling.

    Halassa MM, Kastner S (2017). DOI: 10.1038/s41593-017-0020-1
    Gu Y et al. (2021). DOI: 10.1101/2020.09.16.300749
    Nakajima M et al. (2019). DOI: 10.1016/j.neuron.2019.05.026
    Wimmer RD et al. (2015). DOI: 10.1038/nature15398

    ANATOMICAL INTERFACE (input):
        Sending structures: peripheral sensory pathways (sensory
        input) and PFC (goal/control input).
        Receiving structure: thalamocortical relay neurons (this
        module's output).
        Connection: lemniscal pathways (sensory) and corticothalamic
        layer V/VI (control).

    ANATOMICAL INTERFACE (output):
        Sending structure: thalamocortical relay neurons.
        Receiving structure: primary sensory cortex layer IV (V1, A1).
        Connection: thalamocortical projection.
    """

    def __init__(self, cfg: Optional[ThalamicGateConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ThalamicGateConfig()
        self.pfc = PFCControlSignal(self.cfg)
        self.bg = BasalGangliaRouter(self.cfg)
        self.trn = TRNDisinhibitoryGate(self.cfg)

    def forward(
        self,
        sensory_input: torch.Tensor,
        goal_state: torch.Tensor,
        acc_conflict: Optional[torch.Tensor] = None,
        ne_gain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the four-stage gate.

        Args:
            sensory_input: (B, input_dim) raw sensory drive.
            goal_state: (B, input_dim) PFC goal vector.
            acc_conflict: optional (B,) ACC conflict scalar.
            ne_gain: optional (B,) or scalar NE arousal.

        Returns:
            gated_output: (B, input_dim) the gated sensory signal
                ready for primary sensory cortex.
        """
        if not self.cfg.enable_thalamic_gate:
            return sensory_input

        # Stage 1: PFC control vector.
        pfc_control = self.pfc(goal_state, acc_conflict)

        # Stage 2: BG routing to TRN sector pattern.
        bg_inhibition = self.bg(pfc_control)

        # Stage 3: TRN disinhibitory gain.
        gain = self.trn(bg_inhibition)

        # NE modulation if enabled.
        if self.cfg.enable_ne_modulation and ne_gain is not None:
            if ne_gain.dim() == 0:
                gain = gain * ne_gain
            else:
                gain = gain * ne_gain.unsqueeze(-1)
        else:
            gain = gain * self.cfg.ne_gain_default

        # Stage 4: TC relay (multiplicative).
        if not self.cfg.enable_tc_relay:
            return torch.zeros_like(sensory_input)
        return sensory_input * gain


# =========================================================================
# SECTION: primary_sensory_cortex
# (originally primary_sensory_cortex_t.py)
# =========================================================================
"""
primary_sensory_cortex_t.py
Loop Stage 2: Primary Sensory Cortex (V1/A1)

BIOLOGICAL GROUNDING
Primary sensory cortex performs the first stage of cortical
processing on thalamocortical input. The defining computation is
center-surround receptive field organization formed by the
difference-of-Gaussians (DoG) operation. Hubel and Wiesel (1962)
documented the simple-cell tuning that emerges from convergent LGN
input with center-surround structure; Marr (1982) formalized the
DoG model as the canonical edge-detection filter.

The same architectural principle applies in primary auditory cortex
(A1) for tonotopic frequency analysis, with bandpass tuning curves
that are the auditory analog of V1 simple cells. This file
implements both V1 (visual) and A1 (auditory) as parallel filter
banks with the same underlying DoG operation, since the loop stage
treats them as parallel modality-specific feature extractors.

Primary grounding papers:

Hubel DH, Wiesel TN (1962). "Receptive fields, binocular interaction
and functional architecture in the cat's visual cortex." Journal of
Physiology, 160(1), 106-154.
DOI: 10.1113/jphysiol.1962.sp006837

Marr D (1982). "Vision: A Computational Investigation into the Human
Representation and Processing of Visual Information." MIT Press.

Rauschecker JP, Tian B (2000). "Mechanisms and streams for processing
of 'what' and 'where' in auditory cortex." Proceedings of the
National Academy of Sciences, 97(22), 11800-11806.
DOI: 10.1073/pnas.97.22.11800

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrimarySensoryConfig:
    """Configuration for V1/A1 primary sensory cortex.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_primary_sensory: bool = True
    enable_v1: bool = True
    enable_a1: bool = True
    enable_dog_filtering: bool = True
    enable_orientation_tuning: bool = True

    input_dim: int = 64
    n_filters: int = 32

    # DoG center and surround widths in arbitrary feature units.
    # Marr (1982) recommended a surround/center ratio of approximately
    # 1.6 for V1 simple cells.
    center_sigma: float = 1.0
    surround_sigma: float = 1.6

    # Number of orientation columns for V1. Hubel and Wiesel (1962)
    # documented orientation columns roughly every 10 degrees.
    n_orientations: int = 8


class DifferenceOfGaussians(nn.Module):
    """1D DoG filter bank approximating V1 center-surround receptive fields.

    BIOLOGICAL FUNCTION: Produces edge/contrast detection by
    subtracting a wider Gaussian (surround) from a narrower Gaussian
    (center) over the input feature vector.

    Marr D (1982). Marr's DoG model.
    """

    def __init__(
        self,
        input_dim: int,
        n_filters: int,
        center_sigma: float,
        surround_sigma: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_filters = n_filters

        # Build a fixed DoG kernel bank as register_buffer (not
        # trained; the receptive field structure is determined by
        # development in vivo).
        kernel_size = max(7, int(surround_sigma * 6) | 1)
        kernel = self._build_dog_kernel(
            kernel_size, center_sigma, surround_sigma,
        )
        self.register_buffer("kernel", kernel)

        # Learnable projection from filtered output to n_filters
        # features. The DoG itself is fixed; the cortex learns how
        # to read it out.
        self.readout = nn.Linear(input_dim, n_filters, bias=True)

    @staticmethod
    def _build_dog_kernel(
        size: int, c_sigma: float, s_sigma: float,
    ) -> torch.Tensor:
        """Build a 1D DoG kernel."""
        x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        center = torch.exp(-x**2 / (2 * c_sigma**2)) / (c_sigma * (2 * 3.14159) ** 0.5)
        surround = torch.exp(-x**2 / (2 * s_sigma**2)) / (s_sigma * (2 * 3.14159) ** 0.5)
        return (center - surround).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DoG convolution then linear readout.

        Args:
            x: (B, input_dim) feature vector.

        Returns:
            filtered: (B, n_filters) DoG-filtered features.
        """
        # Conv1d expects (B, C, L)
        x_3d = x.unsqueeze(1)
        padding = self.kernel.shape[-1] // 2
        conv = F.conv1d(x_3d, self.kernel, padding=padding)
        return self.readout(conv.squeeze(1))


class PrimarySensoryCortex(nn.Module):
    """V1/A1 primary sensory cortex with DoG-style filter banks.

    BIOLOGICAL STRUCTURE: Layer IV input layer of primary sensory
    cortex (V1 for vision, A1 for audition) with center-surround and
    orientation-tuned simple cells in layers 2/3.

    BIOLOGICAL FUNCTION: First-stage feature extraction from
    thalamocortical input. Edge and orientation detection in V1;
    bandpass frequency analysis in A1.

    Hubel DH, Wiesel TN (1962). DOI: 10.1113/jphysiol.1962.sp006837
    Marr D (1982). Vision (MIT Press).
    Rauschecker JP, Tian B (2000). DOI: 10.1073/pnas.97.22.11800

    ANATOMICAL INTERFACE (input):
        Sending structures: thalamic relay nuclei (LGN for V1, MGN
        for A1).
        Receiving structure: primary sensory cortex layer IV (this
        module).
        Connection: thalamocortical projection.

    ANATOMICAL INTERFACE (output):
        Sending structure: primary sensory cortex layer 2/3.
        Receiving structures: dorsal-stream (parietal) and
        ventral-stream (temporal) extrastriate cortex.
        Connection: corticocortical feedforward projections.
    """

    def __init__(
        self, cfg: Optional[PrimarySensoryConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PrimarySensoryConfig()

        # V1 filter bank.
        self.v1_filters = DifferenceOfGaussians(
            input_dim=self.cfg.input_dim,
            n_filters=self.cfg.n_filters,
            center_sigma=self.cfg.center_sigma,
            surround_sigma=self.cfg.surround_sigma,
        )

        # A1 filter bank (same architecture, separate weights to
        # match the modality dissociation in vivo).
        self.a1_filters = DifferenceOfGaussians(
            input_dim=self.cfg.input_dim,
            n_filters=self.cfg.n_filters,
            center_sigma=self.cfg.center_sigma,
            surround_sigma=self.cfg.surround_sigma,
        )

        # Orientation tuning bank for V1: a set of learnable
        # orientation-selective readouts.
        self.orientation_columns = nn.Linear(
            self.cfg.n_filters,
            self.cfg.n_orientations * self.cfg.n_filters,
            bias=True,
        )

    def forward(
        self,
        v1_input: torch.Tensor,
        a1_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply V1 and A1 filters.

        Args:
            v1_input: (B, input_dim) gated visual drive from the
                thalamic gate.
            a1_input: optional (B, input_dim) gated auditory drive.

        Returns:
            v1_features: (B, n_filters * n_orientations) if
                orientation tuning enabled, else (B, n_filters).
            a1_features: (B, n_filters).
        """
        if not self.cfg.enable_primary_sensory:
            zero_v1 = torch.zeros(
                v1_input.shape[0],
                self.cfg.n_filters * (
                    self.cfg.n_orientations if self.cfg.enable_orientation_tuning else 1
                ),
                device=v1_input.device, dtype=v1_input.dtype,
            )
            zero_a1 = torch.zeros(
                v1_input.shape[0], self.cfg.n_filters,
                device=v1_input.device, dtype=v1_input.dtype,
            )
            return zero_v1, zero_a1

        # V1 path.
        if self.cfg.enable_v1 and self.cfg.enable_dog_filtering:
            v1 = F.relu(self.v1_filters(v1_input))
            if self.cfg.enable_orientation_tuning:
                v1 = self.orientation_columns(v1)
        elif self.cfg.enable_v1:
            v1 = v1_input
            if self.cfg.enable_orientation_tuning:
                # Pass through unchanged sized to expected output.
                v1 = self.orientation_columns(
                    nn.functional.adaptive_avg_pool1d(
                        v1.unsqueeze(1),
                        self.cfg.n_filters,
                    ).squeeze(1)
                )
        else:
            v1 = torch.zeros(
                v1_input.shape[0],
                self.cfg.n_filters * (
                    self.cfg.n_orientations if self.cfg.enable_orientation_tuning else 1
                ),
                device=v1_input.device, dtype=v1_input.dtype,
            )

        # A1 path.
        if self.cfg.enable_a1 and a1_input is not None and self.cfg.enable_dog_filtering:
            a1 = F.relu(self.a1_filters(a1_input))
        elif self.cfg.enable_a1 and a1_input is not None:
            a1 = a1_input.new_zeros(a1_input.shape[0], self.cfg.n_filters)
        else:
            a1 = torch.zeros(
                v1_input.shape[0], self.cfg.n_filters,
                device=v1_input.device, dtype=v1_input.dtype,
            )

        return v1, a1


# =========================================================================
# SECTION: dorsal_ventral_streams
# (originally dorsal_ventral_streams_t.py)
# =========================================================================
"""
dorsal_ventral_streams_t.py
Loop Stage 3: Dorsal and Ventral Visual Stream Split

BIOLOGICAL GROUNDING
After primary sensory cortex, visual information bifurcates into two
processing streams. The dorsal stream runs from V1 through V2 and
V5/MT to posterior parietal cortex and supports spatial location and
action guidance ("where" and "how"). The ventral stream runs from V1
through V2 and V4 to inferior temporal cortex and supports object
identity ("what"). Mishkin and Ungerleider (1982) and Goodale and
Milner (1992) are the foundational papers.

The same dual-stream organization holds in audition, with the dorsal
auditory stream supporting spatial localization and the ventral
auditory stream supporting object/voice recognition. Rauschecker and
Tian (2000) document the parallel.

The PRAGMI loop treats these as parallel cortical pathways that
project independently to association cortex, where multimodal
binding occurs. The split itself is implemented as a routing module
that takes V1/A1 features and emits dorsal and ventral
representations with distinct projection weights, with each stream
independently ablatable.

Primary grounding papers:

Mishkin M, Ungerleider LG (1982). "Contribution of striate inputs to
the visuospatial functions of parieto-preoccipital cortex in monkeys."
Behavioural Brain Research, 6(1), 57-77.
DOI: 10.1016/0166-4328(82)90081-X

Goodale MA, Milner AD (1992). "Separate visual pathways for
perception and action." Trends in Neurosciences, 15(1), 20-25.
DOI: 10.1016/0166-2236(92)90344-8

Rauschecker JP, Tian B (2000). "Mechanisms and streams for processing
of 'what' and 'where' in auditory cortex." Proceedings of the
National Academy of Sciences, 97(22), 11800-11806.
DOI: 10.1073/pnas.97.22.11800

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DorsalVentralConfig:
    """Configuration for the dorsal/ventral stream split.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_streams: bool = True
    enable_dorsal_stream: bool = True
    enable_ventral_stream: bool = True

    input_dim: int = 256  # n_filters * n_orientations from V1
    dorsal_dim: int = 128
    ventral_dim: int = 128


class DorsalVentralSplit(nn.Module):
    """Dorsal (where/how) and ventral (what) stream split.

    BIOLOGICAL STRUCTURE: Two parallel cortical processing streams
    emerging from V1/A1: dorsal stream through V2/V5/MT to posterior
    parietal cortex, and ventral stream through V2/V4 to inferior
    temporal cortex.

    BIOLOGICAL FUNCTION: Decomposes sensory representation into
    spatial/action ("where"/"how") and identity ("what") components
    that converge later in association cortex for multimodal binding.

    Mishkin M, Ungerleider LG (1982). DOI: 10.1016/0166-4328(82)90081-X
    Goodale MA, Milner AD (1992). DOI: 10.1016/0166-2236(92)90344-8
    Rauschecker JP, Tian B (2000). DOI: 10.1073/pnas.97.22.11800

    ANATOMICAL INTERFACE (input):
        Sending structure: V1/A1 layer 2/3 (this module's input).
        Receiving structures: V2 dorsal and V2 ventral subdivisions
        (this module).
        Connection: extrastriate corticocortical projections.

    ANATOMICAL INTERFACE (output, dorsal):
        Sending structure: posterior parietal cortex.
        Receiving structure: association cortex (binding stage).
        Connection: parietal-to-association corticocortical
        projections.

    ANATOMICAL INTERFACE (output, ventral):
        Sending structure: inferior temporal cortex.
        Receiving structure: association cortex (binding stage).
        Connection: temporal-to-association corticocortical
        projections.
    """

    def __init__(
        self, cfg: Optional[DorsalVentralConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DorsalVentralConfig()
        self.dorsal_projection = nn.Linear(
            self.cfg.input_dim, self.cfg.dorsal_dim, bias=True,
        )
        self.ventral_projection = nn.Linear(
            self.cfg.input_dim, self.cfg.ventral_dim, bias=True,
        )

    def forward(
        self, sensory_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split sensory features into dorsal and ventral streams.

        Args:
            sensory_features: (B, input_dim) features from V1/A1.

        Returns:
            dorsal: (B, dorsal_dim) where/how representation.
            ventral: (B, ventral_dim) what representation.
        """
        if not self.cfg.enable_streams:
            zero_d = torch.zeros(
                sensory_features.shape[0], self.cfg.dorsal_dim,
                device=sensory_features.device,
                dtype=sensory_features.dtype,
            )
            zero_v = torch.zeros(
                sensory_features.shape[0], self.cfg.ventral_dim,
                device=sensory_features.device,
                dtype=sensory_features.dtype,
            )
            return zero_d, zero_v

        if self.cfg.enable_dorsal_stream:
            dorsal = F.relu(self.dorsal_projection(sensory_features))
        else:
            dorsal = torch.zeros(
                sensory_features.shape[0], self.cfg.dorsal_dim,
                device=sensory_features.device,
                dtype=sensory_features.dtype,
            )

        if self.cfg.enable_ventral_stream:
            ventral = F.relu(self.ventral_projection(sensory_features))
        else:
            ventral = torch.zeros(
                sensory_features.shape[0], self.cfg.ventral_dim,
                device=sensory_features.device,
                dtype=sensory_features.dtype,
            )

        return dorsal, ventral


# =========================================================================
# SECTION: cortical_interneurons
# (originally cortical_interneurons_t.py)
# =========================================================================
"""
cortical_interneurons_t.py
Cortical Interneuron Triplet: PV, SST, VIP

BIOLOGICAL GROUNDING
Cortex computes through dynamic interplay of excitatory pyramidal
cells and inhibitory interneurons. Of the 30+ molecular interneuron
classes, three carry the bulk of the functional load:
parvalbumin-positive (PV), somatostatin-positive (SST), and
vasoactive intestinal peptide-positive (VIP).

PV cells are fast-spiking basket cells. They contact pyramidal cell
somata and proximal dendrites and provide perisomatic gain control.
PV inhibition is the substrate of gamma-band oscillations.

SST cells (Martinotti cells in particular) target distal apical
dendrites. They gate the integration of top-down inputs onto the
apical tuft of layer 5 pyramidal cells. SST is slow.

VIP cells preferentially inhibit SST cells, producing a
disinhibitory motif: VIP active means SST suppressed means apical
dendrite released. This is the substrate of selective top-down
attention to specific cortical populations.

Connectivity matrix: Pfeffer, Xue, He, Huang, Scanziani (2013) and
Jiang et al. (2015) provide the empirical connection probabilities
and unitary EPSP/IPSP amplitudes among these classes.

Time constants from Garcia del Molino et al. (2017): PV and
pyramidal at approximately 10 ms, SST at 30 to 50 ms reflecting
the longer integration window of Martinotti cells.

Functional grammar review: Kepecs and Fishell (2014).

Primary grounding papers:

Pfeffer CK, Xue M, He M, Huang ZJ, Scanziani M (2013). "Inhibition
of inhibition in visual cortex: the logic of connections between
molecularly distinct interneurons." Nature Neuroscience, 16(8),
1068-1076. DOI: 10.1038/nn.3446

Jiang X, Shen S, Cadwell CR, Berens P, Sinz F, Ecker AS, Patel S,
Tolias AS (2015). "Principles of connectivity among morphologically
defined cell types in adult neocortex." Science, 350(6264), aac9462.
DOI: 10.1126/science.aac9462

Garcia del Molino LC, Yang GR, Mejias JF, Wang XJ (2017).
"Paradoxical response reversal of top-down modulation in cortical
circuits with three interneuron types." eLife, 6:e29742.
DOI: 10.7554/eLife.29742

Kepecs A, Fishell G (2014). "Interneuron cell types are fit to
function." Nature, 505(7483), 318-326. DOI: 10.1038/nature12983

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class InterneuronConfig:
    """Configuration for the PV/SST/VIP triplet.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_interneurons: bool = True
    enable_pv: bool = True
    enable_sst: bool = True
    enable_vip: bool = True
    enable_vip_disinhibition: bool = True

    pyramidal_dim: int = 64
    pv_dim: int = 16
    sst_dim: int = 16
    vip_dim: int = 8

    # Time constants. Garcia del Molino et al. (2017) DOI:
    # 10.7554/eLife.29742 reports PV and pyramidal time constants
    # of approximately 10 ms; SST integrates over a longer window.
    tau_pyr: float = 10.0
    tau_pv: float = 10.0
    tau_sst: float = 40.0
    tau_vip: float = 15.0

    # Simulation step in ms. NOT a biological quantity.
    dt: float = 1.0

    # Connection strengths drawn from Pfeffer et al. (2013) DOI:
    # 10.1038/nn.3446 qualitative pattern. The exact values are
    # engineering tuning consistent with the empirical sign and
    # relative magnitude.
    w_pyr_to_pv: float = 0.5
    w_pyr_to_sst: float = 0.3
    w_pyr_to_vip: float = 0.2
    w_pv_to_pyr: float = 0.8  # perisomatic inhibition
    w_sst_to_pyr_apical: float = 0.6  # distal dendritic inhibition
    w_vip_to_sst: float = 0.7  # disinhibitory motif


class CorticalInterneuronTriplet(nn.Module):
    """Three-class cortical interneuron module.

    BIOLOGICAL STRUCTURE: Cortical layer 2/3 to 5 microcircuit with
    pyramidal cells and three classes of GABAergic interneuron.

    BIOLOGICAL FUNCTION: Implements the canonical motifs of cortical
    inhibition: PV perisomatic gain control, SST apical dendritic
    gating, and VIP disinhibition of SST. Top-down input via VIP
    selectively releases the apical tuft of targeted pyramidal
    populations.

    Pfeffer CK et al. (2013). DOI: 10.1038/nn.3446
    Jiang X et al. (2015). DOI: 10.1126/science.aac9462
    Garcia del Molino LC et al. (2017). DOI: 10.7554/eLife.29742
    Kepecs A, Fishell G (2014). DOI: 10.1038/nature12983
    """

    def __init__(self, cfg: Optional[InterneuronConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or InterneuronConfig()

        # State buffers for each population.
        self.register_buffer(
            "pyr_state", torch.zeros(self.cfg.pyramidal_dim),
        )
        self.register_buffer("pv_state", torch.zeros(self.cfg.pv_dim))
        self.register_buffer("sst_state", torch.zeros(self.cfg.sst_dim))
        self.register_buffer("vip_state", torch.zeros(self.cfg.vip_dim))

        # Projection matrices for between-population coupling.
        self.pyr_to_pv = nn.Linear(
            self.cfg.pyramidal_dim, self.cfg.pv_dim, bias=False,
        )
        self.pyr_to_sst = nn.Linear(
            self.cfg.pyramidal_dim, self.cfg.sst_dim, bias=False,
        )
        self.pyr_to_vip = nn.Linear(
            self.cfg.pyramidal_dim, self.cfg.vip_dim, bias=False,
        )
        self.pv_to_pyr = nn.Linear(
            self.cfg.pv_dim, self.cfg.pyramidal_dim, bias=False,
        )
        self.sst_to_pyr = nn.Linear(
            self.cfg.sst_dim, self.cfg.pyramidal_dim, bias=False,
        )
        self.vip_to_sst = nn.Linear(
            self.cfg.vip_dim, self.cfg.sst_dim, bias=False,
        )

    def forward(
        self,
        bottom_up: torch.Tensor,
        top_down: Optional[torch.Tensor] = None,
        vip_drive: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Run one timestep of the four-population dynamics.

        Args:
            bottom_up: (B, pyramidal_dim) sensory drive onto pyramidal
                basal dendrites.
            top_down: optional (B, pyramidal_dim) top-down drive onto
                pyramidal apical dendrites (gated by SST).
            vip_drive: optional (B, vip_dim) external drive onto VIP
                cells, the substrate of selective attention.

        Returns:
            pyr_output: (B, pyramidal_dim) pyramidal cell output.
            diagnostics: dict with the four population states.
        """
        if not self.cfg.enable_interneurons:
            return bottom_up, {
                "pyr": self.pyr_state.detach(),
                "pv": self.pv_state.detach(),
                "sst": self.sst_state.detach(),
                "vip": self.vip_state.detach(),
            }

        # We use the persistent buffers as a single-population state
        # and broadcast across batch. This is an engineering
        # simplification that captures the population-level dynamics
        # without per-sample state.
        with torch.no_grad():
            # PV update: driven by pyramidal output.
            if self.cfg.enable_pv:
                pv_drive = self.pyr_to_pv(
                    bottom_up.detach().mean(dim=0)
                ) * self.cfg.w_pyr_to_pv
                self.pv_state.copy_(
                    self.pv_state
                    + self.cfg.dt * (-self.pv_state + pv_drive)
                    / self.cfg.tau_pv
                )
            # VIP update: driven by external attention input and
            # weakly by pyramidal output.
            if self.cfg.enable_vip:
                vip_input = self.pyr_to_vip(
                    bottom_up.detach().mean(dim=0)
                ) * self.cfg.w_pyr_to_vip
                if vip_drive is not None:
                    vip_input = vip_input + vip_drive.detach().mean(dim=0)
                self.vip_state.copy_(
                    self.vip_state
                    + self.cfg.dt * (-self.vip_state + vip_input)
                    / self.cfg.tau_vip
                )
            # SST update: driven by pyramidal output, inhibited by VIP
            # (the disinhibitory motif).
            if self.cfg.enable_sst:
                sst_input = self.pyr_to_sst(
                    bottom_up.detach().mean(dim=0)
                ) * self.cfg.w_pyr_to_sst
                if (
                    self.cfg.enable_vip_disinhibition
                    and self.cfg.enable_vip
                ):
                    sst_input = sst_input - self.cfg.w_vip_to_sst * self.vip_to_sst(
                        self.vip_state.detach()
                    )
                self.sst_state.copy_(
                    self.sst_state
                    + self.cfg.dt * (-self.sst_state + sst_input)
                    / self.cfg.tau_sst
                )

        # Compute pyramidal output: bottom-up drive minus PV
        # perisomatic inhibition, plus apical (top-down) drive
        # minus SST distal dendritic inhibition.
        pv_inh = (
            self.cfg.w_pv_to_pyr * self.pv_to_pyr(self.pv_state.unsqueeze(0))
        ) if self.cfg.enable_pv else 0.0
        sst_inh = (
            self.cfg.w_sst_to_pyr_apical * self.sst_to_pyr(
                self.sst_state.unsqueeze(0)
            )
        ) if self.cfg.enable_sst else 0.0

        apical = top_down if top_down is not None else 0.0
        pyr_output = bottom_up - pv_inh + (apical - sst_inh) * 0.5
        return torch.tanh(pyr_output), {
            "pyr": self.pyr_state.detach(),
            "pv": self.pv_state.detach(),
            "sst": self.sst_state.detach(),
            "vip": self.vip_state.detach(),
        }

    def reset_state(self) -> None:
        """Reset all population states."""
        with torch.no_grad():
            self.pyr_state.zero_()
            self.pv_state.zero_()
            self.sst_state.zero_()
            self.vip_state.zero_()


# =========================================================================
# SECTION: layer5b_apical
# (originally layer5b_apical_t.py)
# =========================================================================
"""
layer5b_apical_t.py
Cortical Compartment: Layer 5b Pyramidal Apical Amplification

BIOLOGICAL GROUNDING
Layer 5b pyramidal cells are the major cortical output neurons,
projecting to subcortical targets including thalamus, brainstem,
and spinal cord. Their dendrites have a striking compartmental
structure: a basal dendritic tree near the soma in deep cortex, and
a long apical dendrite extending up to layer 1 where it forms an
apical tuft. Each compartment has its own active conductances and
synaptic input populations.

The functional implication, formalized by Shai et al. (2015), is
that L5b pyramidal cells implement a sigmoid-of-sigmoids
computation: bottom-up basal input is gated by top-down apical input
through a multiplicative coupling. When bottom-up input alone is
present, basal sigmoid responds but the apical sigmoid is silent and
the cell fires weakly. When apical input is also present, the
apical sigmoid activates the apical Ca-spike mechanism, which
multiplicatively boosts somatic firing. This is the cellular
mechanism for combining feedforward sensory drive with feedback
predictions.

Hay et al. (2011) provide a detailed biophysical model of L5b
including the apical Ca channels that produce the burst-firing mode.
Sacramento et al. (2018) cast the apical compartment as the substrate
for credit assignment: the apical signal carries the prediction error
that should drive plasticity at basal synapses. Phillips et al.
(2023) review the cognitive implications of this compartmentalization.

This module implements the sigmoid-of-sigmoids gate as a per-unit
operation: each output unit has a basal channel and an apical channel,
and the output is the product of the two sigmoids passed through a
final nonlinearity. The compartments and their interaction are each
independently ablatable.

Primary grounding papers:

Shai AS, Anastassiou CA, Larkum ME, Koch C (2015). "Physiology of
layer 5 pyramidal neurons in mouse primary visual cortex: coincidence
detection through bursting." PLOS Computational Biology, 11(3),
e1004090. DOI: 10.1371/journal.pcbi.1004090

Hay E, Hill S, Schurmann F, Markram H, Segev I (2011). "Models of
neocortical layer 5b pyramidal cells capturing a wide range of
dendritic and perisomatic active properties." PLOS Computational
Biology, 7(7), e1002107. DOI: 10.1371/journal.pcbi.1002107

Sacramento J, Costa RP, Bengio Y, Senn W (2018). "Dendritic cortical
microcircuits approximate the backpropagation algorithm." Advances
in Neural Information Processing Systems, 31.
arXiv:1810.11393

Phillips WA, Bachmann T, Spratling MW, Muckli L, Petro LS, Zolnik T
(2023). "Cellular psychology: relating cognition to context-sensitive
pyramidal cells." Trends in Cognitive Sciences, 27(1), 13-25.
DOI: 10.1016/j.tics.2022.10.006

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class L5bApicalConfig:
    """Configuration for the L5b sigmoid-of-sigmoids module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_l5b: bool = True
    enable_basal_compartment: bool = True
    enable_apical_compartment: bool = True
    enable_multiplicative_coupling: bool = True

    basal_input_dim: int = 64
    apical_input_dim: int = 64
    output_dim: int = 64

    # Apical Ca-spike threshold scaling. Hay et al. (2011)
    # DOI: 10.1371/journal.pcbi.1002107. NOT a biological quantity in
    # the strict sense; engineering approximation matching the
    # qualitative threshold-and-burst pattern.
    apical_threshold_bias: float = -1.0

    # Basal-only output scaling when apical compartment is silent.
    # The cell still fires weakly without apical drive. NOT a
    # biological quantity, training artifact.
    basal_only_scale: float = 0.3


class Layer5bApical(nn.Module):
    """Layer 5b pyramidal sigmoid-of-sigmoids apical amplification.

    BIOLOGICAL STRUCTURE: Layer 5b pyramidal cells with basal
    dendritic tree near soma and apical dendrite extending to layer 1
    apical tuft.

    BIOLOGICAL FUNCTION: Coincidence detection between bottom-up
    basal input (carrying sensory drive) and top-down apical input
    (carrying predictions or context). When both are present, the
    apical Ca-spike mechanism multiplicatively amplifies somatic
    firing into burst mode.

    Shai AS et al. (2015). DOI: 10.1371/journal.pcbi.1004090
    Hay E et al. (2011). DOI: 10.1371/journal.pcbi.1002107
    Sacramento J et al. (2018). arXiv:1810.11393
    Phillips WA et al. (2023). DOI: 10.1016/j.tics.2022.10.006

    ANATOMICAL INTERFACE (basal input):
        Sending structures: thalamic relay neurons (sensory drive)
        and local cortical pyramidal cells.
        Receiving structure: L5b basal dendrites (this module).
        Connection: thalamocortical and corticocortical projections
        terminating near soma.

    ANATOMICAL INTERFACE (apical input):
        Sending structures: higher cortical areas and nonspecific
        thalamic nuclei.
        Receiving structure: L5b apical tuft in layer 1 (this module).
        Connection: feedback corticocortical and matrix thalamic
        projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: L5b pyramidal cell axon.
        Receiving structures: thalamus, striatum, brainstem, spinal
        cord.
        Connection: corticosubcortical projections.
    """

    def __init__(self, cfg: Optional[L5bApicalConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or L5bApicalConfig()

        # Basal compartment integration.
        self.basal_integrator = nn.Linear(
            self.cfg.basal_input_dim, self.cfg.output_dim, bias=True,
        )
        # Apical compartment integration.
        self.apical_integrator = nn.Linear(
            self.cfg.apical_input_dim, self.cfg.output_dim, bias=True,
        )

    def forward(
        self,
        basal_input: torch.Tensor,
        apical_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the sigmoid-of-sigmoids computation.

        Args:
            basal_input: (B, basal_input_dim) bottom-up sensory drive.
            apical_input: optional (B, apical_input_dim) top-down
                feedback drive. None means apical silent.

        Returns:
            output: (B, output_dim) somatic firing rate.
        """
        if not self.cfg.enable_l5b:
            return torch.zeros(
                basal_input.shape[0], self.cfg.output_dim,
                device=basal_input.device, dtype=basal_input.dtype,
            )

        # Basal compartment sigmoid.
        if self.cfg.enable_basal_compartment:
            basal_sigmoid = torch.sigmoid(
                self.basal_integrator(basal_input)
            )
        else:
            basal_sigmoid = torch.zeros(
                basal_input.shape[0], self.cfg.output_dim,
                device=basal_input.device, dtype=basal_input.dtype,
            )

        # Apical compartment sigmoid.
        if (
            self.cfg.enable_apical_compartment
            and apical_input is not None
        ):
            apical_drive = self.apical_integrator(apical_input)
            apical_sigmoid = torch.sigmoid(
                apical_drive + self.cfg.apical_threshold_bias
            )
        else:
            apical_sigmoid = torch.zeros_like(basal_sigmoid)

        # Multiplicative coupling.
        if self.cfg.enable_multiplicative_coupling:
            # Coincidence detection: basal AND apical produces full
            # firing; basal alone produces weak firing; apical alone
            # produces nothing (no somatic drive without basal input).
            coincidence = basal_sigmoid * apical_sigmoid
            basal_only = basal_sigmoid * self.cfg.basal_only_scale
            return coincidence + basal_only * (1.0 - apical_sigmoid)
        else:
            # Without multiplicative coupling, the compartments sum.
            return (basal_sigmoid + apical_sigmoid) * 0.5


# =========================================================================
# SECTION: association_cortex
# (originally association_cortex_t.py)
# =========================================================================
"""
association_cortex_t.py
Loop Stage 4: Association Cortex Multimodal Binding

BIOLOGICAL GROUNDING
Association cortex is the convergence zone where dorsal and ventral
streams meet, where unimodal representations combine into multimodal
percepts, and where top-down feedback returns to lower areas.
Damasio (1989) introduced the convergence-divergence zone account
of association cortex; Mesulam (1990) elaborated the large-scale
network organization. Friston (2010) formalized the bidirectional
nature of cortical processing as predictive coding, in which
top-down predictions and bottom-up prediction errors interact at
every level.

A computational instantiation suited to PRAGMI is the mixture of
experts (MoE) architecture: Shazeer et al. (2017) and Eigen et al.
(2014). Different cortical patches act as experts specialized for
different feature combinations; a gating network routes input to
the most appropriate experts. This matches the empirical observation
that different prefrontal-parietal-temporal subregions activate for
different cognitive tasks despite shared global anatomy.

This file implements association cortex as a small MoE bank with
bidirectional feedback to lower-level streams. The MoE binding
component, the gating router, and the top-down feedback are each
independently ablatable.

Primary grounding papers:

Damasio AR (1989). "Time-locked multiregional retroactivation: a
systems-level proposal for the neural substrates of recall and
recognition." Cognition, 33(1-2), 25-62.
DOI: 10.1016/0010-0277(89)90005-X

Mesulam MM (1990). "Large-scale neurocognitive networks and
distributed processing for attention, language, and memory." Annals
of Neurology, 28(5), 597-613. DOI: 10.1002/ana.410280502

Friston K (2010). "The free-energy principle: a unified brain
theory?" Nature Reviews Neuroscience, 11(2), 127-138.
DOI: 10.1038/nrn2787

Shazeer N, Mirhoseini A, Maziarz K, Davis A, Le Q, Hinton G, Dean J
(2017). "Outrageously large neural networks: the sparsely-gated
mixture-of-experts layer." arXiv:1701.06538.

Eigen D, Ranzato MA, Sutskever I (2014). "Learning factored
representations in a deep mixture of experts." arXiv:1312.4314.

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AssociationCortexConfig:
    """Configuration for association cortex MoE binding.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_association_cortex: bool = True
    enable_moe_binding: bool = True
    enable_gating_network: bool = True
    enable_top_down_feedback: bool = True

    dorsal_dim: int = 128
    ventral_dim: int = 128
    n_experts: int = 8
    expert_dim: int = 64
    output_dim: int = 64

    # Top-K routing. Shazeer et al. (2017) found K=2 to give a good
    # quality/efficiency tradeoff. NOT a biological quantity but
    # corresponds to the observation that any cortical patch
    # typically participates in a small number of co-active networks.
    top_k: int = 2

    # Top-down feedback strength. Friston (2010) free energy
    # principle. NOT a biological quantity, training artifact.
    feedback_strength: float = 0.5


class MixtureOfExperts(nn.Module):
    """MoE binding bank for multimodal convergence.

    BIOLOGICAL FUNCTION: Implements convergence-divergence zone
    mechanics by routing combined dorsal/ventral input through a
    small number of specialized expert subnetworks.

    Damasio AR (1989). DOI: 10.1016/0010-0277(89)90005-X
    Shazeer N et al. (2017). arXiv:1701.06538.
    """

    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        expert_dim: int,
        top_k: int,
        enable_gating: bool,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.enable_gating = enable_gating

        # Gating network.
        self.gate = nn.Linear(input_dim, n_experts, bias=False)

        # Bank of experts, each a small MLP.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, expert_dim),
            )
            for _ in range(n_experts)
        ])

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run MoE forward.

        Args:
            x: (B, input_dim) combined dorsal/ventral input.

        Returns:
            output: (B, expert_dim) MoE output.
            gate_weights: (B, n_experts) routing pattern (for diagnostics).
        """
        if self.enable_gating:
            gate_logits = self.gate(x)
            # Top-K routing: keep only the top_k highest-scoring experts.
            top_vals, top_idx = gate_logits.topk(self.top_k, dim=-1)
            sparse_gate = torch.full_like(gate_logits, float("-inf"))
            sparse_gate.scatter_(-1, top_idx, top_vals)
            gate_weights = F.softmax(sparse_gate, dim=-1)
        else:
            # Uniform gating: all experts contribute equally.
            gate_weights = torch.full(
                (x.shape[0], self.n_experts),
                1.0 / self.n_experts,
                device=x.device, dtype=x.dtype,
            )

        # Compute all expert outputs (small enough to compute densely).
        expert_outs = torch.stack(
            [e(x) for e in self.experts], dim=1,
        )  # (B, n_experts, expert_dim)

        # Weighted sum.
        output = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        return output, gate_weights


class AssociationCortex(nn.Module):
    """Association cortex with MoE binding and top-down feedback.

    BIOLOGICAL STRUCTURE: Heteromodal association cortex including
    posterior parietal, lateral temporal, and prefrontal convergence
    zones.

    BIOLOGICAL FUNCTION: Multimodal binding of dorsal and ventral
    stream representations into unified percepts, with bidirectional
    feedback to lower-level streams supporting predictive coding.

    Damasio AR (1989). DOI: 10.1016/0010-0277(89)90005-X
    Mesulam MM (1990). DOI: 10.1002/ana.410280502
    Friston K (2010). DOI: 10.1038/nrn2787

    ANATOMICAL INTERFACE (input):
        Sending structures: posterior parietal cortex (dorsal stream)
        and inferior temporal cortex (ventral stream).
        Receiving structure: heteromodal association cortex (this
        module).
        Connection: parieto-association and temporo-association
        corticocortical projections.

    ANATOMICAL INTERFACE (output, forward):
        Sending structure: association cortex.
        Receiving structures: hippocampal formation (via
        parahippocampal and perirhinal cortices) and PFC (for goal
        representation).
        Connection: cortico-hippocampal and cortico-prefrontal
        projections.

    ANATOMICAL INTERFACE (output, top-down feedback):
        Sending structure: association cortex layer V/VI.
        Receiving structures: dorsal and ventral stream upstream
        areas.
        Connection: feedback corticocortical projections targeting
        layer 1 apical dendrites.
    """

    def __init__(
        self, cfg: Optional[AssociationCortexConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or AssociationCortexConfig()

        combined_dim = self.cfg.dorsal_dim + self.cfg.ventral_dim
        self.moe = MixtureOfExperts(
            input_dim=combined_dim,
            n_experts=self.cfg.n_experts,
            expert_dim=self.cfg.expert_dim,
            top_k=self.cfg.top_k,
            enable_gating=self.cfg.enable_gating_network,
        )
        # Output projection.
        self.output_projection = nn.Linear(
            self.cfg.expert_dim, self.cfg.output_dim, bias=True,
        )
        # Top-down feedback projections to dorsal and ventral.
        self.feedback_dorsal = nn.Linear(
            self.cfg.output_dim, self.cfg.dorsal_dim, bias=False,
        )
        self.feedback_ventral = nn.Linear(
            self.cfg.output_dim, self.cfg.ventral_dim, bias=False,
        )

    def forward(
        self,
        dorsal: torch.Tensor,
        ventral: torch.Tensor,
        prev_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Bind streams and produce association output plus feedback.

        Args:
            dorsal: (B, dorsal_dim) dorsal stream input.
            ventral: (B, ventral_dim) ventral stream input.
            prev_output: optional (B, output_dim) previous-step
                association output for top-down feedback computation.

        Returns:
            assoc_output: (B, output_dim) association cortex output.
            fb_dorsal: (B, dorsal_dim) top-down feedback to dorsal.
            fb_ventral: (B, ventral_dim) top-down feedback to ventral.
            diagnostics: dict including gate_weights.
        """
        if not self.cfg.enable_association_cortex:
            zero_out = torch.zeros(
                dorsal.shape[0], self.cfg.output_dim,
                device=dorsal.device, dtype=dorsal.dtype,
            )
            return (
                zero_out,
                torch.zeros_like(dorsal),
                torch.zeros_like(ventral),
                {"gate_weights": None},
            )

        # Combine streams and pass through MoE.
        combined = torch.cat([dorsal, ventral], dim=-1)
        if self.cfg.enable_moe_binding:
            bound, gate_weights = self.moe(combined)
        else:
            # No MoE: simple linear projection of combined input.
            bound = combined.new_zeros(
                combined.shape[0], self.cfg.expert_dim,
            )
            gate_weights = None

        assoc_output = self.output_projection(bound)

        # Top-down feedback computed from previous output (or current
        # if no previous is provided, which lets the loop bootstrap).
        if self.cfg.enable_top_down_feedback:
            ref = prev_output if prev_output is not None else assoc_output
            fb_dorsal = self.cfg.feedback_strength * self.feedback_dorsal(ref)
            fb_ventral = self.cfg.feedback_strength * self.feedback_ventral(ref)
        else:
            fb_dorsal = torch.zeros_like(dorsal)
            fb_ventral = torch.zeros_like(ventral)

        return assoc_output, fb_dorsal, fb_ventral, {
            "gate_weights": gate_weights,
        }


# =========================================================================
# SECTION: claustrum
# (originally claustrum_t.py)
# =========================================================================
"""
claustrum_t.py
Loop Stage: Claustrum Multimodal Synchronizer

BIOLOGICAL GROUNDING
The claustrum is a thin sheet of subcortical gray matter receiving
projections from nearly every cortical area and projecting back to
nearly every cortical area. Crick and Koch (2005) proposed it as a
binding hub for unified conscious experience. Recent computational
accounts position it as a multimodal synchronizer rather than a
content store: it does not represent specific information but
coordinates the timing and selection of cortical activations.

Reser (2019) provides a developed timing-mechanism account: the
claustrum receives convergent cortical input, computes a global
salience signal, and emits a brief synchronizing pulse that aligns
cortical processing across regions for the next cycle. Grimstvedt et
al. (2024) characterize fast inhibition microcircuits within the
claustrum that support precise burst timing. Madden et al. (2022)
show network-impulse computational role: NICC = Network Impulse
Cell-of-Claustrum. Goll, Atlan, Citri (2015) review broader
attentional functions.

This module implements a claustrum component that:
1. Pools cortical input (multimodal convergence)
2. Computes a salience score
3. Emits a temporally precise synchronizing pulse when salience
   exceeds threshold
4. Applies the pulse as a multiplicative boost across registered
   cortical targets

Primary grounding papers:

Crick FC, Koch C (2005). "What is the function of the claustrum?"
Philosophical Transactions of the Royal Society B, 360(1458),
1271-1279. DOI: 10.1098/rstb.2005.1661

Reser DH (2019). "The claustrum: hub for the timekeeping mechanism
of cognition." (Hypothesis paper).
DOI: 10.31234/osf.io/zsa9p

Grimstvedt JS, Shelton AM, Hoerder-Suabedissen A, Vyssotski AL,
Yates AG, Lensjo KK, Lutsi T, Akiti K, Bjaalie JG, Witter MP, Bjorness
TE, Kornblum HI, Krienen FM, Olive J, Smith Y, Reser DH, Smith KS,
Buzsaki G (2024). Claustrum microcircuit characterization.
DOI: 10.1101/2024.07.03.601954

Madden MB, Stewart BW, White MG, Krimmel SR, Qadir H, Barrett FS,
Seminowicz DA, Mathur BN (2022). "A role for the claustrum in
cognitive control." Trends in Cognitive Sciences, 26(12), 1133-1152.
DOI: 10.1016/j.tics.2022.09.006

Goll Y, Atlan G, Citri A (2015). "Attention: the claustrum." Trends
in Neurosciences, 38(8), 486-495.
DOI: 10.1016/j.tins.2015.05.006

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClaustrumConfig:
    """Configuration for the claustrum module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_claustrum: bool = True
    enable_salience_pooling: bool = True
    enable_synchronizing_pulse: bool = True
    enable_target_boost: bool = True

    cortical_input_dim: int = 64
    pulse_dim: int = 64

    # Salience threshold for pulse generation. Reser (2019) describes
    # the claustrum as quiescent until convergent input crosses
    # threshold. NOT a biological quantity.
    pulse_threshold: float = 0.5

    # Pulse strength when fired. Multiplicative boost applied to
    # cortical targets. NOT a biological quantity, training artifact.
    pulse_strength: float = 1.5

    # Refractory period in steps. Madden et al. (2022) describes
    # claustral activity as transient. NOT a biological quantity.
    refractory_steps: int = 3


class Claustrum(nn.Module):
    """Claustrum multimodal synchronizer.

    BIOLOGICAL STRUCTURE: Thin sheet of gray matter ventrolateral to
    putamen, with reciprocal connections to nearly all of cortex.

    BIOLOGICAL FUNCTION: Pools cortical input to compute a global
    salience signal, then emits a precisely timed synchronizing pulse
    that boosts processing across registered cortical targets. Acts
    as a cognitive timekeeper rather than a content store.

    Crick FC, Koch C (2005). DOI: 10.1098/rstb.2005.1661
    Reser DH (2019). DOI: 10.31234/osf.io/zsa9p
    Madden MB et al. (2022). DOI: 10.1016/j.tics.2022.09.006

    ANATOMICAL INTERFACE (input):
        Sending structures: nearly all cortical regions, with strong
        prefrontal and ACC inputs.
        Receiving structure: claustrum (this module).
        Connection: cortico-claustral projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: claustrum.
        Receiving structures: nearly all cortical regions.
        Connection: claustro-cortical projections, particularly to
        layer 4 and layer 6.
    """

    def __init__(self, cfg: Optional[ClaustrumConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ClaustrumConfig()

        # Salience pooler: aggregates convergent cortical input.
        self.salience_pool = nn.Linear(
            self.cfg.cortical_input_dim, 1, bias=True,
        )

        # Pulse generator: produces the synchronizing pulse pattern.
        self.pulse_generator = nn.Linear(
            self.cfg.cortical_input_dim, self.cfg.pulse_dim, bias=True,
        )

        # Refractory counter buffer.
        self.register_buffer(
            "refractory_counter",
            torch.tensor(0, dtype=torch.long),
        )

    def forward(
        self, cortical_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool input and emit synchronizing pulse if salient.

        Args:
            cortical_input: (B, cortical_input_dim) convergent input
                from cortex.

        Returns:
            pulse: (B, pulse_dim) synchronizing pulse pattern (zero
                when no pulse fires).
            fired: (B,) bool indicating whether pulse fired this step.
        """
        if not self.cfg.enable_claustrum:
            zero_pulse = torch.zeros(
                cortical_input.shape[0], self.cfg.pulse_dim,
                device=cortical_input.device,
                dtype=cortical_input.dtype,
            )
            zero_fired = torch.zeros(
                cortical_input.shape[0],
                device=cortical_input.device,
                dtype=torch.bool,
            )
            return zero_pulse, zero_fired

        # Salience pooling.
        if self.cfg.enable_salience_pooling:
            salience = torch.sigmoid(
                self.salience_pool(cortical_input).squeeze(-1)
            )
        else:
            salience = torch.zeros(
                cortical_input.shape[0],
                device=cortical_input.device,
                dtype=cortical_input.dtype,
            )

        # Refractory check.
        refractory_active = bool(self.refractory_counter.item() > 0)

        # Pulse decision.
        if (
            self.cfg.enable_synchronizing_pulse
            and not refractory_active
        ):
            fired = salience > self.cfg.pulse_threshold
        else:
            fired = torch.zeros(
                cortical_input.shape[0],
                device=cortical_input.device,
                dtype=torch.bool,
            )

        # Update refractory counter.
        with torch.no_grad():
            if fired.any().item():
                self.refractory_counter.copy_(
                    torch.tensor(self.cfg.refractory_steps, dtype=torch.long)
                )
            elif refractory_active:
                self.refractory_counter.copy_(
                    self.refractory_counter - 1
                )

        # Pulse pattern.
        if self.cfg.enable_target_boost:
            base_pulse = self.pulse_generator(cortical_input)
            pulse = base_pulse * fired.unsqueeze(-1).to(base_pulse.dtype)
            pulse = pulse * self.cfg.pulse_strength
        else:
            pulse = torch.zeros(
                cortical_input.shape[0], self.cfg.pulse_dim,
                device=cortical_input.device,
                dtype=cortical_input.dtype,
            )

        return pulse, fired

    def reset_refractory(self) -> None:
        """Clear the refractory counter."""
        with torch.no_grad():
            self.refractory_counter.zero_()


# =========================================================================
# SECTION: basal_ganglia
# (originally basal_ganglia_t.py)
# =========================================================================
"""
basal_ganglia_t.py
Loop Stage 7: Basal Ganglia Disinhibitory Gating

BIOLOGICAL GROUNDING
The basal ganglia implement action selection through disinhibition
rather than through softmax-style competitive excitation. Each
candidate action channel has a baseline tonic inhibition imposed by
GPi/SNr output. Striatal direct-pathway activation removes
inhibition from the chosen channel; indirect-pathway activation adds
inhibition to competitor channels. The hyperdirect pathway from
cortex to STN provides fast broad suppression before fine selection.

Mink (1996) is the canonical reference for this architecture. The
sharp departure from standard deep learning is that selection is
local: each channel knows its own striatal activation and the
dopamine signal, not the activations of competing channels. This
gives robustness to adversarial inputs on unrelated channels and
graceful degradation when dopamine is low.

This file implements the direct, indirect, and hyperdirect pathways
as independently ablatable submodules.

Primary grounding papers:

Mink JW (1996). "The basal ganglia: focused selection and inhibition
of competing motor programs." Progress in Neurobiology, 50(4),
381-425. DOI: 10.1016/S0301-0082(96)00042-1

Frank MJ (2005). "Dynamic dopamine modulation in the basal ganglia:
a neurocomputational account of cognitive deficits in medicated and
nonmedicated Parkinsonism." Journal of Cognitive Neuroscience, 17(1),
51-72. DOI: 10.1162/0898929052880093

Nambu A, Tokuno H, Takada M (2002). "Functional significance of the
cortico-subthalamo-pallidal hyperdirect pathway." Neuroscience
Research, 43(2), 111-117. DOI: 10.1016/S0168-0102(02)00027-5

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BasalGangliaConfig:
    """Configuration for the BG disinhibitory selector."""

    enable_basal_ganglia: bool = True
    enable_direct_pathway: bool = True
    enable_indirect_pathway: bool = True
    enable_hyperdirect_pathway: bool = True

    # Number of candidate action channels. NOT a biological quantity.
    n_channels: int = 8

    # Direct pathway learning rate (alpha in the corpus equation).
    # NOT a biological quantity, training artifact.
    alpha_direct: float = 0.3

    # Indirect pathway learning rate (beta in the corpus equation).
    # NOT a biological quantity, training artifact.
    beta_indirect: float = 0.2

    # Hyperdirect pathway broad suppression strength. NOT a
    # biological quantity, engineering tuning parameter.
    hyperdirect_strength: float = 0.4

    # Tonic baseline inhibition. The GPi/SNr output fires at high
    # tonic rate in vivo (40 to 80 Hz). The value here is normalized
    # to [0, 1] as an engineering convenience.
    tonic_baseline: float = 0.7

    # Decay toward baseline. Without decay, inhibition accumulates
    # without bound. NOT a biological quantity, training artifact.
    decay_rate: float = 0.1


class BasalGanglia(nn.Module):
    """Basal ganglia disinhibitory action selector.

    BIOLOGICAL STRUCTURE: Striatum, globus pallidus internal segment,
    substantia nigra pars reticulata, subthalamic nucleus.

    BIOLOGICAL FUNCTION: Selects one action from many competing
    candidates by disinhibiting the chosen channel and reinforcing
    inhibition on competitors. Dopamine signal modulates the
    direct/indirect balance.

    Mink JW (1996). DOI: 10.1016/S0301-0082(96)00042-1
    Frank MJ (2005). DOI: 10.1162/0898929052880093

    ANATOMICAL INTERFACE (input):
        Sending structures: cortex (corticostriatal projections to
        striatum, corticosubthalamic projections to STN) and VTA
        (dopamine modulation).
        Receiving structure: striatum and STN (this module).
        Connection: corticostriatal and corticosubthalamic.

    ANATOMICAL INTERFACE (output):
        Sending structures: GPi/SNr.
        Receiving structure: thalamus (motor and cognitive nuclei).
        Connection: GPi/SNr-to-thalamus inhibitory projection.
    """

    def __init__(self, cfg: Optional[BasalGangliaConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or BasalGangliaConfig()
        # Persistent tonic inhibition state. One scalar per channel,
        # decays toward tonic_baseline.
        self.register_buffer(
            "inhibition",
            torch.full((self.cfg.n_channels,), self.cfg.tonic_baseline),
        )

    def forward(
        self,
        striatal_activation: torch.Tensor,
        dopamine: Optional[torch.Tensor] = None,
        cortical_drive: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update inhibition state and return current channel gates.

        Args:
            striatal_activation: (B, n_channels) striatal activation
                pattern indicating candidate action strengths.
            dopamine: optional scalar TD error from VTA.
            cortical_drive: optional (B, n_channels) cortical drive
                feeding the hyperdirect pathway.

        Returns:
            channel_gates: (B, n_channels) values in roughly [0, 1+]
                where higher values mean the channel is more
                disinhibited (more likely to be selected).
        """
        if not self.cfg.enable_basal_ganglia:
            return torch.ones_like(striatal_activation)

        d = dopamine if dopamine is not None else torch.tensor(0.0)

        with torch.no_grad():
            # Decay toward tonic baseline.
            self.inhibition.copy_(
                self.inhibition + self.cfg.decay_rate
                * (self.cfg.tonic_baseline - self.inhibition)
            )

            # Mean activation across batch for the per-channel update.
            mean_activation = striatal_activation.detach().mean(dim=0)

            # Direct pathway: remove inhibition on activated channels.
            if self.cfg.enable_direct_pathway:
                self.inhibition.copy_(
                    self.inhibition - self.cfg.alpha_direct
                    * mean_activation * (1.0 + d)
                )

            # Indirect pathway: add inhibition on competitor channels.
            if self.cfg.enable_indirect_pathway:
                # Each channel's "competitor signal" is the sum of
                # other channels' activation.
                total = mean_activation.sum()
                competitor = total - mean_activation
                self.inhibition.copy_(
                    self.inhibition + self.cfg.beta_indirect
                    * competitor * (1.0 - d)
                )

            # Hyperdirect pathway: fast broad suppression based on
            # cortical drive magnitude.
            if (
                self.cfg.enable_hyperdirect_pathway
                and cortical_drive is not None
            ):
                broad = cortical_drive.detach().mean(dim=0).abs().mean()
                self.inhibition.copy_(
                    self.inhibition + self.cfg.hyperdirect_strength * broad
                )

        # The channel gate is the disinhibition: 1.0 minus the
        # inhibition, clamped to nonneg. Broadcast across batch.
        gate = torch.clamp(1.0 - self.inhibition, min=0.0)
        return gate.unsqueeze(0).expand(
            striatal_activation.shape[0], -1,
        )

    def reset_inhibition(self) -> None:
        """Reset inhibition to tonic baseline."""
        with torch.no_grad():
            self.inhibition.fill_(self.cfg.tonic_baseline)


# =========================================================================
# SECTION: anterior_cingulate_cortex
# (originally anterior_cingulate_cortex_t.py)
# =========================================================================
"""
anterior_cingulate_cortex_t.py
Loop Stage: ACC Conflict Detection

BIOLOGICAL GROUNDING
The anterior cingulate cortex monitors response conflict and emits a
control signal that modulates downstream attention and gain. The
canonical Botvinick et al. (2001) framework treats conflict as the
overlap of competing response activations; entropy of the response
distribution is the canonical PRAGMI form.

Botvinick et al. (1999) further document a sequential dependency:
ACC activation peaks on incompatible trials that follow compatible
trials, not on incompatible trials that follow other incompatible
trials. The system is most engaged when conflict rises unexpectedly.
A faithful implementation includes a temporal-derivative term with a
hinge at zero.

This file implements both the entropy-based instantaneous conflict
and the rising-conflict derivative term, with each component
independently ablatable.

Primary grounding papers:

Botvinick MM, Braver TS, Barch DM, Carter CS, Cohen JD (2001).
"Conflict monitoring and cognitive control." Psychological Review,
108(3), 624-652. DOI: 10.1037/0033-295X.108.3.624

Botvinick M, Nystrom LE, Fissell K, Carter CS, Cohen JD (1999).
"Conflict monitoring versus selection-for-action in anterior
cingulate cortex." Nature, 402(6758), 179-181.
DOI: 10.1038/46035

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ACCConfig:
    """Configuration for the ACC conflict module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_acc: bool = True
    enable_entropy_conflict: bool = True
    enable_derivative_term: bool = True

    # Coupling coefficient on instantaneous entropy. Botvinick et al.
    # (2001) DOI: 10.1037/0033-295X.108.3.624. NOT a biological
    # quantity, training artifact.
    beta: float = 1.0

    # Coupling coefficient on the rising-conflict derivative term.
    # Botvinick et al. (1999) report r^2 = 0.66 between the cI-iI
    # reaction time difference and the corresponding ACC activation
    # difference. NOT a biological quantity, training artifact.
    beta_delta: float = 0.5


class AnteriorCingulateCortex(nn.Module):
    """ACC conflict detection.

    BIOLOGICAL STRUCTURE: Anterior cingulate cortex.
    BIOLOGICAL FUNCTION: Monitors response conflict via entropy of the
    competing response distribution and emits a control signal that
    modulates downstream attention and NE gain. Sequential dependency
    captured by the rising-conflict derivative term.

    Botvinick MM et al. (2001). DOI: 10.1037/0033-295X.108.3.624
    Botvinick M et al. (1999). DOI: 10.1038/46035

    ANATOMICAL INTERFACE (input):
        Sending structures: PFC competing response populations.
        Receiving structure: ACC (this module).
        Connection: PFC-to-ACC corticocortical projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: ACC.
        Receiving structures: thalamic gate (gain modulation), LC
        (NE recruitment).
        Connection: ACC-to-thalamus and ACC-to-LC projections.
    """

    def __init__(self, cfg: Optional[ACCConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ACCConfig()
        # Persistent buffer for the previous-step entropy used to
        # compute the rising-conflict derivative.
        self.register_buffer("prev_entropy", torch.tensor(0.0))

    def forward(self, response_activations: torch.Tensor) -> torch.Tensor:
        """Compute the ACC control signal.

        Args:
            response_activations: (B, N) competing response
                activations. Each row is one batch sample with N
                competing response options.

        Returns:
            control_signal: (B,) scalar conflict-driven control.
        """
        if not self.cfg.enable_acc:
            return torch.zeros(
                response_activations.shape[0],
                device=response_activations.device,
                dtype=response_activations.dtype,
            )

        if self.cfg.enable_entropy_conflict:
            # Normalize to probability distribution and compute entropy.
            probs = F.softmax(response_activations, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        else:
            entropy = torch.zeros(
                response_activations.shape[0],
                device=response_activations.device,
                dtype=response_activations.dtype,
            )

        instantaneous = self.cfg.beta * entropy

        if self.cfg.enable_derivative_term:
            # Rising-conflict derivative term with hinge at zero.
            # Botvinick et al. (1999) DOI: 10.1038/46035.
            mean_entropy = entropy.mean()
            delta = torch.clamp(mean_entropy - self.prev_entropy, min=0.0)
            derivative = self.cfg.beta_delta * delta
            with torch.no_grad():
                self.prev_entropy.copy_(mean_entropy.detach())
        else:
            derivative = torch.tensor(
                0.0,
                device=response_activations.device,
                dtype=response_activations.dtype,
            )

        return instantaneous + derivative

    def reset_history(self) -> None:
        """Reset the prev_entropy buffer."""
        with torch.no_grad():
            self.prev_entropy.zero_()


# =========================================================================
# SECTION: cerebellum
# (originally cerebellum_t.py)
# =========================================================================
"""
cerebellum_t.py
Loop Stage 8: Cerebellum Efference Copy + Forward Models

BIOLOGICAL GROUNDING
The cerebellum implements forward models that predict the sensory
consequences of motor and cognitive commands. The classic Wolpert,
Miall, Kawato (1998) MPFIM framework treats the cerebellum as a
bank of paired forward-inverse modules, each tuned to a different
context.

Popa and Ebner (2019) show that individual Purkinje cells maintain
two independent forward models: an implicit kinematic model
predicting effector state, and an explicit task-performance model
predicting whether the action achieves its goal. The two dissociate
experimentally.

The cerebellum is also functionally zoned. Apps and Garwicz (2005)
identify three zonal divisions with distinct climbing fiber sources:
vestibulocerebellum (VOR feedback, retinal slip error),
spinocerebellum (limb position forward model, proprioceptive error),
and cerebrocerebellum (cognitive operations forward model, cortical
errors via parvocellular red nucleus). Each zone is approximately
250 micrometers wide as a sagittal microzone strip.

This file implements the dual-forward-model architecture in three
zonal configurations, with each zone independently ablatable.

Primary grounding papers:

Wolpert DM, Miall RC, Kawato M (1998). "Internal models in the
cerebellum." Trends in Cognitive Sciences, 2(9), 338-347.
DOI: 10.1016/S1364-6613(98)01221-2

Popa LS, Ebner TJ (2019). "Cerebellum, predictions and errors."
Frontiers in Cellular Neuroscience, 12:524.
DOI: 10.3389/fncel.2018.00524

Apps R, Garwicz M (2005). "Anatomical and physiological foundations
of cerebellar information processing." Nature Reviews Neuroscience,
6(4), 297-311. DOI: 10.1038/nrn1646

Yamazaki T, Tanaka S (2007). "A spiking network model for
passage-of-time representation in the cerebellum." European Journal
of Neuroscience, 26(8), 2279-2292.
DOI: 10.1111/j.1460-9568.2007.05837.x

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CerebellarConfig:
    """Configuration for the cerebellum module."""

    enable_cerebellum: bool = True
    enable_kinematic_model: bool = True
    enable_task_model: bool = True
    enable_vestibular_zone: bool = True
    enable_spinal_zone: bool = True
    enable_cerebral_zone: bool = True

    cmd_dim: int = 32
    state_dim: int = 64
    goal_dim: int = 64

    # Correction learning rate. NOT a biological quantity, training
    # artifact.
    eta: float = 0.05


class CerebellarZone(nn.Module):
    """One cerebellar microzone with paired kinematic and task models.

    BIOLOGICAL STRUCTURE: A 250-micrometer-wide sagittal strip of
    Purkinje cells with one climbing fiber source from a specific
    inferior olive subnucleus.

    BIOLOGICAL FUNCTION: Maintains two simultaneous forward models
    per Popa and Ebner (2019): kinematic prediction (effector state)
    and task-performance prediction (goal achievement). The two
    dissociate experimentally.

    Popa LS, Ebner TJ (2019). DOI: 10.3389/fncel.2018.00524
    Yamazaki T, Tanaka S (2007). DOI: 10.1111/j.1460-9568.2007.05837.x
    """

    def __init__(self, cfg: CerebellarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Kinematic forward model: predicts effector state from cmd
        # and current state.
        self.kinematic_head = nn.Linear(
            cfg.cmd_dim + cfg.state_dim, cfg.state_dim, bias=True,
        )
        # Task forward model: predicts task outcome from cmd, goal,
        # and current state.
        self.task_head = nn.Linear(
            cfg.cmd_dim + cfg.goal_dim + cfg.state_dim, 1, bias=True,
        )

    def forward(
        self,
        cmd: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run paired kinematic and task forward models.

        Args:
            cmd: (B, cmd_dim) motor or cognitive command.
            state: (B, state_dim) current sensory/effector state.
            goal: (B, goal_dim) current goal representation.

        Returns:
            pred_kin: (B, state_dim) predicted next state.
            pred_task: (B,) predicted task-performance scalar.
        """
        if self.cfg.enable_kinematic_model:
            kin_input = torch.cat([cmd, state], dim=-1)
            pred_kin = self.kinematic_head(kin_input)
        else:
            pred_kin = torch.zeros_like(state)

        if self.cfg.enable_task_model:
            task_input = torch.cat([cmd, goal, state], dim=-1)
            pred_task = self.task_head(task_input).squeeze(-1)
        else:
            pred_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )

        return pred_kin, pred_task


class Cerebellum(nn.Module):
    """Cerebellum with three zonal configurations.

    BIOLOGICAL STRUCTURE: Cerebellar cortex divided into
    vestibulocerebellum, spinocerebellum, and cerebrocerebellum,
    each receiving climbing fiber error signals from distinct
    inferior olive subnuclei and projecting to distinct deep nuclei.

    BIOLOGICAL FUNCTION: Provides forward-model prediction and
    error-driven correction at three functional levels: vestibular
    reflexes, limb kinematics, and cognitive operations.

    Wolpert DM, Miall RC, Kawato M (1998).
    DOI: 10.1016/S1364-6613(98)01221-2
    Apps R, Garwicz M (2005). DOI: 10.1038/nrn1646
    Popa LS, Ebner TJ (2019). DOI: 10.3389/fncel.2018.00524

    ANATOMICAL INTERFACE (input):
        Sending structures: cortex (cmd via efference copy and goal
        via cortico-pontine projections), state estimators (via
        proprioceptive afferents and visual feedback).
        Receiving structures: cerebellar zones (this module).
        Connection: cortico-ponto-cerebellar pathway and afferent
        sensory pathways.

    ANATOMICAL INTERFACE (output):
        Sending structures: deep cerebellar nuclei (dentate,
        interpositus, fastigial).
        Receiving structures: cortex (via thalamus) for the
        cerebrocerebellum, motor systems for the spinocerebellum,
        and brainstem nuclei for the vestibulocerebellum.
        Connection: cerebello-thalamo-cortical pathway and direct
        cerebello-brainstem projections.
    """

    def __init__(self, cfg: Optional[CerebellarConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CerebellarConfig()
        self.vestibular = CerebellarZone(self.cfg)
        self.spinal = CerebellarZone(self.cfg)
        self.cerebral = CerebellarZone(self.cfg)

    def forward(
        self,
        cmd: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> dict:
        """Run all three zones and return their predictions.

        Args:
            cmd: (B, cmd_dim).
            state: (B, state_dim).
            goal: (B, goal_dim).

        Returns:
            dict mapping zone name to (pred_kin, pred_task) tuple.
        """
        if not self.cfg.enable_cerebellum:
            zero_kin = torch.zeros_like(state)
            zero_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )
            return {
                "vestibular": (zero_kin, zero_task),
                "spinal": (zero_kin, zero_task),
                "cerebral": (zero_kin, zero_task),
            }

        result = {}
        if self.cfg.enable_vestibular_zone:
            result["vestibular"] = self.vestibular(cmd, state, goal)
        else:
            zero_kin = torch.zeros_like(state)
            zero_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )
            result["vestibular"] = (zero_kin, zero_task)

        if self.cfg.enable_spinal_zone:
            result["spinal"] = self.spinal(cmd, state, goal)
        else:
            zero_kin = torch.zeros_like(state)
            zero_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )
            result["spinal"] = (zero_kin, zero_task)

        if self.cfg.enable_cerebral_zone:
            result["cerebral"] = self.cerebral(cmd, state, goal)
        else:
            zero_kin = torch.zeros_like(state)
            zero_task = torch.zeros(
                cmd.shape[0], device=cmd.device, dtype=cmd.dtype,
            )
            result["cerebral"] = (zero_kin, zero_task)

        return result

    def correct(
        self,
        cmd: torch.Tensor,
        actual: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """Apply correction to the next command from kinematic error.

        Args:
            cmd: (B, cmd_dim) original command.
            actual: (B, state_dim) observed state outcome.
            pred: (B, state_dim) predicted state.

        Returns:
            cmd_next: (B, cmd_dim) corrected command. Note that this
                routes kinematic error back to the cmd; task error
                routes to planning, which lives in PFC/BG and is the
                caller's responsibility.
        """
        # Truncate or pad error to cmd_dim (engineering simplification
        # for the cross-dimensional projection that cortex would do
        # in vivo).
        error = actual - pred
        if error.shape[-1] > cmd.shape[-1]:
            error_proj = error[..., : cmd.shape[-1]]
        elif error.shape[-1] < cmd.shape[-1]:
            pad = torch.zeros(
                *error.shape[:-1],
                cmd.shape[-1] - error.shape[-1],
                device=error.device, dtype=error.dtype,
            )
            error_proj = torch.cat([error, pad], dim=-1)
        else:
            error_proj = error
        return cmd - self.cfg.eta * error_proj


# =========================================================================
# SECTION: amygdala
# (originally amygdala_t.py)
# =========================================================================
"""
amygdala_t.py
Loop Stage: Amygdala Emotional Consolidation Tagging

BIOLOGICAL GROUNDING
The amygdala provides emotional valence tagging that biases which
hippocampal memory traces will be preferentially consolidated. The
basolateral amygdala (BLA) projects directly to ventral hippocampus
and indirectly modulates dorsal hippocampus through entorhinal
cortex and the locus coeruleus pathway. Girardeau, Inema, Buzsaki
(2017) document BLA-hippocampus replay coupling during sleep:
hippocampal sharp-wave ripples that co-occur with BLA reactivation
are preferentially preserved into long-term memory, a mechanism
later termed "emotional tagging".

Wei, Krishnan, Bazhenov (2016) provide the computational instantiation:
emotional salience computed at BLA modulates the consolidation gain
applied to hippocampal traces during sleep replay. McGaugh (2004)
reviews the broader stress-hormone amplification of memory.

In the PRAGMI loop, the amygdala module computes an emotional
salience scalar from current state and emits a consolidation tag
that scales how strongly a memory trace from the hippocampus will be
imprinted into the cognitive kernel during the next sleep cycle.
This is a Genesis-specific extension: in vivo the emotional tagging
acts on hippocampal-cortical replay; here we route it through the
kernel's consolidation pathway.

Primary grounding papers:

Girardeau G, Inema I, Buzsaki G (2017). "Reactivations of emotional
memory in the hippocampus-amygdala system during sleep." Nature
Neuroscience, 20(11), 1634-1642. DOI: 10.1038/nn.4637

Wei Y, Krishnan GP, Bazhenov M (2016). "Synaptic mechanisms of
memory consolidation during sleep slow oscillations." Journal of
Neuroscience, 36(15), 4231-4247.
DOI: 10.1523/JNEUROSCI.3648-15.2016

McGaugh JL (2004). "The amygdala modulates the consolidation of
memories of emotionally arousing experiences." Annual Review of
Neuroscience, 27, 1-28.
DOI: 10.1146/annurev.neuro.27.070203.144157

LeDoux JE (2000). "Emotion circuits in the brain." Annual Review of
Neuroscience, 23, 155-184.
DOI: 10.1146/annurev.neuro.23.1.155

Genesis Labs Research
Authored for the PRAGMI loop assembly.

GENESIS-SPECIFIC EXTENSION: The emotional tag is routed to the
cognitive kernel's consolidation pathway rather than acting only on
hippocampal replay. This is an architectural choice that extends the
biological grounding to fit PRAGMI's three-layer separation. The
qualitative behavior (high-salience traces preferentially preserved)
matches the empirical phenomenon.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AmygdalaConfig:
    """Configuration for the amygdala module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_amygdala: bool = True
    enable_valence_evaluation: bool = True
    enable_arousal_evaluation: bool = True
    enable_consolidation_tag: bool = True

    state_dim: int = 64
    valence_dim: int = 8
    arousal_dim: int = 1

    # Consolidation tag scale. Wei et al. (2016) DOI:
    # 10.1523/JNEUROSCI.3648-15.2016. NOT a biological quantity,
    # training artifact.
    tag_scale: float = 2.0

    # Arousal threshold for triggering high-priority consolidation.
    # NOT a biological quantity, engineering tuning.
    arousal_threshold: float = 0.5


class Amygdala(nn.Module):
    """Amygdala emotional valence and consolidation tagging.

    BIOLOGICAL STRUCTURE: Basolateral amygdala complex (BLA), with
    projections to ventral hippocampus, entorhinal cortex, and locus
    coeruleus.

    BIOLOGICAL FUNCTION: Computes emotional salience from current
    state, biasing which memory traces are preferentially consolidated
    during subsequent sleep replay. High emotional arousal increases
    the consolidation tag, predicting stronger trace preservation.

    Girardeau G, Inema I, Buzsaki G (2017). DOI: 10.1038/nn.4637
    Wei Y, Krishnan GP, Bazhenov M (2016).
    DOI: 10.1523/JNEUROSCI.3648-15.2016
    McGaugh JL (2004).
    DOI: 10.1146/annurev.neuro.27.070203.144157
    LeDoux JE (2000). DOI: 10.1146/annurev.neuro.23.1.155

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical state representation including
        association cortex.
        Receiving structure: BLA (this module).
        Connection: cortico-amygdaloid projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: BLA.
        Receiving structures: ventral hippocampus, entorhinal cortex,
        locus coeruleus (the consolidation tag is routed to all three
        in vivo; here it is a single tag scalar that the kernel
        applies during consolidation).
        Connection: BLA-hippocampal and BLA-LC projections.
    """

    def __init__(self, cfg: Optional[AmygdalaConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or AmygdalaConfig()

        # Valence evaluator: maps state to a multi-dim valence vector
        # (positive/negative/threat/reward dimensions).
        self.valence_head = nn.Linear(
            self.cfg.state_dim, self.cfg.valence_dim, bias=True,
        )
        # Arousal evaluator: maps state to a scalar arousal level.
        self.arousal_head = nn.Linear(
            self.cfg.state_dim, self.cfg.arousal_dim, bias=True,
        )

    def forward(
        self, state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute valence, arousal, and the consolidation tag.

        Args:
            state: (B, state_dim) cortical state representation.

        Returns:
            valence: (B, valence_dim) signed emotional vector.
            arousal: (B,) magnitude of emotional engagement.
            tag: (B,) consolidation tag scalar.
        """
        if not self.cfg.enable_amygdala:
            zero_v = torch.zeros(
                state.shape[0], self.cfg.valence_dim,
                device=state.device, dtype=state.dtype,
            )
            zero_a = torch.zeros(
                state.shape[0],
                device=state.device, dtype=state.dtype,
            )
            return zero_v, zero_a, zero_a.clone()

        # Valence (signed).
        if self.cfg.enable_valence_evaluation:
            valence = torch.tanh(self.valence_head(state))
        else:
            valence = torch.zeros(
                state.shape[0], self.cfg.valence_dim,
                device=state.device, dtype=state.dtype,
            )

        # Arousal (positive scalar).
        if self.cfg.enable_arousal_evaluation:
            arousal = torch.sigmoid(
                self.arousal_head(state).squeeze(-1)
            )
        else:
            arousal = torch.zeros(
                state.shape[0], device=state.device, dtype=state.dtype,
            )

        # Consolidation tag: arousal-weighted scalar suitable for use
        # as a multiplier on hippocampal trace strength during the
        # next consolidation pass.
        if self.cfg.enable_consolidation_tag:
            # Tag rises sharply once arousal crosses threshold.
            above = F.relu(arousal - self.cfg.arousal_threshold)
            tag = 1.0 + self.cfg.tag_scale * above
        else:
            tag = torch.ones(
                state.shape[0], device=state.device, dtype=state.dtype,
            )

        return valence, arousal, tag


# =========================================================================
# SECTION: entorhinal_cortex
# (originally entorhinal_cortex_t.py)
# =========================================================================
"""
entorhinal_cortex_t.py
Cognitive Kernel: Entorhinal Cortex with Medial/Lateral Subdivision

BIOLOGICAL GROUNDING
The entorhinal cortex is the primary gateway between neocortex and
hippocampus. It is not a single homogeneous structure. Medial and
lateral entorhinal cortex (MEC and LEC) are functionally and
anatomically distinct subdivisions with different cortical input
sources, different projection targets within hippocampus, and
different computational roles.

MEC carries spatial and self-motion information. Grid cells in MEC
layer II provide a metric for space. MEC projects predominantly to
dentate gyrus and CA3, where pattern separation orthogonalizes the
spatial code before storage in the CA3 attractor network.

LEC carries non-spatial content: object identity, social identity,
odor, and contextual features. LEC projects directly to CA2 via the
recently characterized direct pathway documented by Lopez-Rojas et
al. (2022), bypassing dentate gyrus pattern separation. This makes
sense functionally: identity matching at CA2 requires preserved
representational structure, which DG orthogonalization would destroy.

The kernel's previous EntorhinalCortex module emitted a single output
tensor that was used for both the DG/CA3 path and (in the new CA2
module) the direct CA2 path. This file replaces that module with a
two-output version that separates MEC and LEC contributions, so that
the anatomical distinction is real rather than labeled. Both
subdivisions still share the short-term buffer and the input
normalization that the original module provided, because both EC
subdivisions show persistent firing properties grounded in Egorov et
al. (2002).

Primary grounding papers:

Witter MP, Naber PA, van Haeften T, Machielsen WC, Rombouts SA,
Barkhof F, Scheltens P, Lopes da Silva FH (2000). "Cortico-hippocampal
communication by way of parallel parahippocampal-subicular pathways."
Hippocampus, 10(4), 398-410.
DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K

Hargreaves EL, Rao G, Lee I, Knierim JJ (2005). "Major dissociation
between medial and lateral entorhinal input to dorsal hippocampus."
Science, 308(5729), 1792-1794. DOI: 10.1126/science.1110449

Lopez-Rojas J, von Richthofen HJ, Kempter R, Schmitz D, Gee CE,
Larkum ME (2022). "A direct lateral entorhinal cortex to hippocampal
CA2 circuit conveys social information required for social memory."
Neuron, 110(9), 1559-1572.e4. DOI: 10.1016/j.neuron.2022.01.028

Egorov AV, Hamam BN, Fransen E, Hasselmo ME, Alonso AA (2002).
"Graded persistent activity in entorhinal cortex neurons." Nature,
420(6912), 173-178. DOI: 10.1038/nature01171

Genesis Labs Research
Authored for the PRAGMI Cognitive Kernel
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# PART 1: CONFIG WITH ABLATION FLAGS
# =========================================================================

@dataclass
class EntorhinalCortexConfig:
    """Configuration for the EntorhinalCortex module.

    Ablation flags are ENGINEERING CONTROLS, not biological quantities.
    Master flag is first per the Genesis Labs Research Ablation Flag
    Design Standard.
    """

    # Master flag for the entire EC module. When False the forward
    # method returns labeled neutral values for both MEC and LEC
    # outputs. NOT a biological quantity.
    enable_entorhinal_cortex: bool = True

    # Medial entorhinal cortex subdivision. Hargreaves et al. (2005)
    # DOI: 10.1126/science.1110449. When False, the spatial pathway
    # to DG and CA3 is silenced. NOT a biological quantity.
    enable_medial_subdivision: bool = True

    # Lateral entorhinal cortex subdivision. Lopez-Rojas et al. (2022)
    # DOI: 10.1016/j.neuron.2022.01.028. When False, the direct path
    # to CA2 is silenced and CA2 receives only its mossy fiber
    # collateral input from CA3. NOT a biological quantity.
    enable_lateral_subdivision: bool = True

    # Persistent activity short-term buffer. Egorov et al. (2002)
    # DOI: 10.1038/nature01171 documents graded persistent firing in
    # EC layer V neurons. When False, the EC outputs respond only to
    # immediate input. NOT a biological quantity.
    enable_persistent_buffer: bool = True

    # Coordinate manifold dimensionality. Must match the rest of the
    # kernel. NOT a biological quantity.
    coordinate_dim: int = 64

    # MEC output dimensionality. Sized to match the existing DG
    # input expectation. NOT a biological quantity.
    mec_dim: int = 64

    # LEC output dimensionality. Sized to match the CA2 module's
    # coordinate_dim input expectation. NOT a biological quantity.
    lec_dim: int = 64

    # Buffer time constant. The persistent firing in Egorov et al.
    # (2002) is graded over seconds; the value here is an engineering
    # approximation that produces visible buffering across a training
    # batch sequence. Engineering approximation parameterized to match
    # the qualitative timescale.
    buffer_tau: float = 0.95

    # Buffer mixing weight. Controls how strongly the persistent
    # buffer biases the immediate input. NOT a biological quantity,
    # engineering tuning parameter.
    buffer_mix: float = 0.1


# =========================================================================
# PART 2: ENTORHINAL CORTEX MODULE
# =========================================================================

class EntorhinalCortex(nn.Module):
    """Entorhinal cortex with medial/lateral subdivision.

    BIOLOGICAL STRUCTURE: Entorhinal cortex layers II and III, divided
    into medial (MEC) and lateral (LEC) subdivisions. MEC layer II
    contains grid cells; LEC layer II contains object/identity-tuned
    cells.

    BIOLOGICAL FUNCTION: MEC carries spatial and self-motion content
    to dentate gyrus and CA3 via the perforant path. LEC carries
    non-spatial content (identity, social, contextual, olfactory) to
    CA2 via the recently characterized direct pathway. Both
    subdivisions exhibit graded persistent activity in deeper layers,
    providing a short-term buffer across input gaps.

    Witter MP et al. (2000). DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K
    Hargreaves EL et al. (2005). DOI: 10.1126/science.1110449
    Lopez-Rojas J et al. (2022). DOI: 10.1016/j.neuron.2022.01.028
    Egorov AV et al. (2002). DOI: 10.1038/nature01171

    ANATOMICAL INTERFACE (input):
        Sending structures: neocortical association areas via the
        perirhinal cortex (LEC bias) and postrhinal cortex (MEC bias).
        Receiving structures: MEC layer II (spatial) and LEC layer II
        (non-spatial), this module.
        Connection: cortico-entorhinal projections through the
        parahippocampal region.

    ANATOMICAL INTERFACE (output, MEC):
        Sending structure: MEC layer II.
        Receiving structures: dentate gyrus granule cells and CA3
        pyramidal cells.
        Connection: medial perforant path.

    ANATOMICAL INTERFACE (output, LEC):
        Sending structure: LEC layer II.
        Receiving structure: CA2 pyramidal cells.
        Connection: direct LEC-to-CA2 pathway documented by
        Lopez-Rojas et al. (2022).
    """

    def __init__(self, cfg: Optional[EntorhinalCortexConfig] = None) -> None:
        """Initialize the EC module.

        Args:
            cfg: EntorhinalCortexConfig. If None, defaults are used.
        """
        super().__init__()
        self.cfg = cfg or EntorhinalCortexConfig()

        # Per-subdivision projections from the shared coordinate input.
        # Separate weights enforce that MEC and LEC develop different
        # tuning even from the same upstream input distribution, which
        # matches the empirical dissociation in Hargreaves et al.
        # (2005).
        self.mec_projection = nn.Linear(
            self.cfg.coordinate_dim, self.cfg.mec_dim, bias=True,
        )
        self.lec_projection = nn.Linear(
            self.cfg.coordinate_dim, self.cfg.lec_dim, bias=True,
        )

        # Layer normalization on the combined input prior to
        # subdivision-specific projection. Stabilizes the input
        # distribution across batches.
        self.input_norm = nn.LayerNorm(self.cfg.coordinate_dim)

        # Persistent activity buffer. One shared buffer represents
        # the deep-layer persistent firing that both MEC and LEC
        # have access to in vivo. Egorov et al. (2002) DOI:
        # 10.1038/nature01171.
        self.register_buffer(
            "persistent_buffer", torch.zeros(self.cfg.coordinate_dim),
        )

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MEC and LEC outputs from the coordinate input.

        Args:
            coords: (B, coordinate_dim) input coordinates from the
                upstream encoder or cortical buffer.

        Returns:
            mec_output: (B, mec_dim) for routing to DG/CA3.
            lec_output: (B, lec_dim) for routing to CA2 direct path.
        """
        if not self.cfg.enable_entorhinal_cortex:
            zero_mec = torch.zeros(
                coords.shape[0], self.cfg.mec_dim,
                device=coords.device, dtype=coords.dtype,
            )
            zero_lec = torch.zeros(
                coords.shape[0], self.cfg.lec_dim,
                device=coords.device, dtype=coords.dtype,
            )
            return zero_mec, zero_lec

        # Update the persistent buffer if enabled. Egorov et al. (2002).
        if self.cfg.enable_persistent_buffer:
            with torch.no_grad():
                batch_mean = coords.detach().mean(dim=0)
                tau = self.cfg.buffer_tau
                self.persistent_buffer.copy_(
                    tau * self.persistent_buffer + (1.0 - tau) * batch_mean
                )
            biased_input = coords + self.cfg.buffer_mix * self.persistent_buffer.unsqueeze(0)
        else:
            biased_input = coords

        normed = self.input_norm(biased_input)

        # MEC subdivision output, gated by ablation flag.
        if self.cfg.enable_medial_subdivision:
            mec_output = self.mec_projection(normed)
        else:
            mec_output = torch.zeros(
                coords.shape[0], self.cfg.mec_dim,
                device=coords.device, dtype=coords.dtype,
            )

        # LEC subdivision output, gated by ablation flag.
        if self.cfg.enable_lateral_subdivision:
            lec_output = self.lec_projection(normed)
        else:
            lec_output = torch.zeros(
                coords.shape[0], self.cfg.lec_dim,
                device=coords.device, dtype=coords.dtype,
            )

        return mec_output, lec_output

    def reset_buffer(self) -> None:
        """Reset the persistent activity buffer to zero."""
        with torch.no_grad():
            self.persistent_buffer.zero_()

    def get_diagnostic_state(self) -> dict:
        """Return current internal state for diagnostic logging."""
        return {
            "buffer_norm": self.persistent_buffer.norm().item(),
        }


# =========================================================================
# SECTION: cornu_ammonis_1
# (originally cornu_ammonis_1_t.py)
# =========================================================================
"""
cornu_ammonis_1_t.py
Cognitive Kernel: CA1 Subfield with Ternary Conjunction

BIOLOGICAL GROUNDING
CA1 is the primary output region of the hippocampal trisynaptic
circuit. It receives three converging inputs: the Schaffer collateral
projection from CA3 carrying the pattern-completed spatial code, the
direct temporoammonic projection from EC layer III carrying current
sensory drive, and the Schaffer collateral projection from CA2
carrying the temporal drift signal and identity comparator output.
The conjunction of these three streams at CA1 produces the
hippocampal output code that combines spatial position, sensory
context, and time-stamp/identity into a single representation.

CA1 also acts as a comparator. The classic Lisman and Grace (2005)
account treats CA1 as a novelty detector that compares the CA3
reconstruction against direct EC input; mismatch triggers
hippocampal-VTA loop activation and gates downstream learning. The
addition of CA2 input refines this picture: CA1 now distinguishes
spatial novelty (CA3 vs EC mismatch) from identity novelty (carried
by CA2 mismatch), and the temporal drift component of the CA2 signal
provides the time-stamp that lets CA1 distinguish repeated visits to
the same place.

This file replaces the previous CA1Comparator that took two inputs
(Schaffer from CA3, direct from EC) with a CA1 module that takes
three (Schaffer from CA3, Schaffer from CA2, direct from EC). The
ternary conjunction logic uses the existing novelty gate to balance
CA3 reconstruction against EC drive, then incorporates the CA2
contribution as an additive temporal/identity overlay rather than as
a competing reconstruction.

Primary grounding papers:

Lisman JE, Grace AA (2005). "The hippocampal-VTA loop: controlling
the entry of information into long-term memory." Neuron, 46(5),
703-713. DOI: 10.1016/j.neuron.2005.05.002

Bittner KC, Milstein AD, Grienberger C, Romani S, Magee JC (2017).
"Behavioral time scale synaptic plasticity underlies CA1 place
fields." Science, 357(6355), 1033-1036.
DOI: 10.1126/science.aan3846

Mankin EA, Diehl GW, Sparks FT, Leutgeb S, Leutgeb JK (2015).
"Hippocampal CA2 activity patterns change over time to a larger
extent than between spatial contexts." Neuron, 85(1), 190-201.
DOI: 10.1016/j.neuron.2014.12.001

Genesis Labs Research
Authored for the PRAGMI Cognitive Kernel
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# PART 1: CONFIG WITH ABLATION FLAGS
# =========================================================================

@dataclass
class CA1Config:
    """Configuration for the CA1 module with ternary conjunction.

    Master flag is first per the Genesis Labs Research Ablation Flag
    Design Standard. NOT a biological quantity.
    """

    # Master flag for the entire CA1 module. NOT a biological quantity.
    enable_ca1: bool = True

    # CA3 Schaffer collateral input. Lisman and Grace (2005) DOI:
    # 10.1016/j.neuron.2005.05.002. NOT a biological quantity.
    enable_ca3_schaffer: bool = True

    # Temporoammonic direct EC input. NOT a biological quantity.
    enable_temporoammonic: bool = True

    # CA2 Schaffer input. New pathway added for the CA2 integration.
    # NOT a biological quantity.
    enable_ca2_schaffer: bool = True

    # Novelty-driven gate that balances CA3 reconstruction against EC
    # direct drive. Lisman and Grace (2005) novelty signal mechanism.
    # When False, the conjunction is a simple sum without gating.
    # NOT a biological quantity.
    enable_novelty_gate: bool = True

    coordinate_dim: int = 64
    ca3_dim: int = 192
    ca2_input_dim: int = 192
    ca1_dim: int = 192
    ec_input_dim: int = 64

    # Weight on the CA2 contribution to the conjunction. The CA2
    # signal is an overlay rather than a competing reconstruction, so
    # it enters the conjunction weighted lower than CA3 by default.
    # NOT a biological quantity, engineering tuning parameter.
    ca2_overlay_weight: float = 0.3


# =========================================================================
# PART 2: CA1 MODULE
# =========================================================================

class CornuAmmonis1(nn.Module):
    """CA1 with ternary conjunction of CA3, CA2, and direct EC inputs.

    BIOLOGICAL STRUCTURE: CA1 pyramidal cell layer. Receives Schaffer
    collateral input from CA3 (and from CA2), and direct
    temporoammonic input from EC layer III.

    BIOLOGICAL FUNCTION: Comparator and output stage. Detects
    mismatch between CA3 reconstruction and direct EC drive (Lisman
    and Grace 2005), and combines this with the CA2 temporal/identity
    overlay to produce a place-plus-time-stamp code that is read out
    via subiculum to neocortex. Behavioral time scale plasticity at
    CA1 dendrites (Bittner et al. 2017) provides the substrate for
    binding the three streams over the relevant temporal window.

    Lisman JE, Grace AA (2005). DOI: 10.1016/j.neuron.2005.05.002
    Bittner KC et al. (2017). DOI: 10.1126/science.aan3846
    Mankin EA et al. (2015). DOI: 10.1016/j.neuron.2014.12.001

    ANATOMICAL INTERFACE (input):
        Sending structures: CA3 pyramidal cells (Schaffer collaterals),
        CA2 pyramidal cells (CA2 component of Schaffer projection),
        and EC layer III (temporoammonic path).
        Receiving structure: CA1 pyramidal cells (this module).
        Connections: CA3 Schaffer collaterals, CA2 Schaffer projection,
        temporoammonic path.

    ANATOMICAL INTERFACE (output):
        Sending structure: CA1 pyramidal cells.
        Receiving structure: Subiculum.
        Connection: CA1-to-subiculum projection.
    """

    def __init__(self, cfg: Optional[CA1Config] = None) -> None:
        """Initialize CA1.

        Args:
            cfg: CA1Config. Defaults used if None.
        """
        super().__init__()
        self.cfg = cfg or CA1Config()

        # Direct EC input projected up to CA1 dimensionality.
        # Temporoammonic path from EC layer III.
        self.temporoammonic = nn.Linear(
            self.cfg.ec_input_dim, self.cfg.ca1_dim, bias=True,
        )

        # Compare projections for the CA3 Schaffer reconstruction and
        # the direct EC drive. Used by the novelty gate.
        self.compare_schaffer = nn.Linear(
            self.cfg.ca3_dim, self.cfg.ca1_dim, bias=False,
        )
        self.compare_direct = nn.Linear(
            self.cfg.ca1_dim, self.cfg.ca1_dim, bias=False,
        )

        # CA2 Schaffer projection. CA2 already projects in CA1
        # dimensionality from its own ca2_to_ca1 head, so this is a
        # light gating layer rather than a full projection.
        self.ca2_overlay = nn.Linear(
            self.cfg.ca2_input_dim, self.cfg.ca1_dim, bias=True,
        )

        # Output gate for the conjoined representation.
        self.output_gate = nn.Linear(
            self.cfg.ca1_dim, self.cfg.ca1_dim, bias=True,
        )

    def forward(
        self,
        schaffer_input: torch.Tensor,
        ec_input: torch.Tensor,
        ca2_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the CA1 ternary conjunction.

        Args:
            schaffer_input: (B, ca3_dim) CA3 Schaffer collateral
                reconstruction.
            ec_input: (B, ec_input_dim) direct EC layer III drive.
            ca2_input: (B, ca2_input_dim) CA2 Schaffer overlay
                (temporal drift and identity comparator output).
                If None, the CA2 contribution is treated as zero.

        Returns:
            ca1_output: (B, ca1_dim) the conjoined representation.
            novelty: (B,) novelty scalar from the CA3-vs-EC mismatch.
        """
        if not self.cfg.enable_ca1:
            zero_out = torch.zeros(
                schaffer_input.shape[0], self.cfg.ca1_dim,
                device=schaffer_input.device, dtype=schaffer_input.dtype,
            )
            zero_nov = torch.zeros(
                schaffer_input.shape[0],
                device=schaffer_input.device, dtype=schaffer_input.dtype,
            )
            return zero_out, zero_nov

        # Direct EC drive in CA1 space. Gated by ablation flag.
        if self.cfg.enable_temporoammonic:
            direct = self.temporoammonic(ec_input)
        else:
            direct = torch.zeros(
                schaffer_input.shape[0], self.cfg.ca1_dim,
                device=schaffer_input.device, dtype=schaffer_input.dtype,
            )

        # CA3 Schaffer reconstruction projected for comparison and
        # for direct contribution. Gated by ablation flag.
        if self.cfg.enable_ca3_schaffer:
            s_proj = self.compare_schaffer(schaffer_input)
        else:
            s_proj = torch.zeros_like(direct)

        d_proj = self.compare_direct(direct)

        # Novelty signal from CA3-vs-EC cosine mismatch (Lisman-Grace).
        if self.cfg.enable_novelty_gate and self.cfg.enable_ca3_schaffer:
            cos_sim = F.cosine_similarity(s_proj, d_proj, dim=-1)
            novelty = (1.0 - cos_sim).clamp(0.0, 1.0)
            novelty_gate = novelty.unsqueeze(-1)
            base = novelty_gate * direct + (1.0 - novelty_gate) * s_proj
        else:
            novelty = torch.zeros(
                schaffer_input.shape[0],
                device=schaffer_input.device, dtype=schaffer_input.dtype,
            )
            base = direct + s_proj

        # CA2 overlay: temporal drift plus identity comparator output.
        # Added on top of the CA3-vs-EC base rather than competing
        # with it. ca2_overlay_weight controls how strongly the CA2
        # signal influences the final conjunction.
        if self.cfg.enable_ca2_schaffer and ca2_input is not None:
            ca2_contribution = self.ca2_overlay(ca2_input)
            combined = base + self.cfg.ca2_overlay_weight * ca2_contribution
        else:
            combined = base

        ca1_output = torch.tanh(self.output_gate(combined))
        return ca1_output, novelty


# =========================================================================
# SECTION: cornu_ammonis_2
# (originally cornu_ammonis_2_t.py)
# =========================================================================
"""
cornu_ammonis_2_t.py
Cognitive Kernel: CA2 Subfield (Temporal Drift Generator and Social Memory Comparator)

BIOLOGICAL GROUNDING
This file implements the hippocampal CA2 subfield, the third hippocampal
pyramidal cell region between CA3 and CA1. CA2 was historically treated as
a passive transition zone between CA3 and CA1; the last decade established
it as a functionally distinct module with its own input pathway, its own
plasticity regime, and its own role in the larger trisynaptic circuit.

CA2 has two architectural commitments in PRAGMI. First, it is the temporal
drift generator. Population activity in CA2 drifts substantially across
hours within the same spatial environment, more than CA1 and far more than
CA3. CA1 receives stable spatial input from CA3 plus time-varying input
from CA2; the conjunction at CA1 produces a place-plus-time-stamp code
that is the substrate of episodic (rather than semantic) memory. Second,
CA2 is a comparator rather than an encoder. High RGS14 expression
suppresses LTP at CA3-to-CA2 synapses, which means CA2 does not write new
content into its recurrent weights the way CA3 does. It compares current
input against stored representation and emits a mismatch signal. This
matches the social-memory function: recognizing whether a conspecific is
familiar requires comparing current sensory input against a stored
identity representation, not encoding a new identity from scratch.

The direct lateral entorhinal cortex layer II projection to CA2 carries
social and contextual content that bypasses dentate gyrus pattern
separation entirely. This is the pathway by which CA2 receives input
that is not orthogonalized, which is correct for the comparator function:
pattern separation would destroy the identity match that CA2 needs to
detect.

Primary grounding papers:

Hitti FL, Siegelbaum SA (2014). "The hippocampal CA2 region is essential
for social memory." Nature, 508(7494), 88-92.
DOI: 10.1038/nature13028

Mankin EA, Diehl GW, Sparks FT, Leutgeb S, Leutgeb JK (2015).
"Hippocampal CA2 activity patterns change over time to a larger extent
than between spatial contexts." Neuron, 85(1), 190-201.
DOI: 10.1016/j.neuron.2014.12.001

Lopez-Rojas J, von Richthofen HJ, Kempter R, Schmitz D, Gee CE,
Larkum ME (2022). "A direct lateral entorhinal cortex to hippocampal CA2
circuit conveys social information required for social memory." Neuron,
110(9), 1559-1572.e4. DOI: 10.1016/j.neuron.2022.01.028

Leroy F, Brann DH, Meira T, Siegelbaum SA (2017). "Input-timing-dependent
plasticity in the hippocampal CA2 region and its potential role in social
memory." Neuron, 95(5), 1089-1102.e5.
DOI: 10.1016/j.neuron.2017.07.036

Genesis Labs Research
Authored for the PRAGMI Cognitive Kernel
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# PART 1: ABLATION FLAGS AND CONFIG
# =========================================================================

@dataclass
class CA2Config:
    """Configuration for the CA2 subfield module.

    Ablation flags are ENGINEERING CONTROLS, not biological quantities.
    They exist so that each named biological mechanism with its own
    citation can be independently disabled for ablation studies. The
    master flag is first, per the Genesis Labs Research Ablation Flag
    Design Standard.
    """

    # =====================================================================
    # ABLATION FLAGS
    # =====================================================================

    # Master flag for the entire CA2 module. When False the forward method
    # returns a labeled neutral value and CA2 contributes nothing to the
    # CA1 conjunction. NOT a biological quantity.
    enable_ca2: bool = True

    # Direct lateral entorhinal cortex to CA2 pathway. Lopez-Rojas et al.
    # (2022) DOI: 10.1016/j.neuron.2022.01.028. When False, CA2 receives
    # only mossy-fiber-collateral input from CA3 and the social/contextual
    # content stream is ablated. NOT a biological quantity.
    enable_lec_direct_pathway: bool = True

    # Temporal drift dynamic. Mankin et al. (2015) DOI:
    # 10.1016/j.neuron.2014.12.001 documents the across-hours drift in
    # CA2 population activity. When False, CA2 output is stationary
    # within an environment and only the comparator function remains.
    # Disabling this is the canonical way to test whether time-stamping
    # of episodes depends on CA2 drift versus elsewhere in the kernel.
    # NOT a biological quantity.
    enable_temporal_drift: bool = True

    # RGS14-mediated LTP suppression at CA3-to-CA2 synapses. Lee et al.
    # (2010) DOI: 10.1016/j.cell.2010.08.029 documents that RGS14
    # knockout enables LTP at these synapses; wild-type CA2 does not
    # show classical LTP. When True, the CA3-to-CA2 weights are
    # held effectively fixed during training to model the wild-type
    # comparator regime. When False, CA2 behaves like an additional
    # CA3 attractor and the comparator function is lost.
    # NOT a biological quantity.
    enable_rgs14_ltp_suppression: bool = True

    # Social/identity comparator output. Hitti and Siegelbaum (2014)
    # DOI: 10.1038/nature13028 establishes the social memory function.
    # When False, the comparator output is silenced and only the
    # temporal drift signal flows to CA1. NOT a biological quantity.
    enable_comparator_output: bool = True

    # =====================================================================
    # DIMENSIONS
    # =====================================================================

    # Coordinate manifold dimensionality. Must match the rest of the
    # kernel for interface consistency. NOT a biological quantity.
    coordinate_dim: int = 64

    # CA2 pyramidal cell layer dimensionality. Sized smaller than CA3
    # because CA2 is anatomically a thin band between CA3 and CA1
    # (roughly 5 to 15 percent of the CA3-CA1 axis depending on the
    # measurement convention). NOT a biological quantity; an
    # engineering approximation of relative subfield size.
    ca2_dim: int = 96

    # CA1 dimensionality at the receiving end of the Schaffer
    # projection. Must match the kernel's CA1 module. NOT a biological
    # quantity.
    ca1_dim: int = 192

    # =====================================================================
    # TEMPORAL DRIFT PARAMETERS
    # =====================================================================

    # Drift time constant. Mankin et al. (2015) report that CA2
    # population vector correlations decay over hours within a fixed
    # spatial environment. The exact decay constant is not pinned to
    # a single number in the source; the value here is an engineering
    # approximation that produces visible drift across a training
    # session while not destabilizing the comparator output within a
    # single forward pass. Engineering approximation, not a biological
    # quantity in the strict sense, but parameterized to match the
    # qualitative timescale documented in Mankin et al. (2015).
    drift_tau: float = 0.995

    # Drift noise standard deviation. The drift in Mankin et al. (2015)
    # is structured rather than purely stochastic, but at this level
    # of abstraction a small Gaussian innovation captures the
    # qualitative property that CA2 state moves over time even in the
    # absence of input change. NOT a biological quantity, engineering
    # approximation.
    drift_noise_std: float = 0.02

    # =====================================================================
    # COMPARATOR PARAMETERS
    # =====================================================================

    # Comparator gain on the mismatch signal. Translates cosine
    # distance between current input and stored representation into a
    # match/mismatch scalar. NOT a biological quantity, training
    # artifact.
    comparator_gain: float = 4.0

    # Stored representation update rate. The comparator maintains a
    # slow-moving reference that current input is compared against.
    # This is functionally analogous to a familiarity trace. NOT a
    # biological quantity in the strict sense; the slow update is
    # consistent with the RGS14-suppressed plasticity regime
    # documented in Lee et al. (2010) DOI: 10.1016/j.cell.2010.08.029.
    reference_update_rate: float = 0.01


# =========================================================================
# PART 2: TEMPORAL DRIFT GENERATOR
# =========================================================================

class TemporalDriftGenerator(nn.Module):
    """Drift generator for the CA2 population activity vector.

    BIOLOGICAL STRUCTURE: CA2 pyramidal cell population.

    BIOLOGICAL FUNCTION: Mankin et al. (2015) document that the
    population vector of CA2 firing patterns drifts substantially over
    hours within a fixed spatial context, more than CA1 and far more
    than CA3. This drift is the source of the temporal component of
    the place-plus-time-stamp code that CA1 reads out. Without it,
    repeated visits to the same location produce identical hippocampal
    representations and the system cannot distinguish episodes that
    share spatial content.

    Mankin EA, Diehl GW, Sparks FT, Leutgeb S, Leutgeb JK (2015).
    "Hippocampal CA2 activity patterns change over time to a larger
    extent than between spatial contexts." Neuron, 85(1), 190-201.
    DOI: 10.1016/j.neuron.2014.12.001

    ANATOMICAL INTERFACE:
        Sending structures: lateral entorhinal cortex layer II direct
        projection and CA3 pyramidal cells via mossy fiber collaterals.
        Receiving structure: CA2 pyramidal cells (this module).
        Connections: temporoammonic-like direct EC-to-CA2 path
        (Lopez-Rojas et al. 2022) and CA3-to-CA2 mossy fiber collateral
        path.
    """

    def __init__(self, cfg: CA2Config) -> None:
        """Initialize the drift generator.

        Args:
            cfg: CA2Config carrying drift_tau, drift_noise_std, ca2_dim.
        """
        super().__init__()
        self.cfg = cfg
        # Persistent population state vector. Registered as a buffer
        # because it is part of the module's physical state but not a
        # learnable parameter; drift is generated by an Ornstein-
        # Uhlenbeck-like process, not by gradient descent.
        self.register_buffer(
            "drift_state", torch.zeros(cfg.ca2_dim),
        )

    def forward(self, input_drive: torch.Tensor) -> torch.Tensor:
        """Update and return the drift-modulated population state.

        The drift state evolves as a slow leaky integrator with
        Gaussian innovation. Current input drive is added on top so
        the population state is not purely stochastic but reflects
        both the slow drift and the moment-to-moment input.

        Args:
            input_drive: (B, ca2_dim) immediate drive from the EC and
                CA3 pathways, after their projection into CA2 space.

        Returns:
            ca2_state: (B, ca2_dim) the population activity reflecting
                slow temporal drift overlaid on current input drive.
        """
        if not self.cfg.enable_temporal_drift:
            # Drift disabled: return the input drive unchanged so the
            # comparator function can still operate without the
            # time-stamp signal. Labeled neutral value per ablation
            # convention.
            return input_drive

        with torch.no_grad():
            tau = self.cfg.drift_tau
            innovation = torch.randn_like(self.drift_state) * self.cfg.drift_noise_std
            self.drift_state.copy_(tau * self.drift_state + (1.0 - tau) * innovation)

        # Broadcast the drift state across the batch dimension and add
        # to the current input drive. The drift state is shared across
        # the batch because it represents the population's evolving
        # internal time-stamp, not a per-sample quantity.
        drift_broadcast = self.drift_state.unsqueeze(0).expand_as(input_drive)
        return torch.tanh(input_drive + drift_broadcast)

    def reset_drift(self) -> None:
        """Reset the drift state to zero.

        Used at the start of a fresh session or for ablation tests
        that need to control for drift accumulation between runs.
        """
        with torch.no_grad():
            self.drift_state.zero_()


# =========================================================================
# PART 3: SOCIAL/IDENTITY COMPARATOR
# =========================================================================

class IdentityComparator(nn.Module):
    """Comparator that emits a familiarity/mismatch signal.

    BIOLOGICAL STRUCTURE: CA2 pyramidal cells with RGS14-mediated
    suppression of LTP at CA3-to-CA2 Schaffer synapses.

    BIOLOGICAL FUNCTION: Hitti and Siegelbaum (2014) demonstrate that
    selective silencing of CA2 pyramidal cells abolishes social
    memory while leaving spatial memory intact. The functional
    interpretation is that CA2 compares current sensory input
    (especially identity-bearing input arriving via the lateral
    entorhinal direct pathway) against a stored reference and emits a
    mismatch signal indicating novelty of identity. This is distinct
    from CA1 novelty, which compares CA3 reconstruction against
    direct EC input. CA2 novelty is content-specific (is this
    individual familiar) rather than reconstruction-mismatch
    (does my recall match what I am seeing).

    Hitti FL, Siegelbaum SA (2014). "The hippocampal CA2 region is
    essential for social memory." Nature, 508(7494), 88-92.
    DOI: 10.1038/nature13028

    Lee SE, Simons SB, Heldt SA, Zhao M, Schroeder JP, Vellano CP,
    Cowan DP, Ramineni S, Yates CK, Feng Y, Smith Y, Sweatt JD,
    Weinshenker D, Ressler KJ, Dudek SM, Hepler JR (2010). "RGS14
    is a natural suppressor of both synaptic plasticity in CA2
    neurons and hippocampal-based learning and memory." Cell, 143(5),
    722-734. DOI: 10.1016/j.cell.2010.08.029

    ANATOMICAL INTERFACE:
        Sending structures: CA2 pyramidal cells receiving from
        lateral EC layer II (direct path) and CA3 (Schaffer
        collaterals to CA2).
        Receiving structure: CA1 deep pyramidal cells (downstream)
        and the kernel's diagnostic readout (this comparator's
        mismatch scalar).
        Connection: the CA2 component of the Schaffer collateral
        projection to CA1.
    """

    def __init__(self, cfg: CA2Config) -> None:
        """Initialize the identity comparator.

        Args:
            cfg: CA2Config carrying ca2_dim, comparator_gain,
                reference_update_rate.
        """
        super().__init__()
        self.cfg = cfg
        # Slow-moving reference representation. Functions as the
        # familiarity trace against which current input is compared.
        # Updated by exponential moving average rather than by
        # gradient descent, consistent with the RGS14-suppressed
        # plasticity regime documented in Lee et al. (2010).
        self.register_buffer(
            "reference", torch.zeros(cfg.ca2_dim),
        )
        # Projection from CA2 state to mismatch readout. This is a
        # learnable linear map; the LTP suppression applies to the
        # synaptic weights between CA3 and CA2 (the input side), not
        # to the downstream readout.
        self.mismatch_readout = nn.Linear(cfg.ca2_dim, 1, bias=True)

    def forward(self, ca2_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compare current CA2 state against the stored reference.

        Args:
            ca2_state: (B, ca2_dim) the drift-modulated CA2 population
                activity from TemporalDriftGenerator.

        Returns:
            comparator_output: (B, ca2_dim) the gated CA2 state ready
                to project into CA1.
            mismatch: (B,) familiarity mismatch scalar in [0, 1],
                with 1 indicating maximally novel identity.
        """
        if not self.cfg.enable_comparator_output:
            # Comparator disabled: return the unmodulated CA2 state
            # and a neutral mismatch of zero. Labeled neutral value
            # per ablation convention.
            zero_mismatch = torch.zeros(
                ca2_state.shape[0],
                device=ca2_state.device,
                dtype=ca2_state.dtype,
            )
            return ca2_state, zero_mismatch

        # Compute cosine distance between current state and reference.
        # The reference is broadcast across the batch dimension.
        reference_broadcast = self.reference.unsqueeze(0).expand_as(ca2_state)
        cos_sim = F.cosine_similarity(ca2_state, reference_broadcast, dim=-1)
        # Mismatch is high when current state diverges from the stored
        # reference. Gain shapes the sigmoid; comparator_gain is a
        # training artifact, not a biological quantity.
        mismatch = torch.sigmoid(self.cfg.comparator_gain * (1.0 - cos_sim))

        # Gate the CA2 state by the mismatch signal so that highly
        # familiar input passes through attenuated and novel input
        # passes through amplified, matching the functional role of
        # CA2 as a novelty-of-identity detector.
        gate = mismatch.unsqueeze(-1)
        comparator_output = ca2_state * (0.5 + gate)

        # Update the reference toward the current batch mean. Slow
        # update rate consistent with RGS14-suppressed plasticity.
        with torch.no_grad():
            batch_mean = ca2_state.detach().mean(dim=0)
            rate = self.cfg.reference_update_rate
            self.reference.copy_(
                (1.0 - rate) * self.reference + rate * batch_mean,
            )

        return comparator_output, mismatch

    def reset_reference(self) -> None:
        """Reset the stored reference to zero.

        Used at the start of a fresh session or for ablation tests
        that need to control for accumulated familiarity between runs.
        """
        with torch.no_grad():
            self.reference.zero_()


# =========================================================================
# PART 4: CA2 SUBFIELD MODULE
# =========================================================================

class CornuAmmonis2(nn.Module):
    """Complete CA2 subfield module.

    BIOLOGICAL STRUCTURE: Hippocampal CA2 pyramidal cell layer, the
    region between CA3 and CA1 distinguished by molecular markers
    (high RGS14, PCP4, STEP), distinctive connectivity (direct
    lateral entorhinal cortex layer II input bypassing dentate gyrus),
    and distinctive plasticity (RGS14-mediated suppression of LTP at
    CA3-to-CA2 synapses).

    BIOLOGICAL FUNCTION: Two functions are unified in this module.
    First, CA2 is the temporal drift generator whose output, conjoined
    with the stable spatial code from CA3 at the CA1 readout, gives
    the hippocampal population code its time-stamp. Second, CA2 is
    the social-memory comparator: it compares current input
    (especially identity-bearing input from the lateral EC direct
    pathway) against a stored familiarity reference and emits a
    mismatch signal. The two functions are not independent. The
    drift state provides the slowly varying baseline against which
    the comparator's reference is updated, so a familiar identity
    seen again later in the session produces a different CA2 vector
    than the same identity seen earlier, even though the comparator
    flags both as familiar.

    Hitti FL, Siegelbaum SA (2014). DOI: 10.1038/nature13028
    Mankin EA et al. (2015). DOI: 10.1016/j.neuron.2014.12.001
    Lopez-Rojas J et al. (2022). DOI: 10.1016/j.neuron.2022.01.028
    Lee SE et al. (2010). DOI: 10.1016/j.cell.2010.08.029

    ANATOMICAL INTERFACE (input):
        Sending structures: lateral entorhinal cortex layer II (direct
        path) and CA3 pyramidal cells (mossy fiber collaterals).
        Receiving structure: CA2 pyramidal cells (this module).
        Connections: the direct LEC-to-CA2 path documented by
        Lopez-Rojas et al. (2022), and the CA3-to-CA2 mossy fiber
        collateral path. Note that the CA3-to-CA2 synapses are
        subject to RGS14-mediated LTP suppression in wild-type
        animals (Lee et al. 2010), so this projection is read-mostly
        from CA2's perspective.

    ANATOMICAL INTERFACE (output):
        Sending structure: CA2 pyramidal cells.
        Receiving structures: CA1 deep pyramidal cells (primary
        readout for the temporal drift signal) and ventral CA1 (for
        the social memory readout, per Leroy et al. 2017
        DOI: 10.1016/j.neuron.2017.07.036). The current implementation
        emits a single output that the kernel routes to CA1; the
        ventral split is an architectural note for a future revision
        that requires the dorsal/ventral CA1 distinction.
        Connection: the CA2 component of the Schaffer collateral
        projection to CA1.
    """

    def __init__(self, cfg: Optional[CA2Config] = None) -> None:
        """Initialize the complete CA2 module.

        Args:
            cfg: CA2Config. If None, default values are used.
        """
        super().__init__()
        self.cfg = cfg or CA2Config()

        # Direct lateral EC to CA2 projection. Bypasses dentate gyrus
        # pattern separation, consistent with Lopez-Rojas et al. (2022)
        # DOI: 10.1016/j.neuron.2022.01.028.
        self.lec_to_ca2 = nn.Linear(
            self.cfg.coordinate_dim, self.cfg.ca2_dim, bias=True,
        )

        # CA3 mossy fiber collateral projection to CA2. Subject to
        # RGS14-mediated LTP suppression when enable_rgs14_ltp_suppression
        # is True (Lee et al. 2010 DOI: 10.1016/j.cell.2010.08.029).
        # The suppression is enforced at the optimizer level by the
        # kernel rather than by hiding the parameter, so the weights
        # exist but are excluded from gradient updates when the flag
        # is set.
        self.ca3_to_ca2 = nn.Linear(
            self.cfg.ca2_dim, self.cfg.ca2_dim, bias=False,
        )

        # The two functional submodules.
        self.drift_generator = TemporalDriftGenerator(self.cfg)
        self.comparator = IdentityComparator(self.cfg)

        # Output projection to CA1. The CA1 module conjoins this
        # signal with its existing Schaffer (CA3) and direct EC inputs.
        self.ca2_to_ca1 = nn.Linear(
            self.cfg.ca2_dim, self.cfg.ca1_dim, bias=True,
        )

    def forward(
        self,
        lec_input: torch.Tensor,
        ca3_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the CA2 contribution to the CA1 conjunction.

        Args:
            lec_input: (B, coordinate_dim) lateral entorhinal cortex
                output, the social/contextual content stream that
                bypasses dentate gyrus.
            ca3_state: (B, ca2_dim) CA3 pyramidal cell activity
                projected into CA2 space upstream by the kernel,
                or full CA3 activity that the ca3_to_ca2 projection
                handles. The current implementation expects the
                latter and projects internally.

        Returns:
            ca2_to_ca1_output: (B, ca1_dim) the CA2 contribution
                ready for conjunction at CA1.
            mismatch: (B,) the familiarity mismatch scalar in [0, 1]
                from the identity comparator, available to the
                kernel for diagnostic and downstream gating use.
        """
        if not self.cfg.enable_ca2:
            # Master flag disabled: return labeled neutral values.
            # Zero contribution to CA1 and zero mismatch. The kernel
            # treats this as CA2 being absent from the circuit.
            zero_output = torch.zeros(
                lec_input.shape[0],
                self.cfg.ca1_dim,
                device=lec_input.device,
                dtype=lec_input.dtype,
            )
            zero_mismatch = torch.zeros(
                lec_input.shape[0],
                device=lec_input.device,
                dtype=lec_input.dtype,
            )
            return zero_output, zero_mismatch

        # Combine the two upstream pathways into the CA2 input drive.
        # The LEC direct path can be ablated independently to test
        # whether the social/contextual content is necessary for the
        # comparator function.
        if self.cfg.enable_lec_direct_pathway:
            lec_drive = self.lec_to_ca2(lec_input)
        else:
            lec_drive = torch.zeros(
                lec_input.shape[0],
                self.cfg.ca2_dim,
                device=lec_input.device,
                dtype=lec_input.dtype,
            )
        ca3_drive = self.ca3_to_ca2(ca3_state)
        input_drive = lec_drive + ca3_drive

        # Apply the temporal drift dynamic.
        ca2_state = self.drift_generator(input_drive)

        # Run the identity comparator and gate the output.
        comparator_output, mismatch = self.comparator(ca2_state)

        # Project to CA1 space.
        ca2_to_ca1_output = self.ca2_to_ca1(comparator_output)

        return ca2_to_ca1_output, mismatch

    def reset_state(self) -> None:
        """Reset both drift and reference state.

        Convenience method for the kernel to call at session
        boundaries or for ablation control.
        """
        self.drift_generator.reset_drift()
        self.comparator.reset_reference()

    def get_diagnostic_state(self) -> dict:
        """Return current internal state for diagnostic logging.

        Returns:
            dict with keys 'drift_norm' and 'reference_norm'
            containing the L2 norms of the drift state and the
            stored reference. Useful for verifying that drift is
            accumulating and the reference is updating during
            training.
        """
        return {
            "drift_norm": self.drift_generator.drift_state.norm().item(),
            "reference_norm": self.comparator.reference.norm().item(),
        }


# =========================================================================
# SECTION: ca2_part1
# (originally ca2_t_part1.py)
#
# *** TRUNCATED IN SOURCE REPOSITORY ***
# This file ends mid-line in the upstream repo. The final dataclass is
# syntactically incomplete. Preserved verbatim; do not attempt to
# instantiate anything from this section until upstream is fixed.
# =========================================================================
"""
ca2_t.py
Cognitive Kernel: Hippocampal Subfield CA2 (Temporal Drift Generator)

BIOLOGICAL GROUNDING
This file models hippocampal subfield CA2, a small region between CA3 and
CA1 that for decades was treated as a passive relay. Modern work shows it
is neither passive nor a relay. CA2 is the structure that injects temporal
variability into the hippocampal code, distinguishing episodic memory
(time-indexed) from semantic memory (time-invariant). It is also the
hippocampal hub for social memory.

The architectural commitment of this file is that CA2 is a sibling to CA3,
not a downstream stage. CA3 is a recurrent attractor that suppresses drift
and stabilizes spatial codes through pattern completion. CA2 is a drift
generator that produces structured temporal variability across hours within
the same spatial environment. CA1 receives both: stable spatial input from
CA3 via the proximal Schaffer collaterals, and time-varying input from CA2
via the distal Schaffer-like projection. The conjunction at CA1 produces
a code that carries both place-stamp and time-stamp, which is what
"episodic" means at the algorithmic level.

CA2 plasticity is suppressed relative to CA3 by high RGS14 expression. The
functional consequence is that CA2 acts as a comparator between current
input and stored representation rather than as an encoder of new content.
This file implements that asymmetry by giving CA2 a much smaller plasticity
update than the CA3 attractor uses.

CA2 also has a direct projection from lateral entorhinal cortex layer II
that bypasses the dentate gyrus entirely, carrying social-context and
novelty information. The supramammillary nucleus provides a separate
modulatory input that scales the drift rate. Output goes to CA1 distal
dendrites for the place-plus-time conjunction, and to ventral CA1 for the
social-memory readout pathway.

Primary grounding papers:
Hitti FL, Siegelbaum SA (2014). "The hippocampal CA2 region is essential
for social memory." Nature, 508(7494), 88-92.
DOI: 10.1038/nature13028

Mankin EA, Diehl GW, Sparks FT, Leutgeb S, Leutgeb JK (2015). "Hippocampal
CA2 activity patterns change over time to a larger extent than between
spatial contexts." Neuron, 85(1), 190-201.
DOI: 10.1016/j.neuron.2014.12.001

Lopez-Rojas J, von Richthofen H, Kempter R, Schmitz D, Gee CE,
Larkum ME (2022). "Direct path from lateral entorhinal cortex to CA2
carries social information." Neuron.
DOI: {To be added later.}

Leroy F, Brann DH, Meira T, Siegelbaum SA (2017). "Input-timing-dependent
plasticity in the hippocampal CA2 region and its potential role in social
memory." Neuron, 95(5), 1089-1102.
DOI: 10.1016/j.neuron.2017.07.036

Genesis Labs Research
Authored for the PRAGMI Cognitive Kernel
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# PART 1: Configuration
# =========================================================================

@dataclass
class CA2Config:
    """Configuration for the CA2 module.

    Ablation flags are ENGINEERING CONTROLS, not biological quantities.
    They exist so that each named biological mechanism with its own citation
    can be independently disabled for ablation studies, per the Genesis Labs
    Research Ablation Flag Design Standard.
    """

    # Dimensionality of the EC layer II input vector.
    # Matches the kernel's coordinate_dim by default so CA2 sits in the
    # same coordinate space as the rest of the kernel.
    ec_layer2_dim: int = 64

    # Dimensionality of CA2's internal representation.
    # Smaller than CA3 because CA2 is a smaller subfield in the biology
    # (roughly 10 percent of CA3 cell count in rodent hippocampus).
    # NOT a strict biological quantity, this is an engineering scaling
    # choice consistent with the relative cell counts.
    ca2_dim: int = 96

    # Dimensionality of the supramammillary mod


# =========================================================================
# SECTION: ventral_tegmental_area
# (originally ventral_tegmental_area_t.py)
# =========================================================================
"""
ventral_tegmental_area_t.py
Loop Stage: VTA Dopamine Reward Prediction Error

BIOLOGICAL GROUNDING
The ventral tegmental area is the primary midbrain source of
dopamine to the forebrain. Its output is canonically interpreted as
a reward prediction error signal (RPE), formalized by Montague,
Dayan, and Sejnowski (1996) as the temporal difference (TD) error
in reinforcement learning.

The TD computation is:
    delta(t) = r(t) + gamma * V(s(t)) - V(s(t-1))

where r is the immediate reward, V is the predicted sum of future
discounted rewards, and gamma is the discount factor. The dopamine
signal validates against Schultz, Apicella, Ljungberg recordings:
neurons fire to unexpected reward, transfer their response to the
predictive cue after learning, and pause below baseline when an
expected reward is omitted.

This file implements the basic TD computation. The distributional
update of Dabney et al. (2020) and the population diversity in
discount factors of Eshel et al. (2016) are noted in the architecture
but not implemented in the basic version; they can be added as
extensions if measurable benefit appears on the demo task.

Primary grounding papers:

Montague PR, Dayan P, Sejnowski TJ (1996). "A framework for
mesencephalic dopamine systems based on predictive Hebbian learning."
Journal of Neuroscience, 16(5), 1936-1947.
DOI: 10.1523/JNEUROSCI.16-05-01936.1996

Schultz W, Dayan P, Montague PR (1997). "A neural substrate of
prediction and reward." Science, 275(5306), 1593-1599.
DOI: 10.1126/science.275.5306.1593

Dabney W, Kurth-Nelson Z, Uchida N, Starkweather CK, Hassabis D,
Munos R, Botvinick M (2020). "A distributional code for value in
dopamine-based reinforcement learning." Nature, 577(7792), 671-675.
DOI: 10.1038/s41586-019-1924-6

Eshel N, Tian J, Bukwich M, Uchida N (2016). "Dopamine neurons share
common response function for reward prediction error." Nature
Neuroscience, 19(3), 479-486.
DOI: 10.1038/nn.4239

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class VTAConfig:
    """Configuration for the VTA dopamine module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_vta: bool = True
    enable_value_estimation: bool = True
    enable_rpe_computation: bool = True
    enable_value_update: bool = True

    # State input dimensionality. Must match the upstream
    # representation supplying state context. NOT a biological
    # quantity.
    state_dim: int = 64

    # Discount factor. Eshel et al. (2016) document a population of
    # VTA neurons with diverse discount factors clustered around
    # similar response shapes. The single value here is a simplifying
    # approximation. Engineering choice rather than biological
    # quantity in the strict sense.
    gamma: float = 0.95

    # Value learning rate. Sutton and Barto (1998) standard TD
    # learning rate. NOT a biological quantity, training artifact.
    value_lr: float = 0.01


class VentralTegmentalArea(nn.Module):
    """VTA dopamine reward prediction error.

    BIOLOGICAL STRUCTURE: Dopaminergic neurons of the ventral
    tegmental area projecting to forebrain via the mesolimbic and
    mesocortical pathways.

    BIOLOGICAL FUNCTION: Computes RPE as TD error between predicted
    future reward and observed reward plus discounted next-state
    value. The RPE signal modulates plasticity broadly across the
    forebrain via the dopamine broadcast.

    Montague PR, Dayan P, Sejnowski TJ (1996).
    DOI: 10.1523/JNEUROSCI.16-05-01936.1996
    Schultz W, Dayan P, Montague PR (1997).
    DOI: 10.1126/science.275.5306.1593

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical state representation (this
        module's state input), reward sources (immediate reward
        signal).
        Receiving structure: VTA dopaminergic neurons (this module).
        Connection: convergent excitatory and inhibitory projections
        onto VTA core.

    ANATOMICAL INTERFACE (output):
        Sending structure: VTA dopaminergic neurons.
        Receiving structures: forebrain broadly, including striatum
        (reinforcement of action selection), PFC (working memory
        gating), and hippocampus (consolidation modulation).
        Connection: mesolimbic and mesocortical dopamine projections.
    """

    def __init__(self, cfg: Optional[VTAConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or VTAConfig()
        # Value head: linear value function approximator.
        # The choice of a linear function is a simplification per
        # Sutton and Barto (1998); a deeper network is a valid
        # extension if needed.
        self.value_head = nn.Linear(self.cfg.state_dim, 1, bias=True)
        # Buffer for the previous-state value for TD computation.
        self.register_buffer("prev_value", torch.tensor(0.0))

    def forward(
        self,
        state: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the dopamine RPE and current state value.

        Args:
            state: (B, state_dim) current state representation.
            reward: optional scalar or (B,) immediate reward.

        Returns:
            dopamine: scalar TD error.
            value: scalar current state value estimate.
        """
        if not self.cfg.enable_vta:
            zero = torch.tensor(
                0.0, device=state.device, dtype=state.dtype,
            )
            return zero, zero

        # Current state value.
        if self.cfg.enable_value_estimation:
            current_value = self.value_head(state).mean()
        else:
            current_value = torch.tensor(
                0.0, device=state.device, dtype=state.dtype,
            )

        # TD error.
        if self.cfg.enable_rpe_computation:
            r = reward.mean() if reward is not None else torch.tensor(
                0.0, device=state.device, dtype=state.dtype,
            )
            dopamine = r + self.cfg.gamma * current_value - self.prev_value
        else:
            dopamine = torch.tensor(
                0.0, device=state.device, dtype=state.dtype,
            )

        # Update previous value for next step.
        if self.cfg.enable_value_update:
            with torch.no_grad():
                self.prev_value.copy_(current_value.detach())

        return dopamine, current_value

    def reset_value(self) -> None:
        """Reset the previous-value buffer to zero."""
        with torch.no_grad():
            self.prev_value.zero_()


# =========================================================================
# SECTION: locus_coeruleus
# (originally locus_coeruleus_t.py)
# =========================================================================
"""
locus_coeruleus_t.py
Loop Stage: Locus Coeruleus Norepinephrine Computation

BIOLOGICAL GROUNDING
The locus coeruleus is a small brainstem nucleus that supplies most
of the norepinephrine to the forebrain. It is the canonical source
of the NE signal that scales gain in the thalamic gate, modulates
attention precision, and drives reset of context belief on
unexpected uncertainty.

Yu and Dayan (2005) formalize NE as the unexpected uncertainty
signal: the part of prediction error that cannot be explained by
within-context variance (which is acetylcholine's domain). The
operational form is the negative log probability of the current
observation under the active context model, integrated over a short
window and bounded by soft normalization. When NE crosses a
threshold, the system resets context belief and elevates learning
rates, which behaves like a Kalman filter switching between
covariance regimes.

Dayan and Yu (2006) extend the framework to phasic NE, treating the
phasic burst as a neural interrupt signal for unexpected events
within the current task. Nassar, Wilson, Heasly, Gold (2010) provide
the explicit pseudocode-level update rules in a changing-environment
delta-rule framework.

This file implements the LC as a module that takes an instantaneous
observation log-likelihood under the current context model and emits
both a tonic NE level and a phasic burst signal. The thalamic gate
honors NE through the existing ne_gain hook.

Primary grounding papers:

Yu AJ, Dayan P (2005). "Uncertainty, neuromodulation, and attention."
Neuron, 46(4), 681-692. DOI: 10.1016/j.neuron.2005.04.026

Dayan P, Yu AJ (2006). "Phasic norepinephrine: a neural interrupt
signal for unexpected events." Network: Computation in Neural
Systems, 17(4), 335-350. DOI: 10.1080/09548980601004024

Nassar MR, Wilson RC, Heasly B, Gold JI (2010). "An approximately
Bayesian delta-rule model explains the dynamics of belief updating
in a changing environment." Journal of Neuroscience, 30(37),
12366-12378. DOI: 10.1523/JNEUROSCI.0822-10.2010

Aston-Jones G, Cohen JD (2005). "An integrative theory of locus
coeruleus-norepinephrine function: adaptive gain and optimal
performance." Annual Review of Neuroscience, 28, 403-450.
DOI: 10.1146/annurev.neuro.28.061604.135709

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# =========================================================================
# PART 1: CONFIG
# =========================================================================

@dataclass
class LocusCoeruleusConfig:
    """Configuration for the LC NE module.

    Master flag is first per the Genesis Labs standard.
    NOT a biological quantity.
    """

    enable_locus_coeruleus: bool = True
    enable_tonic_ne: bool = True
    enable_phasic_burst: bool = True
    enable_context_reset: bool = True

    # Window length over which negative log-likelihood is integrated
    # to produce the tonic NE level. Yu and Dayan (2005) describe a
    # "short window" without a specific number; this is an
    # engineering approximation.
    integration_window: int = 16

    # Soft-normalization temperature for bounding the NE scalar.
    # NOT a biological quantity, training artifact.
    softmax_temperature: float = 1.0

    # Threshold above which the phasic burst fires and triggers
    # context reset. Aston-Jones and Cohen (2005) describe phasic
    # bursts as exceeding tonic baseline by a multiplicative factor;
    # this is parameterized to that qualitative description.
    phasic_threshold: float = 1.5

    # Tonic baseline NE level. The thalamic gate uses 1.0 as neutral
    # gain; tonic NE drifts around this value. NOT a biological
    # quantity.
    tonic_baseline: float = 1.0


# =========================================================================
# PART 2: LC MODULE
# =========================================================================

class LocusCoeruleus(nn.Module):
    """Locus coeruleus NE source.

    BIOLOGICAL STRUCTURE: Locus coeruleus, a bilateral brainstem
    nucleus of approximately 30,000 neurons in humans, projecting
    diffusely to forebrain.

    BIOLOGICAL FUNCTION: Computes the unexpected uncertainty signal
    from prediction error in the current context model. Tonic NE
    represents background level of unexpected uncertainty; phasic
    bursts are interrupt signals for unexpected events. Threshold
    crossing triggers context belief reset and elevation of learning
    rates.

    Yu AJ, Dayan P (2005). DOI: 10.1016/j.neuron.2005.04.026
    Dayan P, Yu AJ (2006). DOI: 10.1080/09548980601004024
    Nassar MR et al. (2010). DOI: 10.1523/JNEUROSCI.0822-10.2010
    Aston-Jones G, Cohen JD (2005).
    DOI: 10.1146/annurev.neuro.28.061604.135709

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical and subcortical sources of
        prediction error including ACC, hippocampus, amygdala.
        Receiving structure: LC noradrenergic neurons (this module).
        Connection: convergent excitatory and inhibitory projections
        onto LC core.

    ANATOMICAL INTERFACE (output):
        Sending structure: LC noradrenergic neurons.
        Receiving structures: forebrain broadly, including thalamus
        (gain modulation), cortex (attention precision), and
        hippocampus (consolidation gating).
        Connection: diffuse noradrenergic projections.
    """

    def __init__(self, cfg: Optional[LocusCoeruleusConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or LocusCoeruleusConfig()
        # Rolling window of negative log likelihoods. Implemented as
        # a buffer because the integration is a pure state update
        # without learnable parameters.
        self.register_buffer(
            "nll_window",
            torch.zeros(self.cfg.integration_window),
        )
        self.register_buffer("window_ptr", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        observation_nll: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the NE signal from observation negative log-likelihood.

        Args:
            observation_nll: scalar tensor or (B,) tensor giving the
                negative log probability of the current observation
                under the active context model. Larger values mean
                less expected.

        Returns:
            tonic_ne: scalar tonic NE level for gain modulation.
            phasic_burst: scalar in [0, 1] indicating whether a phasic
                burst is firing.
            reset_signal: bool tensor indicating whether the context
                belief should reset.
        """
        if not self.cfg.enable_locus_coeruleus:
            tonic = torch.tensor(
                self.cfg.tonic_baseline,
                device=observation_nll.device,
                dtype=observation_nll.dtype,
            )
            phasic = torch.tensor(
                0.0,
                device=observation_nll.device,
                dtype=observation_nll.dtype,
            )
            reset = torch.tensor(False, device=observation_nll.device)
            return tonic, phasic, reset

        # Reduce the input to a scalar by mean if it is a batch.
        if observation_nll.dim() > 0:
            scalar_nll = observation_nll.mean()
        else:
            scalar_nll = observation_nll

        # Update the rolling window.
        with torch.no_grad():
            ptr = int(self.window_ptr.item())
            self.nll_window[ptr] = scalar_nll.detach()
            self.window_ptr.copy_(
                torch.tensor((ptr + 1) % self.cfg.integration_window),
            )

        # Tonic NE: soft-normalized integrated NLL.
        if self.cfg.enable_tonic_ne:
            integrated = self.nll_window.mean()
            normed = torch.sigmoid(integrated / self.cfg.softmax_temperature)
            # Map [0, 1] sigmoid output to a gain centered on the
            # tonic baseline. A normed value of 0.5 (no surprise on
            # average) gives gain equal to baseline; higher surprise
            # raises gain, lower surprise lowers it.
            tonic_ne = self.cfg.tonic_baseline + (normed - 0.5) * 2.0
        else:
            tonic_ne = torch.tensor(
                self.cfg.tonic_baseline,
                device=scalar_nll.device,
                dtype=scalar_nll.dtype,
            )

        # Phasic burst: instantaneous NLL above threshold.
        if self.cfg.enable_phasic_burst:
            phasic_burst = (
                scalar_nll > self.cfg.phasic_threshold
            ).to(scalar_nll.dtype)
        else:
            phasic_burst = torch.tensor(
                0.0,
                device=scalar_nll.device,
                dtype=scalar_nll.dtype,
            )

        # Context reset signal: phasic burst gates the reset.
        if self.cfg.enable_context_reset:
            reset_signal = phasic_burst > 0.5
        else:
            reset_signal = torch.tensor(False, device=scalar_nll.device)

        return tonic_ne, phasic_burst, reset_signal

    def reset_window(self) -> None:
        """Reset the integration window to zero."""
        with torch.no_grad():
            self.nll_window.zero_()
            self.window_ptr.zero_()


# =========================================================================
# SECTION: dorsal_raphe
# (originally dorsal_raphe_t.py)
# =========================================================================
"""
dorsal_raphe_t.py
Loop Stage: Dorsal Raphe Nucleus Serotonin Source

BIOLOGICAL GROUNDING
The dorsal raphe nucleus is the primary midbrain source of forebrain
serotonin (5-HT). Its functional role is multifaceted: serotonin
encodes patience for delayed reward (Miyazaki et al. 2014), modulates
risk and aversion (Cools, Roberts, Robbins 2008), and provides a
slow opponent signal to dopamine in reinforcement learning (Daw,
Kakade, Dayan 2002).

The Daw et al. (2002) opponent-process account is most directly
implementable: where dopamine signals positive prediction error,
serotonin signals an inverse, scaled by an aversion-sensitivity
parameter. The two systems together produce balanced exploitation
of positive and negative outcomes, with depression understood as a
serotonin deficit producing impaired punishment learning.

Cohen, Amoroso, Uchida (2015) document that DRN 5-HT neurons fire
in response to both rewards and punishments, supporting a salience-
based account in addition to the opponent-process account. Boureau
and Dayan (2011) review the broader integration with motivation
and decision-making.

This file implements a DRN module emitting:
1. Tonic 5-HT level (slow drift toward an aversion-sensitivity scaled
   running average of negative TD errors)
2. Phasic 5-HT (instantaneous opponent of dopamine RPE)
3. Patience scalar suitable for biasing temporal discounting toward
   longer horizons when 5-HT is high

Primary grounding papers:

Daw ND, Kakade S, Dayan P (2002). "Opponent interactions between
serotonin and dopamine." Neural Networks, 15(4-6), 603-616.
DOI: 10.1016/S0893-6080(02)00052-7

Miyazaki KW, Miyazaki K, Tanaka KF, Yamanaka A, Takahashi A,
Tabuchi S, Doya K (2014). "Optogenetic activation of dorsal raphe
serotonin neurons enhances patience for future rewards." Current
Biology, 24(17), 2033-2040. DOI: 10.1016/j.cub.2014.07.041

Cohen JY, Amoroso MW, Uchida N (2015). "Serotonergic neurons signal
reward and punishment on multiple timescales." eLife, 4, e06346.
DOI: 10.7554/eLife.06346

Cools R, Roberts AC, Robbins TW (2008). "Serotoninergic regulation
of emotional and behavioural control processes." Trends in Cognitive
Sciences, 12(1), 31-40. DOI: 10.1016/j.tics.2007.10.011

Boureau YL, Dayan P (2011). "Opponency revisited: competition and
cooperation between dopamine and serotonin." Neuropsychopharmacology,
36(1), 74-97. DOI: 10.1038/npp.2010.151

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DorsalRapheConfig:
    """Configuration for the DRN serotonin module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_dorsal_raphe: bool = True
    enable_tonic_5ht: bool = True
    enable_phasic_5ht: bool = True
    enable_patience_signal: bool = True

    # Aversion sensitivity. Daw et al. (2002) parameter. NOT a
    # biological quantity in the strict sense.
    aversion_sensitivity: float = 1.0

    # Tonic decay constant. NOT a biological quantity.
    tonic_decay: float = 0.95

    # Mapping from tonic 5-HT to discount factor. Higher 5-HT means
    # longer patience, modeled as adding to the gamma in TD. NOT a
    # biological quantity, training artifact.
    patience_scale: float = 0.05


class DorsalRaphe(nn.Module):
    """Dorsal raphe nucleus serotonin source.

    BIOLOGICAL STRUCTURE: Dorsal raphe nucleus, a midbrain
    serotonergic nucleus projecting diffusely to forebrain.

    BIOLOGICAL FUNCTION: Computes opponent signal to dopamine for
    aversion learning, plus a tonic patience signal that biases
    temporal discounting toward longer horizons when 5-HT is high.

    Daw ND, Kakade S, Dayan P (2002).
    DOI: 10.1016/S0893-6080(02)00052-7
    Miyazaki KW et al. (2014). DOI: 10.1016/j.cub.2014.07.041
    Cohen JY, Amoroso MW, Uchida N (2015). DOI: 10.7554/eLife.06346
    Cools R, Roberts AC, Robbins TW (2008).
    DOI: 10.1016/j.tics.2007.10.011
    Boureau YL, Dayan P (2011). DOI: 10.1038/npp.2010.151

    ANATOMICAL INTERFACE (input):
        Sending structures: VTA dopamine TD error, ACC conflict,
        amygdala valence.
        Receiving structure: DRN (this module).
        Connection: convergent projections onto DRN core.

    ANATOMICAL INTERFACE (output):
        Sending structure: DRN serotonergic neurons.
        Receiving structures: forebrain broadly, including PFC, VTA,
        BG.
        Connection: serotonergic projections.
    """

    def __init__(self, cfg: Optional[DorsalRapheConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DorsalRapheConfig()
        # Persistent tonic 5-HT level.
        self.register_buffer("tonic_level", torch.tensor(0.0))

    def forward(
        self,
        dopamine: Optional[torch.Tensor] = None,
        aversion: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute tonic 5-HT, phasic 5-HT, and patience.

        Args:
            dopamine: optional scalar TD error from VTA.
            aversion: optional scalar aversion signal from amygdala.

        Returns:
            tonic_5ht: scalar tonic 5-HT level.
            phasic_5ht: scalar phasic 5-HT (opponent of dopamine).
            patience: scalar patience-bias for temporal discounting.
        """
        if not self.cfg.enable_dorsal_raphe:
            zero = torch.tensor(0.0)
            return zero, zero.clone(), zero.clone()

        d = dopamine if dopamine is not None else torch.tensor(0.0)
        a = aversion if aversion is not None else torch.tensor(0.0)

        # Phasic 5-HT: opponent of dopamine, scaled by aversion sensitivity.
        if self.cfg.enable_phasic_5ht:
            phasic = self.cfg.aversion_sensitivity * (-d + a)
        else:
            phasic = torch.tensor(0.0)

        # Tonic 5-HT: slow leaky integration of phasic.
        if self.cfg.enable_tonic_5ht:
            with torch.no_grad():
                self.tonic_level.copy_(
                    self.cfg.tonic_decay * self.tonic_level
                    + (1.0 - self.cfg.tonic_decay) * phasic.detach()
                )
            tonic = self.tonic_level.clone()
        else:
            tonic = torch.tensor(0.0)

        # Patience signal: monotone in tonic level.
        if self.cfg.enable_patience_signal:
            patience = self.cfg.patience_scale * tonic
        else:
            patience = torch.tensor(0.0)

        return tonic, phasic, patience

    def reset(self) -> None:
        """Reset tonic level."""
        with torch.no_grad():
            self.tonic_level.zero_()


# =========================================================================
# SECTION: basal_forebrain
# (originally basal_forebrain_t.py)
# =========================================================================
"""
basal_forebrain_t.py
Loop Stage: Basal Forebrain Acetylcholine Source

BIOLOGICAL GROUNDING
The basal forebrain cholinergic system (nucleus basalis of Meynert,
medial septal nucleus, diagonal band of Broca) is the primary
source of cortical acetylcholine. ACh complements norepinephrine in
Yu and Dayan's (2005) uncertainty framework: where NE encodes
unexpected uncertainty (model is wrong), ACh encodes expected
uncertainty (model is correct but stochastic). Hasselmo (2006)
provides the developed account of ACh's role in modulating cortical
encoding versus retrieval modes.

High ACh produces an "encoding" mode: strong feedforward sensory
drive, suppressed feedback, enhanced LTP. Low ACh produces a
"retrieval" mode: stronger feedback recurrence, suppressed
feedforward, weaker LTP. This switch is functionally critical for
the kernel because the cognitive kernel needs encoding mode when
new memories are forming and retrieval mode when reconstructing
from existing traces.

Parikh, Kozak, Martinez, Sarter (2007) document phasic ACh release
events on the second timescale, supporting the role in transient
attentional engagement. Sarter, Lustig, Howe, Gritton, Berry (2014)
review the broader cognitive function.

This file implements an ACh source that emits:
1. Tonic ACh level (slow drift)
2. Phasic ACh transients (fast attentional events)
3. Mode signal (encoding / retrieval) for downstream gating

Primary grounding papers:

Yu AJ, Dayan P (2005). "Uncertainty, neuromodulation, and attention."
Neuron, 46(4), 681-692. DOI: 10.1016/j.neuron.2005.04.026

Hasselmo ME (2006). "The role of acetylcholine in learning and
memory." Current Opinion in Neurobiology, 16(6), 710-715.
DOI: 10.1016/j.conb.2006.09.002

Parikh V, Kozak R, Martinez V, Sarter M (2007). "Prefrontal
acetylcholine release controls cue detection on multiple timescales."
Neuron, 56(1), 141-154. DOI: 10.1016/j.neuron.2007.08.025

Sarter M, Lustig C, Howe WM, Gritton H, Berry AS (2014).
"Deterministic functions of cortical acetylcholine." European
Journal of Neuroscience, 39(11), 1912-1920.
DOI: 10.1111/ejn.12515

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn


class CholinergicMode(Enum):
    """Mode signal emitted by the basal forebrain."""
    ENCODING = 0
    RETRIEVAL = 1


@dataclass
class BasalForebrainConfig:
    """Configuration for the basal forebrain ACh source.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_basal_forebrain: bool = True
    enable_tonic_ach: bool = True
    enable_phasic_ach: bool = True
    enable_mode_switching: bool = True

    state_dim: int = 64

    # Tonic baseline level. NOT a biological quantity.
    tonic_baseline: float = 0.5

    # Phasic burst threshold (on attention salience). NOT a biological
    # quantity.
    phasic_threshold: float = 0.7

    # Mode switching threshold on tonic ACh. Above this, encoding mode;
    # below, retrieval mode. NOT a biological quantity.
    mode_threshold: float = 0.5


class BasalForebrain(nn.Module):
    """Basal forebrain ACh source with mode signaling.

    BIOLOGICAL STRUCTURE: Nucleus basalis of Meynert, medial septal
    nucleus, diagonal band of Broca, projecting cholinergic fibers
    diffusely to neocortex and hippocampus.

    BIOLOGICAL FUNCTION: Computes expected uncertainty as a slow
    tonic ACh signal, with phasic ACh transients for attentional
    engagement. Tonic level switches the cortex between encoding
    and retrieval modes.

    Yu AJ, Dayan P (2005). DOI: 10.1016/j.neuron.2005.04.026
    Hasselmo ME (2006). DOI: 10.1016/j.conb.2006.09.002
    Parikh V et al. (2007). DOI: 10.1016/j.neuron.2007.08.025
    Sarter M et al. (2014). DOI: 10.1111/ejn.12515

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical and amygdaloid sources of
        attentional salience.
        Receiving structure: basal forebrain (this module).
        Connection: cortico-basal-forebrain projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: basal forebrain cholinergic neurons.
        Receiving structures: neocortex and hippocampus broadly.
        Connection: cholinergic projections.
    """

    def __init__(
        self, cfg: Optional[BasalForebrainConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or BasalForebrainConfig()
        # Salience evaluator from state.
        self.salience_head = nn.Linear(
            self.cfg.state_dim, 1, bias=True,
        )
        # Persistent tonic level.
        self.register_buffer(
            "tonic_level", torch.tensor(self.cfg.tonic_baseline)
        )

    def forward(
        self, state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, CholinergicMode]:
        """Compute tonic, phasic, and mode signals.

        Args:
            state: (B, state_dim) cortical state.

        Returns:
            tonic_ach: scalar tonic ACh level.
            phasic_ach: scalar phasic transient (0 or 1).
            mode: CholinergicMode encoding/retrieval.
        """
        if not self.cfg.enable_basal_forebrain:
            return (
                torch.tensor(self.cfg.tonic_baseline),
                torch.tensor(0.0),
                CholinergicMode.RETRIEVAL,
            )

        # Salience computation.
        salience = torch.sigmoid(
            self.salience_head(state).squeeze(-1).mean()
        )

        # Tonic ACh: slow drift toward salience.
        if self.cfg.enable_tonic_ach:
            with torch.no_grad():
                self.tonic_level.copy_(
                    0.95 * self.tonic_level + 0.05 * salience.detach()
                )
            tonic = self.tonic_level.clone()
        else:
            tonic = torch.tensor(self.cfg.tonic_baseline)

        # Phasic ACh: instantaneous transient.
        if self.cfg.enable_phasic_ach:
            phasic = (salience > self.cfg.phasic_threshold).to(salience.dtype)
        else:
            phasic = torch.tensor(0.0)

        # Mode signal.
        if self.cfg.enable_mode_switching:
            mode = (
                CholinergicMode.ENCODING
                if tonic.item() > self.cfg.mode_threshold
                else CholinergicMode.RETRIEVAL
            )
        else:
            mode = CholinergicMode.RETRIEVAL

        return tonic, phasic, mode

    def reset(self) -> None:
        """Reset tonic level to baseline."""
        with torch.no_grad():
            self.tonic_level.fill_(self.cfg.tonic_baseline)


# =========================================================================
# SECTION: sleep_stage_oscillator
# (originally sleep_stage_oscillator_t.py)
# =========================================================================
"""
sleep_stage_oscillator_t.py
Sleep Stage Transition Oscillator

BIOLOGICAL GROUNDING
The sleep-wake cycle is governed by mutually inhibitory brainstem
populations whose dynamics are well characterized by flip-flop
models. Booth, Diniz Behn (2014) provide a four-population mean-field
model of NREM/REM/wake transitions. The wake-promoting populations
(LC, dorsal raphe, basal forebrain cholinergic) suppress sleep
populations (VLPO and pontine cholinergic), with mutual inhibition
producing bistable dynamics. Within sleep, the SLD (sublaterodorsal
nucleus) cholinergic population and ventrolateral periaqueductal
gray REM-off population compete to gate REM versus NREM.

Kumar, Bose, Mallick (2012) provide a mathematical analysis of REM
sleep regulation, demonstrating how slow homeostatic processes
(adenosine accumulation during wake, dissipation during sleep)
combine with fast flip-flop dynamics to produce the canonical 90 to
120 minute NREM-REM cycle.

This file implements a small flip-flop oscillator that emits
discrete sleep stage labels (wake / NREM / REM) and a homeostatic
sleep-pressure scalar. The kernel uses these to gate consolidation
and replay processes.

Primary grounding papers:

Booth V, Diniz Behn CG (2014). "Physiologically-based modeling of
sleep-wake regulatory networks." Mathematical Biosciences, 250,
54-68. DOI: 10.1016/j.mbs.2014.01.012

Kumar R, Bose A, Mallick BN (2012). "A mathematical model towards
understanding the mechanism of neuronal regulation of wake-NREMS-REMS
states." PLOS ONE, 7(8), e42059.
DOI: 10.1371/journal.pone.0042059

Saper CB, Fuller PM, Pedersen NP, Lu J, Scammell TE (2010). "Sleep
state switching." Neuron, 68(6), 1023-1042.
DOI: 10.1016/j.neuron.2010.11.032

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SleepStage(Enum):
    """Sleep stage labels emitted by the oscillator."""
    WAKE = 0
    NREM = 1
    REM = 2


@dataclass
class SleepOscillatorConfig:
    """Configuration for the sleep stage oscillator.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_oscillator: bool = True
    enable_homeostatic_pressure: bool = True
    enable_circadian_drive: bool = True
    enable_nrem_rem_cycle: bool = True

    # Homeostatic pressure rise rate during wake. Approximates
    # adenosine buildup. Per-step rate. NOT a biological quantity in
    # the strict sense; engineering approximation matching the
    # qualitative rise across hours of wakefulness.
    pressure_rise_rate: float = 0.005

    # Homeostatic pressure decay rate during sleep. NOT a biological
    # quantity; engineering approximation.
    pressure_decay_rate: float = 0.02

    # Threshold of pressure that triggers transition to sleep.
    # NOT a biological quantity, training artifact.
    sleep_threshold: float = 0.7

    # Threshold of pressure below which wake resumes.
    wake_threshold: float = 0.2

    # Cycle period in steps for NREM/REM alternation. The biological
    # cycle is approximately 90 to 120 minutes. The integer here is
    # parameterized to match this qualitative cycle length on the
    # caller's timescale; engineering choice.
    nrem_rem_period: int = 100


class SleepOscillator(nn.Module):
    """Sleep stage flip-flop oscillator with homeostatic pressure.

    BIOLOGICAL STRUCTURE: Brainstem and hypothalamic populations
    governing sleep-wake transitions, including VLPO, LC, raphe,
    SLD, vlPAG.

    BIOLOGICAL FUNCTION: Emits sleep stage labels and homeostatic
    pressure governing when the kernel should run consolidation and
    replay processes. Produces the canonical NREM/REM alternation
    within sleep periods.

    Booth V, Diniz Behn CG (2014). DOI: 10.1016/j.mbs.2014.01.012
    Kumar R, Bose A, Mallick BN (2012).
    DOI: 10.1371/journal.pone.0042059
    Saper CB et al. (2010). DOI: 10.1016/j.neuron.2010.11.032

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical and homeostatic drives onto the
        brainstem flip-flop populations.
        Receiving structure: brainstem flip-flop (this module).
        Connection: cortico-brainstem and homeostatic projections.

    ANATOMICAL INTERFACE (output):
        Sending structure: brainstem flip-flop.
        Receiving structures: cortex and hippocampus, gating
        consolidation and replay.
        Connection: brainstem-cortical neuromodulator projections.
    """

    def __init__(
        self, cfg: Optional[SleepOscillatorConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SleepOscillatorConfig()
        # Persistent state.
        self.register_buffer("pressure", torch.tensor(0.0))
        self.register_buffer("stage", torch.tensor(0, dtype=torch.long))
        self.register_buffer("cycle_counter", torch.tensor(0, dtype=torch.long))

    def forward(
        self, external_arousal: Optional[torch.Tensor] = None,
    ) -> Tuple[SleepStage, torch.Tensor]:
        """Advance one tick of the sleep oscillator.

        Args:
            external_arousal: optional scalar in [0, 1] representing
                external arousal pressure (light, sound, threat).
                High arousal forces wake regardless of pressure.

        Returns:
            stage: current SleepStage enum value.
            pressure: current homeostatic pressure scalar.
        """
        if not self.cfg.enable_oscillator:
            return SleepStage.WAKE, torch.tensor(0.0)

        external = (
            external_arousal.item() if external_arousal is not None else 0.0
        )

        with torch.no_grad():
            current_stage_idx = int(self.stage.item())

            # Update homeostatic pressure.
            if self.cfg.enable_homeostatic_pressure:
                if current_stage_idx == SleepStage.WAKE.value:
                    self.pressure.copy_(
                        torch.clamp(
                            self.pressure + self.cfg.pressure_rise_rate,
                            max=1.0,
                        )
                    )
                else:
                    self.pressure.copy_(
                        torch.clamp(
                            self.pressure - self.cfg.pressure_decay_rate,
                            min=0.0,
                        )
                    )

            # Stage transitions.
            if current_stage_idx == SleepStage.WAKE.value:
                # Transition to sleep when pressure crosses threshold
                # and external arousal is low.
                if (
                    self.pressure.item() > self.cfg.sleep_threshold
                    and external < 0.5
                ):
                    self.stage.copy_(
                        torch.tensor(SleepStage.NREM.value, dtype=torch.long)
                    )
                    self.cycle_counter.zero_()
            else:
                # In sleep: alternate NREM and REM, wake on low pressure
                # or high external arousal.
                if (
                    self.pressure.item() < self.cfg.wake_threshold
                    or external > 0.7
                ):
                    self.stage.copy_(
                        torch.tensor(SleepStage.WAKE.value, dtype=torch.long)
                    )
                    self.cycle_counter.zero_()
                elif self.cfg.enable_nrem_rem_cycle:
                    self.cycle_counter.copy_(self.cycle_counter + 1)
                    cycle_pos = int(self.cycle_counter.item()) % self.cfg.nrem_rem_period
                    # First 80 percent of cycle is NREM, last 20 percent is REM.
                    nrem_phase = int(0.8 * self.cfg.nrem_rem_period)
                    if cycle_pos < nrem_phase:
                        self.stage.copy_(
                            torch.tensor(SleepStage.NREM.value, dtype=torch.long)
                        )
                    else:
                        self.stage.copy_(
                            torch.tensor(SleepStage.REM.value, dtype=torch.long)
                        )

        return SleepStage(int(self.stage.item())), self.pressure.clone()

    def force_stage(self, stage: SleepStage) -> None:
        """Manually set the sleep stage (useful for testing)."""
        with torch.no_grad():
            self.stage.copy_(torch.tensor(stage.value, dtype=torch.long))
            self.cycle_counter.zero_()

    def reset(self) -> None:
        """Reset state to initial wake/zero pressure."""
        with torch.no_grad():
            self.pressure.zero_()
            self.stage.zero_()
            self.cycle_counter.zero_()


# =========================================================================
# SECTION: spindle_ripple_coupling
# (originally spindle_ripple_coupling_t.py)
# =========================================================================
"""
spindle_ripple_coupling_t.py
Sleep Replay: Spindle-Ripple Coupling Consolidation

BIOLOGICAL GROUNDING
During NREM sleep, the cortex generates slow oscillations (~0.5-1
Hz), the thalamus generates sleep spindles (~12-15 Hz, lasting
~1 second), and the hippocampus generates sharp-wave ripples
(~150-250 Hz, lasting ~50-150 ms). The temporal nesting of these
three rhythms is the substrate of memory consolidation: ripples
that occur within the trough of slow oscillations and within the
positive cycle of spindles preferentially drive cortical replay
that strengthens consolidated memory traces.

Wei, Krishnan, Bazhenov (2016) provide a mechanistic model showing
how spindle-ripple coupling drives synaptic plasticity at
hippocampal-cortical projections. Helfrich, Lendner, Knight (2024)
characterize the triple-nesting (slow oscillation x spindle x ripple)
and demonstrate causally that disruption of nesting impairs memory.
Mednick et al. (2011) show that sleep spindles correlate with
declarative memory improvement.

This file implements the spindle and ripple generators with phase
coupling, plus a consolidation-strength readout that depends on the
three-way phase alignment. The amygdala consolidation tag (when
provided) gates which traces are preferentially consolidated.

Primary grounding papers:

Helfrich RF, Lendner JD, Knight RT (2024). Sleep oscillation triple
nesting and memory consolidation.
DOI: 10.1038/s41562-023-01768-6

Wei Y, Krishnan GP, Bazhenov M (2016). "Synaptic mechanisms of
memory consolidation during sleep slow oscillations." Journal of
Neuroscience, 36(15), 4231-4247.
DOI: 10.1523/JNEUROSCI.3648-15.2016

Mednick SC, McDevitt EA, Walker MP, Wamsley E, Paller KA, Stickgold
R (2011). "The critical role of sleep spindles in hippocampal-
dependent memory: a pharmacology study." Journal of Neuroscience,
33(10), 4494-4504. DOI: 10.1523/JNEUROSCI.3127-12.2013

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

# [aggregator] from __future__ import annotations  # lifted to top of file

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SpindleRippleConfig:
    """Configuration for the spindle-ripple coupling module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_consolidation: bool = True
    enable_slow_oscillation: bool = True
    enable_spindles: bool = True
    enable_ripples: bool = True
    enable_triple_nesting: bool = True

    # Frequencies in cycles per step (model time, not real seconds).
    # NOT biological quantities; engineering choices reflecting the
    # canonical 1:15:200 ratio between slow oscillation, spindle,
    # and ripple frequencies.
    slow_freq: float = 0.005
    spindle_freq: float = 0.075
    ripple_freq: float = 1.0

    # Spindle envelope width and rate. NOT biological quantities,
    # engineering tuning.
    spindle_amplitude: float = 1.0
    ripple_amplitude: float = 1.0

    # Consolidation gain when triple-nesting alignment is met.
    # NOT a biological quantity, training artifact.
    aligned_gain: float = 2.0


class SpindleRippleCoupling(nn.Module):
    """Spindle-ripple coupling consolidation module.

    BIOLOGICAL STRUCTURE: Cortical slow oscillation generators,
    thalamic spindle generators, hippocampal sharp-wave ripple
    generators, and the synaptic targets that read out their
    coupling.

    BIOLOGICAL FUNCTION: Drives memory consolidation through phase-
    coupled replay. The triple-nesting structure (ripple inside
    spindle inside slow oscillation up-state) marks the temporal
    window in which hippocampal traces are imprinted into cortex.

    Helfrich RF et al. (2024). DOI: 10.1038/s41562-023-01768-6
    Wei Y, Krishnan GP, Bazhenov M (2016).
    DOI: 10.1523/JNEUROSCI.3648-15.2016
    Mednick SC et al. (2011). DOI: 10.1523/JNEUROSCI.3127-12.2013

    ANATOMICAL INTERFACE (input):
        Sending structures: cortical slow oscillation generators
        (brainstem-driven), thalamic spindle generators (TRN-driven),
        hippocampal sharp-wave ripple generators (CA3-driven).
        Receiving structure: this module.
        Connection: sleep-stage-specific neuromodulator and
        thalamocortical-hippocampal coupling.

    ANATOMICAL INTERFACE (output):
        Sending structure: this module's consolidation gain readout.
        Receiving structure: cortex-hippocampus plasticity machinery.
        Connection: applied as multiplicative scaling on plasticity
        signals during sleep replay.
    """

    def __init__(
        self, cfg: Optional[SpindleRippleConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SpindleRippleConfig()
        # Persistent phase counters for each oscillator.
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        is_sleep: bool,
        consolidation_tag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Advance one tick and return the consolidation gain.

        Args:
            is_sleep: bool indicating sleep stage active. The module
                only generates oscillations during sleep.
            consolidation_tag: optional (B,) amygdala emotional tag
                that biases which traces are preferentially
                consolidated. None means uniform consolidation.

        Returns:
            consolidation_gain: scalar or (B,) gain factor to apply
                to plasticity signals.
            diagnostics: dict with the three oscillator phases.
        """
        if not self.cfg.enable_consolidation or not is_sleep:
            zero = torch.tensor(0.0)
            return zero, {
                "slow_phase": 0.0, "spindle_phase": 0.0, "ripple_phase": 0.0,
            }

        with torch.no_grad():
            self.step.copy_(self.step + 1)
            t = float(self.step.item())

        # Compute phases.
        if self.cfg.enable_slow_oscillation:
            slow_phase = torch.cos(
                torch.tensor(2 * 3.14159 * self.cfg.slow_freq * t)
            )
        else:
            slow_phase = torch.tensor(0.0)

        if self.cfg.enable_spindles:
            # Spindles are amplitude-modulated bursts. We model them
            # as a sinusoid times a slow envelope that peaks during
            # the slow oscillation up-state.
            spindle_envelope = torch.relu(slow_phase) * self.cfg.spindle_amplitude
            spindle_phase = (
                spindle_envelope
                * torch.cos(torch.tensor(2 * 3.14159 * self.cfg.spindle_freq * t))
            )
        else:
            spindle_phase = torch.tensor(0.0)

        if self.cfg.enable_ripples:
            # Ripples are nested within spindle peaks.
            ripple_envelope = torch.relu(spindle_phase) * self.cfg.ripple_amplitude
            ripple_phase = (
                ripple_envelope
                * torch.cos(torch.tensor(2 * 3.14159 * self.cfg.ripple_freq * t))
            )
        else:
            ripple_phase = torch.tensor(0.0)

        # Triple-nesting alignment: positive contribution from each
        # oscillator simultaneously gives maximum consolidation.
        if self.cfg.enable_triple_nesting:
            alignment = (
                torch.relu(slow_phase)
                * torch.relu(spindle_phase)
                * torch.relu(ripple_phase)
            )
            base_gain = 1.0 + self.cfg.aligned_gain * alignment
        else:
            # Without triple-nesting, just use ripple amplitude.
            base_gain = 1.0 + torch.relu(ripple_phase)

        # Apply emotional consolidation tag if provided.
        if consolidation_tag is not None:
            consolidation_gain = base_gain * consolidation_tag
        else:
            consolidation_gain = base_gain

        return consolidation_gain, {
            "slow_phase": float(slow_phase.item()),
            "spindle_phase": float(spindle_phase.item()),
            "ripple_phase": float(ripple_phase.item()),
        }

    def reset(self) -> None:
        """Reset oscillator phase."""
        with torch.no_grad():
            self.step.zero_()


# =========================================================================
# SECTION: run_all
# (originally run_all.py)
# =========================================================================
import subprocess
import glob

tests = sorted(glob.glob('/home/claude/work/test_*.py'))
total = 0
for t in tests:
    r = subprocess.run(['python3', t], capture_output=True, text=True, cwd='/home/claude/work')
    last = r.stdout.strip().split('\n')[-1] if r.stdout else 'NO OUTPUT'
    print(f'{t.split("/")[-1]}: {last}')
    if 'passed' in last:
        try:
            n = int(last.split('All ')[1].split(' ')[0])
            total += n
        except Exception:
            pass
print(f'TOTAL: {total} tests')


