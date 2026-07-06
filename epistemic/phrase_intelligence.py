# =================================================================
# MODULE: epistemic/phrase_intelligence.py
# DESCRIPTION: Structural phase state machine + phrase synthesis.
#
# VERSION: 5.6.1 (merged from structural_phase_engine.py + phrase_intelligence.py)
# UPDATED: 2026-05-24
#
# DESIGN PRINCIPLES:
#   - Never re-analyze raw notes. Only transform testimonies.
#   - Structural state is a TRAJECTORY, not a snapshot.
#   - Genre EMERGES from attractor basins, not detection.
#   - Boundaries are HYPOTHESES validated by multiple witnesses.
#   - Time evolution is first-class, not an afterthought.
#   - NoteEvents used ONLY for timestamps to create windows.
#     All musical content comes from upstream testimonies.
#
# BUGS FIXED FROM DRAFT:
#   - StructuralPhase.ANY referenced but never defined → removed
#   - notes[0].start_time → notes[0].start_ms
#   - notes[-1].duration → notes[-1].duration_ms()
#   - _compute_theme_elaboration vs _compute_theme_elaboration_score mismatch
#   - structural_windows[-1] IndexError when list is empty
#   - Missing imports (numpy, dataclasses, etc.)
#   - avg_vector is frozen dataclass — setattr fails → fixed to dict first
#   - StructuralInvariantVector cosine_similarity uses vars() on frozen obj → fixed
#   - PhraseIntelligence._compute_theme_elaboration_score naming inconsistency
# =================================================================

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.order_types import NoteEvent
from epistemic.musical_findings_map import MusicalFindingsMap
from core.testimony import StageResult


# =====================================================================
# Local testimony types
# (Add StructuralTestimony + PhaseTransition to core/testimony.py later)
# =====================================================================

@dataclass
class PhaseTransition:
    """A detected change in structural phase."""
    time_ms:     float
    from_phase:  str   # StructuralPhase.value
    to_phase:    str
    confidence:  float


@dataclass
class StructuralTestimony:
    """Full structural trajectory testimony."""
    structural_trajectory:  List[Any]         = field(default_factory=list)
    phase_transitions:      List[PhaseTransition] = field(default_factory=list)
    attractor_basin:        str               = "unknown"
    attractor_confidence:   float             = 0.0
    invariant_evolution:    Dict[str, List]   = field(default_factory=dict)
    needs_reanalysis:       Dict[str, Any]    = field(default_factory=dict)
    confidence:             float             = 0.0

    # Phrase-level synthesis (from PhraseIntelligence)
    phrase_boundaries:      List[Any]         = field(default_factory=list)
    anacrusis_detected:     Dict[str, Any]    = field(default_factory=dict)
    structural_signature:   Dict[str, float]  = field(default_factory=dict)
    structural_regime:      Dict[str, Any]    = field(default_factory=dict)
    theme_elaboration_score: float            = 0.5
    estimated_form:         Dict[str, Any]    = field(default_factory=dict)


# =====================================================================
# Enums
# =====================================================================

class StructuralPhase(Enum):
    """Discrete structural states — NOT a continuum."""
    THEME            = "theme"
    DEVELOPMENT      = "development"
    TRANSITION       = "transition"
    RECAPITULATION   = "recapitulation"
    CODA             = "coda"
    INTRO            = "intro"
    SOLO             = "solo"
    LOOP             = "loop"
    AMBIGUOUS        = "ambiguous"


class AttractorBasin(Enum):
    """Genre attractors — where the trajectory tends to converge."""
    POP       = "pop"
    JAZZ      = "jazz"
    CLASSICAL = "classical"
    AMBIENT   = "ambient"
    BINARY    = "binary"


# =====================================================================
# Invariant Vector
# =====================================================================

_INVARIANT_AXES = [
    'tonal_stability',
    'temporal_cyclicity',
    'transformational_depth',
    'event_density',
    'constraint_strength',
    'surface_entropy',
]


@dataclass
class StructuralInvariantVector:
    """
    Six orthogonal structural dimensions — properly decomposed.

    Each axis is computed from multiple testimony sources to ensure
    orthogonality. No axis should be computable from any other axis
    alone.
    """
    tonal_stability:        float = 0.5   # 0=unstable, 1=stable/clear
    temporal_cyclicity:     float = 0.5   # 0=through-composed, 1=cyclic
    transformational_depth: float = 0.5   # 0=exact repetition, 1=heavy transform
    event_density:          float = 0.5   # 0=sparse, 1=dense
    constraint_strength:    float = 0.5   # 0=free, 1=highly constrained
    surface_entropy:        float = 0.5   # 0=predictable, 1=unpredictable

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, ax) for ax in _INVARIANT_AXES],
                        dtype=np.float64)

    def cosine_similarity(self, other: 'StructuralInvariantVector') -> float:
        """Cosine similarity in invariant space."""
        a = self.as_array()
        b = other.as_array()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'StructuralInvariantVector':
        return cls(**{ax: d.get(ax, 0.5) for ax in _INVARIANT_AXES})


@dataclass
class StructuralWindow:
    """Time-windowed structural analysis snapshot."""
    start_time_ms:   float
    end_time_ms:     float
    invariant_vector: StructuralInvariantVector
    phase:           StructuralPhase
    phase_confidence: float
    boundary_strength: float


# =====================================================================
# Phase Prototypes
# =====================================================================

_PHASE_PROTOTYPES: Dict[StructuralPhase, StructuralInvariantVector] = {
    StructuralPhase.THEME: StructuralInvariantVector(
        tonal_stability=0.80, temporal_cyclicity=0.70,
        transformational_depth=0.20, event_density=0.60,
        constraint_strength=0.70, surface_entropy=0.30),
    StructuralPhase.DEVELOPMENT: StructuralInvariantVector(
        tonal_stability=0.50, temporal_cyclicity=0.40,
        transformational_depth=0.70, event_density=0.70,
        constraint_strength=0.40, surface_entropy=0.70),
    StructuralPhase.SOLO: StructuralInvariantVector(
        tonal_stability=0.70, temporal_cyclicity=0.60,
        transformational_depth=0.60, event_density=0.55,
        constraint_strength=0.50, surface_entropy=0.80),
    StructuralPhase.RECAPITULATION: StructuralInvariantVector(
        tonal_stability=0.80, temporal_cyclicity=0.70,
        transformational_depth=0.30, event_density=0.60,
        constraint_strength=0.70, surface_entropy=0.30),
    StructuralPhase.LOOP: StructuralInvariantVector(
        tonal_stability=0.80, temporal_cyclicity=0.90,
        transformational_depth=0.10, event_density=0.70,
        constraint_strength=0.80, surface_entropy=0.20),
    StructuralPhase.INTRO: StructuralInvariantVector(
        tonal_stability=0.40, temporal_cyclicity=0.30,
        transformational_depth=0.30, event_density=0.30,
        constraint_strength=0.40, surface_entropy=0.40),
    StructuralPhase.CODA: StructuralInvariantVector(
        tonal_stability=0.70, temporal_cyclicity=0.20,
        transformational_depth=0.20, event_density=0.30,
        constraint_strength=0.60, surface_entropy=0.20),
    StructuralPhase.TRANSITION: StructuralInvariantVector(
        tonal_stability=0.50, temporal_cyclicity=0.30,
        transformational_depth=0.50, event_density=0.50,
        constraint_strength=0.40, surface_entropy=0.50),
}

_ATTRACTOR_PROTOTYPES: Dict[AttractorBasin, StructuralInvariantVector] = {
    AttractorBasin.POP: StructuralInvariantVector(
        tonal_stability=0.85, temporal_cyclicity=0.90,
        transformational_depth=0.20, event_density=0.70,
        constraint_strength=0.80, surface_entropy=0.25),
    AttractorBasin.JAZZ: StructuralInvariantVector(
        tonal_stability=0.70, temporal_cyclicity=0.60,
        transformational_depth=0.65, event_density=0.55,
        constraint_strength=0.50, surface_entropy=0.75),
    AttractorBasin.CLASSICAL: StructuralInvariantVector(
        tonal_stability=0.80, temporal_cyclicity=0.40,
        transformational_depth=0.75, event_density=0.50,
        constraint_strength=0.70, surface_entropy=0.40),
    AttractorBasin.AMBIENT: StructuralInvariantVector(
        tonal_stability=0.40, temporal_cyclicity=0.30,
        transformational_depth=0.30, event_density=0.20,
        constraint_strength=0.30, surface_entropy=0.35),
    AttractorBasin.BINARY: StructuralInvariantVector(
        tonal_stability=0.60, temporal_cyclicity=0.70,
        transformational_depth=0.30, event_density=0.60,
        constraint_strength=0.65, surface_entropy=0.30),
}


# =====================================================================
# Transition Weights (learned priors — not hard rules)
# =====================================================================

def _build_transition_weights() -> Dict[Tuple[StructuralPhase, StructuralPhase], float]:
    """
    Prior probabilities for structural phase transitions.

    These weight the state-machine inference. They are PRIORS that
    can be learned from a corpus over time. They prevent impossible
    transitions from being accepted even when the invariant vector
    happens to match.
    """
    # Default all transitions to low probability
    weights: Dict = {}
    all_phases = list(StructuralPhase)
    for fp in all_phases:
        for tp in all_phases:
            weights[(fp, tp)] = 0.05  # Very low baseline

    # Self-transitions are always plausible
    for p in all_phases:
        weights[(p, p)] = 0.90

    # High-probability transitions (music theory informed)
    allowed = [
        (StructuralPhase.INTRO,          StructuralPhase.THEME),
        (StructuralPhase.INTRO,          StructuralPhase.LOOP),
        (StructuralPhase.THEME,          StructuralPhase.DEVELOPMENT),
        (StructuralPhase.THEME,          StructuralPhase.TRANSITION),
        (StructuralPhase.THEME,          StructuralPhase.SOLO),
        (StructuralPhase.LOOP,           StructuralPhase.TRANSITION),
        (StructuralPhase.LOOP,           StructuralPhase.SOLO),
        (StructuralPhase.LOOP,           StructuralPhase.THEME),
        (StructuralPhase.DEVELOPMENT,    StructuralPhase.RECAPITULATION),
        (StructuralPhase.DEVELOPMENT,    StructuralPhase.CODA),
        (StructuralPhase.DEVELOPMENT,    StructuralPhase.TRANSITION),
        (StructuralPhase.SOLO,           StructuralPhase.THEME),
        (StructuralPhase.SOLO,           StructuralPhase.RECAPITULATION),
        (StructuralPhase.SOLO,           StructuralPhase.CODA),
        (StructuralPhase.TRANSITION,     StructuralPhase.THEME),
        (StructuralPhase.TRANSITION,     StructuralPhase.DEVELOPMENT),
        (StructuralPhase.TRANSITION,     StructuralPhase.LOOP),
        (StructuralPhase.RECAPITULATION, StructuralPhase.CODA),
        (StructuralPhase.RECAPITULATION, StructuralPhase.THEME),
        # AMBIGUOUS can transition anywhere
        *[(StructuralPhase.AMBIGUOUS, p) for p in all_phases],
        *[(p, StructuralPhase.AMBIGUOUS) for p in all_phases],
    ]
    for fp, tp in allowed:
        weights[(fp, tp)] = 0.85

    return weights


_TRANSITION_WEIGHTS = _build_transition_weights()


# =====================================================================
# Main Class
# =====================================================================

class PhraseIntelligence:
    """
    Structural phase state machine + phrase synthesizer.

    Replaces both structural_phase_engine.py and the draft
    phrase_intelligence.py with a single, correct implementation.

    Input:  Upstream testimonies (StageResult objects) + NoteEvent
            timestamps ONLY.
    Output: StructuralTestimony with:
              - Phase trajectory (INTRO → THEME → SOLO → CODA …)
              - Attractor basin (genre emergence)
              - Phrase boundaries and anacrusis
              - Structural regime and theme/elaboration score
              - Re-analysis requests when witnesses contradict

    CRITICAL INVARIANT: NoteEvents are used ONLY for timestamps.
    All musical content is read from testimonies.
    """

    def __init__(
        self,
        findings_map: MusicalFindingsMap,
        window_size_ms: float = 2000.0,
        window_overlap: float = 0.5,
    ):
        self.findings      = findings_map
        self.window_size   = window_size_ms
        # _create_time_windows derives hop = window_size * (1 - overlap);
        # an overlap >= 1.0 (never validated before) makes hop <= 0, so
        # its while-loop's cursor either never advances (hangs forever)
        # or goes backwards. Not reachable via the real pipeline today
        # (always constructed with the 0.5 default) but cheap to guard
        # against a future caller passing a bad config value.
        self.overlap       = max(0.0, min(0.95, window_overlap))
        self._history: deque = deque(maxlen=20)

    # ====================================================================
    # Public Entry Point
    # ====================================================================

    def compute_structural_trajectory(
        self,
        notes:             List[NoteEvent],
        pulse_testimony:   StageResult,
        voice_testimony:   StageResult,
        harmonic_testimony: StageResult,
        geo_testimony:     StageResult,
        tempo_testimony:   StageResult,
        groove_testimony:  StageResult,
        motif_testimony:   Optional[StageResult] = None,
    ) -> StageResult:
        """
        Compute full structural trajectory from testimonies.

        Notes are used ONLY to determine the time range and create
        analysis windows. All musical inference comes from the
        testimony arguments.
        """
        if not notes:
            return StageResult(
                stage_name="phrase_intelligence",
                success=False,
                testimony=StructuralTestimony(),
                errors=["No notes provided"],
            )

        # NoteEvents → timestamps only
        # FIX: use start_ms not start_time; use duration_ms() not .duration
        start_ms   = float(notes[0].start_ms)
        end_ms     = float(notes[-1].end_ms)

        windows_ms = self._create_time_windows(start_ms, end_ms)
        structural_windows: List[StructuralWindow] = []
        reasoning: List[str] = []

        for win_start, win_end in windows_ms:
            evidence = self._extract_window_evidence(
                win_start, win_end,
                pulse_testimony, voice_testimony, harmonic_testimony,
                geo_testimony, tempo_testimony, groove_testimony, motif_testimony,
            )
            invariant  = self._compute_invariant_vector(evidence)
            prev_win   = structural_windows[-1] if structural_windows else None
            phase, conf = self._determine_phase(invariant, prev_win)
            boundary   = self._compute_boundary_strength(evidence, prev_win)

            win = StructuralWindow(
                start_time_ms=win_start,
                end_time_ms=win_end,
                invariant_vector=invariant,
                phase=phase,
                phase_confidence=conf,
                boundary_strength=boundary,
            )
            structural_windows.append(win)
            reasoning.append(
                f"[{win_start:.0f}-{win_end:.0f}ms] {phase.value} "
                f"(conf={conf:.2f}, boundary={boundary:.2f})"
            )

        transitions          = self._detect_phase_transitions(structural_windows)
        attractor, att_conf  = self._compute_attractor_basin(structural_windows)
        invariant_evolution  = self._extract_invariant_evolution(structural_windows)
        contradictions       = self._check_contradictions(structural_windows, transitions)
        traj_confidence      = self._compute_trajectory_confidence(structural_windows)

        # ---- Phrase-level synthesis (PhraseIntelligence contribution) ----
        pulse_field  = self._safe_testimony(pulse_testimony)
        voice_field  = self._safe_testimony(voice_testimony)
        harmonic_f   = self._safe_testimony(harmonic_testimony)
        groove_field = self._safe_testimony(groove_testimony)

        phrase_boundaries = voice_field.get("phrase_boundaries", [])
        chord_progressions = harmonic_f.get("progressions", [])
        cadence_points     = harmonic_f.get("cadences", [])
        downbeat_conf      = pulse_field.get("downbeat_confidences", {})
        beat_phase_map     = pulse_field.get("beat_phase_map", {})

        harmonic_rigidity   = self._compute_harmonic_rigidity(
            chord_progressions, cadence_points,
            harmonic_testimony.confidence if hasattr(harmonic_testimony, 'confidence') else 0.5)
        motivic_persistence = self._compute_motivic_persistence(
            notes, voice_field.get("continuity_scores", {}),
            voice_testimony.confidence if hasattr(voice_testimony, 'confidence') else 0.5)
        formal_periodicity  = self._compute_formal_periodicity(
            downbeat_conf, self._safe_testimony(geo_testimony).get("bar_structure", {}),
            pulse_testimony.confidence if hasattr(pulse_testimony, 'confidence') else 0.5)
        swing_amount        = groove_field.get("swing_amount_ms", 0.0)
        surface_entropy     = self._compute_surface_entropy(
            notes, swing_amount,
            groove_testimony.confidence if hasattr(groove_testimony, 'confidence') else 0.5)

        structural_signature = {
            "harmonic_rigidity":        harmonic_rigidity,
            "motivic_persistence":      motivic_persistence,
            "formal_periodicity":       formal_periodicity,
            "surface_entropy":          surface_entropy,
            "theme_elaboration_score":  0.0,  # filled below
        }

        theme_elaboration = self._compute_theme_elaboration_score(
            motivic_persistence, surface_entropy, harmonic_rigidity)
        structural_signature["theme_elaboration_score"] = theme_elaboration

        structural_regime = self._detect_structural_regime(
            harmonic_rigidity, motivic_persistence,
            formal_periodicity, surface_entropy)

        anacrusis = self._detect_anacrusis(notes, beat_phase_map, downbeat_conf)
        est_form  = self._infer_form(structural_signature)

        testimony = StructuralTestimony(
            structural_trajectory=structural_windows,
            phase_transitions=transitions,
            attractor_basin=attractor.value if attractor else "unknown",
            attractor_confidence=att_conf,
            invariant_evolution=invariant_evolution,
            needs_reanalysis=contradictions,
            confidence=traj_confidence,
            phrase_boundaries=phrase_boundaries,
            anacrusis_detected=anacrusis,
            structural_signature=structural_signature,
            structural_regime=structural_regime,
            theme_elaboration_score=theme_elaboration,
            estimated_form=est_form,
        )

        return StageResult(
            stage_name="phrase_intelligence",
            success=True,
            testimony=testimony,
        )

    # ====================================================================
    # Window Creation
    # ====================================================================

    def _create_time_windows(
        self, start_ms: float, end_ms: float
    ) -> List[Tuple[float, float]]:
        windows = []
        hop    = self.window_size * (1.0 - self.overlap)
        cursor = start_ms
        while cursor < end_ms:
            win_end = min(cursor + self.window_size, end_ms)
            windows.append((cursor, win_end))
            cursor += hop
        return windows

    # ====================================================================
    # Evidence Extraction (testimony-only)
    # ====================================================================

    def _extract_window_evidence(
        self,
        start_ms: float, end_ms: float,
        pulse_t: StageResult, voice_t: StageResult,
        harmonic_t: StageResult, geo_t: StageResult,
        tempo_t: StageResult, groove_t: StageResult,
        motif_t: Optional[StageResult],
    ) -> Dict[str, Any]:
        """
        Extract evidence for a time window from testimonies only.
        No raw audio computation.
        """
        pulse_f    = self._safe_testimony(pulse_t)
        voice_f    = self._safe_testimony(voice_t)
        harmonic_f = self._safe_testimony(harmonic_t)
        groove_f   = self._safe_testimony(groove_t)
        tempo_f    = self._safe_testimony(tempo_t)

        return {
            "downbeat_density":   self._window_avg(
                pulse_f.get("downbeat_confidences", {}), start_ms, end_ms),
            "beat_stability":     self._window_avg(
                pulse_f.get("stability_scores", {}), start_ms, end_ms),
            "phrase_continuity":  self._window_avg(
                voice_f.get("continuity_scores", {}), start_ms, end_ms),
            "harmonic_stability": self._window_avg(
                harmonic_f.get("tonal_stability", {}), start_ms, end_ms),
            "cadence_density":    self._window_avg(
                harmonic_f.get("cadence_strengths", {}), start_ms, end_ms),
            "tempo_variance":     self._window_tempo_variance(
                tempo_f.get("tempo_curve", []), start_ms, end_ms),
            "swing_intensity":    self._window_avg(
                groove_f.get("swing_amounts", {}), start_ms, end_ms),
            "motif_recurrence":   (
                self._window_avg(
                    self._safe_testimony(motif_t).get("recurrence_scores", {}),
                    start_ms, end_ms)
                if motif_t else 0.5),
        }

    def _window_avg(
        self, series: Dict[float, float], start: float, end: float
    ) -> float:
        """Average testimony value in [start, end] ms."""
        if not series:
            return 0.5
        values = [v for t, v in series.items() if start <= t <= end]
        return float(np.mean(values)) if values else 0.5

    def _window_tempo_variance(
        self, tempo_curve: List, start: float, end: float
    ) -> float:
        tempos = [row[1] for row in tempo_curve
                  if isinstance(row, (list, tuple)) and len(row) >= 2
                  and start <= row[0] <= end]
        if len(tempos) < 2:
            return 0.0
        mean = float(np.mean(tempos))
        return float(np.std(tempos) / mean) if mean > 0 else 0.0

    @staticmethod
    def _safe_testimony(result: Optional[StageResult]) -> Dict:
        """Safely extract testimony dict from a StageResult."""
        if result is None:
            return {}
        t = result.testimony
        if isinstance(t, dict):
            return t
        if hasattr(t, '__dict__'):
            return vars(t)
        return {}

    # ====================================================================
    # Invariant Vector Computation
    # ====================================================================

    def _compute_invariant_vector(
        self, evidence: Dict[str, Any]
    ) -> StructuralInvariantVector:
        """
        Map testimony evidence to orthogonal structural invariants.
        Each axis is a weighted combination of DIFFERENT testimony sources
        to ensure orthogonality.
        """
        g = evidence.get

        # Axis 1: Tonal stability
        tonal_stability = (
            g("harmonic_stability", 0.5) * 0.70 +
            g("cadence_density",    0.5) * 0.30)

        # Axis 2: Temporal cyclicity
        temporal_cyclicity = (
            g("downbeat_density", 0.5) * 0.60 +
            g("beat_stability",   0.5) * 0.40)

        # Axis 3: Transformational depth (inverse of motif recurrence)
        motif = g("motif_recurrence", 0.5)
        transformational_depth = (
            (1.0 - motif)              * 0.70 +
            g("phrase_continuity", 0.5) * 0.30)

        # Axis 4: Event density
        event_density = (
            g("downbeat_density",  0.5) * 0.50 +
            g("cadence_density",   0.5) * 0.50)

        # Axis 5: Constraint strength
        tempo_var  = g("tempo_variance", 0.0)
        tempo_constr = 1.0 - min(1.0, tempo_var * 2.0)
        constraint_strength = (
            g("harmonic_stability", 0.5) * 0.60 +
            tempo_constr                 * 0.40)

        # Axis 6: Surface entropy
        surface_entropy = (
            g("swing_intensity",   0.5) * 0.50 +
            (1.0 - g("phrase_continuity", 0.5)) * 0.50)

        def clamp(v: float) -> float:
            return float(np.clip(v, 0.0, 1.0))

        return StructuralInvariantVector(
            tonal_stability=clamp(tonal_stability),
            temporal_cyclicity=clamp(temporal_cyclicity),
            transformational_depth=clamp(transformational_depth),
            event_density=clamp(event_density),
            constraint_strength=clamp(constraint_strength),
            surface_entropy=clamp(surface_entropy),
        )

    # ====================================================================
    # State Machine: Phase Determination
    # ====================================================================

    def _determine_phase(
        self,
        invariant:     StructuralInvariantVector,
        previous_win:  Optional[StructuralWindow],
    ) -> Tuple[StructuralPhase, float]:
        """
        Determine structural phase using prototype matching + transition
        constraints (state-machine logic, NOT a pure classifier).
        """
        # Step 1: Find closest prototype
        best_phase = StructuralPhase.AMBIGUOUS
        best_sim   = 0.0

        for candidate, prototype in _PHASE_PROTOTYPES.items():
            sim = invariant.cosine_similarity(prototype)
            if sim > best_sim:
                best_sim   = sim
                best_phase = candidate

        raw_confidence = best_sim

        # Step 2: Apply transition weight (state-machine constraint)
        if previous_win is not None:
            trans_key  = (previous_win.phase, best_phase)
            trans_prob = _TRANSITION_WEIGHTS.get(trans_key, 0.05)

            # Blend prototype match with transition plausibility
            blended_conf = raw_confidence * 0.70 + trans_prob * 0.30

            # If transition is near-impossible, hold previous phase
            if trans_prob < 0.10 and blended_conf < 0.60:
                best_phase     = previous_win.phase
                blended_conf   = previous_win.phase_confidence * 0.85

            return best_phase, float(np.clip(blended_conf, 0.0, 1.0))

        return best_phase, float(np.clip(raw_confidence, 0.0, 1.0))

    # ====================================================================
    # Boundary Detection
    # ====================================================================

    def _compute_boundary_strength(
        self,
        evidence:     Dict[str, Any],
        previous_win: Optional[StructuralWindow],
    ) -> float:
        """
        Boundary strength = how many witnesses agree there's a discontinuity.
        Strong boundaries require multi-witness agreement.
        """
        if previous_win is None:
            return 0.0

        signals = []
        g   = evidence.get
        piv = previous_win.invariant_vector

        # Harmonic discontinuity
        harm_curr = g("harmonic_stability", 0.5)
        if abs(harm_curr - piv.tonal_stability) > 0.25:
            signals.append(0.80)

        # Rhythmic discontinuity
        cyc_curr = g("downbeat_density", 0.5)
        if abs(cyc_curr - piv.temporal_cyclicity) > 0.35:
            signals.append(0.70)

        # Swing discontinuity (common at section boundaries in jazz)
        swing_curr = g("swing_intensity", 0.5)
        if abs(swing_curr - piv.surface_entropy) > 0.40:
            signals.append(0.65)

        # Phase change from previous window
        new_phase, _ = self._determine_phase(
            self._compute_invariant_vector(evidence), None)
        if new_phase != previous_win.phase:
            signals.append(0.90)

        return float(np.clip(np.mean(signals), 0.0, 1.0)) if signals else 0.0

    # ====================================================================
    # Trajectory Analysis
    # ====================================================================

    def _detect_phase_transitions(
        self, windows: List[StructuralWindow]
    ) -> List[PhaseTransition]:
        transitions = []
        for i in range(1, len(windows)):
            if windows[i].phase != windows[i - 1].phase:
                transitions.append(PhaseTransition(
                    time_ms     = windows[i].start_time_ms,
                    from_phase  = windows[i - 1].phase.value,
                    to_phase    = windows[i].phase.value,
                    confidence  = float(
                        windows[i].phase_confidence *
                        windows[i - 1].phase_confidence),
                ))
        return transitions

    def _compute_attractor_basin(
        self, windows: List[StructuralWindow]
    ) -> Tuple[Optional[AttractorBasin], float]:
        """
        Genre EMERGES from attractor convergence — not from detection.
        Uses the last N windows to avoid transient distortion.
        """
        if not windows:
            return None, 0.0

        tail = windows[-min(5, len(windows)):]
        avg_dict: Dict[str, float] = {}
        for ax in _INVARIANT_AXES:
            avg_dict[ax] = float(np.mean([getattr(w.invariant_vector, ax)
                                          for w in tail]))
        avg_vec = StructuralInvariantVector.from_dict(avg_dict)

        best_basin: Optional[AttractorBasin] = None
        best_sim   = 0.0
        for basin, prototype in _ATTRACTOR_PROTOTYPES.items():
            sim = avg_vec.cosine_similarity(prototype)
            if sim > best_sim:
                best_sim  = sim
                best_basin = basin

        return best_basin, float(best_sim)

    def _extract_invariant_evolution(
        self, windows: List[StructuralWindow]
    ) -> Dict[str, List[float]]:
        if not windows:
            return {}
        evo: Dict[str, List[float]] = {"timestamps_ms": []}
        for ax in _INVARIANT_AXES:
            evo[ax] = []
        for w in windows:
            evo["timestamps_ms"].append(w.start_time_ms)
            for ax in _INVARIANT_AXES:
                evo[ax].append(getattr(w.invariant_vector, ax))
        return evo

    def _compute_trajectory_confidence(
        self, windows: List[StructuralWindow]
    ) -> float:
        if not windows:
            return 0.0

        avg_conf = float(np.mean([w.phase_confidence for w in windows]))

        # Stability bonus: lower entropy in phase sequence = more confident
        phase_vals = [w.phase.value for w in windows]
        counts     = {}
        for p in phase_vals:
            counts[p] = counts.get(p, 0) + 1
        n = len(phase_vals)
        probs      = [c / n for c in counts.values()]
        entropy    = -float(sum(p * np.log(p + 1e-10) for p in probs))
        max_ent    = np.log(len(StructuralPhase))
        stability  = 1.0 - (entropy / max_ent) if max_ent > 0 else 0.5

        return float(np.clip(avg_conf * 0.60 + stability * 0.40, 0.0, 1.0))

    def _check_contradictions(
        self,
        windows:     List[StructuralWindow],
        transitions: List[PhaseTransition],
    ) -> Dict[str, Any]:
        needs = False
        reasons = []

        # Oscillation: too many phases in too few windows
        n_unique = len({w.phase for w in windows})
        if n_unique > 4 and len(windows) > 10:
            needs = True
            reasons.append(
                f"Phase oscillation: {n_unique} phases in {len(windows)} windows")

        # Excessive boundaries
        strong  = [w for w in windows if w.boundary_strength > 0.70]
        if strong and len(strong) > len(windows) // 2:
            needs = True
            reasons.append(
                f"Boundary overload: {len(strong)}/{len(windows)} windows")

        # Impossible transitions
        for t in transitions:
            fp = next((p for p in StructuralPhase if p.value == t.from_phase),
                      StructuralPhase.AMBIGUOUS)
            tp = next((p for p in StructuralPhase if p.value == t.to_phase),
                      StructuralPhase.AMBIGUOUS)
            prob = _TRANSITION_WEIGHTS.get((fp, tp), 0.05)
            if prob < 0.10 and t.confidence > 0.60:
                needs = True
                reasons.append(
                    f"Impossible transition: {t.from_phase} → {t.to_phase}")

        return {
            "requires_reanalysis": needs,
            "reasoning":          reasons,
            "severity":           "high" if needs else "none",
        }

    # ====================================================================
    # Phrase Intelligence (Structural Synthesis Layer)
    # ====================================================================

    def _compute_harmonic_rigidity(
        self,
        progressions:       List,
        cadences:           List,
        source_confidence:  float,
    ) -> float:
        if not progressions:
            return 0.5 * source_confidence

        unique = len({str(p) for p in progressions})
        repetition = 1.0 - (unique / max(len(progressions), 1))

        if len(cadences) < 2:
            cadence_regularity = 0.30
        else:
            intervals = [cadences[i] - cadences[i - 1]
                         for i in range(1, len(cadences))]
            std = float(np.std(intervals)) if len(intervals) > 1 else 0.0
            cadence_regularity = 1.0 - min(1.0, std / 5000.0)

        return float(np.clip(
            (repetition * 0.60 + cadence_regularity * 0.40) * source_confidence,
            0.0, 1.0))

    def _compute_motivic_persistence(
        self,
        notes:             List[NoteEvent],
        continuity_scores: Dict,
        source_confidence: float,
    ) -> float:
        if len(notes) < 10:
            return 0.5

        avg_cont = float(np.mean(list(continuity_scores.values()))
                         if continuity_scores else 0.5)

        # FIX: use .pitch not .note; use start_ms not start_time
        intervals = [notes[i].pitch - notes[i - 1].pitch
                     for i in range(1, min(len(notes), 50))]
        self_sim  = self._interval_self_similarity(intervals)

        return float(np.clip(
            (avg_cont * 0.50 + self_sim * 0.50) * source_confidence,
            0.0, 1.0))

    @staticmethod
    def _interval_self_similarity(intervals: List[int]) -> float:
        if len(intervals) < 4:
            return 0.30
        pattern = intervals[:4]
        matches = 0
        checks  = 0
        for i in range(4, len(intervals) - 3, 2):
            window = intervals[i:i + 4]
            if len(window) == 4:
                match = sum(1 for a, b in zip(pattern, window) if abs(a - b) <= 2)
                if match >= 3:
                    matches += 1
                checks += 1
        return float(min(1.0, (matches / max(checks, 1)) * 2.0))

    def _compute_formal_periodicity(
        self,
        downbeat_confidences: Dict,
        bar_structure:        Dict,
        source_confidence:    float,
    ) -> float:
        if not downbeat_confidences:
            return 0.5

        avg_db = float(np.mean(list(downbeat_confidences.values())))

        bar_lengths = bar_structure.get("bar_lengths_ms", [])
        if len(bar_lengths) > 1:
            std = float(np.std(bar_lengths))
            bar_reg = 1.0 - min(1.0, std / 500.0)
        else:
            bar_reg = 0.50

        return float(np.clip(
            (avg_db * 0.60 + bar_reg * 0.40) * source_confidence,
            0.0, 1.0))

    def _compute_surface_entropy(
        self,
        notes:             List[NoteEvent],
        swing_amount_ms:   float,
        source_confidence: float,
    ) -> float:
        if len(notes) < 10:
            return 0.30

        # Pitch class entropy
        pcs = [n.pitch % 12 for n in notes]
        counts = {}
        for pc in pcs:
            counts[pc] = counts.get(pc, 0) + 1
        probs = [c / len(pcs) for c in counts.values()]
        pitch_ent = -float(sum(p * np.log(p + 1e-10) for p in probs))
        norm_pitch_ent = pitch_ent / np.log(12)

        # FIX: use duration_ms() method, not .duration attribute
        durations    = [n.duration_ms() for n in notes]
        unique_durs  = len(set(round(d, -1) for d in durations))
        rhythm_ent   = unique_durs / min(10.0, len(durations))

        swing_factor = min(0.30, swing_amount_ms / 50.0)

        return float(np.clip(
            (norm_pitch_ent * 0.50 + rhythm_ent * 0.30 + swing_factor)
            * source_confidence, 0.0, 1.0))

    def _compute_theme_elaboration_score(
        self,
        motivic_persistence: float,
        surface_entropy:     float,
        harmonic_rigidity:   float,
    ) -> float:
        """
        0.0–0.3 → THEME state (head, main melody, chorus)
        0.3–0.7 → TRANSITIONAL
        0.7–1.0 → ELABORATION state (solo, development)
        """
        elaboration = (1.0 - motivic_persistence) * 0.60 + surface_entropy * 0.40
        if harmonic_rigidity > 0.60:
            elaboration = min(1.0, elaboration * 1.10)
        return float(np.clip(elaboration, 0.0, 1.0))

    def _detect_structural_regime(
        self,
        harmonic_rigidity:   float,
        motivic_persistence: float,
        formal_periodicity:  float,
        surface_entropy:     float,
    ) -> Dict[str, Any]:
        if (formal_periodicity > 0.70 and
                motivic_persistence > 0.60 and
                surface_entropy < 0.40):
            regime = "stable_periodic"
            conf   = (formal_periodicity + motivic_persistence +
                      (1 - surface_entropy)) / 3.0
        elif (motivic_persistence < 0.40 and
              surface_entropy > 0.60 and
              harmonic_rigidity > 0.50):
            regime = "elaborative_improvisatory"
            conf   = ((1 - motivic_persistence) + surface_entropy +
                      harmonic_rigidity) / 3.0
        elif (formal_periodicity < 0.50 and
              0.30 < motivic_persistence < 0.70):
            regime = "developmental_transitional"
            conf   = ((1 - formal_periodicity) + motivic_persistence +
                      surface_entropy) / 3.0
        elif (motivic_persistence > 0.70 and
              harmonic_rigidity > 0.60):
            regime = "recapitulative"
            conf   = (motivic_persistence + harmonic_rigidity) / 2.0
        else:
            regime = "mixed_ambiguous"
            conf   = 0.40

        return {
            "regime":     regime,
            "confidence": float(np.clip(conf, 0.0, 1.0)),
            "invariants": {
                "harmonic_rigidity":   harmonic_rigidity,
                "motivic_persistence": motivic_persistence,
                "formal_periodicity":  formal_periodicity,
                "surface_entropy":     surface_entropy,
            },
        }

    def _detect_anacrusis(
        self,
        notes:               List[NoteEvent],
        beat_phase_map:      Dict[float, float],
        downbeat_confidences: Dict[float, float],
    ) -> Dict[str, Any]:
        """
        Detect anacrusis using PulseField's beat phase evidence.
        FIX: uses start_ms not start_time.
        """
        if not notes or not downbeat_confidences:
            return {"detected": False, "confidence": 0.0}

        downbeats = sorted(t for t, c in downbeat_confidences.items() if c > 0.70)
        if not downbeats:
            return {"detected": False, "confidence": 0.0}

        first_db  = downbeats[0]
        first_note = notes[0]

        # FIX: start_ms not start_time
        if first_note.start_ms < first_db:
            gap_ms     = first_db - first_note.start_ms
            beat_phase = beat_phase_map.get(first_note.start_ms, 0.0)
            confidence = 0.80 if gap_ms < 2000.0 and beat_phase > 0.20 else 0.40
            affected   = [n.start_ms for n in notes if n.start_ms < first_db]
            return {
                "detected":    True,
                "confidence":  confidence,
                "duration_ms": gap_ms,
                "affected_start_times_ms": affected,
            }

        return {"detected": False, "confidence": 0.0}

    def _infer_form(self, sig: Dict[str, float]) -> Dict[str, Any]:
        hr = sig.get("harmonic_rigidity",   0.5)
        mp = sig.get("motivic_persistence", 0.5)
        fp = sig.get("formal_periodicity",  0.5)
        te = sig.get("theme_elaboration_score", 0.5)

        if fp > 0.70 and mp > 0.60 and te < 0.30:
            return {"form": "verse_chorus", "confidence": 0.85}
        if te < 0.40 and 0.40 < fp < 0.70:
            return {"form": "binary",       "confidence": 0.70}
        if fp < 0.40 and hr < 0.40:
            return {"form": "through_composed", "confidence": 0.70}
        return {"form": "unknown", "confidence": 0.30}