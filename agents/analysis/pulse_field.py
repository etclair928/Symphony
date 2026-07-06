# =================================================================
# MODULE: agents/analysis/pulse_field.py
# DESCRIPTION: Pulse Field analysis - probabilistic pulse grid detection.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# PHILOSOPHY:
#     The Pulse Field represents the underlying pulse grid as a
#     probability field rather than a single tempo value.
#
#     This enables:
#     - Detection of multiple simultaneous pulse possibilities (polymeter)
#     - Graceful handling of tempo changes (rubato, accelerando)
#     - Probabilistic snap targets for Ritornello
#
#     "A note's value is defined by its distance to its neighbor,
#      not its distance to the grid" - the Pulse Field provides the
#      RELATIONAL context, not absolute grid positions.
#
# KEY ARCHITECTURE:
#     - Multi-hypothesis pulse tracking (polymeter support)
#     - Phase coherence detection with circular statistics
#     - Pulse strength annealing over time
#     - Integration with ReverseGeoCrypt for lattice-based pulse
#     - Integration with TempoIntelligence for tempo map
#     - Integration with GrooveField for relational timing
#     - StatusReporter throughout
#
# Authored by: DeepSeek - Complete 5.0 rewrite (2026-05-11)
# Based on 4.7 Pulse_Field and tempo tracking patterns.
# =================================================================

import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from collections import deque
from scipy.signal import find_peaks, correlate, medfilt
from scipy.stats import circmean, circvar

# Core imports - ONLY from bedrock
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    PulseField, PulseHypothesis, TempoMap, TempoEvent, BeatGrid, DecisionType
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    MIN_TEMPO_BPM,
    MAX_TEMPO_BPM,
    DEFAULT_TEMPO_BPM,
    PULSE_FIELD_RESOLUTION_MS,
    BEAT_TRACKING_HOP_MS,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    AnalysisAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)

# Optional imports with safe fallbacks
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy.signal import resample_poly

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ========================================================================
# Enums and Types
# ========================================================================

class PulseStability(str, Enum):
    """Stability level of the detected pulse."""
    LOCKED = "locked"  # Strong, consistent pulse
    STABLE = "stable"  # Good pulse, minor variations
    DRIFTING = "drifting"  # Tempo is gradually changing
    UNSTABLE = "unstable"  # Weak or inconsistent pulse
    POLYMETER = "polymeter"  # Multiple simultaneous pulses detected


@dataclass(frozen=True)
class PulseHypothesisResult:
    """Immutable result of a pulse hypothesis."""
    period_ms: float
    tempo_bpm: float
    probability: float
    harmonic_level: float
    phase_ms: float
    phase_coherence: float
    strength_at_peak: float


@dataclass(frozen=True)
class PulseFieldForensics:
    """Forensic breakdown of pulse field analysis."""
    hypothesis_count: int
    best_hypothesis_period_ms: float
    best_hypothesis_probability: float
    phase_coherence: float
    annealing_applied: bool
    tempo_variance: float
    stability: PulseStability


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class PulseFieldConfig:
    """Configuration for pulse field analysis."""

    # Pulse hypothesis parameters
    min_tempo_bpm: float = MIN_TEMPO_BPM  # 40
    max_tempo_bpm: float = MAX_TEMPO_BPM  # 240
    tempo_resolution_bpm: float = 1.0
    max_hypotheses: int = 5

    # Pulse detection
    pulse_strength_threshold: float = 0.3
    min_pulse_strength: float = 0.1

    # Phase detection (circular statistics)
    phase_resolution_ms: float = PULSE_FIELD_RESOLUTION_MS  # 10ms
    min_phase_coherence: float = 0.4
    phase_bins: int = 36  # 10-degree bins

    # Temporal smoothing
    annealing_window_sec: float = 4.0
    annealing_rate: float = 0.05  # 5% per second
    median_filter_size: int = 5

    # Multi-hypothesis tracking
    track_harmonics: bool = True
    harmonic_levels: List[float] = field(default_factory=lambda: [1, 2, 3, 4, 0.5, 0.25])

    # Beat tracking refinement
    refine_beats: bool = True
    beat_refinement_window_ms: int = 50

    # ReverseGeoCrypt integration
    use_reverse_geo_crypt: bool = True
    reverse_geo_weight: float = 0.3

    # Performance
    max_onset_history: int = 10000
    cleanup_after_analysis: bool = True


# ========================================================================
# Pulse Hypothesis Tracker
# ========================================================================

class PulseHypothesisTracker:
    """
    Tracks multiple pulse hypotheses simultaneously.

    Each hypothesis represents a possible pulse period (tempo)
    with its own phase and strength.
    """

    def __init__(self, config: PulseFieldConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self.hypotheses: Dict[float, PulseHypothesis] = {}
        self._history: deque = deque(maxlen=100)

    def initialize_hypotheses(self, tempo_bpm: float, confidence: float = 0.5):
        """Initialize pulse hypotheses from a base tempo."""
        base_period_ms = 60000 / tempo_bpm

        # Create hypotheses at different harmonic levels
        for level in self.config.harmonic_levels:
            period_ms = base_period_ms / level if level > 0 else base_period_ms * 2
            tempo = 60000 / period_ms if period_ms > 0 else tempo_bpm

            if self.config.min_tempo_bpm <= tempo <= self.config.max_tempo_bpm:
                self.hypotheses[period_ms] = PulseHypothesis(
                    period_ms=period_ms,
                    probability=confidence * (1.0 / (abs(level) if level != 0 else 1.0)),
                    harmonic_level=level
                )

        self._log_status(f"Initialized {len(self.hypotheses)} pulse hypotheses")

    def update_from_onsets(self, onset_times: List[float], onset_strengths: List[float]):
        """Update hypothesis probabilities based on detected onsets."""
        if len(onset_times) < 4:
            return

        # Calculate inter-onset intervals
        intervals = np.diff(onset_times)

        # For each hypothesis, compute how well onsets align
        for period_ms, hypothesis in self.hypotheses.items():
            alignment_score = self._compute_alignment_score(onset_times, period_ms)

            # Update probability with exponential moving average
            new_prob = hypothesis.probability * 0.7 + alignment_score * 0.3
            self.hypotheses[period_ms] = replace(
                hypothesis,
                probability=max(self.config.min_pulse_strength, min(1.0, new_prob))
            )

    def _compute_alignment_score(self, onset_times: List[float], period_ms: float) -> float:
        """Compute how well onsets align with a given period."""
        if len(onset_times) < 2:
            return 0.0

        # Calculate phases
        phases = [(onset % period_ms) / period_ms for onset in onset_times]

        # Find dominant phase using circular statistics
        rad_phases = [2 * np.pi * p for p in phases]
        try:
            mean_rad = circmean(rad_phases)
            variance = circvar(rad_phases)
            coherence = 1 - variance
        except Exception:
            # Fallback to histogram method
            hist, bin_edges = np.histogram(phases, bins=self.config.phase_bins, range=(0, 1))
            peak_idx = np.argmax(hist)
            coherence = hist[peak_idx] / len(phases)

        return coherence

    def update_from_lattice(self, lattice_period_ms: float, lattice_confidence: float):
        """
        Update hypotheses from ReverseGeoCrypt lattice results.

        This integrates the geometric lattice discovery with pulse tracking.
        """
        if lattice_period_ms <= 0:
            return

        # Find closest hypothesis
        closest_period = min(self.hypotheses.keys(),
                             key=lambda p: abs(p - lattice_period_ms))

        if closest_period in self.hypotheses:
            current = self.hypotheses[closest_period]
            weighted_prob = current.probability * (1 - self.config.reverse_geo_weight) + \
                            lattice_confidence * self.config.reverse_geo_weight

            self.hypotheses[closest_period] = replace(current, probability=weighted_prob)

    def get_best_hypothesis(self) -> Optional[PulseHypothesis]:
        """Return the highest probability hypothesis."""
        if not self.hypotheses:
            return None

        return max(self.hypotheses.values(), key=lambda h: h.probability)

    def get_all_hypotheses_sorted(self) -> List[PulseHypothesis]:
        """Return all hypotheses sorted by probability."""
        return sorted(self.hypotheses.values(), key=lambda h: h.probability, reverse=True)

    def get_hypothesis_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all hypotheses."""
        summary = []
        for period_ms, hyp in self.hypotheses.items():
            summary.append({
                "period_ms": period_ms,
                "tempo_bpm": 60000 / period_ms,
                "probability": hyp.probability,
                "harmonic_level": hyp.harmonic_level
            })
        return sorted(summary, key=lambda x: x["probability"], reverse=True)

    def _log_status(self, message: str, level: str = "info"):
        if self.status_reporter:
            getattr(self.status_reporter, level)("PulseHypothesisTracker", message)


# ========================================================================
# Phase Coherence Detector (Circular Statistics)
# ========================================================================

class PhaseCoherenceDetector:
    """
    Detects phase coherence across multiple pulse hypotheses.

    Uses circular statistics for robust phase detection.
    """

    def __init__(self, config: PulseFieldConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._phase_history: Dict[float, deque] = {}

    def detect_phase(self, onset_times: List[float], period_ms: float) -> Tuple[float, float]:
        """
        Detect most likely phase offset for a given period.

        Returns:
            Tuple of (phase_offset_ms, coherence)
        """
        if len(onset_times) < 3:
            return 0.0, 0.0

        # Calculate phases
        phases = [onset % period_ms for onset in onset_times]

        # Convert to radians for circular statistics
        rad_phases = [2 * np.pi * p / period_ms for p in phases]

        try:
            # Circular mean (phase)
            mean_rad = circmean(rad_phases)
            mean_phase_ms = (mean_rad / (2 * np.pi)) * period_ms

            # Circular variance (coherence)
            variance = circvar(rad_phases)
            coherence = 1 - variance

        except Exception:
            # Fallback to histogram method
            hist, bin_edges = np.histogram(phases, bins=self.config.phase_bins)
            peak_idx = np.argmax(hist)
            mean_phase_ms = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
            coherence = hist[peak_idx] / len(phases)

        return mean_phase_ms, coherence

    def refine_phase_with_onsets(self, phase_ms: float, period_ms: float,
                                 onset_times: List[float]) -> float:
        """
        Refine phase using local onset alignment.

        Adjusts phase to align with nearest strong onsets.
        """
        if not onset_times:
            return phase_ms

        # Find onsets near predicted beats
        predicted_beats = [phase_ms + i * period_ms for i in range(-3, 4)]

        refinements = []
        for beat in predicted_beats:
            # Find nearest onset
            distances = [abs(onset - beat) for onset in onset_times]
            min_idx = np.argmin(distances)
            nearest_onset = onset_times[min_idx]
            distance = distances[min_idx]

            if distance < self.config.beat_refinement_window_ms:
                refinements.append(nearest_onset - beat)

        if refinements:
            # Use median refinement for robustness
            median_refinement = np.median(refinements)
            return phase_ms + median_refinement

        return phase_ms

    def get_phase_coherence_over_time(self, period_ms: float) -> float:
        """Get coherence trend over time for a period."""
        if period_ms not in self._phase_history:
            return 0.0

        history = list(self._phase_history[period_ms])
        if len(history) < 2:
            return 0.0

        # Calculate trend (increasing coherence = good)
        return np.mean(history)


# ========================================================================
# Pulse Strength Annealer (Temporal Smoothing)
# ========================================================================

class PulseStrengthAnnealer:
    """
    Smooth pulse strengths over time using annealing.

    Prevents sudden jumps in pulse detection when tempo changes.
    """

    def __init__(self, config: PulseFieldConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._strength_history: Dict[float, deque] = {}
        self._last_time_ms: float = 0.0

    def anneal_strength(self, period_ms: float, current_strength: float,
                        time_ms: float) -> float:
        """
        Apply temporal annealing to pulse strength.

        Strength gradually decays when not reinforced.
        """
        if period_ms not in self._strength_history:
            self._strength_history[period_ms] = deque(maxlen=10)
            self._strength_history[period_ms].append(current_strength)
            return current_strength

        # Calculate time since last update
        delta_sec = (time_ms - self._last_time_ms) / 1000 if self._last_time_ms > 0 else 0
        self._last_time_ms = time_ms

        # Decay factor (exponential decay)
        decay = np.exp(-delta_sec * self.config.annealing_rate)

        # Get last strength
        last_strength = self._strength_history[period_ms][-1]

        # Apply decay and blend with new strength
        annealed = max(current_strength, last_strength * decay)
        annealed = min(1.0, annealed)

        self._strength_history[period_ms].append(annealed)

        return annealed

    def get_strength_trend(self, period_ms: float) -> float:
        """Get trend direction (-1 to 1, positive = increasing)."""
        if period_ms not in self._strength_history or len(self._strength_history[period_ms]) < 3:
            return 0.0

        history = list(self._strength_history[period_ms])
        if len(history) >= 3:
            # Linear trend
            x = np.arange(len(history))
            slope = np.polyfit(x[-10:], history[-10:], 1)[0] if len(history) >= 10 else np.polyfit(x, history, 1)[0]
            return np.clip(slope * 10, -1.0, 1.0)

        return 0.0


# ========================================================================
# Pulse Grid Builder
# ========================================================================

class PulseGridBuilder:
    """Builds beat and sixteenth grids from pulse hypothesis."""

    def __init__(self, config: PulseFieldConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def build_beat_grid(self, phase_ms: float, period_ms: float,
                        duration_ms: float) -> List[float]:
        """Build beat grid from phase and period."""
        if period_ms <= 0:
            return []

        beat_times = []

        # Find first beat before time 0
        start_beat = phase_ms
        while start_beat > 0:
            start_beat -= period_ms

        # Generate beats
        beat = start_beat
        while beat <= duration_ms:
            if beat >= 0:
                beat_times.append(beat)
            beat += period_ms

        return beat_times

    def build_sixteenth_grid(self, beat_grid: List[float], period_ms: float,
                             duration_ms: float) -> List[float]:
        """Build sixteenth note grid from beat grid."""
        if not beat_grid or period_ms <= 0:
            return []

        sixteenth_ms = period_ms / 4
        sixteenth_grid = []

        for beat in beat_grid:
            for offset in [0, sixteenth_ms, sixteenth_ms * 2, sixteenth_ms * 3]:
                sixteenth_time = beat + offset
                if sixteenth_time <= duration_ms + sixteenth_ms:
                    sixteenth_grid.append(sixteenth_time)

        return sorted(set(sixteenth_grid))

    def build_pulse_strengths(self, beat_grid: List[float],
                              onset_times: List[float],
                              onset_strengths: List[float]) -> List[float]:
        """Build pulse strength array for each beat."""
        if not beat_grid or not onset_times:
            return [0.5] * len(beat_grid)

        pulse_strengths = []

        for beat in beat_grid:
            # Find onsets near this beat
            nearby_strengths = []
            for onset, strength in zip(onset_times, onset_strengths):
                distance = abs(onset - beat)
                if distance < 50:  # Within 50ms
                    # Weight by distance
                    weight = 1 - (distance / 50)
                    nearby_strengths.append(strength * weight)

            if nearby_strengths:
                strength = max(nearby_strengths)
            else:
                strength = 0.2  # Default low strength

            pulse_strengths.append(min(1.0, strength))

        # Apply median filter for smoothing
        if len(pulse_strengths) > self.config.median_filter_size:
            pulse_strengths = medfilt(pulse_strengths, self.config.median_filter_size)

        return pulse_strengths


# ========================================================================
# Main Pulse Field Analyzer Agent
# ========================================================================

class PulseFieldAnalyzer:
    """
    Pulse Field analysis - probabilistic pulse grid detection for Grimlock 5.0.

    Features:
        - Multi-hypothesis pulse tracking (polymeter support)
        - Phase coherence detection with circular statistics
        - Pulse strength annealing over time
        - Integration with ReverseGeoCrypt for lattice-based pulse
        - Integration with TempoIntelligence for tempo map
        - Integration with GrooveField for relational timing

    Law 2 Compliance: Does NOT import other agents (receives data via parameters).
    """

    def __init__(
            self,
            config: Optional[PulseFieldConfig] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Args:
            config: Pulse field configuration
            status_reporter: StatusReporter for progress
            music_box: MusicBox for forensic logging
            progress_callback: Optional progress callback
        """
        self._name = "pulse_field_analyzer"
        self._source_type = SourceType.PULSE_FIELD
        self._config = config or PulseFieldConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback

        # Initialize components
        self._hypothesis_tracker = PulseHypothesisTracker(self._config, status_reporter)
        self._phase_detector = PhaseCoherenceDetector(self._config, status_reporter)
        self._strength_annealer = PulseStrengthAnnealer(self._config, status_reporter)
        self._grid_builder = PulseGridBuilder(self._config, status_reporter)

        # State
        self._last_pulse_field: Optional[PulseField] = None
        self._last_forensics: Optional[PulseFieldForensics] = None
        self._last_onset_times: List[float] = []
        self._tempo_history: List[float] = []
        self._last_execution_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0

        self._log_status("PulseFieldAnalyzer initialized")

    # ========================================================================
    # AgentProtocol Implementation
    # ========================================================================

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def name(self) -> str:
        return self._name

    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """Process audio and return StageResult with pulse field."""
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting pulse field analysis")

        try:
            # Extract onsets from audio if not provided in context
            onset_times, onset_strengths = self._extract_onsets(audio_buffer, context)

            # Get tempo hint from context if available
            tempo_hint = self._get_tempo_hint(context)

            # Analyze pulse field
            pulse_field = self.analyze_pulse_field(
                onset_times, onset_strengths, audio_buffer, tempo_hint
            )

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory
            self._last_execution_time_ms = execution_time_ms

            # Log to MusicBox
            if self._music_box:
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="pulse_field_analysis",
                    before_state={"onset_count": len(onset_times)},
                    after_state={
                        "tempo_bpm": pulse_field.tempo_bpm,
                        "confidence": pulse_field.confidence.value,
                        "beat_count": len(pulse_field.beat_grid_ms),
                        "phase_shift_ms": pulse_field.phase_shift_ms,
                        "is_double_time": pulse_field.is_double_time,
                        "is_half_time": pulse_field.is_half_time
                    },
                    reasoning=f"Pulse field: {pulse_field.tempo_bpm:.1f} BPM",
                    reversible=True
                )

            # Force cleanup
            if self._config.cleanup_after_analysis:
                self._force_cleanup()

            self._update_progress(1.0, f"Complete: {pulse_field.tempo_bpm:.1f} BPM")

            return StageResult(
                stage_name=self._name,
                success=True,
                events=[],
                metadata={
                    "tempo_bpm": pulse_field.tempo_bpm,
                    "confidence": pulse_field.confidence.value,
                    "beat_count": len(pulse_field.beat_grid_ms),
                    "sixteenth_count": len(pulse_field.sixteenth_grid_ms),
                    "phase_shift_ms": pulse_field.phase_shift_ms,
                    "is_double_time": pulse_field.is_double_time,
                    "is_half_time": pulse_field.is_half_time,
                    "pulse_stability": self._last_forensics.stability.value if self._last_forensics else "unknown",
                    "hypothesis_count": self._last_forensics.hypothesis_count if self._last_forensics else 0,
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Pulse field analysis failed: {e}", "error")
            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def analyze_pulse_field(
            self,
            onset_times: List[float],
            onset_strengths: List[float],
            audio: Optional[np.ndarray] = None,
            tempo_hint: Optional[float] = None,
            lattice_period_ms: Optional[float] = None,
            lattice_confidence: float = 0.0
    ) -> PulseField:
        """
        Analyze pulse field from onset times.

        This is the main entry point - computes the probabilistic pulse grid.

        Args:
            onset_times: Detected onset timestamps in milliseconds
            onset_strengths: Strength of each onset (0-1)
            audio: Optional audio buffer for refinement
            tempo_hint: Optional tempo hint in BPM
            lattice_period_ms: Optional period from ReverseGeoCrypt
            lattice_confidence: Confidence of ReverseGeoCrypt result

        Returns:
            PulseField object with beat and sixteenth grids
        """
        self._update_progress(0.1, "Validating onset data")

        if len(onset_times) < 4:
            self._log_status(f"Insufficient onsets: {len(onset_times)}", "warn")
            return self._create_default_pulse_field(tempo_hint)

        self._last_onset_times = onset_times[-self._config.max_onset_history:]

        # Step 1: Initialize hypotheses
        self._update_progress(0.2, "Initializing pulse hypotheses")
        if tempo_hint:
            self._hypothesis_tracker.initialize_hypotheses(tempo_hint)
        else:
            estimated_tempo = self._estimate_tempo_from_onsets(onset_times)
            self._hypothesis_tracker.initialize_hypotheses(estimated_tempo, confidence=0.6)

        # Step 2: Integrate ReverseGeoCrypt lattice if provided
        if self._config.use_reverse_geo_crypt and lattice_period_ms and lattice_period_ms > 0:
            self._update_progress(0.3, "Integrating geometric lattice")
            self._hypothesis_tracker.update_from_lattice(lattice_period_ms, lattice_confidence)

        # Step 3: Update hypotheses with onsets
        self._update_progress(0.4, "Tracking pulse hypotheses")
        self._hypothesis_tracker.update_from_onsets(onset_times, onset_strengths)

        # Step 4: Get best hypothesis
        best_hypothesis = self._hypothesis_tracker.get_best_hypothesis()
        if best_hypothesis is None:
            return self._create_default_pulse_field(tempo_hint)

        period_ms = best_hypothesis.period_ms
        tempo_bpm = 60000 / period_ms

        # Step 5: Detect phase
        self._update_progress(0.6, "Detecting phase coherence")
        phase_ms, phase_coherence = self._phase_detector.detect_phase(onset_times, period_ms)

        # Step 6: Refine phase with audio if available
        if audio is not None and self._config.refine_beats:
            self._update_progress(0.7, "Refining phase")
            phase_ms = self._phase_detector.refine_phase_with_onsets(
                phase_ms, period_ms, onset_times
            )

        # Step 7: Apply annealing to pulse strength
        self._update_progress(0.8, "Applying temporal annealing")
        current_time = onset_times[-1] if onset_times else 0
        pulse_strength = self._strength_annealer.anneal_strength(
            period_ms, best_hypothesis.probability, current_time
        )

        # Step 8: Build grids
        self._update_progress(0.85, "Building pulse grids")
        duration_ms = onset_times[-1] + 5000 if onset_times else 60000
        beat_grid = self._grid_builder.build_beat_grid(phase_ms, period_ms, duration_ms)
        sixteenth_grid = self._grid_builder.build_sixteenth_grid(beat_grid, period_ms, duration_ms)
        pulse_strengths = self._grid_builder.build_pulse_strengths(beat_grid, onset_times, onset_strengths)

        # Step 9: Build forensic data
        self._update_progress(0.95, "Building forensic data")
        self._tempo_history.append(tempo_bpm)
        if len(self._tempo_history) > 100:
            self._tempo_history.pop(0)

        tempo_variance = np.var(self._tempo_history) if len(self._tempo_history) > 1 else 0.0
        stability = self._determine_stability(pulse_strength, phase_coherence, tempo_variance)

        hypothesis_count = len(self._hypothesis_tracker.hypotheses)
        self._last_forensics = PulseFieldForensics(
            hypothesis_count=hypothesis_count,
            best_hypothesis_period_ms=period_ms,
            best_hypothesis_probability=best_hypothesis.probability,
            phase_coherence=phase_coherence,
            annealing_applied=True,
            tempo_variance=tempo_variance,
            stability=stability
        )

        self._update_progress(1.0, "Pulse field analysis complete")

        # Create PulseField object
        pulse_field = PulseField(
            tempo_bpm=tempo_bpm,
            confidence=Confidence.from_float(pulse_strength),
            beat_grid_ms=beat_grid,
            sixteenth_grid_ms=sixteenth_grid,
            pulse_strength=pulse_strengths,
            phase_shift_ms=phase_ms,
            is_double_time=best_hypothesis.harmonic_level == 0.5,
            is_half_time=best_hypothesis.harmonic_level == 2
        )

        self._last_pulse_field = pulse_field

        self._log_status(f"Pulse field: {tempo_bpm:.1f} BPM, {len(beat_grid)} beats, "
                         f"confidence {pulse_strength:.2f}, stability {stability.value}")

        if self._music_box is not None:
            self._music_box.log_decision(
                stage_name="pulse_field",
                decision_type=DecisionType.ANALYSIS_EVIDENCE,
                before_state={"onset_count": len(onset_times)},
                after_state={
                    "tempo_bpm": tempo_bpm,
                    "beat_count": len(beat_grid),
                    "pulse_strength": pulse_strength,
                    "stability": stability.value,
                    # forensic breakdown behind that pulse field - computed
                    # into self._last_forensics and never seen again
                    "hypothesis_count": hypothesis_count,
                    "best_hypothesis_probability": best_hypothesis.probability,
                    "phase_coherence": phase_coherence,
                    "tempo_variance": tempo_variance,
                },
                reasoning=f"Pulse field: {tempo_bpm:.1f} BPM, {hypothesis_count} hypotheses tracked, "
                          f"stability {stability.value}",
                reversible=True,
            )

        return pulse_field

    def integrate_with_reverse_geo_crypt(self, lattice_period_ms: float, confidence: float):
        """
        Integrate results from ReverseGeoCrypt.

        Args:
            lattice_period_ms: Period discovered by ReverseGeoCrypt
            confidence: Confidence of the lattice detection
        """
        if not self._config.use_reverse_geo_crypt:
            return

        self._hypothesis_tracker.update_from_lattice(lattice_period_ms, confidence)
        self._log_status(f"Integrated ReverseGeoCrypt lattice: period={lattice_period_ms:.1f}ms")

    def integrate_with_tempo_intelligence(self, tempo_map: TempoMap):
        """
        Integrate results from TempoIntelligence.

        Args:
            tempo_map: TempoMap from TempoIntelligence
        """
        tempo_bpm = tempo_map.initial_tempo_bpm
        self._hypothesis_tracker.initialize_hypotheses(tempo_bpm, tempo_map.confidence.value)
        self._log_status(f"Integrated TempoIntelligence: {tempo_bpm:.1f} BPM")

    def to_tempo_map(self) -> TempoMap:
        """Convert pulse field to TempoMap for export."""
        if self._last_pulse_field is None:
            return TempoMap(initial_tempo_bpm=DEFAULT_TEMPO_BPM)

        # Create tempo events from strong beats
        tempo_events = []
        beat_grid = self._last_pulse_field.beat_grid_ms
        pulse_strengths = self._last_pulse_field.pulse_strength

        for i, (beat_time, strength) in enumerate(zip(beat_grid, pulse_strengths)):
            # Create tempo event for strong downbeats
            if strength > 0.6 and i % 4 == 0:
                tempo_events.append(TempoEvent(
                    time_ms=beat_time,
                    tempo_bpm=self._last_pulse_field.tempo_bpm,
                    confidence=Confidence.from_float(strength)
                ))

        return TempoMap(
            initial_tempo_bpm=self._last_pulse_field.tempo_bpm,
            tempo_events=tempo_events,
            confidence=self._last_pulse_field.confidence
        )

    def to_beat_grid(self) -> Optional[BeatGrid]:
        """Convert pulse field to BeatGrid."""
        if self._last_pulse_field is None:
            return None

        beat_duration_ms = 60000 / self._last_pulse_field.tempo_bpm
        measure_duration_ms = beat_duration_ms * 4  # Assume 4/4

        # Estimate downbeats (every 4th beat, but adjusted for phase)
        downbeats = []
        phase_shift = self._last_pulse_field.phase_shift_ms

        for beat in self._last_pulse_field.beat_grid_ms:
            adjusted = (beat - phase_shift) % measure_duration_ms
            if adjusted < beat_duration_ms:
                downbeats.append(beat)

        return BeatGrid(
            beat_times_ms=self._last_pulse_field.beat_grid_ms,
            downbeat_times_ms=downbeats,
            measure_duration_ms=measure_duration_ms,
            beat_duration_ms=beat_duration_ms,
            tempo_bpm=self._last_pulse_field.tempo_bpm,
            time_signature="4/4"
        )

    def get_pulse_at_time(self, time_ms: float) -> Tuple[float, float]:
        """
        Get pulse information at a specific time.

        Returns:
            Tuple of (nearest_beat_time_ms, pulse_strength)
        """
        if self._last_pulse_field is None:
            return time_ms, 0.5

        nearest_beat = self._last_pulse_field.get_closest_pulse(time_ms)
        beat_index = self._last_pulse_field.get_beat_index(time_ms)

        if beat_index >= 0 and beat_index < len(self._last_pulse_field.pulse_strength):
            strength = self._last_pulse_field.pulse_strength[beat_index]
        else:
            strength = 0.5

        return nearest_beat, strength

    def get_strength_trend(self) -> float:
        """Get trend of pulse strength (-1 to 1, positive = increasing)."""
        if self._last_pulse_field is None:
            return 0.0

        period_ms = 60000 / self._last_pulse_field.tempo_bpm
        return self._strength_annealer.get_strength_trend(period_ms)

    def get_hypotheses_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all tracked hypotheses."""
        return self._hypothesis_tracker.get_hypothesis_summary()

    def _estimate_tempo_from_onsets(self, onset_times: List[float]) -> float:
        """Estimate tempo from inter-onset intervals."""
        if len(onset_times) < 4:
            return DEFAULT_TEMPO_BPM

        intervals = np.diff(onset_times)

        # Remove outliers
        median_interval = np.median(intervals)
        valid_intervals = intervals[
            (intervals > median_interval * 0.5) &
            (intervals < median_interval * 1.5)
            ]

        if len(valid_intervals) > 0:
            mean_interval = np.mean(valid_intervals)
            tempo_bpm = 60000 / mean_interval
        else:
            tempo_bpm = 60000 / median_interval

        return max(self._config.min_tempo_bpm, min(self._config.max_tempo_bpm, tempo_bpm))

    def _extract_onsets(self, audio: np.ndarray, context: AudioContext) -> Tuple[List[float], List[float]]:
        """Extract onsets from audio if not provided."""
        if not LIBROSA_AVAILABLE:
            return [], []

        try:
            import librosa

            hop_length = 512
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=context.working_sample_rate,
                hop_length=hop_length,
                backtrack=True
            )
            onset_times = onset_frames * hop_length / context.working_sample_rate * 1000

            # Get onset strengths
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=context.working_sample_rate,
                hop_length=hop_length
            )
            onset_strengths = [onset_env[frame] / np.max(onset_env) if np.max(onset_env) > 0 else 0.5
                               for frame in onset_frames]

            return onset_times.tolist(), onset_strengths

        except Exception as e:
            self._log_status(f"Onset extraction failed: {e}", "warn")
            return [], []

    def _get_tempo_hint(self, context: AudioContext) -> Optional[float]:
        """Get tempo hint from context if available."""
        if hasattr(context, 'tempo_hint') and context.tempo_hint:
            return context.tempo_hint
        return None

    def _determine_stability(self, pulse_strength: float, phase_coherence: float,
                             tempo_variance: float) -> PulseStability:
        """Determine pulse stability level."""
        if pulse_strength > 0.8 and phase_coherence > 0.7 and tempo_variance < 1.0:
            return PulseStability.LOCKED
        elif pulse_strength > 0.6 and phase_coherence > 0.5 and tempo_variance < 5.0:
            return PulseStability.STABLE
        elif tempo_variance > 20.0:
            return PulseStability.UNSTABLE
        elif tempo_variance > 5.0:
            return PulseStability.DRIFTING
        elif len(self._hypothesis_tracker.hypotheses) > 2:
            return PulseStability.POLYMETER
        else:
            return PulseStability.DRIFTING

    def _create_default_pulse_field(self, tempo_hint: Optional[float] = None) -> PulseField:
        """Create default pulse field when insufficient data."""
        tempo = tempo_hint or DEFAULT_TEMPO_BPM
        period_ms = 60000 / tempo
        duration_ms = 8000  # 8 seconds default

        beat_grid = list(np.arange(0, duration_ms, period_ms))
        sixteenth_grid = []
        for beat in beat_grid:
            for offset in [0, period_ms / 4, period_ms / 2, period_ms * 3 / 4]:
                sixteenth_grid.append(beat + offset)

        return PulseField(
            tempo_bpm=tempo,
            confidence=Confidence.LOW,
            beat_grid_ms=beat_grid,
            sixteenth_grid_ms=sorted(set(sixteenth_grid)),
            pulse_strength=[0.3] * len(beat_grid),
            phase_shift_ms=0.0,
            is_double_time=False,
            is_half_time=False
        )

    def _force_cleanup(self):
        """Force garbage collection."""
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def _update_progress(self, progress: float, message: str):
        if self._progress_callback:
            self._progress_callback(progress, message)
        if self._status_reporter:
            self._status_reporter.progress(self._name, progress, message)

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            getattr(self._status_reporter, level)(self._name, message)

    def _get_current_memory_mb(self) -> float:
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    # ========================================================================
    # ScribeValidatable Implementation
    # ========================================================================

    def validate(self, gate: ValidationGate) -> ValidationResult:
        if self._last_pulse_field is None:
            return ValidationResult(
                is_valid=False,
                gate_used=gate,
                reason=VetoReason.EMPTY_RESULT,
                detail="No pulse field analysis performed"
            )

        if gate == ValidationGate.CONFIDENCE_THRESHOLD:
            is_valid = self._last_pulse_field.confidence.value >= Confidence.LOW.value

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                detail=f"Pulse confidence: {self._last_pulse_field.confidence.value:.2f}",
                confidence_before=self._last_pulse_field.confidence.value,
                confidence_after=self._last_pulse_field.confidence.value if is_valid else 0.0
            )

        elif gate == ValidationGate.TEMPO_REASONABLENESS:
            is_valid = (self._config.min_tempo_bpm <=
                        self._last_pulse_field.tempo_bpm <=
                        self._config.max_tempo_bpm)

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.TEMPO_OUTLIER,
                detail=f"Tempo: {self._last_pulse_field.tempo_bpm:.1f} BPM"
            )

        else:
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                detail=f"Gate {gate.value} not fully supported"
            )

    def get_confidence(self) -> Confidence:
        if self._last_pulse_field is None:
            return Confidence.HALLUCINATION
        return self._last_pulse_field.confidence

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        if self._last_pulse_field is None:
            return (VetoReason.EMPTY_RESULT, "No pulse field")

        if self._last_pulse_field.confidence.value < Confidence.LOW.value:
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Confidence {self._last_pulse_field.confidence.value:.2f}")

        if not (self._config.min_tempo_bpm <= self._last_pulse_field.tempo_bpm <= self._config.max_tempo_bpm):
            return (VetoReason.TEMPO_OUTLIER,
                    f"Tempo {self._last_pulse_field.tempo_bpm:.1f} BPM out of range")

        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        return SchoenbergResult(
            verdict=SchoenbergVerdict.UNCERTAIN,
            zero_crossing_rate=0.0,
            spectral_flatness=0.5,
            reason="PulseFieldAnalyzer analyzes timing, not harmonic series"
        )

    # ========================================================================
    # MemoryManagedProtocol Implementation
    # ========================================================================

    def release_buffer(self, buffer_name: str) -> None:
        pass

    def get_memory_footprint_mb(self) -> float:
        return self._get_current_memory_mb()

    def can_release(self, buffer_name: str) -> bool:
        return True

    def staggered_gc(self) -> Dict[str, Any]:
        before = self._get_current_memory_mb()
        self._force_cleanup()
        after = self._get_current_memory_mb()

        freed = before - after
        self._total_memory_freed_mb += max(0, freed)

        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": freed,
            "triggered_by": self._name,
            "total_freed_mb": self._total_memory_freed_mb
        }

    # ========================================================================
    # Public Methods
    # ========================================================================

    def get_last_pulse_field(self) -> Optional[PulseField]:
        return self._last_pulse_field

    def get_forensics(self) -> Optional[PulseFieldForensics]:
        return self._last_forensics

    def get_statistics(self) -> Dict[str, Any]:
        if self._last_pulse_field:
            return {
                "name": self._name,
                "source_type": self._source_type.value,
                "tempo_bpm": self._last_pulse_field.tempo_bpm,
                "confidence": self._last_pulse_field.confidence.value,
                "beat_count": len(self._last_pulse_field.beat_grid_ms),
                "sixteenth_count": len(self._last_pulse_field.sixteenth_grid_ms),
                "phase_shift_ms": self._last_pulse_field.phase_shift_ms,
                "is_double_time": self._last_pulse_field.is_double_time,
                "is_half_time": self._last_pulse_field.is_half_time,
                "pulse_stability": self._last_forensics.stability.value if self._last_forensics else None,
                "hypothesis_count": self._last_forensics.hypothesis_count if self._last_forensics else 0,
                "phase_coherence": self._last_forensics.phase_coherence if self._last_forensics else 0,
                "last_execution_time_ms": self._last_execution_time_ms,
                "total_memory_freed_mb": self._total_memory_freed_mb
            }
        else:
            return {
                "name": self._name,
                "source_type": self._source_type.value,
                "last_execution_time_ms": self._last_execution_time_ms,
                "total_memory_freed_mb": self._total_memory_freed_mb
            }


# ========================================================================
# Convenience Functions
# ========================================================================

def create_pulse_field_analyzer(
        min_tempo: float = MIN_TEMPO_BPM,
        max_tempo: float = MAX_TEMPO_BPM,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> PulseFieldAnalyzer:
    """Create a configured PulseFieldAnalyzer instance."""
    config = PulseFieldConfig(
        min_tempo_bpm=min_tempo,
        max_tempo_bpm=max_tempo
    )
    return PulseFieldAnalyzer(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )


def quick_pulse_test(audio_path: str) -> Dict[str, Any]:
    """Quick test function for Pulse Field Analyzer."""
    import librosa

    print(f"Testing Pulse Field Analyzer on {audio_path}")

    audio, sr = librosa.load(audio_path, mono=True, sr=TARGET_SAMPLE_RATE)
    print(f"Audio loaded: {len(audio) / sr:.1f}s")

    context = AudioContext(
        file_path=audio_path,
        original_sample_rate=sr,
        duration_seconds=len(audio) / sr,
        num_channels_original=1,
        working_sample_rate=sr,
        is_mono=True
    )

    analyzer = create_pulse_field_analyzer()
    result = analyzer.run(audio, context)
    stats = analyzer.get_statistics()

    print(f"  Success: {result.success}")
    print(f"  Tempo: {stats.get('tempo_bpm', 0):.1f} BPM")
    print(f"  Confidence: {stats.get('confidence', 0):.2f}")
    print(f"  Stability: {stats.get('pulse_stability', 'unknown')}")
    print(f"  Hypothesis count: {stats.get('hypothesis_count', 0)}")
    print(f"  Beats: {stats.get('beat_count', 0)}")

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        quick_pulse_test(sys.argv[1])
    else:
        print("Usage: python pulse_field.py <audio_file>")