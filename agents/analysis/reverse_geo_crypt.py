# =================================================================
# MODULE: agents/analysis/reverse_geo_crypt.py
# DESCRIPTION: Rhythmic Cryptanalysis Engine for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# PHILOSOPHY:
#     Don't ask "what is the tempo?"
#     Ask "what geometric lattice explains these events?"
#
#     Tempo is not an input. Tempo is the OUTPUT of finding the lattice
#     that minimizes structural entropy.
#
# CORE INSIGHT FOR 5.0:
#     The macro pulse (45 BPM) loses elections in 4.7 because witnesses
#     vote on transient density. This module votes on RATIO RELATIONSHIPS.
#
#     It doesn't need stems. It doesn't need Madmom. It needs EVENTS.
#     Events can come from ANYWHERE: onsets, chord changes, accents,
#     silence releases, phrase boundaries.
#
# KEY ARCHITECTURE:
#     - Shared Event/Anchor sources (not recomputed)
#     - Immutable dataclasses (frozen=True)
#     - WitnessTestimony output (not mutated state)
#     - StatusReporter integration
#     - No agent imports - only consumes evidence
#     - Memory management with GC triggers
#     - Forensic logging via MusicBox
#
# Authored by: DeepSeek - Complete 5.0 rewrite (2026-05-11)
# Based on 4.7 reverse_geo_crypt.py philosophy
# =================================================================

import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from collections import defaultdict
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# Core imports - ONLY from bedrock
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    PulseField, TempoMap, TempoEvent, BeatGrid, DecisionType
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    MIN_TEMPO_BPM,
    MAX_TEMPO_BPM,
    DEFAULT_TEMPO_BPM,
    BEAT_TRACKING_HOP_MS,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    AnalysisAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

EPS = 1e-10


# ========================================================================
# Enums and Types
# ========================================================================

class EventType(str, Enum):
    """Types of musical events that can feed into cryptanalysis."""
    ONSET = "onset"
    CHORD_CHANGE = "chord_change"
    ACCENT = "accent"
    SILENCE = "silence"
    PHRASE_BOUNDARY = "phrase_boundary"
    HARMONIC_SHIFT = "harmonic_shift"
    BASS_PEAK = "bass_peak"
    DRUM_ACCENT = "drum_accent"


class RatioFamily(str, Enum):
    """Families of rhythmic ratios."""
    BINARY = "binary"
    TERNARY = "ternary"
    SWING = "swing"
    ADDITIVE = "additive"
    QUINTUPLET = "quintuplet"
    SEPTUPLET = "septuplet"


class LatticeConfidence(str, Enum):
    """Confidence levels for lattice detection."""
    LOCKED = "locked"  # Multiple anchors, low entropy
    PROBABLE = "probable"  # Clear ratio clusters
    AMBIGUOUS = "ambiguous"  # Multiple candidates
    SPARSE = "sparse"  # Insufficient events


# ========================================================================
# Immutable Data Classes
# ========================================================================

@dataclass(frozen=True)
class Event:
    """
    A meaningful musical moment. Source-agnostic.

    LAW: Immutable. Never mutated after creation.
    """
    time: float  # seconds
    strength: float  # 0-1, confidence in this event
    event_type: EventType
    source: SourceType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate event data."""
        if not 0 <= self.strength <= 1:
            object.__setattr__(self, 'strength', max(0.0, min(1.0, self.strength)))


@dataclass(frozen=True)
class Anchor:
    """
    A confirmed structural downbeat. Locks the phase.

    LAW: Immutable. Anchors are sacred and never mutated.
    """
    time: float  # seconds
    confidence: float  # 0-1, how certain this is a downbeat
    source: SourceType
    witness_votes: Tuple[WitnessTestimony, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            object.__setattr__(self, 'confidence', max(0.0, min(1.0, self.confidence)))


@dataclass(frozen=True)
class GeometricLattice:
    """
    The hidden grid that explains rhythmic relationships.

    This is the OUTPUT of ReverseGeoCrypt - the discovered structure.
    """
    subdivision_sec: float  # Smallest grid unit in seconds
    subdivision_ms: float  # Smallest grid unit in milliseconds
    ratio_family: RatioFamily  # binary, ternary, swing, etc.
    error_score: float  # How much events deviate from grid (0-1, lower=better)
    entropy: float  # Rubato / unpredictability (0-1)
    confidence: float  # Overall confidence (0-1)
    confidence_level: LatticeConfidence
    derived_bpm: float  # BPM of the subdivision
    phase: float  # 0-1, where grid starts relative to t=0
    anchor_alignment: float  # How well anchors fit (0-1)

    # Forensic data
    candidate_periods_tested: int = 0
    events_analyzed: int = 0
    anchors_used: int = 0

    def to_tactus_bpm(self) -> float:
        """Convert subdivision to foot-tapping pulse (macro tempo)."""
        if self.ratio_family == RatioFamily.BINARY:
            return self.derived_bpm / 4
        elif self.ratio_family == RatioFamily.TERNARY:
            return self.derived_bpm / 3
        elif self.ratio_family == RatioFamily.SWING:
            return self.derived_bpm / 2
        else:
            return self.derived_bpm / 4

    def to_pulse_field(self, confidence: Optional[Confidence] = None) -> PulseField:
        """Convert lattice to PulseField for downstream quantization."""
        beat_grid_ms = []
        current = self.phase * self.subdivision_ms
        while current < 30000:  # 30 seconds max for grid
            beat_grid_ms.append(current)
            current += self.subdivision_ms * 4  # Quarter note grid

        # subdivision_ms IS the finest discovered grid unit (see the
        # class docstring), so the sixteenth-note grid is just that same
        # period stepped on its own, one point per subdivision, over the
        # same range as beat_grid_ms - this was left as an empty list
        # with a comment saying "will be populated by caller," but no
        # caller ever did, disabling 16th-note-resolution grid snapping
        # in PulseFieldAnalyzer whenever ReverseGeoCrypt was the active
        # tempo source (it falls back to the coarser beat-level grid
        # when this is empty).
        sixteenth_grid_ms = []
        current = self.phase * self.subdivision_ms
        while current < 30000:
            sixteenth_grid_ms.append(current)
            current += self.subdivision_ms

        return PulseField(
            tempo_bpm=self.to_tactus_bpm(),
            confidence=confidence or Confidence.from_float(self.confidence),
            beat_grid_ms=beat_grid_ms,
            sixteenth_grid_ms=sixteenth_grid_ms,
            pulse_strength=[self.confidence] * len(beat_grid_ms),
            phase_shift_ms=self.phase * self.subdivision_ms
        )

    def to_tempo_map(self) -> TempoMap:
        """Convert lattice to TempoMap."""
        return TempoMap(
            initial_tempo_bpm=self.to_tactus_bpm(),
            tempo_events=[],
            confidence=Confidence.from_float(self.confidence)
        )


# ========================================================================
# Event Extractor (from various sources)
# ========================================================================

class EventExtractor:
    """
    Extracts events from various sources for cryptanalysis.

    Pure function - no state. Consumes evidence, produces Events.
    """

    def __init__(self, config: 'ReverseGeoCryptConfig',
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def extract_from_onsets(self, onset_times: List[float],
                            strengths: List[float] = None) -> List[Event]:
        """Extract events from onset detection."""
        events = []
        strengths = strengths or [1.0] * len(onset_times)

        for time, strength in zip(onset_times, strengths):
            events.append(Event(
                time=time,
                strength=strength,
                event_type=EventType.ONSET,
                source=SourceType.RHYTHM
            ))

        return events

    def extract_from_notes(self, notes: List[NoteEvent]) -> List[Event]:
        """Extract events from note events (accents, phrase boundaries)."""
        events = []

        for note in notes:
            # Accent detection based on velocity
            if note.velocity > 100:
                events.append(Event(
                    time=note.start_ms / 1000,
                    strength=min(1.0, note.velocity / 127.0),
                    event_type=EventType.ACCENT,
                    source=note.source
                ))

            # Phrase boundaries (gaps between notes)
            # This would need context - simplified for now

        return events

    def extract_from_harmonic_analysis(self, harmonic_data: Dict[str, Any]) -> List[Event]:
        """Extract events from harmonic analysis (chord changes)."""
        events = []

        chord_changes = harmonic_data.get('chord_changes', [])
        for change_time, confidence in chord_changes:
            events.append(Event(
                time=change_time,
                strength=confidence,
                event_type=EventType.CHORD_CHANGE,
                source=SourceType.TONAL
            ))

        return events

    def extract_from_rms(self, rms_envelope: np.ndarray,
                         times_ms: np.ndarray, threshold: float = 0.3) -> List[Event]:
        """Extract events from RMS envelope (accents)."""
        events = []

        # Find peaks in RMS
        from scipy.signal import find_peaks
        peaks, props = find_peaks(rms_envelope, height=threshold)

        for peak in peaks:
            events.append(Event(
                time=times_ms[peak] / 1000,
                strength=rms_envelope[peak],
                event_type=EventType.ACCENT,
                source=SourceType.PITCH
            ))

        return events


# ========================================================================
# Ratio Cluster Analyzer
# ========================================================================

class RatioClusterAnalyzer:
    """
    Analyzes inter-onset interval ratios to find rhythmic DNA.

    Based on 4.7's ratio clustering but enhanced for 5.0.
    """

    def __init__(self, config: 'ReverseGeoCryptConfig',
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

        # Ideal ratios for different rhythm families
        self.ideal_ratios: Dict[RatioFamily, List[float]] = {
            RatioFamily.BINARY: [1.0, 2.0, 0.5, 4.0, 0.25],
            RatioFamily.TERNARY: [1.3333, 0.6667, 0.3333, 2.6667],
            RatioFamily.SWING: [1.5, 0.6667, 3.0],
            RatioFamily.QUINTUPLET: [1.25, 0.8, 2.5],
            RatioFamily.SEPTUPLET: [1.1429, 0.875, 2.2857]
        }

    def find_ratio_clusters(self, iois: np.ndarray) -> Dict[RatioFamily, float]:
        """
        Find recurring proportions in the rhythm.

        Returns:
            Dict mapping RatioFamily to confidence (0-1)
        """
        if len(iois) < self.config.min_iois_for_analysis:
            return {family: 0.5 for family in RatioFamily}

        # Calculate ratios between adjacent and near-adjacent intervals
        ratios = self._compute_ratios(iois)

        if len(ratios) < self.config.min_ratios_for_clustering:
            return {family: 0.5 for family in RatioFamily}

        try:
            # KDE to find peaks in ratio distribution
            kde = gaussian_kde(ratios, bw_method=self.config.kde_bandwidth)
            x_range = np.linspace(0.4, 2.6, 200)
            densities = kde(x_range)

            # Find peaks
            peaks_idx = np.where((densities[1:-1] > densities[:-2]) &
                                 (densities[1:-1] > densities[2:]))[0]
            peak_ratios = x_range[peaks_idx + 1]
            peak_heights = densities[peaks_idx + 1]

            # Score each ratio family
            scores = {family: 0.0 for family in RatioFamily}

            for ratio, height in zip(peak_ratios, peak_heights):
                for family, ideals in self.ideal_ratios.items():
                    for ideal in ideals:
                        if abs(ratio - ideal) < self.config.ratio_match_tolerance:
                            scores[family] += height
                            break

            # Normalize
            total = sum(scores.values())
            if total > 0:
                for family in scores:
                    scores[family] = min(1.0, scores[family] / total)

            return scores

        except Exception as e:
            if self.status_reporter:
                self.status_reporter.warn("RatioClusterAnalyzer", f"KDE failed: {e}")
            return {family: 0.5 for family in RatioFamily}

    def _compute_ratios(self, iois: np.ndarray) -> List[float]:
        """Compute ratios between intervals."""
        ratios = []

        for i in range(len(iois) - 1):
            if iois[i] > EPS:
                ratios.append(iois[i + 1] / iois[i])

            if i < len(iois) - 2 and iois[i] > EPS:
                ratios.append(iois[i + 2] / iois[i])

        # Filter extreme ratios
        ratios = [r for r in ratios if self.config.min_ratio < r < self.config.max_ratio]

        return ratios


# ========================================================================
# Lattice Search Engine (Core Algorithm)
# ========================================================================

class LatticeSearchEngine:
    """
    Searches for the geometric lattice that explains rhythmic events.

    This is the core algorithm. It tests candidate periods and scores
    how well they explain the rhythm.
    """

    def __init__(self, config: 'ReverseGeoCryptConfig',
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def search(self, iois: np.ndarray, onset_times: np.ndarray,
               ratio_scores: Dict[RatioFamily, float],
               anchors: List[Anchor]) -> Optional[GeometricLattice]:
        """
        Search for the subdivision S that minimizes structural entropy.

        Returns:
            GeometricLattice or None if insufficient data
        """
        if len(iois) < self.config.min_iois_for_search:
            return None

        best_lattice = None
        best_score = -np.inf

        # Generate candidate subdivisions from IOIs
        candidates = np.unique(np.round(iois, 4))
        candidates = candidates[candidates > self.config.min_period_sec]
        candidates = candidates[:self.config.max_candidates]

        periods_tested = 0

        for S in candidates:
            # Test different metric levels
            for multiplier in self.config.multipliers_to_test:
                period = S * multiplier
                bpm = 60.0 / period

                if not (self.config.min_bpm <= bpm <= self.config.max_bpm):
                    continue

                periods_tested += 1

                # Calculate alignment error
                error = self._alignment_error(iois, period)

                # Find phase
                phase = self._find_phase(onset_times, period)

                # Anchor alignment
                anchor_alignment = self._anchor_alignment(anchors, period, phase)

                # Entropy (rubato detection)
                entropy = self._calculate_entropy(iois, period)

                # Determine best ratio family
                ratio_family = max(ratio_scores, key=ratio_scores.get)

                # Calculate confidence
                confidence = self._calculate_confidence(error, anchor_alignment, entropy, ratio_scores)

                # Determine confidence level
                confidence_level = self._get_confidence_level(confidence, anchor_alignment, len(anchors))

                # Combined score for ranking
                score = self._calculate_score(error, anchor_alignment, entropy, ratio_scores)

                if score > best_score:
                    best_score = score
                    best_lattice = GeometricLattice(
                        subdivision_sec=period,
                        subdivision_ms=period * 1000,
                        ratio_family=ratio_family,
                        error_score=error,
                        entropy=entropy,
                        confidence=confidence,
                        confidence_level=confidence_level,
                        derived_bpm=bpm,
                        phase=phase,
                        anchor_alignment=anchor_alignment,
                        candidate_periods_tested=periods_tested,
                        events_analyzed=len(onset_times),
                        anchors_used=len(anchors)
                    )

        return best_lattice

    def _alignment_error(self, iois: np.ndarray, period: float) -> float:
        """
        How much do intervals deviate from integer multiples of period?

        Returns:
            0 = perfect alignment, 1 = random
        """
        if len(iois) == 0 or period <= 0:
            return 1.0

        # Calculate how close each IOI is to an integer multiple
        multiples = iois / period
        nearest_int = np.round(multiples)
        error = np.mean(np.abs(multiples - nearest_int))

        return min(1.0, error)

    def _find_phase(self, times: np.ndarray, period: float) -> float:
        """Find the phase offset that best aligns events."""
        if len(times) < 2:
            return 0.0

        # Calculate remainder for each event
        remainders = times % period

        # Use histogram to find most common remainder
        hist, bins = np.histogram(remainders, bins=self.config.phase_bins)
        peak_bin = np.argmax(hist)
        phase = (bins[peak_bin] + bins[peak_bin + 1]) / 2

        return phase / period

    def _anchor_alignment(self, anchors: List[Anchor], period: float, phase: float) -> float:
        """How well do structural downbeats align with integer multiples?"""
        if not anchors:
            return 0.5

        phase_offset = phase * period
        matches = 0
        total_weight = 0

        for anchor in anchors[:self.config.max_anchors_to_check]:
            # Check if anchor lands near an integer multiple
            adjusted_time = anchor.time - phase_offset
            nearest_int = round(adjusted_time / period)
            grid_time = phase_offset + nearest_int * period
            distance = abs(anchor.time - grid_time)
            tolerance = period * self.config.anchor_tolerance_ratio

            if distance < tolerance:
                weight = anchor.confidence * (1.0 - (distance / tolerance))
                matches += weight

            total_weight += anchor.confidence

        if total_weight == 0:
            return 0.5

        return min(1.0, matches / total_weight)

    def _calculate_entropy(self, iois: np.ndarray, period: float) -> float:
        """Calculate rhythmic entropy (rubato / unpredictability)."""
        if len(iois) == 0 or period <= 0:
            return 0.5

        # Normalize IOIs to period
        normalized = (iois % period) / period

        # Entropy as standard deviation
        entropy = np.std(normalized) if len(normalized) > 0 else 0.5

        return min(1.0, entropy * 2)  # Scale to 0-1

    def _calculate_confidence(self, error: float, anchor_alignment: float,
                              entropy: float, ratio_scores: Dict[RatioFamily, float]) -> float:
        """Calculate overall confidence in the lattice."""
        # Higher anchor alignment = more confidence
        # Lower error = more confidence
        # Lower entropy = more confidence (less rubato)
        # Higher ratio score = more confidence

        best_ratio_score = max(ratio_scores.values())

        confidence = (
                (1.0 - error) * 0.3 +
                anchor_alignment * 0.3 +
                (1.0 - entropy) * 0.2 +
                best_ratio_score * 0.2
        )

        return min(1.0, max(0.0, confidence))

    def _calculate_score(self, error: float, anchor_alignment: float,
                         entropy: float, ratio_scores: Dict[RatioFamily, float]) -> float:
        """Calculate ranking score for candidate lattice."""
        best_ratio_score = max(ratio_scores.values())

        return (1.0 - error) * 0.4 + anchor_alignment * 0.4 + (1.0 - entropy) * 0.1 + best_ratio_score * 0.1

    def _get_confidence_level(self, confidence: float, anchor_alignment: float,
                              anchor_count: int) -> LatticeConfidence:
        """Determine confidence level based on metrics."""
        if anchor_count >= 3 and anchor_alignment > 0.8 and confidence > 0.8:
            return LatticeConfidence.LOCKED
        elif confidence > 0.6:
            return LatticeConfidence.PROBABLE
        elif confidence > 0.3:
            return LatticeConfidence.AMBIGUOUS
        else:
            return LatticeConfidence.SPARSE


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class ReverseGeoCryptConfig:
    """Configuration for ReverseGeoCrypt engine."""

    # BPM range
    min_bpm: float = MIN_TEMPO_BPM  # 40
    max_bpm: float = MAX_TEMPO_BPM  # 240

    # IOI filtering
    min_ioi_sec: float = 0.05  # 50ms
    max_ioi_sec: float = 2.0  # 2 seconds

    # Ratio analysis
    min_iois_for_analysis: int = 4
    min_ratios_for_clustering: int = 5
    ratio_match_tolerance: float = 0.1
    min_ratio: float = 0.4
    max_ratio: float = 2.6
    kde_bandwidth: float = 0.15

    # Lattice search
    min_iois_for_search: int = 3
    min_period_sec: float = 0.05
    max_candidates: int = 20
    multipliers_to_test: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    phase_bins: int = 20

    # Anchors
    max_anchors_to_check: int = 10
    anchor_tolerance_ratio: float = 0.1

    # Performance
    cleanup_after_analysis: bool = True


# ========================================================================
# Main ReverseGeoCrypt Agent
# ========================================================================

class ReverseGeoCrypt:
    """
    Rhythmic Cryptanalysis Engine for Grimlock 5.0.

    LAW 2 COMPLIANT: Does NOT import other agents.
    Consumes Events and Anchors. Produces GeometricLattice.

    Philosophy:
        Don't ask "what is the tempo?"
        Ask "what geometric lattice explains these events?"

    Usage (5.0):
        crypt = ReverseGeoCrypt()
        lattice = crypt.decrypt(events, anchors=downbeats)

        macro_bpm = lattice.to_tactus_bpm()
        confidence = lattice.confidence
    """

    def __init__(
            self,
            config: Optional[ReverseGeoCryptConfig] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Args:
            config: Configuration for cryptanalysis
            status_reporter: StatusReporter for progress
            music_box: MusicBox for forensic logging
            progress_callback: Optional progress callback
        """
        self._name = "reverse_geo_crypt"
        self._source_type = SourceType.TEMPO_INTELLIGENCE
        self._config = config or ReverseGeoCryptConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback

        # Initialize components
        self._event_extractor = EventExtractor(self._config, status_reporter)
        self._ratio_analyzer = RatioClusterAnalyzer(self._config, status_reporter)
        self._lattice_searcher = LatticeSearchEngine(self._config, status_reporter)

        # State
        self._last_lattice: Optional[GeometricLattice] = None
        self._last_execution_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0

        self._log_status("ReverseGeoCrypt initialized")

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
        """
        Process audio and return StageResult with tempo information.

        Note: ReverseGeoCrypt primarily works with events, not raw audio.
        This method extracts basic events from audio for convenience.
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting rhythmic cryptanalysis")

        try:
            # Extract basic events from audio
            events = self._extract_basic_events(audio_buffer, context)
            anchors = self._extract_basic_anchors(audio_buffer, context)

            # Decrypt the rhythm
            lattice = self.decrypt(events, anchors)
            self._last_lattice = lattice

            # Convert to PulseField and TempoMap
            pulse_field = lattice.to_pulse_field() if lattice else None
            tempo_map = lattice.to_tempo_map() if lattice else None

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory
            self._last_execution_time_ms = execution_time_ms

            # Log to MusicBox
            if self._music_box and lattice:
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="rhythmic_cryptanalysis",
                    before_state={"events": len(events), "anchors": len(anchors)},
                    after_state={
                        "subdivision_ms": lattice.subdivision_ms,
                        "derived_bpm": lattice.derived_bpm,
                        "tactus_bpm": lattice.to_tactus_bpm(),
                        "ratio_family": lattice.ratio_family.value,
                        "confidence": lattice.confidence,
                        "confidence_level": lattice.confidence_level.value
                    },
                    reasoning=f"Found {lattice.ratio_family.value} lattice at {lattice.to_tactus_bpm():.1f} BPM",
                    reversible=True
                )
            elif self._music_box:
                # A null result is itself worth recording - previously
                # this whole stage went silent whenever no lattice was
                # found, indistinguishable from the stage never running.
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type=DecisionType.ANALYSIS_EVIDENCE,
                    before_state={"events": len(events), "anchors": len(anchors)},
                    after_state={"lattice_found": False},
                    reasoning="No geometric lattice found from the extracted events/anchors",
                    reversible=True
                )

            self._update_progress(1.0,
                                  f"Complete: {lattice.to_tactus_bpm():.1f} BPM" if lattice else "No lattice found")

            return StageResult(
                stage_name=self._name,
                success=lattice is not None,
                events=[],  # No NoteEvents from this agent
                metadata={
                    "lattice_found": lattice is not None,
                    "subdivision_ms": lattice.subdivision_ms if lattice else 0,
                    "derived_bpm": lattice.derived_bpm if lattice else 0,
                    "tactus_bpm": lattice.to_tactus_bpm() if lattice else 0,
                    "ratio_family": lattice.ratio_family.value if lattice else "none",
                    "confidence": lattice.confidence if lattice else 0,
                    "confidence_level": lattice.confidence_level.value if lattice else "none",
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Cryptanalysis failed: {e}", "error")
            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def decrypt(self, events: List[Event], anchors: List[Anchor] = None) -> Optional[GeometricLattice]:
        """
        Decrypt the rhythmic code from events.

        Args:
            events: List of musical events (onsets, chords, accents)
            anchors: Optional structural downbeats (locks the phase)

        Returns:
            GeometricLattice or None if insufficient data
        """
        self._update_progress(0.1, "Validating events")

        if len(events) < self._config.min_iois_for_analysis:
            self._log_status(f"Insufficient events: {len(events)}", "warn")
            return None

        # Extract onset times (prioritize onsets, fallback to all events)
        self._update_progress(0.2, "Extracting event times")
        onset_times = self._extract_times(events)

        if len(onset_times) < self._config.min_iois_for_analysis:
            self._log_status(f"Insufficient onset times: {len(onset_times)}", "warn")
            return None

        # Compute IOIs
        self._update_progress(0.3, "Computing inter-onset intervals")
        iois = self._compute_iois(onset_times)

        if len(iois) < self._config.min_iois_for_search:
            self._log_status(f"Insufficient IOIs: {len(iois)}", "warn")
            return None

        # Find ratio clusters
        self._update_progress(0.5, "Analyzing ratio clusters")
        ratio_scores = self._ratio_analyzer.find_ratio_clusters(iois)

        # Search for lattice
        self._update_progress(0.7, "Searching for geometric lattice")
        anchors = anchors or []
        lattice = self._lattice_searcher.search(iois, onset_times, ratio_scores, anchors)

        # Force cleanup
        if self._config.cleanup_after_analysis:
            self._force_cleanup()

        return lattice

    def _extract_times(self, events: List[Event]) -> np.ndarray:
        """Extract timestamps, preferring onsets but using all events if needed."""
        onset_times = [e.time for e in events if e.event_type == EventType.ONSET]

        if len(onset_times) < self._config.min_iois_for_analysis:
            # Fall back to all events
            onset_times = [e.time for e in events]

        return np.array(sorted(onset_times))

    def _compute_iois(self, times: np.ndarray) -> np.ndarray:
        """Compute inter-onset intervals, filtering extremes."""
        iois = np.diff(times)
        # Filter out implausible intervals
        iois = iois[(iois > self._config.min_ioi_sec) & (iois < self._config.max_ioi_sec)]
        return iois

    def _extract_basic_events(self, audio: np.ndarray, context: AudioContext) -> List[Event]:
        """Extract basic events from audio for standalone operation."""
        events = []

        if not LIBROSA_AVAILABLE:
            return events

        try:
            import librosa

            # Onset detection
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=context.working_sample_rate,
                hop_length=512,
                backtrack=True
            )

            hop_ms = 512 / context.working_sample_rate * 1000
            for frame in onset_frames:
                events.append(Event(
                    time=frame * hop_ms / 1000,
                    strength=0.7,
                    event_type=EventType.ONSET,
                    source=SourceType.RHYTHM
                ))

            # RMS accent detection
            hop_length = 512
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            times_ms = np.arange(len(rms)) * hop_length / context.working_sample_rate * 1000

            rms_events = self._event_extractor.extract_from_rms(rms, times_ms)
            events.extend(rms_events)

        except Exception as e:
            self._log_status(f"Basic event extraction failed: {e}", "warn")

        return events

    def _extract_basic_anchors(self, audio: np.ndarray, context: AudioContext) -> List[Anchor]:
        """Extract basic anchors from audio for standalone operation."""
        anchors = []

        if not LIBROSA_AVAILABLE:
            return anchors

        try:
            import librosa

            # Beat tracking for anchor candidates
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio,
                sr=context.working_sample_rate,
                hop_length=512
            )

            hop_ms = 512 / context.working_sample_rate * 1000
            for i, frame in enumerate(beat_frames[:8]):  # First 8 beats
                anchors.append(Anchor(
                    time=frame * hop_ms / 1000,
                    confidence=0.6 if i == 0 else 0.5,
                    source=SourceType.TEMPO_INTELLIGENCE
                ))

        except Exception as e:
            self._log_status(f"Basic anchor extraction failed: {e}", "warn")

        return anchors

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
        """Update progress via callback and StatusReporter."""
        if self._progress_callback:
            self._progress_callback(progress, message)
        if self._status_reporter:
            self._status_reporter.progress(self._name, progress, message)

    def _log_status(self, message: str, level: str = "info"):
        """Log status message."""
        if self._status_reporter:
            getattr(self._status_reporter, level)(self._name, message)
        else:
            print(f"[{self._name}] {message}")

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
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
        """Run validation gate on this agent's output."""
        if self._last_lattice is None:
            return ValidationResult(
                is_valid=False,
                gate_used=gate,
                reason=VetoReason.EMPTY_RESULT,
                detail="No lattice found"
            )

        if gate == ValidationGate.CONFIDENCE_THRESHOLD:
            is_valid = self._last_lattice.confidence >= Confidence.LOW.value

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                detail=f"Lattice confidence: {self._last_lattice.confidence:.2f}",
                confidence_before=self._last_lattice.confidence,
                confidence_after=self._last_lattice.confidence if is_valid else 0.0
            )

        elif gate == ValidationGate.TEMPO_REASONABLENESS:
            tactus_bpm = self._last_lattice.to_tactus_bpm()
            is_valid = self._config.min_bpm <= tactus_bpm <= self._config.max_bpm

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.TEMPO_OUTLIER,
                detail=f"Tactus BPM: {tactus_bpm:.1f}"
            )

        else:
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                detail=f"Gate {gate.value} not fully supported"
            )

    def get_confidence(self) -> Confidence:
        """Return overall confidence of last output."""
        if self._last_lattice is None:
            return Confidence.HALLUCINATION
        return Confidence.from_float(self._last_lattice.confidence)

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        """Return (reason, detail) if currently vetoed, else None."""
        if self._last_lattice is None:
            return (VetoReason.EMPTY_RESULT, "No lattice found")

        if self._last_lattice.confidence < Confidence.LOW.value:
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Confidence {self._last_lattice.confidence:.2f}")

        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        """Run harmonic series check (N/A for rhythm analysis)."""
        return SchoenbergResult(
            verdict=SchoenbergVerdict.UNCERTAIN,
            zero_crossing_rate=0.0,
            spectral_flatness=0.5,
            reason="ReverseGeoCrypt analyzes rhythm, not harmonic series"
        )

    # ========================================================================
    # MemoryManagedProtocol Implementation
    # ========================================================================

    def release_buffer(self, buffer_name: str) -> None:
        """Explicit deletion request for audio buffers."""
        pass

    def get_memory_footprint_mb(self) -> float:
        """Current memory usage in MB."""
        return self._get_current_memory_mb()

    def can_release(self, buffer_name: str) -> bool:
        """Return True if this buffer is safe to delete."""
        return True

    def staggered_gc(self) -> Dict[str, Any]:
        """Run garbage collection and return stats."""
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
    # AnalysisAgentProtocol Properties
    # ========================================================================

    @property
    def analysis_threshold(self) -> Confidence:
        return Confidence.LOW

    def get_last_lattice(self) -> Optional[GeometricLattice]:
        """Get the last detected lattice."""
        return self._last_lattice

    def get_statistics(self) -> Dict[str, Any]:
        """Get cryptanalysis statistics."""
        if self._last_lattice:
            return {
                "name": self._name,
                "source_type": self._source_type.value,
                "lattice_found": True,
                "subdivision_ms": self._last_lattice.subdivision_ms,
                "tactus_bpm": self._last_lattice.to_tactus_bpm(),
                "ratio_family": self._last_lattice.ratio_family.value,
                "confidence": self._last_lattice.confidence,
                "confidence_level": self._last_lattice.confidence_level.value,
                "error_score": self._last_lattice.error_score,
                "entropy": self._last_lattice.entropy,
                "last_execution_time_ms": self._last_execution_time_ms,
                "total_memory_freed_mb": self._total_memory_freed_mb
            }
        else:
            return {
                "name": self._name,
                "source_type": self._source_type.value,
                "lattice_found": False,
                "last_execution_time_ms": self._last_execution_time_ms,
                "total_memory_freed_mb": self._total_memory_freed_mb
            }


# ========================================================================
# Convenience Functions
# ========================================================================

def create_reverse_geo_crypt(
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> ReverseGeoCrypt:
    """Create a configured ReverseGeoCrypt instance."""
    config = ReverseGeoCryptConfig()
    return ReverseGeoCrypt(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )


def quick_crypt_test(onset_times: List[float]) -> Dict[str, Any]:
    """
    Quick test function for ReverseGeoCrypt.

    Args:
        onset_times: List of onset times in seconds

    Returns:
        Dictionary with test results
    """
    print(f"Testing ReverseGeoCrypt on {len(onset_times)} onsets")

    # Create events from onsets
    events = [
        Event(time=t, strength=0.8, event_type=EventType.ONSET, source=SourceType.RHYTHM)
        for t in onset_times
    ]

    # Create mock anchors
    anchors = [
        Anchor(time=0.0, confidence=0.9, source=SourceType.TEMPO_INTELLIGENCE),
        Anchor(time=2.0, confidence=0.85, source=SourceType.TEMPO_INTELLIGENCE),
        Anchor(time=4.0, confidence=0.8, source=SourceType.TEMPO_INTELLIGENCE),
    ]

    crypt = create_reverse_geo_crypt()
    lattice = crypt.decrypt(events, anchors)

    if lattice:
        print(f"  Subdivision: {lattice.subdivision_ms:.1f}ms")
        print(f"  Derived BPM: {lattice.derived_bpm:.1f}")
        print(f"  Tactus BPM: {lattice.to_tactus_bpm():.1f}")
        print(f"  Ratio Family: {lattice.ratio_family.value}")
        print(f"  Confidence: {lattice.confidence:.1%} ({lattice.confidence_level.value})")
        print(f"  Error: {lattice.error_score:.3f}")
        print(f"  Entropy: {lattice.entropy:.3f}")

        return {
            "tactus_bpm": lattice.to_tactus_bpm(),
            "ratio_family": lattice.ratio_family.value,
            "confidence": lattice.confidence,
            "confidence_level": lattice.confidence_level.value
        }
    else:
        print("  No lattice found")
        return {"error": "No lattice found"}


# For standalone testing
if __name__ == "__main__":
    # Mock event times (simulating a 60 BPM pulse with 8th notes)
    mock_onsets = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    quick_crypt_test(mock_onsets)