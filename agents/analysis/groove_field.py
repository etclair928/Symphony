# =================================================================
# MODULE: agents/analysis/groove_field.py
# DESCRIPTION: Groove Field analyzer - Relational Physics for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# PHILOSOPHY:
#     "A note's value is defined by its distance to its neighbor,
#      not its distance to the grid."
#
#     This measures the Phase Delta between Bass and Kick to detect
#     intentional "wide swing" or "Dilla pocket" rather than timing errors.
#
#     If bass and kick are consistently 12ms apart, that is a physical
#     law of the performance that the engine must respect.
#
# KEY ARCHITECTURE:
#     - Relational Physics: Measures neighbor distances, not grid alignment
#     - Phase delta analysis between bass and kick events
#     - Swing type detection (wide swing, Dilla pocket, laid back, rushed)
#     - Groove confidence scoring with forensic breakdown
#     - Quantization recommendations based on groove strength
#     - StatusReporter integration throughout
#     - No agent imports - only consumes NoteEvents
#
# Authored by: DeepSeek - Complete 5.0 rewrite (2026-05-11)
# Based on 4.7 GrooveFieldAnalyzer and relational physics wisdom.
# =================================================================

import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from collections import deque
from scipy.stats import gaussian_kde

# Core imports - ONLY from bedrock
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    GrooveField, NoteWithContext, DecisionType
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    PHASE_DELTA_SWING_THRESHOLD_MS,
    DILLA_POCKET_VARIANCE_MS,
    MAX_GROOVE_SNAP_MS,
    MIN_PHASE_DELTA_TO_CONSIDER_MS,
    INTER_NOTE_DISTANCE_HISTORY_WINDOW,
    GROOVE_MAX_PAIRING_DISTANCE_MS,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    AnalysisAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)


# ========================================================================
# Enums and Types
# ========================================================================

class GrooveType(str, Enum):
    """Types of detected groove feels."""
    TIGHT_POCKET = "tight_pocket"  # Very consistent, <2ms variance
    WIDE_SWING = "wide_swing"  # Intentional large offset
    DILLA_POCKET = "dilla_pocket"  # J Dilla style push/pull
    LAID_BACK = "laid_back"  # Consistently behind the beat
    RUSHED = "rushed"  # Consistently ahead of the beat
    FLUID = "fluid"  # Rubato, expressive timing
    ERRATIC = "erratic"  # Inconsistent, likely errors


class QuantizationRecommendation(str, Enum):
    """Recommendation for quantization based on groove analysis."""
    PRESERVE_GROOVE = "preserve_groove"  # Don't quantize - intentional feel
    LIGHT_SNAP = "light_snap"  # Small corrections only
    NORMAL_QUANTIZE = "normal_quantize"  # Standard quantization
    HEAVY_QUANTIZE = "heavy_quantize"  # Aggressive correction


@dataclass(frozen=True)
class PhaseDeltaResult:
    """Immutable result of phase delta analysis."""
    deltas: Tuple[float, ...]
    average_ms: float
    std_ms: float
    consistency: float
    is_significant: bool
    sample_count: int


@dataclass(frozen=True)
class GrooveConfidenceComponents:
    """Forensic confidence breakdown for groove detection."""
    phase_consistency: float = 0.0
    temporal_regularity: float = 0.0
    sample_size_confidence: float = 0.0
    swing_definition: float = 0.0
    ensemble_agreement: float = 0.0

    def total(self) -> float:
        """Calculate total confidence score."""
        weights = {
            'phase_consistency': 0.35,
            'temporal_regularity': 0.25,
            'sample_size_confidence': 0.15,
            'swing_definition': 0.15,
            'ensemble_agreement': 0.10
        }

        return (
                self.phase_consistency * weights['phase_consistency'] +
                self.temporal_regularity * weights['temporal_regularity'] +
                self.sample_size_confidence * weights['sample_size_confidence'] +
                self.swing_definition * weights['swing_definition'] +
                self.ensemble_agreement * weights['ensemble_agreement']
        )


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class GrooveConfig:
    """Configuration for Groove Field analyzer."""

    # Phase detection
    swing_threshold_ms: float = PHASE_DELTA_SWING_THRESHOLD_MS  # 8.0ms
    dilla_variance_threshold: float = DILLA_POCKET_VARIANCE_MS  # 4.0ms
    max_snap_ms: float = MAX_GROOVE_SNAP_MS  # 100ms
    min_phase_delta_to_consider_ms: float = MIN_PHASE_DELTA_TO_CONSIDER_MS  # 2.0ms
    max_pairing_distance_ms: float = GROOVE_MAX_PAIRING_DISTANCE_MS  # 500.0ms
    history_window: int = INTER_NOTE_DISTANCE_HISTORY_WINDOW  # 8 notes

    # Alignment options
    align_to_kick: bool = True
    align_to_downbeat: bool = True

    # Confidence scoring
    min_notes_for_groove: int = 8
    confidence_boost_for_consistent_swing: float = 0.2
    confidence_penalty_for_high_variance: float = 0.3

    # Multi-resolution analysis
    use_multi_resolution: bool = True
    resolutions_ms: List[int] = field(default_factory=lambda: [2, 5, 10])

    # Performance
    max_events_to_analyze: int = 10000
    cleanup_after_analysis: bool = True


# ========================================================================
# Phase Delta Analyzer (Core Relational Physics)
# ========================================================================

class PhaseDeltaAnalyzer:
    """
    Analyzes phase delta between bass and kick events.

    Core of Relational Physics: Measures the consistent offset between
    bass and kick to detect intentional feel.

    Pure function - no state beyond configuration.
    """

    def __init__(self, config: GrooveConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def compute_phase_deltas(
            self,
            bass_events: List[NoteEvent],
            kick_events: List[NoteEvent]
    ) -> PhaseDeltaResult:
        """
        Compute phase deltas between bass and kick events.

        For each bass note, find the nearest kick and compute offset.
        Positive = bass after kick, Negative = bass before kick.

        Returns immutable PhaseDeltaResult.
        """
        if not bass_events or not kick_events:
            return PhaseDeltaResult(
                deltas=(),
                average_ms=0.0,
                std_ms=0.0,
                consistency=0.0,
                is_significant=False,
                sample_count=0
            )

        deltas = []

        for bass in bass_events:
            # Find nearest kick
            nearest_kick = min(
                kick_events,
                key=lambda k: abs(k.get_active_start_ms() - bass.get_active_start_ms())
            )

            delta = bass.get_active_start_ms() - nearest_kick.get_active_start_ms()

            # A bass note whose "nearest" kick is still very far away (e.g.
            # a bass note in a kick-less passage) isn't a real rhythmic
            # pair - including it would let one outlier drag the whole
            # groove estimate to nonsense (seen in practice: one such pair
            # produced a "phase delta" of several seconds).
            if abs(delta) > self.config.max_pairing_distance_ms:
                continue

            deltas.append(delta)

        if not deltas:
            return PhaseDeltaResult(
                deltas=(),
                average_ms=0.0,
                std_ms=0.0,
                consistency=0.0,
                is_significant=False,
                sample_count=0
            )

        deltas_array = np.array(deltas)
        avg_delta = float(np.mean(deltas_array))
        std_delta = float(np.std(deltas_array))

        # Calculate consistency (inverse of normalized variance)
        variance = np.var(deltas_array) if len(deltas) > 1 else 0.0
        consistency = 1.0 / (1.0 + variance / 100) if variance > 0 else 1.0

        # Check if delta is significant enough to consider
        is_significant = abs(avg_delta) > self.config.min_phase_delta_to_consider_ms

        return PhaseDeltaResult(
            deltas=tuple(deltas),
            average_ms=avg_delta,
            std_ms=std_delta,
            consistency=consistency,
            is_significant=is_significant,
            sample_count=len(deltas)
        )

    def detect_swing_type(self, result: PhaseDeltaResult) -> Tuple[GrooveType, bool, bool]:
        """
        Detect type of swing/groove from phase delta result.

        Returns:
            (groove_type, is_wide_swing, is_dilla_pocket)
        """
        if result.sample_count < 2:
            return GrooveType.ERRATIC, False, False

        avg_delta = result.average_ms
        std_delta = result.std_ms

        # Wide swing: consistent offset > threshold
        is_wide_swing = abs(avg_delta) > self.config.swing_threshold_ms and result.consistency > 0.6

        # Dilla pocket: moderate variance but intentional micro-timing
        # Named after J Dilla's characteristic push/pull
        is_dilla_pocket = (
                self.config.dilla_variance_threshold * 0.5 <= std_delta <= self.config.dilla_variance_threshold * 1.5 and
                result.consistency > 0.5 and
                abs(avg_delta) < self.config.swing_threshold_ms
        )

        # Determine groove type
        if is_dilla_pocket:
            groove_type = GrooveType.DILLA_POCKET
        elif is_wide_swing:
            groove_type = GrooveType.WIDE_SWING
        elif abs(avg_delta) < 2 and result.consistency > 0.8:
            groove_type = GrooveType.TIGHT_POCKET
        elif avg_delta > 0:
            groove_type = GrooveType.LAID_BACK
        elif avg_delta < 0:
            groove_type = GrooveType.RUSHED
        elif result.consistency < 0.3:
            groove_type = GrooveType.ERRATIC
        else:
            groove_type = GrooveType.FLUID

        return groove_type, is_wide_swing, is_dilla_pocket


# ========================================================================
# Onset Swing Analyzer (kick-independent)
# ========================================================================

class OnsetSwingAnalyzer:
    """
    Detects swing/shuffle feel from general note-onset timing against a
    real beat grid - unlike PhaseDeltaAnalyzer's bass-vs-kick relational
    model, this needs no specific instrument to be present. Confirmed
    necessary on real audio: a real ~60s section of Hopeful.mp3 produced
    544 real drum hits (verified via DrumIntelligence) with a full
    articulation spread (hi-hat, ride, rimshot, tom, crash, snare) but
    zero classified as kick - PhaseDeltaAnalyzer's bass-vs-kick model
    structurally cannot produce a verdict for a song like that, no matter
    how good the underlying kick/bass data is.

    Works by finding onsets that land near the halfway point between two
    consecutive real beats (candidate swung eighth notes) and measuring
    how far past the exact midpoint they actually sit on average - swing
    is exactly this: a "long-short" eighth pair instead of two even
    eighths. A straight, unswung performance averages close to the exact
    midpoint; real swing pushes it later.
    """

    def __init__(self, config: GrooveConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def detect_swing_ratio(
            self, beat_grid_ms: List[float], onset_times_ms: List[float]
    ) -> Tuple[float, float, int]:
        """
        Returns (swing_ratio, confidence, sample_count).

        swing_ratio is in TemporalLattice's own [0.5, 0.85] convention
        (0.5 = straight, higher = more swung) so it can be used directly
        without another layer of bucketing/translation.
        """
        if len(beat_grid_ms) < 4 or len(onset_times_ms) < 8:
            return 0.5, 0.0, 0

        beat_grid = np.asarray(sorted(beat_grid_ms), dtype=float)
        onsets = np.asarray(sorted(onset_times_ms), dtype=float)

        offbeat_positions = []
        for onset in onsets:
            idx = int(np.searchsorted(beat_grid, onset)) - 1
            if idx < 0 or idx >= len(beat_grid) - 1:
                continue
            beat_start = beat_grid[idx]
            beat_dur = beat_grid[idx + 1] - beat_start
            if beat_dur <= 0:
                continue
            norm_pos = (onset - beat_start) / beat_dur
            # Only onsets plausibly landing near the halfway point are
            # candidate swung eighth notes - onsets near 0.0/1.0 are on
            # the beat itself and say nothing about swing.
            if 0.3 <= norm_pos <= 0.8:
                offbeat_positions.append(norm_pos)

        if len(offbeat_positions) < 5:
            return 0.5, 0.0, len(offbeat_positions)

        mean_pos = float(np.mean(offbeat_positions))
        swing_ratio = max(0.5, min(0.85, mean_pos))

        # Confidence: real swing is a consistent, repeated pattern, not
        # scattered noise - tightly clustered off-beat positions (low
        # std) plus a reasonable sample size both raise confidence.
        std_pos = float(np.std(offbeat_positions))
        consistency = max(0.0, 1.0 - std_pos / 0.15)
        sample_confidence = min(1.0, len(offbeat_positions) / 20.0)
        confidence = consistency * 0.7 + sample_confidence * 0.3

        return swing_ratio, max(0.0, min(1.0, confidence)), len(offbeat_positions)


# ========================================================================
# Inter-Note Distance Analyzer
# ========================================================================

class InterNoteDistanceAnalyzer:
    """
    Analyzes distances between consecutive notes.

    Relational Physics: The space between notes defines the feel,
    not the absolute position on a grid.
    """

    def __init__(self, config: GrooveConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def compute_distances(self, events: List[NoteEvent]) -> List[float]:
        """Compute distances between consecutive notes."""
        if len(events) < 2:
            return []

        sorted_events = sorted(events, key=lambda e: e.get_active_start_ms())
        distances = []

        for i in range(len(sorted_events) - 1):
            distance = sorted_events[i + 1].get_active_start_ms() - sorted_events[i].get_active_start_ms()
            if distance > 0:  # Only positive distances
                distances.append(distance)

        return distances

    def compute_statistics(self, distances: List[float]) -> Dict[str, float]:
        """Compute statistics on inter-note distances."""
        if not distances:
            return {
                "average_ms": 0.0,
                "std_ms": 0.0,
                "regularity": 0.0,
                "median_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0
            }

        distances_array = np.array(distances)
        variance = np.var(distances_array) if len(distances) > 1 else 0.0
        regularity = 1.0 / (1.0 + variance / 100) if variance > 0 else 1.0

        return {
            "average_ms": float(np.mean(distances_array)),
            "std_ms": float(np.std(distances_array)),
            "regularity": regularity,
            "median_ms": float(np.median(distances_array)),
            "min_ms": float(np.min(distances_array)),
            "max_ms": float(np.max(distances_array))
        }

    def estimate_tempo(self, distances: List[float]) -> float:
        """
        Estimate tempo from inter-note distances.

        Average distance between beats = 60000 / BPM
        """
        if not distances:
            return 120.0

        # Use median for robustness against outliers
        avg_distance_ms = np.median(distances)
        if avg_distance_ms <= 0:
            return 120.0

        tempo_bpm = 60000 / avg_distance_ms

        # Clamp to reasonable range
        return max(40.0, min(240.0, tempo_bpm))


# ========================================================================
# Groove Confidence Scorer (Forensic)
# ========================================================================

class GrooveConfidenceScorer:
    """
    Scores confidence of detected groove with forensic breakdown.

    High confidence = intentional feel that should be preserved.
    Low confidence = timing errors that should be corrected.
    """

    def __init__(self, config: GrooveConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def compute_confidence(
            self,
            phase_result: PhaseDeltaResult,
            distance_stats: Dict[str, float],
            note_count: int,
            groove_type: GrooveType
    ) -> Tuple[Confidence, GrooveConfidenceComponents]:
        """
        Compute overall confidence in groove detection with forensic breakdown.

        Returns:
            Tuple of (Confidence enum, ConfidenceComponents)
        """
        # Sample size confidence
        if note_count >= self.config.min_notes_for_groove * 2:
            sample_size_conf = 1.0
        elif note_count >= self.config.min_notes_for_groove:
            sample_size_conf = 0.7
        elif note_count >= 4:
            sample_size_conf = 0.4
        else:
            sample_size_conf = 0.2

        # Phase consistency (from phase analysis)
        phase_consistency = phase_result.consistency

        # Temporal regularity (from distance analysis)
        temporal_regularity = distance_stats.get("regularity", 0.0)

        # Swing definition - how clearly does this match a known groove type?
        if groove_type in [GrooveType.TIGHT_POCKET, GrooveType.WIDE_SWING, GrooveType.DILLA_POCKET]:
            swing_definition = 0.9
        elif groove_type in [GrooveType.LAID_BACK, GrooveType.RUSHED]:
            swing_definition = 0.7
        elif groove_type == GrooveType.FLUID:
            swing_definition = 0.5
        else:
            swing_definition = 0.3

        # Ensemble agreement (placeholder - would come from multiple witnesses)
        ensemble_agreement = 0.7  # Default moderate agreement

        components = GrooveConfidenceComponents(
            phase_consistency=phase_consistency,
            temporal_regularity=temporal_regularity,
            sample_size_confidence=sample_size_conf,
            swing_definition=swing_definition,
            ensemble_agreement=ensemble_agreement
        )

        total_score = components.total()

        # Boost for consistent swing patterns
        if groove_type in [GrooveType.WIDE_SWING, GrooveType.DILLA_POCKET] and phase_consistency > 0.7:
            total_score = min(1.0, total_score + self.config.confidence_boost_for_consistent_swing)

        # Penalty for erratic timing
        if groove_type == GrooveType.ERRATIC:
            total_score = max(0.0, total_score - self.config.confidence_penalty_for_high_variance)

        # Map to Confidence enum
        if total_score >= 0.85:
            confidence = Confidence.PERFECT
        elif total_score >= 0.7:
            confidence = Confidence.HIGH
        elif total_score >= 0.5:
            confidence = Confidence.MEDIUM
        elif total_score >= 0.25:
            confidence = Confidence.LOW
        else:
            confidence = Confidence.HALLUCINATION

        return confidence, components


# ========================================================================
# Main Groove Field Analyzer Agent
# ========================================================================

class GrooveFieldAnalyzer:
    """
    Groove Field analyzer - Relational Physics implementation for Grimlock 5.0.

    Measures the phase delta between bass and kick to detect intentional
    feel. Recognizes "wide swing" and "Dilla pocket" as high-confidence
    intentional acts rather than timing errors.

    Law of Relational Physics: A note's value is defined by its distance
    to its neighbor, not its distance to the grid.

    Law 2 Compliance: Does NOT import other agents.
    """

    def __init__(
            self,
            config: Optional[GrooveConfig] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Args:
            config: Groove analyzer configuration
            status_reporter: StatusReporter for progress
            music_box: MusicBox for forensic logging
            progress_callback: Optional progress callback
        """
        self._name = "groove_field_analyzer"
        self._source_type = SourceType.GROOVE_FIELD
        self._config = config or GrooveConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback

        # Initialize components
        self._phase_analyzer = PhaseDeltaAnalyzer(self._config, status_reporter)
        self._distance_analyzer = InterNoteDistanceAnalyzer(self._config, status_reporter)
        self._confidence_scorer = GrooveConfidenceScorer(self._config, status_reporter)

        # State
        self._last_groove_field: Optional[GrooveField] = None
        self._last_phase_result: Optional[PhaseDeltaResult] = None
        self._last_confidence_components: Optional[GrooveConfidenceComponents] = None
        self._last_execution_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0
        self._phase_history: deque = deque(maxlen=self._config.history_window)

        self._log_status("GrooveFieldAnalyzer initialized")

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
        Process audio and return StageResult with groove analysis.

        Note: This analyzer works with NoteEvents, not raw audio.
        Expected usage: bass and kick events come from previous pipeline stages.
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting groove analysis")

        try:
            # Get events from context
            bass_events = self._get_events_from_context(context, 'bass')
            kick_events = self._get_events_from_context(context, 'kick')

            if len(bass_events) < self._config.min_notes_for_groove // 2:
                self._log_status(f"Insufficient bass events: {len(bass_events)}", "warn")

            # Analyze groove
            groove_field = self.analyze(bass_events, kick_events, context)

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory
            self._last_execution_time_ms = execution_time_ms

            # Log to MusicBox
            if self._music_box:
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="groove_analysis",
                    before_state={
                        "bass_count": len(bass_events),
                        "kick_count": len(kick_events)
                    },
                    after_state={
                        "phase_delta_ms": groove_field.bass_kick_phase_delta_ms,
                        "groove_signature": groove_field.groove_signature,
                        "confidence": groove_field.confidence.name,
                        "is_wide_swing": groove_field.is_wide_swing,
                        "is_dilla_pocket": groove_field.is_dilla_pocket
                    },
                    reasoning=f"Groove: {groove_field.groove_signature}, delta={groove_field.bass_kick_phase_delta_ms:.1f}ms",
                    reversible=True
                )

            # Force cleanup
            if self._config.cleanup_after_analysis:
                self._force_cleanup()

            self._update_progress(1.0, f"Complete: {groove_field.groove_signature}")

            return StageResult(
                stage_name=self._name,
                success=True,
                events=[],  # No NoteEvents from this agent
                metadata={
                    "phase_delta_ms": groove_field.bass_kick_phase_delta_ms,
                    "is_wide_swing": groove_field.is_wide_swing,
                    "is_dilla_pocket": groove_field.is_dilla_pocket,
                    "groove_signature": groove_field.groove_signature,
                    "confidence": groove_field.confidence.name,
                    "avg_inter_note_distance_ms": groove_field.average_inter_note_distance_ms,
                    "tempo_estimate_bpm": groove_field.tempo_estimate_bpm,
                    "phase_delta_variance_ms": groove_field.phase_delta_variance_ms,
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Groove analysis failed: {e}", "error")
            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def analyze(
            self,
            bass_events: List[NoteEvent],
            kick_events: List[NoteEvent],
            context: AudioContext
    ) -> GrooveField:
        """
        Analyze groove from bass and kick events.

        This is the primary interface for GrooveFieldAnalyzerProtocol.

        Process:
        1. Compute phase deltas between bass and kick
        2. Detect swing type (wide swing, Dilla pocket, etc.)
        3. Compute inter-note distances and tempo estimate
        4. Compute confidence with forensic breakdown
        5. Build GrooveField result

        Args:
            bass_events: Detected bass note events
            kick_events: Detected kick drum events
            context: Audio context (for tempo reference)

        Returns:
            GrooveField containing relational physics data
        """
        self._update_progress(0.1, "Validating events")

        # Limit events for performance
        if len(bass_events) > self._config.max_events_to_analyze:
            bass_events = bass_events[:self._config.max_events_to_analyze]
        if len(kick_events) > self._config.max_events_to_analyze:
            kick_events = kick_events[:self._config.max_events_to_analyze]

        # Step 1: Compute phase deltas between bass and kick
        self._update_progress(0.2, "Computing phase deltas")
        phase_result = self._phase_analyzer.compute_phase_deltas(bass_events, kick_events)
        self._last_phase_result = phase_result

        # Update history
        for delta in phase_result.deltas:
            self._phase_history.append(delta)

        # Step 2: Detect swing type
        self._update_progress(0.4, "Detecting swing type")
        groove_type, is_wide_swing, is_dilla_pocket = self._phase_analyzer.detect_swing_type(phase_result)

        # Step 3: Compute inter-note distances
        self._update_progress(0.6, "Computing inter-note distances")
        all_notes = bass_events + kick_events
        distances = self._distance_analyzer.compute_distances(all_notes)
        distance_stats = self._distance_analyzer.compute_statistics(distances)
        tempo_estimate = self._distance_analyzer.estimate_tempo(distances)

        avg_distance = distance_stats.get("average_ms", 0.0)

        # Step 4: Compute confidence with forensic breakdown
        self._update_progress(0.8, "Computing confidence")
        total_note_count = len(bass_events) + len(kick_events)
        confidence, components = self._confidence_scorer.compute_confidence(
            phase_result, distance_stats, total_note_count, groove_type
        )
        self._last_confidence_components = components

        # Step 5: Determine groove signature
        groove_signature = self._determine_groove_signature(groove_type, phase_result.average_ms)

        # Step 6: Create GrooveField
        groove_field = GrooveField(
            bass_kick_phase_delta_ms=phase_result.average_ms,
            is_wide_swing=is_wide_swing,
            is_dilla_pocket=is_dilla_pocket,
            average_inter_note_distance_ms=avg_distance,
            tempo_estimate_bpm=tempo_estimate,
            confidence=confidence,
            groove_signature=groove_signature,
            phase_delta_variance_ms=phase_result.std_ms ** 2 if phase_result.std_ms else 0.0
        )

        self._last_groove_field = groove_field

        self._log_status(f"Groove: {groove_signature}, delta={phase_result.average_ms:.1f}ms, {confidence.name}")

        if self._music_box is not None:
            self._music_box.log_decision(
                stage_name="groove_field",
                decision_type=DecisionType.ANALYSIS_EVIDENCE,
                before_state={"bass_events": len(bass_events), "kick_events": len(kick_events)},
                after_state={
                    "groove_signature": groove_signature,
                    "phase_delta_ms": phase_result.average_ms,
                    "is_wide_swing": is_wide_swing,
                    "is_dilla_pocket": is_dilla_pocket,
                    "tempo_estimate_bpm": tempo_estimate,
                    "confidence": confidence.name,
                    # the forensic breakdown behind that confidence -
                    # previously computed, stashed on self._last_confidence_components,
                    # and never seen again
                    "phase_consistency": components.phase_consistency,
                    "temporal_regularity": components.temporal_regularity,
                    "sample_size_confidence": components.sample_size_confidence,
                    "swing_definition": components.swing_definition,
                    "ensemble_agreement": components.ensemble_agreement,
                },
                reasoning=f"Groove: {groove_signature}, phase delta {phase_result.average_ms:.1f}ms",
                reversible=True,
            )

        return groove_field

    def build_note_context(self, events: List[NoteEvent]) -> List[NoteWithContext]:
        """
        Wrap each note with its neighbors for relational analysis.

        Args:
            events: List of NoteEvent objects

        Returns:
            List of NoteWithContext with neighbor references
        """
        if len(events) < 2:
            return [NoteWithContext(note=e) for e in events]

        sorted_events = sorted(events, key=lambda e: e.get_active_start_ms())
        result = []

        for i, note in enumerate(sorted_events):
            prev_note = sorted_events[i - 1] if i > 0 else None
            next_note = sorted_events[i + 1] if i < len(sorted_events) - 1 else None

            result.append(NoteWithContext(
                note=note,
                previous_note=NoteWithContext(note=prev_note) if prev_note else None,
                next_note=NoteWithContext(note=next_note) if next_note else None
            ))

        return result

    def get_phase_consistency(self) -> float:
        """Get consistency of phase deltas (0-1)."""
        if self._last_phase_result is None:
            return 0.0
        return self._last_phase_result.consistency

    def get_groove_strength(self) -> float:
        """
        Get overall groove strength (0-1).

        High value = strong, intentional groove that should be preserved.
        """
        if self._last_groove_field is None:
            return 0.0

        if self._last_groove_field.is_wide_swing or self._last_groove_field.is_dilla_pocket:
            return self._last_groove_field.confidence.value
        else:
            return 0.3

    def should_preserve_groove(self) -> bool:
        """
        Determine if groove should be preserved (not quantized).

        High-confidence wide swing or Dilla pocket should be preserved.
        """
        if self._last_groove_field is None:
            return False

        return (
                (self._last_groove_field.is_wide_swing or self._last_groove_field.is_dilla_pocket) and
                self._last_groove_field.confidence in [Confidence.HIGH, Confidence.PERFECT]
        )

    def get_quantization_recommendation(self) -> Dict[str, Any]:
        """
        Get recommendation for quantization based on groove.

        Returns:
            Dictionary with recommendation, reason, and max snap distance
        """
        if self.should_preserve_groove():
            return {
                "recommendation": QuantizationRecommendation.PRESERVE_GROOVE,
                "reason": "Preserve intentional groove",
                "max_snap_ms": 0,
                "confidence": self._last_groove_field.confidence.value if self._last_groove_field else 0
            }
        elif self._last_groove_field and self._last_groove_field.bass_kick_phase_delta_ms > 10:
            return {
                "recommendation": QuantizationRecommendation.NORMAL_QUANTIZE,
                "reason": "Large phase delta likely timing error",
                "max_snap_ms": self._config.max_snap_ms,
                "confidence": max(0.5, 1.0 - abs(self._last_groove_field.bass_kick_phase_delta_ms) / 50)
            }
        elif self._last_groove_field and self._last_phase_result and self._last_phase_result.consistency < 0.4:
            return {
                "recommendation": QuantizationRecommendation.HEAVY_QUANTIZE,
                "reason": "Inconsistent timing",
                "max_snap_ms": self._config.max_snap_ms,
                "confidence": 0.3
            }
        else:
            return {
                "recommendation": QuantizationRecommendation.LIGHT_SNAP,
                "reason": "Light quantization recommended",
                "max_snap_ms": self._config.max_snap_ms // 2,
                "confidence": 0.7
            }

    def _determine_groove_signature(self, groove_type: GrooveType, avg_delta_ms: float) -> str:
        """Determine human-readable groove signature."""
        if groove_type == GrooveType.DILLA_POCKET:
            return "dilla_pocket"
        elif groove_type == GrooveType.WIDE_SWING:
            return "wide_swing"
        elif groove_type == GrooveType.TIGHT_POCKET:
            return "tight_pocket"
        elif groove_type == GrooveType.LAID_BACK:
            return "laid_back"
        elif groove_type == GrooveType.RUSHED:
            return "rushed"
        elif groove_type == GrooveType.FLUID:
            return "fluid"
        else:
            return "erratic"

    def _get_events_from_context(self, context: AudioContext, event_type: str) -> List[NoteEvent]:
        """Extract events from context."""
        if event_type == 'bass' and hasattr(context, 'bass_events'):
            return context.bass_events
        elif event_type == 'kick' and hasattr(context, 'kick_events'):
            return context.kick_events
        return []

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
        if self._last_groove_field is None:
            return ValidationResult(
                is_valid=False,
                gate_used=gate,
                reason=VetoReason.EMPTY_RESULT,
                detail="No groove analysis performed"
            )

        if gate == ValidationGate.CONFIDENCE_THRESHOLD:
            is_valid = self._last_groove_field.confidence.value >= Confidence.LOW.value

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                detail=f"Groove confidence: {self._last_groove_field.confidence.value:.2f}",
                confidence_before=self._last_groove_field.confidence.value,
                confidence_after=self._last_groove_field.confidence.value if is_valid else 0.0
            )

        elif gate == ValidationGate.PHASE_CONSISTENCY:
            phase_consistency = self.get_phase_consistency()
            is_valid = phase_consistency > 0.4

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.PHASE_INCOHERENCE,
                detail=f"Phase consistency: {phase_consistency:.2f}"
            )

        else:
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                detail=f"Gate {gate.value} not fully supported"
            )

    def get_confidence(self) -> Confidence:
        """Return overall confidence of last output."""
        if self._last_groove_field is None:
            return Confidence.HALLUCINATION
        return self._last_groove_field.confidence

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        """Return (reason, detail) if currently vetoed, else None."""
        if self._last_groove_field is None:
            return (VetoReason.EMPTY_RESULT, "No groove analysis")

        if self._last_groove_field.confidence.value < Confidence.LOW.value:
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Confidence {self._last_groove_field.confidence.value:.2f}")

        if self.get_phase_consistency() < 0.3:
            return (VetoReason.PHASE_INCOHERENCE,
                    f"Phase consistency {self.get_phase_consistency():.2f}")

        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        """Run harmonic series check (N/A for groove analysis)."""
        return SchoenbergResult(
            verdict=SchoenbergVerdict.UNCERTAIN,
            zero_crossing_rate=0.0,
            spectral_flatness=0.5,
            reason="GrooveFieldAnalyzer analyzes relational timing, not harmonic series"
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
    # Public Methods
    # ========================================================================

    def get_last_groove_field(self) -> Optional[GrooveField]:
        """Get the last analyzed groove field."""
        return self._last_groove_field

    def get_confidence_breakdown(self) -> Optional[Dict[str, float]]:
        """Get forensic confidence breakdown."""
        if self._last_confidence_components is None:
            return None

        return {
            "phase_consistency": self._last_confidence_components.phase_consistency,
            "temporal_regularity": self._last_confidence_components.temporal_regularity,
            "sample_size_confidence": self._last_confidence_components.sample_size_confidence,
            "swing_definition": self._last_confidence_components.swing_definition,
            "ensemble_agreement": self._last_confidence_components.ensemble_agreement,
            "total": self._last_confidence_components.total()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get groove analysis statistics."""
        return {
            "name": self._name,
            "source_type": self._source_type.value,
            "last_groove": {
                "phase_delta_ms": self._last_groove_field.bass_kick_phase_delta_ms if self._last_groove_field else None,
                "groove_signature": self._last_groove_field.groove_signature if self._last_groove_field else None,
                "confidence": self._last_groove_field.confidence.name if self._last_groove_field else None,
                "is_wide_swing": self._last_groove_field.is_wide_swing if self._last_groove_field else None,
                "is_dilla_pocket": self._last_groove_field.is_dilla_pocket if self._last_groove_field else None
            } if self._last_groove_field else None,
            "phase_consistency": self.get_phase_consistency(),
            "groove_strength": self.get_groove_strength(),
            "confidence_breakdown": self.get_confidence_breakdown(),
            "last_execution_time_ms": self._last_execution_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb
        }


# ========================================================================
# Convenience Functions
# ========================================================================

def create_groove_analyzer(
        swing_threshold_ms: float = PHASE_DELTA_SWING_THRESHOLD_MS,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> GrooveFieldAnalyzer:
    """
    Convenience function to create a groove field analyzer.

    Args:
        swing_threshold_ms: Threshold for wide swing detection (ms)
        status_reporter: StatusReporter for progress
        music_box: MusicBox for logging

    Returns:
        Configured GrooveFieldAnalyzer instance
    """
    config = GrooveConfig(swing_threshold_ms=swing_threshold_ms)
    return GrooveFieldAnalyzer(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )


def quick_test_groove_analyzer(
        bass_times: List[float],
        kick_times: List[float]
) -> Dict[str, Any]:
    """
    Quick test function for groove analyzer with synthetic data.

    Args:
        bass_times: List of bass note times in ms
        kick_times: List of kick times in ms

    Returns:
        Dictionary with analysis results
    """
    print(f"Testing GrooveFieldAnalyzer on {len(bass_times)} bass, {len(kick_times)} kick events")

    # Create events
    bass_events = [
        NoteEvent(
            pitch=36, start_ms=t, end_ms=t + 100, velocity=80,
            confidence=0.9, zero_crossing_rate=0.0, source=SourceType.PITCH
        )
        for t in bass_times
    ]

    kick_events = [
        NoteEvent(
            pitch=36, start_ms=t, end_ms=t + 50, velocity=90,
            confidence=0.9, zero_crossing_rate=0.0, source=SourceType.RHYTHM
        )
        for t in kick_times
    ]

    # Create mock context
    class MockContext:
        def __init__(self):
            self.bass_events = bass_events
            self.kick_events = kick_events
            self.file_path = "test"
            self.duration_seconds = max(bass_times + kick_times, default=1) / 1000
            self.working_sample_rate = 16000
            self.num_channels_original = 1
            self.is_mono = True
            self.original_sample_rate = 16000

    context = MockContext()

    # Analyze
    analyzer = create_groove_analyzer()

    # Use analyze directly instead of run for testing
    groove = analyzer.analyze(bass_events, kick_events, context)

    print(f"  Phase delta: {groove.bass_kick_phase_delta_ms:.1f}ms")
    print(f"  Wide swing: {groove.is_wide_swing}")
    print(f"  Dilla pocket: {groove.is_dilla_pocket}")
    print(f"  Groove signature: {groove.groove_signature}")
    print(f"  Confidence: {groove.confidence.name}")
    print(f"  Tempo estimate: {groove.tempo_estimate_bpm:.1f} BPM")

    # Get quantization recommendation
    recommendation = analyzer.get_quantization_recommendation()
    print(f"  Quantization: {recommendation['recommendation']} - {recommendation['reason']}")

    return {
        "phase_delta_ms": groove.bass_kick_phase_delta_ms,
        "is_wide_swing": groove.is_wide_swing,
        "is_dilla_pocket": groove.is_dilla_pocket,
        "groove_signature": groove.groove_signature,
        "confidence": groove.confidence.name,
        "tempo_estimate_bpm": groove.tempo_estimate_bpm,
        "quantization_recommendation": recommendation['recommendation']
    }


# For standalone testing
if __name__ == "__main__":
    # Test with synthetic wide swing pattern (bass 12ms behind kick)
    print("\n=== Wide Swing Test ===")
    kick_times_wide = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    bass_times_wide = [t + 12 for t in kick_times_wide]
    quick_test_groove_analyzer(bass_times_wide, kick_times_wide)

    # Test with tight pocket (bass aligned with kick)
    print("\n=== Tight Pocket Test ===")
    kick_times_tight = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    bass_times_tight = kick_times_tight.copy()
    quick_test_groove_analyzer(bass_times_tight, kick_times_tight)

    # Test with Dilla pocket (slightly variable)
    print("\n=== Dilla Pocket Test ===")
    import random

    random.seed(42)
    kick_times_dilla = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    bass_times_dilla = [t + random.uniform(-3, 8) for t in kick_times_dilla]
    quick_test_groove_analyzer(bass_times_dilla, kick_times_dilla)