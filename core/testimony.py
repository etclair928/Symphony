# =================================================================
# MODULE: core/testimony.py
# DESCRIPTION: Canonical testimony objects for all pipeline stages.
# VERSION: 5.6.1 (Added QuaverTestimony + StructuralTestimony)
# UPDATED: 2026-06-02
#
# CHANGES IN 5.6.1:
#   1. Added QuaverTestimony (symbolic duration witness output)
#   2. Added DurationHypothesis and WitnessContribution
#   3. Added StructuralTestimony (phrase intelligence output)
#   4. Added PhaseTransition and StructuralWindow
#   5. Added StructuralInvariantVector
#   6. Updated TestimonyNormalizer for new types
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Generic, TypeVar
from datetime import datetime
from enum import Enum

from core.order_types import NoteEvent, GrooveField, TempoMap, PulseField, Voice

# ========================================================================
# Result Types
# ========================================================================

T = TypeVar('T')


@dataclass
class StageResult(Generic[T]):
    """
    Canonical stage output - used by EVERY pipeline stage.
    No raw tuples. No ad-hoc dicts. No None returns.
    """
    stage_name: str
    success: bool
    testimony: T  # Type-specific testimony object
    execution_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Evidence tracking
    evidence_acquired: List[str] = field(default_factory=list)
    evidence_released: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    # Epistemic metadata (for witness testimony)
    confidence: float = 1.0
    reasoning_trace: List[str] = field(default_factory=list)
    witness_type: str = ""  # "symbolic_witness", "field_generator", "acoustic_witness", "pitch_witness"

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def is_usable(self) -> bool:
        return self.success and not self.has_errors


# ========================================================================
# Quaver Intelligence Testimony Types (NEW in 5.6.1)
# ========================================================================

@dataclass(frozen=True)
class WitnessContribution:
    """
    One agent's contribution to a hypothesis.
    Used by QuaverIntelligence to track support/opposition provenance.
    """
    witness_name: str
    contribution: float  # Positive = support, negative = opposition
    reliability: float  # Prior reliability weight (0-1)

    def weighted_contribution(self) -> float:
        return self.contribution * self.reliability


@dataclass
class DurationHypothesis:
    """
    One competing hypothesis about a note's symbolic duration.

    Multiple hypotheses are generated per note. The EpistemicCouncil
    arbitrates. The ConsensusEngine stabilizes truth.

    NOTE: This is NOT frozen because probabilities get normalized.
    """
    symbolic_value: str  # "quarter", "eighth", "swing_eighth", "triplet", etc.
    duration_ms: float  # Exact ms this hypothesis implies
    probability: float  # Normalized probability (0-1)

    # Epistemic scores
    support_score: float = 0.0  # Total weighted support
    opposition_score: float = 0.0  # Total weighted opposition
    epistemic_tension: float = 0.0  # How contested is this hypothesis? (high = support ≈ opposition)
    uncertainty: float = 0.0  # Net uncertainty score

    # Provenance
    supporting_witnesses: List[WitnessContribution] = field(default_factory=list)
    opposing_witnesses: List[WitnessContribution] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)


@dataclass
class QuaverTestimony:
    """
    Full symbolic-duration testimony for a single note.
    Produced by QuaverIntelligence, consumed by EpistemicCouncil.

    This is EVIDENCE, not truth. The ConsensusEngine will weigh this
    against other witnesses. No notes are mutated.
    """
    note_key: Tuple[int, float]  # (pitch, start_ms) — stable identifier
    start_ms: float
    raw_duration_ms: float
    duration_hypotheses: List[DurationHypothesis] = field(default_factory=list)

    # Context
    phrase_context: Dict[str, Any] = field(default_factory=dict)
    instrument_profile: str = "unknown"
    beat_ms: float = 500.0

    # Epistemic metadata
    witness_reliability: Dict[str, float] = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)
    uncertainty: float = 0.0
    contradiction_detected: bool = False

    @property
    def primary_hypothesis(self) -> Optional[DurationHypothesis]:
        """The highest-probability hypothesis."""
        if not self.duration_hypotheses:
            return None
        return max(self.duration_hypotheses, key=lambda h: h.probability)

    @property
    def has_contention(self) -> bool:
        """True if two hypotheses are within 0.15 of each other in probability."""
        if len(self.duration_hypotheses) < 2:
            return False
        sorted_h = sorted(self.duration_hypotheses, key=lambda h: -h.probability)
        return (sorted_h[0].probability - sorted_h[1].probability) < 0.15

    @property
    def max_epistemic_tension(self) -> float:
        """Maximum epistemic tension across all hypotheses."""
        if not self.duration_hypotheses:
            return 0.0
        return max(h.epistemic_tension for h in self.duration_hypotheses)


# ========================================================================
# Phrase Intelligence Testimony Types (NEW in 5.6.1)
# ========================================================================

@dataclass
class StructuralInvariantVector:
    """
    Six orthogonal structural dimensions — properly decomposed.

    Each axis is computed from multiple testimony sources to ensure
    orthogonality. No axis should be computable from any other axis alone.
    """
    tonal_stability: float = 0.5        # 0=unstable/ambiguous, 1=stable/clear
    temporal_cyclicity: float = 0.5     # 0=through-composed, 1=highly cyclic
    transformational_depth: float = 0.5 # 0=exact repetition, 1=heavy transformation
    event_density: float = 0.5          # 0=sparse, 1=dense
    constraint_strength: float = 0.5    # 0=free, 1=highly constrained
    surface_entropy: float = 0.5        # 0=low unpredictability, 1=high

    def as_dict(self) -> Dict[str, float]:
        return {
            "tonal_stability": self.tonal_stability,
            "temporal_cyclicity": self.temporal_cyclicity,
            "transformational_depth": self.transformational_depth,
            "event_density": self.event_density,
            "constraint_strength": self.constraint_strength,
            "surface_entropy": self.surface_entropy,
        }


@dataclass
class StructuralWindow:
    """Time-windowed structural analysis snapshot."""
    start_time_ms: float
    end_time_ms: float
    invariant_vector: StructuralInvariantVector
    phase: str  # StructuralPhase.value
    phase_confidence: float
    boundary_strength: float


@dataclass
class PhaseTransition:
    """A detected change in structural phase."""
    time_ms: float
    from_phase: str
    to_phase: str
    confidence: float


@dataclass
class StructuralTestimony:
    """
    Full structural trajectory testimony from PhraseIntelligence.

    Contains competing structural hypotheses and phase transitions.
    Genre EMERGES from attractor basins — this is not detection.
    """
    # Trajectory
    structural_trajectory: List[StructuralWindow] = field(default_factory=list)
    phase_transitions: List[PhaseTransition] = field(default_factory=list)

    # Attractor basin (genre emergence)
    attractor_basin: str = "unknown"  # "pop", "jazz", "classical", "ambient", "binary"
    attractor_confidence: float = 0.0

    # Invariant evolution over time
    invariant_evolution: Dict[str, List[float]] = field(default_factory=dict)

    # Re-analysis request
    needs_reanalysis: Dict[str, Any] = field(default_factory=dict)

    # Overall confidence
    confidence: float = 0.0

    # Phrase-level synthesis (from PhraseIntelligence)
    phrase_boundaries: List[float] = field(default_factory=list)
    anacrusis_detected: Dict[str, Any] = field(default_factory=dict)
    structural_signature: Dict[str, float] = field(default_factory=dict)
    structural_regime: Dict[str, Any] = field(default_factory=dict)
    theme_elaboration_score: float = 0.5
    estimated_form: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_anacrusis(self) -> bool:
        return self.anacrusis_detected.get("detected", False)

    @property
    def anacrusis_duration_ms(self) -> float:
        return self.anacrusis_detected.get("duration_ms", 0.0)

    @property
    def requires_reanalysis(self) -> bool:
        return self.needs_reanalysis.get("requires_reanalysis", False)

    @property
    def reanalysis_severity(self) -> str:
        return self.needs_reanalysis.get("severity", "none")


# ========================================================================
# Testimony Objects by Stage
# ========================================================================

@dataclass(frozen=True)
class IngestionTestimony:
    """Stage 1: Audio loading."""
    audio_views: 'AudioViews'
    original_duration_seconds: float
    is_truncated: bool
    truncation_duration_seconds: float
    file_hash: str
    memory_mb: float


@dataclass(frozen=True)
class FeatureExtractionTestimony:
    """Stage 2: Spectral features."""
    feature_bundle_id: str  # BufferID in arena
    memory_mb: float
    feature_types: List[str]
    hop_length: int
    n_fft: int


@dataclass(frozen=True)
class TempoTestimony:
    """Stage 3: Tempo analysis."""
    tempo_bpm: float
    confidence: float
    source: str  # "librosa", "madmom", "guided", "default"
    beat_times_ms: List[float]
    tempo_map: Optional[TempoMap] = None

    # Time series for window-based analysis
    tempo_curve: List[Tuple[float, float]] = field(default_factory=list)  # (time_ms, tempo_bpm)

    # For QuaverIntelligence's per-note lookup: "local_tempo_{start_ms}" ->
    # the BPM of whichever confirmed tempo segment that note falls in
    local_tempo: Dict[str, float] = field(default_factory=dict)

    # Set by TempoOctaveCorrector when the blended tempo estimate was
    # replaced with a rational-multiple candidate that fit real onset
    # times meaningfully better (e.g. "2x (double tempo)"); None when no
    # correction was applied.
    octave_correction: Optional[str] = None


@dataclass(frozen=True)
class SeparationTestimony:
    """Stage 4: Source separation."""
    stems_available: List[str]  # "drums", "bass", "other", "vocals"
    separation_time_seconds: float
    model_used: str
    fallback_used: bool


@dataclass(frozen=True)
class PitchTestimony:
    """Stages 6,7,8: Pitch detection."""
    notes: List[NoteEvent]
    confidence: float
    model_used: str  # "crepe", "basic_pitch", "epistemic_council", "librosa"
    stem_type: str  # "bass", "master", "other"
    frame_count: int
    rejected_frames: int

    @property
    def acceptance_rate(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return (self.frame_count - self.rejected_frames) / self.frame_count


@dataclass(frozen=True)
class RhythmTestimony:
    """Stage 5: Rhythm detection."""
    onset_events: List[Any]  # OnsetEvent list
    kick_notes: List[NoteEvent]
    pattern_count: int
    tempo_estimate_bpm: float
    confidence: float


@dataclass(frozen=True)
class DrumTestimony:
    """Stage 10: Drum detection."""
    drum_events: List[Any]  # DrumEvent list
    note_events: List[NoteEvent]
    drum_types_detected: List[str]
    confidence: float

    # For QuaverIntelligence
    nearest_beat_confidence: float = 0.5
    anti_groove: float = 0.0
    # Per-note percussive anchoring: str(int(start_ms)) -> {"anchor_confidence", "anti_groove"}
    per_note: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class TonalTestimony:
    """Stage 9: Tonal/harmonic detection."""
    chord_map: List[Dict[str, Any]]
    detected_key: Optional[str]
    key_confidence: float
    frame_count: int

    # For PhraseIntelligence
    progressions: List[Any] = field(default_factory=list)
    cadences: List[float] = field(default_factory=list)  # timestamps in ms
    tonal_stability: Dict[float, float] = field(default_factory=dict)  # time_ms -> stability
    cadence_strengths: Dict[float, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TimbreTestimony:
    """Stage 11: Timbre analysis."""
    instrument_counts: Dict[str, int]
    entities: List[Dict[str, Any]]
    total_analyzed: int
    confidence: float


@dataclass(frozen=True)
class HarmonicValidationTestimony:
    """Stage 12: Harmonic validation."""
    validated_notes: List[NoteEvent]
    vetoed_count: int
    detected_key: Optional[str]
    key_confidence: float
    method: str  # "harmonic_validator", "fallback"


@dataclass(frozen=True)
class AnechoicMATestimony:
    """Stage 11.5: Anechoic MA silence/resonance analysis."""
    frame_count: int
    region_count: int
    silent_region_count: int
    resonant_region_count: int
    profile: str  # "studio", "live_room", "church", "outdoor", "auto_detected"
    applied: bool = True

    # For QuaverIntelligence - time series
    resonance_map: Dict[str, float] = field(default_factory=dict)  # "resonance_{pitch}_{start_ms}" -> value


@dataclass(frozen=True)
class GrooveTestimony:
    """Stage 13: Groove analysis."""
    phase_delta_ms: float
    is_wide_swing: bool
    is_dilla_pocket: bool
    confidence: float
    groove_field: Optional[GrooveField] = None

    # For QuaverIntelligence - time series
    swing_amounts: Dict[float, float] = field(default_factory=dict)  # time_ms -> swing_ms
    swing_beats: List[float] = field(default_factory=list)

    # For QuaverIntelligence's exact lookup shape: a flat global magnitude
    # ("swing_amount_ms") plus per-note alignment_{start_ms} -> 0-1 score
    swing_amount_ms: float = 0.0
    alignment: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PulseTestimony:
    """Stage 14: Pulse analysis."""
    tempo_bpm: float
    confidence: float
    beat_grid_ms: List[float]
    pulse_field: Optional[PulseField] = None

    # For QuaverIntelligence - detailed downbeat info
    downbeat_confidences: Dict[float, float] = field(default_factory=dict)  # time_ms -> confidence
    beat_phase_map: Dict[float, float] = field(default_factory=dict)  # time_ms -> phase (0-1)
    stability_scores: Dict[float, float] = field(default_factory=dict)  # time_ms -> stability

    # For QuaverIntelligence's exact lookup shape: str(int(start_ms)) ->
    # {"strength": .., "opposition": ..}
    per_note: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class LatticeTestimony:
    """Stage 15: Reverse Geo Crypt."""
    period_ms: float
    confidence: float
    events_analyzed: int
    ratio_family: str  # "binary", "ternary", "swing"

    # For QuaverIntelligence
    bar_structure: Dict[str, Any] = field(default_factory=dict)  # bar_lengths_ms, etc.


@dataclass(frozen=True)
class VoiceContinuityTestimony:
    """Stage 16: Voice separation."""
    voices: List[Voice]
    voice_count: int
    crossing_count: int
    confidence: float

    # For QuaverIntelligence and PhraseIntelligence
    continuity_scores: Dict[float, float] = field(default_factory=dict)  # time_ms -> score
    phrase_boundaries: List[float] = field(default_factory=list)  # timestamps in ms
    phrase_id_map: Dict[str, str] = field(default_factory=dict)  # "note_{pitch}_{start_ms}" -> phrase_id


@dataclass(frozen=True)
class QuantizationTestimony:
    """Stage 17: Quantization."""
    quantized_notes: List[NoteEvent]
    original_count: int
    snapped_count: int
    total_confidence_loss: float
    lattice_summary: str


@dataclass(frozen=True)
class MergeTestimony:
    """Stage 18: Velocity merge."""
    merged_notes: List[NoteEvent]
    original_count: int
    removed_count: int
    merge_count: int
    velocity_stats: Dict[str, float]


@dataclass(frozen=True)
class ConsensusTestimony:
    """Stage 19: Consensus."""
    final_decision: bool
    confidence: float
    veto_triggered: bool
    veto_source: Optional[str]
    witness_count: int


@dataclass(frozen=True)
class SchoenbergMirrorTestimony:
    """Schoenberg Mirror: harmonic-series audit / epistemic veto on consensus notes."""
    audited_count: int
    vetoed_count: int
    tonal_count: int
    percussion_count: int
    uncertain_count: int
    noise_count: int
    hallucination_count: int


@dataclass(frozen=True)
class ValidationTestimony:
    """Stage 20: Validation."""
    passed: bool
    hard_veto_count: int
    soft_veto_count: int
    verdict_summary: str
    is_trustworthy: bool


@dataclass(frozen=True)
class ExportTestimony:
    """Stage 21: Export."""
    exported_paths: List[str]
    formats: List[str]
    note_count: int


# ========================================================================
# Testimony Normalizer (Fixes tuple drift at the boundary)
# ========================================================================

class TestimonyNormalizer:
    """
    Singleton normalizer that converts ANY raw output into canonical testimony.
    This is the ONLY place where tuple unpacking happens.
    """

    @staticmethod
    def normalize_pitch(raw: Any, stage_name: str, stem_type: str) -> PitchTestimony:
        """
        Normalize ANY pitch detection output into PitchTestimony.

        Handles:
            - PitchTestimony (already canonical)
            - List[NoteEvent]
            - Tuple[List[NoteEvent], Dict]
            - Tuple[List[ConsensusNote], MusicalFindingsMap]
            - None
        """
        if raw is None:
            return PitchTestimony(
                notes=[],
                confidence=0.0,
                model_used="none",
                stem_type=stem_type,
                frame_count=0,
                rejected_frames=0
            )

        # Case 1: Already canonical
        if isinstance(raw, PitchTestimony):
            return raw

        notes = []
        confidence = 0.5
        frame_count = 0
        rejected_frames = 0

        # Case 2: Tuple (notes, metadata) - THE BUG PATTERN
        if isinstance(raw, tuple):
            import warnings
            warnings.warn(
                f"[{stage_name}] Received tuple, unpacking. "
                f"This should be fixed at source.",
                RuntimeWarning,
                stacklevel=2
            )

            if len(raw) > 0:
                notes_raw = raw[0]
                if len(raw) > 1:
                    metadata = raw[1]
                    if isinstance(metadata, dict):
                        confidence = metadata.get('confidence', 0.5)
                        frame_count = metadata.get('frame_count', 0)
                        rejected_frames = metadata.get('rejected_frames', 0)
            else:
                notes_raw = raw

        # Case 3: List (raw notes)
        elif isinstance(raw, list):
            notes_raw = raw

        # Case 4: Council output (special)
        elif hasattr(raw, 'notes') and hasattr(raw, 'confidence'):
            # ConsensusNote list
            notes_raw = raw.notes
            confidence = raw.confidence if hasattr(raw, 'confidence') else 0.7

        else:
            notes_raw = []
            warnings.warn(
                f"[{stage_name}] Unknown pitch output type: {type(raw)}",
                RuntimeWarning
            )

        # Convert notes to NoteEvent if needed
        for item in notes_raw:
            if hasattr(item, 'pitch') and hasattr(item, 'start_ms'):
                notes.append(item)
            elif isinstance(item, dict):
                from core.order_types import NoteEvent, SourceType
                notes.append(NoteEvent(
                    pitch=item.get('pitch', 60),
                    start_ms=item.get('start_ms', 0),
                    end_ms=item.get('end_ms', 100),
                    velocity=item.get('velocity', 80),
                    confidence=item.get('confidence', 0.5),
                    zero_crossing_rate=0.0,
                    source=SourceType.PITCH
                ))

        return PitchTestimony(
            notes=notes,
            confidence=confidence,
            model_used="unknown",
            stem_type=stem_type,
            frame_count=frame_count or len(notes),
            rejected_frames=rejected_frames
        )

    @staticmethod
    def normalize_tempo(raw: Any) -> TempoTestimony:
        """Normalize tempo detection output."""
        if raw is None:
            return TempoTestimony(
                tempo_bpm=120.0,
                confidence=0.0,
                source="default",
                beat_times_ms=[]
            )

        if isinstance(raw, TempoTestimony):
            return raw

        if isinstance(raw, dict):
            return TempoTestimony(
                tempo_bpm=raw.get('tempo_bpm', 120.0),
                confidence=raw.get('confidence', 0.5),
                source=raw.get('source', 'unknown'),
                beat_times_ms=raw.get('beat_times_ms', []),
                tempo_curve=raw.get('tempo_curve', [])
            )

        # Simple tuple (tempo, confidence)
        if isinstance(raw, tuple) and len(raw) >= 2:
            return TempoTestimony(
                tempo_bpm=float(raw[0]),
                confidence=float(raw[1]),
                source="detected",
                beat_times_ms=[]
            )

        # Just a float
        if isinstance(raw, (int, float)):
            return TempoTestimony(
                tempo_bpm=float(raw),
                confidence=0.5,
                source="estimate",
                beat_times_ms=[]
            )

        return TempoTestimony(
            tempo_bpm=120.0,
            confidence=0.0,
            source="default",
            beat_times_ms=[]
        )

    @staticmethod
    def normalize_groove(raw: Any) -> GrooveTestimony:
        """Normalize groove analysis output."""
        if raw is None:
            return GrooveTestimony(
                phase_delta_ms=0.0,
                is_wide_swing=False,
                is_dilla_pocket=False,
                confidence=0.0
            )

        if isinstance(raw, GrooveTestimony):
            return raw

        if isinstance(raw, GrooveField):
            return GrooveTestimony(
                phase_delta_ms=raw.bass_kick_phase_delta_ms,
                is_wide_swing=raw.is_wide_swing,
                is_dilla_pocket=raw.is_dilla_pocket,
                confidence=raw.confidence.value_f,
                groove_field=raw
            )

        if isinstance(raw, dict):
            return GrooveTestimony(
                phase_delta_ms=raw.get('phase_delta_ms', 0.0),
                is_wide_swing=raw.get('is_wide_swing', False),
                is_dilla_pocket=raw.get('is_dilla_pocket', False),
                confidence=raw.get('confidence', 0.0),
                swing_amounts=raw.get('swing_amounts', {}),
                swing_beats=raw.get('swing_beats', [])
            )

        return GrooveTestimony(
            phase_delta_ms=0.0,
            is_wide_swing=False,
            is_dilla_pocket=False,
            confidence=0.0
        )

    @staticmethod
    def normalize_pulse(raw: Any) -> PulseTestimony:
        """Normalize pulse analysis output."""
        if raw is None:
            return PulseTestimony(
                tempo_bpm=120.0,
                confidence=0.0,
                beat_grid_ms=[]
            )

        if isinstance(raw, PulseTestimony):
            return raw

        if isinstance(raw, dict):
            return PulseTestimony(
                tempo_bpm=raw.get('tempo_bpm', 120.0),
                confidence=raw.get('confidence', 0.0),
                beat_grid_ms=raw.get('beat_grid_ms', []),
                downbeat_confidences=raw.get('downbeat_confidences', {}),
                beat_phase_map=raw.get('beat_phase_map', {}),
                stability_scores=raw.get('stability_scores', {})
            )

        return PulseTestimony(
            tempo_bpm=120.0,
            confidence=0.0,
            beat_grid_ms=[]
        )

    @staticmethod
    def normalize_anechoic(raw: Any) -> AnechoicMATestimony:
        """Normalize anechoic MA output."""
        if raw is None:
            return AnechoicMATestimony(
                frame_count=0,
                region_count=0,
                silent_region_count=0,
                resonant_region_count=0,
                profile="unknown",
                applied=False
            )

        if isinstance(raw, AnechoicMATestimony):
            return raw

        if isinstance(raw, dict):
            return AnechoicMATestimony(
                frame_count=raw.get('frame_count', 0),
                region_count=raw.get('region_count', 0),
                silent_region_count=raw.get('silent_region_count', 0),
                resonant_region_count=raw.get('resonant_region_count', 0),
                profile=raw.get('profile', 'unknown'),
                applied=raw.get('applied', True),
                resonance_map=raw.get('resonance_map', {})
            )

        return AnechoicMATestimony(
            frame_count=0,
            region_count=0,
            silent_region_count=0,
            resonant_region_count=0,
            profile="unknown",
            applied=False
        )

    @staticmethod
    def normalize_voice_continuity(raw: Any) -> VoiceContinuityTestimony:
        """Normalize voice continuity output."""
        if raw is None:
            return VoiceContinuityTestimony(
                voices=[],
                voice_count=0,
                crossing_count=0,
                confidence=0.0
            )

        if isinstance(raw, VoiceContinuityTestimony):
            return raw

        if isinstance(raw, dict):
            return VoiceContinuityTestimony(
                voices=raw.get('voices', []),
                voice_count=raw.get('voice_count', 0),
                crossing_count=raw.get('crossing_count', 0),
                confidence=raw.get('confidence', 0.0),
                continuity_scores=raw.get('continuity_scores', {}),
                phrase_boundaries=raw.get('phrase_boundaries', []),
                phrase_id_map=raw.get('phrase_id_map', {})
            )

        return VoiceContinuityTestimony(
            voices=[],
            voice_count=0,
            crossing_count=0,
            confidence=0.0
        )

    @staticmethod
    def normalize_quaver(raw: Any) -> List[QuaverTestimony]:
        """Normalize QuaverIntelligence output to list of QuaverTestimony."""
        if raw is None:
            return []

        if isinstance(raw, list):
            # Already list of QuaverTestimony
            if all(isinstance(item, QuaverTestimony) for item in raw):
                return raw
            # List of dicts
            return [QuaverTestimony(**item) if isinstance(item, dict) else item for item in raw]

        if isinstance(raw, dict):
            # Single dict
            return [QuaverTestimony(**raw)]

        if isinstance(raw, QuaverTestimony):
            return [raw]

        return []

    @staticmethod
    def normalize_structural(raw: Any) -> Optional[StructuralTestimony]:
        """Normalize PhraseIntelligence output to StructuralTestimony."""
        if raw is None:
            return None

        if isinstance(raw, StructuralTestimony):
            return raw

        if isinstance(raw, dict):
            return StructuralTestimony(**raw)

        return None


# Singleton instance
normalizer = TestimonyNormalizer()


# ========================================================================
# Exports
# ========================================================================

__all__ = [
    # Base
    "StageResult",
    "TestimonyNormalizer",
    "normalizer",

    # Quaver Intelligence (NEW)
    "WitnessContribution",
    "DurationHypothesis",
    "QuaverTestimony",

    # Phrase Intelligence (NEW)
    "StructuralInvariantVector",
    "StructuralWindow",
    "PhaseTransition",
    "StructuralTestimony",

    # Stage testimonies
    "IngestionTestimony",
    "FeatureExtractionTestimony",
    "TempoTestimony",
    "SeparationTestimony",
    "PitchTestimony",
    "RhythmTestimony",
    "DrumTestimony",
    "TonalTestimony",
    "TimbreTestimony",
    "HarmonicValidationTestimony",
    "AnechoicMATestimony",
    "GrooveTestimony",
    "PulseTestimony",
    "LatticeTestimony",
    "VoiceContinuityTestimony",
    "QuantizationTestimony",
    "MergeTestimony",
    "ConsensusTestimony",
    "ValidationTestimony",
    "ExportTestimony",
]