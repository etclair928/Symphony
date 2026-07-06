# =================================================================
# MODULE: core/protocols.py
# DESCRIPTION: Abstract base classes and Protocols for all agents.
# NO IMPLEMENTATIONS. Only interfaces that define WHAT agents do.
#
# These ensure unidirectional flow and prevent circular imports.
# Agents import from here; protocols never import agents.
#
# DEPENDS ON: order_types.py, constants.py, feature_bundle.py
# DEPENDENTS: All agent implementations.
#
# Laws encoded here:
#   - Every agent must declare its SourceType (for forensic logging)
#   - Validatable outputs support the Epistemic Veto system
#   - Memory-managed agents support Resource Survival
#   - No stateful protocols (StateManager is FORBIDDEN)
#
# VERSION: 5.6.1 (refactored)
# UPDATED: 2026-05-12
# =================================================================

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Union, Protocol, runtime_checkable
from enum import Enum
import numpy as np

# Import ONLY from core modules - NO CIRCULAR IMPORTS
from core.order_types import (
    NoteEvent, GrooveField, AudioContext, StageResult,
    SourceType, StemType, ValidationResult, VetoReason,
    ConsensusPackage, WitnessVote, DrumEvent, TempoMap,
    PulseField, SeparationResult, Voice, NoteWithContext,
    RhythmPattern, OnsetEvent, SchoenbergResult, AnechoicMask,
    VelocityMergeResult, MultiStageQuantization, ValidationGate,
    Confidence, QuantizationStrategy, ConsensusStrategy, MergeStrategy,
    WitnessTestimony, ConsensusConfig, TempoEvent, BeatTrackingResult,
    TimeSignatureCandidate, TempoIntelligenceResult, RhythmTrackResult,
    DrumTrackResult, DrumKitMapping, VoiceContinuityResult,
    HarmonicSeries, PartialTrack, DecisionType
)

from core.constants import (
    CONFIDENCE_WEIGHTS, VETO_HIERARCHY, SPICE_MODEL_SIZE,
    DEMUCS_MODEL, LIBROSA_ONSET_THRESHOLD,
    MIN_CONFIDENCE_TO_PASS, VALIDATION_GATE_TO_VETO,
    MIN_QUALIFIED_WITNESSES_FOR_CONSENSUS
)

# Import from feature_bundle for spectral evidence
from core.feature_bundle import (
    EvidenceType,
    FeatureBundle,
    MemoryPriority as FeatureMemoryPriority
)

# ============================================================================
# TYPE CHECKING FORWARD REFERENCES
# ============================================================================

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These are imported at runtime but need forward references for type hints
    from agents.base import BaseAgent


# ============================================================================
# LAW 2: UNIDIRECTIONAL INTEGRITY
# ============================================================================

class AgentProtocol(Protocol):
    """Base protocol for all Grimlock agents."""

    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """Process audio and return events."""
        ...

    @property
    def source_type(self) -> SourceType:
        """Which source type this agent represents."""
        ...

    @property
    def name(self) -> str:
        """Agent identifier for MusicBox logging."""
        ...


# ============================================================================
# LAW 1: RESOURCE SURVIVAL
# ============================================================================

class MemoryManagedProtocol(Protocol):
    """Law of Resource Survival: Agents must release buffers."""

    def release_buffer(self, buffer_name: str) -> None:
        """Explicit deletion request for audio buffers."""
        ...

    def get_memory_footprint_mb(self) -> float:
        """Current memory usage in MB."""
        ...

    def can_release(self, buffer_name: str) -> bool:
        """Return True if this buffer is safe to delete."""
        ...

    def staggered_gc(self) -> Dict[str, Any]:
        """Run garbage collection and return stats."""
        ...


# ============================================================================
# LAW 3: SCRIBE'S TRUTH (Epistemic Veto)
# ============================================================================

class ScribeValidatable(Protocol):
    """Law of Scribe's Truth: Output can be validated."""

    def validate(self, gate: ValidationGate) -> ValidationResult:
        """Run validation gate on this agent's output."""
        ...

    def get_confidence(self) -> Confidence:
        """Return overall confidence of last output."""
        ...

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        """Return (reason, detail) if currently vetoed, else None."""
        ...

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        """Run harmonic series check."""
        ...


# ============================================================================
# FEATURE BUNDLE PROTOCOL (Shared Spectral Evidence)
# ============================================================================

class FeatureBundleProtocol(Protocol):
    """
    Protocol for shared spectral evidence.

    LAW: This is the ONLY way agents should access spectral data.
    Agents borrow FeatureBundle via MemoryArena, never own it directly.
    """

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio this bundle represents."""
        ...

    @property
    def memory_mb(self) -> float:
        """Current memory footprint in MB."""
        ...

    @property
    def total_frames(self) -> int:
        """Number of time frames in the bundle."""
        ...

    def get_available_evidence(self) -> List[EvidenceType]:
        """Return list of evidence types currently available."""
        ...

    def has_evidence(self, evidence_type: EvidenceType) -> bool:
        """Check if evidence of this type is currently available."""
        ...

    def get_spectral_slice(self, start_ms: float, end_ms: float,
                           freq_min_hz: float, freq_max_hz: float) -> Optional[np.ndarray]:
        """Extract spectral region. Returns a VIEW, not a copy."""
        ...

    def acquire_lease(self, evidence_type: EvidenceType, stage: str,
                      ttl_seconds: float = 300,
                      priority: Optional[FeatureMemoryPriority] = None) -> bool:
        """Acquire a lease on evidence. Returns False if already leased."""
        ...

    def release(self, evidence_type: EvidenceType, stage: Optional[str] = None) -> bool:
        """Explicitly release evidence memory."""
        ...

    def can_release(self, evidence_type: EvidenceType, stage: str) -> bool:
        """Check if evidence can be safely released."""
        ...


# ============================================================================
# SEPARATION AGENTS
# ============================================================================

class SeparationAgentProtocol(AgentProtocol, Protocol):
    """Separates audio into stems."""

    def separate(self, audio_buffer: np.ndarray, context: AudioContext) -> SeparationResult:
        """Return dict of stem_type -> audio_buffer."""
        ...

    @property
    def available_stems(self) -> List[StemType]:
        """Which stems this separator can produce."""
        ...

    @property
    def separation_quality(self) -> str:
        """'fast', 'balanced', 'high'"""
        ...

    def release_stem(self, stem_type: StemType) -> None:
        """Explicitly free a stem buffer when no longer needed."""
        ...


class DemucsProtocol(SeparationAgentProtocol, Protocol):
    """Demucs-specific interface."""

    @property
    def demucs_model_name(self) -> str:
        return DEMUCS_MODEL

    @property
    def segment_duration_seconds(self) -> float:
        ...

    @property
    def overlap_seconds(self) -> float:
        ...


class BSRoformerProtocol(SeparationAgentProtocol, Protocol):
    """BS_Roformer_Tracker interface."""

    @property
    def roformer_sample_rate(self) -> int:
        ...

    @property
    def hop_length_samples(self) -> int:
        ...


class HybridSeparatorProtocol(SeparationAgentProtocol, Protocol):
    """Hybrid separator that combines multiple separation models."""

    def set_model_weights(self, demucs_weight: float, roformer_weight: float) -> None:
        ...

    def get_ensemble_vote(self, stem_type: StemType) -> float:
        ...


# ============================================================================
# DETECTION AGENTS
# ============================================================================

class DetectionAgentProtocol(AgentProtocol, ScribeValidatable, Protocol):
    """Base for all detection agents."""

    def detect(self, stem_buffer: np.ndarray, context: AudioContext) -> List[NoteEvent]:
        """Detect events from audio stem."""
        ...

    @property
    def detection_threshold(self) -> Confidence:
        return Confidence.LOW

    def get_raw_confidence(self) -> float:
        """Get raw confidence score for last detection."""
        ...


class RhythmEngineProtocol(DetectionAgentProtocol, Protocol):
    """Rhythm detection."""

    def detect_onsets(self, audio_buffer: np.ndarray) -> List[OnsetEvent]:
        """Detect onset events."""
        ...

    def find_patterns(self, onsets: List[OnsetEvent]) -> List[RhythmPattern]:
        """Find repeating rhythmic patterns."""
        ...

    def estimate_tempo(self, audio_buffer: np.ndarray) -> Tuple[float, Confidence]:
        """Estimate tempo from audio."""
        ...

    def build_rhythm_track(self, audio_buffer: np.ndarray) -> RhythmTrackResult:
        """Complete rhythm analysis."""
        ...


class PitchEngineProtocol(DetectionAgentProtocol, Protocol):
    """Pitch detection."""

    @property
    def pitch_strategy(self) -> str:
        """Name of pitch detection strategy used."""
        ...

    def detect_pitch_contour(self, audio_buffer: np.ndarray) -> np.ndarray:
        """Detect continuous pitch contour."""
        ...

    def get_voicing_flags(self, audio_buffer: np.ndarray) -> List[bool]:
        """Get voiced/unvoiced flags per frame."""
        ...

    def basic_pitch_specific(self) -> Dict[str, Any]:
        """Basic Pitch-specific configuration."""
        ...


class TonalDetectorProtocol(DetectionAgentProtocol, Protocol):
    """Tonal detection with harmonic series confirmation."""

    def detect_harmonic_series(self, audio_buffer: np.ndarray, fundamental_hz: float) -> HarmonicSeries:
        """Detect harmonic series for a given fundamental."""
        ...

    def get_schoenberg_verdict(self, note: NoteEvent,
                               feature_bundle: Optional[FeatureBundleProtocol] = None) -> SchoenbergResult:
        """Get Schoenberg Mirror verdict for a note."""
        ...

    @property
    def max_harmonics(self) -> int:
        return 8


class DrumIntelligenceProtocol(DetectionAgentProtocol, Protocol):
    """Drum detection using SPICE or similar."""

    def detect_drums(self, drum_buffer: np.ndarray, feature_bundle: Optional[FeatureBundleProtocol] = None) -> List[
        DrumEvent]:
        """Detect drum events from audio."""
        ...

    def get_drum_track(self, drum_buffer: np.ndarray) -> DrumTrackResult:
        """Complete drum track analysis."""
        ...

    @property
    def spice_model_size(self) -> str:
        return SPICE_MODEL_SIZE

    @property
    def drum_kit_mapping(self) -> DrumKitMapping:
        """Current drum kit mapping."""
        ...


# ============================================================================
# ANALYSIS AGENTS
# ============================================================================

class GrooveFieldAnalyzerProtocol(Protocol):
    """Relational Physics: measures distance between notes."""

    def analyze(self, bass_events: List[NoteEvent], kick_events: List[NoteEvent],
                context: AudioContext) -> GrooveField:
        """Analyze groove from bass and kick events."""
        ...

    def get_phase_delta(self, note1: NoteEvent, note2: NoteEvent) -> float:
        """Calculate phase delta between two notes in ms."""
        ...

    def build_note_context(self, events: List[NoteEvent]) -> List[NoteWithContext]:
        """Build neighbor-aware note context."""
        ...


class VoiceContinuityProtocol(Protocol):
    """Voice continuity across polyphonic texture."""

    def separate_voices(self, notes: List[NoteEvent]) -> List[Voice]:
        """Separate notes into monophonic voices."""
        ...

    def assign_voice_roles(self, voices: List[Voice]) -> List[Voice]:
        """Assign melodic roles to voices."""
        ...

    def detect_voice_crossings(self, voices: List[Voice]) -> List[Tuple[Voice, Voice, float]]:
        """Detect where voices cross."""
        ...

    def get_continuity_result(self, notes: List[NoteEvent]) -> VoiceContinuityResult:
        """Complete voice continuity analysis."""
        ...


class TempoIntelligenceProtocol(Protocol):
    """Tempo Intelligence: detects tempo changes, pulse field, time signature."""

    def analyze_tempo(self, audio_buffer: np.ndarray,
                      feature_bundle: Optional[FeatureBundleProtocol] = None) -> TempoMap:
        """Analyze tempo map from audio."""
        ...

    def build_pulse_field(self, tempo_map: TempoMap, audio_buffer: np.ndarray) -> PulseField:
        """Build pulse field from tempo map."""
        ...

    def detect_time_signature(self, beat_track: List[float]) -> List[TimeSignatureCandidate]:
        """Detect time signature candidates from beat track."""
        ...

    def get_intelligence_result(self, audio_buffer: np.ndarray) -> TempoIntelligenceResult:
        """Complete tempo intelligence result."""
        ...


# ============================================================================
# QUANTIZATION AGENTS
# ============================================================================

class RitornelloProtocol(Protocol):
    """Non-destructive quantization."""

    def quantize(self, events: List[NoteEvent],
                 groove_field: Optional[GrooveField] = None,
                 pulse_field: Optional[PulseField] = None,
                 strategy: QuantizationStrategy = QuantizationStrategy.MULTI_STAGE) -> List[NoteEvent]:
        """Quantize notes to grid."""
        ...

    def get_quantization_history(self) -> MultiStageQuantization:
        """Get audit trail of quantization passes."""
        ...

    def get_confidence_penalty(self, shift_ms: float) -> float:
        """Calculate confidence penalty for a given shift."""
        ...


class QuantizationAgentProtocol(Protocol):
    """Non-destructive quantization agent interface."""

    def quantize(self, events: List[NoteEvent], context: AudioContext) -> List[NoteEvent]:
        """Apply quantization to note events."""
        ...

    @property
    def quantization_strength(self) -> float:
        """0.0 to 1.0 - how strongly to snap to grid."""
        ...

    @property
    def swing_factor(self) -> float:
        """Swing amount for quantization."""
        ...

    def get_quantization_grid(self, tempo_map: TempoMap) -> List[float]:
        """Return grid positions in seconds."""
        ...

    def set_strategy(self, strategy: QuantizationStrategy) -> None:
        """Change quantization strategy."""
        ...


class VelocityMergeProtocol(Protocol):
    """Velocity Merge: merges overlapping/adjacent notes."""

    def merge_overlaps(self, events: List[NoteEvent]) -> VelocityMergeResult:
        """Merge notes that overlap in time."""
        ...

    def merge_adjacent(self, events: List[NoteEvent], max_gap_ms: float) -> VelocityMergeResult:
        """Merge notes that are very close together."""
        ...

    def set_merge_strategy(self, strategy: MergeStrategy) -> None:
        """Change merge strategy."""
        ...

    def get_merge_candidates(self, events: List[NoteEvent]) -> List[List[NoteEvent]]:
        """Find groups of notes that could be merged."""
        ...


# ============================================================================
# VALIDATION AGENTS
# ============================================================================

class ValidationAgentProtocol(AgentProtocol, ScribeValidatable, Protocol):
    """Base protocol for all validation agents."""

    def validate(self, stage_result: StageResult, context: AudioContext) -> ValidationResult:
        """Validate a stage result and return validation outcome."""
        ...

    @property
    def validation_gate(self) -> ValidationGate:
        """Which validation gate this agent uses."""
        ...

    def get_veto_power(self) -> bool:
        """Whether this agent can issue binding vetoes."""
        ...

    def get_validation_history(self) -> List[ValidationResult]:
        """Return history of validations performed."""
        ...


class ScribeProtocol(Protocol):
    """Epistemic Veto authority."""

    def validate_stage(self, result: StageResult, source: SourceType) -> ValidationResult:
        """Validate a stage result."""
        ...

    def apply_schoenberg_mirror(self, events: List[NoteEvent],
                                feature_bundle: Optional[FeatureBundleProtocol] = None) -> SchoenbergResult:
        """Apply harmonic series validation."""
        ...

    def check_silence(self, events: List[NoteEvent], duration_seconds: float) -> ValidationResult:
        """Check if result has too much silence."""
        ...

    def get_veto_chain(self) -> List[Tuple[SourceType, VetoReason, str]]:
        """Get full veto history."""
        ...

    def build_validation_gate(self, gate: ValidationGate, events: List[NoteEvent]) -> ValidationResult:
        """Build a specific validation gate."""
        ...


class ConsensusEngineProtocol(Protocol):
    """Consensus Engine: multiple witnesses vote; any veto wins."""

    def gather_testimonies(self, agents: List[AgentProtocol]) -> List[WitnessTestimony]:
        """Gather testimonies from all witness agents."""
        ...

    def convert_to_votes(self, testimonies: List[WitnessTestimony]) -> List[WitnessVote]:
        """Convert testimonies to structured votes."""
        ...

    def check_vetoes(self, votes: List[WitnessVote]) -> Optional[Tuple[SourceType, VetoReason, str]]:
        """Check if any vote constitutes a veto."""
        ...

    def reach_consensus(self, testimonies: List[WitnessTestimony]) -> ConsensusPackage:
        """Reach consensus from all testimonies."""
        ...

    def get_confidence_weight(self, source: SourceType) -> float:
        """Get confidence weight for a source."""
        return CONFIDENCE_WEIGHTS.get(source, 0.5)

    def get_config(self) -> ConsensusConfig:
        """Get consensus engine configuration."""
        ...


# ============================================================================
# TRACKER AGENTS
# ============================================================================

class OnsetTrackerProtocol(Protocol):
    """Onset detection using librosa or similar."""

    def detect_onsets(self, audio_buffer: np.ndarray) -> List[float]:
        """Detect onset timestamps in seconds."""
        ...

    @property
    def threshold(self) -> float:
        return LIBROSA_ONSET_THRESHOLD

    def get_onset_strengths(self) -> List[float]:
        """Get onset strength for each detection."""
        ...


class BeatTrackerProtocol(Protocol):
    """Beat tracking using madmom."""

    def detect_beats(self, audio_buffer: np.ndarray) -> List[float]:
        """Detect beat timestamps in seconds."""
        ...

    def get_beat_confidences(self) -> List[float]:
        """Get confidence for each detected beat."""
        ...

    def get_beat_grid(self) -> Optional[List[float]]:
        """Get full beat grid if available."""
        ...

    @property
    def fps(self) -> int:
        return 100


# ============================================================================
# EXPORT AGENTS
# ============================================================================

class MidiExporterProtocol(Protocol):
    """MIDI export."""

    def export_midi(self, events: List[NoteEvent], tempo_map: TempoMap,
                    context: AudioContext, output_path: str) -> bool:
        """Export notes to MIDI file."""
        ...

    def apply_octave_restoration(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Apply octave restoration for bass."""
        ...

    def add_heartbeat_if_empty(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Add heartbeat note if result is empty."""
        ...


class JsonExporterProtocol(Protocol):
    """JSON export with forensic audit trail."""

    def export_json(self, events: List[NoteEvent], music_box_logs: List[Dict[str, Any]],
                    context: AudioContext, output_path: str) -> bool:
        """Export results to JSON file."""
        ...

    def include_forensic_audit(self, include: bool) -> None:
        """Toggle forensic audit inclusion."""
        ...

    def get_exporter_state(self) -> Dict[str, Any]:
        """Get current exporter state."""
        ...


# ============================================================================
# ORCHESTRATION PROTOCOLS
# ============================================================================

class PipelineStageProtocol(Protocol):
    """A single stage in the linear pipeline."""

    def execute(self, input_data: Any, context: AudioContext) -> StageResult:
        """Execute this pipeline stage."""
        ...

    @property
    def stage_name(self) -> str:
        """Name of this stage."""
        ...

    @property
    def source_type(self) -> SourceType:
        """Source type for events from this stage."""
        ...


class MemoryGuardianProtocol(Protocol):
    """Resource management."""

    def load_and_downmix(self, file_path: str) -> Tuple[np.ndarray, AudioContext]:
        """Load audio with mono downmixing."""
        ...

    def staggered_gc(self, stage_name: str) -> Dict[str, Any]:
        """Run staggered garbage collection."""
        ...

    def mark_for_deletion(self, buffer_name: str, buffer_obj: Any, stage_cleared: str) -> None:
        """Mark buffer for deletion."""
        ...

    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory status."""
        ...

    def is_critical(self) -> bool:
        """Check if memory is at critical level."""
        ...


class OrchestratorProtocol(Protocol):
    """Pipeline orchestrator."""

    def run_pipeline(self, audio_path: str) -> StageResult:
        """Run the complete pipeline."""
        ...

    def add_stage(self, stage: PipelineStageProtocol, index: Optional[int] = None) -> None:
        """Add a stage to the pipeline."""
        ...

    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result from a specific stage."""
        ...

    def pause_on_veto(self, pause: bool) -> None:
        """Set whether to pause on veto."""
        ...


# ============================================================================
# MUSIC BOX PROTOCOL
# ============================================================================

class MusicBoxProtocol(Protocol):
    """Append-only logger. Every decision recorded for forensic audit."""

    def log_decision(self, stage_name: str, decision_type: Union[str, DecisionType],
                     before_state: Dict[str, Any],
                     after_state: Dict[str, Any], reasoning: str, reversible: bool = True) -> None:
        """Log a decision."""
        ...

    def log_veto(self, source: SourceType, reason: VetoReason, detail: str, gate: ValidationGate) -> None:
        """Log a veto event."""
        ...

    def log_error(self, stage_name: str, error: Exception, context: Dict[str, Any]) -> None:
        """Log an error."""
        ...

    def get_session_logs(self) -> List[Dict[str, Any]]:
        """Get all logs for current session."""
        ...

    def flush(self) -> None:
        """Flush log buffer."""
        ...

    def get_session_id(self) -> str:
        """Get current session ID."""
        ...


# ============================================================================
# STATUS REPORTER PROTOCOL
# ============================================================================

class StatusReporterProtocol(Protocol):
    """Protocol for reporting status updates."""

    def report_status(self, status: str, progress: Optional[float] = None, **kwargs) -> None:
        """Report a status update."""
        ...

    def report_error(self, error: str, fatal: bool = False) -> None:
        """Report an error."""
        ...

    def report_progress(self, percent: float, message: str) -> None:
        """Report progress percentage."""
        ...


# ============================================================================
# ANALYSIS AGENT PROTOCOL
# ============================================================================

class AnalysisAgentProtocol(Protocol):
    """Protocol for analysis agents (drums, pitch, rhythm)."""

    @property
    def name(self) -> str:
        """Agent name."""
        ...

    @property
    def capabilities(self) -> List[str]:
        """List of capabilities."""
        ...

    async def can_handle(self, stem: Any, context: AudioContext) -> bool:
        """Check if agent can handle this audio."""
        ...

    async def analyze(self, stem: Any, context: AudioContext) -> Any:
        """Run analysis."""
        ...

    def get_confidence(self, result: Any) -> float:
        """Get confidence score for result."""
        ...


# ============================================================================
# FACTORY PROTOCOLS
# ============================================================================

class AgentFactoryProtocol(Protocol):
    """Creates agents without circular imports."""

    def create_separator(self, separator_type: str, **kwargs) -> SeparationAgentProtocol:
        """Create a separator agent."""
        ...

    def create_detector(self, detector_type: str, **kwargs) -> DetectionAgentProtocol:
        """Create a detection agent."""
        ...

    def create_analyzer(self, analyzer_type: str, **kwargs) -> Any:
        """Create an analysis agent."""
        ...

    def register_agent(self, name: str, agent_class: type) -> None:
        """Register a custom agent."""
        ...


# ============================================================================
# UTILITY PROTOCOLS
# ============================================================================

@runtime_checkable
class AnechoicMaskProtocol(Protocol):
    """De-reverberation agent."""

    def apply_mask(self, audio_buffer: np.ndarray, context: AudioContext) -> Tuple[np.ndarray, AnechoicMask]:
        """Apply de-reverberation mask."""
        ...

    def detect_profile(self, audio_buffer: np.ndarray) -> AnechoicMask:
        """Detect room profile."""
        ...


@runtime_checkable
class AnechoicMaProtocol(Protocol):
    """Alias for AnechoicMaskProtocol - legacy name."""

    def apply_mask(self, audio_buffer: np.ndarray, context: AudioContext) -> Tuple[np.ndarray, AnechoicMask]:
        """Apply de-reverberation mask."""
        ...

    def detect_profile(self, audio_buffer: np.ndarray) -> AnechoicMask:
        """Detect room profile."""
        ...


@runtime_checkable
class MusicBrainzProtocol(Protocol):
    """MusicBrainz integration for metadata lookup."""

    def lookup_artist(self, artist_name: str) -> Dict[str, Any]:
        """Look up artist metadata."""
        ...

    def get_expected_tempo_range(self, genre: str) -> Tuple[float, float]:
        """Get expected tempo range for a genre."""
        ...


@runtime_checkable
class HeartbeatGeneratorProtocol(Protocol):
    """Last-resort generator for empty transcriptions."""

    def generate_heartbeat(self, context: AudioContext) -> List[NoteEvent]:
        """Generate a heartbeat note."""
        ...

    def should_generate_heartbeat(self, events: List[NoteEvent]) -> bool:
        """Check if heartbeat should be generated."""
        ...


# ============================================================================
# DEPRECATED PROTOCOLS (4.7 compatibility only)
# ============================================================================

class ConfidenceRouterProtocol(Protocol):
    """DEPRECATED: Routing is forbidden in 5.0."""

    def route_to_agent(self, confidence_score: float) -> str:
        """DEPRECATED: Do not use."""
        ...


class FusionLayerProtocol(Protocol):
    """DEPRECATED: Fusion is forbidden in 5.0."""

    def fuse_stems(self, stems: Dict[StemType, np.ndarray]) -> np.ndarray:
        """DEPRECATED: Do not use."""
        ...


class StateManagerProtocol(Protocol):
    """DEPRECATED: State causes split-brain."""

    def get_state(self, key: str) -> Any:
        """DEPRECATED: Do not use."""
        ...

    def set_state(self, key: str, value: Any) -> None:
        """DEPRECATED: Do not use."""
        ...

    def clear_state(self) -> None:
        """DEPRECATED: Do not use."""
        ...


# ============================================================================
# DEPRECATION WARNING SHIM
# ============================================================================

import warnings


def __getattr__(name: str):
    """
    Provide deprecation warnings for legacy protocol names.
    """
    deprecated_protocols = {
        "ConfidenceRouterProtocol": "Routing is forbidden in 5.0",
        "FusionLayerProtocol": "Fusion is forbidden in 5.0",
        "StateManagerProtocol": "State causes split-brain",
    }

    if name in deprecated_protocols:
        warnings.warn(
            f"'{name}' is deprecated in Grimlock 5.0. "
            f"{deprecated_protocols[name]}.",
            DeprecationWarning,
            stacklevel=2
        )

        # Return a placeholder that raises on use
        class _DeprecatedPlaceholder:
            def __getattr__(self, _):
                raise RuntimeError(f"{name} is deprecated and cannot be used")

        return _DeprecatedPlaceholder()

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ============================================================================
# __ALL__ EXPORTS
# ============================================================================

__all__ = [
    # Base protocols
    "AgentProtocol",
    "MemoryManagedProtocol",
    "ScribeValidatable",

    # Feature Bundle
    "FeatureBundleProtocol",

    # Separation
    "SeparationAgentProtocol",
    "DemucsProtocol",
    "BSRoformerProtocol",
    "HybridSeparatorProtocol",

    # Detection
    "DetectionAgentProtocol",
    "RhythmEngineProtocol",
    "PitchEngineProtocol",
    "TonalDetectorProtocol",
    "DrumIntelligenceProtocol",

    # Analysis
    "GrooveFieldAnalyzerProtocol",
    "VoiceContinuityProtocol",
    "TempoIntelligenceProtocol",

    # Quantization
    "RitornelloProtocol",
    "QuantizationAgentProtocol",
    "VelocityMergeProtocol",

    # Validation
    "ValidationAgentProtocol",
    "ScribeProtocol",
    "ConsensusEngineProtocol",

    # Trackers
    "OnsetTrackerProtocol",
    "BeatTrackerProtocol",

    # Export
    "MidiExporterProtocol",
    "JsonExporterProtocol",

    # Orchestration
    "PipelineStageProtocol",
    "MemoryGuardianProtocol",
    "OrchestratorProtocol",

    # Music Box
    "MusicBoxProtocol",
    "StatusReporterProtocol",

    # Analysis Agent
    "AnalysisAgentProtocol",

    # Factory
    "AgentFactoryProtocol",

    # Deprecated
    "ConfidenceRouterProtocol",
    "FusionLayerProtocol",
    "StateManagerProtocol",

    # Utility
    "AnechoicMaskProtocol",
    "AnechoicMaProtocol",
    "MusicBrainzProtocol",
    "HeartbeatGeneratorProtocol",
]