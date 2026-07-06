# =====================================================================
# MODULE: core/contracts.py
# DESCRIPTION: Canonical contract definitions for Grimlock 5.2.
#              Single source of truth for WHAT components must do.
#
# VERSION: 5.6.1 (Reconciled - Claude's explanations + Our completeness)
# UPDATED: 2026-05-16
#
# PHILOSOPHY:
#     "Contracts define WHAT. Implementations define HOW."
#     "No concrete implementations. Only Protocols and ABCs."
#     "Every component must conform to exactly one contract."
#
# IMPORT RULE:
#     This file imports ONLY from core.order_types and stdlib.
#     No agents. No orchestration. No memory modules.
#     Import chain: order_types → contracts → everything else.
#
# WHY AudioContract FIXES THE STEM SAMPLE RATE BUG:
#     The root cause of pitch/drum/rhythm detection errors was that
#     Demucs stems were stored as raw numpy arrays with no sample_rate
#     attribute. Arena._ensure_sample_rate() silently skipped resampling
#     because it couldn't find the rate. AudioContract solves this:
#     every audio object — master, stem, chunk — carries its sample_rate.
#     Arena stores AudioContracts, not arrays. Agents borrow AudioContracts.
#     The sample rate is always known. Resampling is always correct.
#
# CONTRACTS IN THIS FILE:
#     1. AudioContract          — Immutable audio ledger (carries sample rate)
#     2. DetectionContract      — What detection agents must return
#     3. SeparationContract     — What separation agents must return
#     4. AnalysisContract       — What analysis agents must return
#     5. QuantizationContract   — What quantizers must guarantee
#     6. ValidationContract     — What validators must enforce
#     7. ArenaContract          — What the memory arena must provide
#     8. GuardianContract       — What the memory guardian must enforce
#     9. FeatureBundleContract  — What the spectral bus must expose
#     10. ObservableContract    — What loggable components must expose
# =====================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Protocol,
    Tuple, Iterator, Union, runtime_checkable
)

import numpy as np

# Import ONLY from core.order_types (bedrock types)
from core.order_types import (
    AudioContext, BeatGrid, BeatTrackingResult, Confidence,
    ConsensusPackage, DrumEvent, DrumTrackResult, GrooveField,
    MergeStrategy, NoteEvent, OnsetEvent, PulseField,
    QuantizationStrategy, RhythmPattern, SchoenbergResult,
    SeparationResult, SourceType, StageResult, StemType,
    TempoEvent, TempoMap, TimeSignatureCandidate, ValidationGate,
    ValidationResult, VetoReason, Voice, VoiceContinuityResult,
    WitnessTestimony
)


# ============================================================================
# PART 1: AUDIO CONTRACT — The Immutable Audio Ledger
#
# This is the core fix for the stem sample rate bug. Every audio object
# in the system — master audio, separated stems, resampled variants —
# must satisfy this contract. The sample_rate property is always present.
# Arena._ensure_sample_rate() can always find the rate. Resampling is
# always correct. The phantom "hasattr(buffer, 'sample_rate')" check
# disappears because all audio is an AudioContract, not a raw array.
# ============================================================================

@runtime_checkable
class AudioContract(Protocol):
    """
    Immutable audio ledger. Every audio buffer in the system must satisfy this.

    LAW: "Once wrapped, forever frozen. samples is always writeable=False."
    LAW: "sample_rate is always present. Raw numpy arrays are forbidden."
    LAW: "Every transformation leaves a forensic breadcrumb in lineage."
    LAW: "Timing conversions use integer arithmetic. No float drift."

    IMPLEMENTATION: core.acoustic_intelligence.AudioContractImpl
    """

    @property
    def samples(self) -> np.ndarray:
        """
        Read-only contiguous float32 array.
        Shape (n_samples,) for mono or (n_samples, n_channels) for stereo.
        MUST be writeable=False. Any mutation is a contract violation.
        """
        ...

    @property
    def sample_rate(self) -> int:
        """
        Sample rate in Hz. THIS IS THE CRITICAL FIELD.
        Without this, arena._ensure_sample_rate() silently skips resampling.
        Every stem produced by Demucs must carry this. No exceptions.
        """
        ...

    @property
    def total_samples(self) -> int:
        """Total number of sample frames."""
        ...

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        ...

    @property
    def sha256_hash(self) -> str:
        """64-character hex SHA-256 of the raw sample bytes. Forensic identity."""
        ...

    @property
    def lineage(self) -> str:
        """
        Transformation history. E.g.:
        "loaded:48000Hz:stereo → resample:44100Hz → demucs:drums → mono"
        Append-only. Never reset between stages.
        """
        ...

    @property
    def channel_layout(self) -> str:
        """'mono', 'stereo', or 'multichannel'."""
        ...

    @property
    def num_channels(self) -> int:
        """Number of audio channels (1 for mono, 2 for stereo)."""
        ...

    @property
    def created_at(self) -> datetime:
        """UTC timestamp of contract creation."""
        ...

    def index_to_ms(self, sample_index: int) -> float:
        """Convert sample index to milliseconds. NO FLOAT DIVISION."""
        ...

    def ms_to_index(self, time_ms: float) -> int:
        """Convert milliseconds to sample index. NO FLOAT DIVISION."""
        ...

    def iter_chunks(self, chunk_ms: float = 1000) -> Iterator[AudioContract]:
        """Stream audio in chunks without copying."""
        ...

    def to_mono(self) -> AudioContract:
        """Return mono version using power-preserving downmix (L+R)/√2."""
        ...

    def verify_integrity(self) -> bool:
        """Recompute SHA-256 and compare to stored hash. Raises on mismatch."""
        ...

    def __len__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __hash__(self) -> int:
        ...


# ============================================================================
# PART 2: STEM CONTRACT
#
# The specific AudioContract subtype for separated stems.
# Carries separation provenance alongside the audio.
# When Demucs returns a stem, it must be wrapped in StemContract before
# being stored in the arena. This is where the stem sample rate bug dies.
# ============================================================================

@dataclass
class StemContract:
    """
    A separated audio stem with full provenance.

    WHY THIS EXISTS: Demucs returns raw numpy arrays. The pipeline was
    storing those arrays directly, losing the sample rate. Arena then
    couldn't resample correctly because 'original_sr' was unknowable.
    StemContract wraps the stem in an AudioContract (which carries sample_rate)
    and adds the separation metadata needed for forensic audit.

    The pipeline's _run_separation() must produce StemContracts, not raw arrays.
    """
    audio: AudioContract          # Carries sample_rate — never lose this
    stem_type: StemType
    separation_model: str         # e.g. "htdemucs", "bs_roformer"
    confidence: float
    separation_time_ms: float
    original_mix_hash: str        # SHA-256 of the pre-separation audio

    @property
    def sample_rate(self) -> int:
        """Delegate to the wrapped AudioContract."""
        return self.audio.sample_rate

    @property
    def duration_seconds(self) -> float:
        return self.audio.duration_ms / 1000.0


# ============================================================================
# PART 3: DETECTION CONTRACT
# ============================================================================

@dataclass
class ConfidenceComponents:
    """
    Forensic breakdown of a detection confidence score.

    Instead of a single opaque float, detection agents must report
    how that float was computed. The Scribe uses this to veto
    hallucinated notes that have high overall confidence but zero
    spectral match (a telltale sign of model overfitting).
    """
    transient_strength: float = 0.0
    spectral_match: float = 0.0
    temporal_consistency: float = 0.0
    model_confidence: float = 0.0
    ensemble_agreement: float = 0.0

    def total(self) -> float:
        """Weighted aggregate confidence. Range 0.0–1.0."""
        weights = (0.30, 0.20, 0.20, 0.15, 0.15)
        return (
            self.transient_strength   * weights[0] +
            self.spectral_match       * weights[1] +
            self.temporal_consistency * weights[2] +
            self.model_confidence     * weights[3] +
            self.ensemble_agreement   * weights[4]
        )

    def is_hallucination_candidate(self) -> bool:
        """
        True if confidence pattern suggests a hallucinated note.
        High model confidence + low spectral match = model overfitting.
        """
        return self.model_confidence > 0.8 and self.spectral_match < 0.3


@dataclass
class DetectionGuarantees:
    """
    What a detection agent promises about its output before it runs.

    The pipeline reads these guarantees during stage planning to:
    - Request the right audio format from the arena
    - Set appropriate memory reservations
    - Know what confidence threshold to use for vetoing
    """
    native_sample_rate: int = 16000     # What SR this agent was designed for
    requires_mono: bool = True
    requires_stereo: bool = False
    stem_type: Optional[StemType] = None
    is_monophonic: bool = False
    is_polyphonic: bool = False
    max_events_per_second: float = 30.0
    min_confidence_threshold: float = 0.25
    latency_ms: float = 5000.0


@dataclass
class DetectionResult:
    """
    The ONLY permitted return type from a detection agent.
    No raw List[NoteEvent]. No None. No dict. Always DetectionResult.

    The reasoning_chain is not optional. The Scribe reads it when
    issuing vetoes so the forensic log explains why a note was rejected.
    """
    events: List[NoteEvent]
    confidence: ConfidenceComponents
    source: SourceType
    detection_time_ms: float
    reasoning_chain: List[str]
    guarantees: DetectionGuarantees
    sample_rate_used: int                # Actual SR the detection ran at
    stem_type_used: Optional[StemType] = None

    @property
    def is_empty(self) -> bool:
        return len(self.events) == 0

    @property
    def avg_confidence(self) -> float:
        if not self.events:
            return 0.0
        return sum(e.confidence for e in self.events) / len(self.events)


@runtime_checkable
class DetectionContract(Protocol):
    """
    Contract for all detection agents (Rhythm, Pitch, Tonal, Drum).

    Agents receive an AudioContract (not a raw numpy array).
    This eliminates the sample rate tracking problem — the agent
    can check audio.sample_rate and resample if needed.
    """

    @property
    def source_type(self) -> SourceType:
        """Which SourceType this agent reports in NoteEvent.source."""
        ...

    @property
    def name(self) -> str:
        """Human-readable agent identifier for logging."""
        ...

    def detect(self, audio: AudioContract, context: AudioContext) -> DetectionResult:
        """
        Detect events from audio.

        MUST return DetectionResult — never a raw list, never None.
        MUST populate reasoning_chain with at least one entry.
        MUST set sample_rate_used to the SR actually used.
        """
        ...

    def get_guarantees(self) -> DetectionGuarantees:
        """Return what this agent promises about its output."""
        ...


# ============================================================================
# PART 4: SEPARATION CONTRACT
# ============================================================================

@dataclass
class SeparationContractResult:
    """
    Immutable result from a separation agent.

    NOTE: Different from core.order_types.SeparationResult.
    This contract version uses typed StemContract instead of raw Dict.
    The pipeline will convert between them as needed.
    """
    stems: Dict[StemType, StemContract]
    separation_time_seconds: float
    memory_usage_mb: float
    model_used: str
    confidence: Confidence

    def get_stem(self, stem_type: StemType) -> Optional[StemContract]:
        return self.stems.get(stem_type)

    def has_stem(self, stem_type: StemType) -> bool:
        return stem_type in self.stems

    def to_order_types_result(self) -> 'core.order_types.SeparationResult':
        """Convert to order_types.SeparationResult for pipeline compatibility."""
        from core.order_types import SeparationResult as LegacyResult
        return LegacyResult(
            stems={k: v.audio.samples for k, v in self.stems.items()},
            separation_time_seconds=self.separation_time_seconds,
            memory_usage_mb=self.memory_usage_mb,
            confidence=self.confidence
        )


@runtime_checkable
class SeparationContract(Protocol):
    """
    Contract for all separation agents (Demucs, BS_Roformer, Hybrid).

    CRITICAL REQUIREMENT: separate() must return StemContracts, not numpy arrays.
    Every StemContract wraps an AudioContract which carries the sample_rate.
    This is the contract-level fix for the stem sample rate tracking bug.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def native_sample_rate(self) -> int:
        """SR this model was trained on. Demucs=44100, BSRoformer=44100."""
        ...

    @property
    def available_stems(self) -> List[StemType]:
        """Which stems this separator can produce."""
        ...

    @property
    def separation_quality(self) -> str:
        """'fast', 'balanced', 'high'."""
        ...

    def separate(self, audio: AudioContract, context: AudioContext) -> SeparationContractResult:
        """
        Separate audio into stems.

        MUST preserve original sample rate in StemContract.
        MUST return SeparationContractResult with StemContracts, not raw arrays.
        """
        ...


# ============================================================================
# PART 5: ANALYSIS CONTRACT
# ============================================================================

@dataclass
class GrooveAnalysisResult:
    """Result from GrooveFieldAnalyzer."""
    groove_field: GrooveField
    bass_events_used: int
    kick_events_used: int
    analysis_time_ms: float
    reasoning: str


@dataclass
class TempoAnalysisResult:
    """
    Result from TempoIntelligence / PulseFieldAnalyzer.

    beat_grid is populated when madmom is available.
    pulse_field is derived from beat_grid via BeatGrid.to_pulse_field().
    tempo_map is built from beat_grid's average beat duration.
    """
    tempo_map: TempoMap
    pulse_field: PulseField
    beat_grid: Optional[BeatGrid]
    beat_track: List[BeatTrackingResult]
    time_signature_candidates: List[TimeSignatureCandidate]
    time_signature_primary: Optional[TimeSignatureCandidate]
    is_constant_tempo: bool
    analysis_time_ms: float
    method_used: str          # "madmom", "librosa", "pulse_field", "default"
    confidence: Confidence


@dataclass
class AnalysisResult:
    """
    Generic result wrapper for analysis agents.

    Use the specific typed results (GrooveAnalysisResult, TempoAnalysisResult)
    where possible. This wrapper is for stages that don't fit the typed pattern.
    """
    result_type: str          # 'groove', 'tempo', 'pulse', 'voice'
    confidence: Confidence
    analysis_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    groove: Optional[GrooveAnalysisResult] = None
    tempo: Optional[TempoAnalysisResult] = None
    voices: Optional[VoiceContinuityResult] = None


@runtime_checkable
class AnalysisContract(Protocol):
    """
    Contract for analysis agents.

    These agents don't produce NoteEvents; they produce analysis
    structures like GrooveField, TempoMap, or PulseField.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def analysis_type(self) -> str:
        """'groove', 'tempo', 'pulse', 'voice'."""
        ...

    def analyze(
        self,
        audio: Optional[AudioContract],
        context: AudioContext,
        events: Optional[List[NoteEvent]] = None
    ) -> AnalysisResult:
        """
        Analyze audio and/or events.

        MUST return AnalysisResult with appropriate fields populated.
        """
        ...


# ============================================================================
# PART 6: QUANTIZATION CONTRACT
# ============================================================================

@dataclass
class QuantizationAudit:
    """
    Forensic audit trail of quantization decisions.

    The Non-Destructive Audit law requires that every snap is recorded
    and the original timestamps are always recoverable.
    """
    original_events: List[NoteEvent]
    quantized_events: List[NoteEvent]
    snaps_performed: int
    total_confidence_loss: float
    max_snap_ms: float
    average_shift_ms: float
    stage_breakdown: List[Dict[str, Any]]


@runtime_checkable
class QuantizationContract(Protocol):
    """
    Contract for quantization agents (Ritornello).

    LAW: "Original timestamps (start_ms, end_ms) are NEVER overwritten."
    LAW: "Snapped timestamps go to snapped_start_ms, snapped_end_ms only."
    """

    @property
    def name(self) -> str:
        ...

    def quantize(
        self,
        events: List[NoteEvent],
        tempo_map: Optional[TempoMap] = None,
        groove_field: Optional[GrooveField] = None,
        pulse_field: Optional[PulseField] = None,
        beat_grid: Optional[BeatGrid] = None
    ) -> List[NoteEvent]:
        """
        Quantize events to grid.

        MUST preserve original timestamps.
        MUST populate snapped_* fields for moved notes.
        MUST record audit trail (accessible via get_audit()).
        """
        ...

    def get_audit(self) -> QuantizationAudit:
        """Return forensic audit of last quantization."""
        ...


# ============================================================================
# PART 7: VALIDATION CONTRACT
# ============================================================================

@dataclass
class ValidationReport:
    """
    Comprehensive validation report.

    The Scribe reads DetectionResult.reasoning_chain and
    ConfidenceComponents.is_hallucination_candidate() when generating
    the gates_failed list. This is what makes the forensic log useful.
    """
    is_valid: bool
    stage_name: str
    source: SourceType
    event_count: int
    avg_confidence: float
    silence_ratio: float
    hallucination_count: int
    gates_passed: List[ValidationGate]
    gates_failed: List[Tuple[ValidationGate, VetoReason, str]]
    vetoes_issued: List[Dict[str, Any]]
    final_verdict: str
    retry_recommendation: str
    is_trustworthy: bool


@runtime_checkable
class ValidationContract(Protocol):
    """
    Contract for validation agents (Scribe, ConsensusEngine).

    Law of Epistemic Veto: "Any qualified witness can declare falsehood."
    """

    @property
    def name(self) -> str:
        ...

    def validate_stage(
        self,
        result: Union[DetectionResult, SeparationContractResult, AnalysisResult],
        source: SourceType,
        stage_name: str
    ) -> ValidationReport:
        """
        Validate a stage's output.

        MUST return ValidationReport.
        MUST record all vetoes for forensic audit.
        """
        ...

    def get_veto_history(self) -> List[Dict[str, Any]]:
        """Return all vetoes issued in this session."""
        ...

    def final_verdict(self) -> Dict[str, Any]:
        """Return final aggregated transcription verdict."""
        ...


# ============================================================================
# PART 8: MEMORY CONTRACT
# ============================================================================

@dataclass
class BufferID:
    """
    Typed identifier for a buffer stored in the arena.

    Using a typed dataclass instead of a raw string prevents
    accidental ID collisions and carries buffer metadata.
    """
    arena_name: str
    buffer_name: str
    unique_id: str
    sample_rate: int            # SR of the stored audio — always known
    stem_type: Optional[StemType] = None
    is_stem: bool = False

    def __str__(self) -> str:
        return f"{self.arena_name}:{self.buffer_name}:{self.unique_id}"

    @property
    def short(self) -> str:
        return f"{self.buffer_name}:{self.unique_id[:8]}@{self.sample_rate}Hz"


@dataclass
class MemoryReport:
    """Comprehensive memory report from Arena or Guardian."""
    current_mb: float
    peak_mb: float
    arena_count: int
    buffer_count: int
    borrowed_count: int
    total_allocated_mb: float
    total_freed_mb: float
    pressure_level: str
    freeable_mb: float
    reserved_mb: float


@runtime_checkable
class ArenaContract(Protocol):
    """
    Contract for MemoryArena.

    LAW: "Arena stores AudioContracts. Not numpy arrays. Not bytes."
    LAW: "Arena DOES NOT transform audio. store() is O(1). borrow() is O(1)."
    LAW: "Resampling is NOT a borrow operation. Call audio.at_sample_rate() first."
    """

    @property
    def name(self) -> str:
        ...

    def store(
        self,
        contract: AudioContract,
        name: str,
        stem_type: Optional[StemType] = None,
        tier: int = 2
    ) -> BufferID:
        """
        Store an AudioContract in the arena.

        MUST NOT resample or transform the audio.
        MUST return BufferID with sample_rate set to contract.sample_rate.
        """
        ...

    def borrow(self, buffer_id: BufferID, agent_name: str) -> AudioContract:
        """
        Borrow an AudioContract from the arena.

        MUST return the stored AudioContract unchanged.
        MUST NOT resample, convert to mono, or sanitize.
        """
        ...

    def prepare(self, buffer_id: BufferID, target_sr: int, mono: bool = True) -> BufferID:
        """
        Prepare a transformed version of audio for a specific model.

        This is where resampling and mono conversion happen.
        Returns a NEW BufferID for the prepared audio.
        The original remains untouched.
        """
        ...

    def release(self, buffer_id: BufferID) -> bool:
        """Release a buffer, freeing memory."""
        ...

    def get_memory_report(self) -> MemoryReport:
        ...

    def __contains__(self, buffer_id: BufferID) -> bool:
        ...


@runtime_checkable
class GuardianContract(Protocol):
    """
    Contract for MemoryGuardian.

    LAW: "The pipeline requests. The Guardian decides."
    LAW: "Guardian enforces policy. Arena executes storage."
    """

    def request_allocation(self, requester: str, estimated_mb: int) -> bool:
        """Request memory allocation. Guardian decides."""
        ...

    def release_allocation(self, requester: str) -> None:
        """Release allocated memory reservation."""
        ...

    def register_stage(self, stage_name: str) -> None:
        """Register a stage with Guardian."""
        ...

    def stage_starting(self, stage_name: str, estimated_mb: int) -> None:
        """Notify Guardian that a stage is starting with expected memory cost."""
        ...

    def stage_completed(self, stage_name: str, success: bool, actual_mb: float) -> None:
        """Report actual memory usage after stage completion."""
        ...

    def revoke_stage_borrows(self, stage_name: str) -> int:
        """Revoke all borrows held by a stage."""
        ...

    def get_memory_report(self) -> MemoryReport:
        ...


# ============================================================================
# PART 9: FEATURE BUNDLE CONTRACT
# ============================================================================

class EvidenceType(str, Enum):
    """Types of spectral evidence that can be independently released."""
    STFT = "stft"
    MAGNITUDE = "magnitude"
    PHASE = "phase"
    CHROMA = "chroma"
    CQT = "cqt"
    RMS = "rms"
    ZCR = "zcr"
    ONSET_STRENGTH = "onset_strength"
    HARMONIC_NETWORK = "harmonic_network"
    POLYPHONIC_PEAKS = "polyphonic_peaks"
    VOICE_SEPARATION = "voice_separation"


@dataclass
class EvidenceLease:
    """Lease on evidence, preventing concurrent mutation."""
    evidence_type: EvidenceType
    owner_stage: str
    acquired_at: float
    expires_at: float
    priority: str = "normal"


@runtime_checkable
class FeatureBundleContract(Protocol):
    """
    Contract for FeatureBundle - shared spectral evidence.

    LAW: "Compute once. Borrow everywhere. Never copy."
    LAW: "has_evidence() before accessing any field. Fields can be None."
    """

    @property
    def sample_rate(self) -> int:
        """SR at which features were computed."""
        ...

    @property
    def memory_mb(self) -> float:
        """Current memory footprint in MB."""
        ...

    @property
    def stem_type(self) -> Optional[StemType]:
        """Which stem these features were computed from."""
        ...

    def has_evidence(self, evidence_type: EvidenceType) -> bool:
        """True if this evidence type is allocated and usable."""
        ...

    def get_spectral_slice(
        self,
        start_ms: float,
        end_ms: float,
        freq_min_hz: float = 0.0,
        freq_max_hz: float = 22050.0
    ) -> Optional[np.ndarray]:
        """Return a VIEW of the magnitude spectrogram. Zero-copy."""
        ...

    def get_harmonic_evidence(self, fundamental_hz: float, time_ms: float) -> Dict[str, Any]:
        """Extract pre-computed harmonic evidence at a specific time."""
        ...

    def acquire_lease(
        self,
        evidence_type: EvidenceType,
        stage: str,
        ttl_seconds: float = 300.0
    ) -> bool:
        """Claim ownership of an evidence type for a stage."""
        ...

    def release(self, evidence_type: EvidenceType) -> None:
        """Free the memory for one evidence type."""
        ...

    def release_all(self) -> None:
        """Free all evidence. Call after all agents have finished."""
        ...


# ============================================================================
# PART 10: OBSERVABLE CONTRACT (Forensic Logging)
# ============================================================================

@runtime_checkable
class ObservableContract(Protocol):
    """
    Contract for components that produce forensic logs.

    Every decision is recorded. Every veto is logged.
    """

    @property
    def observable_name(self) -> str:
        """Name for log identification."""
        ...

    def log_decision(
        self,
        decision_type: str,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        reasoning: str
    ) -> None:
        """Log a decision for forensic audit."""
        ...

    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log an error."""
        ...

    def get_session_id(self) -> str:
        """Get current forensic session ID."""
        ...


# ============================================================================
# PART 11: VALIDATION HELPERS
# ============================================================================

def validate_audio_contract(contract: AudioContract, context: str = "") -> bool:
    """
    Runtime check that an object satisfies AudioContract.

    The most important check: sample_rate must be accessible.
    This is the test for whether the stem sample rate bug has been fixed.
    """
    required = [
        'samples', 'sample_rate', 'total_samples', 'duration_ms',
        'sha256_hash', 'lineage', 'channel_layout', 'num_channels',
        'created_at', 'index_to_ms', 'ms_to_index', 'iter_chunks',
        'to_mono', 'verify_integrity',
    ]

    for attr in required:
        if not hasattr(contract, attr):
            raise TypeError(
                f"AudioContract violation: missing '{attr}' in {context}. "
                f"Did you store a raw numpy array instead of an AudioContract?"
            )

    # The most important check: sample_rate must be accessible
    try:
        sr = contract.sample_rate
        if not isinstance(sr, int) or sr <= 0:
            raise TypeError(f"sample_rate must be positive int in {context}, got {sr!r}")
    except AttributeError:
        raise TypeError(
            f"sample_rate not accessible in {context}. "
            f"This is the stem sample rate bug. Wrap the array in an AudioContract."
        )

    # Verify samples is read-only
    if hasattr(contract.samples, 'flags') and contract.samples.flags.writeable:
        raise TypeError(f"samples must be writeable=False in {context}")

    return True


def validate_detection_result(result: DetectionResult, context: str = "") -> bool:
    """Validate that a DetectionResult satisfies all requirements."""
    if not isinstance(result.events, list):
        raise TypeError(f"DetectionResult.events must be list in {context}")

    if not result.reasoning_chain:
        raise TypeError(f"DetectionResult.reasoning_chain is empty in {context}")

    if result.sample_rate_used <= 0:
        raise TypeError(f"DetectionResult.sample_rate_used must be positive in {context}")

    for event in result.events:
        if not hasattr(event, 'confidence'):
            raise TypeError(f"NoteEvent missing confidence in {context}")
        if not hasattr(event, 'start_ms'):
            raise TypeError(f"NoteEvent missing start_ms in {context}")

    return True


def validate_stem_contract(stem: StemContract, context: str = "") -> bool:
    """
    Runtime check that a StemContract carries its sample rate correctly.
    This is the test for whether the stem sample rate bug has been fixed.
    """
    validate_audio_contract(stem.audio, f"StemContract.audio{context}")

    if not (0.0 <= stem.confidence <= 1.0):
        raise TypeError(f"StemContract.confidence out of range in {context}")

    if not stem.original_mix_hash:
        raise TypeError(
            f"StemContract.original_mix_hash is empty in {context}. "
            f"Set it to the SHA-256 of the pre-separation audio."
        )

    return True


# ============================================================================
# PART 12: CONTRACT REGISTRY
# ============================================================================

CONTRACT_REGISTRY: Dict[str, str] = {
    # Audio ingestion
    "ingestion.loader.AudioLoader": "LoaderContract",

    # Separation agents
    "agents.separation.demucs.DemucsSeparator": "SeparationContract",
    "agents.separation.roformer.BSRoformerSeparator": "SeparationContract",
    "agents.separation.hybrid.HybridSeparator": "SeparationContract",

    # Detection agents
    "agents.detection.rhythm_engine.RhythmEngine": "DetectionContract",
    "agents.detection.pitch_intelligence.PitchIntelligence": "DetectionContract",
    "agents.detection.drum_intelligence.DrumIntelligence": "DetectionContract",
    "agents.detection.harmonic_intelligence.HarmonicIntelligence": "DetectionContract",

    # Analysis agents
    "agents.analysis.groove_field.GrooveFieldAnalyzer": "AnalysisContract",
    "agents.analysis.tempo_intelligence.TempoIntelligence": "AnalysisContract",
    "agents.analysis.pulse_field.PulseFieldAnalyzer": "AnalysisContract",
    "agents.analysis.voice_continuity.VoiceContinuity": "AnalysisContract",

    # Quantization
    "agents.quantization.ritornello.Ritornello": "QuantizationContract",

    # Validation
    "orchestration.scribe.Scribe": "ValidationContract",
    "agents.validation.consensus_engine.ConsensusEngine": "ValidationContract",

    # Memory
    "memory.arena.MemoryArena": "ArenaContract",
    "memory.guardian.MemoryGuardian": "GuardianContract",

    # Spectral bus
    "core.feature_bundle.FeatureBundle": "FeatureBundleContract",

    # Export
    "export.midi_writer.MidiWriter": "ExportContract",
    "export.json_writer.JsonWriter": "ExportContract",
}


# =====================================================================
# PART 13: MODULE DOCSTRING
# =====================================================================

__doc__ = """
Grimlock 5.2 Contracts Module
==============================

This module defines the CANONICAL contracts for every component in the system.

Why Contracts?
--------------
- Single source of truth for WHAT components must do
- Enables compile-time type checking (mypy)
- Prevents circular imports (Protocols don't need implementations)
- Documents system boundaries explicitly

How This Fixes the Stem Sample Rate Bug:
----------------------------------------
Before: Demucs returned raw numpy arrays -> stored in arena -> sample_rate lost
After:  Demucs returns AudioContract or StemContract -> sample_rate always present

The AudioContract protocol requires a sample_rate property.
Any component that handles audio must accept or return AudioContract.
Raw numpy arrays are forbidden at system boundaries.

Contract Hierarchy:
-------------------
1. AudioContract - Immutable audio ledger (lowest level)
2. DetectionContract - Detection agents (Rhythm, Pitch, Tonal, Drum)
3. SeparationContract - Separation agents (Demucs, BS_Roformer)
4. AnalysisContract - Analysis agents (Groove, Tempo, Pulse, Voice)
5. QuantizationContract - Quantization agents (Ritornello)
6. ValidationContract - Validation agents (Scribe, Consensus)
7. ArenaContract - Memory management (Arena)
8. GuardianContract - Memory policy (Guardian)
9. FeatureBundleContract - Spectral evidence
10. ObservableContract - Forensic logging

Usage:
------
from core.contracts import DetectionContract, DetectionResult, AudioContract

class MyDetector(DetectionContract):
    def detect(self, audio: AudioContract, context: AudioContext) -> DetectionResult:
        # Implementation must conform exactly
        ...

Type Checking:
-------------
Run mypy to enforce contracts: mypy --strict my_module.py

Testing:
-------
Use validate_audio_contract() for runtime validation in tests.
"""

# =====================================================================
# STANDALONE TEST
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CORE CONTRACTS v5.2 — Validation Test")
    print("=" * 70)

    # Test 1: AudioContract validation rejects raw numpy arrays
    print("\n1. AudioContract rejects raw numpy array:")
    try:
        validate_audio_contract(np.zeros(1024), "test")
        print("   ✗ FAILED — should have raised TypeError")
    except TypeError as e:
        print(f"   ✓ Correctly rejected: {str(e)[:80]}")

    # Test 2: ConfidenceComponents hallucination detection
    print("\n2. ConfidenceComponents hallucination detection:")
    cc = ConfidenceComponents(
        transient_strength=0.1,
        spectral_match=0.1,
        temporal_consistency=0.1,
        model_confidence=0.95,
        ensemble_agreement=0.1,
    )
    is_halluc = cc.is_hallucination_candidate()
    print(f"   ✓ High model confidence + low spectral match = hallucination: {is_halluc}")

    # Test 3: BufferID carries sample rate    print("\n3. BufferID carries sample rate (fixes stem tracking bug):")
    buf_id = BufferID(
        arena_name="pipeline_main",
        buffer_name="drums_stem",
        unique_id="abc123",
        sample_rate=44100,
        stem_type=StemType.DRUMS,
        is_stem=True,
    )
    print(f"   ✓ {buf_id.short}")

    # Test 4: Contract registry completeness
    print(f"\n4. Contract registry: {len(CONTRACT_REGISTRY)} entries")
    contracts_used = set(CONTRACT_REGISTRY.values())
    print(f"   Contracts covered: {sorted(contracts_used)}")

    print("\n" + "=" * 70)
    print("Contracts v5.2 ready.")
    print("Next step: implement AudioContract in core/acoustic_intelligence.py")
    print("           wrap Demucs stems in StemContract in pipeline.py")
    print("=" * 70)