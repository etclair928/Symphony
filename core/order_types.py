# =================================================================
# MODULE: core/order_types.py
# DESCRIPTION: Root type definitions for Grimlock 5.6.1.
#
# VERSION: 5.6.1 (Added Quaver + Phrase testimony types)
# UPDATED: 2026-06-02
#
# CHANGES IN 5.6.1:
#   1. Added DurationHypothesis and WitnessContribution
#   2. Added QuaverTestimony and QuaverConfig
#   3. Added StructuralInvariantVector, StructuralWindow, PhaseTransition
#   4. Added StructuralTestimony
#   5. Added StructuralPhase enum
#   6. Added AttractorBasin enum
#   7. Updated NoteEvent with midi_channel for drum separation
#   8. Added phrase_id to NoteEvent for voice continuity
#
# RULES FOR THIS FILE:
#   - NO numpy. NO librosa. NO torch. Standard library ONLY.
#   - Every agent, pipeline stage, and exporter imports from here.
#   - Required dataclass fields MUST come before optional fields.
#   - Methods on dataclasses are allowed; business logic is not.
#   - Add carefully. Never remove without a migration plan.
#   - ALL TIME VALUES ARE IN MILLISECONDS (float)
#   - ALL FREQUENCY VALUES ARE IN HZ OR BPM (explicit suffix required)
#   - ALL PITCH VALUES ARE MIDI 0-127
#
# INVARIANTS:
#   - Non-Destructive Audit: start_ms/end_ms NEVER change
#   - snapped_* fields are derived-only optional overlays
#   - Confidence: Use .value_f for numeric, never treat enum as float
#   - All lists assumed sorted where noted (debug validation available)
#   - AudioContext: file_hash REQUIRED for stable identity
#   - AudioContext: NO inference, NO guessing, NO silent fallbacks
#
# Laws encoded here:
#   - Non-Destructive Audit: start_ms is never overwritten.
#     snapped_start_ms shadows it. The original is always there.
#   - Confidence must be earned: Confidence.HALLUCINATION = 0.0.
#     A float below 0.25 is not a LOW confidence — it is nothing.
#   - Relational Physics: GrooveField measures note-to-note distance,
#     not note-to-grid distance.
#   - Truth Anchor: AudioContext is deterministic identity, not container.
# =================================================================

import bisect
import math
import gc
import time
import hashlib
import warnings
from enum import Enum, auto
from dataclasses import dataclass, field, replace
from typing import Optional, List, Any, Dict, Tuple, Union
from datetime import datetime
from copy import deepcopy

# ============================================================================
# CONSTANTS FOR VALIDATION
# ============================================================================

MIN_VALID_TEMPO_BPM: float = 20.0
MAX_VALID_TEMPO_BPM: float = 300.0
MIN_VALID_DURATION_MS: float = 1.0
SORT_TOLERANCE_MS: float = 0.001
UNKNOWN_SAMPLE_RATE: int = -1
UNKNOWN_HASH_PLACEHOLDER: str = "0000000000000000000000000000000000000000000000000000000000000000"


def _is_sorted_ascending(xs: List[float], tolerance: float = SORT_TOLERANCE_MS) -> bool:
    """Check if list is sorted in ascending order within tolerance."""
    if len(xs) < 2:
        return True
    return all(xs[i] <= xs[i + 1] + tolerance for i in range(len(xs) - 1))


def _warn_unsorted(name: str, list_name: str) -> None:
    """Emit warning for unsorted list."""
    warnings.warn(
        f"{name}: {list_name} is not sorted. This may cause incorrect behavior.",
        UserWarning,
        stacklevel=3
    )


# ============================================================================
# PART 1: FOUNDATION ENUMS
# ============================================================================

class Confidence(Enum):
    """
    Earned confidence levels. Not assumed.

    LAW: Does NOT inherit from float to prevent arithmetic abuse.
    Use .value_f for numeric operations.

    ENUM TAXONOMY: Validation system
    """
    HALLUCINATION = 0.0
    LOW = 0.25
    MEDIUM = 0.50
    HIGH = 0.80
    PERFECT = 1.0

    @property
    def value_f(self) -> float:
        """Numeric value as float. Use this for arithmetic."""
        return self._value_

    def __ge__(self, other: Union[float, 'Confidence']) -> bool:
        """Compare with float or other Confidence."""
        if isinstance(other, Confidence):
            other = other.value_f
        return self.value_f >= other

    def __le__(self, other: Union[float, 'Confidence']) -> bool:
        if isinstance(other, Confidence):
            other = other.value_f
        return self.value_f <= other

    def __gt__(self, other: Union[float, 'Confidence']) -> bool:
        if isinstance(other, Confidence):
            other = other.value_f
        return self.value_f > other

    def __lt__(self, other: Union[float, 'Confidence']) -> bool:
        if isinstance(other, Confidence):
            other = other.value_f
        return self.value_f < other

    def __float__(self) -> float:
        """Deprecated: Use .value_f. Kept for backward compatibility."""
        warnings.warn(
            "Converting Confidence to float is deprecated. Use .value_f",
            DeprecationWarning,
            stacklevel=2
        )
        return self.value_f

    @classmethod
    def from_float(cls, value: float) -> 'Confidence':
        """Convert float to Confidence enum."""
        if value < 0.10:
            return cls.HALLUCINATION
        elif value < 0.40:
            return cls.LOW
        elif value < 0.70:
            return cls.MEDIUM
        elif value < 0.95:
            return cls.HIGH
        return cls.PERFECT

    def as_float(self) -> float:
        """Alias for value_f."""
        return self.value_f


class DecisionType(str, Enum):
    """Forensic decision types for MusicBox logging. Append-only."""
    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Stage lifecycle
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    STAGE_INPUT = "stage_input"
    STAGE_OUTPUT = "stage_output"

    # Agent testimony
    SEPARATION_TESTIMONY = "separation_testimony"
    PITCH_TESTIMONY = "pitch_testimony"
    RHYTHM_TESTIMONY = "rhythm_testimony"
    TONAL_TESTIMONY = "tonal_testimony"
    DRUM_TESTIMONY = "drum_testimony"
    QUAVER_TESTIMONY = "quaver_testimony"  # NEW in 5.6.1
    PHRASE_TESTIMONY = "phrase_testimony"   # NEW in 5.6.1

    # Analysis findings - a generic bucket for a stage's own computed
    # evidence (confidence breakdowns, per-note costs, witness votes,
    # etc.) that isn't itself a pass/fail veto or a full agent testimony,
    # logged purely so it's inspectable later even when nothing
    # downstream actually consumes it.
    ANALYSIS_EVIDENCE = "analysis_evidence"
    # A specific contention (two witnesses/hypotheses disagreeing about
    # the same thing) being arbitrated to a single answer - distinct from
    # ANALYSIS_EVIDENCE since it's a genuine decision, not just a finding.
    CONTENTION_RESOLVED = "contention_resolved"

    # Model management
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"

    # Validation
    VETO_TRIGGERED = "veto_triggered"
    CONSENSUS_REACHED = "consensus_reached"
    CONSENSUS_DECISION = "consensus_decision"
    VALIDATION = "validation"

    # Processing
    QUANTIZATION_APPLIED = "quantization_applied"
    EXPORT_COMPLETED = "export_completed"

    # Errors and warnings
    ERROR_OCCURRED = "error_occurred"
    MEMORY_PRESSURE = "memory_pressure"
    GC_TRIGGERED = "gc_triggered"

    # Memory management events
    REGISTER = "register"
    RELEASE = "release"
    MEMORY_ALLOC = "memory_alloc"
    MEMORY_RELEASE = "memory_release"
    MEMORY_EVICT = "memory_evict"
    ARENA_ALLOCATION = "arena_allocation"
    ARENA_RELEASE = "arena_release"
    ARENA_EXIT = "arena_exit"
    ARENA_EXCEPTION = "arena_exception"
    BUFFER_BORROW = "buffer_borrow"
    BUFFER_RELEASE = "buffer_release"

    # Pool events
    POOL_HIT = "pool_hit"
    POOL_MISS = "pool_miss"
    POOL_EVICT = "pool_evict"

    # Transfer events
    TRANSFER = "transfer"
    TRANSFER_BLOCKED = "transfer_blocked"

    # Pin events
    PIN = "pin"
    UNPIN = "unpin"

    # Arena cleanup
    ARENA_FREE = "arena_free"
    ARENA_FREE_PINNED = "arena_free_pinned"
    ARENA_KEEP = "arena_keep"

    # Borrow management
    BORROW_REVOKE = "borrow_revoke"
    BORROW_REVOKE_EXPIRED = "borrow_revoke_expired"
    RELEASE_BLOCKED = "release_blocked"

    # GC events
    FORCED_GC = "forced_gc"
    STAGGERED_GC = "staggered_gc"
    EMERGENCY_CLEANUP = "emergency_cleanup"

    # Load events
    LOAD_COMPLETE = "load_complete"
    LOAD_START = "load_start"


class VetoReason(str, Enum):
    """Why the Scribe rejected a stage's output. Append-only."""
    EXCESS_SILENCE = "excess_silence"
    HALLUCINATED_NOTE = "hallucinated_note"
    MISSING_MANDATORY_FIELD = "missing_mandatory_field"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    EMPTY_RESULT = "empty_result"
    HARMONIC_SERIES_FAILURE = "harmonic_series_failure"
    PHASE_INCOHERENCE = "phase_incoherence"
    TEMPO_OUTLIER = "tempo_outlier"
    NO_HARMONIC_STRUCTURE = "no_harmonic_structure"
    EXCESS_NOISE = "excess_noise"
    STRUCTURAL_INCONSISTENCY = "structural_inconsistency"  # NEW in 5.6.1


class ValidationGate(str, Enum):
    """Which validation check the Scribe applied."""
    SCHOENBERG_MIRROR = "schoenberg_mirror"
    HARMONIC_SERIES = "harmonic_series"
    SILENCE_DETECTOR = "silence_detector"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    PHASE_CONSISTENCY = "phase_consistency"
    TEMPO_REASONABLENESS = "tempo_reasonableness"
    STRUCTURAL_PHASE = "structural_phase"  # NEW in 5.6.1


class SourceType(str, Enum):
    """
    Which agent generated an event.

    ENUM TAXONOMY: Source identification for forensic traceability.
    """
    # Detection
    RHYTHM = "rhythm_engine"
    PITCH = "pitch_engine"
    TONAL = "tonal_detector"
    PERCUSSION = "percussion_intelligence"
    DRUM_INTELLIGENCE = "drum_intelligence"
    BASS_STEM = "bass_stem"
    BASS = "bass_engine"
    MASTER_STEM = "master_stem"
    OTHER_STEM = "other_stem"

    # Separation
    DEMUCS = "demucs_separator"
    BS_ROFORMER = "bs_roformer"      # DEPRECATED in 5.0, removed in 5.1
    MEL_ROFORMER = "mel_roformer"    # NEW in 5.0 - replacement for BS-Roformer
    HYBRID_SEPARATOR = "hybrid_separator"

    # Analysis
    GROOVE_FIELD = "groove_field"
    PULSE_FIELD = "pulse_field"
    TEMPO_INTELLIGENCE = "tempo_intelligence"
    VOICE_CONTINUITY = "voice_continuity"

    # Quantization
    RITORNELLO = "ritornello"
    VELOCITY_MERGE = "velocity_merge"
    ANECHOIC_MA = "anechoic_ma"
    QUANTIZATION_STAGES = "quantization_stages"

    # Epistemic Witnesses (NEW in 5.6.1)
    QUAVER = "quaver_intelligence"       # Symbolic duration witness
    PHRASE = "phrase_intelligence"       # Structural phase witness

    # Orchestration / validation
    SCRIBE = "scribe"
    CONSENSUS_ENGINE = "consensus_engine"
    SCHOENBERG_MIRROR = "schoenberg_mirror"
    GUIDED = "guided_mode"
    PIPELINE = "pipeline"

    @property
    def is_deprecated(self) -> bool:
        """Check if this source type is deprecated."""
        return self == SourceType.BS_ROFORMER

    @property
    def replacement(self) -> Optional['SourceType']:
        """Get replacement for deprecated source types."""
        if self == SourceType.BS_ROFORMER:
            return SourceType.MEL_ROFORMER
        return None


class StemType(str, Enum):
    """
    Which audio stem a buffer belongs to.

    ENUM TAXONOMY: Stem identification for source separation.
    """
    FULL_MIX = "full_mix"
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    OTHER = "other"
    PERCUSSION = "percussion"
    RHYTHM_SECTION = "rhythm_section"
    RESIDUAL = "residual"  # NEW in 5.0 - unassigned energy/ambience


class QuantizationStrategy(str, Enum):
    """How notes are snapped to the timing grid. Non-destructive only."""
    RITORNELLO_SAFE = "ritornello_safe"
    GROOVE_AWARE = "groove_aware"
    PULSE_AWARE = "pulse_aware"
    MULTI_STAGE = "multi_stage"


class QuantizationStage(str, Enum):
    """Which pass of multi-stage quantization."""
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"
    MICRO = "micro"
    GROOVE_AWARE = "groove_aware"


class MergeStrategy(str, Enum):
    """How overlapping notes are resolved."""
    MAX_VELOCITY = "max_velocity"
    WEIGHTED_AVERAGE = "weighted_average"
    LONGEST_DURATION = "longest_duration"
    EARLIEST_ONSET = "earliest_onset"


class ConsensusStrategy(str, Enum):
    """How the ConsensusEngine resolves conflicting witness votes."""
    UNANIMOUS_REQUIRED = "unanimous"
    MAJORITY_VOTE = "majority"
    WEIGHTED_BY_CONFIDENCE = "weighted"
    ANY_VETO_WINS = "veto"


class DrumType(str, Enum):
    """Drum/percussion instrument types."""
    KICK = "kick"
    SNARE = "snare"
    HI_HAT_CLOSED = "hi_hat_closed"
    HI_HAT_OPEN = "hi_hat_open"
    CRASH = "crash"
    RIDE = "ride"
    TOM_HIGH = "tom_high"
    TOM_MID = "tom_mid"
    TOM_LOW = "tom_low"
    RIMSHOT = "rimshot"
    COWBELL = "cowbell"
    PERCUSSION = "aux_percussion"


class AnechoicProfile(str, Enum):
    """Room acoustic profile detected by AnechoicMa."""
    STUDIO = "studio"
    LIVE_ROOM = "live_room"
    CHURCH = "church"
    OUTDOOR = "outdoor"
    AUTO_DETECTED = "auto"


class VoiceRole(str, Enum):
    """The melodic role of a traced Voice within polyphonic texture."""
    BASS = "bass"
    TENOR = "tenor"
    ALTO = "alto"
    SOPRANO = "soprano"
    MELODY = "melody"
    INNER_VOICE = "inner_voice"


class SchoenbergVerdict(str, Enum):
    """Harmonic Mirror verdict on whether a detected pitch is real."""
    TONAL = "tonal"
    NOISE = "noise"
    PERCUSSION = "percussion"
    UNCERTAIN = "uncertain"
    HALLUCINATION = "hallucination"


# ============================================================================
# NEW ENUMS FOR 5.6.1
# ============================================================================

class StructuralPhase(str, Enum):
    """
    Discrete structural states — NOT a continuum.
    Used by PhraseIntelligence for state machine inference.
    """
    THEME = "theme"
    DEVELOPMENT = "development"
    TRANSITION = "transition"
    RECAPITULATION = "recapitulation"
    CODA = "coda"
    INTRO = "intro"
    SOLO = "solo"
    LOOP = "loop"
    AMBIGUOUS = "ambiguous"


class AttractorBasin(str, Enum):
    """
    Genre attractors — where the structural trajectory tends to converge.
    This is EMERGENT, not detected.
    """
    POP = "pop"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    AMBIENT = "ambient"
    BINARY = "binary"


class EpistemicRole(str, Enum):
    """What an agent does in the epistemic pipeline."""
    GENERATE_FIELD = "generate_field"   # Derives fields from audio
    TESTIFY = "testify"                 # Produces testimony about hypotheses
    ARBITRATE = "arbitrate"             # Resolves contradictions
    STABILIZE = "stabilize"             # Converges to truth
    INSCRIBE = "inscribe"               # Commits symbolic reality


# ============================================================================
# PART 2: AUDIO CONTEXT - DETERMINISTIC TRUTH ANCHOR (COMPLETELY REBUILT)
# ============================================================================

@dataclass(frozen=True, slots=True)
class AudioContext:
    """
    IMMUTABLE TRUTH ANCHOR for distributed audio reasoning.

    LAW: "AudioContext is NOT a container. It is a deterministic truth anchor."
    LAW: "No inference. No guessing. No silent fallbacks."
    LAW: "Identity is NON-conditional. Hash required or context is unstable."

    CONTRACT:
        - file_hash MUST be present for stable identity
        - num_channels is semantic channel count, NOT tensor shape
        - working_sample_rate is the ACTUAL sample rate after any processing
        - UNKNOWN_SAMPLE_RATE (-1) indicates unknown - use ONLY for error/fallback contexts

    INVARIANTS:
        - file_hash, when present, is normalized (lowercase, no whitespace)
        - duration_seconds > 0 for valid audio
        - sample_rate > 0 or exactly UNKNOWN_SAMPLE_RATE
    """

    # REQUIRED FIELDS (must be provided, no defaults)
    file_path: str
    original_sample_rate: int
    working_sample_rate: int
    duration_seconds: float
    num_channels: int

    # REQUIRED FOR IDENTITY STABILITY
    file_hash: str

    # METADATA (not identity-critical)
    memory_mb: float = 0.0

    def __post_init__(self):
        """STRICT INVARIANT ENFORCEMENT - NO EXCEPTIONS."""
        # FILE PATH VALIDATION
        if not self.file_path:
            raise ValueError("AudioContext: file_path cannot be empty")

        # SAMPLE RATE VALIDATION
        if self.original_sample_rate <= 0 and self.original_sample_rate != UNKNOWN_SAMPLE_RATE:
            raise ValueError(
                f"AudioContext: original_sample_rate must be > 0 or == {UNKNOWN_SAMPLE_RATE}, "
                f"got {self.original_sample_rate}"
            )

        if self.working_sample_rate <= 0 and self.working_sample_rate != UNKNOWN_SAMPLE_RATE:
            raise ValueError(
                f"AudioContext: working_sample_rate must be > 0 or == {UNKNOWN_SAMPLE_RATE}, "
                f"got {self.working_sample_rate}"
            )

        # DURATION VALIDATION
        if self.duration_seconds <= 0:
            raise ValueError(f"AudioContext: duration_seconds must be > 0, got {self.duration_seconds}")

        # CHANNEL VALIDATION
        if self.num_channels <= 0:
            raise ValueError(f"AudioContext: num_channels must be > 0, got {self.num_channels}")

        if self.num_channels > 2:
            warnings.warn(
                f"AudioContext: {self.num_channels}-channel audio detected. "
                f"Grimlock primarily supports mono (1) and stereo (2).",
                UserWarning,
                stacklevel=3
            )

        # HASH VALIDATION (CRITICAL FOR IDENTITY)
        if not self.file_hash:
            raise ValueError(
                "AudioContext: file_hash is required for identity stability. "
                "Use UNKNOWN_HASH_PLACEHOLDER only for truly unstable contexts."
            )

        normalized_hash = self.file_hash.lower().strip()

        if normalized_hash != UNKNOWN_HASH_PLACEHOLDER:
            if len(normalized_hash) != 64:
                raise ValueError(
                    f"AudioContext: file_hash must be 64-character SHA-256 hex string, "
                    f"got length {len(normalized_hash)}"
                )
            try:
                int(normalized_hash, 16)
            except ValueError:
                raise ValueError(f"AudioContext: file_hash must be hexadecimal, got {normalized_hash[:16]}...")

        if normalized_hash != self.file_hash:
            object.__setattr__(self, "file_hash", normalized_hash)

        # MEMORY VALIDATION
        if self.memory_mb < 0:
            raise ValueError(f"AudioContext: memory_mb cannot be negative, got {self.memory_mb}")

    # ========================================================================
    # DERIVED PROPERTIES
    # ========================================================================

    @property
    def is_mono(self) -> bool:
        """True if audio has exactly 1 semantic channel."""
        return self.num_channels == 1

    @property
    def is_stereo(self) -> bool:
        """True if audio has exactly 2 semantic channels."""
        return self.num_channels == 2

    @property
    def is_multichannel(self) -> bool:
        """True if audio has more than 2 semantic channels."""
        return self.num_channels > 2

    @property
    def is_unknown_sample_rate(self) -> bool:
        """True if working sample rate is unknown."""
        return self.working_sample_rate == UNKNOWN_SAMPLE_RATE

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_seconds * 1000.0

    @property
    def has_stable_identity(self) -> bool:
        """True if this context can be used as a truth anchor."""
        return self.file_hash != UNKNOWN_HASH_PLACEHOLDER

    # ========================================================================
    # IDENTITY METHODS (Deterministic, Non-conditional)
    # ========================================================================

    def __eq__(self, other: object) -> bool:
        """Equality is HASH-BASED ONLY."""
        if not isinstance(other, AudioContext):
            return False
        if not self.has_stable_identity or not other.has_stable_identity:
            return False
        return self.file_hash == other.file_hash

    def __hash__(self) -> int:
        """Hash is HASH-BASED ONLY, mirrors equality."""
        if not self.has_stable_identity:
            return hash(UNKNOWN_HASH_PLACEHOLDER)
        return hash(self.file_hash)

    # ========================================================================
    # FACTORY METHODS (Explicit, No Guessing)
    # ========================================================================

    @classmethod
    def from_loaded_audio(cls, loaded_audio: Any, file_path: str) -> 'AudioContext':
        """Create from LoadedAudio object. REQUIRES stable hash."""
        if not hasattr(loaded_audio, 'sha256_hash') or not loaded_audio.sha256_hash:
            raise ValueError(
                "AudioContext.from_loaded_audio: loaded_audio missing sha256_hash. "
                "Cannot establish stable identity."
            )

        return cls(
            file_path=file_path,
            original_sample_rate=loaded_audio.sample_rate,
            working_sample_rate=loaded_audio.sample_rate,
            duration_seconds=loaded_audio.duration_seconds,
            num_channels=loaded_audio.info.num_channels,
            file_hash=loaded_audio.sha256_hash,
            memory_mb=loaded_audio.memory_mb
        )

    @classmethod
    def from_stem(cls, stem: Any, file_path: str, original_sr: int) -> 'AudioContext':
        """
        Create from stem. REQUIRES audio_context_meta dict.
        NO INFERENCE. If metadata missing, raise.
        """
        if not hasattr(stem, 'audio_context_meta'):
            raise ValueError(
                "AudioContext.from_stem: stem missing audio_context_meta. "
                "Stem must provide explicit metadata dict with keys: "
                "sample_rate, duration_seconds, num_channels, file_hash"
            )

        meta = stem.audio_context_meta
        required_keys = ['sample_rate', 'duration_seconds', 'num_channels', 'file_hash']
        missing_keys = [k for k in required_keys if k not in meta]
        if missing_keys:
            raise ValueError(
                f"AudioContext.from_stem: stem.audio_context_meta missing: {missing_keys}"
            )

        return cls(
            file_path=file_path,
            original_sample_rate=original_sr,
            working_sample_rate=meta['sample_rate'],
            duration_seconds=meta['duration_seconds'],
            num_channels=meta['num_channels'],
            file_hash=meta['file_hash'],
            memory_mb=meta.get('memory_mb', 0.0)
        )

    @classmethod
    def from_contract(cls, contract: Any, file_path: str, original_sr: int) -> 'AudioContext':
        """Create from AudioContract."""
        if not hasattr(contract, 'sample_rate') or not hasattr(contract, 'duration_ms'):
            raise ValueError(
                "AudioContext.from_contract: contract missing required attributes. "
                "Expected AudioContract with sample_rate and duration_ms."
            )

        if not hasattr(contract, 'sha256_hash'):
            raise ValueError(
                "AudioContext.from_contract: contract missing sha256_hash. "
                "AudioContract must provide forensic hash."
            )

        num_channels = getattr(contract, 'num_channels', 1)

        return cls(
            file_path=file_path,
            original_sample_rate=original_sr,
            working_sample_rate=contract.sample_rate,
            duration_seconds=contract.duration_ms / 1000.0,
            num_channels=num_channels,
            file_hash=contract.sha256_hash,
            memory_mb=0.0
        )

    @classmethod
    def unstable(cls, file_path: str, duration_seconds: float, num_channels: int = 1) -> 'AudioContext':
        """
        Create UNSTABLE context for error/fallback ONLY.

        WARNING: Cannot be used for caching, deduplication, or identity-critical operations.
        """
        warnings.warn(
            "AudioContext.unstable() creates context with UNSTABLE IDENTITY. "
            "Use only for error recovery and fallback paths.",
            RuntimeWarning,
            stacklevel=2
        )

        return cls(
            file_path=file_path,
            original_sample_rate=UNKNOWN_SAMPLE_RATE,
            working_sample_rate=UNKNOWN_SAMPLE_RATE,
            duration_seconds=duration_seconds,
            num_channels=num_channels,
            file_hash=UNKNOWN_HASH_PLACEHOLDER,
            memory_mb=0.0
        )

    @classmethod
    def from_components(
            cls,
            file_path: str,
            original_sample_rate: int,
            working_sample_rate: int,
            duration_seconds: float,
            num_channels: int,
            file_hash: str,
            memory_mb: float = 0.0
    ) -> 'AudioContext':
        """Direct construction with explicit components."""
        return cls(
            file_path=file_path,
            original_sample_rate=original_sample_rate,
            working_sample_rate=working_sample_rate,
            duration_seconds=duration_seconds,
            num_channels=num_channels,
            file_hash=file_hash,
            memory_mb=memory_mb
        )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "original_sample_rate": self.original_sample_rate,
            "working_sample_rate": self.working_sample_rate,
            "duration_seconds": self.duration_seconds,
            "duration_ms": self.duration_ms,
            "num_channels": self.num_channels,
            "is_mono": self.is_mono,
            "is_stereo": self.is_stereo,
            "is_multichannel": self.is_multichannel,
            "has_stable_identity": self.has_stable_identity,
            "file_hash": self.file_hash[:16] + "..." if self.has_stable_identity else "UNKNOWN",
            "memory_mb": self.memory_mb
        }

    def is_compatible_with(self, other: 'AudioContext') -> bool:
        """Check if two contexts are processing-compatible."""
        if self.is_unknown_sample_rate or other.is_unknown_sample_rate:
            return False
        return (self.working_sample_rate == other.working_sample_rate and
                self.num_channels == other.num_channels)

    def __repr__(self) -> str:
        hash_part = self.file_hash[:16] if self.has_stable_identity else "UNSTABLE"
        return (
            f"AudioContext("
            f"file={self.file_path!r}, "
            f"sr={self.working_sample_rate}Hz, "
            f"dur={self.duration_seconds:.2f}s, "
            f"ch={self.num_channels}, "
            f"hash={hash_part}...)"
        )


@dataclass
class MemoryMarker:
    """Tracks one audio buffer through the staggered deletion lifecycle."""
    buffer_name: str
    size_bytes: int
    created_at: datetime
    deleted_at: Optional[datetime] = None
    stage_cleared: Optional[str] = None
    stem_type: Optional[StemType] = None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


# ============================================================================
# PART 3: NOTE & RHYTHM EVENTS (UPDATED for 5.6.1)
# ============================================================================

@dataclass
class NoteEvent:
    """
    A single transcribed note. Non-destructive audit.

    INVARIANTS:
        - original start_ms/end_ms NEVER change
        - snapped_* fields are derived-only optional overlays
        - confidence MUST be 0.0-1.0 float, not Confidence enum

    NEW in 5.6.1:
        - midi_channel: For drum separation (channel 9 = drums)
        - phrase_id: For voice continuity and phrase grouping
    """
    pitch: int
    start_ms: float
    end_ms: float
    velocity: int
    confidence: float
    zero_crossing_rate: float
    source: SourceType
    snapped_start_ms: Optional[float] = None
    snapped_end_ms: Optional[float] = None
    snap_reason: Optional[str] = None
    snap_confidence_delta: Optional[float] = None
    quantization_strategy: Optional[QuantizationStrategy] = None

    consensus_votes: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    veto_attempts: List[Tuple[ValidationGate, VetoReason]] = field(default_factory=list)

    fundamental_freq_hz: Optional[float] = None
    harmonic_series_match_ratio: Optional[float] = None

    # NEW in 5.6.1
    midi_channel: int = 0  # 0 = pitched, 9 = drums
    phrase_id: Optional[str] = None  # From VoiceContinuity

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if self.snapped_start_ms is not None:
            if self.snapped_start_ms < self.start_ms - 5:
                raise ValueError(
                    f"snapped_start_ms ({self.snapped_start_ms}) < start_ms ({self.start_ms}) - 5ms"
                )
        if self.midi_channel not in (0, 9):
            # Auto-detect drums from source
            if self.source == SourceType.DRUM_INTELLIGENCE:
                self.midi_channel = 9

    def is_snapped(self) -> bool:
        return self.snapped_start_ms is not None

    def is_drum(self) -> bool:
        """Check if this note should go to MIDI channel 9."""
        return self.midi_channel == 9 or self.source == SourceType.DRUM_INTELLIGENCE

    def get_active_start_ms(self) -> float:
        return self.snapped_start_ms if self.snapped_start_ms is not None else self.start_ms

    def get_active_end_ms(self) -> float:
        return self.snapped_end_ms if self.snapped_end_ms is not None else self.end_ms

    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def active_duration_ms(self) -> float:
        return self.get_active_end_ms() - self.get_active_start_ms()

    def get_active_duration_ms(self) -> float:
        """Get active duration (snapped if available, otherwise original)."""
        return self.get_active_end_ms() - self.get_active_start_ms()

    def get_confidence_enum(self) -> Confidence:
        return Confidence.from_float(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "pitch": self.pitch,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "velocity": self.velocity,
            "confidence": self.confidence,
            "confidence_level": self.get_confidence_enum().name,
            "zero_crossing_rate": self.zero_crossing_rate,
            "source": self.source.value,
            "midi_channel": self.midi_channel,
            "is_drum": self.is_drum(),
        }
        if self.phrase_id:
            result["phrase_id"] = self.phrase_id
        if self.snapped_start_ms is not None:
            result["snapped_start_ms"] = self.snapped_start_ms
            result["snapped_end_ms"] = self.snapped_end_ms
            result["snap_reason"] = self.snap_reason
            result["snap_confidence_delta"] = self.snap_confidence_delta
        if self.fundamental_freq_hz:
            result["fundamental_freq_hz"] = self.fundamental_freq_hz
        if self.harmonic_series_match_ratio:
            result["harmonic_series_match_ratio"] = self.harmonic_series_match_ratio
        if self.reasoning_chain:
            result["reasoning_chain"] = self.reasoning_chain[:10]
        return result


@dataclass
class RhythmEvent:
    """A raw rhythmic hit before pitch assignment."""
    instrument: str
    start_ms: float
    end_ms: float
    source: SourceType
    confidence: float
    amplitude: float = 0.0
    velocity: int = 80
    drum_type: Optional[str] = None
    is_ghost_note: bool = False
    is_flam: bool = False
    flam_delta_ms: float = 0.0


@dataclass
class OnsetEvent:
    """A single detected rhythmic onset."""
    timestamp_ms: float
    strength: float
    width_ms: float
    spectral_flux: float


@dataclass
class NoteWithContext:
    """NoteEvent wrapped with its immediate neighbors for Relational Physics."""
    note: NoteEvent
    previous_note: Optional["NoteWithContext"] = None
    next_note: Optional["NoteWithContext"] = None
    phase_delta_to_kick_ms: Optional[float] = None
    phase_delta_to_previous_ms: Optional[float] = None
    is_grid_aligned: bool = False
    groove_weight: float = 1.0


# ============================================================================
# PART 4: SEPARATION RESULTS
# ============================================================================

@dataclass
class SeparationResult:
    """Output from a stem separation agent."""
    stems: Dict[StemType, Any]
    separation_time_seconds: float
    memory_usage_mb: float
    confidence: Confidence = Confidence.HIGH
    separator_used: Optional[str] = None  # "mel_roformer", "demucs", etc.

    def get_stem(self, stem_type: StemType) -> Any:
        return self.stems.get(stem_type)

    def has_stem(self, stem_type: StemType) -> bool:
        return stem_type in self.stems and self.stems[stem_type] is not None

    def release_stem(self, stem_type: StemType) -> None:
        self.stems.pop(stem_type, None)


# ============================================================================
# PART 5: PIPELINE RESULTS, EXPORT OPTIONS, VALIDATION
# ============================================================================

@dataclass
class StageResult:
    """Output from a single pipeline stage."""
    stage_name: str
    success: bool
    events: List[NoteEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    veto_reason: Optional[VetoReason] = None
    execution_time_ms: float = 0.0
    memory_delta_mb: float = 0.0

    def get_event_count(self) -> int:
        return len(self.events)

    def get_average_confidence(self) -> float:
        if not self.events:
            return 0.0
        return sum(e.confidence for e in self.events) / len(self.events)

    def get_average_confidence_enum(self) -> Confidence:
        return Confidence.from_float(self.get_average_confidence())


@dataclass
class ExportOptions:
    """Configuration for MIDI and JSON export stages."""
    midi_ppqn: int = 480
    json_pretty: bool = True
    include_original_timestamps: bool = True
    include_snapped_timestamps: bool = True
    include_veto_history: bool = True
    include_consensus_votes: bool = True
    heartbeat_note_pitch: Optional[int] = 60
    heartbeat_note_velocity: Optional[int] = 10
    octave_restoration_enabled: bool = True
    bass_octave_shift: int = -12


@dataclass
class ValidationResult:
    """Output from a single Scribe validation gate."""
    is_valid: bool
    gate_used: ValidationGate
    reason: Optional[VetoReason] = None
    detail: str = ""
    events_validated: int = 0
    events_rejected: int = 0
    confidence_before: float = 0.0
    confidence_after: float = 0.0


@dataclass
class ForensicRecord:
    """Append-only record of a pipeline decision."""
    session_id: str
    timestamp_utc: datetime
    stage_name: str
    decision_type: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    reasoning: str
    human_reversible: bool = True


# ============================================================================
# PART 6: GROOVE, PULSE, AND BEAT GRID
# ============================================================================

@dataclass
class GrooveField:
    """Relational Physics: note timing defined by neighbor distance."""
    bass_kick_phase_delta_ms: float
    is_wide_swing: bool
    is_dilla_pocket: bool
    average_inter_note_distance_ms: float
    tempo_estimate_bpm: float
    confidence: Confidence
    groove_signature: Optional[str] = None
    phase_delta_variance_ms: float = 0.0


@dataclass
class PulseField:
    """
    Probabilistic beat grid from TempoIntelligence.

    INVARIANT: beat_grid_ms MUST be sorted ascending.
    """
    tempo_bpm: float
    confidence: Confidence
    beat_grid_ms: List[float]
    sixteenth_grid_ms: List[float]
    pulse_strength: List[float]
    phase_shift_ms: float = 0.0
    is_double_time: bool = False
    is_half_time: bool = False

    def __post_init__(self):
        if self.beat_grid_ms and not _is_sorted_ascending(self.beat_grid_ms):
            _warn_unsorted("PulseField", "beat_grid_ms")
        if self.sixteenth_grid_ms and not _is_sorted_ascending(self.sixteenth_grid_ms):
            _warn_unsorted("PulseField", "sixteenth_grid_ms")

    def get_closest_pulse(self, timestamp_ms: float) -> float:
        if not self.beat_grid_ms:
            return timestamp_ms
        return min(self.beat_grid_ms, key=lambda t: abs(t - timestamp_ms))

    def get_beat_index(self, timestamp_ms: float) -> int:
        if not self.beat_grid_ms:
            return -1
        pos = bisect.bisect_left(self.beat_grid_ms, timestamp_ms)
        if pos == 0:
            return 0
        if pos >= len(self.beat_grid_ms):
            return len(self.beat_grid_ms) - 1
        before = self.beat_grid_ms[pos - 1]
        after = self.beat_grid_ms[pos]
        return pos - 1 if abs(timestamp_ms - before) <= abs(timestamp_ms - after) else pos

    def distance_to_nearest_beat_ms(self, timestamp_ms: float) -> float:
        nearest = self.get_closest_pulse(timestamp_ms)
        return abs(timestamp_ms - nearest)


@dataclass
class PulseHypothesis:
    """A candidate pulse period with probability weight."""
    period_ms: float
    probability: float
    harmonic_level: float


@dataclass
class BeatGrid:
    """
    Structured beat grid from madmom DBNBeatTrackingProcessor.

    INVARIANT: beat_times_ms MUST be sorted ascending.
    """
    beat_times_ms: List[float]
    downbeat_times_ms: List[float]
    measure_duration_ms: float
    beat_duration_ms: float
    tempo_bpm: float
    time_signature: str

    def __post_init__(self):
        if self.beat_times_ms and not _is_sorted_ascending(self.beat_times_ms):
            _warn_unsorted("BeatGrid", "beat_times_ms")
        if self.downbeat_times_ms and not _is_sorted_ascending(self.downbeat_times_ms):
            _warn_unsorted("BeatGrid", "downbeat_times_ms")

    def get_closest_beat(self, time_ms: float) -> Tuple[float, int]:
        if not self.beat_times_ms:
            return time_ms, -1
        pos = bisect.bisect_left(self.beat_times_ms, time_ms)
        if pos == 0:
            return self.beat_times_ms[0], 0
        if pos >= len(self.beat_times_ms):
            return self.beat_times_ms[-1], len(self.beat_times_ms) - 1
        before = self.beat_times_ms[pos - 1]
        after = self.beat_times_ms[pos]
        if abs(time_ms - before) <= abs(time_ms - after):
            return before, pos - 1
        return after, pos

    def is_downbeat(self, beat_index: int) -> bool:
        if not self.beat_times_ms or beat_index < 0 or beat_index >= len(self.beat_times_ms):
            return False
        beat_time = self.beat_times_ms[beat_index]
        return any(abs(beat_time - db) < 5.0 for db in self.downbeat_times_ms)

    def to_pulse_field(self, confidence: Confidence) -> PulseField:
        sixteenth_ms: List[float] = []
        if self.beat_times_ms and self.beat_duration_ms > 0:
            sixteenth_interval = self.beat_duration_ms / 4.0
            for beat_ms in self.beat_times_ms:
                for i in range(4):
                    sixteenth_ms.append(beat_ms + i * sixteenth_interval)
            sixteenth_ms.sort()

        return PulseField(
            tempo_bpm=self.tempo_bpm,
            confidence=confidence,
            beat_grid_ms=list(self.beat_times_ms),
            sixteenth_grid_ms=sixteenth_ms,
            pulse_strength=[1.0] * len(self.beat_times_ms),
        )


# ============================================================================
# PART 7: TEMPO & TIMING
# ============================================================================

@dataclass
class TempoEvent:
    """A tempo change at a specific point in time."""
    time_ms: float
    tempo_bpm: float
    confidence: Confidence


@dataclass
class TempoMap:
    """
    Tempo information for a complete track.

    INVARIANT: tempo_events MUST be sorted by time_ms ascending.
    """
    initial_tempo_bpm: float
    tempo_events: List[TempoEvent] = field(default_factory=list)
    time_signature_numerator: int = 4
    time_signature_denominator: int = 4
    confidence: Confidence = Confidence.MEDIUM

    def __post_init__(self):
        if self.initial_tempo_bpm < MIN_VALID_TEMPO_BPM or self.initial_tempo_bpm > MAX_VALID_TEMPO_BPM:
            raise ValueError(
                f"Initial tempo {self.initial_tempo_bpm} BPM out of range "
                f"[{MIN_VALID_TEMPO_BPM}, {MAX_VALID_TEMPO_BPM}]"
            )
        for i in range(1, len(self.tempo_events)):
            if self.tempo_events[i].time_ms < self.tempo_events[i - 1].time_ms:
                raise ValueError(
                    f"Tempo events not sorted: {self.tempo_events[i - 1].time_ms}ms > {self.tempo_events[i].time_ms}ms")

    def get_tempo_at_ms(self, ms: float) -> float:
        if not self.tempo_events:
            return self.initial_tempo_bpm
        if ms <= self.tempo_events[0].time_ms:
            return self.initial_tempo_bpm
        if ms >= self.tempo_events[-1].time_ms:
            return self.tempo_events[-1].tempo_bpm
        for i in range(len(self.tempo_events) - 1):
            t1 = self.tempo_events[i].time_ms
            bpm1 = self.tempo_events[i].tempo_bpm
            t2 = self.tempo_events[i + 1].time_ms
            bpm2 = self.tempo_events[i + 1].tempo_bpm
            if t1 <= ms <= t2:
                span = t2 - t1
                if span <= 0:
                    return bpm2
                return bpm1 + (ms - t1) / span * (bpm2 - bpm1)
        return self.tempo_events[-1].tempo_bpm

    def beat_duration_ms(self, at_ms: float = 0.0) -> float:
        bpm = self.get_tempo_at_ms(at_ms)
        return 60000.0 / bpm if bpm > 0 else 500.0


@dataclass
class BeatTrackingResult:
    """A single detected beat with metadata."""
    timestamp_ms: float
    beat_number: int
    confidence: float
    is_accented: bool = False
    tempo_instant_bpm: float = 0.0


@dataclass
class TimeSignatureCandidate:
    """A possible time signature for a segment."""
    numerator: int
    denominator: int
    confidence: float
    sample_segment_start_ms: float
    sample_segment_end_ms: float
    # Which beat_times index (0..numerator-1) the downbeat-salience search
    # found to be the real "beat 1" - e.g. a pickup/anacrusis measure means
    # beat_times[0] is NOT the downbeat, so this can be nonzero.
    downbeat_phase: int = 0


@dataclass
class TempoIntelligenceResult:
    """Complete output from TempoIntelligence."""
    tempo_map: TempoMap
    pulse_field: PulseField
    beat_track: List[BeatTrackingResult]
    time_signature_candidates: List[TimeSignatureCandidate]
    beat_grid: Optional[BeatGrid] = None
    time_signature_primary: Optional[TimeSignatureCandidate] = None
    is_constant_tempo: bool = True


@dataclass
class RhythmPattern:
    """A detected repeating rhythmic pattern."""
    pattern_id: str
    pattern_interval_ms: float
    confidence: Confidence
    onsets_in_pattern: List[OnsetEvent]
    is_swing: bool = False
    swing_ratio: float = 0.0


@dataclass
class RhythmTrackResult:
    """Complete output from RhythmEngine."""
    onsets: List[OnsetEvent]
    patterns: List[RhythmPattern]
    dominant_tempo_bpm: float
    tempo_confidence: Confidence
    time_signature_estimate: str
    has_polymeter: bool = False


# ============================================================================
# PART 8: DRUM INTELLIGENCE
# ============================================================================

@dataclass
class DrumKitMapping:
    """Maps DrumType to MIDI note numbers."""
    drum_to_midi: Dict[DrumType, int] = field(default_factory=dict)
    midi_to_drum: Dict[int, DrumType] = field(default_factory=dict)
    kit_name: str = "GM_Standard"

    def __post_init__(self) -> None:
        for drum, note in self.drum_to_midi.items():
            if note not in self.midi_to_drum:
                self.midi_to_drum[note] = drum

    @classmethod
    def gm_standard(cls) -> "DrumKitMapping":
        mapping = {
            DrumType.KICK: 36,
            DrumType.SNARE: 38,
            DrumType.HI_HAT_CLOSED: 42,
            DrumType.HI_HAT_OPEN: 46,
            DrumType.CRASH: 49,
            DrumType.RIDE: 51,
            DrumType.TOM_HIGH: 48,
            DrumType.TOM_MID: 45,
            DrumType.TOM_LOW: 43,
            DrumType.RIMSHOT: 37,
            DrumType.COWBELL: 56,
            DrumType.PERCUSSION: 39,
        }
        return cls(drum_to_midi=mapping)

    def get_midi(self, drum_type: DrumType, fallback: int = 38) -> int:
        return self.drum_to_midi.get(drum_type, fallback)


@dataclass
class DrumEvent:
    """A detected drum hit with classification."""
    drum_type: DrumType
    start_ms: float
    end_ms: float
    velocity: int
    confidence: float
    midi_note: int = 0
    is_ghost_note: bool = False
    is_flam: bool = False
    flam_delta_ms: float = 0.0
    is_rimshot: bool = False

    def __post_init__(self) -> None:
        if self.midi_note == 0:
            _gm = {
                DrumType.KICK: 36, DrumType.SNARE: 38, DrumType.HI_HAT_CLOSED: 42,
                DrumType.HI_HAT_OPEN: 46, DrumType.CRASH: 49, DrumType.RIDE: 51,
                DrumType.TOM_HIGH: 48, DrumType.TOM_MID: 45, DrumType.TOM_LOW: 43,
                DrumType.RIMSHOT: 37, DrumType.COWBELL: 56, DrumType.PERCUSSION: 39,
            }
            self.midi_note = _gm.get(self.drum_type, 38)

    def to_note_event(self, source: SourceType = SourceType.DRUM_INTELLIGENCE) -> NoteEvent:
        return NoteEvent(
            pitch=self.midi_note,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            velocity=self.velocity,
            confidence=self.confidence,
            zero_crossing_rate=0.0,
            source=source,
            midi_channel=9,  # Drums go to channel 9
            reasoning_chain=[f"Drum hit: {self.drum_type.value}"],
        )


@dataclass
class DrumTrackResult:
    """Complete output from DrumIntelligence."""
    events: List[DrumEvent]
    drum_types_detected: List[DrumType]
    kit_mapping: DrumKitMapping
    total_drum_events: int
    average_density_events_per_second: float


# ============================================================================
# PART 9: WITNESS & CONSENSUS
# ============================================================================

@dataclass
class WitnessVote:
    """One agent's vote about a note."""
    source: SourceType
    note_present: bool
    pitch_midi: Optional[int] = None
    start_ms: Optional[float] = None
    end_ms: Optional[float] = None
    confidence: float = 0.0
    zero_crossing_rate: float = 0.0
    veto: Optional[Tuple[ValidationGate, VetoReason]] = None
    reasoning: str = ""

    def is_qualified(self) -> bool:
        return self.confidence >= Confidence.LOW.value_f


@dataclass
class WitnessTestimony:
    """Complete testimony from one witness agent."""
    witness_id: SourceType
    testimony_time_ms: float
    events: List[NoteEvent]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_veto_vote(self) -> Optional[Tuple[VetoReason, str]]:
        if not self.events:
            return (VetoReason.EMPTY_RESULT, "No events detected by this witness")
        all_low = all(e.confidence < Confidence.LOW.value_f for e in self.events)
        if all_low:
            max_conf = max(e.confidence for e in self.events)
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Max confidence {max_conf:.2f} below LOW threshold {Confidence.LOW.value_f}")
        return None


@dataclass
class ConsensusConfig:
    """Configuration for ConsensusEngine."""
    strategy: ConsensusStrategy = ConsensusStrategy.ANY_VETO_WINS
    min_qualified_witnesses: int = 2
    veto_threshold: float = 0.25
    require_confidence_consensus: bool = True


@dataclass
class ConsensusPackage:
    """ConsensusEngine's final verdict."""
    witness_votes: List[WitnessVote]
    final_decision: bool
    final_pitch: Optional[int] = None
    final_start_ms: Optional[float] = None
    final_end_ms: Optional[float] = None
    consensus_confidence: float = 0.0
    veto_triggered_by: Optional[SourceType] = None
    veto_reason: Optional[VetoReason] = None
    timestamp_utc: datetime = field(default_factory=datetime.now)

    def get_qualified_witnesses(self) -> List[WitnessVote]:
        return [w for w in self.witness_votes if w.is_qualified()]

    def unanimously_approved(self) -> bool:
        qualified = self.get_qualified_witnesses()
        if not qualified:
            return False
        return all(w.note_present == self.final_decision for w in qualified)


# ============================================================================
# PART 10: SCHOENBERG MIRROR
# ============================================================================

@dataclass
class PartialTrack:
    """A single harmonic partial detected in the spectrum."""
    partial_number: int
    frequency_hz: float
    amplitude_db: float
    confidence: float


@dataclass
class HarmonicSeries:
    """Complete harmonic series analysis for a detected pitch."""
    fundamental_freq_hz: float
    fundamental_confidence: float
    partials: List[PartialTrack]
    missing_partials: List[int]
    inharmonicity: float = 0.0

    def get_highest_present_partial(self) -> int:
        present = [p.partial_number for p in self.partials if p.confidence > 0.5]
        return max(present) if present else 1

    def match_ratio(self) -> float:
        if not self.partials:
            return 0.0
        expected_count = self.partials[-1].partial_number if self.partials else 1
        present_count = sum(1 for p in self.partials if p.confidence > 0.5)
        return present_count / max(expected_count, 1)


@dataclass
class SchoenbergResult:
    """Verdict from the Schoenberg Mirror."""
    verdict: SchoenbergVerdict
    harmonic_series: Optional[HarmonicSeries] = None
    zero_crossing_rate: float = 0.0
    spectral_flatness: float = 0.0
    reason: str = ""

    def is_tonal(self) -> bool:
        return self.verdict == SchoenbergVerdict.TONAL

    def should_veto(self) -> bool:
        return self.verdict in (SchoenbergVerdict.NOISE, SchoenbergVerdict.HALLUCINATION)


# ============================================================================
# PART 11: VOICE CONTINUITY
# ============================================================================

@dataclass
class Voice:
    """A traced monophonic voice line."""
    voice_id: str
    role: VoiceRole
    notes: List[NoteEvent]
    start_time_ms: float
    end_time_ms: float
    average_pitch: float
    pitch_range_semitones: float
    is_monophonic: bool = True


@dataclass
class VoiceContinuityResult:
    """Complete output from VoiceContinuity."""
    voices: List[Voice]
    voice_crossings: List[Tuple[Voice, Voice, float]]
    octave_jumps: List[Tuple[NoteEvent, NoteEvent]]
    confidence: Confidence


# ============================================================================
# PART 12: QUANTIZATION
# ============================================================================

@dataclass
class QuantizationPass:
    """Record of one pass in multi-stage quantization."""
    stage: QuantizationStage
    grid_resolution_ms: float
    snap_distance_allowance_ms: float
    confidence_penalty: float
    events_affected: int
    average_shift_ms: float


@dataclass
class MultiStageQuantization:
    """Complete audit trail of Ritornello quantization."""
    original_events: List[NoteEvent]
    passes: List[QuantizationPass]
    final_events: List[NoteEvent]
    total_confidence_loss: float
    human_readable_audit: str


@dataclass
class MergeCandidate:
    """A group of overlapping notes identified for merging."""
    notes: List[NoteEvent]
    merge_start_ms: float
    merge_end_ms: float
    merge_velocity: int
    merge_confidence: float
    strategy_used: Optional[MergeStrategy] = None


@dataclass
class VelocityMergeResult:
    """Output from VelocityMerge."""
    original_notes: List[NoteEvent]
    merged_notes: List[NoteEvent]
    merges_performed: List[MergeCandidate]
    notes_removed: int


# ============================================================================
# PART 13: ANECHOIC MA
# ============================================================================

@dataclass
class AnechoicMask:
    """De-reverberation profile for a recording."""
    profile: AnechoicProfile
    reverb_time_rt60_ms: float
    direct_to_reverb_ratio_db: float
    confidence: Confidence
    pre_delay_ms: float = 0.0


# ============================================================================
# PART 14: DEPRECATED TYPES
# ============================================================================

class RouterType(str, Enum):
    """DEPRECATED: Routing is forbidden in 5.0."""
    CONFIDENCE_ROUTER = "FORBIDDEN_IN_5.0"
    STATE_MANAGER = "FORBIDDEN_IN_5.0"
    FUSION_LAYER = "FORBIDDEN_IN_5.0"


class FusionStrategy(str, Enum):
    """DEPRECATED: Fusion is forbidden in 5.0."""
    WEIGHTED_AVERAGE = "DEPRECATED"
    MAX_CONFIDENCE = "DEPRECATED"
    CONSENSUS = "DEPRECATED"


# ============================================================================
# PART 15: IMMUTABLE COPY UTILITIES (G6 Fix)
# ============================================================================

def copy_note_event(event: NoteEvent) -> NoteEvent:
    """Create a deep copy of a NoteEvent for immutable passing."""
    return deepcopy(event)


def copy_note_events(events: List[NoteEvent]) -> List[NoteEvent]:
    """Create deep copies of all note events."""
    return [deepcopy(e) for e in events]


# ============================================================================
# PART 16: DEPRECATION SHIMS (Safe, Deterministic)
# ============================================================================

_SHIM_MAP: Dict[str, Any] = {}

try:
    from core.feature_bundle import EvidenceType, MemoryPriority, EvidenceLease, FeatureBundle

    _SHIM_MAP.update({
        "EvidenceType": EvidenceType,
        "MemoryPriority": MemoryPriority,
        "EvidenceLease": EvidenceLease,
        "FeatureBundle": FeatureBundle,
    })
except ImportError:
    pass


def __getattr__(name: str) -> Any:
    """Lazy import for types moved to feature_bundle.py."""
    if name in _SHIM_MAP:
        warnings.warn(
            f"Importing {name} from core.order_types is deprecated. "
            f"Use 'from core.feature_bundle import {name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return _SHIM_MAP[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ============================================================================
# PART 17: UTILITY FUNCTIONS
# ============================================================================

def confidence_bucket(value: float) -> Confidence:
    """Convert float confidence to Confidence enum."""
    return Confidence.from_float(value)


def validate_pitch(pitch: int) -> bool:
    return 0 <= pitch <= 127


def validate_velocity(velocity: int) -> bool:
    return 0 <= velocity <= 127


def ms_to_samples(ms: float, sample_rate: int) -> int:
    return int(ms * sample_rate / 1000.0)


def samples_to_ms(samples: int, sample_rate: int) -> float:
    return samples * 1000.0 / sample_rate


def hz_to_midi(frequency_hz: float) -> float:
    if frequency_hz <= 0:
        return 0.0
    return 12.0 * math.log2(frequency_hz / 440.0) + 69.0


def midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def beat_ms_to_bpm(beat_duration_ms: float) -> float:
    if beat_duration_ms <= 0:
        return 120.0
    return 60000.0 / beat_duration_ms


def bpm_to_beat_ms(bpm: float) -> float:
    if bpm <= 0:
        return 500.0
    return 60000.0 / bpm


def is_sorted_ascending(xs: List[float], tolerance: float = SORT_TOLERANCE_MS) -> bool:
    """Check if list is sorted in ascending order within tolerance."""
    return _is_sorted_ascending(xs, tolerance)