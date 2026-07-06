# =================================================================
# MODULE: core/__init__.py
# DESCRIPTION: Core module exports for Grimlock 5.5
#
# This file defines the public API for the core module.
# All imports from core should come from here.
#
# LAW: NEVER import from core.submodule directly outside of core.
#      Use `from core import X` instead.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-26
# =================================================================

import sys
from typing import Any
import warnings

# ============================================================================
# PART 1: ORDER TYPES (Foundation - stdlib only, no numpy)
# ============================================================================

from core.order_types import (
    # Enums
    Confidence,
    DecisionType,
    VetoReason,
    ValidationGate,
    SourceType,
    StemType,
    QuantizationStrategy,
    QuantizationStage,
    MergeStrategy,
    ConsensusStrategy,
    DrumType,
    AnechoicProfile,
    VoiceRole,
    SchoenbergVerdict,
    RouterType,
    FusionStrategy,

    # Core Data Structures
    AudioContext,
    MemoryMarker,
    NoteEvent,
    RhythmEvent,
    OnsetEvent,
    NoteWithContext,
    SeparationResult,
    StageResult,
    ExportOptions,
    ValidationResult,
    ForensicRecord,
    GrooveField,
    PulseField,
    PulseHypothesis,
    BeatGrid,
    TempoEvent,
    TempoMap,
    BeatTrackingResult,
    TimeSignatureCandidate,
    TempoIntelligenceResult,
    RhythmPattern,
    RhythmTrackResult,
    DrumKitMapping,
    DrumEvent,
    DrumTrackResult,
    WitnessVote,
    WitnessTestimony,
    ConsensusConfig,
    ConsensusPackage,
    PartialTrack,
    HarmonicSeries,
    SchoenbergResult,
    Voice,
    VoiceContinuityResult,
    QuantizationPass,
    MultiStageQuantization,
    MergeCandidate,
    VelocityMergeResult,
    AnechoicMask,

    # Utility Functions
    validate_pitch,
    validate_velocity,
    ms_to_samples,
    samples_to_ms,
    hz_to_midi,
    midi_to_hz,
    beat_ms_to_bpm,
    bpm_to_beat_ms,

    # Constants
    UNKNOWN_HASH_PLACEHOLDER,
)

# ============================================================================
# PART 2: TESTIMONY (Canonical testimony objects for pipeline stages)
# ============================================================================

from core.testimony import (
    # Base Result Type
    StageResult as TestimonyStageResult,

    # Testimony Objects
    IngestionTestimony,
    FeatureExtractionTestimony,
    TempoTestimony,
    SeparationTestimony,
    PitchTestimony,
    RhythmTestimony,
    DrumTestimony,
    TonalTestimony,
    TimbreTestimony,
    HarmonicValidationTestimony,
    AnechoicMATestimony,
    GrooveTestimony,
    PulseTestimony,
    LatticeTestimony,
    VoiceContinuityTestimony,
    QuantizationTestimony,
    MergeTestimony,
    ConsensusTestimony,
    ValidationTestimony,
    ExportTestimony,

    # Testimony Normalizer
    TestimonyNormalizer,
    normalizer,
)

# ============================================================================
# PART 3: AUDIO VIEWS (Immutable audio with multi-proxy pattern)
# ============================================================================

from core.audio_views import (
    ChannelLayout,
    ImmutableAudio,
    AudioViews,
    create_audio_views,
    create_stereo_views,
    create_mono_views,
    views_from_loaded_audio,
)

# ============================================================================
# PART 4: CONTRACTS (WHAT components must do - no implementations)
# ============================================================================

from core.contracts import (
    # Audio Contract
    AudioContract,

    # Detection Contract
    ConfidenceComponents,
    DetectionGuarantees,
    DetectionResult,
    DetectionContract,

    # Separation Contract
    StemContract,
    SeparationContractResult,
    SeparationContract,

    # Analysis Contract
    GrooveAnalysisResult,
    TempoAnalysisResult,
    AnalysisResult,
    AnalysisContract,

    # Quantization Contract
    QuantizationAudit,
    QuantizationContract,

    # Validation Contract
    ValidationReport,
    ValidationContract,

    # Memory Contract
    BufferID,
    MemoryReport,
    ArenaContract,
    GuardianContract,

    # Feature Bundle Contract
    EvidenceType,
    EvidenceLease,
    FeatureBundleContract,

    # Observable Contract
    ObservableContract,

    # Validation Helpers
    validate_audio_contract,
    validate_detection_result,
    validate_stem_contract,

    # Registry
    CONTRACT_REGISTRY,
)

# ============================================================================
# PART 5: ACOUSTIC INTELLIGENCE (Audio truth engine)
# ============================================================================

from core.acoustic_intelligence import (
    # Constants
    MIN_SAMPLE_RATE,
    MAX_SAMPLE_RATE,
    MIN_DURATION_MS,
    MAX_DURATION_MS,
    MAX_CHANNELS,
    MAX_ABSOLUTE_AMPLITUDE,
    ANTI_ALIAS_MARGIN,
    DEFAULT_KAISER_BETA,
    FLOAT_EPSILON,
    QUALITY_PRESETS,

    # Channel Layout
    ChannelLayout as AcousticChannelLayout,

    # Immutable Audio Implementation
    ImmutableAudio as AcousticImmutableAudio,

    # Acoustic Intelligence (Transformation Authority)
    AcousticIntelligence,
)

# ============================================================================
# PART 6: FEATURE BUNDLE (Spectral evidence - lazy calculation with validation)
# ============================================================================

from core.feature_bundle import (
    # Main class
    FeatureBundle,
    FeatureProvenance,
    DSPThresholds,

    # Factory functions
    create_feature_bundle,
    extract_features,
    bundle_for_madmom,
    bundle_for_pitch,
    bundle_for_analysis,

    # Constants
    MADMOM_SR,
    BASIC_PITCH_SR,
    CREPE_SR,
    DEMUCS_SR,
    BS_ROFORMER_SR,
    ANALYSIS_SR,
    MIN_DURATION_SECONDS,

    # Validation functions
    validate_sample_rate,
    validate_audio_duration,

    # DSP Utilities
    sanitize_audio,
    validate_fft_size,
    safe_normalize,
    compute_audio_statistics,
    compute_feature_statistics,
)

# ============================================================================
# PART 7: EVIDENCE TRACKER (Reference counting)
# ============================================================================

from core.evidence_tracker import (
    EvidenceTracker,
    EvidenceLeaseRecord,
)

# ============================================================================
# PART 8: CONSTANTS (Configuration thresholds)
# ============================================================================

from core.constants import (
    # Sample rates
    TARGET_SAMPLE_RATE,
    ANALYSIS_SR,
    BASIC_PITCH_SR,
    CREPE_SR,
    MADMOM_SR,
    DEMUCS_SR,
    TARGET_SAMPLE_RATE_DEPRECATED,
    PROFESSIONAL_FIDELITY_SR,

    # STFT
    MONO_DOWNMIX_FORCED,
    STFT_HOP_LENGTH,
    STFT_WINDOW_SIZE,
    STFT_WINDOW_TYPE,

    # Truncation
    DEFAULT_PROCESSING_DURATION_SECONDS,
    TRUNCATION_PRESETS,
    MIN_TRUNCATION_DURATION_SECONDS,
    MAX_TRUNCATION_DURATION_SECONDS,
    validate_truncation_duration,
    get_truncation_preset,

    # Memory limits
    MEMORY_LIMIT_MB,
    MEMORY_WARNING_THRESHOLD_MB,
    MEMORY_CRITICAL_THRESHOLD_MB,
    STAGGERED_GC_TRIGGER_MB,
    BUFFER_LIFECYCLE_STAGES,
    RETAIN_FULL_MIX_AFTER_SEPARATION,
    RETAIN_SEPARATED_STEMS_AFTER_DETECTION,
    MAX_CONCURRENT_BUFFERS,
    GC_COLLECT_AFTER_EACH_STAGE,
    DELAY_BETWEEN_STAGGERED_DELETIONS_MS,

    # Pipeline timeouts
    STAGE_TIMEOUT_LOAD_SECONDS,
    STAGE_TIMEOUT_SEPARATION_SECONDS,
    STAGE_TIMEOUT_DETECTION_SECONDS,
    STAGE_TIMEOUT_ANALYSIS_SECONDS,
    STAGE_TIMEOUT_QUANTIZATION_SECONDS,
    STAGE_TIMEOUT_EXPORT_SECONDS,
    MAX_STAGE_RETRIES,
    ALLOWED_CROSS_STAGE_IMPORTS,
    CORE_MODULE_ALLOWED_IMPORTS,

    # Scribe validation
    MIN_CONFIDENCE_TO_PASS,
    SILENCE_RATIO_MAX,
    ZERO_CROSSING_MAX,
    MAX_HALLUCINATION_RATIO,
    MIN_EVENTS_REQUIRED,
    MAX_EVENTS_FOR_SANITY_CHECK,
    VALIDATION_GATE_TO_VETO,
    CONFIDENCE_WEIGHTS,
    MIN_QUALIFIED_WITNESSES_FOR_CONSENSUS,

    # Schoenberg Mirror
    HARMONIC_SERIES_MIN_PARTIALS,
    HARMONIC_SERIES_MATCH_THRESHOLD,
    MAX_INHARMONICITY,
    ZERO_CROSSING_HYSTERESIS_MS,
    SPECTRAL_FLATNESS_TONAL_THRESHOLD,
    SPECTRAL_FLATNESS_NOISE_THRESHOLD,
    VERDICT_BASE_CONFIDENCE,

    # Groove Field
    PHASE_DELTA_SWING_THRESHOLD_MS,
    DILLA_POCKET_VARIANCE_MS,
    MAX_GROOVE_SNAP_MS,
    MIN_PHASE_DELTA_TO_CONSIDER_MS,
    INTER_NOTE_DISTANCE_HISTORY_WINDOW,
    GROOVE_FIELD_CONFIDENCE_THRESHOLDS,

    # Pulse Field
    PULSE_FIELD_RESOLUTION_MS,
    MIN_TEMPO_BPM,
    MAX_TEMPO_BPM,
    DEFAULT_TEMPO_BPM,
    TEMPO_FORECAST_WINDOW_SECONDS,
    BEAT_TRACKING_HOP_MS,
    DEFAULT_TIME_SIGNATURE_NUMERATOR,
    DEFAULT_TIME_SIGNATURE_DENOMINATOR,
    TIME_SIGNATURE_CANDIDATE_LIMIT,

    # Ritornello quantization
    RITORNELLO_MAX_SNAP_MS,
    PRESERVE_ORIGINAL_MS_ALWAYS,
    MIN_SNAP_DISTANCE_TO_CONSIDER_MS,
    SNAP_CONFIDENCE_PENALTY_PER_MS,
    MAX_CONSECUTIVE_SNAPS,
    QUANTIZATION_GRID_RESOLUTIONS_MS,
    QUANTIZATION_STAGE_CONFIDENCE_PENALTIES,
    MIN_CONFIDENCE_FOR_QUANTIZATION,

    # Velocity Merge
    OVERLAP_TOLERANCE_MS,
    DEFAULT_MERGE_STRATEGY,
    MIN_VELOCITY_TO_KEEP,
    MAX_VELOCITY,
    DEFAULT_GHOST_NOTE_VELOCITY,
    MAX_VOICE_GAP_MS,
    MIN_NOTE_DURATION_MS,
    MAX_OCTAVE_JUMP_SEMITONES,
    VOICE_CROSSING_DETECTION_WINDOW_MS,

    # Drum Intelligence
    SPICE_MODEL_SIZE,
    SPICE_HOP_LENGTH_MS,
    SPICE_ONSET_THRESHOLD,
    DRUM_CONFIDENCE_MIN,
    DRUM_GHOST_NOTE_VELOCITY_MAX,
    DRUM_FLAM_DETECTION_WINDOW_MS,
    DRUM_ROLL_DETECTION_MIN_HITS,
    DRUM_ROLL_MAX_INTERVAL_MS,

    # Pitch Intelligence
    MIN_PITCH_MIDI,
    MAX_PITCH_MIDI,
    MIN_FREQUENCY_HZ,
    MAX_FREQUENCY_HZ,
    TONAL_CORRELATION_THRESHOLD,
    TONAL_HARMONIC_WEIGHT_DECAY,
    TONAL_MAX_HARMONICS,
    BASIC_PITCH_MODEL,
    BASIC_PITCH_HOP_MS,
    BASIC_PITCH_MIN_FREQ_HZ,
    BASIC_PITCH_MAX_FREQ_HZ,

    # Mel-Roformer
    MEL_ROFORMER_MODEL_NAME,
    MEL_ROFORMER_N_MELS,
    MEL_ROFORMER_SAMPLE_RATE,
    MEL_ROFORMER_DEVICE,
    MEL_ROFORMER_NUM_LAYERS,
    MEL_ROFORMER_HIDDEN_SIZE,
    MEL_ROFORMER_NUM_HEADS,
    MEL_ROFORMER_DROPOUT,
    MEL_ROFORMER_N_FFT,
    MEL_ROFORMER_HOP_LENGTH,
    MEL_ROFORMER_WINDOW_TYPE,
    MEL_ROFORMER_STEMS,
    MEL_ROFORMER_INCLUDE_RESIDUAL_STEM,
    MEL_ROFORMER_STEREO_MODE,
    MEL_ROFORMER_CHUNK_DURATION_SECONDS,
    MEL_ROFORMER_OVERLAP_SECONDS,
    MEL_ROFORMER_USE_CHUNKED_INFERENCE,
    MEL_ROFORMER_MAX_DURATION_SECONDS,
    MEL_ROFORMER_TIMEOUT_SECONDS,
    MEL_ROFORMER_FORCE_CPU,
    MEL_ROFORMER_GPU_MEMORY_FRACTION,
    MEL_ROFORMER_BATCH_SIZE,
    MEL_ROFORMER_EPSILON,
    MEL_ROFORMER_DETERMINISTIC,
    MEL_ROFORMER_RANDOM_SEED,
    MEL_ROFORMER_MIN_SPECTRAL_COHERENCE,
    MEL_ROFORMER_MAX_ACCEPTABLE_BLEED,
    MEL_ROFORMER_MAX_RECONSTRUCTION_ERROR,
    MEL_ROFORMER_MAX_MASK_ENTROPY,
    MEL_ROFORMER_FALLBACK_ENABLED,
    MEL_ROFORMER_FALLBACK_MODE,
    MEL_ROFORMER_QUALITY_PROFILES,
    get_mel_roformer_quality_profile,

    # Demucs
    DEMUCS_MODEL,
    DEMUCS_SEGMENT_SECONDS,
    DEMUCS_OVERLAP_SECONDS,
    DEMUCS_STEMS,
    DEMUCS_INCLUDE_RESIDUAL_STEM,
    DEMUCS_MAX_DURATION_SECONDS,
    DEMUCS_TIMEOUT_SECONDS,
    DEMUCS_FORCE_CPU,
    DEMUCS_GPU_MEMORY_FRACTION,
    DEMUCS_WINDOW_SECONDS,
    DEMUCS_HOP_SECONDS,
    DEMUCS_FALLBACK_ENABLED,
    DEMUCS_FALLBACK_MODE,
    DEMUCS_QUALITY_PROFILES,
    get_demucs_quality_profile,

    # BS-Roformer (deprecated)
    BS_ROFORMER_DEPRECATED,
    BS_ROFORMER_MIGRATION_WARNING,
    BS_ROFORMER_MODEL,
    BS_ROFORMER_HOP_LENGTH_SAMPLES,
    BS_ROFORMER_WINDOW_SIZE_SAMPLES,

    # Unified separation
    SEPARATION_FALLBACK_ORDER,
    SEPARATION_CACHE_ENABLED,
    SEPARATION_CACHE_MAX_SIZE_MB,
    SEPARATION_VALIDATE_OUTPUT,
    SEPARATION_MIN_STEM_ENERGY_RATIO,
    SEPARATION_MAX_STEM_ENERGY_RATIO,

    # Hybrid separator
    HYBRID_SEPARATOR_QUALITY,
    HYBRID_CACHE_STEMS,

    # Trackers
    LIBROSA_ONSET_HOP_LENGTH,
    LIBROSA_ONSET_BACKTRACK,
    LIBROSA_ONSET_THRESHOLD,
    MADMOM_BEAT_MODEL,
    MADMOM_FPS,
    TRACKER_ENSEMBLE_AGREEMENT_THRESHOLD,

    # Anechoic MA
    ANECHOIC_DEFAULT_PROFILE,
    ANECHOIC_MAX_REVERB_MS,
    ANECHOIC_MIN_DIRECT_TO_REVERB_RATIO_DB,
    ANECHOIC_FFT_SIZE_FOR_REVERB,

    # Export
    MIDI_PPQN,
    MIDI_TEMPO_DEFAULT,
    MIDI_FORMAT_VERSION,
    JSON_PRETTY_PRINT,
    JSON_INCLUDE_FORENSIC_AUDIT,
    JSON_INCLUDE_MUSIC_BOX,
    JSON_MAX_FILE_SIZE_MB,
    HEARTBEAT_NOTE_PITCH,
    HEARTBEAT_NOTE_VELOCITY,
    HEARTBEAT_NOTE_DURATION_MS,
    OCTAVE_RESTORE_BASS,
    BASS_OCTAVE_SHIFT_SEMITONES,
    BASS_NOTE_RANGE_LOW_MIDI,
    BASS_NOTE_RANGE_HIGH_MIDI,

    # Logging
    MUSIC_BOX_LOG_PATH,
    MUSIC_BOX_BUFFER_SIZE,
    MUSIC_BOX_ROTATE_BYTES,
    MUSIC_BOX_MAX_BACKUPS,
    LOG_LEVEL,
    LOG_SHOW_TIMESTAMPS,
    LOG_SHOW_MEMORY_USAGE,
    LOG_SHOW_STAGE_DURATIONS,

    # Performance
    MAX_WORKERS,
    USE_GPU_IF_AVAILABLE,
    GPU_MEMORY_FRACTION,
    BATCH_SIZE_PITCH_DETECTION,
    BATCH_SIZE_DRUM_DETECTION,
    CACHE_SEPARATION_RESULTS,
    CACHE_PITCH_DETECTION,
    CACHE_TTL_SECONDS,

    # Epistemic Veto
    VETO_HIERARCHY,
    VETO_RECOVERY_STRATEGY,
    FALLBACK_SOURCE_ORDER,

    # Deprecated (4.7 compatibility)
    DEPRECATED_CONFIDENCE_ROUTER_THRESHOLDS,
)

# ============================================================================
# PART 9: PROTOCOLS (Legacy interfaces - need for StatusReporterProtocol)
# ============================================================================

from core.protocols import (
    # Base protocols
    AgentProtocol,
    MemoryManagedProtocol,
    ScribeValidatable,

    # Feature Bundle
    FeatureBundleProtocol,

    # Separation agents
    SeparationAgentProtocol,
    DemucsProtocol,
    BSRoformerProtocol,
    HybridSeparatorProtocol,

    # Detection agents
    DetectionAgentProtocol,
    RhythmEngineProtocol,
    PitchEngineProtocol,
    TonalDetectorProtocol,
    DrumIntelligenceProtocol,

    # Analysis agents
    GrooveFieldAnalyzerProtocol,
    VoiceContinuityProtocol,
    TempoIntelligenceProtocol,

    # Quantization agents
    RitornelloProtocol,
    QuantizationAgentProtocol,
    VelocityMergeProtocol,

    # Validation agents
    ValidationAgentProtocol,
    ScribeProtocol,
    ConsensusEngineProtocol,

    # Trackers
    OnsetTrackerProtocol,
    BeatTrackerProtocol,

    # Exporters
    MidiExporterProtocol,
    JsonExporterProtocol,

    # Orchestration
    PipelineStageProtocol,
    MemoryGuardianProtocol,
    OrchestratorProtocol,

    # Forensic
    MusicBoxProtocol,
    StatusReporterProtocol,  # <-- CRITICAL: Added for main.py

    # Factory
    AgentFactoryProtocol,

    # Utility
    AnechoicMaProtocol,
    MusicBrainzProtocol,
    HeartbeatGeneratorProtocol,

    # Deprecated (4.7 compatibility)
    ConfidenceRouterProtocol,
    FusionLayerProtocol,
    StateManagerProtocol,
)

# ============================================================================
# PART 10: GUIDED CONTROL (Human-in-the-loop)
# ============================================================================

from core.guided_control import (
    GenreType,
    TimeSignature,
    GuidedParams,
    GuidedBeatGrid,
    GuidedBeatGridGenerator,
    KeyConstraint,
    GuidedModeController,
    create_guided_params_from_form,
    get_scale_notes,
    validate_guided_params,
)

# ============================================================================
# PART 11: MODEL REGISTRY (Witness specifications)
# ============================================================================

from core.model_registry import (
    ModelDomain,
    StreamingCompatibility,
    OutputSemantics,
    ModelSpecification,
    CanonicalRegistry,
    get_model_summary,
    compare_models,
)

# ============================================================================
# PART 12: VERSION AND METADATA
# ============================================================================

__version__ = "5.6.1"

# ============================================================================
# PART 13: COMPLETE __all__ EXPORTS (what main.py expects)
# ============================================================================

__all__ = [
    # Version
    "__version__",

    # Order Types
    "Confidence",
    "VetoReason",
    "ValidationGate",
    "SourceType",
    "StemType",
    "NoteEvent",
    "AudioContext",
    "GrooveField",
    "PulseField",
    "TempoMap",
    "Voice",
    "VoiceContinuityResult",
    "WitnessTestimony",
    "ConsensusPackage",
    "ExportOptions",
    "SeparationResult",
    "StageResult",
    "ValidationResult",
    "AnechoicProfile",
    "AnechoicMask",
    "UNKNOWN_HASH_PLACEHOLDER",

    # Utility functions
    "beat_ms_to_bpm",
    "bpm_to_beat_ms",
    "hz_to_midi",
    "midi_to_hz",
    "ms_to_samples",
    "samples_to_ms",
    "validate_pitch",
    "validate_velocity",

    # Testimony
    "TestimonyStageResult",
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
    "TestimonyNormalizer",
    "normalizer",

    # Audio Views
    "ChannelLayout",
    "ImmutableAudio",
    "AudioViews",
    "create_audio_views",
    "create_stereo_views",
    "create_mono_views",
    "views_from_loaded_audio",

    # Contracts
    "AudioContract",
    "DetectionResult",
    "EvidenceType",
    "validate_audio_contract",
    "validate_detection_result",
    "validate_stem_contract",

    # Acoustic Intelligence
    "AcousticIntelligence",
    "AcousticChannelLayout",
    "AcousticImmutableAudio",
    "MIN_SAMPLE_RATE",
    "MAX_SAMPLE_RATE",
    "QUALITY_PRESETS",

    # Feature Bundle
    "FeatureBundle",
    "extract_features",
    "create_feature_bundle",
    "MADMOM_SR",
    "BASIC_PITCH_SR",
    "CREPE_SR",
    "ANALYSIS_SR",
    "DEMUCS_SR",
    "compute_audio_statistics",
    "compute_feature_statistics",
    "sanitize_audio",
    "safe_normalize",
    "validate_fft_size",
    "validate_sample_rate",
    "validate_audio_duration",

    # Evidence Tracker
    "EvidenceTracker",

    # Constants - CRITICAL for main.py
    "DEFAULT_PROCESSING_DURATION_SECONDS",
    "TRUNCATION_PRESETS",
    "validate_truncation_duration",
    "get_truncation_preset",
    "MEMORY_LIMIT_MB",
    "MEMORY_WARNING_THRESHOLD_MB",
    "MEMORY_CRITICAL_THRESHOLD_MB",
    "STAGGERED_GC_TRIGGER_MB",
    "MIN_CONFIDENCE_TO_PASS",
    "RITORNELLO_MAX_SNAP_MS",
    "HEARTBEAT_NOTE_PITCH",
    "HEARTBEAT_NOTE_VELOCITY",
    "HEARTBEAT_NOTE_DURATION_MS",
    "MIN_TEMPO_BPM",
    "MAX_TEMPO_BPM",
    "DEFAULT_TEMPO_BPM",
    "MIN_PITCH_MIDI",
    "MAX_PITCH_MIDI",
    "STAGE_TIMEOUT_LOAD_SECONDS",
    "STAGE_TIMEOUT_SEPARATION_SECONDS",
    "STAGE_TIMEOUT_DETECTION_SECONDS",
    "STAGE_TIMEOUT_ANALYSIS_SECONDS",
    "STAGE_TIMEOUT_QUANTIZATION_SECONDS",
    "STAGE_TIMEOUT_EXPORT_SECONDS",
    "TARGET_SAMPLE_RATE",
    "PROFESSIONAL_FIDELITY_SR",
    "TARGET_SAMPLE_RATE_DEPRECATED",

    # Mel-Roformer
    "MEL_ROFORMER_QUALITY_PROFILES",
    "get_mel_roformer_quality_profile",
    "MEL_ROFORMER_CHUNK_DURATION_SECONDS",
    "MEL_ROFORMER_OVERLAP_SECONDS",
    "MEL_ROFORMER_USE_CHUNKED_INFERENCE",

    # Demucs
    "DEMUCS_QUALITY_PROFILES",
    "get_demucs_quality_profile",
    "DEMUCS_SEGMENT_SECONDS",
    "DEMUCS_OVERLAP_SECONDS",

    # Protocols - CRITICAL for main.py
    "AgentProtocol",
    "MemoryManagedProtocol",
    "ScribeValidatable",
    "MusicBoxProtocol",
    "StatusReporterProtocol",  # <-- CRITICAL: Added for main.py

    # Guided Control
    "GuidedParams",
    "GuidedModeController",
    "KeyConstraint",
    "GuidedBeatGrid",
    "GuidedBeatGridGenerator",
    "GenreType",
    "TimeSignature",
    "create_guided_params_from_form",
    "get_scale_notes",
    "validate_guided_params",

    # Model Registry
    "CanonicalRegistry",
    "ModelSpecification",
    "ModelDomain",
    "get_model_summary",
    "compare_models",
]

# ============================================================================
# PART 14: MODULE DOCSTRING
# ============================================================================

__doc__ = """
Grimlock 5.5 Core Module
========================

This module exports all core types, constants, protocols, and utilities
needed by pipeline.py and main.py.

CRITICAL EXPORTS FOR main.py:
    - DEFAULT_PROCESSING_DURATION_SECONDS
    - StatusReporterProtocol
    - FeatureBundle
    - AcousticIntelligence
    - AudioViews, create_audio_views
    - CanonicalRegistry
    - ALL testimony classes
"""


# ============================================================================
# PART 15: BACKWARD COMPATIBILITY SHIMS
# ============================================================================

def __getattr__(name: str) -> Any:
    """
    Backward compatibility shims for 4.x / early 5.0 code.
    """
    legacy_renames = {
        "CONFIDENCE_THRESHOLD_LOW": "MIN_CONFIDENCE_TO_PASS",
        "TEMPO_MIN": "MIN_TEMPO_BPM",
        "TEMPO_MAX": "MAX_TEMPO_BPM",
        "GRID_SNAP_LIMIT_MS": "RITORNELLO_MAX_SNAP_MS",
        "TARGET_SAMPLE_RATE": "TARGET_SAMPLE_RATE_DEPRECATED",
    }

    if name in legacy_renames:
        warnings.warn(
            f"'{name}' is deprecated. Use '{legacy_renames[name]}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from core import constants
        return getattr(constants, legacy_renames[name])

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ============================================================================
# PART 16: LAZY LOADING HELPERS
# ============================================================================

def get_core_info() -> dict:
    """Get information about the core module."""
    return {
        "version": __version__,
        "exports_count": len(__all__),
        "export_categories": {
            "order_types": ["NoteEvent", "Confidence", "AudioContext"],
            "testimony": ["IngestionTestimony", "TempoTestimony", "TestimonyNormalizer"],
            "constants": ["DEFAULT_PROCESSING_DURATION_SECONDS", "MEMORY_LIMIT_MB"],
            "protocols": ["StatusReporterProtocol", "MusicBoxProtocol"],
            "guided_control": ["GuidedParams", "GuidedModeController"],
            "model_registry": ["CanonicalRegistry"],
        }
    }