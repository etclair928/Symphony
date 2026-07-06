# =================================================================
# MODULE: core/constants.py
# DESCRIPTION: Hardcoded thresholds, limits, and configuration values.
# NO LOGIC. Only constants. Imported by everything.
#
# These values come from empirical tuning of Grimlock 4.5-4.7.
# Changing these affects transcription behavior globally.
#
# DEPENDS ON: order_types.py (Confidence enum, VetoReason, etc.)
# DEPENDENTS: Everything.
#
# Laws encoded here:
#   - Confidence thresholds map to order_types.Confidence levels
#   - Veto rules reference order_types.VetoReason and ValidationGate
#   - All thresholds respect non-destructive quantization
#
# VERSION: 5.6.1 (Added QuaverIntelligence + PhraseIntelligence constants)
# UPDATED: 2026-06-02
# =================================================================

from core.order_types import Confidence, VetoReason, ValidationGate, SourceType, MergeStrategy

# ============================================================================
# LAW 1: RESOURCE SURVIVAL (The 4.5 Legacy)
# "The math must never exceed the machine."
# ============================================================================

# ===== MASTER SAMPLE RATES =====
# Primary working sample rate for the pipeline
TARGET_SAMPLE_RATE = 44100  # CD quality - used as master working rate

# ===== ANALYSIS SAMPLE RATES (5.0 Feature) =====
# Each processor has its optimal sample rate
# Arena handles resampling via borrow_mono(target_sr=xxx)

# Sample rate for spectral analysis (STFT, chroma, CQT)
# Balances frequency resolution and computational cost
ANALYSIS_SR = 22050  # Half of CD quality, sufficient for most analysis

# Sample rate for Basic Pitch (Spotify's model)
# Optimized for polyphonic instrument detection
BASIC_PITCH_SR = 22050

# Sample rate for CREPE pitch detection
# CREPE expects 16kHz for optimal monophonic pitch tracking
CREPE_SR = 16000

# Sample rate for Madmom tempo detection
# Higher rate for better onset timing accuracy
MADMOM_SR = 44100

# Sample rate for Demucs separation
# Demucs works best at original rate, but 44.1k is standard
DEMUCS_SR = 44100

# ===== LEGACY DEPRECATED =====
# DEPRECATED: Use TARGET_SAMPLE_RATE or specific processor SR instead
# Kept for backward compatibility only
TARGET_SAMPLE_RATE_DEPRECATED = 16000  # DO NOT USE - legacy fallback only

# Professional fidelity reference
PROFESSIONAL_FIDELITY_SR = 44100  # Reference for documentation only

MONO_DOWNMIX_FORCED = True  # Halves RAM footprint at load
STFT_HOP_LENGTH = 512  # 32ms hops at 16kHz (adjust based on sample rate)
STFT_WINDOW_SIZE = 2048  # 128ms windows
STFT_WINDOW_TYPE = "hann"  # Hann window for spectral analysis

# ===== TRUNCATION DEFAULTS (5.0 Feature) =====
# Default processing duration in seconds
# 0 = full song, >0 = truncate to first N seconds
DEFAULT_PROCESSING_DURATION_SECONDS: float = 30.0

# Available truncation presets for CLI
TRUNCATION_PRESETS = {
    "full": 0.0,  # Process entire song
    "30sec": 30.0,  # First 30 seconds
    "60sec": 60.0,  # First 60 seconds
    "90sec": 90.0,  # First 90 seconds
    "120sec": 120.0,  # First 120 seconds (2 minutes)
    "180sec": 180.0,  # First 180 seconds (3 minutes)
    "300sec": 300.0,  # First 300 seconds (5 minutes)
}

# Minimum duration for meaningful analysis (less than this fails fast)
MIN_TRUNCATION_DURATION_SECONDS: float = 5.0

# Maximum duration for truncation (longer than this triggers warning)
MAX_TRUNCATION_DURATION_SECONDS: float = 300.0  # 5 minutes max for analysis

# ===== MEMORY LIMITS (from MemoryGuardian) =====
MEMORY_LIMIT_MB = 2048  # 2GB max RSS (safety limit)
MEMORY_WARNING_THRESHOLD_MB = 1536  # 1.5GB triggers warning
MEMORY_CRITICAL_THRESHOLD_MB = 1840  # 1.8GB triggers forced GC
STAGGERED_GC_TRIGGER_MB = 512  # Force GC after 512MB allocated

BUFFER_LIFECYCLE_STAGES = [
    "source",  # Raw loaded audio
    "separated",  # Demucs/Mel-Roformer output
    "detected",  # Note events before validation
    "quantized"  # After Ritornello
]

# ===== BUFFER RETENTION POLICY =====
RETAIN_FULL_MIX_AFTER_SEPARATION = False  # Delete source after separation
RETAIN_SEPARATED_STEMS_AFTER_DETECTION = True  # Keep stems until detection done
MAX_CONCURRENT_BUFFERS = 3  # Never hold more than 3 audio buffers

# ===== STAGGERED DELETION TIMING =====
GC_COLLECT_AFTER_EACH_STAGE = True  # Explicit gc.collect() between stages
DELAY_BETWEEN_STAGGERED_DELETIONS_MS = 10  # Brief pause for system to breathe

# ============================================================================
# LAW 2: UNIDIRECTIONAL INTEGRITY (The 4.7 Legacy)
# "Data flows forward; dependencies never look back."
# ============================================================================

# ===== PIPELINE STAGE TIMEOUTS (seconds) =====
STAGE_TIMEOUT_LOAD_SECONDS = 30
STAGE_TIMEOUT_SEPARATION_SECONDS = 600  # Placeholder before audio duration is known;
                                        # the pipeline overwrites this dynamically once
                                        # ingestion reports the real clip length (see
                                        # SEPARATION_*_PER_AUDIO_SECOND below).
STAGE_TIMEOUT_DETECTION_SECONDS = 60
STAGE_TIMEOUT_ANALYSIS_SECONDS = 30
STAGE_TIMEOUT_QUANTIZATION_SECONDS = 10
STAGE_TIMEOUT_EXPORT_SECONDS = 10

# ===== SEPARATION TIMEOUT SCALING (Demucs -> Mel-Roformer fallback chain) =====
# Demucs (CPU, htdemucs, streaming) is always tried first; if it fails or
# times out, Mel-Roformer is tried next. Each model's budget scales with the
# audio duration being processed instead of a flat cutoff, since a fixed
# timeout that's tight enough for a 30s clip can be far too short for a
# 60s+ clip and starve the fallback chain of any time to run at all.
# Observed CPU-only htdemucs throughput has been ~9-10s of wall time per
# second of audio on this hardware; the multiplier below adds headroom.
DEMUCS_SECONDS_PER_AUDIO_SECOND = 12.0
DEMUCS_TIMEOUT_FLOOR_SECONDS = 180.0
DEMUCS_TIMEOUT_CAP_SECONDS = 2400.0

# Mel-Roformer has no measured throughput yet (Demucs has always succeeded
# in testing so far) - budget it slightly leaner than Demucs pending data.
MEL_ROFORMER_SECONDS_PER_AUDIO_SECOND = 8.0
MEL_ROFORMER_TIMEOUT_FLOOR_SECONDS = 120.0
MEL_ROFORMER_TIMEOUT_CAP_SECONDS = 1800.0

# Extra headroom added on top of (demucs_timeout + mel_roformer_timeout) for
# the outer per-stage watchdog, to cover STFT-fallback time and bookkeeping.
SEPARATION_STAGE_TIMEOUT_BUFFER_SECONDS = 60.0

# New stage timeouts for 5.6.1
STAGE_TIMEOUT_QUAVER_INTELLIGENCE_SECONDS = 60
STAGE_TIMEOUT_PHRASE_INTELLIGENCE_SECONDS = 60
STAGE_TIMEOUT_STRUCTURAL_FEEDBACK_SECONDS = 30

# ===== MAXIMUM PIPELINE RETRIES =====
# Stages cannot retry themselves - only Orchestrator
MAX_STAGE_RETRIES = 0  # No retries in 5.0 (fail fast)

# ===== CIRCULAR IMPORT PROTECTION =====
ALLOWED_CROSS_STAGE_IMPORTS = []  # EMPTY - no cross-stage imports allowed
CORE_MODULE_ALLOWED_IMPORTS = ["typing", "enum", "dataclasses", "datetime", "collections", "order_types"]

# ============================================================================
# LAW 3: SCRIBE'S TRUTH (Epistemic Veto)
# "Confidence must be earned, not assumed."
# ============================================================================

# ===== SCRIBE VALIDATION THRESHOLDS =====
# These map to order_types.Confidence values
MIN_CONFIDENCE_TO_PASS = Confidence.LOW.value  # 0.25 - below this = veto
SILENCE_RATIO_MAX = 0.90  # 90% silence → veto
ZERO_CROSSING_MAX = 0.35  # Above this = noise/snare, not tonal
MAX_HALLUCINATION_RATIO = 0.05  # Max 5% hallucinated notes allowed
MIN_EVENTS_REQUIRED = 1  # Empty result always vetoes
MAX_EVENTS_FOR_SANITY_CHECK = 10000  # More than this triggers warning

# ===== VALIDATION GATE TO VETO REASON MAPPING =====
# Used by Scribe to convert gate failures to specific veto reasons
VALIDATION_GATE_TO_VETO = {
    ValidationGate.SCHOENBERG_MIRROR: VetoReason.HARMONIC_SERIES_FAILURE,
    ValidationGate.HARMONIC_SERIES: VetoReason.HARMONIC_SERIES_FAILURE,
    ValidationGate.SILENCE_DETECTOR: VetoReason.EXCESS_SILENCE,
    ValidationGate.CONFIDENCE_THRESHOLD: VetoReason.CONFIDENCE_TOO_LOW,
    ValidationGate.PHASE_CONSISTENCY: VetoReason.PHASE_INCOHERENCE,
    ValidationGate.TEMPO_REASONABLENESS: VetoReason.TEMPO_OUTLIER,
}

# ===== PER-SOURCE CONFIDENCE WEIGHTS =====
# These multiply against the witness's raw confidence
CONFIDENCE_WEIGHTS = {
    SourceType.RHYTHM: 0.7,
    SourceType.PITCH: 0.8,
    SourceType.TONAL: 0.6,
    SourceType.DRUM_INTELLIGENCE: 0.75,
    SourceType.DEMUCS: 0.65,
    SourceType.MEL_ROFORMER: 0.80,  # Added for 5.0 - higher weight due to better separation
    SourceType.GROOVE_FIELD: 0.85,
    SourceType.RITORNELLO: 0.9,
}

# ===== MINIMUM QUALIFIED WITNESSES =====
MIN_QUALIFIED_WITNESSES_FOR_CONSENSUS = 2

# ============================================================================
# SCHOENBERG MIRROR (Harmonic Series Auditor)
# ============================================================================

# ===== HARMONIC SERIES DETECTION THRESHOLDS =====
HARMONIC_SERIES_MIN_PARTIALS = 2  # Need at least 2 partials to confirm pitch
HARMONIC_SERIES_MATCH_THRESHOLD = 0.7  # 70% of partials must match
MAX_INHARMONICITY = 0.05  # 5% max deviation from perfect harmonics
ZERO_CROSSING_HYSTERESIS_MS = 5  # 5ms hysteresis for zero-crossing detection
SPECTRAL_FLATNESS_TONAL_THRESHOLD = 0.3  # Below 0.3 = tonal
SPECTRAL_FLATNESS_NOISE_THRESHOLD = 0.7  # Above 0.7 = noise

# ===== SCHOENBERG VERDICT TO CONFIDENCE MAPPING =====
VERDICT_BASE_CONFIDENCE = {
    "harmonic_series_detected": Confidence.HIGH.value,  # 0.80
    "impulsive_no_harmonics": Confidence.LOW.value,  # 0.25
    "no_harmonic_structure": Confidence.HALLUCINATION.value,  # 0.0
    "borderline_case": Confidence.MEDIUM.value,  # 0.50
    "false_positive_detected": Confidence.HALLUCINATION.value,  # 0.0
}

# ============================================================================
# RELATIONAL PHYSICS (Groove Field & Pulse Field)
# ============================================================================

# ===== GROOVE FIELD THRESHOLDS =====
PHASE_DELTA_SWING_THRESHOLD_MS = 8.0  # >8ms = intentional "wide swing"
DILLA_POCKET_VARIANCE_MS = 4.0  # J Dilla pocket tolerance
MAX_GROOVE_SNAP_MS = 100  # Don't snap notes more than 100ms
MIN_PHASE_DELTA_TO_CONSIDER_MS = 2.0  # Less than 2ms is rounding error
INTER_NOTE_DISTANCE_HISTORY_WINDOW = 8  # Look at last 8 notes for pattern
GROOVE_MAX_PAIRING_DISTANCE_MS = 500.0  # Bass/kick farther apart than this
                                        # aren't a real rhythmic pair (e.g. a
                                        # bass note in a kick-less passage) -
                                        # exclude from the phase-delta average
                                        # instead of letting one outlier drag
                                        # a groove estimate to nonsense.

# ===== GROOVE FIELD CONFIDENCE MAPPING =====
GROOVE_FIELD_CONFIDENCE_THRESHOLDS = {
    "excellent_swing": Confidence.HIGH.value,  # > 0.80
    "detectable_swing": Confidence.MEDIUM.value,  # 0.50-0.80
    "straight": Confidence.LOW.value,  # 0.25-0.50
    "random": Confidence.HALLUCINATION.value,  # < 0.25
}

# ===== PULSE FIELD PARAMETERS =====
PULSE_FIELD_RESOLUTION_MS = 10  # 10ms grid for pulse field
MIN_TEMPO_BPM = 40  # Below 40 BPM is unlikely
MAX_TEMPO_BPM = 240  # Above 240 BPM is unlikely
DEFAULT_TEMPO_BPM = 120  # Fallback tempo
TEMPO_FORECAST_WINDOW_SECONDS = 4  # Look ahead 4 seconds
BEAT_TRACKING_HOP_MS = 20  # 20ms hops for beat tracking

# ===== TEMPO WITNESS WEIGHTING (tempo_analysis stage) =====
# ReverseGeoCrypt runs as an independent tempo witness alongside the
# TempoIntelligence ensemble (Librosa/Spectral/Madmom) - it votes on
# inter-onset ratio geometry rather than beat-tracking, so it's a
# genuinely different failure mode and useful for catching octave
# (half/double tempo) errors. Weighted slightly below the ensemble
# since it's a single method vs. a 3-witness ensemble.
TEMPO_INTELLIGENCE_ENSEMBLE_WEIGHT = 1.0
REVERSE_GEO_CRYPT_WITNESS_WEIGHT = 0.7

# tempo_analysis now runs TempoIntelligence's 3-witness ensemble (Librosa,
# Spectral autocorrelation, Madmom RNN) plus the ReverseGeoCrypt witness in
# sequence - a flat 60s timeout wasn't enough even for a 30s clip. Scale it
# with duration the same way separation's timeout is scaled. Measured ~118s
# for a 30s clip (a 3.93 ratio) with a 4.0 multiplier - that left only ~2s
# of margin before the stage's own timeout, so bumped to 6.0 for real headroom.
TEMPO_ANALYSIS_SECONDS_PER_AUDIO_SECOND = 6.0
TEMPO_ANALYSIS_TIMEOUT_FLOOR_SECONDS = 120.0
TEMPO_ANALYSIS_TIMEOUT_CAP_SECONDS = 900.0

# drum_detection uses the flat STAGE_TIMEOUT_DETECTION_SECONDS (60s) like
# every other detection stage, but its NMF/HFC coprocessor decomposition
# is far more expensive than the plain CREPE calls the other detection
# stages make - confirmed on a real full-length (294s) track: rhythm/
# pitch_bass/pitch_master/tonal all finished in 5-7s each, but
# drum_detection hit the 60s cap and returned zero drum events for the
# entire song. 30s-clip testing earlier this session measured ~13.2s
# (a 0.44 ratio) - scale with real headroom above that, same reasoning
# as separation/tempo_analysis above.
DRUM_DETECTION_SECONDS_PER_AUDIO_SECOND = 2.0
DRUM_DETECTION_TIMEOUT_FLOOR_SECONDS = 60.0
DRUM_DETECTION_TIMEOUT_CAP_SECONDS = 600.0

# ===== TIME SIGNATURE DEFAULTS =====
DEFAULT_TIME_SIGNATURE_NUMERATOR = 4
DEFAULT_TIME_SIGNATURE_DENOMINATOR = 4
TIME_SIGNATURE_CANDIDATE_LIMIT = 3

# ============================================================================
# QUAVER INTELLIGENCE (Symbolic Duration Witness) - NEW in 5.6.1
# ============================================================================

# ===== QUAVER DURATION HYPOTHESIS PARAMETERS =====
TEMPO_RELATIVE_TOLERANCE = 0.08  # 8% of beat = acceptable error for duration matching
TUPLET_TOLERANCE_FACTOR = 1.5  # Tuplets get 50% wider tolerance
SWING_THRESHOLD_MS = 15.0  # Min swing deviation to consider groove interpretation
SUSTAIN_RESONANCE_THRESHOLD = 0.30  # Resonance below this → no sustain hypothesis
ARTIFACT_OPPOSITION_THRESHOLD = 0.80  # Opposition above this → mark as artifact

# ===== QUAVER EPISTEMIC THRESHOLDS =====
MIN_HYPOTHESIS_PROBABILITY = 0.05  # Below this, discard hypothesis
HIGH_TENSION_THRESHOLD = 0.70  # Epistemic tension above this indicates strong contradiction
CRITICAL_TENSION_THRESHOLD = 0.85  # Above this triggers re-analysis
UNCERTAINTY_HIGH_THRESHOLD = 0.65  # Above this, flag as uncertain

# ===== QUAVER CONTENTION DETECTION =====
CONTENTION_PROBABILITY_GAP = 0.15  # If top two hypotheses within this gap, contention exists
MAX_HYPOTHESES_PER_NOTE = 5  # Maximum competing hypotheses per note
MIN_HYPOTHESIS_SUPPORT = 0.25  # Minimum support score to include hypothesis

# ===== QUAVER WITNESS RELIABILITY (Priors) =====
QUAVER_WITNESS_RELIABILITY = {
    "reverse_geo_crypt": 0.92,
    "pulse_field": 0.88,
    "groove_field": 0.85,
    "anechoic_ma": 0.91,
    "tempo_intelligence": 0.87,
    "drum_intelligence": 0.94,
    "voice_continuity": 0.83,
    "instrument_prior": 0.78,
}

# ===== QUAVER SYMBOLIC DURATION SPACE =====
# Multiples of subdivision for standard durations
SYMBOLIC_DURATION_MULTIPLES = {
    "whole": 4.0,
    "dotted_half": 3.0,
    "half": 2.0,
    "dotted_quarter": 1.5,
    "quarter": 1.0,
    "dotted_eighth": 0.75,
    "eighth": 0.5,
    "dotted_sixteenth": 0.375,
    "sixteenth": 0.25,
    "thirty_second": 0.125,
}

# ===== QUAVER TUPLET RATIOS =====
TUPLET_RATIOS = {
    "triplet": 2.0 / 3.0,
    "quintuplet": 4.0 / 5.0,
    "septuplet": 4.0 / 7.0,
    "compound_triplet": 3.0 / 4.0,
}

# ============================================================================
# PHRASE INTELLIGENCE (Structural Phase Engine) - NEW in 5.6.1
# ============================================================================

# ===== PHRASE BOUNDARY DETECTION =====
PHRASE_BOUNDARY_REST_THRESHOLD_BARS = 0.75  # Rest longer than 0.75 bars = boundary
PHRASE_BOUNDARY_PITCH_JUMP_THRESHOLD_SEMITONES = 7  # Jump > 7 semitones = potential boundary
PHRASE_BOUNDARY_MIN_GAP_MS = 30  # Minimum gap to consider as boundary
PHRASE_BOUNDARY_MAX_GAP_MS = 5000  # Maximum gap before forced boundary

# ===== PHRASE WINDOWING =====
STRUCTURAL_WINDOW_SIZE_MS = 2000  # 2-second analysis windows
STRUCTURAL_WINDOW_OVERLAP = 0.5  # 50% overlap between windows
STRUCTURAL_HISTORY_WINDOWS = 10  # Keep 10 windows in history (20 seconds)

# ===== STRUCTURAL INVARIANT THRESHOLDS =====
INVARIANT_HIGH_THRESHOLD = 0.70
INVARIANT_MEDIUM_THRESHOLD = 0.50
INVARIANT_LOW_THRESHOLD = 0.30

# ===== STRUCTURAL REGIME DETECTION =====
STABLE_PERIODIC_THRESHOLDS = {
    "formal_periodicity": 0.70,
    "motivic_persistence": 0.60,
    "surface_entropy": 0.40,
}

ELABORATIVE_THRESHOLDS = {
    "motivic_persistence": 0.40,
    "surface_entropy": 0.60,
    "harmonic_rigidity": 0.50,
}

DEVELOPMENTAL_THRESHOLDS = {
    "formal_periodicity": 0.50,
    "motivic_persistence_low": 0.30,
    "motivic_persistence_high": 0.70,
}

RECAPITULATIVE_THRESHOLDS = {
    "motivic_persistence": 0.70,
    "harmonic_rigidity": 0.60,
}

# ===== STRUCTURAL PHASE TRANSITION WEIGHTS =====
# Used in state machine for phase transitions
PHASE_TRANSITION_SELF_WEIGHT = 0.90  # Probability of staying in same phase
PHASE_TRANSITION_HIGH_WEIGHT = 0.85  # High-probability transitions
PHASE_TRANSITION_LOW_WEIGHT = 0.05  # Near-impossible transitions
PHASE_TRANSITION_MEDIUM_WEIGHT = 0.50  # Default for unknown transitions

# ===== ATTRACTOR BASIN PROTOTYPES (Genre Emergence) =====
# These are NOT genre detection - they're attractor basins for structural convergence
ATTRACTOR_POP = {
    "tonal_stability": 0.85,
    "temporal_cyclicity": 0.90,
    "transformational_depth": 0.20,
    "event_density": 0.70,
    "constraint_strength": 0.80,
    "surface_entropy": 0.25,
}

ATTRACTOR_JAZZ = {
    "tonal_stability": 0.70,
    "temporal_cyclicity": 0.60,
    "transformational_depth": 0.65,
    "event_density": 0.55,
    "constraint_strength": 0.50,
    "surface_entropy": 0.75,
}

ATTRACTOR_CLASSICAL = {
    "tonal_stability": 0.80,
    "temporal_cyclicity": 0.40,
    "transformational_depth": 0.75,
    "event_density": 0.50,
    "constraint_strength": 0.70,
    "surface_entropy": 0.40,
}

ATTRACTOR_AMBIENT = {
    "tonal_stability": 0.40,
    "temporal_cyclicity": 0.30,
    "transformational_depth": 0.30,
    "event_density": 0.20,
    "constraint_strength": 0.30,
    "surface_entropy": 0.35,
}

ATTRACTOR_BINARY = {
    "tonal_stability": 0.60,
    "temporal_cyclicity": 0.70,
    "transformational_depth": 0.30,
    "event_density": 0.60,
    "constraint_strength": 0.65,
    "surface_entropy": 0.30,
}

# ===== PHRASE FEEDBACK PARAMETERS =====
REANALYSIS_SEVERITY_HIGH = "high"
REANALYSIS_SEVERITY_MEDIUM = "medium"
REANALYSIS_SEVERITY_LOW = "low"
REANALYSIS_SEVERITY_NONE = "none"

REANALYSIS_PHRASE_OSCILLATION_THRESHOLD = 4  # More than 4 phases in 10 windows = oscillation
REANALYSIS_BOUNDARY_OVERLOAD_THRESHOLD = 0.5  # More than 50% strong boundaries = overload
REANALYSIS_IMPOSSIBLE_TRANSITION_THRESHOLD = 0.10  # Transition weight < 0.10 with confidence > 0.60

# ============================================================================
# RITORNELLO (Non-Destructive Quantization)
# ============================================================================

# ===== QUANTIZATION LIMITS =====
# Law: start_ms NEVER overwritten
RITORNELLO_MAX_SNAP_MS = 100  # Never snap more than 100ms
PRESERVE_ORIGINAL_MS_ALWAYS = True  # Non-destructive is law
MIN_SNAP_DISTANCE_TO_CONSIDER_MS = 5  # Don't snap if change <5ms
SNAP_CONFIDENCE_PENALTY_PER_MS = 0.001  # -0.1 confidence for 100ms snap
MAX_CONSECUTIVE_SNAPS = 50  # Audit limit per session

# ===== MULTI-STAGE QUANTIZATION =====
QUANTIZATION_GRID_RESOLUTIONS_MS = {
    "coarse": 500,  # Quarter note at 120 BPM
    "medium": 250,  # Eighth note
    "fine": 125,  # 16th note
    "micro": 62.5,  # 32nd note
}
QUANTIZATION_STAGE_CONFIDENCE_PENALTIES = {
    "coarse": 0.01,
    "medium": 0.02,
    "fine": 0.03,
    "micro": 0.05,
}

# ===== MINIMUM CONFIDENCE TO QUANTIZE =====
# Below this, snapped_* fields remain None
MIN_CONFIDENCE_FOR_QUANTIZATION = Confidence.MEDIUM.value  # 0.50

# ============================================================================
# VELOCITY MERGE (Voice & Note Merging)
# ============================================================================

# ===== NOTE MERGING THRESHOLDS =====
OVERLAP_TOLERANCE_MS = 10  # Notes within 10ms considered overlapping
DEFAULT_MERGE_STRATEGY = MergeStrategy.WEIGHTED_AVERAGE
MIN_VELOCITY_TO_KEEP = 1  # Below 1 is silence (MIDI standard)
MAX_VELOCITY = 127
DEFAULT_GHOST_NOTE_VELOCITY = 20  # Quiet notes are ghost notes

# ===== VOICE CONTINUITY =====
MAX_VOICE_GAP_MS = 50  # Gap >50ms is new voice
MIN_NOTE_DURATION_MS = 20  # Shorter than 20ms is probably noise
MAX_OCTAVE_JUMP_SEMITONES = 14  # Jump > octave+2 is suspicious
VOICE_CROSSING_DETECTION_WINDOW_MS = 100

# ============================================================================
# DRUM INTELLIGENCE (SPICE-based)
# ============================================================================

# ===== SPICE MODEL DEFAULTS =====
SPICE_MODEL_SIZE = "medium"  # "small", "medium", "large"
SPICE_HOP_LENGTH_MS = 10
SPICE_ONSET_THRESHOLD = 0.5

# ===== DRUM DETECTION THRESHOLDS =====
DRUM_CONFIDENCE_MIN = Confidence.LOW.value  # 0.25
DRUM_GHOST_NOTE_VELOCITY_MAX = 30
DRUM_FLAM_DETECTION_WINDOW_MS = 50  # Two hits within 50ms is a flam
DRUM_ROLL_DETECTION_MIN_HITS = 4
DRUM_ROLL_MAX_INTERVAL_MS = 80

# ============================================================================
# PITCH INTELLIGENCE (Basic Pitch & Tonal Detection)
# ============================================================================

# ===== PITCH DETECTION BOUNDS =====
MIN_PITCH_MIDI = 21  # A0 (lowest piano)
MAX_PITCH_MIDI = 108  # C8 (highest piano)
MIN_FREQUENCY_HZ = 27.5  # A0
MAX_FREQUENCY_HZ = 4186.01  # C8

# ===== TONAL DETECTION PARAMETERS =====
TONAL_CORRELATION_THRESHOLD = 0.6
TONAL_HARMONIC_WEIGHT_DECAY = 0.5  # Each harmonic half the weight
TONAL_MAX_HARMONICS = 8

# ===== BASIC PITCH SPECIFIC =====
BASIC_PITCH_MODEL = "spotify_basic_pitch"
BASIC_PITCH_HOP_MS = 32
BASIC_PITCH_MIN_FREQ_HZ = 50
BASIC_PITCH_MAX_FREQ_HZ = 2000

# ============================================================================
# MEL-ROFORMER CONFIGURATION (5.0 - Replacement for BS-Roformer)
# ============================================================================

# ===== MEL-ROFORMER MODEL DEFAULTS =====
MEL_ROFORMER_MODEL_NAME = "mel_roformer_16k"
MEL_ROFORMER_N_MELS = 128  # Mel bands (64/128/256 - 128 is sweet spot)
MEL_ROFORMER_SAMPLE_RATE = 16000  # 16kHz optimal for mel processing
MEL_ROFORMER_DEVICE = "auto"  # "auto", "cuda", "cpu"

# ===== MEL-ROFORMER ARCHITECTURE =====
MEL_ROFORMER_NUM_LAYERS = 6  # Transformer layers
MEL_ROFORMER_HIDDEN_SIZE = 512  # Hidden dimension
MEL_ROFORMER_NUM_HEADS = 8  # Attention heads
MEL_ROFORMER_DROPOUT = 0.1

# ===== MEL-ROFORMER STFT PARAMETERS =====
MEL_ROFORMER_N_FFT = 2048
MEL_ROFORMER_HOP_LENGTH = 512  # 32ms hops at 16kHz
MEL_ROFORMER_WINDOW_TYPE = "hann"

# ===== MEL-ROFORMER STEM CONFIGURATION =====
MEL_ROFORMER_STEMS = ["drums", "bass", "other", "vocals"]  # Primary stems
MEL_ROFORMER_INCLUDE_RESIDUAL_STEM = True  # Preserve ambience/unassigned energy
MEL_ROFORMER_STEREO_MODE = "mid_side"  # "mid_side", "stereo", "mono_fallback"

# ===== CHUNKED INFERENCE (Memory Management) =====
MEL_ROFORMER_CHUNK_DURATION_SECONDS = 30.0  # Process 30-second chunks
MEL_ROFORMER_OVERLAP_SECONDS = 5.0  # 5-second overlap between chunks
MEL_ROFORMER_USE_CHUNKED_INFERENCE = True  # Enabled by default for long files
MEL_ROFORMER_MAX_DURATION_SECONDS = 300.0  # 5 minutes max per file

# ===== PERFORMANCE & MEMORY =====
MEL_ROFORMER_TIMEOUT_SECONDS = 120.0  # Max time to wait for separation
MEL_ROFORMER_FORCE_CPU = False  # Can use GPU efficiently
MEL_ROFORMER_GPU_MEMORY_FRACTION = 0.5  # Use 50% of GPU memory max
MEL_ROFORMER_BATCH_SIZE = 1  # Process one chunk at a time

# ===== NUMERICAL STABILITY =====
MEL_ROFORMER_EPSILON = 1e-8  # Avoid division by zero
MEL_ROFORMER_DETERMINISTIC = True  # Forensic reproducibility
MEL_ROFORMER_RANDOM_SEED = 42

# ===== CONFIDENCE METRICS THRESHOLDS =====
MEL_ROFORMER_MIN_SPECTRAL_COHERENCE = 0.3  # Below this = unstable stem
MEL_ROFORMER_MAX_ACCEPTABLE_BLEED = 0.4  # Above 40% bleed = poor separation
MEL_ROFORMER_MAX_RECONSTRUCTION_ERROR = 0.25  # Above 25% error = hallucination
MEL_ROFORMER_MAX_MASK_ENTROPY = 0.7  # Above 0.7 = uncertain separation

# ===== FALLBACK BEHAVIOR =====
MEL_ROFORMER_FALLBACK_ENABLED = True  # Use simple frequency-based separation if model fails
MEL_ROFORMER_FALLBACK_MODE = "frequency_band"  # "frequency_band", "silence", "passthrough"

# ===== QUALITY PROFILES =====
MEL_ROFORMER_QUALITY_PROFILES = {
    "fast": {
        "n_mels": 64,
        "num_layers": 4,
        "hidden_size": 256,
        "num_heads": 4,
        "chunk_duration_seconds": 15.0,
    },
    "balanced": {
        "n_mels": 128,
        "num_layers": 6,
        "hidden_size": 512,
        "num_heads": 8,
        "chunk_duration_seconds": 30.0,
    },
    "high": {
        "n_mels": 256,
        "num_layers": 8,
        "hidden_size": 768,
        "num_heads": 12,
        "chunk_duration_seconds": 60.0,
    }
}

# ============================================================================
# DEMUCS CONFIGURATION (Legacy, maintained for compatibility)
# ============================================================================

# ===== DEMUCS MODEL DEFAULTS =====
DEMUCS_MODEL = "htdemucs"  # Hybrid Transformer Demucs
DEMUCS_SAMPLE_RATE = 44100  # Demucs works best at original rate
DEMUCS_DEVICE = "auto"  # "auto", "cuda", "cpu"

# ===== DEMUCS PROCESSING =====
DEMUCS_SEGMENT_SECONDS = 10.0  # Process in 10-second chunks
DEMUCS_OVERLAP_SECONDS = 0.5  # 500ms overlap between chunks
DEMUCS_STEMS = ["drums", "bass", "other"]  # We don't need vocals usually
DEMUCS_INCLUDE_RESIDUAL_STEM = False  # Demucs doesn't produce residual

# ===== PERFORMANCE & MEMORY =====
DEMUCS_MAX_DURATION_SECONDS = 60.0  # Max audio length for Demucs
DEMUCS_TIMEOUT_SECONDS = 180.0  # Max time to wait for Demucs
DEMUCS_FORCE_CPU = True  # Use CPU to avoid GPU memory issues
DEMUCS_GPU_MEMORY_FRACTION = 0.6  # Use 60% of GPU memory max

# ===== STREAMING MODE (for long files) =====
DEMUCS_WINDOW_SECONDS = 3.0  # Streaming window size (3 seconds)
DEMUCS_HOP_SECONDS = 1.5  # Hop between windows (1.5 seconds)

# ===== FALLBACK =====
DEMUCS_FALLBACK_ENABLED = True
DEMUCS_FALLBACK_MODE = "mel_roformer"  # Fallback to Mel-Roformer if Demucs fails

# ===== QUALITY PROFILES =====
DEMUCS_QUALITY_PROFILES = {
    "fast": {
        "model": "htdemucs_ft",  # Fine-tuned, smaller
        "segment_seconds": 6.0,
    },
    "balanced": {
        "model": "htdemucs",
        "segment_seconds": 10.0,
    },
    "high": {
        "model": "htdemucs",
        "segment_seconds": 20.0,
    }
}

# ============================================================================
# BS-ROFORMER DEPRECATION (5.0 → 5.1 Migration)
# ============================================================================

# BS-Roformer is deprecated in 5.0, removed in 5.1
BS_ROFORMER_DEPRECATED = True
BS_ROFORMER_MIGRATION_WARNING = "BS-Roformer is deprecated in 5.0, removed in 5.1. Use Mel-Roformer."
BS_ROFORMER_MODEL = None  # No longer loaded by default
BS_ROFORMER_HOP_LENGTH_SAMPLES = None
BS_ROFORMER_WINDOW_SIZE_SAMPLES = None

# ============================================================================
# UNIFIED SEPARATION FALLBACK CONFIGURATION
# ============================================================================

# Order of fallback attempts when primary separator fails
SEPARATION_FALLBACK_ORDER = [
    "mel_roformer",  # Primary in 5.0
    "demucs",  # Fallback for complex mixes
    "frequency_band",  # Simple frequency-based separation
    "passthrough"  # Last resort: return original audio as "other"
]

# Whether to cache separation results per file (hash-based)
SEPARATION_CACHE_ENABLED = False  # Too big to cache usually
SEPARATION_CACHE_MAX_SIZE_MB = 500  # 500MB max cache

# Whether to validate separation results (sanity checks)
SEPARATION_VALIDATE_OUTPUT = True
SEPARATION_MIN_STEM_ENERGY_RATIO = 0.01  # Stem must have at least 1% of original energy
SEPARATION_MAX_STEM_ENERGY_RATIO = 1.1  # Stem can't exceed original by more than 10%

# ============================================================================
# HYBRID SEPARATOR (Legacy)
# ============================================================================

HYBRID_SEPARATOR_QUALITY = "balanced"  # "fast", "balanced", "high"
HYBRID_CACHE_STEMS = True  # Cache stems to disk to save RAM

# ============================================================================
# TRACKER DEFAULTS (Librosa & Madmom)
# ============================================================================

# ===== LIBROSA ONSET DETECTION =====
LIBROSA_ONSET_HOP_LENGTH = 512
LIBROSA_ONSET_BACKTRACK = True
LIBROSA_ONSET_THRESHOLD = 0.5

# ===== MADMOM BEAT TRACKING =====
MADMOM_BEAT_MODEL = "dbpnn"  # Deep beat tracking
MADMOM_FPS = 100  # Frames per second for processing

# ===== ENSEMBLE TRACKING =====
TRACKER_ENSEMBLE_AGREEMENT_THRESHOLD = 0.7  # 70% agreement to use ensemble

# ============================================================================
# ANECHOIC MA (De-reverberation)
# ============================================================================

ANECHOIC_DEFAULT_PROFILE = "auto_detected"
ANECHOIC_MAX_REVERB_MS = 5000  # 5 seconds max reverb tail
ANECHOIC_MIN_DIRECT_TO_REVERB_RATIO_DB = -20  # Below -20dB is too wet
ANECHOIC_FFT_SIZE_FOR_REVERB = 4096

# ============================================================================
# EXPORT DEFAULTS
# ============================================================================

# ===== MIDI EXPORT =====
MIDI_PPQN = 480  # Pulses per quarter note
MIDI_TEMPO_DEFAULT = 500000  # 120 BPM in microseconds per quarter
MIDI_FORMAT_VERSION = 1  # Type 1 MIDI (multi-track)

# ===== JSON EXPORT =====
JSON_PRETTY_PRINT = True
JSON_INCLUDE_FORENSIC_AUDIT = True
JSON_INCLUDE_MUSIC_BOX = True
JSON_MAX_FILE_SIZE_MB = 100

# ===== HEARTBEAT NOTE (for empty transcriptions - Law 3 compliance) =====
HEARTBEAT_NOTE_PITCH = 60  # C4
HEARTBEAT_NOTE_VELOCITY = 10  # Barely audible (not a real note)
HEARTBEAT_NOTE_DURATION_MS = 500

# ===== OCTAVE RESTORATION =====
OCTAVE_RESTORE_BASS = True
BASS_OCTAVE_SHIFT_SEMITONES = -12  # Shift bass down one octave
BASS_NOTE_RANGE_LOW_MIDI = 28  # E1
BASS_NOTE_RANGE_HIGH_MIDI = 52  # E3

# ============================================================================
# LOGGING & FORENSICS
# ============================================================================

# ===== MUSIC BOX (forensic flight recorder) =====
MUSIC_BOX_LOG_PATH = "music_box.jsonl"
MUSIC_BOX_BUFFER_SIZE = 100  # Flush after 100 entries
MUSIC_BOX_ROTATE_BYTES = 100 * 1024 * 1024  # 100MB rotation
MUSIC_BOX_MAX_BACKUPS = 5

# ===== CONSOLE LOGGING =====
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_SHOW_TIMESTAMPS = True
LOG_SHOW_MEMORY_USAGE = True  # Memory pressure logging
LOG_SHOW_STAGE_DURATIONS = True

# ============================================================================
# PERFORMANCE TUNING (from 4.7 empirical optimization)
# ============================================================================

# ===== PARALLEL PROCESSING =====
MAX_WORKERS = 4  # CPU threads for parallel stages
USE_GPU_IF_AVAILABLE = True  # For SPICE, Demucs, Mel-Roformer
GPU_MEMORY_FRACTION = 0.6  # Use 60% of GPU memory max

# PyTorch defaults to spawning one thread per CPU core for its intra-op
# matrix math, which can pin every core during CPU-only Demucs inference
# and starve the rest of the system (including this process's own
# orchestration/monitoring code) of scheduling time. Clamp it.
TORCH_CPU_INTRAOP_THREADS = 2
TORCH_CPU_INTEROP_THREADS = 2

# ===== BATCH PROCESSING =====
BATCH_SIZE_PITCH_DETECTION = 32  # Frames per batch
BATCH_SIZE_DRUM_DETECTION = 16

# ===== CACHING =====
CACHE_SEPARATION_RESULTS = False  # Too big to cache
CACHE_PITCH_DETECTION = True  # Small enough to cache
CACHE_TTL_SECONDS = 3600  # 1 hour cache

# ============================================================================
# EPISTEMIC VETO RULES (From 4.7 ConsensusEngine)
# ============================================================================

# ===== VETO HIERARCHY =====
# Which sources can veto which other sources
# "*" means any source
VETO_HIERARCHY = {
    SourceType.SCRIBE: ["*"],  # Scribe can veto ANYONE
    SourceType.CONSENSUS_ENGINE: ["*"],
}

# ===== VETO RECOVERY =====
VETO_RECOVERY_STRATEGY = "fail_fast"  # "fail_fast", "fallback_to_secondary"
FALLBACK_SOURCE_ORDER = [
    SourceType.PITCH,
    SourceType.TONAL,
    SourceType.RHYTHM
]

# ============================================================================
# INSTRUMENT PRIORS (for QuaverIntelligence)
# ============================================================================

# Sustain and articulation priors per instrument family
INSTRUMENT_SUSTAIN_PRIORS = {
    "cello": 0.85,
    "violin": 0.80,
    "double_bass": 0.90,
    "piano": 0.30,
    "guitar": 0.40,
    "flute": 0.75,
    "trumpet": 0.70,
    "drum": 0.10,
    "strings": 0.82,
    "brass": 0.65,
    "unknown": 0.50,
}

INSTRUMENT_ARTICULATION_CLARITY = {
    "cello": 0.60,
    "violin": 0.65,
    "double_bass": 0.50,
    "piano": 0.90,
    "guitar": 0.85,
    "flute": 0.70,
    "trumpet": 0.75,
    "drum": 0.95,
    "strings": 0.62,
    "brass": 0.78,
    "unknown": 0.70,
}

# ============================================================================
# DEPRECATED CONSTANTS (4.7 compatibility - DO NOT USE IN NEW CODE)
# ============================================================================

# These are kept for backward compatibility but will be removed in 5.1
DEPRECATED_CONFIDENCE_ROUTER_THRESHOLDS = {
    "legacy_high": 0.8,
    "legacy_medium": 0.5,
    "legacy_low": 0.2,
}


# ============================================================================
# TRUNCATION VALIDATION
# ============================================================================

def validate_truncation_duration(duration: float) -> float:
    """
    Validate and clamp truncation duration.

    Args:
        duration: Requested duration in seconds (0 = full song)

    Returns:
        Validated duration (clamped to valid range)
    """
    if duration == 0:
        return 0.0

    if duration < MIN_TRUNCATION_DURATION_SECONDS:
        import warnings
        warnings.warn(
            f"Truncation duration {duration}s is below minimum {MIN_TRUNCATION_DURATION_SECONDS}s. "
            f"Using minimum value.",
            UserWarning
        )
        return MIN_TRUNCATION_DURATION_SECONDS

    if duration > MAX_TRUNCATION_DURATION_SECONDS:
        import warnings
        warnings.warn(
            f"Truncation duration {duration}s exceeds maximum {MAX_TRUNCATION_DURATION_SECONDS}s. "
            f"Clipping to maximum.",
            UserWarning
        )
        return MAX_TRUNCATION_DURATION_SECONDS

    return duration


def get_truncation_preset(preset_name: str) -> float:
    """
    Get truncation duration from preset name.

    Args:
        preset_name: Preset name (e.g., "30sec", "60sec", "full")

    Returns:
        Duration in seconds (0 = full song)
    """
    if preset_name not in TRUNCATION_PRESETS:
        raise ValueError(f"Unknown truncation preset: {preset_name}. "
                         f"Available: {list(TRUNCATION_PRESETS.keys())}")
    return TRUNCATION_PRESETS[preset_name]


def get_mel_roformer_quality_profile(profile_name: str = "balanced") -> dict:
    """
    Get Mel-Roformer configuration for a quality profile.

    Args:
        profile_name: "fast", "balanced", or "high"

    Returns:
        Dictionary of configuration overrides
    """
    if profile_name not in MEL_ROFORMER_QUALITY_PROFILES:
        raise ValueError(f"Unknown quality profile: {profile_name}. "
                         f"Available: {list(MEL_ROFORMER_QUALITY_PROFILES.keys())}")
    return MEL_ROFORMER_QUALITY_PROFILES[profile_name].copy()


def get_demucs_quality_profile(profile_name: str = "balanced") -> dict:
    """
    Get Demucs configuration for a quality profile.

    Args:
        profile_name: "fast", "balanced", or "high"

    Returns:
        Dictionary of configuration overrides
    """
    if profile_name not in DEMUCS_QUALITY_PROFILES:
        raise ValueError(f"Unknown quality profile: {profile_name}. "
                         f"Available: {list(DEMUCS_QUALITY_PROFILES.keys())}")
    return DEMUCS_QUALITY_PROFILES[profile_name].copy()


# ============================================================================
# EXPORTS (for __init__.py)
# ============================================================================

__all__ = [
    # Sample rates
    "TARGET_SAMPLE_RATE",
    "ANALYSIS_SR",
    "BASIC_PITCH_SR",
    "CREPE_SR",
    "MADMOM_SR",
    "DEMUCS_SR",
    "TARGET_SAMPLE_RATE_DEPRECATED",
    "PROFESSIONAL_FIDELITY_SR",

    # STFT
    "MONO_DOWNMIX_FORCED",
    "STFT_HOP_LENGTH",
    "STFT_WINDOW_SIZE",
    "STFT_WINDOW_TYPE",

    # Truncation
    "DEFAULT_PROCESSING_DURATION_SECONDS",
    "TRUNCATION_PRESETS",
    "MIN_TRUNCATION_DURATION_SECONDS",
    "MAX_TRUNCATION_DURATION_SECONDS",
    "validate_truncation_duration",
    "get_truncation_preset",

    # Memory
    "MEMORY_LIMIT_MB",
    "MEMORY_WARNING_THRESHOLD_MB",
    "MEMORY_CRITICAL_THRESHOLD_MB",
    "STAGGERED_GC_TRIGGER_MB",
    "BUFFER_LIFECYCLE_STAGES",
    "RETAIN_FULL_MIX_AFTER_SEPARATION",
    "RETAIN_SEPARATED_STEMS_AFTER_DETECTION",
    "MAX_CONCURRENT_BUFFERS",
    "GC_COLLECT_AFTER_EACH_STAGE",
    "DELAY_BETWEEN_STAGGERED_DELETIONS_MS",

    # Timeouts
    "STAGE_TIMEOUT_LOAD_SECONDS",
    "STAGE_TIMEOUT_SEPARATION_SECONDS",
    "STAGE_TIMEOUT_DETECTION_SECONDS",
    "STAGE_TIMEOUT_ANALYSIS_SECONDS",
    "STAGE_TIMEOUT_QUANTIZATION_SECONDS",
    "STAGE_TIMEOUT_EXPORT_SECONDS",
    "STAGE_TIMEOUT_QUAVER_INTELLIGENCE_SECONDS",
    "STAGE_TIMEOUT_PHRASE_INTELLIGENCE_SECONDS",
    "STAGE_TIMEOUT_STRUCTURAL_FEEDBACK_SECONDS",
    "DEMUCS_SECONDS_PER_AUDIO_SECOND",
    "DEMUCS_TIMEOUT_FLOOR_SECONDS",
    "DEMUCS_TIMEOUT_CAP_SECONDS",
    "MEL_ROFORMER_SECONDS_PER_AUDIO_SECOND",
    "MEL_ROFORMER_TIMEOUT_FLOOR_SECONDS",
    "MEL_ROFORMER_TIMEOUT_CAP_SECONDS",
    "SEPARATION_STAGE_TIMEOUT_BUFFER_SECONDS",
    "MAX_STAGE_RETRIES",
    "ALLOWED_CROSS_STAGE_IMPORTS",
    "CORE_MODULE_ALLOWED_IMPORTS",

    # Scribe validation
    "MIN_CONFIDENCE_TO_PASS",
    "SILENCE_RATIO_MAX",
    "ZERO_CROSSING_MAX",
    "MAX_HALLUCINATION_RATIO",
    "MIN_EVENTS_REQUIRED",
    "MAX_EVENTS_FOR_SANITY_CHECK",
    "VALIDATION_GATE_TO_VETO",
    "CONFIDENCE_WEIGHTS",
    "MIN_QUALIFIED_WITNESSES_FOR_CONSENSUS",

    # Schoenberg Mirror
    "HARMONIC_SERIES_MIN_PARTIALS",
    "HARMONIC_SERIES_MATCH_THRESHOLD",
    "MAX_INHARMONICITY",
    "ZERO_CROSSING_HYSTERESIS_MS",
    "SPECTRAL_FLATNESS_TONAL_THRESHOLD",
    "SPECTRAL_FLATNESS_NOISE_THRESHOLD",
    "VERDICT_BASE_CONFIDENCE",

    # Groove Field
    "PHASE_DELTA_SWING_THRESHOLD_MS",
    "DILLA_POCKET_VARIANCE_MS",
    "MAX_GROOVE_SNAP_MS",
    "MIN_PHASE_DELTA_TO_CONSIDER_MS",
    "INTER_NOTE_DISTANCE_HISTORY_WINDOW",
    "GROOVE_MAX_PAIRING_DISTANCE_MS",
    "GROOVE_FIELD_CONFIDENCE_THRESHOLDS",

    # Pulse Field
    "PULSE_FIELD_RESOLUTION_MS",
    "MIN_TEMPO_BPM",
    "MAX_TEMPO_BPM",
    "DEFAULT_TEMPO_BPM",
    "TEMPO_FORECAST_WINDOW_SECONDS",
    "BEAT_TRACKING_HOP_MS",
    "TEMPO_INTELLIGENCE_ENSEMBLE_WEIGHT",
    "REVERSE_GEO_CRYPT_WITNESS_WEIGHT",
    "TEMPO_ANALYSIS_SECONDS_PER_AUDIO_SECOND",
    "TEMPO_ANALYSIS_TIMEOUT_FLOOR_SECONDS",
    "TEMPO_ANALYSIS_TIMEOUT_CAP_SECONDS",
    "DRUM_DETECTION_SECONDS_PER_AUDIO_SECOND",
    "DRUM_DETECTION_TIMEOUT_FLOOR_SECONDS",
    "DRUM_DETECTION_TIMEOUT_CAP_SECONDS",
    "DEFAULT_TIME_SIGNATURE_NUMERATOR",
    "DEFAULT_TIME_SIGNATURE_DENOMINATOR",
    "TIME_SIGNATURE_CANDIDATE_LIMIT",

    # Quaver Intelligence (NEW)
    "TEMPO_RELATIVE_TOLERANCE",
    "TUPLET_TOLERANCE_FACTOR",
    "SWING_THRESHOLD_MS",
    "SUSTAIN_RESONANCE_THRESHOLD",
    "ARTIFACT_OPPOSITION_THRESHOLD",
    "MIN_HYPOTHESIS_PROBABILITY",
    "HIGH_TENSION_THRESHOLD",
    "CRITICAL_TENSION_THRESHOLD",
    "UNCERTAINTY_HIGH_THRESHOLD",
    "CONTENTION_PROBABILITY_GAP",
    "MAX_HYPOTHESES_PER_NOTE",
    "MIN_HYPOTHESIS_SUPPORT",
    "QUAVER_WITNESS_RELIABILITY",
    "SYMBOLIC_DURATION_MULTIPLES",
    "TUPLET_RATIOS",

    # Phrase Intelligence (NEW)
    "PHRASE_BOUNDARY_REST_THRESHOLD_BARS",
    "PHRASE_BOUNDARY_PITCH_JUMP_THRESHOLD_SEMITONES",
    "PHRASE_BOUNDARY_MIN_GAP_MS",
    "PHRASE_BOUNDARY_MAX_GAP_MS",
    "STRUCTURAL_WINDOW_SIZE_MS",
    "STRUCTURAL_WINDOW_OVERLAP",
    "STRUCTURAL_HISTORY_WINDOWS",
    "INVARIANT_HIGH_THRESHOLD",
    "INVARIANT_MEDIUM_THRESHOLD",
    "INVARIANT_LOW_THRESHOLD",
    "STABLE_PERIODIC_THRESHOLDS",
    "ELABORATIVE_THRESHOLDS",
    "DEVELOPMENTAL_THRESHOLDS",
    "RECAPITULATIVE_THRESHOLDS",
    "PHASE_TRANSITION_SELF_WEIGHT",
    "PHASE_TRANSITION_HIGH_WEIGHT",
    "PHASE_TRANSITION_LOW_WEIGHT",
    "PHASE_TRANSITION_MEDIUM_WEIGHT",
    "ATTRACTOR_POP",
    "ATTRACTOR_JAZZ",
    "ATTRACTOR_CLASSICAL",
    "ATTRACTOR_AMBIENT",
    "ATTRACTOR_BINARY",
    "REANALYSIS_SEVERITY_HIGH",
    "REANALYSIS_SEVERITY_MEDIUM",
    "REANALYSIS_SEVERITY_LOW",
    "REANALYSIS_SEVERITY_NONE",
    "REANALYSIS_PHRASE_OSCILLATION_THRESHOLD",
    "REANALYSIS_BOUNDARY_OVERLOAD_THRESHOLD",
    "REANALYSIS_IMPOSSIBLE_TRANSITION_THRESHOLD",

    # Instrument Priors (NEW)
    "INSTRUMENT_SUSTAIN_PRIORS",
    "INSTRUMENT_ARTICULATION_CLARITY",

    # Ritornello
    "RITORNELLO_MAX_SNAP_MS",
    "PRESERVE_ORIGINAL_MS_ALWAYS",
    "MIN_SNAP_DISTANCE_TO_CONSIDER_MS",
    "SNAP_CONFIDENCE_PENALTY_PER_MS",
    "MAX_CONSECUTIVE_SNAPS",
    "QUANTIZATION_GRID_RESOLUTIONS_MS",
    "QUANTIZATION_STAGE_CONFIDENCE_PENALTIES",
    "MIN_CONFIDENCE_FOR_QUANTIZATION",

    # Velocity Merge
    "OVERLAP_TOLERANCE_MS",
    "DEFAULT_MERGE_STRATEGY",
    "MIN_VELOCITY_TO_KEEP",
    "MAX_VELOCITY",
    "DEFAULT_GHOST_NOTE_VELOCITY",
    "MAX_VOICE_GAP_MS",
    "MIN_NOTE_DURATION_MS",
    "MAX_OCTAVE_JUMP_SEMITONES",
    "VOICE_CROSSING_DETECTION_WINDOW_MS",

    # Drum Intelligence
    "SPICE_MODEL_SIZE",
    "SPICE_HOP_LENGTH_MS",
    "SPICE_ONSET_THRESHOLD",
    "DRUM_CONFIDENCE_MIN",
    "DRUM_GHOST_NOTE_VELOCITY_MAX",
    "DRUM_FLAM_DETECTION_WINDOW_MS",
    "DRUM_ROLL_DETECTION_MIN_HITS",
    "DRUM_ROLL_MAX_INTERVAL_MS",

    # Pitch Intelligence
    "MIN_PITCH_MIDI",
    "MAX_PITCH_MIDI",
    "MIN_FREQUENCY_HZ",
    "MAX_FREQUENCY_HZ",
    "TONAL_CORRELATION_THRESHOLD",
    "TONAL_HARMONIC_WEIGHT_DECAY",
    "TONAL_MAX_HARMONICS",
    "BASIC_PITCH_MODEL",
    "BASIC_PITCH_HOP_MS",
    "BASIC_PITCH_MIN_FREQ_HZ",
    "BASIC_PITCH_MAX_FREQ_HZ",

    # Mel-Roformer
    "MEL_ROFORMER_MODEL_NAME",
    "MEL_ROFORMER_N_MELS",
    "MEL_ROFORMER_SAMPLE_RATE",
    "MEL_ROFORMER_DEVICE",
    "MEL_ROFORMER_NUM_LAYERS",
    "MEL_ROFORMER_HIDDEN_SIZE",
    "MEL_ROFORMER_NUM_HEADS",
    "MEL_ROFORMER_DROPOUT",
    "MEL_ROFORMER_N_FFT",
    "MEL_ROFORMER_HOP_LENGTH",
    "MEL_ROFORMER_WINDOW_TYPE",
    "MEL_ROFORMER_STEMS",
    "MEL_ROFORMER_INCLUDE_RESIDUAL_STEM",
    "MEL_ROFORMER_STEREO_MODE",
    "MEL_ROFORMER_CHUNK_DURATION_SECONDS",
    "MEL_ROFORMER_OVERLAP_SECONDS",
    "MEL_ROFORMER_USE_CHUNKED_INFERENCE",
    "MEL_ROFORMER_MAX_DURATION_SECONDS",
    "MEL_ROFORMER_TIMEOUT_SECONDS",
    "MEL_ROFORMER_FORCE_CPU",
    "MEL_ROFORMER_GPU_MEMORY_FRACTION",
    "MEL_ROFORMER_BATCH_SIZE",
    "MEL_ROFORMER_EPSILON",
    "MEL_ROFORMER_DETERMINISTIC",
    "MEL_ROFORMER_RANDOM_SEED",
    "MEL_ROFORMER_MIN_SPECTRAL_COHERENCE",
    "MEL_ROFORMER_MAX_ACCEPTABLE_BLEED",
    "MEL_ROFORMER_MAX_RECONSTRUCTION_ERROR",
    "MEL_ROFORMER_MAX_MASK_ENTROPY",
    "MEL_ROFORMER_FALLBACK_ENABLED",
    "MEL_ROFORMER_FALLBACK_MODE",
    "MEL_ROFORMER_QUALITY_PROFILES",
    "get_mel_roformer_quality_profile",

    # Demucs
    "DEMUCS_MODEL",
    "DEMUCS_SAMPLE_RATE",
    "DEMUCS_DEVICE",
    "DEMUCS_SEGMENT_SECONDS",
    "DEMUCS_OVERLAP_SECONDS",
    "DEMUCS_STEMS",
    "DEMUCS_INCLUDE_RESIDUAL_STEM",
    "DEMUCS_MAX_DURATION_SECONDS",
    "DEMUCS_TIMEOUT_SECONDS",
    "DEMUCS_FORCE_CPU",
    "DEMUCS_GPU_MEMORY_FRACTION",
    "DEMUCS_WINDOW_SECONDS",
    "DEMUCS_HOP_SECONDS",
    "DEMUCS_FALLBACK_ENABLED",
    "DEMUCS_FALLBACK_MODE",
    "DEMUCS_QUALITY_PROFILES",
    "get_demucs_quality_profile",

    # BS-Roformer (deprecated)
    "BS_ROFORMER_DEPRECATED",
    "BS_ROFORMER_MIGRATION_WARNING",
    "BS_ROFORMER_MODEL",
    "BS_ROFORMER_HOP_LENGTH_SAMPLES",
    "BS_ROFORMER_WINDOW_SIZE_SAMPLES",

    # Unified separation
    "SEPARATION_FALLBACK_ORDER",
    "SEPARATION_CACHE_ENABLED",
    "SEPARATION_CACHE_MAX_SIZE_MB",
    "SEPARATION_VALIDATE_OUTPUT",
    "SEPARATION_MIN_STEM_ENERGY_RATIO",
    "SEPARATION_MAX_STEM_ENERGY_RATIO",

    # Hybrid separator
    "HYBRID_SEPARATOR_QUALITY",
    "HYBRID_CACHE_STEMS",

    # Trackers
    "LIBROSA_ONSET_HOP_LENGTH",
    "LIBROSA_ONSET_BACKTRACK",
    "LIBROSA_ONSET_THRESHOLD",
    "MADMOM_BEAT_MODEL",
    "MADMOM_FPS",
    "TRACKER_ENSEMBLE_AGREEMENT_THRESHOLD",

    # Anechoic MA
    "ANECHOIC_DEFAULT_PROFILE",
    "ANECHOIC_MAX_REVERB_MS",
    "ANECHOIC_MIN_DIRECT_TO_REVERB_RATIO_DB",
    "ANECHOIC_FFT_SIZE_FOR_REVERB",

    # Export
    "MIDI_PPQN",
    "MIDI_TEMPO_DEFAULT",
    "MIDI_FORMAT_VERSION",
    "JSON_PRETTY_PRINT",
    "JSON_INCLUDE_FORENSIC_AUDIT",
    "JSON_INCLUDE_MUSIC_BOX",
    "JSON_MAX_FILE_SIZE_MB",
    "HEARTBEAT_NOTE_PITCH",
    "HEARTBEAT_NOTE_VELOCITY",
    "HEARTBEAT_NOTE_DURATION_MS",
    "OCTAVE_RESTORE_BASS",
    "BASS_OCTAVE_SHIFT_SEMITONES",
    "BASS_NOTE_RANGE_LOW_MIDI",
    "BASS_NOTE_RANGE_HIGH_MIDI",

    # Logging
    "MUSIC_BOX_LOG_PATH",
    "MUSIC_BOX_BUFFER_SIZE",
    "MUSIC_BOX_ROTATE_BYTES",
    "MUSIC_BOX_MAX_BACKUPS",
    "LOG_LEVEL",
    "LOG_SHOW_TIMESTAMPS",
    "LOG_SHOW_MEMORY_USAGE",
    "LOG_SHOW_STAGE_DURATIONS",

    # Performance
    "MAX_WORKERS",
    "USE_GPU_IF_AVAILABLE",
    "GPU_MEMORY_FRACTION",
    "TORCH_CPU_INTRAOP_THREADS",
    "TORCH_CPU_INTEROP_THREADS",
    "BATCH_SIZE_PITCH_DETECTION",
    "BATCH_SIZE_DRUM_DETECTION",
    "CACHE_SEPARATION_RESULTS",
    "CACHE_PITCH_DETECTION",
    "CACHE_TTL_SECONDS",

    # Epistemic Veto
    "VETO_HIERARCHY",
    "VETO_RECOVERY_STRATEGY",
    "FALLBACK_SOURCE_ORDER",

    # Deprecated
    "DEPRECATED_CONFIDENCE_ROUTER_THRESHOLDS",
]


# ============================================================================
# DEPRECATION WARNING SHIM
# ============================================================================

def __getattr__(name: str):
    """
    Provide deprecation warnings for legacy constant names.
    """
    legacy_renames = {
        "TARGET_SAMPLE_RATE": "TARGET_SAMPLE_RATE_DEPRECATED",
    }

    if name in legacy_renames:
        import warnings
        warnings.warn(
            f"'{name}' is deprecated in Grimlock 5.6.1. "
            f"Use AudioContext.working_sample_rate or ModelRegistry instead. "
            f"Falling back to '{legacy_renames[name]}'.",
            DeprecationWarning,
            stacklevel=2
        )
        return globals()[legacy_renames[name]]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")