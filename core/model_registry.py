# =====================================================================
# FILE: core/model_registry.py
# DESCRIPTION:
#     Canonical witness registry and interoperability contracts
#     for Grimlock / Symphony 5.0.
#
# PHILOSOPHY:
#     This registry defines the PHYSICS of every witness.
#
#     Every model must explicitly declare:
#         - what it consumes
#         - what it produces
#         - what assumptions it makes
#         - what timing system it uses
#         - what memory profile it requires
#         - whether it is chunk/stream safe
#         - what spectral evidence it needs (FeatureBundle)
#
#     The registry becomes:
#         - orchestration authority
#         - scheduling authority
#         - conversion authority
#         - compatibility authority
#         - memory governance assistant
#
# DESIGN GOALS:
#     - Immutable
#     - Lightweight
#     - Queryable
#     - Future-proof
#     - Arena-compatible
#     - Runtime availability checking
#
# VERSION: 5.6.1 (refactored with availability checking)
# UPDATED: 2026-05-13
# =====================================================================

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.feature_bundle import EvidenceType


# =====================================================================
# DOMAIN ENUMS
# =====================================================================

class ModelDomain(Enum):
    """Primary witness capability domain."""
    SEPARATION = auto()
    PITCH = auto()
    RHYTHM = auto()
    HARMONY = auto()
    TRANSCRIPTION = auto()
    SPECTRAL = auto()
    VALIDATION = auto()


class StreamingCompatibility(Enum):
    """
    Defines how safely a witness can operate
    under chunked/streamed conditions.
    """
    NONE = auto()  # Cannot handle streaming at all
    LIMITED = auto()  # Limited chunk-wise processing
    WINDOWED = auto()  # Works with overlapping windows
    REALTIME = auto()  # Real-time capable


class OutputSemantics(Enum):
    """
    Canonical meaning of witness outputs.
    """
    MONOPHONIC_PITCH_TRACK = auto()
    POLYPHONIC_NOTE_EVENTS = auto()
    CHORD_ESTIMATION = auto()
    BEAT_TRACK = auto()
    DOWNBEAT_TRACK = auto()
    STEM_AUDIO = auto()
    SPECTROGRAM = auto()
    ONSET_ACTIVATIONS = auto()
    FRAME_ACTIVATIONS = auto()
    MULTI_PITCH = auto()


# =====================================================================
# MODEL SPECIFICATION
# =====================================================================

@dataclass(frozen=True, slots=True)
class ModelSpecification:
    """
    Canonical blueprint for a Grimlock-compatible witness.

    This defines the PHYSICS and CONTRACTS of a witness.
    """

    # ================================================================
    # Identity
    # ================================================================

    model_id: str
    canonical_name: str
    display_name: str
    version: str
    domain: ModelDomain

    # ================================================================
    # Audio Input Contracts
    # ================================================================

    preferred_sr: int
    accepts_variable_sr: bool
    accepts_stereo: bool
    internally_downmixes: bool

    required_dtype: str = "float32"
    requires_normalized_audio: bool = True
    input_range: Tuple[float, float] = (-1.0, 1.0)

    # ================================================================
    # FeatureBundle Integration
    # ================================================================

    can_use_feature_bundle: bool = False
    required_evidence: Tuple[str, ...] = field(default_factory=tuple)  # EvidenceType names
    optional_evidence: Tuple[str, ...] = field(default_factory=tuple)

    # ================================================================
    # Temporal Contracts
    # ================================================================

    frame_size: Optional[int] = None
    hop_size: Optional[int] = None
    centered_windows: bool = True
    causal_processing: bool = False
    latency_ms: float = 0.0

    # ================================================================
    # Output Contracts
    # ================================================================

    output_semantics: OutputSemantics = OutputSemantics.MONOPHONIC_PITCH_TRACK
    output_units: str = "unknown"
    output_shape: str = "[frames]"
    output_standard: str = "grimlock_v1"

    # ================================================================
    # Confidence Contracts
    # ================================================================

    confidence_threshold: float = 0.5
    confidence_range: Tuple[float, float] = (0.0, 1.0)
    confidence_meaning: str = "generic_confidence"

    # ================================================================
    # Chunking / Streaming
    # ================================================================

    streaming_compatibility: StreamingCompatibility = StreamingCompatibility.NONE
    chunk_safe: bool = True
    minimum_context_seconds: float = 0.0
    preferred_overlap_seconds: float = 0.0

    # ================================================================
    # Memory / Compute
    # ================================================================

    gpu_accelerated: bool = False
    gpu_required: bool = False
    estimated_vram_mb: int = 0
    estimated_ram_mb: int = 0

    # ================================================================
    # Arena / Infrastructure
    # ================================================================

    arena_managed: bool = True
    immutable_output: bool = True

    # ================================================================
    # Reliability / Failure Modes
    # ================================================================

    known_failure_modes: Tuple[str, ...] = field(default_factory=tuple)
    quirks: Tuple[str, ...] = field(default_factory=tuple)

    # ================================================================
    # Runtime Availability
    # ================================================================

    package_name: Optional[str] = None  # Python package to import
    import_name: Optional[str] = None  # Alternative import name


# =====================================================================
# CANONICAL REGISTRY
# =====================================================================

class CanonicalRegistry:
    """
    The authoritative witness registry for Symphony 5.0.

    This registry defines the interoperability laws
    governing all witnesses.

    Features:
        - Runtime availability checking
        - Sample rate compatibility
        - Memory requirement queries
        - FeatureBundle evidence requirements
    """

    _MODELS: Dict[str, ModelSpecification] = {

        # ============================================================
        # SEPARATION MODELS
        # ============================================================

        "demucs": ModelSpecification(
            model_id="demucs",
            canonical_name="htdemucs_ft",
            display_name="HTDemucs FT",
            version="ft",
            domain=ModelDomain.SEPARATION,
            preferred_sr=44100,
            accepts_variable_sr=False,
            accepts_stereo=True,
            internally_downmixes=False,
            can_use_feature_bundle=False,
            output_semantics=OutputSemantics.STEM_AUDIO,
            output_units="pcm_audio",
            output_shape="[channels, samples]",
            streaming_compatibility=StreamingCompatibility.LIMITED,
            chunk_safe=True,
            minimum_context_seconds=8.0,
            preferred_overlap_seconds=2.0,
            gpu_accelerated=True,
            gpu_required=False,
            estimated_vram_mb=6000,
            estimated_ram_mb=4000,
            known_failure_modes=(
                "stem_bleed",
                "boundary_artifacts",
                "phase_smearing",
            ),
            quirks=(
                "Very high memory usage",
                "Overlap-add reconstruction critical",
            ),
            package_name="demucs",
        ),

        "bs_roformer": ModelSpecification(
            model_id="bs_roformer",
            canonical_name="bs_roformer_viper",
            display_name="BS Roformer Viper",
            version="viper",
            domain=ModelDomain.SEPARATION,
            preferred_sr=44100,
            accepts_variable_sr=False,
            accepts_stereo=True,
            internally_downmixes=False,
            can_use_feature_bundle=False,
            output_semantics=OutputSemantics.STEM_AUDIO,
            output_units="pcm_audio",
            output_shape="[channels, samples]",
            streaming_compatibility=StreamingCompatibility.LIMITED,
            chunk_safe=True,
            minimum_context_seconds=10.0,
            preferred_overlap_seconds=2.5,
            gpu_accelerated=True,
            estimated_vram_mb=8000,
            estimated_ram_mb=5000,
            known_failure_modes=(
                "boundary_artifacts",
                "high_gpu_fragmentation",
            ),
            quirks=(
                "Very high VRAM usage",
                "Large tensor allocations",
            ),
            package_name="demucs",  # Part of demucs
        ),

        "mel_roformer": ModelSpecification(
            model_id="mel_roformer",
            canonical_name="mel_roformer_v2",
            display_name="Mel Roformer V2",
            version="v2",
            domain=ModelDomain.SEPARATION,
            preferred_sr=44100,
            accepts_variable_sr=False,
            accepts_stereo=True,
            internally_downmixes=False,
            can_use_feature_bundle=False,
            output_semantics=OutputSemantics.STEM_AUDIO,
            output_units="pcm_audio",
            output_shape="[channels, samples]",
            streaming_compatibility=StreamingCompatibility.LIMITED,
            chunk_safe=True,
            minimum_context_seconds=10.0,
            preferred_overlap_seconds=2.5,
            gpu_accelerated=True,
            estimated_vram_mb=7500,
            estimated_ram_mb=5000,
            known_failure_modes=(
                "boundary_artifacts",
                "melodic_smearing",
            ),
            quirks=(
                "Strong melodic separation",
                "High GPU pressure",
            ),
            package_name="demucs",
        ),

        # ============================================================
        # PITCH MODELS
        # ============================================================

        "crepe": ModelSpecification(
            model_id="crepe",
            canonical_name="crepe_full",
            display_name="CREPE Full",
            version="full",
            domain=ModelDomain.PITCH,
            preferred_sr=16000,
            accepts_variable_sr=False,
            accepts_stereo=False,
            internally_downmixes=False,
            can_use_feature_bundle=False,
            required_dtype="float32",
            frame_size=1024,
            hop_size=160,
            centered_windows=True,
            output_semantics=OutputSemantics.MONOPHONIC_PITCH_TRACK,
            output_units="Hz",
            output_shape="[frames]",
            confidence_threshold=0.60,
            confidence_meaning="periodicity",
            streaming_compatibility=StreamingCompatibility.NONE,
            chunk_safe=True,
            minimum_context_seconds=1.0,
            preferred_overlap_seconds=0.5,
            gpu_accelerated=True,
            estimated_vram_mb=400,
            estimated_ram_mb=600,
            known_failure_modes=(
                "octave_jumps",
                "bass_instability",
                "transient_confusion",
            ),
            quirks=(
                "Excellent monophonic pitch tracking",
                "Outputs Hz requiring MIDI conversion",
            ),
            package_name="crepe",
        ),

        "spice": ModelSpecification(
            model_id="spice",
            canonical_name="spice_model",
            display_name="SPICE",
            version="1.0",
            domain=ModelDomain.PITCH,
            preferred_sr=16000,
            accepts_variable_sr=False,
            accepts_stereo=False,
            internally_downmixes=True,
            can_use_feature_bundle=False,
            output_semantics=OutputSemantics.MONOPHONIC_PITCH_TRACK,
            output_units="Hz",
            output_shape="[frames]",
            confidence_meaning="pitch_confidence",
            streaming_compatibility=StreamingCompatibility.WINDOWED,
            chunk_safe=True,
            minimum_context_seconds=1.5,
            preferred_overlap_seconds=0.75,
            gpu_accelerated=True,
            estimated_vram_mb=1200,
            estimated_ram_mb=2000,
            known_failure_modes=(
                "pitch_drift",
                "polyphonic_instability",
            ),
            quirks=(
                "TensorFlow-heavy memory profile",
                "Good pitch smoothness",
            ),
            package_name="spice",
        ),

        # ============================================================
        # TRANSCRIPTION MODELS
        # ============================================================

        "basic_pitch": ModelSpecification(
            model_id="basic_pitch",
            canonical_name="basic_pitch_spotify",
            display_name="Spotify Basic Pitch",
            version="spotify",
            domain=ModelDomain.TRANSCRIPTION,
            preferred_sr=22050,
            accepts_variable_sr=False,
            accepts_stereo=False,
            internally_downmixes=True,
            can_use_feature_bundle=False,
            output_semantics=OutputSemantics.POLYPHONIC_NOTE_EVENTS,
            output_units="MIDI",
            output_shape="[events]",
            confidence_meaning="activation_probability",
            streaming_compatibility=StreamingCompatibility.WINDOWED,
            chunk_safe=True,
            minimum_context_seconds=3.0,
            preferred_overlap_seconds=1.0,
            gpu_accelerated=True,
            estimated_vram_mb=1500,
            estimated_ram_mb=2500,
            known_failure_modes=(
                "note_over_sustain",
                "onset_blurring",
            ),
            quirks=(
                "Excellent MIDI-friendly outputs",
                "Strong polyphonic handling",
            ),
            package_name="basic_pitch",
        ),

        "omnizart": ModelSpecification(
            model_id="omnizart",
            canonical_name="omnizart_chord_v2",
            display_name="Omnizart Chord V2",
            version="v2",
            domain=ModelDomain.HARMONY,
            preferred_sr=44100,
            accepts_variable_sr=False,
            accepts_stereo=False,
            internally_downmixes=True,
            can_use_feature_bundle=True,
            required_evidence=("CHROMA", "CQT"),
            optional_evidence=("MAGNITUDE",),
            output_semantics=OutputSemantics.CHORD_ESTIMATION,
            output_units="chord_labels",
            output_shape="[frames]",
            streaming_compatibility=StreamingCompatibility.NONE,
            chunk_safe=False,
            minimum_context_seconds=10.0,
            preferred_overlap_seconds=3.0,
            gpu_accelerated=True,
            estimated_vram_mb=3500,
            estimated_ram_mb=6000,
            known_failure_modes=(
                "dense_mix_confusion",
                "boundary_instability",
            ),
            quirks=(
                "Heavy TensorFlow footprint",
                "Excellent piano/chord transcription",
            ),
            package_name="omnizart",
        ),

        # ============================================================
        # RHYTHM MODELS
        # ============================================================

        "madmom": ModelSpecification(
            model_id="madmom",
            canonical_name="madmom_dbn_beat",
            display_name="Madmom DBN Beat Tracker",
            version="dbn",
            domain=ModelDomain.RHYTHM,
            preferred_sr=44100,
            accepts_variable_sr=True,
            accepts_stereo=False,
            internally_downmixes=True,
            can_use_feature_bundle=True,
            required_evidence=("ONSET_STRENGTH",),
            optional_evidence=("MAGNITUDE",),
            output_semantics=OutputSemantics.BEAT_TRACK,
            output_units="seconds",
            output_shape="[events]",
            confidence_meaning="beat_activation",
            streaming_compatibility=StreamingCompatibility.WINDOWED,
            chunk_safe=True,
            minimum_context_seconds=8.0,
            preferred_overlap_seconds=4.0,
            gpu_accelerated=False,
            estimated_vram_mb=0,
            estimated_ram_mb=1200,
            known_failure_modes=(
                "triplet_confusion",
                "tempo_halving",
                "tempo_doubling",
            ),
            quirks=(
                "Extremely strong beat tracking",
                "Long-context analysis improves reliability",
            ),
            package_name="madmom",
        ),
    }

    # =================================================================
    # BASIC ACCESS API
    # =================================================================

    @classmethod
    def get_spec(cls, model_id: str) -> ModelSpecification:
        """Retrieve the canonical specification for a witness."""
        if model_id not in cls._MODELS:
            raise ValueError(
                f"Model '{model_id}' is not registered "
                f"in CanonicalRegistry."
            )
        return cls._MODELS[model_id]

    @classmethod
    def has_model(cls, model_id: str) -> bool:
        """Check if a witness exists in registry."""
        return model_id in cls._MODELS

    @classmethod
    def list_models(cls) -> List[str]:
        """Return all registered witness IDs."""
        return list(cls._MODELS.keys())

    # =================================================================
    # RUNTIME AVAILABILITY CHECKING (NEW)
    # =================================================================

    @classmethod
    def is_model_available(cls, model_id: str) -> bool:
        """
        Check if a model is both registered AND installed.

        Returns:
            True if model is registered and its package is installed
        """
        if not cls.has_model(model_id):
            return False

        spec = cls.get_spec(model_id)
        package_name = spec.package_name

        if package_name is None:
            return True  # No package to check - assume available

        try:
            __import__(package_name)
            return True
        except ImportError:
            return False

    @classmethod
    def get_available_models(cls, domain: Optional[ModelDomain] = None) -> List[str]:
        """
        Return list of models that are both registered AND installed.
        """
        models = cls.list_by_domain(domain) if domain else cls.list_models()
        return [m for m in models if cls.is_model_available(m)]

    @classmethod
    def get_unavailable_models(cls, domain: Optional[ModelDomain] = None) -> List[str]:
        """
        Return list of registered models that are NOT installed.
        Useful for logging warnings about missing dependencies.
        """
        models = cls.list_by_domain(domain) if domain else cls.list_models()
        return [m for m in models if not cls.is_model_available(m)]

    @classmethod
    def log_missing_models(cls) -> None:
        """Log warnings for all registered but unavailable models."""
        missing = cls.get_unavailable_models()
        if missing:
            warnings.warn(
                f"The following models are registered but not installed: {missing}",
                UserWarning,
                stacklevel=2
            )

    # =================================================================
    # DOMAIN QUERIES
    # =================================================================

    @classmethod
    def list_by_domain(cls, domain: ModelDomain) -> List[str]:
        """Return all witnesses for a domain."""
        return [
            model_id
            for model_id, spec in cls._MODELS.items()
            if spec.domain == domain
        ]

    @classmethod
    def get_models_by_domain(cls, domain: ModelDomain) -> List[ModelSpecification]:
        """Return all model specifications for a domain."""
        return [
            spec for spec in cls._MODELS.values()
            if spec.domain == domain
        ]

    @classmethod
    def get_available_by_domain(cls, domain: ModelDomain) -> List[str]:
        """Return installed models for a domain."""
        return cls.get_available_models(domain)

    # =================================================================
    # SAMPLE RATE QUERIES
    # =================================================================

    @classmethod
    def get_preferred_sample_rate(cls, model_id: str) -> int:
        """Get the preferred sample rate for a model."""
        return cls.get_spec(model_id).preferred_sr

    @classmethod
    def can_handle_sample_rate(cls, model_id: str, sample_rate: int) -> bool:
        """Check if a model can handle a given sample rate."""
        spec = cls.get_spec(model_id)
        return spec.accepts_variable_sr or spec.preferred_sr == sample_rate

    @classmethod
    def get_models_for_sample_rate(cls, sample_rate: int) -> List[str]:
        """Return all models that can handle the given sample rate."""
        return [
            model_id for model_id, spec in cls._MODELS.items()
            if spec.accepts_variable_sr or spec.preferred_sr == sample_rate
        ]

    @classmethod
    def get_available_for_sample_rate(cls, sample_rate: int) -> List[str]:
        """Return installed models that can handle the given sample rate."""
        all_models = cls.get_models_for_sample_rate(sample_rate)
        return [m for m in all_models if cls.is_model_available(m)]

    @classmethod
    def get_required_sample_rate_for_domain(cls, domain: ModelDomain) -> Optional[int]:
        """
        Get the preferred sample rate for a domain.
        Returns None if models in domain have conflicting requirements.
        """
        models = cls.get_models_by_domain(domain)
        if not models:
            return None

        sample_rates = set(m.preferred_sr for m in models if not m.accepts_variable_sr)
        if len(sample_rates) == 1:
            return sample_rates.pop()
        elif len(sample_rates) > 1:
            return None
        else:
            return 44100

    # =================================================================
    # MEMORY QUERIES
    # =================================================================

    @classmethod
    def get_memory_requirements(cls, model_id: str) -> Tuple[int, int]:
        """Get (estimated_ram_mb, estimated_vram_mb) for a model."""
        spec = cls.get_spec(model_id)
        return (spec.estimated_ram_mb, spec.estimated_vram_mb)

    @classmethod
    def get_total_memory_requirement(cls, model_ids: List[str]) -> Tuple[int, int]:
        """
        Get total memory requirements for a sequence of models.
        Returns (total_ram_mb, peak_vram_mb).
        """
        total_ram = 0
        peak_vram = 0
        for model_id in model_ids:
            if cls.is_model_available(model_id):
                ram, vram = cls.get_memory_requirements(model_id)
                total_ram += ram
                peak_vram = max(peak_vram, vram)
        return (total_ram, peak_vram)

    @classmethod
    def would_exceed_memory_limit(cls, model_ids: List[str], limit_mb: int) -> bool:
        """Check if a sequence of models would exceed memory limit."""
        total_ram, _ = cls.get_total_memory_requirement(model_ids)
        return total_ram > limit_mb

    # =================================================================
    # FEATURE BUNDLE QUERIES
    # =================================================================

    @classmethod
    def can_use_feature_bundle(cls, model_id: str) -> bool:
        """Check if a model can accept a FeatureBundle instead of raw audio."""
        return cls.get_spec(model_id).can_use_feature_bundle

    @classmethod
    def get_required_evidence(cls, model_id: str) -> Tuple[str, ...]:
        """Get the EvidenceTypes required by this model."""
        return cls.get_spec(model_id).required_evidence

    @classmethod
    def get_optional_evidence(cls, model_id: str) -> Tuple[str, ...]:
        """Get the EvidenceTypes optionally used by this model."""
        return cls.get_spec(model_id).optional_evidence

    @classmethod
    def get_all_required_evidence_for_models(cls, model_ids: List[str]) -> Set[str]:
        """Get the union of all required evidence for a list of models."""
        required = set()
        for model_id in model_ids:
            if cls.is_model_available(model_id):
                required.update(cls.get_required_evidence(model_id))
        return required

    # =================================================================
    # STREAMING QUERIES
    # =================================================================

    @classmethod
    def get_streaming_compatible_models(cls, level: StreamingCompatibility) -> List[str]:
        """Return all models matching a streaming level."""
        return [
            model_id
            for model_id, spec in cls._MODELS.items()
            if spec.streaming_compatibility == level
        ]

    @classmethod
    def get_chunk_safe_models(cls) -> List[str]:
        """Return all chunk-safe witnesses."""
        return [
            model_id
            for model_id, spec in cls._MODELS.items()
            if spec.chunk_safe
        ]

    @classmethod
    def get_available_chunk_safe_models(cls) -> List[str]:
        """Return installed chunk-safe witnesses."""
        return [m for m in cls.get_chunk_safe_models() if cls.is_model_available(m)]

    # =================================================================
    # GPU QUERIES
    # =================================================================

    @classmethod
    def get_gpu_models(cls) -> List[str]:
        """Return all GPU-accelerated witnesses."""
        return [
            model_id
            for model_id, spec in cls._MODELS.items()
            if spec.gpu_accelerated
        ]

    @classmethod
    def get_gpu_required_models(cls) -> List[str]:
        """Return all models that require GPU."""
        return [
            model_id
            for model_id, spec in cls._MODELS.items()
            if spec.gpu_required
        ]

    @classmethod
    def get_available_gpu_models(cls) -> List[str]:
        """Return installed GPU-accelerated witnesses."""
        return [m for m in cls.get_gpu_models() if cls.is_model_available(m)]

    # =================================================================
    # VALIDATION HELPERS
    # =================================================================

    @classmethod
    def validate_compatibility(
            cls,
            model_id: str,
            sample_rate: int,
            has_gpu: bool = False,
            available_ram_mb: int = 0,
            available_vram_mb: int = 0
    ) -> Tuple[bool, List[str]]:
        """
        Validate if a model can run given constraints.

        Returns:
            (is_compatible, list_of_issues)
        """
        issues = []

        if not cls.is_model_available(model_id):
            issues.append(f"Model '{model_id}' is not installed.")
            return (False, issues)

        spec = cls.get_spec(model_id)

        # Check sample rate
        if not cls.can_handle_sample_rate(model_id, sample_rate):
            issues.append(
                f"Sample rate {sample_rate}Hz incompatible. "
                f"Model prefers {spec.preferred_sr}Hz."
            )

        # Check GPU
        if spec.gpu_required and not has_gpu:
            issues.append(f"Model requires GPU but GPU not available.")

        # Check RAM
        if available_ram_mb > 0 and spec.estimated_ram_mb > available_ram_mb:
            issues.append(
                f"Insufficient RAM: model needs {spec.estimated_ram_mb}MB, "
                f"only {available_ram_mb}MB available."
            )

        # Check VRAM
        if available_vram_mb > 0 and spec.estimated_vram_mb > available_vram_mb:
            issues.append(
                f"Insufficient VRAM: model needs {spec.estimated_vram_mb}MB, "
                f"only {available_vram_mb}MB available."
            )

        return (len(issues) == 0, issues)

    # =================================================================
    # BEST MODEL SELECTION
    # =================================================================

    @classmethod
    def get_best_model_for_domain(
            cls,
            domain: ModelDomain,
            sample_rate: int,
            has_gpu: bool = False,
            available_ram_mb: int = 0
    ) -> Optional[str]:
        """
        Select the best available model for a domain given constraints.

        Returns:
            Model ID of the best match, or None if none available.
        """
        candidates = cls.get_available_by_domain(domain)
        if not candidates:
            return None

        # Score each candidate
        best_model = None
        best_score = -1

        for model_id in candidates:
            spec = cls.get_spec(model_id)
            score = 0

            # Prefer models that match sample rate exactly
            if spec.preferred_sr == sample_rate:
                score += 10
            elif spec.accepts_variable_sr:
                score += 5

            # Prefer GPU-accelerated if GPU available
            if has_gpu and spec.gpu_accelerated:
                score += 3

            # Prefer lower memory usage
            if available_ram_mb > 0:
                memory_efficiency = 1.0 - (spec.estimated_ram_mb / available_ram_mb)
                score += int(memory_efficiency * 5)

            if score > best_score:
                best_score = score
                best_model = model_id

        return best_model

    # =================================================================
    # REGISTRATION (For future extensibility)
    # =================================================================

    @classmethod
    def register_model(cls, spec: ModelSpecification) -> None:
        """
        Register a new model at runtime.
        Used for custom or dynamically loaded models.
        """
        if spec.model_id in cls._MODELS:
            warnings.warn(
                f"Overwriting existing model '{spec.model_id}' in registry.",
                UserWarning,
                stacklevel=2
            )
        cls._MODELS[spec.model_id] = spec

    @classmethod
    def get_all_specs(cls) -> Dict[str, ModelSpecification]:
        """Return all registered specifications (read-only view)."""
        return cls._MODELS.copy()


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def get_model_summary(model_id: str) -> Dict[str, Any]:
    """Get a human-readable summary of a model's specifications."""
    spec = CanonicalRegistry.get_spec(model_id)
    return {
        "model_id": spec.model_id,
        "display_name": spec.display_name,
        "domain": spec.domain.name,
        "preferred_sample_rate": spec.preferred_sr,
        "accepts_variable_sr": spec.accepts_variable_sr,
        "gpu_accelerated": spec.gpu_accelerated,
        "estimated_ram_mb": spec.estimated_ram_mb,
        "estimated_vram_mb": spec.estimated_vram_mb,
        "chunk_safe": spec.chunk_safe,
        "can_use_feature_bundle": spec.can_use_feature_bundle,
        "required_evidence": list(spec.required_evidence),
        "streaming_compatibility": spec.streaming_compatibility.name,
        "is_installed": CanonicalRegistry.is_model_available(model_id),
    }


def compare_models(model_id_1: str, model_id_2: str) -> Dict[str, Any]:
    """Compare two models and highlight differences."""
    spec1 = CanonicalRegistry.get_spec(model_id_1)
    spec2 = CanonicalRegistry.get_spec(model_id_2)

    differences = {}
    for field_name in ["preferred_sr", "estimated_ram_mb", "gpu_accelerated", "chunk_safe"]:
        val1 = getattr(spec1, field_name)
        val2 = getattr(spec2, field_name)
        if val1 != val2:
            differences[field_name] = {"model1": val1, "model2": val2}

    return {
        "model1": model_id_1,
        "model2": model_id_2,
        "same_domain": spec1.domain == spec2.domain,
        "differences": differences,
        "model1_installed": CanonicalRegistry.is_model_available(model_id_1),
        "model2_installed": CanonicalRegistry.is_model_available(model_id_2),
    }


def print_registry_summary() -> None:
    """Print a summary of all registered models and their availability."""
    print("\n" + "=" * 70)
    print("GRIMLOCK MODEL REGISTRY SUMMARY")
    print("=" * 70)

    for domain in ModelDomain:
        models = CanonicalRegistry.list_by_domain(domain)
        if models:
            print(f"\n{domain.name}:")
            for model_id in models:
                available = "✓" if CanonicalRegistry.is_model_available(model_id) else "✗"
                spec = CanonicalRegistry.get_spec(model_id)
                print(f"  [{available}] {model_id} - {spec.display_name} ({spec.preferred_sr}Hz)")


# =====================================================================
# DEPRECATION WARNING SHIM
# =====================================================================

def __getattr__(name: str):
    """
    Provide deprecation warnings for legacy access patterns.
    """
    deprecated_aliases = {
        "MODELS": "_MODELS",
    }

    if name in deprecated_aliases:
        warnings.warn(
            f"'{name}' is deprecated. Access models via CanonicalRegistry methods.",
            DeprecationWarning,
            stacklevel=2
        )
        return getattr(CanonicalRegistry, deprecated_aliases[name])

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =====================================================================
# MODULE DOCSTRING
# =====================================================================

__doc__ = """
Model Registry for Grimlock 5.0

This module defines the canonical specifications for all models (witnesses)
that can be used in the Grimlock pipeline. It serves as the authority for:

- Sample rate requirements
- Memory usage estimates
- GPU requirements
- Streaming compatibility
- FeatureBundle evidence requirements
- Runtime availability checking

Usage:
    from core.model_registry import CanonicalRegistry, ModelDomain

    # Check if model is installed
    if CanonicalRegistry.is_model_available("crepe"):
        spec = CanonicalRegistry.get_spec("crepe")
        print(f"CREPE needs {spec.preferred_sr}Hz")

    # Get only installed models for a domain
    models = CanonicalRegistry.get_available_by_domain(ModelDomain.PITCH)

    # Validate before loading
    ok, issues = CanonicalRegistry.validate_compatibility(
        "demucs", 
        sample_rate=44100,
        available_ram_mb=16000,
        has_gpu=True
    )

    # Get best model for your constraints
    best = CanonicalRegistry.get_best_model_for_domain(
        ModelDomain.PITCH,
        sample_rate=44100,
        has_gpu=True
    )
"""

# =====================================================================
# EXPORTS
# =====================================================================

__all__ = [
    "ModelDomain",
    "StreamingCompatibility",
    "OutputSemantics",
    "ModelSpecification",
    "CanonicalRegistry",
    "get_model_summary",
    "compare_models",
    "print_registry_summary",
]