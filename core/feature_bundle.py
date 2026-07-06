# =================================================================
# MODULE: core/feature_bundle.py
# DESCRIPTION: Shared spectral evidence bus with validation & provenance
# VERSION: 5.6.1 (FIXED: Validation, statistics, silent detection)
# UPDATED: 2026-05-17
#
# FIXES IMPLEMENTED:
#   3.1 - Feature statistics (max, mean, std)
#   3.2 - Silent feature detection with warnings
#   3.3 - Feature provenance tracking
#   3.4 - Feature validation with assertions
#   5.1 - Audio statistics (RMS, peak, DC offset, silence ratio)
#   5.2 - Arena asset inspector
#   7.2 - DSP validation utilities
# =================================================================

from __future__ import annotations

import gc
import time
import warnings
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from functools import cached_property

import numpy as np


# =====================================================================
# TYPES
# =====================================================================

class EvidenceType(Enum):
    """Types of spectral evidence that can be independently released."""
    STFT = auto()
    MAGNITUDE = auto()
    PHASE = auto()
    CHROMA = auto()
    CQT = auto()
    RMS = auto()
    ZCR = auto()
    ONSET_STRENGTH = auto()
    HARMONIC_NETWORK = auto()
    POLYPHONIC_PEAKS = auto()
    VOICE_SEPARATION = auto()


class MemoryPriority(Enum):
    """Priority for evidence retention during memory pressure."""
    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()
    EVICTABLE = auto()


# =====================================================================
# VALIDATION FUNCTIONS (required by core/__init__.py)
# =====================================================================

def validate_sample_rate(sample_rate: int, context: str = "") -> None:
    """
    Validate that a sample rate is appropriate for processing.

    Args:
        sample_rate: Sample rate in Hz
        context: Optional context string for error messages

    Raises:
        ValueError: If sample rate is invalid
    """
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate {sample_rate}{': ' + context if context else ''}")

    standard_rates = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000]
    if sample_rate not in standard_rates:
        warnings.warn(f"Non-standard sample rate {sample_rate}Hz{': ' + context if context else ''}", UserWarning)


def validate_audio_duration(audio: np.ndarray, sample_rate: int, context: str = "") -> Tuple[float, bool]:
    """
    Validate audio duration and return (duration_seconds, is_valid).

    Args:
        audio: Audio array
        sample_rate: Sample rate in Hz
        context: Optional context string for warnings

    Returns:
        Tuple of (duration_seconds, is_valid) - is_valid is True if duration >= MIN_DURATION_SECONDS
    """
    duration = len(audio) / sample_rate

    if duration < MIN_DURATION_SECONDS:
        raise ValueError(
            f"Audio too short: {duration:.3f}s < minimum {MIN_DURATION_SECONDS}s"
            f"{': ' + context if context else ''}"
        )

    if duration > 3600:
        warnings.warn(
            f"Audio very long: {duration / 60:.1f} minutes{': ' + context if context else ''}. Processing may take a while.",
            UserWarning)

    return duration, True


# =====================================================================
# DSP VALIDATION UTILITIES (FIX 7.2)
# =====================================================================

def sanitize_audio(audio: np.ndarray, context: str = "") -> np.ndarray:
    """
    Centralized audio sanitization.

    Handles:
        - NaN/Inf replacement
        - Range clipping
        - Type conversion to float32
    """
    if audio is None:
        raise ValueError(f"Audio is None in context: {context}")

    audio = np.asarray(audio, dtype=np.float32)

    # Replace non-finite values
    if np.any(~np.isfinite(audio)):
        nonfinite_count = np.sum(~np.isfinite(audio))
        warnings.warn(
            f"[{context}] Found {nonfinite_count} non-finite values, replacing with 0",
            RuntimeWarning
        )
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip to valid range
    if np.max(np.abs(audio)) > 1.5:
        warnings.warn(
            f"[{context}] Audio exceeds range [-1,1], clipping (max={np.max(audio):.2f})",
            RuntimeWarning
        )
        audio = np.clip(audio, -1.0, 1.0)

    return audio


def validate_fft_size(n_fft: int, hop_length: int, audio_len: int, context: str = "") -> Tuple[int, int]:
    """
    Validate and adjust FFT parameters to be compatible with audio length.

    Returns:
        (adjusted_n_fft, adjusted_hop_length)
    """
    if n_fft > audio_len:
        warnings.warn(
            f"[{context}] n_fft={n_fft} > audio_len={audio_len}, reducing to {audio_len}",
            RuntimeWarning
        )
        n_fft = audio_len

    if hop_length > n_fft:
        hop_length = n_fft // 2

    return n_fft, hop_length


def safe_normalize(arr: np.ndarray, axis: int = None, eps: float = 1e-8) -> np.ndarray:
    """
    Centralized safe normalization.

    Prevents division by zero and handles all-zero arrays.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if axis is not None:
        norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    else:
        norms = np.linalg.norm(arr)

    # Handle all-zero case
    zero_mask = norms < eps
    if np.any(zero_mask):
        if axis is not None:
            # For each zero norm, set to uniform
            zero_indices = np.where(zero_mask.ravel())[0]
            for idx in zero_indices:
                if axis == 0:
                    arr[:, idx] = 1.0 / arr.shape[0]
                elif axis == 1:
                    arr[idx, :] = 1.0 / arr.shape[1]
        else:
            return np.ones_like(arr) / len(arr)

    return arr / (norms + eps)


def compute_audio_statistics(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Compute comprehensive audio statistics for debugging. (FIX 5.1)

    Returns:
        Dictionary with:
            - rms: Root mean square amplitude
            - peak: Maximum absolute amplitude
            - dc_offset: Mean value (should be near 0)
            - silence_ratio: Fraction of samples below -60dB
            - clipping_ratio: Fraction of samples at ±1.0
            - crest_factor: Peak/RMS ratio
            - dynamic_range_db: 20*log10(peak/rms)
    """
    audio = sanitize_audio(audio, "statistics")

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    dc_offset = float(np.mean(audio))

    # Silence detection (-60dB threshold)
    silence_threshold = 10 ** (-60 / 20)  # 0.001
    silence_ratio = float(np.mean(np.abs(audio) < silence_threshold))

    # Clipping detection
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))

    # Crest factor (peak to RMS ratio)
    crest_factor = peak / (rms + 1e-8)

    # Dynamic range in dB
    dynamic_range_db = 20 * np.log10(peak / (rms + 1e-8)) if rms > 0 else 0

    return {
        "rms": rms,
        "peak": peak,
        "dc_offset": dc_offset,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "crest_factor": crest_factor,
        "dynamic_range_db": dynamic_range_db,
        "duration_seconds": len(audio) / sample_rate,
        "sample_rate": sample_rate,
        "num_samples": len(audio)
    }


def compute_feature_statistics(feature: np.ndarray, name: str) -> Dict[str, Any]:
    """
    Compute comprehensive feature statistics for debugging. (FIX 3.1)
    """
    if feature is None:
        return {"available": False, "name": name}

    feature = np.asarray(feature, dtype=np.float32)

    stats = {
        "available": True,
        "name": name,
        "shape": list(feature.shape),
        "dtype": str(feature.dtype),
        "max": float(np.max(feature)),
        "min": float(np.min(feature)),
        "mean": float(np.mean(feature)),
        "std": float(np.std(feature)),
        "energy": float(np.sum(feature ** 2)),
        "sparsity": float(np.mean(feature == 0)),
    }

    # Detect silent features (FIX 3.2)
    if stats["max"] < 1e-5:
        warnings.warn(f"[FEATURE] {name} is nearly silent (max={stats['max']:.2e})", RuntimeWarning)
        stats["is_silent"] = True
    else:
        stats["is_silent"] = False

    return stats


# =====================================================================
# FEATURE PROVENANCE (FIX 3.3)
# =====================================================================

@dataclass
class FeatureProvenance:
    """Provenance metadata for a feature."""
    source_stem: Optional[str] = None
    sample_rate: int = 0
    fft_size: int = 0
    hop_length: int = 0
    window: str = "hann"
    normalization: str = "none"
    computed_at: float = field(default_factory=time.time)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_stem": self.source_stem,
            "sample_rate": self.sample_rate,
            "fft_size": self.fft_size,
            "hop_length": self.hop_length,
            "window": self.window,
            "normalization": self.normalization,
            "computed_at": self.computed_at,
            "duration_seconds": self.duration_seconds
        }


@dataclass
class EvidenceLease:
    """Lease on evidence, preventing concurrent mutation."""
    evidence_type: EvidenceType
    owner_stage: str
    acquired_at: float
    expires_at: float
    priority: MemoryPriority = MemoryPriority.NORMAL
    dependencies: frozenset[EvidenceType] = field(default_factory=frozenset)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        return time.time() - self.acquired_at


# =====================================================================
# PER-MODEL NATIVE SAMPLE RATES
# =====================================================================

MADMOM_SR: int = 44100
BASIC_PITCH_SR: int = 22050
CREPE_SR: int = 16000
DEMUCS_SR: int = 44100
BS_ROFORMER_SR: int = 44100
ANALYSIS_SR: int = 22050

MIN_DURATION_SECONDS: float = 0.1


# =====================================================================
# FEATURE BUNDLE - WITH VALIDATION & PROVENANCE
# =====================================================================

@dataclass
class FeatureBundle:
    """
    Shared spectral evidence with validation, statistics, and provenance.

    FIXES IMPLEMENTED:
        - Feature statistics on demand (FIX 3.1)
        - Silent feature detection (FIX 3.2)
        - Provenance tracking (FIX 3.3)
        - Validation on storage (FIX 3.4)
        - Audio statistics (FIX 5.1)
    """

    # ========== REQUIRED FIELDS ==========
    sample_rate: int
    hop_length: int
    audio: np.ndarray

    # ========== PROVENANCE (FIX 3.3) ==========
    provenance: FeatureProvenance = field(default_factory=FeatureProvenance)

    # ========== COORDINATE AXES ==========
    _time_ms: Optional[np.ndarray] = field(default=None, repr=False)
    _freq_hz: Optional[np.ndarray] = field(default=None, repr=False)

    # ========== LAZY CACHE ==========
    _stft: Optional[np.ndarray] = field(default=None, repr=False)
    _magnitude: Optional[np.ndarray] = field(default=None, repr=False)
    _phase: Optional[np.ndarray] = field(default=None, repr=False)
    _chroma: Optional[np.ndarray] = field(default=None, repr=False)
    _cqt: Optional[np.ndarray] = field(default=None, repr=False)
    _rms: Optional[np.ndarray] = field(default=None, repr=False)
    _zcr: Optional[np.ndarray] = field(default=None, repr=False)
    _onset_strength: Optional[np.ndarray] = field(default=None, repr=False)
    _polyphonic_peaks: Optional[np.ndarray] = field(default=None, repr=False)
    _harmonic_network: Optional[np.ndarray] = field(default=None, repr=False)
    _voice_separation: Optional[np.ndarray] = field(default=None, repr=False)

    # ========== STATISTICS CACHE (FIX 3.1) ==========
    _audio_stats: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _feature_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    # ========== LEASE MANAGEMENT ==========
    _evidence_leases: Dict[EvidenceType, EvidenceLease] = field(
        default_factory=dict, repr=False, compare=False
    )
    _memory_mb: float = field(default=0.0, init=False, repr=False, compare=False)

    # ========== COMPUTATION FLAGS ==========
    _computed_flags: Dict[str, bool] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize with validation and audio statistics."""
        # Sanitize audio first
        self.audio = sanitize_audio(self.audio, "FeatureBundle")

        # Validate audio duration
        duration = len(self.audio) / self.sample_rate
        if duration < MIN_DURATION_SECONDS:
            raise ValueError(
                f"Audio too short: {duration:.3f}s < minimum {MIN_DURATION_SECONDS}s"
            )

        # Store duration in provenance
        self.provenance.duration_seconds = duration
        self.provenance.sample_rate = self.sample_rate
        self.provenance.hop_length = self.hop_length

        # Compute and cache audio statistics (FIX 5.1)
        self._audio_stats = compute_audio_statistics(self.audio, self.sample_rate)

        # Initialize coordinates
        self._init_coordinates()

        self._refresh_memory_estimate()
        self._log_status(f"FeatureBundle initialized: {duration:.2f}s at {self.sample_rate}Hz")

        # Print audio statistics for debugging
        print(f"[AUDIO STATS] RMS={self._audio_stats['rms']:.4f}, "
              f"Peak={self._audio_stats['peak']:.2f}, "
              f"DC={self._audio_stats['dc_offset']:.4f}, "
              f"Silence={self._audio_stats['silence_ratio']:.1%}, "
              f"Clipping={self._audio_stats['clipping_ratio']:.1%}")

    def _init_coordinates(self) -> None:
        """Initialize coordinate axes."""
        if self._time_ms is None:
            import librosa
            n_frames = self._estimate_n_frames()
            self._time_ms = librosa.frames_to_time(
                np.arange(n_frames), sr=self.sample_rate, hop_length=self.hop_length
            ) * 1000.0

        if self._freq_hz is None:
            import librosa
            self._freq_hz = librosa.fft_frequencies(sr=self.sample_rate)

    def _estimate_n_frames(self) -> int:
        return int(np.ceil(len(self.audio) / self.hop_length)) + 1

    # =================================================================
    # AUDIO STATISTICS (FIX 5.1)
    # =================================================================

    def get_audio_statistics(self) -> Dict[str, Any]:
        """Get comprehensive audio statistics."""
        if self._audio_stats is None:
            self._audio_stats = compute_audio_statistics(self.audio, self.sample_rate)
        return self._audio_stats.copy()

    def print_audio_stats(self) -> None:
        """Print audio statistics to console."""
        stats = self.get_audio_statistics()
        print("\n" + "=" * 50)
        print("AUDIO STATISTICS")
        print("=" * 50)
        print(f"  Duration:     {stats['duration_seconds']:.2f}s")
        print(f"  Sample rate:  {stats['sample_rate']}Hz")
        print(f"  RMS:          {stats['rms']:.6f}")
        print(f"  Peak:         {stats['peak']:.4f}")
        print(f"  DC offset:    {stats['dc_offset']:.6f}")
        print(f"  Silence:      {stats['silence_ratio']:.1%}")
        print(f"  Clipping:     {stats['clipping_ratio']:.1%}")
        print(f"  Crest factor: {stats['crest_factor']:.2f}")
        print(f"  Dynamic range: {stats['dynamic_range_db']:.1f}dB")
        print("=" * 50)

    # =================================================================
    # FEATURE STATISTICS (FIX 3.1 & 3.2)
    # =================================================================

    def get_feature_statistics(self, feature_name: str) -> Dict[str, Any]:
        """Get statistics for a specific feature."""
        if feature_name in self._feature_stats:
            return self._feature_stats[feature_name]

        feature = getattr(self, feature_name, None)
        stats = compute_feature_statistics(feature, feature_name)
        self._feature_stats[feature_name] = stats
        return stats

    def print_feature_stats(self, feature_name: str) -> None:
        """Print statistics for a specific feature."""
        stats = self.get_feature_statistics(feature_name)
        if not stats.get("available", False):
            print(f"[{feature_name}] Not available")
            return

        print(f"\n[{feature_name.upper()}]")
        print(f"  Shape: {stats['shape']}")
        print(f"  Max:   {stats['max']:.4e}")
        print(f"  Mean:  {stats['mean']:.4e}")
        print(f"  Std:   {stats['std']:.4e}")
        print(f"  Energy:{stats['energy']:.4e}")
        print(f"  Sparsity: {stats['sparsity']:.1%}")
        if stats.get('is_silent', False):
            print(f"  ⚠️ WARNING: Feature is nearly silent!")

    def print_all_feature_stats(self) -> None:
        """Print statistics for all computed features."""
        print("\n" + "=" * 60)
        print("FEATURE STATISTICS")
        print("=" * 60)

        feature_names = [
            "stft", "magnitude", "phase", "chroma", "cqt",
            "rms", "zcr", "onset_strength"
        ]

        for name in feature_names:
            self.print_feature_stats(name)

    # =================================================================
    # FEATURE VALIDATION (FIX 3.4)
    # =================================================================

    def validate_features(self) -> bool:
        """
        Validate all computed features.

        Checks:
            - No NaN/Inf values
            - Chroma has 12 bins
            - CQT has reasonable values
        """
        all_valid = True

        # Validate chroma
        if self._chroma is not None:
            if np.any(~np.isfinite(self._chroma)):
                warnings.warn("[VALIDATION] Chroma contains NaN/Inf", RuntimeWarning)
                all_valid = False

            if self._chroma.shape[0] != 12:
                warnings.warn(f"[VALIDATION] Chroma has {self._chroma.shape[0]} bins, expected 12", RuntimeWarning)
                all_valid = False

        # Validate CQT
        if self._cqt is not None:
            if np.any(~np.isfinite(self._cqt)):
                warnings.warn("[VALIDATION] CQT contains NaN/Inf", RuntimeWarning)
                all_valid = False

            if np.max(self._cqt) < 1e-5:
                warnings.warn("[VALIDATION] CQT is nearly silent", RuntimeWarning)
                all_valid = False

        # Validate magnitude
        if self._magnitude is not None:
            if np.any(~np.isfinite(self._magnitude)):
                warnings.warn("[VALIDATION] Magnitude contains NaN/Inf", RuntimeWarning)
                all_valid = False

        return all_valid

    # =================================================================
    # PROPERTIES (Lazy Computation with Validation)
    # =================================================================

    @property
    def time_ms(self) -> np.ndarray:
        """Time axis in milliseconds."""
        if self._time_ms is None:
            self._init_coordinates()
        return self._time_ms

    @property
    def freq_hz(self) -> np.ndarray:
        """Frequency axis in Hz."""
        if self._freq_hz is None:
            self._init_coordinates()
        return self._freq_hz

    @property
    def stft(self) -> Optional[np.ndarray]:
        """STFT complex array (lazy)."""
        if self._stft is None and not self._computed_flags.get('stft_failed', False):
            self._compute_stft()
        return self._stft

    @property
    def magnitude(self) -> Optional[np.ndarray]:
        """Magnitude spectrogram (lazy)."""
        if self._magnitude is None and not self._computed_flags.get('magnitude_failed', False):
            if self.stft is not None:
                self._magnitude = np.abs(self.stft)
                # Validate magnitude (FIX 3.4)
                if np.any(~np.isfinite(self._magnitude)):
                    warnings.warn("[FEATURE] Magnitude contains NaN/Inf after computation", RuntimeWarning)
                    self._magnitude = np.nan_to_num(self._magnitude, nan=0.0, posinf=0.0, neginf=0.0)

                # Check for silent magnitude (FIX 3.2)
                if np.max(self._magnitude) < 1e-5:
                    warnings.warn("[FEATURE] Magnitude is nearly silent", RuntimeWarning)

                self._refresh_memory_estimate()
        return self._magnitude

    @property
    def phase(self) -> Optional[np.ndarray]:
        """Phase spectrogram (lazy)."""
        if self._phase is None and not self._computed_flags.get('phase_failed', False):
            if self.stft is not None:
                self._phase = np.angle(self.stft)
                self._refresh_memory_estimate()
        return self._phase

    @property
    def chroma(self) -> Optional[np.ndarray]:
        """Chroma features (lazy)."""
        if self._chroma is None and not self._computed_flags.get('chroma_failed', False):
            self._compute_chroma()

            # Validate chroma (FIX 3.4)
            if self._chroma is not None:
                # Check shape
                if self._chroma.shape[0] != 12:
                    warnings.warn(f"[FEATURE] Chroma has {self._chroma.shape[0]} bins, expected 12", RuntimeWarning)

                # Check for silence
                if np.max(self._chroma) < 1e-5:
                    warnings.warn("[FEATURE] Chroma is nearly silent - possible empty stem", RuntimeWarning)

                # Print stats for debugging (FIX 3.1)
                stats = compute_feature_statistics(self._chroma, "chroma")
                print(
                    f"[CHROMA] max={stats['max']:.4e}, mean={stats['mean']:.4e}, silent={stats.get('is_silent', False)}")

        return self._chroma

    @property
    def cqt(self) -> Optional[np.ndarray]:
        """Constant-Q Transform (lazy)."""
        if self._cqt is None and not self._computed_flags.get('cqt_failed', False):
            self._compute_cqt()

            if self._cqt is not None and np.max(self._cqt) < 1e-5:
                warnings.warn("[FEATURE] CQT is nearly silent", RuntimeWarning)

            stats = compute_feature_statistics(self._cqt, "cqt")
            print(f"[CQT] max={stats['max']:.4e}, mean={stats['mean']:.4e}, silent={stats.get('is_silent', False)}")

        return self._cqt

    @property
    def rms(self) -> Optional[np.ndarray]:
        """RMS energy (lazy)."""
        if self._rms is None and not self._computed_flags.get('rms_failed', False):
            self._compute_rms()
        return self._rms

    @property
    def zcr(self) -> Optional[np.ndarray]:
        """Zero-crossing rate (lazy)."""
        if self._zcr is None and not self._computed_flags.get('zcr_failed', False):
            self._compute_zcr()
        return self._zcr

    @property
    def onset_strength(self) -> Optional[np.ndarray]:
        """Onset strength envelope (lazy)."""
        if self._onset_strength is None and not self._computed_flags.get('onset_failed', False):
            self._compute_onset_strength()

            if self._onset_strength is not None and np.max(self._onset_strength) < 1e-5:
                warnings.warn("[FEATURE] Onset strength is nearly silent", RuntimeWarning)
        return self._onset_strength

    @property
    def polyphonic_peaks(self) -> Optional[np.ndarray]:
        """Multi-pitch activation map (lazy)."""
        if self._polyphonic_peaks is None and not self._computed_flags.get('polyphonic_failed', False):
            self._compute_polyphonic_peaks()
        return self._polyphonic_peaks

    @property
    def harmonic_network(self) -> Optional[np.ndarray]:
        """Harmonic partial relationship matrix (lazy)."""
        if self._harmonic_network is None and not self._computed_flags.get('harmonic_failed', False):
            self._compute_harmonic_network()
        return self._harmonic_network

    @property
    def voice_separation(self) -> Optional[np.ndarray]:
        """Per-frame dominant timbral-cluster index (lazy). -1 = no cluster active."""
        if self._voice_separation is None and not self._computed_flags.get('voice_failed', False):
            self._compute_voice_separation()
        return self._voice_separation

    # =================================================================
    # COMPUTATION METHODS (Internal)
    # =================================================================

    def _compute_stft(self) -> None:
        """Compute STFT with validation."""
        try:
            import librosa

            # Validate FFT parameters
            n_fft, hop_length = validate_fft_size(
                2048, self.hop_length, len(self.audio), "STFT"
            )

            self._stft = librosa.stft(self.audio, n_fft=n_fft, hop_length=hop_length)

            # Validate output
            if np.any(~np.isfinite(self._stft)):
                warnings.warn("[STFT] Output contains NaN/Inf", RuntimeWarning)
                self._stft = np.nan_to_num(self._stft, nan=0.0, posinf=0.0, neginf=0.0)

            self._refresh_memory_estimate()
            self._log_status(f"Computed STFT: {self._stft.shape}")

        except Exception as e:
            self._computed_flags['stft_failed'] = True
            warnings.warn(f"STFT computation failed: {e}")
            self._stft = None

    def _compute_chroma(self) -> None:
        """Compute chroma features with validation."""
        try:
            import librosa

            if self.magnitude is not None:
                self._chroma = librosa.feature.chroma_stft(
                    S=self.magnitude, sr=self.sample_rate, hop_length=self.hop_length
                )
            else:
                self._chroma = librosa.feature.chroma_stft(
                    y=self.audio, sr=self.sample_rate, hop_length=self.hop_length
                )

            # Validate output
            if self._chroma is not None:
                if np.any(~np.isfinite(self._chroma)):
                    warnings.warn("[CHROMA] Output contains NaN/Inf", RuntimeWarning)
                    self._chroma = np.nan_to_num(self._chroma, nan=0.0, posinf=0.0, neginf=0.0)

                # Ensure 12 bins
                if self._chroma.shape[0] != 12:
                    warnings.warn(f"[CHROMA] Expected 12 bins, got {self._chroma.shape[0]}", RuntimeWarning)

            self._refresh_memory_estimate()

        except Exception as e:
            self._computed_flags['chroma_failed'] = True
            warnings.warn(f"Chroma computation failed: {e}")
            self._chroma = None

    def _compute_cqt(self) -> None:
        """Compute Constant-Q Transform."""
        try:
            import librosa

            # Validate parameters
            n_bins = 84  # 7 octaves * 12 bins
            bins_per_octave = 12

            self._cqt = np.abs(librosa.cqt(
                self.audio, sr=self.sample_rate, hop_length=self.hop_length,
                n_bins=n_bins, bins_per_octave=bins_per_octave
            ))

            if np.any(~np.isfinite(self._cqt)):
                warnings.warn("[CQT] Output contains NaN/Inf", RuntimeWarning)
                self._cqt = np.nan_to_num(self._cqt, nan=0.0, posinf=0.0, neginf=0.0)

            self._refresh_memory_estimate()

        except Exception as e:
            self._computed_flags['cqt_failed'] = True
            warnings.warn(f"CQT computation failed: {e}")
            self._cqt = None

    def _compute_rms(self) -> None:
        """Compute RMS energy."""
        try:
            import librosa

            if self.magnitude is not None:
                self._rms = librosa.feature.rms(S=self.magnitude)[0]
            else:
                self._rms = librosa.feature.rms(y=self.audio, hop_length=self.hop_length)[0]

            self._refresh_memory_estimate()

        except Exception as e:
            self._computed_flags['rms_failed'] = True
            warnings.warn(f"RMS computation failed: {e}")
            self._rms = None

    def _compute_zcr(self) -> None:
        """Compute zero-crossing rate."""
        try:
            import librosa
            self._zcr = librosa.feature.zero_crossing_rate(
                self.audio, hop_length=self.hop_length
            )[0]
            self._refresh_memory_estimate()
        except Exception as e:
            self._computed_flags['zcr_failed'] = True
            warnings.warn(f"ZCR computation failed: {e}")
            self._zcr = None

    def _compute_onset_strength(self) -> None:
        """Compute onset strength envelope."""
        try:
            import librosa
            self._onset_strength = librosa.onset.onset_strength(
                y=self.audio, sr=self.sample_rate, hop_length=self.hop_length
            )

            # Normalize
            if np.max(self._onset_strength) > 0:
                self._onset_strength = self._onset_strength / np.max(self._onset_strength)

            self._refresh_memory_estimate()

        except Exception as e:
            self._computed_flags['onset_failed'] = True
            warnings.warn(f"Onset strength computation failed: {e}")
            self._onset_strength = None

    def _compute_polyphonic_peaks(self) -> None:
        """Compute multi-pitch activation map."""
        try:
            if self.magnitude is not None:
                from scipy.signal import argrelmax
                peaks = np.zeros_like(self.magnitude, dtype=np.float32)
                for t in range(self.magnitude.shape[1]):
                    col = self.magnitude[:, t]
                    peak_indices = argrelmax(col, order=3)[0]
                    peaks[peak_indices, t] = col[peak_indices]
                self._polyphonic_peaks = peaks
            else:
                self._polyphonic_peaks = None
            self._refresh_memory_estimate()
        except Exception as e:
            self._computed_flags['polyphonic_failed'] = True
            warnings.warn(f"Polyphonic peaks computation failed: {e}")
            self._polyphonic_peaks = None

    def _compute_harmonic_network(self) -> None:
        """Compute harmonic partial relationship matrix."""
        try:
            if self.magnitude is not None:
                n_harmonics = 8
                harmonic_network = np.zeros((n_harmonics, self.magnitude.shape[1]), dtype=np.float32)
                for k in range(1, n_harmonics + 1):
                    if self.magnitude.shape[0] // k > 0:
                        harmonic_row = self.magnitude[::k].mean(axis=0)
                        harmonic_network[k - 1, :len(harmonic_row)] = harmonic_row
                self._harmonic_network = harmonic_network
            else:
                self._harmonic_network = None
            self._refresh_memory_estimate()
        except Exception as e:
            self._computed_flags['harmonic_failed'] = True
            warnings.warn(f"Harmonic network computation failed: {e}")
            self._harmonic_network = None

    def _compute_voice_separation(self) -> None:
        """
        Per-frame dominant timbral-cluster index, built from StemScanner's
        distinct-instrument clustering over this bundle's own audio.

        This was a literal stub (always all-zeros, no real computation) -
        now reuses the same real cluster-discovery machinery wired into
        the pipeline's per-note cluster_id tagging
        (agents/analysis/spectral_masker.py's StemScanner). Clusters are
        returned dominant-first (by n_windows), so a frame where multiple
        clusters' active_windows overlap is assigned the more dominant one.
        """
        try:
            if self.magnitude is None:
                self._voice_separation = None
                return

            from agents.analysis.spectral_masker import StemScanner

            scanner = StemScanner(sample_rate=self.sample_rate)
            clusters = scanner.scan(self.audio, tempo_bpm=120.0)

            n_frames = self.magnitude.shape[1]
            voice_idx = np.full(n_frames, -1, dtype=np.int8)

            if clusters:
                time_axis = self.time_ms
                for t in range(min(n_frames, len(time_axis))):
                    ts = time_axis[t]
                    for cluster in clusters:
                        if cluster.is_active_at(ts):
                            voice_idx[t] = cluster.cluster_id
                            break

            self._voice_separation = voice_idx
            self._refresh_memory_estimate()
        except Exception as e:
            self._computed_flags['voice_failed'] = True
            warnings.warn(f"Voice separation computation failed: {e}")
            self._voice_separation = None

    def _refresh_memory_estimate(self) -> None:
        """Recalculate total memory footprint in MB."""
        total = 0
        for arr in [
            self._stft, self._magnitude, self._phase,
            self._chroma, self._cqt,
            self._rms, self._zcr, self._onset_strength,
            self._time_ms, self._freq_hz,
            self._polyphonic_peaks, self._harmonic_network, self._voice_separation,
        ]:
            if arr is not None and hasattr(arr, 'nbytes'):
                total += arr.nbytes
        self._memory_mb = total / (1024 * 1024)

    def _log_status(self, message: str) -> None:
        """Log status for debugging."""
        if hasattr(self, '_debug') and self._debug:
            print(f"[FeatureBundle] {message}")

    # =================================================================
    # PUBLIC API
    # =================================================================

    @property
    def memory_mb(self) -> float:
        return self._memory_mb

    @property
    def total_frames(self) -> int:
        if self._magnitude is not None:
            return self._magnitude.shape[1]
        if self._onset_strength is not None:
            return len(self._onset_strength)
        return len(self.time_ms)

    @property
    def duration_seconds(self) -> float:
        return self.provenance.duration_seconds

    @property
    def provenance_dict(self) -> Dict[str, Any]:
        """Get provenance as dictionary."""
        return self.provenance.to_dict()

    def has_evidence(self, evidence_type: EvidenceType) -> bool:
        """Check if evidence is available."""
        mapping = {
            EvidenceType.STFT: self._stft,
            EvidenceType.MAGNITUDE: self._magnitude,
            EvidenceType.PHASE: self._phase,
            EvidenceType.CHROMA: self._chroma,
            EvidenceType.CQT: self._cqt,
            EvidenceType.RMS: self._rms,
            EvidenceType.ZCR: self._zcr,
            EvidenceType.ONSET_STRENGTH: self._onset_strength,
            EvidenceType.HARMONIC_NETWORK: self._harmonic_network,
            EvidenceType.POLYPHONIC_PEAKS: self._polyphonic_peaks,
            EvidenceType.VOICE_SEPARATION: self._voice_separation,
        }
        return mapping.get(evidence_type) is not None

    def get_available_evidence(self) -> List[EvidenceType]:
        """Return list of evidence currently available."""
        available = []
        for et in EvidenceType:
            if self.has_evidence(et):
                available.append(et)
        return available

    # =================================================================
    # DEBUG DUMP (FIX 5.2)
    # =================================================================

    def debug_dump(self) -> Dict[str, Any]:
        """
        Comprehensive debug dump of bundle state.

        Returns:
            Dictionary with all statistics, provenance, and validation results.
        """
        dump = {
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "memory_mb": self.memory_mb,
            "provenance": self.provenance_dict,
            "audio_statistics": self.get_audio_statistics(),
            "available_evidence": [e.name for e in self.get_available_evidence()],
            "feature_statistics": {},
            "validation_passed": self.validate_features(),
            "active_leases": {
                et.name: {
                    "owner": lease.owner_stage,
                    "expires_in": max(0, lease.expires_at - time.time())
                }
                for et, lease in self._evidence_leases.items()
            }
        }

        # Add feature statistics
        for name in ["stft", "magnitude", "chroma", "cqt", "rms", "zcr", "onset_strength"]:
            stats = self.get_feature_statistics(name)
            if stats.get("available"):
                dump["feature_statistics"][name] = stats

        return dump

    def print_debug(self) -> None:
        """Print debug dump to console."""
        import json
        dump = self.debug_dump()

        print("\n" + "=" * 70)
        print("FEATURE BUNDLE DEBUG DUMP")
        print("=" * 70)

        print(f"\n[CONFIGURATION]")
        print(f"  Sample rate: {dump['sample_rate']}Hz")
        print(f"  Hop length: {dump['hop_length']}")
        print(f"  Duration: {dump['duration_seconds']:.2f}s")
        print(f"  Memory: {dump['memory_mb']:.2f}MB")
        print(f"  Validation: {'✓ PASS' if dump['validation_passed'] else '✗ FAIL'}")

        print(f"\n[AUDIO STATISTICS]")
        stats = dump['audio_statistics']
        print(f"  RMS: {stats['rms']:.6f}")
        print(f"  Peak: {stats['peak']:.4f}")
        print(f"  DC offset: {stats['dc_offset']:.6f}")
        print(f"  Silence ratio: {stats['silence_ratio']:.1%}")
        print(f"  Clipping ratio: {stats['clipping_ratio']:.1%}")

        print(f"\n[AVAILABLE EVIDENCE]")
        for ev in dump['available_evidence']:
            print(f"  ✓ {ev}")

        print(f"\n[FEATURE STATISTICS]")
        for name, stats in dump['feature_statistics'].items():
            print(f"\n  {name.upper()}:")
            print(f"    max: {stats['max']:.4e}")
            print(f"    mean: {stats['mean']:.4e}")
            if stats.get('is_silent', False):
                print(f"    ⚠️ SILENT FEATURE")

        print(f"\n[ACTIVE LEASES]")
        for et, lease in dump['active_leases'].items():
            print(f"  {et}: {lease['owner']} (expires in {lease['expires_in']:.1f}s)")

        print("=" * 70 + "\n")

    # =================================================================
    # VIEW-BASED DATA ACCESS
    # =================================================================

    def get_audio_slice(self, start_ms: float, end_ms: float) -> np.ndarray:
        """Return a VIEW of audio between timestamps."""
        start_idx = int((start_ms / 1000.0) * self.sample_rate)
        end_idx = int((end_ms / 1000.0) * self.sample_rate)
        start_idx = max(0, start_idx)
        end_idx = min(len(self.audio), end_idx)
        return self.audio[start_idx:end_idx]

    def get_spectral_slice(
            self,
            start_ms: float,
            end_ms: float,
            freq_min_hz: float = 0.0,
            freq_max_hz: float = 22050.0,
    ) -> Optional[np.ndarray]:
        """Extract a region of the magnitude spectrogram as a VIEW."""
        if self.magnitude is None:
            return None

        t_start = max(0, int(np.searchsorted(self.time_ms, start_ms)))
        t_end = min(len(self.time_ms), int(np.searchsorted(self.time_ms, end_ms)))
        f_start = max(0, int(np.searchsorted(self.freq_hz, freq_min_hz)))
        f_end = min(len(self.freq_hz), int(np.searchsorted(self.freq_hz, freq_max_hz)))
        f_end = min(f_end, self.magnitude.shape[0])

        return self.magnitude[f_start:f_end, t_start:t_end]

    def get_harmonic_evidence(self, note: Any) -> Dict[str, Any]:
        """
        Build per-note harmonic partial evidence for HarmonicValidator.

        This method didn't exist at all until now - HarmonicValidator.
        validate() called features.get_harmonic_evidence(note) every
        single time it ran, which raised AttributeError immediately.
        The pipeline's caller wraps that call in a bare except that
        treats any failure as "note passed," so harmonic validation
        (partial detection, inharmonicity scoring, key-weighting) never
        actually evaluated a single real note - every candidate was
        silently rubber-stamped regardless of its real harmonic content.

        Uses CQT (constant-Q transform), not linear STFT/magnitude, to
        locate each expected harmonic partial: CQT's logarithmic
        frequency spacing gives far better relative resolution for
        distinguishing closely-spaced low-frequency harmonics, which is
        exactly the situation real (polyphonic) music creates - multiple
        simultaneous notes' harmonic series overlapping and interleaving,
        especially in the bass register where linear FFT bins are much
        too coarse to tell one note's 2nd/3rd partial from a
        neighboring note's fundamental.
        """
        fundamental_hz = float(getattr(note, 'fundamental_freq_hz', 0.0) or 0.0)
        if fundamental_hz <= 0:
            pitch = getattr(note, 'pitch', None)
            if pitch is not None:
                fundamental_hz = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        if fundamental_hz <= 0:
            return {'partials': []}

        cqt = self.cqt
        if cqt is None or cqt.size == 0:
            return {'partials': []}

        cqt_freqs = self._cqt_frequencies()
        if cqt_freqs is None:
            return {'partials': []}

        start_ms = note.get_active_start_ms() if hasattr(note, 'get_active_start_ms') else getattr(note, 'start_ms', 0.0)
        end_ms = note.get_active_end_ms() if hasattr(note, 'get_active_end_ms') else getattr(note, 'end_ms', start_ms)
        end_ms = max(end_ms, start_ms + 1.0)

        time_axis = self.time_ms
        n_frames = min(len(time_axis), cqt.shape[1])
        t_start = int(np.searchsorted(time_axis[:n_frames], start_ms))
        t_end = int(np.searchsorted(time_axis[:n_frames], end_ms))
        if t_end <= t_start:
            t_end = min(t_start + 1, n_frames)
        t_start = max(0, min(t_start, n_frames - 1))
        t_end = max(t_start + 1, min(t_end, n_frames))

        note_energy = np.mean(cqt[:, t_start:t_end], axis=1)
        noise_floor = float(np.median(note_energy)) if note_energy.size else 0.0

        max_harmonics = 8
        partials: List[Dict[str, Any]] = []
        for k in range(1, max_harmonics + 1):
            target_hz = fundamental_hz * k
            if target_hz > cqt_freqs[-1] * 1.05:
                break

            bin_idx = int(np.argmin(np.abs(cqt_freqs - target_hz)))
            amplitude = float(note_energy[bin_idx])

            # Only count a real local peak, not background/noise floor -
            # otherwise every harmonic slot would trivially "detect" a
            # partial regardless of whether any real energy is there.
            if amplitude <= 0 or amplitude < noise_floor * 1.5:
                continue

            partials.append({
                'partial_number': k,
                'frequency_hz': float(cqt_freqs[bin_idx]),
                'amplitude_db': float(20.0 * np.log10(amplitude + 1e-8)),
            })

        return {'partials': partials}

    def is_fundamental_prominent_peak(self, note: Any) -> bool:
        """
        Corroborate a note's fundamental against polyphonic_peaks - a real
        multi-pitch activation map built from per-frame local maxima
        (argrelmax) on the raw magnitude spectrogram, independent of
        get_harmonic_evidence's CQT/noise-floor-relative partial detection.

        A fundamental that sits on a genuine local spectral peak during the
        note's own active window is materially less likely to be a
        hallucinated pitch than one that's merely "above the noise floor"
        along a plausible harmonic series. This was a real, working
        property computed by FeatureBundle but never consumed anywhere.
        """
        peaks = self.polyphonic_peaks
        if peaks is None or peaks.size == 0:
            return False

        fundamental_hz = float(getattr(note, 'fundamental_freq_hz', 0.0) or 0.0)
        if fundamental_hz <= 0:
            pitch = getattr(note, 'pitch', None)
            if pitch is None:
                return False
            fundamental_hz = 440.0 * (2.0 ** ((pitch - 69) / 12.0))

        freq_axis = self.freq_hz
        if freq_axis is None or len(freq_axis) == 0:
            return False

        start_ms = note.get_active_start_ms() if hasattr(note, 'get_active_start_ms') else getattr(note, 'start_ms', 0.0)
        end_ms = note.get_active_end_ms() if hasattr(note, 'get_active_end_ms') else getattr(note, 'end_ms', start_ms)
        end_ms = max(end_ms, start_ms + 1.0)

        time_axis = self.time_ms
        n_frames = min(len(time_axis), peaks.shape[1])
        t_start = int(np.searchsorted(time_axis[:n_frames], start_ms))
        t_end = int(np.searchsorted(time_axis[:n_frames], end_ms))
        if t_end <= t_start:
            t_end = min(t_start + 1, n_frames)
        t_start = max(0, min(t_start, n_frames - 1))
        t_end = max(t_start + 1, min(t_end, n_frames))

        bin_idx = int(np.argmin(np.abs(freq_axis - fundamental_hz)))
        bin_tolerance = 2  # matches linear-STFT bin resolution, not a musical unit
        f_lo = max(0, bin_idx - bin_tolerance)
        f_hi = min(peaks.shape[0], bin_idx + bin_tolerance + 1)

        window = peaks[f_lo:f_hi, t_start:t_end]
        if not np.any(window > 0):
            return False

        # argrelmax finds ANY local maximum, including tiny numerical bumps
        # in otherwise-silent/noise-floor bins, so a bare "> 0" check would
        # trivially corroborate almost any claimed frequency. Require the
        # peak to clear this frame's own noise floor, same relative-
        # amplitude principle get_harmonic_evidence already applies to CQT.
        if self.magnitude is None or self.magnitude.size == 0:
            return True
        frame_magnitude = self.magnitude[:, t_start:t_end]
        noise_floor = float(np.median(frame_magnitude)) if frame_magnitude.size else 0.0
        threshold = noise_floor * 2.0

        # A single frame clearing the threshold isn't enough - with a
        # realistic noisy background, an isolated random bump exceeding
        # 2x the noise floor somewhere in a multi-frame/multi-bin window
        # is ordinary, not rare (more "chances" the wider the window). A
        # genuine sustained fundamental should clear it in most frames
        # across the note's own duration, not just one.
        per_frame_hit = np.any(window > threshold, axis=0)
        return bool(np.mean(per_frame_hit) >= 0.5)

    def _cqt_frequencies(self) -> Optional[np.ndarray]:
        """Center frequency of each CQT bin, matching _compute_cqt's parameters."""
        try:
            import librosa
            return librosa.cqt_frequencies(
                n_bins=84, fmin=librosa.note_to_hz('C1'), bins_per_octave=12
            )
        except Exception:
            return None

    # =================================================================
    # LEASE MANAGEMENT
    # =================================================================

    def acquire_lease(
            self,
            evidence_type: EvidenceType,
            stage: str,
            priority: MemoryPriority = MemoryPriority.NORMAL,
            ttl_seconds: float = 300.0,
    ) -> bool:
        existing = self._evidence_leases.get(evidence_type)
        if existing is not None:
            still_valid = time.time() < existing.expires_at
            owned_by_other = existing.owner_stage != stage
            if still_valid and owned_by_other:
                return False

        now = time.time()
        self._evidence_leases[evidence_type] = EvidenceLease(
            evidence_type=evidence_type,
            owner_stage=stage,
            acquired_at=now,
            expires_at=now + ttl_seconds,
            priority=priority,
            dependencies=frozenset(),
        )
        return True

    def can_release(self, evidence_type: EvidenceType, stage: str) -> bool:
        lease = self._evidence_leases.get(evidence_type)
        if lease is None:
            return True
        is_owner = lease.owner_stage == stage
        is_expired = time.time() > lease.expires_at
        return is_owner or is_expired

    def release(self, evidence_type: EvidenceType, stage: Optional[str] = None) -> bool:
        if stage is not None and not self.can_release(evidence_type, stage):
            return False

        collect_needed = False

        if evidence_type == EvidenceType.STFT:
            self._stft = None
            self._magnitude = None
            self._phase = None
            collect_needed = True
        elif evidence_type == EvidenceType.MAGNITUDE:
            self._magnitude = None
            self._phase = None
        elif evidence_type == EvidenceType.PHASE:
            self._phase = None
        elif evidence_type == EvidenceType.CHROMA:
            self._chroma = None
        elif evidence_type == EvidenceType.CQT:
            self._cqt = None
            collect_needed = True
        elif evidence_type == EvidenceType.RMS:
            self._rms = None
        elif evidence_type == EvidenceType.ZCR:
            self._zcr = None
        elif evidence_type == EvidenceType.ONSET_STRENGTH:
            self._onset_strength = None
        elif evidence_type == EvidenceType.HARMONIC_NETWORK:
            self._harmonic_network = None
        elif evidence_type == EvidenceType.POLYPHONIC_PEAKS:
            self._polyphonic_peaks = None
        elif evidence_type == EvidenceType.VOICE_SEPARATION:
            self._voice_separation = None
        else:
            return False

        self._evidence_leases.pop(evidence_type, None)
        self._refresh_memory_estimate()

        if collect_needed:
            gc.collect()

        return True

    def release_all(self) -> None:
        self._stft = None
        self._magnitude = None
        self._phase = None
        self._chroma = None
        self._cqt = None
        self._rms = None
        self._zcr = None
        self._onset_strength = None
        self._polyphonic_peaks = None
        self._harmonic_network = None
        self._voice_separation = None
        self._evidence_leases.clear()
        self._memory_mb = 0.0
        gc.collect()

    def __repr__(self) -> str:
        present = [et.name for et in self.get_available_evidence()]
        return (
            f"FeatureBundle("
            f"sr={self.sample_rate}, "
            f"duration={self.duration_seconds:.2f}s, "
            f"frames={self.total_frames}, "
            f"memory={self._memory_mb:.1f}MB, "
            f"evidence={present})"
        )


# =====================================================================
# FACTORY FUNCTIONS
# =====================================================================

def create_feature_bundle(
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int = 512,
        source_stem: Optional[str] = None,
        precompute: List[EvidenceType] = None,
        debug: bool = False,
) -> FeatureBundle:
    """
    Create a FeatureBundle with optional precomputation.

    Args:
        audio: Mono audio array
        sample_rate: Sample rate in Hz
        hop_length: STFT hop length
        source_stem: Optional stem name for provenance
        precompute: List of evidence types to precompute (None = compute nothing)
        debug: Enable debug output

    Returns:
        Configured FeatureBundle
    """
    # Create provenance
    provenance = FeatureProvenance(
        source_stem=source_stem,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )

    bundle = FeatureBundle(
        sample_rate=sample_rate,
        hop_length=hop_length,
        audio=audio,
        provenance=provenance,
    )

    # Set debug flag
    bundle._debug = debug

    # Precompute requested evidence
    if precompute:
        for et in precompute:
            if et == EvidenceType.STFT:
                _ = bundle.stft
            elif et == EvidenceType.CHROMA:
                _ = bundle.chroma
            elif et == EvidenceType.CQT:
                _ = bundle.cqt
            elif et == EvidenceType.ONSET_STRENGTH:
                _ = bundle.onset_strength

    return bundle


def extract_features(
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int = 512,
        source_stem: Optional[str] = None,
        debug: bool = False,
) -> FeatureBundle:
    """
    Extract all features (legacy interface).

    Note: This precomputes everything. Use create_feature_bundle with
    precompute list for better memory efficiency.
    """
    return create_feature_bundle(
        audio=audio,
        sample_rate=sample_rate,
        hop_length=hop_length,
        source_stem=source_stem,
        precompute=[
            EvidenceType.STFT,
            EvidenceType.CHROMA,
            EvidenceType.CQT,
            EvidenceType.RMS,
            EvidenceType.ZCR,
            EvidenceType.ONSET_STRENGTH,
        ],
        debug=debug,
    )


def bundle_for_madmom(audio: np.ndarray, source_stem: Optional[str] = None) -> FeatureBundle:
    """Bundle optimized for madmom beat tracking."""
    return create_feature_bundle(
        audio=audio,
        sample_rate=MADMOM_SR,
        hop_length=441,
        source_stem=source_stem,
        precompute=[EvidenceType.ONSET_STRENGTH, EvidenceType.ZCR, EvidenceType.RMS]
    )


def bundle_for_pitch(audio: np.ndarray, source_stem: Optional[str] = None) -> FeatureBundle:
    """Bundle optimized for pitch detection (Basic Pitch)."""
    return create_feature_bundle(
        audio=audio,
        sample_rate=BASIC_PITCH_SR,
        hop_length=256,
        source_stem=source_stem,
        precompute=[EvidenceType.CQT, EvidenceType.CHROMA, EvidenceType.RMS]
    )


def bundle_for_analysis(
        audio: np.ndarray,
        sample_rate: int = ANALYSIS_SR,
        source_stem: Optional[str] = None,
        include_harmonic_network: bool = False,
        debug: bool = False,
) -> FeatureBundle:
    """General-purpose analysis bundle."""
    precompute = [EvidenceType.STFT, EvidenceType.CHROMA, EvidenceType.RMS, EvidenceType.ONSET_STRENGTH]
    if include_harmonic_network:
        precompute.append(EvidenceType.HARMONIC_NETWORK)

    return create_feature_bundle(
        audio=audio,
        sample_rate=sample_rate,
        hop_length=512,
        source_stem=source_stem,
        precompute=precompute,
        debug=debug,
    )


# =====================================================================
# DSP CONSTANTS (FIX 7.1 - Centralized)
# =====================================================================

class DSPThresholds:
    """Centralized DSP thresholds."""

    # Audio validation
    MIN_VALID_RMS: float = 1e-6
    MAX_VALID_PEAK: float = 1.5
    MIN_VALID_DURATION_SECONDS: float = 0.1

    # Silence detection
    SILENCE_THRESHOLD_DB: float = -60.0
    SILENCE_THRESHOLD_LINEAR: float = 10 ** (SILENCE_THRESHOLD_DB / 20)  # 0.001

    # Feature validation
    MIN_FEATURE_ENERGY: float = 1e-5
    MIN_CHROMA_ENERGY: float = 1e-5
    MIN_CQT_ENERGY: float = 1e-5

    # Chroma validation
    EXPECTED_CHROMA_BINS: int = 12

    # STFT defaults
    DEFAULT_FFT_SIZE: int = 2048
    DEFAULT_HOP_LENGTH: int = 512
    DEFAULT_WINDOW: str = "hann"

    # Compression
    MAGNITUDE_EPS: float = 1e-8
    LOG_COMPRESSION_FACTOR: float = 1.0


# Export thresholds for easy access
DSP_THRESHOLDS = DSPThresholds()