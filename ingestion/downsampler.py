# =================================================================
# MODULE: ingestion/downsampler.py
# DESCRIPTION: Legacy resampling engine - Delegates to AcousticIntelligence
#
# VERSION: 5.6.1 (Delegation Layer)
# UPDATED: 2026-05-16
#
# DEPRECATION NOTICE:
#     This module is now a DEPRECATION LAYER that delegates to
#     core.acoustic_intelligence.AcousticIntelligence.
#
#     New code should use AcousticIntelligence directly.
#     This module exists only for backward compatibility.
#
# MIGRATION PATH:
#     OLD: from ingestion.downsampler import DownsamplingEngine
#     NEW: from core.acoustic_intelligence import AcousticIntelligence
#
#     OLD: downsampled = DownsamplingEngine.resample(audio, 44100, 16000)
#     NEW: downsampled = AcousticIntelligence.resample(contract, 16000)
#
# =================================================================

from __future__ import annotations

import warnings
from typing import Dict, Any, Optional, Tuple
from fractions import Fraction

import numpy as np

# Import the new canonical authority
from core.acoustic_intelligence import (
    AcousticIntelligence,
    ImmutableAudio,
    ChannelLayout,
    QUALITY_PRESETS as AI_QUALITY_PRESETS,
    DEFAULT_KAISER_BETA as AI_DEFAULT_BETA
)

# =================================================================
# Model Sample Rate Requirements (Forwarded to contracts)
# =================================================================

# These are HARD requirements - models expect these exact rates
# DEPRECATED: Use core.contracts or core.model_registry instead
MODEL_REQUIRED_RATES: Dict[str, int] = {
    "spice": 16000,
    "crepe": 16000,
    "basic_pitch": 22050,
    "omnizart": 22050,
    "demucs": 44100,
    "mel_roformer": 44100,
    "madmom": 44100,
}

# Analysis types with optimal rates (soft requirements)
# DEPRECATED: Use core.contracts instead
ANALYSIS_OPTIMAL_RATES: Dict[str, int] = {
    "pitch_detection": 16000,
    "rhythm_detection": 16000,
    "drum_detection": 22050,
    "harmonic_analysis": 16000,
    "groove_analysis": 16000,
    "voice_separation": 22050,
}

# =================================================================
# Quality Presets (Forwarded to AcousticIntelligence)
# =================================================================

# DEPRECATED: Use core.acoustic_intelligence.QUALITY_PRESETS
QUALITY_PRESETS: Dict[str, Tuple[int, float]] = {
    'fast': (4, 5.0),
    'normal': (8, 8.0),
    'high': (12, 12.0),
    'best': (16, 14.0)
}

# Anti-aliasing cutoff margin (5% below Nyquist)
# DEPRECATED: Use core.acoustic_intelligence.ANTI_ALIAS_MARGIN
ANTI_ALIAS_MARGIN: float = 0.95


# =================================================================
# DEPRECATION WARNING
# =================================================================

def _deprecation_warning(method_name: str):
    """Emit deprecation warning with migration guidance."""
    warnings.warn(
        f"ingestion.downsampler.{method_name} is deprecated in Grimlock 5.2. "
        f"Use core.acoustic_intelligence.AcousticIntelligence instead. "
        f"Migration: Create an AudioContract first, then use resample().",
        DeprecationWarning,
        stacklevel=3
    )


# =================================================================
# Downsampling Engine - DEPRECATION LAYER
# =================================================================

class DownsamplingEngine:
    """
    DEPRECATED: Legacy resampling engine.

    This class now delegates to core.acoustic_intelligence.AcousticIntelligence.
    New code should use AcousticIntelligence directly.

    Migration Example:
        # OLD
        downsampled = DownsamplingEngine.resample(audio, 44100, 16000)

        # NEW
        from core.acoustic_intelligence import AcousticIntelligence, ChannelLayout
        contract = AcousticIntelligence.create_contract(
            audio, 44100, "my_source", force_mono=True
        )
        resampled = AcousticIntelligence.resample(contract, 16000)
        downsampled = resampled.samples
    """

    @classmethod
    def resample(
            cls,
            audio: np.ndarray,
            orig_sr: int,
            target_sr: int,
            quality: str = "high"
    ) -> np.ndarray:
        """
        DEPRECATED: Resample audio using AcousticIntelligence.

        This method creates a temporary AudioContract, resamples it,
        and returns the samples. For multiple operations, create and
        reuse a contract directly.
        """
        _deprecation_warning("resample()")

        if orig_sr == target_sr:
            return audio

        if target_sr > orig_sr:
            # No upsampling - return original
            return audio

        try:
            # Create temporary contract
            contract = AcousticIntelligence.create_contract(
                audio=audio,
                sample_rate=orig_sr,
                source="downsampler_legacy",
                force_mono=True
            )

            # Resample using canonical engine
            resampled_contract = AcousticIntelligence.resample(
                contract=contract,
                target_sample_rate=target_sr,
                quality=quality
            )

            return resampled_contract.samples

        except Exception as e:
            # Fallback for backward compatibility
            warnings.warn(
                f"AcousticIntelligence resample failed: {e}. "
                f"Falling back to legacy implementation.",
                RuntimeWarning
            )
            return cls._legacy_resample(audio, orig_sr, target_sr, quality)

    @classmethod
    def for_model(
            cls,
            audio: np.ndarray,
            orig_sr: int,
            model_name: str,
            quality: str = "high"
    ) -> np.ndarray:
        """
        DEPRECATED: Resample to a model's required sample rate.
        """
        _deprecation_warning("for_model()")

        target_sr = MODEL_REQUIRED_RATES.get(model_name.lower(), orig_sr)

        if target_sr >= orig_sr:
            return audio

        return cls.resample(audio, orig_sr, target_sr, quality)

    @classmethod
    def for_analysis(
            cls,
            audio: np.ndarray,
            orig_sr: int,
            analysis_type: str,
            quality: str = "high"
    ) -> np.ndarray:
        """
        DEPRECATED: Resample to optimal rate for an analysis type.
        """
        _deprecation_warning("for_analysis()")

        optimal = ANALYSIS_OPTIMAL_RATES.get(analysis_type, orig_sr)

        if optimal >= orig_sr:
            return audio

        return cls.resample(audio, orig_sr, optimal, quality)

    # =============================================================
    # Legacy Fallback Methods (Keep for compatibility)
    # =============================================================

    @classmethod
    def _legacy_resample(
            cls,
            audio: np.ndarray,
            orig_sr: int,
            target_sr: int,
            quality: str
    ) -> np.ndarray:
        """
        Legacy resampling implementation (kept for fallback).
        """
        try:
            from scipy import signal
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False

        if HAS_SCIPY:
            order, beta = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['high'])
            return cls._legacy_resample_scipy(audio, orig_sr, target_sr, order, beta)
        else:
            return cls._legacy_resample_fallback(audio, orig_sr, target_sr)

    @classmethod
    def _legacy_resample_scipy(
            cls,
            audio: np.ndarray,
            orig_sr: int,
            target_sr: int,
            order: int,
            beta: float
    ) -> np.ndarray:
        """Legacy scipy-based resampling."""
        from scipy import signal

        nyquist = target_sr / 2
        cutoff = nyquist * ANTI_ALIAS_MARGIN
        normalized_cutoff = cutoff / (orig_sr / 2)
        normalized_cutoff = min(max(normalized_cutoff, 0.001), 0.999)

        b, a = signal.butter(order, normalized_cutoff, btype='low')
        filtered = signal.filtfilt(b, a, audio)

        ratio = Fraction(target_sr, orig_sr).limit_denominator(1000)
        up = ratio.numerator
        down = ratio.denominator

        return signal.resample_poly(
            filtered, up, down,
            window=('kaiser', beta),
            padtype='constant'
        )

    @classmethod
    def _legacy_resample_fallback(
            cls,
            audio: np.ndarray,
            orig_sr: int,
            target_sr: int
    ) -> np.ndarray:
        """Legacy fallback resampling."""
        try:
            import librosa
            return librosa.resample(
                audio, orig_sr=orig_sr, target_sr=target_sr,
                res_type='kaiser_best'
            )
        except ImportError:
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(audio), ratio)
            indices = indices[indices < len(audio)]
            x = np.arange(len(audio))
            return np.interp(indices, x, audio)

    # =============================================================
    # Utility Methods (Deprecated)
    # =============================================================

    @classmethod
    def get_optimal_rate(cls, model_name: str) -> int:
        """DEPRECATED: Get the optimal sample rate for a model."""
        _deprecation_warning("get_optimal_rate()")
        return MODEL_REQUIRED_RATES.get(model_name.lower(), 44100)

    @classmethod
    def needs_downsampling(cls, model_name: str, current_sr: int) -> bool:
        """DEPRECATED: Check if a model needs downsampling."""
        _deprecation_warning("needs_downsampling()")
        optimal = cls.get_optimal_rate(model_name)
        return current_sr > optimal

    @classmethod
    def get_quality_presets(cls) -> Dict[str, Tuple[int, float]]:
        """DEPRECATED: Return available quality presets."""
        _deprecation_warning("get_quality_presets()")
        return QUALITY_PRESETS.copy()

    @classmethod
    def is_available(cls) -> bool:
        """DEPRECATED: Check if scipy is available."""
        try:
            from scipy import signal
            return True
        except ImportError:
            return False


# =================================================================
# Backward Compatibility Wrapper (for old code)
# =================================================================

class Downsampler:
    """
    DEPRECATED: Legacy wrapper for DownsamplingEngine.

    Use AcousticIntelligence directly in new code.
    """

    def __init__(self, default_quality: str = "high"):
        _deprecation_warning("Downsampler class")
        self.default_quality = default_quality
        self._operation_count = 0

    def downsample(
            self,
            audio: np.ndarray,
            orig_sr: int,
            target_sr: int,
            quality: Optional[str] = None
    ) -> np.ndarray:
        """Legacy method - forwards to DownsamplingEngine."""
        self._operation_count += 1
        return DownsamplingEngine.resample(
            audio, orig_sr, target_sr,
            quality or self.default_quality
        )

    def for_model(
            self,
            audio: np.ndarray,
            orig_sr: int,
            model_name: str,
            force: bool = False
    ) -> np.ndarray:
        """Legacy method - forwards to DownsamplingEngine."""
        if not force and not DownsamplingEngine.needs_downsampling(model_name, orig_sr):
            return audio

        self._operation_count += 1
        return DownsamplingEngine.for_model(audio, orig_sr, model_name, self.default_quality)

    def for_analysis(
            self,
            audio: np.ndarray,
            orig_sr: int,
            analysis_type: str,
            force: bool = False
    ) -> np.ndarray:
        """Legacy method - forwards to DownsamplingEngine."""
        self._operation_count += 1
        return DownsamplingEngine.for_analysis(audio, orig_sr, analysis_type, self.default_quality)

    def get_statistics(self) -> Dict[str, Any]:
        """Return legacy statistics."""
        return {
            "operation_count": self._operation_count,
            "scipy_available": DownsamplingEngine.is_available(),
            "default_quality": self.default_quality,
            "deprecated": True,
            "migration_target": "core.acoustic_intelligence.AcousticIntelligence"
        }


# =============================================================
# DEPRECATED Convenience Functions
# =============================================================

def resample(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
        quality: str = "high"
) -> np.ndarray:
    """
    DEPRECATED: Quick one-shot resampling.

    Use AcousticIntelligence.create_contract() + resample() instead.
    """
    _deprecation_warning("resample()")
    return DownsamplingEngine.resample(audio, orig_sr, target_sr, quality)


def downsample_for_model(
        audio: np.ndarray,
        orig_sr: int,
        model_name: str,
        quality: str = "high"
) -> np.ndarray:
    """
    DEPRECATED: Quick one-shot model-specific downsampling.
    """
    _deprecation_warning("downsample_for_model()")
    return DownsamplingEngine.for_model(audio, orig_sr, model_name, quality)


def get_required_sample_rate(model_name: str) -> int:
    """
    DEPRECATED: Get the required sample rate for a model.

    Use core.model_registry.CanonicalRegistry.get_spec(model_id).preferred_sr
    """
    _deprecation_warning("get_required_sample_rate()")
    return DownsamplingEngine.get_optimal_rate(model_name)


# =============================================================
# Standalone Test (Updated to show migration path)
# =============================================================

def quick_test():
    """Quick test function showing both legacy and new patterns."""
    print("\n" + "=" * 60)
    print("DownsamplingEngine Test - DEPRECATION LAYER")
    print("=" * 60)
    print("\n⚠️  This module is DEPRECATED. Use AcousticIntelligence instead.")
    print("   Migration example shown below.\n")

    # Create test audio
    duration = 1.0
    orig_sr = 44100
    t = np.linspace(0, duration, int(orig_sr * duration))
    test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("\n1. LEGACY PATTERN (Deprecated):")
    downsampled = DownsamplingEngine.resample(test_audio, orig_sr, 16000)
    print(f"   Legacy resample: {len(downsampled)} samples")

    print("\n2. NEW PATTERN (Recommended):")
    from core.acoustic_intelligence import AcousticIntelligence

    contract = AcousticIntelligence.create_contract(
        test_audio, orig_sr, "test", force_mono=True
    )
    resampled_contract = AcousticIntelligence.resample(contract, 16000)
    print(f"   New resample: {len(resampled_contract.samples)} samples")
    print(f"   Contract: {resampled_contract}")

    print("\n3. Backend status:")
    print(f"   scipy available: {DownsamplingEngine.is_available()}")
    print(f"   Quality presets: {list(DownsamplingEngine.get_quality_presets().keys())}")

    print("\n" + "=" * 60)
    print("⚠️  This module is deprecated. Update code to use AcousticIntelligence.")
    print("=" * 60)


if __name__ == "__main__":
    quick_test()