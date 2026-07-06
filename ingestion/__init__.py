# =================================================================
# MODULE: ingestion/__init__.py
# DESCRIPTION: Audio ingestion - loader and downsampler.
#
# VERSION: 5.6.1 (clean slate with integrated engine)
# UPDATED: 2026-05-13
#
# PHILOSOPHY:
#     Law of Non-Destructive Audit: The original signal is the only
#     source of truth. Downsampling is OPTIONAL and REVERSIBLE.
#
#     Law of Resource Survival: Memory is managed, but not at the
#     cost of destroying information.
#
#     Single Source of Truth: DownsamplingEngine is the ONLY resampler.
#     Both LoadedAudio proxies and direct calls use the same math.
#
# COMPONENTS:
#     - LoadedAudio: Master audio with forensic hash and multi-rate proxies
#     - AudioLoader: Load audio with header parsing and integrity verification
#     - DownsamplingEngine: Centralized resampling (polyphase FIR + Kaiser)
#     - Downsampler: Legacy wrapper (backward compatibility)
#
# 5.0 CHANGES (Clean Slate):
#     - DownsamplingEngine as single source of truth
#     - LoadedAudio with at_rate() proxy pattern
#     - Forensic SHA-256 hash for integrity verification
#     - Power-preserving mono downmix (L+R)/√2
#     - Polyphase FIR with Kaiser window for anti-aliasing
#     - No global downsampling - preserve original quality
#
# USAGE:
#     from ingestion import load_audio, LoadedAudio, DownsamplingEngine
#
#     # Load master audio (source of truth)
#     master = load_audio("song.wav")
#     print(f"Loaded {master.duration_seconds:.1f}s at {master.sample_rate}Hz")
#
#     # Multi-rate proxies (cached, first call downsamples)
#     audio_44k = master.at_44k
#     audio_22k = master.at_22k
#     audio_16k = master.at_16k
#
#     # Direct resampling using the engine
#     from ingestion import resample
#     downsampled = resample(audio, 44100, 16000)
#
#     # Model-specific downsampling
#     from ingestion import downsample_for_model
#     audio_16k = downsample_for_model(audio, 44100, "crepe")
# =================================================================

# =================================================================
# Loader Components
# =================================================================

from ingestion.loader import (
    AudioLoader,
    LoadedAudio,
    AudioInfo,
    load_audio,
    get_audio_info,
)

# =================================================================
# Downsampler Components (Single Source of Truth)
# =================================================================

from ingestion.downsampler import (
    # Core engine
    DownsamplingEngine,

    # Model requirements tables
    MODEL_REQUIRED_RATES,
    ANALYSIS_OPTIMAL_RATES,

    # Quality presets
    QUALITY_PRESETS,

    # Legacy wrapper (backward compatibility)
    Downsampler,

    # Convenience functions
    resample,
    downsample_for_model,
    get_required_sample_rate,
)

# =================================================================
# Version and Metadata
# =================================================================

__version__ = "5.6.1"
__author__ = "DeepSeek"

# =================================================================
# Module Docstring
# =================================================================

__doc__ = """
Grimlock 5.0 Ingestion Module
==============================

Law of Non-Destructive Audit: The original signal is the only source of truth.

This module handles audio loading and resampling for Grimlock 5.0.

Components:
-----------
LoadedAudio:
    Master audio object with forensic hash and multi-rate proxy pattern.

    Features:
    - SHA-256 hash for integrity verification
    - at_rate() method for on-demand resampling
    - Cached proxies: first call downsamples, subsequent calls return cached
    - Convenience properties: at_44k, at_22k, at_16k
    - verify_integrity() to check if audio was modified

AudioLoader:
    Loads audio files with forensic tracking.

    Features:
    - Header parsing before loading
    - float64 internal processing (headroom preservation)
    - Power-preserving mono downmix (L+R)/√2
    - SHA-256 hash for verification
    - Memory-mapped fallback for large files (optional)

DownsamplingEngine:
    Centralized resampling engine (SINGLE SOURCE OF TRUTH).

    Features:
    - Polyphase FIR with Kaiser window
    - Anti-aliasing Butterworth filter
    - Model-aware resampling (spice→16k, crepe→16k, basic_pitch→22k)
    - Deterministic: same inputs → same outputs
    - Stateless: no hidden state between calls

5.0 Philosophy:
--------------
4.x assumption: "Jazz bass doesn't need 48kHz" → WRONG.
5.0 reality:   Preserve original, downsample only when models require it.

The original signal is sacred. Downsampled versions are derivatives,
never replacements. Every resampling operation uses the same engine.

Usage Examples:
--------------
# Basic loading (preserves original quality)
from ingestion import load_audio

master = load_audio("song.wav")
print(f"Loaded {master.duration_seconds:.1f}s at {master.sample_rate}Hz")
print(f"Hash: {master.sha256_hash[:16]}...")

# Multi-rate proxies
audio_44k = master.at_44k    # First call - downsamples
audio_44k_again = master.at_44k  # Returns cached version
audio_16k = master.at_16k    # Different rate, separate cache

# Direct resampling using the engine
from ingestion import resample
downsampled = resample(audio, 44100, 16000, quality="high")

# Model-specific downsampling
from ingestion import downsample_for_model
audio_16k = downsample_for_model(audio, 44100, "crepe")
audio_22k = downsample_for_model(audio, 44100, "basic_pitch")

# Get model requirements
from ingestion import get_required_sample_rate, MODEL_REQUIRED_RATES
print(f"CREPE requires: {get_required_sample_rate('crepe')}Hz")
print(f"All requirements: {MODEL_REQUIRED_RATES}")

# Verify integrity
if not master.verify_integrity():
    print("WARNING: Audio data corrupted!")

# Check what's cached
print(f"Cached rates: {master.get_cached_rates()}")

# Clear cache to free memory
master.clear_cache()

5 Strategies Integration:
------------------------
Strategy 1 (Streaming):   Future: load_streaming() with chunk callbacks
Strategy 2 (Disk-Backed): File hashing and info caching
Strategy 3 (mmap):        Memory-mapped file reading (optional)
Strategy 4 (GC):          Explicit cleanup, cache clearing
Strategy 5 (Sparse):      get_audio_info() without full load

Law Compliance:
--------------
- Law of Resource Survival:   Memory limits, duration limits, mono downmix
- Law of Non-Destructive Audit: Original preserved, resampling reversible
- Law of Unidirectional Integrity: No circular imports
"""

# =================================================================
# All Exports
# =================================================================

__all__ = [
    # Version
    "__version__",

    # Loader
    "AudioLoader",
    "LoadedAudio",
    "AudioInfo",
    "load_audio",
    "get_audio_info",

    # Downsampler (Engine)
    "DownsamplingEngine",
    "MODEL_REQUIRED_RATES",
    "ANALYSIS_OPTIMAL_RATES",
    "QUALITY_PRESETS",

    # Downsampler (Legacy wrapper)
    "Downsampler",

    # Convenience functions
    "resample",
    "downsample_for_model",
    "get_required_sample_rate",
]


# =================================================================
# Module Initialization Check
# =================================================================

def _check_imports() -> bool:
    """Verify all ingestion components are importable."""
    missing = []

    try:
        from ingestion.loader import AudioLoader, LoadedAudio
    except ImportError as e:
        missing.append(f"loader: {e}")

    try:
        from ingestion.downsampler import DownsamplingEngine
    except ImportError as e:
        missing.append(f"downsampler: {e}")

    if missing:
        import warnings
        warnings.warn(f"Some ingestion components failed to import: {missing}", ImportWarning)
        return False

    return True


_IMPORTS_OK = _check_imports()

# =================================================================
# Module Metadata
# =================================================================

version_info = {
    "module": "ingestion",
    "version": __version__,
    "imports_ok": _IMPORTS_OK,
    "components": ["loader", "downsampler_engine", "legacy_wrapper"],
    "resampling_backend": "scipy" if DownsamplingEngine.is_available() else "fallback"
}

# =================================================================
# Standalone Test
# =================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRIMLOCK 5.0 - INGESTION MODULE")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")
    print(f"Resampling backend: {version_info['resampling_backend']}")

    print("\nAvailable Components:")
    print("  - AudioLoader: Load audio with forensic tracking")
    print("  - LoadedAudio: Master audio with multi-rate proxies")
    print("  - DownsamplingEngine: Centralized resampling (single source of truth)")
    print("  - Downsampler: Legacy wrapper (backward compatibility)")

    print("\n" + "=" * 70)
    print("Quick Test:")
    print("=" * 70)

    # Test loader info
    print("\n1. AudioLoader:")
    from ingestion.loader import AudioLoader

    loader = AudioLoader()
    stats = loader.get_statistics()
    print(f"   Backends: soundfile={stats['soundfile_available']}, "
          f"librosa={stats['librosa_available']}, "
          f"scipy={stats['scipy_available']}")

    # Test downsampler engine
    print("\n2. DownsamplingEngine:")
    from ingestion.downsampler import DownsamplingEngine, MODEL_REQUIRED_RATES

    print(f"   scipy available: {DownsamplingEngine.is_available()}")
    print(f"   Quality presets: {list(DownsamplingEngine.get_quality_presets().keys())}")
    print(f"   Model requirements:")
    for model in ["spice", "crepe", "basic_pitch", "demucs"]:
        rate = DownsamplingEngine.get_optimal_rate(model)
        needs = DownsamplingEngine.needs_downsampling(model, 44100)
        print(f"     {model}: {rate}Hz (needs downsampling from 44.1k: {needs})")

    # Test convenience functions
    print("\n3. Convenience functions:")
    from ingestion.downsampler import get_required_sample_rate

    print(f"   get_required_sample_rate('crepe'): {get_required_sample_rate('crepe')}Hz")

    print("\n" + "=" * 70)
    print("Ingestion module ready.")
    print("=" * 70)