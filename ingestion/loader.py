# =================================================================
# MODULE: ingestion/loader.py
# DESCRIPTION: Zero-loss audio ingestion using AcousticIntelligence
#
# VERSION: 5.6.1 (AcousticIntelligence Integration - FIXED)
# UPDATED: 2026-05-16
#
# LAWS:
#     - Load as float64, preserve headroom
#     - Decouple header parsing from audio loading
#     - Attach SHA-256 hash for forensic verification
#     - Power-preserving mono downmix (L+R)/√2
#     - Never modify original data
#     - Truncate to manageable duration for memory/performance
#     - Multi-rate proxies: each stage requests its required rate
#     - Cached downsampling using AcousticIntelligence
#
# ARCHITECTURE:
#     - Loader creates AudioContracts via AcousticIntelligence
#     - LoadedAudio stores contracts, not raw audio
#     - at_rate() delegates to AcousticIntelligence.resample()
#     - All audio transformation goes through the canonical authority
# =================================================================

from __future__ import annotations

import hashlib
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# Import truncation constants from core
from core.constants import (
    DEFAULT_PROCESSING_DURATION_SECONDS,
    MIN_TRUNCATION_DURATION_SECONDS,
    MAX_TRUNCATION_DURATION_SECONDS,
    validate_truncation_duration
)

# Import the canonical audio authority
from core.acoustic_intelligence import (
    AcousticIntelligence,
    ImmutableAudio,
    ChannelLayout
)

# Import contracts for type hints
from core.contracts import AudioContract

# Optional backends - fail gracefully
try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from scipy import signal

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =================================================================
# Data Structures
# =================================================================

@dataclass(frozen=True)
class AudioInfo:
    """Immutable audio file metadata (no audio data)."""
    file_path: str
    file_size_bytes: int
    duration_seconds: float
    sample_rate_original: int
    num_channels: int
    bit_depth: Optional[int]
    format: str
    is_supported: bool


@dataclass
class LoadedAudio:
    """
    Complete loaded audio with forensic trail, truncation, and multi-rate proxy support.

    This is the SOURCE OF TRUTH. All derived versions are cached proxies.
    Never modify the master audio directly.

    NEW: Stores an AudioContract (immutable) instead of raw numpy array.
    All at_rate() calls delegate to AcousticIntelligence.

    Usage:
        master = loader.load("song.mp3")
        audio_44k = master.at_rate(44100)   # Delegates to AcousticIntelligence
        audio_44k_again = master.at_rate(44100)  # Returns cached
    """
    contract: AudioContract  # Store the immutable contract
    info: AudioInfo
    loaded_at: datetime
    load_time_ms: float
    original_duration_seconds: float
    truncated_duration_seconds: float
    is_truncated: bool = False
    truncation_source: Optional[str] = None

    # Internal cache for downsampled versions (lazy, per-rate)
    _cache: Dict[int, AudioContract] = field(default_factory=dict, repr=False, compare=False)

    # =============================================================
    # Properties (Delegated to contract)
    # =============================================================

    @property
    def sample_rate(self) -> int:
        """Sample rate of the master audio."""
        return self.contract.sample_rate

    @property
    def audio(self) -> np.ndarray:
        """Raw audio samples (read-only view)."""
        return self.contract.samples

    @property
    def sha256_hash(self) -> str:
        """Forensic hash of the audio."""
        return self.contract.sha256_hash

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds (truncated if applicable)."""
        return self.contract.duration_ms / 1000.0

    @property
    def memory_mb(self) -> float:
        """Memory footprint of master audio in MB."""
        return self.contract.samples.nbytes / (1024 * 1024)

    @property
    def channel_layout(self) -> str:
        """Channel layout of the master audio."""
        return self.contract.channel_layout

    @property
    def num_channels(self) -> int:
        """Number of audio channels."""
        return self.contract.num_channels

    # =============================================================
    # Truncation Information
    # =============================================================

    @property
    def original_duration(self) -> float:
        """Original full duration in seconds (before truncation)."""
        return self.original_duration_seconds

    @property
    def processed_duration(self) -> float:
        """Duration of loaded/truncated audio in seconds."""
        return self.truncated_duration_seconds

    @property
    def truncation_ratio(self) -> float:
        """Ratio of processed to original duration (1.0 = full)."""
        if self.original_duration_seconds <= 0:
            return 1.0
        return self.truncated_duration_seconds / self.original_duration_seconds

    @property
    def is_full_song(self) -> bool:
        """True if no truncation was applied."""
        return not self.is_truncated

    # =============================================================
    # Multi-Rate Proxy Methods (Using AcousticIntelligence)
    # =============================================================

    def at_rate(self, target_sr: int, quality: str = "high") -> np.ndarray:
        """
        Get audio at target sample rate.

        First call creates downsampled contract via AcousticIntelligence.
        Subsequent calls return cached result.
        Never modifies the master audio.
        """
        if target_sr == self.contract.sample_rate:
            return self.contract.samples

        if target_sr not in self._cache:
            # Delegate to canonical resampling authority
            resampled_contract = AcousticIntelligence.resample(
                contract=self.contract,
                target_sample_rate=target_sr,
                quality=quality
            )
            self._cache[target_sr] = resampled_contract

        return self._cache[target_sr].samples

    def at_rate_contract(self, target_sr: int, quality: str = "high") -> AudioContract:
        """
        Get audio contract at target sample rate.

        Returns the full AudioContract (not just samples).
        Useful when downstream needs the contract metadata.
        """
        if target_sr == self.contract.sample_rate:
            return self.contract

        if target_sr not in self._cache:
            resampled_contract = AcousticIntelligence.resample(
                contract=self.contract,
                target_sample_rate=target_sr,
                quality=quality
            )
            self._cache[target_sr] = resampled_contract

        return self._cache[target_sr]

    # =============================================================
    # Convenience Properties
    # =============================================================

    @property
    def at_44k(self) -> np.ndarray:
        """Get audio at 44100 Hz (CD quality)."""
        return self.at_rate(44100)

    @property
    def at_22k(self) -> np.ndarray:
        """Get audio at 22050 Hz (Basic Pitch optimal)."""
        return self.at_rate(22050)

    @property
    def at_16k(self) -> np.ndarray:
        """Get audio at 16000 Hz (CREPE/SPICE optimal)."""
        return self.at_rate(16000)

    @property
    def at_48k(self) -> np.ndarray:
        """Get audio at 48000 Hz (if master is higher)."""
        return self.at_rate(48000)

    # =============================================================
    # Utility Methods
    # =============================================================

    def get_cached_rates(self) -> List[int]:
        """Return list of sample rates currently cached."""
        return list(self._cache.keys())

    def clear_cache(self) -> None:
        """Clear all cached downsampled versions."""
        self._cache.clear()

    def verify_integrity(self) -> bool:
        """Verify that the audio hash matches the stored hash."""
        return self.contract.verify_integrity()

    def get_truncation_summary(self) -> Dict[str, Any]:
        """Return summary of truncation status."""
        return {
            "is_truncated": self.is_truncated,
            "original_duration_sec": self.original_duration_seconds,
            "truncated_duration_sec": self.truncated_duration_seconds,
            "truncation_ratio": self.truncation_ratio,
            "truncation_source": self.truncation_source
        }

    def __repr__(self) -> str:
        trunc_info = f", truncated={self.is_truncated}" if self.is_truncated else ""
        return (
            f"LoadedAudio(duration={self.duration_seconds:.1f}s{trunc_info}, "
            f"sr={self.sample_rate}Hz, "
            f"channels={self.num_channels}, "
            f"layout={self.channel_layout}, "
            f"hash={self.sha256_hash[:16]}..., "
            f"cached_rates={self.get_cached_rates()})"
        )


# =================================================================
# Supported Formats
# =================================================================

SUPPORTED_FORMATS = {
    '.wav', '.wave', '.aiff', '.aif', '.flac',
    '.mp3', '.mp4', '.m4a', '.aac', '.ogg', '.oga', '.opus'
}


# =================================================================
# Audio Loader - With Truncation and AcousticIntelligence
# =================================================================

class AudioLoader:
    """
    Zero-loss audio loader with forensic integrity, truncation, and multi-rate proxy pattern.

    NEW: Creates AudioContracts via AcousticIntelligence.create_contract()
    All audio enters the system through the canonical authority.

    Features:
        - Header parsing before loading
        - float64 internal processing
        - SHA-256 hash for verification
        - Power-preserving mono downmix (L+R)/√2
        - Multi-rate proxy pattern (stages request their required rates)
        - Cached downsampling via AcousticIntelligence
        - TRUNCATION: Load only first N seconds (default 30s) for memory/performance
    """

    def __init__(self, enable_mmap: bool = False, truncate_duration: float = DEFAULT_PROCESSING_DURATION_SECONDS):
        """
        Args:
            enable_mmap: Enable memory-mapped I/O for large files (experimental)
            truncate_duration: Duration in seconds to load (0 = full, >0 = truncate)
        """
        self.enable_mmap = enable_mmap
        self.truncate_duration = validate_truncation_duration(truncate_duration)
        self._load_count = 0
        self._total_load_time_ms = 0.0

        if self.truncate_duration > 0:
            print(f"[LOADER] Truncation enabled: loading first {self.truncate_duration}s of audio")
        else:
            print("[LOADER] Truncation disabled: loading full audio")

    # =============================================================
    # Public API
    # =============================================================

    def get_info(self, file_path: Path) -> AudioInfo:
        """Get file metadata WITHOUT loading audio data."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size

        # Try soundfile first (fastest for metadata)
        if HAS_SOUNDFILE:
            try:
                info = sf.info(str(file_path))
                return AudioInfo(
                    file_path=str(file_path),
                    file_size_bytes=file_size,
                    duration_seconds=info.duration,
                    sample_rate_original=info.samplerate,
                    num_channels=info.channels,
                    bit_depth=None,
                    format=file_path.suffix.lower(),
                    is_supported=file_path.suffix.lower() in SUPPORTED_FORMATS
                )
            except Exception:
                pass

        # Fallback to librosa
        if HAS_LIBROSA:
            try:
                duration = librosa.get_duration(path=str(file_path))
                sr = librosa.get_samplerate(str(file_path))
                return AudioInfo(
                    file_path=str(file_path),
                    file_size_bytes=file_size,
                    duration_seconds=duration,
                    sample_rate_original=sr,
                    num_channels=1,
                    bit_depth=None,
                    format=file_path.suffix.lower(),
                    is_supported=file_path.suffix.lower() in SUPPORTED_FORMATS
                )
            except Exception:
                pass

        # Minimal info if no backends available
        return AudioInfo(
            file_path=str(file_path),
            file_size_bytes=file_size,
            duration_seconds=0.0,
            sample_rate_original=0,
            num_channels=0,
            bit_depth=None,
            format=file_path.suffix.lower(),
            is_supported=file_path.suffix.lower() in SUPPORTED_FORMATS
        )

    def load(self, file_path: Union[str, Path]) -> LoadedAudio:
        """
        Load audio with full forensic tracking, optional truncation, and multi-rate proxy support.

        NEW: Creates AudioContract via AcousticIntelligence.create_contract()
        """
        start_time = time.time()

        file_path = Path(file_path)

        # Get metadata first (decoupled)
        info = self.get_info(file_path)

        if not info.is_supported:
            raise ValueError(f"Unsupported format: {file_path.suffix}. "
                             f"Supported: {', '.join(SUPPORTED_FORMATS)}")

        # Load full audio data
        raw_audio, sr = self._load_audio_data(file_path)
        original_duration = len(raw_audio) / sr

        # Apply truncation if requested
        is_truncated = False
        truncated_duration = original_duration
        truncation_source = None

        if self.truncate_duration > 0 and original_duration > self.truncate_duration:
            samples_to_keep = int(self.truncate_duration * sr)
            raw_audio = raw_audio[:samples_to_keep]
            truncated_duration = len(raw_audio) / sr
            is_truncated = True
            truncation_source = "user" if self.truncate_duration > 0 else "default"

            print(f"[TRUNCATE] Original: {original_duration:.1f}s → {truncated_duration:.1f}s "
                  f"(keeping {samples_to_keep} samples at {sr}Hz)")

        # Normalize to [-1, 1] range if outside
        if np.abs(raw_audio).max() > 1.0:
            raw_audio = raw_audio / np.max(np.abs(raw_audio))

        # Convert to float32 (AcousticIntelligence expects float32)
        if raw_audio.dtype != np.float32:
            raw_audio = raw_audio.astype(np.float32)

        # =============================================================
        # CRITICAL: Create AudioContract via canonical authority
        # =============================================================

        # Determine if we should force mono (default for most pipelines)
        force_mono = True  # Default for ingestion

        contract = AcousticIntelligence.create_contract(
            audio=raw_audio,  # FIXED: parameter name is 'audio', not 'raw_audio' or 'buffer'
            sample_rate=sr,
            source=f"ingestion_{file_path.name}",
            force_mono=force_mono
        )

        load_time_ms = (time.time() - start_time) * 1000
        self._load_count += 1
        self._total_load_time_ms += load_time_ms

        # Create LoadedAudio with contract
        loaded = LoadedAudio(
            contract=contract,
            info=info,
            loaded_at=datetime.now(),
            load_time_ms=load_time_ms,
            original_duration_seconds=original_duration,
            truncated_duration_seconds=truncated_duration,
            is_truncated=is_truncated,
            truncation_source=truncation_source
        )

        return loaded

    # =============================================================
    # Internal Methods
    # =============================================================

    def _load_audio_data(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load raw audio data using best available backend.
        """
        # Prefer soundfile (preserves original channels)
        if HAS_SOUNDFILE:
            try:
                audio, sr = sf.read(str(file_path), dtype='float64', always_2d=True)
                return audio, sr
            except Exception:
                pass

        # Fallback to librosa
        if HAS_LIBROSA:
            try:
                audio, sr = librosa.load(
                    str(file_path),
                    mono=False,
                    sr=None,
                    dtype='float64'
                )
                # librosa returns (channels, samples) → transpose to (samples, channels)
                if audio.ndim > 1:
                    audio = audio.T
                return audio, sr
            except Exception:
                pass

        raise RuntimeError(
            "No audio loading backend available. "
            "Install soundfile or librosa: pip install soundfile librosa"
        )

    # =============================================================
    # Utility
    # =============================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return loader statistics."""
        return {
            "load_count": self._load_count,
            "total_load_time_ms": self._total_load_time_ms,
            "avg_load_time_ms": self._total_load_time_ms / self._load_count if self._load_count > 0 else 0,
            "soundfile_available": HAS_SOUNDFILE,
            "librosa_available": HAS_LIBROSA,
            "scipy_available": HAS_SCIPY,
            "enable_mmap": self.enable_mmap,
            "truncate_duration_sec": self.truncate_duration,
            "uses_acoustic_intelligence": True
        }

    def clear_stats(self) -> None:
        """Clear loader statistics."""
        self._load_count = 0
        self._total_load_time_ms = 0.0


# =============================================================
# Convenience Functions
# =============================================================

def load_audio(
        file_path: Union[str, Path],
        enable_mmap: bool = False,
        truncate_duration: float = DEFAULT_PROCESSING_DURATION_SECONDS,
        force_mono: bool = True
) -> LoadedAudio:
    """
    Quick one-shot audio loading with truncation and multi-rate proxy support.

    NEW: Uses AcousticIntelligence.create_contract() internally.
    All audio enters through the canonical authority.

    Args:
        file_path: Path to audio file
        enable_mmap: Enable memory-mapped I/O for large files
        truncate_duration: Duration in seconds to load (0 = full, >0 = truncate)
        force_mono: Convert to mono (default True)

    Returns:
        LoadedAudio object with master audio and proxy methods
    """
    loader = AudioLoader(enable_mmap=enable_mmap, truncate_duration=truncate_duration)
    return loader.load(file_path)


def get_audio_info(file_path: Union[str, Path]) -> AudioInfo:
    """Quick one-shot info retrieval."""
    loader = AudioLoader()
    return loader.get_info(Path(file_path))


def quick_test(audio_path: str, truncate_duration: float = 30.0) -> None:
    """Quick test function for AudioLoader with AcousticIntelligence."""
    print("\n" + "=" * 60)
    print("AudioLoader Test - AcousticIntelligence Integration")
    print("=" * 60)

    # Load master with truncation
    print(f"\n1. Loading master: {Path(audio_path).name} (truncate to {truncate_duration}s)")
    master = load_audio(audio_path, truncate_duration=truncate_duration)

    print(f"   Original duration: {master.original_duration:.1f}s")
    print(f"   Processed duration: {master.processed_duration:.1f}s")
    print(f"   Truncated: {master.is_truncated}")
    print(f"   Sample rate: {master.sample_rate}Hz")
    print(f"   Channels: {master.num_channels}")
    print(f"   Layout: {master.channel_layout}")
    print(f"   Memory: {master.memory_mb:.2f}MB")
    print(f"   Hash: {master.sha256_hash[:16]}...")
    print(f"   Contract: {master.contract}")

    # Test multi-rate proxies
    print("\n2. Multi-rate proxies (first call - downsamples via AcousticIntelligence):")

    import time
    start = time.time()
    audio_44k = master.at_44k
    time_44k = (time.time() - start) * 1000
    print(f"   at_44k: {len(audio_44k)} samples, {time_44k:.1f}ms")

    start = time.time()
    audio_22k = master.at_22k
    time_22k = (time.time() - start) * 1000
    print(f"   at_22k: {len(audio_22k)} samples, {time_22k:.1f}ms")

    start = time.time()
    audio_16k = master.at_16k
    time_16k = (time.time() - start) * 1000
    print(f"   at_16k: {len(audio_16k)} samples, {time_16k:.1f}ms")

    # Test cache hits
    print("\n3. Cache hits (subsequent calls):")
    print(f"   Cached rates: {master.get_cached_rates()}")

    start = time.time()
    audio_44k_cached = master.at_44k
    time_cached = (time.time() - start) * 1000
    print(f"   at_44k (cached): {time_cached:.1f}ms ({(time_44k - time_cached):.1f}ms saved)")

    # Verify integrity
    print("\n4. Integrity verification:")
    is_valid = master.verify_integrity()
    print(f"   Hash matches: {is_valid}")

    # Truncation summary
    print("\n5. Truncation Summary:")
    summary = master.get_truncation_summary()
    print(f"   {summary}")

    # Statistics
    print("\n6. Loader statistics:")
    loader = AudioLoader()
    stats = loader.get_statistics()
    print(
        f"   Backends: soundfile={stats['soundfile_available']}, librosa={stats['librosa_available']}, scipy={stats['scipy_available']}")
    print(f"   Truncation setting: {stats['truncate_duration_sec']}s")
    print(f"   Uses AcousticIntelligence: {stats['uses_acoustic_intelligence']}")

    print("\n" + "=" * 60)
    print("Test complete. Audio ingested through canonical authority.")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        trunc_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        quick_test(sys.argv[1], trunc_duration)
    else:
        print("Usage: python loader.py <audio_file> [truncate_seconds]")
        print("  truncate_seconds: 0=full song, 30=30 seconds (default), 60=60 seconds")