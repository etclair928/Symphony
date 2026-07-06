# =================================================================
# MODULE: core/audio_views.py
# DESCRIPTION: Immutable audio views with multi-proxy pattern
#
# VERSION: 5.6.1
# UPDATED: 2026-05-25
#
# LAW OF EVIDENCE:
#     Master audio is the SOURCE OF TRUTH.
#     All views are derived from the master, never modified.
#     Stages request proxies at required sample rates.
#
# MULTI-PROXY PATTERN:
#     - at_44k: 44100 Hz (Demucs, Madmom, Tonal detection)
#     - at_22k: 22050 Hz (Basic Pitch, Rhythm detection)
#     - at_16k: 16000 Hz (CREPE, SPICE, Epistemic Council)
#     - at_rate(): Generic rate conversion
#     - mono_sum_normalized: Mono downmix for analysis
# =================================================================

from __future__ import annotations

import hashlib
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import librosa
import scipy.signal as signal


# ====================================================================
# Channel Layout
# ====================================================================

class ChannelLayout(Enum):
    """Channel layout of audio."""
    MONO = "mono"
    STEREO = "stereo"
    MULTI_CHANNEL = "multi_channel"


# ====================================================================
# ImmutableAudio (Core audio container)
# ====================================================================

@dataclass(frozen=True)
class ImmutableAudio:
    """
    Immutable audio container with hash and metadata.

    This is the source of truth for all audio data in the pipeline.
    """
    samples: np.ndarray
    sample_rate: int
    channel_layout: ChannelLayout
    source_hash: str
    duration_seconds: float
    num_channels: int

    @classmethod
    def from_audio(cls, audio: np.ndarray, sample_rate: int,
                   source: str = "unknown") -> 'ImmutableAudio':
        """Create ImmutableAudio from audio array."""
        if audio.ndim == 1:
            channel_layout = ChannelLayout.MONO
            num_channels = 1
            samples = audio
        elif audio.ndim == 2:
            if audio.shape[0] == 2:
                channel_layout = ChannelLayout.STEREO
                num_channels = 2
                samples = audio
            elif audio.shape[1] == 2:
                channel_layout = ChannelLayout.STEREO
                num_channels = 2
                samples = audio.T
            else:
                channel_layout = ChannelLayout.MULTI_CHANNEL
                num_channels = audio.shape[0]
                samples = audio
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

        # Compute hash
        audio_bytes = samples.astype(np.float32).tobytes()[:10000]  # Sample for hash
        source_hash = hashlib.sha256(f"{source}:{audio_bytes}".encode()).hexdigest()

        duration = samples.shape[-1] / sample_rate

        return cls(
            samples=samples.astype(np.float32),
            sample_rate=sample_rate,
            channel_layout=channel_layout,
            source_hash=source_hash,
            duration_seconds=duration,
            num_channels=num_channels,
        )

    def to_mono(self) -> np.ndarray:
        """Convert to mono by averaging channels."""
        if self.channel_layout == ChannelLayout.MONO:
            return self.samples.copy()
        return np.mean(self.samples, axis=0)

    def get_channel(self, channel_idx: int) -> np.ndarray:
        """Get a specific channel."""
        if self.channel_layout == ChannelLayout.MONO:
            return self.samples if channel_idx == 0 else self.samples
        return self.samples[channel_idx]

    def resample(self, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if target_sr == self.sample_rate:
            return self.to_mono()
        return librosa.resample(self.to_mono(), orig_sr=self.sample_rate, target_sr=target_sr)


# ====================================================================
# AudioViews (Multi-proxy pattern)
# ====================================================================

class AudioViews:
    """
    Immutable audio views with multi-proxy pattern.

    Master audio is the SOURCE OF TRUTH. All views are derived
    from the master, never modified. Stages request proxies at
    required sample rates.

    Usage:
        views = AudioViews(master_stereo=stereo, sample_rate=44100)
        audio_44k = views.at_44k      # 44100 Hz (original)
        audio_22k = views.at_22k      # 22050 Hz (Basic Pitch)
        audio_16k = views.at_16k      # 16000 Hz (CREPE, SPICE)
        mono = views.mono_sum_normalized
    """

    def __init__(
            self,
            master_stereo: np.ndarray,
            sample_rate: int,
            channel_layout: ChannelLayout = ChannelLayout.STEREO,
            drums_stereo: Optional[np.ndarray] = None,
            bass_stereo: Optional[np.ndarray] = None,
            other_stereo: Optional[np.ndarray] = None,
            sha256_hash: Optional[str] = None,
    ):
        """
        Initialize AudioViews with master stereo audio.

        Args:
            master_stereo: Master audio array, shape (2, samples) for stereo
            sample_rate: Sample rate of master audio
            channel_layout: Channel layout (default: STEREO)
            drums_stereo: Drums stem from separation, shape (2, samples)
            bass_stereo: Bass stem from separation, shape (2, samples)
            other_stereo: Other stem from separation, shape (2, samples)
            sha256_hash: Optional hash of master audio
        """
        # Validate master shape
        if master_stereo.ndim == 1:
            master_stereo = np.stack([master_stereo, master_stereo])
        elif master_stereo.ndim == 2 and master_stereo.shape[1] == 2:
            master_stereo = master_stereo.T

        if master_stereo.shape[0] != 2:
            raise ValueError(f"Master audio must have 2 channels, got {master_stereo.shape[0]}")

        self._master = master_stereo.astype(np.float32)
        self._sample_rate = sample_rate
        self._channel_layout = channel_layout
        self._drums = drums_stereo.astype(np.float32) if drums_stereo is not None else None
        self._bass = bass_stereo.astype(np.float32) if bass_stereo is not None else None
        self._other = other_stereo.astype(np.float32) if other_stereo is not None else None
        self._sha256_hash = sha256_hash

        # Validate stem shapes
        for stem, name in [(self._drums, "drums"), (self._bass, "bass"), (self._other, "other")]:
            if stem is not None and stem.shape[0] != 2:
                raise ValueError(f"{name} stem must have 2 channels, got {stem.shape[0]}")

        # Cache for resampled audio
        self._cache_44k: Optional[np.ndarray] = None
        self._cache_22k: Optional[np.ndarray] = None
        self._cache_16k: Optional[np.ndarray] = None
        self._cache_mono: Optional[np.ndarray] = None

        # Compute mono once
        self._mono = np.mean(self._master, axis=0)

        # Normalized mono (peak-normalized to 1.0)
        max_val = np.max(np.abs(self._mono))
        if max_val > 0:
            self._mono_norm = self._mono / max_val
        else:
            self._mono_norm = self._mono

    # ====================================================================
    # Properties
    # ====================================================================

    @property
    def sample_rate(self) -> int:
        """Original sample rate of master audio."""
        return self._sample_rate

    @property
    def channel_layout(self) -> ChannelLayout:
        """Channel layout of master audio."""
        return self._channel_layout

    @property
    def master_stereo(self) -> np.ndarray:
        """Master stereo audio, shape (2, samples)."""
        return self._master

    @property
    def left_channel(self) -> np.ndarray:
        """Left channel (channel 0)."""
        return self._master[0]

    @property
    def right_channel(self) -> np.ndarray:
        """Right channel (channel 1)."""
        return self._master[1]

    @property
    def mono_sum(self) -> np.ndarray:
        """Mono sum of left+right channels (not normalized)."""
        return self._mono

    @property
    def mono_sum_normalized(self) -> np.ndarray:
        """Mono sum normalized to [-1, 1] range."""
        return self._mono_norm

    @property
    def sha256_hash(self) -> Optional[str]:
        """SHA-256 hash of master audio."""
        return self._sha256_hash

    # ====================================================================
    # Stem access
    # ====================================================================

    @property
    def drums_stereo(self) -> Optional[np.ndarray]:
        """Drums stem from separation, shape (2, samples)."""
        return self._drums

    @property
    def bass_stereo(self) -> Optional[np.ndarray]:
        """Bass stem from separation, shape (2, samples)."""
        return self._bass

    @property
    def other_stereo(self) -> Optional[np.ndarray]:
        """Other stem from separation, shape (2, samples)."""
        return self._other

    # ====================================================================
    # Multi-proxy access (lazy resampling)
    # ====================================================================

    @property
    def at_44k(self) -> np.ndarray:
        """Audio at 44100 Hz (original or resampled)."""
        if self._sample_rate == 44100:
            return self.mono_sum_normalized
        if self._cache_44k is None:
            self._cache_44k = librosa.resample(
                self.mono_sum_normalized,
                orig_sr=self._sample_rate,
                target_sr=44100
            )
        return self._cache_44k

    @property
    def at_22k(self) -> np.ndarray:
        """Audio at 22050 Hz (Basic Pitch, Rhythm detection)."""
        if self._sample_rate == 22050:
            return self.mono_sum_normalized
        if self._cache_22k is None:
            self._cache_22k = librosa.resample(
                self.mono_sum_normalized,
                orig_sr=self._sample_rate,
                target_sr=22050
            )
        return self._cache_22k

    @property
    def at_16k(self) -> np.ndarray:
        """Audio at 16000 Hz (CREPE, SPICE, Epistemic Council)."""
        if self._sample_rate == 16000:
            return self.mono_sum_normalized
        if self._cache_16k is None:
            self._cache_16k = librosa.resample(
                self.mono_sum_normalized,
                orig_sr=self._sample_rate,
                target_sr=16000
            )
        return self._cache_16k

    def at_rate(self, target_sr: int) -> np.ndarray:
        """Get audio at arbitrary sample rate."""
        if target_sr == self._sample_rate:
            return self.mono_sum_normalized
        if target_sr == 44100:
            return self.at_44k
        if target_sr == 22050:
            return self.at_22k
        if target_sr == 16000:
            return self.at_16k
        return librosa.resample(
            self.mono_sum_normalized,
            orig_sr=self._sample_rate,
            target_sr=target_sr
        )

    # ====================================================================
    # Utility methods
    # ====================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get audio statistics."""
        stats = {
            "duration_seconds": len(self._master[0]) / self._sample_rate,
            "sample_rate": self._sample_rate,
            "channel_layout": self._channel_layout.value,
            "num_channels": 2,
            "rms": float(np.sqrt(np.mean(self._mono ** 2))),
            "peak": float(np.max(np.abs(self._mono))),
            "stereo_width": float(np.corrcoef(self._master[0], self._master[1])[0, 1]),
        }

        # Stem info
        stats["has_drums"] = self._drums is not None
        stats["has_bass"] = self._bass is not None
        stats["has_other"] = self._other is not None

        if self._drums is not None:
            stats["drums_energy"] = float(np.sum(self._drums ** 2))
        if self._bass is not None:
            stats["bass_energy"] = float(np.sum(self._bass ** 2))
        if self._other is not None:
            stats["other_energy"] = float(np.sum(self._other ** 2))

        return stats

    def clear_cache(self):
        """Clear resampled audio cache to free memory."""
        self._cache_44k = None
        self._cache_22k = None
        self._cache_16k = None

    def __repr__(self) -> str:
        return (f"AudioViews({self._master.shape[1] / self._sample_rate:.1f}s, "
                f"{self._sample_rate}Hz, stems: drums={self._drums is not None}, "
                f"bass={self._bass is not None}, other={self._other is not None})")


# ====================================================================
# Factory Functions
# ====================================================================

def create_audio_views(
        master_stereo: np.ndarray,
        sample_rate: int,
        channel_layout: ChannelLayout = ChannelLayout.STEREO,
        drums_stereo: Optional[np.ndarray] = None,
        bass_stereo: Optional[np.ndarray] = None,
        other_stereo: Optional[np.ndarray] = None,
        sha256_hash: Optional[str] = None,
) -> AudioViews:
    """Create AudioViews instance."""
    return AudioViews(
        master_stereo=master_stereo,
        sample_rate=sample_rate,
        channel_layout=channel_layout,
        drums_stereo=drums_stereo,
        bass_stereo=bass_stereo,
        other_stereo=other_stereo,
        sha256_hash=sha256_hash,
    )


def create_stereo_views(
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        sha256_hash: Optional[str] = None,
) -> AudioViews:
    """Create AudioViews from separate left/right channels."""
    stereo = np.stack([left, right])
    return AudioViews(
        master_stereo=stereo,
        sample_rate=sample_rate,
        channel_layout=ChannelLayout.STEREO,
        sha256_hash=sha256_hash,
    )


def create_mono_views(
        mono: np.ndarray,
        sample_rate: int,
        sha256_hash: Optional[str] = None,
) -> AudioViews:
    """Create AudioViews from mono audio (duplicates to stereo)."""
    stereo = np.stack([mono, mono])
    return AudioViews(
        master_stereo=stereo,
        sample_rate=sample_rate,
        channel_layout=ChannelLayout.MONO,
        sha256_hash=sha256_hash,
    )


def views_from_loaded_audio(loaded_audio) -> AudioViews:
    """Create AudioViews from LoadedAudio object."""
    audio = loaded_audio.audio
    if audio.ndim == 1:
        return create_mono_views(audio, loaded_audio.sample_rate, loaded_audio.sha256_hash)
    elif audio.ndim == 2:
        if audio.shape[0] == 2:
            return create_stereo_views(audio[0], audio[1], loaded_audio.sample_rate, loaded_audio.sha256_hash)
        elif audio.shape[1] == 2:
            return create_stereo_views(audio[:, 0], audio[:, 1], loaded_audio.sample_rate, loaded_audio.sha256_hash)
    raise ValueError(f"Cannot create views from audio shape {audio.shape}")


# ====================================================================
# Exports
# ====================================================================

__all__ = [
    "ChannelLayout",
    "ImmutableAudio",
    "AudioViews",
    "create_audio_views",
    "create_stereo_views",
    "create_mono_views",
    "views_from_loaded_audio",
]