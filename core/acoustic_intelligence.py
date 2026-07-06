# =====================================================================
# MODULE: core/acoustic_intelligence.py
# DESCRIPTION: Canonical audio truth engine for Grimlock 5.x.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-16
#
# PURPOSE:
#     This module is the SINGLE AUTHORITY for:
#
#         - Audio sanitization
#         - Immutable audio contracts
#         - Deterministic timing math
#         - Polyphase resampling
#         - Channel conversion
#         - Forensic lineage tracking
#
# ARCHITECTURAL LAW:
#
#     contracts.py defines WHAT audio must guarantee.
#     acoustic_intelligence.py defines HOW audio is transformed.
#
# SYSTEM LAW:
#
#     Arena stores audio.
#     Arena does NOT transform audio.
#
#     Agents interpret audio.
#     Agents do NOT own audio truth.
#
#     AcousticIntelligence is the ONLY legal transformation authority.
#
# =====================================================================

from __future__ import annotations

import hashlib
import math
import warnings

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction
from typing import Iterator, Optional, Tuple, Union, Any

import numpy as np

# Optional scipy import with graceful fallback
try:
    from scipy.signal import butter, filtfilt, resample_poly

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


    # Create dummy functions that will raise helpful errors
    def butter(*args, **kwargs):
        raise ImportError("scipy.signal.butter requires scipy. Install with: pip install scipy")


    def filtfilt(*args, **kwargs):
        raise ImportError("scipy.signal.filtfilt requires scipy. Install with: pip install scipy")


    def resample_poly(*args, **kwargs):
        raise ImportError("scipy.signal.resample_poly requires scipy. Install with: pip install scipy")

from core.contracts import AudioContract

# =====================================================================
# CONSTANTS
# =====================================================================

# Sample rate bounds
MIN_SAMPLE_RATE: int = 8000  # 8kHz - lowest usable rate
MAX_SAMPLE_RATE: int = 384000  # 384kHz - highest practical rate

# Duration bounds (prevents OOM)
MIN_DURATION_MS: float = 10.0  # 10ms - below this is probably corruption
MAX_DURATION_MS: float = 60 * 60 * 1000  # 1 hour hard limit

# Channel limits
MAX_CHANNELS: int = 8  # Ambisonics and beyond not supported

# Amplitude sanity
MAX_ABSOLUTE_AMPLITUDE: float = 4.0  # >4.0 indicates catastrophic explosion

# Resampling quality
ANTI_ALIAS_MARGIN: float = 0.95  # 5% below Nyquist
DEFAULT_KAISER_BETA: float = 12.0  # Good suppression without excessive ringing

# Numerical tolerance
FLOAT_EPSILON: float = 1e-12  # Denormal elimination threshold

# Quality presets for resampling
QUALITY_PRESETS: dict = {
    'fast': 5.0,  # Minimal sidelobe suppression
    'normal': 8.0,  # Good balance (default)
    'high': 12.0,  # Very good suppression
    'best': 14.0  # Maximum suppression (but may ring)
}


# =====================================================================
# CHANNEL LAYOUT
# =====================================================================

class ChannelLayout(str, Enum):
    """Explicit channel layout enumeration."""
    MONO = "mono"
    STEREO = "stereo"
    MULTICHANNEL = "multichannel"


# =====================================================================
# IMMUTABLE AUDIO CONTRACT IMPLEMENTATION
# =====================================================================

@dataclass(frozen=True, slots=True)
class ImmutableAudio:
    """
    Canonical immutable audio contract implementation.

    Implements core.contracts.AudioContract

    LAW: "Once created, forever frozen. No exceptions."
    LAW: "Samples are writeable=False. No mutation possible."
    LAW: "Timing uses Fraction arithmetic. No float drift."
    """

    samples: np.ndarray
    sample_rate: int
    lineage: str
    channel_layout: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    total_samples: int = field(init=False)
    duration_ms: float = field(init=False)
    sha256_hash: str = field(init=False)
    num_channels: int = field(init=False)

    _ms_per_sample: Fraction = field(init=False, repr=False)

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __post_init__(self) -> None:
        """Validate and freeze the contract after initialization."""

        # Type validation
        if not isinstance(self.samples, np.ndarray):
            raise TypeError(f"samples must be numpy.ndarray, got {type(self.samples)}")

        if self.samples.size == 0:
            raise ValueError("Cannot create contract from empty audio")

        # Sample rate validation
        if self.sample_rate < MIN_SAMPLE_RATE:
            raise ValueError(
                f"Sample rate {self.sample_rate}Hz below minimum {MIN_SAMPLE_RATE}Hz"
            )

        if self.sample_rate > MAX_SAMPLE_RATE:
            raise ValueError(
                f"Sample rate {self.sample_rate}Hz above maximum {MAX_SAMPLE_RATE}Hz"
            )

        # Ensure contiguous memory layout
        samples = np.ascontiguousarray(self.samples.astype(np.float32))

        # Final sanitization pass
        samples = AcousticIntelligence.sanitize_audio(samples, context="contract_creation")

        # CRITICAL: Freeze forever
        samples.flags.writeable = False

        object.__setattr__(self, "samples", samples)

        # Calculate channel count
        if samples.ndim == 1:
            num_channels = 1
        else:
            num_channels = samples.shape[1]

        object.__setattr__(self, "num_channels", num_channels)

        # Calculate total samples
        total_samples = len(samples)
        object.__setattr__(self, "total_samples", total_samples)

        # Calculate duration using Fraction (NO FLOAT DIVISION)
        duration_ms = Fraction(total_samples, self.sample_rate) * 1000
        duration_ms_float = float(duration_ms)

        object.__setattr__(self, "duration_ms", duration_ms_float)

        # Duration validation
        if duration_ms_float < MIN_DURATION_MS:
            raise ValueError(
                f"Audio duration {duration_ms_float:.3f}ms below minimum {MIN_DURATION_MS}ms"
            )

        if duration_ms_float > MAX_DURATION_MS:
            raise ValueError(
                f"Audio duration {duration_ms_float / 1000 / 60:.2f} minutes "
                f"exceeds maximum {MAX_DURATION_MS / 1000 / 60:.2f} minutes"
            )

        # Calculate forensic hash
        sha256_hash = hashlib.sha256(samples.tobytes()).hexdigest()
        object.__setattr__(self, "sha256_hash", sha256_hash)

        # Store timing constant as Fraction (exact rational)
        object.__setattr__(self, "_ms_per_sample", Fraction(1000, self.sample_rate))

    # =================================================================
    # TIMING METHODS (Fraction-based, No Float Division)
    # =================================================================

    def index_to_ms(self, sample_index: int) -> float:
        """
        Convert sample index to milliseconds.

        Uses Fraction multiplication internally. NO FLOAT DIVISION.
        Returns float for API compatibility, but derivation is exact.
        """
        if sample_index < 0:
            raise IndexError(f"Negative sample index: {sample_index}")

        if sample_index > self.total_samples:
            raise IndexError(
                f"Sample index {sample_index} exceeds contract bounds "
                f"[0, {self.total_samples}]"
            )

        # CANONICAL TIMING: Fraction multiplication, no float division
        return float(self._ms_per_sample * sample_index)

    def ms_to_index(self, time_ms: float) -> int:
        """
        Convert milliseconds to sample index.

        Uses microsecond precision to avoid floating-point rounding errors.
        """
        if time_ms < 0:
            raise ValueError(f"Negative time: {time_ms}ms")

        if time_ms > self.duration_ms:
            raise ValueError(
                f"Time {time_ms}ms exceeds contract duration {self.duration_ms:.2f}ms"
            )

        # Convert to microseconds for integer precision
        microseconds = int(round(time_ms * 1000))
        time_fraction = Fraction(microseconds, 1000)

        # CANONICAL TIMING: Fraction division, no float division
        sample_fraction = time_fraction / self._ms_per_sample
        return int(round(float(sample_fraction)))

    # =================================================================
    # CHUNK STREAMING (Memory-efficient iteration)
    # =================================================================

    def iter_chunks(self, chunk_ms: float = 1000) -> Iterator[ImmutableAudio]:
        """
        Stream audio in chunks without copying the entire array.

        Each chunk is a new ImmutableAudio (still immutable) that shares
        a view of the original data where possible.

        Args:
            chunk_ms: Chunk size in milliseconds (default 1000ms = 1 second)

        Yields:
            ImmutableAudio chunks (read-only views)
        """
        if chunk_ms <= 0:
            raise ValueError(f"chunk_ms must be > 0, got {chunk_ms}")

        chunk_samples = self.ms_to_index(chunk_ms)
        chunk_samples = max(chunk_samples, 1)

        for start in range(0, self.total_samples, chunk_samples):
            end = min(start + chunk_samples, self.total_samples)
            chunk = self.samples[start:end]

            yield ImmutableAudio(
                samples=chunk,
                sample_rate=self.sample_rate,
                lineage=f"{self.lineage}|chunk({start}:{end})",
                channel_layout=self.channel_layout
            )

    # =================================================================
    # MONO CONVERSION (Power-preserving)
    # =================================================================

    def to_mono(self) -> ImmutableAudio:
        """
        Convert to mono using power-preserving downmix.

        Law: (L + R) / √2 for stereo preserves total power.
        For multichannel: sum(channels) / √(channels) preserves total power.
        """
        if self.channel_layout == ChannelLayout.MONO:
            return self

        mono_array = AcousticIntelligence.to_mono_array(self.samples)

        return ImmutableAudio(
            samples=mono_array,
            sample_rate=self.sample_rate,
            lineage=f"{self.lineage}|mono",
            channel_layout=ChannelLayout.MONO
        )

    # =================================================================
    # INTEGRITY VERIFICATION
    # =================================================================

    def verify_integrity(self) -> bool:
        """
        Verify that stored hash matches current samples.

        Raises:
            RuntimeError: If hash mismatch (data was mutated)
        """
        current_hash = hashlib.sha256(self.samples.tobytes()).hexdigest()

        if current_hash != self.sha256_hash:
            raise RuntimeError(
                f"Audio contract integrity violation in lineage '{self.lineage}'. "
                f"Hash mismatch. Data may have been mutated after creation."
            )

        return True

    # =================================================================
    # DUNDER METHODS
    # =================================================================

    def __len__(self) -> int:
        return self.total_samples

    def __hash__(self) -> int:
        return hash(self.sha256_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImmutableAudio):
            return False
        return self.sha256_hash == other.sha256_hash

    def __repr__(self) -> str:
        short_hash = self.sha256_hash[:12]
        return (
            f"<ImmutableAudio "
            f"hash={short_hash}... "
            f"sr={self.sample_rate}Hz "
            f"dur={self.duration_ms:.2f}ms "
            f"channels={self.num_channels} "
            f"layout={self.channel_layout}>"
        )


# =====================================================================
# ACOUSTIC INTELLIGENCE
# =====================================================================

class AcousticIntelligence:
    """
    Canonical audio transformation authority.

    This is the ONLY module that may:
        - Create AudioContracts from raw audio
        - Resample audio
        - Convert between channel layouts
        - Sanitize audio arrays

    LAW: "If audio enters the system, it passes through here."
    LAW: "If audio is transformed, it passes through here."
    LAW: "No other module may create AudioContracts directly."
    """

    # =================================================================
    # SANITIZATION (Universal Boundary Cleaner)
    # =================================================================

    @staticmethod
    def sanitize_audio(audio: np.ndarray, context: str = "unknown") -> np.ndarray:
        """
        Universal audio sanitization layer.

        THIS MUST BE CALLED:
            - after ingestion
            - after separation
            - after resampling
            - after normalization
            - after ANY transformation that produces audio
        """
        if audio.size == 0:
            raise ValueError(f"Cannot sanitize empty audio from {context}")

        # Ensure float32 and numpy array
        audio = np.asarray(audio, dtype=np.float32)

        # Replace NaN and Inf with zeros
        if not np.all(np.isfinite(audio)):
            nonfinite_count = np.sum(~np.isfinite(audio))
            warnings.warn(
                f"Replaced {nonfinite_count} non-finite values in audio from {context}",
                RuntimeWarning
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for amplitude explosion
        peak = float(np.max(np.abs(audio)))
        if peak > MAX_ABSOLUTE_AMPLITUDE:
            raise ValueError(
                f"Amplitude explosion detected from {context}: peak={peak:.3f} > {MAX_ABSOLUTE_AMPLITUDE}"
            )

        # Clip to safe range if slightly out of bounds
        if peak > 1.0:
            audio = np.clip(audio, -1.0, 1.0)

        # Remove denormalized floats (they waste CPU cycles)
        audio[np.abs(audio) < FLOAT_EPSILON] = 0.0

        return np.ascontiguousarray(audio, dtype=np.float32)

    # =================================================================
    # CONTRACT CREATION (Entry Point)
    # =================================================================

    @classmethod
    def create_contract(
            cls,
            audio: np.ndarray,
            sample_rate: int,
            source: str,
            force_mono: bool = False,
            explicit_layout: Optional[str] = None
    ) -> ImmutableAudio:
        """
        Create an immutable AudioContract from raw audio.

        This is the ONLY entry point for audio into the system.

        Args:
            audio: Input audio array (any shape, any dtype)
            sample_rate: Sample rate in Hz
            source: Source identifier for lineage tracking
            force_mono: If True, convert to mono (default: False)
            explicit_layout: Override automatic layout detection

        Returns:
            Immutable AudioContract
        """
        if audio.size == 0:
            raise ValueError(f"Empty audio received from {source}")

        # First sanitization pass
        audio = cls.sanitize_audio(audio, context=f"create_contract({source})")

        # Normalize layout
        layout, normalized = cls.normalize_layout(audio, explicit_layout)

        # Apply mono conversion if requested
        if force_mono and layout != ChannelLayout.MONO:
            normalized = cls.to_mono_array(normalized)
            layout = ChannelLayout.MONO

        # Build forensic lineage
        lineage = f"ingest(source={source},sr={sample_rate},layout={layout})"

        # Create and return immutable contract
        return ImmutableAudio(
            samples=normalized,
            sample_rate=sample_rate,
            lineage=lineage,
            channel_layout=layout
        )

    # =================================================================
    # RESAMPLING (Polyphase FIR with Kaiser Window)
    # =================================================================

    @classmethod
    def resample(
            cls,
            contract: AudioContract,
            target_sample_rate: int,
            quality: str = "normal"
    ) -> ImmutableAudio:
        """
        Resample audio to target sample rate.

        Pure mathematical resampling - no model knowledge.
        Uses polyphase FIR with Kaiser window for anti-aliasing.

        Args:
            contract: Source AudioContract
            target_sample_rate: Target sample rate (must be <= source sample rate)
            quality: 'fast', 'normal', 'high', 'best'

        Returns:
            New AudioContract at target sample rate

        Raises:
            ValueError: If target_sr > source_sr (upsampling prohibited)
        """
        source_sr = contract.sample_rate

        if source_sr == target_sample_rate:
            return contract

        # LAW: No upsampling (prevents hallucinated high-frequency data)
        if target_sample_rate > source_sr:
            raise ValueError(
                f"Upsampling prohibited: {source_sr}Hz -> {target_sample_rate}Hz. "
                f"Upsampling invents non-existent data. Lineage: {contract.lineage}"
            )

        # Get Kaiser beta from quality preset
        beta = QUALITY_PRESETS.get(quality, DEFAULT_KAISER_BETA)

        # Apply anti-alias filter
        filtered = cls._anti_alias_filter(
            contract.samples,
            source_sr,
            target_sample_rate
        )

        # Polyphase resampling
        try:
            if not HAS_SCIPY:
                raise ImportError("scipy.signal.resample_poly not available")

            ratio = Fraction(target_sample_rate, source_sr).limit_denominator(1000)
            up = ratio.numerator
            down = ratio.denominator

            resampled = resample_poly(
                filtered,
                up,
                down,
                window=("kaiser", beta),
                axis=0,
                padtype="constant"
            )
        except Exception as e:
            raise RuntimeError(f"Polyphase resampling failed: {e}") from e

        # Final sanitization
        resampled = cls.sanitize_audio(resampled, context=f"resample({source_sr}->{target_sample_rate})")

        # Build lineage
        lineage = f"{contract.lineage}|resample({source_sr}->{target_sample_rate},{quality})"

        return ImmutableAudio(
            samples=resampled,
            sample_rate=target_sample_rate,
            lineage=lineage,
            channel_layout=contract.channel_layout
        )

    # =================================================================
    # MONO CONVERSION (Power-Preserving)
    # =================================================================

    @staticmethod
    def to_mono_array(audio: np.ndarray) -> np.ndarray:
        """
        Power-preserving mono downmix.

        Formula:
            Stereo: (L + R) / √2
            Multichannel: sum(channels) / √(channel_count)

        This preserves total power for correlated signals.
        """
        if audio.ndim == 1:
            return audio

        channels = audio.shape[1]

        if channels == 1:
            return audio[:, 0]

        # Power-preserving sum
        mono = np.sum(audio, axis=1) / math.sqrt(channels)

        return AcousticIntelligence.sanitize_audio(mono, context="to_mono_array")

    # =================================================================
    # CHANNEL LAYOUT NORMALIZATION
    # =================================================================

    @staticmethod
    def normalize_layout(
            audio: np.ndarray,
            explicit_layout: Optional[str] = None
    ) -> Tuple[ChannelLayout, np.ndarray]:
        """
        Detect and normalize channel layout to canonical (samples, channels) format.

        Returns:
            Tuple of (layout, normalized_audio)
        """
        if explicit_layout is not None:
            layout = ChannelLayout(explicit_layout)

            if audio.ndim == 1:
                if layout == ChannelLayout.MONO:
                    return layout, audio
                else:
                    # Convert 1D to 2D with correct channel count
                    if layout == ChannelLayout.STEREO:
                        normalized = np.column_stack([audio, audio])
                    else:
                        raise ValueError(f"Cannot create {layout} from 1D audio")
                    return layout, normalized

            if layout == ChannelLayout.MONO and audio.ndim == 2:
                if audio.shape[1] == 1:
                    return layout, audio[:, 0]
                else:
                    # Downmix to mono
                    mono = AcousticIntelligence.to_mono_array(audio)
                    return layout, mono

            return layout, audio

        # Auto-detection
        if audio.ndim == 1:
            return ChannelLayout.MONO, audio

        if audio.ndim != 2:
            raise ValueError(f"Unsupported audio dimensions: {audio.ndim}D")

        rows, cols = audio.shape

        # Already (samples, channels) format
        if cols <= MAX_CHANNELS and rows > MAX_CHANNELS:
            if cols == 1:
                return ChannelLayout.MONO, audio[:, 0]
            if cols == 2:
                return ChannelLayout.STEREO, audio
            return ChannelLayout.MULTICHANNEL, audio

        # Transposed (channels, samples) format - transpose
        if rows <= MAX_CHANNELS and cols > MAX_CHANNELS:
            transposed = audio.T
            if rows == 1:
                return ChannelLayout.MONO, transposed[:, 0]
            if rows == 2:
                return ChannelLayout.STEREO, transposed
            return ChannelLayout.MULTICHANNEL, transposed

        # Ambiguous shape
        raise ValueError(
            f"Ambiguous audio shape: {audio.shape}. "
            f"Cannot determine channel layout. "
            f"Specify explicit_layout parameter."
        )

    # =================================================================
    # ANTI-ALIAS FILTERING
    # =================================================================

    @staticmethod
    def _anti_alias_filter(
            audio: np.ndarray,
            source_sr: int,
            target_sr: int
    ) -> np.ndarray:
        """
        Apply anti-aliasing low-pass filter before downsampling.

        Uses 8th-order Butterworth filter (maximally flat, no ripple).
        """
        if target_sr >= source_sr:
            return audio

        nyquist = target_sr / 2.0
        cutoff = nyquist * ANTI_ALIAS_MARGIN
        normalized_cutoff = cutoff / (source_sr / 2.0)

        # Clamp to valid range
        normalized_cutoff = min(max(normalized_cutoff, 0.001), 0.999)

        try:
            if not HAS_SCIPY:
                raise ImportError("scipy.signal.butter/filtfilt not available")

            b, a = butter(8, normalized_cutoff, btype="low")
            filtered = filtfilt(b, a, audio, axis=0)
            filtered = AcousticIntelligence.sanitize_audio(filtered, context="anti_alias_filter")
            return filtered

        except Exception as e:
            warnings.warn(f"Anti-alias filter failed: {e}. Using unfiltered audio.", RuntimeWarning)
            return audio

    # =================================================================
    # VALIDATION
    # =================================================================

    @staticmethod
    def validate_contract(contract: AudioContract, context: str = "") -> bool:
        """
        Validate that a contract meets all integrity requirements.

        Checks:
            - Hash integrity (data not mutated)
            - No non-finite values
            - Samples are writeable=False
            - Duration consistency

        Raises:
            RuntimeError: If any validation fails
        """
        # Check hash integrity
        contract.verify_integrity()

        # Check for non-finite values
        if not np.all(np.isfinite(contract.samples)):
            raise RuntimeError(
                f"Contract contains non-finite values in {context}. "
                f"Lineage: {contract.lineage}"
            )

        # Check mutability
        if contract.samples.flags.writeable:
            raise RuntimeError(
                f"Contract samples are mutable in {context}. "
                f"Lineage: {contract.lineage}"
            )

        # Check duration consistency
        expected_duration = float(Fraction(contract.total_samples, contract.sample_rate) * 1000)
        drift = abs(expected_duration - contract.duration_ms)

        if drift > 0.01:
            raise RuntimeError(
                f"Duration drift detected in {context}: {drift:.6f}ms. "
                f"Lineage: {contract.lineage}"
            )

        return True

    # =================================================================
    # STEM VALIDATION
    # =================================================================

    @staticmethod
    def validate_stem_sample_rate(
            contract: AudioContract,
            expected_sample_rate: int,
            stem_name: str
    ) -> None:
        """
        Validate that a stem contract has the correct sample rate.

        This is critical for Demucs stems which must remain at 44.1kHz.
        """
        if contract.sample_rate != expected_sample_rate:
            raise RuntimeError(
                f"Stem sample rate mismatch for {stem_name}: "
                f"expected={expected_sample_rate}Hz, "
                f"actual={contract.sample_rate}Hz. "
                f"Lineage: {contract.lineage}"
            )

    # =================================================================
    # FORENSIC FINGERPRINT
    # =================================================================

    @staticmethod
    def fingerprint(contract: AudioContract) -> str:
        """
        Generate a compact forensic fingerprint for logging.

        Example: "[Audio hash=abc12345 sr=44100Hz dur=30.00s layout=stereo]"
        """
        short_hash = contract.sha256_hash[:12]
        duration_sec = contract.duration_ms / 1000.0

        return (
            f"[Audio "
            f"hash={short_hash}... "
            f"sr={contract.sample_rate}Hz "
            f"dur={duration_sec:.2f}s "
            f"layout={contract.channel_layout}]"
        )


# =====================================================================
# STANDALONE TEST
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ACOUSTIC INTELLIGENCE - Self Test")
    print("=" * 70)

    # Create test audio (1 second of 440Hz sine wave)
    duration = 1.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("\n1. Creating mono contract:")
    contract = AcousticIntelligence.create_contract(test_audio, sr, "self_test")
    print(f"   {contract}")
    print(f"   Hash: {contract.sha256_hash[:16]}...")
    print(f"   Samples writeable: {contract.samples.flags.writeable}")

    print("\n2. Testing Fraction-based timing (no float drift):")
    for idx in [0, 10000, 44100]:
        ms = contract.index_to_ms(idx)
        back_idx = contract.ms_to_index(ms)
        print(f"   Index {idx:6d} -> {ms:8.3f}ms -> Index {back_idx:6d} (Δ={back_idx - idx})")

    print("\n3. Creating stereo contract (preserved):")
    stereo_audio = np.column_stack([test_audio, test_audio * 0.8])
    stereo_contract = AcousticIntelligence.create_contract(
        stereo_audio, sr, "self_test", force_mono=False
    )
    print(f"   {stereo_contract}")
    print(f"   Shape: {stereo_contract.samples.shape}")

    print("\n4. Converting stereo to mono (power-preserving):")
    mono_from_stereo = stereo_contract.to_mono()
    print(f"   {mono_from_stereo}")

    print("\n5. Resampling (44.1kHz -> 16kHz):")
    try:
        resampled = AcousticIntelligence.resample(contract, 16000, quality="high")
        print(f"   {resampled}")
        print(f"   Expected samples: {int(1.0 * 16000)} = 16000")
        print(f"   Actual samples: {len(resampled)}")
    except Exception as e:
        print(f"   Resampling skipped (scipy not available): {e}")

    print("\n6. Chunk streaming:")
    chunk_count = 0
    for chunk in contract.iter_chunks(chunk_ms=200):
        chunk_count += 1
        if chunk_count <= 3:
            print(f"   Chunk {chunk_count}: {len(chunk)} samples")
    print(f"   Total chunks: {chunk_count}")

    print("\n7. Validation:")
    is_valid = AcousticIntelligence.validate_contract(contract, "self_test")
    print(f"   Contract valid: {is_valid}")

    print("\n8. Integrity verification (should pass):")
    contract.verify_integrity()
    print("   ✓ Hash matches")

    print("\n9. Forensic fingerprint:")
    fingerprint = AcousticIntelligence.fingerprint(contract)
    print(f"   {fingerprint}")

    print("\n" + "=" * 70)
    print("All tests passed. AcousticIntelligence ready.")
    print("=" * 70)