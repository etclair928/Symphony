#!/usr/bin/env python3
"""
agents/separation/roformer.py — BS_Roformer Separation Agent for Grimlock 5.6.1
VERSION: 5.6.1

5.0 LAW:
    - Agents OBSERVE. They do not DECIDE.
    - Agents import from CORE only (order_types, constants, protocols)
    - Agents return TESTIMONY (observations + confidence), not verdicts.
    - Agents MUST declare their SourceType (for forensic logging).

RESPONSIBILITIES:
    - Separate audio into stems using BS_Roformer
    - Return stems as floating-point arrays
    - Report confidence per stem based on separation quality
    - Support spectral gating for cleaner separation
    - NEVER decide which stem is "correct" — only testify.

MIGRATION NOTE:
    BS_Roformer is being replaced by Mel Roformer in 5.1.
    This code is kept for backward compatibility but will be deprecated.
    New implementations should use MelRoformerSeparator (commented below).
"""

import gc
import time
import warnings
import numpy as np
from typing import Dict, Optional, Callable, Any, Tuple, List
from dataclasses import dataclass, field

# Core imports ONLY — no other agents
from core.order_types import (
    SourceType, StemType, Confidence, AudioContext,
    StageResult, SeparationResult
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    STAGE_TIMEOUT_SEPARATION_SECONDS,
    MEMORY_CRITICAL_THRESHOLD_MB,
    GC_COLLECT_AFTER_EACH_STAGE,
    BS_ROFORMER_MODEL,
    BS_ROFORMER_HOP_LENGTH_SAMPLES,
    BS_ROFORMER_WINDOW_SIZE_SAMPLES
)
from core.protocols import (
    SeparationAgentProtocol, MemoryManagedProtocol, MusicBoxProtocol
)

# Optional imports — gracefully degrade
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. BS_Roformer will not work.")

try:
    from bs_roformer import BSRoformer

    BS_ROFORMER_AVAILABLE = True
except ImportError:
    BS_ROFORMER_AVAILABLE = False
    warnings.warn("bs_roformer not available. BSRoformerSeparator will use fallback.")

# Optional spectral gating for cleaner separation
try:
    import noisereduce as nr

    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    warnings.warn("noisereduce not available. Spectral gating disabled.")

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ============================================================================
# MEL ROFORMER (Future replacement - COMMENTED OUT FOR NOW)
# ============================================================================

"""
# TODO: Uncomment and implement for 5.1
# from mel_roformer import MelRoformer  # New model for 5.1

class MelRoformerSeparator:
    '''
    Mel Roformer separator — the future replacement for BS_Roformer.
    Better separation quality, lower memory footprint.

    MIGRATION: This will replace BSRoformerSeparator in 5.1.
    Keep BS_Roformer as fallback during transition.
    '''

    def __init__(self, model_name: str = "mel_roformer_16k", device: str = "auto"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._load_time_ms = 0

    def separate(self, audio: np.ndarray, sample_rate: int) -> Dict[StemType, np.ndarray]:
        '''Separate using Mel Roformer.'''
        # Implementation coming in 5.1
        pass
"""


# ============================================================================
# BS ROFORMER TESTIMONY (Observation Record)
# ============================================================================

@dataclass
class BSRoformerTestimony:
    """
    What BS_Roformer observed — not a verdict, just testimony.
    This is the data that gets passed to Scribe for validation.
    """
    stems: Dict[StemType, np.ndarray]
    confidence: Dict[StemType, Confidence]
    model_name: str
    device: str
    separation_time_ms: float
    peak_memory_mb: float
    sample_rate: int
    spectral_gating_applied: bool = False
    fallback_used: bool = False

    def to_stage_result(self, stage_name: str = "bs_roformer") -> StageResult:
        """Convert testimony to StageResult for pipeline."""
        return StageResult(
            stage_name=stage_name,
            success=len(self.stems) > 0,
            events=[],  # Separation agents don't produce NoteEvents
            metadata={
                "stems": {k.value: v.shape for k, v in self.stems.items()},
                "confidence": {k.value: v.value for k, v in self.confidence.items()},
                "model_name": self.model_name,
                "device": self.device,
                "separation_time_ms": self.separation_time_ms,
                "peak_memory_mb": self.peak_memory_mb,
                "sample_rate": self.sample_rate,
                "spectral_gating_applied": self.spectral_gating_applied,
                "fallback_used": self.fallback_used
            },
            veto_reason=None,
            execution_time_ms=self.separation_time_ms,
            memory_delta_mb=self.peak_memory_mb
        )


# ============================================================================
# MAIN BS ROFORMER SEPARATOR AGENT
# ============================================================================

class BSRoformerSeparator(SeparationAgentProtocol, MemoryManagedProtocol):
    """
    BS_Roformer separation agent for Grimlock 5.0.

    TESTIFIES (does not decide):
        - Here are the separated stems (drums, bass, vocals, other)
        - Here is my confidence for each stem
        - Here is how long it took
        - Here is my memory usage

    DOES NOT:
        - Decide which stem is "correct"
        - Route to other agents
        - Make decisions about what to do with stems

    DEPRECATION WARNING:
        BS_Roformer is being replaced by Mel Roformer in 5.1.
        This agent is maintained for backward compatibility.
        New code should use MelRoformerSeparator when available.
    """

    # Known stem order for BS_Roformer
    STEM_ORDER = ["drums", "bass", "vocals", "other"]

    def __init__(
            self,
            model_name: str = BS_ROFORMER_MODEL,
            hop_length: int = BS_ROFORMER_HOP_LENGTH_SAMPLES,
            window_size: int = BS_ROFORMER_WINDOW_SIZE_SAMPLES,
            device: str = "auto",
            use_spectral_gating: bool = True,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Args:
            model_name: BS_Roformer model ('bs_roformer_16k', 'bs_roformer_32k', 'bs_roformer_44k')
            hop_length: Hop length in samples
            window_size: Window size in samples
            device: 'auto', 'cpu', or 'cuda'
            use_spectral_gating: Apply noise reduction after separation
            music_box: Optional forensic logger
            progress_callback: Optional progress reporter
        """
        self._name = "bs_roformer_separator"
        self._source_type = SourceType.BS_ROFORMER
        self._model_name = model_name
        self._hop_length = hop_length
        self._window_size = window_size
        self._device = self._resolve_device(device)
        self._use_spectral_gating = use_spectral_gating and NOISE_REDUCE_AVAILABLE
        self._music_box = music_box
        self._progress_callback = progress_callback
        self._model = None
        self._sample_rate = self._extract_sample_rate(model_name)
        self._load_time_ms = 0
        self._memory_pressure = False

        # Lazy load model
        self._init_model()

    # ========================================================================
    # AgentProtocol Implementation
    # ========================================================================

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """
        Main entry point for pipeline.
        Converts testimony to StageResult.
        """
        testimony = self.separate(audio_buffer, context)
        return testimony.to_stage_result(self._name)

    # ========================================================================
    # SeparationAgentProtocol Implementation
    # ========================================================================

    def separate(
            self,
            audio_buffer: np.ndarray,
            context: AudioContext
    ) -> SeparationResult:
        """
        Separate audio into stems.

        Preserves:
            - Original sample rate (resamples internally)
            - Channel count (mono output typical for BS_Roformer)

        Returns:
            SeparationResult with stems dict
        """
        start_time = time.time()
        self._update_progress(0.0, "Starting BS_Roformer separation")

        # Use original sample rate from context or detect
        sample_rate = context.original_sample_rate if context else self._sample_rate

        # Validate input
        if audio_buffer is None or len(audio_buffer) == 0:
            return self._empty_result(sample_rate, context)

        # Track memory before
        memory_before = self.get_memory_footprint_mb()

        # Ensure mono for BS_Roformer (it expects mono)
        audio, was_stereo = self._ensure_mono_format(audio_buffer)

        # Run separation
        if BS_ROFORMER_AVAILABLE and TORCH_AVAILABLE and self._model is not None:
            stems, confidences, fallback = self._run_roformer(audio, sample_rate)
        else:
            stems, confidences = self._fallback_separation(audio)
            fallback = True

        separation_time = time.time() - start_time
        memory_after = self.get_memory_footprint_mb()

        self._update_progress(1.0, "Separation complete")

        # Log testimony
        self._log_separation(stems, confidences, separation_time, memory_after, fallback)

        # Return SeparationResult (from order_types)
        return SeparationResult(
            stems=stems,
            separation_time_seconds=separation_time,
            memory_usage_mb=memory_after,
            confidence=Confidence.HIGH if not fallback else Confidence.LOW
        )

    # ========================================================================
    # SeparationAgentProtocol Additional Properties
    # ========================================================================

    @property
    def available_stems(self) -> List[StemType]:
        """Which stems this separator can produce."""
        return [StemType.DRUMS, StemType.BASS, StemType.VOCALS, StemType.OTHER]

    @property
    def separation_quality(self) -> str:
        """'fast', 'balanced', 'high'"""
        if "44k" in self._model_name:
            return "high"
        elif "32k" in self._model_name:
            return "balanced"
        return "fast"

    # ========================================================================
    # MemoryManagedProtocol Implementation
    # ========================================================================

    def release_buffer(self, buffer_name: str) -> None:
        """Release a buffer by name."""
        if hasattr(self, buffer_name):
            delattr(self, buffer_name)
            self._maybe_gc()

    def get_memory_footprint_mb(self) -> float:
        """Current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Fallback: estimate from model
        if self._model is not None:
            return 250.0  # Rough estimate for BS_Roformer
        return 0.0

    def can_release(self, buffer_name: str) -> bool:
        """Return True if this buffer is safe to delete."""
        # Stems are safe to release after they've been passed downstream
        return buffer_name.startswith("stem_")

    def staggered_gc(self) -> Dict[str, Any]:
        """Run garbage collection and return stats."""
        before = self.get_memory_footprint_mb()
        gc.collect()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        after = self.get_memory_footprint_mb()

        return {
            "memory_before_mb": before,
            "memory_after_mb": after,
            "freed_mb": before - after,
            "timestamp": time.time()
        }

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _resolve_device(self, device: str) -> str:
        """Resolve device string — observation only, no decision."""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _extract_sample_rate(self, model_name: str) -> int:
        """Extract sample rate from model name."""
        model_lower = model_name.lower()
        if "44k" in model_lower or "44.1k" in model_lower:
            return 44100
        elif "32k" in model_lower:
            return 32000
        elif "16k" in model_lower:
            return 16000
        else:
            return 16000  # Default

    def _init_model(self):
        """Lazy initialization of BS_Roformer model."""
        if not BS_ROFORMER_AVAILABLE or not TORCH_AVAILABLE:
            self._model = None
            return

        load_start = time.time()

        try:
            # BS_Roformer initialization pattern from 4.7
            self._model = BSRoformer(
                model_name=self._model_name,
                device=self._device,
                hop_length=self._hop_length,
                window_size=self._window_size
            )
            self._load_time_ms = (time.time() - load_start) * 1000

            if self._music_box:
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="model_loaded",
                    before_state={"model": None},
                    after_state={
                        "model": self._model_name,
                        "device": self._device,
                        "sample_rate": self._sample_rate
                    },
                    reasoning=f"BS_Roformer {self._model_name} on {self._device} in {self._load_time_ms:.0f}ms",
                    reversible=False
                )
        except Exception as e:
            warnings.warn(f"Failed to load BS_Roformer: {e}")
            self._model = None

    def _ensure_mono_format(self, audio: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Ensure audio is mono for BS_Roformer.
        Returns (mono_audio, was_stereo)
        """
        was_stereo = False

        if audio.ndim == 1:
            # Already mono
            mono = audio
        elif audio.ndim == 2:
            was_stereo = True
            if audio.shape[0] == 2:
                # (2, N) format -> average to mono
                mono = np.mean(audio, axis=0)
            elif audio.shape[1] == 2:
                # (N, 2) format -> average to mono
                mono = np.mean(audio, axis=1)
            else:
                # Unknown shape, take first channel
                mono = audio[0] if audio.shape[0] > 0 else audio.flatten()
        else:
            # Fallback: flatten
            mono = audio.flatten()

        return mono.astype(np.float32), was_stereo

    def _run_roformer(
            self,
            audio: np.ndarray,
            sample_rate: int
    ) -> Tuple[Dict[StemType, np.ndarray], Dict[StemType, Confidence], bool]:
        """Run actual BS_Roformer separation — pure observation."""
        fallback = False

        self._update_progress(0.2, "Converting to tensor")

        # Resample if needed
        if sample_rate != self._sample_rate and LIBROSA_AVAILABLE:
            self._update_progress(0.1, f"Resampling to {self._sample_rate}Hz")
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self._sample_rate)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Add batch and channel dimensions if needed
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        elif len(audio_tensor.shape) == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, channels, N)

        # Move to device
        device = torch.device(self._device)
        audio_tensor = audio_tensor.to(device)

        self._update_progress(0.4, "Running BS_Roformer model")

        # Run separation with no_grad (observation only)
        try:
            with torch.no_grad():
                stems_raw = self._model.separate(audio_tensor) if hasattr(self._model, 'separate') else {}
        except Exception as e:
            warnings.warn(f"BS_Roformer separation failed: {e}. Using fallback.")
            fallback = True
            stems_raw = {}

        self._update_progress(0.8, "Extracting stems")

        # Extract stems
        stems = {}
        confidences = {}

        for idx, stem_name in enumerate(self.STEM_ORDER):
            stem_type = self._map_stem_name(stem_name)

            if stems_raw and stem_name in stems_raw:
                stem = stems_raw[stem_name]
                if hasattr(stem, 'cpu'):
                    stem = stem.cpu().numpy()
                if len(stem.shape) > 1:
                    stem = np.mean(stem, axis=0)  # Downmix to mono
                stems[stem_type] = stem.astype(np.float32)
                confidences[stem_type] = self._estimate_confidence(stem)
            elif stems_raw and isinstance(stems_raw, (list, tuple)) and idx < len(stems_raw):
                stem = stems_raw[idx]
                if hasattr(stem, 'cpu'):
                    stem = stem.cpu().numpy()
                if len(stem.shape) > 1:
                    stem = np.mean(stem, axis=0)
                stems[stem_type] = stem.astype(np.float32)
                confidences[stem_type] = self._estimate_confidence(stem)
            else:
                # Empty stem
                stems[stem_type] = np.zeros_like(audio, dtype=np.float32)
                confidences[stem_type] = Confidence.HALLUCINATION

        # Apply spectral gating if enabled
        if self._use_spectral_gating and not fallback:
            stems = self._apply_spectral_gating(stems, audio)

        # Resample back if needed
        if sample_rate != self._sample_rate and LIBROSA_AVAILABLE:
            for stem_type in stems:
                stems[stem_type] = librosa.resample(
                    stems[stem_type],
                    orig_sr=self._sample_rate,
                    target_sr=sample_rate
                )

        # Clean up GPU memory
        del audio_tensor
        if stems_raw:
            del stems_raw
        self._maybe_gc()

        return stems, confidences, fallback

    def _map_stem_name(self, roformer_name: str) -> StemType:
        """Map BS_Roformer stem name to our StemType."""
        name_lower = roformer_name.lower()
        if "drum" in name_lower:
            return StemType.DRUMS
        elif "bass" in name_lower:
            return StemType.BASS
        elif "vocal" in name_lower:
            return StemType.VOCALS
        else:
            return StemType.OTHER

    def _estimate_confidence(self, stem: np.ndarray) -> Confidence:
        """
        Estimate separation confidence based on stem energy.

        Higher energy = more confident this stem contains actual audio.
        """
        if stem is None or stem.size == 0:
            return Confidence.HALLUCINATION

        # RMS energy
        rms = np.sqrt(np.mean(stem ** 2))

        # Peak amplitude
        peak = np.max(np.abs(stem))

        # Combined confidence (0-1)
        confidence_value = min(0.9, (rms * 2 + peak * 0.3))

        if confidence_value < Confidence.LOW.value:
            return Confidence.HALLUCINATION
        elif confidence_value < Confidence.MEDIUM.value:
            return Confidence.LOW
        elif confidence_value < Confidence.HIGH.value:
            return Confidence.MEDIUM
        else:
            return Confidence.HIGH

    def _apply_spectral_gating(
            self,
            stems: Dict[StemType, np.ndarray],
            original_audio: np.ndarray
    ) -> Dict[StemType, np.ndarray]:
        """
        Apply spectral gating noise reduction to stems.

        Based on 4.7's spectral gating technique for cleaner separation.
        Less aggressive for drums to preserve transients.
        """
        if not NOISE_REDUCE_AVAILABLE:
            return stems

        for stem_type, stem in stems.items():
            if stem is not None and len(stem) > 0:
                try:
                    # Adjust aggression based on stem type
                    if stem_type == StemType.DRUMS:
                        # Less aggressive for drums (preserve transients)
                        prop_decrease = 0.4
                    elif stem_type == StemType.BASS:
                        # Medium for bass
                        prop_decrease = 0.6
                    else:
                        # More aggressive for vocals and other
                        prop_decrease = 0.7

                    stems[stem_type] = nr.reduce_noise(
                        y=stem,
                        sr=self._sample_rate,
                        prop_decrease=prop_decrease,
                        verbose=False
                    )
                except Exception as e:
                    warnings.warn(f"Spectral gating failed for {stem_type}: {e}")

        return stems

    def _fallback_separation(
            self,
            audio: np.ndarray
    ) -> Tuple[Dict[StemType, np.ndarray], Dict[StemType, Confidence]]:
        """Fallback when BS_Roformer unavailable."""
        stems = {}
        confidences = {}

        # Original as "other" stem
        stems[StemType.OTHER] = audio.astype(np.float32)
        confidences[StemType.OTHER] = Confidence.HIGH

        # Empty stems for drums, bass, vocals
        empty_stem = np.zeros_like(audio, dtype=np.float32)
        stems[StemType.DRUMS] = empty_stem
        stems[StemType.BASS] = empty_stem
        stems[StemType.VOCALS] = empty_stem

        confidences[StemType.DRUMS] = Confidence.HALLUCINATION
        confidences[StemType.BASS] = Confidence.HALLUCINATION
        confidences[StemType.VOCALS] = Confidence.HALLUCINATION

        return stems, confidences

    def _empty_result(self, sample_rate: int, context: Optional[AudioContext]) -> SeparationResult:
        """Return empty result for silence/error cases."""
        duration = context.duration_seconds if context else 0
        empty_stem = np.zeros(int(sample_rate * duration), dtype=np.float32)

        stems = {
            StemType.DRUMS: empty_stem.copy(),
            StemType.BASS: empty_stem.copy(),
            StemType.VOCALS: empty_stem.copy(),
            StemType.OTHER: empty_stem.copy()
        }

        return SeparationResult(
            stems=stems,
            separation_time_seconds=0,
            memory_usage_mb=0,
            confidence=Confidence.HALLUCINATION
        )

    def _update_progress(self, progress: float, message: str):
        """Report progress — observation only."""
        if self._progress_callback:
            self._progress_callback(progress, message)

    def _log_separation(
            self,
            stems: Dict,
            confidences: Dict,
            elapsed: float,
            memory: float,
            fallback: bool
    ):
        """Log testimony for forensic audit."""
        if not self._music_box:
            return

        stem_shapes = {k.value: list(v.shape) for k, v in stems.items()}
        confidence_values = {k.value: v.value for k, v in confidences.items()}

        self._music_box.log_decision(
            stage_name=self._name,
            decision_type="separation_testimony",
            before_state={"audio_loaded": True},
            after_state={
                "stems": list(stems.keys()),
                "stem_shapes": stem_shapes,
                "confidence": confidence_values,
                "fallback_used": fallback,
                "spectral_gating": self._use_spectral_gating
            },
            reasoning=(
                    f"BS_Roformer separated {len(stems)} stems in {elapsed:.1f}s"
                    + (f" (FALLBACK MODE)" if fallback else "")
            ),
            reversible=False
        )

    def _maybe_gc(self):
        """Conditional garbage collection based on memory pressure."""
        memory_mb = self.get_memory_footprint_mb()
        self._memory_pressure = memory_mb > MEMORY_CRITICAL_THRESHOLD_MB

        if GC_COLLECT_AFTER_EACH_STAGE or self._memory_pressure:
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release(self):
        """Release model and clear memory."""
        if self._model is not None:
            if TORCH_AVAILABLE and hasattr(self._model, 'cpu'):
                self._model.cpu()
            del self._model
            self._model = None

        self._maybe_gc()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "available": BS_ROFORMER_AVAILABLE,
            "model": self._model_name,
            "device": self._device,
            "sample_rate": self._sample_rate,
            "hop_length": self._hop_length,
            "window_size": self._window_size,
            "load_time_ms": self._load_time_ms,
            "spectral_gating": self._use_spectral_gating
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_roformer_separator(
        model_name: str = BS_ROFORMER_MODEL,
        device: str = "auto",
        use_spectral_gating: bool = True,
        music_box: Optional[MusicBoxProtocol] = None
) -> BSRoformerSeparator:
    """Create a BS_Roformer separator agent."""
    return BSRoformerSeparator(
        model_name=model_name,
        device=device,
        use_spectral_gating=use_spectral_gating,
        music_box=music_box
    )


"""
# TODO: Uncomment for 5.1 when Mel Roformer is ready
def create_mel_roformer_separator(
    model_name: str = "mel_roformer_16k",
    device: str = "auto",
    music_box: Optional[MusicBoxProtocol] = None
) -> MelRoformerSeparator:
    '''Create a Mel Roformer separator agent.'''
    return MelRoformerSeparator(
        model_name=model_name,
        device=device,
        music_box=music_box
    )
"""


# ============================================================================
# DEPRECATION WARNING
# ============================================================================

def __getattr__(name: str):
    """Emit deprecation warning for BS_Roformer access."""
    if name == "BSRoformerSeparator":
        warnings.warn(
            "BSRoformerSeparator is deprecated and will be removed in 5.1. "
            "Use MelRoformerSeparator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return BSRoformerSeparator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")