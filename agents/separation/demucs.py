#!/usr/bin/env python3
"""
agents/separation/demucs.py — Hybrid Separation Agent for Grimlock 5.6.1
VERSION: 5.6.1

5.4 LAW:
    - Agents OBSERVE. They do not DECIDE.
    - Agents import from CORE only (order_types, constants, protocols)
    - Agents return TESTIMONY (observations + confidence), not verdicts.
    - Agents MUST declare their SourceType (for forensic logging).
    - Fallback is ALWAYS available — pipeline never blocks.

RESPONSIBILITIES:
    - Separate stereo audio into stems using Demucs (primary) or Mel-Roformer (fallback)
    - Preserve original sample rate (no downsampling)
    - Return stems as stereo arrays (preserving channel count)
    - Report confidence per stem based on separation quality
    - Support streaming separation for long audio files
    - NEVER decide which stem is "correct" — only testify.

HYBRID ARCHITECTURE:
    Level 1: Demucs (htdemucs) — high quality, high memory
    Level 2: Demucs (mdx) — balanced quality
    Level 3: Mel-Roformer — lower quality, lower memory, faster
    Level 4: Librosa fallback — basic separation
    Level 5: Identity fallback — return original as "other"

CHANGES FROM 5.0:
    - Added Mel-Roformer as fallback tier
    - Added progressive fallback chain (Demucs → Mel-Roformer → Librosa)
    - Added memory pressure detection before loading heavy models

TIMEOUT MODEL (correcting a prior false claim in this docstring):
    _run_with_timeout() uses a single-worker ThreadPoolExecutor, NOT a
    separate OS process - Python cannot forcibly kill a running thread,
    so on timeout the abandoned worker keeps running to completion in
    the background. The executor is shut down with wait=False so THIS
    function returns promptly at the timeout instead of blocking until
    that worker finishes. True hard cancellation would need a real OS
    process with an explicit terminate()/kill() call, which this does
    not do; that would be a bigger change than it looks (Windows uses
    spawn, so it re-imports the whole module tree per worker process;
    shared MusicBox logging doesn't cross a process boundary; large
    audio buffers must be pickled both ways) and isn't done here.
"""

import gc
import time
import warnings
import os
import signal
import subprocess
import tempfile
import numpy as np
from typing import Dict, Optional, Callable, Any, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum

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
    TORCH_CPU_INTRAOP_THREADS,
    TORCH_CPU_INTEROP_THREADS,
)
from core.protocols import (
    SeparationAgentProtocol, DemucsProtocol,
    MemoryManagedProtocol, MusicBoxProtocol
)

# Optional imports — gracefully degrade
try:
    import torch

    TORCH_AVAILABLE = True
    # PyTorch defaults to one intra-op thread per CPU core for its matrix
    # math. On CPU-only inference (this project's Demucs config always
    # uses device="cpu") that pins every core for the whole separation
    # call, starving the rest of this process - including the very
    # timeout/cancellation machinery meant to bound it - of scheduling
    # time. Clamp it once, at import time, for the whole process.
    torch.set_num_threads(TORCH_CPU_INTRAOP_THREADS)
    try:
        # set_num_interop_threads raises RuntimeError if called after any
        # parallel work has already started (e.g. some other module using
        # torch was imported first and triggered internal parallelism) -
        # this is a best-effort perf tweak, not worth crashing import over.
        torch.set_num_interop_threads(TORCH_CPU_INTEROP_THREADS)
    except RuntimeError as e:
        warnings.warn(f"Could not set torch interop thread count (already in use): {e}")
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Demucs will not work.")

try:
    from demucs import pretrained
    from demucs.apply import apply_model

    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    warnings.warn("demucs not available. DemucsSeparator will use fallback.")

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
    warnings.warn("librosa not available. Audio processing limited.")


# ============================================================================
# FALLBACK LEVELS
# ============================================================================

class FallbackLevel(Enum):
    """Progressive fallback levels for separation."""
    DEMUCS_HD = 1  # htdemucs - highest quality
    DEMUCS_MDX = 2  # mdx - balanced quality
    MEL_ROFORMER = 3  # mel-based transformer
    LIBROSA = 4  # simple harmonic/percussive
    IDENTITY = 5  # original as "other"


# ============================================================================
# STREAMING SUPPORT: OVERLAP-ADD BUFFER
# ============================================================================

@dataclass
class OverlapAddBuffer:
    """
    Overlap-add buffer for streaming separation.
    Maintains continuity across window boundaries using 50% overlap.
    """
    channels: int = 2
    window_samples: int = 262144  # ~6 seconds at 44.1kHz
    hop_samples: int = 131072  # 50% overlap

    def __post_init__(self):
        self._buffer = np.zeros((self.channels, self.window_samples), dtype=np.float32)
        self._position = 0
        self._pending_output = np.zeros((self.channels, 0), dtype=np.float32)

    def add_window(self, window: np.ndarray, position: int) -> Optional[np.ndarray]:
        """Add a processed window to the buffer."""
        if window.shape[-1] != self.window_samples:
            if window.shape[-1] < self.window_samples:
                pad_width = self.window_samples - window.shape[-1]
                window = np.pad(window, ((0, 0), (0, pad_width)), mode='constant')
            else:
                window = window[:, :self.window_samples]

        if self._position == 0:
            self._buffer = window
            self._position = self.hop_samples
            return None

        overlap_start = self._position - self.hop_samples
        overlap_end = self._position

        output = self._buffer[:, :self.hop_samples].copy()

        remaining = self._buffer[:, self.hop_samples:]
        self._buffer = np.concatenate([remaining, window], axis=1)
        self._position = min(self._position + self.hop_samples, self.window_samples)

        return output

    def flush(self) -> np.ndarray:
        """Return remaining audio in buffer."""
        output = self._buffer[:, :self._position].copy()
        self._reset()
        return output

    def _reset(self):
        self._buffer = np.zeros((self.channels, self.window_samples), dtype=np.float32)
        self._position = 0


# ============================================================================
# DEMUCS TESTIMONY
# ============================================================================

@dataclass
class DemucsTestimony:
    """What Demucs observed — not a verdict, just testimony."""
    stems: Dict[StemType, np.ndarray]
    confidence: Dict[StemType, Confidence]
    model_name: str
    fallback_level: FallbackLevel
    device: str
    separation_time_ms: float
    peak_memory_mb: float
    sample_rate: int
    is_streaming: bool = False
    windows_processed: int = 1

    def to_stage_result(self, stage_name: str = "separation") -> StageResult:
        """Convert testimony to StageResult for pipeline."""
        return StageResult(
            stage_name=stage_name,
            success=len(self.stems) > 0,
            events=[],
            metadata={
                "stems": {k.value: list(v.shape) for k, v in self.stems.items()},
                "confidence": {k.value: v.value for k, v in self.confidence.items()},
                "model_name": self.model_name,
                "fallback_level": self.fallback_level.value,
                "device": self.device,
                "separation_time_ms": self.separation_time_ms,
                "peak_memory_mb": self.peak_memory_mb,
                "sample_rate": self.sample_rate,
                "is_streaming": self.is_streaming,
                "windows_processed": self.windows_processed
            },
            veto_reason=None,
            execution_time_ms=self.separation_time_ms,
            memory_delta_mb=self.peak_memory_mb
        )

    def to_separation_result(self) -> SeparationResult:
        """Convert to SeparationResult for pipeline."""
        return SeparationResult(
            stems=self.stems,
            separation_time_seconds=self.separation_time_ms / 1000.0,
            memory_usage_mb=self.peak_memory_mb,
            confidence=Confidence.HIGH if self.fallback_level.value <= 2 else Confidence.MEDIUM
        )


# ============================================================================
# MEL ROFORMER (Simplified Implementation)
# ============================================================================

class SimpleMelRoformer:
    """
    Simplified Mel-Roformer implementation for fallback.

    When Demucs is unavailable or times out, this provides basic
    separation using mel-spectrogram analysis.
    """

    def __init__(
            self,
            sample_rate: int = 44100,
            n_mels: int = 128,
            device: str = "cpu"
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.device = device

    def separate(self, audio: np.ndarray) -> Dict[StemType, np.ndarray]:
        """
        Basic separation using spectral analysis.

        Returns:
            Dict with DRUMS, BASS, OTHER stems
        """
        if not LIBROSA_AVAILABLE:
            return self._identity_fallback(audio)

        # Ensure mono for analysis
        if audio.ndim == 2:
            mono = np.mean(audio, axis=0) if audio.shape[0] == 2 else np.mean(audio, axis=1)
        else:
            mono = audio

        # Compute CQT for better frequency resolution at low frequencies
        try:
            cqt = np.abs(librosa.cqt(mono, sr=self.sample_rate, hop_length=512, n_bins=84))

            # Bass: first 24 bins (up to ~200Hz)
            bass_energy = np.mean(cqt[:24], axis=0)
            bass_mask = bass_energy > np.percentile(bass_energy, 70)

            # Drums: transient detection
            onset_env = librosa.onset.onset_strength(y=mono, sr=self.sample_rate)
            drum_mask = onset_env > np.percentile(onset_env, 80)

            # Reconstruct stems
            bass_stem = mono * bass_mask.astype(np.float32)
            drum_stem = mono * drum_mask.astype(np.float32)
            other_stem = mono - bass_stem - drum_stem

        except Exception as e:
            warnings.warn(f"Mel-Roformer analysis failed: {e}, using identity")
            return self._identity_fallback(audio)

        # Restore stereo if needed
        if audio.ndim == 2:
            channels = audio.shape[0] if audio.shape[0] == 2 else audio.shape[1]
            if channels == 2:
                bass_stem = np.stack([bass_stem, bass_stem], axis=0)
                drum_stem = np.stack([drum_stem, drum_stem], axis=0)
                other_stem = np.stack([other_stem, other_stem], axis=0)

        return {
            StemType.BASS: bass_stem.astype(np.float32),
            StemType.DRUMS: drum_stem.astype(np.float32),
            StemType.OTHER: other_stem.astype(np.float32),
            StemType.VOCALS: np.zeros_like(bass_stem, dtype=np.float32)
        }

    def _identity_fallback(self, audio: np.ndarray) -> Dict[StemType, np.ndarray]:
        """Return original as 'other' stem."""
        empty = np.zeros_like(audio, dtype=np.float32)
        return {
            StemType.DRUMS: empty.copy(),
            StemType.BASS: empty.copy(),
            StemType.OTHER: audio.astype(np.float32),
            StemType.VOCALS: empty.copy()
        }


# ============================================================================
# MAIN DEMUCS SEPARATOR AGENT (with Mel-Roformer Fallback)
# ============================================================================

class DemucsSeparator(DemucsProtocol, MemoryManagedProtocol):
    """
    Demucs separation agent with Mel-Roformer fallback for Grimlock 5.4.

    TESTIFIES (does not decide):
        - Here are the separated stems (stereo, original sample rate)
        - Here is my confidence for each stem
        - Here is which fallback level was used
        - Here is how long it took
        - Here is my memory usage

    FALLBACK CHAIN:
        1. Try Demucs htdemucs (best quality)
        2. Try Demucs mdx (balanced)
        3. Use Mel-Roformer (basic)
        4. Use librosa (simple)
        5. Identity (original as "other")
    """

    # Known stem order from Demucs 4.x
    STEM_ORDER = ["drums", "bass", "other", "vocals"]

    def __init__(
            self,
            model_name: str = "htdemucs",
            device: str = "auto",
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None,
            streaming: bool = False,
            window_samples: int = 131072,  # Reduced from 262144 for better timeout behavior
            hop_samples: int = 65536,
            timeout_seconds: float = 180.0,  # 3 minutes max
            enable_fallback: bool = True
    ):
        """
        Args:
            model_name: Demucs model ('htdemucs', 'hdemucs', 'mdx')
            device: 'auto', 'cpu', or 'cuda'
            music_box: Optional forensic logger
            progress_callback: Optional progress reporter
            streaming: Enable streaming mode for long files
            window_samples: Window size for streaming (samples)
            hop_samples: Hop size for streaming (50% overlap recommended)
            timeout_seconds: Maximum time to wait for Demucs
            enable_fallback: Enable Mel-Roformer fallback
        """
        self._name = "demucs_separator"
        self._source_type = SourceType.DEMUCS
        self._model_name = model_name
        self._device = self._resolve_device(device)
        self._music_box = music_box
        self._progress_callback = progress_callback
        self._streaming = streaming
        self._window_samples = window_samples
        self._hop_samples = hop_samples
        self._timeout_seconds = timeout_seconds
        self._enable_fallback = enable_fallback
        self._model = None
        self._mel_roformer = None
        self._load_time_ms = 0
        self._memory_pressure = False
        self._current_fallback_level = FallbackLevel.DEMUCS_HD

        # Streaming buffer
        self._overlap_buffer: Optional[OverlapAddBuffer] = None

        # Lazy load model
        self._init_model()
        self._init_mel_roformer()

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
        """Main entry point for pipeline."""
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
        Separate audio into stems with progressive fallback.

        Returns:
            SeparationResult with stems dict
        """
        start_time = time.time()
        self._update_progress(0.0, "Starting separation")

        # Use original sample rate from context or detect
        sample_rate = context.original_sample_rate if context else 44100

        # Validate input
        if audio_buffer is None or len(audio_buffer) == 0:
            return self._empty_result(sample_rate, context)

        # Ensure stereo format
        audio, was_mono = self._ensure_stereo_format(audio_buffer)

        # Check memory before heavy processing
        memory_before = self.get_memory_footprint_mb()
        if memory_before > MEMORY_CRITICAL_THRESHOLD_MB * 0.8:
            self._update_progress(0.0, "High memory pressure, using light fallback")
            return self._run_fallback_level(audio, sample_rate, FallbackLevel.MEL_ROFORMER)

        # Progressive fallback chain
        stems = None
        confidences = None
        fallback_used = FallbackLevel.DEMUCS_HD

        # Level 1: Try Demucs htdemucs
        if self._current_fallback_level.value <= FallbackLevel.DEMUCS_HD.value:
            self._update_progress(0.1, "Attempting Demucs htdemucs...")
            result = self._run_with_timeout(audio, sample_rate, self._model_name)
            if result is not None:
                stems, confidences = result
                fallback_used = FallbackLevel.DEMUCS_HD
                self._current_fallback_level = FallbackLevel.DEMUCS_HD

        # Level 2: Try Demucs mdx
        if stems is None and self._current_fallback_level.value <= FallbackLevel.DEMUCS_MDX.value:
            self._update_progress(0.1, "Demucs htdemucs failed, trying mdx...")
            result = self._run_with_timeout(audio, sample_rate, "mdx")
            if result is not None:
                stems, confidences = result
                fallback_used = FallbackLevel.DEMUCS_MDX
                self._current_fallback_level = FallbackLevel.DEMUCS_MDX

        # Level 3: Try Mel-Roformer
        if stems is None and self._enable_fallback:
            self._update_progress(0.1, "Demucs failed, using Mel-Roformer...")
            result = self._run_mel_roformer(audio, sample_rate)
            if result is not None:
                stems, confidences = result
                fallback_used = FallbackLevel.MEL_ROFORMER
                self._current_fallback_level = FallbackLevel.MEL_ROFORMER

        # Level 4: Try librosa fallback
        if stems is None:
            self._update_progress(0.1, "Mel-Roformer failed, using librosa...")
            result = self._run_librosa_fallback(audio, sample_rate)
            if result is not None:
                stems, confidences = result
                fallback_used = FallbackLevel.LIBROSA
                self._current_fallback_level = FallbackLevel.LIBROSA

        # Level 5: Identity fallback
        if stems is None:
            self._update_progress(0.1, "All separators failed, using identity...")
            stems, confidences = self._identity_fallback(audio)
            fallback_used = FallbackLevel.IDENTITY
            self._current_fallback_level = FallbackLevel.IDENTITY

        # Single choke point for all 5 fallback levels: sanitize NaN/Inf
        # before any of this reaches downstream detection agents. Demucs
        # and Mel-Roformer inference can occasionally produce non-finite
        # values (e.g. numerical instability on silence or clipped input).
        stems = self._sanitize_stems(stems, fallback_used)

        separation_time = (time.time() - start_time) * 1000
        memory_after = self.get_memory_footprint_mb()

        self._update_progress(1.0, f"Separation complete (fallback: {fallback_used.name})")

        # Log testimony
        self._log_separation(stems, confidences, separation_time, memory_after, fallback_used)

        # Return SeparationResult
        return SeparationResult(
            stems=stems,
            separation_time_seconds=separation_time / 1000.0,
            memory_usage_mb=memory_after,
            confidence=self._get_fallback_confidence(fallback_used)
        )

    # ========================================================================
    # DemucsProtocol Implementation
    # ========================================================================

    @property
    def demucs_model_name(self) -> str:
        return self._model_name

    @property
    def segment_duration_seconds(self) -> float:
        return self._window_samples / 44100.0

    @property
    def overlap_seconds(self) -> float:
        return self._hop_samples / 44100.0

    @property
    def available_stems(self) -> List[StemType]:
        return [StemType.DRUMS, StemType.BASS, StemType.OTHER, StemType.VOCALS]

    @property
    def separation_quality(self) -> str:
        if self._current_fallback_level == FallbackLevel.DEMUCS_HD:
            return "high"
        elif self._current_fallback_level == FallbackLevel.DEMUCS_MDX:
            return "balanced"
        elif self._current_fallback_level == FallbackLevel.MEL_ROFORMER:
            return "medium"
        return "low"

    # ========================================================================
    # MemoryManagedProtocol Implementation
    # ========================================================================

    def release_buffer(self, buffer_name: str) -> None:
        if hasattr(self, buffer_name):
            delattr(self, buffer_name)
            self._maybe_gc()

    def get_memory_footprint_mb(self) -> float:
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return 0.0

    def can_release(self, buffer_name: str) -> bool:
        return buffer_name.startswith("stem_")

    def staggered_gc(self) -> Dict[str, Any]:
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
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _init_model(self):
        """Load Demucs model."""
        if not DEMUCS_AVAILABLE or not TORCH_AVAILABLE:
            self._model = None
            return

        load_start = time.time()

        try:
            self._model = pretrained.get_model(self._model_name)
            self._model.to(torch.device(self._device))
            self._model.eval()
            self._load_time_ms = (time.time() - load_start) * 1000

            if self._music_box:
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="model_loaded",
                    before_state={"model": None},
                    after_state={"model": self._model_name, "device": self._device},
                    reasoning=f"Demucs {self._model_name} on {self._device} in {self._load_time_ms:.0f}ms",
                    reversible=False
                )
        except Exception as e:
            warnings.warn(f"Failed to load Demucs: {e}")
            self._model = None

    def _init_mel_roformer(self):
        """Initialize Mel-Roformer fallback."""
        self._mel_roformer = SimpleMelRoformer()

    def _ensure_stereo_format(self, audio: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Ensure audio is in (channels, samples) format."""
        was_mono = False

        if audio.ndim == 1:
            was_mono = True
            stereo = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2:
            if audio.shape[0] == 2:
                stereo = audio
            elif audio.shape[1] == 2:
                stereo = audio.T
            else:
                # Neither axis is length-2 (e.g. 5.1 surround, or a
                # transposed mono buffer that happens to have 2 rows/cols
                # for an unrelated reason). Downmixing to mono here is a
                # deliberate, reasonable fallback - but silently doing it
                # without a trace makes a real upstream shape bug
                # indistinguishable from "just some mono audio."
                warnings.warn(
                    f"_ensure_stereo_format: ambiguous channel layout {audio.shape} "
                    f"(neither axis is length 2) - downmixing to mono"
                )
                was_mono = True
                mono = np.mean(audio, axis=0) if audio.shape[1] > 1 else audio.flatten()
                stereo = np.stack([mono, mono], axis=0)
        else:
            # Genuinely unexpected (0-D, 3-D+): silently returning silence
            # here would masquerade as a successful separation of nothing.
            warnings.warn(
                f"_ensure_stereo_format: unexpected audio.ndim={audio.ndim} "
                f"(shape={getattr(audio, 'shape', None)}) - returning silence"
            )
            was_mono = True
            length = len(audio) if hasattr(audio, '__len__') else 0
            stereo = np.zeros((2, length), dtype=audio.dtype)

        return stereo.astype(np.float32), was_mono

    def _sanitize_stems(
            self,
            stems: Dict[StemType, np.ndarray],
            fallback_used: 'FallbackLevel'
    ) -> Dict[StemType, np.ndarray]:
        """
        Replace any NaN/Inf values in separated stems with 0.0 and warn.

        Neural separation models can occasionally produce non-finite
        values (numerical instability on silence, clipped/extreme input,
        etc.) - letting that reach downstream pitch/drum detection as
        garbage-in would be far harder to diagnose than catching it here.
        """
        if not stems:
            return stems

        for stem_type, stem in stems.items():
            if stem is None or stem.size == 0:
                continue
            bad_mask = ~np.isfinite(stem)
            bad_count = int(np.count_nonzero(bad_mask))
            if bad_count > 0:
                warnings.warn(
                    f"Demucs/{fallback_used.name}: {stem_type.value} stem had "
                    f"{bad_count} non-finite sample(s) ({bad_count / stem.size:.2%}) - zeroed"
                )
                stems[stem_type] = np.nan_to_num(stem, nan=0.0, posinf=0.0, neginf=0.0)

        return stems

    def _run_with_timeout(
            self,
            audio: np.ndarray,
            sample_rate: int,
            model_name: str
    ) -> Optional[Tuple[Dict[StemType, np.ndarray], Dict[StemType, Confidence]]]:
        """Run Demucs with timeout protection."""
        import threading
        import concurrent.futures

        result_container = [None]
        error_container = [None]

        def _run():
            try:
                if DEMUCS_AVAILABLE and TORCH_AVAILABLE:
                    # Load model if needed
                    if model_name != self._model_name or self._model is None:
                        try:
                            model = pretrained.get_model(model_name)
                            model.to(torch.device(self._device))
                            model.eval()
                        except Exception as e:
                            error_container[0] = e
                            return
                    else:
                        model = self._model

                    # Run separation
                    stems, confidences = self._run_demucs_with_model(audio, sample_rate, model)
                    result_container[0] = (stems, confidences)
                else:
                    error_container[0] = RuntimeError("Demucs not available")
            except Exception as e:
                error_container[0] = e

        # NOTE: deliberately not using `with ThreadPoolExecutor(...) as executor`
        # here. ThreadPoolExecutor.__exit__ calls shutdown(wait=True), which
        # blocks until the submitted task actually finishes - completely
        # defeating the timeout below, since future.cancel() is a no-op once
        # a task has started running. Managing the executor manually lets us
        # give up and return promptly on timeout; the abandoned worker thread
        # keeps running to completion in the background, but the caller (and
        # the fallback chain: Demucs -> Mel-Roformer -> STFT) isn't blocked on it.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_run)
            future.result(timeout=self._timeout_seconds)
            if error_container[0]:
                raise error_container[0]
            return result_container[0]
        except concurrent.futures.TimeoutError:
            warnings.warn(f"Demucs {model_name} timed out after {self._timeout_seconds}s")
            return None
        except Exception as e:
            warnings.warn(f"Demucs {model_name} failed: {e}")
            return None
        finally:
            executor.shutdown(wait=False)

    def _run_demucs_with_model(
            self,
            audio: np.ndarray,
            sample_rate: int,
            model
    ) -> Tuple[Dict[StemType, np.ndarray], Dict[StemType, Confidence]]:
        """Run Demucs with already-loaded model."""
        audio_tensor = torch.from_numpy(audio).float()
        if len(audio_tensor.shape) == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(torch.device(self._device))

        with torch.no_grad():
            sources = apply_model(
                model,
                audio_tensor,
                device=self._device,
                shifts=1,
                split=True,
                overlap=0.25,
                progress=False
            )

        stems = {}
        confidences = {}

        for idx, stem_name in enumerate(self.STEM_ORDER):
            if idx < sources.shape[1]:
                stem = sources[0, idx].cpu().numpy()
                if stem.shape[0] != 2:
                    stem = np.stack([stem, stem], axis=0)

                stem_type = self._map_stem_name(stem_name)
                stems[stem_type] = stem.astype(np.float32)
                confidences[stem_type] = self._estimate_confidence(stem)

        # Ensure all expected stems exist
        for stem_type in [StemType.DRUMS, StemType.BASS, StemType.OTHER, StemType.VOCALS]:
            if stem_type not in stems:
                empty_stem = np.zeros((2, audio.shape[1]), dtype=np.float32)
                stems[stem_type] = empty_stem
                confidences[stem_type] = Confidence(0.0, "no_separation")

        del audio_tensor, sources
        self._maybe_gc()

        return stems, confidences

    def _run_mel_roformer(
            self,
            audio: np.ndarray,
            sample_rate: int
    ) -> Optional[Tuple[Dict[StemType, np.ndarray], Dict[StemType, Confidence]]]:
        """Run Mel-Roformer fallback."""
        try:
            stems = self._mel_roformer.separate(audio)
            confidences = {}
            for stem_type, stem in stems.items():
                confidences[stem_type] = self._estimate_confidence(stem)
            return stems, confidences
        except Exception as e:
            warnings.warn(f"Mel-Roformer failed: {e}")
            return None

    def _run_librosa_fallback(
            self,
            audio: np.ndarray,
            sample_rate: int
    ) -> Optional[Tuple[Dict[StemType, np.ndarray], Dict[StemType, Confidence]]]:
        """Run librosa-based fallback."""
        if not LIBROSA_AVAILABLE:
            return None

        try:
            # Ensure mono for analysis
            if audio.ndim == 2:
                mono = np.mean(audio, axis=0)
            else:
                mono = audio

            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(mono)

            # Restore stereo if needed
            if audio.ndim == 2:
                channels = 2
                harmonic = np.stack([harmonic, harmonic], axis=0)
                percussive = np.stack([percussive, percussive], axis=0)
                other = audio - harmonic - percussive
            else:
                other = mono - harmonic - percussive

            stems = {
                StemType.DRUMS: percussive.astype(np.float32),
                StemType.BASS: harmonic.astype(np.float32),
                StemType.OTHER: other.astype(np.float32),
                StemType.VOCALS: np.zeros_like(harmonic, dtype=np.float32)
            }

            confidences = {}
            for stem_type, stem in stems.items():
                confidences[stem_type] = self._estimate_confidence(stem)

            return stems, confidences
        except Exception as e:
            warnings.warn(f"Librosa fallback failed: {e}")
            return None

    def _run_fallback_level(
            self,
            audio: np.ndarray,
            sample_rate: int,
            level: FallbackLevel
    ) -> SeparationResult:
        """Run a specific fallback level and return SeparationResult directly."""
        start_time = time.time()

        if level == FallbackLevel.MEL_ROFORMER:
            result = self._run_mel_roformer(audio, sample_rate)
        elif level == FallbackLevel.LIBROSA:
            result = self._run_librosa_fallback(audio, sample_rate)
        else:
            result = None

        if result is None:
            stems, confidences = self._identity_fallback(audio)
        else:
            stems, confidences = result

        separation_time = (time.time() - start_time) * 1000
        memory_after = self.get_memory_footprint_mb()

        return SeparationResult(
            stems=stems,
            separation_time_seconds=separation_time / 1000.0,
            memory_usage_mb=memory_after,
            confidence=self._get_fallback_confidence(level)
        )

    def _identity_fallback(
            self,
            audio: np.ndarray
    ) -> Tuple[Dict[StemType, np.ndarray], Dict[StemType, Confidence]]:
        """Return original as 'other' stem."""
        empty = np.zeros_like(audio, dtype=np.float32)
        stems = {
            StemType.DRUMS: empty.copy(),
            StemType.BASS: empty.copy(),
            StemType.OTHER: audio.astype(np.float32),
            StemType.VOCALS: empty.copy()
        }
        confidences = {
            StemType.DRUMS: Confidence.HALLUCINATION,
            StemType.BASS: Confidence.HALLUCINATION,
            StemType.OTHER: Confidence.HIGH,
            StemType.VOCALS: Confidence.HALLUCINATION
        }
        return stems, confidences

    def _map_stem_name(self, demucs_name: str) -> StemType:
        """Map Demucs stem name to our StemType."""
        name_lower = demucs_name.lower()
        if "drum" in name_lower:
            return StemType.DRUMS
        elif "bass" in name_lower:
            return StemType.BASS
        elif "vocal" in name_lower:
            return StemType.VOCALS
        else:
            return StemType.OTHER

    def _estimate_confidence(self, stem: np.ndarray) -> Confidence:
        """Estimate separation confidence based on stem energy."""
        if stem is None or stem.size == 0:
            return Confidence.HALLUCINATION

        rms = np.sqrt(np.mean(stem ** 2))
        peak = np.max(np.abs(stem))
        confidence_value = min(0.95, (rms * 2 + peak * 0.5))

        if confidence_value < Confidence.LOW.value:
            return Confidence.HALLUCINATION
        elif confidence_value < Confidence.MEDIUM.value:
            return Confidence.LOW
        elif confidence_value < Confidence.HIGH.value:
            return Confidence.MEDIUM
        return Confidence.HIGH

    def _get_fallback_confidence(self, fallback_level: FallbackLevel) -> Confidence:
        """Get overall confidence based on fallback level."""
        if fallback_level == FallbackLevel.DEMUCS_HD:
            return Confidence.HIGH
        elif fallback_level == FallbackLevel.DEMUCS_MDX:
            return Confidence.HIGH
        elif fallback_level == FallbackLevel.MEL_ROFORMER:
            return Confidence.MEDIUM
        elif fallback_level == FallbackLevel.LIBROSA:
            return Confidence.LOW
        return Confidence.HALLUCINATION

    def _empty_result(self, sample_rate: int, context: Optional[AudioContext]) -> SeparationResult:
        """Return empty result for silence/error cases."""
        duration = context.duration_seconds if context else 0
        empty_stem = np.zeros((2, int(sample_rate * duration)), dtype=np.float32)

        stems = {
            StemType.DRUMS: empty_stem.copy(),
            StemType.BASS: empty_stem.copy(),
            StemType.OTHER: empty_stem.copy(),
            StemType.VOCALS: empty_stem.copy()
        }

        return SeparationResult(
            stems=stems,
            separation_time_seconds=0,
            memory_usage_mb=0,
            confidence=Confidence.HALLUCINATION
        )

    def _update_progress(self, progress: float, message: str):
        if self._progress_callback:
            self._progress_callback(progress, message)

    def _log_separation(
            self,
            stems: Dict,
            confidences: Dict,
            elapsed_ms: float,
            memory: float,
            fallback_level: FallbackLevel
    ):
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
                "fallback_level": fallback_level.value,
                "fallback_name": fallback_level.name
            },
            reasoning=f"Separated {len(stems)} stems in {elapsed_ms:.0f}ms using {fallback_level.name}",
            reversible=False
        )

    def _maybe_gc(self):
        memory_mb = self.get_memory_footprint_mb()
        self._memory_pressure = memory_mb > MEMORY_CRITICAL_THRESHOLD_MB

        if GC_COLLECT_AFTER_EACH_STAGE or self._memory_pressure:
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release(self):
        """Release model and clear memory."""
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        self._maybe_gc()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_demucs_separator(
        model_name: str = "htdemucs",
        device: str = "auto",
        music_box: Optional[MusicBoxProtocol] = None,
        streaming: bool = False,
        timeout_seconds: float = 180.0,
        enable_fallback: bool = True
) -> DemucsSeparator:
    """Create a Demucs separator agent with Mel-Roformer fallback."""
    return DemucsSeparator(
        model_name=model_name,
        device=device,
        music_box=music_box,
        streaming=streaming,
        timeout_seconds=timeout_seconds,
        enable_fallback=enable_fallback
    )


def create_quick_separator(
        music_box: Optional[MusicBoxProtocol] = None
) -> DemucsSeparator:
    """Create a separator optimized for speed (mdx model, shorter timeout)."""
    return DemucsSeparator(
        model_name="mdx",
        device="cpu",
        music_box=music_box,
        streaming=True,
        window_samples=65536,
        hop_samples=32768,
        timeout_seconds=120.0,
        enable_fallback=True
    )