#!/usr/bin/env python3
"""
agents/separation/demucs.py — Demucs Separation Agent for Grimlock 5.6.1
VERSION: 5.6.1

5.4 LAW:
    - Agents OBSERVE. They do not DECIDE.
    - Agents import from CORE only (order_types, constants, protocols)
    - Agents return TESTIMONY (observations + confidence), not verdicts.
    - Agents MUST declare their SourceType (for forensic logging).

RESPONSIBILITIES:
    - Separate stereo audio into stems using real Demucs (htdemucs)
    - Preserve original sample rate (no downsampling)
    - Return stems as stereo arrays (preserving channel count)
    - Report confidence per stem based on separation quality
    - Try, and tell the truth about whether it worked. Nothing else.

RESTRUCTURED (this version) — what changed and why:
    Previously this file owned its own internal 5-level fallback ladder
    (htdemucs -> mdx -> a class literally named "Mel-Roformer" -> librosa
    HPSS -> identity/all-zeros), nested INSIDE the separate fallback ladder
    orchestration/pipeline.py's _run_separation already ran on top of it
    (Demucs -> Mel-Roformer -> STFT). Two overlapping fallback systems,
    and the inner one had two real bugs: (1) the enable_fallback=False
    flag callers used to mean "only try real Demucs" only actually gated
    ONE of the four internal fallback levels (Mel-Roformer) - MDX and
    librosa fired regardless, so "disable fallback" never fully worked;
    (2) "Mel-Roformer" was not a Roformer or any neural model at all - it
    was a crude CQT-band-energy + onset-strength heuristic, misleadingly
    named such that it read as a legitimate second model when investigating
    a real separation bug this session.

    Verified directly this session that a genuine neural MelBand-Roformer/
    BS-Roformer checkpoint doing the same 4-stem (drums/bass/other/vocals)
    job as Demucs does not exist publicly - every Roformer checkpoint
    available (audio-separator's full catalog, which wraps the actual UVR
    community models) is a narrow 2-stem specialist (vocals/instrumental,
    karaoke, de-reverb, drum-only sub-separation). The only real 4-stem
    models anywhere in that ecosystem are Demucs variants themselves. So
    "a different real architecture as a second opinion" isn't an available
    option here - by explicit decision, the fallback tier below Demucs
    stays the honest heuristic (renamed HeuristicStemSplitter, its logic
    unchanged from the already-fixed version), not a fake neural model.

    This file's DemucsSeparator now does exactly one job: try real htdemucs,
    report honestly whether it worked. It returns None on any failure
    (timeout, exception, or memory pressure) instead of silently
    substituting a different method - the caller (pipeline.py's
    _run_separation) owns the ONE remaining fallback ladder: Demucs ->
    HeuristicStemSplitter -> STFT/HPSS -> identity.

CONFIRMED-DEAD CODE REMOVED IN THIS REWRITE (never instantiated anywhere,
verified by search before deleting):
    - OverlapAddBuffer (streaming buffer for a chunking scheme
      _run_demucs_with_model never actually used - the real inference call
      uses demucs's own apply_model(split=True), which handles long audio
      internally, independent of this class)
    - DemucsTestimony (a to_stage_result()/to_separation_result() dataclass
      whose to_stage_result the also-dead run() method called - .separate()
      is what every real caller actually uses, and it returns
      SeparationResult directly)
    - create_demucs_separator() / create_quick_separator() factory
      functions (never called anywhere)
    - streaming / window_samples / hop_samples constructor parameters
      (stored, exposed via segment_duration_seconds/overlap_seconds
      properties, but never read by the actual inference call - dead
      configuration that looked load-bearing but wasn't)

TIMEOUT MODEL:
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
import numpy as np
from typing import Dict, Optional, Callable, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum

# Core imports ONLY — no other agents
from core.order_types import (
    SourceType, StemType, Confidence, AudioContext, SeparationResult
)
from core.constants import (
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
# HEURISTIC STEM SPLITTER (honest name - not a neural model)
# ============================================================================
# Renamed from "SimpleMelRoformer". It never was a Roformer, or any neural
# network - it's a CQT low-band-energy mask (bass) plus an onset-strength
# mask (drums), applied directly in the sample domain. Real signal
# processing, genuinely useful as a last-resort split when real Demucs
# isn't available - but the old name made it read like a legitimate
# second model when investigating a real separation bug this session,
# which is exactly the kind of confusion an honest name avoids.

class HeuristicStemSplitter:
    """
    Cheap non-neural fallback: bass via CQT low-band energy, drums via
    onset-strength transients. Used by pipeline.py as the tier below real
    Demucs - not wired into DemucsSeparator itself (see module docstring).
    """

    def __init__(
            self,
            sample_rate: int = 44100,
            device: str = "cpu"
    ):
        self.sample_rate = sample_rate
        self.device = device

    def separate(self, audio: np.ndarray) -> Dict[StemType, np.ndarray]:
        """
        Basic separation using spectral analysis.

        Returns:
            Dict with DRUMS, BASS, OTHER, VOCALS stems
        """
        if not LIBROSA_AVAILABLE:
            return self._identity_fallback(audio)

        # Ensure mono for analysis
        if audio.ndim == 2:
            mono = np.mean(audio, axis=0) if audio.shape[0] == 2 else np.mean(audio, axis=1)
        else:
            mono = audio

        try:
            hop_length = 512
            cqt = np.abs(librosa.cqt(mono, sr=self.sample_rate, hop_length=hop_length, n_bins=84))

            # Bass: first 24 bins (up to ~200Hz)
            bass_energy = np.mean(cqt[:24], axis=0)
            bass_mask_frames = bass_energy > np.percentile(bass_energy, 70)

            # Drums: transient detection
            onset_env = librosa.onset.onset_strength(y=mono, sr=self.sample_rate, hop_length=hop_length)
            drum_mask_frames = onset_env > np.percentile(onset_env, 80)

            # CQT/onset_strength both operate on hop_length-sized frames
            # (len(mono)/hop_length of them), not one value per raw sample -
            # multiplying either mask directly against mono (sample-domain)
            # is a shape mismatch that fires on every real invocation, not
            # an edge case (frame count is always ~hop_length times smaller
            # than sample count). Upsample each frame's mask value across
            # its hop_length samples, then pad/trim to exactly len(mono)
            # since frame_count * hop_length rarely lands on it exactly.
            def _frames_to_samples(mask_frames: np.ndarray, n_samples: int) -> np.ndarray:
                upsampled = np.repeat(mask_frames, hop_length)
                if len(upsampled) < n_samples:
                    upsampled = np.pad(upsampled, (0, n_samples - len(upsampled)), mode='edge')
                else:
                    upsampled = upsampled[:n_samples]
                return upsampled

            bass_mask = _frames_to_samples(bass_mask_frames, len(mono))
            drum_mask = _frames_to_samples(drum_mask_frames, len(mono))

            # Reconstruct stems
            bass_stem = mono * bass_mask.astype(np.float32)
            drum_stem = mono * drum_mask.astype(np.float32)
            other_stem = mono - bass_stem - drum_stem

        except Exception as e:
            warnings.warn(f"HeuristicStemSplitter analysis failed: {e}, using identity")
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
        """Return original as 'other' stem - correctly labeled HALLUCINATION
        confidence by whoever wraps this (see pipeline.py's _run_separation),
        never silently passed off as a real 3-way split."""
        empty = np.zeros_like(audio, dtype=np.float32)
        return {
            StemType.DRUMS: empty.copy(),
            StemType.BASS: empty.copy(),
            StemType.OTHER: audio.astype(np.float32),
            StemType.VOCALS: empty.copy()
        }


# ============================================================================
# MAIN DEMUCS SEPARATOR AGENT
# ============================================================================

class DemucsSeparator(DemucsProtocol, MemoryManagedProtocol):
    """
    Demucs separation agent for Grimlock 5.6.1.

    TESTIFIES (does not decide):
        - Here are the separated stems (stereo, original sample rate)
        - Here is my confidence for each stem
        - Here is how long it took
        - Here is my memory usage

    Does exactly one thing: try real htdemucs, report honestly whether it
    worked. Returns None on any failure - timeout, exception, or memory
    pressure - rather than silently substituting a different method. The
    caller (orchestration/pipeline.py's _run_separation) owns the actual
    fallback ladder (Demucs -> HeuristicStemSplitter -> STFT/HPSS ->
    identity), so a failure here is a clean, honest signal to move to the
    next tier, not something masked by an internal ladder of its own.
    """

    # Known stem order from Demucs 4.x
    STEM_ORDER = ["drums", "bass", "other", "vocals"]

    def __init__(
            self,
            model_name: str = "htdemucs",
            device: str = "auto",
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None,
            timeout_seconds: float = 180.0,
    ):
        """
        Args:
            model_name: Demucs model ('htdemucs', 'hdemucs_mmi', 'mdx', ...)
            device: 'auto', 'cpu', or 'cuda'
            music_box: Optional forensic logger
            progress_callback: Optional progress reporter
            timeout_seconds: Maximum time to wait for Demucs
        """
        self._name = "demucs_separator"
        self._source_type = SourceType.DEMUCS
        self._model_name = model_name
        self._device = self._resolve_device(device)
        self._music_box = music_box
        self._progress_callback = progress_callback
        self._timeout_seconds = timeout_seconds
        self._model = None
        self._load_time_ms = 0
        self._succeeded = False

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

    # ========================================================================
    # SeparationAgentProtocol Implementation
    # ========================================================================

    def separate(
            self,
            audio_buffer: np.ndarray,
            context: AudioContext
    ) -> Optional[SeparationResult]:
        """
        Try real Demucs separation.

        Returns:
            SeparationResult if htdemucs actually ran successfully, else
            None - callers should move to their own next fallback tier
            rather than receive a substitute method dressed up as success.
        """
        start_time = time.time()
        self._update_progress(0.0, "Starting separation")
        self._succeeded = False

        sample_rate = context.original_sample_rate if context else 44100

        if audio_buffer is None or len(audio_buffer) == 0:
            return None

        audio, was_mono = self._ensure_stereo_format(audio_buffer)

        # Check memory before heavy processing. This is Law 1 (Resource
        # Survival) working as intended - if the process is already near
        # the critical threshold, don't even attempt a large neural model
        # load/inference. Returns None honestly instead of silently
        # substituting a different separation method under the same name.
        memory_before = self.get_memory_footprint_mb()
        if memory_before > MEMORY_CRITICAL_THRESHOLD_MB * 0.8:
            self._update_progress(0.0, "High memory pressure - declining to attempt Demucs")
            warnings.warn(
                f"DemucsSeparator: memory pressure ({memory_before:.0f}MB > "
                f"{MEMORY_CRITICAL_THRESHOLD_MB * 0.8:.0f}MB threshold) - "
                f"declining to run, caller should use next fallback tier"
            )
            return None

        result = self._run_with_timeout(audio, sample_rate, self._model_name)
        if result is None:
            return None

        stems, confidences = result
        stems = self._sanitize_stems(stems)

        separation_time = (time.time() - start_time) * 1000
        memory_after = self.get_memory_footprint_mb()
        self._succeeded = True

        self._update_progress(1.0, "Separation complete (htdemucs)")
        self._log_separation(stems, confidences, separation_time, memory_after)

        return SeparationResult(
            stems=stems,
            separation_time_seconds=separation_time / 1000.0,
            memory_usage_mb=memory_after,
            confidence=Confidence.HIGH,
            separator_used="DEMUCS_HD" if self._model_name == "htdemucs" else self._model_name.upper()
        )

    # ========================================================================
    # DemucsProtocol Implementation
    # ========================================================================

    @property
    def demucs_model_name(self) -> str:
        return self._model_name

    @property
    def segment_duration_seconds(self) -> float:
        # Real segment length as actually used by demucs's own apply_model
        # chunking (split=True), not a separately-configured value that
        # doesn't drive anything - see module docstring on the removed
        # streaming/window_samples/hop_samples parameters.
        return float(getattr(self._model, "segment", 0.0) or 0.0)

    @property
    def overlap_seconds(self) -> float:
        # Matches the fixed overlap=0.25 ratio passed to apply_model in
        # _run_demucs_with_model below.
        return self.segment_duration_seconds * 0.25

    @property
    def available_stems(self) -> List[StemType]:
        return [StemType.DRUMS, StemType.BASS, StemType.OTHER, StemType.VOCALS]

    @property
    def separation_quality(self) -> str:
        return "high" if self._succeeded else "unavailable"

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
            stems: Dict[StemType, np.ndarray]
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
                    f"Demucs: {stem_type.value} stem had "
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
        # keeps running to completion in the background, but the caller
        # isn't blocked on it.
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
                # Confidence is a plain single-value Enum - calling it
                # with a second positional arg ("no_separation") doesn't
                # set some description field, it hits EnumMeta's
                # functional-API call signature (value, names, ...) and
                # would either raise or silently do something bizarre,
                # not attach a label. This path is only reachable when a
                # source model produces fewer than 4 stems (never hit by
                # the real htdemucs 4-stem path), but was never actually
                # exercised/caught until now.
                confidences[stem_type] = Confidence.HALLUCINATION

        del audio_tensor, sources
        self._maybe_gc()

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

    def _raw_confidence_score(self, stem: np.ndarray) -> float:
        """
        Continuous energy-based confidence score, before bucketing into
        a discrete Confidence tier. Split out from _estimate_confidence
        so _log_separation can report the real per-stem number - logging
        only the bucketed tier's fixed representative constant (e.g.
        every stem that lands anywhere in [0.25, 0.50) reports as an
        identical "0.25") silently collapsed genuinely different stems
        into what looked like one flat, suspicious constant.
        """
        if stem is None or stem.size == 0:
            return 0.0
        rms = np.sqrt(np.mean(stem ** 2))
        peak = np.max(np.abs(stem))
        return float(min(0.95, (rms * 2 + peak * 0.5)))

    def _estimate_confidence(self, stem: np.ndarray) -> Confidence:
        """Estimate separation confidence based on stem energy."""
        if stem is None or stem.size == 0:
            return Confidence.HALLUCINATION

        confidence_value = self._raw_confidence_score(stem)

        if confidence_value < Confidence.LOW.value:
            return Confidence.HALLUCINATION
        elif confidence_value < Confidence.MEDIUM.value:
            return Confidence.LOW
        elif confidence_value < Confidence.HIGH.value:
            return Confidence.MEDIUM
        return Confidence.HIGH

    def _update_progress(self, progress: float, message: str):
        if self._progress_callback:
            self._progress_callback(progress, message)

    def _log_separation(
            self,
            stems: Dict,
            confidences: Dict,
            elapsed_ms: float,
            memory: float,
    ):
        if not self._music_box:
            return

        stem_shapes = {k.value: list(v.shape) for k, v in stems.items()}
        confidence_tiers = {k.value: v.name for k, v in confidences.items()}
        confidence_scores = {k.value: self._raw_confidence_score(v) for k, v in stems.items()}

        self._music_box.log_decision(
            stage_name=self._name,
            decision_type="separation_testimony",
            before_state={"audio_loaded": True},
            after_state={
                "stems": list(stems.keys()),
                "stem_shapes": stem_shapes,
                "confidence_tier": confidence_tiers,
                "confidence_score": confidence_scores,
                "model": self._model_name,
            },
            reasoning=f"Separated {len(stems)} stems in {elapsed_ms:.0f}ms using real {self._model_name}",
            reversible=False
        )

    def _maybe_gc(self):
        memory_mb = self.get_memory_footprint_mb()
        if GC_COLLECT_AFTER_EACH_STAGE or memory_mb > MEMORY_CRITICAL_THRESHOLD_MB:
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
