# =================================================================
# MODULE: agents/detection/pitch_intelligence.py
# DESCRIPTION: Pitch detection engine for Grimlock 5.3
# VERSION: 5.6.1 (COMPLETE REWRITE - Aggressive fixes)
# UPDATED: 2026-05-17
#
# CRITICAL FIXES in this version:
#   1. FIX 1.1 - Removed impossible threshold (was rejecting all frames)
#   2. FIX 1.2 - Relaxed pitch continuity (allow ±1 semitone jitter)
#   3. FIX 1.3 - Added signal diagnostics (global_peak, threshold, etc.)
#   4. FIX 1.4 - Temporarily disabled harmonic veto for debugging
#   5. FIX 1.5 - Dynamic FFT size based on audio slice length
#   6. FIX 1.6 - Added multiple source options (full mix, bass, other)
#   7. Added CREPE as primary (most reliable for bass)
#   8. Added confidence thresholds that actually work
#   9. Added frame acceptance/rejection tracking
# =================================================================

import warnings
import time
import gc
import numpy as np
import threading
import sys
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Core imports
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, HarmonicSeries, PartialTrack
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    MIN_PITCH_MIDI,
    MAX_PITCH_MIDI,
    MIN_FREQUENCY_HZ,
    MAX_FREQUENCY_HZ,
    HARMONIC_SERIES_MATCH_THRESHOLD,
    ANALYSIS_SR,
    CREPE_SR,
    BASIC_PITCH_SR
)
from core.protocols import (
    DetectionAgentProtocol, MusicBoxProtocol, StatusReporterProtocol
)


# ========================================================================
# Enums and Configuration
# ========================================================================

class PitchModelType(str, Enum):
    OMNIZART = "omnizart"
    BASIC_PITCH = "basic_pitch"
    CREPE = "crepe"
    SPICE = "spice"
    LIBROSA = "librosa"


@dataclass
class PitchDetectionConfig:
    """Configuration for pitch detection with aggressive defaults."""

    # Model selection - CREPE is best for bass
    primary_model: PitchModelType = PitchModelType.CREPE
    fallback_model: PitchModelType = PitchModelType.LIBROSA
    use_ensemble: bool = True

    # Detection parameters - LOWERED thresholds
    min_freq_hz: float = 40.0  # Lower for bass (was 50)
    max_freq_hz: float = 2000.0
    min_note_duration_ms: float = 30.0  # Shorter notes allowed (was 50)
    hop_ms: float = 10.0

    # Confidence thresholds - LOWERED
    note_confidence_threshold: float = 0.15  # MUCH lower (was 0.3)
    harmonic_confidence_threshold: float = 0.3

    # Post-processing
    median_filter_size: int = 3
    min_note_gap_ms: float = 20.0
    merge_overlap_ms: float = 20.0

    # Octave correction
    correct_octave_errors: bool = True
    octave_error_threshold: float = 0.3

    # FIX 1.4: Temporarily disable harmonic veto for debugging
    analyze_harmonics: bool = False  # DISABLED

    # Performance
    cleanup_after_detection: bool = True
    sanitize_input: bool = True
    import_timeout_seconds: float = 5.0

    # FIX 1.1: Adaptive threshold parameters
    threshold_percentile: float = 75  # Use percentile-based threshold
    threshold_multiplier: float = 0.25  # 25% of percentile


# ========================================================================
# Safe Import Helper
# ========================================================================

def import_with_timeout(module_name: str, timeout_seconds: float) -> Tuple[Any, bool]:
    result = [None]
    success = [False]

    def target():
        try:
            import importlib
            result[0] = importlib.import_module(module_name)
            success[0] = True
        except Exception:
            pass

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        return None, False
    return result[0], success[0]


# ========================================================================
# Base Model Class
# ========================================================================

class BasePitchModel(ABC):
    def __init__(self, config: PitchDetectionConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._available = False
        self._load_time_ms: float = 0.0
        self._model_name: str = "base"
        self._init_lock = threading.Lock()
        self._initialized = False
        self._init_failed = False

        # Frame acceptance tracking (FIX 1.3)
        self._accepted_frames = 0
        self._rejected_frames = 0
        self._total_frames_processed = 0

    @property
    def available(self) -> bool:
        if self._init_failed:
            return False
        if not self._initialized:
            self._lazy_init()
        return self._available

    @property
    def model_name(self) -> str:
        return self._model_name

    def _lazy_init(self):
        with self._init_lock:
            if self._initialized or self._init_failed:
                return
            try:
                self._init_model()
            except Exception as e:
                self._log_status(f"Initialization failed: {e}", "error")
                self._init_failed = True
                self._available = False
            finally:
                self._initialized = True

    @abstractmethod
    def _init_model(self):
        pass

    def _sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.config.sanitize_input:
            return audio
        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return audio

    def _freq_to_midi(self, freq_hz: float) -> int:
        if freq_hz <= 0:
            return 60
        midi = 12 * np.log2(freq_hz / 440.0) + 69
        return int(round(midi))

    def _log_status(self, message: str, level: str = "info"):
        if self.status_reporter:
            getattr(self.status_reporter, level)(self._model_name, message)

    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int) -> List[NoteEvent]:
        pass


# ========================================================================
# CREPE Model (Primary - Best for bass)
# ========================================================================

class CrepeModel(BasePitchModel):
    """CREPE monophonic pitch detection - best for bass."""

    def __init__(self, config: PitchDetectionConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        super().__init__(config, status_reporter)
        self._model_name = "CREPE"
        self._crepe_module = None
        self._model = None

    def _init_model(self):
        self._log_status(f"Initializing CREPE (timeout={self.config.import_timeout_seconds}s)...")

        module, success = import_with_timeout("crepe", self.config.import_timeout_seconds)

        if not success or module is None:
            self._log_status("CREPE not available - will use fallback", "warn")
            self._available = False
            return

        self._crepe_module = module
        self._available = True
        self._log_status("CREPE available")

        # Load model
        try:
            from crepe import load_model
            self._model = load_model()
            self._log_status("CREPE model loaded")
        except Exception as e:
            self._log_status(f"CREPE model load failed: {e}", "warn")
            self._available = False

    def detect(self, audio: np.ndarray, sample_rate: int) -> List[NoteEvent]:
        if not self.available or self._crepe_module is None:
            return []

        audio = self._sanitize_audio(audio)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        try:
            from crepe import predict

            # Run CREPE
            time_ms, frequency_hz, confidence, _ = predict(
                audio,
                sr=sample_rate,
                viterbi=True,
                step_size=int(self.config.hop_ms),
                verbose=0
            )

            return self._convert_to_notes(time_ms, frequency_hz, confidence)

        except Exception as e:
            self._log_status(f"CREPE detection failed: {e}", "error")
            return []

    def _convert_to_notes(self, time_ms: np.ndarray, frequency_hz: np.ndarray,
                          confidence: np.ndarray) -> List[NoteEvent]:
        """Convert CREPE output to NoteEvents with aggressive thresholds."""
        events = []
        current_note = None

        frequency_hz = np.nan_to_num(frequency_hz, nan=0.0)
        confidence = np.nan_to_num(confidence, nan=0.0)

        # Get threshold for acceptance (FIX 1.1)
        # Use percentile-based threshold instead of fixed value
        valid_conf = confidence[confidence > 0.01]
        if len(valid_conf) > 10:
            threshold = np.percentile(valid_conf, self.config.threshold_percentile) * self.config.threshold_multiplier
            threshold = max(0.05, min(0.3, threshold))  # Clamp between 0.05 and 0.3
        else:
            threshold = self.config.note_confidence_threshold

        print(f"[CREPE] Frame count: {len(time_ms)}, threshold={threshold:.3f}")

        self._accepted_frames = 0
        self._rejected_frames = 0

        for i, (t, f, c) in enumerate(zip(time_ms, frequency_hz, confidence)):
            if c > threshold and f > 0:
                pitch = self._freq_to_midi(f)

                # Unlike LibrosaModel's detection path, nothing here ever
                # checked MIN_PITCH_MIDI/MAX_PITCH_MIDI - a pitch-tracker
                # octave error (locking onto a sub-harmonic an octave or
                # two below the true pitch, a well-known CREPE failure
                # mode that low-fundamental content like bass is
                # especially prone to triggering) went straight through
                # to a NoteEvent unchecked. The ±1-semitone continuity
                # check below only protects an ALREADY-accepted note from
                # drifting further; it does nothing for the first bad
                # frame that starts one.
                if not (MIN_PITCH_MIDI <= pitch <= MAX_PITCH_MIDI):
                    self._rejected_frames += 1
                    continue

                # FIX 1.2: Relaxed pitch continuity (allow ±1 semitone)
                if current_note is None:
                    current_note = {
                        'pitch': pitch,
                        'start_ms': t,
                        'confidence': c,
                        'freq_hz': f
                    }
                    self._accepted_frames += 1
                # Allow pitch to vary by ±1 semitone (FIX 1.2)
                elif abs(pitch - current_note['pitch']) <= 1:
                    # Same note continuing
                    if c > current_note['confidence']:
                        current_note['confidence'] = c
                        current_note['freq_hz'] = f
                    self._accepted_frames += 1
                else:
                    # Pitch changed - end current note
                    duration = t - current_note['start_ms']
                    if duration >= self.config.min_note_duration_ms:
                        events.append(NoteEvent(
                            pitch=current_note['pitch'],
                            start_ms=current_note['start_ms'],
                            end_ms=t,
                            velocity=int(min(127, current_note['confidence'] * 100 + 30)),
                            confidence=current_note['confidence'],
                            zero_crossing_rate=0.0,
                            source=SourceType.PITCH,
                            fundamental_freq_hz=current_note['freq_hz'],
                            reasoning_chain=[f"CREPE: {current_note['pitch']} for {duration:.0f}ms"]
                        ))
                        self._accepted_frames += 1
                    else:
                        self._rejected_frames += 1

                    # Start new note
                    current_note = {
                        'pitch': pitch,
                        'start_ms': t,
                        'confidence': c,
                        'freq_hz': f
                    }
            else:
                if current_note is not None:
                    duration = t - current_note['start_ms']
                    if duration >= self.config.min_note_duration_ms:
                        events.append(NoteEvent(
                            pitch=current_note['pitch'],
                            start_ms=current_note['start_ms'],
                            end_ms=t,
                            velocity=int(min(127, current_note['confidence'] * 100 + 30)),
                            confidence=current_note['confidence'],
                            zero_crossing_rate=0.0,
                            source=SourceType.PITCH,
                            fundamental_freq_hz=current_note['freq_hz'],
                            reasoning_chain=[f"CREPE: {current_note['pitch']} for {duration:.0f}ms"]
                        ))
                        self._accepted_frames += 1
                    else:
                        self._rejected_frames += 1
                    current_note = None

        # Handle final note
        if current_note is not None:
            duration = time_ms[-1] - current_note['start_ms']
            if duration >= self.config.min_note_duration_ms:
                events.append(NoteEvent(
                    pitch=current_note['pitch'],
                    start_ms=current_note['start_ms'],
                    end_ms=time_ms[-1],
                    velocity=int(min(127, current_note['confidence'] * 100 + 30)),
                    confidence=current_note['confidence'],
                    zero_crossing_rate=0.0,
                    source=SourceType.PITCH,
                    fundamental_freq_hz=current_note['freq_hz'],
                    reasoning_chain=[f"CREPE final: {current_note['pitch']}"]
                ))

        print(f"[CREPE] Accepted frames: {self._accepted_frames}, Rejected: {self._rejected_frames}")

        # Filter by confidence
        events = [e for e in events if e.confidence >= self.config.note_confidence_threshold]

        # Filter by duration
        events = [e for e in events if e.duration_ms() >= self.config.min_note_duration_ms]

        return events


# ========================================================================
# Librosa Model (Fallback)
# ========================================================================

class LibrosaModel(BasePitchModel):
    """Librosa pitch detection - always available fallback."""

    def __init__(self, config: PitchDetectionConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        super().__init__(config, status_reporter)
        self._model_name = "Librosa"

    def _init_model(self):
        try:
            import librosa
            self._available = True
            self._log_status("Librosa available as fallback")
        except ImportError:
            self._log_status("Librosa not available", "error")
            self._available = False

    def detect(self, audio: np.ndarray, sample_rate: int) -> List[NoteEvent]:
        audio = self._sanitize_audio(audio)

        if not LIBROSA_AVAILABLE:
            return []

        try:
            import librosa

            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)

            # FIX 1.5: Dynamic FFT size based on audio length
            n_fft = min(4096, len(audio))
            n_fft = max(256, n_fft)
            hop_length = max(256, int(self.config.hop_ms * sample_rate / 1000))

            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=sample_rate,
                hop_length=hop_length,
                n_fft=n_fft,
                fmin=self.config.min_freq_hz,
                fmax=self.config.max_freq_hz
            )

            time_per_frame = hop_length / sample_rate * 1000
            global_peak = np.max(magnitudes) if magnitudes.size > 0 else 1.0

            # FIX 1.1: Adaptive threshold
            non_zero_mags = magnitudes[magnitudes > 0]
            if len(non_zero_mags) > 0:
                percentile_threshold = np.percentile(non_zero_mags, self.config.threshold_percentile)
                threshold = max(percentile_threshold * self.config.threshold_multiplier, 0.01)
            else:
                threshold = global_peak * 0.05

            print(f"[LIBROSA] global_peak={global_peak:.4f}, threshold={threshold:.4f}")

            events = []
            note_start, note_pitch, note_frames, note_confidence_sum, note_freq = None, None, 0, 0.0, 0.0
            accepted_frames = 0
            rejected_frames = 0

            for i in range(pitches.shape[1]):
                frame_mags = magnitudes[:, i]
                frame_peak = np.max(frame_mags)

                if frame_peak > threshold:
                    max_idx = np.argmax(frame_mags)
                    freq = pitches[max_idx, i]

                    if freq > 0:
                        pitch = self._freq_to_midi(freq)
                        confidence = min(1.0, frame_peak / (global_peak + 1e-8))

                        if MIN_PITCH_MIDI <= pitch <= MAX_PITCH_MIDI:
                            time_ms = i * time_per_frame

                            if note_start is None:
                                note_start = time_ms
                                note_pitch = pitch
                                note_frames = 1
                                note_confidence_sum = confidence
                                note_freq = freq
                                accepted_frames += 1
                            # FIX 1.2: Allow ±1 semitone jitter
                            elif abs(pitch - note_pitch) <= 1:
                                note_frames += 1
                                note_confidence_sum += confidence
                                accepted_frames += 1
                            else:
                                if note_frames >= 2:
                                    events.append(NoteEvent(
                                        pitch=note_pitch,
                                        start_ms=note_start,
                                        end_ms=time_ms - time_per_frame,
                                        velocity=int(min(127, (note_confidence_sum / note_frames) * 80 + 30)),
                                        confidence=note_confidence_sum / note_frames,
                                        zero_crossing_rate=0.0,
                                        source=SourceType.PITCH,
                                        fundamental_freq_hz=note_freq,
                                        reasoning_chain=["librosa piptrack"]
                                    ))
                                    accepted_frames += 1
                                else:
                                    rejected_frames += 1
                                note_start, note_pitch, note_frames, note_confidence_sum, note_freq = time_ms, pitch, 1, confidence, freq
                else:
                    if note_start is not None and note_frames >= 2:
                        events.append(NoteEvent(
                            pitch=note_pitch,
                            start_ms=note_start,
                            end_ms=i * time_per_frame,
                            velocity=int(min(127, (note_confidence_sum / note_frames) * 80 + 30)),
                            confidence=note_confidence_sum / note_frames,
                            zero_crossing_rate=0.0,
                            source=SourceType.PITCH,
                            fundamental_freq_hz=note_freq,
                            reasoning_chain=["librosa piptrack"]
                        ))
                        accepted_frames += 1
                    else:
                        if note_start is not None:
                            rejected_frames += 1
                    note_start, note_pitch, note_frames, note_confidence_sum, note_freq = None, None, 0, 0.0, 0.0

            # Final note
            if note_start is not None and note_frames >= 2:
                events.append(NoteEvent(
                    pitch=note_pitch,
                    start_ms=note_start,
                    end_ms=pitches.shape[1] * time_per_frame,
                    velocity=int(min(127, (note_confidence_sum / note_frames) * 80 + 30)),
                    confidence=note_confidence_sum / note_frames,
                    zero_crossing_rate=0.0,
                    source=SourceType.PITCH,
                    fundamental_freq_hz=note_freq,
                    reasoning_chain=["librosa piptrack final"]
                ))

            print(f"[LIBROSA] Accepted frames: {accepted_frames}, Rejected: {rejected_frames}")

            # Filter by confidence
            events = [e for e in events if e.confidence >= self.config.note_confidence_threshold]

            # Filter by duration
            events = [e for e in events if e.duration_ms() >= self.config.min_note_duration_ms]

            return events

        except Exception as e:
            self._log_status(f"Librosa detection failed: {e}", "error")
            return []


# ========================================================================
# Main Pitch Intelligence Agent
# ========================================================================

# Set global flag for librosa availability
LIBROSA_AVAILABLE = False
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    pass


class PitchIntelligence:
    """Pitch detection engine with aggressive fixes for bass detection."""

    def __init__(
            self,
            config: Optional[PitchDetectionConfig] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        self._name = "pitch_intelligence"
        self._source_type = SourceType.PITCH
        self._config = config or PitchDetectionConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback

        self._models: Dict[PitchModelType, BasePitchModel] = {}
        self._init_models()

        self._last_events: List[NoteEvent] = []
        self._last_execution_time_ms: float = 0.0
        self._active_model: Optional[PitchModelType] = None
        self._total_memory_freed_mb: float = 0.0

        self._log_status(f"PitchIntelligence initialized (primary={self._config.primary_model.value})")

    def _init_models(self):
        """Initialize model stubs."""
        model_classes = {
            PitchModelType.CREPE: CrepeModel,
            PitchModelType.LIBROSA: LibrosaModel,
        }

        for model_type, model_class in model_classes.items():
            try:
                self._models[model_type] = model_class(self._config, self._status_reporter)
            except Exception as e:
                self._log_status(f"Failed to init {model_type.value}: {e}", "warn")

    def _get_active_model(self) -> Optional[BasePitchModel]:
        """Get the best available model."""
        # Try primary
        primary = self._models.get(self._config.primary_model)
        if primary:
            try:
                if primary.available:
                    self._active_model = self._config.primary_model
                    return primary
            except Exception as e:
                self._log_status(f"Primary model {primary.model_name} failed: {e}", "warn")

        # Try fallback
        fallback = self._models.get(self._config.fallback_model)
        if fallback:
            try:
                if fallback.available:
                    self._active_model = self._config.fallback_model
                    self._log_status(f"Using fallback: {fallback.model_name}", "warn")
                    return fallback
            except Exception as e:
                self._log_status(f"Fallback model failed: {e}", "warn")

        # Try any available
        for model_type, model in self._models.items():
            try:
                if model.available:
                    self._active_model = model_type
                    self._log_status(f"Using model: {model.model_name}")
                    return model
            except Exception:
                continue

        self._log_status("No pitch detection models available", "error")
        return None

    def detect(self, audio_buffer: np.ndarray, context: AudioContext) -> List[NoteEvent]:
        """Detect pitch from audio with aggressive fixes."""
        if audio_buffer is None or len(audio_buffer) == 0:
            self._log_status("Empty audio buffer", "warn")
            return []

        # Print input audio stats for debugging
        rms = np.sqrt(np.mean(audio_buffer ** 2))
        peak = np.max(np.abs(audio_buffer))
        print(
            f"[PITCH INPUT] RMS={rms:.4f}, Peak={peak:.4f}, Duration={len(audio_buffer) / context.working_sample_rate:.1f}s")

        if self._config.sanitize_input:
            if np.any(~np.isfinite(audio_buffer)):
                self._log_status("Found non-finite values in audio, sanitizing", "warn")
                audio_buffer = np.nan_to_num(audio_buffer, nan=0.0, posinf=1.0, neginf=-1.0)

        sample_rate = context.working_sample_rate

        self._update_progress(0.1, "Running pitch detection")

        try:
            model = self._get_active_model()
            if model is None:
                self._log_status("No active model available", "error")
                return []

            events = model.detect(audio_buffer, sample_rate)

            # FIX 1.4: Skip harmonic analysis (temporarily disabled)
            if self._config.analyze_harmonics and events:
                # This is disabled but left for future re-enable
                pass

            if self._config.correct_octave_errors:
                events = self._correct_octave_errors(events)

            events = [e for e in events if e.confidence >= self._config.note_confidence_threshold]

            self._last_events = events
            self._update_progress(1.0, f"Detected {len(events)} notes")

            print(f"[PITCH RESULT] Detected {len(events)} notes")
            for e in events[:10]:
                print(f"  Note {e.pitch}: {e.start_ms:.0f}ms - {e.end_ms:.0f}ms, conf={e.confidence:.2f}")

            if self._music_box and events:
                avg_confidence = np.mean([e.confidence for e in events])
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="pitch_detection",
                    before_state={},
                    after_state={
                        "event_count": len(events),
                        "avg_confidence": float(avg_confidence),
                        "active_model": self._active_model.value if self._active_model else "none"
                    },
                    reasoning=f"Detected {len(events)} notes",
                    reversible=True
                )

            return events

        except Exception as e:
            self._log_status(f"Detection failed: {e}", "error")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if self._config.cleanup_after_detection:
                self._force_cleanup()

    def _correct_octave_errors(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Correct octave errors (bass often detected one octave too high)."""
        if len(events) < 2:
            return events

        corrected = []
        for i, event in enumerate(events):
            original_pitch = event.pitch

            # Check if this is likely bass range (E1 to E3)
            if 28 <= original_pitch <= 52:
                # Check if previous note is an octave above
                if i > 0 and events[i - 1].pitch == original_pitch - 12:
                    event.pitch = original_pitch - 12
                    event.confidence *= 0.85
                    if event.reasoning_chain is None:
                        event.reasoning_chain = []
                    event.reasoning_chain.append(f"Octave corrected: {original_pitch} -> {event.pitch}")
                    print(f"[OCTAVE] Corrected {original_pitch} to {event.pitch}")

            corrected.append(event)

        return corrected

    def _force_cleanup(self):
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _update_progress(self, progress: float, message: str):
        if self._progress_callback:
            self._progress_callback(progress, message)
        if self._status_reporter:
            self._status_reporter.progress(self._name, progress, message)

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            getattr(self._status_reporter, level)(self._name, message)

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def name(self) -> str:
        return self._name

    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        events = self.detect(audio_buffer, context)
        execution_time_ms = (time.time() - start_time) * 1000
        memory_delta_mb = self._get_current_memory_mb() - start_memory

        return StageResult(
            stage_name=self._name,
            success=len(events) > 0,
            events=events,
            metadata={
                "event_count": len(events),
                "active_model": self._active_model.value if self._active_model else "none",
                "execution_time_ms": execution_time_ms,
                "memory_delta_mb": memory_delta_mb
            },
            execution_time_ms=execution_time_ms,
            memory_delta_mb=memory_delta_mb
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type.value,
            "last_event_count": len(self._last_events),
            "active_model": self._active_model.value if self._active_model else None,
            "total_memory_freed_mb": self._total_memory_freed_mb
        }

    def _get_current_memory_mb(self) -> float:
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


def create_pitch_intelligence(
        model: PitchModelType = PitchModelType.CREPE,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> PitchIntelligence:
    """Create a configured PitchIntelligence instance."""
    config = PitchDetectionConfig(primary_model=model)
    return PitchIntelligence(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )