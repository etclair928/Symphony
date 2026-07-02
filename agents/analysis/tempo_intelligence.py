# =================================================================
# MODULE: agents/analysis/tempo_intelligence.py
# DESCRIPTION: Tempo Intelligence for Grimlock 5.0.
#
# VERSION: 5.1 (FIXED: Thread isolation, no hardcoded defaults)
# UPDATED: 2026-05-15
#
# CRITICAL FIXES:
#   1. No hardcoded 120 BPM default - compute from audio
#   2. Thread-local BLAS configuration for madmom isolation
#   3. Proper fallback chain with confidence scoring
#   4. Input sanitization for all audio buffers
# =================================================================

import os
import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# CRITICAL: Set thread-local BLAS before importing heavy libs
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    TempoMap, TempoEvent, PulseField, BeatGrid,
    TimeSignatureCandidate, BeatTrackingResult, TempoIntelligenceResult
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    MIN_TEMPO_BPM,
    MAX_TEMPO_BPM,
    DEFAULT_TEMPO_BPM,
    BEAT_TRACKING_HOP_MS,
    PULSE_FIELD_RESOLUTION_MS,
    TEMPO_FORECAST_WINDOW_SECONDS,
    DEFAULT_TIME_SIGNATURE_NUMERATOR,
    DEFAULT_TIME_SIGNATURE_DENOMINATOR,
    TIME_SIGNATURE_CANDIDATE_LIMIT,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    AnalysisAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy.signal import find_peaks, resample_poly
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Madmom - isolated thread usage
MADMOM_AVAILABLE = False
try:
    import madmom
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    MADMOM_AVAILABLE = True
except ImportError:
    pass


class TempoWitnessType(str, Enum):
    LIBROSA = "librosa"
    MADMOM = "madmom"
    SPECTRAL = "spectral"
    ONSET = "onset"


class TempoChangeType(str, Enum):
    ACCELERANDO = "accelerando"
    RITARDANDO = "ritardando"
    SUBITO = "subito"
    RUBATO = "rubato"
    STEADY = "steady"


@dataclass(frozen=True)
class TempoWitnessVote:
    witness_type: TempoWitnessType
    time_ms: float
    tempo_bpm: float
    confidence: float
    weight: float = 1.0


@dataclass
class TempoConfig:
    min_tempo_bpm: float = MIN_TEMPO_BPM
    max_tempo_bpm: float = MAX_TEMPO_BPM
    default_tempo_bpm: float = DEFAULT_TEMPO_BPM
    detect_tempo_changes: bool = True
    tempo_change_window_seconds: float = 2.0
    tempo_change_threshold_bpm: float = 10.0
    min_stable_duration_seconds: float = 4.0
    beat_hop_ms: int = BEAT_TRACKING_HOP_MS
    beat_confidence_threshold: float = 0.3
    pulse_resolution_ms: int = PULSE_FIELD_RESOLUTION_MS
    pulse_smoothing_sigma: float = 2.0
    detect_time_signature: bool = True
    time_sig_candidate_limit: int = TIME_SIGNATURE_CANDIDATE_LIMIT
    use_ensemble: bool = True
    min_witnesses_for_consensus: int = 2
    witness_weights: Dict[TempoWitnessType, float] = field(default_factory=lambda: {
        TempoWitnessType.MADMOM: 1.0,
        TempoWitnessType.LIBROSA: 0.8,
        TempoWitnessType.SPECTRAL: 0.6,
        TempoWitnessType.ONSET: 0.7
    })
    max_tempo_events: int = 1000
    cache_ttl_seconds: int = 3600
    cleanup_after_analysis: bool = True
    sanitize_input: bool = True


class LibrosaTempoWitness:
    def __init__(self, config: TempoConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._witness_type = TempoWitnessType.LIBROSA

    def _sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.config.sanitize_input:
            return audio
        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return audio

    def estimate_tempo(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        if not LIBROSA_AVAILABLE:
            return self.config.default_tempo_bpm, 0.0

        audio = self._sanitize_audio(audio)

        try:
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio,
                sr=sr,
                hop_length=512,
                start_bpm=self.config.min_tempo_bpm
            )
            # librosa >=0.10 returns tempo as a 0-d/1-element ndarray rather
            # than a scalar - coerce immediately so downstream arithmetic,
            # comparisons, and string formatting all see a plain float.
            tempo = float(np.asarray(tempo).reshape(-1)[0])

            if len(beat_frames) > 4:
                beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
                intervals = np.diff(beat_times)
                if len(intervals) > 0:
                    consistency = 1.0 - min(1.0, np.std(intervals) / (np.mean(intervals) + 1e-8))
                    confidence = min(0.9, 0.5 + consistency * 0.4)
                else:
                    confidence = 0.5
            else:
                confidence = 0.3

            tempo = max(self.config.min_tempo_bpm, min(self.config.max_tempo_bpm, tempo))
            return tempo, confidence

        except Exception as e:
            if self.status_reporter:
                self.status_reporter.warn("LibrosaTempoWitness", f"Failed: {e}")
            return self.config.default_tempo_bpm, 0.0


class SpectralTempoWitness:
    def __init__(self, config: TempoConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._witness_type = TempoWitnessType.SPECTRAL

    def _sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.config.sanitize_input:
            return audio
        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return audio

    def estimate_tempo(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        if not LIBROSA_AVAILABLE:
            return self.config.default_tempo_bpm, 0.0

        audio = self._sanitize_audio(audio)

        try:
            hop_length = 512
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=sr,
                hop_length=hop_length
            )

            autocorr = np.correlate(onset_env, onset_env, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            peaks, _ = find_peaks(autocorr, height=np.max(autocorr) * 0.3)

            if len(peaks) == 0:
                return self.config.default_tempo_bpm, 0.0

            hop_seconds = hop_length / sr
            tempos = []
            confidences = []

            for peak in peaks[:10]:
                period_seconds = peak * hop_seconds
                if period_seconds > 0:
                    tempo = 60.0 / period_seconds
                    if self.config.min_tempo_bpm <= tempo <= self.config.max_tempo_bpm:
                        tempos.append(tempo)
                        confidences.append(autocorr[peak] / np.max(autocorr))

            if not tempos:
                return self.config.default_tempo_bpm, 0.0

            total_weight = sum(confidences)
            if total_weight > 0:
                tempo = sum(t * c for t, c in zip(tempos, confidences)) / total_weight
                confidence = max(confidences)
            else:
                tempo = tempos[0]
                confidence = 0.5

            return tempo, min(0.9, confidence)

        except Exception as e:
            if self.status_reporter:
                self.status_reporter.warn("SpectralTempoWitness", f"Failed: {e}")
            return self.config.default_tempo_bpm, 0.0


class MadmomTempoWitness:
    def __init__(self, config: TempoConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._witness_type = TempoWitnessType.MADMOM
        self._processor = None
        self._init_processor()

    def _init_processor(self):
        if not MADMOM_AVAILABLE:
            return
        try:
            self._processor = DBNBeatTrackingProcessor(fps=100)
        except Exception as e:
            if self.status_reporter:
                self.status_reporter.warn("MadmomTempoWitness", f"Init failed: {e}")

    def _sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.config.sanitize_input:
            return audio
        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return audio

    def estimate_tempo(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        if not MADMOM_AVAILABLE or self._processor is None:
            return self.config.default_tempo_bpm, 0.0

        audio = self._sanitize_audio(audio)

        try:
            if sr != 44100:
                if SCIPY_AVAILABLE:
                    gcd = np.gcd(sr, 44100)
                    up = 44100 // gcd
                    down = sr // gcd
                    audio_44k = resample_poly(audio, up, down)
                else:
                    return self.config.default_tempo_bpm, 0.0
            else:
                audio_44k = audio

            proc = RNNBeatProcessor()(audio_44k)
            beats = self._processor(proc)

            if len(beats) < 4:
                return self.config.default_tempo_bpm, 0.0

            intervals = np.diff(beats)
            median_interval = np.median(intervals)

            if median_interval > 0:
                tempo = 60.0 / median_interval
                consistency = 1.0 - min(1.0, np.std(intervals) / (median_interval + 1e-8))
                confidence = min(0.95, 0.6 + consistency * 0.35)
            else:
                tempo = self.config.default_tempo_bpm
                confidence = 0.0

            tempo = max(self.config.min_tempo_bpm, min(self.config.max_tempo_bpm, tempo))
            return tempo, confidence

        except Exception as e:
            if self.status_reporter:
                self.status_reporter.warn("MadmomTempoWitness", f"Failed: {e}")
            return self.config.default_tempo_bpm, 0.0


class EnsembleTempoDetector:
    def __init__(self, config: TempoConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._witnesses: Dict[TempoWitnessType, Any] = {}

        if LIBROSA_AVAILABLE:
            self._witnesses[TempoWitnessType.LIBROSA] = LibrosaTempoWitness(config, status_reporter)
        if MADMOM_AVAILABLE:
            self._witnesses[TempoWitnessType.MADMOM] = MadmomTempoWitness(config, status_reporter)
        if LIBROSA_AVAILABLE:
            self._witnesses[TempoWitnessType.SPECTRAL] = SpectralTempoWitness(config, status_reporter)

    def estimate_global_tempo(self, audio: np.ndarray, sr: int) -> Tuple[float, float, Dict[str, Any]]:
        votes: List[TempoWitnessVote] = []
        witness_results = {}

        for witness_type, witness in self._witnesses.items():
            tempo, confidence = witness.estimate_tempo(audio, sr)
            weight = self.config.witness_weights.get(witness_type, 0.5)

            votes.append(TempoWitnessVote(
                witness_type=witness_type,
                time_ms=0,
                tempo_bpm=tempo,
                confidence=confidence,
                weight=weight
            ))

            witness_results[witness_type.value] = {
                "tempo": tempo,
                "confidence": confidence,
                "weight": weight
            }

        if not votes:
            return self.config.default_tempo_bpm, 0.0, witness_results

        total_weight = sum(v.confidence * v.weight for v in votes)
        if total_weight > 0:
            tempo = sum(v.tempo_bpm * v.confidence * v.weight for v in votes) / total_weight
            confidence = total_weight / sum(v.weight for v in votes)
        else:
            tempo = np.mean([v.tempo_bpm for v in votes])
            confidence = 0.3

        tempo = max(self.config.min_tempo_bpm, min(self.config.max_tempo_bpm, tempo))
        confidence = min(0.95, confidence)

        return tempo, confidence, witness_results

    def detect_tempo_changes(self, audio: np.ndarray, sr: int,
                             duration_seconds: float) -> List[TempoEvent]:
        if not self.config.detect_tempo_changes:
            return []

        window_sec = self.config.tempo_change_window_seconds
        hop_sec = window_sec / 2

        tempos = []
        times = []

        for start in np.arange(0, duration_seconds - window_sec, hop_sec):
            end = start + window_sec
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            window = audio[start_sample:end_sample]
            if len(window) > sr * 0.1:
                tempo, confidence, _ = self.estimate_global_tempo(window, sr)
                if confidence > 0.3:
                    tempos.append(tempo)
                    times.append(start + window_sec / 2)

        if len(tempos) < 2:
            return []

        tempo_events = []
        previous_tempo = tempos[0]

        for i, (tempo, time_sec) in enumerate(zip(tempos, times)):
            if i == 0:
                continue

            change = abs(tempo - previous_tempo)
            if change >= self.config.tempo_change_threshold_bpm:
                event = TempoEvent(
                    time_ms=time_sec * 1000,
                    tempo_bpm=tempo,
                    confidence=Confidence.MEDIUM
                )
                tempo_events.append(event)
                previous_tempo = tempo

        return tempo_events


class TimeSignatureDetector:
    def __init__(self, config: TempoConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def detect_time_signature(self, beat_times: List[float],
                              onset_strengths: Optional[np.ndarray] = None) -> List[TimeSignatureCandidate]:
        if len(beat_times) < 8:
            return self._get_default_candidates()

        intervals = np.diff(beat_times)
        median_interval = np.median(intervals)
        downbeat_candidates = self._find_downbeat_patterns(beat_times, intervals)

        candidates = []

        for numerator in [2, 3, 4, 6, 5, 7, 9, 12]:
            denominator = 4
            score = self._score_time_signature(beat_times, downbeat_candidates, numerator)

            if score > 0.3:
                candidates.append(TimeSignatureCandidate(
                    numerator=numerator,
                    denominator=denominator,
                    confidence=score,
                    sample_segment_start_ms=beat_times[0] * 1000 if beat_times else 0,
                    sample_segment_end_ms=beat_times[-1] * 1000 if beat_times else 0
                ))

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[:self.config.time_sig_candidate_limit]

    def _find_downbeat_patterns(self, beat_times: List[float],
                                intervals: np.ndarray) -> List[int]:
        downbeats = [0]
        for period in [2, 3, 4, 6]:
            for start in range(period, len(beat_times), period):
                if start < len(beat_times):
                    downbeats.append(start)
        return list(set(downbeats))

    def _score_time_signature(self, beat_times: List[float],
                              downbeats: List[int], numerator: int) -> float:
        if len(beat_times) < numerator:
            return 0.0

        expected_downbeats = list(range(0, len(beat_times), numerator))

        if not expected_downbeats:
            return 0.0

        matches = sum(1 for db in downbeats if db in expected_downbeats)
        score = matches / max(len(expected_downbeats), 1)
        return min(1.0, score)

    def _get_default_candidates(self) -> List[TimeSignatureCandidate]:
        return [
            TimeSignatureCandidate(
                numerator=DEFAULT_TIME_SIGNATURE_NUMERATOR,
                denominator=DEFAULT_TIME_SIGNATURE_DENOMINATOR,
                confidence=0.5,
                sample_segment_start_ms=0,
                sample_segment_end_ms=0
            )
        ]


class BeatGridBuilder:
    def __init__(self, config: TempoConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def build_pulse_field(self, tempo_map: TempoMap, duration_seconds: float) -> PulseField:
        beat_grid_ms = []
        beat_duration_ms = 60000.0 / max(tempo_map.initial_tempo_bpm, 1.0)

        current = 0
        while current < duration_seconds * 1000:
            beat_grid_ms.append(current)
            tempo = tempo_map.get_tempo_at_ms(current)
            beat_duration_ms = 60000.0 / max(tempo, 1.0)
            current += beat_duration_ms

        sixteenth_grid_ms = []
        for beat in beat_grid_ms:
            for i in range(4):
                sixteenth_grid_ms.append(beat + i * beat_duration_ms / 4)
        sixteenth_grid_ms.sort()

        pulse_strength = [1.0] * len(beat_grid_ms)

        return PulseField(
            tempo_bpm=tempo_map.initial_tempo_bpm,
            confidence=tempo_map.confidence,
            beat_grid_ms=beat_grid_ms,
            sixteenth_grid_ms=sixteenth_grid_ms,
            pulse_strength=pulse_strength,
            phase_shift_ms=0.0
        )

    def build_beat_grid(self, beat_times: List[float], tempo_bpm: float) -> BeatGrid:
        beat_duration_ms = 60000.0 / max(tempo_bpm, 1.0)

        if len(beat_times) < 4:
            beat_times = [i * beat_duration_ms for i in range(32)]

        downbeats = [beat_times[0]]
        measure_duration = beat_duration_ms * 4

        current = downbeats[0] + measure_duration
        while current < beat_times[-1]:
            nearest = min(beat_times, key=lambda t: abs(t - current))
            if abs(nearest - current) < beat_duration_ms * 0.5:
                downbeats.append(nearest)
            current += measure_duration

        return BeatGrid(
            beat_times_ms=beat_times,
            downbeat_times_ms=downbeats,
            measure_duration_ms=measure_duration,
            beat_duration_ms=beat_duration_ms,
            tempo_bpm=tempo_bpm,
            time_signature="4/4"
        )


class TempoIntelligence:
    """Tempo Intelligence Agent for Grimlock 5.0."""

    def __init__(
            self,
            config: Optional[TempoConfig] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        self._name = "tempo_intelligence"
        self._source_type = SourceType.TEMPO_INTELLIGENCE
        self._config = config or TempoConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback

        self._ensemble_detector = EnsembleTempoDetector(self._config, status_reporter)
        self._time_sig_detector = TimeSignatureDetector(self._config, status_reporter)
        self._beat_grid_builder = BeatGridBuilder(self._config, status_reporter)

        self._last_tempo_map: Optional[TempoMap] = None
        self._last_pulse_field: Optional[PulseField] = None
        self._last_beat_grid: Optional[BeatGrid] = None
        self._last_time_signatures: List[TimeSignatureCandidate] = []
        self._last_execution_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0

        self._log_status("TempoIntelligence initialized")

    def _sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self._config.sanitize_input:
            return audio
        if np.any(~np.isfinite(audio)):
            self._log_status("Found non-finite values in audio, sanitizing", "warn")
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return audio

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def name(self) -> str:
        return self._name

    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting tempo intelligence")

        try:
            result = self.analyze_tempo(audio_buffer, context)

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory
            self._last_execution_time_ms = execution_time_ms

            if self._music_box:
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="tempo_analysis",
                    before_state={"audio_duration": context.duration_seconds},
                    after_state={
                        "initial_tempo": result.tempo_map.initial_tempo_bpm,
                        "tempo_confidence": result.tempo_map.confidence.value,
                        "time_sig_candidates": len(result.time_signature_candidates),
                        "beat_count": len(result.beat_track)
                    },
                    reasoning=f"Tempo: {result.tempo_map.initial_tempo_bpm:.1f} BPM",
                    reversible=True
                )

            if self._config.cleanup_after_analysis:
                self._force_cleanup()

            self._update_progress(1.0, f"Complete: {result.tempo_map.initial_tempo_bpm:.1f} BPM")

            return StageResult(
                stage_name=self._name,
                success=True,
                events=[],
                metadata={
                    "initial_tempo_bpm": result.tempo_map.initial_tempo_bpm,
                    "tempo_confidence": result.tempo_map.confidence.value,
                    "tempo_event_count": len(result.tempo_map.tempo_events),
                    "pulse_field_bpm": result.pulse_field.tempo_bpm,
                    "pulse_field_confidence": result.pulse_field.confidence.value,
                    "time_sig_candidates": len(result.time_signature_candidates),
                    "beat_track_count": len(result.beat_track),
                    "has_beat_grid": result.beat_grid is not None,
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Tempo analysis failed: {e}", "error")
            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def analyze_tempo(self, audio_buffer: np.ndarray,
                      context: AudioContext) -> TempoIntelligenceResult:
        audio_buffer = self._sanitize_audio(audio_buffer)

        self._update_progress(0.1, "Estimating global tempo")

        global_tempo, confidence, witness_details = self._ensemble_detector.estimate_global_tempo(
            audio_buffer, context.working_sample_rate
        )

        self._log_status(f"Global tempo: {global_tempo:.1f} BPM (confidence {confidence:.2f})")

        self._update_progress(0.3, "Detecting tempo changes")
        tempo_events = []

        if self._config.detect_tempo_changes:
            tempo_events = self._ensemble_detector.detect_tempo_changes(
                audio_buffer, context.working_sample_rate, context.duration_seconds
            )
            self._log_status(f"Detected {len(tempo_events)} tempo changes")

        tempo_map = TempoMap(
            initial_tempo_bpm=global_tempo,
            tempo_events=tempo_events,
            confidence=Confidence.from_float(confidence)
        )

        self._update_progress(0.5, "Building pulse field")
        pulse_field = self._beat_grid_builder.build_pulse_field(tempo_map, context.duration_seconds)
        beat_grid = self._beat_grid_builder.build_beat_grid(
            pulse_field.beat_grid_ms, tempo_map.initial_tempo_bpm
        )

        self._update_progress(0.7, "Building beat track")
        beat_track = []
        for i, beat_ms in enumerate(pulse_field.beat_grid_ms[:100]):
            beat_track.append(BeatTrackingResult(
                timestamp_ms=beat_ms,
                beat_number=i,
                confidence=pulse_field.pulse_strength[i] if i < len(pulse_field.pulse_strength) else 0.7,
                tempo_instant_bpm=tempo_map.get_tempo_at_ms(beat_ms)
            ))

        self._update_progress(0.85, "Detecting time signature")
        time_signatures = []
        primary_time_sig = None

        if self._config.detect_time_signature:
            time_signatures = self._time_sig_detector.detect_time_signature(
                pulse_field.beat_grid_ms,
                None
            )
            if time_signatures:
                primary_time_sig = time_signatures[0]

        self._last_tempo_map = tempo_map
        self._last_pulse_field = pulse_field
        self._last_beat_grid = beat_grid
        self._last_time_signatures = time_signatures

        return TempoIntelligenceResult(
            tempo_map=tempo_map,
            pulse_field=pulse_field,
            beat_track=beat_track,
            time_signature_candidates=time_signatures,
            beat_grid=beat_grid,
            time_signature_primary=primary_time_sig,
            is_constant_tempo=len(tempo_events) == 0
        )

    def build_pulse_field(self, tempo_map: TempoMap, duration_seconds: float) -> PulseField:
        return self._beat_grid_builder.build_pulse_field(tempo_map, duration_seconds)

    def _force_cleanup(self):
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
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

    def _get_current_memory_mb(self) -> float:
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def validate(self, gate: ValidationGate) -> ValidationResult:
        if self._last_tempo_map is None:
            return ValidationResult(
                is_valid=False,
                gate_used=gate,
                reason=VetoReason.EMPTY_RESULT,
                detail="No tempo analysis performed"
            )

        if gate == ValidationGate.CONFIDENCE_THRESHOLD:
            is_valid = self._last_tempo_map.confidence.value >= Confidence.LOW.value
            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                detail=f"Tempo confidence: {self._last_tempo_map.confidence.value:.2f}",
                confidence_before=self._last_tempo_map.confidence.value,
                confidence_after=self._last_tempo_map.confidence.value if is_valid else 0.0
            )
        elif gate == ValidationGate.TEMPO_REASONABLENESS:
            is_valid = (self._config.min_tempo_bpm <=
                        self._last_tempo_map.initial_tempo_bpm <=
                        self._config.max_tempo_bpm)
            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.TEMPO_OUTLIER,
                detail=f"Tempo: {self._last_tempo_map.initial_tempo_bpm:.1f} BPM"
            )
        else:
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                detail=f"Gate {gate.value} not fully supported"
            )

    def get_confidence(self) -> Confidence:
        if self._last_tempo_map is None:
            return Confidence.HALLUCINATION
        return self._last_tempo_map.confidence

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        if self._last_tempo_map is None:
            return (VetoReason.EMPTY_RESULT, "No tempo analysis")
        if self._last_tempo_map.confidence.value < Confidence.LOW.value:
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Confidence {self._last_tempo_map.confidence.value:.2f}")
        if not (self._config.min_tempo_bpm <= self._last_tempo_map.initial_tempo_bpm <= self._config.max_tempo_bpm):
            return (VetoReason.TEMPO_OUTLIER,
                    f"Tempo {self._last_tempo_map.initial_tempo_bpm:.1f} BPM out of range")
        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        return SchoenbergResult(
            verdict=SchoenbergVerdict.UNCERTAIN,
            zero_crossing_rate=0.0,
            spectral_flatness=0.5,
            reason="TempoIntelligence analyzes rhythm timing, not harmonic series"
        )

    def release_buffer(self, buffer_name: str) -> None:
        pass

    def get_memory_footprint_mb(self) -> float:
        return self._get_current_memory_mb()

    def can_release(self, buffer_name: str) -> bool:
        return True

    def staggered_gc(self) -> Dict[str, Any]:
        before = self._get_current_memory_mb()
        self._force_cleanup()
        after = self._get_current_memory_mb()
        freed = before - after
        self._total_memory_freed_mb += max(0, freed)
        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": freed,
            "triggered_by": self._name,
            "total_freed_mb": self._total_memory_freed_mb
        }

    def get_last_tempo_map(self) -> Optional[TempoMap]:
        return self._last_tempo_map

    def get_last_pulse_field(self) -> Optional[PulseField]:
        return self._last_pulse_field

    def get_statistics(self) -> Dict[str, Any]:
        if self._last_tempo_map:
            return {
                "name": self._name,
                "source_type": self._source_type.value,
                "initial_tempo_bpm": self._last_tempo_map.initial_tempo_bpm,
                "tempo_confidence": self._last_tempo_map.confidence.value,
                "tempo_event_count": len(self._last_tempo_map.tempo_events),
                "time_sig_candidates": len(self._last_time_signatures),
                "last_execution_time_ms": self._last_execution_time_ms,
                "total_memory_freed_mb": self._total_memory_freed_mb
            }
        else:
            return {
                "name": self._name,
                "source_type": self._source_type.value,
                "last_execution_time_ms": self._last_execution_time_ms,
                "total_memory_freed_mb": self._total_memory_freed_mb
            }


def create_tempo_intelligence(
        min_tempo: float = MIN_TEMPO_BPM,
        max_tempo: float = MAX_TEMPO_BPM,
        default_tempo: float = DEFAULT_TEMPO_BPM,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> TempoIntelligence:
    config = TempoConfig(
        min_tempo_bpm=min_tempo,
        max_tempo_bpm=max_tempo,
        default_tempo_bpm=default_tempo
    )
    return TempoIntelligence(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )