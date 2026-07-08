# =================================================================
# MODULE: agents/drum_intelligence/orchestrator.py
# VERSION: 6.0-draft
# DESCRIPTION: Thin DrumIntelligence orchestrator - sequencing only. All
# real detection/classification/articulation/interpretation/consensus
# logic lives in this package's other modules; this class just calls
# them in order and satisfies the pipeline-facing protocols
# (DetectionAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
# MusicBoxProtocol/StatusReporterProtocol consumers).
#
# REPLACES: agents/detection/drum_intelligence.py's DrumIntelligence,
# SilenceGater, create_drum_intelligence.
#
# Two real fixes land here that the old orchestrator never had:
#   1. _get_cached_tempo used to return a hardcoded 120.0 for ANY new
#      cache key and cache it as if it were measured - only reachable
#      when a caller fails to pass tempo_bpm (the real pipeline always
#      does), but a landmine if that ever changes. Replaced with an
#      honest, clearly-labeled fallback constant - no fake caching.
#   2. SilenceGater was fully built and instantiated in __init__ but
#      get_active_regions() was never actually called anywhere - the
#      whole audio buffer, silence included, always ran through the full
#      ensemble. Now genuinely wired: detect() skips silent regions
#      (title-count-in gaps, dead air) before running the ensemble,
#      re-offsetting event times back to absolute position afterward.
# =================================================================

from __future__ import annotations

import gc
import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import replace
from collections import deque

from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, DrumKitMapping, DecisionType
)
from core.constants import TARGET_SAMPLE_RATE, STAGGERED_GC_TRIGGER_MB
from core.protocols import MusicBoxProtocol, StatusReporterProtocol

from .drum_types import DrumIntelligenceConfig, ForensicDrumEvent, DetectionPass
from .onset_detection import SafeResampler
from .classification import SpectralDrumClassifier
from .articulation import ArticulationClassifier
from .interpretation import PatternHashDetector, MicrotimingAnalyzer, MetricalPositionEvidence
from .consensus import EnsembleDrumDetector

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Honest fallback tempo when a caller genuinely doesn't pass tempo_bpm.
# AudioContext carries no tempo field by design (a deterministic
# identity/format anchor, not an analysis result) - the real pipeline
# always passes the actual detected tempo explicitly since drum
# detection runs after tempo_analysis. This constant only fires for a
# caller that skips that contract.
_FALLBACK_TEMPO_BPM = 120.0


# ========================================================================
# Silence Gater
# ========================================================================

class SilenceGater:
    """Detects active (non-silent) regions so the ensemble skips dead air."""

    def __init__(self, config: DrumIntelligenceConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate

    def get_active_regions(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        if not LIBROSA_AVAILABLE:
            return [(0, len(audio))]

        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            hop_length = 512
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

            threshold_linear = 10 ** (self.config.silence_threshold_db / 20)
            active = rms > threshold_linear

            window_ms = self.config.min_energy_window_ms
            min_active_frames = int(window_ms * self.sample_rate / 1000 / hop_length)

            regions = []
            in_region = False
            start = 0

            for i, is_active in enumerate(active):
                if is_active and not in_region:
                    in_region = True
                    start = i * hop_length
                elif not is_active and in_region:
                    if (i * hop_length - start) > min_active_frames * hop_length:
                        regions.append((start, i * hop_length))
                    in_region = False

            if in_region:
                regions.append((start, len(audio)))

            return regions if regions else [(0, len(audio))]
        except Exception:
            return [(0, len(audio))]


# ========================================================================
# Main Drum Intelligence Agent
# ========================================================================

class DrumIntelligence:
    """Complete drum intelligence orchestrator - sequencing only."""

    def __init__(
            self,
            config: Optional[DrumIntelligenceConfig] = None,
            drum_kit_mapping: Optional[DrumKitMapping] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        self._name = "drum_intelligence"
        self._source_type = SourceType.DRUM_INTELLIGENCE
        self._config = config or DrumIntelligenceConfig()
        self._kit_mapping = drum_kit_mapping or DrumKitMapping.gm_standard()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback
        self._detection_threshold = Confidence.LOW.value

        # Real sample rate is only known once real audio/context arrives at
        # detect() - see _configure_for_sample_rate(), called at the top of
        # every detect() call. This constructor-time value is just a
        # placeholder until then; it used to be the ONLY value ever used,
        # permanently hardcoded regardless of what sample rate the caller's
        # audio actually was in. Confirmed as a real, active bug: pipeline.py
        # resamples the drums stem to 22050Hz before calling this, while
        # every internal onset/offset time computation kept assuming
        # 44100Hz - a systematic 2x time-compression that misreported real
        # hits in the back half of a clip as if they occurred at half their
        # true timestamp (verified directly: a real pipeline run only found
        # events within ms of half the timestamps of independently-measured
        # strong energy peaks in the same stem).
        self.sample_rate = TARGET_SAMPLE_RATE
        self._ensemble = EnsembleDrumDetector(self._config, self.sample_rate)
        self._articulation = ArticulationClassifier(self._config, self.sample_rate)
        self._silence_gater = SilenceGater(self._config, self.sample_rate)

        self._last_events: List[ForensicDrumEvent] = []
        self._last_patterns: List[Dict[str, Any]] = []
        self._detected_tempo: float = _FALLBACK_TEMPO_BPM
        self._last_execution_time_ms: float = 0.0
        self._history: deque = deque(maxlen=self._config.max_retained_events)

        self._log_status("DrumIntelligence initialized")

    def _configure_for_sample_rate(self, sample_rate: int) -> None:
        """
        Rebuilds the sub-detectors for the real sample rate of the audio
        about to be processed. Called at the top of every detect() - audio
        sample rate is a per-call fact (the caller's context, not something
        fixed at construction time), so this is the only place that can
        actually guarantee correctness regardless of what rate the caller
        resampled to. A no-op when the rate hasn't changed since last call.
        """
        if sample_rate == self.sample_rate:
            return
        self.sample_rate = sample_rate
        self._ensemble = EnsembleDrumDetector(self._config, self.sample_rate)
        self._articulation = ArticulationClassifier(self._config, self.sample_rate)
        self._silence_gater = SilenceGater(self._config, self.sample_rate)

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

    def run(self, audio_buffer: np.ndarray, context: AudioContext,
            tempo_bpm: Optional[float] = None,
            time_sig_numerator: int = 4,
            downbeat_offset_ms: float = 0.0) -> StageResult:
        """Process audio and return StageResult with NoteEvents."""
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting drum detection")

        try:
            events = self.detect(audio_buffer, context, tempo_bpm,
                                 time_sig_numerator, downbeat_offset_ms)

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory
            self._last_execution_time_ms = execution_time_ms

            if self._get_current_memory_mb() > STAGGERED_GC_TRIGGER_MB:
                self._force_cleanup()

            if self._music_box:
                self._log_decision_to_music_box(audio_buffer, events)

            self._update_progress(1.0, f"Complete: {len(events)} events")

            return StageResult(
                stage_name=self._name,
                success=len(events) > 0,
                events=events,
                metadata={
                    "event_count": len(events),
                    "pattern_count": len(self._last_patterns),
                    "tempo": self._detected_tempo,
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Detection failed: {e}", "error")
            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def _log_decision_to_music_box(self, audio_buffer: np.ndarray, events: List[NoteEvent]) -> None:
        articulation_counts: Dict[str, int] = {}
        nmf_recovered = 0
        contention_resolved = 0
        confidence_components_sum = {
            "transient_strength": 0.0, "spectral_match": 0.0,
            "temporal_consistency": 0.0, "model_confidence": 0.0,
            "ensemble_agreement": 0.0, "metrical_fit": 0.0,
        }
        for e in self._last_events:
            articulation_counts[e.articulation.value] = articulation_counts.get(e.articulation.value, 0) + 1
            for tag in (e.reasoning_chain or ()):
                if "NMF polyphony split" in tag:
                    nmf_recovered += 1
                elif "Contention resolved" in tag:
                    contention_resolved += 1
            for field_name in confidence_components_sum:
                confidence_components_sum[field_name] += getattr(e.confidence, field_name)
        n = max(1, len(self._last_events))
        mean_confidence_components = {k: v / n for k, v in confidence_components_sum.items()}

        self._music_box.log_decision(
            stage_name=self._name,
            decision_type=DecisionType.DRUM_TESTIMONY,
            before_state={"audio_shape": list(audio_buffer.shape)},
            after_state={
                "event_count": len(events),
                "pattern_count": len(self._last_patterns),
                "tempo": self._detected_tempo,
                "articulation_counts": articulation_counts,
                "nmf_recovered_count": nmf_recovered,
                "contention_resolved_count": contention_resolved,
                "mean_confidence_components": mean_confidence_components,
            },
            reasoning=f"Detected {len(events)} drum events",
            reversible=True
        )

    def detect(self, audio_buffer: np.ndarray, context: AudioContext,
               tempo_bpm: Optional[float] = None,
               time_sig_numerator: int = 4,
               downbeat_offset_ms: float = 0.0) -> List[NoteEvent]:
        """Run the full detection pipeline: witnesses -> classification ->
        consensus -> articulation -> interpretation -> NoteEvents."""
        if audio_buffer is None or len(audio_buffer) == 0:
            return []

        self._configure_for_sample_rate(context.working_sample_rate)

        audio_buffer = self._sanitize_audio(audio_buffer)
        if audio_buffer.dtype != np.float32:
            audio_buffer = audio_buffer.astype(np.float32)

        tempo = tempo_bpm if tempo_bpm and tempo_bpm > 0 else self._fallback_tempo()
        self._detected_tempo = tempo

        density = self._estimate_rolling_density(audio_buffer)
        self._log_status(f"Tempo: {tempo:.1f} BPM, Density: {density:.2f}")

        self._update_progress(0.15, "Locating active regions")
        raw_events = self._detect_over_active_regions(audio_buffer, density, tempo)

        self._update_progress(0.4, "Cleaning events")
        cleaned_events = self._deduplicate_events(raw_events, window_ms=self._config.dedup_window_ms)
        cleaned_events.sort(key=lambda e: (e.start_ms, e.velocity))

        self._update_progress(0.55, "Classifying articulations")
        classified_events = self._articulation.classify(cleaned_events, audio_buffer)

        self._update_progress(0.68, "Detecting patterns")
        # Pattern annotation is enrichment metadata only (pattern_count in
        # StageResult.metadata) - it never feeds the returned NoteEvents,
        # so an isolated failure here can only ever cost pattern metadata,
        # never the actual drum notes.
        try:
            pattern_detector = PatternHashDetector(self._config, tempo)
            self._last_patterns = pattern_detector.detect_patterns(classified_events)
        except Exception as e:
            self._log_status(f"Pattern detection failed (non-fatal): {e}", "warn")
            self._last_patterns = []

        self._update_progress(0.78, "Analyzing microtiming")
        microtiming_analyzer = MicrotimingAnalyzer(self._config, tempo, downbeat_offset_ms)
        timed_events = microtiming_analyzer.analyze(classified_events)

        self._update_progress(0.85, "Weighing metrical position evidence")
        metrical_evidence = MetricalPositionEvidence(
            self._config, tempo, time_sig_numerator, downbeat_offset_ms)
        interpreted_events = metrical_evidence.apply(timed_events)

        self._update_progress(0.9, "Adding forensic data")
        forensic_events = self._add_forensic_data(interpreted_events)

        # min_confidence existed but was, at one point in this system's
        # history, never actually applied anywhere - every detected blip
        # became a NoteEvent regardless of confidence. Kept as a real gate
        # here.
        before_filter = len(forensic_events)
        forensic_events = [e for e in forensic_events
                           if e.confidence.total() >= self._config.min_confidence]
        if before_filter != len(forensic_events):
            self._log_status(
                f"Confidence filter: {before_filter - len(forensic_events)} "
                f"of {before_filter} events below {self._config.min_confidence} removed")

        self._update_progress(0.97, "Converting to NoteEvents")
        note_events = self._to_note_events(forensic_events)

        self._last_events = forensic_events[-self._config.max_retained_events:]

        return note_events

    def _detect_over_active_regions(self, audio_buffer: np.ndarray,
                                    density: float, tempo: float) -> List[ForensicDrumEvent]:
        """
        Runs the ensemble only over non-silent regions, skipping dead air
        (count-ins, intros, outros) entirely rather than wasting the full
        ensemble's cost on silence. Falls back to running the whole
        buffer when the gater finds one region spanning it (nothing to
        skip) or fails.
        """
        regions = self._silence_gater.get_active_regions(audio_buffer)

        if len(regions) == 1 and regions[0] == (0, len(audio_buffer)):
            return self._ensemble.detect(audio_buffer, density, tempo)

        all_events: List[ForensicDrumEvent] = []
        for start, end in regions:
            if end <= start:
                continue
            segment = audio_buffer[start:end]
            segment_events = self._ensemble.detect(segment, density, tempo)
            offset_ms = start / self.sample_rate * 1000.0
            all_events.extend(
                replace(e, start_ms=e.start_ms + offset_ms, end_ms=e.end_ms + offset_ms)
                for e in segment_events
            )

        all_events.sort(key=lambda e: (e.start_ms, e.velocity))
        return all_events

    def _deduplicate_events(self, events: List[ForensicDrumEvent], window_ms: float = 12.0) -> List[ForensicDrumEvent]:
        if not events:
            return []

        events = sorted(events, key=lambda e: e.start_ms)
        result = []

        for event in events:
            if not result:
                result.append(event)
            else:
                last = result[-1]
                if event.drum_type == last.drum_type and abs(event.start_ms - last.start_ms) < window_ms:
                    if event.confidence.total() > last.confidence.total():
                        result[-1] = event
                else:
                    result.append(event)

        return result

    def _add_forensic_data(self, events: List[ForensicDrumEvent]) -> List[ForensicDrumEvent]:
        return [
            replace(
                event,
                spectral_centroid_hz=0.0,
                spectral_rolloff_hz=0.0,
                transient_sharpness=event.confidence.transient_strength,
                detection_pass=DetectionPass.FORENSIC
            )
            for event in events
        ]

    def _to_note_events(self, events: List[ForensicDrumEvent]) -> List[NoteEvent]:
        return [e.to_note_event(self._kit_mapping) for e in events]

    def _fallback_tempo(self) -> float:
        self._log_status(
            f"No tempo_bpm passed by caller - using fallback {_FALLBACK_TEMPO_BPM} BPM. "
            "The real pipeline should always pass the actual detected tempo "
            "(drum detection runs after tempo_analysis).", "warn")
        return _FALLBACK_TEMPO_BPM

    def _estimate_rolling_density(self, audio: np.ndarray) -> float:
        if not LIBROSA_AVAILABLE:
            return 0.5

        audio = self._sanitize_audio(audio)

        try:
            hop_length = 512
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=hop_length)

            window_frames = int(2 * self.sample_rate / hop_length)
            if len(onset_env) <= window_frames:
                return 0.5

            densities = []
            for i in range(0, len(onset_env) - window_frames, window_frames // 2):
                window = onset_env[i:i + window_frames]
                density = np.mean(window > np.percentile(window, 75))
                densities.append(density)

            return float(np.mean(densities)) if densities else 0.5
        except Exception:
            return 0.5

    def _force_cleanup(self):
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
        if gate == ValidationGate.SILENCE_DETECTOR:
            is_valid = len(self._last_events) > 0
            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.EXCESS_SILENCE,
                events_validated=len(self._last_events),
                events_rejected=0
            )
        elif gate == ValidationGate.CONFIDENCE_THRESHOLD:
            if not self._last_events:
                return ValidationResult(is_valid=False, gate_used=gate, reason=VetoReason.CONFIDENCE_TOO_LOW)

            avg_confidence = np.mean([e.confidence.total() for e in self._last_events])
            is_valid = avg_confidence >= self._detection_threshold

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                confidence_before=avg_confidence,
                confidence_after=avg_confidence if is_valid else 0.0
            )
        else:
            return ValidationResult(is_valid=True, gate_used=gate)

    def get_confidence(self) -> Confidence:
        if not self._last_events:
            return Confidence.HALLUCINATION
        avg_confidence = np.mean([e.confidence.total() for e in self._last_events])
        return Confidence.from_float(avg_confidence)

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        if len(self._last_events) < 2:
            return (VetoReason.EXCESS_SILENCE, f"Only {len(self._last_events)} events detected")
        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        return SchoenbergResult(
            verdict=SchoenbergVerdict.PERCUSSION,
            zero_crossing_rate=0.0,
            spectral_flatness=0.7,
            reason="Percussion events by design"
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

        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": before - after,
            "triggered_by": self._name
        }

    @property
    def detection_threshold(self) -> Confidence:
        return Confidence.LOW

    def get_raw_confidence(self) -> float:
        if not self._last_events:
            return 0.0
        return float(np.mean([e.confidence.total() for e in self._last_events]))


def create_drum_intelligence(
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> DrumIntelligence:
    config = DrumIntelligenceConfig()
    return DrumIntelligence(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )
