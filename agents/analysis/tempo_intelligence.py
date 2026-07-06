# =================================================================
# MODULE: agents/analysis/tempo_intelligence.py
# DESCRIPTION: Tempo Intelligence for Grimlock 5.0.
#
# VERSION: 5.6.1 (FIXED: Thread isolation, no hardcoded defaults)
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
import queue
import threading
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

    # Minimum time budget given to each individual witness in the
    # ensemble (see EnsembleTempoDetector.estimate_global_tempo) before
    # it's excluded from the vote rather than blocking every other
    # witness - scales up for longer audio, but never below this floor.
    witness_timeout_seconds: float = 20.0


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
        """
        Runs each witness with its own bounded timeout instead of calling
        them in a plain loop. Previously, if any one witness (Madmom in
        particular - a heavyweight neural beat tracker, prone to stalling
        under system memory/CPU pressure even though it's fast and light
        when the system isn't under load) hung or ran long, the whole
        ensemble - including whatever the OTHER witnesses had already
        computed - was lost to the outer stage-level timeout, falling
        back to a bare default tempo with zero real signal. A stalled
        witness now just gets excluded from the vote; the others still
        get to contribute.
        """
        votes: List[TempoWitnessVote] = []
        witness_results = {}

        duration_seconds = len(audio) / sr if sr > 0 else 0.0
        per_witness_timeout = max(self.config.witness_timeout_seconds, duration_seconds * 0.5)

        # Launch every witness on its own daemon thread FIRST, all at once,
        # then join each with its timeout. If two witnesses both stall,
        # joining them one at a time after calling them one at a time
        # would cost the SUM of their timeouts; since they're all started
        # together here, joining is bounded by whichever one is slowest -
        # the wall-clock cost of N stalled witnesses is one timeout, not N.
        pending = {
            witness_type: self._start_witness_call(witness, audio, sr)
            for witness_type, witness in self._witnesses.items()
        }

        for witness_type, (thread, result_queue) in pending.items():
            weight = self.config.witness_weights.get(witness_type, 0.5)
            outcome = self._collect_witness_result(thread, result_queue, per_witness_timeout)

            if outcome is None:
                if self.status_reporter:
                    self.status_reporter.warn(
                        "EnsembleTempoDetector",
                        f"{witness_type.value} witness timed out after "
                        f"{per_witness_timeout:.0f}s - excluded from vote")
                witness_results[witness_type.value] = {
                    "tempo": None, "confidence": 0.0, "weight": weight, "timed_out": True}
                continue

            status, payload = outcome
            if status == "error":
                if self.status_reporter:
                    self.status_reporter.warn(
                        "EnsembleTempoDetector", f"{witness_type.value} failed: {payload}")
                witness_results[witness_type.value] = {
                    "tempo": None, "confidence": 0.0, "weight": weight, "error": str(payload)}
                continue

            tempo, confidence = payload

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

    @staticmethod
    def _start_witness_call(
            witness: Any, audio: np.ndarray, sr: int
    ) -> Tuple[threading.Thread, "queue.Queue"]:
        """
        Starts witness.estimate_tempo(audio, sr) on a daemon thread and
        returns immediately with the thread and the queue its result will
        land in - pairs with _collect_witness_result(). Split into a
        separate launch step (rather than one call that starts-and-joins)
        specifically so every witness can be started together before any
        of them are waited on: if two witnesses both stall, waiting on
        them one at a time after calling them one at a time would cost
        the SUM of their timeouts, while starting all of them up front
        means the wall-clock cost of N stalled witnesses is bounded by
        one timeout, not N, since they're all running concurrently while
        we wait.

        Uses a plain daemon thread rather than ThreadPoolExecutor
        deliberately: concurrent.futures registers an atexit hook that
        joins every worker thread at interpreter shutdown, so a witness
        that's still stuck when the process wants to exit would hang
        the whole process anyway even after eating a "timeout." A daemon
        thread carries no such guarantee - if it never finishes, it's
        simply abandoned when the process exits.
        """
        result_queue: "queue.Queue" = queue.Queue(maxsize=1)

        def target():
            try:
                result_queue.put(("ok", witness.estimate_tempo(audio, sr)))
            except Exception as e:
                result_queue.put(("error", e))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread, result_queue

    @staticmethod
    def _collect_witness_result(
            thread: threading.Thread, result_queue: "queue.Queue", timeout_seconds: float
    ) -> Optional[Tuple[str, Any]]:
        """
        Waits up to timeout_seconds for a thread started by
        _start_witness_call() to finish. Returns None on timeout, or
        ("ok", (tempo, confidence)) / ("error", exception) otherwise.
        """
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            return None

        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return None

    def detect_tempo_changes(self, audio: np.ndarray, sr: int,
                             duration_seconds: float) -> List[TempoEvent]:
        """
        Scan for local tempo drift across the track using a lightweight
        single witness, not the full ensemble.

        This used to call estimate_global_tempo() - launching all three
        witnesses (Madmom included) on their own threads - for every
        window. With a 2s window and 1s hop, that's ~2 * duration_seconds
        separate ensemble calls (measured: 146s on a real 60s clip,
        instead of the ~20-30s a single global estimate takes). Madmom is
        both the most expensive witness to run repeatedly and the least
        meaningful one on a bare 2-second slice anyway (barely 2-4 beats
        of context for a beat tracker). A single fast, deterministic
        witness is enough here: _confirm_tempo_changes already requires
        a candidate deviation to persist across several consecutive
        windows (min_stable_duration_seconds) before accepting it as a
        real tempo change, so this doesn't trade away reliability - it
        just stops paying Madmom's cost dozens of times for a coarse
        drift scan the global estimate already anchors.
        """
        if not self.config.detect_tempo_changes:
            return []

        window_sec = self.config.tempo_change_window_seconds
        hop_sec = window_sec / 2

        tempos = []
        times = []

        window_witness = (self._witnesses.get(TempoWitnessType.LIBROSA)
                          or self._witnesses.get(TempoWitnessType.SPECTRAL))

        for start in np.arange(0, duration_seconds - window_sec, hop_sec):
            end = start + window_sec
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            window = audio[start_sample:end_sample]
            if len(window) > sr * 0.1:
                if window_witness is not None:
                    tempo, confidence = window_witness.estimate_tempo(window, sr)
                else:
                    tempo, confidence = self.config.default_tempo_bpm, 0.0
                if confidence > 0.3:
                    tempos.append(tempo)
                    times.append(start + window_sec / 2)

        return self._confirm_tempo_changes(tempos, times)

    def _confirm_tempo_changes(self, tempos: List[float], times: List[float]) -> List[TempoEvent]:
        """
        Convert raw per-window tempo estimates into confirmed tempo-change
        events.

        A single window's estimate is noisy almost by construction:
        tempo_change_window_seconds is only 2s by default, which is barely
        2-4 beats of evidence - nowhere near enough for any tempo detector
        (librosa, madmom, or the autocorrelation witness) to be stable.
        Treating every adjacent-window jump past tempo_change_threshold_bpm
        as a real musical event produced several spurious "tempo changes"
        a second apart on a track that was almost certainly constant-tempo
        throughout (confirmed live: 6 jumps between 110-128 BPM crammed
        into a 30s clip, exported into the MIDI tempo track and visibly
        confusing downstream tools).

        min_stable_duration_seconds already existed on TempoConfig for
        exactly this purpose but was never actually enforced - a candidate
        deviation now has to persist for at least that long (i.e. be seen
        consistently across several consecutive windows, not just once)
        before it's accepted as a real tempo change.
        """
        if len(tempos) < 2:
            return []

        tempo_events = []
        accepted_tempo = tempos[0]
        candidate_tempo: Optional[float] = None
        candidate_start_time: Optional[float] = None

        for tempo, time_sec in zip(tempos[1:], times[1:]):
            change = abs(tempo - accepted_tempo)

            if change < self.config.tempo_change_threshold_bpm:
                # Back within tolerance of the accepted tempo - whatever
                # deviation was being tracked didn't hold up.
                candidate_tempo = None
                candidate_start_time = None
                continue

            if (candidate_tempo is not None
                    and abs(tempo - candidate_tempo) < self.config.tempo_change_threshold_bpm):
                # Same deviation as before - has it persisted long enough?
                if time_sec - candidate_start_time >= self.config.min_stable_duration_seconds:
                    tempo_events.append(TempoEvent(
                        time_ms=candidate_start_time * 1000,
                        tempo_bpm=candidate_tempo,
                        confidence=Confidence.MEDIUM
                    ))
                    accepted_tempo = candidate_tempo
                    candidate_tempo = None
                    candidate_start_time = None
            else:
                # A new deviation - start tracking it from here.
                candidate_tempo = tempo
                candidate_start_time = time_sec

        return tempo_events


class TempoOctaveCorrector:
    """
    Tests whether a tempo estimate is off by a musically-common rational
    factor - the classic "tempo octave error" where a beat tracker reports
    double/half/triplet/dotted-etc. of the tempo a listener actually feels.
    None of the existing witnesses (Librosa, Madmom, Spectral) or the
    ReverseGeoCrypt blend ever explicitly test for this: they vote between
    independent algorithms, but if multiple algorithms agree on the same
    wrong octave, nothing catches it.

    Each rational-multiple candidate is scored by how well real onset
    times align to its beat grid, using a CHANCE-CORRECTED score: a denser
    grid (higher BPM) catches more onsets within tolerance purely by
    coincidence, so raw "fraction of onsets near a grid line" is biased
    toward picking higher multiples regardless of whether they're
    musically real. This mirrors (and fixes) a real bug found in
    Grimlock 4.7's analogous TempoOctaveCorrector, which had no such
    correction and would confidently pick a wrong, denser-grid tempo.

    Deliberately conservative: only overrides the input tempo when a
    candidate beats the unmodified ("exact") candidate by a clear margin
    (min_improvement) - see _confirm_tempo_changes's docstring for why a
    single-signal jump should never be accepted without one.
    """

    # (numerator, denominator, label, description)
    RATIOS: List[Tuple[int, int, str, str]] = [
        (1, 1, "exact", "no correction"),
        (1, 2, "1/2", "half tempo"),
        (2, 1, "2x", "double tempo"),
        (1, 3, "1/3", "compound meter (12/8, 6/4)"),
        (3, 1, "3x", "triple tempo"),
        (2, 3, "2/3", "triplet pair"),
        (3, 2, "3/2", "dotted quarter"),
        (3, 4, "3/4", "dotted eighth"),
        (4, 3, "4/3", "inverse dotted"),
        (5, 6, "5/6", "compound correction"),
        (4, 5, "4/5", "quintuplet expansion"),
        (5, 4, "5/4", "quintuplet compression"),
    ]

    def __init__(self, config: TempoConfig, min_improvement: float = 0.15):
        self.config = config
        self.min_improvement = min_improvement

    def correct(self, raw_tempo: float,
                onset_times_sec: np.ndarray) -> Tuple[float, str, Dict[str, Any]]:
        """
        Returns (corrected_tempo, reason, debug_info). reason is
        "no_correction_needed" (or similar) when raw_tempo is kept.
        """
        if raw_tempo <= 0 or len(onset_times_sec) < 8:
            return raw_tempo, "insufficient_onsets", {}

        candidates = []
        for n, d, label, desc in self.RATIOS:
            candidate_bpm = raw_tempo * n / d
            if not (self.config.min_tempo_bpm <= candidate_bpm <= self.config.max_tempo_bpm):
                continue
            score = self._chance_corrected_score(candidate_bpm, onset_times_sec)
            candidates.append({"bpm": candidate_bpm, "label": label,
                               "desc": desc, "score": score})

        if not candidates:
            return raw_tempo, "no_candidates", {}

        candidates.sort(key=lambda c: -c["score"])
        exact = next((c for c in candidates if c["label"] == "exact"),
                     candidates[0])
        best = candidates[0]

        debug_info = {"candidates": candidates, "exact_score": exact["score"],
                     "best_score": best["score"]}

        if (best["label"] == "exact" or
                (best["score"] - exact["score"]) < self.min_improvement):
            return raw_tempo, "no_correction_needed", debug_info

        return best["bpm"], f"{best['label']} ({best['desc']})", debug_info

    def _chance_corrected_score(self, candidate_bpm: float,
                                onset_times_sec: np.ndarray,
                                tolerance_fraction: float = 0.06) -> float:
        """
        Combines two chance-corrected alignment checks and takes their
        minimum:

        - precision: fraction of ONSETS landing near some grid line.
        - recall: fraction of the candidate's own GRID LINES that have a
          nearby onset.

        Precision alone isn't enough to disambiguate a true tempo from an
        exact multiple of it: if the real beat is 120 BPM, the 240 BPM
        grid contains every one of those beats too, so onsets always land
        on "some" 240 BPM grid line - precision alone would score both
        candidates near-perfectly. Recall is what actually catches this:
        the 240 BPM candidate's grid has twice as many lines as there are
        real onsets, so roughly half of them have nothing nearby, while
        the true 120 BPM candidate's grid lines are all accounted for.
        Requiring both (via min) is what makes this genuinely tell octave
        multiples apart instead of just detecting "is periodic at all."

        Both directions get their own chance-correction (precision scales
        with a fixed tolerance fraction of the beat; recall scales with
        the actual onset rate), so denser candidate grids aren't favored
        by coincidence in either direction.
        """
        if candidate_bpm <= 0 or len(onset_times_sec) == 0:
            return 0.0
        beat_dur = 60.0 / candidate_bpm
        tol = beat_dur * tolerance_fraction

        # Precision: onsets -> nearest grid line
        phases = np.mod(onset_times_sec, beat_dur)
        dist_to_grid = np.minimum(phases, beat_dur - phases)
        precision_observed = float(np.mean(dist_to_grid < tol))
        chance_precision = min(1.0, 2.0 * tolerance_fraction)
        precision_score = max(0.0, (precision_observed - chance_precision)
                              / (1.0 - chance_precision + 1e-8))

        # Recall: this candidate's grid lines -> nearest onset
        duration_sec = (float(onset_times_sec[-1] - onset_times_sec[0])
                        if len(onset_times_sec) > 1 else beat_dur)
        grid_lines = np.arange(0.0, duration_sec, beat_dur)
        if len(grid_lines) == 0:
            return precision_score

        insert_idx = np.clip(np.searchsorted(onset_times_sec, grid_lines),
                             1, len(onset_times_sec) - 1)
        dist_left = np.abs(grid_lines - onset_times_sec[insert_idx - 1])
        dist_right = np.abs(onset_times_sec[insert_idx] - grid_lines)
        dist_to_onset = np.minimum(dist_left, dist_right)
        recall_observed = float(np.mean(dist_to_onset < tol))

        onset_rate = len(onset_times_sec) / max(duration_sec, 1e-8)
        chance_recall = min(1.0, onset_rate * 2.0 * tol)
        recall_score = max(0.0, (recall_observed - chance_recall)
                           / (1.0 - chance_recall + 1e-8))

        return min(precision_score, recall_score)


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