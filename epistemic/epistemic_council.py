# =================================================================
# MODULE: epistemic/epistemic_council.py
# DESCRIPTION: The Council of Witnesses for polyphonic pitch detection.
#
# VERSION: 5.6.1 (Added analyze_sync method for pipeline compatibility)
# UPDATED: 2026-05-26
#
# KEY FIXES:
#   1. Fixed Basic Pitch call - passes numpy array correctly, not str(array)
#   2. Unified with ConsensusEngine - removed manual list merging
#   3. Returns StageResult with proper testimony
#   4. Proper temp file handling for models that require file paths
#   5. Added proper async error handling
#   6. Graceful degradation when models fail
#   7. Added analyze_sync() method for synchronous pipeline usage
#   8. Fixed CREPE refinement to handle None returns
# =================================================================

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np

from core.order_types import NoteEvent, SourceType, Confidence
from core.testimony import PitchTestimony
from epistemic.musical_findings_map import MusicalFindingsMap

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class CouncilConfig:
    """
    Configuration for the Epistemic Council.
    """
    # Pre-flight proxy
    proxy_sample_rate: int = 22050
    full_sample_rate: int = 16000

    # Model enable flags
    use_librosa: bool = True
    use_spice: bool = True
    use_basic_pitch: bool = True
    use_crepe: bool = True
    # Disabled by default. _run_omnizart's model_path="pop" was a real bug
    # (MusicTranscription._load_model looks the string up as an exact dict
    # key in checkpoint_path, whose real keys are capitalized - 'Pop', not
    # 'pop') that made every single call fail instantly with a
    # FileNotFoundError, caught silently and logged only at warning level -
    # so Omnizart contributed zero votes all session despite the checkpoint
    # being genuinely installed. That's now fixed (see _run_omnizart), but
    # bare-model testing with the fix applied - isolated 'other' stem, the
    # full 20s mix, and the entire untruncated 293s song - all produced
    # ZERO note events from the music_pop checkpoint. Its own feature
    # extraction falls back from Spleeter to plain librosa (Spleeter isn't
    # installed), which may be degrading it, or the checkpoint may simply
    # be poorly calibrated for this material - either way, with the fix
    # verified correct, it's still pure dead weight: ~2+ minutes of load +
    # inference per call for zero yield, which is exactly the kind of cost
    # that made pitch_detection_bass's timeout unreliable earlier this
    # session. Leave off until Spleeter is installed and re-tested, or a
    # checkpoint that actually transcribes this project's material is
    # found.
    use_omnizart: bool = False

    # Confidence thresholds
    spice_confidence_floor: float = 0.50
    basic_pitch_onset_threshold: float = 0.35
    basic_pitch_frame_threshold: float = 0.20
    librosa_threshold: float = 0.10
    crepe_confidence_floor: float = 0.75

    # Basic Pitch model path
    basic_pitch_model: Optional[str] = None

    # Consensus
    agreement_bonus: float = 0.15
    spice_basic_pitch_lock_conf: float = 0.99
    agreement_window_ms: float = 50.0
    agreement_semitones: int = 1

    # Constraint map
    energy_threshold: float = 0.02
    min_active_window_ms: float = 100.0

    # Thread pool
    max_workers: int = 4

    # Note filtering
    min_note_duration_ms: float = 30.0
    min_frequency_hz: float = 65.4
    max_frequency_hz: float = 4186.0

    # Timeout per model (seconds)
    model_timeout_seconds: float = 30.0


# =====================================================================
# Individual Model Runners
# =====================================================================

def _run_librosa_cqt(
        audio: np.ndarray,
        sr: int,
        config: CouncilConfig,
) -> Tuple[List[NoteEvent], Optional[np.ndarray]]:
    """Librosa CQT Accountant."""
    try:
        import librosa

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        harmonic, _ = librosa.effects.hpss(audio)

        cqt = np.abs(librosa.cqt(
            harmonic, sr=sr,
            fmin=librosa.note_to_hz('C2'),
            n_bins=84,
            bins_per_octave=12,
        ))

        pitches, magnitudes = librosa.piptrack(
            y=harmonic, sr=sr,
            threshold=config.librosa_threshold,
            hop_length=256,
        )

        events: List[NoteEvent] = []
        time_per_frame = 256 / sr * 1000.0
        global_max = np.max(magnitudes) + 1e-8

        note_start = None
        note_pitch = None
        note_frames = 0
        note_conf_sum = 0.0
        note_freq = 0.0

        for i in range(pitches.shape[1]):
            frame_mags = magnitudes[:, i]
            idx = np.argmax(frame_mags)
            freq = pitches[idx, i]

            if freq > 0:
                pitch = int(round(12 * np.log2(freq / 440.0) + 69))
                conf = float(frame_mags[idx] / global_max)
                t = i * time_per_frame

                if note_start is None:
                    note_start, note_pitch, note_frames = t, pitch, 1
                    note_conf_sum, note_freq = conf, freq
                elif abs(pitch - note_pitch) <= 1:
                    note_frames += 1
                    note_conf_sum += conf
                else:
                    if note_frames >= 2:
                        dur = t - note_start
                        if dur >= config.min_note_duration_ms:
                            avg_conf = note_conf_sum / note_frames
                            events.append(NoteEvent(
                                pitch=note_pitch,
                                start_ms=note_start,
                                end_ms=t,
                                velocity=int(min(127, avg_conf * 80 + 30)),
                                confidence=avg_conf,
                                zero_crossing_rate=0.0,
                                source=SourceType.PITCH,
                                fundamental_freq_hz=note_freq,
                                reasoning_chain=["librosa_cqt_accountant"],
                            ))
                    note_start, note_pitch, note_frames = t, pitch, 1
                    note_conf_sum, note_freq = conf, freq
            else:
                if note_start is not None and note_frames >= 2:
                    t = i * time_per_frame
                    dur = t - note_start
                    if dur >= config.min_note_duration_ms:
                        avg_conf = note_conf_sum / note_frames
                        events.append(NoteEvent(
                            pitch=note_pitch,
                            start_ms=note_start,
                            end_ms=t,
                            velocity=int(min(127, avg_conf * 80 + 30)),
                            confidence=avg_conf,
                            zero_crossing_rate=0.0,
                            source=SourceType.PITCH,
                            fundamental_freq_hz=note_freq,
                            reasoning_chain=["librosa_cqt_accountant"],
                        ))
                note_start = None
                note_pitch = None
                note_frames = 0
                note_conf_sum = 0.0
                note_freq = 0.0

        logger.debug(f"[Librosa] {len(events)} notes")
        return events, cqt

    except Exception as e:
        logger.warning(f"[Librosa] Failed: {e}")
        return [], None


_SPICE_MODEL = None
_SPICE_MODEL_LOCK = __import__("threading").Lock()
_SPICE_SR = 16000

# Official Google SPICE output->Hz conversion (from the model's own
# published colab/model card) - the model's "pitch" output is NOT a
# frequency, it's a normalized position on a fixed log-frequency axis
# and must be de-normalized with these exact constants before it means
# anything in Hz.
_SPICE_PT_OFFSET = 25.58
_SPICE_PT_SLOPE = 63.07
_SPICE_FMIN_HZ = 10.0
_SPICE_BINS_PER_OCTAVE = 12.0


def _get_spice_model():
    """
    Lazily loads the real Google SPICE model (self-supervised pitch
    estimation, https://tfhub.dev/google/spice/2) exactly once per
    process and reuses it. There is no pip package literally called
    "spice" that matches the `from spice import SPICE, SPICEConfig` API
    this file used to assume - that import always raised ImportError,
    silently contributing zero votes to the council forever. The real
    model is distributed as a TF-Hub SavedModel; `tensorflow_hub` is
    already a project dependency (confirmed installed, v0.16.1) and
    loading this model directly was verified to work end-to-end this
    session (~12s load, real pitch/uncertainty outputs).
    """
    global _SPICE_MODEL
    if _SPICE_MODEL is None:
        with _SPICE_MODEL_LOCK:
            if _SPICE_MODEL is None:
                import tensorflow_hub as hub
                _SPICE_MODEL = hub.load("https://tfhub.dev/google/spice/2")
    return _SPICE_MODEL


def _spice_pitch_to_hz(pitch_output: np.ndarray) -> np.ndarray:
    cqt_bin = pitch_output * _SPICE_PT_SLOPE + _SPICE_PT_OFFSET
    return _SPICE_FMIN_HZ * (2.0 ** (cqt_bin / _SPICE_BINS_PER_OCTAVE))


_BASIC_PITCH_MODEL = None
_BASIC_PITCH_MODEL_LOCK = __import__("threading").Lock()


def _get_basic_pitch_model(model_path_override: Optional[str]):
    """
    Lazily loads Basic Pitch's Model exactly once per process and reuses
    it. predict()'s model_or_model_path param accepts either a loaded
    Model or a path/str - passing a path (the previous behavior here)
    makes it reload and rebuild the model from disk on EVERY single
    call, which is most of what made analyze_bass_sync's two full
    council passes take ~466s for just 15s of audio in this session's
    verification run.
    """
    global _BASIC_PITCH_MODEL
    if _BASIC_PITCH_MODEL is None:
        with _BASIC_PITCH_MODEL_LOCK:
            if _BASIC_PITCH_MODEL is None:
                from basic_pitch.inference import Model
                from basic_pitch import ICASSP_2022_MODEL_PATH
                _BASIC_PITCH_MODEL = Model(model_path_override or ICASSP_2022_MODEL_PATH)
    return _BASIC_PITCH_MODEL


def _run_spice(
        audio: np.ndarray,
        sr: int,
        config: CouncilConfig,
) -> List[NoteEvent]:
    """
    SPICE Melodic Thread - real model, monophonic pitch tracking.

    SPICE is self-supervised and monophonic by design (it tracks ONE
    predominant pitch contour per input, like CREPE) - appropriate here
    because this witness's real job is corroborating/refining the
    melodic line other witnesses already proposed, not independently
    finding polyphony (Basic Pitch/Omnizart own that job).
    """
    try:
        import tensorflow as tf
        import librosa as _lib

        model = _get_spice_model()

        audio = audio.astype(np.float32)
        if sr != _SPICE_SR:
            audio = _lib.resample(audio, orig_sr=sr, target_sr=_SPICE_SR)

        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak

        if len(audio) < _SPICE_SR // 10:
            return []

        output = model.signatures["serving_default"](tf.constant(audio, tf.float32))
        pitch_output = output["pitch"].numpy()
        uncertainty = output["uncertainty"].numpy()
        confidence = 1.0 - uncertainty

        if len(pitch_output) == 0:
            return []

        freq_hz = _spice_pitch_to_hz(pitch_output)
        # SPICE's internal hop isn't part of its public output - frames
        # are evenly spaced across the input duration by construction
        # (fixed-hop analysis), so dividing total duration by frame
        # count recovers the real per-frame timestamp exactly.
        duration_ms = len(audio) / _SPICE_SR * 1000.0
        time_per_frame_ms = duration_ms / len(pitch_output)

        events: List[NoteEvent] = []
        note_start = None
        note_pitch = None
        note_frames = 0
        note_conf_sum = 0.0
        note_freq = 0.0

        for i in range(len(pitch_output)):
            t = i * time_per_frame_ms
            c = float(confidence[i])
            f = float(freq_hz[i])

            if c >= config.spice_confidence_floor and config.min_frequency_hz <= f <= config.max_frequency_hz:
                pitch = int(round(12.0 * np.log2(f / 440.0) + 69))

                if note_start is None:
                    note_start, note_pitch, note_frames = t, pitch, 1
                    note_conf_sum, note_freq = c, f
                elif abs(pitch - note_pitch) <= 1:
                    note_frames += 1
                    note_conf_sum += c
                else:
                    if note_frames >= 2:
                        dur = t - note_start
                        if dur >= config.min_note_duration_ms:
                            avg_conf = note_conf_sum / note_frames
                            events.append(NoteEvent(
                                pitch=note_pitch, start_ms=note_start, end_ms=t,
                                velocity=int(min(127, avg_conf * 100)),
                                confidence=avg_conf, zero_crossing_rate=0.0,
                                source=SourceType.PITCH, fundamental_freq_hz=note_freq,
                                reasoning_chain=["spice_melodic_thread"],
                            ))
                    note_start, note_pitch, note_frames = t, pitch, 1
                    note_conf_sum, note_freq = c, f
            else:
                if note_start is not None and note_frames >= 2:
                    dur = t - note_start
                    if dur >= config.min_note_duration_ms:
                        avg_conf = note_conf_sum / note_frames
                        events.append(NoteEvent(
                            pitch=note_pitch, start_ms=note_start, end_ms=t,
                            velocity=int(min(127, avg_conf * 100)),
                            confidence=avg_conf, zero_crossing_rate=0.0,
                            source=SourceType.PITCH, fundamental_freq_hz=note_freq,
                            reasoning_chain=["spice_melodic_thread"],
                        ))
                note_start, note_pitch, note_frames, note_conf_sum, note_freq = None, None, 0, 0.0, 0.0

        if note_start is not None and note_frames >= 2:
            dur = duration_ms - note_start
            if dur >= config.min_note_duration_ms:
                avg_conf = note_conf_sum / note_frames
                events.append(NoteEvent(
                    pitch=note_pitch, start_ms=note_start, end_ms=duration_ms,
                    velocity=int(min(127, avg_conf * 100)),
                    confidence=avg_conf, zero_crossing_rate=0.0,
                    source=SourceType.PITCH, fundamental_freq_hz=note_freq,
                    reasoning_chain=["spice_melodic_thread_final"],
                ))

        logger.debug(f"[SPICE] {len(events)} melody notes")
        return events

    except ImportError:
        logger.debug("[SPICE] tensorflow_hub not installed — skipping")
        return []
    except Exception as e:
        logger.warning(f"[SPICE] Failed: {e}")
        return []


def _run_basic_pitch(
        audio: np.ndarray,
        sr: int,
        config: CouncilConfig,
) -> List[NoteEvent]:
    """
    Basic Pitch Architect.

    basic_pitch.inference.predict()'s real signature is
    predict(audio_path: Union[Path, str], model_or_model_path=..., ...) -
    its first parameter loads audio FROM A FILE, it does not accept a raw
    numpy array at all, and its SECOND positional slot is already
    model_or_model_path. Calling predict(audio, sr, model_or_model_path=...)
    put the sample rate into that same slot positionally while also
    naming it as a keyword, which is exactly why this failed on every
    single call with "got multiple values for argument
    'model_or_model_path'" - Basic Pitch contributed zero votes to the
    ensemble the entire time, regardless of audio content. Fixed using
    the same temp-WAV-file pattern _run_omnizart already uses below for
    the same reason (a file-based model, not an array-based one).
    """
    try:
        import soundfile as sf
        from basic_pitch.inference import predict

        model = _get_basic_pitch_model(config.basic_pitch_model)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, audio, sr)

        try:
            model_output, midi_data, note_events = predict(
                temp_path,
                model_or_model_path=model,
                onset_threshold=config.basic_pitch_onset_threshold,
                frame_threshold=config.basic_pitch_frame_threshold,
                minimum_note_length=int(config.min_note_duration_ms),
                minimum_frequency=config.min_frequency_hz,
                maximum_frequency=config.max_frequency_hz,
            )
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        events: List[NoteEvent] = []
        for note in note_events:
            start_s, end_s, pitch_midi, amplitude, _ = note
            start_ms = start_s * 1000.0
            end_ms = end_s * 1000.0

            events.append(NoteEvent(
                pitch=int(pitch_midi),
                start_ms=start_ms,
                end_ms=end_ms,
                velocity=int(min(127, amplitude * 100)),
                confidence=float(amplitude),
                zero_crossing_rate=0.0,
                source=SourceType.PITCH,
                reasoning_chain=["basic_pitch_architect"],
            ))

        logger.debug(f"[Basic Pitch] {len(events)} notes")
        return events

    except ImportError:
        logger.debug("[Basic Pitch] Not installed — skipping")
        return []
    except Exception as e:
        logger.warning(f"[Basic Pitch] Failed: {e}")
        return []


def _run_crepe_refinement(
        audio: np.ndarray,
        sr: int,
        candidates: List[NoteEvent],
        config: CouncilConfig,
) -> List[NoteEvent]:
    """CREPE Forensic Specialist - refines pitch of high-confidence notes."""
    if not candidates:
        return []

    try:
        import crepe

        refined: List[NoteEvent] = []

        for note in candidates:
            start_sample = int(note.start_ms * sr / 1000.0)
            end_sample = int(note.end_ms * sr / 1000.0)

            if end_sample <= start_sample or end_sample > len(audio):
                refined.append(note)
                continue

            segment = audio[start_sample:end_sample]
            if len(segment) < 512:
                refined.append(note)
                continue

            try:
                time_arr, frequency, confidence, _ = crepe.predict(
                    segment, sr,
                    viterbi=True,
                    model_capacity="small",
                    step_size=10,
                )

                high_conf_mask = confidence > config.crepe_confidence_floor
                if np.any(high_conf_mask):
                    median_freq = float(np.median(frequency[high_conf_mask]))
                    if median_freq > 0:
                        refined_pitch = int(round(
                            12.0 * np.log2(median_freq / 440.0) + 69
                        ))
                        avg_crepe_conf = float(np.mean(confidence[high_conf_mask]))

                        note.reasoning_chain = list(note.reasoning_chain or [])
                        note.reasoning_chain.append(
                            f"crepe_refined:{note.pitch}→{refined_pitch} "
                            f"({median_freq:.1f}Hz, conf={avg_crepe_conf:.2f})"
                        )
                        note.pitch = refined_pitch
                        note.confidence = min(1.0, note.confidence + 0.05)

                refined.append(note)

            except Exception:
                refined.append(note)

        logger.debug(f"[CREPE] Refined {len(refined)} notes")
        return refined

    except ImportError:
        logger.debug("[CREPE] Not installed — returning candidates unrefined")
        return candidates
    except Exception as e:
        logger.warning(f"[CREPE] Failed: {e}")
        return candidates


def _run_omnizart(
        audio: np.ndarray,
        sr: int,
        config: CouncilConfig,
) -> List[NoteEvent]:
    """Omnizart Harmonic Logician - uses temp file for file-based models."""
    try:
        import soundfile as sf
        from omnizart.music import app as omnizart_music

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        sf.write(temp_path, audio, sr)

        try:
            # Omnizart's own default (no model_path given) is "music_piano",
            # tuned for solo piano - the wrong instrument prior for the
            # general pop/hip-hop material this project actually
            # transcribes. "Pop" selects the music_pop checkpoint instead -
            # MusicTranscription._load_model looks this string up as an
            # exact dict key in self.settings.checkpoint_path, whose real
            # keys are capitalized ('Piano', 'Pop', 'Stream', 'PianoV2');
            # the lowercase "pop" used here previously never matched,
            # silently fell through to being treated as a literal
            # filesystem path, and made Omnizart contribute zero votes on
            # every single call all session - caught by the bare except
            # Exception below and logged only at warning level, so it
            # never surfaced in any "success" pipeline run.
            midi = omnizart_music.transcribe(temp_path, model_path="Pop")
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        events: List[NoteEvent] = []
        if midi:
            for instrument in midi.instruments:
                for note in instrument.notes:
                    start_ms = note.start * 1000.0
                    end_ms = note.end * 1000.0

                    events.append(NoteEvent(
                        pitch=note.pitch,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        velocity=note.velocity,
                        confidence=0.75,
                        zero_crossing_rate=0.0,
                        source=SourceType.PITCH,
                        reasoning_chain=[
                            "omnizart_harmonic_logician",
                            f"instrument:{instrument.name}",
                        ],
                    ))

        logger.debug(f"[Omnizart] {len(events)} notes")
        return events

    except ImportError:
        logger.debug("[Omnizart] Not installed — skipping")
        return []
    except Exception as e:
        logger.warning(f"[Omnizart] Failed: {e}")
        return []


# =====================================================================
# The Epistemic Council
# =====================================================================

class EpistemicCouncil:
    """
    The Council of Witnesses for polyphonic pitch detection.

    Provides both async analyze() and sync analyze_sync() methods.
    """

    def __init__(self, config: Optional[CouncilConfig] = None):
        self.config = config or CouncilConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="epistemic_council",
        )

    # =================================================================
    # ASYNC METHOD (for asyncio contexts)
    # =================================================================

    async def analyze(
            self,
            audio: np.ndarray,
            sr: int,
            findings: Optional[MusicalFindingsMap] = None,
    ) -> Tuple[List[NoteEvent], MusicalFindingsMap]:
        """Async version - run council with asyncio concurrency."""
        if findings is None:
            findings = MusicalFindingsMap()

        loop = asyncio.get_event_loop()
        config = self.config

        def run_in_thread(fn, *args):
            return loop.run_in_executor(self._executor, fn, *args)

        if audio is None or len(audio) == 0:
            logger.warning("[Council] Empty audio input")
            return [], findings

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Pre-flight proxy
        try:
            import librosa as _lib
            if sr != config.proxy_sample_rate:
                proxy = _lib.resample(audio, orig_sr=sr, target_sr=config.proxy_sample_rate)
                proxy_sr = config.proxy_sample_rate
            else:
                proxy = audio
                proxy_sr = sr
        except Exception:
            proxy = audio
            proxy_sr = sr

        # Stage 1: Librosa + SPICE
        logger.debug("[Council] Stage 1: Librosa + SPICE...")
        t1_start = time.time()

        tasks_s1 = []
        if config.use_librosa:
            tasks_s1.append(run_in_thread(_run_librosa_cqt, proxy, proxy_sr, config))
        if config.use_spice:
            tasks_s1.append(run_in_thread(_run_spice, proxy, proxy_sr, config))

        results_s1 = await asyncio.gather(*tasks_s1, return_exceptions=True)

        librosa_notes, cqt_heatmap = [], None
        spice_notes = []

        result_idx = 0
        if config.use_librosa:
            r = results_s1[result_idx]
            result_idx += 1
            if not isinstance(r, Exception):
                librosa_notes, cqt_heatmap = r

        if config.use_spice:
            r = results_s1[result_idx]
            result_idx += 1
            if not isinstance(r, Exception):
                spice_notes = r

        if cqt_heatmap is not None:
            findings.librosa_cqt_heatmap = cqt_heatmap

        # Stage 2: Basic Pitch + Omnizart
        logger.debug("[Council] Stage 2: Basic Pitch + Omnizart...")
        t2_start = time.time()

        tasks_s2 = []
        if config.use_basic_pitch:
            tasks_s2.append(run_in_thread(_run_basic_pitch, audio, sr, config))
        if config.use_omnizart:
            tasks_s2.append(run_in_thread(_run_omnizart, audio, sr, config))

        results_s2 = await asyncio.gather(*tasks_s2, return_exceptions=True)

        basic_pitch_notes = []
        omnizart_notes = []

        result_idx = 0
        if config.use_basic_pitch:
            r = results_s2[result_idx]
            result_idx += 1
            if not isinstance(r, Exception):
                basic_pitch_notes = r

        if config.use_omnizart:
            r = results_s2[result_idx]
            result_idx += 1
            if not isinstance(r, Exception):
                omnizart_notes = r

        # Stage 3: Merge witnesses
        all_notes = []
        all_notes.extend([(n, "librosa") for n in librosa_notes])
        all_notes.extend([(n, "spice") for n in spice_notes])
        all_notes.extend([(n, "basic_pitch") for n in basic_pitch_notes])
        all_notes.extend([(n, "omnizart") for n in omnizart_notes])

        if not all_notes:
            return [], findings

        all_notes.sort(key=lambda x: x[0].start_ms)
        groups: List[List[Tuple[NoteEvent, str]]] = []

        for note, source in all_notes:
            placed = False
            for group in groups:
                rep_note, _ = group[0]
                time_close = abs(note.start_ms - rep_note.start_ms) <= config.agreement_window_ms
                pitch_close = abs(note.pitch - rep_note.pitch) <= config.agreement_semitones

                if time_close and pitch_close:
                    group.append((note, source))
                    placed = True
                    break

            if not placed:
                groups.append([(note, source)])

        consensus_notes: List[NoteEvent] = []

        for group in groups:
            sources = [src for _, src in group]
            notes = [n for n, _ in group]
            n_sources = len(set(sources))

            base = max(notes, key=lambda n: n.confidence)

            has_spice = any(s == "spice" for s in sources)
            has_bp = any(s == "basic_pitch" for s in sources)

            if has_spice and has_bp:
                final_conf = config.spice_basic_pitch_lock_conf
                reasoning = list(base.reasoning_chain or []) + ["council:SPICE+BP_LOCKED"]
            elif n_sources >= 2:
                final_conf = min(1.0, base.confidence + config.agreement_bonus)
                reasoning = list(base.reasoning_chain or []) + [
                    f"council:{n_sources}_witnesses_{','.join(set(sources))}"
                ]
            else:
                final_conf = base.confidence
                reasoning = list(base.reasoning_chain or []) + [
                    f"council:single_{sources[0]}"
                ]

            median_pitch = int(round(float(np.median([n.pitch for n in notes]))))
            weights = [n.confidence for n in notes]
            total_w = sum(weights) + 1e-8
            start_ms = sum(n.start_ms * w for n, w in zip(notes, weights)) / total_w
            end_ms = max(n.end_ms for n in notes)
            velocity = max(n.velocity for n in notes)

            # This used to always construct a fresh NoteEvent with
            # fundamental_freq_hz left at its None default and
            # zero_crossing_rate hardcoded to 0.0, discarding real evidence
            # several constituent witnesses (SPICE, CREPE) DO compute -
            # even though this consensus note is what everything downstream
            # actually sees. SchoenbergMirror's harmonic-series corroboration
            # only runs `if note.fundamental_freq_hz`, so losing it here
            # silently forced every council-derived note through ZCR-only
            # classification with no harmonic evidence to counter a
            # percussive-sounding ATTACK transient - which real piano/
            # plucked-string/mallet attacks commonly have despite being
            # clearly tonal instruments. Prefer base's own value; fall back
            # to any other constituent vote in the group that has one.
            fundamental_freq_hz = base.fundamental_freq_hz
            if not fundamental_freq_hz:
                for n in notes:
                    if n.fundamental_freq_hz:
                        fundamental_freq_hz = n.fundamental_freq_hz
                        break

            # Basic Pitch and Omnizart - likely the dominant contributors
            # for polyphonic "other"-stem content - never set
            # fundamental_freq_hz on their own votes at all (confirmed by
            # direct source read), so a group made up entirely of their
            # votes still has nothing real to fall back to above. A
            # standard equal-temperament estimate from the note's own
            # detected pitch is still real, useful evidence for
            # SchoenbergMirror's partial-alignment test - far better than
            # leaving it None and forcing every such note through ZCR-only
            # classification with no harmonic corroboration available at all.
            if not fundamental_freq_hz:
                fundamental_freq_hz = 440.0 * (2.0 ** ((median_pitch - 69) / 12.0))

            consensus_notes.append(NoteEvent(
                pitch=median_pitch,
                start_ms=start_ms,
                end_ms=end_ms,
                velocity=velocity,
                confidence=final_conf,
                zero_crossing_rate=base.zero_crossing_rate,
                source=SourceType.PITCH,
                reasoning_chain=reasoning,
                fundamental_freq_hz=fundamental_freq_hz,
            ))

        consensus_notes = [n for n in consensus_notes if n.confidence >= 0.15]
        consensus_notes.sort(key=lambda n: n.start_ms)

        # Stage 4: CREPE refinement
        if config.use_crepe:
            high_conf_candidates = [n for n in consensus_notes if n.confidence >= config.crepe_confidence_floor]
            if high_conf_candidates:
                logger.debug(f"[Council] Stage 4: CREPE refining {len(high_conf_candidates)} candidates...")
                t4_start = time.time()
                refined = await run_in_thread(
                    _run_crepe_refinement, audio, sr, high_conf_candidates, config
                )
                if refined:
                    low_conf = [n for n in consensus_notes if n.confidence < config.crepe_confidence_floor]
                    consensus_notes = low_conf + refined
                    consensus_notes.sort(key=lambda n: n.start_ms)

        findings.record_stage("epistemic_council")
        return consensus_notes, findings

    # =================================================================
    # SYNC METHOD (for pipeline threads - CRITICAL FIX)
    # =================================================================

    def analyze_sync(
            self,
            audio: np.ndarray,
            sr: int,
            findings: Optional[MusicalFindingsMap] = None,
    ) -> Tuple[List[NoteEvent], MusicalFindingsMap]:
        """
        Synchronous version of analyze() - runs the council without asyncio.

        This is the recommended method for use inside pipeline threads
        where asyncio event loops may not be available.

        Args:
            audio:    Mono float32 audio array
            sr:       Sample rate (16000 recommended)
            findings: Existing MusicalFindingsMap to update (or new one)

        Returns:
            (consensus_notes, updated_findings)
        """
        if findings is None:
            findings = MusicalFindingsMap()

        if audio is None or len(audio) == 0:
            logger.warning("[Council] Empty audio input")
            return [], findings

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Pre-flight proxy
        try:
            import librosa as _lib
            if sr != self.config.proxy_sample_rate:
                proxy = _lib.resample(audio, orig_sr=sr, target_sr=self.config.proxy_sample_rate)
                proxy_sr = self.config.proxy_sample_rate
            else:
                proxy = audio
                proxy_sr = sr
        except Exception:
            proxy = audio
            proxy_sr = sr

        # Stage 1: Librosa + SPICE (sequential)
        logger.debug("[Council] Stage 1: Librosa + SPICE...")
        t1_start = time.time()

        librosa_notes, cqt_heatmap = [], None
        spice_notes = []

        if self.config.use_librosa:
            try:
                librosa_notes, cqt_heatmap = _run_librosa_cqt(proxy, proxy_sr, self.config)
            except Exception as e:
                logger.warning(f"[Council] Librosa failed: {e}")

        if self.config.use_spice:
            try:
                spice_notes = _run_spice(proxy, proxy_sr, self.config)
            except Exception as e:
                logger.warning(f"[Council] SPICE failed: {e}")

        if cqt_heatmap is not None:
            findings.librosa_cqt_heatmap = cqt_heatmap

        # Stage 2: Basic Pitch + Omnizart (sequential)
        logger.debug("[Council] Stage 2: Basic Pitch + Omnizart...")
        t2_start = time.time()

        basic_pitch_notes = []
        omnizart_notes = []

        if self.config.use_basic_pitch:
            try:
                basic_pitch_notes = _run_basic_pitch(audio, sr, self.config)
            except Exception as e:
                logger.warning(f"[Council] Basic Pitch failed: {e}")

        if self.config.use_omnizart:
            try:
                omnizart_notes = _run_omnizart(audio, sr, self.config)
            except Exception as e:
                logger.warning(f"[Council] Omnizart failed: {e}")

        # Stage 3: Merge witnesses
        all_notes = []
        all_notes.extend([(n, "librosa") for n in librosa_notes])
        all_notes.extend([(n, "spice") for n in spice_notes])
        all_notes.extend([(n, "basic_pitch") for n in basic_pitch_notes])
        all_notes.extend([(n, "omnizart") for n in omnizart_notes])

        if not all_notes:
            return [], findings

        all_notes.sort(key=lambda x: x[0].start_ms)
        groups: List[List[Tuple[NoteEvent, str]]] = []

        for note, source in all_notes:
            placed = False
            for group in groups:
                rep_note, _ = group[0]
                time_close = abs(note.start_ms - rep_note.start_ms) <= self.config.agreement_window_ms
                pitch_close = abs(note.pitch - rep_note.pitch) <= self.config.agreement_semitones

                if time_close and pitch_close:
                    group.append((note, source))
                    placed = True
                    break

            if not placed:
                groups.append([(note, source)])

        consensus_notes: List[NoteEvent] = []

        for group in groups:
            sources = [src for _, src in group]
            notes = [n for n, _ in group]
            n_sources = len(set(sources))

            base = max(notes, key=lambda n: n.confidence)

            has_spice = any(s == "spice" for s in sources)
            has_bp = any(s == "basic_pitch" for s in sources)

            if has_spice and has_bp:
                final_conf = self.config.spice_basic_pitch_lock_conf
                reasoning = list(base.reasoning_chain or []) + ["council:SPICE+BP_LOCKED"]
            elif n_sources >= 2:
                final_conf = min(1.0, base.confidence + self.config.agreement_bonus)
                reasoning = list(base.reasoning_chain or []) + [
                    f"council:{n_sources}_witnesses_{','.join(set(sources))}"
                ]
            else:
                final_conf = base.confidence
                reasoning = list(base.reasoning_chain or []) + [
                    f"council:single_{sources[0]}"
                ]

            median_pitch = int(round(float(np.median([n.pitch for n in notes]))))
            weights = [n.confidence for n in notes]
            total_w = sum(weights) + 1e-8
            start_ms = sum(n.start_ms * w for n, w in zip(notes, weights)) / total_w
            end_ms = max(n.end_ms for n in notes)
            velocity = max(n.velocity for n in notes)

            # This used to always construct a fresh NoteEvent with
            # fundamental_freq_hz left at its None default and
            # zero_crossing_rate hardcoded to 0.0, discarding real evidence
            # several constituent witnesses (SPICE, CREPE) DO compute -
            # even though this consensus note is what everything downstream
            # actually sees. SchoenbergMirror's harmonic-series corroboration
            # only runs `if note.fundamental_freq_hz`, so losing it here
            # silently forced every council-derived note through ZCR-only
            # classification with no harmonic evidence to counter a
            # percussive-sounding ATTACK transient - which real piano/
            # plucked-string/mallet attacks commonly have despite being
            # clearly tonal instruments. Prefer base's own value; fall back
            # to any other constituent vote in the group that has one.
            fundamental_freq_hz = base.fundamental_freq_hz
            if not fundamental_freq_hz:
                for n in notes:
                    if n.fundamental_freq_hz:
                        fundamental_freq_hz = n.fundamental_freq_hz
                        break

            # Basic Pitch and Omnizart - likely the dominant contributors
            # for polyphonic "other"-stem content - never set
            # fundamental_freq_hz on their own votes at all (confirmed by
            # direct source read), so a group made up entirely of their
            # votes still has nothing real to fall back to above. A
            # standard equal-temperament estimate from the note's own
            # detected pitch is still real, useful evidence for
            # SchoenbergMirror's partial-alignment test - far better than
            # leaving it None and forcing every such note through ZCR-only
            # classification with no harmonic corroboration available at all.
            if not fundamental_freq_hz:
                fundamental_freq_hz = 440.0 * (2.0 ** ((median_pitch - 69) / 12.0))

            consensus_notes.append(NoteEvent(
                pitch=median_pitch,
                start_ms=start_ms,
                end_ms=end_ms,
                velocity=velocity,
                confidence=final_conf,
                zero_crossing_rate=base.zero_crossing_rate,
                source=SourceType.PITCH,
                reasoning_chain=reasoning,
                fundamental_freq_hz=fundamental_freq_hz,
            ))

        consensus_notes = [n for n in consensus_notes if n.confidence >= 0.15]
        consensus_notes.sort(key=lambda n: n.start_ms)

        # Stage 4: CREPE refinement
        if self.config.use_crepe:
            high_conf_candidates = [n for n in consensus_notes if n.confidence >= self.config.crepe_confidence_floor]
            if high_conf_candidates:
                logger.debug(f"[Council] Stage 4: CREPE refining {len(high_conf_candidates)} candidates...")
                t4_start = time.time()
                refined = _run_crepe_refinement(audio, sr, high_conf_candidates, self.config)
                if refined:
                    low_conf = [n for n in consensus_notes if n.confidence < self.config.crepe_confidence_floor]
                    consensus_notes = low_conf + refined
                    consensus_notes.sort(key=lambda n: n.start_ms)

        logger.debug(f"[Council] Final: {len(consensus_notes)} consensus notes")
        findings.record_stage("epistemic_council")
        return consensus_notes, findings

    # =================================================================
    # BASS OCTAVE-SHIFT PASS
    # =================================================================
    #
    # A bass fundamental sitting at, say, E1 (~41Hz) is right at the
    # noisy, ambiguous low end of what most pitch trackers were tuned
    # and validated on - weak/absent fundamental energy relative to
    # overtones (very common on electric bass through a cabinet/DI) makes
    # it easy for a tracker to lock onto a partial or a sub-harmonic
    # instead of the true pitch. Shifting the audio up an octave before
    # analysis moves the same fundamental into a range every witness here
    # is more reliable in, without changing the actual musical content -
    # then the result is shifted back down. Run BOTH the direct-pitch
    # pass and the shift-and-correct pass and treat them as two
    # independent measurements of the same physical pitch: agreement is
    # much stronger evidence than either alone, and disagreement is a
    # real, visible flag (not silently discarded) that the direct pass
    # may have caught a wrong octave/partial.

    def analyze_bass_sync(
            self,
            audio: np.ndarray,
            sr: int,
            findings: Optional[MusicalFindingsMap] = None,
    ) -> Tuple[List[NoteEvent], MusicalFindingsMap]:
        """
        Bass-specific variant of analyze_sync(): runs the full council
        twice (direct pitch, and pitch-shifted-up-an-octave-then-
        corrected-back-down) and reconciles the two into one note list.
        """
        if findings is None:
            findings = MusicalFindingsMap()

        if audio is None or len(audio) == 0:
            logger.warning("[Council/Bass] Empty audio input")
            return [], findings

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        direct_notes, findings = self.analyze_sync(audio, sr, findings)
        for n in direct_notes:
            n.reasoning_chain = list(n.reasoning_chain or []) + ["bass:direct"]

        try:
            import librosa as _lib
            shifted_audio = _lib.effects.pitch_shift(audio, sr=sr, n_steps=12)
        except Exception as e:
            logger.warning(f"[Council/Bass] Octave-shift failed, direct-only: {e}")
            return direct_notes, findings

        shifted_notes, findings = self.analyze_sync(shifted_audio, sr, findings)

        corrected_notes: List[NoteEvent] = []
        for n in shifted_notes:
            n.pitch = n.pitch - 12
            if n.fundamental_freq_hz:
                n.fundamental_freq_hz = n.fundamental_freq_hz / 2.0
            n.reasoning_chain = list(n.reasoning_chain or []) + ["bass:octave_shift_corrected"]
            corrected_notes.append(n)

        merged = self._merge_bass_passes(direct_notes, corrected_notes)
        logger.debug(
            f"[Council/Bass] direct={len(direct_notes)} shifted={len(corrected_notes)} "
            f"merged={len(merged)}")
        return merged, findings

    def _merge_bass_passes(
            self,
            direct_notes: List[NoteEvent],
            shifted_notes: List[NoteEvent],
    ) -> List[NoteEvent]:
        """
        Groups direct/shifted notes by time+pitch proximity (same
        agreement_window_ms/agreement_semitones the main council uses)
        and reconciles each group:
          - both passes agree (same group) -> confidence boosted, this
            is the strongest possible corroboration for a bass note.
          - only the shift-corrected pass found it -> kept, but at
            reduced confidence and explicitly tagged, since it's an
            indirect measurement the direct pass either missed or
            rejected as ambiguous - a real disagreement, not hidden.
          - only the direct pass found it -> kept as-is; the octave
            trick doesn't invalidate a note it simply didn't re-detect.
        """
        tagged = [(n, "direct") for n in direct_notes] + [(n, "shifted") for n in shifted_notes]
        tagged.sort(key=lambda x: x[0].start_ms)

        groups: List[List[Tuple[NoteEvent, str]]] = []
        for note, src in tagged:
            placed = False
            for group in groups:
                rep_note, _ = group[0]
                time_close = abs(note.start_ms - rep_note.start_ms) <= self.config.agreement_window_ms
                pitch_close = abs(note.pitch - rep_note.pitch) <= self.config.agreement_semitones
                if time_close and pitch_close:
                    group.append((note, src))
                    placed = True
                    break
            if not placed:
                groups.append([(note, src)])

        merged: List[NoteEvent] = []
        for group in groups:
            sources = {src for _, src in group}
            notes = [n for n, _ in group]
            base = max(notes, key=lambda n: n.confidence)

            if len(sources) == 2:
                base.confidence = min(1.0, base.confidence + self.config.agreement_bonus)
                base.reasoning_chain = list(base.reasoning_chain or []) + ["bass:direct+shift_agree"]
            elif "shifted" in sources:
                base.confidence = base.confidence * 0.85
                base.reasoning_chain = list(base.reasoning_chain or []) + ["bass:shift_only_no_direct_corroboration"]

            merged.append(base)

        merged.sort(key=lambda n: n.start_ms)
        return merged

    # =================================================================
    # DURATION CONTENTION ARBITRATION
    # =================================================================
    #
    # QuaverIntelligence testifies with competing duration hypotheses per
    # note, but until now nothing ever arbitrated the genuinely contested
    # ones (has_contention=True, i.e. its top two candidates land within
    # 0.15 probability of each other) - the pipeline's own duration-
    # consensus step deliberately skips those as too risky, and Quaver
    # itself only "testifies", it doesn't decide. That's exactly the gap
    # an epistemic council is for: reconcile disagreement using evidence
    # that isn't already baked into either candidate's own probability.

    @staticmethod
    def resolve_duration_contention(
            testimony: Any,
            pulse_field: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Break a tie between a note's top two competing duration
        hypotheses using two signals neither hypothesis's own
        probability already accounts for:

        - epistemic_tension: how internally confident a hypothesis is
          (a low support-vs-opposition balance means it was a close
          call even on its OWN terms, independent of how its
          probability compares to its rival's).
        - end-of-note grid alignment: does the position this hypothesis
          implies the note actually ENDS at land near a real beat-grid
          line, rather than an arbitrary spot? Every other grid-
          alignment check in this pipeline only ever looks at a note's
          START; nothing evaluates where a candidate duration would
          leave it ending.

        Only meaningful when Quaver itself flagged real contention -
        for anything else, primary_hypothesis already stands on its own
        and this just returns it unchanged.
        """
        if not testimony.duration_hypotheses:
            return None

        hyps = sorted(testimony.duration_hypotheses, key=lambda h: -h.probability)
        if len(hyps) < 2 or not testimony.has_contention:
            return hyps[0]

        top, runner_up = hyps[0], hyps[1]
        grid = (pulse_field or {}).get("beat_grid_ms") or []

        def score(hyp: Any) -> float:
            tension_score = 1.0 - hyp.epistemic_tension
            implied_end_ms = testimony.start_ms + hyp.duration_ms
            end_alignment = EpistemicCouncil._grid_alignment_score(implied_end_ms, grid)
            return hyp.probability * 0.5 + tension_score * 0.3 + end_alignment * 0.2

        return max([top, runner_up], key=score)

    @staticmethod
    def resolve_drum_type_contention(votes: List[Any]) -> Optional[Tuple[Any, float]]:
        """
        Break a tie between competing single-hit drum-type
        interpretations (e.g. the centroid classifier says "ride", the
        NMF decomposer says "snare" for the same onset) using a signal
        neither vote's own confidence already accounts for:
        cross-source corroboration - how much OTHER votes (not just the
        top one) agree with a given type, weighted by their own
        confidence*weight. A lone high-confidence vote from one source
        is less trustworthy than two independent sources converging on
        the same answer.

        This is for CONTENTION only: one physical hit, disagreement
        about what it is. It is deliberately NOT used to resolve genuine
        polyphony (two different simultaneous hits) - forcing a single
        winner there would destroy the very thing multi-source
        decomposition exists to recover, so that case is handled
        structurally by letting both survive rather than by arbitration.

        Each vote is duck-typed: needs `.drum_type`, `.confidence`, and
        optionally `.weight` (defaults to 1.0).
        """
        if not votes:
            return None
        if len(votes) == 1:
            v = votes[0]
            return v.drum_type, v.confidence

        type_support: Dict[Any, float] = {}
        for v in votes:
            w = getattr(v, 'weight', 1.0)
            type_support[v.drum_type] = type_support.get(v.drum_type, 0.0) + v.confidence * w

        total_support = sum(type_support.values())
        if total_support <= 0:
            best = max(votes, key=lambda v: v.confidence)
            return best.drum_type, best.confidence

        def score(drum_type: Any) -> float:
            corroboration = type_support[drum_type] / total_support
            best_single = max(
                (v.confidence * getattr(v, 'weight', 1.0) for v in votes if v.drum_type == drum_type),
                default=0.0)
            return corroboration * 0.6 + min(1.0, best_single) * 0.4

        best_type = max(type_support.keys(), key=score)
        total_weight = sum(getattr(v, 'weight', 1.0) for v in votes)
        final_confidence = min(0.95, type_support[best_type] / max(1.0, total_weight))
        return best_type, final_confidence

    @staticmethod
    def resolve_tempo_contention(
            votes: List[Tuple[float, float, float, str]],
            music_box: Optional[Any] = None,
    ) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        """
        Arbitrate between independent tempo witnesses, the same way this
        council already arbitrates drum-type and note-duration disputes -
        the one place TempoIntelligence's blend, ReverseGeoCrypt's
        lattice tactus, and RhythmEngine's drum-stem estimate all submit
        to, instead of each being consumed directly by whatever happens
        to read them (see ARCHITECTURE_CONSOLIDATION_PLAN.md §3.3/§4.1:
        the actual gap was never a missing consensus mechanism, it was
        tempo witnesses not being routed through the one that exists).

        The reconciliation math itself is anchor_and_align_tempo_witnesses
        (agents/analysis/tempo_intelligence.py) - anchor on the most
        trusted witness, rescale octave/ratio-related witnesses into its
        octave, exclude unrelated ones - deliberately delegated rather
        than duplicated here, since a second implementation of the same
        arbitration would recreate the exact redundancy problem this
        method exists to close.

        What this adds on top of the raw math: disagreement is treated as
        a first-class, logged event. A witness that doesn't relate to the
        anchor by any musical ratio isn't just silently excluded - the
        exclusion is logged to music_box as its own decision record, and
        the returned confidence already reflects it (the excluded
        witness's weight counts against total confidence by construction).

        Args:
            votes: (tempo_bpm, confidence, weight, source_name) per witness.
            music_box: optional forensic logger for disagreement records.

        Returns:
            (resolved_tempo_bpm, confidence, details) or None if no votes.
            details includes per-witness alignment outcomes for callers
            that want to report/log them.
        """
        if not votes:
            return None

        from agents.analysis.tempo_intelligence import (
            anchor_and_align_tempo_witnesses, TempoOctaveCorrector,
        )

        resolved_tempo, confidence, anchor_tempo = anchor_and_align_tempo_witnesses(
            [(t, c, w) for t, c, w, _ in votes]
        )

        # Re-derive which witnesses aligned vs. were excluded, for the
        # disagreement record (the alignment function returns only the
        # blend; the per-witness outcome is what makes a disagreement
        # visible instead of silent).
        outcomes = []
        for tempo_bpm, conf, weight, source in votes:
            if tempo_bpm <= 0 or anchor_tempo <= 0:
                outcomes.append({"source": source, "tempo_bpm": tempo_bpm,
                                 "confidence": conf, "aligned": False})
                continue
            ratio = tempo_bpm / anchor_tempo
            aligned = any(
                abs(ratio - (n / d)) / (n / d) < 0.08
                for n, d, _, _ in TempoOctaveCorrector.RATIOS
            )
            outcomes.append({"source": source, "tempo_bpm": tempo_bpm,
                             "confidence": conf, "aligned": aligned})

        excluded = [o for o in outcomes if not o["aligned"]]
        details = {"anchor_tempo_bpm": anchor_tempo, "outcomes": outcomes,
                   "excluded_count": len(excluded)}

        if music_box is not None:
            try:
                music_box.log_decision(
                    stage_name="epistemic_council",
                    decision_type="tempo_contention",
                    before_state={"votes": [
                        {"source": s, "tempo_bpm": t, "confidence": c, "weight": w}
                        for t, c, w, s in votes]},
                    after_state={"resolved_tempo_bpm": resolved_tempo,
                                 "confidence": confidence,
                                 "anchor_tempo_bpm": anchor_tempo,
                                 "excluded_witnesses": [o["source"] for o in excluded]},
                    reasoning=(f"Tempo witnesses reconciled to {resolved_tempo:.1f} BPM"
                               + (f"; DISAGREEMENT: {len(excluded)} witness(es) unrelated "
                                  f"to anchor by any musical ratio "
                                  f"({', '.join(o['source'] for o in excluded)})"
                                  if excluded else "; all witnesses ratio-consistent")),
                    reversible=True,
                )
            except Exception:
                pass  # forensic logging must never break arbitration

        return resolved_tempo, confidence, details

    @staticmethod
    def _grid_alignment_score(time_ms: float, grid_ms: List[float]) -> float:
        """How close time_ms lands to the nearest beat-grid line (0=far, 1=exact)."""
        if not grid_ms:
            return 0.5
        grid_arr = np.array(grid_ms)
        nearest_dist = float(np.min(np.abs(grid_arr - time_ms)))
        spacing = float(np.median(np.diff(grid_arr))) if len(grid_arr) > 1 else 500.0
        half_spacing = spacing / 2.0 if spacing > 0 else 1.0
        return max(0.0, 1.0 - min(1.0, nearest_dist / half_spacing))

    def close(self):
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=False)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =====================================================================
# Convenience Functions
# =====================================================================

def create_epistemic_council(config: Optional[CouncilConfig] = None) -> EpistemicCouncil:
    """Create an EpistemicCouncil instance."""
    return EpistemicCouncil(config=config)


def quick_test_council(audio_path: str = None):
    """Quick test for Epistemic Council."""
    import librosa

    print("=" * 60)
    print("Epistemic Council Test")
    print("=" * 60)

    if audio_path:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    else:
        duration = 2.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print(f"Audio: {len(audio) / sr:.1f}s at {sr}Hz")

    config = CouncilConfig(
        use_librosa=True,
        use_spice=False,
        use_basic_pitch=True,
        use_crepe=False,
        use_omnizart=False,
    )
    council = create_epistemic_council(config)

    # Test sync method (what pipeline uses)
    notes, findings = council.analyze_sync(audio, sr)

    print(f"\nResults: {len(notes)} consensus notes")
    for i, note in enumerate(notes[:10]):
        print(f"  Note {i}: pitch={note.pitch}, start={note.start_ms:.0f}ms, "
              f"conf={note.confidence:.2f}")

    council.close()
    return notes, findings


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        quick_test_council(sys.argv[1])
    else:
        quick_test_council()