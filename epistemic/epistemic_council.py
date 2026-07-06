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


def _run_spice(
        audio: np.ndarray,
        sr: int,
        config: CouncilConfig,
) -> List[NoteEvent]:
    """SPICE Melodic Thread."""
    try:
        from spice import SPICE, SPICEConfig

        spice_config = SPICEConfig(
            model_size="small",
            hop_length_ms=10,
        )
        model = SPICE(config=spice_config)
        model.load()

        result = model.predict(audio, sr)
        events: List[NoteEvent] = []

        if hasattr(result, 'note_events'):
            for note in result.note_events:
                if note.confidence >= config.spice_confidence_floor:
                    events.append(NoteEvent(
                        pitch=note.pitch_midi,
                        start_ms=note.start_ms,
                        end_ms=note.end_ms,
                        velocity=int(min(127, note.confidence * 100)),
                        confidence=note.confidence,
                        zero_crossing_rate=0.0,
                        source=SourceType.PITCH,
                        reasoning_chain=["spice_melodic_thread"],
                    ))

        logger.debug(f"[SPICE] {len(events)} melody notes")
        return events

    except ImportError:
        logger.debug("[SPICE] Not installed — skipping")
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
        from basic_pitch import ICASSP_2022_MODEL_PATH

        model_path = config.basic_pitch_model or ICASSP_2022_MODEL_PATH

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, audio, sr)

        try:
            model_output, midi_data, note_events = predict(
                temp_path,
                model_or_model_path=model_path,
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
            midi = omnizart_music.transcribe(temp_path)
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

            consensus_notes.append(NoteEvent(
                pitch=median_pitch,
                start_ms=start_ms,
                end_ms=end_ms,
                velocity=velocity,
                confidence=final_conf,
                zero_crossing_rate=0.0,
                source=SourceType.PITCH,
                reasoning_chain=reasoning,
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

            consensus_notes.append(NoteEvent(
                pitch=median_pitch,
                start_ms=start_ms,
                end_ms=end_ms,
                velocity=velocity,
                confidence=final_conf,
                zero_crossing_rate=0.0,
                source=SourceType.PITCH,
                reasoning_chain=reasoning,
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