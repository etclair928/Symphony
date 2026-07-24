# =================================================================
# MODULE: core/basic_pitch_utils.py
# VERSION: 5.6.1
# DESCRIPTION: Shared Basic Pitch model loading + inference call,
# extracted from epistemic/epistemic_council.py so every caller (the
# pitch council, and now agents/quantization/ritornello.py's
# CorrectionScribe) shares one implementation instead of drifting
# apart on "how to call Basic Pitch correctly" - a real bug already
# had to be fixed here once (predict()'s actual signature loads audio
# from a file path, not a raw array; a second copy of this logic could
# silently regress the same way).
# =================================================================

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# scipy >= 1.13 removed the top-level scipy.signal.gaussian re-export (the
# function only ever lived in scipy.signal.windows). Basic Pitch 0.3.0's
# note_creation.get_pitch_bends still calls scipy.signal.gaussian, so on
# modern scipy EVERY predict() raises "module 'scipy.signal' has no attribute
# 'gaussian'" and Basic Pitch - the primary detector - silently contributes
# nothing. We cannot downgrade scipy (audio-separator and madmom both pin
# scipy>=1.13), so restore the alias here; it is a pure re-export, so this is
# behaviour-identical to the version Basic Pitch was written against.
try:  # pragma: no cover - trivial compat shim
    import scipy.signal as _sps
    if not hasattr(_sps, "gaussian"):
        from scipy.signal.windows import gaussian as _gaussian
        _sps.gaussian = _gaussian
except Exception:
    pass

_BASIC_PITCH_MODEL = None
_BASIC_PITCH_MODEL_LOCK = threading.Lock()

# Basic Pitch's "note" posteriorgram is a fixed 88-bin grid (one bin per
# semitone) starting at A0 - basic_pitch.constants.ANNOTATIONS_BASE_FREQUENCY
# (27.5Hz = MIDI 21) / ANNOTATIONS_N_SEMITONES (88). Confirmed directly
# against the installed package rather than assumed: bin = midi_pitch - 21
# is an exact, non-interpolated mapping (no fractional bins).
BASIC_PITCH_POSTERIORGRAM_BASE_MIDI_PITCH = 21
BASIC_PITCH_POSTERIORGRAM_N_BINS = 88


@dataclass(frozen=True)
class BasicPitchPosteriorgram:
    """
    The model's own per-pitch, per-frame "is this note sounding" evidence
    - frame_note[i, pitch_midi - 21] is Basic Pitch's belief (0-1) that
    pitch_midi is actively sounding at frame i. This is the same evidence
    the model already used internally to decide where notes begin and
    end; consulting it again for "is this note still ringing" reuses the
    model's own judgment instead of a separate, cruder energy heuristic.
    """
    frame_note: np.ndarray       # (n_frames, 88) float32, values in [0, 1]
    frame_hop_ms: float          # ~11.6ms at Basic Pitch's internal 22050Hz/256-hop

    def level_at(self, pitch_midi: int, time_ms: float) -> float:
        """Posterior for pitch_midi at time_ms, or 0.0 if out of range."""
        bin_idx = pitch_midi - BASIC_PITCH_POSTERIORGRAM_BASE_MIDI_PITCH
        if not (0 <= bin_idx < self.frame_note.shape[1]):
            return 0.0
        frame_idx = int(time_ms / self.frame_hop_ms)
        if not (0 <= frame_idx < self.frame_note.shape[0]):
            return 0.0
        return float(self.frame_note[frame_idx, bin_idx])

    def duration_ms(self) -> float:
        return self.frame_note.shape[0] * self.frame_hop_ms


def get_basic_pitch_model(model_path_override: Optional[str] = None):
    """
    Lazily loads Basic Pitch's Model exactly once per process and reuses
    it. predict()'s model_or_model_path param accepts either a loaded
    Model or a path/str - passing a path makes it reload and rebuild the
    model from disk on EVERY call, which is what made repeated Basic
    Pitch calls expensive before this caching was added.
    """
    global _BASIC_PITCH_MODEL
    if _BASIC_PITCH_MODEL is None:
        with _BASIC_PITCH_MODEL_LOCK:
            if _BASIC_PITCH_MODEL is None:
                from basic_pitch.inference import Model
                from basic_pitch import ICASSP_2022_MODEL_PATH
                _BASIC_PITCH_MODEL = Model(model_path_override or ICASSP_2022_MODEL_PATH)
    return _BASIC_PITCH_MODEL


def run_basic_pitch_on_slice(
        audio: np.ndarray,
        sr: int,
        onset_threshold: float,
        frame_threshold: float,
        min_note_duration_ms: float,
        min_frequency_hz: float,
        max_frequency_hz: float,
        model_path_override: Optional[str] = None,
) -> List[Tuple[float, float, int, float]]:
    """
    Runs Basic Pitch on an arbitrary audio slice, returning raw
    (start_s, end_s, pitch_midi, amplitude) tuples - NOT NoteEvents.
    Callers build their own NoteEvent with their own source/
    reasoning_chain (the council tags "basic_pitch_architect";
    CorrectionScribe tags "correction_scribe:basic_pitch").

    predict()'s real signature loads audio FROM A FILE
    (predict(audio_path, model_or_model_path=..., ...)) - it does not
    accept a raw numpy array - so this always goes through a temp WAV,
    same as the pattern this was extracted from.
    """
    import soundfile as sf
    from basic_pitch.inference import predict

    model = get_basic_pitch_model(model_path_override)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    sf.write(temp_path, audio, sr)

    try:
        _model_output, _midi_data, note_events = predict(
            temp_path,
            model_or_model_path=model,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=int(min_note_duration_ms),
            minimum_frequency=min_frequency_hz,
            maximum_frequency=max_frequency_hz,
        )
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    return [
        (float(start_s), float(end_s), int(pitch_midi), float(amplitude))
        for start_s, end_s, pitch_midi, amplitude, _pitch_bend in note_events
    ]


def run_basic_pitch_with_posteriorgram(
        audio: np.ndarray,
        sr: int,
        onset_threshold: float,
        frame_threshold: float,
        min_note_duration_ms: float,
        min_frequency_hz: float,
        max_frequency_hz: float,
        model_path_override: Optional[str] = None,
) -> Tuple[List[Tuple[float, float, int, float]], BasicPitchPosteriorgram]:
    """
    Same call as run_basic_pitch_on_slice(), but also returns the model's
    raw frame-level note posteriorgram - previously discarded as
    `_model_output` at every call site. Kept as a separate function
    rather than changing run_basic_pitch_on_slice()'s return shape, so
    existing callers (CorrectionScribe, the council's normal per-note
    detection) are unaffected; only the caller that actually wants the
    posteriorgram (Ritornello's SustainRecovery) uses this one.
    """
    import soundfile as sf
    from basic_pitch.inference import predict

    model = get_basic_pitch_model(model_path_override)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    sf.write(temp_path, audio, sr)

    try:
        model_output, _midi_data, note_events = predict(
            temp_path,
            model_or_model_path=model,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=int(min_note_duration_ms),
            minimum_frequency=min_frequency_hz,
            maximum_frequency=max_frequency_hz,
        )
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    notes = [
        (float(start_s), float(end_s), int(pitch_midi), float(amplitude))
        for start_s, end_s, pitch_midi, amplitude, _pitch_bend in note_events
    ]
    frame_note = np.asarray(model_output["note"], dtype=np.float32)
    frame_hop_ms = (len(audio) / sr * 1000.0) / max(1, frame_note.shape[0])
    posteriorgram = BasicPitchPosteriorgram(frame_note=frame_note, frame_hop_ms=frame_hop_ms)
    return notes, posteriorgram
