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
from typing import List, Optional, Tuple

import numpy as np

_BASIC_PITCH_MODEL = None
_BASIC_PITCH_MODEL_LOCK = threading.Lock()


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
