# =================================================================
# MODULE: core/duration_check.py
# The duration gate - ported from Grimlock 6.0 (Jazz), 2026-07-21.
#
# THE PRINCIPLE (the user's, and the cheapest objective ground truth this
# project has): if a song is 3:43, its MIDI transcription should play back
# for 3:43. When the exported MIDI's total playback time matches the source
# audio's duration, the whole timeline is correct end to end - tempo, tempo
# map, and note placement all agree. When it doesn't, something in the
# export is broken, and no ear is needed to know it.
#
# This is a PURE VERIFIER - it reads a finished .mid and its source audio
# and returns a verdict. It touches no pipeline, mutates nothing, and can
# be run on any export, current or historical. Applied to Symphony's own
# Output/ it instantly flagged three stale exports at 0.66x-0.88x (tempo
# written too fast / truncated) that had shipped silently.
#
# WHAT IT CATCHES and WHAT IT DOES NOT: a wrong tempo map, a truncated
# render, a flat-tempo/tick mismatch, a note run that stops early - all
# make the ratio miss 1.0. It does NOT by itself prove the tempo NUMBER is
# right (a wrong tempo with a compensating map can still total the right
# duration), so pair it with "does a notation host read back the tempo we
# wrote" for the full check. Necessary, not sufficient - but it is the one
# test that would have caught every timeline regression in Output/.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Within this band of the true duration, the timeline is trustworthy. A
# few percent covers trailing silence (the last note usually ends a beat
# or two before the audio does) without letting a real error through.
DURATION_PASS_LO = 0.95
DURATION_PASS_HI = 1.05


@dataclass(frozen=True)
class DurationCheck:
    midi_seconds: float
    audio_seconds: float
    ratio: float
    passed: bool
    verdict: str            # "PASS" | "MIDI_SHORT" | "MIDI_LONG"

    def as_line(self) -> str:
        def ds(s: float) -> str:
            return f"{int(s // 60)}:{int(s % 60):02d}"
        return (f"duration: audio {ds(self.audio_seconds)}  midi {ds(self.midi_seconds)}  "
                f"ratio {self.ratio:.3f}x  {self.verdict}")


def check_export_duration(midi_path: str, audio_path: str,
                          truncate_seconds: float = 0.0) -> DurationCheck:
    """Compares a rendered MIDI's playback length to its source audio's
    length. Honors the MIDI's own tempo map (pretty_midi.get_end_time).

    `truncate_seconds`: when the pipeline processed only the first N seconds
    (truncate_duration), the MIDI is legitimately that long, not the full
    file's length - compare against the PROCESSED span so a deliberate
    truncation is not mis-flagged as a timeline bug. 0 means full file."""
    import pretty_midi
    import soundfile as sf

    midi_seconds = float(pretty_midi.PrettyMIDI(midi_path).get_end_time())
    file_seconds = float(sf.info(audio_path).duration)
    audio_seconds = (min(file_seconds, truncate_seconds)
                     if truncate_seconds and truncate_seconds > 0 else file_seconds)
    ratio = midi_seconds / audio_seconds if audio_seconds > 0 else 0.0

    if DURATION_PASS_LO <= ratio <= DURATION_PASS_HI:
        verdict = "PASS"
    elif ratio < DURATION_PASS_LO:
        verdict = "MIDI_SHORT"
    else:
        verdict = "MIDI_LONG"

    return DurationCheck(
        midi_seconds=midi_seconds, audio_seconds=audio_seconds,
        ratio=ratio, passed=(verdict == "PASS"), verdict=verdict,
    )


def assert_export_duration(midi_path: str, audio_path: str,
                           logger: Optional[object] = None) -> DurationCheck:
    """Same check, but SURFACES a failure instead of returning it silently -
    the whole point of a gate (the inherited reflex this project fights is
    'emit a confident-looking output and fail silently'). Logs a warning
    on failure if a logger is given; never raises into the pipeline."""
    result = check_export_duration(midi_path, audio_path)
    if not result.passed and logger is not None:
        try:
            logger.warning(
                "EXPORT DURATION GATE FAILED: %s - the MIDI timeline does not "
                "match the source audio (likely a tempo-map or truncation bug).",
                result.as_line(),
            )
        except Exception:
            pass
    return result


__all__ = ["DurationCheck", "check_export_duration", "assert_export_duration"]
