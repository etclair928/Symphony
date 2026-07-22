"""Regression test for export/musicxml_writer.py.

Guards the invariant that was learned the corrupted-file way (Jazz 6.0,
ported into Symphony): whatever polyphony the exporter is handed, the
MusicXML it writes must OPEN - voice numbers dense from 1 (never 0), at
most 4 voices per staff, and re-parseable by music21.

Runnable two ways:
  python test_musicxml_writer.py      # prints PASS / raises on failure
  pytest test_musicxml_writer.py      # picks up the test_* functions
"""
import os
import tempfile
import xml.etree.ElementTree as ET

from core.order_types import NoteEvent, SourceType
from export.musicxml_writer import export_musicxml, MAX_VOICES_PER_STAFF


def _mk(pitch, start_ms, end_ms, family, vel=80):
    # Instrument identity lives in reasoning_chain tags (voice_family:),
    # the real pipeline path - notes carry no .instrument_family field.
    return NoteEvent(
        pitch=pitch, start_ms=float(start_ms), end_ms=float(end_ms),
        velocity=vel, confidence=0.9, zero_crossing_rate=0.1,
        source=SourceType.PITCH, reasoning_chain=[f"voice_family:{family}"],
    )


def _dense_polyphony_events():
    """A chord struck together, a staggered melody over it (forces a 2nd
    voice), and a separate-family bass line (forces a 2nd part)."""
    ev = []
    # three-note chord struck together -> ONE chord, one voice
    ev += [_mk(60, 0, 500, "piano"), _mk(64, 5, 500, "piano"), _mk(67, 10, 500, "piano")]
    # melody staggered over the chord -> a 2nd voice on the piano staff
    ev += [_mk(72, 100, 300, "piano"), _mk(74, 320, 520, "piano"), _mk(76, 540, 900, "piano")]
    # bass, a different family -> its own part, bass clef
    ev += [_mk(40, 0, 480, "bass"), _mk(43, 500, 980, "bass"), _mk(38, 1000, 1480, "bass")]
    return ev


def test_exports_valid_openable_multipart_score():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "regression.musicxml")
        ok = export_musicxml(_dense_polyphony_events(), out,
                             tempo_bpm=145.0, time_signature=(4, 4), key="F minor")
        assert ok, "export_musicxml returned False on valid input"
        assert os.path.exists(out), "no MusicXML file written"

        root = ET.parse(out).getroot()
        parts = root.findall(".//part")
        assert len(parts) >= 2, f"expected >=2 parts (piano + bass), got {len(parts)}"

        # The corruption trap: every <voice> must be >= 1, and no staff may
        # carry more than MAX_VOICES_PER_STAFF voices.
        for part in parts:
            nums = [int(v.text) for v in part.iter("voice")]
            if nums:
                assert min(nums) >= 1, "voice number < 1 (MuseScore would reject the file)"
                assert max(nums) <= MAX_VOICES_PER_STAFF, \
                    f"more than {MAX_VOICES_PER_STAFF} voices on one staff (invalid)"

        # It must actually re-open.
        from music21 import converter
        sc = converter.parse(out)
        assert len(sc.parts) >= 2, "music21 re-parse lost parts"
        assert len(list(sc.recurse().notes)) > 0, "music21 re-parse found no notes"


def test_readable_mode_reduces_confetti_without_losing_notes():
    """Readable mode (fill short gaps + eighth grid) must turn a line of
    short, truncated notes into eighths-or-coarser without adding or dropping
    a single note - the whole point of the legibility policy."""
    from export.musicxml_writer import build_music21_score

    # A 145 BPM line: a note every eighth (~207ms) but each only ~150ms long
    # (Basic-Pitch-style truncation) -> every note is sub-eighth in raw mode.
    events, t = [], 0.0
    for i in range(16):
        events.append(NoteEvent(
            pitch=60 + (i % 5), start_ms=t, end_ms=t + 150.0, velocity=80,
            confidence=0.9, zero_crossing_rate=0.1, source=SourceType.PITCH,
            reasoning_chain=["voice_family:piano"]))
        t += 206.9

    def sub_eighth(score):
        return sum(1 for n in score.recurse().notes if n.quarterLength < 0.5)

    raw = build_music21_score(events, 145.0, (4, 4), "C", duration_policy="raw")
    readable = build_music21_score(events, 145.0, (4, 4), "C",
                                   duration_policy="fill", grid_ql=0.5)
    n_raw = len(list(raw.recurse().notes))
    n_readable = len(list(readable.recurse().notes))
    assert n_readable == n_raw, "readable mode changed the note count"
    assert sub_eighth(readable) < sub_eighth(raw), \
        "readable mode did not reduce sub-eighth 'confetti'"


def test_all_drums_returns_false_not_crash():
    """Drums (channel 9) are not engraved; an all-drum input must fail
    softly (return False), never raise into the pipeline."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "drums_only.musicxml")
        drum = NoteEvent(pitch=38, start_ms=0.0, end_ms=100.0, velocity=90,
                         confidence=0.9, zero_crossing_rate=0.1,
                         source=SourceType.DRUM_INTELLIGENCE, midi_channel=9)
        ok = export_musicxml([drum], out, tempo_bpm=120.0)
        assert ok is False, "all-drum input should return False, not succeed"
        assert not os.path.exists(out), "no file should be written for all-drum input"


if __name__ == "__main__":
    test_exports_valid_openable_multipart_score()
    test_readable_mode_reduces_confetti_without_losing_notes()
    test_all_drums_returns_false_not_crash()
    print("PASS")
