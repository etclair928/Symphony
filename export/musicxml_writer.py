# =================================================================
# MODULE: export/musicxml_writer.py
# Serializes Symphony's transcribed NoteEvents to MusicXML via music21.
#
# WHY THIS EXISTS (the notation frontier, A.V "the MIDI ceiling"):
# Symphony's only symbolic outputs were MIDI and JSON. MIDI is a
# performance format - it cannot carry the things a *page* needs:
# independent voices as separate staves, tied notes across a barline,
# rests, clefs, a key signature. The guitar's two independent lines
# collapse into one polyphonic mush in MIDI. MusicXML is the notation
# format every engraver (MuseScore, Dorico, Finale, Sibelius) reads, so
# this is the first Symphony output that can actually become a score.
#
# This is a VIEW. It reads finished NoteEvents and their attached
# instrument_family and never mutates a note - the non-destructive-audit
# law holds by construction (it is read-only over the events).
#
# SINGLE SOURCE OF TRUTH: grouping into instruments and the General MIDI
# program per instrument are taken from export/midi_writer.py
# (InstrumentFamily + FAMILY_TO_PROGRAM), NOT re-invented here, so a
# MusicXML render and the MIDI render pick the same instruments and the
# same playback voices. If the MIDI routing changes, this follows.
#
# HARD VALIDITY CONSTRAINT (learned the corrupted-file way in Jazz 6.0,
# ported here): a family's notes are NOT monophonic - staggered polyphony
# and chords are normal. MusicXML voice numbers must start at 1 (never 0)
# and MuseScore represents at most 4 voices per staff, or it rejects the
# whole file as "corrupted." So this view guarantees validity
# structurally whatever it is fed: it splits each part into monophonic
# sub-voices itself, numbers them from 1, caps at MAX_VOICES_PER_STAFF,
# and spills overflow onto extra staves. A view may render an upstream
# mess uglily; it may never emit a file that will not open.
# =================================================================

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.order_types import NoteEvent, SourceType
from export.midi_writer import InstrumentFamily, FAMILY_TO_PROGRAM

# music21 quarterLength is in quarter notes; ms -> quarterLength needs the
# tempo (at T BPM one quarter note is 60000/T ms).
_MIN_QUARTER_LENGTH = 0.125     # a 32nd note - the finest a duration rounds to

# MuseScore (and most engravers) represent at most four voices per staff.
# More than this is invalid and the file will not open; overflow voices
# spill onto a new staff of the same instrument instead.
MAX_VOICES_PER_STAFF = 4

# Notes whose onsets fall within this window are treated as struck
# together - one chord in one voice - rather than exploded into separate
# voices. Keeps a strummed chord on one staff.
_CHORD_ONSET_WINDOW_MS = 30.0

# READABLE-MODE defaults. Jazz 6.0's MusicXML A/B experiments (recorded in
# the notation-frontier notes) showed that a VALID score is still an
# illegible one when it honors Basic Pitch's literal, often-truncated note
# durations on a fine grid ("confetti" of 16ths/32nds). Two decisions WE
# control fix the rhythm, independent of the format: (1) a coarser grid, and
# (2) a duration policy that holds a note through a short gap to the next
# onset (like a human transcriber) instead of writing a 16th + a 16th rest.
# The fill threshold is BEAT-RELATIVE (a musical unit derived from tempo),
# not an absolute magic millisecond - a gap shorter than this fraction of a
# beat reads as legato/detachment within one gesture; a longer gap is a real
# rest and is kept. These are the defaults ONLY for opt-in readable mode;
# the default export policy stays "raw" (literal durations, fine grid).
_READABLE_GRID_QL = 0.5          # eighth note
_FILL_MAX_GAP_BEATS = 1.0        # hold through gaps up to one beat

# Human-readable part names per family (the enum .value is lowercase).
_FAMILY_DISPLAY = {
    InstrumentFamily.PIANO: "Piano",
    InstrumentFamily.BASS: "Bass",
    InstrumentFamily.GUITAR: "Guitar",
    InstrumentFamily.STRINGS: "Strings",
    InstrumentFamily.BRASS: "Brass",
    InstrumentFamily.WOODWINDS: "Woodwinds",
    InstrumentFamily.SYNTH: "Synth",
    InstrumentFamily.VOCAL: "Vocals",
    InstrumentFamily.OTHER: "Other",
}


# The pipeline records instrument identity as reasoning_chain TAGS, not as
# a NoteEvent field: `voice_family:X` (the stable, per-voice consolidated
# vote - see pipeline._consolidate_voice_family) and the noisier per-note
# `family:X`. Those X values come from the *timbre* InstrumentFamily
# vocabulary (strings/plucked/voice/...), which is a different enum from the
# MIDI writer's routing families (a known B.5 type collision). This table
# translates the timbre vocabulary into the MIDI routing family so the
# score's parts line up with how the MIDI is voiced.
_TIMBRE_TO_FAMILY = {
    "strings": InstrumentFamily.STRINGS,
    "plucked": InstrumentFamily.GUITAR,
    "piano": InstrumentFamily.PIANO,
    "brass": InstrumentFamily.BRASS,
    "woodwind": InstrumentFamily.WOODWINDS,
    "woodwinds": InstrumentFamily.WOODWINDS,
    "voice": InstrumentFamily.VOCAL,
    "vocal": InstrumentFamily.VOCAL,
    "synth": InstrumentFamily.SYNTH,
    "percussion": InstrumentFamily.PERCUSSION,
    "drums": InstrumentFamily.DRUMS,
    "bass": InstrumentFamily.BASS,
    "guitar": InstrumentFamily.GUITAR,
    "unknown": InstrumentFamily.OTHER,
    "other": InstrumentFamily.OTHER,
}

# Source stem -> routing family, mirroring MidiWriter.SOURCE_TO_FAMILY, used
# only when a note carries no timbre tag at all.
_SOURCE_TO_FAMILY = {
    SourceType.BASS_STEM: InstrumentFamily.BASS,
    SourceType.BASS: InstrumentFamily.BASS,
    SourceType.DRUM_INTELLIGENCE: InstrumentFamily.DRUMS,
}


def _family_tag(event: NoteEvent) -> Optional[str]:
    """The consolidated voice_family tag if present, else the per-note family
    tag, read off reasoning_chain. Returns the raw timbre-family string."""
    chain = getattr(event, "reasoning_chain", None) or []
    voice_fam = None
    per_note = None
    for t in chain:
        if not isinstance(t, str):
            continue
        if t.startswith("voice_family:"):
            voice_fam = t.split(":", 1)[1].split(":")[0]  # ignore _contested suffix marker
        elif t.startswith("voice_family_contested:"):
            voice_fam = voice_fam or t.split(":", 1)[1]
        elif t.startswith("family:"):
            per_note = t.split(":", 1)[1]
    return voice_fam or per_note


def _resolve_family(event: NoteEvent) -> InstrumentFamily:
    """Group key for a note: the consolidated timbre family (normalized into
    the MIDI routing vocabulary), falling back to the stem source, then
    OTHER. Notes never carry an .instrument_family field in this pipeline -
    identity lives in reasoning_chain tags."""
    tag = _family_tag(event)
    if tag:
        fam = _TIMBRE_TO_FAMILY.get(tag.strip().lower())
        if fam is not None:
            return fam
    src_fam = _SOURCE_TO_FAMILY.get(getattr(event, "source", None))
    if src_fam is not None:
        return src_fam
    return InstrumentFamily.OTHER


def _is_drum(event: NoteEvent) -> bool:
    if getattr(event, "midi_channel", 0) == 9:
        return True
    if getattr(event, "source", None) == SourceType.DRUM_INTELLIGENCE:
        return True
    return _resolve_family(event) in (InstrumentFamily.DRUMS, InstrumentFamily.PERCUSSION)


def _quarter_length(duration_ms: float, tempo_bpm: float,
                    grid_ql: float = _MIN_QUARTER_LENGTH) -> float:
    ms_per_quarter = 60000.0 / max(tempo_bpm, 1.0)
    ql = duration_ms / ms_per_quarter
    snapped = round(ql / grid_ql) * grid_ql
    return max(grid_ql, snapped)


def _offset_quarter_length(start_ms: float, origin_ms: float, tempo_bpm: float,
                           grid_ql: float = _MIN_QUARTER_LENGTH) -> float:
    ms_per_quarter = 60000.0 / max(tempo_bpm, 1.0)
    ql = (start_ms - origin_ms) / ms_per_quarter
    return max(0.0, round(ql / grid_ql) * grid_ql)


def _clef_for(mean_pitch: float):
    from music21 import clef
    # Guitar/vocals read an octave high in treble (8vb); low parts get bass.
    if mean_pitch >= 60:
        return clef.TrebleClef()
    if mean_pitch >= 52:
        return clef.Treble8vbClef()
    return clef.BassClef()


def _m21_key(key_str: str):
    """Parse Symphony's key string ('F minor', 'Fm', 'C major', 'C#',
    'Bb minor', ...) into a music21 Key. Returns None if unparseable -
    a missing key signature is fine; a wrong crash is not."""
    from music21 import key as m21key
    if not key_str:
        return None
    s = key_str.strip()
    if not s:
        return None
    low = s.lower()
    mode = "minor" if ("min" in low or (low.endswith("m") and "maj" not in low)) else "major"
    tonic = s[0].upper()
    if len(s) > 1 and s[1] in ("#", "b", "♯", "♭"):
        tonic += "#" if s[1] in ("#", "♯") else "-"   # music21 flat is '-'
    try:
        return m21key.Key(tonic, mode)
    except Exception:
        return None


def _chord_events(notes: List[NoteEvent]) -> List[List[NoteEvent]]:
    """Group notes whose onsets fall within _CHORD_ONSET_WINDOW_MS into one
    event. Notes struck together are a chord, not separate voices - seven
    notes at once is ONE seven-note chord (correct, and keeps them on one
    staff instead of exploding into seven voices)."""
    events: List[List[NoteEvent]] = []
    cur: List[NoteEvent] = []
    cur_onset = 0.0
    for n in sorted(notes, key=lambda e: e.get_active_start_ms()):
        s = n.get_active_start_ms()
        if cur and s - cur_onset > _CHORD_ONSET_WINDOW_MS:
            events.append(cur)
            cur = []
        if not cur:
            cur_onset = s
        cur.append(n)
    if cur:
        events.append(cur)
    return events


def _events_to_voices(events: List[List[NoteEvent]]) -> List[List[List[NoteEvent]]]:
    """Assign chord-events to the fewest monophonic voices such that no two
    events in a voice overlap in time. Only genuinely staggered polyphony (a
    note beginning while another still sounds) forces a new voice.

    Rendering-only: this guarantees legal MusicXML, not musical truth.
    Meaningful voicing is the upstream data-association problem; this is the
    floor that keeps whatever we are handed openable."""
    voices: List[List[List[NoteEvent]]] = []
    free_at: List[float] = []
    for ev in sorted(events, key=lambda e: min(n.get_active_start_ms() for n in e)):
        start = min(n.get_active_start_ms() for n in ev)
        end = max(n.get_active_end_ms() for n in ev)
        placed = False
        for i, f in enumerate(free_at):
            if start >= f:
                voices[i].append(ev)
                free_at[i] = end
                placed = True
                break
        if not placed:
            voices.append([ev])
            free_at.append(end)
    return voices


def _staff_groups(voices: List) -> List[List]:
    """Chunk voices into staves of at most MAX_VOICES_PER_STAFF."""
    return [voices[i:i + MAX_VOICES_PER_STAFF]
            for i in range(0, len(voices), MAX_VOICES_PER_STAFF)]


def _normalize_voice_numbers(path: str) -> None:
    """Belt-and-suspenders: rewrite every part's <voice> numbers to a dense
    1..N per part. music21's own export numbering is not reliably 1-based
    (it has emitted voice 0, which MuseScore rejects), so make the invariant
    true in the bytes on disk regardless of what music21 chose."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    for part in root.findall(".//part"):
        seen: Dict[str, str] = {}
        for v in part.iter("voice"):
            old = v.text
            if old not in seen:
                seen[old] = str(len(seen) + 1)
            v.text = seen[old]
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def build_music21_score(
    events: List[NoteEvent],
    tempo_bpm: float,
    time_signature: Tuple[int, int] = (4, 4),
    key: Optional[str] = None,
    duration_policy: str = "raw",
    grid_ql: float = _MIN_QUARTER_LENGTH,
    fill_max_gap_beats: float = _FILL_MAX_GAP_BEATS,
):
    """Returns a music21.stream.Score. Each pitched instrument family becomes
    one or more staves; each staff carries at most MAX_VOICES_PER_STAFF
    voices, numbered from 1 - always valid, whatever the input polyphony.

    duration_policy:
      "raw"  - each note's own (SustainRecovery-aware) duration, snapped to
               grid_ql. Faithful to what was detected; can look like confetti.
      "fill" - hold a note through a gap shorter than fill_max_gap_beats to
               the next onset in its voice (legato, like a human transcriber),
               keeping the real duration + a rest on longer gaps. With a
               coarser grid this is what makes the PAGE readable (Jazz 6.0's
               A/B result), separate from the format itself.

    Drums/percussion (channel 9) are intentionally not engraved here: an
    unpitched kit belongs on a percussion staff with its own mapping, a
    separate problem. MIDI already carries the kit on channel 10."""
    from music21 import stream, note as m21note, chord as m21chord, tempo as m21tempo
    from music21 import meter as m21meter, instrument as m21instrument

    pitched = [e for e in events if not _is_drum(e)]
    if not pitched:
        raise ValueError("no pitched notes to engrave (all events were drums/empty)")

    # Group by instrument family - one part (possibly multi-staff) per family.
    by_family: Dict[InstrumentFamily, List[NoteEvent]] = {}
    for e in pitched:
        by_family.setdefault(_resolve_family(e), []).append(e)

    # Anchor the score at absolute t=0, NOT at the first note. The page's
    # timeline should match the audio's (and the MIDI's) - the project's
    # success metric is that a 3:43 song reads back as 3:43. A note that
    # sounds 2s in is 2s of leading rest on the page, which is correct;
    # collapsing to the first onset would silently shift everything earlier.
    origin_ms = 0.0
    m21_key = _m21_key(key) if key else None
    num, den = time_signature
    ms_per_beat = 60000.0 / max(tempo_bpm, 1.0)
    fill_gap_ms = fill_max_gap_beats * ms_per_beat

    def _event_bounds(event: List[NoteEvent]) -> Tuple[float, float]:
        return (min(n.get_active_start_ms() for n in event),
                max(n.get_active_end_ms() for n in event))

    def _make_element(event: List[NoteEvent], duration_ms: float):
        ql = _quarter_length(duration_ms, tempo_bpm, grid_ql)
        if len(event) == 1:
            el = m21note.Note(int(event[0].pitch))
        else:
            el = m21chord.Chord(sorted(int(n.pitch) for n in event))
        el.quarterLength = ql
        el.volume.velocity = max(int(n.velocity) for n in event)
        return el

    m_score = stream.Score()
    # Stable, readable ordering of instruments in the score.
    family_order = sorted(by_family.keys(), key=lambda f: f.value)
    for family in family_order:
        fam_notes = by_family[family]
        events_grouped = _chord_events(fam_notes)
        voices = _events_to_voices(events_grouped)
        staves = _staff_groups(voices)
        program = FAMILY_TO_PROGRAM.get(family)

        for staff_idx, staff_voices in enumerate(staves):
            m_part = stream.Part()
            base = _FAMILY_DISPLAY.get(family, family.value.title())
            m_part.partName = base if len(staves) == 1 else f"{base} [{staff_idx + 1}]"

            inst = m21instrument.Instrument()
            if program is not None:
                inst.midiProgram = int(program)
            m_part.insert(0.0, inst)
            m_part.insert(0.0, m21tempo.MetronomeMark(number=round(tempo_bpm)))
            m_part.insert(0.0, m21meter.TimeSignature(f"{num}/{den}"))
            if m21_key is not None:
                m_part.insert(0.0, m21_key)

            staff_mean = sum(n.pitch for v in staff_voices for ev in v for n in ev) / \
                sum(len(ev) for v in staff_voices for ev in v)
            m_part.insert(0.0, _clef_for(staff_mean))

            for v_num, voice_events in enumerate(staff_voices, start=1):
                m_voice = stream.Voice(id=str(v_num))
                # voice_events are monophonic and non-overlapping; sort by
                # onset so the fill policy can look ahead to the next note.
                ordered = sorted(voice_events, key=lambda e: _event_bounds(e)[0])
                for idx, event in enumerate(ordered):
                    onset_ms, real_end_ms = _event_bounds(event)
                    dur_ms = real_end_ms - onset_ms
                    if duration_policy == "fill" and idx + 1 < len(ordered):
                        next_onset = _event_bounds(ordered[idx + 1])[0]
                        gap = next_onset - real_end_ms
                        # Hold through a short gap (legato); a longer gap is a
                        # real rest and keeps the note's own duration.
                        if 0.0 < gap <= fill_gap_ms:
                            dur_ms = next_onset - onset_ms
                    offset = _offset_quarter_length(onset_ms, origin_ms,
                                                    tempo_bpm, grid_ql)
                    m_voice.insert(offset, _make_element(event, dur_ms))
                m_voice.makeRests(fillGaps=True, inPlace=True)
                m_part.insert(0.0, m_voice)

            m_part = m_part.makeNotation(inPlace=False)
            m_score.insert(0.0, m_part)

    return m_score


def export_musicxml(
    events: List[NoteEvent],
    output_path: str,
    tempo_bpm: float,
    time_signature: Tuple[int, int] = (4, 4),
    key: Optional[str] = None,
    readable: bool = False,
) -> bool:
    """Write `events` to `output_path` as MusicXML, guaranteed voice-legal
    (voices 1..N per part, <= MAX_VOICES_PER_STAFF per staff). Returns True
    on success. A view must never take the pipeline down: on any failure it
    returns False rather than raising.

    readable=False (default): faithful "raw" durations on a 32nd grid.
    readable=True: the legibility policy (fill short gaps + eighth grid) that
    turns a valid-but-confetti score into one a human can actually read."""
    try:
        if readable:
            m_score = build_music21_score(
                events, tempo_bpm, time_signature, key,
                duration_policy="fill", grid_ql=_READABLE_GRID_QL)
        else:
            m_score = build_music21_score(events, tempo_bpm, time_signature, key)
        m_score.write("musicxml", fp=output_path)
        _normalize_voice_numbers(output_path)
        return True
    except Exception:
        return False


__all__ = ["build_music21_score", "export_musicxml", "MAX_VOICES_PER_STAFF"]
