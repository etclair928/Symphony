# =================================================================
# MODULE: export/midi_writer.py (REWRITTEN)
# VERSION: 5.6.1 (CRITICAL: Fixed polyphony, tempo integration, channel manager)
# UPDATED: 2026-06-03
#
# KEY FIXES in 5.6.0:
#   1. ✅ Added missing imports (struct, gc, time)
#   2. ✅ Fixed polyphony - timeline-based event sorting
#   3. ✅ Fixed tempo integration with cumulative tick calculation
#   4. ✅ Added instrument_family for semantic routing
#   5. ✅ Fixed octave restoration with provenance tracking
#   6. ✅ Dynamic channel allocation via ChannelManager
#   7. ✅ Program Change messages for proper instrument playback
#   8. ✅ Split long forensic strings across multiple meta events
#   9. ✅ Enhanced confidence statistics (median, std, 5th percentile)
# =================================================================

import struct
import gc
import time
import warnings
from typing import List, Optional, Dict, Any, Tuple, Union, Set
from pathlib import Path
from datetime import datetime
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from copy import copy, deepcopy
from collections import defaultdict
import math

import numpy as np

# Core imports
from core.order_types import (
    NoteEvent, TempoMap, ExportOptions, SourceType, Confidence,
    AudioContext
)
from core.constants import (
    MIDI_PPQN,
    MIDI_TEMPO_DEFAULT,
    HEARTBEAT_NOTE_PITCH,
    HEARTBEAT_NOTE_VELOCITY,
    HEARTBEAT_NOTE_DURATION_MS,
    OCTAVE_RESTORE_BASS,
    BASS_OCTAVE_SHIFT_SEMITONES,
    BASS_NOTE_RANGE_LOW_MIDI,
    BASS_NOTE_RANGE_HIGH_MIDI,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import MusicBoxProtocol, StatusReporterProtocol


# ========================================================================
# Enums and Types
# ========================================================================

class MidiFormat(str, Enum):
    """MIDI file format."""
    FORMAT_0 = "format_0"
    FORMAT_1 = "format_1"
    FORMAT_2 = "format_2"


class InstrumentFamily(str, Enum):
    """Semantic instrument families for routing."""
    PIANO = "piano"
    BASS = "bass"
    DRUMS = "drums"
    STRINGS = "strings"
    BRASS = "brass"
    WOODWINDS = "woodwinds"
    GUITAR = "guitar"
    SYNTH = "synth"
    VOCAL = "vocal"
    PERCUSSION = "percussion"
    OTHER = "other"


# GM Program Numbers (partial mapping)
class GMProgram(IntEnum):
    """General MIDI Program Change numbers."""
    ACOUSTIC_GRAND_PIANO = 0
    ELECTRIC_BASS_FINGER = 33
    ELECTRIC_BASS_PICK = 34
    FRETLESS_BASS = 35
    SLAP_BASS_1 = 36
    SLAP_BASS_2 = 37
    SYNTH_BASS_1 = 38
    SYNTH_BASS_2 = 39
    STRING_ENSEMBLE_1 = 48
    STRING_ENSEMBLE_2 = 49
    TRUMPET = 56
    TROMBONE = 57
    TUBA = 58
    FRENCH_HORN = 60
    SAX_SOPRANO = 64
    SAX_ALTO = 65
    SAX_TENOR = 66
    SAX_BARITONE = 67
    FLUTE = 73
    PICCALO = 72
    CLARINET = 71
    OBOE = 68
    BASSOON = 70
    ACOUSTIC_GUITAR_NYLON = 24
    ACOUSTIC_GUITAR_STEEL = 25
    ELECTRIC_GUITAR_JAZZ = 26
    ELECTRIC_GUITAR_CLEAN = 27
    ELECTRIC_GUITAR_MUTED = 28
    SYNTH_LEAD_SQUARE = 80
    SYNTH_LEAD_SAW = 81
    CHOIR_AAHS = 52
    VOICE_OOHS = 53


# Instrument Family to GM Program mapping
FAMILY_TO_PROGRAM = {
    InstrumentFamily.PIANO: GMProgram.ACOUSTIC_GRAND_PIANO,
    InstrumentFamily.BASS: GMProgram.ELECTRIC_BASS_FINGER,
    InstrumentFamily.STRINGS: GMProgram.STRING_ENSEMBLE_1,
    InstrumentFamily.BRASS: GMProgram.TRUMPET,
    InstrumentFamily.WOODWINDS: GMProgram.FLUTE,
    InstrumentFamily.GUITAR: GMProgram.ACOUSTIC_GUITAR_STEEL,
    InstrumentFamily.SYNTH: GMProgram.SYNTH_LEAD_SQUARE,
    InstrumentFamily.VOCAL: GMProgram.CHOIR_AAHS,
    InstrumentFamily.DRUMS: None,  # Drums use channel 9, no program change
    InstrumentFamily.PERCUSSION: None,
    InstrumentFamily.OTHER: GMProgram.ACOUSTIC_GRAND_PIANO,
}


# ========================================================================
# Channel Manager - Dynamic Channel Allocation
# ========================================================================

class ChannelManager:
    """
    Dynamic MIDI channel allocator for semantic instrument routing.

    Channel 9 (index 9) is reserved for drums (GM Channel 10)
    Channel 15 is reserved for forensic/metadata track
    Channels 0-8, 10-14 available for dynamic allocation
    """

    RESERVED_DRUMS = 9
    RESERVED_FORENSIC = 15
    AVAILABLE_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]

    def __init__(self):
        self._family_to_channel: Dict[InstrumentFamily, int] = {}
        self._channel_to_family: Dict[int, InstrumentFamily] = {}
        self._used_channels: Set[int] = set()

    def allocate_channel(self, family: InstrumentFamily) -> int:
        """Allocate a channel for an instrument family."""
        if family == InstrumentFamily.DRUMS:
            self._family_to_channel[family] = self.RESERVED_DRUMS
            self._channel_to_family[self.RESERVED_DRUMS] = family
            return self.RESERVED_DRUMS

        if family in self._family_to_channel:
            return self._family_to_channel[family]

        # Find next available channel
        for channel in self.AVAILABLE_CHANNELS:
            if channel not in self._used_channels:
                self._family_to_channel[family] = channel
                self._channel_to_family[channel] = family
                self._used_channels.add(channel)
                return channel

        # Fallback: use channel 0
        return 0

    def get_channel(self, family: InstrumentFamily) -> Optional[int]:
        """Get allocated channel for a family."""
        return self._family_to_channel.get(family)

    def get_family(self, channel: int) -> Optional[InstrumentFamily]:
        """Get family for a channel."""
        return self._channel_to_family.get(channel)

    def get_allocation_map(self) -> Dict[str, int]:
        """Get all allocations as dict."""
        return {fam.value: ch for fam, ch in self._family_to_channel.items()}


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class MidiExportConfig:
    """Configuration for MIDI export."""

    # Basic settings
    ppqn: int = MIDI_PPQN
    format: MidiFormat = MidiFormat.FORMAT_1
    tempo_default_bpm: float = 120.0

    # Octave restoration with provenance
    octave_restore_bass: bool = OCTAVE_RESTORE_BASS
    bass_octave_shift: int = BASS_OCTAVE_SHIFT_SEMITONES
    bass_range_low: int = BASS_NOTE_RANGE_LOW_MIDI
    bass_range_high: int = BASS_NOTE_RANGE_HIGH_MIDI

    # Heartbeat note
    heartbeat_enabled: bool = True
    heartbeat_pitch: int = HEARTBEAT_NOTE_PITCH
    heartbeat_velocity: int = HEARTBEAT_NOTE_VELOCITY
    heartbeat_duration_ms: int = HEARTBEAT_NOTE_DURATION_MS

    # Velocity handling
    min_velocity: int = 1
    max_velocity: int = 127
    default_velocity: int = 80

    # Forensic markers
    include_forensic_markers: bool = True
    include_reasoning_chains: bool = True
    include_original_timestamps: bool = True

    # Multi-track
    separate_stems: bool = True
    create_forensic_track: bool = True

    # Program changes
    send_program_changes: bool = True

    # Drum channel (DO NOT CHANGE - must be 9 for GM compliance)
    drum_channel: int = 9

    # Performance
    max_notes_per_track: int = 50000
    cleanup_after_export: bool = True
    verbose: bool = False

    # Polyphony limit per track (0 = unlimited)
    max_polyphony: int = 64


# ========================================================================
# MIDI Event Types
# ========================================================================

@dataclass
class TimelineEvent:
    """Event in the MIDI timeline for polyphonic sorting."""
    tick: int
    event_type: str  # 'note_on' or 'note_off'
    channel: int
    pitch: int
    velocity: int
    note_event_id: int = -1  # For matching note_on with note_off


# ========================================================================
# MidiWriter
# ========================================================================

class MidiWriter:
    """
    MIDI exporter with Grimlock 5.6 enhancements.

    CRITICAL FIXES:
    - Polyphony: Timeline-based event sorting for chords and overlapping notes
    - Tempo: Cumulative tick calculation across tempo segments
    - Routing: Semantic instrument_family instead of fragile source detection
    - Channels: Dynamic channel allocation for unlimited instrument types
    """

    VALID_DRUM_PITCHES = range(35, 82)
    DRUM_PITCH_MAPPING = {
        36: "Kick", 38: "Snare", 42: "Closed Hi-Hat", 46: "Open Hi-Hat",
        49: "Crash", 51: "Ride", 35: "Kick 2", 40: "Snare 2",
        41: "Low Tom", 43: "High Tom", 45: "Mid Tom", 48: "Mid Tom 2",
        47: "Low-Mid Tom", 50: "High Tom 2"
    }

    # SourceType to InstrumentFamily mapping
    SOURCE_TO_FAMILY = {
        SourceType.PITCH: InstrumentFamily.OTHER,
        SourceType.DRUM_INTELLIGENCE: InstrumentFamily.DRUMS,
        SourceType.BASS_STEM: InstrumentFamily.BASS,
        SourceType.BASS: InstrumentFamily.BASS,
        SourceType.SCRIBE: InstrumentFamily.OTHER,
        SourceType.CONSENSUS_ENGINE: InstrumentFamily.OTHER,
    }

    def __init__(
            self,
            config: Optional[MidiExportConfig] = None,
            options: Optional[ExportOptions] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        self.config = config or MidiExportConfig()
        self.options = options or ExportOptions()
        self._status_reporter = status_reporter
        self._music_box = music_box

        # MIDI file state
        self._tracks: Dict[int, bytearray] = {}  # channel_index -> track_data
        self._channel_manager = ChannelManager()
        self._tick_resolution = self.config.ppqn
        self._export_time = datetime.now()

        # Tempo state
        self._tempo_cache = {}
        self._cumulative_tick_cache = {}
        self._tempo_map: Optional[TempoMap] = None

        # Performance tracking
        self._last_export_time_ms = 0.0
        self._total_memory_freed_mb = 0.0

        # Polyphony tracking per channel
        self._active_notes: Dict[int, Set[int]] = defaultdict(set)

        self._log_status(f"MidiWriter 5.6.0 initialized (PPQN: {self.config.ppqn}, "
                         f"drum_channel={self.config.drum_channel} [GM Channel {self.config.drum_channel + 1}])")

    # ========================================================================
    # Helper: Map NoteEvent to InstrumentFamily
    # ========================================================================

    def _get_instrument_family(self, event: NoteEvent) -> InstrumentFamily:
        """Determine instrument family from NoteEvent with semantic intent."""
        # Prefer explicit instrument_family if set
        if hasattr(event, 'instrument_family') and event.instrument_family:
            if isinstance(event.instrument_family, InstrumentFamily):
                return event.instrument_family
            if isinstance(event.instrument_family, str):
                try:
                    return InstrumentFamily(event.instrument_family)
                except ValueError:
                    pass

        # Check for percussion/drum detection
        is_drum = (
                event.source == SourceType.DRUM_INTELLIGENCE or
                (hasattr(event, '_midi_channel') and getattr(event, '_midi_channel') == 9)
        )
        if is_drum:
            return InstrumentFamily.DRUMS

        # Use source mapping
        family = self.SOURCE_TO_FAMILY.get(event.source, InstrumentFamily.OTHER)

        # Refine based on pitch range for bass detection
        if family == InstrumentFamily.OTHER and event.pitch <= self.config.bass_range_high:
            family = InstrumentFamily.BASS

        return family

    # ========================================================================
    # Main Export Interface
    # ========================================================================

    def export_midi_with_drums(
            self,
            pitched_events: List[NoteEvent],
            drum_events: List[NoteEvent],
            tempo_map: Optional[TempoMap] = None,
            output_path: Optional[str] = None,
            tempo_source: str = "default",
            timbre_analysis: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Export notes to MIDI with proper polyphony and tempo integration.

        CRITICAL: Drums go to channel 9 (GM Channel 10).
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        if output_path is None:
            output_path = f"grimlock_export_{self._export_time.strftime('%Y%m%d_%H%M%S')}.mid"

        output_path = Path(output_path)

        # Count events
        drum_note_count = len(drum_events)
        pitched_note_count = len(pitched_events)

        self._log_status(
            f"Exporting {pitched_note_count} pitched notes + "
            f"{drum_note_count} drum notes to {output_path.name}"
        )

        if self._music_box:
            self._music_box.log_decision(
                stage_name="midi_writer",
                decision_type="export_start",
                before_state={},
                after_state={
                    "output_path": str(output_path),
                    "pitched_count": pitched_note_count,
                    "drum_count": drum_note_count,
                    "drum_channel": self.config.drum_channel,
                    "tempo_source": tempo_source
                },
                reasoning=f"Exporting {pitched_note_count} pitched + {drum_note_count} drum notes",
                reversible=False
            )

        # Work on copies
        pitched_events = self._copy_events(pitched_events)
        drum_events = self._copy_events(drum_events)

        # Validate drum pitches
        if drum_events:
            validated_drums = []
            for i, event in enumerate(drum_events):
                validated_drums.append(self._validate_drum_pitch(event, i))
            drum_events = validated_drums

        # Apply octave restoration with provenance
        if self.config.octave_restore_bass:
            pitched_events = self._apply_octave_restoration_with_provenance(pitched_events)

        # Reset track/channel state for this export BEFORE allocating channels,
        # so the allocations below survive into the track-writing loop.
        self._init_midi_file()
        # _ms_to_ticks_cumulative() needs this to place notes against the
        # real tempo map (including any tempo changes) instead of a
        # hardcoded default - see its docstring for why that used to be
        # a stub.
        self._tempo_map = tempo_map

        # Apply family-based routing
        events_by_family: Dict[InstrumentFamily, List[NoteEvent]] = defaultdict(list)

        for event in pitched_events:
            family = self._get_instrument_family(event)
            events_by_family[family].append(event)

        # Allocate channels for each family
        for family in events_by_family.keys():
            self._channel_manager.allocate_channel(family)

        # Drums go to channel 9
        if drum_events:
            events_by_family[InstrumentFamily.DRUMS] = drum_events
            self._channel_manager.allocate_channel(InstrumentFamily.DRUMS)

        # Law of Scribe's Truth: never write a completely empty file
        if not events_by_family and self.config.heartbeat_enabled:
            heartbeat_family = InstrumentFamily.OTHER
            events_by_family[heartbeat_family] = self._create_heartbeat_note()
            self._log_status("Empty transcription: added heartbeat note", "warn")

        # Clamp velocities for all events
        for family in events_by_family:
            events_by_family[family] = self._clamp_velocities(events_by_family[family])

        # Add tempo track with cumulative tick calculation
        self._add_tempo_track_integrated(tempo_map, tempo_source)

        # Add instrument tracks with program changes
        for family, events in events_by_family.items():
            channel = self._channel_manager.get_channel(family)
            if channel is None:
                continue

            if family == InstrumentFamily.DRUMS:
                track_name = "Grimlock - Drums"
            else:
                track_name = f"Grimlock - {family.value.title()}"

            self._add_track_with_polyphony(
                events=events,
                track_name=track_name,
                channel=channel,
                instrument_family=family
            )

        # Forensic audit track
        if self.config.include_forensic_markers:
            all_events = pitched_events + drum_events
            self._add_forensic_track_split(all_events, tempo_map, tempo_source, timbre_analysis)

        # Write to disk
        success = self._write_file(output_path)

        execution_time_ms = (time.time() - start_time) * 1000
        memory_delta_mb = self._get_current_memory_mb() - start_memory
        self._last_export_time_ms = execution_time_ms

        if self.config.cleanup_after_export and memory_delta_mb > STAGGERED_GC_TRIGGER_MB:
            self._force_cleanup()

        if success and self._music_box:
            self._music_box.log_decision(
                stage_name="midi_writer",
                decision_type="export_complete",
                before_state={},
                after_state={
                    "output_path": str(output_path),
                    "file_size_bytes": output_path.stat().st_size,
                    "track_count": len(self._tracks),
                    "execution_time_ms": execution_time_ms
                },
                reasoning=f"MIDI exported to {output_path.name}",
                reversible=False
            )

        self._log_status(f"Export complete in {execution_time_ms:.0f}ms")
        return success

    def export_midi(
            self,
            events: List[NoteEvent],
            tempo_map: Optional[TempoMap] = None,
            output_path: Optional[str] = None,
            track_name: str = "Grimlock Transcription",
            separate_stems: Optional[bool] = None,
            tempo_source: str = "default",
            timbre_analysis: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Legacy export method - routes events intelligently."""
        pitched_events = []
        drum_events = []

        for event in events:
            family = self._get_instrument_family(event)
            if family == InstrumentFamily.DRUMS:
                drum_events.append(event)
            else:
                pitched_events.append(event)

        return self.export_midi_with_drums(
            pitched_events=pitched_events,
            drum_events=drum_events,
            tempo_map=tempo_map,
            output_path=output_path,
            tempo_source=tempo_source,
            timbre_analysis=timbre_analysis
        )

    # ========================================================================
    # Polyphony-Aware Track Writing (FIX 1)
    # ========================================================================

    def _add_track_with_polyphony(
            self,
            events: List[NoteEvent],
            track_name: str,
            channel: int,
            instrument_family: Optional[InstrumentFamily] = None
    ):
        """
        Write notes to MIDI track with proper polyphony support.

        Uses timeline-based event sorting for correct chords and overlapping notes.
        """
        if not events:
            return

        track = bytearray()

        # MTrk chunk header
        track.extend(self._make_chunk(b'MTrk', 0))

        # Track name meta event
        name_bytes = track_name.encode('ascii')[:255]
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x03, name_bytes))

        # Program Change (if configured)
        if self.config.send_program_changes and instrument_family and instrument_family != InstrumentFamily.DRUMS:
            program = FAMILY_TO_PROGRAM.get(instrument_family, GMProgram.ACOUSTIC_GRAND_PIANO)
            if program is not None:
                track.extend(self._encode_var_length(0))
                track.append(0xC0 | (channel & 0x0F))  # Program Change status
                track.append(program.value & 0x7F)  # Program number

        # Build timeline of note_on and note_off events
        timeline: List[TimelineEvent] = []

        for i, event in enumerate(events):
            start_ms = event.get_active_start_ms()
            end_ms = start_ms + max(10.0, event.active_duration_ms())

            # Convert to ticks
            start_tick = self._ms_to_ticks_cumulative(start_ms)
            end_tick = self._ms_to_ticks_cumulative(end_ms)

            # Ensure end_tick > start_tick
            if end_tick <= start_tick:
                end_tick = start_tick + 1

            timeline.append(TimelineEvent(
                tick=start_tick,
                event_type='note_on',
                channel=channel,
                pitch=event.pitch,
                velocity=int(event.velocity),
                note_event_id=i
            ))

            timeline.append(TimelineEvent(
                tick=end_tick,
                event_type='note_off',
                channel=channel,
                pitch=event.pitch,
                velocity=0,
                note_event_id=i
            ))

        # Sort timeline by tick
        timeline.sort(key=lambda e: (e.tick, 0 if e.event_type == 'note_off' else 1))

        # Write events with delta times
        last_tick = 0
        active_note_offs: Dict[int, int] = {}  # note_event_id -> expected off_tick

        for event in timeline:
            delta_ticks = max(0, event.tick - last_tick)

            # Every MIDI event must be preceded by a delta-time byte, even
            # when the delta is zero - omitting it corrupts the byte stream
            # for every subsequent event in the track.
            track.extend(self._encode_var_length(delta_ticks))

            if event.event_type == 'note_on':
                # Check polyphony limit
                forced_off = False
                if self.config.max_polyphony > 0:
                    active_count = len(self._active_notes[channel])
                    if active_count >= self.config.max_polyphony:
                        # Force oldest note off
                        self._log_status(f"Polyphony limit reached on channel {channel}, force releasing notes", "warn")
                        # Find and force-off the oldest note. note_id is
                        # the OLD note's index into events (note_event_id),
                        # not the new note-on's pitch - using event.pitch
                        # here (the timeline entry for the NEW note-on
                        # about to be triggered) sent a Note-Off for the
                        # wrong pitch: the new note being added, which
                        # hadn't even had its Note-On sent yet, while the
                        # actual old note being released got no Note-Off
                        # at all and would play forever (stuck note).
                        for note_id in list(self._active_notes[channel])[:10]:
                            if note_id in active_note_offs:
                                # Already scheduled for off
                                continue
                            # Force off now. This inserts extra events into
                            # the byte stream mid-loop. The delta_ticks
                            # written at the top of this outer loop already
                            # covers the FIRST inserted event (that's the
                            # thing it's advancing the clock to) - every
                            # event AFTER that first one still needs its
                            # OWN delta-time byte (0, since nothing else
                            # advances the clock further), which this was
                            # never writing at all. Confirmed directly:
                            # mido failed to parse a real exported file
                            # with "data byte must be in range 0..127" the
                            # moment this path was exercised with 2+ active
                            # notes forced off at once - tracing the raw
                            # bytes showed two delta-times back to back
                            # with no event between them.
                            if forced_off:
                                track.extend(self._encode_var_length(0))
                            old_pitch = events[note_id].pitch
                            track.extend(self._make_note_off(channel, old_pitch, 0))
                            self._active_notes[channel].discard(note_id)
                            forced_off = True

                self._active_notes[channel].add(event.note_event_id)
                if forced_off:
                    track.extend(self._encode_var_length(0))
                track.extend(self._make_note_on(channel, event.pitch, event.velocity))

            else:  # note_off
                self._active_notes[channel].discard(event.note_event_id)
                track.extend(self._make_note_off(channel, event.pitch, 0))

            last_tick = event.tick

        # End of track
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x2F, bytes([])))

        # Back-fill track length
        self._update_track_length(track)
        self._tracks[channel] = track

    # ========================================================================
    # Tempo Track with Cumulative Tick Integration (FIX 2)
    # ========================================================================

    def _add_tempo_track_integrated(
            self,
            tempo_map: Optional[TempoMap],
            tempo_source: str = "default"
    ):
        """Add tempo track with correct cumulative tick calculation."""
        track = bytearray()

        track.extend(self._make_chunk(b'MTrk', 0))

        # Track name
        name_bytes = f"Grimlock 5.6 Tempo Map (source: {tempo_source})".encode('ascii')
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x03, name_bytes))

        # Initial tempo
        tempo_bpm = tempo_map.initial_tempo_bpm if tempo_map else self.config.tempo_default_bpm
        tempo_us = self._bpm_to_us(tempo_bpm)
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x51, self._encode_three_bytes(tempo_us)))

        # Time signature
        if tempo_map:
            numerator = tempo_map.time_signature_numerator
            denominator = tempo_map.time_signature_denominator
        else:
            numerator = 4
            denominator = 4
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(
            0x58,
            bytes([numerator, self._denom_to_power(denominator), 24, 8])
        ))

        # Gradual tempo changes with cumulative tick calculation
        if tempo_map and tempo_map.tempo_events:
            last_tick = 0
            last_time_ms = 0
            last_tempo_bpm = tempo_bpm

            for event in tempo_map.tempo_events:
                # Calculate cumulative ticks to this event
                segment_duration_ms = event.time_ms - last_time_ms
                segment_ticks = self._duration_to_ticks(segment_duration_ms, last_tempo_bpm)
                current_tick = last_tick + segment_ticks

                delta_ticks = max(0, current_tick - last_tick)
                track.extend(self._encode_var_length(delta_ticks))

                new_tempo_us = self._bpm_to_us(event.tempo_bpm)
                track.extend(self._make_meta_event(0x51, self._encode_three_bytes(new_tempo_us)))

                last_tick = current_tick
                last_time_ms = event.time_ms
                last_tempo_bpm = event.tempo_bpm

        # End of track
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x2F, bytes([])))

        self._update_track_length(track)
        # Tempo track is channel -1 (special)
        self._tracks[-1] = track

    def _duration_to_ticks(self, duration_ms: float, tempo_bpm: float) -> float:
        """Convert duration in ms to ticks with double precision."""
        if duration_ms <= 0:
            return 0.0
        microseconds_per_beat = self._bpm_to_us(tempo_bpm)
        beats = (duration_ms * 1000.0) / microseconds_per_beat
        return beats * self._tick_resolution

    def _ms_to_ticks_cumulative(self, ms: float) -> int:
        """
        Convert milliseconds to cumulative MIDI ticks with tempo integration.

        Integrates across tempo_events using each segment's own tempo,
        exactly matching the tempo track _add_tempo_track_integrated()
        writes. This used to always use a hardcoded 120 BPM default
        regardless of the real tempo_map passed to export - the tempo
        meta-track declared the correct BPM, but every note's tick
        position was computed as if the song were 120 BPM, silently
        misplacing every note relative to the tempo the file itself
        declares whenever the real tempo wasn't ~120 BPM.
        """
        if ms <= 0:
            return 0

        if ms in self._cumulative_tick_cache:
            return self._cumulative_tick_cache[ms]

        tempo_map = self._tempo_map
        if tempo_map is None:
            # No tempo map available (e.g. a direct/legacy call) - fall
            # back to the configured default, same as before.
            tempo_bpm = self.config.tempo_default_bpm
            microseconds_per_beat = self._bpm_to_us(tempo_bpm)
            ticks = int((ms * 1000.0 * self._tick_resolution) / microseconds_per_beat)
            self._cumulative_tick_cache[ms] = ticks
            return ticks

        events = (sorted(tempo_map.tempo_events, key=lambda e: e.time_ms)
                 if tempo_map.tempo_events else [])

        accumulated_ticks = 0.0
        last_time_ms = 0.0
        current_tempo_bpm = tempo_map.initial_tempo_bpm

        for event in events:
            if event.time_ms >= ms:
                break
            accumulated_ticks += self._duration_to_ticks(
                event.time_ms - last_time_ms, current_tempo_bpm)
            last_time_ms = event.time_ms
            current_tempo_bpm = event.tempo_bpm

        accumulated_ticks += self._duration_to_ticks(ms - last_time_ms, current_tempo_bpm)

        ticks = int(accumulated_ticks)
        self._cumulative_tick_cache[ms] = ticks
        return ticks

    # ========================================================================
    # Octave Restoration with Provenance (FIX 4)
    # ========================================================================

    def _apply_octave_restoration_with_provenance(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """
        Shift bass notes down one octave with provenance tracking.

        Prevents double-shifting by checking if note was already restored.
        """
        BASS_SOURCE_TYPES = {SourceType.BASS_STEM, SourceType.PITCH, SourceType.BASS}
        restored = []

        for event in events:
            # Check if already restored (provenance)
            already_restored = False
            if event.reasoning_chain:
                for entry in event.reasoning_chain:
                    if "octave restored" in entry.lower() or "octave_restored" in entry.lower():
                        already_restored = True
                        break

            # Also check for explicit flag
            if hasattr(event, 'flags') and event.flags.get('octave_restored', False):
                already_restored = True

            pitch = event.pitch
            is_bass = (
                    event.source in BASS_SOURCE_TYPES and
                    self.config.bass_range_low <= pitch <= self.config.bass_range_high
            )

            if is_bass and not already_restored:
                new_pitch = max(0, min(127, pitch + self.config.bass_octave_shift))
                new_event = copy(event)
                new_event.pitch = new_pitch

                # Add reasoning with provenance
                if new_event.reasoning_chain:
                    new_event.reasoning_chain.append(f"Octave restored (provenance): {pitch} → {new_pitch}")
                else:
                    new_event.reasoning_chain = [f"Octave restored (provenance): {pitch} → {new_pitch}"]

                # Set flag for future checks
                if not hasattr(new_event, 'flags'):
                    new_event.flags = {}
                new_event.flags['octave_restored'] = True
                new_event.flags['original_pitch'] = pitch

                restored.append(new_event)
            else:
                if already_restored and self.config.verbose:
                    self._log_status(f"Skipping octave restoration for note {pitch} - already restored", "debug")
                restored.append(event)

        return restored

    # ========================================================================
    # Drum Pitch Validation (FIX 3 - part of it)
    # ========================================================================

    def _validate_drum_pitch(self, event: NoteEvent, index: int) -> NoteEvent:
        """Validate and optionally remap invalid drum pitches."""
        if event.pitch not in self.VALID_DRUM_PITCHES:
            old_pitch = event.pitch
            # Find nearest valid drum pitch
            nearest = min(self.VALID_DRUM_PITCHES, key=lambda x: abs(x - event.pitch))
            new_event = copy(event)
            new_event.pitch = nearest

            if new_event.reasoning_chain:
                new_event.reasoning_chain.append(
                    f"Drum pitch corrected: {old_pitch} → {nearest} "
                    f"({self.DRUM_PITCH_MAPPING.get(nearest, 'GM Drum')})"
                )

            self._log_status(
                f"Fixed drum note {index}: pitch {old_pitch} → {nearest}",
                "warn"
            )
            return new_event
        return event

    # ========================================================================
    # Forensic Track with Split Strings (FIX 5)
    # ========================================================================

    def _add_forensic_track_split(
            self,
            events: List[NoteEvent],
            tempo_map: Optional[TempoMap],
            tempo_source: str = "default",
            timbre_analysis: Optional[Dict[str, Any]] = None
    ):
        """Write forensic metadata track with proper string splitting."""
        track = bytearray()

        track.extend(self._make_chunk(b'MTrk', 0))

        # Track name
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x03, b"Grimlock 5.6 Forensic Audit"))

        # Enhanced statistics
        stats = self._compute_enhanced_statistics(events)

        metadata_lines = [
            "Grimlock 5.6 Transcription",
            f"Exported: {self._export_time.isoformat()}",
            f"Total notes: {len(events)}",
            f"PPQN: {self.config.ppqn}",
            f"Tempo source: {tempo_source}",
            f"Drum channel: {self.config.drum_channel} (GM {self.config.drum_channel + 1})",
            "",
            "=== Confidence Statistics ===",
            f"Mean confidence: {stats['mean_confidence']:.3f}",
            f"Median confidence: {stats['median_confidence']:.3f}",
            f"Std deviation: {stats['std_confidence']:.3f}",
            f"5th percentile: {stats['p5_confidence']:.3f}",
            f"95th percentile: {stats['p95_confidence']:.3f}",
            f"Min confidence: {stats['min_confidence']:.3f}",
            f"Max confidence: {stats['max_confidence']:.3f}",
        ]

        if tempo_map:
            metadata_lines.append(f"Tempo: {tempo_map.initial_tempo_bpm:.1f} BPM")
            if tempo_map.tempo_events:
                metadata_lines.append(f"Tempo changes: {len(tempo_map.tempo_events)}")

        if timbre_analysis:
            instrument_counts = timbre_analysis.get("instrument_counts", {})
            if instrument_counts:
                metadata_lines.append("Timbre Analysis:")
                for inst, count in instrument_counts.items():
                    metadata_lines.append(f"  {inst}: {count} notes")

        # Channel allocation map
        metadata_lines.append("Channel Allocation:")
        for family, channel in self._channel_manager.get_allocation_map().items():
            metadata_lines.append(f"  {family}: channel {channel}")

        # Source distribution
        source_counts: Dict[str, int] = {}
        for event in events:
            source_counts[event.source.value] = source_counts.get(event.source.value, 0) + 1
        if source_counts:
            metadata_lines.append("Source distribution:")
            for source, count in source_counts.items():
                metadata_lines.append(f"  {source}: {count}")

        # Snapped notes info
        snapped_notes = [e for e in events if e.is_snapped()]
        if snapped_notes and self.config.include_original_timestamps:
            metadata_lines.append(f"Snapped notes: {len(snapped_notes)}")
            for i, note in enumerate(snapped_notes[:10]):
                metadata_lines.append(
                    f"  Note {i}: original {note.start_ms:.1f}ms "
                    f"→ snapped {note.snapped_start_ms:.1f}ms"
                )
            if len(snapped_notes) > 10:
                metadata_lines.append(f"  ... and {len(snapped_notes) - 10} more")

        # Write each line, splitting long strings
        for line in metadata_lines:
            self._write_split_meta_event(track, line.encode('utf-8'), event_type=0x01)  # Text event

        # Reasoning chains from notes
        if self.config.include_reasoning_chains:
            metadata_lines = ["", "=== Reasoning Chains (sample) ==="]
            for line in metadata_lines:
                self._write_split_meta_event(track, line.encode('utf-8'), event_type=0x01)

            shown = 0
            for event in events[:50]:
                if event.reasoning_chain and shown < 20:
                    chain_text = f"Note {event.pitch} @ {event.start_ms:.1f}ms: {' → '.join(event.reasoning_chain[:5])}"
                    self._write_split_meta_event(track, chain_text.encode('utf-8'), event_type=0x01)
                    shown += 1

        # End of track
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x2F, bytes([])))

        self._update_track_length(track)
        self._tracks[ChannelManager.RESERVED_FORENSIC] = track

    def _write_split_meta_event(self, track: bytearray, data: bytes, event_type: int = 0x01, max_chunk: int = 250):
        """
        Write a meta event, splitting long data into multiple events.

        Many DAWs have limits on meta event size. Splitting ensures all data is preserved.
        """
        for i in range(0, len(data), max_chunk):
            chunk = data[i:i + max_chunk]
            track.extend(self._encode_var_length(0))
            track.extend(self._make_meta_event(event_type, chunk))

    # ========================================================================
    # Enhanced Statistics (FIX 6)
    # ========================================================================

    def _compute_enhanced_statistics(self, events: List[NoteEvent]) -> Dict[str, float]:
        """Compute comprehensive confidence statistics."""
        if not events:
            return {
                'mean_confidence': 0.0,
                'median_confidence': 0.0,
                'std_confidence': 0.0,
                'p5_confidence': 0.0,
                'p95_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'total_notes': 0
            }

        confidences = [e.confidence for e in events]
        conf_array = np.array(confidences)

        return {
            'mean_confidence': float(np.mean(conf_array)),
            'median_confidence': float(np.median(conf_array)),
            'std_confidence': float(np.std(conf_array)),
            'p5_confidence': float(np.percentile(conf_array, 5)),
            'p95_confidence': float(np.percentile(conf_array, 95)),
            'min_confidence': float(np.min(conf_array)),
            'max_confidence': float(np.max(conf_array)),
            'total_notes': len(events)
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _copy_events(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Shallow copy events to avoid mutating originals."""
        return [copy(e) for e in events]

    def _clamp_velocities(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Clamp velocities to valid MIDI range."""
        for event in events:
            event.velocity = max(
                self.config.min_velocity,
                min(self.config.max_velocity, event.velocity)
            )
        return events

    def _create_heartbeat_note(self) -> List[NoteEvent]:
        """Create a barely-audible marker note for empty transcriptions."""
        heartbeat = NoteEvent(
            pitch=self.config.heartbeat_pitch,
            start_ms=0.0,
            end_ms=float(self.config.heartbeat_duration_ms),
            velocity=self.config.heartbeat_velocity,
            confidence=Confidence.HALLUCINATION.value_f,
            zero_crossing_rate=0.0,
            source=SourceType.SCRIBE,
            reasoning_chain=[
                "Heartbeat note — empty transcription marker",
                f"Confidence: {Confidence.HALLUCINATION.value_f} — not a real note"
            ]
        )
        # Add provenance flags
        if not hasattr(heartbeat, 'flags'):
            heartbeat.flags = {}
        heartbeat.flags['is_heartbeat'] = True
        return [heartbeat]

    def _init_midi_file(self):
        """Reset tracks for a new export."""
        self._tracks = {}
        self._active_notes.clear()
        self._tempo_cache.clear()
        self._cumulative_tick_cache.clear()
        self._channel_manager = ChannelManager()

    # ========================================================================
    # MIDI Encoding Helpers
    # ========================================================================

    def _encode_var_length(self, value: int) -> bytes:
        """Encode a value as a MIDI variable-length quantity."""
        # Callers sometimes pass numpy floats (e.g. tempo-change tick deltas
        # derived from TempoEvent.time_ms, which can be numpy.float64) -
        # bitwise ops on those raise "ufunc 'bitwise_and' not supported"
        # instead of the plain-int MIDI byte math this is meant to do.
        value = int(value)
        if value < 0:
            return b'\x00'

        buffer = bytearray()
        buffer.append(value & 0x7F)
        value >>= 7

        while value > 0:
            buffer.append(0x80 | (value & 0x7F))
            value >>= 7

        return bytes(reversed(buffer))

    def _make_chunk(self, chunk_type: bytes, length: int) -> bytes:
        """Build a MIDI chunk header (8 bytes)."""
        return chunk_type + struct.pack('>I', length)

    def _make_meta_event(self, meta_type: int, data: bytes) -> bytes:
        """Build a MIDI meta event."""
        return bytes([0xFF, meta_type]) + self._encode_var_length(len(data)) + data

    def _make_note_on(self, channel: int, pitch: int, velocity: int) -> bytes:
        """Build a MIDI Note On message with the specified channel."""
        status = 0x90 | (channel & 0x0F)
        return bytes([status, pitch & 0x7F, velocity & 0x7F])

    def _make_note_off(self, channel: int, pitch: int, velocity: int) -> bytes:
        """Build a MIDI Note Off message with the specified channel."""
        status = 0x80 | (channel & 0x0F)
        return bytes([status, pitch & 0x7F, velocity & 0x7F])

    def _encode_three_bytes(self, value: int) -> bytes:
        """Encode a 24-bit integer as 3 big-endian bytes."""
        return struct.pack('>I', value)[1:4]

    def _bpm_to_us(self, bpm: float) -> int:
        """Convert BPM to microseconds per quarter note."""
        if bpm <= 0:
            bpm = 120.0
        return int(60_000_000 / bpm)

    def _denom_to_power(self, denominator: int) -> int:
        """Convert time signature denominator to its power-of-2 exponent."""
        return {2: 1, 4: 2, 8: 3, 16: 4}.get(denominator, 2)

    def _update_track_length(self, track: bytearray):
        """Back-fill the 4-byte length field in the MTrk header."""
        track_length = len(track) - 8
        track[4:8] = struct.pack('>I', track_length)

    def _write_file(self, output_path: Path) -> bool:
        """Write the complete MIDI file to disk."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            format_type = 1 if self.config.format == MidiFormat.FORMAT_1 else 0
            track_count = len(self._tracks)

            with open(output_path, 'wb') as f:
                # MThd header
                header = (
                        b'MThd' +
                        struct.pack('>I', 6) +
                        struct.pack('>HHH', format_type, track_count, self._tick_resolution)
                )
                f.write(header)

                # Write tracks in order (tempo track first, then by channel)
                # Get tempo track (key -1)
                if -1 in self._tracks:
                    f.write(self._tracks[-1])

                # Write other tracks by channel number
                for channel in sorted([k for k in self._tracks.keys() if k != -1]):
                    f.write(self._tracks[channel])

            return True

        except Exception as e:
            self._log_status(f"Failed to write MIDI file: {e}", "error")
            if self._music_box:
                self._music_box.log_decision(
                    stage_name="midi_writer",
                    decision_type="error",
                    before_state={},
                    after_state={"error": str(e), "output_path": str(output_path)},
                    reasoning=f"MIDI export failed: {e}",
                    reversible=False
                )
            return False

    def _force_cleanup(self):
        """Force garbage collection and CUDA cache clearing."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def _get_current_memory_mb(self) -> float:
        """Return current process RSS memory usage in MB."""
        try:
            import psutil
            import os
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _log_status(self, message: str, level: str = "info"):
        """Forward a status message to the StatusReporter."""
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info("MidiWriter", message)
            elif level == "warn":
                self._status_reporter.warn("MidiWriter", message)
            elif level == "error":
                self._status_reporter.error("MidiWriter", message)
            elif level == "debug" and self.config.verbose:
                self._status_reporter.info("MidiWriter", f"[DEBUG] {message}")

    # ========================================================================
    # Public Info
    # ========================================================================

    def get_export_info(self) -> Dict[str, Any]:
        """Return metadata about the last export."""
        return {
            "version": "5.6.0",
            "export_time": self._export_time.isoformat(),
            "tick_resolution": self._tick_resolution,
            "track_count": len(self._tracks),
            "last_export_time_ms": self._last_export_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb,
            "drum_channel": self.config.drum_channel,
            "gm_channel": self.config.drum_channel + 1,
            "channel_allocation": self._channel_manager.get_allocation_map(),
            "config": {
                "octave_restoration": self.config.octave_restore_bass,
                "separate_stems": self.config.separate_stems,
                "heartbeat_enabled": self.config.heartbeat_enabled,
                "include_forensic_markers": self.config.include_forensic_markers,
                "max_polyphony": self.config.max_polyphony
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return writer statistics."""
        return {
            "name": "MidiWriter",
            "version": "5.6.0",
            "ppqn": self._tick_resolution,
            "last_export_time_ms": self._last_export_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb,
            "track_count": len(self._tracks)
        }


# ========================================================================
# Convenience Functions
# ========================================================================

def export_to_midi(
        events: List[NoteEvent],
        output_path: str,
        tempo_map: Optional[TempoMap] = None,
        options: Optional[ExportOptions] = None,
        config: Optional[MidiExportConfig] = None,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None,
        tempo_source: str = "default",
        timbre_analysis: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Convenience function — create a MidiWriter and export in one call.
    Automatically routes drums to channel 9.
    """
    writer = MidiWriter(config, options, status_reporter, music_box)
    return writer.export_midi(
        events=events,
        tempo_map=tempo_map,
        output_path=output_path,
        tempo_source=tempo_source,
        timbre_analysis=timbre_analysis
    )


def export_midi_with_drums(
        pitched_events: List[NoteEvent],
        drum_events: List[NoteEvent],
        output_path: str,
        tempo_map: Optional[TempoMap] = None,
        config: Optional[MidiExportConfig] = None,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None,
        tempo_source: str = "default"
) -> bool:
    """
    Convenience function — export with explicit drum routing to channel 9.

    CRITICAL: Drums go to channel 9 (0-indexed) = GM Channel 10.
    This is the standard percussion channel.
    """
    writer = MidiWriter(config, None, status_reporter, music_box)
    return writer.export_midi_with_drums(
        pitched_events=pitched_events,
        drum_events=drum_events,
        tempo_map=tempo_map,
        output_path=output_path,
        tempo_source=tempo_source
    )


def quick_midi_test():
    """Quick test to verify MIDI export with all fixes."""
    print("\n" + "=" * 60)
    print("MIDI Writer Test v5.6.0")
    print("=" * 60)

    # Create test notes with chords (testing polyphony fix)
    pitched = [
        NoteEvent(pitch=60, start_ms=0, end_ms=1000, velocity=80, confidence=0.9,
                  zero_crossing_rate=0.1, source=SourceType.PITCH),
        NoteEvent(pitch=64, start_ms=0, end_ms=1000, velocity=75, confidence=0.8,
                  zero_crossing_rate=0.1, source=SourceType.PITCH),  # Chord note at same time
        NoteEvent(pitch=67, start_ms=500, end_ms=1500, velocity=70, confidence=0.85,
                  zero_crossing_rate=0.1, source=SourceType.PITCH),  # Overlapping
    ]

    drums = [
        NoteEvent(pitch=36, start_ms=0, end_ms=200, velocity=100, confidence=0.9,
                  zero_crossing_rate=0.5, source=SourceType.DRUM_INTELLIGENCE),
        NoteEvent(pitch=38, start_ms=500, end_ms=700, velocity=90, confidence=0.85,
                  zero_crossing_rate=0.5, source=SourceType.DRUM_INTELLIGENCE),
    ]

    tempo_map = TempoMap(initial_tempo_bpm=120, tempo_events=[], confidence=Confidence.HIGH)

    # Test config with verbose output
    config = MidiExportConfig(verbose=True, max_polyphony=32)
    writer = MidiWriter(config=config)
    output_path = "test_midi_5_6_0.mid"

    success = writer.export_midi_with_drums(
        pitched_events=pitched,
        drum_events=drums,
        tempo_map=tempo_map,
        output_path=output_path,
        tempo_source="test"
    )

    print(f"\nTest result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Output file: {output_path}")

    if success:
        info = writer.get_export_info()
        print(f"Channel allocation: {info.get('channel_allocation', {})}")
        print(f"Track count: {info.get('track_count', 0)}")

    print("=" * 60)
    return success


if __name__ == "__main__":
    quick_midi_test()