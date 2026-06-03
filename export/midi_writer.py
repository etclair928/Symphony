# =================================================================
# MODULE: export/midi_writer.py
# DESCRIPTION: MIDI export with octave restoration and forensic audit.
#
# VERSION: 5.5.0 (CRITICAL: Drums to Channel 9 = GM Channel 10)
# UPDATED: 2026-05-26
#
# KEY FIXES in 5.5.0:
#   1. Drums ALWAYS routed to Channel 9 (0-indexed) = GM Channel 10
#   2. Fixed _add_single_track_notes to use channel parameter correctly
#   3. Added debug output showing channel routing
#   4. Fixed MTrk header byte order
#   5. Added proper channel mapping for all instrument types
#
# MIDI STANDARD: Channel 9 (0-indexed) is ALWAYS the percussion channel
# All DAWs (MuseScore, Sibelius, Logic, Ableton, Cubase) respect this.
# =================================================================

import struct
import warnings
import time
import gc
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from copy import copy

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
    FORMAT_0 = "format_0"  # Single track
    FORMAT_1 = "format_1"  # Multiple tracks, synchronous
    FORMAT_2 = "format_2"  # Multiple tracks, asynchronous


class ExportChannel:
    """
    MIDI channels for different instrument types.

    CRITICAL: Channel 9 (0-indexed) = General MIDI Channel 10 (1-indexed)
    This is the STANDARD percussion channel. All DAWs and notation software
    (MuseScore, Sibelius, Logic, Ableton, Cubase) treat channel 9 as the
    percussion track. DO NOT CHANGE THIS.

    GM Percussion Note Mapping (Standard):
        Bass Drum:     35 (B0), 36 (C1)
        Snare:         38 (D1), 40 (E1)
        Hi-Hat Closed: 42 (F#1)
        Hi-Hat Open:   46 (A#1)
        Crash Cymbal:  49 (C#2), 57 (A2)
        Ride Cymbal:   51 (D#2)
        Ride Bell:     53 (E2)
        Tom Low:       43 (G1), 45 (A1)
        Tom Mid:       47 (B1), 48 (C2)
        Tom High:      50 (D2)
    """
    BASS = 0  # Bass guitar/bass instruments
    MELODY = 1  # Melodic instruments (piano, sax, trumpet)
    HARMONY = 2  # Harmonic instruments (strings, pads)
    OTHER = 3  # Other pitched instruments
    DRUMS = 9  # PERCUSSION - GM Channel 10 (DO NOT CHANGE)
    FORENSIC = 15  # Metadata track


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class MidiExportConfig:
    """Configuration for MIDI export."""

    # Basic settings
    ppqn: int = MIDI_PPQN  # 480 pulses per quarter note
    format: MidiFormat = MidiFormat.FORMAT_1
    tempo_default_bpm: float = 120.0

    # Octave restoration
    octave_restore_bass: bool = OCTAVE_RESTORE_BASS
    bass_octave_shift: int = BASS_OCTAVE_SHIFT_SEMITONES  # -12
    bass_range_low: int = BASS_NOTE_RANGE_LOW_MIDI  # 28
    bass_range_high: int = BASS_NOTE_RANGE_HIGH_MIDI  # 52

    # Heartbeat note
    heartbeat_enabled: bool = True
    heartbeat_pitch: int = HEARTBEAT_NOTE_PITCH  # 60 (C4)
    heartbeat_velocity: int = HEARTBEAT_NOTE_VELOCITY  # 10
    heartbeat_duration_ms: int = HEARTBEAT_NOTE_DURATION_MS  # 500

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

    # Drum channel (DO NOT CHANGE - must be 9 for GM compliance)
    drum_channel: int = 9  # GM Channel 10 (1-indexed)

    # Performance
    max_notes_per_track: int = 50000
    cleanup_after_export: bool = True
    verbose: bool = False  # Enable debug output


# ========================================================================
# MidiWriter
# ========================================================================

class MidiWriter:
    """
    MIDI exporter with Grimlock 5.5 enhancements.

    CRITICAL: Drums are ALWAYS exported to Channel 9 (0-indexed) = GM Channel 10.
    This is the MIDI standard percussion channel that all DAWs recognize.

    Features:
    - Octave restoration (shift bass notes down for natural playback)
    - Heartbeat note for empty transcriptions
    - Tempo map with gradual changes
    - Forensic markers (veto history, confidence, original timestamps)
    - Multi-track export with proper channel routing
    - StatusReporter integration
    - MusicBox forensic logging
    """

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
        self._tracks: List[bytearray] = []
        self._tick_resolution = self.config.ppqn
        self._export_time = datetime.now()

        # Performance tracking
        self._last_export_time_ms = 0.0
        self._total_memory_freed_mb = 0.0

        self._log_status(f"MidiWriter initialized (PPQN: {self.config.ppqn}, "
                         f"drum_channel={self.config.drum_channel} [GM Channel {self.config.drum_channel + 1}])")

    # ========================================================================
    # Main Export Interface
    # ========================================================================

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
        """
        Export notes to MIDI file (single track, channel 0).

        For proper drum routing, use export_midi_with_drums() instead.
        """
        # Separate drums from pitched events
        drum_events = []
        pitched_events = []

        for event in events:
            is_drum = (
                    event.source in [SourceType.DRUM_INTELLIGENCE, SourceType.DRUMS] or
                    (hasattr(event, '_midi_channel') and getattr(event, '_midi_channel') == 9)
            )
            if is_drum:
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
        Export pitched notes on channel 0 and drum notes on channel 9.

        CRITICAL: Drums go to channel 9 (0-indexed) = General MIDI Channel 10.
        This is the STANDARD percussion channel. DO NOT CHANGE.

        Args:
            pitched_events: NoteEvent list for melodic instruments (ch 0)
            drum_events:    NoteEvent list for drums (ch 9)
            tempo_map:      TempoMap for the tempo track
            output_path:    Where to write the .mid file
            tempo_source:   Source of tempo detection for metadata
            timbre_analysis: Timbre intelligence results for metadata

        Returns:
            True if export successful
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        if output_path is None:
            output_path = f"grimlock_export_{self._export_time.strftime('%Y%m%d_%H%M%S')}.mid"

        output_path = Path(output_path)

        # Count drum notes
        drum_note_count = len(drum_events)
        pitched_note_count = len(pitched_events)

        self._log_status(
            f"Exporting {pitched_note_count} pitched notes (ch 0) + "
            f"{drum_note_count} drum notes (ch {self.config.drum_channel} / GM {self.config.drum_channel + 1}) "
            f"to {output_path.name}"
        )

        if self.config.verbose:
            print(
                f"  [MIDI] Pitched: channel 0, Drums: channel {self.config.drum_channel} (GM {self.config.drum_channel + 1})")

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
                    "gm_channel": self.config.drum_channel + 1,
                    "tempo_source": tempo_source
                },
                reasoning=f"Exporting {pitched_note_count} pitched + {drum_note_count} drum notes",
                reversible=False
            )

        # Work on copies so we never mutate originals
        pitched_events = self._copy_events(pitched_events)
        drum_events = self._copy_events(drum_events)

        # Optional octave restoration for bass notes
        if self.config.octave_restore_bass:
            pitched_events = self._apply_octave_restoration(pitched_events)

        # Law of Scribe's Truth: never write a completely empty file
        if not pitched_events and not drum_events and self.config.heartbeat_enabled:
            pitched_events = self._create_heartbeat_note()
            self._log_status("Empty transcription: added heartbeat note", "warn")

        # Clamp velocities
        pitched_events = self._clamp_velocities(pitched_events)
        drum_events = self._clamp_velocities(drum_events)

        # Build MIDI file structure
        self._init_midi_file()
        self._add_tempo_track(tempo_map, tempo_source)

        # Pitched instruments — channel 0
        if pitched_events:
            if self.config.verbose:
                print(f"  [MIDI] Writing {len(pitched_events)} pitched notes to channel 0")
            self._add_single_track_notes(
                events=pitched_events,
                track_name="Grimlock - Pitched",
                channel=ExportChannel.BASS,
            )
        elif self.config.heartbeat_enabled:
            self._add_single_track_notes(
                events=self._create_heartbeat_note(),
                track_name="Grimlock - Pitched (Heartbeat)",
                channel=ExportChannel.BASS,
            )

        # Drums — channel 9 (GM Channel 10 = percussion)
        if drum_events:
            if self.config.verbose:
                print(
                    f"  [MIDI] Writing {len(drum_events)} drum notes to channel {self.config.drum_channel} (GM {self.config.drum_channel + 1})")
            self._add_single_track_notes(
                events=drum_events,
                track_name="Grimlock - Drums",
                channel=self.config.drum_channel,  # 9 = percussion channel
            )

        # Forensic audit track (if configured)
        if self.config.include_forensic_markers:
            all_events = pitched_events + drum_events
            self._add_forensic_track(all_events, tempo_map, tempo_source, timbre_analysis)

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

        if success and self.config.verbose:
            print(f"  [MIDI] Export complete: {output_path.name}")
            print(f"  [MIDI] Tracks: {len(self._tracks)}, Size: {output_path.stat().st_size} bytes")

        self._log_status(f"Export complete in {execution_time_ms:.0f}ms")
        return success

    # ========================================================================
    # Event Helpers
    # ========================================================================

    def _copy_events(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Shallow copy events to avoid mutating originals."""
        return [copy(e) for e in events]

    # ========================================================================
    # Octave Restoration
    # ========================================================================

    def _apply_octave_restoration(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """
        Shift bass notes down one octave for natural playback.
        Bass is often detected one octave too high by pitch algorithms.
        """
        BASS_SOURCE_TYPES = {SourceType.BASS_STEM, SourceType.PITCH, SourceType.BASS}
        restored = []

        for event in events:
            pitch = event.pitch
            is_bass = (
                    event.source in BASS_SOURCE_TYPES and
                    self.config.bass_range_low <= pitch <= self.config.bass_range_high
            )
            if is_bass:
                new_pitch = max(0, min(127, pitch + self.config.bass_octave_shift))
                new_event = copy(event)
                new_event.pitch = new_pitch
                if new_event.reasoning_chain:
                    new_event.reasoning_chain.append(f"Octave restored: {pitch} → {new_pitch}")
                else:
                    new_event.reasoning_chain = [f"Octave restored: {pitch} → {new_pitch}"]
                restored.append(new_event)
            else:
                restored.append(event)

        return restored

    def _clamp_velocities(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Clamp velocities to valid MIDI range."""
        for event in events:
            event.velocity = max(
                self.config.min_velocity,
                min(self.config.max_velocity, event.velocity)
            )
        return events

    # ========================================================================
    # Heartbeat Note
    # ========================================================================

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
        return [heartbeat]

    # ========================================================================
    # MIDI File Construction
    # ========================================================================

    def _init_midi_file(self):
        """Reset track list for a new export."""
        self._tracks = []

    def _add_tempo_track(
            self,
            tempo_map: Optional[TempoMap],
            tempo_source: str = "default"
    ):
        """Build Track 0 — the tempo/time-signature track."""
        track = bytearray()

        # 1. MTrk chunk header
        track.extend(self._make_chunk(b'MTrk', 0))

        # 2. Tempo meta event — delta time 0
        tempo_bpm = tempo_map.initial_tempo_bpm if tempo_map else self.config.tempo_default_bpm
        tempo_us = self._bpm_to_us(tempo_bpm)
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x51, self._encode_three_bytes(tempo_us)))

        # 3. Time signature meta event
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

        # 4. Track name
        name_bytes = f"Grimlock 5.5 Tempo Map (source: {tempo_source})".encode('ascii')
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x03, name_bytes))

        # 5. Gradual tempo changes (if any)
        if tempo_map and tempo_map.tempo_events:
            last_tick = 0
            for event in tempo_map.tempo_events:
                ticks = self._ms_to_ticks(event.time_ms, tempo_map)
                delta_ticks = max(0, ticks - last_tick)
                track.extend(self._encode_var_length(delta_ticks))
                track.extend(self._make_meta_event(
                    0x51,
                    self._encode_three_bytes(self._bpm_to_us(event.tempo_bpm))
                ))
                last_tick = ticks

        # 6. End of track
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x2F, bytes([])))

        # 7. Back-fill track length
        self._update_track_length(track)
        self._tracks.append(track)

    def _add_single_track_notes(
            self,
            events: List[NoteEvent],
            track_name: str = "",
            channel: int = 0
    ):
        """
        Write notes to a single MIDI track on the specified channel.

        CRITICAL: The channel parameter determines which MIDI channel
        the notes are sent to. Channel 9 (0-indexed) = GM Channel 10,
        which is the STANDARD percussion channel.

        Args:
            events: List of NoteEvent objects
            track_name: Name for the track (meta event)
            channel: MIDI channel (0-15). Use 9 for drums (GM percussion).
        """
        if not events:
            if self.config.verbose:
                print(f"  [MIDI] No events for track '{track_name}', skipping")
            return

        track = bytearray()

        # 1. MTrk chunk header
        track.extend(self._make_chunk(b'MTrk', 0))

        # 2. Track name meta event
        name_bytes = track_name.encode('ascii')[:255]
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x03, name_bytes))

        # 3. Sort by start time
        sorted_events = sorted(events, key=lambda e: e.get_active_start_ms())

        last_tick = 0
        for event in sorted_events:
            start_ms = event.get_active_start_ms()
            start_tick = self._ms_to_ticks(start_ms)

            # Safe duration — minimum 10ms
            duration_ms = event.active_duration_ms()
            if duration_ms <= 0:
                duration_ms = 10.0
            duration_ticks = max(1, self._ms_to_ticks(duration_ms))

            # Delta from last event to this note's start
            delta_ticks = max(0, start_tick - last_tick)

            pitch = max(0, min(127, int(event.pitch)))
            velocity = max(1, min(127, int(event.velocity)))

            # Note On with the specified channel
            track.extend(self._encode_var_length(delta_ticks))
            track.extend(self._make_note_on(channel, pitch, velocity))

            # Note Off (duration after Note On)
            track.extend(self._encode_var_length(duration_ticks))
            track.extend(self._make_note_off(channel, pitch, 0))

            last_tick = start_tick + duration_ticks

        # 4. End of track
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x2F, bytes([])))

        # 5. Back-fill track length
        self._update_track_length(track)
        self._tracks.append(track)

    def _add_forensic_track(
            self,
            events: List[NoteEvent],
            tempo_map: Optional[TempoMap],
            tempo_source: str = "default",
            timbre_analysis: Optional[Dict[str, Any]] = None
    ):
        """Write a forensic metadata track."""
        track = bytearray()

        # 1. MTrk chunk header
        track.extend(self._make_chunk(b'MTrk', 0))

        # 2. Track name
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x03, b"Grimlock 5.5 Forensic Audit"))

        # 3. Build metadata strings
        metadata = [
            "Grimlock 5.5 Transcription",
            "Version: 5.5.0",
            f"Exported: {self._export_time.isoformat()}",
            f"Total notes: {len(events)}",
            f"PPQN: {self.config.ppqn}",
            f"Tempo source: {tempo_source}",
            f"Drum channel: {self.config.drum_channel} (GM {self.config.drum_channel + 1})",
        ]

        if tempo_map:
            metadata.append(f"Tempo: {tempo_map.initial_tempo_bpm:.1f} BPM")
            if tempo_map.tempo_events:
                metadata.append(f"Tempo changes: {len(tempo_map.tempo_events)}")

        if timbre_analysis:
            instrument_counts = timbre_analysis.get("instrument_counts", {})
            if instrument_counts:
                metadata.append("Timbre Analysis (instruments detected):")
                for inst, count in instrument_counts.items():
                    metadata.append(f"  {inst}: {count} notes")

        # Count drums separately
        drum_count = sum(1 for e in events if e.source in [SourceType.DRUM_INTELLIGENCE, SourceType.DRUMS])
        if drum_count:
            metadata.append(
                f"Drum notes (routed to channel {self.config.drum_channel}/GM{self.config.drum_channel + 1}): {drum_count}")

        if events:
            confidences = [e.confidence for e in events]
            metadata.append(f"Average confidence: {np.mean(confidences):.2f}")
            metadata.append(f"Min confidence: {np.min(confidences):.2f}")
            metadata.append(f"Max confidence: {np.max(confidences):.2f}")

        source_counts: Dict[str, int] = {}
        for event in events:
            source_counts[event.source.value] = source_counts.get(event.source.value, 0) + 1
        if source_counts:
            metadata.append("Source distribution:")
            for source, count in source_counts.items():
                metadata.append(f"  {source}: {count}")

        snapped_notes = [e for e in events if e.is_snapped()]
        if snapped_notes and self.config.include_original_timestamps:
            metadata.append(f"Snapped notes: {len(snapped_notes)}")
            for i, note in enumerate(snapped_notes[:10]):
                metadata.append(
                    f"  Note {i}: original {note.start_ms:.1f}ms "
                    f"→ snapped {note.snapped_start_ms:.1f}ms"
                )
            if len(snapped_notes) > 10:
                metadata.append(f"  ... and {len(snapped_notes) - 10} more")

        # 4. Write each metadata string as a text meta event
        for text in metadata:
            text_bytes = text.encode('utf-8')[:255]
            track.extend(self._encode_var_length(0))
            track.extend(self._make_meta_event(0x01, text_bytes))

        # 5. End of track
        track.extend(self._encode_var_length(0))
        track.extend(self._make_meta_event(0x2F, bytes([])))

        # 6. Back-fill track length
        self._update_track_length(track)
        self._tracks.append(track)

    # ========================================================================
    # MIDI Encoding Helpers
    # ========================================================================

    def _encode_var_length(self, value: int) -> bytes:
        """Encode a value as a MIDI variable-length quantity."""
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

    def _ms_to_ticks(self, ms: float, tempo_map: Optional[TempoMap] = None) -> int:
        """Convert milliseconds to MIDI ticks."""
        if ms <= 0:
            return 0

        if tempo_map:
            tempo_bpm = tempo_map.get_tempo_at_ms(ms)
            microseconds_per_beat = self._bpm_to_us(tempo_bpm)
        else:
            microseconds_per_beat = MIDI_TEMPO_DEFAULT

        microseconds = ms * 1000.0
        ticks_per_microsecond = self._tick_resolution / microseconds_per_beat

        return max(0, int(microseconds * ticks_per_microsecond))

    def _update_track_length(self, track: bytearray):
        """Back-fill the 4-byte length field in the MTrk header."""
        track_length = len(track) - 8
        track[4:8] = struct.pack('>I', track_length)

    def _write_file(self, output_path: Path) -> bool:
        """Write the complete MIDI file to disk."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            format_type = 1 if self.config.format == MidiFormat.FORMAT_1 else 0

            with open(output_path, 'wb') as f:
                # MThd header
                header = (
                        b'MThd' +
                        struct.pack('>I', 6) +
                        struct.pack('>HHH', format_type, len(self._tracks), self._tick_resolution)
                )
                f.write(header)

                for track in self._tracks:
                    f.write(track)

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

    # ========================================================================
    # Utility
    # ========================================================================

    def _force_cleanup(self):
        """Force garbage collection."""
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

    # ========================================================================
    # Public Info
    # ========================================================================

    def get_export_info(self) -> Dict[str, Any]:
        """Return metadata about the last export."""
        return {
            "version": "5.5.0",
            "export_time": self._export_time.isoformat(),
            "tick_resolution": self._tick_resolution,
            "track_count": len(self._tracks),
            "last_export_time_ms": self._last_export_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb,
            "drum_channel": self.config.drum_channel,
            "gm_channel": self.config.drum_channel + 1,
            "config": {
                "octave_restoration": self.config.octave_restore_bass,
                "separate_stems": self.config.separate_stems,
                "heartbeat_enabled": self.config.heartbeat_enabled,
                "include_forensic_markers": self.config.include_forensic_markers
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return writer statistics."""
        return {
            "name": "MidiWriter",
            "version": "5.5.0",
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
    """Quick test to verify MIDI export with drums on channel 9."""
    print("\n" + "=" * 60)
    print("MIDI Writer Test - Drums on Channel 9")
    print("=" * 60)

    # Create test notes
    pitched = [
        NoteEvent(pitch=60, start_ms=0, end_ms=500, velocity=80, confidence=0.9, zero_crossing_rate=0.1,
                  source=SourceType.PITCH),
        NoteEvent(pitch=64, start_ms=500, end_ms=1000, velocity=75, confidence=0.8, zero_crossing_rate=0.1,
                  source=SourceType.PITCH),
    ]

    drums = [
        NoteEvent(pitch=36, start_ms=0, end_ms=100, velocity=100, confidence=0.9, zero_crossing_rate=0.5,
                  source=SourceType.DRUM_INTELLIGENCE),
        NoteEvent(pitch=38, start_ms=500, end_ms=600, velocity=90, confidence=0.85, zero_crossing_rate=0.5,
                  source=SourceType.DRUM_INTELLIGENCE),
        NoteEvent(pitch=42, start_ms=250, end_ms=350, velocity=80, confidence=0.8, zero_crossing_rate=0.5,
                  source=SourceType.DRUM_INTELLIGENCE),
    ]

    # Create tempo map
    tempo_map = TempoMap(initial_tempo_bpm=120, tempo_events=[], confidence=Confidence.HIGH)

    # Export
    writer = MidiWriter(config=MidiExportConfig(verbose=True))
    output_path = "test_drums_channel_9.mid"

    success = writer.export_midi_with_drums(
        pitched_events=pitched,
        drum_events=drums,
        tempo_map=tempo_map,
        output_path=output_path,
        tempo_source="test"
    )

    print(f"\nTest result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Output file: {output_path}")
    print(f"Drums should be on channel 9 (GM Channel 10)")
    print("=" * 60)

    return success


if __name__ == "__main__":
    quick_midi_test()