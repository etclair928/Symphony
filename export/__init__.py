# =================================================================
# MODULE: export/__init__.py
# DESCRIPTION: Export module - MIDI and JSON writers for Grimlock 5.6.
#
# VERSION: 5.6.1
# UPDATED: 2026-06-03
#
# PHILOSOPHY:
#     Law of Non-Destructive Audit: All exported data preserves
#     both original and transformed versions for forensic audit.
#
#     Law of Scribe's Truth: Empty transcriptions receive a heartbeat
#     note (C4, velocity 10, 500ms) rather than an empty file.
#
# COMPONENTS:
#     - MidiWriter: MIDI file export with polyphony, tempo integration
#     - JsonWriter: JSON forensic export with streaming and validation
#
# UPDATES FOR 5.6.0:
#     - Fixed polyphony (timeline-based event sorting for chords)
#     - Fixed tempo integration (cumulative tick calculation)
#     - Added semantic instrument_family routing
#     - Added dynamic channel allocation (ChannelManager)
#     - Added Program Change messages for proper instrument playback
#     - Added streaming JSON export for large transcriptions (10k+ notes)
#     - Added note validation before export
#     - Enhanced confidence statistics (median, std, percentiles)
#
# 5 STRATEGIES INTEGRATION:
#     Strategy 2 (Disk-Backed): Write directly to disk
#     Strategy 3 (mmap for Audit): Human-readable formats for audit
#     Strategy 4 (GC Discipline): Clean up after export
#     Strategy 5 (Sparse Representations): Only export essential data
#
# USAGE:
#     from export import export_to_midi, export_to_json
#
#     # Export to MIDI with tempo source and timbre analysis
#     success = export_to_midi(
#         notes, "output.mid", tempo_map,
#         tempo_source="librosa",
#         timbre_analysis=timbre_data
#     )
#
#     # Export to JSON with forensic audit
#     success = export_to_json(
#         notes, "output.json", audio_context, groove_field,
#         tempo_source="librosa",
#         timbre_analysis=timbre_data
#     )
# =================================================================

from typing import Dict, List, Any, Optional
from pathlib import Path

# Import main classes from updated modules
from export.midi_writer import (
    MidiWriter,
    MidiExportConfig,
    MidiFormat,
    InstrumentFamily,
    GMProgram,
    ChannelManager,
    export_to_midi,
    export_midi_with_drums
)

from export.json_writer import (
    JsonWriter,
    JSONEncoder,
    ExportValidator,
    export_to_json
)

# Import core types needed for type hints
from core.order_types import (
    NoteEvent, TempoMap, AudioContext, GrooveField, ExportOptions
)

# Version identifier - Updated to 5.6.0
__version__ = "5.6.1"
__author__ = "DeepSeek"

# Module docstring
__doc__ = """
Grimlock 5.6 Export Module
===========================

Law of Non-Destructive Audit: The original signal is the only source of truth.
All exports preserve both original and transformed values for forensic audit.

Components:
-----------
MidiWriter (v5.6.0):
    MIDI file export with Grimlock 5.6 enhancements.

    NEW FEATURES:
    - ✅ Fixed polyphony: Timeline-based event sorting for chords
    - ✅ Fixed tempo integration: Cumulative tick calculation across segments
    - ✅ Semantic instrument_family routing (no fragile source detection)
    - ✅ Dynamic channel allocation (supports unlimited instrument types)
    - ✅ Program Change messages for proper instrument playback
    - ✅ Octave restoration with provenance (no double-shifting)

    Existing Features:
    - Octave restoration (optional, for bass notes)
    - Heartbeat note for empty transcriptions
    - Velocity clamping and scaling
    - Tempo map integration with gradual changes
    - Forensic metadata in MIDI markers
    - Multi-track export for stems
    - Configurable PPQN and format
    - Tempo source tracking (guided/librosa/rhythm/default)
    - Timbre analysis summary in forensic track

JsonWriter (v5.5.0):
    JSON forensic export with complete audit trail.

    NEW FEATURES:
    - ✅ Streaming export for large transcriptions (10k+ notes)
    - ✅ Note validation before export (repairs invalid data)
    - ✅ Enhanced confidence statistics (median, std, percentiles)
    - ✅ Instrument family serialization

    Existing Features:
    - Complete note data with original and snapped timestamps
    - Veto history and reasoning chain from Scribe
    - Groove field and tempo map export
    - Timbre analysis export (instruments detected)
    - MusicBox integration for forensic logs
    - Pretty print option for human readability
    - File size limiting to prevent OOM

5 Strategies Integration:
------------------------
Strategy 2 (Disk-Backed): Both writers write directly to disk
Strategy 3 (mmap for Audit): JSON and MIDI are human-readable/auditable
Strategy 4 (GC Discipline): Cleanup after export operations
Strategy 5 (Sparse Representations): Only export essential note metadata

Law Compliance:
--------------
- Law of Non-Destructive Audit: Original timestamps preserved
- Law of Scribe's Truth: Heartbeat notes for empty transcriptions
- Law of Resource Survival: Memory limits and file size limits
- Law of Unidirectional Integrity: Copies made before modifications

Usage Examples:
--------------
# Basic MIDI export
from export import export_to_midi

success = export_to_midi(notes, "output.mid", tempo_map)

# MIDI export with tempo source and timbre analysis
success = export_to_midi(
    notes, "output.mid", tempo_map,
    tempo_source="librosa",
    timbre_analysis=timbre_data
)

# Advanced MIDI export with configuration
from export import MidiWriter, MidiExportConfig

config = MidiExportConfig(
    separate_stems=True,
    octave_restore_bass=True,
    max_polyphony=64,
    verbose=True
)
writer = MidiWriter(config)
success = writer.export_midi(
    notes, tempo_map, "output.mid",
    track_name="My Track",
    tempo_source="guided",
    timbre_analysis=timbre_data
)

# Export with explicit drum routing
from export import export_midi_with_drums

success = export_midi_with_drums(
    pitched_events=melody_notes,
    drum_events=drum_notes,
    output_path="output.mid",
    tempo_map=tempo_map
)

# Basic JSON export
from export import export_to_json

success = export_to_json(
    notes, "output.json",
    audio_context=ctx,
    groove_field=groove,
    tempo_source="librosa",
    timbre_analysis=timbre_data
)

# Advanced JSON export with streaming
from export import JsonWriter, ExportOptions

options = ExportOptions(json_pretty=True, include_veto_history=True)
writer = JsonWriter(options)
# Automatically uses streaming for >10,000 notes
success = writer.export_json(
    notes, "output.json",
    audio_context=ctx,
    scribe_verdict=verdict,
    tempo_source="rhythm",
    timbre_analysis=timbre_data
)

# Combined export (MIDI + JSON)
from export import export_all

results = export_all(
    events=notes,
    base_path="output/transcription",
    tempo_map=tempo_map,
    audio_context=ctx,
    groove_field=groove,
    tempo_source="librosa",
    timbre_analysis=timbre_data
)
print(f"MIDI: {results['midi']}, JSON: {results['json']}")
"""

# All exports
__all__ = [
    # Version
    "__version__",

    # MidiWriter (v5.6.0)
    "MidiWriter",
    "MidiExportConfig",
    "MidiFormat",
    "InstrumentFamily",
    "GMProgram",
    "ChannelManager",
    "export_to_midi",
    "export_midi_with_drums",

    # JsonWriter (v5.5.0)
    "JsonWriter",
    "JSONEncoder",
    "ExportValidator",
    "export_to_json",

    # Core types
    "ExportOptions",
]


# ========================================================================
# Module Initialization Check
# ========================================================================

def _check_imports() -> bool:
    """Verify all export components are importable."""
    missing = []

    try:
        from export.midi_writer import MidiWriter, MidiExportConfig, MidiFormat
    except ImportError as e:
        missing.append(f"midi_writer: {e}")

    try:
        from export.json_writer import JsonWriter, ExportValidator
    except ImportError as e:
        missing.append(f"json_writer: {e}")

    if missing:
        import warnings
        warnings.warn(f"Some export components failed to import: {missing}", ImportWarning)
        return False

    return True


_IMPORTS_OK = _check_imports()

# ========================================================================
# Module Metadata
# ========================================================================

version_info = {
    "module": "export",
    "version": __version__,
    "imports_ok": _IMPORTS_OK,
    "components": ["midi_writer", "json_writer"],
    "features": [
        "tempo_source_tracking",
        "timbre_analysis_export",
        "polyphony_fixed",
        "tempo_integration",
        "instrument_family_routing",
        "dynamic_channel_allocation",
        "program_change_messages",
        "streaming_json_export",
        "note_validation",
        "enhanced_statistics"
    ],
    "midi_version": "5.6.0",
    "json_version": "5.5.0"
}


# ========================================================================
# Combined Export Functions
# ========================================================================

def export_all(
        events: List[NoteEvent],
        base_path: str,
        tempo_map: Optional[TempoMap] = None,
        audio_context: Optional[AudioContext] = None,
        groove_field: Optional[GrooveField] = None,
        scribe_verdict: Optional[Dict[str, Any]] = None,
        timbre_analysis: Optional[Dict[str, Any]] = None,
        tempo_source: str = "default",
        options: Optional[ExportOptions] = None,
        status_reporter: Optional[Any] = None,
        music_box: Optional[Any] = None,
        separate_drums: bool = False
) -> Dict[str, bool]:
    """
    Export to both MIDI and JSON formats.

    Args:
        events: List of NoteEvent objects
        base_path: Base path for output files (without extension)
        tempo_map: Tempo map for MIDI export
        audio_context: Audio context for JSON export
        groove_field: Groove field for JSON export
        scribe_verdict: Scribe verdict for JSON export
        timbre_analysis: Timbre intelligence results
        tempo_source: Source of tempo detection (guided/librosa/rhythm/default)
        options: Export options
        status_reporter: StatusReporter for progress
        music_box: MusicBox for logging
        separate_drums: If True, use export_midi_with_drums for drum separation

    Returns:
        Dictionary with 'midi' and 'json' success flags
    """
    base = Path(base_path)
    midi_path = base.with_suffix('.mid')
    json_path = base.with_suffix('.json')

    results = {}

    # Export MIDI - choose method based on separate_drums flag
    if separate_drums:
        # Separate pitched from drum events
        pitched_events = []
        drum_events = []

        from export.midi_writer import MidiWriter
        temp_writer = MidiWriter()

        for event in events:
            family = temp_writer._get_instrument_family(event)
            if family.value == "drums":
                drum_events.append(event)
            else:
                pitched_events.append(event)

        results['midi'] = export_midi_with_drums(
            pitched_events=pitched_events,
            drum_events=drum_events,
            output_path=str(midi_path),
            tempo_map=tempo_map,
            status_reporter=status_reporter,
            music_box=music_box,
            tempo_source=tempo_source
        )
    else:
        results['midi'] = export_to_midi(
            events=events,
            output_path=str(midi_path),
            tempo_map=tempo_map,
            options=options,
            status_reporter=status_reporter,
            music_box=music_box,
            tempo_source=tempo_source,
            timbre_analysis=timbre_analysis
        )

    # Export JSON
    results['json'] = export_to_json(
        events=events,
        output_path=str(json_path),
        audio_context=audio_context,
        groove_field=groove_field,
        tempo_map=tempo_map,
        scribe_verdict=scribe_verdict,
        timbre_analysis=timbre_analysis,
        tempo_source=tempo_source,
        options=options,
        status_reporter=status_reporter,
        music_box=music_box
    )

    return results


def export_simple(
        events: List[NoteEvent],
        output_path: str,
        tempo_map: Optional[TempoMap] = None,
        audio_context: Optional[AudioContext] = None,
        **kwargs
) -> bool:
    """
    Simple combined export (MIDI only by default, JSON optional).

    Args:
        events: List of NoteEvent objects
        output_path: Base output path (extension will be added)
        tempo_map: Tempo map for MIDI export
        audio_context: Audio context for JSON export
        **kwargs: Additional arguments:
            - export_json (bool): Also export JSON
            - tempo_source (str): Tempo detection source
            - timbre_analysis (dict): Timbre intelligence results
            - status_reporter: StatusReporter
            - music_box: MusicBox
            - options: ExportOptions
            - separate_drums (bool): Use drum separation

    Returns:
        True if at least one export succeeded
    """
    base = Path(output_path).with_suffix('')
    export_json = kwargs.get('export_json', False)
    tempo_source = kwargs.get('tempo_source', 'default')
    timbre_analysis = kwargs.get('timbre_analysis', None)
    status_reporter = kwargs.get('status_reporter', None)
    music_box = kwargs.get('music_box', None)
    options = kwargs.get('options', None)
    separate_drums = kwargs.get('separate_drums', False)

    results = {}

    # Export MIDI - choose method based on separate_drums flag
    if separate_drums:
        pitched_events = []
        drum_events = []

        from export.midi_writer import MidiWriter
        temp_writer = MidiWriter()

        for event in events:
            family = temp_writer._get_instrument_family(event)
            if family.value == "drums":
                drum_events.append(event)
            else:
                pitched_events.append(event)

        results['midi'] = export_midi_with_drums(
            pitched_events=pitched_events,
            drum_events=drum_events,
            output_path=str(base.with_suffix('.mid')),
            tempo_map=tempo_map,
            status_reporter=status_reporter,
            music_box=music_box,
            tempo_source=tempo_source
        )
    else:
        results['midi'] = export_to_midi(
            events=events,
            output_path=str(base.with_suffix('.mid')),
            tempo_map=tempo_map,
            tempo_source=tempo_source,
            timbre_analysis=timbre_analysis,
            options=options,
            status_reporter=status_reporter,
            music_box=music_box
        )

    # Optionally export JSON
    if export_json:
        results['json'] = export_to_json(
            events=events,
            output_path=str(base.with_suffix('.json')),
            audio_context=audio_context,
            tempo_map=tempo_map,
            timbre_analysis=timbre_analysis,
            tempo_source=tempo_source,
            options=options,
            status_reporter=status_reporter,
            music_box=music_box
        )

    return results.get('midi', False) or results.get('json', False)


def export_with_drum_routing(
        pitched_events: List[NoteEvent],
        drum_events: List[NoteEvent],
        output_path: str,
        tempo_map: Optional[TempoMap] = None,
        **kwargs
) -> bool:
    """
    Export with explicit drum routing to channel 9 (GM Channel 10).

    Args:
        pitched_events: List of pitched NoteEvent objects
        drum_events: List of drum NoteEvent objects
        output_path: Output MIDI file path
        tempo_map: Tempo map
        **kwargs: Additional arguments (tempo_source, status_reporter, music_box)

    Returns:
        True if export successful
    """
    tempo_source = kwargs.get('tempo_source', 'default')
    status_reporter = kwargs.get('status_reporter', None)
    music_box = kwargs.get('music_box', None)
    config = kwargs.get('config', None)

    return export_midi_with_drums(
        pitched_events=pitched_events,
        drum_events=drum_events,
        output_path=output_path,
        tempo_map=tempo_map,
        config=config,
        status_reporter=status_reporter,
        music_box=music_box,
        tempo_source=tempo_source
    )


# Add to __all__
__all__.extend(["export_all", "export_simple", "export_with_drum_routing"])


# ========================================================================
# Quick Test Function
# ========================================================================

def _quick_test():
    """Quick test for export module."""
    from core.order_types import NoteEvent, SourceType, TempoMap, Confidence

    print("\n" + "=" * 70)
    print("GRIMLOCK 5.6 - EXPORT MODULE")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")
    print(f"JSON Version: {version_info['json_version']}")
    print(f"MIDI Version: {version_info['midi_version']}")

    # Create test events
    events = [
        NoteEvent(
            pitch=60, start_ms=0, end_ms=500, velocity=80,
            confidence=0.9, zero_crossing_rate=0.0,
            source=SourceType.PITCH,
            reasoning_chain=["Test note 1"]
        ),
        NoteEvent(
            pitch=64, start_ms=0, end_ms=500, velocity=90,  # Same time = chord
            confidence=0.95, zero_crossing_rate=0.0,
            source=SourceType.PITCH,
            reasoning_chain=["Test note 2"]
        ),
        NoteEvent(
            pitch=57, start_ms=600, end_ms=1200, velocity=85,
            confidence=0.85, zero_crossing_rate=0.0,
            source=SourceType.BASS_STEM,
            reasoning_chain=["Bass test note"]
        ),
        NoteEvent(
            pitch=36, start_ms=0, end_ms=200, velocity=100,
            confidence=0.9, zero_crossing_rate=0.5,
            source=SourceType.DRUM_INTELLIGENCE,  # Will be routed to channel 9
            reasoning_chain=["Kick drum"]
        )
    ]

    tempo_map = TempoMap(initial_tempo_bpm=120.0, confidence=Confidence.HIGH)

    # Mock timbre analysis
    mock_timbre = {
        "instrument_counts": {"piano": 45, "bass": 25, "drums": 30},
        "total_analyzed": 100,
        "entities": []
    }

    print("\nAvailable Components:")
    print("  - MidiWriter v5.6.0: MIDI export with polyphony + tempo integration")
    print("  - JsonWriter v5.5.0: JSON export with streaming + validation")
    print("  - ChannelManager: Dynamic channel allocation")
    print("  - ExportValidator: Note validation before export")

    print("\n" + "=" * 70)
    print("Quick Test:")
    print("=" * 70)

    # Test MIDI export
    print("\n1. Testing MIDI export with polyphony (chord at time 0)...")
    midi_success = export_to_midi(
        events=events,
        output_path="/tmp/test_export.mid",
        tempo_map=tempo_map,
        tempo_source="librosa",
        timbre_analysis=mock_timbre
    )
    print(f"   MIDI export: {'✓ SUCCESS' if midi_success else '✗ FAILED'}")

    # Test export with drum routing
    print("\n2. Testing export with explicit drum routing to channel 9...")
    pitched = [e for e in events if e.source != SourceType.DRUM_INTELLIGENCE]
    drums = [e for e in events if e.source == SourceType.DRUM_INTELLIGENCE]

    drum_success = export_midi_with_drums(
        pitched_events=pitched,
        drum_events=drums,
        output_path="/tmp/test_drums.mid",
        tempo_map=tempo_map,
        tempo_source="librosa"
    )
    print(f"   Drum routing export: {'✓ SUCCESS' if drum_success else '✗ FAILED'}")

    # Test JSON export
    print("\n3. Testing JSON export with validation...")
    json_success = export_to_json(
        events=events,
        output_path="/tmp/test_export.json",
        tempo_source="librosa",
        timbre_analysis=mock_timbre
    )
    print(f"   JSON export: {'✓ SUCCESS' if json_success else '✗ FAILED'}")

    # Test combined export
    print("\n4. Combined export with all features...")
    results = export_all(
        events=events,
        base_path="/tmp/test_export_all",
        tempo_map=tempo_map,
        tempo_source="rhythm",
        timbre_analysis=mock_timbre,
        separate_drums=True
    )
    print(f"   MIDI: {'✓' if results['midi'] else '✗'}")
    print(f"   JSON: {'✓' if results['json'] else '✗'}")

    # Cleanup
    import os
    test_files = [
        "/tmp/test_export.mid", "/tmp/test_export.json",
        "/tmp/test_drums.mid",
        "/tmp/test_export_all.mid", "/tmp/test_export_all.json",
    ]
    for f in test_files:
        if os.path.exists(f):
            os.unlink(f)
            print(f"   Cleaned: {f}")

    print("\n" + "=" * 70)
    print(f"Export module ready (v{__version__}).")
    print("=" * 70)
    return True


# ========================================================================
# Standalone Test Runner
# ========================================================================

if __name__ == "__main__":
    _quick_test()