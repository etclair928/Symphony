# =================================================================
# MODULE: export/__init__.py
# DESCRIPTION: Export module - MIDI and JSON writers for Grimlock 5.4.
#
# VERSION: 5.4
# UPDATED: 2026-05-17
#
# PHILOSOPHY:
#     Law of Non-Destructive Audit: All exported data preserves
#     both original and transformed versions for forensic audit.
#
#     Law of Scribe's Truth: Empty transcriptions receive a heartbeat
#     note (C4, velocity 10, 500ms) rather than an empty file.
#
# COMPONENTS:
#     - MidiWriter: MIDI file export with octave restoration
#     - JsonWriter: JSON forensic export with complete audit trail
#
# UPDATES FOR 5.4:
#     - Added timbre analysis to export functions
#     - Added tempo source tracking
#     - Updated version to 5.4.0
#     - Added tempo_source parameter to export functions
#     - Added timbre_analysis parameter to export functions
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

# Import main classes
from export.midi_writer import (
    MidiWriter,
    MidiExportConfig,
    MidiFormat,
    export_to_midi
)

from export.json_writer import (
    JsonWriter,
    JSONEncoder,
    export_to_json
)

# Import core types needed for type hints
from core.order_types import (
    NoteEvent, TempoMap, AudioContext, GrooveField, ExportOptions
)

# Version identifier
__version__ = "5.4.0"
__author__ = "DeepSeek"

# Module docstring
__doc__ = """
Grimlock 5.4 Export Module
===========================

Law of Non-Destructive Audit: The original signal is the only source of truth.
All exports preserve both original and transformed values for forensic audit.

Components:
-----------
MidiWriter:
    MIDI file export with Grimlock 5.4 enhancements.

    Features:
    - Octave restoration (optional, for bass notes)
    - Heartbeat note for empty transcriptions
    - Velocity clamping and scaling
    - Tempo map integration with gradual changes
    - Forensic metadata in MIDI markers
    - Multi-track export for stems
    - Configurable PPQN and format
    - Tempo source tracking (guided/librosa/rhythm/default)
    - Timbre analysis summary in forensic track

JsonWriter:
    JSON forensic export with complete audit trail.

    Features:
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

config = MidiExportConfig(separate_stems=True, octave_restore_bass=True)
writer = MidiWriter(config)
success = writer.export_midi(
    notes, tempo_map, "output.mid",
    track_name="My Track",
    tempo_source="guided",
    timbre_analysis=timbre_data
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

# Advanced JSON export
from export import JsonWriter, ExportOptions

options = ExportOptions(json_pretty=True, include_veto_history=True)
writer = JsonWriter(options)
success = writer.export_json(
    notes, "output.json",
    audio_context=ctx,
    scribe_verdict=verdict,
    tempo_source="rhythm",
    timbre_analysis=timbre_data
)
"""

# All exports
__all__ = [
    # Version
    "__version__",

    # MidiWriter
    "MidiWriter",
    "MidiExportConfig",
    "MidiFormat",
    "export_to_midi",

    # JsonWriter
    "JsonWriter",
    "JSONEncoder",
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
        from export.midi_writer import MidiWriter
    except ImportError as e:
        missing.append(f"midi_writer: {e}")

    try:
        from export.json_writer import JsonWriter
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
    "features": ["tempo_source_tracking", "timbre_analysis_export"]
}


# ========================================================================
# Combined Export Function
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
        music_box: Optional[Any] = None
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

    Returns:
        Dictionary with 'midi' and 'json' success flags
    """
    base = Path(base_path)
    midi_path = base.with_suffix('.mid')
    json_path = base.with_suffix('.json')

    results = {}

    # Export MIDI
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


# Add to __all__
__all__.append("export_all")

# ========================================================================
# Simplified Combined Export
# ========================================================================

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
        **kwargs: Additional arguments (export_json, tempo_source, timbre_analysis, etc.)

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

    results = {}

    # Always export MIDI
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


# Add to __all__
__all__.append("export_simple")

# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    from core.order_types import NoteEvent, SourceType, TempoMap, Confidence

    print("\n" + "=" * 70)
    print("GRIMLOCK 5.4 - EXPORT MODULE")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")

    # Create test events
    events = [
        NoteEvent(
            pitch=60, start_ms=0, end_ms=500, velocity=80,
            confidence=0.9, zero_crossing_rate=0.0,
            source=SourceType.PITCH,
            reasoning_chain=["Test note 1"]
        ),
        NoteEvent(
            pitch=64, start_ms=600, end_ms=1000, velocity=90,
            confidence=0.95, zero_crossing_rate=0.0,
            source=SourceType.PITCH,
            reasoning_chain=["Test note 2"]
        ),
        NoteEvent(
            pitch=57, start_ms=1200, end_ms=1800, velocity=85,
            confidence=0.85, zero_crossing_rate=0.0,
            source=SourceType.BASS_STEM,
            reasoning_chain=["Bass test note"]
        )
    ]

    tempo_map = TempoMap(initial_tempo_bpm=120.0, confidence=Confidence.HIGH)

    # Mock timbre analysis
    mock_timbre = {
        "instrument_counts": {"piano": 45, "bass": 25},
        "total_analyzed": 70,
        "entities": []
    }

    print("\nAvailable Components:")
    print("  - MidiWriter: MIDI export with octave restoration + tempo source tracking")
    print("  - JsonWriter: JSON export with forensic audit + timbre analysis")

    print("\n" + "=" * 70)
    print("Quick Test:")
    print("=" * 70)

    # Test export
    print("\n1. Testing exports with tempo source and timbre analysis...")

    # MIDI export with tempo source
    midi_success = export_to_midi(
        events=events,
        output_path="/tmp/test_export.mid",
        tempo_map=tempo_map,
        tempo_source="librosa",
        timbre_analysis=mock_timbre
    )
    print(f"   MIDI export (tempo_source=librosa): {'✓' if midi_success else '✗'}")

    # JSON export with timbre analysis
    json_success = export_to_json(
        events=events,
        output_path="/tmp/test_export.json",
        tempo_source="librosa",
        timbre_analysis=mock_timbre
    )
    print(f"   JSON export (with timbre analysis): {'✓' if json_success else '✗'}")

    # Combined export
    print("\n2. Combined export with all features...")
    results = export_all(
        events=events,
        base_path="/tmp/test_export_all",
        tempo_map=tempo_map,
        tempo_source="rhythm",
        timbre_analysis=mock_timbre
    )
    print(f"   MIDI: {'✓' if results['midi'] else '✗'}")
    print(f"   JSON: {'✓' if results['json'] else '✗'}")

    # Simple export
    print("\n3. Simple export (MIDI only)...")
    simple_success = export_simple(
        events=events,
        output_path="/tmp/test_simple",
        tempo_map=tempo_map,
        tempo_source="guided",
        timbre_analysis=mock_timbre
    )
    print(f"   Simple export: {'✓' if simple_success else '✗'}")

    # Cleanup
    import os

    test_files = [
        "/tmp/test_export.mid", "/tmp/test_export.json",
        "/tmp/test_export_all.mid", "/tmp/test_export_all.json",
        "/tmp/test_simple.mid"
    ]
    for f in test_files:
        if os.path.exists(f):
            os.unlink(f)
            print(f"   Cleaned: {f}")

    print("\n" + "=" * 70)
    print("Export module ready (v5.4).")
    print("=" * 70)