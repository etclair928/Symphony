# =================================================================
# MODULE: __init__.py (root)
# DESCRIPTION: Root package initializer for Grimlock 5.6.1.
# Exports main public interfaces and version information.
#
# Authored by: {DeepSeek} - Complete package structure (2026-05-09)
# =================================================================

"""
Grimlock 5.6.1 - Audio Transcription with Epistemic Veto
=======================================================

Grimlock is an intelligent audio transcription system that applies
epistemic veto and relational physics to produce high-quality
MIDI transcriptions from audio.

Key Features:
- Law of Resource Survival: Memory-efficient processing
- Law of Unidirectional Integrity: No circular dependencies
- Law of Scribe's Truth: Confidence must be earned
- Relational Physics: Groove-aware quantization
- Epistemic Veto: Any qualified witness can declare falsehood
- Non-Destructive Audit: All transformations reversible

Usage:
    from grimlock_5 import transcribe, GrimlockPipeline

    # Simple transcription
    result = transcribe("input.wav", "output.mid")

    # Advanced usage
    pipeline = GrimlockPipeline()
    result = pipeline.transcribe("input.wav")
"""

__version__ = "5.6.1"
__author__ = "Grimlock Team"
__all__ = [
    "transcribe",
    "GrimlockPipeline",
    "MusicBox",
    "Scribe",
    "MemoryGuardian",
    "NoteEvent",
    "GrooveField",
    "TempoMap",
    "Confidence",
    "SourceType"
]

# Version check
import sys

if sys.version_info < (3, 8):
    raise RuntimeError("Grimlock 5.0 requires Python 3.8 or higher")


def transcribe(
        audio_path: str,
        output_path: str = None,
        export_midi: bool = True,
        export_json: bool = False,
        **kwargs
):
    """
    Quick transcription function.

    Args:
        audio_path: Path to input audio file
        output_path: Path for output file (auto-generated if None)
        export_midi: Export MIDI file
        export_json: Export JSON with forensic data
        **kwargs: Additional pipeline configuration

    Returns:
        Dictionary with transcription results
    """
    from orchestration.pipeline import GrimlockPipeline
    from core.order_types import ExportOptions

    # Create export options
    export_opts = ExportOptions(
        midi_ppqn=kwargs.get("midi_ppqn", 480),
        json_pretty=kwargs.get("json_pretty", True),
        include_original_timestamps=True,
        include_snapped_timestamps=True,
        include_veto_history=True,
        octave_restoration_enabled=kwargs.get("octave_restore", True)
    )

    # Create pipeline
    pipeline = GrimlockPipeline(export_options=export_opts)

    # Run transcription
    result = pipeline.transcribe(audio_path, output_path)

    return result


# Export main classes for easier imports
from orchestration.pipeline import GrimlockPipeline
from orchestration.music_box import MusicBox
from orchestration.scribe import Scribe
from memory.guardian import MemoryGuardian
from memory.arena import MemoryArena
from core.order_types import (
    NoteEvent, GrooveField, TempoMap, Confidence,
    SourceType, AudioContext, StageResult
)
from agents.detection.rhythm_engine import RhythmEngine
from agents.detection.pitch_intelligence import PitchIntelligence
from agents.detection.drum_intelligence import DrumIntelligence
from agents.analysis.groove_field import GrooveFieldAnalyzer
from agents.analysis.tempo_intelligence import TempoIntelligence
from agents.quantization.ritornello import RitornelloQuantizer
from export.midi_writer import MidiWriter
from export.json_writer import JsonWriter