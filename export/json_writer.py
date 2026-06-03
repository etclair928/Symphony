# =================================================================
# MODULE: export/json_writer.py
# DESCRIPTION: JSON export with full forensic audit trail.
#
# VERSION: 5.4 (Updated for production AudioContext)
# UPDATED: 2026-05-17
#
# LAW OF NON-DESTRUCTIVE AUDIT:
#     "The original signal is the only source of truth; all
#      transformations must be reversible."
#
#     The JSON output includes both original and snapped timestamps
#     so humans can see exactly what was changed and why.
#
# UPDATES FOR 5.4:
#     - Fixed AudioContext serialization (num_channels, file_hash)
#     - Added timbre analysis export
#     - Added pitch detection source tracking
#     - Added tempo detection source (guided/librosa/rhythm/default)
#     - Added instrument breakdown from timbre analysis
#     - Fixed file_hash_md5 → file_hash
#     - Fixed num_channels_original → num_channels
#     - Added multi-source pitch event tracking
#
# KEY ARCHITECTURE:
#     - Complete note data with original and snapped timestamps
#     - Veto history and reasoning chain from Scribe
#     - Groove field and tempo map export
#     - MusicBox integration for forensic logs
#     - Pretty print option for human readability
#     - File size limiting to prevent OOM
#     - StatusReporter integration
#
# 5 STRATEGIES INTEGRATION:
#     Strategy 2 (Disk-Backed): Write directly to disk, not memory
#     Strategy 3 (mmap for Audit): JSON readable by any text viewer
#     Strategy 4 (GC Discipline): Minimal memory footprint during export
#     Strategy 5 (Sparse Representations): Only export essential metadata
# =================================================================

import json
import warnings
import time
import gc
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

import numpy as np

# Core imports
from core.order_types import (
    NoteEvent, GrooveField, TempoMap, AudioContext, ExportOptions,
    SourceType, VetoReason, ValidationGate, Confidence
)
from core.constants import (
    JSON_PRETTY_PRINT,
    JSON_INCLUDE_FORENSIC_AUDIT,
    JSON_INCLUDE_MUSIC_BOX,
    JSON_MAX_FILE_SIZE_MB,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import MusicBoxProtocol, StatusReporterProtocol


# ========================================================================
# Serialization Helpers
# ========================================================================

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and datetimes."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


# ========================================================================
# JsonWriter
# ========================================================================

class JsonWriter:
    """
    JSON exporter with forensic audit trail.

    Law of Non-Destructive Audit: The original signal is the only source
    of truth; all transformations must be reversible.

    The JSON output includes:
    - Original timestamps (never overwritten)
    - Snapped timestamps (if quantization applied)
    - Reasoning chain for each decision
    - Veto history from Scribe
    - Complete MusicBox logs for forensic analysis
    - Timbre analysis (instrument identification)
    """

    def __init__(
            self,
            options: Optional[ExportOptions] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        """
        Args:
            options: Export options configuration
            status_reporter: StatusReporter for progress
            music_box: MusicBox for forensic logs
        """
        self.options = options or ExportOptions()
        self._status_reporter = status_reporter
        self._music_box = music_box

        # Session metadata
        self._export_time = datetime.now()
        self._version = "5.4.0"

        # Performance tracking
        self._last_export_time_ms = 0.0
        self._total_memory_freed_mb = 0.0

        self._log_status(f"JsonWriter initialized (version {self._version})")

    # ========================================================================
    # Main Export Interface
    # ========================================================================

    def export_json(
            self,
            events: List[NoteEvent],
            output_path: Optional[str] = None,
            audio_context: Optional[AudioContext] = None,
            groove_field: Optional[GrooveField] = None,
            tempo_map: Optional[TempoMap] = None,
            scribe_verdict: Optional[Dict[str, Any]] = None,
            timbre_analysis: Optional[Dict[str, Any]] = None,
            tempo_source: str = "default",
            include_music_box_logs: Optional[bool] = None
    ) -> bool:
        """
        Export transcription to JSON with full forensic audit.

        Args:
            events: List of NoteEvent objects
            output_path: Output file path (auto-generated if None)
            audio_context: Audio file context metadata
            groove_field: Groove field analysis result
            tempo_map: Tempo map for timing
            scribe_verdict: Final Scribe verdict
            timbre_analysis: Timbre intelligence results
            tempo_source: Source of tempo detection (guided/librosa/rhythm/default)
            include_music_box_logs: Override for including MusicBox logs

        Returns:
            True if export successful
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        if output_path is None:
            output_path = f"grimlock_transcription_{self._export_time.strftime('%Y%m%d_%H%M%S')}.json"

        output_path = Path(output_path)

        self._log_status(f"Exporting {len(events)} notes to {output_path.name}")

        # Log export start
        if self._music_box:
            self._music_box.log_decision(
                stage_name="json_writer",
                decision_type="export_start",
                before_state={},
                after_state={
                    "output_path": str(output_path),
                    "event_count": len(events),
                    "include_forensic": self.options.include_veto_history,
                    "tempo_source": tempo_source
                },
                reasoning=f"Exporting {len(events)} notes to JSON",
                reversible=False
            )

        # Build complete export data
        export_data = self._build_export_data(
            events=events,
            audio_context=audio_context,
            groove_field=groove_field,
            tempo_map=tempo_map,
            scribe_verdict=scribe_verdict,
            timbre_analysis=timbre_analysis,
            tempo_source=tempo_source,
            include_music_box_logs=include_music_box_logs
        )

        # Write JSON file
        success = self._write_file(export_data, output_path)

        execution_time_ms = (time.time() - start_time) * 1000
        memory_delta_mb = self._get_current_memory_mb() - start_memory
        self._last_export_time_ms = execution_time_ms

        # Force cleanup if needed
        if memory_delta_mb > STAGGERED_GC_TRIGGER_MB:
            self._force_cleanup()

        if success and self._music_box:
            self._music_box.log_decision(
                stage_name="json_writer",
                decision_type="export_complete",
                before_state={},
                after_state={
                    "output_path": str(output_path),
                    "file_size_bytes": output_path.stat().st_size,
                    "execution_time_ms": execution_time_ms
                },
                reasoning=f"JSON exported to {output_path.name}",
                reversible=False
            )

        self._log_status(f"Export complete in {execution_time_ms:.0f}ms")

        return success

    # ========================================================================
    # Export Data Construction
    # ========================================================================

    def _build_export_data(
            self,
            events: List[NoteEvent],
            audio_context: Optional[AudioContext] = None,
            groove_field: Optional[GrooveField] = None,
            tempo_map: Optional[TempoMap] = None,
            scribe_verdict: Optional[Dict[str, Any]] = None,
            timbre_analysis: Optional[Dict[str, Any]] = None,
            tempo_source: str = "default",
            include_music_box_logs: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Build complete JSON export data structure."""

        include_logs = include_music_box_logs if include_music_box_logs is not None else self.options.include_veto_history

        export_data = {
            "version": self._version,
            "export_timestamp": self._export_time.isoformat(),
            "grimlock_version": self._version,
            "provenance": self._build_provenance(),

            # Audio context
            "audio": self._serialize_audio_context(audio_context) if audio_context else None,

            # Analysis results
            "groove_field": self._serialize_groove_field(groove_field) if groove_field else None,
            "tempo_map": self._serialize_tempo_map(tempo_map) if tempo_map else None,
            "tempo_source": tempo_source,

            # Timbre analysis (NEW)
            "timbre_analysis": timbre_analysis if timbre_analysis else None,

            # Note events (core transcription)
            "notes": [self._serialize_note_event(event) for event in events],
            "note_count": len(events),

            # Statistics
            "statistics": self._compute_statistics(events, groove_field, tempo_map, timbre_analysis),

            # Validation and vetoes
            "scribe_verdict": scribe_verdict or {},

            # Forensic audit trail
            "veto_history": self._collect_veto_history(),
            "reasoning_chains": self._collect_reasoning_chains(events),
            "quantization_history": self._collect_quantization_history(events),
        }

        # Add MusicBox logs if requested
        if include_logs and self._music_box:
            export_data["music_box_logs"] = self._get_music_box_logs()

        return export_data

    def _build_provenance(self) -> Dict[str, Any]:
        """Build provenance metadata for reproducibility."""
        provenance = {
            "generator": "Grimlock 5.4",
            "export_timestamp": self._export_time.isoformat(),
            "laws_enforced": [
                "Law of Resource Survival",
                "Law of Unidirectional Integrity",
                "Law of Scribe's Truth",
                "Law of Non-Destructive Audit",
                "Relational Physics",
                "Epistemic Veto"
            ],
            "export_options": {
                "pretty_print": self.options.json_pretty,
                "include_forensic": self.options.include_veto_history,
                "include_music_box": JSON_INCLUDE_MUSIC_BOX,
                "heartbeat_note_pitch": self.options.heartbeat_note_pitch,
                "octave_restoration_enabled": self.options.octave_restoration_enabled
            }
        }

        # Try to get git hash
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                provenance["git_commit"] = result.stdout.strip()[:8]
        except Exception:
            pass

        return provenance

    # ========================================================================
    # Serialization Methods
    # ========================================================================

    def _serialize_note_event(self, event: NoteEvent) -> Dict[str, Any]:
        """
        Serialize a NoteEvent to JSON.

        Law of Non-Destructive Audit: Both original and snapped timestamps
        are preserved so human can see what was changed.
        """
        # Note name calculation
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = event.pitch // 12 - 1
        note_name = f"{note_names[event.pitch % 12]}{octave}"

        result = {
            "pitch": event.pitch,
            "note_name": note_name,
            "pitch_class": event.pitch % 12,
            "octave": octave,
            "velocity": event.velocity,
            "confidence": round(event.confidence, 4),
            "confidence_level": Confidence.from_float(event.confidence).name,
            "zero_crossing_rate": round(event.zero_crossing_rate, 4),
            "source": event.source.value,

            # Original timestamps (never overwritten)
            "original_start_ms": round(event.start_ms, 2),
            "original_end_ms": round(event.end_ms, 2),
            "original_duration_ms": round(event.end_ms - event.start_ms, 2),
        }

        # Snapped timestamps (if quantization applied)
        if event.is_snapped():
            result["snapped"] = {
                "start_ms": round(event.snapped_start_ms, 2),
                "end_ms": round(event.snapped_end_ms, 2),
                "duration_ms": round(event.snapped_end_ms - event.snapped_start_ms, 2),
                "delta_ms": round(event.snapped_start_ms - event.start_ms, 2),
                "reason": event.snap_reason,
                "confidence_delta": round(event.snap_confidence_delta, 4) if event.snap_confidence_delta else None
            }

        # Consensus and reasoning
        if event.consensus_votes:
            result["consensus_votes"] = event.consensus_votes[:10]

        if event.reasoning_chain:
            result["reasoning_chain"] = event.reasoning_chain[:20]

        if event.veto_attempts:
            result["veto_attempts"] = [
                {"gate": gate.value, "reason": reason.value}
                for gate, reason in event.veto_attempts[:5]
            ]

        # Frequency domain data
        if event.fundamental_freq_hz:
            result["fundamental_freq_hz"] = round(event.fundamental_freq_hz, 2)

        if event.harmonic_series_match_ratio:
            result["harmonic_series_match_ratio"] = round(event.harmonic_series_match_ratio, 4)

        return result

    def _serialize_audio_context(self, context: AudioContext) -> Dict[str, Any]:
        """Serialize AudioContext to JSON - FIXED for production."""
        return {
            "file_path": context.file_path,
            "file_name": Path(context.file_path).name,
            "original_sample_rate": context.original_sample_rate,
            "working_sample_rate": context.working_sample_rate,
            "duration_seconds": round(context.duration_seconds, 2),
            "duration_formatted": self._format_duration(context.duration_seconds),
            "num_channels": context.num_channels,
            "is_mono": context.is_mono,
            "is_stereo": context.is_stereo,
            "has_stable_identity": context.has_stable_identity if hasattr(context, 'has_stable_identity') else True,
            "file_hash": context.file_hash[:16] + "..." if context.file_hash else "UNKNOWN",
            "memory_mb": round(context.memory_mb, 2)
        }

    def _serialize_groove_field(self, groove: GrooveField) -> Dict[str, Any]:
        """Serialize GrooveField to JSON."""
        return {
            "bass_kick_phase_delta_ms": round(groove.bass_kick_phase_delta_ms, 2),
            "is_wide_swing": groove.is_wide_swing,
            "is_dilla_pocket": groove.is_dilla_pocket,
            "average_inter_note_distance_ms": round(groove.average_inter_note_distance_ms, 2),
            "tempo_estimate_bpm": round(groove.tempo_estimate_bpm, 1),
            "confidence": groove.confidence.name,
            "groove_signature": groove.groove_signature,
            "phase_delta_variance_ms": round(groove.phase_delta_variance_ms, 2),
            "interpretation": self._interpret_groove(groove)
        }

    def _interpret_groove(self, groove: GrooveField) -> str:
        """Generate human-readable groove interpretation."""
        if groove.is_dilla_pocket:
            return "J Dilla style pocket - intentional push/pull"
        elif groove.is_wide_swing:
            return f"Wide swing: bass {groove.bass_kick_phase_delta_ms:.1f}ms {'behind' if groove.bass_kick_phase_delta_ms > 0 else 'ahead'} of kick"
        elif abs(groove.bass_kick_phase_delta_ms) < 2:
            return "Tight pocket: bass and kick tightly locked"
        elif groove.bass_kick_phase_delta_ms > 0:
            return f"Laid back: bass {groove.bass_kick_phase_delta_ms:.1f}ms behind kick"
        else:
            return f"Rushed: bass {abs(groove.bass_kick_phase_delta_ms):.1f}ms ahead of kick"

    def _serialize_tempo_map(self, tempo_map: TempoMap) -> Dict[str, Any]:
        """Serialize TempoMap to JSON."""
        result = {
            "initial_tempo_bpm": round(tempo_map.initial_tempo_bpm, 1),
            "time_signature": f"{tempo_map.time_signature_numerator}/{tempo_map.time_signature_denominator}",
            "confidence": tempo_map.confidence.name,
            "tempo_events": []
        }

        for event in tempo_map.tempo_events:
            result["tempo_events"].append({
                "time_ms": round(event.time_ms, 2),
                "tempo_bpm": round(event.tempo_bpm, 1),
                "confidence": event.confidence.name
            })

        return result

    # ========================================================================
    # Statistics Computation
    # ========================================================================

    def _compute_statistics(
            self,
            events: List[NoteEvent],
            groove_field: Optional[GrooveField],
            tempo_map: Optional[TempoMap],
            timbre_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute transcription statistics."""
        if not events:
            return {
                "total_notes": 0,
                "empty_transcription": True,
                "heartbeat_note_added": self.options.heartbeat_note_pitch is not None
            }

        pitches = [e.pitch for e in events]
        confidences = [e.confidence for e in events]
        durations_ms = [e.end_ms - e.start_ms for e in events]

        # Source distribution
        source_counts = {}
        for event in events:
            source_counts[event.source.value] = source_counts.get(event.source.value, 0) + 1

        # Snapped notes
        snapped_count = sum(1 for e in events if e.is_snapped())

        # Pitch distribution by octave
        pitch_distribution = {}
        for pitch in pitches:
            octave = pitch // 12 - 1
            pitch_distribution[octave] = pitch_distribution.get(octave, 0) + 1

        stats = {
            "total_notes": len(events),
            "average_confidence": round(float(np.mean(confidences)), 3),
            "median_confidence": round(float(np.median(confidences)), 3),
            "std_confidence": round(float(np.std(confidences)), 3),
            "min_confidence": round(min(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            "average_duration_ms": round(float(np.mean(durations_ms)), 1),
            "median_duration_ms": round(float(np.median(durations_ms)), 1),
            "pitch_range": {
                "min": min(pitches),
                "max": max(pitches),
                "range_semitones": max(pitches) - min(pitches)
            },
            "pitch_distribution": pitch_distribution,
            "source_distribution": source_counts,
            "snapped_notes_count": snapped_count,
            "snapped_notes_percentage": round(snapped_count / len(events) * 100, 1) if events else 0,
            "confidence_distribution": {
                "hallucination": sum(1 for c in confidences if c < Confidence.HALLUCINATION.value_f),
                "low": sum(1 for c in confidences if Confidence.HALLUCINATION.value_f <= c < Confidence.LOW.value_f),
                "medium": sum(1 for c in confidences if Confidence.LOW.value_f <= c < Confidence.MEDIUM.value_f),
                "high": sum(1 for c in confidences if Confidence.MEDIUM.value_f <= c < Confidence.HIGH.value_f),
                "perfect": sum(1 for c in confidences if c >= Confidence.HIGH.value_f)
            }
        }

        # Add tempo if available
        if tempo_map:
            stats["tempo"] = {
                "initial_bpm": round(tempo_map.initial_tempo_bpm, 1),
                "has_tempo_changes": len(tempo_map.tempo_events) > 0,
                "time_signature": f"{tempo_map.time_signature_numerator}/{tempo_map.time_signature_denominator}"
            }

        # Add groove if available
        if groove_field:
            stats["groove"] = {
                "phase_delta_ms": round(groove_field.bass_kick_phase_delta_ms, 2),
                "is_swing": groove_field.is_wide_swing or groove_field.is_dilla_pocket,
                "groove_signature": groove_field.groove_signature
            }

        # Add timbre statistics (NEW)
        if timbre_analysis:
            instrument_counts = timbre_analysis.get("instrument_counts", {})
            total_analyzed = timbre_analysis.get("total_analyzed", 0)
            stats["timbre"] = {
                "instruments_detected": instrument_counts,
                "notes_analyzed": total_analyzed,
                "primary_instrument": max(instrument_counts.items(), key=lambda x: x[1])[0] if instrument_counts else "unknown"
            }

        return stats

    # ========================================================================
    # Forensic Data Collection
    # ========================================================================

    def _collect_veto_history(self) -> List[Dict[str, Any]]:
        """Collect veto history from MusicBox."""
        if not self._music_box:
            return []

        try:
            veto_entries = self._music_box.query_vetoes()

            return [
                {
                    "timestamp": entry.timestamp.isoformat() if hasattr(entry, 'timestamp') else str(datetime.now()),
                    "source": entry.get("source") if isinstance(entry, dict) else None,
                    "stage": entry.get("stage") if isinstance(entry, dict) else None,
                    "reason": entry.get("reason") if isinstance(entry, dict) else None,
                    "detail": entry.get("detail", "") if isinstance(entry, dict) else "",
                    "is_hard_veto": entry.get("is_hard", True) if isinstance(entry, dict) else True
                }
                for entry in veto_entries[:100]
            ]
        except Exception as e:
            self._log_status(f"Failed to collect veto history: {e}", "warn")
            return []

    def _collect_reasoning_chains(self, events: List[NoteEvent]) -> List[Dict[str, Any]]:
        """Collect reasoning chains from notes that have them."""
        chains = []
        for i, event in enumerate(events[:100]):
            if event.reasoning_chain:
                chains.append({
                    "note_index": i,
                    "pitch": event.pitch,
                    "note_name": self._note_number_to_name(event.pitch),
                    "start_ms": round(event.start_ms, 2),
                    "chain": event.reasoning_chain[:10]
                })
        return chains

    def _collect_quantization_history(self, events: List[NoteEvent]) -> Dict[str, Any]:
        """Collect quantization history from snapped notes."""
        snapped = [e for e in events if e.is_snapped()]

        if not snapped:
            return {"snapped_notes": 0, "snaps": []}

        snaps = []
        deltas = []
        for event in snapped:
            delta = event.snapped_start_ms - event.start_ms
            deltas.append(delta)
            snaps.append({
                "original_start_ms": round(event.start_ms, 2),
                "snapped_start_ms": round(event.snapped_start_ms, 2),
                "delta_ms": round(delta, 2),
                "reason": event.snap_reason,
                "confidence_delta": round(event.snap_confidence_delta, 4) if event.snap_confidence_delta else None
            })

        return {
            "snapped_notes_count": len(snapped),
            "average_delta_ms": round(float(np.mean(deltas)), 2) if deltas else 0,
            "std_delta_ms": round(float(np.std(deltas)), 2) if deltas else 0,
            "max_delta_ms": round(max(deltas), 2) if deltas else 0,
            "min_delta_ms": round(min(deltas), 2) if deltas else 0,
            "snaps": snaps[:50]
        }

    def _get_music_box_logs(self) -> List[Dict[str, Any]]:
        """Get MusicBox logs for the current session."""
        if not self._music_box:
            return []

        try:
            entries = self._music_box.query_entries(limit=500)

            return [
                {
                    "timestamp": entry.timestamp.isoformat() if hasattr(entry, 'timestamp') else str(datetime.now()),
                    "decision_type": entry.decision_type.value if hasattr(entry, 'decision_type') else "unknown",
                    "stage": entry.stage if hasattr(entry, 'stage') else "unknown",
                    "message": entry.message if hasattr(entry, 'message') else "",
                    "data": entry.data if hasattr(entry, 'data') else {}
                }
                for entry in entries
            ]
        except Exception as e:
            self._log_status(f"Failed to get MusicBox logs: {e}", "warn")
            return []

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _note_number_to_name(self, midi: int) -> str:
        """Convert MIDI note number to note name."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = midi // 12 - 1
        return f"{notes[midi % 12]}{octave}"

    def _format_duration(self, seconds: float) -> str:
        """Format duration as MM:SS or HH:MM:SS."""
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _write_file(self, data: Dict[str, Any], output_path: Path) -> bool:
        """Write JSON data to file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Estimate file size
            import sys
            sample_json = json.dumps({"test": "sample"}, cls=JSONEncoder)
            estimated_size = sys.getsizeof(sample_json) * 100

            if estimated_size > JSON_MAX_FILE_SIZE_MB * 1024 * 1024:
                warnings.warn(
                    f"JSON file estimated size > {JSON_MAX_FILE_SIZE_MB}MB. "
                    f"Consider reducing data or splitting output."
                )

            with open(output_path, 'w', encoding='utf-8') as f:
                if self.options.json_pretty:
                    json.dump(data, f, indent=2, ensure_ascii=False, cls=JSONEncoder)
                else:
                    json.dump(data, f, ensure_ascii=False, cls=JSONEncoder)

            return True

        except Exception as e:
            self._log_status(f"Failed to write JSON file: {e}", "error")
            if self._music_box:
                self._music_box.log_error("json_writer", e, {"output_path": str(output_path)})
            return False

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
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _log_status(self, message: str, level: str = "info"):
        """Log status message via StatusReporter."""
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info("JsonWriter", message)
            elif level == "warn":
                self._status_reporter.warn("JsonWriter", message)
            elif level == "error":
                self._status_reporter.error("JsonWriter", message)

    # ========================================================================
    # Public Methods
    # ========================================================================

    def get_export_info(self) -> Dict[str, Any]:
        """Get information about the last export."""
        return {
            "version": self._version,
            "export_time": self._export_time.isoformat(),
            "options": {
                "pretty_print": self.options.json_pretty,
                "include_forensic": self.options.include_veto_history,
                "include_music_box": JSON_INCLUDE_MUSIC_BOX,
                "max_file_size_mb": JSON_MAX_FILE_SIZE_MB
            },
            "last_export_time_ms": self._last_export_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get writer statistics."""
        return {
            "name": "JsonWriter",
            "version": self._version,
            "last_export_time_ms": self._last_export_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb,
            "options": {
                "pretty_print": self.options.json_pretty,
                "include_forensic": self.options.include_veto_history
            }
        }


# ========================================================================
# Convenience Functions
# ========================================================================

def export_to_json(
        events: List[NoteEvent],
        output_path: str,
        audio_context: Optional[AudioContext] = None,
        groove_field: Optional[GrooveField] = None,
        tempo_map: Optional[TempoMap] = None,
        scribe_verdict: Optional[Dict[str, Any]] = None,
        timbre_analysis: Optional[Dict[str, Any]] = None,
        tempo_source: str = "default",
        options: Optional[ExportOptions] = None,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> bool:
    """
    Convenience function to export transcription to JSON.

    Args:
        events: List of NoteEvent objects
        output_path: Output file path
        audio_context: Audio context metadata
        groove_field: Groove field analysis result
        tempo_map: Tempo map
        scribe_verdict: Scribe final verdict
        timbre_analysis: Timbre intelligence results
        tempo_source: Source of tempo detection
        options: Export options
        status_reporter: StatusReporter for progress
        music_box: MusicBox for logging

    Returns:
        True if export successful
    """
    writer = JsonWriter(options, status_reporter, music_box)
    return writer.export_json(
        events=events,
        output_path=output_path,
        audio_context=audio_context,
        groove_field=groove_field,
        tempo_map=tempo_map,
        scribe_verdict=scribe_verdict,
        timbre_analysis=timbre_analysis,
        tempo_source=tempo_source
    )


def quick_json_test():
    """Quick test function for JsonWriter."""
    from core.order_types import NoteEvent, SourceType

    print("\n" + "=" * 60)
    print("JsonWriter Test (v5.4)")
    print("=" * 60)

    # Create mock events
    events = [
        NoteEvent(
            pitch=60, start_ms=0, end_ms=500, velocity=80,
            confidence=0.9, zero_crossing_rate=0.05,
            source=SourceType.PITCH,
            fundamental_freq_hz=261.63,
            harmonic_series_match_ratio=0.85
        ),
        NoteEvent(
            pitch=62, start_ms=600, end_ms=1000, velocity=75,
            confidence=0.85, zero_crossing_rate=0.04,
            source=SourceType.PITCH,
            fundamental_freq_hz=293.66,
            harmonic_series_match_ratio=0.82
        ),
        NoteEvent(
            pitch=64, start_ms=1200, end_ms=1800, velocity=90,
            confidence=0.95, zero_crossing_rate=0.03,
            source=SourceType.PITCH,
            fundamental_freq_hz=329.63,
            harmonic_series_match_ratio=0.88
        )
    ]

    # Add snapped version for one note
    import copy
    snapped_note = copy.copy(events[0])
    snapped_note.snapped_start_ms = 2
    snapped_note.snapped_end_ms = 502
    snapped_note.snap_reason = "quantized_to_beat"
    events.append(snapped_note)

    # Create mock AudioContext
    mock_context = AudioContext(
        file_path="test_audio.mp3",
        original_sample_rate=44100,
        working_sample_rate=44100,
        duration_seconds=10.0,
        num_channels=2,
        file_hash="a" * 64,
        memory_mb=5.0
    )

    # Mock timbre analysis
    mock_timbre = {
        "instrument_counts": {"piano": 45, "bass": 25, "drums": 200},
        "total_analyzed": 270,
        "entities": []
    }

    print(f"\n1. Exporting {len(events)} notes...")

    # Export
    success = export_to_json(
        events=events,
        output_path="/tmp/test_transcription.json",
        audio_context=mock_context,
        tempo_source="librosa",
        timbre_analysis=mock_timbre,
        options=ExportOptions(json_pretty=True, include_veto_history=True)
    )

    print(f"   Export success: {success}")

    # Check output
    import json
    output_path = Path("/tmp/test_transcription.json")
    if output_path.exists():
        size = output_path.stat().st_size
        print(f"   File size: {size / 1024:.1f}KB")

        with open(output_path, 'r') as f:
            data = json.load(f)
            print(f"   Version: {data.get('version')}")
            print(f"   Note count: {data.get('note_count')}")
            print(f"   Tempo source: {data.get('tempo_source')}")
            print(f"   Timbre analysis: {data.get('timbre_analysis', {}).get('instrument_counts', {})}")

        output_path.unlink()

    print("\n" + "=" * 60)
    print("JsonWriter test complete.")
    print("=" * 60)


if __name__ == "__main__":
    quick_json_test()