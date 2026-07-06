# =================================================================
# MODULE: export/json_writer.py (REWRITTEN)
# VERSION: 5.6.1 (Enhanced with streaming, validation, better stats)
# UPDATED: 2026-06-03
#
# KEY FIXES in 5.5.0:
#   1. ✅ Fixed file size estimation (actual size after write)
#   2. ✅ Added streaming export for large transcriptions (>10k notes)
#   3. ✅ Enhanced confidence statistics (median, std, percentiles)
#   4. ✅ Added note validation before export
#   5. ✅ Improved veto history collection with limits
#   6. ✅ Added instrument_family to serialization
# =================================================================

import json
import warnings
import time
import gc
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum
import math

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
        if isinstance(obj, np.bool_):
            return bool(obj)
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
# ExportValidator
# ========================================================================

class ExportValidator:
    """Validate and repair note events before export."""

    @staticmethod
    def validate_notes(
            events: List[NoteEvent],
            context: str = "export"
    ) -> Tuple[List[NoteEvent], Dict[str, int]]:
        """Validate and optionally repair note events before export."""
        if not events:
            return [], {"empty": 1}

        stats = {
            "total": len(events),
            "invalid_pitch": 0,
            "invalid_velocity": 0,
            "invalid_confidence": 0,
            "negative_timing": 0,
            "repaired": 0,
            "zero_duration": 0
        }

        validated = []
        for i, event in enumerate(events):
            # Validate pitch
            if not hasattr(event, 'pitch') or event.pitch is None:
                stats["invalid_pitch"] += 1
                continue

            if event.pitch < 0 or event.pitch > 127:
                stats["invalid_pitch"] += 1
                event.pitch = max(0, min(127, event.pitch))
                stats["repaired"] += 1

            # Validate velocity
            if not hasattr(event, 'velocity') or event.velocity is None:
                event.velocity = 64
                stats["repaired"] += 1
            elif event.velocity < 1 or event.velocity > 127:
                event.velocity = max(1, min(127, event.velocity))
                stats["repaired"] += 1

            # Validate timing
            if event.start_ms < 0:
                event.start_ms = 0
                stats["repaired"] += 1
                stats["negative_timing"] += 1

            if event.end_ms <= event.start_ms:
                if event.end_ms == event.start_ms:
                    stats["zero_duration"] += 1
                event.end_ms = event.start_ms + 10
                stats["repaired"] += 1

            # Validate confidence
            if not hasattr(event, 'confidence') or event.confidence is None:
                event.confidence = 0.5
                stats["repaired"] += 1
            else:
                event.confidence = max(0.0, min(1.0, event.confidence))

            validated.append(event)

        return validated, stats


# ========================================================================
# JsonWriter
# ========================================================================

class JsonWriter:
    """
    JSON exporter with forensic audit trail.

    Enhanced with streaming export for large transcriptions.
    """

    STREAMING_THRESHOLD = 10000  # Notes threshold for streaming mode

    def __init__(
            self,
            options: Optional[ExportOptions] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        self.options = options or ExportOptions()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._validator = ExportValidator()

        # Session metadata
        self._export_time = datetime.now()
        self._version = "5.5.0"

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
        Export transcription to JSON with validation.

        Automatically uses streaming for large transcriptions.
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        if output_path is None:
            output_path = f"grimlock_transcription_{self._export_time.strftime('%Y%m%d_%H%M%S')}.json"

        output_path = Path(output_path)

        # Validate and repair notes
        events, validation_stats = self._validator.validate_notes(events, "JSON export")
        if not events:
            self._log_status("No valid notes to export after validation", "error")
            return False

        if validation_stats["repaired"] > 0:
            self._log_status(
                f"Repaired {validation_stats['repaired']}/{validation_stats['total']} notes "
                f"(invalid pitch: {validation_stats['invalid_pitch']}, "
                f"zero duration: {validation_stats['zero_duration']})",
                "warn"
            )

        self._log_status(f"Exporting {len(events)} notes to {output_path.name}")

        if self._music_box:
            self._music_box.log_decision(
                stage_name="json_writer",
                decision_type="export_start",
                before_state={},
                after_state={
                    "output_path": str(output_path),
                    "event_count": len(events),
                    "include_forensic": self.options.include_veto_history,
                    "tempo_source": tempo_source,
                    "validation_stats": validation_stats
                },
                reasoning=f"Exporting {len(events)} notes to JSON",
                reversible=False
            )

        # Choose export method based on size
        if len(events) > self.STREAMING_THRESHOLD:
            self._log_status(f"Using streaming export for {len(events)} notes", "info")
            success = self._export_json_streaming(
                events=events,
                output_path=output_path,
                audio_context=audio_context,
                groove_field=groove_field,
                tempo_map=tempo_map,
                scribe_verdict=scribe_verdict,
                timbre_analysis=timbre_analysis,
                tempo_source=tempo_source,
                include_music_box_logs=include_music_box_logs
            )
        else:
            export_data = self._build_export_data(
                events=events,
                audio_context=audio_context,
                groove_field=groove_field,
                tempo_map=tempo_map,
                scribe_verdict=scribe_verdict,
                timbre_analysis=timbre_analysis,
                tempo_source=tempo_source,
                include_music_box_logs=include_music_box_logs,
                validation_stats=validation_stats
            )
            success = self._write_file(export_data, output_path)

        execution_time_ms = (time.time() - start_time) * 1000
        memory_delta_mb = self._get_current_memory_mb() - start_memory
        self._last_export_time_ms = execution_time_ms

        if memory_delta_mb > STAGGERED_GC_TRIGGER_MB:
            self._force_cleanup()

        if success and self._music_box:
            self._music_box.log_decision(
                stage_name="json_writer",
                decision_type="export_complete",
                before_state={},
                after_state={
                    "output_path": str(output_path),
                    "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
                    "execution_time_ms": execution_time_ms
                },
                reasoning=f"JSON exported to {output_path.name}",
                reversible=False
            )

        self._log_status(f"Export complete in {execution_time_ms:.0f}ms")
        return success

    # ========================================================================
    # Streaming Export (for large transcriptions)
    # ========================================================================

    def _export_json_streaming(
            self,
            events: List[NoteEvent],
            output_path: Path,
            chunk_size: int = 1000,
            **kwargs
    ) -> bool:
        """Stream export for large transcriptions to prevent OOM."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write('{\n')
                f.write(f'  "version": "{self._version}",\n')
                f.write(f'  "export_timestamp": "{self._export_time.isoformat()}",\n')
                f.write(f'  "grimlock_version": "{self._version}",\n')
                f.write(f'  "streaming_export": true,\n')
                f.write(f'  "chunk_size": {chunk_size},\n')
                f.write('  "notes": [\n')

                # Stream notes in chunks
                for i in range(0, len(events), chunk_size):
                    chunk = events[i:i + chunk_size]
                    chunk_data = [self._serialize_note_event(e) for e in chunk]
                    chunk_json = json.dumps(chunk_data, indent=2, cls=JSONEncoder)

                    # Remove outer brackets and add commas
                    chunk_content = chunk_json[1:-1]
                    if i > 0:
                        f.write(',\n')
                    f.write(chunk_content)

                    # Force flush to disk periodically
                    if i % (chunk_size * 10) == 0:
                        f.flush()

                    # Optional GC after each chunk
                    if i % (chunk_size * 5) == 0:
                        gc.collect()

                # Write footer with statistics
                stats = self._compute_enhanced_statistics(events)
                f.write('\n  ],\n')
                f.write(f'  "note_count": {len(events)},\n')
                f.write(f'  "statistics": {json.dumps(stats, indent=2, cls=JSONEncoder)}\n')
                f.write('}\n')

            return True
        except Exception as e:
            self._log_status(f"Streaming export failed: {e}", "error")
            return False

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
            include_music_box_logs: Optional[bool] = None,
            validation_stats: Optional[Dict[str, int]] = None
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

            # Timbre analysis
            "timbre_analysis": timbre_analysis if timbre_analysis else None,

            # Note events
            "notes": [self._serialize_note_event(event) for event in events],
            "note_count": len(events),

            # Enhanced statistics
            "statistics": self._compute_enhanced_statistics(events, groove_field, tempo_map, timbre_analysis),

            # Validation stats
            "validation": validation_stats or {},

            # Validation and vetoes
            "scribe_verdict": scribe_verdict or {},

            # Forensic audit trail
            "veto_history": self._collect_veto_history(),
            "reasoning_chains": self._collect_reasoning_chains(events),
            "quantization_history": self._collect_quantization_history(events),
        }

        if include_logs and self._music_box:
            export_data["music_box_logs"] = self._get_music_box_logs()

        return export_data

    def _build_provenance(self) -> Dict[str, Any]:
        """Build provenance metadata for reproducibility."""
        provenance = {
            "generator": "Grimlock 5.5",
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
    # Enhanced Statistics (FIX: median, std, percentiles)
    # ========================================================================

    def _compute_enhanced_statistics(
            self,
            events: List[NoteEvent],
            groove_field: Optional[GrooveField] = None,
            tempo_map: Optional[TempoMap] = None,
            timbre_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive transcription statistics."""
        if not events:
            return {
                "total_notes": 0,
                "empty_transcription": True,
                "heartbeat_note_added": self.options.heartbeat_note_pitch is not None
            }

        pitches = [e.pitch for e in events]
        confidences = [e.confidence for e in events]
        durations_ms = [e.end_ms - e.start_ms for e in events]

        conf_array = np.array(confidences)

        # Source distribution
        source_counts = {}
        for event in events:
            source_counts[event.source.value] = source_counts.get(event.source.value, 0) + 1

        # Snapped notes: is_snapped() is true for every note the quantizer
        # ever touched, even when it preserved the raw position exactly
        # (in-pocket) - only count it as "snapped" if it was actually
        # displaced by a musically meaningful amount.
        shift_deltas = [abs(e.snapped_start_ms - e.start_ms) for e in events if e.is_snapped()]
        snapped_count = sum(1 for d in shift_deltas if d >= 1.0)
        average_shift_ms = round(float(np.mean(shift_deltas)), 1) if shift_deltas else 0.0

        # Pitch distribution by octave
        pitch_distribution = {}
        for pitch in pitches:
            octave = pitch // 12 - 1
            pitch_distribution[octave] = pitch_distribution.get(octave, 0) + 1

        stats = {
            "total_notes": len(events),
            # Enhanced confidence stats
            "average_confidence": round(float(np.mean(conf_array)), 3),
            "median_confidence": round(float(np.median(conf_array)), 3),
            "std_confidence": round(float(np.std(conf_array)), 3),
            "p5_confidence": round(float(np.percentile(conf_array, 5)), 3),
            "p95_confidence": round(float(np.percentile(conf_array, 95)), 3),
            "min_confidence": round(min(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            # Duration stats
            "average_duration_ms": round(float(np.mean(durations_ms)), 1),
            "median_duration_ms": round(float(np.median(durations_ms)), 1),
            "std_duration_ms": round(float(np.std(durations_ms)), 1),
            # Pitch stats
            "pitch_range": {
                "min": min(pitches),
                "max": max(pitches),
                "range_semitones": max(pitches) - min(pitches)
            },
            "pitch_distribution": pitch_distribution,
            # Source distribution
            "source_distribution": source_counts,
            # Quantization stats
            "snapped_notes_count": snapped_count,
            "snapped_notes_percentage": round(snapped_count / len(events) * 100, 1) if events else 0,
            "average_shift_ms": average_shift_ms,
            # Confidence bins
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

        # Add timbre statistics
        if timbre_analysis:
            instrument_counts = timbre_analysis.get("instrument_counts", {})
            total_analyzed = timbre_analysis.get("total_analyzed", 0)
            stats["timbre"] = {
                "instruments_detected": instrument_counts,
                "notes_analyzed": total_analyzed,
                "primary_instrument": max(instrument_counts.items(), key=lambda x: x[1])[
                    0] if instrument_counts else "unknown"
            }

        return stats

    # ========================================================================
    # Serialization Methods
    # ========================================================================

    def _serialize_note_event(self, event: NoteEvent) -> Dict[str, Any]:
        """Serialize a NoteEvent to JSON with instrument family."""
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

            # Original timestamps
            "original_start_ms": round(event.start_ms, 2),
            "original_end_ms": round(event.end_ms, 2),
            "original_duration_ms": round(event.end_ms - event.start_ms, 2),
        }

        # Add instrument family if available
        if hasattr(event, 'instrument_family') and event.instrument_family:
            if isinstance(event.instrument_family, Enum):
                result["instrument_family"] = event.instrument_family.value
            else:
                result["instrument_family"] = str(event.instrument_family)

        # Snapped timestamps
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
        """Serialize AudioContext to JSON."""
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
    # Forensic Data Collection (with limits)
    # ========================================================================

    def _collect_veto_history(self) -> List[Dict[str, Any]]:
        """Collect veto history from MusicBox with limits."""
        if not self._music_box:
            return []

        try:
            veto_entries = self._music_box.query_vetoes()
            if len(veto_entries) > 500:
                self._log_status(f"Truncating veto history from {len(veto_entries)} to 100 entries", "warn")

            return [
                {
                    "timestamp": str(entry.get("timestamp", "")),
                    "source": entry.get("source", "unknown"),
                    "stage": entry.get("stage", "unknown"),
                    "reason": entry.get("reason", "unknown"),
                    "detail": entry.get("detail", ""),
                    "is_hard_veto": entry.get("is_hard", True)
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
                    "timestamp": str(entry.get("timestamp", "")),
                    "decision_type": entry.get("decision_type", "unknown"),
                    "stage": entry.get("stage", "unknown"),
                    "message": entry.get("message", ""),
                    "data": entry.get("data", {})
                }
                for entry in entries[:200]  # Limit to 200 entries
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
        """Write JSON data to file with accurate size reporting."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                if self.options.json_pretty:
                    json.dump(data, f, indent=2, ensure_ascii=False, cls=JSONEncoder)
                else:
                    json.dump(data, f, ensure_ascii=False, cls=JSONEncoder)

            # Check actual size after write
            actual_size_mb = output_path.stat().st_size / (1024 * 1024)
            if actual_size_mb > JSON_MAX_FILE_SIZE_MB:
                warnings.warn(
                    f"JSON file size {actual_size_mb:.1f}MB exceeds limit {JSON_MAX_FILE_SIZE_MB}MB. "
                    f"Consider using streaming export for future exports."
                )

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
                "max_file_size_mb": JSON_MAX_FILE_SIZE_MB,
                "streaming_threshold": self.STREAMING_THRESHOLD
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
            "streaming_threshold": self.STREAMING_THRESHOLD,
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
    """Convenience function to export transcription to JSON."""
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
    print("JsonWriter Test (v5.5)")
    print("=" * 60)

    # Create test notes
    events = [
        NoteEvent(pitch=60, start_ms=0, end_ms=500, velocity=80,
                  confidence=0.9, zero_crossing_rate=0.05,
                  source=SourceType.PITCH),
        NoteEvent(pitch=64, start_ms=600, end_ms=1000, velocity=75,
                  confidence=0.85, zero_crossing_rate=0.04,
                  source=SourceType.PITCH),
    ]

    # Add instrument families
    if hasattr(events[0], 'instrument_family'):
        events[0].instrument_family = "piano"
        events[1].instrument_family = "piano"

    print(f"\n1. Exporting {len(events)} notes...")

    success = export_to_json(
        events=events,
        output_path="/tmp/test_transcription.json",
        tempo_source="test",
        options=ExportOptions(json_pretty=True, include_veto_history=True)
    )

    print(f"   Export success: {success}")

    if success:
        output_path = Path("/tmp/test_transcription.json")
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"   File size: {size / 1024:.1f}KB")

            with open(output_path, 'r') as f:
                data = json.load(f)
                print(f"   Version: {data.get('version')}")
                print(f"   Note count: {data.get('note_count')}")
                print(f"   Stats median confidence: {data.get('statistics', {}).get('median_confidence', 'N/A')}")

            output_path.unlink()

    print("\n" + "=" * 60)
    print("JsonWriter test complete.")
    print("=" * 60)


if __name__ == "__main__":
    quick_json_test()