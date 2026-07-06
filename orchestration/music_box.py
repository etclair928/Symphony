# =================================================================
# MODULE: orchestration/music_box.py
# DESCRIPTION: Forensic flight recorder for Grimlock 5.0.
#
# VERSION: 5.6.1 (Updated for Pipeline Integration)
# UPDATED: 2026-05-13
#
# PHILOSOPHY:
#     Every decision is recorded. Every veto is logged.
#     The MusicBox is the append-only source of truth for
#     what the pipeline did and why.
#
# FIXES & IMPROVEMENTS:
#     - Added missing ForensicRecord import
#     - Added LogLevel enum
#     - Safe DecisionType handling (doesn't crash on unknown values)
#     - StageOutput/StageInput logging support
#     - Memory allocation/release tracking
#     - Eviction event logging
#     - Proper datetime import
#     - Arena event logging
# =================================================================

from __future__ import annotations

import json
import time
import uuid
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

from core.order_types import DecisionType, VetoReason, ValidationGate, SourceType, ForensicRecord
from orchestration.stage_context import StageOutput, StageStatus, StageInput


# ========================================================================
# Log Level Enum
# ========================================================================

class LogLevel(str, Enum):
    """Log level for MusicBox console output."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ========================================================================
# Log Entry and Session Summary
# ========================================================================

@dataclass
class LogEntry:
    """A single log entry in the MusicBox."""
    timestamp: datetime
    stage: str
    decision_type: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage,
            "decision_type": self.decision_type,
            "message": self.message,
            "metadata": self.metadata
        }


@dataclass
class SessionSummary:
    """Summary of a MusicBox session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    decision_count: int = 0
    error_count: int = 0
    veto_count: int = 0
    memory_peak_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "decision_count": self.decision_count,
            "error_count": self.error_count,
            "veto_count": self.veto_count,
            "memory_peak_mb": self.memory_peak_mb
        }


# ========================================================================
# Music Box Configuration
# ========================================================================

@dataclass
class MusicBoxConfig:
    """Configuration for MusicBox."""
    log_path: Optional[Path] = None
    buffer_size: int = 100
    rotate_bytes: int = 100 * 1024 * 1024  # 100MB
    max_backups: int = 5
    console_log_level: LogLevel = LogLevel.INFO
    include_memory_snapshot: bool = True
    include_timestamps: bool = True
    log_memory_events: bool = True
    log_stage_io: bool = True


# ========================================================================
# Music Box (Forensic Flight Recorder)
# ========================================================================

class MusicBox:
    """
    Append-only logger for pipeline decisions.

    Every decision is recorded with before/after state and reasoning.
    The MusicBox never deletes or modifies records.

    Safe DecisionType handling: unknown DecisionType strings are logged
    as warnings but do NOT crash the pipeline.
    """

    def __init__(self, config: Optional[MusicBoxConfig] = None):
        self.config = config or MusicBoxConfig()
        self._session_id = str(uuid.uuid4())[:8]
        self._buffer: List[ForensicRecord] = []
        self._flush_count = 0
        self._error_log: List[Dict[str, Any]] = []
        self._session_start_time = datetime.now()
        self._memory_peak_mb = 0.0

        if self.config.log_path:
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Core Logging
    # ========================================================================

    def log_decision(
            self,
            stage_name: str,
            decision_type: Union[str, DecisionType],
            before_state: Dict[str, Any],
            after_state: Dict[str, Any],
            reasoning: str,
            reversible: bool = True
    ) -> None:
        """
        Log a decision. Safe for unknown DecisionType values.
        """
        decision_type_str = None

        if isinstance(decision_type, DecisionType):
            decision_type_str = decision_type.value
        elif isinstance(decision_type, str):
            try:
                dt = DecisionType(decision_type)
                decision_type_str = dt.value
            except ValueError:
                # A free-form decision_type string, not one of the
                # pre-registered DecisionType values - this is the
                # NORMAL case, not an error: nearly every agent in this
                # codebase logs its own descriptive string here
                # ("fallback", "harmonic_validation", "model_loaded",
                # "export_start", ...) rather than a registered enum
                # value. This used to call log_error() for every single
                # one of those calls, which flooded the session log
                # with a fake "error_occurred" record for completely
                # normal, successful logging - actively undermining the
                # forensic record's trustworthiness (real errors get
                # buried under dozens of manufactured ones). Just record
                # the string as-is.
                decision_type_str = decision_type
        else:
            decision_type_str = str(decision_type)

        record = ForensicRecord(
            session_id=self._session_id,
            timestamp_utc=datetime.now(),
            stage_name=stage_name,
            decision_type=decision_type_str,
            before_state=self._sanitize_state(before_state),
            after_state=self._sanitize_state(after_state),
            reasoning=reasoning,
            human_reversible=reversible
        )

        self._buffer.append(record)

        if len(self._buffer) >= self.config.buffer_size:
            self.flush()

    # ========================================================================
    # Stage Lifecycle Logging (New)
    # ========================================================================

    def log_stage_input(self, stage_name: str, stage_input: StageInput) -> None:
        """Log stage input for forensic tracking."""
        if not self.config.log_stage_io:
            return

        self.log_decision(
            stage_name=stage_name,
            decision_type="stage_input",
            before_state={},
            after_state={
                "inputs": {k: str(v) for k, v in stage_input.inputs.items()},
                "evidence": stage_input.evidence,
                "guided_params": stage_input.guided_params is not None,
                "timestamp": stage_input.timestamp.isoformat()
            },
            reasoning=f"Stage {stage_name} received input",
            reversible=False
        )

    def log_stage_output(self, stage_name: str, stage_output: StageOutput) -> None:
        """Log stage output for forensic tracking."""
        if not self.config.log_stage_io:
            return

        self.log_decision(
            stage_name=stage_name,
            decision_type="stage_output",
            before_state={},
            after_state={
                "success": stage_output.success,
                "status": stage_output.status.value if hasattr(stage_output.status, 'value') else str(
                    stage_output.status),
                "produced_buffers": [str(b) for b in stage_output.produced_buffers],
                "event_count": len(stage_output.produced_events),
                "error": stage_output.error,
                "execution_time_ms": stage_output.execution_time_ms
            },
            reasoning=f"Stage {stage_name} completed with status {stage_output.status}",
            reversible=False
        )

    def log_stage_start(self, stage_name: str, metadata: Dict[str, Any]) -> None:
        """Log that a pipeline stage has started."""
        self.log_decision(
            stage_name=stage_name,
            decision_type=DecisionType.STAGE_START,
            before_state={},
            after_state=metadata,
            reasoning=f"Stage {stage_name} started",
            reversible=False
        )

    def log_stage_end(
            self,
            stage_name: str,
            result: Any,
            elapsed_ms: float
    ) -> None:
        """Log that a pipeline stage has ended."""
        # Handle both StageOutput and legacy result objects
        if hasattr(result, 'status'):
            status = result.status.value if hasattr(result.status, 'value') else str(result.status)
            success = result.success
            event_count = len(result.produced_events)
        else:
            status = "SUCCESS" if getattr(result, 'success', True) else "FAILED"
            success = getattr(result, 'success', True)
            event_count = len(getattr(result, 'events', []))

        self.log_decision(
            stage_name=stage_name,
            decision_type=DecisionType.STAGE_END,
            before_state={},
            after_state={
                "success": success,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "event_count": event_count
            },
            reasoning=f"Stage {stage_name} completed in {elapsed_ms:.0f}ms",
            reversible=False
        )

    # ========================================================================
    # Veto and Validation Logging
    # ========================================================================

    def log_veto(
            self,
            source: SourceType,
            reason: VetoReason,
            detail: str,
            gate: ValidationGate
    ) -> None:
        """Log a veto event."""
        self.log_decision(
            stage_name="scribe",
            decision_type=DecisionType.VETO_TRIGGERED,
            before_state={"source": source.value, "gate": gate.value},
            after_state={"reason": reason.value, "detail": detail},
            reasoning=f"Veto by {source.value}: {reason.value}",
            reversible=False
        )

    def log_validation(
            self,
            stage_name: str,
            gate: ValidationGate,
            passed: bool,
            details: Dict[str, Any]
    ) -> None:
        """Log a validation gate result."""
        self.log_decision(
            stage_name=stage_name,
            decision_type="validation",
            before_state={"gate": gate.value},
            after_state={"passed": passed, "details": details},
            reasoning=f"Validation gate {gate.value}: {'PASSED' if passed else 'FAILED'}",
            reversible=False
        )

    # ========================================================================
    # Error Logging
    # ========================================================================

    def log_error(
            self,
            stage_name: str,
            error: Exception,
            context: Dict[str, Any]
    ) -> None:
        """Log an error."""
        self.log_decision(
            stage_name=stage_name,
            decision_type=DecisionType.ERROR_OCCURRED,
            before_state=context,
            after_state={"error": str(error), "error_type": type(error).__name__},
            reasoning=f"Error in {stage_name}: {error}",
            reversible=False
        )

        self._error_log.append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "error": str(error),
            "type": type(error).__name__,
            "context": context
        })

    # ========================================================================
    # Memory Events
    # ========================================================================

    def log_memory_allocation(
            self,
            arena_name: str,
            buffer_name: str,
            size_bytes: int,
            tier: Optional[str] = None
    ) -> None:
        """Log a memory allocation event."""
        if not self.config.log_memory_events:
            return

        after_state = {
            "buffer": buffer_name,
            "size_bytes": size_bytes,
            "size_mb": size_bytes / (1024 * 1024)
        }
        if tier:
            after_state["tier"] = tier

        self.log_decision(
            stage_name=f"arena_{arena_name}",
            decision_type="memory_alloc",
            before_state={},
            after_state=after_state,
            reasoning=f"Allocated {buffer_name} ({size_bytes / (1024 * 1024):.1f}MB)",
            reversible=True
        )

    def log_memory_release(
            self,
            arena_name: str,
            buffer_name: str,
            size_bytes: int
    ) -> None:
        """Log a memory release event."""
        if not self.config.log_memory_events:
            return

        self.log_decision(
            stage_name=f"arena_{arena_name}",
            decision_type="memory_release",
            before_state={},
            after_state={
                "buffer": buffer_name,
                "size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 * 1024)
            },
            reasoning=f"Released {buffer_name}",
            reversible=True
        )

    def log_memory_eviction(
            self,
            arena_name: str,
            buffer_name: str,
            size_bytes: int,
            strategy: str
    ) -> None:
        """Log a memory eviction event."""
        if not self.config.log_memory_events:
            return

        self.log_decision(
            stage_name=f"arena_{arena_name}",
            decision_type="memory_evict",
            before_state={},
            after_state={
                "buffer": buffer_name,
                "size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 * 1024),
                "strategy": strategy
            },
            reasoning=f"Evicted {buffer_name} due to {strategy} strategy",
            reversible=False
        )

    def log_memory_pressure(self, current_mb: float, limit_mb: float, freeable_mb: float = 0.0) -> None:
        """Log memory pressure warning."""
        self._memory_peak_mb = max(self._memory_peak_mb, current_mb)

        self.log_decision(
            stage_name="memory_guardian",
            decision_type=DecisionType.MEMORY_PRESSURE,
            before_state={"current_mb": current_mb, "limit_mb": limit_mb},
            after_state={"action": "gc_scheduled", "freeable_mb": freeable_mb, "peak_mb": self._memory_peak_mb},
            reasoning=f"Memory pressure: {current_mb:.1f}MB / {limit_mb:.1f}MB (freeable: {freeable_mb:.1f}MB)",
            reversible=False
        )

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_session_logs(self) -> List[Dict[str, Any]]:
        """Get all logs for current session."""
        self.flush()
        if self.config.log_path and self.config.log_path.exists():
            logs = []
            with open(self.config.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return logs
        return [asdict(r) for r in self._buffer]

    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all errors logged in this session."""
        return self._error_log.copy()

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    def get_session_summary(self) -> SessionSummary:
        """Get summary of current session."""
        return SessionSummary(
            session_id=self._session_id,
            start_time=self._session_start_time,
            end_time=datetime.now(),
            decision_count=len(self._buffer) + (self._flush_count * self.config.buffer_size),
            error_count=len(self._error_log),
            veto_count=sum(1 for r in self._buffer if r.decision_type == "veto_triggered"),
            memory_peak_mb=self._memory_peak_mb
        )

    def query_vetoes(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        All veto_triggered records logged this session, flattened to
        source/stage/reason/detail/is_hard - log_veto() itself splits
        this same information across before_state/after_state, which
        this unpacks back out. export/json_writer.py's veto-history
        collector already called this method; it never existed, so
        that code path crashed the moment it actually ran.
        """
        vetoes = []
        for r in self.get_session_logs():
            if r.get("decision_type") != "veto_triggered":
                continue
            before = r.get("before_state") or {}
            after = r.get("after_state") or {}
            vetoes.append({
                "timestamp": r.get("timestamp_utc"),
                "source": before.get("source", "unknown"),
                "stage": r.get("stage_name", "unknown"),
                "reason": after.get("reason", "unknown"),
                "detail": after.get("detail", ""),
                "is_hard": True,
            })
        return vetoes[:limit] if limit else vetoes

    def query_entries(self, decision_type: Optional[Union[str, DecisionType]] = None,
                      stage_name: Optional[str] = None,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        All logged records this session, optionally filtered by
        decision type and/or stage, flattened to stage/message/data -
        the same missing-method gap as query_vetoes() above.
        """
        records = self.get_session_logs()
        if decision_type is not None:
            dt_str = decision_type.value if isinstance(decision_type, DecisionType) else decision_type
            records = [r for r in records if r.get("decision_type") == dt_str]
        if stage_name is not None:
            records = [r for r in records if r.get("stage_name") == stage_name]

        entries = [{
            "timestamp": r.get("timestamp_utc"),
            "decision_type": r.get("decision_type", "unknown"),
            "stage": r.get("stage_name", "unknown"),
            "message": r.get("reasoning", ""),
            "data": r.get("after_state", {}),
        } for r in records]
        return entries[:limit] if limit else entries

    def get_stats(self) -> Dict[str, Any]:
        """Get MusicBox statistics."""
        return {
            "session_id": self._session_id,
            "buffer_size": len(self._buffer),
            "flush_count": self._flush_count,
            "error_count": len(self._error_log),
            "log_path": str(self.config.log_path) if self.config.log_path else None,
            "memory_peak_mb": self._memory_peak_mb,
            "session_duration_seconds": (datetime.now() - self._session_start_time).total_seconds()
        }

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def flush(self) -> None:
        """
        Flush buffer to disk.

        With no log_path configured, there's nowhere to flush TO - so
        this now leaves the buffer alone instead of clearing it. It
        previously cleared unconditionally, and get_session_logs()
        always calls flush() first, which meant the very common case
        of no log_path (e.g. the real pipeline's MusicBox() with no
        arguments - orchestration/pipeline.py never sets one) silently
        discarded everything the moment more than buffer_size records
        had been logged, or the instant anyone called get_session_logs()
        at all. Verified directly: logging one veto then calling
        get_session_logs() returned zero entries.
        """
        if not self.config.log_path:
            return

        with open(self.config.log_path, 'a', encoding='utf-8') as f:
            for record in self._buffer:
                f.write(json.dumps(asdict(record), default=str) + '\n')

        self._flush_count += 1
        self._buffer.clear()

        if self.config.log_path.exists():
            size = self.config.log_path.stat().st_size
            if size > self.config.rotate_bytes:
                self._rotate_logs()

    def _rotate_logs(self) -> None:
        """Rotate log files."""
        if not self.config.log_path:
            return

        for i in range(self.config.max_backups - 1, 0, -1):
            src = self.config.log_path.with_suffix(f".{i}.jsonl")
            dst = self.config.log_path.with_suffix(f".{i + 1}.jsonl")
            if src.exists():
                src.rename(dst)

        if self.config.log_path.exists():
            self.config.log_path.rename(
                self.config.log_path.with_suffix(".1.jsonl")
            )

    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state dict for JSON serialization."""
        sanitized = {}
        for key, value in state.items():
            if hasattr(value, 'to_dict'):
                sanitized[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                sanitized[key] = str(value)
            elif hasattr(value, 'shape'):
                sanitized[key] = f"<array shape={value.shape}>"
            elif isinstance(value, (bytes, bytearray)):
                sanitized[key] = f"<bytes length={len(value)}>"
            elif isinstance(value, (list, tuple)) and len(value) > 100:
                sanitized[key] = f"<list length={len(value)}>"
            else:
                try:
                    json.dumps(value)
                    sanitized[key] = value
                except (TypeError, ValueError):
                    sanitized[key] = str(value)
        return sanitized

    def end_session(self) -> None:
        """End the current session and flush remaining logs."""
        summary = self.get_session_summary()

        self.log_decision(
            stage_name="music_box",
            decision_type=DecisionType.SESSION_END,
            before_state={},
            after_state=summary.to_dict(),
            reasoning=f"Session ended. Duration: {(datetime.now() - self._session_start_time).total_seconds():.1f}s",
            reversible=False
        )
        self.flush()

    def start_session(self, audio_path: str, guided_params: Optional[Any] = None) -> None:
        """Start a new session."""
        self._session_id = str(uuid.uuid4())[:8]
        self._session_start_time = datetime.now()
        self._memory_peak_mb = 0.0

        self.log_decision(
            stage_name="music_box",
            decision_type=DecisionType.SESSION_START,
            before_state={},
            after_state={
                "audio_path": audio_path,
                "guided_mode": guided_params is not None,
                "session_id": self._session_id,
                "timestamp": self._session_start_time.isoformat()
            },
            reasoning=f"Session started for {audio_path}",
            reversible=False
        )

    def log_consensus_decision(
            self,
            final_decision: bool,
            confidence: float,
            veto_triggered_by: Optional[SourceType] = None,
            veto_reason: Optional[VetoReason] = None
    ) -> None:
        """Log a consensus decision."""
        self.log_decision(
            stage_name="consensus",
            decision_type="consensus_decision",
            before_state={},
            after_state={
                "decision": final_decision,
                "confidence": confidence,
                "veto_triggered_by": veto_triggered_by.value if veto_triggered_by else None,
                "veto_reason": veto_reason.value if veto_reason else None
            },
            reasoning=f"Consensus reached: {'PASS' if final_decision else 'FAIL'} (confidence: {confidence:.2f})",
            reversible=False
        )

    def __repr__(self) -> str:
        return f"MusicBox(session={self._session_id}, buffer={len(self._buffer)}, errors={len(self._error_log)}, peak_memory={self._memory_peak_mb:.1f}MB)"


# ========================================================================
# Convenience Functions
# ========================================================================

def create_music_box(
        log_path: Optional[str] = None,
        buffer_size: int = 100,
        console_log_level: Union[str, LogLevel] = LogLevel.INFO,
        log_memory_events: bool = True
) -> MusicBox:
    """Create a configured MusicBox instance."""
    if isinstance(console_log_level, str):
        try:
            console_log_level = LogLevel(console_log_level.upper())
        except ValueError:
            console_log_level = LogLevel.INFO

    config = MusicBoxConfig(
        log_path=Path(log_path) if log_path else None,
        buffer_size=buffer_size,
        console_log_level=console_log_level,
        log_memory_events=log_memory_events
    )
    return MusicBox(config)


# ========================================================================
# Standalone Test
# ========================================================================

def quick_music_box_test():
    """Quick test function for MusicBox."""
    print("\n" + "=" * 60)
    print("MusicBox Test - Pipeline Integration")
    print("=" * 60)

    music_box = create_music_box()

    print("\n1. Session Management:")
    music_box.start_session("test_audio.mp3")
    print(f"   Session ID: {music_box.get_session_id()}")

    print("\n2. Stage Input/Output Logging:")
    from orchestration.stage_context import StageInput, StageOutput, StageStatus
    from core.order_types import AudioContext

    mock_context = AudioContext(
        file_path="test.mp3",
        original_sample_rate=44100,
        working_sample_rate=44100,
        duration_seconds=10.0,
        num_channels_original=2,
        is_mono=True
    )

    stage_input = StageInput(
        context=mock_context,
        inputs={"master": "test_id"},
        evidence={}
    )

    music_box.log_stage_input("ingestion", stage_input)
    print("   ✓ Stage input logged")

    stage_output = StageOutput(
        stage_name="ingestion",
        success=True,
        status=StageStatus.SUCCESS,
        produced_buffers=["master_id"]
    )

    music_box.log_stage_output("ingestion", stage_output)
    print("   ✓ Stage output logged")

    print("\n3. Memory Events:")
    music_box.log_memory_allocation("arena_main", "test_buffer", 1024 * 1024, "TIER_1")
    music_box.log_memory_pressure(1500.0, 2048.0, 500.0)
    print("   ✓ Memory events logged")

    print("\n4. Consensus Logging:")
    music_box.log_consensus_decision(True, 0.85)
    print("   ✓ Consensus decision logged")

    print("\n5. Statistics:")
    stats = music_box.get_stats()
    print(f"   Session: {stats['session_id']}")
    print(f"   Buffer size: {stats['buffer_size']}")
    print(f"   Error count: {stats['error_count']}")
    print(f"   Peak memory: {stats['memory_peak_mb']:.1f}MB")

    print("\n" + "=" * 60)
    print("MusicBox test complete.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        pass
    quick_music_box_test()