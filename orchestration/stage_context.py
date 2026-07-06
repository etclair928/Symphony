# =================================================================
# MODULE: orchestration/stage_context.py
# VERSION: 5.6.1
# DESCRIPTION: Immutable stage packets for data flow.
#
# FIXES: #2, #12, #14 - No shared mutable state, typed outputs
# =================================================================

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum

from core.order_types import NoteEvent, AudioContext
from memory.arena import BufferID


class StageStatus(str, Enum):
    """Honest stage completion status."""
    SUCCESS = "success"  # Stage completed fully
    DEGRADED = "degraded"  # Stage completed with fallback
    EMPTY_BUT_VALID = "empty_but_valid"  # No output but that's expected
    UNIMPLEMENTED = "unimplemented"  # Stub - placeholder
    FAILED = "failed"  # Stage failed, cannot proceed
    TIMEOUT = "timeout"  # Stage timed out
    CANCELLED = "cancelled"  # Stage was cancelled


@dataclass(frozen=True)
class AudioBuffer:
    """Immutable audio buffer with full provenance."""
    buffer_id: BufferID
    sample_rate: int
    channels: int
    duration_seconds: float
    origin_stage: str
    transform_lineage: List[str] = field(default_factory=list)
    original_sample_rate: Optional[int] = None

    @property
    def is_downsampled(self) -> bool:
        return self.original_sample_rate is not None and self.original_sample_rate != self.sample_rate


@dataclass(frozen=True)
class StageInput:
    """Immutable input packet for a stage."""
    context: AudioContext
    inputs: Dict[str, BufferID]  # named inputs by role
    evidence: Dict[str, Any]  # available evidence types
    guided_params: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StageOutput:
    """Typed output packet from a stage."""
    stage_name: str
    success: bool
    status: StageStatus
    produced_buffers: List[BufferID] = field(default_factory=list)
    produced_events: List[NoteEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    @property
    def is_usable(self) -> bool:
        """Output can be used by downstream stages."""
        return self.status in (StageStatus.SUCCESS, StageStatus.DEGRADED, StageStatus.EMPTY_BUT_VALID)

    @property
    def is_catastrophic(self) -> bool:
        """Output indicates pipeline should abort."""
        return self.status == StageStatus.FAILED and not self.metadata.get("retryable", False)