# =================================================================
# MODULE: memory/guardian.py
# DESCRIPTION: Allocation Authority & Borrow Enforcement.
#              "The pipeline requests. The Guardian decides."
#
# VERSION: 5.6.1 (Allocation Broker Architecture)
# UPDATED: 2026-05-14
#
# KEY CHANGES FROM 5.0:
#     1. request_allocation() - Broker pattern replaces passive monitoring
#     2. AllocationResult dataclass - Structured approval/denial
#     3. Stage registration - Guardian knows who is running
#     4. Borrow enforcement - Can revoke misbehaving stages
#     5. Progressive degradation - Suggest model downgrades on pressure
#     6. No direct buffer access - All access through Arena.borrow()
#
# PHILOSOPHY:
#     "The pipeline requests. The Guardian decides."
#     "Memory is a finite resource. The Guardian is the broker."
# =================================================================

from __future__ import annotations

import gc
import time
import warnings
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from enum import Enum, auto

# Core imports
from core.constants import (
    MEMORY_LIMIT_MB,
    MEMORY_WARNING_THRESHOLD_MB,
    MEMORY_CRITICAL_THRESHOLD_MB,
    STAGGERED_GC_TRIGGER_MB,
    BUFFER_LIFECYCLE_STAGES,
    GC_COLLECT_AFTER_EACH_STAGE,
    DELAY_BETWEEN_STAGGERED_DELETIONS_MS
)
from core.protocols import (
    MusicBoxProtocol, StatusReporterProtocol
)

# Import from arena (will be defined after guardian in practice)
# Using forward references for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.arena import MemoryArena, EvictionStrategy, MemoryTier, BufferID
else:
    # Runtime imports
    from memory.arena import MemoryArena, EvictionStrategy, MemoryTier, BufferID


# ========================================================================
# Enums and Types
# ========================================================================

class MemoryPressureLevel(str, Enum):
    """Memory pressure levels for auto-GC decisions."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AllocationStatus(str, Enum):
    """Status of an allocation request."""
    APPROVED = "approved"
    APPROVED_WITH_DEGRADATION = "approved_with_degradation"
    DENIED_INSUFFICIENT_MEMORY = "denied_insufficient_memory"
    DENIED_PRESSURE_TOO_HIGH = "denied_pressure_too_high"
    DENIED_RESERVATION_CONFLICT = "denied_reservation_conflict"
    DEFERRED_EVICTION_NEEDED = "deferred_eviction_needed"


class StageStatus(str, Enum):
    """Status of a registered stage."""
    PENDING = "pending"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    EVICTED = "evicted"


@dataclass(frozen=True)
class AllocationResult:
    """Result of an allocation request."""
    status: AllocationStatus
    allocated_mb: float
    requested_mb: float
    degradation_suggestion: Optional[str] = None
    alternative_model: Optional[str] = None
    required_eviction_mb: float = 0.0
    reason: Optional[str] = None

    @property
    def approved(self) -> bool:
        return self.status in (
            AllocationStatus.APPROVED,
            AllocationStatus.APPROVED_WITH_DEGRADATION
        )

    @classmethod
    def approved(cls, allocated_mb: float) -> 'AllocationResult':
        return cls(
            status=AllocationStatus.APPROVED,
            allocated_mb=allocated_mb,
            requested_mb=allocated_mb
        )

    @classmethod
    def approved_with_degradation(
            cls,
            allocated_mb: float,
            requested_mb: float,
            suggestion: str
    ) -> 'AllocationResult':
        return cls(
            status=AllocationStatus.APPROVED_WITH_DEGRADATION,
            allocated_mb=allocated_mb,
            requested_mb=requested_mb,
            degradation_suggestion=suggestion
        )

    @classmethod
    def denied(cls, status: AllocationStatus, reason: str, requested_mb: float) -> 'AllocationResult':
        return cls(
            status=status,
            allocated_mb=0.0,
            requested_mb=requested_mb,
            reason=reason
        )


@dataclass
class StageRecord:
    """Record of a stage registered with Guardian."""
    name: str
    status: StageStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    active_borrows: Set[str] = field(default_factory=set)  # BufferID strings
    allocated_mb: float = 0.0
    memory_peak_mb: float = 0.0
    eviction_count: int = 0


@dataclass
class MemoryReport:
    """Memory report for forensic logging."""
    current_mb: float
    peak_mb: float
    tracked_buffers: int
    tracked_bytes: int
    arena_count: int
    total_freed_mb: float
    gc_count: int
    emergency_gc_count: int
    pressure_level: MemoryPressureLevel
    freeable_mb: float = 0.0
    reserved_mb: float = 0.0
    active_stages: int = 0
    active_borrows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_mb": self.current_mb,
            "peak_mb": self.peak_mb,
            "tracked_buffers": self.tracked_buffers,
            "tracked_mb": self.tracked_bytes / (1024 * 1024),
            "arena_count": self.arena_count,
            "total_freed_mb": self.total_freed_mb,
            "gc_count": self.gc_count,
            "emergency_gc_count": self.emergency_gc_count,
            "pressure_level": self.pressure_level.value,
            "freeable_mb": self.freeable_mb,
            "reserved_mb": self.reserved_mb,
            "active_stages": self.active_stages,
            "active_borrows": self.active_borrows
        }


# ========================================================================
# MemoryGuardian - Allocation Broker
# ========================================================================

class MemoryGuardian:
    """
    Allocation Authority & Borrow Enforcement.

    Law 1: "The pipeline requests. The Guardian decides."
    Law 2: "No stage touches memory without borrowing through Arena."
    Law 3: "The Guardian can revoke any borrow at any time."
    """

    def __init__(
            self,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        self._status_reporter = status_reporter
        self._music_box = music_box

        # Arena registry
        self._arenas: Dict[str, 'MemoryArena'] = {}
        self._current_arena: Optional[str] = None

        # Stage registry (NEW - Guardian knows who is running)
        self._stages: Dict[str, StageRecord] = {}
        self._current_stage: Optional[str] = None

        # Buffer tracking (simplified - Arena is primary owner)
        self._global_buffer_count: int = 0
        self._global_buffer_bytes: int = 0

        # Statistics
        self._peak_memory_mb: float = 0.0
        self._total_freed_mb: float = 0.0
        self._gc_count: int = 0
        self._emergency_gc_count: int = 0

        # Memory budget
        self._memory_limit_mb = MEMORY_LIMIT_MB
        self._reserved_memory_mb: float = 0.0
        self._reservations: Dict[str, float] = {}  # stage -> reserved_mb

        # Pressure callbacks
        self._pressure_callbacks: List[Callable[[MemoryPressureLevel], None]] = []

        self._log_status("MemoryGuardian 5.1 initialized - Allocation Broker active")

    # ========================================================================
    # Allocation Broker API (NEW)
    # ========================================================================

    def request_allocation(
            self,
            requester: str,
            estimated_mb: int,
            priority: 'MemoryTier' = None,
            can_evict: bool = True,
            can_degrade: bool = True,
            timeout_seconds: float = 30.0
    ) -> AllocationResult:
        """
        Request memory allocation. Guardian decides.

        This is the PRIMARY entry point for any stage needing memory.

        Args:
            requester: Stage name requesting allocation
            estimated_mb: Estimated memory needed
            priority: Memory tier (defaults to TIER_2 for normal operations)
            can_evict: Whether Guardian can evict other buffers
            can_degrade: Whether Guardian can suggest model degradation
            timeout_seconds: How long to wait for eviction

        Returns:
            AllocationResult with approval/denial and suggestions
        """
        from memory.arena import MemoryTier
        if priority is None:
            priority = MemoryTier.TIER_2

        # Update stage registration
        self._ensure_stage_registered(requester)

        # Check current state
        current_mb = self._get_current_memory_mb()
        pressure = self._get_pressure_level(current_mb)

        # Emergency: reject all non-critical allocations
        if pressure == MemoryPressureLevel.EMERGENCY and priority.value > 0:
            return AllocationResult.denied(
                AllocationStatus.DENIED_PRESSURE_TOO_HIGH,
                f"Emergency pressure ({current_mb:.1f}MB > {self._memory_limit_mb}MB)",
                estimated_mb
            )

        # Calculate available memory
        available_mb = self._memory_limit_mb - current_mb - self._reserved_memory_mb
        freeable_mb = self._estimate_freeable_memory_mb() if can_evict else 0.0
        total_available = available_mb + freeable_mb

        self._log_status(
            f"Allocation request: {requester} needs {estimated_mb}MB "
            f"(available: {available_mb:.1f}MB, freeable: {freeable_mb:.1f}MB, "
            f"reserved: {self._reserved_memory_mb:.1f}MB)",
            "info" if estimated_mb < total_available else "warn"
        )

        # Case 1: Enough memory available
        if available_mb >= estimated_mb:
            self._reserve_memory(requester, estimated_mb)
            self._update_stage_allocation(requester, estimated_mb)
            return AllocationResult.approved(estimated_mb)

        # Case 2: Can evict to make room
        if can_evict and total_available >= estimated_mb:
            needed = estimated_mb - available_mb
            freed = self._evict_for_space(needed, requester)

            if freed >= needed:
                self._reserve_memory(requester, estimated_mb)
                self._update_stage_allocation(requester, estimated_mb)
                self._log_status(f"Eviction successful: freed {freed:.1f}MB for {requester}")
                return AllocationResult.approved(estimated_mb)

        # Case 3: Can degrade model to reduce memory
        if can_degrade and priority.value > 0:
            degraded_estimate = self._suggest_degradation(requester, estimated_mb)
            if degraded_estimate <= total_available:
                return AllocationResult.approved_with_degradation(
                    degraded_estimate,
                    estimated_mb,
                    f"Use degraded model (needs {degraded_estimate:.0f}MB vs {estimated_mb}MB)"
                )

        # Case 4: Denied
        return AllocationResult.denied(
            AllocationStatus.DENIED_INSUFFICIENT_MEMORY,
            f"Need {estimated_mb}MB, only {total_available:.1f}MB available",
            estimated_mb
        )

    def release_allocation(self, requester: str, actual_mb: Optional[float] = None) -> None:
        """Release previously allocated memory reservation."""
        if requester in self._reservations:
            released = self._reservations.pop(requester)
            self._reserved_memory_mb -= released
            self._log_status(f"Released {released:.1f}MB reservation for {requester}")

        if requester in self._stages:
            self._stages[requester].allocated_mb = 0
            if actual_mb:
                self._stages[requester].memory_peak_mb = max(
                    self._stages[requester].memory_peak_mb, actual_mb
                )

    def _reserve_memory(self, requester: str, mb: float) -> None:
        """Internal: Reserve memory for a stage."""
        if requester in self._reservations:
            self._reserved_memory_mb -= self._reservations[requester]
        self._reservations[requester] = mb
        self._reserved_memory_mb += mb

    def _evict_for_space(self, needed_mb: float, requester: str) -> float:
        """Evict buffers to free memory."""
        from memory.arena import EvictionStrategy

        freed = 0.0

        # Try balanced eviction first
        for arena in self._arenas.values():
            if freed >= needed_mb:
                break
            if hasattr(arena, 'release_evictable'):
                arena_freed = arena.release_evictable(EvictionStrategy.BALANCED)
                freed += arena_freed / (1024 * 1024)

        # If still not enough, try aggressive
        if freed < needed_mb:
            for arena in self._arenas.values():
                if freed >= needed_mb:
                    break
                if hasattr(arena, 'release_evictable'):
                    arena_freed = arena.release_evictable(EvictionStrategy.AGGRESSIVE)
                    freed += arena_freed / (1024 * 1024)

        # Force GC
        if freed < needed_mb:
            gc.collect()
            self._gc_count += 1

        return freed

    def _suggest_degradation(self, requester: str, requested_mb: int) -> float:
        """Suggest a degraded model with lower memory footprint."""
        # Simple degradation suggestions based on stage
        degradation_map = {
            "separation": 0.5,  # demucs -> lightweight
            "pitch_detection": 0.6,  # crepe -> basic_pitch
            "rhythm_detection": 0.7,
            "feature_extraction": 0.5
        }

        factor = degradation_map.get(requester, 0.8)
        return requested_mb * factor

    # ========================================================================
    # Stage Registration & Enforcement (NEW)
    # ========================================================================

    def register_stage(self, stage_name: str) -> None:
        """Register a stage with Guardian."""
        self._ensure_stage_registered(stage_name)
        self._stages[stage_name].status = StageStatus.PENDING
        self._log_status(f"Stage registered: {stage_name}")

    def stage_starting(self, stage_name: str) -> None:
        """Notify Guardian that a stage is starting."""
        self._ensure_stage_registered(stage_name)
        self._stages[stage_name].status = StageStatus.RUNNING
        self._stages[stage_name].started_at = datetime.now()
        self._current_stage = stage_name
        self._log_status(f"Stage starting: {stage_name}")

    def stage_completed(self, stage_name: str, success: bool) -> None:
        """Notify Guardian that a stage completed."""
        if stage_name in self._stages:
            self._stages[stage_name].status = StageStatus.COMPLETED if success else StageStatus.FAILED
            self._stages[stage_name].completed_at = datetime.now()

            # Release any remaining reservations
            self.release_allocation(stage_name)

            self._log_status(f"Stage {'completed' if success else 'failed'}: {stage_name}")

    def revoke_stage_borrows(self, stage_name: str, force: bool = False) -> int:
        """
        Revoke all borrows held by a stage.

        Called when a stage times out or misbehaves.

        Returns:
            Number of borrows revoked
        """
        if stage_name not in self._stages:
            return 0

        revoked = 0
        stage = self._stages[stage_name]

        for arena in self._arenas.values():
            if hasattr(arena, 'revoke_borrows_by_agent'):
                revoked += arena.revoke_borrows_by_agent(stage_name, force)

        stage.active_borrows.clear()
        self._log_status(f"Revoked {revoked} borrows for stage {stage_name}" +
                         (" (force)" if force else ""), "warn")

        return revoked

    def _ensure_stage_registered(self, stage_name: str) -> None:
        """Ensure a stage is registered."""
        if stage_name not in self._stages:
            self._stages[stage_name] = StageRecord(name=stage_name, status=StageStatus.PENDING)

    def _update_stage_allocation(self, stage_name: str, mb: float) -> None:
        """Update stage's memory allocation tracking."""
        if stage_name in self._stages:
            self._stages[stage_name].allocated_mb = mb
            self._stages[stage_name].memory_peak_mb = max(
                self._stages[stage_name].memory_peak_mb, mb
            )

    # ========================================================================
    # Arena Management
    # ========================================================================

    def register_arena(self, arena: 'MemoryArena') -> None:
        """Register an arena with Guardian."""
        self._arenas[arena.name] = arena
        self._log_status(f"Arena '{arena.name}' registered")

    def unregister_arena(self, arena_name: str) -> None:
        """Unregister an arena."""
        if arena_name in self._arenas:
            del self._arenas[arena_name]
            self._log_status(f"Arena '{arena_name}' unregistered")

    def get_arena(self, name: str) -> Optional['MemoryArena']:
        """Get an arena by name."""
        return self._arenas.get(name)

    def set_current_arena(self, arena_name: str) -> None:
        """Set the current arena (for context)."""
        self._current_arena = arena_name

    # ========================================================================
    # Borrow Tracking (Enforcement)
    # ========================================================================

    def record_borrow(self, buffer_id: 'BufferID', agent_name: str) -> None:
        """Record that a borrow occurred (called by Arena)."""
        # Track globally for enforcement
        stage_name = self._current_stage or agent_name
        self._ensure_stage_registered(stage_name)
        self._stages[stage_name].active_borrows.add(str(buffer_id))

    def record_release(self, buffer_id: 'BufferID', agent_name: str) -> None:
        """Record that a borrow was released (called by Arena)."""
        stage_name = self._current_stage or agent_name
        if stage_name in self._stages:
            self._stages[stage_name].active_borrows.discard(str(buffer_id))

    # ========================================================================
    # Predictive Memory Management
    # ========================================================================

    def get_eviction_summary(self) -> Dict[str, Any]:
        """Get summary of evictable memory across all arenas."""
        from memory.arena import EvictionStrategy

        total_freeable = 0.0
        by_strategy = {}

        for strategy in EvictionStrategy:
            if strategy == EvictionStrategy.NONE:
                continue

            freeable = 0.0
            for arena in self._arenas.values():
                if hasattr(arena, 'get_evictable_buffers'):
                    buffers = arena.get_evictable_buffers(strategy)
                    for buf in buffers:
                        freeable += buf.size_mb
            by_strategy[strategy.value] = freeable
            total_freeable += freeable

        return {
            "total_freeable_mb": total_freeable,
            "by_strategy": by_strategy,
            "reserved_mb": self._reserved_memory_mb
        }

    def get_freeable_memory_mb(self) -> float:
        """Estimate how much memory could be freed by eviction."""
        freeable = 0.0
        for arena in self._arenas.values():
            if hasattr(arena, 'get_freeable_memory_mb'):
                freeable += arena.get_freeable_memory_mb()
        return freeable

    # ========================================================================
    # Memory Monitoring
    # ========================================================================

    def _get_current_memory_mb(self) -> float:
        """Get current process memory usage."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except (ImportError, psutil.NoSuchProcess, psutil.AccessDenied):
            # Fallback to tracked buffers
            total_bytes = 0
            for arena in self._arenas.values():
                stats = arena.get_stats()
                total_bytes += stats.total_bytes
            return total_bytes / (1024 * 1024)

    def _get_pressure_level(self, current_mb: float) -> MemoryPressureLevel:
        """Determine pressure level based on current memory."""
        if current_mb > self._memory_limit_mb:
            return MemoryPressureLevel.EMERGENCY
        if current_mb > MEMORY_CRITICAL_THRESHOLD_MB:
            return MemoryPressureLevel.CRITICAL
        if current_mb > MEMORY_WARNING_THRESHOLD_MB:
            return MemoryPressureLevel.WARNING
        return MemoryPressureLevel.NORMAL

    def _estimate_freeable_memory_mb(self) -> float:
        """Estimate freeable memory across all arenas."""
        return self.get_freeable_memory_mb()

    def check_memory_pressure(self) -> MemoryPressureLevel:
        """Check current memory pressure and trigger callbacks."""
        current = self._get_current_memory_mb()

        if current > self._peak_memory_mb:
            self._peak_memory_mb = current

        pressure = self._get_pressure_level(current)

        if pressure == MemoryPressureLevel.EMERGENCY:
            self._log_status(f"EMERGENCY: Memory at {current:.1f}MB > {self._memory_limit_mb}MB", "error")
            self._notify_pressure_callbacks(pressure)
        elif pressure == MemoryPressureLevel.CRITICAL:
            self._log_status(f"CRITICAL: Memory at {current:.1f}MB > {MEMORY_CRITICAL_THRESHOLD_MB}MB", "warn")
            self._notify_pressure_callbacks(pressure)
        elif pressure == MemoryPressureLevel.WARNING:
            self._log_status(f"WARNING: Memory at {current:.1f}MB > {MEMORY_WARNING_THRESHOLD_MB}MB", "warn")

        return pressure

    def register_pressure_callback(self, callback: Callable[[MemoryPressureLevel], None]) -> None:
        """Register callback for memory pressure events."""
        self._pressure_callbacks.append(callback)

    def _notify_pressure_callbacks(self, level: MemoryPressureLevel) -> None:
        """Notify all pressure callbacks."""
        for callback in self._pressure_callbacks:
            try:
                callback(level)
            except Exception as e:
                self._log_status(f"Pressure callback failed: {e}", "error")

    # ========================================================================
    # Staggered GC
    # ========================================================================

    def staggered_gc(self, stage_name: str = "") -> Dict[str, Any]:
        """Run staggered GC after stage completion."""
        stage = stage_name or self._current_stage or "unknown"
        self._log_status(f"Running staggered GC after stage: {stage}")

        if GC_COLLECT_AFTER_EACH_STAGE:
            collected = gc.collect()
            self._gc_count += 1
        else:
            collected = 0

        current_mb = self._get_current_memory_mb()

        result = {
            "stage": stage,
            "gc_collected_objects": collected,
            "current_memory_mb": current_mb,
            "peak_memory_mb": self._peak_memory_mb,
            "total_freed_mb": self._total_freed_mb,
            "gc_count": self._gc_count,
            "arena_count": len(self._arenas)
        }

        pressure = self.check_memory_pressure()
        if pressure in (MemoryPressureLevel.CRITICAL, MemoryPressureLevel.EMERGENCY):
            second_pass = gc.collect()
            result["auto_second_pass"] = True
            result["second_pass_objects"] = second_pass
            self._emergency_gc_count += 1

            for arena in self._arenas.values():
                if hasattr(arena, 'release_evictable'):
                    from memory.arena import EvictionStrategy
                    arena.release_evictable(EvictionStrategy.BALANCED)

        if DELAY_BETWEEN_STAGGERED_DELETIONS_MS > 0:
            time.sleep(DELAY_BETWEEN_STAGGERED_DELETIONS_MS / 1000.0)

        return result

    def force_gc(self, reason: str = "manual") -> Dict[str, Any]:
        """Force emergency garbage collection."""
        self._log_status(f"Forcing emergency GC: {reason}", "warn")

        before_mb = self._get_current_memory_mb()
        collected = gc.collect()
        after_mb = self._get_current_memory_mb()
        self._gc_count += 1
        self._emergency_gc_count += 1

        result = {
            "reason": reason,
            "collected_objects": collected,
            "memory_before_mb": before_mb,
            "memory_after_mb": after_mb,
            "freed_mb": before_mb - after_mb,
            "gc_count": self._gc_count,
            "emergency_gc_count": self._emergency_gc_count
        }

        if self._music_box:
            self._music_box.log_decision(
                stage_name="memory_guardian",
                decision_type="forced_gc",
                before_state={"memory_mb": before_mb},
                after_state={"memory_mb": after_mb},
                reasoning=reason,
                reversible=False
            )

        return result

    # ========================================================================
    # Emergency Cleanup
    # ========================================================================

    def emergency_cleanup(self) -> Dict[str, Any]:
        """Emergency cleanup - release as much as possible."""
        self._log_status("EMERGENCY CLEANUP - releasing all arenas", "error")

        freed_mb = 0
        freed_count = 0

        for arena in self._arenas.values():
            if hasattr(arena, 'release_all'):
                result = arena.release_all(keep_pinned=False)
                freed_mb += result.get("freed_mb", 0)
                freed_count += result.get("freed_count", 0)

            if hasattr(arena, 'release_evictable'):
                from memory.arena import EvictionStrategy
                arena.release_evictable(EvictionStrategy.AGGRESSIVE)

        gc.collect()

        # Reset all stage borrow tracking
        for stage in self._stages.values():
            stage.active_borrows.clear()

        result = {
            "emergency": True,
            "freed_mb": freed_mb,
            "freed_count": freed_count,
            "remaining_arenas": len(self._arenas)
        }

        self._log_status(f"Emergency cleanup complete: {freed_mb:.1f}MB freed", "warn")
        return result

    def clear_all_buffers(self) -> Dict[str, Any]:
        """Clear all buffers (alias for emergency_cleanup)."""
        return self.emergency_cleanup()

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_memory_report(self) -> MemoryReport:
        """Get comprehensive memory report."""
        current = self._get_current_memory_mb()
        pressure = self._get_pressure_level(current)

        tracked_bytes = 0
        arena_count = len(self._arenas)

        for arena in self._arenas.values():
            stats = arena.get_stats()
            tracked_bytes += stats.total_bytes

        freeable_mb = self.get_freeable_memory_mb()
        active_borrows = sum(len(s.active_borrows) for s in self._stages.values())

        return MemoryReport(
            current_mb=current,
            peak_mb=self._peak_memory_mb,
            tracked_buffers=0,  # No longer tracked directly
            tracked_bytes=tracked_bytes,
            arena_count=arena_count,
            total_freed_mb=self._total_freed_mb,
            gc_count=self._gc_count,
            emergency_gc_count=self._emergency_gc_count,
            pressure_level=pressure,
            freeable_mb=freeable_mb,
            reserved_mb=self._reserved_memory_mb,
            active_stages=len([s for s in self._stages.values() if s.status == StageStatus.RUNNING]),
            active_borrows=active_borrows
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get guardian statistics."""
        report = self.get_memory_report()
        return {
            "current_mb": report.current_mb,
            "peak_mb": report.peak_mb,
            "arena_count": report.arena_count,
            "total_freed_mb": self._total_freed_mb,
            "gc_count": self._gc_count,
            "emergency_gc_count": self._emergency_gc_count,
            "pressure_level": report.pressure_level.value,
            "reserved_memory_mb": self._reserved_memory_mb,
            "freeable_mb": report.freeable_mb,
            "active_stages": report.active_stages,
            "active_borrows": report.active_borrows
        }

    def get_stage_status(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific stage."""
        if stage_name not in self._stages:
            return None

        stage = self._stages[stage_name]
        return {
            "name": stage.name,
            "status": stage.status.value,
            "started_at": stage.started_at.isoformat() if stage.started_at else None,
            "completed_at": stage.completed_at.isoformat() if stage.completed_at else None,
            "active_borrows": len(stage.active_borrows),
            "allocated_mb": stage.allocated_mb,
            "memory_peak_mb": stage.memory_peak_mb,
            "eviction_count": stage.eviction_count
        }

    def is_critical(self) -> bool:
        """Check if memory is at critical level."""
        current = self._get_current_memory_mb()
        return current > MEMORY_CRITICAL_THRESHOLD_MB

    def is_emergency(self) -> bool:
        """Check if memory is at emergency level."""
        current = self._get_current_memory_mb()
        return current > self._memory_limit_mb

    # ========================================================================
    # Context Manager
    # ========================================================================

    @contextmanager
    def session(self):
        """Context manager for a memory session."""
        self._log_status("Starting memory session")

        try:
            yield self
        finally:
            self.emergency_cleanup()
            gc.collect()

            report = self.get_memory_report()
            self._log_status(
                f"Session ended. Peak: {report.peak_mb:.1f}MB, "
                f"Total freed: {self._total_freed_mb:.1f}MB, "
                f"GC cycles: {self._gc_count}"
            )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _log_status(self, message: str, level: str = "info"):
        """Log status message."""
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info("MemoryGuardian", message)
            elif level == "warn":
                self._status_reporter.warn("MemoryGuardian", message)
            elif level == "error":
                self._status_reporter.error("MemoryGuardian", message)


# ========================================================================
# Convenience Functions
# ========================================================================

def create_memory_guardian(
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> MemoryGuardian:
    """Create a configured MemoryGuardian."""
    return MemoryGuardian(
        status_reporter=status_reporter,
        music_box=music_box
    )


# ========================================================================
# Standalone Test
# ========================================================================

def quick_guardian_test():
    """Quick test of Guardian 5.1 features."""
    print("\n" + "=" * 60)
    print("MemoryGuardian 5.1 Test - Allocation Broker")
    print("=" * 60)

    guardian = create_memory_guardian()

    # Test allocation requests
    print("\n1. Allocation Requests:")

    # Request that should be approved
    result1 = guardian.request_allocation("test_stage", 100)
    print(f"   Request 100MB: {result1.status.value} - {result1.reason or 'OK'}")

    # Request that might need eviction
    result2 = guardian.request_allocation("big_stage", 500, can_evict=True)
    print(f"   Request 500MB with eviction: {result2.status.value}")

    # Release
    guardian.release_allocation("test_stage")
    guardian.release_allocation("big_stage")

    print("\n2. Stage Registration:")
    guardian.register_stage("rhythm_detection")
    guardian.stage_starting("rhythm_detection")
    print(f"   Stage status: {guardian.get_stage_status('rhythm_detection')}")
    guardian.stage_completed("rhythm_detection", True)

    print("\n3. Memory Report:")
    report = guardian.get_memory_report()
    print(f"   Current: {report.current_mb:.1f}MB")
    print(f"   Peak: {report.peak_mb:.1f}MB")
    print(f"   Pressure: {report.pressure_level.value}")
    print(f"   Freeable: {report.freeable_mb:.1f}MB")
    print(f"   Reserved: {report.reserved_mb:.1f}MB")
    print(f"   Active stages: {report.active_stages}")

    print("\n" + "=" * 60)
    print("MemoryGuardian 5.1 test complete.")
    print("=" * 60)


if __name__ == "__main__":
    quick_guardian_test()