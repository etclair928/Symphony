# =================================================================
# MODULE: memory/__init__.py
# DESCRIPTION: Memory management - guardian and arena.
#
# VERSION: 5.6.1 (Fixed - removed non-existent borrowed_buffer import)
# UPDATED: 2026-05-16
#
# PHILOSOPHY:
#     Law of Resource Survival: "The math must never exceed the machine."
#     Law of Borrowing: "Agents borrow, arena owns. Pipeline never touches raw arrays."
#     Law of Allocation: "The pipeline requests. The Guardian decides."
#
#     These components enforce strict memory discipline, preventing the
#     "Memory Wall" crashes that plagued versions 4.5-4.7 when processing
#     long, high-fidelity recordings.
#
# COMPONENTS:
#     - MemoryGuardian: Allocation broker, enforce borrow discipline
#     - MemoryArena: Scoped memory arena with typed borrows
#     - Buffer pooling, zero-copy transfer, staggered GC, eviction tiers
#
# 5 STRATEGIES INTEGRATION:
#     Strategy 1 (Streaming Windows): windowed_arena() for per-window buffers
#     Strategy 2 (Disk-Backed): Buffer pooling reduces allocations
#     Strategy 3 (mmap for Audit): Zero-copy buffer transfer
#     Strategy 4 (GC Discipline): Staggered garbage collection
#     Strategy 5 (Sparse Representations): Track metadata only
#
# ARCHITECTURE LAYERS:
#     Guardian (Policy) → Arena (Resource Owner) → Pipeline (Coordinator)
#
# USAGE:
#     from memory import MemoryGuardian, MemoryArena, scoped_arena
#
#     guardian = MemoryGuardian()
#     with scoped_arena(guardian, "pitch_stage") as arena:
#         # Store buffer (returns BufferID)
#         buffer_id = arena.store("pitch_buffer", audio_data)
#
#         # Borrow with type guarantees
#         with arena.borrow_mono(buffer_id, "pitch_agent") as audio:
#             process(audio)  # Guaranteed: 1D, float32, correct SR
#
#         # Buffer auto-freed when context exits
# =================================================================

# Import main classes
from memory.guardian import (
    MemoryGuardian,
    MemoryReport,
    MemoryPressureLevel,
    AllocationResult,
    AllocationStatus,
    StageStatus,
    create_memory_guardian
)

from memory.arena import (
    MemoryArena,
    BufferID,
    BufferSlot,
    BufferStats,
    ArenaStats,
    BufferState,
    ArenaPriority,
    BorrowScope,
    MemoryTier,
    EvictionStrategy,
    EvictableBuffer,
    BorrowRecord,
    scoped_arena,
    nested_arenas,
)

# Version identifier
__version__ = "5.6.1"
__author__ = "DeepSeek"

# Module docstring
__doc__ = """
Grimlock 5.3 Memory Management
===============================

Three Laws of Memory Management:
-------------------------------
1. Law of Resource Survival: "The math must never exceed the machine."
2. Law of Borrowing: "Agents borrow, arena owns. Pipeline never touches raw arrays."
3. Law of Allocation: "The pipeline requests. The Guardian decides."

This module enforces strict memory discipline to prevent the "Memory Wall"
crashes that plagued versions 4.5-4.7 when processing long, high-fidelity
recordings.

Components:
-----------
MemoryGuardian (Policy Layer):
    Allocation broker enforcing memory policy.
    Features: allocation requests, stage registration, borrow revocation,
              memory pressure monitoring, progressive degradation.

MemoryArena (Resource Layer):
    Scoped memory arena with enforced ownership.
    Features: typed borrows (mono/stereo/rate), buffer pooling, zero-copy
              transfer, eviction tiers, borrow tracking, revocation.

Architecture Layers:
--------------------
Guardian (Policy) → Arena (Resource Owner) → Pipeline (Coordinator)

- Guardian decides IF memory can be allocated
- Arena owns the memory and provides safe access
- Pipeline coordinates stages using BufferIDs only

Usage Examples:
--------------
# Basic arena with typed borrow
from memory import MemoryGuardian, scoped_arena, MemoryTier

guardian = MemoryGuardian()
with scoped_arena(guardian, "pitch_stage") as arena:
    # Store buffer (returns BufferID, not raw array)
    buffer_id = arena.store("audio", audio_data, tier=MemoryTier.TIER_1)

    # Borrow with type guarantees - Arena handles all transformations
    with arena.borrow_mono(buffer_id, "pitch_agent") as audio:
        # audio is guaranteed: 1D, float32
        process(audio)

# Guardian allocation request (automatic within pipeline)
result = guardian.request_allocation("separation", estimated_mb=800)
if result.approved:
    # Proceed with separation
    pass

# Stage registration for tracking
guardian.register_stage("rhythm_detection")
guardian.stage_starting("rhythm_detection")
# ... stage execution ...
guardian.stage_completed("rhythm_detection", success=True)

# Revocation on timeout (Guardian enforces)
guardian.revoke_stage_borrows("misbehaving_stage", force=True)

# Nested arenas (parent can share with child)
with nested_arenas(guardian, "separation", "detection") as (parent, child):
    parent.store("mix", full_audio)
    child.store("drums", drums_stem)
    # child can access parent buffers via borrow

# Zero-copy buffer transfer
with scoped_arena(guardian, "source") as source:
    with scoped_arena(guardian, "target") as target:
        buffer_id = source.store("buffer", data)
        source.transfer_to(buffer_id, target)  # Zero-copy!

5 Strategies Integration:
-------------------------
1. Streaming Windows: scoped_arena() for per-window buffers
2. Disk-Backed: Buffer pooling reduces allocations (arena._pool)
3. mmap for Audit: Zero-copy buffer transfer (transfer_to)
4. GC Discipline: Staggered garbage collection (guardian.staggered_gc)
5. Sparse Representations: Track BufferIDs only, not raw arrays
"""

# All exports
__all__ = [
    # Version
    "__version__",

    # Guardian (Policy)
    "MemoryGuardian",
    "MemoryReport",
    "MemoryPressureLevel",
    "AllocationResult",
    "AllocationStatus",
    "StageStatus",
    "create_memory_guardian",

    # Arena (Resource)
    "MemoryArena",
    "BufferID",
    "BufferSlot",
    "BufferStats",
    "ArenaStats",
    "BufferState",
    "ArenaPriority",
    "BorrowScope",
    "MemoryTier",
    "EvictionStrategy",
    "EvictableBuffer",
    "BorrowRecord",
    "scoped_arena",
    "nested_arenas",
]


# ========================================================================
# Module Initialization Check
# ========================================================================

def _check_imports() -> bool:
    """Verify all memory components are importable."""
    missing = []

    try:
        from memory.guardian import MemoryGuardian
    except ImportError as e:
        missing.append(f"guardian: {e}")

    try:
        from memory.arena import MemoryArena
    except ImportError as e:
        missing.append(f"arena: {e}")

    if missing:
        import warnings
        warnings.warn(f"Some memory components failed to import: {missing}", ImportWarning)
        return False

    return True


_IMPORTS_OK = _check_imports()


# ========================================================================
# Convenience Windowed Arena (Strategy 1 Integration)
# ========================================================================

def windowed_arena(guardian: MemoryGuardian, base_name: str, window_idx: int, **kwargs):
    """
    Create a windowed arena for streaming processing.

    This is a convenience wrapper around scoped_arena for Strategy 1
    (Streaming Windows). Each window gets its own arena that auto-clears
    when the window is done.

    Args:
        guardian: MemoryGuardian instance
        base_name: Base name for the arena (will be suffixed with window index)
        window_idx: Window index (for unique naming)
        **kwargs: Additional arguments to MemoryArena

    Example:
        for i in range(num_windows):
            with windowed_arena(guardian, "analysis", i) as arena:
                window_data = get_window(i)
                buffer_id = arena.store(f"window_{i}_data", window_data)
                with arena.borrow_mono(buffer_id, "processor") as audio:
                    process(audio)
                # Arena cleared after each window
    """
    arena_name = f"{base_name}_window_{window_idx:04d}"
    return scoped_arena(guardian, arena_name, **kwargs)


# ========================================================================
# Module Metadata
# ========================================================================

version_info = {
    "module": "memory",
    "version": __version__,
    "imports_ok": _IMPORTS_OK,
    "architecture": {
        "guardian_role": "allocation_broker",
        "arena_role": "resource_owner",
        "borrow_semantics": "enforced",
        "typed_borrows": ["mono", "stereo", "at_rate", "feature_bundle"],
        "eviction_tiers": ["TIER_0", "TIER_1", "TIER_2", "TIER_3", "SPILLABLE"]
    }
}


# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    import numpy as np

    print("\n" + "=" * 70)
    print("GRIMLOCK 5.3 - MEMORY MANAGEMENT")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")
    print(f"Architecture: {version_info['architecture']}")

    # Quick test
    print("\n" + "-" * 40)
    print("Quick Test:")
    print("-" * 40)

    from memory.guardian import MemoryGuardian, create_memory_guardian
    from memory.arena import scoped_arena, nested_arenas, MemoryTier

    guardian = create_memory_guardian()

    # Test 1: Basic arena with typed borrow
    print("\n1. Basic Arena with Typed Borrow:")
    with scoped_arena(guardian, "test") as arena:
        # Create stereo test audio
        stereo_audio = np.random.rand(100000, 2).astype(np.float32)

        # Store buffer (returns BufferID)
        buffer_id = arena.store("test_stereo", stereo_audio, tier=MemoryTier.TIER_2)
        print(f"   Buffer stored: {buffer_id.short}")

        # Test borrow_mono (auto-converts to mono)
        with arena.borrow_mono(buffer_id, "test_agent") as mono:
            print(f"   borrow_mono: shape={mono.shape}, dtype={mono.dtype}")
            assert mono.ndim == 1, "Should be mono"
            assert mono.dtype == np.float32, "Should be float32"

        # Test arena stats
        stats = arena.get_stats()
        print(f"   Arena buffer count: {stats.buffer_count}")
        print(f"   Arena memory: {stats.total_mb:.2f}MB")

    # Test 2: Guardian allocation request
    print("\n2. Guardian Allocation Request:")
    result = guardian.request_allocation("test_stage", 100, can_evict=True)
    print(f"   Request 100MB: {result.status.value}")
    if result.approved:
        guardian.release_allocation("test_stage")
        print("   Allocation released")

    # Test 3: Nested arenas
    print("\n3. Nested Arenas:")
    with nested_arenas(guardian, "parent", "child") as (parent, child):
        parent_id = parent.store("parent_data", np.random.rand(50000).astype(np.float32))
        child_id = child.store("child_data", np.random.rand(30000).astype(np.float32))
        print(f"   Parent buffers: {len(parent)}")
        print(f"   Child buffers: {len(child)}")

        # Child can access parent buffer
        with child.borrow_mono(parent_id, "child_agent") as audio:
            print(f"   Child accessed parent buffer: shape={audio.shape}")

    # Test 4: Windowed arena
    print("\n4. Windowed Arena:")
    for i in range(3):
        with windowed_arena(guardian, "stream", i) as arena:
            buffer_id = arena.store(f"window_data", np.random.rand(20000).astype(np.float32))
            with arena.borrow_mono(buffer_id, "window_processor") as audio:
                print(f"   Window {i}: {audio.shape[0]} samples")
            # Arena auto-clears after each window

    # Test 5: Stage registration (simulated)
    print("\n5. Stage Registration:")
    guardian.register_stage("rhythm_detection")
    guardian.stage_starting("rhythm_detection")
    print(f"   Stage status: {guardian.get_stage_status('rhythm_detection')}")
    guardian.stage_completed("rhythm_detection", True)

    # Test 6: Eviction summary
    print("\n6. Eviction Summary:")
    with scoped_arena(guardian, "evict_test") as arena:
        arena.store("tier2", np.random.rand(500000).astype(np.float32), tier=MemoryTier.TIER_2)
        arena.store("tier3", np.random.rand(300000).astype(np.float32), tier=MemoryTier.TIER_3)

        evictable = arena.get_evictable_buffers(EvictionStrategy.BALANCED)
        print(f"   Evictable buffers: {len(evictable)}")
        for buf in evictable[:2]:
            print(f"     - {buf.name}: {buf.size_mb:.1f}MB, tier={buf.tier.name}")

    # Test 7: Guardian stats
    print("\n7. Guardian Statistics:")
    report = guardian.get_memory_report()
    print(f"   Current memory: {report.current_mb:.1f}MB")
    print(f"   Peak memory: {report.peak_mb:.1f}MB")
    print(f"   Pressure level: {report.pressure_level.value}")
    print(f"   Freeable memory: {report.freeable_mb:.1f}MB")
    print(f"   Reserved memory: {report.reserved_mb:.1f}MB")
    print(f"   Active stages: {report.active_stages}")
    print(f"   Active borrows: {report.active_borrows}")
    print(f"   GC count: {report.gc_count}")

    print("\n" + "=" * 70)
    print("Memory module ready.")
    print("=" * 70)