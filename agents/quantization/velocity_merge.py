# =================================================================
# MODULE: agents/quantization/velocity_merge.py
# DESCRIPTION: Velocity Merge - merges overlapping or adjacent notes
# with intelligent velocity weighting.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# PHILOSOPHY:
#     "Overlapping notes are usually detection errors or voice stealing.
#      Merging them improves readability while preserving musical intent."
#
#     This agent produces immutable evidence about note groups that
#     should be merged. It does NOT mutate original events.
#
# KEY ARCHITECTURE:
#     - Immutable transformations using replace()
#     - Multiple merge strategies (weighted average, max velocity, etc.)
#     - Overlap and adjacency detection
#     - Forensic audit trail of all merges
#     - Ghost note preservation
#     - StatusReporter integration
#     - No agent imports - pure transformer
#
# Authored by: DeepSeek - Complete 5.0 rewrite (2026-05-11)
# Based on 4.7 Velocity_Merge and voice continuity patterns.
# =================================================================

import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable, Set
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from collections import defaultdict

# Core imports - ONLY from bedrock
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    MergeStrategy, MergeCandidate, VelocityMergeResult, DecisionType
)
from core.constants import (
    OVERLAP_TOLERANCE_MS,
    MIN_VELOCITY_TO_KEEP,
    DEFAULT_GHOST_NOTE_VELOCITY,
    MAX_VELOCITY,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    QuantizationAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)


# ========================================================================
# Enums and Types
# ========================================================================

class MergePriority(str, Enum):
    """Priority for which note to keep when merging in single-note strategies."""
    HIGHEST_CONFIDENCE = "highest_confidence"
    HIGHEST_VELOCITY = "highest_velocity"
    LONGEST_DURATION = "longest_duration"
    EARLIEST_ONSET = "earliest_onset"
    LATEST_OFFSET = "latest_offset"


class MergeOutcome(str, Enum):
    """Outcome of a merge attempt."""
    MERGED = "merged"
    PRESERVED = "preserved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class MergeRecord:
    """Forensic record of a single merge operation."""
    original_note_indices: Tuple[int, ...]
    original_pitches: Tuple[int, ...]
    merged_pitch: int
    merged_start_ms: float
    merged_end_ms: float
    merged_velocity: int
    merged_confidence: float
    strategy_used: MergeStrategy
    confidence_change: float


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class VelocityMergeConfig:
    """Configuration for velocity merge."""

    # Merge detection
    overlap_tolerance_ms: float = OVERLAP_TOLERANCE_MS  # 10ms
    adjacent_gap_ms: float = 20.0  # Max gap to consider adjacent
    min_overlap_ratio: float = 0.1  # Minimum overlap ratio (10%)

    # Merge strategy
    default_strategy: MergeStrategy = MergeStrategy.WEIGHTED_AVERAGE
    priority: MergePriority = MergePriority.HIGHEST_CONFIDENCE

    # Velocity handling
    max_velocity: int = MAX_VELOCITY  # 127
    min_velocity: int = MIN_VELOCITY_TO_KEEP  # 1
    ghost_note_threshold: int = DEFAULT_GHOST_NOTE_VELOCITY  # 20

    # Weighting for weighted average
    confidence_weight: float = 0.6
    velocity_weight: float = 0.4
    duration_weight: float = 0.2

    # Post-processing
    merge_adjacent: bool = True
    preserve_ghost_notes: bool = True
    min_merge_confidence: float = 0.3

    # Pitch matching
    merge_different_pitches: bool = False  # Only merge same pitch by default
    pitch_tolerance_semitones: int = 2

    # Performance
    max_merge_candidates_per_event: int = 10
    cleanup_after_merge: bool = True


# ========================================================================
# Overlap and Adjacency Detector
# ========================================================================

class OverlapDetector:
    """
    Detects overlapping and adjacent notes.

    Pure function - no state. Returns evidence about relationships.
    """

    def __init__(self, config: VelocityMergeConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def detect_overlaps(self, events: List[NoteEvent]) -> List[Tuple[int, int, float]]:
        """
        Detect overlapping note pairs.

        Returns:
            List of (idx1, idx2, overlap_duration_ms) for overlapping pairs
        """
        overlaps = []

        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event1 = events[i]
                event2 = events[j]

                # Check if pitches are compatible for merging
                if not self._pitches_compatible(event1.pitch, event2.pitch):
                    continue

                start1 = event1.get_active_start_ms()
                end1 = event1.get_active_end_ms()
                start2 = event2.get_active_start_ms()
                end2 = event2.get_active_end_ms()

                # Check for overlap
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > self.config.overlap_tolerance_ms:
                    overlaps.append((i, j, overlap_duration))

        return overlaps

    def detect_adjacent(self, events: List[NoteEvent]) -> List[Tuple[int, int, float]]:
        """
        Detect adjacent notes (small gaps between them).

        Returns:
            List of (idx1, idx2, gap_duration_ms) for adjacent pairs
        """
        if len(events) < 2:
            return []

        # Sort by start time
        sorted_indices = sorted(range(len(events)), key=lambda i: events[i].get_active_start_ms())
        adjacent = []

        for k in range(len(sorted_indices) - 1):
            i = sorted_indices[k]
            j = sorted_indices[k + 1]

            event1 = events[i]
            event2 = events[j]

            # Check if pitches are compatible
            if not self._pitches_compatible(event1.pitch, event2.pitch):
                continue

            gap = event2.get_active_start_ms() - event1.get_active_end_ms()

            # detect_overlaps only counts overlap_duration > tolerance (a
            # positive threshold), and this required gap strictly > 0 -
            # two notes touching with EXACTLY zero gap (one ends exactly
            # where the next starts) fell through both checks and were
            # never considered mergeable at all, even though they're the
            # clearest possible legato-adjacent case.
            if 0 <= gap <= self.config.adjacent_gap_ms:
                adjacent.append((i, j, gap))

        return adjacent

    def build_merge_groups(self, events: List[NoteEvent],
                           overlaps: List[Tuple[int, int, float]],
                           adjacent: List[Tuple[int, int, float]]) -> List[List[int]]:
        """
        Build groups of notes that should be merged together.

        Uses graph connectivity to find connected components.
        """
        if not overlaps and not adjacent:
            return []

        # Build adjacency graph
        adj = defaultdict(set)

        for i, j, _ in overlaps:
            adj[i].add(j)
            adj[j].add(i)

        for i, j, _ in adjacent:
            adj[i].add(j)
            adj[j].add(i)

        # Find connected components
        visited = set()
        groups = []

        for idx in range(len(events)):
            if idx in visited:
                continue

            # BFS to find component
            queue = [idx]
            group = []

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                group.append(current)

                for neighbor in adj[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _pitches_compatible(self, pitch1: int, pitch2: int) -> bool:
        """Check if two pitches can be merged."""
        if self.config.merge_different_pitches:
            return abs(pitch1 - pitch2) <= self.config.pitch_tolerance_semitones
        return pitch1 == pitch2


# ========================================================================
# Velocity Merger (Core Logic)
# ========================================================================

class VelocityMerger:
    """
    Merges groups of notes using various strategies.

    Pure function - returns new NoteEvent, does not mutate.
    """

    def __init__(self, config: VelocityMergeConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._merge_records: List[MergeRecord] = []

    def merge_group(self, events: List[NoteEvent], indices: List[int],
                    strategy: MergeStrategy) -> Optional[NoteEvent]:
        """
        Merge a group of notes into a single note.

        Returns:
            New merged NoteEvent or None if merge failed
        """
        if len(indices) < 2:
            return None

        group = [events[i] for i in indices]

        # Sort by start time
        group.sort(key=lambda e: e.get_active_start_ms())

        # Determine merged boundaries
        merge_start = min(e.get_active_start_ms() for e in group)
        merge_end = max(e.get_active_end_ms() for e in group)

        # Choose primary pitch (most common or earliest)
        primary_pitch = self._determine_primary_pitch(group)

        # Apply merge strategy
        if strategy == MergeStrategy.MAX_VELOCITY:
            merged_velocity, merged_confidence, source = self._max_velocity_merge(group)
        elif strategy == MergeStrategy.WEIGHTED_AVERAGE:
            merged_velocity, merged_confidence, source = self._weighted_average_merge(group)
        elif strategy == MergeStrategy.LONGEST_DURATION:
            merged_velocity, merged_confidence, source = self._longest_duration_merge(group)
        elif strategy == MergeStrategy.EARLIEST_ONSET:
            merged_velocity, merged_confidence, source = self._earliest_onset_merge(group)
        else:
            # Default to weighted average
            merged_velocity, merged_confidence, source = self._weighted_average_merge(group)

        # Clamp values
        merged_velocity = max(self.config.min_velocity,
                              min(self.config.max_velocity, merged_velocity))
        merged_confidence = max(0.0, min(1.0, merged_confidence))

        # Check if this should be a ghost note
        is_ghost = merged_velocity <= self.config.ghost_note_threshold

        # Build reasoning chain
        reasoning = self._build_reasoning_chain(group, strategy, merged_velocity, is_ghost)

        # Preserve snapped timestamps from the most confident note
        snapped_start = None
        snapped_end = None
        best_note = max(group, key=lambda e: e.confidence)
        if best_note.snapped_start_ms is not None:
            snapped_start = best_note.snapped_start_ms
            snapped_end = best_note.snapped_end_ms

        # Create new immutable note. Previously built via NoteEvent(...)
        # naming only the fields the merge logic itself computed - every
        # other field (harmonic_series_match_ratio, fundamental_freq_hz,
        # midi_channel, phrase_id, consensus_votes, veto_attempts, ...)
        # silently reset to its default for every merged note. Basing it
        # on replace(best_note, ...) instead carries that upstream
        # evidence forward from the group's most-confident member (the
        # same note snapped_start_ms/snapped_end_ms already get inherited
        # from above), while still explicitly overriding every field the
        # merge computed fresh.
        merged_note = replace(
            best_note,
            pitch=primary_pitch,
            start_ms=merge_start,
            end_ms=merge_end,
            velocity=merged_velocity,
            confidence=merged_confidence,
            zero_crossing_rate=float(np.mean([e.zero_crossing_rate for e in group])),
            source=source,
            snapped_start_ms=snapped_start,
            snapped_end_ms=snapped_end,
            reasoning_chain=reasoning
        )

        # Record merge for forensic audit
        self._merge_records.append(MergeRecord(
            original_note_indices=tuple(indices),
            original_pitches=tuple(e.pitch for e in group),
            merged_pitch=primary_pitch,
            merged_start_ms=merge_start,
            merged_end_ms=merge_end,
            merged_velocity=merged_velocity,
            merged_confidence=merged_confidence,
            strategy_used=strategy,
            confidence_change=merged_confidence - np.mean([e.confidence for e in group])
        ))

        return merged_note

    def _determine_primary_pitch(self, group: List[NoteEvent]) -> int:
        """Determine the primary pitch for merged note."""
        # Use most common pitch
        from collections import Counter
        pitches = [e.pitch for e in group]
        counter = Counter(pitches)
        most_common = counter.most_common(1)
        if most_common:
            return most_common[0][0]
        return group[0].pitch

    def _max_velocity_merge(self, group: List[NoteEvent]) -> Tuple[int, float, SourceType]:
        """Keep the note with highest velocity."""
        best = max(group, key=lambda e: e.velocity)
        return best.velocity, best.confidence, best.source

    def _weighted_average_merge(self, group: List[NoteEvent]) -> Tuple[int, float, SourceType]:
        """Weighted average of velocities by confidence and duration."""
        total_weight = 0.0
        weighted_velocity = 0.0
        weighted_confidence = 0.0

        for event in group:
            # Weight by confidence and duration. confidence and
            # velocity/127 are both naturally bounded [0,1], but a bare
            # duration/1000 is not - a 5000ms note contributed 5.0
            # regardless of duration_weight, letting long notes dominate
            # the weighted average far more than the configured weight
            # ratios intended (a note twice as long as another would
            # outweigh it by that unbounded factor, not by the ~0.2/1.2
            # share duration_weight's default suggests). Capping at 1.0
            # (a full second) makes it comparable in scale to the other
            # two terms.
            duration = event.get_active_end_ms() - event.get_active_start_ms()
            weight = (event.confidence * self.config.confidence_weight +
                      (event.velocity / 127.0) * self.config.velocity_weight +
                      min(1.0, duration / 1000.0) * self.config.duration_weight)

            weighted_velocity += event.velocity * weight
            weighted_confidence += event.confidence * weight
            total_weight += weight

        if total_weight > 0:
            merged_velocity = int(weighted_velocity / total_weight)
            merged_confidence = weighted_confidence / total_weight
        else:
            merged_velocity = 64
            merged_confidence = 0.5

        # Source with highest confidence
        source = max(group, key=lambda e: e.confidence).source

        return merged_velocity, merged_confidence, source

    def _longest_duration_merge(self, group: List[NoteEvent]) -> Tuple[int, float, SourceType]:
        """Keep the note with longest duration."""
        durations = [e.get_active_end_ms() - e.get_active_start_ms() for e in group]
        best_idx = np.argmax(durations)
        best = group[best_idx]
        return best.velocity, best.confidence, best.source

    def _earliest_onset_merge(self, group: List[NoteEvent]) -> Tuple[int, float, SourceType]:
        """Keep the note that started first."""
        best = min(group, key=lambda e: e.get_active_start_ms())
        return best.velocity, best.confidence, best.source

    def _build_reasoning_chain(self, group: List[NoteEvent], strategy: MergeStrategy,
                               merged_velocity: int, is_ghost: bool) -> List[str]:
        """Build reasoning chain for merged note."""
        reasoning = [f"Merged {len(group)} notes using {strategy.value}"]

        if is_ghost:
            reasoning.append(f"Ghost note (velocity {merged_velocity} ≤ {self.config.ghost_note_threshold})")

        # Add original velocities for context
        velocities = [e.velocity for e in group]
        reasoning.append(f"Original velocities: {velocities}")

        return reasoning

    def get_merge_records(self) -> List[MergeRecord]:
        """Get all merge records from last operation."""
        return self._merge_records

    def clear_records(self):
        """Clear merge records."""
        self._merge_records = []


# ========================================================================
# Main Velocity Merge Agent
# ========================================================================

class VelocityMerge:
    """
    Velocity Merge - merges overlapping/adjacent notes intelligently.

    Features:
        - Multiple merge strategies (weighted average, max velocity, etc.)
        - Overlap and adjacency detection
        - Forensic audit trail of all merges
        - Ghost note preservation
        - Pitch-based grouping option

    Law 2 Compliance: Does NOT import other agents.
    """

    def __init__(
            self,
            config: Optional[VelocityMergeConfig] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Args:
            config: Velocity merge configuration
            status_reporter: StatusReporter for progress
            music_box: MusicBox for forensic logging
            progress_callback: Optional progress callback
        """
        self._name = "velocity_merge"
        self._source_type = SourceType.VELOCITY_MERGE
        self._config = config or VelocityMergeConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback

        # Initialize components
        self._overlap_detector = OverlapDetector(self._config, status_reporter)
        self._merger = VelocityMerger(self._config, status_reporter)

        # State
        self._last_original_events: List[NoteEvent] = []
        self._last_merged_events: List[NoteEvent] = []
        self._last_result: Optional[VelocityMergeResult] = None
        self._last_execution_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0

        self._log_status("VelocityMerge initialized")

    # ========================================================================
    # AgentProtocol Implementation
    # ========================================================================

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def name(self) -> str:
        return self._name

    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """
        Process and return StageResult with merged events.

        Note: This agent works with NoteEvents, not raw audio.
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting velocity merge")

        try:
            # Get events from context
            events = self._get_events_from_context(context)

            if not events:
                self._log_status("No events to merge", "warn")
                return StageResult(
                    stage_name=self._name,
                    success=False,
                    events=[],
                    metadata={"error": "No events provided"},
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            # Get merge parameters from context
            strategy = getattr(context, 'merge_strategy', self._config.default_strategy)
            merge_adjacent = getattr(context, 'merge_adjacent', self._config.merge_adjacent)

            # Merge
            result = self.merge_overlaps(events, strategy, merge_adjacent)

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory
            self._last_execution_time_ms = execution_time_ms

            # Force cleanup
            if self._config.cleanup_after_merge:
                self._force_cleanup()

            self._update_progress(1.0, f"Complete: {result.notes_removed} notes removed")

            return StageResult(
                stage_name=self._name,
                success=True,
                events=result.merged_notes,
                metadata={
                    "original_count": len(result.original_notes),
                    "merged_count": len(result.merged_notes),
                    "removed_count": result.notes_removed,
                    "merge_groups": len(result.merges_performed),
                    "reduction_ratio": result.notes_removed / max(len(result.original_notes), 1),
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Merge failed: {e}", "error")
            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def merge_overlaps(self, events: List[NoteEvent],
                       strategy: Optional[MergeStrategy] = None,
                       merge_adjacent: Optional[bool] = None) -> VelocityMergeResult:
        """
        Merge overlapping and adjacent notes.

        This is the main entry point for velocity merge.

        Args:
            events: List of NoteEvents to merge
            strategy: Merge strategy (default from config)
            merge_adjacent: Whether to merge adjacent notes (legato)

        Returns:
            VelocityMergeResult with original, merged notes, and statistics
        """
        self._update_progress(0.1, "Validating events")

        if not events:
            return VelocityMergeResult(
                original_notes=[],
                merged_notes=[],
                merges_performed=[],
                notes_removed=0
            )

        strategy = strategy or self._config.default_strategy
        merge_adjacent = merge_adjacent if merge_adjacent is not None else self._config.merge_adjacent

        self._last_original_events = list(events)
        self._merger.clear_records()

        # Step 1: Detect overlaps
        self._update_progress(0.2, "Detecting overlaps")
        overlaps = self._overlap_detector.detect_overlaps(events)

        # Step 2: Detect adjacent notes
        self._update_progress(0.3, "Detecting adjacent notes")
        adjacent = self._overlap_detector.detect_adjacent(events) if merge_adjacent else []

        # Step 3: Build merge groups
        self._update_progress(0.4, "Building merge groups")
        merge_groups = self._overlap_detector.build_merge_groups(events, overlaps, adjacent)

        if not merge_groups:
            self._update_progress(1.0, "No merges needed")
            if self._music_box:
                # A null result (nothing overlapped or was adjacent
                # enough to merge) is itself worth recording - previously
                # this early-return path was indistinguishable from the
                # stage never running at all.
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type=DecisionType.ANALYSIS_EVIDENCE,
                    before_state={"original_count": len(events)},
                    after_state={"merge_groups": 0, "notes_removed": 0},
                    reasoning="No overlapping or adjacent notes found to merge",
                    reversible=True
                )
            return VelocityMergeResult(
                original_notes=events,
                merged_notes=events,
                merges_performed=[],
                notes_removed=0
            )

        # Step 4: Perform merges
        self._update_progress(0.6, f"Merging {len(merge_groups)} groups")

        merged_indices: Set[int] = set()
        merged_notes: List[NoteEvent] = []

        for group in merge_groups:
            merged_note = self._merger.merge_group(events, group, strategy)
            if merged_note:
                merged_notes.append(merged_note)
                merged_indices.update(group)

        # Add notes that weren't merged
        for i, event in enumerate(events):
            if i not in merged_indices:
                merged_notes.append(event)

        # Sort by start time
        merged_notes.sort(key=lambda e: e.get_active_start_ms())

        self._last_merged_events = merged_notes
        notes_removed = len(events) - len(merged_notes)

        # Step 5: Build result
        self._update_progress(0.9, "Building result")
        merge_records = self._merger.get_merge_records()

        # Convert merge records to MergeCandidate objects
        merge_candidates = []
        for record in merge_records:
            # Need to reconstruct the original notes for the candidate
            original_notes = [events[i] for i in record.original_note_indices if i < len(events)]
            merge_candidates.append(MergeCandidate(
                notes=original_notes,
                merge_start_ms=record.merged_start_ms,
                merge_end_ms=record.merged_end_ms,
                merge_velocity=record.merged_velocity,
                merge_confidence=record.merged_confidence,
                strategy_used=record.strategy_used
            ))

        result = VelocityMergeResult(
            original_notes=events,
            merged_notes=merged_notes,
            merges_performed=merge_candidates,
            notes_removed=notes_removed
        )

        self._last_result = result

        # Log to MusicBox
        if self._music_box and notes_removed > 0:
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="velocity_merge",
                before_state={"original_count": len(events)},
                after_state={
                    "merged_count": len(merged_notes),
                    "removed_count": notes_removed,
                    "merge_groups": len(merge_groups),
                    "strategy": strategy.value
                },
                reasoning=f"Merged {len(merge_groups)} groups, removed {notes_removed} notes",
                reversible=True
            )

        self._log_status(f"Merge complete: {len(events)} → {len(merged_notes)} notes ({notes_removed} removed)")

        return result

    def merge_by_pitch(self, events: List[NoteEvent],
                       strategy: Optional[MergeStrategy] = None) -> VelocityMergeResult:
        """
        Merge notes by pitch (separate analysis per pitch class).

        Useful for instruments where different pitches shouldn't be merged.
        """
        self._update_progress(0.1, "Grouping by pitch")

        # Group by pitch
        pitch_groups: Dict[int, List[NoteEvent]] = defaultdict(list)

        for event in events:
            pitch_groups[event.pitch].append(event)

        # Merge each group
        all_merged = []
        total_removed = 0
        all_merges = []

        for idx, (pitch, pitch_events) in enumerate(pitch_groups.items()):
            progress = 0.1 + (idx / len(pitch_groups)) * 0.8
            self._update_progress(progress, f"Merging pitch {pitch}")

            if len(pitch_events) <= 1:
                all_merged.extend(pitch_events)
                continue

            result = self.merge_overlaps(pitch_events, strategy, merge_adjacent=False)
            all_merged.extend(result.merged_notes)
            total_removed += result.notes_removed
            all_merges.extend(result.merges_performed)

        # Sort by start time
        all_merged.sort(key=lambda e: e.get_active_start_ms())

        return VelocityMergeResult(
            original_notes=events,
            merged_notes=all_merged,
            merges_performed=all_merges,
            notes_removed=total_removed
        )

    def merge_legato(self, events: List[NoteEvent],
                     max_gap_ms: float = 30.0) -> VelocityMergeResult:
        """
        Specifically merge legato passages (adjacent notes with small gaps).

        Args:
            events: List of NoteEvents
            max_gap_ms: Maximum gap to consider as legato

        Returns:
            VelocityMergeResult with merged notes
        """
        # Temporarily override max gap
        original_gap = self._config.adjacent_gap_ms
        self._config.adjacent_gap_ms = max_gap_ms

        result = self.merge_overlaps(events, merge_adjacent=True)

        # Restore
        self._config.adjacent_gap_ms = original_gap

        return result

    def _get_events_from_context(self, context: AudioContext) -> List[NoteEvent]:
        """Extract events from context."""
        if hasattr(context, 'events') and context.events:
            return context.events
        if hasattr(context, 'notes') and context.notes:
            return context.notes
        return []

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

    def _update_progress(self, progress: float, message: str):
        if self._progress_callback:
            self._progress_callback(progress, message)
        if self._status_reporter:
            self._status_reporter.progress(self._name, progress, message)

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            getattr(self._status_reporter, level)(self._name, message)

    def _get_current_memory_mb(self) -> float:
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    # ========================================================================
    # ScribeValidatable Implementation
    # ========================================================================

    def validate(self, gate: ValidationGate) -> ValidationResult:
        if self._last_result is None:
            return ValidationResult(
                is_valid=False,
                gate_used=gate,
                reason=VetoReason.EMPTY_RESULT,
                detail="No merge performed"
            )

        if gate == ValidationGate.CONFIDENCE_THRESHOLD:
            avg_confidence = np.mean(
                [e.confidence for e in self._last_result.merged_notes]) if self._last_result.merged_notes else 0
            is_valid = avg_confidence >= 0.25

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                detail=f"Avg confidence: {avg_confidence:.2f}",
                confidence_before=avg_confidence,
                confidence_after=avg_confidence if is_valid else 0.0
            )

        else:
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                detail=f"Gate {gate.value} not fully supported"
            )

    def get_confidence(self) -> Confidence:
        if not self._last_merged_events:
            return Confidence.HALLUCINATION

        avg_confidence = np.mean([e.confidence for e in self._last_merged_events])
        return Confidence.from_float(avg_confidence)

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        if self._last_result and self._last_result.notes_removed > len(self._last_result.original_notes) * 0.8:
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Excessive merging: {self._last_result.notes_removed} removed")
        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        return SchoenbergResult(
            verdict=SchoenbergVerdict.UNCERTAIN,
            zero_crossing_rate=0.0,
            spectral_flatness=0.5,
            reason="VelocityMerge processes note timing, not harmonic series"
        )

    # ========================================================================
    # MemoryManagedProtocol Implementation
    # ========================================================================

    def release_buffer(self, buffer_name: str) -> None:
        pass

    def get_memory_footprint_mb(self) -> float:
        return self._get_current_memory_mb()

    def can_release(self, buffer_name: str) -> bool:
        return True

    def staggered_gc(self) -> Dict[str, Any]:
        before = self._get_current_memory_mb()
        self._force_cleanup()
        after = self._get_current_memory_mb()

        freed = before - after
        self._total_memory_freed_mb += max(0, freed)

        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": freed,
            "triggered_by": self._name,
            "total_freed_mb": self._total_memory_freed_mb
        }

    # ========================================================================
    # Public Methods
    # ========================================================================

    def get_merge_records(self) -> List[MergeRecord]:
        """Get forensic merge records from last operation."""
        return self._merger.get_merge_records()

    def analyze_velocity_distribution(self, events: List[NoteEvent]) -> Dict[str, Any]:
        """Analyze velocity distribution."""
        if not events:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        velocities = [e.velocity for e in events]

        return {
            "mean": float(np.mean(velocities)),
            "std": float(np.std(velocities)),
            "min": int(np.min(velocities)),
            "max": int(np.max(velocities)),
            "quartiles": {
                "q25": float(np.percentile(velocities, 25)),
                "q50": float(np.percentile(velocities, 50)),
                "q75": float(np.percentile(velocities, 75))
            }
        }

    def get_ghost_notes(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Get notes classified as ghost notes (low velocity)."""
        return [e for e in events if e.velocity <= self._config.ghost_note_threshold]

    def get_statistics(self) -> Dict[str, Any]:
        """Get merge statistics."""
        return {
            "name": self._name,
            "source_type": self._source_type.value,
            "original_count": len(self._last_original_events),
            "merged_count": len(self._last_merged_events),
            "removed_count": len(self._last_original_events) - len(
                self._last_merged_events) if self._last_original_events else 0,
            "merge_records": len(self._merger.get_merge_records()),
            "strategy": self._config.default_strategy.value,
            "overlap_tolerance_ms": self._config.overlap_tolerance_ms,
            "ghost_note_threshold": self._config.ghost_note_threshold,
            "last_execution_time_ms": self._last_execution_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb
        }


# ========================================================================
# Convenience Functions
# ========================================================================

def create_velocity_merge(
        overlap_tolerance_ms: float = OVERLAP_TOLERANCE_MS,
        strategy: MergeStrategy = MergeStrategy.WEIGHTED_AVERAGE,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> VelocityMerge:
    """Create a configured VelocityMerge instance."""
    config = VelocityMergeConfig(
        overlap_tolerance_ms=overlap_tolerance_ms,
        default_strategy=strategy
    )
    return VelocityMerge(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )


def quick_test_velocity_merge():
    """Quick test function for velocity merge."""
    from core.order_types import NoteEvent, SourceType

    print("=" * 60)
    print("Velocity Merge Test")
    print("=" * 60)

    # Create test events
    events = [
        NoteEvent(pitch=60, start_ms=0, end_ms=500, velocity=80, confidence=0.9,
                  zero_crossing_rate=0.0, source=SourceType.PITCH),
        NoteEvent(pitch=60, start_ms=100, end_ms=400, velocity=50, confidence=0.6,
                  zero_crossing_rate=0.0, source=SourceType.PITCH),
        NoteEvent(pitch=62, start_ms=520, end_ms=1000, velocity=70, confidence=0.8,
                  zero_crossing_rate=0.0, source=SourceType.PITCH),
        NoteEvent(pitch=64, start_ms=1500, end_ms=2000, velocity=90, confidence=0.95,
                  zero_crossing_rate=0.0, source=SourceType.PITCH),
    ]

    print(f"\nOriginal: {len(events)} events")
    for e in events:
        print(f"  pitch={e.pitch}, {e.start_ms:.0f}-{e.end_ms:.0f}ms, vel={e.velocity}, conf={e.confidence:.2f}")

    # Create merger
    merger = create_velocity_merge()

    # Merge
    result = merger.merge_overlaps(events)

    print(f"\nMerged: {len(result.merged_notes)} events")
    for e in result.merged_notes:
        print(f"  pitch={e.pitch}, {e.start_ms:.0f}-{e.end_ms:.0f}ms, vel={e.velocity}, conf={e.confidence:.2f}")

    print(f"\nRemoved: {result.notes_removed} notes")
    print(f"Merge groups: {len(result.merges_performed)}")

    # Show merge details
    for i, merge in enumerate(result.merges_performed):
        print(f"\nMerge {i + 1}: {len(merge.notes)} notes → vel={merge.merge_velocity}")
        for note in merge.notes:
            print(f"  - vel={note.velocity}, conf={note.confidence:.2f}")

    # Velocity analysis
    stats = merger.analyze_velocity_distribution(result.merged_notes)
    print(f"\nVelocity stats: mean={stats['mean']:.1f}, std={stats['std']:.1f}")

    return result


if __name__ == "__main__":
    quick_test_velocity_merge()