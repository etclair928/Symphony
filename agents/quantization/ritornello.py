# =================================================================
# MODULE: agents/quantization/ritornello.py
# DESCRIPTION: Non-destructive quantization for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# PHILOSOPHY:
#     "The original signal is the only source of truth; all
#      transformations must be reversible."
#
#     Non-Destructive Audit law: start_ms is NEVER overwritten.
#     snapped_start_ms shadows it. The original is always there.
#
#     If you (the human) disagree with the transcription later,
#     you can look at the MusicBox and see exactly which witness
#     led the engine astray.
#
# KEY ARCHITECTURE:
#     - Non-destructive (original timestamps preserved)
#     - Multi-stage quantization (coarse → medium → fine → micro)
#     - Groove-aware snapping (preserves intentional feel)
#     - Pulse field integration (probabilistic grid)
#     - Forensic audit trail (every snap recorded)
#     - Confidence penalties based on snap distance
#     - StatusReporter integration
#     - No agent imports - pure quantizer
#
# Authored by: DeepSeek - Complete 5.0 rewrite (2026-05-11)
# Based on 4.7 Ritornello and non-destructive quantization wisdom.
# =================================================================

import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from copy import deepcopy

# Core imports - ONLY from bedrock
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    GrooveField, PulseField, QuantizationStrategy, QuantizationStage,
    MultiStageQuantization, QuantizationPass, BeatGrid
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    RITORNELLO_MAX_SNAP_MS,
    PRESERVE_ORIGINAL_MS_ALWAYS,
    MIN_SNAP_DISTANCE_TO_CONSIDER_MS,
    SNAP_CONFIDENCE_PENALTY_PER_MS,
    MAX_CONSECUTIVE_SNAPS,
    QUANTIZATION_GRID_RESOLUTIONS_MS,
    QUANTIZATION_STAGE_CONFIDENCE_PENALTIES,
    MIN_CONFIDENCE_FOR_QUANTIZATION,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    QuantizationAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)


# ========================================================================
# Enums and Types
# ========================================================================

class SnapDecision(str, Enum):
    """Decision made about a note during quantization."""
    SNAPPED = "snapped"  # Note was moved to grid
    PRESERVED = "preserved"  # Note kept original position (intentional)
    REJECTED = "rejected"  # Note confidence too low to quantize
    GROOVE_PRESERVED = "groove_preserved"  # Moved but preserved relative timing


@dataclass(frozen=True)
class SnapRecord:
    """Forensic record of a single snap operation."""
    note_id: int
    original_start_ms: float
    original_end_ms: float
    snapped_start_ms: float
    snapped_end_ms: float
    delta_ms: float
    confidence_penalty: float
    stage: QuantizationStage
    reason: str
    strategy_used: QuantizationStrategy


@dataclass(frozen=True)
class QuantizationResult:
    """Result of quantizing a note (immutable)."""
    note: NoteEvent
    snapped: bool
    delta_ms: float
    confidence_penalty: float
    snap_record: Optional[SnapRecord] = None


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class RitornelloConfig:
    """Configuration for Ritornello quantizer."""

    # Core limits
    max_snap_ms: float = RITORNELLO_MAX_SNAP_MS  # 100ms
    min_snap_distance_ms: float = MIN_SNAP_DISTANCE_TO_CONSIDER_MS  # 5ms
    preserve_original: bool = PRESERVE_ORIGINAL_MS_ALWAYS  # True

    # Confidence
    min_confidence_for_quantization: float = MIN_CONFIDENCE_FOR_QUANTIZATION  # 0.50
    snap_penalty_per_ms: float = SNAP_CONFIDENCE_PENALTY_PER_MS  # 0.001
    max_consecutive_snaps: int = MAX_CONSECUTIVE_SNAPS  # 50

    # Multi-stage quantization
    use_multi_stage: bool = True
    grid_resolutions_ms: Dict[str, float] = field(default_factory=lambda: {
        "coarse": 500.0,  # Quarter note at 120 BPM
        "medium": 250.0,  # Eighth note
        "fine": 125.0,  # 16th note
        "micro": 62.5  # 32nd note
    })
    stage_penalties: Dict[str, float] = field(default_factory=lambda: {
        "coarse": 0.01,
        "medium": 0.02,
        "fine": 0.03,
        "micro": 0.05
    })

    # Groove awareness
    groove_preservation: bool = True
    groove_preservation_threshold: float = 0.7  # Confidence threshold for groove preservation

    # Pulse field
    use_pulse_field: bool = True
    pulse_field_weight: float = 0.6

    # Beat grid
    use_beat_grid: bool = True
    beat_grid_weight: float = 0.4

    # Performance
    cleanup_after_quantization: bool = True


# ========================================================================
# Grid Builders
# ========================================================================

class GridBuilder:
    """Builds quantization grids from various sources."""

    @staticmethod
    def from_beat_grid(beat_grid: BeatGrid, stage: QuantizationStage) -> List[float]:
        """Build grid from BeatGrid."""
        if stage == QuantizationStage.COARSE:
            # Quarter note grid
            return beat_grid.beat_times_ms
        elif stage == QuantizationStage.MEDIUM:
            # Eighth note grid
            grid = []
            for beat in beat_grid.beat_times_ms:
                grid.append(beat)
                grid.append(beat + beat_grid.beat_duration_ms / 2)
            return sorted(grid)
        elif stage == QuantizationStage.FINE:
            # 16th note grid
            grid = []
            sixteenth = beat_grid.beat_duration_ms / 4
            for beat in beat_grid.beat_times_ms:
                for i in range(4):
                    grid.append(beat + i * sixteenth)
            return sorted(grid)
        else:  # MICRO
            # 32nd note grid
            grid = []
            thirty_second = beat_grid.beat_duration_ms / 8
            for beat in beat_grid.beat_times_ms[:100]:  # Limit for memory
                for i in range(8):
                    grid.append(beat + i * thirty_second)
            return sorted(grid)

    @staticmethod
    def from_pulse_field(pulse_field: PulseField, stage: QuantizationStage) -> List[float]:
        """Build grid from PulseField."""
        if stage == QuantizationStage.COARSE:
            return pulse_field.beat_grid_ms
        elif stage in [QuantizationStage.MEDIUM, QuantizationStage.FINE]:
            return pulse_field.sixteenth_grid_ms
        else:
            # For micro, subdivide sixteenths
            grid = []
            for sixteenth in pulse_field.sixteenth_grid_ms[:400]:
                for i in range(2):
                    grid.append(sixteenth + i * 31.25)  # 32nd notes
            return sorted(grid)

    @staticmethod
    def from_tempo(tempo_bpm: float, duration_ms: float, stage: QuantizationStage) -> List[float]:
        """Build grid from simple tempo (fallback)."""
        beat_ms = 60000.0 / tempo_bpm

        if stage == QuantizationStage.COARSE:
            step = beat_ms
        elif stage == QuantizationStage.MEDIUM:
            step = beat_ms / 2
        elif stage == QuantizationStage.FINE:
            step = beat_ms / 4
        else:  # MICRO
            step = beat_ms / 8

        return list(np.arange(0, duration_ms + step, step))


# ========================================================================
# Groove-Aware Target Finder
# ========================================================================

class GrooveAwareTargetFinder:
    """
    Finds snap targets while preserving intentional groove.

    If the GrooveField indicates intentional swing, we preserve
    the relative timing between notes rather than snapping to absolute grid.
    """

    def __init__(self, config: RitornelloConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def find_target(self, note: NoteEvent, grid: List[float],
                    groove_field: Optional[GrooveField] = None,
                    previous_note: Optional[NoteEvent] = None) -> Tuple[float, bool, str]:
        """
        Find the best snap target for a note.

        Returns:
            Tuple of (target_time_ms, preserve_groove, reason)
        """
        original_time = note.get_active_start_ms()

        # If no groove field or low confidence, use absolute grid
        if not groove_field or groove_field.confidence.value < self.config.groove_preservation_threshold:
            target = self._find_nearest_grid(original_time, grid)
            return target, False, "absolute_grid"

        # If groove should be preserved, use relative positioning
        if groove_field.is_wide_swing or groove_field.is_dilla_pocket:
            if previous_note:
                # Preserve the phase delta from previous note
                prev_original = previous_note.get_active_start_ms()
                prev_snapped = previous_note.snapped_start_ms if previous_note.is_snapped() else prev_original

                original_delta = original_time - prev_original
                target = prev_snapped + original_delta

                return target, True, "groove_preserved"

        # Default to absolute grid
        target = self._find_nearest_grid(original_time, grid)
        return target, False, "absolute_grid"

    def _find_nearest_grid(self, time_ms: float, grid: List[float]) -> float:
        """Find nearest grid point."""
        if not grid:
            return time_ms

        idx = np.searchsorted(grid, time_ms)

        if idx == 0:
            return grid[0]
        elif idx >= len(grid):
            return grid[-1]
        else:
            before = grid[idx - 1]
            after = grid[idx]

            if abs(time_ms - before) <= abs(time_ms - after):
                return before
            return after


# ========================================================================
# Multi-Stage Quantizer
# ========================================================================

class MultiStageQuantizer:
    """
    Multi-stage quantizer for progressive refinement.

    Stages: Coarse → Medium → Fine → Micro
    Each stage applies progressively smaller adjustments.
    """

    def __init__(self, config: RitornelloConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def quantize_multi_stage(
            self,
            notes: List[NoteEvent],
            grid_builder: GridBuilder,
            stage_config: Dict[str, float],
            tempo_bpm: float,
            duration_ms: float,
            pulse_field: Optional[PulseField] = None,
            beat_grid: Optional[BeatGrid] = None,
            groove_field: Optional[GrooveField] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ) -> Tuple[List[NoteEvent], List[QuantizationPass]]:
        """
        Apply multi-stage quantization.

        Returns:
            Tuple of (quantized_notes, quantization_passes)
        """
        current_notes = list(notes)
        passes = []
        target_finder = GrooveAwareTargetFinder(self.config, self.status_reporter)

        stage_order = [QuantizationStage.COARSE, QuantizationStage.MEDIUM,
                       QuantizationStage.FINE, QuantizationStage.MICRO]

        for stage in stage_order:
            stage_name = stage.value
            grid_resolution = stage_config.get(f"{stage_name}_grid_ms",
                                               self.config.grid_resolutions_ms.get(stage_name, 100))
            confidence_penalty = self.config.stage_penalties.get(stage_name, 0.02)

            # Build grid for this stage
            if pulse_field and self.config.use_pulse_field:
                grid = GridBuilder.from_pulse_field(pulse_field, stage)
            elif beat_grid and self.config.use_beat_grid:
                grid = GridBuilder.from_beat_grid(beat_grid, stage)
            else:
                grid = GridBuilder.from_tempo(tempo_bpm, duration_ms, stage)

            # Apply quantization at this stage
            notes_affected = 0
            total_shift_ms = 0.0
            stage_notes = []

            for i, note in enumerate(current_notes):
                # Skip if confidence too low
                if note.confidence < self.config.min_confidence_for_quantization:
                    stage_notes.append(note)
                    continue

                previous = current_notes[i - 1] if i > 0 else None
                target, preserve_groove, reason = target_finder.find_target(
                    note, grid, groove_field, previous
                )

                delta = target - note.get_active_start_ms()

                # Only snap if shift is significant
                if abs(delta) >= self.config.min_snap_distance_ms:
                    # Check max snap limit
                    if abs(delta) <= self.config.max_snap_ms:
                        # Apply snap
                        snapped_start = target
                        snapped_end = note.get_active_end_ms() + delta

                        # Apply confidence penalty
                        delta_penalty = abs(delta) * self.config.snap_penalty_per_ms
                        total_penalty = delta_penalty + confidence_penalty
                        new_confidence = max(0.0, note.confidence - total_penalty)

                        # Create snapped note
                        snapped_note = replace(
                            note,
                            snapped_start_ms=snapped_start,
                            snapped_end_ms=snapped_end,
                            snap_reason=f"{stage_name}_{reason}",
                            snap_confidence_delta=-total_penalty,
                            quantization_strategy=QuantizationStrategy.MULTI_STAGE,
                            confidence=new_confidence
                        )
                        stage_notes.append(snapped_note)
                        notes_affected += 1
                        total_shift_ms += abs(delta)

                        # Log to MusicBox
                        if music_box:
                            music_box.log_decision(
                                stage_name=f"ritornello_{stage_name}",
                                decision_type="quantization",
                                before_state={"original_start": note.start_ms},
                                after_state={"snapped_start": snapped_start, "delta": delta},
                                reasoning=reason,
                                reversible=True
                            )
                    else:
                        # Shift too large - preserve original
                        stage_notes.append(note)
                else:
                    # Shift too small - preserve original
                    stage_notes.append(note)

            # Record this pass
            passes.append(QuantizationPass(
                stage=stage,
                grid_resolution_ms=grid_resolution,
                snap_distance_allowance_ms=self.config.max_snap_ms,
                confidence_penalty=confidence_penalty,
                events_affected=notes_affected,
                average_shift_ms=total_shift_ms / max(notes_affected, 1)
            ))

            current_notes = stage_notes

        return current_notes, passes


# ========================================================================
# Main Ritornello Quantizer
# ========================================================================

class Ritornello:
    """
    Non-destructive quantization for Grimlock 5.0.

    LAW: Original timestamps (start_ms, end_ms) are NEVER overwritten.
    Snapped timestamps go to snapped_start_ms, snapped_end_ms.

    Features:
        - Multi-stage quantization (coarse → medium → fine → micro)
        - Groove-aware snapping (preserves intentional feel)
        - Pulse field integration (probabilistic grid)
        - Forensic audit trail (every snap recorded)
        - Confidence penalties based on snap distance

    Law 2 Compliance: Does NOT import other agents.
    """

    def __init__(
            self,
            config: Optional[RitornelloConfig] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Args:
            config: Quantization configuration
            status_reporter: StatusReporter for progress
            music_box: MusicBox for forensic logging
            progress_callback: Optional progress callback
        """
        self._name = "ritornello"
        self._source_type = SourceType.RITORNELLO
        self._config = config or RitornelloConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback

        # Initialize components
        self._grid_builder = GridBuilder()
        self._target_finder = GrooveAwareTargetFinder(self._config, status_reporter)
        self._multi_stage = MultiStageQuantizer(self._config, status_reporter)

        # State
        self._last_quantization_history: Optional[MultiStageQuantization] = None
        self._last_events: List[NoteEvent] = []
        self._snap_count: int = 0
        self._last_execution_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0

        self._log_status("Ritornello initialized")

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
        Process and return StageResult with quantized events.

        Note: This agent works with NoteEvents, not raw audio.
        Events should be provided via context.
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting quantization")

        try:
            # Get events from context
            events = self._get_events_from_context(context)

            if not events:
                self._log_status("No events to quantize", "warn")
                return StageResult(
                    stage_name=self._name,
                    success=False,
                    events=[],
                    metadata={"error": "No events provided"},
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            # Get quantization parameters from context
            groove_field = getattr(context, 'groove_field', None)
            pulse_field = getattr(context, 'pulse_field', None)
            beat_grid = getattr(context, 'beat_grid', None)
            tempo_bpm = getattr(context, 'tempo_bpm', 120.0)
            duration_ms = context.duration_seconds * 1000

            # Quantize
            quantized_events = self.quantize(
                events, groove_field, pulse_field, beat_grid,
                tempo_bpm, duration_ms
            )

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory
            self._last_execution_time_ms = execution_time_ms

            # Force cleanup
            if self._config.cleanup_after_quantization:
                self._force_cleanup()

            self._update_progress(1.0, f"Complete: {self._snap_count} snaps")

            return StageResult(
                stage_name=self._name,
                success=True,
                events=quantized_events,
                metadata={
                    "original_count": len(events),
                    "quantized_count": len(quantized_events),
                    "snap_count": self._snap_count,
                    "total_confidence_loss": self._last_quantization_history.total_confidence_loss if self._last_quantization_history else 0,
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Quantization failed: {e}", "error")
            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def quantize(
            self,
            events: List[NoteEvent],
            groove_field: Optional[GrooveField] = None,
            pulse_field: Optional[PulseField] = None,
            beat_grid: Optional[BeatGrid] = None,
            tempo_bpm: float = 120.0,
            duration_ms: float = 60000.0
    ) -> List[NoteEvent]:
        """
        Quantize events non-destructively.

        Args:
            events: List of NoteEvent objects
            groove_field: Optional GrooveField for groove preservation
            pulse_field: Optional PulseField for probabilistic grid
            beat_grid: Optional BeatGrid for absolute grid
            tempo_bpm: Tempo in beats per minute (fallback)
            duration_ms: Duration in milliseconds

        Returns:
            Quantized NoteEvent objects (original preserved in snapped_* fields)
        """
        self._update_progress(0.1, "Preparing quantization")

        if not events:
            return []

        original_events = list(events)
        self._snap_count = 0

        # Filter by confidence
        filtered_events = [
            e for e in original_events
            if e.confidence >= self._config.min_confidence_for_quantization
        ]

        self._update_progress(0.2, f"Filtered to {len(filtered_events)} events")

        # Choose quantization strategy
        if self._config.use_multi_stage:
            self._update_progress(0.3, "Multi-stage quantization")
            quantized_events, passes = self._multi_stage.quantize_multi_stage(
                filtered_events, self._grid_builder,
                self._config.grid_resolutions_ms,
                tempo_bpm, duration_ms,
                pulse_field, beat_grid, groove_field,
                self._music_box
            )

            # Count snaps
            self._snap_count = sum(1 for e in quantized_events if e.is_snapped())

            # Build history
            self._last_quantization_history = MultiStageQuantization(
                original_events=original_events,
                passes=passes,
                final_events=quantized_events,
                total_confidence_loss=sum(p.confidence_penalty * p.events_affected for p in passes),
                human_readable_audit=self._build_audit_string(passes, self._snap_count)
            )

        else:
            # Single-stage quantization
            self._update_progress(0.3, "Single-stage quantization")
            quantized_events = self._single_stage_quantize(
                filtered_events, groove_field, pulse_field, beat_grid, tempo_bpm, duration_ms
            )

        self._last_events = quantized_events

        # Log to MusicBox
        if self._music_box:
            history = self._last_quantization_history
            after_state = {
                "quantized_count": len(quantized_events),
                "snap_count": self._snap_count,
                "total_confidence_loss": history.total_confidence_loss if history else 0,
            }
            if history is not None:
                # Per-grid-resolution pass breakdown (confidence_penalty,
                # events_affected per pass) - previously only ever
                # collapsed into the single total_confidence_loss number
                # above, with the human-readable audit trail discarded
                # entirely.
                after_state["passes"] = [
                    {"grid_ms": p.grid_resolution_ms, "events_affected": p.events_affected,
                     "confidence_penalty": p.confidence_penalty}
                    for p in history.passes
                ]
                after_state["audit"] = history.human_readable_audit
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="quantization",
                before_state={"original_count": len(original_events)},
                after_state=after_state,
                reasoning=f"Quantized {self._snap_count} notes",
                reversible=True
            )

        return quantized_events

    def _single_stage_quantize(
            self,
            events: List[NoteEvent],
            groove_field: Optional[GrooveField],
            pulse_field: Optional[PulseField],
            beat_grid: Optional[BeatGrid],
            tempo_bpm: float,
            duration_ms: float
    ) -> List[NoteEvent]:
        """Single-stage quantization (fallback)."""
        # Build grid
        if pulse_field and self._config.use_pulse_field:
            grid = GridBuilder.from_pulse_field(pulse_field, QuantizationStage.FINE)
        elif beat_grid and self._config.use_beat_grid:
            grid = GridBuilder.from_beat_grid(beat_grid, QuantizationStage.FINE)
        else:
            grid = GridBuilder.from_tempo(tempo_bpm, duration_ms, QuantizationStage.FINE)

        quantized = []

        for i, note in enumerate(events):
            previous = events[i - 1] if i > 0 else None
            target, preserve_groove, reason = self._target_finder.find_target(
                note, grid, groove_field, previous
            )

            delta = target - note.get_active_start_ms()

            if preserve_groove or abs(delta) < self._config.min_snap_distance_ms:
                quantized.append(note)
                continue

            if abs(delta) <= self._config.max_snap_ms:
                snapped_start = target
                snapped_end = note.get_active_end_ms() + delta
                delta_penalty = abs(delta) * self._config.snap_penalty_per_ms
                new_confidence = max(0.0, note.confidence - delta_penalty)

                snapped_note = replace(
                    note,
                    snapped_start_ms=snapped_start,
                    snapped_end_ms=snapped_end,
                    snap_reason=reason,
                    snap_confidence_delta=-delta_penalty,
                    quantization_strategy=QuantizationStrategy.RITORNELLO_SAFE,
                    confidence=new_confidence
                )
                quantized.append(snapped_note)
                self._snap_count += 1
            else:
                quantized.append(note)

        return quantized

    def get_quantization_history(self) -> Optional[MultiStageQuantization]:
        """Return complete audit trail of what was snapped and why."""
        return self._last_quantization_history

    def get_confidence_penalty(self, shift_ms: float) -> float:
        """
        Calculate confidence penalty for snapping a note.

        Larger shifts penalize more.
        """
        return abs(shift_ms) * self._config.snap_penalty_per_ms

    def _build_audit_string(self, passes: List[QuantizationPass], snap_count: int) -> str:
        """Build human-readable audit string."""
        parts = [f"Quantized {snap_count} notes in {len(passes)} stages"]
        for p in passes:
            parts.append(f"  {p.stage.value}: {p.events_affected} notes, avg shift {p.average_shift_ms:.1f}ms")
        return "\n".join(parts)

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
        if self._last_quantization_history is None:
            return ValidationResult(
                is_valid=False,
                gate_used=gate,
                reason=VetoReason.EMPTY_RESULT,
                detail="No quantization performed"
            )

        if gate == ValidationGate.CONFIDENCE_THRESHOLD:
            avg_confidence = np.mean([e.confidence for e in self._last_events]) if self._last_events else 0
            is_valid = avg_confidence >= MIN_CONFIDENCE_FOR_QUANTIZATION

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
        if not self._last_events:
            return Confidence.HALLUCINATION

        avg_confidence = np.mean([e.confidence for e in self._last_events])
        return Confidence.from_float(avg_confidence)

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        if self._snap_count > self._config.max_consecutive_snaps:
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Excessive snapping: {self._snap_count} snaps")

        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        return SchoenbergResult(
            verdict=SchoenbergVerdict.UNCERTAIN,
            zero_crossing_rate=0.0,
            spectral_flatness=0.5,
            reason="Ritornello quantizes timing, not harmonic series"
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

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "source_type": self._source_type.value,
            "last_snap_count": self._snap_count,
            "last_event_count": len(self._last_events),
            "total_confidence_loss": self._last_quantization_history.total_confidence_loss if self._last_quantization_history else 0,
            "last_execution_time_ms": self._last_execution_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb
        }


# ========================================================================
# Convenience Functions
# ========================================================================

def create_ritornello(
        max_snap_ms: float = RITORNELLO_MAX_SNAP_MS,
        min_confidence: float = MIN_CONFIDENCE_FOR_QUANTIZATION,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> Ritornello:
    """Create a configured Ritornello quantizer."""
    config = RitornelloConfig(
        max_snap_ms=max_snap_ms,
        min_confidence_for_quantization=min_confidence
    )
    return Ritornello(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )


def quick_quantize_test(notes: List[NoteEvent], tempo_bpm: float = 120.0) -> Dict[str, Any]:
    """
    Quick test function for Ritornello quantizer.

    Args:
        notes: List of NoteEvent objects
        tempo_bpm: Tempo for grid generation

    Returns:
        Dictionary with test results
    """
    print(f"Testing Ritornello on {len(notes)} notes at {tempo_bpm} BPM")

    # Create mock context
    class MockContext:
        def __init__(self, notes, tempo):
            self.events = notes
            self.tempo_bpm = tempo
            self.duration_seconds = max([n.end_ms for n in notes], default=1000) / 1000
            self.file_path = "test"
            self.working_sample_rate = 16000
            self.num_channels_original = 1
            self.original_sample_rate = 16000
            self.is_mono = True

    context = MockContext(notes, tempo_bpm)

    # Create quantizer
    quantizer = create_ritornello()

    # Quantize
    result = quantizer.run(np.array([]), context)
    stats = quantizer.get_statistics()

    original_snapped = sum(1 for n in notes if n.is_snapped())
    quantized_snapped = sum(1 for n in result.events if n.is_snapped())

    print(f"  Original notes: {len(notes)}")
    print(f"  Quantized notes: {len(result.events)}")
    print(f"  Snapped: {quantized_snapped}")
    print(f"  Confidence loss: {stats.get('total_confidence_loss', 0):.3f}")

    # Show sample
    for n in result.events[:5]:
        if n.is_snapped():
            print(
                f"    MIDI {n.pitch}: {n.start_ms:.0f}→{n.snapped_start_ms:.0f}ms (Δ={n.snapped_start_ms - n.start_ms:.1f}ms)")

    return stats


if __name__ == "__main__":
    # Create test notes
    from core.order_types import SourceType

    test_notes = [
        NoteEvent(pitch=60, start_ms=i * 500, end_ms=i * 500 + 100, velocity=80,
                  confidence=0.8, zero_crossing_rate=0.0, source=SourceType.PITCH)
        for i in range(8)
    ]

    # Add one with timing offset
    test_notes.append(NoteEvent(pitch=62, start_ms=3750, end_ms=3850, velocity=80,
                                confidence=0.8, zero_crossing_rate=0.0, source=SourceType.PITCH))

    quick_quantize_test(test_notes)