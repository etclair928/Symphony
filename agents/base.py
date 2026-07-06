# =================================================================
# MODULE: agents/base.py
# DESCRIPTION: Abstract base classes for all Grimlock 5.0 agents.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# PHILOSOPHY:
#     These base classes provide common functionality so individual
#     agents don't have to reimplement logging, memory management,
#     error handling, and protocol compliance.
#
#     LAW 2: Base classes do NOT import other agents. They only import
#     from core modules and provide interfaces.
#
# KEY FEATURES:
#     - Common logging and error handling
#     - Performance tracking decorators
#     - Memory management mixin
#     - StatusReporter integration
#     - ScribeValidatable base implementation
#     - Fallback implementations for graceful degradation
#
# Authored by: DeepSeek - Complete 5.0 rewrite (2026-05-11)
# Based on 4.7 agent patterns and 5.0 architectural requirements.
# =================================================================

import time
import functools
import warnings
import gc
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# Core imports - ONLY from bedrock
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult, VetoReason,
    StemType, SeparationResult, GrooveField, TempoMap, PulseField,
    Confidence, ValidationGate, ValidationResult, SchoenbergResult,
    SchoenbergVerdict, WitnessTestimony, WitnessVote
)
from core.constants import (
    STAGE_TIMEOUT_DETECTION_SECONDS,
    STAGGERED_GC_TRIGGER_MB,
    TARGET_SAMPLE_RATE
)
from core.protocols import (
    AgentProtocol, SeparationAgentProtocol, DetectionAgentProtocol,
    AnalysisAgentProtocol, QuantizationAgentProtocol, ValidationAgentProtocol,
    MemoryManagedProtocol, ScribeValidatable, MusicBoxProtocol,
    StatusReporterProtocol
)


# ========================================================================
# Decorators for Common Agent Functionality
# ========================================================================

def timed(logger: Optional[MusicBoxProtocol] = None):
    """
    Decorator to time agent execution.

    Records execution time to MusicBox for forensic audit.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            try:
                result = func(self, *args, **kwargs)
                elapsed_ms = (time.time() - start) * 1000

                # Log timing if MusicBox available
                music_box = getattr(self, 'music_box', None) or logger
                if music_box and hasattr(music_box, 'log_decision'):
                    music_box.log_decision(
                        stage_name=getattr(self, '_name', func.__name__),
                        decision_type="timing",
                        before_state={},
                        after_state={"duration_ms": elapsed_ms},
                        reasoning=f"Execution took {elapsed_ms:.1f}ms",
                        reversible=False
                    )
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                music_box = getattr(self, 'music_box', None) or logger
                if music_box and hasattr(music_box, 'log_decision'):
                    music_box.log_decision(
                        stage_name=getattr(self, '_name', func.__name__),
                        decision_type="error",
                        before_state={},
                        after_state={"error": str(e), "duration_ms": elapsed_ms},
                        reasoning=f"Failed after {elapsed_ms:.1f}ms",
                        reversible=False
                    )
                raise

        return wrapper

    return decorator


def handle_errors(default_return=None, log_errors: bool = True):
    """
    Decorator to handle errors gracefully.

    Prevents agent crashes from bringing down the pipeline.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                if log_errors:
                    music_box = getattr(self, 'music_box', None)
                    if music_box and hasattr(music_box, 'log_decision'):
                        music_box.log_decision(
                            stage_name=getattr(self, '_name', func.__name__),
                            decision_type="error_recovery",
                            before_state={},
                            after_state={"error": str(e)},
                            reasoning=f"Recovered from error: {e}",
                            reversible=False
                        )
                    warnings.warn(f"{getattr(self, '_name', func.__name__)}.{func.__name__} failed: {e}")

                if default_return is not None:
                    return default_return
                raise

        return wrapper

    return decorator


# ========================================================================
# Base Agent Class
# ========================================================================

class BaseAgent(ABC):
    """
    Base class for all Grimlock agents.

    Provides common functionality:
    - Configuration management
    - MusicBox logging
    - StatusReporter integration
    - Error handling
    - Performance tracking
    """

    def __init__(
            self,
            name: str,
            source_type: SourceType,
            config: Optional[Dict[str, Any]] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Args:
            name: Agent name for logging
            source_type: SourceType enum value
            config: Agent-specific configuration
            status_reporter: StatusReporter for progress updates
            music_box: MusicBox for forensic logging
            progress_callback: Optional progress callback
        """
        self._name = name
        self._source_type = source_type
        self._config = config or {}
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback

        # Performance tracking
        self._execution_times: List[float] = []
        self._call_count = 0
        self._error_count = 0
        self._total_processing_time_ms = 0.0

        # Memory tracking
        self._memory_peak_mb = 0.0

        self._log_status(f"{self.__class__.__name__} initialized")

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @timed()
    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """
        Run the agent on audio buffer.

        This is the main entry point required by AgentProtocol.
        """
        self._call_count += 1
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        try:
            result = self._run_impl(audio_buffer, context)
            elapsed_ms = (time.time() - start_time) * 1000
            memory_delta = self._get_current_memory_mb() - start_memory

            self._execution_times.append(elapsed_ms)
            self._total_processing_time_ms += elapsed_ms
            self._memory_peak_mb = max(self._memory_peak_mb, self._get_current_memory_mb())

            # Update result metadata
            result.execution_time_ms = elapsed_ms
            result.memory_delta_mb = memory_delta

            return result

        except Exception as e:
            self._error_count += 1
            self._log_status(f"Run failed: {e}", "error")

            if self._music_box:
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="error",
                    before_state={},
                    after_state={"error": str(e)},
                    reasoning=f"Agent failed with error: {e}",
                    reversible=False
                )

            return StageResult(
                stage_name=self._name,
                success=False,
                veto_reason=VetoReason.EMPTY_RESULT,
                metadata={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                memory_delta_mb=self._get_current_memory_mb() - start_memory
            )

    @abstractmethod
    def _run_impl(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """Implementation of agent logic. Override in subclasses."""
        pass

    def _update_progress(self, progress: float, message: str):
        """Update progress via callback and StatusReporter."""
        if self._progress_callback:
            self._progress_callback(progress, message)
        if self._status_reporter:
            self._status_reporter.progress(self._name, progress, message)

    def _log_status(self, message: str, level: str = "info"):
        """Log status message via StatusReporter."""
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info(self._name, message)
            elif level == "warn":
                self._status_reporter.warn(self._name, message)
            elif level == "error":
                self._status_reporter.error(self._name, message)

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Return agent performance statistics."""
        return {
            "name": self._name,
            "source_type": self._source_type.value,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "avg_execution_time_ms": np.mean(self._execution_times) if self._execution_times else 0,
            "total_execution_time_ms": self._total_processing_time_ms,
            "memory_peak_mb": self._memory_peak_mb
        }


# ========================================================================
# Base Detection Agent
# ========================================================================

class BaseDetectionAgent(BaseAgent, DetectionAgentProtocol):
    """
    Base class for detection agents (Rhythm, Pitch, Tonal, Drum).

    Handles common detection tasks:
    - Onset/note detection
    - Confidence calculation
    - Post-processing (filtering, merging)
    - ScribeValidatable implementation
    """

    def __init__(
            self,
            name: str,
            source_type: SourceType,
            config: Optional[Dict[str, Any]] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        super().__init__(name, source_type, config, status_reporter, music_box, progress_callback)

        self._detection_threshold = config.get("detection_threshold", Confidence.LOW.value)
        self._min_note_duration_ms = config.get("min_note_duration_ms", 20)
        self._max_notes_per_second = config.get("max_notes_per_second", 30)

        # State for ScribeValidatable
        self._last_events: List[NoteEvent] = []
        self._last_raw_confidence: float = 0.0

    def detect(self, stem_buffer: np.ndarray, context: AudioContext) -> List[NoteEvent]:
        """Run detection on a stem."""
        events = self._detect_impl(stem_buffer, context)

        # Post-process events
        events = self._filter_by_confidence(events)
        events = self._filter_by_duration(events)
        events = self._limit_density(events, context.duration_seconds)

        self._last_events = events
        self._last_raw_confidence = np.mean([e.confidence for e in events]) if events else 0.0

        # Log detection results
        if self._music_box:
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="detection",
                before_state={"audio_duration": context.duration_seconds},
                after_state={
                    "event_count": len(events),
                    "avg_confidence": self._last_raw_confidence
                },
                reasoning=f"Detected {len(events)} events with confidence {self._last_raw_confidence:.2f}",
                reversible=True
            )

        return events

    @abstractmethod
    def _detect_impl(self, stem_buffer: np.ndarray, context: AudioContext) -> List[NoteEvent]:
        """Implementation of detection logic. Override in subclasses."""
        pass

    def _filter_by_confidence(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Remove low-confidence events."""
        return [e for e in events if e.confidence >= self._detection_threshold]

    def _filter_by_duration(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """Remove very short notes (likely noise)."""
        return [e for e in events if (e.end_ms - e.start_ms) >= self._min_note_duration_ms]

    def _limit_density(self, events: List[NoteEvent], duration_seconds: float) -> List[NoteEvent]:
        """Limit note density if too high."""
        if duration_seconds <= 0:
            return events

        max_allowed = int(duration_seconds * self._max_notes_per_second)
        if len(events) <= max_allowed:
            return events

        # Keep highest confidence events
        events.sort(key=lambda e: e.confidence, reverse=True)
        return events[:max_allowed]

    # ========================================================================
    # DetectionAgentProtocol Properties
    # ========================================================================

    @property
    def detection_threshold(self) -> Confidence:
        return Confidence.from_float(self._detection_threshold)

    def get_raw_confidence(self) -> float:
        return self._last_raw_confidence

    # ========================================================================
    # ScribeValidatable Implementation
    # ========================================================================

    def validate(self, gate: ValidationGate) -> ValidationResult:
        """Run validation gate on this agent's output."""
        if gate == ValidationGate.CONFIDENCE_THRESHOLD:
            is_valid = self._last_raw_confidence >= self._detection_threshold

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                detail=f"Detection confidence: {self._last_raw_confidence:.2f}",
                events_validated=len(self._last_events),
                events_rejected=0,
                confidence_before=self._last_raw_confidence,
                confidence_after=self._last_raw_confidence if is_valid else 0.0
            )

        elif gate == ValidationGate.SILENCE_DETECTOR:
            is_valid = len(self._last_events) > 0

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.EXCESS_SILENCE,
                detail=f"Event count: {len(self._last_events)}",
                events_validated=len(self._last_events),
                events_rejected=0
            )

        else:
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                detail=f"Gate {gate.value} not fully supported",
                events_validated=len(self._last_events)
            )

    def get_confidence(self) -> Confidence:
        return Confidence.from_float(self._last_raw_confidence)

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        if len(self._last_events) == 0:
            return (VetoReason.EMPTY_RESULT, "No events detected")

        if self._last_raw_confidence < self._detection_threshold:
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Confidence {self._last_raw_confidence:.2f} below threshold")

        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        """Default implementation - override in subclasses."""
        return SchoenbergResult(
            verdict=SchoenbergVerdict.UNCERTAIN,
            zero_crossing_rate=0.0,
            spectral_flatness=0.5,
            reason="Base detection agent - override apply_schoenberg_mirror"
        )

    def _run_impl(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """Implement AgentProtocol.run"""
        events = self.detect(audio_buffer, context)
        return StageResult(
            stage_name=self._name,
            success=len(events) > 0,
            events=events,
            metadata={
                "event_count": len(events),
                "avg_confidence": self._last_raw_confidence
            }
        )


# ========================================================================
# Base Analysis Agent
# ========================================================================

class BaseAnalysisAgent(BaseAgent, AnalysisAgentProtocol):
    """
    Base class for analysis agents (Groove Field, Voice, Tempo, Pulse Field).

    These agents don't produce NoteEvents directly; they produce
    analysis structures like GrooveField, TempoMap, or PulseField.
    """

    def __init__(
            self,
            name: str,
            source_type: SourceType,
            config: Optional[Dict[str, Any]] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        super().__init__(name, source_type, config, status_reporter, music_box, progress_callback)

        # State for analysis results
        self._last_analysis_result: Optional[Any] = None

    @property
    def analysis_threshold(self) -> Confidence:
        """Minimum confidence to consider analysis valid."""
        return Confidence.MEDIUM

    def get_last_result(self) -> Optional[Any]:
        """Get the last analysis result."""
        return self._last_analysis_result

    def _run_impl(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """Analysis agents typically don't use the simple run method."""
        raise NotImplementedError("Use specific analysis method instead")


# ========================================================================
# Base Quantization Agent
# ========================================================================

class BaseQuantizationAgent(BaseAgent, QuantizationAgentProtocol):
    """
    Base class for quantization agents (Ritornello, Velocity Merge).

    Handles non-destructive quantization with audit trails.
    """

    def __init__(
            self,
            name: str,
            source_type: SourceType,
            config: Optional[Dict[str, Any]] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        super().__init__(name, source_type, config, status_reporter, music_box, progress_callback)

        self._max_snap_ms = config.get("max_snap_ms", 100)
        self._min_confidence = config.get("min_confidence", 0.25)

        # State
        self._last_original_events: List[NoteEvent] = []
        self._last_quantized_events: List[NoteEvent] = []

    @property
    def quantizer_name(self) -> str:
        return self._name

    def get_quantization_audit(self) -> Dict[str, Any]:
        """Get audit trail of quantization decisions."""
        return {
            "agent": self._name,
            "original_count": len(self._last_original_events),
            "quantized_count": len(self._last_quantized_events),
            "max_snap_ms": self._max_snap_ms,
            "min_confidence": self._min_confidence
        }


# ========================================================================
# Base Validation Agent
# ========================================================================

class BaseValidationAgent(BaseAgent, ValidationAgentProtocol):
    """
    Base class for validation agents (Consensus Engine, Scribe).

    These agents validate outputs from other agents and may issue vetoes.
    """

    def __init__(
            self,
            name: str,
            source_type: SourceType,
            config: Optional[Dict[str, Any]] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        super().__init__(name, source_type, config, status_reporter, music_box, progress_callback)

        self._veto_threshold = config.get("veto_threshold", 0.25)
        self._last_validation_result: Optional[ValidationResult] = None

    def validate_stage(self, result: StageResult, source: SourceType) -> ValidationResult:
        """Validate a stage's output."""
        self._last_validation_result = self._validate_impl(result, source)
        return self._last_validation_result

    @abstractmethod
    def _validate_impl(self, result: StageResult, source: SourceType) -> ValidationResult:
        """Implementation of validation logic."""
        pass

    def get_last_validation(self) -> Optional[ValidationResult]:
        """Get the last validation result."""
        return self._last_validation_result

    def _run_impl(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """Validation agents typically don't use run."""
        raise NotImplementedError("Use validate_stage() method instead")


# ========================================================================
# Memory Management Mixin
# ========================================================================

class MemoryManagedMixin(MemoryManagedProtocol):
    """
    Mixin for agents that manage their own memory.

    Implements MemoryManagedProtocol for staggered deletion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._managed_buffers: Dict[str, Tuple[Any, int]] = {}
        self._total_memory_mb = 0.0
        self._gc_count = 0
        self._total_freed_mb = 0.0

    def register_buffer(self, name: str, buffer: Any, size_bytes: Optional[int] = None):
        """Register a buffer for memory tracking."""
        if size_bytes is None and hasattr(buffer, 'nbytes'):
            size_bytes = buffer.nbytes
        elif size_bytes is None:
            size_bytes = 0

        self._managed_buffers[name] = (buffer, size_bytes)
        self._total_memory_mb += size_bytes / (1024 * 1024)

    def release_buffer(self, buffer_name: str) -> None:
        """Release a specific buffer."""
        if buffer_name in self._managed_buffers:
            buffer, size_bytes = self._managed_buffers.pop(buffer_name)
            del buffer
            mb_freed = size_bytes / (1024 * 1024)
            self._total_memory_mb -= mb_freed
            self._total_freed_mb += mb_freed

    def get_memory_footprint_mb(self) -> float:
        """Return current memory usage in MB."""
        return self._total_memory_mb

    def can_release(self, buffer_name: str) -> bool:
        """Check if buffer can be safely released."""
        return buffer_name in self._managed_buffers

    def release_all_buffers(self):
        """Release all managed buffers."""
        for name in list(self._managed_buffers.keys()):
            self.release_buffer(name)

    def staggered_gc(self) -> Dict[str, Any]:
        """Run staggered garbage collection and return stats."""
        before = self.get_memory_footprint_mb()
        gc.collect()
        after = self.get_memory_footprint_mb()

        self._gc_count += 1
        freed = before - after
        self._total_freed_mb += freed

        return {
            "gc_count": self._gc_count,
            "before_mb": before,
            "after_mb": after,
            "freed_mb": freed,
            "total_freed_mb": self._total_freed_mb,
            "managed_buffers": len(self._managed_buffers)
        }


# ========================================================================
# Fallback Implementations for Graceful Degradation
# ========================================================================

class FallbackDetector(BaseDetectionAgent):
    """
    Fallback detector when real detection models unavailable.
    Returns empty list (no notes detected).
    """

    def __init__(
            self,
            detector_type: str,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        super().__init__(
            name=f"fallback_{detector_type}",
            source_type=SourceType.PITCH if detector_type == "pitch" else SourceType.RHYTHM,
            config={"detection_threshold": 0.0},
            status_reporter=status_reporter,
            music_box=music_box
        )
        self.detector_type = detector_type

    def _detect_impl(self, stem_buffer: np.ndarray, context: AudioContext) -> List[NoteEvent]:
        self._log_status(f"Using fallback detector for {self.detector_type} - returning empty results", "warn")
        return []

    def get_confidence(self) -> Confidence:
        return Confidence.HALLUCINATION


class FallbackAnalyzer(BaseAnalysisAgent):
    """Fallback analyzer when real analyzers unavailable."""

    def __init__(
            self,
            analyzer_type: str,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        super().__init__(
            name=f"fallback_{analyzer_type}",
            source_type=SourceType.GROOVE_FIELD,
            status_reporter=status_reporter,
            music_box=music_box
        )
        self.analyzer_type = analyzer_type

    def _run_impl(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        self._log_status(f"Using fallback analyzer for {self.analyzer_type}", "warn")
        return StageResult(
            stage_name=self._name,
            success=True,
            events=[],
            metadata={"fallback": True, "analyzer_type": self.analyzer_type}
        )


class FallbackQuantizer(BaseQuantizationAgent):
    """Fallback quantizer that does nothing (identity)."""

    def __init__(
            self,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        super().__init__(
            name="fallback_quantizer",
            source_type=SourceType.RITORNELLO,
            config={"max_snap_ms": 0, "min_confidence": 0.0},
            status_reporter=status_reporter,
            music_box=music_box
        )

    def quantize(self, events: List[NoteEvent], **kwargs) -> List[NoteEvent]:
        """Identity quantization - no changes."""
        return events

    def get_quantization_audit(self) -> Dict[str, Any]:
        audit = super().get_quantization_audit()
        audit["fallback"] = True
        return audit

    def _run_impl(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        self._log_status("Using fallback quantizer - identity transform", "warn")
        return StageResult(
            stage_name=self._name,
            success=True,
            events=[],
            metadata={"fallback": True}
        )


def create_fallback_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    Factory function to create appropriate fallback agent.

    Args:
        agent_type: One of 'detection', 'analysis', 'quantization'
        **kwargs: Additional arguments for the agent

    Returns:
        Fallback agent instance
    """
    if agent_type == "detection":
        detector_type = kwargs.get("detector_type", "unknown")
        return FallbackDetector(detector_type, **kwargs)
    elif agent_type == "analysis":
        analyzer_type = kwargs.get("analyzer_type", "unknown")
        return FallbackAnalyzer(analyzer_type, **kwargs)
    elif agent_type == "quantization":
        return FallbackQuantizer(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")