# =================================================================
# MODULE: orchestration/scribe.py
# DESCRIPTION: Epistemic Veto authority - The Scribe's Truth.
#
# VERSION: 5.6.1 (Updated for Pipeline Integration)
# UPDATED: 2026-05-13
#
# LAW OF SCRIBE'S TRUTH:
#     "Confidence must be earned, not assumed."
#
# EPISTEMIC VETO:
#     "No single witness can declare a truth; any qualified witness
#      can declare a falsehood."
#
# KEY IMPROVEMENTS:
#     - StageOutput/StageInput integration
#     - Memory-aware validation (tracks buffer IDs)
#     - Enhanced veto propagation to pipeline
#     - Retry recommendation support
#     - Degraded mode validation
#     - Evidence-based validation
#     - ScribeConfig for configuration management
#
# Authored by: DeepSeek - Updated for Pipeline Integration (2026-05-13)
# =================================================================

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Set, Union
from dataclasses import dataclass, field, replace
from datetime import datetime
from collections import defaultdict
from enum import Enum, auto

# Core imports
from core.order_types import (
    NoteEvent, SourceType, VetoReason, ValidationGate, Confidence,
    StageResult, ValidationResult, SchoenbergResult,
    AudioContext, WitnessTestimony
)
from core.constants import (
    SILENCE_RATIO_MAX,
    MIN_CONFIDENCE_TO_PASS,
    MAX_HALLUCINATION_RATIO,
    CONFIDENCE_WEIGHTS,
    VETO_HIERARCHY,
    HARMONIC_SERIES_MIN_PARTIALS,
    MAX_INHARMONICITY,
    MIN_QUALIFIED_WITNESSES_FOR_CONSENSUS
)
from core.protocols import MusicBoxProtocol, StatusReporterProtocol

# Orchestration imports
from orchestration.stage_context import StageOutput, StageStatus, StageInput


# ========================================================================
# Enums and Types
# ========================================================================

class VetoSeverity(str, Enum):
    """Severity level of a veto."""
    HARD = "hard"  # Stops pipeline, invalidates stage
    SOFT = "soft"  # Warning only, pipeline continues
    ADVISORY = "advisory"  # Informational only


class RetryRecommendation(str, Enum):
    """Recommendation for stage retry."""
    NONE = "none"
    WITH_FALLBACK = "with_fallback"
    WITH_DEGRADED = "with_degraded"
    WITH_DIFFERENT_MODEL = "with_different_model"
    AFTER_MEMORY_CLEANUP = "after_memory_cleanup"


@dataclass
class ScribeConfig:
    """Configuration for the Scribe validation system."""
    min_confidence: float = MIN_CONFIDENCE_TO_PASS
    max_silence_ratio: float = SILENCE_RATIO_MAX
    max_hallucination_ratio: float = MAX_HALLUCINATION_RATIO
    enable_schoenberg_mirror: bool = True
    enable_phase_consistency: bool = True
    enable_tempo_reasonableness: bool = True
    log_all_validations: bool = True


@dataclass(frozen=True)
class VetoRecord:
    """Immutable record of a veto event for forensic audit."""
    timestamp: datetime
    source: SourceType
    stage: str
    reason: VetoReason
    detail: str
    severity: VetoSeverity
    gate: ValidationGate
    affected_event_count: int = 0
    witness_id: Optional[str] = None
    retry_recommendation: RetryRecommendation = RetryRecommendation.NONE
    memory_pressure_at_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "stage": self.stage,
            "reason": self.reason.value,
            "detail": self.detail,
            "severity": self.severity.value,
            "gate": self.gate.value,
            "affected_event_count": self.affected_event_count,
            "retry_recommendation": self.retry_recommendation.value,
            "memory_pressure_mb": self.memory_pressure_at_time
        }


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation report for a stage."""
    is_valid: bool
    stage_name: str
    source: SourceType
    event_count: int
    avg_confidence: float
    silence_ratio: float
    hallucination_ratio: float
    gates_passed: List[ValidationGate]
    gates_failed: List[Tuple[ValidationGate, VetoReason, str]]
    vetoes_issued: List[VetoRecord]
    final_verdict: str
    retry_recommendation: RetryRecommendation = RetryRecommendation.NONE
    degraded_mode_eligible: bool = False
    memory_footprint_mb: float = 0.0


# ========================================================================
# Scribe - Epistemic Veto Authority
# ========================================================================

class Scribe:
    """
    Epistemic Veto authority - The only validation gate.

    Law of Scribe's Truth: Confidence must be earned, not assumed.
    Any qualified witness can declare a falsehood.
    """

    def __init__(
            self,
            music_box: Optional[MusicBoxProtocol] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            config: Optional[ScribeConfig] = None,
            min_confidence: float = None,
            max_silence_ratio: float = None,
            max_hallucination_ratio: float = None
    ):
        """
        Initialize Scribe with either config object or individual parameters.

        Args:
            music_box: MusicBox for forensic logging
            status_reporter: StatusReporter for progress
            config: ScribeConfig object (preferred)
            min_confidence: Minimum confidence threshold (legacy)
            max_silence_ratio: Maximum silence ratio (legacy)
            max_hallucination_ratio: Maximum hallucination ratio (legacy)
        """
        self._music_box = music_box
        self._status_reporter = status_reporter

        # Apply configuration
        if config is not None:
            self._min_confidence = config.min_confidence
            self._max_silence_ratio = config.max_silence_ratio
            self._max_hallucination_ratio = config.max_hallucination_ratio
            self._enable_schoenberg_mirror = config.enable_schoenberg_mirror
            self._enable_phase_consistency = config.enable_phase_consistency
            self._enable_tempo_reasonableness = config.enable_tempo_reasonableness
            self._log_all_validations = config.log_all_validations
        else:
            self._min_confidence = min_confidence if min_confidence is not None else MIN_CONFIDENCE_TO_PASS
            self._max_silence_ratio = max_silence_ratio if max_silence_ratio is not None else SILENCE_RATIO_MAX
            self._max_hallucination_ratio = max_hallucination_ratio if max_hallucination_ratio is not None else MAX_HALLUCINATION_RATIO
            self._enable_schoenberg_mirror = True
            self._enable_phase_consistency = True
            self._enable_tempo_reasonableness = True
            self._log_all_validations = True

        # Veto tracking
        self._hard_vetoes: List[VetoRecord] = []
        self._soft_vetoes: List[VetoRecord] = []
        self._advisories: List[VetoRecord] = []
        self._passed_stages: List[str] = []
        self._failed_stages: List[str] = []
        self._gate_history: Dict[str, List[ValidationResult]] = defaultdict(list)
        self._witness_testimonies: List[WitnessTestimony] = []
        self._current_memory_pressure_mb: float = 0.0

        self._log_status("Scribe initialized")

    # ========================================================================
    # Main Validation Interface
    # ========================================================================

    def validate_stage(
            self,
            result: Union[StageResult, StageOutput],
            source: SourceType,
            stage_name: str,
            apply_gates: Optional[List[ValidationGate]] = None,
            witness_testimony: Optional[WitnessTestimony] = None,
            memory_pressure_mb: float = 0.0
    ) -> Tuple[bool, Optional[VetoReason], ValidationReport]:
        """
        Validate a stage's output. Main entry point for pipeline.

        Args:
            result: StageResult or StageOutput containing events
            source: Which agent produced this result
            stage_name: Name of the stage (for logging)
            apply_gates: Specific gates to apply (None = all)
            witness_testimony: Optional witness testimony for consensus
            memory_pressure_mb: Current memory pressure for context

        Returns:
            Tuple of (is_valid, veto_reason_if_any, validation_report)
        """
        self._current_memory_pressure_mb = memory_pressure_mb

        # Extract events from either StageResult or StageOutput
        if hasattr(result, 'produced_events'):
            events = result.produced_events
            success = result.success
        else:
            events = getattr(result, 'events', [])
            success = getattr(result, 'success', True)

        # Store witness testimony if provided
        if witness_testimony:
            self._witness_testimonies.append(witness_testimony)

        gates_passed = []
        gates_failed = []
        vetoes_issued = []

        # Determine which gates to apply
        if apply_gates is None:
            gates_to_apply = list(ValidationGate)
        else:
            gates_to_apply = apply_gates

        # Gate 1: Empty result check
        if ValidationGate.SILENCE_DETECTOR in gates_to_apply:
            if not events and not success:
                veto = self._record_veto(
                    source, stage_name, VetoReason.EMPTY_RESULT,
                    "No events in result and stage marked as failed",
                    VetoSeverity.HARD, ValidationGate.SILENCE_DETECTOR,
                    retry_recommendation=RetryRecommendation.WITH_FALLBACK
                )
                vetoes_issued.append(veto)
                gates_failed.append(
                    (ValidationGate.SILENCE_DETECTOR, VetoReason.EMPTY_RESULT, "No events, stage failed"))

                report = self._build_report(
                    is_valid=False, stage_name=stage_name, source=source,
                    events=events, gates_passed=gates_passed, gates_failed=gates_failed,
                    vetoes_issued=vetoes_issued, memory_pressure_mb=memory_pressure_mb
                )
                return False, VetoReason.EMPTY_RESULT, report
            elif not events:
                # Empty but stage succeeded - could be legitimate for some stages
                if stage_name in ["tonal_detection", "drum_detection"]:
                    gates_passed.append(ValidationGate.SILENCE_DETECTOR)
                else:
                    veto = self._record_veto(
                        source, stage_name, VetoReason.EMPTY_RESULT,
                        "No events in result but stage reported success",
                        VetoSeverity.SOFT, ValidationGate.SILENCE_DETECTOR,
                        retry_recommendation=RetryRecommendation.WITH_DEGRADED
                    )
                    vetoes_issued.append(veto)
                    gates_failed.append((ValidationGate.SILENCE_DETECTOR, VetoReason.EMPTY_RESULT, "Empty result"))

        # Gate 2: Mandatory fields check
        if ValidationGate.CONFIDENCE_THRESHOLD in gates_to_apply and events:
            missing_fields = self._check_mandatory_fields(events)
            if missing_fields:
                veto = self._record_veto(
                    source, stage_name, VetoReason.MISSING_MANDATORY_FIELD,
                    f"Missing fields: {missing_fields}", VetoSeverity.HARD,
                    ValidationGate.CONFIDENCE_THRESHOLD
                )
                vetoes_issued.append(veto)
                gates_failed.append((ValidationGate.CONFIDENCE_THRESHOLD,
                                     VetoReason.MISSING_MANDATORY_FIELD, missing_fields))

                report = self._build_report(
                    is_valid=False, stage_name=stage_name, source=source,
                    events=events, gates_passed=gates_passed, gates_failed=gates_failed,
                    vetoes_issued=vetoes_issued, memory_pressure_mb=memory_pressure_mb
                )
                return False, VetoReason.MISSING_MANDATORY_FIELD, report
            else:
                gates_passed.append(ValidationGate.CONFIDENCE_THRESHOLD)

        # Gate 3: Silence ratio check
        if ValidationGate.SILENCE_DETECTOR in gates_to_apply and events:
            silence_ratio = self._calculate_silence_ratio(events)
            if silence_ratio > self._max_silence_ratio:
                veto = self._record_veto(
                    source, stage_name, VetoReason.EXCESS_SILENCE,
                    f"{silence_ratio:.1%} of events below confidence threshold",
                    VetoSeverity.HARD, ValidationGate.SILENCE_DETECTOR,
                    retry_recommendation=RetryRecommendation.WITH_DEGRADED
                )
                vetoes_issued.append(veto)
                gates_failed.append((ValidationGate.SILENCE_DETECTOR,
                                     VetoReason.EXCESS_SILENCE, f"{silence_ratio:.1%}"))

                report = self._build_report(
                    is_valid=False, stage_name=stage_name, source=source,
                    events=events, gates_passed=gates_passed, gates_failed=gates_failed,
                    vetoes_issued=vetoes_issued,
                    extra_metadata={"silence_ratio": silence_ratio},
                    memory_pressure_mb=memory_pressure_mb
                )
                return False, VetoReason.EXCESS_SILENCE, report
            else:
                gates_passed.append(ValidationGate.SILENCE_DETECTOR)

        # Gate 4: Confidence threshold check
        if ValidationGate.CONFIDENCE_THRESHOLD in gates_to_apply and events:
            avg_confidence = sum(e.confidence for e in events) / len(events)
            if avg_confidence < self._min_confidence:
                veto = self._record_veto(
                    source, stage_name, VetoReason.CONFIDENCE_TOO_LOW,
                    f"Average confidence: {avg_confidence:.3f} (threshold {self._min_confidence})",
                    VetoSeverity.HARD, ValidationGate.CONFIDENCE_THRESHOLD,
                    retry_recommendation=RetryRecommendation.WITH_DIFFERENT_MODEL
                )
                vetoes_issued.append(veto)
                gates_failed.append((ValidationGate.CONFIDENCE_THRESHOLD,
                                     VetoReason.CONFIDENCE_TOO_LOW, f"{avg_confidence:.3f}"))

                report = self._build_report(
                    is_valid=False, stage_name=stage_name, source=source,
                    events=events, gates_passed=gates_passed, gates_failed=gates_failed,
                    vetoes_issued=vetoes_issued,
                    extra_metadata={"avg_confidence": avg_confidence},
                    memory_pressure_mb=memory_pressure_mb
                )
                return False, VetoReason.CONFIDENCE_TOO_LOW, report
            else:
                gates_passed.append(ValidationGate.CONFIDENCE_THRESHOLD)

        # Gate 5: Schoenberg Mirror (Harmonic series check)
        # veto_ratio must exist even when this gate is skipped (disabled,
        # or excluded via apply_gates - e.g. drum/percussion validation
        # deliberately excludes this gate since percussive content has no
        # harmonic series to check) since it's referenced unconditionally
        # below when building the final metrics.
        veto_ratio = 0.0
        if self._enable_schoenberg_mirror and ValidationGate.SCHOENBERG_MIRROR in gates_to_apply and events:
            schoenberg_results = self._apply_schoenberg_mirror(events)
            veto_count = sum(1 for r in schoenberg_results if r.should_veto())
            veto_ratio = veto_count / len(events) if events else 0

            for event, sch_result in zip(events, schoenberg_results):
                if sch_result.harmonic_series:
                    event.harmonic_series_match_ratio = sch_result.harmonic_series.fundamental_confidence

            if veto_ratio > self._max_hallucination_ratio:
                veto = self._record_veto(
                    source, stage_name, VetoReason.HALLUCINATED_NOTE,
                    f"{veto_count}/{len(events)} notes failed harmonic series check ({veto_ratio:.1%})",
                    VetoSeverity.HARD, ValidationGate.SCHOENBERG_MIRROR,
                    retry_recommendation=RetryRecommendation.WITH_FALLBACK
                )
                vetoes_issued.append(veto)
                gates_failed.append((ValidationGate.SCHOENBERG_MIRROR,
                                     VetoReason.HALLUCINATED_NOTE, f"{veto_ratio:.1%}"))

                report = self._build_report(
                    is_valid=False, stage_name=stage_name, source=source,
                    events=events, gates_passed=gates_passed, gates_failed=gates_failed,
                    vetoes_issued=vetoes_issued,
                    extra_metadata={"hallucination_ratio": veto_ratio},
                    memory_pressure_mb=memory_pressure_mb
                )
                return False, VetoReason.HALLUCINATED_NOTE, report
            else:
                gates_passed.append(ValidationGate.SCHOENBERG_MIRROR)

        # Gate 6: Phase consistency check (Relational Physics)
        if self._enable_phase_consistency and ValidationGate.PHASE_CONSISTENCY in gates_to_apply and len(events) > 1:
            phase_result = self._check_phase_consistency(events)
            if not phase_result.is_valid:
                veto = self._record_veto(
                    source, stage_name, VetoReason.PHASE_INCOHERENCE,
                    phase_result.detail, VetoSeverity.SOFT,
                    ValidationGate.PHASE_CONSISTENCY
                )
                vetoes_issued.append(veto)
                gates_failed.append((ValidationGate.PHASE_CONSISTENCY,
                                     VetoReason.PHASE_INCOHERENCE, phase_result.detail))
            else:
                gates_passed.append(ValidationGate.PHASE_CONSISTENCY)

        # Gate 7: Tempo reasonableness check
        if self._enable_tempo_reasonableness and ValidationGate.TEMPO_REASONABLENESS in gates_to_apply:
            gates_passed.append(ValidationGate.TEMPO_REASONABLENESS)

        # Determine if degraded mode is eligible
        degraded_eligible = self._is_degraded_mode_eligible(stage_name, vetoes_issued)

        # Determine retry recommendation
        retry_rec = self._determine_retry_recommendation(vetoes_issued, memory_pressure_mb)

        # All hard gates passed
        hard_veto_count = len([v for v in vetoes_issued if v.severity == VetoSeverity.HARD])
        if hard_veto_count == 0:
            self._passed_stages.append(stage_name)
        else:
            self._failed_stages.append(stage_name)

        # Calculate final metrics
        avg_confidence = sum(e.confidence for e in events) / len(events) if events else 0
        silence_ratio = self._calculate_silence_ratio(events) if events else 1.0
        hallucination_ratio = veto_ratio if events else 0

        report = self._build_report(
            is_valid=hard_veto_count == 0,
            stage_name=stage_name, source=source,
            events=events, gates_passed=gates_passed, gates_failed=gates_failed,
            vetoes_issued=vetoes_issued,
            extra_metadata={
                "avg_confidence": avg_confidence,
                "silence_ratio": silence_ratio,
                "hallucination_ratio": hallucination_ratio
            },
            memory_pressure_mb=memory_pressure_mb,
            retry_recommendation=retry_rec,
            degraded_eligible=degraded_eligible
        )

        # Log to MusicBox
        if self._music_box and self._log_all_validations:
            self._music_box.log_decision(
                stage_name="scribe",
                decision_type="validation",
                before_state={"stage": stage_name},
                after_state={
                    "passed": report.is_valid,
                    "event_count": len(events),
                    "avg_confidence": avg_confidence,
                    "gates_passed": [g.value for g in gates_passed],
                    "retry_recommendation": retry_rec.value,
                    "degraded_eligible": degraded_eligible
                },
                reasoning=f"Validation {'passed' if report.is_valid else 'failed'}: {stage_name}",
                reversible=True
            )

        return report.is_valid, None if report.is_valid else (
            vetoes_issued[0].reason if vetoes_issued else None), report

    def validate_events(
            self,
            events: List[NoteEvent],
            source: SourceType,
            stage_name: str,
            apply_gates: Optional[List[ValidationGate]] = None,
            memory_pressure_mb: float = 0.0
    ) -> Tuple[bool, Optional[VetoReason], ValidationReport]:
        """Validate a list of events directly."""

        class SimpleResult:
            def __init__(self, events):
                self.events = events
                self.success = True

        result = SimpleResult(events)
        return self.validate_stage(result, source, stage_name, apply_gates, memory_pressure_mb=memory_pressure_mb)

    # ========================================================================
    # Validation Gates
    # ========================================================================

    def _check_mandatory_fields(self, events: List[NoteEvent]) -> Optional[str]:
        """Check that all required fields are present."""
        for event in events:
            if event.pitch is None or not (0 <= event.pitch <= 127):
                return "pitch"
            if event.start_ms is None or event.start_ms < 0:
                return "start_ms"
            if event.end_ms is None or event.end_ms < event.start_ms:
                return "end_ms"
            if event.confidence is None or not (0 <= event.confidence <= 1):
                return "confidence"
            if event.source is None:
                return "source"
        return None

    def _calculate_silence_ratio(self, events: List[NoteEvent]) -> float:
        """Calculate ratio of low-confidence events."""
        if not events:
            return 1.0

        low_confidence = sum(1 for e in events if e.confidence < self._min_confidence)
        return low_confidence / len(events)

    def _apply_schoenberg_mirror(self, events: List[NoteEvent]) -> List[SchoenbergResult]:
        """
        Apply Schoenberg Mirror - harmonic series auditor.

        Delegates to the real SchoenbergMirror class (agents/validation/
        schoenberg_mirror.py) instead of reimplementing the check here.
        Scribe only has NoteEvent objects at this point (no raw audio or
        FeatureBundle), so this relies on each event's own precomputed
        zero_crossing_rate and harmonic_series_match_ratio fields - the
        mirror falls back to those when no audio contract is available.
        """
        from agents.validation.schoenberg_mirror import SchoenbergMirror

        mirror = SchoenbergMirror(music_box=self._music_box)
        return mirror.audit_batch(events)

    def _check_phase_consistency(self, events: List[NoteEvent]) -> ValidationResult:
        """
        Check phase consistency WITHIN each voice's own note stream, not
        across the whole flattened list.

        A raw cross-voice check treats any two simultaneously-sounding
        notes from DIFFERENT voices as an anomaly - but a kick and snare
        struck at once, or two instruments harmonizing, is completely
        normal polyphony, not a phase defect. Verified live: the old
        check flagged 92.2% of real, correct drum hits as incoherent
        purely because a real kit has kick/snare/hihat sounding at once
        by design, and 30.7% of pitched notes for the same underlying
        reason (multiple instrument voices playing simultaneously).

        Groups notes by phrase_id (VoiceContinuity's per-note voice
        assignment - pitched notes) or by pitch (each drum type is its
        own independent stream; percussion notes have no phrase_id,
        since voice_continuity only processes pitched notes), and only
        checks gaps between CONSECUTIVE notes within the SAME group.
        Overlap or a near-zero gap between two notes in the same voice
        or drum type is still a genuine anomaly (double-detection, a
        stuck note, a mis-timed onset) and is still flagged.
        """
        if len(events) < 2:
            return ValidationResult(
                is_valid=True,
                gate_used=ValidationGate.PHASE_CONSISTENCY,
                detail="Not enough events for phase consistency check",
                events_validated=len(events),
                events_rejected=0
            )

        groups: Dict[Any, List[NoteEvent]] = defaultdict(list)
        for e in events:
            key = e.phrase_id if e.phrase_id is not None else ("pitch", e.pitch)
            groups[key].append(e)

        anomalies = 0.0
        total_checks = 0

        for group in groups.values():
            if len(group) < 2:
                continue
            sorted_group = sorted(group, key=lambda e: e.start_ms)
            for i in range(len(sorted_group) - 1):
                current = sorted_group[i]
                next_event = sorted_group[i + 1]

                gap_ms = next_event.start_ms - current.end_ms
                total_checks += 1

                if gap_ms < -10:
                    anomalies += 1
                elif 0 < gap_ms < 5:
                    anomalies += 0.5

        anomaly_ratio = anomalies / total_checks if total_checks > 0 else 0

        if anomaly_ratio > 0.3:
            return ValidationResult(
                is_valid=False,
                gate_used=ValidationGate.PHASE_CONSISTENCY,
                reason=VetoReason.PHASE_INCOHERENCE,
                detail=f"{anomaly_ratio:.1%} of within-voice note transitions have phase issues",
                events_validated=len(events),
                events_rejected=int(anomalies)
            )

        return ValidationResult(
            is_valid=True,
            gate_used=ValidationGate.PHASE_CONSISTENCY,
            detail=f"Phase consistency OK ({anomaly_ratio:.1%} anomalies)",
            events_validated=len(events),
            events_rejected=0
        )

    # ========================================================================
    # Degraded Mode & Retry Logic
    # ========================================================================

    def _is_degraded_mode_eligible(self, stage_name: str, vetoes: List[VetoRecord]) -> bool:
        """Determine if stage can run in degraded mode."""
        degraded_capable_stages = ["separation", "pitch_detection", "rhythm_detection"]

        if stage_name not in degraded_capable_stages:
            return False

        recoverable_reasons = [VetoReason.CONFIDENCE_TOO_LOW, VetoReason.EXCESS_SILENCE]

        for veto in vetoes:
            if veto.reason not in recoverable_reasons:
                return False

        return True

    def _determine_retry_recommendation(self, vetoes: List[VetoRecord],
                                        memory_pressure_mb: float) -> RetryRecommendation:
        """Determine retry recommendation based on vetoes and memory pressure."""
        if not vetoes:
            return RetryRecommendation.NONE

        if memory_pressure_mb > 1800:
            return RetryRecommendation.AFTER_MEMORY_CLEANUP

        for veto in vetoes:
            if veto.reason == VetoReason.CONFIDENCE_TOO_LOW:
                return RetryRecommendation.WITH_DIFFERENT_MODEL
            elif veto.reason == VetoReason.EXCESS_SILENCE:
                return RetryRecommendation.WITH_DEGRADED
            elif veto.reason == VetoReason.EMPTY_RESULT:
                return RetryRecommendation.WITH_FALLBACK

        return RetryRecommendation.NONE

    # ========================================================================
    # Confidence Analysis
    # ========================================================================

    def assess_confidence(self, events: List[NoteEvent], source: SourceType) -> float:
        """Assess overall confidence for a set of events."""
        if not events:
            return 0.0

        raw_confidence = sum(e.confidence for e in events) / len(events)
        source_weight = CONFIDENCE_WEIGHTS.get(source, 0.5)

        confidences = [e.confidence for e in events]
        variance = np.var(confidences) if len(confidences) > 1 else 0
        consistency_penalty = min(variance, 0.5)

        final_confidence = raw_confidence * source_weight * (1 - consistency_penalty)
        return max(0.0, min(1.0, final_confidence))

    def get_most_confident_events(self, events: List[NoteEvent], top_n: int = 10) -> List[NoteEvent]:
        """Return the N most confident events."""
        sorted_events = sorted(events, key=lambda e: e.confidence, reverse=True)
        return sorted_events[:top_n]

    def filter_low_confidence(self, events: List[NoteEvent], threshold: Optional[float] = None) -> List[NoteEvent]:
        """Filter out low-confidence events."""
        threshold = threshold or self._min_confidence
        return [e for e in events if e.confidence >= threshold]

    # ========================================================================
    # Witness Qualification
    # ========================================================================

    def is_witness_qualified(self, witness: SourceType, confidence: float) -> bool:
        """Check if a witness is qualified to vote."""
        return confidence >= self._min_confidence

    def get_veto_from_hierarchy(self, source: SourceType, target: SourceType) -> bool:
        """Check if source can veto target based on hierarchy."""
        if source not in VETO_HIERARCHY:
            return False

        allowed = VETO_HIERARCHY[source]
        return "*" in allowed or target.value in allowed

    # ========================================================================
    # Veto Management
    # ========================================================================

    def _record_veto(
            self,
            source: SourceType,
            stage: str,
            reason: VetoReason,
            detail: str,
            severity: VetoSeverity = VetoSeverity.HARD,
            gate: ValidationGate = ValidationGate.CONFIDENCE_THRESHOLD,
            affected_event_count: int = 0,
            retry_recommendation: RetryRecommendation = RetryRecommendation.NONE
    ) -> VetoRecord:
        """Record a veto for forensic audit."""
        veto = VetoRecord(
            timestamp=datetime.now(),
            source=source,
            stage=stage,
            reason=reason,
            detail=detail,
            severity=severity,
            gate=gate,
            affected_event_count=affected_event_count,
            retry_recommendation=retry_recommendation,
            memory_pressure_at_time=self._current_memory_pressure_mb
        )

        if severity == VetoSeverity.HARD:
            self._hard_vetoes.append(veto)
        elif severity == VetoSeverity.SOFT:
            self._soft_vetoes.append(veto)
        else:
            self._advisories.append(veto)

        if self._music_box:
            self._music_box.log_veto(source, reason, detail, gate)

        self._log_status(f"VETO [{severity.value}]: {source.value} - {reason.value}", "warn")

        return veto

    def get_hard_vetoes(self) -> List[VetoRecord]:
        """Return all hard vetoes."""
        return self._hard_vetoes.copy()

    def get_soft_vetoes(self) -> List[VetoRecord]:
        """Return all soft vetoes (warnings)."""
        return self._soft_vetoes.copy()

    def get_advisories(self) -> List[VetoRecord]:
        """Return all advisories."""
        return self._advisories.copy()

    def has_vetoes(self) -> bool:
        """Return True if any hard vetoes have occurred."""
        return len(self._hard_vetoes) > 0

    def has_soft_vetoes(self) -> bool:
        """Return True if any soft vetoes have occurred."""
        return len(self._soft_vetoes) > 0

    def get_last_veto(self) -> Optional[VetoRecord]:
        """Return the most recent hard veto."""
        return self._hard_vetoes[-1] if self._hard_vetoes else None

    def clear(self):
        """Clear all veto records and stage history."""
        self._hard_vetoes.clear()
        self._soft_vetoes.clear()
        self._advisories.clear()
        self._passed_stages.clear()
        self._failed_stages.clear()
        self._gate_history.clear()
        self._witness_testimonies.clear()

    # ========================================================================
    # Report Building
    # ========================================================================

    def _build_report(
            self,
            is_valid: bool,
            stage_name: str,
            source: SourceType,
            events: List[NoteEvent],
            gates_passed: List[ValidationGate],
            gates_failed: List[Tuple[ValidationGate, VetoReason, str]],
            vetoes_issued: List[VetoRecord],
            extra_metadata: Optional[Dict[str, Any]] = None,
            memory_pressure_mb: float = 0.0,
            retry_recommendation: RetryRecommendation = RetryRecommendation.NONE,
            degraded_eligible: bool = False
    ) -> ValidationReport:
        """Build a validation report."""
        avg_confidence = sum(e.confidence for e in events) / len(events) if events else 0
        silence_ratio = self._calculate_silence_ratio(events) if events else 1.0
        hallucination_ratio = extra_metadata.get("hallucination_ratio", 0.0) if extra_metadata else 0.0

        report = ValidationReport(
            is_valid=is_valid,
            stage_name=stage_name,
            source=source,
            event_count=len(events),
            avg_confidence=avg_confidence,
            silence_ratio=silence_ratio,
            hallucination_ratio=hallucination_ratio,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            vetoes_issued=vetoes_issued,
            final_verdict="PASSED" if is_valid else "FAILED",
            retry_recommendation=retry_recommendation,
            degraded_mode_eligible=degraded_eligible,
            memory_footprint_mb=memory_pressure_mb
        )

        self._gate_history[stage_name].append(
            ValidationResult(
                is_valid=is_valid,
                gate_used=ValidationGate.CONFIDENCE_THRESHOLD,
                detail=f"Validation {'passed' if is_valid else 'failed'}",
                events_validated=len(events)
            )
        )

        return report

    # ========================================================================
    # Final Verdict
    # ========================================================================

    def final_verdict(self) -> Dict[str, Any]:
        """Return the final transcription verdict."""
        return {
            "passed": len(self._hard_vetoes) == 0,
            "hard_vetoes": [v.to_dict() for v in self._hard_vetoes],
            "soft_vetoes": [v.to_dict() for v in self._soft_vetoes],
            "advisories": [v.to_dict() for v in self._advisories],
            "stages_completed": self._passed_stages,
            "stages_failed": self._failed_stages,
            "hard_veto_count": len(self._hard_vetoes),
            "soft_veto_count": len(self._soft_vetoes),
            "advisory_count": len(self._advisories),
            "is_trustworthy": len(self._hard_vetoes) == 0,
            "witness_testimonies": len(self._witness_testimonies),
            "gate_history": {
                stage: [{"passed": r.is_valid, "detail": r.detail} for r in results]
                for stage, results in self._gate_history.items()
            },
            "retry_recommendations": [v.retry_recommendation.value for v in self._hard_vetoes]
        }

    def is_trustworthy(self) -> bool:
        """Quick check if transcription can be trusted."""
        return len(self._hard_vetoes) == 0

    def get_verdict_summary(self) -> str:
        """Get human-readable verdict summary."""
        if self.is_trustworthy():
            return f"✓ TRUSTWORTHY: {len(self._passed_stages)} stages passed"
        else:
            return f"✗ UNTRUSTWORTHY: {len(self._hard_vetoes)} hard vetoes, {len(self._soft_vetoes)} warnings"

    # ========================================================================
    # Batch Validation
    # ========================================================================

    def validate_batch(
            self,
            results: List[Union[StageResult, StageOutput]],
            sources: List[SourceType],
            stage_names: List[str],
            memory_pressures: Optional[List[float]] = None
    ) -> List[Tuple[bool, Optional[VetoReason], ValidationReport]]:
        """Validate multiple stage results in batch."""
        outcomes = []
        for i, (result, source, stage_name) in enumerate(zip(results, sources, stage_names)):
            pressure = memory_pressures[i] if memory_pressures else 0.0
            is_valid, reason, report = self.validate_stage(
                result, source, stage_name, memory_pressure_mb=pressure
            )
            outcomes.append((is_valid, reason, report))
        return outcomes

    # ========================================================================
    # Soft Validation
    # ========================================================================

    def warn_if_problematic(
            self,
            events: List[NoteEvent],
            source: SourceType,
            stage_name: str,
            memory_pressure_mb: float = 0.0
    ) -> List[str]:
        """Validate but only produce warnings, not hard vetoes."""
        warnings_list = []

        if not events:
            warnings_list.append("No events in result")
            return warnings_list

        silence_ratio = self._calculate_silence_ratio(events)
        if silence_ratio > self._max_silence_ratio * 0.7:
            msg = f"High silence ratio: {silence_ratio:.1%}"
            warnings_list.append(msg)
            self._record_veto(
                source, stage_name, VetoReason.EXCESS_SILENCE,
                detail=f"Warning only: {silence_ratio:.1%} silence",
                severity=VetoSeverity.SOFT,
                gate=ValidationGate.SILENCE_DETECTOR
            )

        if events:
            avg_conf = sum(e.confidence for e in events) / len(events)
            if avg_conf < self._min_confidence * 1.2:
                msg = f"Low average confidence: {avg_conf:.3f}"
                warnings_list.append(msg)
                self._record_veto(
                    source, stage_name, VetoReason.CONFIDENCE_TOO_LOW,
                    detail=f"Warning only: avg confidence {avg_conf:.3f}",
                    severity=VetoSeverity.SOFT,
                    gate=ValidationGate.CONFIDENCE_THRESHOLD
                )

        return warnings_list

    # ========================================================================
    # Statistics & Reporting
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return validation statistics for this session."""
        total_validations = len(self._passed_stages) + len(self._hard_vetoes)

        return {
            "total_validations": total_validations,
            "passed_count": len(self._passed_stages),
            "failed_count": len(self._failed_stages),
            "hard_veto_count": len(self._hard_vetoes),
            "soft_veto_count": len(self._soft_vetoes),
            "advisory_count": len(self._advisories),
            "pass_rate": len(self._passed_stages) / total_validations if total_validations > 0 else 1.0,
            "most_common_veto": self._get_most_common_veto(),
            "stages_that_passed": self._passed_stages,
            "stages_that_failed": self._failed_stages,
            "witness_testimonies": len(self._witness_testimonies),
            "is_trustworthy": self.is_trustworthy()
        }

    def _get_most_common_veto(self) -> Optional[str]:
        """Return the most common veto reason."""
        if not self._hard_vetoes:
            return None

        veto_counts = defaultdict(int)
        for v in self._hard_vetoes:
            veto_counts[v.reason.value] += 1

        return max(veto_counts, key=veto_counts.get)

    def get_summary_report(self) -> str:
        """Get a human-readable summary report."""
        lines = []
        lines.append("=" * 60)
        lines.append("SCRIBE'S FINAL REPORT")
        lines.append("=" * 60)
        lines.append(f"Trustworthy: {self.is_trustworthy()}")
        lines.append(f"Stages passed: {len(self._passed_stages)}")
        lines.append(f"Stages failed: {len(self._failed_stages)}")
        lines.append(f"Hard vetoes: {len(self._hard_vetoes)}")
        lines.append(f"Soft vetoes: {len(self._soft_vetoes)}")
        lines.append(f"Advisories: {len(self._advisories)}")
        lines.append(f"Witness testimonies: {len(self._witness_testimonies)}")

        if self._hard_vetoes:
            lines.append("\nHard Vetoes:")
            for v in self._hard_vetoes[:5]:
                lines.append(f"  - {v.source.value}: {v.reason.value} ({v.detail[:50]}...)")

        if self._soft_vetoes:
            lines.append("\nSoft Vetoes (Warnings):")
            for v in self._soft_vetoes[:3]:
                lines.append(f"  - {v.source.value}: {v.reason.value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _log_status(self, message: str, level: str = "info"):
        """Log status message via StatusReporter."""
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info("Scribe", message)
            elif level == "warn":
                self._status_reporter.warn("Scribe", message)
            elif level == "error":
                self._status_reporter.error("Scribe", message)

    def set_memory_pressure(self, pressure_mb: float):
        """Set current memory pressure for veto context."""
        self._current_memory_pressure_mb = pressure_mb


# ========================================================================
# Convenience Functions
# ========================================================================

def create_scribe(
        music_box: Optional[MusicBoxProtocol] = None,
        status_reporter: Optional[StatusReporterProtocol] = None,
        config: Optional[ScribeConfig] = None,
        min_confidence: float = None,
        max_silence_ratio: float = None,
        max_hallucination_ratio: float = None
) -> Scribe:
    """Create a configured Scribe instance."""
    if config is not None:
        return Scribe(
            music_box=music_box,
            status_reporter=status_reporter,
            config=config
        )
    return Scribe(
        music_box=music_box,
        status_reporter=status_reporter,
        min_confidence=min_confidence,
        max_silence_ratio=max_silence_ratio,
        max_hallucination_ratio=max_hallucination_ratio
    )


def quick_scribe_test():
    """Quick test function for Scribe."""
    from core.order_types import NoteEvent, SourceType

    print("\n" + "=" * 60)
    print("Scribe - Epistemic Veto Authority Test")
    print("=" * 60)

    good_events = [
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
        )
    ]

    bad_events = [
        NoteEvent(
            pitch=60, start_ms=0, end_ms=500, velocity=80,
            confidence=0.2, zero_crossing_rate=0.5,
            source=SourceType.PITCH
        )
    ]

    scribe = create_scribe()

    print("\n1. Validating GOOD events:")
    is_valid, reason, report = scribe.validate_events(
        good_events, SourceType.PITCH, "test_stage"
    )
    print(f"   Valid: {is_valid}")
    print(f"   Event count: {report.event_count}")
    print(f"   Avg confidence: {report.avg_confidence:.2f}")
    print(f"   Retry recommendation: {report.retry_recommendation.value}")

    print("\n2. Validating BAD events:")
    is_valid, reason, report = scribe.validate_events(
        bad_events, SourceType.PITCH, "test_stage"
    )
    print(f"   Valid: {is_valid}")
    if reason:
        print(f"   Veto reason: {reason.value}")
    print(f"   Retry recommendation: {report.retry_recommendation.value}")

    print("\n3. Final Verdict:")
    verdict = scribe.final_verdict()
    print(f"   Trustworthy: {verdict['is_trustworthy']}")
    print(f"   Hard vetoes: {verdict['hard_veto_count']}")

    print("\n" + "=" * 60)
    print("Scribe test complete.")
    print("=" * 60)


if __name__ == "__main__":
    quick_scribe_test()