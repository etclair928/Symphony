# =================================================================
# MODULE: agents/validation/consensus_engine.py
# DESCRIPTION: Consensus Engine - Multi-witness epistemic veto system.
#
# VERSION: 5.6.1 (Unified with testimony.py and epistemic_council.py)
# UPDATED: 2026-05-26
#
# PHILOSOPHY:
#     "No single witness can declare a truth; any qualified witness
#      can declare a falsehood."
#
#     This is the EPISTEMIC VETO - the cornerstone of Grimlock's
#     truth-seeking architecture. The engine would rather miss a
#     ghost note than hallucinate a false one.
#
# UNIFIED ARCHITECTURE:
#     - TestimonyCollector gathers PitchTestimony from witnesses
#     - WitnessQualifier filters by confidence and blacklist
#     - ConsensusVoter applies strategies (ANY_VETO_WINS is default)
#     - Returns ConsensusPackage with final decision
#     - Works seamlessly with EpistemicCouncil output
#
# INTEGRATION WITH EPISTEMIC_COUNCIL:
#     council = EpistemicCouncil()
#     notes, findings = await council.analyze(audio, sr)
#     testimony = PitchTestimony(notes=notes, confidence=0.7, ...)
#     consensus = engine.reach_consensus(testimonies=[testimony])
# =================================================================

import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field, replace
from datetime import datetime
from collections import defaultdict
from enum import Enum
from uuid import uuid4

# Core imports
from core.order_types import (
    NoteEvent, SourceType, VetoReason, Confidence, WitnessVote,
    WitnessTestimony, ConsensusPackage, ConsensusStrategy, ConsensusConfig,
    StageResult, ValidationGate, ValidationResult, SchoenbergResult,
    SchoenbergVerdict, AudioContext
)
from core.testimony import (
    PitchTestimony, TempoTestimony, GrooveTestimony,
    HarmonicValidationTestimony, StageResult as TestimonyStageResult
)
from core.constants import (
    CONFIDENCE_WEIGHTS,
    VETO_HIERARCHY,
    MIN_CONFIDENCE_TO_PASS,
    MIN_QUALIFIED_WITNESSES_FOR_CONSENSUS,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    ValidationAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)


# ========================================================================
# Enums and Types
# ========================================================================

class WitnessStatus(str, Enum):
    """Status of a witness in the consensus process."""
    QUALIFIED = "qualified"
    DISQUALIFIED = "disqualified"
    BLACKLISTED = "blacklisted"
    PENDING = "pending"


class ConsensusOutcome(str, Enum):
    """Outcome of a consensus round."""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    VETOED = "vetoed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ConsensusRound:
    """Immutable record of a single consensus round for forensic audit."""
    round_id: str
    timestamp: datetime
    target_note_id: Optional[str]
    witnesses_consulted: Tuple[SourceType, ...]
    testimonies: Tuple[Any, ...]  # Can be PitchTestimony, WitnessTestimony, etc.
    vetoes_cast: Tuple[Tuple[SourceType, VetoReason, str], ...]
    final_decision: bool
    final_events: Tuple[NoteEvent, ...]
    strategy_used: ConsensusStrategy
    confidence: float
    outcome: ConsensusOutcome


@dataclass(frozen=True)
class WitnessRegistration:
    """Immutable record of a witness registration."""
    witness_id: SourceType
    agent_name: str
    registered_at: datetime
    initial_confidence: float
    testimony_type: str  # "pitch", "tempo", "groove", "harmonic"


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class ConsensusEngineConfig:
    """Configuration for Consensus Engine."""

    # Strategy
    strategy: ConsensusStrategy = ConsensusStrategy.ANY_VETO_WINS
    min_qualified_witnesses: int = MIN_QUALIFIED_WITNESSES_FOR_CONSENSUS
    veto_threshold: float = Confidence.LOW.value  # 0.25

    # Confidence
    require_confidence_consensus: bool = True
    min_consensus_confidence: float = 0.5

    # Witness management
    auto_blacklist_on_veto: bool = True
    max_blacklist_per_witness: int = 3

    # Testimony conversion
    use_testimony_normalizer: bool = True

    # Performance
    max_rounds_stored: int = 1000
    cleanup_after_consensus: bool = True
    parallel_testimony_collection: bool = False


# ========================================================================
# Testimony Adapters (Convert between testimony types)
# ========================================================================

class TestimonyAdapter:
    """Converts between different testimony types for consensus."""

    @staticmethod
    def to_witness_testimony(
            testimony: Any,
            source: Optional[SourceType] = None,
            confidence: Optional[float] = None
    ) -> WitnessTestimony:
        """
        Convert any testimony object to WitnessTestimony for consensus.

        Handles:
            - PitchTestimony
            - TempoTestimony
            - GrooveTestimony
            - HarmonicValidationTestimony
            - Raw NoteEvent list
            - WitnessTestimony (passthrough)
        """
        # Already WitnessTestimony
        if isinstance(testimony, WitnessTestimony):
            return testimony

        # PitchTestimony
        if hasattr(testimony, 'notes'):
            events = testimony.notes
            conf = confidence if confidence is not None else getattr(testimony, 'confidence', 0.5)
            src = source if source is not None else SourceType.PITCH

            return WitnessTestimony(
                witness_id=src,
                testimony_time_ms=datetime.now().timestamp() * 1000,
                events=events,
                confidence=conf,
                metadata={"testimony_type": "pitch"}
            )

        # List of NoteEvents
        if isinstance(testimony, list) and all(isinstance(n, NoteEvent) for n in testimony):
            events = testimony
            conf = confidence if confidence is not None else 0.5
            src = source if source is not None else SourceType.PITCH

            return WitnessTestimony(
                witness_id=src,
                testimony_time_ms=datetime.now().timestamp() * 1000,
                events=events,
                confidence=conf,
                metadata={"testimony_type": "raw_notes"}
            )

        # Single NoteEvent
        if isinstance(testimony, NoteEvent):
            events = [testimony]
            conf = confidence if confidence is not None else testimony.confidence
            src = source if source is not None else SourceType.PITCH

            return WitnessTestimony(
                witness_id=src,
                testimony_time_ms=datetime.now().timestamp() * 1000,
                events=events,
                confidence=conf,
                metadata={"testimony_type": "single_note"}
            )

        # Fallback: empty testimony
        return WitnessTestimony(
            witness_id=source or SourceType.SCRIBE,
            testimony_time_ms=datetime.now().timestamp() * 1000,
            events=[],
            confidence=confidence or 0.0,
            metadata={"testimony_type": "unknown", "error": str(type(testimony))}
        )

    @staticmethod
    def from_witness_testimony(
            testimony: WitnessTestimony,
            target_type: str = "pitch"
    ) -> Any:
        """
        Convert WitnessTestimony back to specific testimony type.

        Args:
            testimony: WitnessTestimony to convert
            target_type: "pitch", "tempo", "groove", "notes"

        Returns:
            Appropriate testimony object
        """
        if target_type == "pitch":
            from core.testimony import PitchTestimony
            return PitchTestimony(
                notes=testimony.events,
                confidence=testimony.confidence,
                model_used="consensus",
                stem_type="consensus",
                frame_count=len(testimony.events),
                rejected_frames=0
            )
        elif target_type == "notes":
            return testimony.events
        else:
            return testimony


# ========================================================================
# Witness Qualifier
# ========================================================================

class WitnessQualifier:
    """
    Determines which witnesses are qualified to vote.

    From 4.7 wisdom: A witness is qualified if:
    - Confidence >= LOW threshold
    - Not blacklisted for this context
    - Has valid testimony
    """

    def __init__(self, config: ConsensusEngineConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._blacklist: Dict[SourceType, List[Tuple[str, datetime]]] = defaultdict(list)
        self._blacklist_count: Dict[SourceType, int] = defaultdict(int)

    def is_qualified(self, witness: SourceType, confidence: float) -> bool:
        """Check if a witness is qualified to vote."""
        # Check confidence threshold
        if confidence < self.config.veto_threshold:
            return False

        # Check blacklist
        if witness in self._blacklist and self._blacklist_count[witness] >= self.config.max_blacklist_per_witness:
            return False

        return True

    def blacklist_witness(self, witness: SourceType, reason: str):
        """Add a witness to blacklist for the remainder of session."""
        self._blacklist[witness].append((reason, datetime.now()))
        self._blacklist_count[witness] += 1

        if self.status_reporter:
            self.status_reporter.warn("WitnessQualifier",
                                      f"Blacklisted {witness.value}: {reason}")

    def clear_blacklist(self):
        """Clear all blacklists."""
        self._blacklist.clear()
        self._blacklist_count.clear()

    def get_blacklisted(self) -> Dict[SourceType, List[Tuple[str, str]]]:
        """Get all blacklisted witnesses with reasons and times."""
        return {
            k: [(reason, dt.isoformat()) for reason, dt in v]
            for k, v in self._blacklist.items()
        }

    def get_status(self, witness: SourceType, confidence: float) -> WitnessStatus:
        """Get detailed status of a witness."""
        if witness in self._blacklist and self._blacklist_count[witness] >= self.config.max_blacklist_per_witness:
            return WitnessStatus.BLACKLISTED
        elif confidence < self.config.veto_threshold:
            return WitnessStatus.DISQUALIFIED
        else:
            return WitnessStatus.QUALIFIED


# ========================================================================
# Testimony Collector (Unified)
# ========================================================================

class TestimonyCollector:
    """
    Collects testimonies from multiple witnesses (agents or direct inputs).

    Now works with both agent instances and direct testimony objects
    (like those from EpistemicCouncil).
    """

    def __init__(self, config: ConsensusEngineConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._adapter = TestimonyAdapter()

    def collect_testimonies(
            self,
            testimonies: List[Any],
            audio_context: Optional[AudioContext] = None
    ) -> List[WitnessTestimony]:
        """
        Collect and normalize testimonies from various sources.

        Args:
            testimonies: List of testimony objects (PitchTestimony, WitnessTestimony, etc.)
            audio_context: Optional audio context for agents

        Returns:
            List of normalized WitnessTestimony objects
        """
        normalized = []

        for testimony in testimonies:
            try:
                # Already WitnessTestimony
                if isinstance(testimony, WitnessTestimony):
                    normalized.append(testimony)
                else:
                    # Convert using adapter
                    witness = self._adapter.to_witness_testimony(testimony)
                    normalized.append(witness)
            except Exception as e:
                self._log_error(SourceType.SCRIBE, e)
                # Add empty testimony for failed conversion
                normalized.append(WitnessTestimony(
                    witness_id=SourceType.SCRIBE,
                    testimony_time_ms=datetime.now().timestamp() * 1000,
                    events=[],
                    confidence=0.0,
                    metadata={"error": str(e)}
                ))

        return normalized

    def collect_from_agents(
            self,
            target_note: Optional[NoteEvent],
            agents: Dict[SourceType, Any],
            audio_context: Optional[AudioContext] = None
    ) -> List[WitnessTestimony]:
        """
        Collect testimonies from agent instances.

        Args:
            target_note: The note being evaluated (or None for general consensus)
            agents: Dictionary of agent instances by SourceType
            audio_context: Optional audio context for agents

        Returns:
            List of WitnessTestimony objects
        """
        testimonies = []

        for source, agent in agents.items():
            try:
                testimony = self._get_testimony_from_agent(agent, target_note, audio_context)
                if testimony:
                    testimonies.append(testimony)
                else:
                    testimonies.append(WitnessTestimony(
                        witness_id=source,
                        testimony_time_ms=datetime.now().timestamp() * 1000,
                        events=[],
                        confidence=0.0,
                        metadata={"reason": "abstained"}
                    ))
            except Exception as e:
                self._log_error(source, e)
                testimonies.append(WitnessTestimony(
                    witness_id=source,
                    testimony_time_ms=datetime.now().timestamp() * 1000,
                    events=[],
                    confidence=0.0,
                    metadata={"error": str(e)}
                ))

        return testimonies

    def _get_testimony_from_agent(
            self,
            agent: Any,
            target_note: Optional[NoteEvent],
            audio_context: Optional[AudioContext]
    ) -> Optional[WitnessTestimony]:
        """Get testimony from a single agent."""
        events = []
        confidence = 0.0
        metadata = {}

        # Try different methods based on agent type
        if hasattr(agent, 'get_events'):
            events = agent.get_events()
            confidence = agent.get_confidence() if hasattr(agent, 'get_confidence') else 0.5
            metadata["method"] = "get_events"

        elif hasattr(agent, 'validate_note') and target_note:
            is_valid, confidence, reason = agent.validate_note(target_note)
            if is_valid:
                events = [target_note]
            metadata = {"validation_reason": reason}

        elif hasattr(agent, 'get_vote_on_note') and target_note:
            vote = agent.get_vote_on_note(target_note)
            if vote.note_present:
                events = [target_note]
            confidence = vote.confidence
            metadata = {"vote": vote.reasoning}

        elif hasattr(agent, 'run') and audio_context:
            result = agent.run(np.array([]), audio_context)
            if result.events:
                events = result.events
            confidence = result.metadata.get('confidence', 0.5)
            metadata["method"] = "run"

        else:
            if hasattr(agent, 'confidence'):
                confidence = agent.confidence
            metadata["method"] = "fallback"

        source = agent.source_type if hasattr(agent, 'source_type') else SourceType.SCRIBE

        return WitnessTestimony(
            witness_id=source,
            testimony_time_ms=datetime.now().timestamp() * 1000,
            events=events,
            confidence=confidence,
            metadata=metadata
        )

    def _log_error(self, source: SourceType, error: Exception):
        """Log error during testimony collection."""
        if self.status_reporter:
            self.status_reporter.error("TestimonyCollector",
                                       f"Failed to collect from {source.value}: {error}")


# ========================================================================
# Consensus Voter (Core Logic)
# ========================================================================

class ConsensusVoter:
    """
    Applies consensus strategies to reach a decision.

    Strategies:
    - UNANIMOUS_REQUIRED: All witnesses must agree
    - MAJORITY_VOTE: Simple majority wins
    - WEIGHTED_BY_CONFIDENCE: Weighted by witness confidence
    - ANY_VETO_WINS: Epistemic veto - any veto = false (DEFAULT)
    """

    def __init__(self, config: ConsensusEngineConfig,
                 status_reporter: Optional[StatusReporterProtocol] = None,
                 music_box: Optional[MusicBoxProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self.music_box = music_box
        self._rounds: List[ConsensusRound] = []
        self._round_counter = 0

    def reach_consensus(
            self,
            testimonies: List[WitnessTestimony],
            target_note: Optional[NoteEvent] = None
    ) -> ConsensusPackage:
        """
        Reach consensus from collected testimonies.

        Args:
            testimonies: List of witness testimonies
            target_note: The note being evaluated (if any)

        Returns:
            ConsensusPackage with final decision
        """
        # Filter qualified witnesses
        qualified = self._filter_qualified(testimonies)

        # Check minimum qualified witnesses
        if len(qualified) < self.config.min_qualified_witnesses:
            return self._create_inconclusive_package(testimonies, qualified,
                                                     "Insufficient qualified witnesses")

        # Apply veto rule first (epistemic veto overrides everything)
        veto_triggered, veto_source, veto_reason, veto_detail = self._check_vetoes(qualified)

        if veto_triggered and self.config.strategy == ConsensusStrategy.ANY_VETO_WINS:
            return self._create_veto_package(
                testimonies, qualified, veto_source, veto_reason, veto_detail
            )

        # Otherwise, apply selected strategy
        if self.config.strategy == ConsensusStrategy.UNANIMOUS_REQUIRED:
            decision, confidence, final_events, outcome = self._unanimous_consensus(qualified, target_note)
        elif self.config.strategy == ConsensusStrategy.MAJORITY_VOTE:
            decision, confidence, final_events, outcome = self._majority_vote(qualified, target_note)
        elif self.config.strategy == ConsensusStrategy.WEIGHTED_BY_CONFIDENCE:
            decision, confidence, final_events, outcome = self._weighted_consensus(qualified, target_note)
        else:
            decision, confidence, final_events, outcome = self._any_veto_wins(qualified, target_note)

        # Record consensus round
        self._record_round(testimonies, qualified, decision, final_events,
                           confidence, outcome, veto_source if veto_triggered else None)
        self._log_round_to_music_box(len(testimonies), len(qualified), decision, confidence, outcome)

        return ConsensusPackage(
            witness_votes=self._to_witness_votes(qualified),
            final_decision=decision,
            final_pitch=final_events[0].pitch if final_events else None,
            final_start_ms=final_events[0].start_ms if final_events else None,
            final_end_ms=final_events[0].end_ms if final_events else None,
            consensus_confidence=confidence,
            veto_triggered_by=veto_source if veto_triggered else None,
            veto_reason=veto_reason if veto_triggered else None,
            timestamp_utc=datetime.now()
        )

    def _filter_qualified(self, testimonies: List[WitnessTestimony]) -> List[WitnessTestimony]:
        """Filter to only qualified witnesses."""
        return [t for t in testimonies if t.confidence >= self.config.veto_threshold]

    def _check_vetoes(
            self,
            testimonies: List[WitnessTestimony]
    ) -> Tuple[bool, Optional[SourceType], Optional[VetoReason], str]:
        """Check if any witness issues a veto."""
        for testimony in testimonies:
            veto = testimony.get_veto_vote()
            if veto is not None:
                return True, testimony.witness_id, veto[0], veto[1]
        return False, None, None, ""

    def _unanimous_consensus(
            self,
            testimonies: List[WitnessTestimony],
            target_note: Optional[NoteEvent]
    ) -> Tuple[bool, float, List[NoteEvent], ConsensusOutcome]:
        """Unanimous agreement required."""
        if not testimonies:
            return False, 0.0, [], ConsensusOutcome.INCONCLUSIVE

        first_decision = len(testimonies[0].events) > 0

        for testimony in testimonies:
            has_events = len(testimony.events) > 0
            if has_events != first_decision:
                return False, 0.0, [], ConsensusOutcome.REJECTED

        all_events = []
        for testimony in testimonies:
            all_events.extend(testimony.events)

        confidence = min(t.confidence for t in testimonies)
        outcome = ConsensusOutcome.ACCEPTED if first_decision else ConsensusOutcome.REJECTED

        return first_decision, confidence, all_events, outcome

    def _majority_vote(
            self,
            testimonies: List[WitnessTestimony],
            target_note: Optional[NoteEvent]
    ) -> Tuple[bool, float, List[NoteEvent], ConsensusOutcome]:
        """Simple majority vote."""
        if not testimonies:
            return False, 0.0, [], ConsensusOutcome.INCONCLUSIVE

        votes_for = sum(1 for t in testimonies if len(t.events) > 0)
        votes_against = len(testimonies) - votes_for

        decision = votes_for > votes_against

        margin = abs(votes_for - votes_against) / len(testimonies)
        confidence = 0.5 + margin * 0.5
        confidence = max(0.0, min(1.0, confidence))

        all_events = []
        for testimony in testimonies:
            if len(testimony.events) > 0:
                all_events.extend(testimony.events)

        outcome = ConsensusOutcome.ACCEPTED if decision else ConsensusOutcome.REJECTED

        return decision, confidence, all_events, outcome

    def _weighted_consensus(
            self,
            testimonies: List[WitnessTestimony],
            target_note: Optional[NoteEvent]
    ) -> Tuple[bool, float, List[NoteEvent], ConsensusOutcome]:
        """Weighted by witness confidence."""
        if not testimonies:
            return False, 0.0, [], ConsensusOutcome.INCONCLUSIVE

        weighted_for = 0.0
        weighted_against = 0.0
        total_weight = 0.0

        for testimony in testimonies:
            weight = testimony.confidence * CONFIDENCE_WEIGHTS.get(testimony.witness_id, 0.5)
            total_weight += weight

            if len(testimony.events) > 0:
                weighted_for += weight
            else:
                weighted_against += weight

        decision = weighted_for > weighted_against

        if total_weight > 0:
            confidence = max(weighted_for, weighted_against) / total_weight
        else:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        all_events = []
        for testimony in testimonies:
            if len(testimony.events) > 0:
                all_events.extend(testimony.events)

        outcome = ConsensusOutcome.ACCEPTED if decision else ConsensusOutcome.REJECTED

        return decision, confidence, all_events, outcome

    def _any_veto_wins(
            self,
            testimonies: List[WitnessTestimony],
            target_note: Optional[NoteEvent]
    ) -> Tuple[bool, float, List[NoteEvent], ConsensusOutcome]:
        """Epistemic veto: any qualified witness can declare falsehood."""
        for testimony in testimonies:
            veto = testimony.get_veto_vote()
            if veto is not None:
                return False, 0.0, [], ConsensusOutcome.VETOED

        all_events = []
        for testimony in testimonies:
            all_events.extend(testimony.events)

        if testimonies:
            confidence = np.mean([t.confidence for t in testimonies])
        else:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        outcome = ConsensusOutcome.ACCEPTED if all_events else ConsensusOutcome.REJECTED

        return True, confidence, all_events, outcome

    def _to_witness_votes(self, testimonies: List[WitnessTestimony]) -> List[WitnessVote]:
        """Convert testimonies to WitnessVote objects."""
        votes = []
        for testimony in testimonies:
            has_events = len(testimony.events) > 0

            best_event = None
            if testimony.events:
                best_event = max(testimony.events, key=lambda e: e.confidence)

            votes.append(WitnessVote(
                source=testimony.witness_id,
                note_present=has_events,
                pitch_midi=best_event.pitch if best_event else None,
                start_ms=best_event.start_ms if best_event else None,
                end_ms=best_event.end_ms if best_event else None,
                confidence=testimony.confidence,
                zero_crossing_rate=best_event.zero_crossing_rate if best_event else 0.0,
                reasoning=str(testimony.metadata)
            ))

        return votes

    def _create_veto_package(
            self,
            all_testimonies: List[WitnessTestimony],
            qualified: List[WitnessTestimony],
            veto_source: SourceType,
            veto_reason: VetoReason,
            veto_detail: str
    ) -> ConsensusPackage:
        """Create a consensus package for veto case."""
        self._record_round(all_testimonies, qualified, False, [], 0.0,
                           ConsensusOutcome.VETOED, veto_source)
        # Vetoes are the entire point of this class ("any qualified
        # witness can declare a falsehood") yet this path never logged
        # to music_box at all before - only the non-veto success path
        # below did. Confirmed by direct test: an empty-witness veto
        # produced zero music_box entries despite reach_consensus()
        # returning a real VETOED ConsensusPackage.
        self._log_round_to_music_box(len(all_testimonies), len(qualified), False, 0.0,
                                     ConsensusOutcome.VETOED)

        return ConsensusPackage(
            witness_votes=self._to_witness_votes(qualified),
            final_decision=False,
            final_pitch=None,
            final_start_ms=None,
            final_end_ms=None,
            consensus_confidence=0.0,
            veto_triggered_by=veto_source,
            veto_reason=veto_reason,
            timestamp_utc=datetime.now()
        )

    def _create_inconclusive_package(
            self,
            all_testimonies: List[WitnessTestimony],
            qualified: List[WitnessTestimony],
            reason: str
    ) -> ConsensusPackage:
        """Create a consensus package for inconclusive case."""
        self._record_round(all_testimonies, qualified, False, [], 0.0,
                           ConsensusOutcome.INCONCLUSIVE, None)
        self._log_round_to_music_box(len(all_testimonies), len(qualified), False, 0.0,
                                     ConsensusOutcome.INCONCLUSIVE)

        return ConsensusPackage(
            witness_votes=self._to_witness_votes(qualified),
            final_decision=False,
            final_pitch=None,
            final_start_ms=None,
            final_end_ms=None,
            consensus_confidence=0.0,
            veto_triggered_by=None,
            veto_reason=None,
            timestamp_utc=datetime.now()
        )

    def _record_round(
            self,
            testimonies: List[WitnessTestimony],
            qualified: List[WitnessTestimony],
            decision: bool,
            final_events: List[NoteEvent],
            confidence: float,
            outcome: ConsensusOutcome,
            veto_source: Optional[SourceType] = None
    ):
        """Record consensus round for forensic audit."""
        self._round_counter += 1

        vetoes = []
        for t in testimonies:
            veto = t.get_veto_vote()
            if veto:
                vetoes.append((t.witness_id, veto[0], veto[1]))

        round_record = ConsensusRound(
            round_id=f"round_{self._round_counter}_{uuid4().hex[:8]}",
            timestamp=datetime.now(),
            target_note_id=str(id(final_events[0])) if final_events else None,
            witnesses_consulted=tuple(qualified_t.witness_id for qualified_t in qualified),
            testimonies=tuple(testimonies),
            vetoes_cast=tuple(vetoes),
            final_decision=decision,
            final_events=tuple(final_events),
            strategy_used=self.config.strategy,
            confidence=confidence,
            outcome=outcome
        )

        self._rounds.append(round_record)

        if len(self._rounds) > self.config.max_rounds_stored:
            self._rounds = self._rounds[-self.config.max_rounds_stored:]

    def _log_round_to_music_box(
            self,
            testimony_count: int,
            qualified_count: int,
            decision: bool,
            confidence: float,
            outcome: ConsensusOutcome
    ) -> None:
        """
        Log the round _record_round() just appended to self._rounds.

        Shared by all three reach_consensus() exit paths (success, veto,
        inconclusive) so every outcome reaches the forensic log, not just
        the success path.
        """
        if not self.music_box:
            return

        last_round = self._rounds[-1] if self._rounds else None
        after_state = {
            "decision": decision,
            "confidence": confidence,
            "outcome": outcome.value,
            "qualified_witnesses": qualified_count,
        }
        if last_round is not None:
            after_state["witnesses_consulted"] = [w.value for w in last_round.witnesses_consulted]
            after_state["vetoes_cast"] = [
                {"witness": w.value, "reason": r.value, "detail": d}
                for w, r, d in last_round.vetoes_cast
            ]
        self.music_box.log_decision(
            stage_name="consensus_engine",
            decision_type="consensus",
            before_state={"testimony_count": testimony_count},
            after_state=after_state,
            reasoning=f"Consensus reached: {decision} with confidence {confidence:.2f}"
                      if outcome not in (ConsensusOutcome.VETOED, ConsensusOutcome.INCONCLUSIVE)
                      else f"Consensus {outcome.value}",
            reversible=True
        )

    def get_rounds(self) -> List[ConsensusRound]:
        """Get all consensus rounds for forensic audit."""
        return self._rounds.copy()

    def clear_rounds(self):
        """Clear all recorded rounds."""
        self._rounds.clear()
        self._round_counter = 0


# ========================================================================
# Main Consensus Engine Agent
# ========================================================================

class ConsensusEngine:
    """
    Consensus Engine - Multi-witness epistemic veto system.

    Law of Epistemic Veto: "No single witness can declare a truth;
    any qualified witness can declare a falsehood."

    This engine acts as the final authority for resolving conflicts
    between multiple agents (witnesses). It implements the epistemic
    veto rule: if any qualified witness declares a falsehood, the
    consensus is false.

    UNIFIED with EpistemicCouncil:
        The Council produces PitchTestimony, which this engine consumes.
        No manual list merging needed - the engine handles weighting.
    """

    def __init__(
            self,
            config: Optional[ConsensusEngineConfig] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        self._name = "consensus_engine"
        self._source_type = SourceType.CONSENSUS_ENGINE
        self._config = config or ConsensusEngineConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback

        self._qualifier = WitnessQualifier(self._config, status_reporter)
        self._collector = TestimonyCollector(self._config, status_reporter)
        self._voter = ConsensusVoter(self._config, status_reporter, music_box)
        self._adapter = TestimonyAdapter()

        self._last_consensus: Optional[ConsensusPackage] = None
        self._registered_witnesses: Dict[SourceType, Any] = {}
        self._witness_registrations: Dict[SourceType, WitnessRegistration] = {}
        self._last_execution_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0

        self._log_status("ConsensusEngine initialized")

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
        """Process and return StageResult with consensus verdict."""
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting consensus process")

        try:
            target_note = getattr(context, 'target_note', None)

            consensus = self.reach_consensus(target_note=target_note, audio_context=context)

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory
            self._last_execution_time_ms = execution_time_ms

            if self._config.cleanup_after_consensus:
                self._force_cleanup()

            self._update_progress(1.0, f"Consensus: {consensus.final_decision}")

            return StageResult(
                stage_name=self._name,
                success=consensus.final_decision,
                events=list(consensus.witness_votes),
                metadata={
                    "final_decision": consensus.final_decision,
                    "confidence": consensus.consensus_confidence,
                    "veto_triggered": consensus.veto_triggered_by is not None,
                    "veto_source": consensus.veto_triggered_by.value if consensus.veto_triggered_by else None,
                    "veto_reason": consensus.veto_reason.value if consensus.veto_reason else None,
                    "witness_count": len(consensus.witness_votes),
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Consensus failed: {e}", "error")
            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    # ========================================================================
    # Core Consensus Interface (Unified)
    # ========================================================================

    def register_witness(self, source: SourceType, agent: Any, testimony_type: str = "pitch"):
        """Register a witness (agent) for consensus."""
        self._registered_witnesses[source] = agent
        self._witness_registrations[source] = WitnessRegistration(
            witness_id=source,
            agent_name=agent.__class__.__name__,
            registered_at=datetime.now(),
            initial_confidence=self._get_witness_confidence(agent),
            testimony_type=testimony_type
        )
        self._log_status(f"Registered witness: {source.value}")

    def unregister_witness(self, source: SourceType):
        """Unregister a witness."""
        if source in self._registered_witnesses:
            del self._registered_witnesses[source]
            if source in self._witness_registrations:
                del self._witness_registrations[source]
            self._log_status(f"Unregistered witness: {source.value}")

    def reach_consensus(
            self,
            testimonies: Optional[List[Any]] = None,
            target_note: Optional[NoteEvent] = None,
            audio_context: Optional[AudioContext] = None,
            witnesses: Optional[List[SourceType]] = None
    ) -> ConsensusPackage:
        """
        Reach consensus from testimonies or registered witnesses.

        UNIFIED ENTRY POINT:
            - Direct testimonies: pass list of PitchTestimony objects
            - Agent witnesses: use registered witnesses
            - Mixed: both work

        Args:
            testimonies: Direct testimony objects (PitchTestimony, etc.)
            target_note: The note being evaluated (or None for general consensus)
            audio_context: Optional audio context for witnesses
            witnesses: Optional subset of registered witnesses to consult

        Returns:
            ConsensusPackage with final decision
        """
        self._update_progress(0.1, "Preparing consensus process")

        # Collect testimonies
        if testimonies:
            # Direct testimonies provided
            collected = self._collector.collect_testimonies(testimonies, audio_context)
        elif self._registered_witnesses:
            # Use registered agents
            witness_sources = witnesses or list(self._registered_witnesses.keys())
            selected_witnesses = {
                s: self._registered_witnesses[s]
                for s in witness_sources
                if s in self._registered_witnesses
            }
            collected = self._collector.collect_from_agents(target_note, selected_witnesses, audio_context)
        else:
            # No sources
            self._log_status("No testimonies or witnesses available", "warn")
            return self._create_empty_consensus()

        # Reach consensus
        self._update_progress(0.7, "Reaching consensus")
        consensus = self._voter.reach_consensus(collected, target_note)

        self._last_consensus = consensus

        # Auto-blacklist witnesses that issued invalid vetoes
        if self._config.auto_blacklist_on_veto and consensus.veto_triggered_by:
            self._qualifier.blacklist_witness(
                consensus.veto_triggered_by,
                f"Issued veto: {consensus.veto_reason.value if consensus.veto_reason else 'unknown'}"
            )

        self._log_status(f"Consensus: {consensus.final_decision} (confidence {consensus.consensus_confidence:.2f})")

        return consensus

    def reach_consensus_on_pitch_testimonies(
            self,
            testimonies: List[PitchTestimony],
            target_note: Optional[NoteEvent] = None
    ) -> ConsensusPackage:
        """
        Convenience method for reaching consensus on PitchTestimony objects.

        This is the primary way to use ConsensusEngine with EpistemicCouncil output.

        Args:
            testimonies: List of PitchTestimony from various witnesses
            target_note: Optional target note for validation

        Returns:
            ConsensusPackage with final decision
        """
        return self.reach_consensus(testimonies=testimonies, target_note=target_note)

    def reach_consensus_on_note_events(
            self,
            note_lists: List[List[NoteEvent]],
            sources: List[SourceType],
            confidences: Optional[List[float]] = None
    ) -> ConsensusPackage:
        """
        Convenience method for reaching consensus on raw note lists.

        Args:
            note_lists: List of note lists from different witnesses
            sources: Corresponding source types
            confidences: Optional per-witness confidence scores

        Returns:
            ConsensusPackage with final decision
        """
        testimonies = []
        for i, notes in enumerate(note_lists):
            conf = confidences[i] if confidences and i < len(confidences) else 0.5
            testimony = PitchTestimony(
                notes=notes,
                confidence=conf,
                model_used="unknown",
                stem_type="unknown",
                frame_count=len(notes),
                rejected_frames=0
            )
            testimonies.append(testimony)

        return self.reach_consensus_on_pitch_testimonies(testimonies)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _create_empty_consensus(self) -> ConsensusPackage:
        """Create empty consensus package when no sources available."""
        return ConsensusPackage(
            witness_votes=[],
            final_decision=False,
            final_pitch=None,
            final_start_ms=None,
            final_end_ms=None,
            consensus_confidence=0.0,
            veto_triggered_by=None,
            veto_reason=None,
            timestamp_utc=datetime.now()
        )

    def _get_witness_confidence(self, agent: Any) -> float:
        """Get confidence of a witness agent."""
        if hasattr(agent, 'get_confidence'):
            return agent.get_confidence()
        elif hasattr(agent, 'confidence'):
            return agent.confidence if isinstance(agent.confidence, (int, float)) else 0.5
        else:
            return 0.5

    def get_qualified_witnesses(self) -> List[SourceType]:
        """Get list of qualified witnesses based on their confidence."""
        qualified = []
        for source, agent in self._registered_witnesses.items():
            confidence = self._get_witness_confidence(agent)
            if self._qualifier.is_qualified(source, confidence):
                qualified.append(source)
        return qualified

    def get_witness_status(self) -> Dict[SourceType, Dict[str, Any]]:
        """Get status of all registered witnesses."""
        status = {}
        for source, agent in self._registered_witnesses.items():
            confidence = self._get_witness_confidence(agent)
            status[source] = {
                "registered": True,
                "confidence": confidence,
                "status": self._qualifier.get_status(source, confidence).value,
                "agent_type": agent.__class__.__name__,
                "registration": self._witness_registrations[source] if source in self._witness_registrations else None
            }
        return status

    def blacklist_witness(self, witness: SourceType, reason: str):
        """Blacklist a witness for the remainder of session."""
        self._qualifier.blacklist_witness(witness, reason)

    def get_last_consensus(self) -> Optional[ConsensusPackage]:
        """Get the last consensus result."""
        return self._last_consensus

    def get_consensus_history(self) -> List[ConsensusRound]:
        """Get all consensus rounds for forensic audit."""
        return self._voter.get_rounds()

    def clear_history(self):
        """Clear consensus history and blacklists."""
        self._voter.clear_rounds()
        self._qualifier.clear_blacklist()
        self._last_consensus = None

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
        if self._last_consensus is None:
            return ValidationResult(
                is_valid=False,
                gate_used=gate,
                reason=VetoReason.EMPTY_RESULT,
                detail="No consensus reached"
            )

        if gate == ValidationGate.CONFIDENCE_THRESHOLD:
            is_valid = self._last_consensus.consensus_confidence >= MIN_CONFIDENCE_TO_PASS
            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                detail=f"Consensus confidence: {self._last_consensus.consensus_confidence:.2f}",
                confidence_before=self._last_consensus.consensus_confidence,
                confidence_after=self._last_consensus.consensus_confidence if is_valid else 0.0
            )

        return ValidationResult(
            is_valid=True,
            gate_used=gate,
            detail=f"Gate {gate.value} not fully supported"
        )

    def get_confidence(self) -> Confidence:
        if self._last_consensus is None:
            return Confidence.HALLUCINATION
        return Confidence.from_float(self._last_consensus.consensus_confidence)

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        if self._last_consensus is None:
            return (VetoReason.EMPTY_RESULT, "No consensus")

        if self._last_consensus.veto_triggered_by:
            return (self._last_consensus.veto_reason or VetoReason.CONFIDENCE_TOO_LOW,
                    f"Veto by {self._last_consensus.veto_triggered_by.value}")

        if self._last_consensus.consensus_confidence < MIN_CONFIDENCE_TO_PASS:
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Confidence {self._last_consensus.consensus_confidence:.2f}")

        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        return SchoenbergResult(
            verdict=SchoenbergVerdict.UNCERTAIN,
            zero_crossing_rate=0.0,
            spectral_flatness=0.5,
            reason="ConsensusEngine evaluates witnesses, not harmonic series"
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
    # Statistics
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus engine statistics."""
        rounds = self._voter.get_rounds()
        return {
            "name": self._name,
            "source_type": self._source_type.value,
            "strategy": self._config.strategy.value,
            "registered_witnesses": len(self._registered_witnesses),
            "qualified_witnesses": len(self.get_qualified_witnesses()),
            "consensus_rounds": len(rounds),
            "veto_count": sum(1 for r in rounds if r.outcome == ConsensusOutcome.VETOED),
            "acceptance_count": sum(1 for r in rounds if r.outcome == ConsensusOutcome.ACCEPTED),
            "rejection_count": sum(1 for r in rounds if r.outcome == ConsensusOutcome.REJECTED),
            "blacklisted_witnesses": self._qualifier.get_blacklisted(),
            "last_consensus": {
                "decision": self._last_consensus.final_decision if self._last_consensus else None,
                "confidence": self._last_consensus.consensus_confidence if self._last_consensus else None,
                "veto_triggered": self._last_consensus.veto_triggered_by is not None if self._last_consensus else None
            } if self._last_consensus else None,
            "last_execution_time_ms": self._last_execution_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb
        }


# ========================================================================
# Convenience Functions
# ========================================================================

def create_consensus_engine(
        strategy: ConsensusStrategy = ConsensusStrategy.ANY_VETO_WINS,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> ConsensusEngine:
    """Create a configured ConsensusEngine instance."""
    config = ConsensusEngineConfig(strategy=strategy)
    return ConsensusEngine(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )


def quick_consensus_test():
    """Quick test for Consensus Engine."""
    from core.testimony import PitchTestimony
    from core.order_types import NoteEvent, SourceType

    print("=" * 60)
    print("Consensus Engine Test (Unified)")
    print("=" * 60)

    # Create mock note events
    notes_1 = [
        NoteEvent(pitch=60, start_ms=1000, end_ms=1500, velocity=80, confidence=0.9, zero_crossing_rate=0.1,
                  source=SourceType.PITCH),
        NoteEvent(pitch=64, start_ms=2000, end_ms=2500, velocity=75, confidence=0.8, zero_crossing_rate=0.1,
                  source=SourceType.PITCH),
    ]
    notes_2 = [
        NoteEvent(pitch=60, start_ms=1000, end_ms=1500, velocity=78, confidence=0.85, zero_crossing_rate=0.1,
                  source=SourceType.PITCH),
        NoteEvent(pitch=64, start_ms=2000, end_ms=2500, velocity=80, confidence=0.82, zero_crossing_rate=0.1,
                  source=SourceType.PITCH),
    ]
    notes_3 = []  # Empty testimony (veto)

    # Create testimonies
    testimony_1 = PitchTestimony(notes=notes_1, confidence=0.9, model_used="basic_pitch", stem_type="other",
                                 frame_count=2, rejected_frames=0)
    testimony_2 = PitchTestimony(notes=notes_2, confidence=0.85, model_used="spice", stem_type="other", frame_count=2,
                                 rejected_frames=0)
    testimony_3 = PitchTestimony(notes=notes_3, confidence=0.0, model_used="crepe", stem_type="other", frame_count=0,
                                 rejected_frames=0)

    # Create engine
    engine = create_consensus_engine()

    print("\nTest 1: Two agreeing witnesses")
    consensus = engine.reach_consensus_on_pitch_testimonies([testimony_1, testimony_2])
    print(f"  Decision: {consensus.final_decision}")
    print(f"  Confidence: {consensus.consensus_confidence:.2f}")
    print(f"  Veto: {consensus.veto_triggered_by is not None}")

    print("\nTest 2: One witness with veto")
    consensus = engine.reach_consensus_on_pitch_testimonies([testimony_1, testimony_3])
    print(f"  Decision: {consensus.final_decision}")
    print(f"  Confidence: {consensus.consensus_confidence:.2f}")
    if consensus.veto_triggered_by:
        print(f"  Veto by: {consensus.veto_triggered_by.value}")

    print("\nTest 3: Single witness")
    consensus = engine.reach_consensus_on_pitch_testimonies([testimony_1])
    print(f"  Decision: {consensus.final_decision}")
    print(f"  Confidence: {consensus.consensus_confidence:.2f}")

    stats = engine.get_statistics()
    print(f"\nStatistics:")
    print(f"  Strategy: {stats['strategy']}")
    print(f"  Consensus rounds: {stats['consensus_rounds']}")

    return engine


if __name__ == "__main__":
    quick_consensus_test()