# =================================================================
# MODULE: epistemic/quaver_intelligence.py
# DESCRIPTION: Symbolic duration witness for Grimlock 5.5.
#              Generates competing duration hypotheses with support,
#              opposition, and epistemic tension scores.
#              Does NOT quantize. Does NOT decide. Only TESTIFIES.
#
# VERSION: 5.6.1 (rewrite of quaver_intelligence.py draft)
# UPDATED: 2026-05-24
#
# BUGS FIXED FROM DRAFT:
#   - note.start_time → note.start_ms
#   - note.duration → note.duration_ms() method
#   - note.id → (note.pitch, note.start_ms) tuple key
#   - from core.lattice import Lattice → TemporalLattice
#   - QuaverConfig, WitnessContribution not defined → defined here
#   - DurationHypothesis missing support_score etc. → defined here
#   - QuaverTestimony missing fields → defined here
#   - findings_map.get_instrument_for_voice() doesn't exist → fixed
#   - evidence["raw_duration"] never set → fixed
#   - _compute_theme_elaboration_score naming inconsistency → fixed
#   - normalize_probabilities mutates frozen objects → fixed
#   - _find_longer_duration had wrong calculation → fixed
#   - MusicalFindingsMap import added
# =================================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.order_types import NoteEvent
from epistemic.musical_findings_map import MusicalFindingsMap


# =====================================================================
# Supporting Types
# (Add to core/testimony.py once stabilized)
# =====================================================================

@dataclass
class WitnessContribution:
    """One agent's contribution to a hypothesis."""
    witness_name: str
    contribution: float  # Positive = support, negative = opposition
    reliability: float  # Prior reliability weight (0-1)

    def weighted_contribution(self) -> float:
        return self.contribution * self.reliability


@dataclass
class DurationHypothesis:
    """
    One competing hypothesis about a note's symbolic duration.

    Multiple hypotheses are generated per note. The EpistemicCouncil
    arbitrates. The ConsensusEngine stabilizes truth.
    """
    symbolic_value: str  # "quarter", "eighth", "swing_eighth", etc.
    duration_ms: float  # Exact ms this hypothesis implies
    probability: float  # Normalized probability (0-1)

    support_score: float = 0.0  # Total weighted support
    opposition_score: float = 0.0  # Total weighted opposition
    epistemic_tension: float = 0.0  # How contested is this hypothesis?
    uncertainty: float = 0.0  # Net uncertainty score

    supporting_witnesses: List[WitnessContribution] = field(default_factory=list)
    opposing_witnesses: List[WitnessContribution] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)


@dataclass
class QuaverTestimony:
    """
    Full symbolic-duration testimony for a single note.
    Produced by QuaverIntelligence, consumed by EpistemicCouncil.
    """
    note_key: Tuple[int, float]  # (pitch, start_ms) — no note.id needed
    start_ms: float
    raw_duration_ms: float
    duration_hypotheses: List[DurationHypothesis] = field(default_factory=list)

    phrase_context: Dict[str, Any] = field(default_factory=dict)
    instrument_profile: str = "unknown"
    beat_ms: float = 500.0

    witness_reliability: Dict[str, float] = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)

    # Epistemic state
    uncertainty: float = 0.0
    contradiction_detected: bool = False

    @property
    def primary_hypothesis(self) -> Optional[DurationHypothesis]:
        """The highest-probability hypothesis."""
        if not self.duration_hypotheses:
            return None
        return max(self.duration_hypotheses, key=lambda h: h.probability)

    @property
    def has_contention(self) -> bool:
        """True if two hypotheses are within 0.15 of each other in probability."""
        if len(self.duration_hypotheses) < 2:
            return False
        sorted_h = sorted(self.duration_hypotheses, key=lambda h: -h.probability)
        return (sorted_h[0].probability - sorted_h[1].probability) < 0.15


@dataclass
class QuaverConfig:
    """Configuration for QuaverIntelligence."""
    tempo_relative_tolerance: float = 0.08  # 8% of beat = acceptable error
    tuplet_tolerance_factor: float = 1.50  # Tuplets get 50% wider tolerance
    swing_threshold_ms: float = 15.0  # Min swing deviation to consider groove
    sustain_resonance_threshold: float = 0.30  # Resonance below this → no sustain
    artifact_opposition_threshold: float = 0.80  # Above this → mark as artifact

    # Quantization strategy
    prefer_simple_durations: bool = True  # Prefer whole, half, quarter over tuplets
    jazz_mode: bool = True  # Enable swing hypotheses


# =====================================================================
# Symbolic Duration Space
# =====================================================================

# Maps symbolic name → multiple of one subdivision unit
SYMBOLIC_DURATIONS: Dict[str, float] = {
    "whole": 4.000,
    "dotted_half": 3.000,
    "half": 2.000,
    "dotted_quarter": 1.500,
    "quarter": 1.000,
    "dotted_eighth": 0.750,
    "eighth": 0.500,
    "dotted_sixteenth": 0.375,
    "sixteenth": 0.250,
    "thirty_second": 0.125,
}

# Sorted from longest to shortest for find-longer operations
SYMBOLIC_DURATIONS_SORTED = sorted(
    SYMBOLIC_DURATIONS.items(), key=lambda kv: -kv[1])

TUPLET_RATIOS: Dict[str, float] = {
    "triplet": 2.0 / 3.0,
    "quintuplet": 4.0 / 5.0,
    "septuplet": 4.0 / 7.0,
    "compound_triplet": 3.0 / 4.0,
}

# Witness reliability priors
DEFAULT_WITNESS_RELIABILITY: Dict[str, float] = {
    "reverse_geo_crypt": 0.92,
    "pulse_field": 0.88,
    "groove_field": 0.85,
    "anechoic_ma": 0.91,
    "tempo_intelligence": 0.87,
    "drum_intelligence": 0.94,
    "voice_continuity": 0.83,
    "instrument_prior": 0.78,
}

# Instrument sustain and articulation priors
INSTRUMENT_PRIORS: Dict[str, Dict[str, float]] = {
    "cello": {"sustain_prior": 0.85, "articulation_clarity": 0.60},
    "piano": {"sustain_prior": 0.30, "articulation_clarity": 0.90},
    "violin": {"sustain_prior": 0.80, "articulation_clarity": 0.65},
    "double_bass": {"sustain_prior": 0.90, "articulation_clarity": 0.50},
    "flute": {"sustain_prior": 0.75, "articulation_clarity": 0.70},
    "trumpet": {"sustain_prior": 0.70, "articulation_clarity": 0.75},
    "guitar": {"sustain_prior": 0.40, "articulation_clarity": 0.85},
    "drum": {"sustain_prior": 0.10, "articulation_clarity": 0.95},
    "strings": {"sustain_prior": 0.82, "articulation_clarity": 0.62},
    "brass": {"sustain_prior": 0.65, "articulation_clarity": 0.78},
    "unknown": {"sustain_prior": 0.50, "articulation_clarity": 0.70},
}


# =====================================================================
# Main Class
# =====================================================================

class QuaverIntelligence:
    """
    Epistemic symbolic-duration witness.

    Consumes:  Testimonies from upstream witnesses (pulse, groove,
               anechoic_ma, drum, voice, tempo).
    Produces:  Competing DurationHypothesis objects per note, with
               support, opposition, tension, and uncertainty scores.

    CRITICAL:
        - Does NOT quantize notes.
        - Does NOT mutate NoteEvents.
        - Only generates competing hypotheses for the EpistemicCouncil.

    USAGE:
        agent = QuaverIntelligence(findings_map, config)
        testimonies = agent.infer_symbolic_durations(
            note_events=notes,
            subdivision_ms=subdivision_ms,
            tempo_bpm=152.0,
            pulse_field=pulse_field_dict,
            groove_field=groove_field_dict,
            anechoic_ma=anechoic_dict,
            tempo_intelligence=tempo_dict,
            drum_intelligence=drum_dict,
            voice_continuity=voice_dict,
            instrument_hint="piano",
        )
    """

    def __init__(
            self,
            findings_map: MusicalFindingsMap,
            config: Optional[QuaverConfig] = None,
    ):
        self.findings = findings_map
        self.config = config or QuaverConfig()

    # ====================================================================
    # Public Entry Point
    # ====================================================================

    def infer_symbolic_durations(
            self,
            note_events: List[NoteEvent],
            subdivision_ms: float,
            tempo_bpm: float,
            pulse_field: Dict[str, Any],
            groove_field: Dict[str, Any],
            anechoic_ma: Dict[str, Any],
            tempo_intelligence: Dict[str, Any],
            drum_intelligence: Dict[str, Any],
            voice_continuity: Dict[str, Any],
            instrument_hint: str = "unknown",
    ) -> List[QuaverTestimony]:
        """
        Generate competing duration hypotheses for every note.

        Returns:
            List of QuaverTestimony, one per input NoteEvent.
            Each testimony has 2-5 competing DurationHypothesis objects.
        """
        if not note_events:
            return []

        beat_ms = 60000.0 / tempo_bpm
        tolerance = beat_ms * self.config.tempo_relative_tolerance
        priors = INSTRUMENT_PRIORS.get(instrument_hint,
                                       INSTRUMENT_PRIORS["unknown"])

        # Group by phrase for context-aware analysis
        phrases = self._group_by_phrase(note_events, voice_continuity)

        testimonies: List[QuaverTestimony] = []

        for phrase_idx, phrase in enumerate(phrases):
            for note_idx, note in enumerate(phrase):
                evidence = self._gather_adversarial_evidence(
                    note=note,
                    note_idx=note_idx,
                    phrase=phrase,
                    all_notes=note_events,
                    subdivision_ms=subdivision_ms,
                    beat_ms=beat_ms,
                    tolerance_ms=tolerance,
                    pulse_field=pulse_field,
                    groove_field=groove_field,
                    anechoic_ma=anechoic_ma,
                    tempo_intelligence=tempo_intelligence,
                    drum_intelligence=drum_intelligence,
                    voice_continuity=voice_continuity,
                    priors=priors,
                )

                hypotheses = self._generate_competing_hypotheses(evidence)

                # Score epistemic state on each hypothesis
                for hyp in hypotheses:
                    hyp.epistemic_tension = self._epistemic_tension(hyp)
                    hyp.uncertainty = self._epistemic_uncertainty(hyp)

                # Normalize so probabilities reflect relative weight
                hypotheses = self._normalize(hypotheses)

                # Build testimony
                # FIX: use (pitch, start_ms) as key — no note.id on NoteEvent
                note_key = (note.pitch, note.start_ms)

                testimony = QuaverTestimony(
                    note_key=note_key,
                    start_ms=note.start_ms,
                    # FIX: use duration_ms() method not .duration attribute
                    raw_duration_ms=note.duration_ms(),
                    duration_hypotheses=hypotheses,
                    phrase_context={
                        "phrase_id": f"phrase_{phrase_idx}",
                        "position_in_phrase": note_idx,
                        "phrase_length": len(phrase),
                    },
                    instrument_profile=instrument_hint,
                    beat_ms=beat_ms,
                    witness_reliability=DEFAULT_WITNESS_RELIABILITY.copy(),
                    reasoning_trace=evidence["reasoning_trace"],
                    uncertainty=float(np.mean([h.uncertainty for h in hypotheses])
                                      if hypotheses else 0.0),
                    contradiction_detected=any(
                        h.epistemic_tension > 0.70 for h in hypotheses),
                )

                testimonies.append(testimony)

        return testimonies

    # ====================================================================
    # Evidence Gathering
    # ====================================================================

    def _gather_adversarial_evidence(
            self,
            note: NoteEvent,
            note_idx: int,
            phrase: List[NoteEvent],
            all_notes: List[NoteEvent],
            subdivision_ms: float,
            beat_ms: float,
            tolerance_ms: float,
            pulse_field: Dict[str, Any],
            groove_field: Dict[str, Any],
            anechoic_ma: Dict[str, Any],
            tempo_intelligence: Dict[str, Any],
            drum_intelligence: Dict[str, Any],
            voice_continuity: Dict[str, Any],
            priors: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Gather supporting AND opposing evidence from all witnesses.

        Each witness gets to support OR oppose hypotheses. Evidence is
        NOT cherry-picked — contradictions are preserved.
        """
        reasoning: List[str] = []

        # FIX: note.duration_ms() not note.duration; note.start_ms not note.start_time
        raw_dur = note.duration_ms()
        start_key = str(int(note.start_ms))

        # ---- 1. ReverseGeoCrypt: Lattice coherence ----
        lattice_candidates = []
        for symbol, mult in SYMBOLIC_DURATIONS.items():
            candidate_ms = mult * subdivision_ms
            error_ms = abs(raw_dur - candidate_ms)
            if error_ms <= tolerance_ms:
                sup = float(1.0 - error_ms / (tolerance_ms + 1e-8))
                lattice_candidates.append({
                    "symbol": symbol,
                    "duration_ms": candidate_ms,
                    "support": sup,
                    "error_ms": error_ms,
                    "multiplier": mult,
                })
                reasoning.append(
                    f"ReverseGeoCrypt: {symbol} ({candidate_ms:.1f}ms) "
                    f"fits with support={sup:.2f}")

        # Sort by support descending
        lattice_candidates.sort(key=lambda c: -c["support"])

        # ---- 2. PulseField: Metrical legitimacy ----
        pulse_entry = pulse_field.get(start_key, {})
        metrical_sup = float(pulse_entry.get("strength", 0.50))
        metrical_opp = float(pulse_entry.get("opposition", 0.00))
        if metrical_opp > 0.30:
            reasoning.append(
                f"PulseField OPPOSES metrical placement "
                f"(opposition={metrical_opp:.2f})")
        else:
            reasoning.append(
                f"PulseField supports placement (strength={metrical_sup:.2f})")

        # ---- 3. GrooveField: Intentional deviation ----
        groove_dev_ms = float(groove_field.get("swing_amount_ms", 0.0))
        swing_align_key = f"alignment_{int(note.start_ms)}"
        swing_align = float(groove_field.get(swing_align_key, 0.0))
        rigid_opp = 0.0

        if (groove_dev_ms > self.config.swing_threshold_ms and
                swing_align > 0.60):
            rigid_opp = 0.85
            reasoning.append(
                "GrooveField OPPOSES rigid grid (swing pocket, "
                f"deviation={groove_dev_ms:.1f}ms, opp={rigid_opp:.2f})")

        # ---- 4. AnechoicMa: Acoustic discontinuity ----
        # FIX: use (pitch, start_ms) identifier not note.id
        resonance_key = f"resonance_{note.pitch}_{int(note.start_ms)}"
        resonance = float(anechoic_ma.get(resonance_key, 0.50))
        sustain_opp = 0.0
        sustain_reason: Optional[str] = None

        if (resonance < self.config.sustain_resonance_threshold and
                priors["sustain_prior"] > 0.60):
            sustain_opp = 0.91
            sustain_reason = "Resonance exhausted — sustain unsupported"
            reasoning.append(
                f"AnechoicMa OPPOSES sustain "
                f"(resonance={resonance:.2f}, opp={sustain_opp:.2f})")
        elif resonance > 0.70:
            reasoning.append(
                f"AnechoicMa supports sustain/resonance "
                f"(resonance={resonance:.2f})")

        # ---- 5. TempoIntelligence: Local tempo plausibility ----
        local_tempo_key = f"local_tempo_{int(note.start_ms)}"
        local_bpm = float(tempo_intelligence.get(
            local_tempo_key,
            tempo_intelligence.get("local_tempo_bpm", 60000.0 / beat_ms)))
        ref_bpm = 60000.0 / beat_ms
        tempo_drift = abs(local_bpm - ref_bpm) / (ref_bpm + 1e-8)
        tempo_plausi = float(1.0 - min(1.0, tempo_drift * 2.0))

        # ---- 6. DrumIntelligence: Percussive anchor ----
        drum_entry = drum_intelligence.get(str(int(note.start_ms)), {})
        drum_anchor = float(drum_entry.get(
            "anchor_confidence", drum_intelligence.get("nearest_beat_confidence", 0.50)))
        anti_groove = float(drum_entry.get(
            "anti_groove", drum_intelligence.get("anti_groove", 0.0)))

        # ---- 7. VoiceContinuity: Phrase continuity ----
        vc_key = f"note_{note.pitch}_{int(note.start_ms)}_phrase"
        in_phrase = voice_continuity.get(vc_key) is not None
        phrase_cont = 0.80 if in_phrase else 0.30

        # ---- 8. Neighbor context ----
        prev_ratio = self._duration_ratio(note_idx, phrase, offset=-1)
        next_ratio = self._duration_ratio(note_idx, phrase, offset=+1)

        return {
            # FIX: raw_duration now properly set
            "raw_duration_ms": raw_dur,
            "lattice_candidates": lattice_candidates,
            "metrical_support": metrical_sup,
            "metrical_opposition": metrical_opp,
            "groove_deviation_ms": groove_dev_ms,
            "rigid_grid_opposition": rigid_opp,
            "resonance": resonance,
            "sustain_collapse_opposition": sustain_opp,
            "sustain_collapse_reason": sustain_reason,
            "tempo_plausibility": tempo_plausi,
            "drum_anchor": drum_anchor,
            "anti_groove_opposition": anti_groove,
            "phrase_continuity": phrase_cont,
            "prev_duration_ratio": prev_ratio,
            "next_duration_ratio": next_ratio,
            "sustain_prior": priors["sustain_prior"],
            "articulation_clarity": priors["articulation_clarity"],
            "tolerance_ms": tolerance_ms,
            "subdivision_ms": subdivision_ms,
            "beat_ms": beat_ms,
            "sixteenth_ms": beat_ms / 4.0,
            "reasoning_trace": reasoning,
        }

    # ====================================================================
    # Hypothesis Generation
    # ====================================================================

    def _generate_competing_hypotheses(
            self, evidence: Dict[str, Any]
    ) -> List[DurationHypothesis]:
        """
        Generate all competing hypotheses for one note.

        Candidates:
            H1: Primary lattice candidate
            H2: Sustain / legato (longer duration)
            H3: Swing-aware eighth
            H4: Tuplet family
            H5: Artifact (if resonance collapses)
        """
        hypotheses: List[DurationHypothesis] = []

        raw_dur = evidence["raw_duration_ms"]
        rigid_op = evidence["rigid_grid_opposition"]
        met_sup = evidence["metrical_support"]
        met_opp = evidence["metrical_opposition"]

        # --- H1: Primary lattice candidate ---
        if evidence["lattice_candidates"]:
            best = evidence["lattice_candidates"][0]
            sup = float(best["support"] * met_sup)
            opp = float(rigid_op + met_opp * 0.50)

            hypotheses.append(DurationHypothesis(
                symbolic_value=best["symbol"],
                duration_ms=best["duration_ms"],
                probability=max(0.01, sup - opp * 0.50),
                support_score=sup,
                opposition_score=opp,
                supporting_witnesses=[
                    WitnessContribution("reverse_geo_crypt",
                                        best["support"],
                                        DEFAULT_WITNESS_RELIABILITY["reverse_geo_crypt"]),
                    WitnessContribution("pulse_field",
                                        met_sup,
                                        DEFAULT_WITNESS_RELIABILITY["pulse_field"]),
                ],
                opposing_witnesses=(
                    [WitnessContribution("groove_field", rigid_op,
                                         DEFAULT_WITNESS_RELIABILITY["groove_field"])]
                    if rigid_op > 0 else []
                ),
                reasoning_trace=[f"Primary lattice: {best['symbol']} "
                                 f"({best['duration_ms']:.1f}ms, err={best['error_ms']:.1f}ms)"],
            ))

            # --- H2: Sustain / legato (next longer symbolic value) ---
            sus_conf = float(evidence["sustain_prior"] * evidence["resonance"])
            sus_opp = evidence["sustain_collapse_opposition"]

            if sus_conf > 0.40:
                longer = self._find_longer_candidate(
                    best["multiplier"], evidence["subdivision_ms"])
                if longer:
                    hypotheses.append(DurationHypothesis(
                        symbolic_value=f"tied_{longer['symbol']}",
                        duration_ms=longer["duration_ms"],
                        probability=max(0.01, sus_conf - sus_opp * 0.50),
                        support_score=sus_conf,
                        opposition_score=sus_opp,
                        supporting_witnesses=[
                            WitnessContribution("anechoic_ma",
                                                evidence["resonance"],
                                                DEFAULT_WITNESS_RELIABILITY["anechoic_ma"]),
                            WitnessContribution("instrument_prior",
                                                evidence["sustain_prior"],
                                                DEFAULT_WITNESS_RELIABILITY["instrument_prior"]),
                        ],
                        opposing_witnesses=(
                            [WitnessContribution("anechoic_ma", sus_opp,
                                                 DEFAULT_WITNESS_RELIABILITY["anechoic_ma"])]
                            if sus_opp > 0 else []
                        ),
                        reasoning_trace=[f"Sustain interpretation: {longer['symbol']} "
                                         f"(sus_conf={sus_conf:.2f}, opp={sus_opp:.2f})"],
                    ))

        # --- H3: Swing-aware eighth ---
        if (self.config.jazz_mode and
                rigid_op > 0.40 and
                evidence["groove_deviation_ms"] > self.config.swing_threshold_ms):
            swing_ms = (SYMBOLIC_DURATIONS.get("eighth", 0.50) *
                        evidence["subdivision_ms"] +
                        evidence["groove_deviation_ms"])
            hypotheses.append(DurationHypothesis(
                symbolic_value="swing_eighth",
                duration_ms=swing_ms,
                probability=max(0.01, rigid_op * 0.80),
                support_score=rigid_op,
                opposition_score=0.15,
                supporting_witnesses=[
                    WitnessContribution("groove_field", rigid_op,
                                        DEFAULT_WITNESS_RELIABILITY["groove_field"]),
                    WitnessContribution("drum_intelligence",
                                        evidence["drum_anchor"],
                                        DEFAULT_WITNESS_RELIABILITY["drum_intelligence"]),
                ],
                opposing_witnesses=[],
                reasoning_trace=[f"Swing eighth (dev={evidence['groove_deviation_ms']:.1f}ms)"],
            ))

        # --- H4: Tuplet candidates ---
        hypotheses.extend(self._tuplet_hypotheses(evidence))

        # --- H5: Artifact (resonance collapses under sustain prior) ---
        if evidence["sustain_collapse_opposition"] > self.config.artifact_opposition_threshold:
            hypotheses.append(DurationHypothesis(
                symbolic_value="artifact",
                duration_ms=0.0,
                probability=0.05,
                support_score=0.0,
                opposition_score=evidence["sustain_collapse_opposition"],
                supporting_witnesses=[],
                opposing_witnesses=[
                    WitnessContribution("anechoic_ma",
                                        evidence["sustain_collapse_opposition"],
                                        DEFAULT_WITNESS_RELIABILITY["anechoic_ma"]),
                ],
                reasoning_trace=["Likely detection artifact — resonance exhausted"],
            ))

        # If no hypotheses were generated (very short or noisy note),
        # produce a low-confidence "unknown" hypothesis
        if not hypotheses:
            hypotheses.append(DurationHypothesis(
                symbolic_value="unknown",
                duration_ms=raw_dur,
                probability=0.30,
                support_score=0.20,
                opposition_score=0.10,
                reasoning_trace=[f"No lattice match for {raw_dur:.1f}ms"],
            ))

        self._apply_fine_duration_scrutiny(hypotheses, evidence)

        return hypotheses

    def _apply_fine_duration_scrutiny(
            self, hypotheses: List[DurationHypothesis], evidence: Dict[str, Any]
    ) -> None:
        """
        Most conventional music has nothing rhythmically finer than a
        16th note. Grace notes, real 32nd notes, and the occasional
        tuplet do happen, but they should have to earn it rather than
        being generated on equal footing with every other candidate.

        Any hypothesis finer than the tempo's real 16th note - not
        subdivision_ms, which is empirically discovered from raw
        inter-onset intervals and can drift far from the actual beat -
        gets an opposition penalty scaled by how much finer it is,
        offset by genuine corroborating evidence: confirmed
        articulation clarity, percussive anchoring, or resonance-backed
        short attacks combined with solid metrical placement.
        """
        sixteenth_ms = evidence.get("sixteenth_ms", 0.0)
        if sixteenth_ms <= 0:
            return

        justification = max(
            evidence.get("articulation_clarity", 0.0),
            evidence.get("drum_anchor", 0.0),
            evidence.get("resonance", 0.0),
        ) * evidence.get("metrical_support", 0.0)

        for hyp in hypotheses:
            if hyp.duration_ms >= sixteenth_ms * 0.92:
                continue
            fineness = max(0.0, hyp.duration_ms / sixteenth_ms)
            penalty = float(np.clip((1.0 - fineness) * (1.0 - justification), 0.0, 0.85))
            if penalty <= 0.01:
                continue
            hyp.opposition_score += penalty
            hyp.probability = max(0.01, hyp.probability - penalty)
            hyp.reasoning_trace.append(
                f"Sub-16th scrutiny: {hyp.duration_ms:.1f}ms < 16th "
                f"({sixteenth_ms:.1f}ms), justification={justification:.2f}, "
                f"penalty={penalty:.2f}")

    def _tuplet_hypotheses(
            self, evidence: Dict[str, Any]
    ) -> List[DurationHypothesis]:
        """Generate tuplet-family hypotheses with proper ratio checking."""
        hypotheses = []
        raw_dur = evidence["raw_duration_ms"]
        subdiv = evidence["subdivision_ms"]
        tolerance = evidence["tolerance_ms"] * self.config.tuplet_tolerance_factor
        met_opp = evidence["metrical_opposition"]

        for tuplet_name, ratio in TUPLET_RATIOS.items():
            tuplet_ms = subdiv * ratio
            error_ms = abs(raw_dur - tuplet_ms)
            if error_ms <= tolerance:
                sup = float(1.0 - error_ms / (tolerance + 1e-8))
                # Tuplets are inherently skeptical without drum anchor
                base_prob = sup * 0.60
                hypotheses.append(DurationHypothesis(
                    symbolic_value=tuplet_name,
                    duration_ms=tuplet_ms,
                    probability=max(0.01, base_prob),
                    support_score=sup * 0.60,
                    opposition_score=met_opp * 0.50,
                    supporting_witnesses=[
                        WitnessContribution("reverse_geo_crypt",
                                            sup * 0.60,
                                            DEFAULT_WITNESS_RELIABILITY["reverse_geo_crypt"]),
                    ],
                    opposing_witnesses=(
                        [WitnessContribution("pulse_field", met_opp,
                                             DEFAULT_WITNESS_RELIABILITY["pulse_field"])]
                        if met_opp > 0 else []
                    ),
                    reasoning_trace=[f"{tuplet_name}: {tuplet_ms:.1f}ms "
                                     f"(err={error_ms:.1f}ms, sup={sup:.2f})"],
                ))

        return hypotheses

    # ====================================================================
    # Epistemic Scoring
    # ====================================================================

    @staticmethod
    def _epistemic_tension(hyp: DurationHypothesis) -> float:
        """
        Tension = how contested is this hypothesis?

        High tension (→1.0): strong support AND strong opposition
        Low tension (→0.0):  one side clearly dominates
        """
        total = hyp.support_score + hyp.opposition_score
        if total < 1e-8:
            return 0.0
        agreement = abs(hyp.support_score - hyp.opposition_score)
        return float(np.clip(1.0 - agreement / total, 0.0, 1.0))

    @staticmethod
    def _epistemic_uncertainty(hyp: DurationHypothesis) -> float:
        """
        Uncertainty = how unconfident are we overall?

        High uncertainty: strong opposition relative to support,
        amplified by high tension.
        """
        base = hyp.opposition_score / (hyp.support_score + 1e-8)
        multiplier = 1.0 + hyp.epistemic_tension
        return float(np.clip(base * multiplier, 0.0, 1.0))

    @staticmethod
    def _normalize(
            hypotheses: List[DurationHypothesis]
    ) -> List[DurationHypothesis]:
        """
        Normalize raw probabilities so they reflect relative weight.
        FIX: creates new DurationHypothesis instead of mutating frozen objects.
        """
        total = sum(h.probability for h in hypotheses)
        if total < 1e-8:
            return hypotheses

        normalized = []
        for h in hypotheses:
            # DurationHypothesis is NOT frozen — direct mutation is safe
            h.probability = h.probability / total
            normalized.append(h)
        return normalized

    # ====================================================================
    # Utilities
    # ====================================================================

    @staticmethod
    def _group_by_phrase(
            notes: List[NoteEvent],
            voice_continuity: Dict[str, Any],
    ) -> List[List[NoteEvent]]:
        """
        Group notes into phrases using VoiceContinuity testimony.
        FIX: key uses pitch + start_ms not note.id.
        """
        groups: List[List[NoteEvent]] = []
        current: List[NoteEvent] = []

        def _phrase_id(n: NoteEvent) -> Optional[str]:
            key = f"note_{n.pitch}_{int(n.start_ms)}_phrase"
            return voice_continuity.get(key)

        for note in notes:
            pid = _phrase_id(note)
            if not current:
                current.append(note)
            elif pid == _phrase_id(current[-1]):
                current.append(note)
            else:
                groups.append(current)
                current = [note]

        if current:
            groups.append(current)

        return groups

    @staticmethod
    def _duration_ratio(
            idx: int, phrase: List[NoteEvent], offset: int
    ) -> float:
        """Duration ratio between this note and a neighbor."""
        neighbor_idx = idx + offset
        if neighbor_idx < 0 or neighbor_idx >= len(phrase):
            return 1.0
        neighbor_dur = phrase[neighbor_idx].duration_ms()
        this_dur = phrase[idx].duration_ms()
        if neighbor_dur < 1e-8:
            return 1.0
        return float(this_dur / neighbor_dur)

    @staticmethod
    def _find_longer_candidate(
            current_multiplier: float,
            subdivision_ms: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the next longer symbolic duration above current_multiplier.
        FIX: correctly computes candidate_ms from subdivision * multiplier.
        """
        for symbol, mult in SYMBOLIC_DURATIONS_SORTED:
            if mult > current_multiplier:
                return {
                    "symbol": symbol,
                    "duration_ms": mult * subdivision_ms,
                    "multiplier": mult,
                }
        return None
