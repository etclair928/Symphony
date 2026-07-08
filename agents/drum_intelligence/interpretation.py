# =================================================================
# MODULE: agents/drum_intelligence/interpretation.py
# VERSION: 6.0-draft
# DESCRIPTION: The "what does this pattern of hits actually mean"
# layer - repeating-pattern detection, phase-aware microtiming, and
# metrical-position evidence (where in the bar a hit lands, and whether
# that's typical for its drum type). This is the layer the old two-file
# system never had: everything else here only asks "what onset is this
# and what type is it," never "does its position in the bar make sense."
#
# REPLACES: agents/detection/drum_intelligence.py's PatternHashDetector,
# MicrotimingAnalyzer.
#
# NEW: MetricalPositionEvidence - gives task #83 (wire real metrical-
# position evidence into DrumIntelligence classification) an actual
# home. The user's own listening notes on Hopeful-Drums.mp3 ("kick drum
# on 1 always... sometimes the and of 6," "cross stick thing on 4 every
# bar") are exactly this kind of evidence - a drum type's beat-in-bar
# placement is informative, and a candidate hit that matches its type's
# usual placement deserves more confidence than one that doesn't.
# Rather than hardcode genre assumptions about which slot each drum type
# "should" land on, priors are learned per-track from this same run's
# own confidently-classified events (same self-supervised philosophy
# NMFDrumDecomposer already uses for its spectral templates) - so it
# adapts to whatever meter/genre/pattern the actual song has, instead of
# enforcing a 4/4-rock assumption on a 6/8 or hip-hop track.
# =================================================================

from __future__ import annotations

import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import replace
from collections import defaultdict

from .drum_types import DrumIntelligenceConfig, ForensicDrumEvent, DrumMicrotiming
from core.order_types import DrumType


# ========================================================================
# Pattern Hash Detector
# ========================================================================

class PatternHashDetector:
    """Hash-based repeating-pattern detection - O(n) complexity."""

    def __init__(self, config: DrumIntelligenceConfig, tempo: float = 120.0):
        self.config = config
        self.tempo = tempo
        self._hash_cache: Dict[str, List[float]] = {}

    def detect_patterns(self, events: List[ForensicDrumEvent]) -> List[Dict[str, Any]]:
        if not self.config.detect_patterns or len(events) < 8:
            return []

        sixteenth_ms = 60000 / self.tempo / 4
        pattern_length = 16
        pattern_map: Dict[str, List[int]] = defaultdict(list)

        for start_idx in range(0, max(1, len(events) - pattern_length), pattern_length // 4):
            window = events[start_idx:start_idx + pattern_length]

            pattern_parts = []
            for event in window:
                pos = int(event.start_ms / sixteenth_ms) % pattern_length
                pattern_parts.append(f"{event.drum_type.value}:{pos}")

            pattern_str = '|'.join(pattern_parts)
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()[:self.config.pattern_hash_bits]
            pattern_map[pattern_hash].append(start_idx)

        patterns = []
        for pattern_hash, occurrences in pattern_map.items():
            if len(occurrences) >= 2:
                patterns.append({
                    'hash': pattern_hash,
                    'occurrences': occurrences,
                    'count': len(occurrences),
                    'positions_ms': [events[o].start_ms for o in occurrences if o < len(events)]
                })

        return patterns


# ========================================================================
# Microtiming Analyzer - now phase-aware
# ========================================================================

class MicrotimingAnalyzer:
    """
    Subdivision-aware microtiming analysis, measured against grids
    anchored to the REAL detected downbeat rather than assuming beat 1
    always lands at t=0. The old version built grids as flat multiples
    of beat_ms starting from zero - correct only when the very first
    sample happened to be a downbeat. A pickup measure, an intro count-
    in, or simply audio that doesn't start on beat 1 (Hopeful.mp3's own
    "starts on the pickup, beat 4 5 6" case) would silently misjudge
    every event's swing/pocket/rushing/dragging classification relative
    to the wrong phase.

    downbeat_offset_ms: the absolute time (mod beat_ms) at which a real
    downbeat lands - pass 0.0 only when no real downbeat phase is known.
    """

    def __init__(self, config: DrumIntelligenceConfig, tempo: float = 120.0,
                downbeat_offset_ms: float = 0.0):
        self.config = config
        self.tempo = tempo
        self.downbeat_offset_ms = downbeat_offset_ms
        self._steps = self._build_steps()

    def _build_steps(self) -> Dict[str, float]:
        beat_ms = 60000 / self.tempo
        return {
            'quarter': beat_ms,
            'eighth': beat_ms / 2,
            'triplet': beat_ms / 3,
            'sixteenth': beat_ms / 4,
        }

    def _nearest_grid_deviation(self, t_ms: float, step_ms: float) -> float:
        """Signed ms deviation from the nearest phase-aligned grid line."""
        phase = (t_ms - self.downbeat_offset_ms) % step_ms
        if phase > step_ms / 2:
            return phase - step_ms
        return phase

    def analyze(self, events: List[ForensicDrumEvent]) -> List[ForensicDrumEvent]:
        updated_events = []

        for event in events:
            best_grid = None
            best_deviation = float('inf')

            for grid_name, step_ms in self._steps.items():
                if step_ms <= 0:
                    continue
                deviation = self._nearest_grid_deviation(event.start_ms, step_ms)
                if abs(deviation) < abs(best_deviation):
                    best_deviation = deviation
                    best_grid = grid_name

            if abs(best_deviation) < 3:
                category = DrumMicrotiming.POCKET
            elif best_deviation > 0:
                category = DrumMicrotiming.LAID_BACK if best_deviation < 10 else DrumMicrotiming.DRAGGING
            else:
                category = DrumMicrotiming.ON_TOP if best_deviation > -10 else DrumMicrotiming.RUSHING

            updated = replace(
                event,
                microtiming_offset_ms=best_deviation,
                microtiming_category=category,
                reasoning_chain=event.reasoning_chain + (
                    f"Microtiming: {category.value} ({best_deviation:.1f}ms vs {best_grid})",)
            )
            updated_events.append(updated)

        return updated_events


# ========================================================================
# Metrical Position Evidence (NEW)
# ========================================================================

class MetricalPositionEvidence:
    """
    Where in the bar does this drum type usually land, and does THIS hit
    match that? Priors are learned from this same run's own confidently-
    classified events (self-supervised, per-track) rather than a
    hardcoded genre assumption - a kick that clusters on beat 1 in this
    particular song lends support to a marginal beat-1 kick candidate;
    a hi-hat with no real positional preference neither helps nor hurts
    ambiguous hi-hat candidates.
    """

    def __init__(self, config: DrumIntelligenceConfig, tempo: float = 120.0,
                numerator: int = 4, downbeat_offset_ms: float = 0.0):
        self.config = config
        self.tempo = tempo
        self.numerator = max(1, numerator)
        self.downbeat_offset_ms = downbeat_offset_ms
        self.beat_ms = 60000 / tempo if tempo > 0 else 500.0

    def beat_slot(self, start_ms: float) -> int:
        """0-indexed beat position within the bar (0 = downbeat)."""
        beats_elapsed = (start_ms - self.downbeat_offset_ms) / self.beat_ms
        return int(round(beats_elapsed)) % self.numerator

    def learn_priors(self, events: List[ForensicDrumEvent]) -> Dict[DrumType, np.ndarray]:
        """Confidence-weighted histogram of beat-slot placement per drum type."""
        sums: Dict[DrumType, np.ndarray] = defaultdict(lambda: np.zeros(self.numerator))
        for event in events:
            slot = self.beat_slot(event.start_ms)
            sums[event.drum_type][slot] += max(event.confidence.total(), 0.05)

        priors: Dict[DrumType, np.ndarray] = {}
        for drum_type, hist in sums.items():
            total = float(np.sum(hist))
            if total <= 0:
                continue
            priors[drum_type] = hist / total
        return priors

    def apply(self, events: List[ForensicDrumEvent]) -> List[ForensicDrumEvent]:
        """Tag each event with its metrical_position and nudge
        confidence.metrical_fit toward the learned per-type prior."""
        if not self.config.metrical_evidence_enabled or not events:
            return events

        priors = self.learn_priors(events)
        updated_events = []

        for event in events:
            slot = self.beat_slot(event.start_ms)
            prior = priors.get(event.drum_type)

            if prior is None or np.max(prior) <= 0:
                fit = 0.5  # no evidence either way - neutral, not penalized
            else:
                # Normalize against this type's OWN peak so a uniformly-
                # spread drum type (e.g. straight 8th hi-hats with no real
                # peak) doesn't get punished just for lacking a strong
                # preference - only a hit landing somewhere its type
                # almost NEVER lands should lose confidence.
                fit = float(prior[slot] / np.max(prior))

            old_conf = event.confidence
            blended = (1.0 - self.config.metrical_fit_weight) * 0.5 + self.config.metrical_fit_weight * fit
            new_conf = replace(old_conf, metrical_fit=blended)

            updated_events.append(replace(
                event,
                metrical_position=slot,
                confidence=new_conf,
                reasoning_chain=event.reasoning_chain + (
                    f"Metrical position: beat {slot + 1}/{self.numerator} (fit={fit:.2f})",)
            ))

        return updated_events
