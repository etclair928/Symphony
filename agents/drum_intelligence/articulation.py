# =================================================================
# MODULE: agents/drum_intelligence/articulation.py
# VERSION: 6.0-draft
# DESCRIPTION: Playing-technique tagging - ghost notes, flams, rolls,
# buzz rolls, rimshot/cross-stick carry-through. Takes a time-ordered
# list of classified ForensicDrumEvents (+ optional raw audio for
# envelope-based refinement) and returns the same events with
# `articulation` (and flam pairing) filled in.
#
# REPLACES: agents/detection/drum_coprocessor.py's ArticulationRefiner,
# plus the flam/roll/buzz-roll logic that used to live inline in
# agents/detection/drum_intelligence.py's DrumIntelligence._classify_articulations.
# =================================================================

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import replace

from .drum_types import DrumIntelligenceConfig, ForensicDrumEvent, DrumArticulation
from core.constants import TARGET_SAMPLE_RATE

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ========================================================================
# Envelope-based refinement primitives
# ========================================================================

class ArticulationRefiner:
    """
    Envelope-based playing-technique analysis beyond simple onset-timing
    proximity checks.
    """

    def __init__(self, config: DrumIntelligenceConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate

    def has_energy_dip(self, audio: np.ndarray, t1_ms: float, t2_ms: float) -> bool:
        """
        True if there's a measurable energy trough between two nearby
        onsets - distinguishes a genuine flam (grace note then main
        stroke: two separate attacks with a dip between) from two onsets
        that just land close together during continuously loud energy
        (e.g. a fast, evenly-loud double stroke that isn't really a flam).
        """
        if not self.config.envelope_flam_gap_required:
            return True

        start = int(min(t1_ms, t2_ms) * self.sample_rate / 1000)
        end = int(max(t1_ms, t2_ms) * self.sample_rate / 1000)
        if end - start < 32:
            return True  # too short a gap to measure meaningfully

        segment = np.abs(audio[start:end])
        if len(segment) < 8:
            return True

        edge_level = max(segment[0], segment[-1])
        trough = float(np.min(segment))
        if edge_level <= 1e-8:
            return True

        return (trough / (edge_level + 1e-8)) < self.config.flam_dip_ratio_threshold

    def detect_buzz_roll_regions(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        A genuine buzz/press roll blurs into a sustained high-zero-
        crossing-rate, low-variance texture rather than discrete
        separable strokes - the discrete-onset-run roll detector below
        can only catch a roll the onset detector manages to split into
        4+ clean hits; a true buzz roll often doesn't. Returns
        (start_ms, end_ms) regions.
        """
        if not LIBROSA_AVAILABLE or len(audio) == 0:
            return []

        window_samples = max(64, int(self.config.buzz_zcr_window_ms * self.sample_rate / 1000))
        hop = max(1, window_samples // 4)

        try:
            zcr = librosa.feature.zero_crossing_rate(
                audio, frame_length=window_samples, hop_length=hop)[0]
        except Exception:
            return []

        if len(zcr) < 5:
            return []

        rolling_variance = np.array([
            np.var(zcr[max(0, i - 2):i + 3]) for i in range(len(zcr))
        ])
        zcr_floor = np.percentile(zcr, self.config.buzz_zcr_percentile)
        is_buzz = (zcr > zcr_floor) & (rolling_variance < self.config.buzz_zcr_variance_threshold)

        regions: List[Tuple[float, float]] = []
        in_region = False
        region_start = 0.0
        for i, flag in enumerate(is_buzz):
            t_ms = i * hop / self.sample_rate * 1000.0
            if flag and not in_region:
                in_region = True
                region_start = t_ms
            elif not flag and in_region:
                in_region = False
                if t_ms - region_start >= self.config.buzz_zcr_window_ms:
                    regions.append((region_start, t_ms))
        if in_region:
            final_ms = len(audio) / self.sample_rate * 1000.0
            if final_ms - region_start >= self.config.buzz_zcr_window_ms:
                regions.append((region_start, final_ms))
        return regions


# ========================================================================
# Articulation Classifier - orchestrates ghost/flam/roll/buzz tagging
# ========================================================================

class ArticulationClassifier:
    """
    Assigns DrumArticulation to a time-ordered list of ForensicDrumEvents:
    ghost notes by velocity floor, flams by same-type proximity + a real
    energy dip, discrete-onset-run rolls by interval, and buzz rolls by
    sustained ZCR texture (audio-dependent, skipped when no audio given).
    """

    def __init__(self, config: DrumIntelligenceConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.refiner = ArticulationRefiner(config, sample_rate)

    def classify(self, events: List[ForensicDrumEvent],
                audio: Optional[np.ndarray] = None) -> List[ForensicDrumEvent]:
        updated_events = self._tag_ghost_notes(events)
        updated_events = self._tag_flams(updated_events, audio)
        updated_events = self._tag_rolls(updated_events)
        if audio is not None:
            updated_events = self._tag_buzz_rolls(updated_events, audio)
        return updated_events

    def _tag_ghost_notes(self, events: List[ForensicDrumEvent]) -> List[ForensicDrumEvent]:
        updated_events = []
        for event in events:
            articulation = DrumArticulation.NORMAL
            if event.velocity <= self.config.ghost_note_max_velocity:
                articulation = DrumArticulation.GHOST
            updated_events.append(replace(
                event,
                articulation=articulation,
                reasoning_chain=event.reasoning_chain + (f"Articulation: {articulation.value}",)
            ))
        return updated_events

    def _tag_flams(self, events: List[ForensicDrumEvent],
                  audio: Optional[np.ndarray]) -> List[ForensicDrumEvent]:
        """
        Two same-type hits close together (a grace note just before the
        main stroke). Requires a real energy dip between the two onsets
        when audio is available - timing proximity alone can't
        distinguish a genuine flam from a fast, evenly-loud double stroke
        that just happens to land in the same window.
        """
        updated_events = list(events)
        for i in range(len(updated_events) - 1):
            e1 = updated_events[i]
            e2 = updated_events[i + 1]

            if e1.drum_type != e2.drum_type:
                continue
            interval = e2.start_ms - e1.start_ms
            if not (20 <= interval <= self.config.flam_window_ms):
                continue

            has_dip = True
            if audio is not None:
                try:
                    has_dip = self.refiner.has_energy_dip(audio, e1.start_ms, e2.start_ms)
                except Exception:
                    has_dip = True

            if has_dip:
                flam_id = f"flam_{i}_{int(e1.start_ms)}"
                updated_events[i] = replace(
                    updated_events[i], flam_pair_id=flam_id, articulation=DrumArticulation.FLAM)
                updated_events[i + 1] = replace(
                    updated_events[i + 1], flam_pair_id=flam_id, articulation=DrumArticulation.FLAM)

        return updated_events

    def _tag_rolls(self, events: List[ForensicDrumEvent]) -> List[ForensicDrumEvent]:
        """roll_min_hits+ consecutive same-type hits, each within
        roll_max_interval_ms of the previous one."""
        updated_events = list(events)
        i = 0
        while i < len(updated_events):
            run = [i]
            j = i + 1
            while j < len(updated_events):
                prev = updated_events[run[-1]]
                curr = updated_events[j]
                if (curr.drum_type == prev.drum_type
                        and 0 < (curr.start_ms - prev.start_ms) <= self.config.roll_max_interval_ms):
                    run.append(j)
                    j += 1
                else:
                    break

            if len(run) >= self.config.roll_min_hits:
                for idx in run:
                    e = updated_events[idx]
                    updated_events[idx] = replace(
                        e, articulation=DrumArticulation.ROLL,
                        reasoning_chain=e.reasoning_chain + (f"Roll: {len(run)}-hit run",)
                    )
                i = run[-1] + 1
            else:
                i += 1

        return updated_events

    def _tag_buzz_rolls(self, events: List[ForensicDrumEvent],
                        audio: np.ndarray) -> List[ForensicDrumEvent]:
        """
        Sustained high-ZCR/low-variance texture gets flagged as a roll
        even when it produced few or no discrete onsets of its own - the
        discrete-run detection above only catches a roll the onset
        detector manages to split into roll_min_hits+ separate clean
        hits, which a real buzz roll often doesn't do.
        """
        try:
            buzz_regions = self.refiner.detect_buzz_roll_regions(audio)
        except Exception:
            buzz_regions = []

        updated_events = list(events)
        for start_ms, end_ms in buzz_regions:
            for idx, e in enumerate(updated_events):
                if start_ms <= e.start_ms <= end_ms and e.articulation != DrumArticulation.ROLL:
                    updated_events[idx] = replace(
                        e, articulation=DrumArticulation.ROLL,
                        reasoning_chain=e.reasoning_chain + (
                            f"Buzz roll region: {start_ms:.0f}-{end_ms:.0f}ms",)
                    )
        return updated_events
