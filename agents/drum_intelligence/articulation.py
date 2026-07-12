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
from core.order_types import DrumType
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

    def is_brush_swell(self, audio: np.ndarray, start_ms: float) -> bool:
        """
        True if the audio near start_ms shows a brush-played snare's real
        envelope: a quiet, bright attack followed 40-120ms later by a
        much louder, lower-centroid "bloom" as the shell/snare-wire
        resonance swells in. See DrumIntelligenceConfig's brush_* fields
        for the full derivation - verified directly on a real recording,
        not assumed from first principles.

        Searches a small tolerance window of alignment offsets around
        start_ms rather than trusting it exactly - verified directly that
        this matters: the real onset ensemble's backtracked start times
        landed early enough into the attack's own onset-detection
        function that a literal-start_ms check alone missed roughly half
        of the same real brush hits a slightly later alignment caught
        (same class of onset-timing-precision issue found elsewhere in
        this project's onset detection this session).
        """
        if not LIBROSA_AVAILABLE:
            return False

        for offset_ms in self.config.brush_alignment_offsets_ms:
            if self._check_brush_swell_at(audio, start_ms + offset_ms):
                return True
        return False

    def _check_brush_swell_at(self, audio: np.ndarray, start_ms: float) -> bool:
        if start_ms < 0:
            return False

        sub_samples = max(64, int(self.config.brush_sub_window_ms * self.sample_rate / 1000))
        start = int(start_ms * self.sample_rate / 1000)
        window_ms = self.config.brush_swell_max_delay_ms + self.config.brush_sub_window_ms
        end = min(len(audio), start + int(window_ms * self.sample_rate / 1000))
        if end - start < sub_samples * 2:
            return False
        segment = audio[start:end]

        rms_trace = []
        centroid_trace = []
        for i in range(0, len(segment) - sub_samples, sub_samples):
            sub = segment[i:i + sub_samples]
            rms_trace.append(float(np.sqrt(np.mean(sub.astype(np.float64) ** 2))))
            try:
                centroid_trace.append(float(np.mean(
                    librosa.feature.spectral_centroid(y=sub, sr=self.sample_rate))))
            except Exception:
                centroid_trace.append(0.0)

        if len(rms_trace) < 3:
            return False

        attack_subwindows = max(1, int(self.config.brush_attack_window_ms / self.config.brush_sub_window_ms))
        attack_rms = float(np.mean(rms_trace[:attack_subwindows]))
        attack_centroid = float(np.mean(centroid_trace[:attack_subwindows]))
        if attack_rms <= 1e-8:
            return False

        peak_idx = int(np.argmax(rms_trace))
        peak_delay_ms = peak_idx * self.config.brush_sub_window_ms
        if not (self.config.brush_swell_min_delay_ms <= peak_delay_ms <= self.config.brush_swell_max_delay_ms):
            return False
        if peak_idx < attack_subwindows:
            return False

        peak_rms = rms_trace[peak_idx]
        peak_centroid = centroid_trace[peak_idx]

        rms_ratio_ok = (peak_rms / attack_rms) >= self.config.brush_swell_min_rms_ratio
        centroid_drop_ok = (attack_centroid - peak_centroid) >= self.config.brush_swell_min_centroid_drop_hz
        return rms_ratio_ok and centroid_drop_ok

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
        # _tag_ghost_notes unconditionally sets articulation to NORMAL/GHOST
        # for every event, with no notion of "already classified as
        # something more specific" - it must run before any stage whose
        # verdict should actually stick, exactly like flam/roll/buzz-roll
        # already rely on by running after it. Brush must follow the same
        # rule (confirmed directly: running it before ghost-tagging meant
        # every one of its verdicts got silently overwritten back to
        # NORMAL/GHOST one line later, despite classify() itself looking
        # correct in isolation).
        updated_events = self._tag_ghost_notes(events)
        if audio is not None:
            updated_events = self._tag_brush_strokes(updated_events, audio)
        updated_events = self._tag_flams(updated_events, audio)
        updated_events = self._tag_rolls(updated_events)
        if audio is not None:
            updated_events = self._tag_buzz_rolls(updated_events, audio)
        return updated_events

    def _tag_brush_strokes(self, events: List[ForensicDrumEvent],
                           audio: np.ndarray) -> List[ForensicDrumEvent]:
        """
        Reclassifies events showing a brush-played snare's real envelope
        (see ArticulationRefiner.is_brush_swell) to SNARE + BRUSH,
        overriding whatever drum_type the base spectral classifier guessed
        - that classifier judges on a single whole-slice centroid/crest
        reading and has no notion of an envelope that changes shape over
        time, so it routes this signature to KICK (tripped by the later
        low-centroid bloom) or HI_HAT_OPEN/RIDE (tripped by the initial
        bright attack) depending on which part of the slice happened to
        dominate its average - never SNARE, regardless of playing
        technique. Runs after ghost-tagging (which unconditionally sets
        articulation to NORMAL/GHOST for every event, with no notion of
        "already classified as something more specific") and before flam/
        roll/buzz-roll (which read drum_type for same-instrument
        grouping) - the same "more specific evidence wins, applied later"
        ordering those already rely on.
        """
        updated_events = list(events)
        for i, e in enumerate(updated_events):
            try:
                is_brush = self.refiner.is_brush_swell(audio, e.start_ms)
            except Exception:
                is_brush = False
            if is_brush:
                updated_events[i] = replace(
                    e, drum_type=DrumType.SNARE, articulation=DrumArticulation.BRUSH,
                    reasoning_chain=e.reasoning_chain + (
                        f"Brush stroke: reclassified from {e.drum_type.value} "
                        "(quiet attack -> louder low-centroid bloom envelope)",)
                )
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
