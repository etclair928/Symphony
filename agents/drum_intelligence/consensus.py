# =================================================================
# MODULE: agents/drum_intelligence/consensus.py
# VERSION: 6.0-draft
# DESCRIPTION: Voting and contention resolution - takes votes from
# multiple independent witnesses (SPICE, onset+spectral classification,
# NMF polyphony recovery) and produces one consensus list of
# ForensicDrumEvents, arbitrating same-hit disagreements through
# EpistemicCouncil rather than silently trusting whichever source ran
# first.
#
# REPLACES: agents/detection/drum_intelligence.py's EnsembleDrumDetector,
# and the polyphony/contention half of agents/detection/
# drum_coprocessor.py's DrumCoprocessor (the NMF math itself lives in
# classification.py's NMFDrumDecomposer - this module only orchestrates
# what to do with its output).
# =================================================================

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import replace
from collections import defaultdict

from .drum_types import DrumIntelligenceConfig, ForensicDrumEvent, ConfidenceComponents, DrumTypeVote
from .onset_detection import MultiBandOnsetDetector, SimpleBandFilter
from .classification import SpectralDrumClassifier, AdaptiveSPICEWrapper, NMFDrumDecomposer
from core.order_types import DrumType
from core.constants import TARGET_SAMPLE_RATE


# ========================================================================
# Polyphony Resolver - orchestrates NMFDrumDecomposer's output
# ========================================================================

class PolyphonyResolver:
    """
    Turns NMFDrumDecomposer's raw activation solve into two outcomes:

    polyphony_events: genuinely simultaneous DIFFERENT drum types NMF
    recovered from an overlap candidate - these should coexist, never be
    forced to compete for a single winner.

    contentions: (disputed_event, votes) pairs for candidates where NMF's
    decomposition settled on a SINGLE dominant type that disagrees with
    what the existing classifier already said for that same onset - this
    is one hit, two interpretations, and belongs to
    EpistemicCouncil.resolve_drum_type_contention, not the polyphony path
    (forcing a single winner among genuinely different simultaneous hits
    would defeat the point of NMF). disputed_event is the actual object
    from onset_events so the caller can replace it in place once
    arbitrated, rather than risk both the original guess and the
    arbitrated answer surviving as two separate phantom events at the
    same instant.
    """

    def __init__(self, config: DrumIntelligenceConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.nmf = NMFDrumDecomposer(config, sample_rate)

    def resolve(self, audio: np.ndarray, onset_events: List[ForensicDrumEvent],
               onsets_by_band: Dict[str, List[float]],
               band_ranges: Optional[Dict[str, Tuple[float, float]]] = None
               ) -> Tuple[List[ForensicDrumEvent], List[Tuple[ForensicDrumEvent, List[DrumTypeVote]]]]:
        if not self.config.nmf_enabled:
            return [], []

        polyphony_times = self.nmf.find_polyphony_candidates(onsets_by_band, audio, band_ranges)
        if not polyphony_times:
            return [], []

        templates = self.nmf.bootstrap_templates(onset_events, audio, polyphony_times)
        if len(templates) < 2:
            return [], []

        drum_types = list(templates.keys())[:self.config.nmf_max_components]
        W = np.stack([templates[dt] for dt in drum_types], axis=1)

        polyphony_events: List[ForensicDrumEvent] = []
        contentions: List[Tuple[ForensicDrumEvent, List[DrumTypeVote]]] = []

        for t_ms in polyphony_times:
            H = self.nmf.decompose_window(audio, t_ms, W)
            if H is None:
                continue
            total = float(np.sum(H)) + 1e-8
            active = [(drum_types[i], H[i] / total) for i in range(len(drum_types))
                     if H[i] / total >= self.config.nmf_activation_threshold]

            if len(active) >= 2:
                for drum_type, ratio in active:
                    polyphony_events.append(ForensicDrumEvent(
                        drum_type=drum_type,
                        start_ms=t_ms,
                        end_ms=t_ms + 150,
                        velocity=int(max(1, min(127, ratio * 127))),
                        confidence=ConfidenceComponents(
                            transient_strength=0.6,
                            spectral_match=float(min(1.0, ratio)),
                            temporal_consistency=0.5,
                            model_confidence=0.6,
                            ensemble_agreement=0.0,
                        ),
                        source_model="nmf_decomposer",
                        reasoning_chain=(f"NMF polyphony split: {drum_type.value} activation={ratio:.2f}",),
                    ))
            elif len(active) == 1:
                nmf_type, nmf_ratio = active[0]
                nearest = min(onset_events, key=lambda e: abs(e.start_ms - t_ms), default=None)
                if nearest is not None and abs(nearest.start_ms - t_ms) <= self.config.nmf_polyphony_window_ms \
                        and nearest.drum_type != nmf_type:
                    votes = [
                        DrumTypeVote(source="nmf_decomposer", drum_type=nmf_type,
                                    confidence=float(min(1.0, nmf_ratio)), weight=1.0),
                        DrumTypeVote(source="spectral_classifier", drum_type=nearest.drum_type,
                                    confidence=nearest.confidence.total(), weight=1.0),
                    ]
                    contentions.append((nearest, votes))

        return polyphony_events, contentions


# ========================================================================
# Ensemble Detector - combines SPICE, onset+spectral, and NMF votes
# ========================================================================

class EnsembleDrumDetector:
    """Ensemble detector combining SPICE, onset, and spectral classification."""

    def __init__(self, config: DrumIntelligenceConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate
        self._spice = AdaptiveSPICEWrapper(config, sample_rate)
        self._onset = MultiBandOnsetDetector(config, sample_rate)
        self._spectral = SpectralDrumClassifier(sample_rate)
        self._polyphony = PolyphonyResolver(config, sample_rate)

    def detect(self, audio: np.ndarray, density: float = 0.5, tempo: float = 120.0) -> List[ForensicDrumEvent]:
        """Detect drums using ensemble voting."""
        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        votes: List[List[ForensicDrumEvent]] = []

        if self._spice.available:
            spice_events = self._spice.detect(audio, density, tempo)
            votes.append(spice_events)

        onsets_by_band = self._onset.detect_onsets(audio)
        onset_events = self._onset_to_events(onsets_by_band, audio)
        votes.append(onset_events)

        # NMF polyphony recovery: resolves genuinely simultaneous
        # different-type hits (e.g. kick+tom on the same millisecond) that
        # the centroid classifier above would blend into one wrong guess.
        # Contention votes (NMF and the classifier disagree about a
        # SINGLE hit, not two coexisting ones) get arbitrated through
        # EpistemicCouncil rather than silently trusting either source.
        if self.config.use_ensemble and self.config.nmf_enabled:
            try:
                band_ranges = {'low': self.config.low_band_hz, 'mid': self.config.mid_band_hz,
                              'high': self.config.high_band_hz}
                polyphony_events, contentions = self._polyphony.resolve(
                    audio, onset_events, onsets_by_band, band_ranges)
                if polyphony_events:
                    votes.append(polyphony_events)
                if contentions:
                    # onset_events is the same list object already appended
                    # to votes above - mutating it in place here (instead
                    # of appending a second, competing vote) is what keeps
                    # the arbitrated answer from surviving alongside the
                    # original guess as a phantom duplicate at the same
                    # instant.
                    self._resolve_contentions(contentions, onset_events)
            except Exception:
                pass  # best-effort second layer - never blocks the primary ensemble

        if not votes:
            return []

        consensus = self._find_consensus(votes)

        updated_consensus = []
        for event in consensus:
            agreement_count = sum(1 for v in votes if self._has_event(v, event))
            agreement_ratio = agreement_count / len(votes)

            new_confidence = replace(event.confidence, ensemble_agreement=agreement_ratio)
            updated_event = replace(event, confidence=new_confidence)
            updated_consensus.append(updated_event)

        return updated_consensus

    def _resolve_contentions(self, contentions: List[Tuple[ForensicDrumEvent, List[DrumTypeVote]]],
                             onset_events: List[ForensicDrumEvent]) -> None:
        """
        Arbitrate same-hit disagreements between the centroid classifier
        and NMF via EpistemicCouncil.resolve_drum_type_contention, then
        replace the disputed event's type in place - not append a second
        vote for it, which would let the original guess and the
        arbitrated answer both survive as separate phantom events at the
        same instant.
        """
        from epistemic.epistemic_council import EpistemicCouncil

        for disputed_event, type_votes in contentions:
            result = EpistemicCouncil.resolve_drum_type_contention(type_votes)
            if result is None:
                continue
            winning_type, winning_confidence = result
            if winning_type == disputed_event.drum_type:
                continue  # the classifier's original guess already won

            idx = next((i for i, e in enumerate(onset_events) if e is disputed_event), None)
            if idx is None:
                continue

            new_confidence = replace(disputed_event.confidence, model_confidence=winning_confidence)
            onset_events[idx] = replace(
                disputed_event, drum_type=winning_type, confidence=new_confidence,
                reasoning_chain=disputed_event.reasoning_chain + (
                    f"Contention resolved: {winning_type.value} (was {disputed_event.drum_type.value})",)
            )

    def _onset_to_events(self, onsets: Dict[str, List[float]], audio: np.ndarray) -> List[ForensicDrumEvent]:
        """
        Convert onset detections to forensic events, classifying each
        onset's own band-filtered audio slice via SpectralDrumClassifier
        (falling back to the band's typical drum type only when the
        slice is too short to classify). Classifies from the SAME band-
        filtered audio the onset was found in, not the raw full mix -
        classifying the raw mix lets simultaneous cymbal bleed dominate
        the spectral centroid regardless of which band actually
        triggered the onset.
        """
        events = []
        band_mapping = {
            'low': DrumType.KICK,
            'mid': DrumType.SNARE,
            'high': DrumType.HI_HAT_CLOSED
        }
        band_ranges = {
            'low': self.config.low_band_hz,
            'mid': self.config.mid_band_hz,
            'high': self.config.high_band_hz,
        }
        slice_ms = 80.0

        for band, onset_times in onsets.items():
            band_guess = band_mapping.get(band, DrumType.PERCUSSION)

            band_range = band_ranges.get(band)
            band_audio = (SimpleBandFilter.filter_band(audio, self.sample_rate, band_range[0], band_range[1])
                         if band_range is not None else audio)

            sorted_onset_times = sorted(onset_times)
            for onset_idx, time_ms in enumerate(sorted_onset_times):
                start_sample = int(time_ms * self.sample_rate / 1000)
                end_sample = min(len(band_audio), start_sample + int(slice_ms * self.sample_rate / 1000))
                audio_slice = band_audio[start_sample:end_sample] if start_sample < end_sample else np.array([])

                if len(audio_slice) >= 512:
                    drum_type = self._spectral.classify(audio_slice, time_ms)
                    spectral_match = 0.7
                    source_model = "onset_detector+spectral_classifier"
                else:
                    drum_type = band_guess
                    spectral_match = 0.5
                    source_model = "onset_detector"

                # Every real hit was reporting an identical, hardcoded
                # velocity=70 regardless of how hard it was actually
                # struck - a flat plateau with zero dynamics, audibly
                # lifeless on playback even when the pitches/timing were
                # right. Peak amplitude of this same band-filtered slice
                # (already computed above for classification) is a real,
                # per-hit transient-strength measurement - scaled against
                # typical separated-drum-stem amplitudes observed on real
                # audio (quiet ghost-adjacent hits ~0.03-0.1 peak, loud
                # accented hits ~0.5-0.8 peak).
                peak = float(np.max(np.abs(audio_slice))) if len(audio_slice) > 0 else 0.1
                velocity = int(np.clip(20 + peak * 150, 20, 127))

                confidence = ConfidenceComponents(
                    transient_strength=0.7,
                    model_confidence=0.5,
                    temporal_consistency=0.6,
                    spectral_match=spectral_match,
                    ensemble_agreement=0.0
                )

                # A flat end-time offset for every drum type regardless of
                # how fast it's actually being played caused Scribe's
                # phase-consistency gate to flag consecutive same-type
                # drum hits as "incoherent overlap" (a hi-hat pattern at
                # 8th/16th notes routinely has hits under 150ms apart).
                # Reuses AdaptiveSPICEWrapper's per-type duration table.
                # Additionally caps against the next onset in this same
                # band so a slow-decaying type (crash=800ms) still can't
                # overlap a genuinely fast next hit - capped exactly AT
                # the next onset (a true zero gap), not a hair before it,
                # since even a 1ms buffer lands in the near-zero-gap
                # anomaly band Scribe checks for.
                estimated_end = time_ms + self._spice._estimate_duration(drum_type)
                if onset_idx + 1 < len(sorted_onset_times):
                    estimated_end = min(estimated_end, sorted_onset_times[onset_idx + 1])
                estimated_end = max(estimated_end, time_ms + 10.0)

                events.append(ForensicDrumEvent(
                    drum_type=drum_type,
                    start_ms=time_ms,
                    end_ms=estimated_end,
                    velocity=velocity,
                    confidence=confidence,
                    is_cross_stick=(drum_type == DrumType.RIMSHOT),
                    source_model=source_model,
                    reasoning_chain=(f"Onset detection in {band} band",)
                ))

        events.sort(key=lambda e: (e.start_ms, e.velocity))
        return events

    def _has_event(self, events: List[ForensicDrumEvent], target: ForensicDrumEvent) -> bool:
        for e in events:
            if e.drum_type == target.drum_type and abs(e.start_ms - target.start_ms) < 20:
                return True
        return False

    def _find_consensus(self, votes: List[List[ForensicDrumEvent]]) -> List[ForensicDrumEvent]:
        """
        Groups by (time_bucket, drum_type) rather than time_bucket alone,
        so different-type votes at the same instant coexist (real
        polyphony NMF recovered), while same-type votes from different
        sources still compete/merge into one event.
        """
        all_events = []
        for vote in votes:
            all_events.extend(vote)

        groups = defaultdict(list)
        for event in all_events:
            time_bucket = int(event.start_ms / 25) * 25
            groups[(time_bucket, event.drum_type)].append(event)

        result = []
        for group in groups.values():
            if group:
                best = max(group, key=lambda e: e.confidence.total())
                result.append(best)

        return result
