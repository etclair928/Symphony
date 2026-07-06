# =================================================================
# MODULE: agents/detection/drum_coprocessor.py
# VERSION: 5.6.1
# DESCRIPTION: Second, MIR-technique-based analytical layer for drum
# detection - runs ALONGSIDE (not instead of) the centroid/crest-factor
# classifier and onset-band ensemble already in drum_intelligence.py.
#
# Covers four gaps a hard-threshold, single-frame-average classifier
# structurally cannot close:
#   1. NMF polyphony recovery - two drums struck at the same instant
#      average their frequencies together under a centroid rule and
#      produce one blended, often-wrong guess. NMFDrumDecomposer solves
#      per-track-bootstrapped template activations (H) with the
#      templates (W) held fixed, splitting the overlapping energy back
#      into separate simultaneous events.
#   2. HFC (High-Frequency Content) onset detection for mid/high bands -
#      HFC(n) = sum_k |X_n(k)|^2 * k weights each bin by its own index,
#      so it's far more sensitive to the sudden broadband bursts typical
#      of cymbal/snare attacks than flat spectral-flux energy is.
#   3. Adaptive median-filtered thresholding - a sliding-window median
#      floor (delta(n) = alpha*median(O(n-k..n+k)) + beta) that hugs the
#      ambient level of the performance, instead of one flat threshold
#      that's wrong for both the quiet ghost-note roll AND the loud fill.
#   4. Envelope-based articulation refinement - a flam requires a real
#      energy dip between the two onsets (not just timing proximity),
#      and a genuine buzz/press roll shows up as a sustained high-ZCR,
#      low-variance texture over a rolling window rather than discrete
#      separable onsets - a real gap the discrete-onset-run roll
#      detector (drum_intelligence.py's _classify_articulations) cannot
#      resolve, since a true buzz roll often doesn't decompose into 4+
#      cleanly separable onsets at all.
# =================================================================

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core.order_types import DrumType
from core.constants import TARGET_SAMPLE_RATE

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class CoprocessorConfig:
    """Configuration for the NMF/HFC drum coprocessor layer."""

    # NMF polyphony recovery
    nmf_enabled: bool = True
    nmf_max_components: int = 4  # kick/snare/tom/other - a practical cap
    nmf_iterations: int = 60
    # Two onsets in different bands within this window are a polyphony
    # candidate - the "kick and tom on the same millisecond" scenario a
    # centroid rule averages away instead of separating.
    nmf_polyphony_window_ms: float = 15.0
    # Skip bootstrapping a drum type's template with fewer than this many
    # clean exemplars - too few examples make a poor, noisy template.
    nmf_min_bootstrap_examples: int = 2
    # Both bootstrap templates and decomposition windows use this same
    # slice length so their FFTs always share the same bin count.
    nmf_template_slice_ms: float = 40.0
    # An NMF component's share of total activation must clear this to
    # count as "present" rather than decomposition noise.
    nmf_activation_threshold: float = 0.15
    # Two co-occurring bands only count as genuine polyphony (not one
    # band's transient bleeding into the other) if the weaker band's
    # instantaneous energy is at least this fraction of the stronger
    # band's - calibrated well above the ~4-19% bleed ratios measured
    # directly from synthetic isolated kick/tom hits.
    nmf_min_energy_ratio: float = 0.3

    # HFC (mid/high bands only - see module docstring for why low/kick
    # is deliberately excluded)
    hfc_enabled: bool = True
    hfc_n_fft: int = 1024
    hfc_hop_length: int = 256

    # Adaptive median-filtered thresholding. Window must be wide relative
    # to a single transient's rise time, or the "ambient floor" gets
    # contaminated by the very peak it's supposed to be judged against -
    # verified directly: with hop_length=256 @ 44100Hz (~5.8ms/frame), a
    # 7-frame (~40ms) centered window was comparable to a real drum
    # attack's own rise time, so median_filter's window captured a big
    # chunk of the rising transient itself, and alpha=1.25 pushed the
    # resulting threshold ABOVE the actual peak - never detectable. 31
    # frames (~180ms) keeps a single attack a small minority of the
    # window, matching standard onset-detection practice.
    median_threshold_enabled: bool = True
    median_window_frames: int = 31
    median_alpha: float = 1.25
    median_beta: float = 0.02

    # Envelope-based articulation refinement
    envelope_flam_gap_required: bool = True
    flam_dip_ratio_threshold: float = 0.6  # trough must be below this fraction of the edges
    buzz_zcr_window_ms: float = 80.0
    buzz_zcr_variance_threshold: float = 0.002
    buzz_zcr_percentile: float = 60.0

    def __post_init__(self):
        assert self.nmf_max_components >= 2, "nmf_max_components must allow at least 2 simultaneous types"
        assert self.nmf_min_bootstrap_examples >= 1
        assert self.median_window_frames >= 3, "median_window_frames must be at least 3"
        if self.median_window_frames % 2 == 0:
            self.median_window_frames += 1


# ========================================================================
# HFC Onset Detector
# ========================================================================

class HFCOnsetDetector:
    """
    High-Frequency Content onset function: HFC(n) = sum_k |X_n(k)|^2 * k.

    Weighting by bin index makes this highly sensitive to sudden bursts
    of high-frequency energy - a washy cymbal or a loose snare's buzz -
    while flat spectral-flux energy treats a bright cymbal crash the same
    as a dull, equally-loud tom thump. Deliberately not used for the
    low/kick band: weighting up by frequency would suppress exactly the
    low-frequency content that defines a kick.
    """

    def __init__(self, config: CoprocessorConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate

    def compute_hfc_curve(self, audio: np.ndarray) -> np.ndarray:
        if not LIBROSA_AVAILABLE or len(audio) < self.config.hfc_n_fft:
            return np.zeros(0)

        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            stft = librosa.stft(audio, n_fft=self.config.hfc_n_fft, hop_length=self.config.hfc_hop_length)
        except Exception:
            return np.zeros(0)

        magnitude = np.abs(stft)
        bin_weights = np.arange(magnitude.shape[0], dtype=np.float64)
        hfc = np.sum((magnitude ** 2) * bin_weights[:, None], axis=0)
        return hfc


# ========================================================================
# Adaptive Median Threshold
# ========================================================================

class AdaptiveMedianThreshold:
    """
    Sliding-window adaptive threshold: delta(n) = alpha*median(O(n-k..n+k)) + beta.

    Hugs the ambient level of the performance instead of a single flat
    value: drops during a quiet ghost-note passage so those notes clear
    it, rises during a loud fill so acoustic ring/bleed doesn't.
    """

    def __init__(self, config: CoprocessorConfig):
        self.config = config

    def compute_threshold(self, curve: np.ndarray) -> np.ndarray:
        if len(curve) == 0:
            return np.zeros(0)
        if not SCIPY_AVAILABLE:
            floor = np.full_like(curve, float(np.median(curve)))
        else:
            floor = median_filter(curve, size=self.config.median_window_frames, mode='reflect')
        return self.config.median_alpha * floor + self.config.median_beta

    def find_peaks_adaptive(self, curve: np.ndarray) -> List[int]:
        if not SCIPY_AVAILABLE or len(curve) < 3:
            return []
        threshold = self.compute_threshold(curve)
        peaks, _ = find_peaks(curve, height=threshold)
        return [int(p) for p in peaks]


# ========================================================================
# NMF Drum Decomposer
# ========================================================================

class NMFDrumDecomposer:
    """
    Resolves simultaneous same-instant drum hits that a single centroid
    guess would blend into one wrong answer, via a plain multiplicative-
    update NMF solve: V ~= W . H, with W (spectral templates) held fixed
    and only H (activation) solved for. Deliberately not sklearn's
    NMF.fit_transform (updates both W and H by default, and "hold W fixed"
    support isn't stable across sklearn versions) - a manual ~10-line
    multiplicative-update loop is simpler and matches the textbook
    formulation directly.

    Templates (W) are bootstrapped from THIS TRACK's own cleanly-
    classified, non-overlapping hits rather than a generic hardcoded
    drum spectrum - self-supervised, calibrated to this specific kit/
    room/mic setup, same philosophy as PersistentWitnessLog/StemScanner
    elsewhere in this codebase (learn from what's actually been
    witnessed rather than assume a universal shape).
    """

    def __init__(self, config: CoprocessorConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate

    def _band_energy_at(self, band_audio: np.ndarray, time_ms: float, window_ms: float = 20.0) -> float:
        start = int(time_ms * self.sample_rate / 1000)
        end = min(len(band_audio), start + int(window_ms * self.sample_rate / 1000))
        if end <= start:
            return 0.0
        segment = band_audio[start:end]
        return float(np.sqrt(np.mean(segment.astype(np.float64) ** 2))) if len(segment) else 0.0

    def find_polyphony_candidates(self, onsets_by_band: Dict[str, List[float]],
                                  audio: Optional[np.ndarray] = None,
                                  band_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> List[float]:
        """
        Timestamps where onsets in 2+ different bands land within
        nmf_polyphony_window_ms of each other AND both bands show
        comparably strong instantaneous energy - not just co-occurring
        onset detections.

        A single sharp transient's broadband click registers as a
        technically-detected onset in every band at once, but at much
        weaker energy than its own dominant band (verified directly on a
        synthetic isolated kick: low-band RMS 0.48 vs. mid-band bleed
        only 0.02, an ~4% ratio; an isolated tom similarly bled only
        ~19% into the low band). A genuine simultaneous kick+tom instead
        shows BOTH bands near their own full strength at once. Requires
        the weaker band's energy to be at least
        nmf_min_energy_ratio (default well above any bleed ratio
        observed) of the stronger band's, at that same instant -
        distinguishes real polyphony from single-transient bleed without
        needing any track-wide statistics.
        """
        window = self.config.nmf_polyphony_window_ms
        tagged: List[Tuple[float, str]] = []
        for band, times in onsets_by_band.items():
            for t in times:
                tagged.append((t, band))
        tagged.sort(key=lambda x: x[0])

        band_audio: Dict[str, np.ndarray] = {}
        if audio is not None and band_ranges is not None:
            from agents.detection.drum_intelligence import SimpleBandFilter
            for band in onsets_by_band.keys():
                band_range = band_ranges.get(band)
                if band_range is not None:
                    band_audio[band] = SimpleBandFilter.filter_band(
                        audio, self.sample_rate, band_range[0], band_range[1])

        def both_prominent(t: float, band: str, t2: float, band2: str) -> bool:
            if band not in band_audio or band2 not in band_audio:
                return True  # no strength data available - fall back to timing-only
            e1 = self._band_energy_at(band_audio[band], t)
            e2 = self._band_energy_at(band_audio[band2], t2)
            if max(e1, e2) <= 1e-10:
                return True
            return (min(e1, e2) / max(e1, e2)) >= self.config.nmf_min_energy_ratio

        candidates = []
        for t, band in tagged:
            bands_nearby = {band}
            for t2, band2 in tagged:
                if abs(t2 - t) <= window and both_prominent(t, band, t2, band2):
                    bands_nearby.add(band2)
            if len(bands_nearby) >= 2:
                candidates.append(t)

        if not candidates:
            return []

        candidates = sorted(set(candidates))
        deduped = [candidates[0]]
        for c in candidates[1:]:
            if c - deduped[-1] > window:
                deduped.append(c)
        return deduped

    def _slice_magnitude(self, audio: np.ndarray, time_ms: float) -> Optional[np.ndarray]:
        slice_samples = int(self.config.nmf_template_slice_ms * self.sample_rate / 1000)
        start = int(time_ms * self.sample_rate / 1000)
        end = min(len(audio), start + slice_samples)
        if end - start < 512:
            return None
        segment = audio[start:end]
        if np.any(~np.isfinite(segment)):
            segment = np.nan_to_num(segment, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.abs(np.fft.rfft(segment, n=slice_samples))

    def bootstrap_templates(self, events: List[Any], audio: np.ndarray,
                            polyphony_times_ms: List[float]) -> Dict[DrumType, np.ndarray]:
        """
        Build one empirical spectral template per drum type from events
        the existing classifier already labeled confidently AND that
        aren't themselves sitting on a polyphony candidate (an event near
        a flagged overlap may itself be the blended/wrong guess NMF
        exists to fix, so it would make a poor exemplar).
        """
        window = self.config.nmf_polyphony_window_ms
        by_type: Dict[DrumType, List[np.ndarray]] = defaultdict(list)

        for event in events:
            if any(abs(event.start_ms - p) <= window for p in polyphony_times_ms):
                continue
            spec = self._slice_magnitude(audio, event.start_ms)
            if spec is not None:
                by_type[event.drum_type].append(spec)

        templates: Dict[DrumType, np.ndarray] = {}
        for drum_type, specs in by_type.items():
            if len(specs) >= self.config.nmf_min_bootstrap_examples:
                templates[drum_type] = np.mean(np.stack(specs), axis=0)
        return templates

    def decompose_window(self, audio: np.ndarray, time_ms: float, W: np.ndarray) -> Optional[np.ndarray]:
        """Solve H via multiplicative updates with W fixed: H <- H * (W^T V) / (W^T W H + eps)."""
        V = self._slice_magnitude(audio, time_ms)
        if V is None:
            return None
        if len(V) != W.shape[0]:
            n = W.shape[0]
            V = V[:n] if len(V) > n else np.pad(V, (0, n - len(V)))

        n_components = W.shape[1]
        H = np.ones(n_components, dtype=np.float64)
        eps = 1e-10
        WtW = W.T @ W
        WtV = W.T @ V
        for _ in range(self.config.nmf_iterations):
            denom = WtW @ H + eps
            H = H * (WtV / denom)
        return H


# ========================================================================
# Articulation Refiner
# ========================================================================

class ArticulationRefiner:
    """
    Envelope-based playing-technique analysis beyond the discrete-onset
    articulation tagging already in drum_intelligence.py.
    """

    def __init__(self, config: CoprocessorConfig, sample_rate: int = TARGET_SAMPLE_RATE):
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
        separable strokes - the discrete-onset-run roll detector
        (drum_intelligence.py's _classify_articulations, added earlier
        this session) can only catch a roll the onset detector manages
        to split into 4+ clean hits; a true buzz roll often doesn't.
        Returns (start_ms, end_ms) regions.
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
# Coordinator
# ========================================================================

class DrumCoprocessor:
    """
    Coordinates the four new mechanisms and exposes the entry points
    EnsembleDrumDetector/DrumIntelligence actually call.
    """

    def __init__(self, config: Optional[CoprocessorConfig] = None, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config or CoprocessorConfig()
        self.sample_rate = sample_rate
        self.hfc = HFCOnsetDetector(self.config, sample_rate)
        self.median_threshold = AdaptiveMedianThreshold(self.config)
        self.nmf = NMFDrumDecomposer(self.config, sample_rate)
        self.articulation = ArticulationRefiner(self.config, sample_rate)

    def detect_onsets_hfc(self, audio: np.ndarray) -> Optional[List[float]]:
        """
        HFC + adaptive-median onset detection, for mid/high bands only.

        Returns None (not []) when HFC genuinely couldn't run (audio too
        short, curve computation failed) so the caller can fall back to
        the standard onset-strength path - but returns [] when HFC ran
        fine and legitimately found no onsets in this band, so a quiet/
        silent band doesn't get double-checked by both methods.
        """
        if not self.config.hfc_enabled:
            return None
        curve = self.hfc.compute_hfc_curve(audio)
        if len(curve) == 0:
            return None

        if self.config.median_threshold_enabled:
            peak_frames = self.median_threshold.find_peaks_adaptive(curve)
        elif SCIPY_AVAILABLE:
            peak_frames = [int(p) for p in find_peaks(curve)[0]]
        else:
            peak_frames = []

        hop = self.config.hfc_hop_length
        return [float(f) * hop / self.sample_rate * 1000.0 for f in peak_frames]

    def decompose_polyphony(self, audio: np.ndarray, onset_events: List[Any],
                            onsets_by_band: Dict[str, List[float]],
                            band_ranges: Optional[Dict[str, Tuple[float, float]]] = None
                            ) -> Tuple[List[Any], List[Tuple[Any, List["DrumTypeVote"]]]]:
        """
        Returns (polyphony_events, contentions).

        polyphony_events: genuinely simultaneous DIFFERENT drum types
        NMF recovered from an overlap candidate - these should coexist,
        never be forced to compete for a single winner.

        contentions: (disputed_event, votes) pairs for candidates where
        NMF's decomposition settled on a SINGLE dominant type that
        disagrees with what the existing classifier already said for
        that same onset - this is one hit, two interpretations, and
        belongs to EpistemicCouncil.resolve_drum_type_contention, not the
        polyphony path (forcing a single winner among genuinely
        different simultaneous hits would defeat the point of NMF).
        disputed_event is the actual object from onset_events so the
        caller can replace it in place once arbitrated, rather than
        risk both the original guess and the arbitrated answer
        surviving as two separate phantom events at the same instant.
        """
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

        polyphony_events: List[Any] = []
        contentions: List[Tuple[Any, List["DrumTypeVote"]]] = []

        # Deferred import (ForensicDrumEvent/ConfidenceComponents live in
        # drum_intelligence.py, which imports THIS module - importing at
        # call time, not module load time, avoids a circular import).
        from agents.detection.drum_intelligence import ForensicDrumEvent, ConfidenceComponents

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


@dataclass(frozen=True)
class DrumTypeVote:
    """One source's vote about what type a single hit is - consumed by
    EpistemicCouncil.resolve_drum_type_contention when two sources
    disagree about the same physical event."""
    source: str
    drum_type: DrumType
    confidence: float
    weight: float = 1.0


# ========================================================================
# Factory
# ========================================================================

def create_drum_coprocessor(sample_rate: int = TARGET_SAMPLE_RATE,
                            config: Optional[CoprocessorConfig] = None) -> DrumCoprocessor:
    return DrumCoprocessor(config=config or CoprocessorConfig(), sample_rate=sample_rate)
