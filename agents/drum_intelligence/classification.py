# =================================================================
# MODULE: agents/drum_intelligence/classification.py
# VERSION: 6.0-draft
# DESCRIPTION: Per-family classification experts - each takes a witnessed
# onset (or, for NMF, a polyphony candidate) and produces a drum-type
# guess with a confidence breakdown. No voting/contention logic here -
# that's consensus.py's job.
#
# REPLACES: agents/detection/drum_intelligence.py's SpectralDrumClassifier,
# AdaptiveSPICEWrapper, and agents/detection/drum_coprocessor.py's
# NMFDrumDecomposer.
# =================================================================

from __future__ import annotations

import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from .drum_types import DrumIntelligenceConfig, ConfidenceComponents, ForensicDrumEvent
from .onset_detection import SafeResampler, SimpleBandFilter
from core.order_types import DrumType
from core.constants import TARGET_SAMPLE_RATE

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

SPICE_AVAILABLE = False
try:
    from spice import SPICE, SPICEConfig
    SPICE_AVAILABLE = True
except ImportError:
    pass


# ========================================================================
# Spectral Classifier - centroid + crest-factor ladder
# ========================================================================

class SpectralDrumClassifier:
    """Spectral-based drum classification for fallback mode."""

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def classify(self, audio_slice: np.ndarray, onset_time_ms: float) -> DrumType:
        """Classify drum type from audio slice."""
        if not LIBROSA_AVAILABLE or len(audio_slice) < 512:
            return DrumType.PERCUSSION

        if np.any(~np.isfinite(audio_slice)):
            audio_slice = np.nan_to_num(audio_slice, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(
                y=audio_slice, sr=self.sample_rate
            )))

            rms = np.sqrt(np.mean(audio_slice ** 2))
            peak = np.max(np.abs(audio_slice))
            crest_factor = peak / (rms + 1e-8)

            # Centroid alone misses kicks that land simultaneously with a
            # cymbal/hi-hat hit (extremely common - kick+hihat on beat 1)
            # because centroid is energy-WEIGHTED: broadband cymbal energy
            # dominates the average even when a real kick fundamental is
            # also present, pushing the whole hit's centroid up into the
            # hi-hat/ride/crash range regardless of the kick underneath.
            # Verified directly on a real isolated drum stem: raw low-
            # frequency energy ratio is strongly BIMODAL (a large cluster
            # with ~0% energy below 150Hz - genuinely no kick - and another
            # large cluster at 85-100% - real kick fundamentals - with a
            # sparse valley between), while spectral_centroid alone showed
            # the same onsets pushed almost entirely into the >3000Hz
            # buckets. Checking low-frequency content FIRST, independent
            # of whatever else is happening in the rest of the spectrum,
            # catches these masked kicks without needing to guess whether
            # a hit was "pure" kick or kick-plus-something-else.
            #
            # Measured over the ATTACK only (first ~30ms), not the whole
            # slice - a real kick's fundamental dominates from the very
            # first instant. Measuring over the full ~150ms slice instead
            # let a completely different, common pattern trip this same
            # shortcut: a quiet, bright transient (centroid 4000-10000Hz)
            # followed 40-120ms later by a much louder, low-centroid
            # "bloom" as a drum shell/snare-wire resonance swells in -
            # verified directly on a real brush-played snare recording
            # (Copper_and_Skin.mp3): 60/225 onsets showed a >5x RMS swell
            # at a tightly clustered 40-120ms delay (median 100ms), too
            # consistent to be incidental overlap with a separate kick -
            # a genuine acoustic signature of this playing technique, not
            # noise. Integrating low-frequency energy across that whole
            # window let the LATER bloom's energy dominate the ratio and
            # misroute every one of these to KICK, even though the attack
            # itself was never kick-like.
            attack_samples = min(len(audio_slice), int(0.03 * self.sample_rate))
            attack = audio_slice[:attack_samples]
            spectrum = np.abs(np.fft.rfft(attack)) ** 2
            freqs = np.fft.rfftfreq(len(attack), 1.0 / self.sample_rate)
            total_energy = float(np.sum(spectrum)) + 1e-12
            low_freq_ratio = float(np.sum(spectrum[freqs < 150])) / total_energy

            if low_freq_ratio > 0.15:
                return DrumType.KICK

            if spectral_centroid < 150:
                return DrumType.KICK
            elif spectral_centroid < 800:
                if crest_factor > 5:
                    return DrumType.SNARE
                else:
                    return DrumType.TOM_MID
            elif spectral_centroid < 3000:
                # Cross-stick/rimshot technique (stick on rim, not a full
                # stroke) - its bright, short "click" legitimately sits in
                # this same centroid range as hi-hat/ride. Verified
                # directly on a real second-line snare performance (heavy
                # cross-stick use): mid-band onsets clustered at centroid
                # 1200-2500Hz (median ~1470Hz), crest 3-8.35 (median ~4.3).
                if crest_factor > 6:
                    return DrumType.RIMSHOT
                else:
                    return DrumType.RIDE
            elif spectral_centroid < 4000:
                if crest_factor > 8:
                    return DrumType.HI_HAT_CLOSED
                else:
                    return DrumType.RIDE
            else:
                if crest_factor > 10:
                    return DrumType.CRASH
                else:
                    return DrumType.HI_HAT_OPEN
        except Exception:
            return DrumType.PERCUSSION

    def get_confidence_components(self, audio_slice: np.ndarray) -> ConfidenceComponents:
        """Get confidence breakdown for classification."""
        if not LIBROSA_AVAILABLE:
            return ConfidenceComponents()

        try:
            rms = np.sqrt(np.mean(audio_slice ** 2))
            peak = np.max(np.abs(audio_slice))
            transient_strength = min(1.0, peak / (rms + 1e-8) / 10)

            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(
                y=audio_slice, sr=self.sample_rate
            )))
            spectral_match = 1.0 - min(1.0, spectral_centroid / 10000)

            envelope = np.abs(audio_slice)
            envelope_smooth = gaussian_filter1d(envelope, sigma=5)
            temporal_consistency = 1.0 - np.var(envelope - envelope_smooth) / (np.var(envelope) + 1e-8)

            return ConfidenceComponents(
                transient_strength=transient_strength,
                spectral_match=spectral_match,
                temporal_consistency=float(temporal_consistency),
                model_confidence=0.5,
                ensemble_agreement=0.0
            )
        except Exception:
            return ConfidenceComponents()


# ========================================================================
# Adaptive SPICE Wrapper
# ========================================================================

class AdaptiveSPICEWrapper:
    """SPICE wrapper with adaptive threshold and forensic output.

    SPICE_AVAILABLE is False whenever the `spice` package isn't installed
    in the environment - `available` reflects that honestly rather than
    pretending a lazily-loaded model exists.
    """

    def __init__(self, config: DrumIntelligenceConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate
        self._model = None
        self._model_load_time_ms = 0.0
        self._available = False
        self._init_model()

    def _init_model(self):
        if not SPICE_AVAILABLE:
            return
        try:
            load_start = time.time()
            spice_config = SPICEConfig(
                model_size=self.config.spice_model_size,
                hop_length_ms=self.config.spice_hop_length_ms
            )
            self._model = SPICE(spice_config)
            self._model_load_time_ms = (time.time() - load_start) * 1000
            self._available = True
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, audio: np.ndarray, density: float = 0.5, tempo: float = 120.0) -> List[ForensicDrumEvent]:
        """Detect drums using SPICE with adaptive threshold."""
        if not self._available:
            return []

        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        threshold = self.config.spice_onset_threshold
        if self.config.spice_adaptive_threshold:
            # Denser passages and faster tempos mean more overlapping
            # transients, so weak/spurious activations are MORE likely to
            # get mistaken for real onsets there - require more evidence
            # (raise the threshold), not less.
            threshold += density * 0.1
            threshold += ((tempo - 60) / 180) * 0.05
            threshold = max(0.2, min(0.8, threshold))

        audio_22050 = SafeResampler.resample(audio, self.sample_rate, 22050)

        try:
            activations = self._model.predict(audio_22050, threshold=threshold)
            return self._convert_activations(activations)
        except Exception:
            return []

    def _convert_activations(self, activations: Dict[str, np.ndarray]) -> List[ForensicDrumEvent]:
        events = []

        for drum_name, activation in activations.items():
            drum_type = self._map_drum_name(drum_name)
            if drum_type is None:
                continue

            peaks, _ = find_peaks(
                activation,
                height=self.config.spice_onset_threshold,
                distance=int(self.config.spice_hop_length_ms / 1000 * 22050)
            )

            for peak in peaks:
                time_ms = peak * self.config.spice_hop_length_ms
                strength = activation[peak]
                velocity = int(strength * 127)

                confidence = ConfidenceComponents(
                    model_confidence=strength,
                    transient_strength=strength,
                    temporal_consistency=0.8,
                    spectral_match=0.7,
                    ensemble_agreement=0.0
                )

                events.append(ForensicDrumEvent(
                    drum_type=drum_type,
                    start_ms=time_ms,
                    end_ms=time_ms + self._estimate_duration(drum_type),
                    velocity=velocity,
                    confidence=confidence,
                    source_model="spice",
                    reasoning_chain=(f"SPICE detection: {drum_name}",)
                ))

        events.sort(key=lambda e: (e.start_ms, e.velocity))
        return events

    def _map_drum_name(self, name: str) -> Optional[DrumType]:
        mapping = {
            'kick': DrumType.KICK,
            'snare': DrumType.SNARE,
            'hihat': DrumType.HI_HAT_CLOSED,
            'hihat_closed': DrumType.HI_HAT_CLOSED,
            'hihat_open': DrumType.HI_HAT_OPEN,
            'crash': DrumType.CRASH,
            'ride': DrumType.RIDE,
            'tom': DrumType.TOM_MID,
        }
        return mapping.get(name.lower())

    def _estimate_duration(self, drum_type: DrumType) -> float:
        durations = {
            DrumType.KICK: 300,
            DrumType.SNARE: 250,
            DrumType.HI_HAT_CLOSED: 80,
            DrumType.HI_HAT_OPEN: 400,
            DrumType.CRASH: 800,
            DrumType.RIDE: 500,
            DrumType.TOM_HIGH: 400,
            DrumType.TOM_MID: 500,
            DrumType.TOM_LOW: 600,
        }
        return durations.get(drum_type, 200)


# ========================================================================
# NMF Drum Decomposer - polyphony recovery
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
    room/mic setup.
    """

    def __init__(self, config: DrumIntelligenceConfig, sample_rate: int = TARGET_SAMPLE_RATE):
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
        shows BOTH bands near their own full strength at once.
        """
        window = self.config.nmf_polyphony_window_ms
        tagged: List[Tuple[float, str]] = []
        for band, times in onsets_by_band.items():
            for t in times:
                tagged.append((t, band))
        tagged.sort(key=lambda x: x[0])

        band_audio: Dict[str, np.ndarray] = {}
        if audio is not None and band_ranges is not None:
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
