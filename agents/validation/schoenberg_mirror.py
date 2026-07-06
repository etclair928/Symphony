#!/usr/bin/env python3
"""
agents/validation/schoenberg_mirror.py - Harmonic series auditor, Epistemic Veto for pitch legitimacy.

VERSION: 5.6.1 (rebuilt - the prior 5.2 file was corrupted: UTF-16 encoded with
its entire body commented out, evidently the leftover of a botched stub edit)
UPDATED: 2026-07-02

PHILOSOPHY:
    "The Schoenberg Mirror does not transcribe. It audits."
    "It asks: Is this pitch real, or is it hallucinated?"

    A note passes the mirror if it exhibits a legitimate harmonic series.
    Percussive sounds (drums) are exempt and receive PERCUSSION verdict.
    Noise (unpitched, non-harmonic) receives NOISE verdict and triggers veto.

REBUILD NOTES (6.0):
    This rebuild restores the 5.2 design (HarmonicSeriesAnalyzer doing its
    own STFT + partial-picking on a raw audio slice, ZeroCrossingAnalyzer,
    SpectralFlatnessAnalyzer, FeatureBundleAnalyzer) and fixes the stale
    import: `core.acoustic_intelligence.AudioContract` no longer exists -
    AudioContract is now a Protocol in core.contracts, and the concrete
    implementation is core.acoustic_intelligence.ImmutableAudio, built via
    AcousticIntelligence.create_contract(). Everything else in the 5.2
    design (verdict logic, data types) was still fully compatible with the
    current core.order_types.

    It also borrows a genuinely useful idea from an earlier, differently-
    scoped predecessor of this module ("The Four Mirrors of Truth", which
    validated drum-hit onsets via four literal-transformation tests: ZCR,
    reversed-audio envelope comparison, frequency-inverted HPS, and NMF
    template matching). That module's data model doesn't fit here (it was
    validating "is this a real drum hit", not "is this pitch legitimate"),
    but its central principle - apply a literal transformation and measure
    what survives, rather than relying on a single statistical threshold -
    is worth keeping. Two of its checks are folded in as independent
    corroborating witnesses:
      - RetrogradeSymmetryAnalyzer: reverses the note's audio window and
        compares the amplitude envelope forward vs backward. A real
        percussive attack looks structurally different played backwards
        (asymmetric decay); a sustained tonal note looks similar either
        way (symmetric). This corroborates (or contradicts) the
        ZCR-based percussion call instead of relying on ZCR alone.
      - SpectralInversionAnalyzer: runs Harmonic Product Spectrum on both
        the normal and frequency-flipped spectrum. This is a different
        mathematical test for harmonic structure than direct partial-
        picking, so agreement between the two raises confidence and
        disagreement is a real signal, not noise.
    When independent witnesses disagree, the verdict downgrades to
    UNCERTAIN rather than picking a side - this is what
    ConfidenceComponents.ensemble_agreement is for, and it's now actually
    populated instead of hardcoded to 0.0.

INTEGRATION WITH ARCHITECTURE:
    - Uses AcousticIntelligence.create_contract() for canonical sample access
    - Uses FeatureBundle for spectral evidence (lazy, shared) as a fallback
      when a raw audio slice isn't available
    - Produces SchoenbergResult for Scribe validation
    - Callable directly, via a dedicated pipeline stage, or via Scribe's
      validation gate (Scribe delegates to this class rather than
      reimplementing the check)

LAW OF EPISTEMIC VETO:
    "Any qualified witness can declare a falsehood."
    The Schoenberg Mirror is a qualified witness with veto power.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any

import numpy as np

# Core imports
from core.order_types import (
    NoteEvent, SchoenbergResult, SchoenbergVerdict, HarmonicSeries,
    PartialTrack, SourceType, Confidence, AudioContext
)
from core.constants import (
    HARMONIC_SERIES_MIN_PARTIALS,
    HARMONIC_SERIES_MATCH_THRESHOLD,
    MAX_INHARMONICITY,
    ZERO_CROSSING_MAX,
    SPECTRAL_FLATNESS_TONAL_THRESHOLD,
    SPECTRAL_FLATNESS_NOISE_THRESHOLD,
    VERDICT_BASE_CONFIDENCE,
)
from core.contracts import AudioContract, ConfidenceComponents
from core.acoustic_intelligence import AcousticIntelligence
from core.feature_bundle import FeatureBundle, EvidenceType
from core.protocols import MusicBoxProtocol, StatusReporterProtocol

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# =====================================================================
# Constants
# =====================================================================

# Partial detection thresholds
MIN_PARTIAL_AMPLITUDE_DB: float = -80.0  # Below this is noise floor
PARTIAL_PROMINENCE_THRESHOLD: float = 0.3  # Must be this prominent vs neighbors

# Tolerance bands (cents) - musical, not linear
POWER_PARTIAL_TOLERANCE_CENTS: float = 30.0  # Harmonics 1-3: strict
MID_PARTIAL_TOLERANCE_CENTS: float = 50.0  # Harmonics 4-6: moderate
HIGH_PARTIAL_TOLERANCE_CENTS: float = 70.0  # Harmonics 7+: loose (string stretch)

# Frequency analysis parameters
FFT_SIZE: int = 4096
HOP_LENGTH: int = 512
WINDOW_TYPE: str = "hann"

# Retrograde symmetry analysis (literal-transformation witness for
# percussion vs tonal, corroborating the ZCR-based call)
RETROGRADE_WINDOW_PRE_MS: float = 20.0
RETROGRADE_WINDOW_POST_MS: float = 150.0
RETROGRADE_PERCUSSION_THRESHOLD: float = 0.6  # similarity below this = asymmetric = percussive
RETROGRADE_TONAL_THRESHOLD: float = 0.85  # similarity above this = strongly symmetric = tonal

# Spectral inversion (HPS on normal + frequency-flipped spectrum) - second,
# independent corroborating witness for tonal/harmonic content
HPS_NUM_HARMONICS: int = 4
SPECTRAL_INVERSION_WINDOW_MS: float = 50.0


# =====================================================================
# Harmonic Series Analyzer
# =====================================================================

class HarmonicSeriesAnalyzer:
    """
    Analyzes audio to detect harmonic series for a candidate fundamental.

    This is the core mathematical engine of the Schoenberg Mirror.
    Pure function - no persistent state across notes beyond an STFT cache
    that must be reset() between notes.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._cached_stft: Optional[np.ndarray] = None
        self._cached_freqs: Optional[np.ndarray] = None

    def analyze(
            self,
            audio_slice: np.ndarray,
            fundamental_hz: float,
            sample_rate: int
    ) -> Optional[HarmonicSeries]:
        """
        Analyze harmonic series for a candidate fundamental.

        Args:
            audio_slice: Audio segment containing the note
            fundamental_hz: Candidate fundamental frequency in Hz
            sample_rate: Sample rate of audio

        Returns:
            HarmonicSeries if enough partials detected, None otherwise
        """
        if audio_slice is None or len(audio_slice) < sample_rate * 0.01:
            return None

        if fundamental_hz <= 0 or fundamental_hz > sample_rate / 2:
            return None

        self.sample_rate = sample_rate
        if self._cached_stft is None:
            self._compute_stft(audio_slice, sample_rate)

        partials = self._detect_partials(fundamental_hz)

        if len(partials) < HARMONIC_SERIES_MIN_PARTIALS:
            return None

        inharmonicity = self._calculate_inharmonicity(partials, fundamental_hz)

        expected_partials = list(range(1, 9))  # Harmonics 1-8
        present_numbers = {p.partial_number for p in partials}
        missing = [n for n in expected_partials if n not in present_numbers]
        match_ratio = len(partials) / len(expected_partials)

        confidence = match_ratio * (1.0 - min(inharmonicity, 0.1) / 0.1)
        confidence = max(0.0, min(1.0, confidence))

        return HarmonicSeries(
            fundamental_freq_hz=fundamental_hz,
            fundamental_confidence=confidence,
            partials=partials,
            missing_partials=missing,
            inharmonicity=inharmonicity
        )

    def _compute_stft(self, audio: np.ndarray, sample_rate: int) -> None:
        """Compute STFT for the audio slice (averaged over time frames)."""
        if LIBROSA_AVAILABLE:
            stft = librosa.stft(audio, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, window=WINDOW_TYPE)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=FFT_SIZE)
            self._cached_stft = np.mean(magnitude, axis=1)
            self._cached_freqs = freqs
        else:
            fft = np.fft.rfft(audio, n=FFT_SIZE)
            self._cached_stft = np.abs(fft)
            self._cached_freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sample_rate)

    def _detect_partials(self, fundamental_hz: float) -> List[PartialTrack]:
        """Detect harmonic partials in the spectrum."""
        if self._cached_stft is None or self._cached_freqs is None:
            return []

        partials = []

        for harmonic in range(1, 9):
            target_freq = fundamental_hz * harmonic
            if target_freq > self.sample_rate / 2:
                continue

            idx = int(np.argmin(np.abs(self._cached_freqs - target_freq)))
            detected_freq = self._cached_freqs[idx]
            amplitude = self._cached_stft[idx]

            amplitude_db = 20 * np.log10(amplitude + 1e-8)
            if amplitude_db < MIN_PARTIAL_AMPLITUDE_DB:
                continue

            if 0 < idx < len(self._cached_stft) - 1:
                left = self._cached_stft[idx - 1]
                right = self._cached_stft[idx + 1]
                if amplitude < max(left, right) * (1 + PARTIAL_PROMINENCE_THRESHOLD):
                    continue  # Not a clear peak - might be noise

            cent_deviation = 1200 * np.log2(detected_freq / target_freq) if target_freq > 0 else 0.0

            if harmonic <= 3:
                tolerance_cents, weight = POWER_PARTIAL_TOLERANCE_CENTS, 1.0
            elif harmonic <= 5:
                tolerance_cents, weight = MID_PARTIAL_TOLERANCE_CENTS, 0.7
            else:
                tolerance_cents, weight = HIGH_PARTIAL_TOLERANCE_CENTS, 0.4

            if abs(cent_deviation) <= tolerance_cents:
                deviation_ratio = abs(cent_deviation) / tolerance_cents
                confidence = max(0.0, min(1.0, (1.0 - deviation_ratio) * weight))

                partials.append(PartialTrack(
                    partial_number=harmonic,
                    frequency_hz=float(detected_freq),
                    amplitude_db=float(amplitude_db),
                    confidence=float(confidence)
                ))

        return partials

    def _calculate_inharmonicity(self, partials: List[PartialTrack], fundamental_hz: float) -> float:
        """Weighted inharmonicity: deviation from perfect integer-multiple harmonics."""
        if not partials:
            return 1.0

        total_deviation = 0.0
        total_weight = 0.0

        for p in partials:
            expected = fundamental_hz * p.partial_number
            if expected > 0:
                deviation = abs(p.frequency_hz - expected) / expected
                weight = 1.0 / p.partial_number  # Lower harmonics matter more for pitch perception
                total_deviation += deviation * weight
                total_weight += weight

        return total_deviation / total_weight if total_weight > 0 else 1.0

    def reset(self) -> None:
        """Reset cached STFT before analyzing the next note."""
        self._cached_stft = None
        self._cached_freqs = None


# =====================================================================
# Retrograde Symmetry Analyzer (literal transformation: reverse and compare)
# =====================================================================

class RetrogradeSymmetryAnalyzer:
    """
    Reverses the note's audio window in time and compares the amplitude
    envelope forward vs backward.

    A real percussive attack (drum hit) has a sharp onset and long decay -
    played backwards, that looks completely different (slow swell into a
    sharp cutoff). A sustained tonal note has a much more symmetric
    envelope either way. This gives an independent, non-statistical
    signal for percussion-vs-tonal that doesn't depend on spectral content
    at all, so it corroborates (or contradicts) the ZCR-based call.
    """

    @staticmethod
    def _envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        envelope = np.abs(audio)
        window_size = max(1, int(0.005 * sample_rate))
        if window_size > 1 and len(envelope) > window_size:
            kernel = np.ones(window_size) / window_size
            envelope = np.convolve(envelope, kernel, mode='same')
        return envelope

    @staticmethod
    def _analyze_envelope(envelope: np.ndarray) -> Optional[Dict[str, float]]:
        if len(envelope) == 0:
            return None

        peak_idx = int(np.argmax(envelope))
        peak_val = float(envelope[peak_idx])
        peak_position = peak_idx / len(envelope)

        attack_slope = peak_val / (peak_idx + 1e-8) if peak_idx > 0 else 0.0
        decay_samples = len(envelope) - peak_idx
        decay_slope = peak_val / (decay_samples + 1e-8) if decay_samples > 0 else 0.0

        total_energy = float(np.sum(envelope))
        early_energy = float(np.sum(envelope[:len(envelope) // 2]))
        skewness = early_energy / (total_energy + 1e-8)

        return {
            "peak_position": peak_position,
            "attack_slope": attack_slope,
            "decay_slope": decay_slope,
            "skewness": skewness,
        }

    @staticmethod
    def _similarity(fwd: Dict[str, float], rev: Dict[str, float]) -> float:
        """Higher = more symmetric (tonal-like). Lower = more asymmetric (percussive-like)."""
        position_similarity = 1.0 - min(1.0, abs(fwd["peak_position"] - rev["peak_position"]) / 0.5)
        skew_similarity = 1.0 - min(1.0, abs(fwd["skewness"] - rev["skewness"]))

        ratio_fwd = fwd["attack_slope"] / (fwd["decay_slope"] + 1e-8)
        ratio_rev = rev["attack_slope"] / (rev["decay_slope"] + 1e-8)
        slope_similarity = 1.0 - min(1.0, abs(ratio_fwd - ratio_rev) / 10.0)

        return position_similarity * 0.4 + skew_similarity * 0.3 + slope_similarity * 0.3

    def analyze(self, audio_slice: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Returns a dict with 'similarity' (0-1, higher = more symmetric/tonal)
        and 'verdict_hint' ('percussive', 'tonal', or 'uncertain').
        """
        if audio_slice is None or len(audio_slice) < sample_rate * 0.02:
            return {"similarity": 0.5, "verdict_hint": "uncertain"}

        fwd_analysis = self._analyze_envelope(self._envelope(audio_slice, sample_rate))
        rev_analysis = self._analyze_envelope(self._envelope(audio_slice[::-1], sample_rate))

        if fwd_analysis is None or rev_analysis is None:
            return {"similarity": 0.5, "verdict_hint": "uncertain"}

        similarity = self._similarity(fwd_analysis, rev_analysis)

        if similarity < RETROGRADE_PERCUSSION_THRESHOLD:
            hint = "percussive"
        elif similarity > RETROGRADE_TONAL_THRESHOLD:
            hint = "tonal"
        else:
            hint = "uncertain"

        return {"similarity": similarity, "verdict_hint": hint}


# =====================================================================
# Spectral Inversion Analyzer (literal transformation: flip spectrum, run HPS)
# =====================================================================

def harmonic_product_spectrum(spectrum: np.ndarray, num_harmonics: int = HPS_NUM_HARMONICS) -> Tuple[int, float]:
    """
    Harmonic Product Spectrum: multiplies downsampled copies of the
    spectrum together, so a true fundamental (whose harmonics all line up)
    produces a strong product peak even when the fundamental itself is weak.
    """
    if len(spectrum) == 0:
        return 0, 0.0

    hps = spectrum.astype(np.float64).copy()
    for h in range(2, num_harmonics + 1):
        downsampled = spectrum[::h]
        n = min(len(hps), len(downsampled))
        hps[:n] *= downsampled[:n]

    peak_idx = int(np.argmax(hps))
    return peak_idx, float(hps[peak_idx])


class SpectralInversionAnalyzer:
    """
    Runs HPS on the normal spectrum, then LITERALLY flips the frequency
    axis (spectrum[::-1]) and runs HPS again. A genuinely harmonic sound
    still produces a comparatively strong HPS peak even mirrored, because
    its energy really is concentrated in a small number of related bins;
    noise stays uniformly weak either way. This is a different
    mathematical test than direct partial-picking, so it's a real second
    opinion rather than restating the same evidence.
    """

    @staticmethod
    def analyze(audio_slice: np.ndarray, sample_rate: int) -> float:
        """Returns a harmonic_ratio in [0, 1] - higher means more harmonic/tonal."""
        if audio_slice is None or len(audio_slice) < 256:
            return 0.5

        segment = audio_slice[:int(SPECTRAL_INVERSION_WINDOW_MS / 1000 * sample_rate)]
        if len(segment) < 256:
            segment = audio_slice
        segment = segment * np.hanning(len(segment))

        spectrum = np.abs(np.fft.rfft(segment, n=FFT_SIZE))
        if len(spectrum) < 10 or np.max(spectrum) <= 0:
            return 0.5

        _, normal_strength = harmonic_product_spectrum(spectrum)
        _, inverted_strength = harmonic_product_spectrum(spectrum[::-1])

        max_possible = np.max(spectrum) * HPS_NUM_HARMONICS
        normal_ratio = normal_strength / max_possible
        inverted_ratio = inverted_strength / max_possible

        return float(max(0.0, min(1.0, (normal_ratio + inverted_ratio) / 2)))


# =====================================================================
# Zero-Crossing Analyzer
# =====================================================================

class ZeroCrossingAnalyzer:
    """
    Analyzes zero-crossing rate to distinguish tonal from noisy signals.

    - Tonal sounds have low, stable ZCR
    - Noise has high ZCR throughout
    - Percussive attacks have high ZCR at onset that settles quickly
    """

    @staticmethod
    def analyze(audio_slice: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        if audio_slice is None or len(audio_slice) < sample_rate * 0.005:
            return {"verdict": "insufficient", "noise_ratio": 0.5, "onset_zcr": 0.0}

        zcr = ZeroCrossingAnalyzer._compute_zcr(audio_slice, sample_rate)
        if len(zcr) < 3:
            return {"verdict": "insufficient", "noise_ratio": 0.5, "onset_zcr": 0.0}

        onset_zcr = float(zcr[0])
        sustained_zcr = float(np.mean(zcr[1:])) if len(zcr) > 1 else onset_zcr

        if onset_zcr > ZERO_CROSSING_MAX and sustained_zcr < ZERO_CROSSING_MAX * 0.6:
            verdict, noise_ratio = "percussive", 0.1
        elif onset_zcr > ZERO_CROSSING_MAX and sustained_zcr > ZERO_CROSSING_MAX:
            verdict, noise_ratio = "noise", 0.9
        elif onset_zcr < ZERO_CROSSING_MAX and sustained_zcr < ZERO_CROSSING_MAX:
            verdict, noise_ratio = "tonal", 0.0
        else:
            verdict, noise_ratio = "uncertain", 0.3

        return {
            "onset_zcr": onset_zcr,
            "sustained_zcr": sustained_zcr,
            "verdict": verdict,
            "noise_ratio": noise_ratio,
        }

    @staticmethod
    def _compute_zcr(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if LIBROSA_AVAILABLE:
            return librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]

        hop, frame_size = HOP_LENGTH, FFT_SIZE
        n_frames = max(0, (len(audio) - frame_size) // hop + 1)
        zcr = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + frame_size]
            zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / 2 / len(frame)
        return zcr


# =====================================================================
# Spectral Flatness Analyzer
# =====================================================================

class SpectralFlatnessAnalyzer:
    """
    Spectral flatness (geometric mean / arithmetic mean of the spectrum).
    Tonal signals have low flatness (energy concentrated in peaks).
    Noise has high flatness (energy spread uniformly).
    """

    @staticmethod
    def analyze(audio_slice: np.ndarray, sample_rate: int) -> float:
        if audio_slice is None or len(audio_slice) < sample_rate * 0.01:
            return 0.5

        if LIBROSA_AVAILABLE:
            stft = librosa.stft(audio_slice, n_fft=FFT_SIZE, hop_length=HOP_LENGTH)
            flatness = librosa.feature.spectral_flatness(S=np.abs(stft))
            return float(np.mean(flatness))

        fft = np.fft.rfft(audio_slice, n=FFT_SIZE)
        magnitude = np.abs(fft)
        magnitude = magnitude[magnitude > 1e-8]
        if len(magnitude) == 0:
            return 0.5

        geometric = np.exp(np.mean(np.log(magnitude)))
        arithmetic = np.mean(magnitude)
        return float(geometric / arithmetic) if arithmetic > 0 else 0.5


# =====================================================================
# FeatureBundle-Based Analyzer (secondary path - no raw audio needed)
# =====================================================================

class FeatureBundleAnalyzer:
    """
    Harmonic analysis using a pre-computed FeatureBundle's harmonic_network
    evidence. Used when a raw audio slice isn't available (or as a cheap
    O(1) lookup alternative to re-running STFT per note).
    """

    def __init__(self, feature_bundle: FeatureBundle):
        self.bundle = feature_bundle

    def analyze_note(self, note: NoteEvent) -> Optional[HarmonicSeries]:
        if not self.bundle.has_evidence(EvidenceType.HARMONIC_NETWORK):
            return None

        harmonic_network = self.bundle.harmonic_network
        if harmonic_network is None or not note.fundamental_freq_hz:
            return None

        time_ms = note.get_active_start_ms()
        time_idx = int(np.argmin(np.abs(self.bundle.time_ms - time_ms)))
        time_idx = min(time_idx, harmonic_network.shape[1] - 1) if harmonic_network.ndim == 2 else 0

        harmonic_data = harmonic_network[:, time_idx] if harmonic_network.ndim == 2 else harmonic_network

        partials = []
        for harmonic in range(1, 9):
            if harmonic - 1 < len(harmonic_data):
                strength = harmonic_data[harmonic - 1]
                if strength > 0.1:
                    partials.append(PartialTrack(
                        partial_number=harmonic,
                        frequency_hz=note.fundamental_freq_hz * harmonic,
                        amplitude_db=20 * np.log10(strength + 1e-8),
                        confidence=float(strength)
                    ))

        if len(partials) < HARMONIC_SERIES_MIN_PARTIALS:
            return None

        present_numbers = {p.partial_number for p in partials}
        missing = [i for i in range(1, 9) if i not in present_numbers]

        return HarmonicSeries(
            fundamental_freq_hz=note.fundamental_freq_hz,
            fundamental_confidence=len(partials) / 8,
            partials=partials,
            missing_partials=missing,
            inharmonicity=0.0,  # Not computable from precomputed evidence
        )


# =====================================================================
# Schoenberg Mirror - Main Class
# =====================================================================

class SchoenbergMirror:
    """
    The Schoenberg Mirror - Harmonic series auditor and epistemic veto authority.

    This agent does not transcribe. It audits existing pitch detections.

    A note passes if:
        1. It has a detectable harmonic series (tonal sounds)
        2. OR it's percussive (drums, percussion) - PERCUSSION verdict
        3. OR it's ambiguous but not clearly false - UNCERTAIN verdict

    A note is vetoed if:
        1. It has no harmonic structure AND isn't clearly percussive
        2. Zero-crossing profile indicates persistent noise
        3. Independent witnesses (retrograde symmetry, spectral inversion)
           disagree with the primary call strongly enough to be unconvincing
    """

    def __init__(
            self,
            music_box: Optional[MusicBoxProtocol] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
    ):
        self._music_box = music_box
        self._status_reporter = status_reporter
        self._harmonic_analyzer = HarmonicSeriesAnalyzer()
        self._retrograde_analyzer = RetrogradeSymmetryAnalyzer()
        self._spectral_inversion_analyzer = SpectralInversionAnalyzer()
        self._zcr_analyzer = ZeroCrossingAnalyzer()
        self._flatness_analyzer = SpectralFlatnessAnalyzer()

    # =================================================================
    # Primary API
    # =================================================================

    def audit_note(
            self,
            note: NoteEvent,
            audio_contract: Optional[AudioContract] = None,
            feature_bundle: Optional[FeatureBundle] = None,
            context: Optional[AudioContext] = None
    ) -> SchoenbergResult:
        """
        Audit a single note for harmonic legitimacy.

        Args:
            audio_contract: Canonical audio source (preferred - enables the
                full STFT/retrograde/spectral-inversion analysis)
            feature_bundle: Precomputed evidence, used as a fallback when
                no raw audio slice is available
            context: Sample-rate fallback if neither of the above is given
        """
        audio_slice = None
        sample_rate = 44100

        if audio_contract is not None:
            start_ms = note.get_active_start_ms()
            end_ms = min(note.get_active_end_ms(), audio_contract.duration_ms)
            if end_ms > start_ms:
                start_idx = audio_contract.ms_to_index(start_ms)
                end_idx = audio_contract.ms_to_index(end_ms)
                audio_slice = audio_contract.samples[start_idx:end_idx]
            sample_rate = audio_contract.sample_rate
        elif context is not None:
            sample_rate = context.working_sample_rate

        # Zero-crossing profile (percussion/noise/tonal signal 1). Without a
        # raw audio slice we can't split onset-vs-sustained ZCR (so we can't
        # detect "percussive"), but the note's own precomputed
        # zero_crossing_rate is still real evidence and shouldn't be
        # discarded - Scribe's validation gate runs exactly this way (notes
        # only, no audio) and needs this to still catch persistent noise.
        if audio_slice is not None and len(audio_slice) > 0:
            zcr_profile = self._zcr_analyzer.analyze(audio_slice, sample_rate)
        elif note.zero_crossing_rate > ZERO_CROSSING_MAX:
            zcr_profile = {"verdict": "noise", "noise_ratio": 0.9, "onset_zcr": note.zero_crossing_rate}
        elif note.zero_crossing_rate < ZERO_CROSSING_MAX * 0.6:
            zcr_profile = {"verdict": "tonal", "noise_ratio": 0.0, "onset_zcr": note.zero_crossing_rate}
        else:
            zcr_profile = {"verdict": "uncertain", "noise_ratio": 0.3, "onset_zcr": note.zero_crossing_rate}

        # Quick veto for persistent noise - no point running the rest
        if zcr_profile.get("verdict") == "noise":
            return SchoenbergResult(
                verdict=SchoenbergVerdict.NOISE,
                harmonic_series=None,
                zero_crossing_rate=zcr_profile.get("onset_zcr", 0.0),
                spectral_flatness=SPECTRAL_FLATNESS_NOISE_THRESHOLD,
                reason=f"Persistent noise: ZCR {zcr_profile.get('onset_zcr', 0):.3f} > {ZERO_CROSSING_MAX}"
            )

        # Retrograde symmetry profile (percussion/tonal signal 2, corroborating)
        retrograde_profile = {"similarity": 0.5, "verdict_hint": "uncertain"}
        if audio_slice is not None and len(audio_slice) > 0:
            retrograde_profile = self._retrograde_analyzer.analyze(audio_slice, sample_rate)

        # Harmonic series: prefer direct STFT analysis of the raw slice;
        # fall back to precomputed FeatureBundle evidence if no slice exists;
        # fall back further to a note's own precomputed
        # harmonic_series_match_ratio (set upstream by harmonic_intelligence)
        # when neither audio nor a feature bundle is available at all - this
        # is the situation Scribe's validation gate is usually in (it only
        # has NoteEvent objects, not raw audio).
        harmonic_series = None
        if audio_slice is not None and note.fundamental_freq_hz:
            self._harmonic_analyzer.reset()
            harmonic_series = self._harmonic_analyzer.analyze(audio_slice, note.fundamental_freq_hz, sample_rate)
        elif feature_bundle is not None and note.fundamental_freq_hz:
            harmonic_series = FeatureBundleAnalyzer(feature_bundle).analyze_note(note)
        elif note.fundamental_freq_hz and getattr(note, 'harmonic_series_match_ratio', None) is not None:
            harmonic_series = HarmonicSeries(
                fundamental_freq_hz=note.fundamental_freq_hz,
                fundamental_confidence=note.harmonic_series_match_ratio,
                partials=[],
                missing_partials=[],
                inharmonicity=0.0,
            )

        # Spectral inversion profile (tonal signal 2, corroborating)
        spectral_inversion_ratio = 0.5
        if audio_slice is not None and len(audio_slice) > 0:
            spectral_inversion_ratio = self._spectral_inversion_analyzer.analyze(audio_slice, sample_rate)

        verdict, reason = self._determine_verdict(
            harmonic_series, zcr_profile, retrograde_profile, spectral_inversion_ratio, note
        )

        spectral_flatness = 0.5
        if audio_slice is not None and len(audio_slice) > 0:
            spectral_flatness = self._flatness_analyzer.analyze(audio_slice, sample_rate)

        result = SchoenbergResult(
            verdict=verdict,
            harmonic_series=harmonic_series,
            zero_crossing_rate=zcr_profile.get("onset_zcr", note.zero_crossing_rate),
            spectral_flatness=spectral_flatness,
            reason=reason,
        )

        if self._music_box:
            self._music_box.log_decision(
                stage_name="schoenberg_mirror",
                decision_type="pitch_legitimacy_audit",
                before_state={"pitch": note.pitch, "confidence": note.confidence},
                after_state={"verdict": verdict.value, "reason": reason},
                reasoning=reason,
                reversible=True,
            )

        return result

    def audit_batch(
            self,
            notes: List[NoteEvent],
            audio_contract: Optional[AudioContract] = None,
            feature_bundle: Optional[FeatureBundle] = None,
            context: Optional[AudioContext] = None
    ) -> List[SchoenbergResult]:
        """Audit multiple notes. Provide audio_contract for full-fidelity analysis."""
        return [self.audit_note(note, audio_contract, feature_bundle, context) for note in notes]

    # =================================================================
    # Veto Integration
    # =================================================================

    def should_veto(self, result: SchoenbergResult, confidence_threshold: float = 0.25) -> bool:
        """
        Law of Epistemic Veto: any qualified witness can declare falsehood.
        The Schoenberg Mirror is a qualified witness.
        """
        return result.verdict in (SchoenbergVerdict.HALLUCINATION, SchoenbergVerdict.NOISE)

    def adjust_confidence(self, note: NoteEvent, result: SchoenbergResult) -> float:
        """
        Tonal and percussion notes keep their confidence. Uncertain notes
        (including witness disagreement) get a 25% haircut. Noise/
        hallucination should be vetoed outright (confidence -> 0).
        """
        if result.verdict in (SchoenbergVerdict.TONAL, SchoenbergVerdict.PERCUSSION):
            return note.confidence
        if result.verdict == SchoenbergVerdict.UNCERTAIN:
            return note.confidence * 0.75
        return 0.0

    # =================================================================
    # Internal Methods
    # =================================================================

    def _determine_verdict(
            self,
            harmonic_series: Optional[HarmonicSeries],
            zcr_profile: Dict[str, Any],
            retrograde_profile: Dict[str, Any],
            spectral_inversion_ratio: float,
            note: NoteEvent,
    ) -> Tuple[SchoenbergVerdict, str]:
        """
        Determine verdict from all available witnesses. Two independent
        witnesses agreeing raises confidence in a call; disagreeing
        downgrades it to UNCERTAIN rather than picking a side.
        """
        zcr_says_percussive = zcr_profile.get("verdict") == "percussive"
        retrograde_hint = retrograde_profile.get("verdict_hint", "uncertain")

        # Percussion: ZCR onset/decay pattern is the primary signal;
        # retrograde symmetry corroborates or contradicts it.
        if zcr_says_percussive:
            if retrograde_hint == "percussive":
                return SchoenbergVerdict.PERCUSSION, (
                    "Percussive attack confirmed by two independent witnesses "
                    f"(ZCR decay pattern, retrograde asymmetry={retrograde_profile['similarity']:.2f})"
                )
            if retrograde_hint == "tonal":
                return SchoenbergVerdict.UNCERTAIN, (
                    f"Witnesses disagree: ZCR suggests percussive but retrograde envelope is "
                    f"symmetric (similarity={retrograde_profile['similarity']:.2f})"
                )
            return SchoenbergVerdict.PERCUSSION, "Percussive attack with tonal decay (ZCR pattern)"

        # Harmonic series present: corroborate with spectral inversion HPS.
        if harmonic_series is not None:
            match_ok = harmonic_series.fundamental_confidence >= HARMONIC_SERIES_MATCH_THRESHOLD
            inharmonicity_ok = harmonic_series.inharmonicity < MAX_INHARMONICITY

            if match_ok and inharmonicity_ok:
                if spectral_inversion_ratio >= 0.4:
                    return SchoenbergVerdict.TONAL, (
                        f"Valid harmonic series ({len(harmonic_series.partials)} partials, "
                        f"confidence {harmonic_series.fundamental_confidence:.1%}), corroborated by "
                        f"spectral inversion HPS ratio {spectral_inversion_ratio:.2f}"
                    )
                return SchoenbergVerdict.UNCERTAIN, (
                    f"Harmonic partials matched ({harmonic_series.fundamental_confidence:.1%}) but "
                    f"spectral inversion HPS disagrees (ratio={spectral_inversion_ratio:.2f})"
                )
            if not inharmonicity_ok:
                return SchoenbergVerdict.UNCERTAIN, f"High inharmonicity: {harmonic_series.inharmonicity:.3f}"
            return SchoenbergVerdict.UNCERTAIN, f"Partial match only: {harmonic_series.fundamental_confidence:.1%}"

        # No harmonic series detected at all.
        if note.source in (SourceType.RHYTHM, SourceType.DRUM_INTELLIGENCE):
            return SchoenbergVerdict.PERCUSSION, "Rhythm/drum source - harmonic analysis not applicable"

        if retrograde_hint == "percussive":
            return SchoenbergVerdict.PERCUSSION, (
                f"No harmonic series, but retrograde envelope is asymmetric "
                f"(similarity={retrograde_profile['similarity']:.2f}) - likely an untagged percussive hit"
            )

        if note.confidence < 0.5:
            return SchoenbergVerdict.HALLUCINATION, (
                f"Low confidence ({note.confidence:.1%}) without harmonic support"
            )

        return SchoenbergVerdict.NOISE, "No harmonic series detected"

    # =================================================================
    # Utility Methods
    # =================================================================

    def get_confidence_components(self, result: SchoenbergResult, note: NoteEvent) -> ConfidenceComponents:
        """Forensic confidence breakdown for Scribe."""
        harmonic_confidence = result.harmonic_series.fundamental_confidence if result.harmonic_series else 0.0

        if result.zero_crossing_rate < 0.1:
            zcr_confidence = 1.0
        elif result.zero_crossing_rate < 0.2:
            zcr_confidence = 0.8
        elif result.zero_crossing_rate < 0.35:
            zcr_confidence = 0.5
        else:
            zcr_confidence = 0.2

        if result.spectral_flatness < 0.3:
            flatness_confidence = 1.0
        elif result.spectral_flatness < 0.5:
            flatness_confidence = 0.7
        elif result.spectral_flatness < 0.7:
            flatness_confidence = 0.4
        else:
            flatness_confidence = 0.1

        verdict_confidence = {
            SchoenbergVerdict.TONAL: VERDICT_BASE_CONFIDENCE["harmonic_series_detected"],
            SchoenbergVerdict.PERCUSSION: VERDICT_BASE_CONFIDENCE["impulsive_no_harmonics"] + 0.6,
            SchoenbergVerdict.UNCERTAIN: VERDICT_BASE_CONFIDENCE["borderline_case"],
            SchoenbergVerdict.NOISE: VERDICT_BASE_CONFIDENCE["no_harmonic_structure"] + 0.1,
            SchoenbergVerdict.HALLUCINATION: VERDICT_BASE_CONFIDENCE["false_positive_detected"],
        }.get(result.verdict, 0.5)

        # Ensemble agreement: how many independent witnesses (ZCR-implied
        # tonal/percussive call, harmonic-series presence, and the verdict
        # itself all pointing the same way) actually agree with the final
        # call, rather than a hardcoded placeholder.
        witness_votes = []
        is_tonal_verdict = result.verdict == SchoenbergVerdict.TONAL
        is_percussion_verdict = result.verdict == SchoenbergVerdict.PERCUSSION
        if result.zero_crossing_rate < ZERO_CROSSING_MAX:
            witness_votes.append(is_tonal_verdict)
        else:
            witness_votes.append(is_percussion_verdict or result.verdict == SchoenbergVerdict.NOISE)
        if result.harmonic_series is not None:
            witness_votes.append(is_tonal_verdict)
        if result.spectral_flatness < SPECTRAL_FLATNESS_TONAL_THRESHOLD:
            witness_votes.append(is_tonal_verdict)
        elif result.spectral_flatness > SPECTRAL_FLATNESS_NOISE_THRESHOLD:
            witness_votes.append(not is_tonal_verdict)

        ensemble_agreement = (sum(1 for v in witness_votes if v) / len(witness_votes)) if witness_votes else 0.5

        return ConfidenceComponents(
            transient_strength=zcr_confidence,
            spectral_match=harmonic_confidence,
            temporal_consistency=flatness_confidence,
            model_confidence=verdict_confidence,
            ensemble_agreement=ensemble_agreement,
        )


# =================================================================
# Convenience Functions
# =================================================================

def create_schoenberg_mirror(
        music_box: Optional[MusicBoxProtocol] = None,
        status_reporter: Optional[StatusReporterProtocol] = None,
) -> SchoenbergMirror:
    """Create a configured Schoenberg Mirror instance."""
    return SchoenbergMirror(music_box=music_box, status_reporter=status_reporter)


def quick_schoenberg_test(audio_path: str, notes: List[NoteEvent]) -> None:
    """Quick standalone test: audits a handful of notes against a real audio file."""
    from core.feature_bundle import create_feature_bundle

    print("\n" + "=" * 70)
    print("SCHOENBERG MIRROR - Quick Test")
    print("=" * 70)

    print(f"\n1. Loading audio contract from {audio_path}")
    audio, sr = librosa.load(audio_path, mono=True, sr=None)
    contract = AcousticIntelligence.create_contract(audio, sr, "schoenberg_test", force_mono=True)
    print(f"   Contract: {contract.duration_ms:.0f}ms @ {contract.sample_rate}Hz")

    print("\n2. Creating FeatureBundle")
    bundle = create_feature_bundle(audio, sr, precompute=[EvidenceType.HARMONIC_NETWORK])
    print(f"   Bundle available: {bundle.get_available_evidence()}")

    mirror = create_schoenberg_mirror()

    print(f"\n3. Auditing {len(notes)} notes")
    for i, note in enumerate(notes[:5]):
        result = mirror.audit_note(note, contract, bundle)
        print(f"\n   Note {i + 1}: MIDI {note.pitch}")
        print(f"      Verdict: {result.verdict.value}")
        print(f"      Reason: {result.reason[:80]}")
        print(f"      ZCR: {result.zero_crossing_rate:.3f}  Flatness: {result.spectral_flatness:.3f}")
        if result.harmonic_series:
            print(f"      Partials: {len(result.harmonic_series.partials)}  "
                  f"Match: {result.harmonic_series.fundamental_confidence:.1%}")
        print(f"      {'VETO' if mirror.should_veto(result) else 'PASS'}")

    print("\n" + "=" * 70)
    print("Test complete.")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    mock_notes = [
        NoteEvent(
            pitch=60, start_ms=1000, end_ms=1500, velocity=80,
            confidence=0.9, zero_crossing_rate=0.05,
            source=SourceType.PITCH, fundamental_freq_hz=261.63
        ),
        NoteEvent(
            pitch=64, start_ms=2000, end_ms=2500, velocity=90,
            confidence=0.85, zero_crossing_rate=0.04,
            source=SourceType.PITCH, fundamental_freq_hz=329.63
        ),
        NoteEvent(
            pitch=36, start_ms=3000, end_ms=3500, velocity=100,
            confidence=0.95, zero_crossing_rate=0.15,
            source=SourceType.DRUM_INTELLIGENCE, fundamental_freq_hz=65.41
        ),
    ]

    if len(sys.argv) > 1:
        quick_schoenberg_test(sys.argv[1], mock_notes)
    else:
        print("Usage: python schoenberg_mirror.py <audio_file>")
