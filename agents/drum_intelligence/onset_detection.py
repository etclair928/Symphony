# =================================================================
# MODULE: agents/drum_intelligence/onset_detection.py
# VERSION: 6.0-draft
# DESCRIPTION: Raw onset witnesses - band-filtering, resampling, and the
# two onset-function families (flat spectral-flux, HFC) that feed
# classification.py. Produces float onset times, no drum-type guesses.
#
# REPLACES: agents/detection/drum_intelligence.py's SafeResampler,
# SimpleBandFilter, MultiBandOnsetDetector, and agents/detection/
# drum_coprocessor.py's HFCOnsetDetector, AdaptiveMedianThreshold.
#
# HFC + adaptive-median thresholding used to be reachable only through an
# optional external "coprocessor" object MultiBandOnsetDetector had to be
# handed and null-check before using. There was never a real scenario
# where HFC onset detection ran without the rest of this package present
# - it's native here now, selected per-band by config like any other
# onset function, not bolted on through an optional collaborator.
# =================================================================

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict

from .drum_types import DrumIntelligenceConfig
from core.constants import TARGET_SAMPLE_RATE

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy.signal import find_peaks, medfilt
    from scipy.ndimage import gaussian_filter1d, median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ========================================================================
# Safe Resampler
# ========================================================================

class SafeResampler:
    """Safe polyphase resampling - no aliasing."""

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio

        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        if SCIPY_AVAILABLE:
            gcd = np.gcd(orig_sr, target_sr)
            up = target_sr // gcd
            down = orig_sr // gcd
            from scipy.signal import resample_poly
            return resample_poly(audio, up, down)
        elif LIBROSA_AVAILABLE:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        else:
            raise RuntimeError("No resampling library available")


# ========================================================================
# Simple Band Filter (FFT-based, deterministic, no recursive state)
# ========================================================================

class SimpleBandFilter:
    """Simple FFT-based band filter - deterministic, no state."""

    @staticmethod
    def filter_band(audio: np.ndarray, sample_rate: int,
                    low_hz: float, high_hz: float) -> np.ndarray:
        if not SCIPY_AVAILABLE:
            return audio

        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        n = len(audio)
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
        fft = np.fft.rfft(audio)

        mask = (freqs >= low_hz) & (freqs <= high_hz)
        fft_filtered = fft * mask

        return np.fft.irfft(fft_filtered, n=n)


# ========================================================================
# HFC Onset Function
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

    def __init__(self, config: DrumIntelligenceConfig, sample_rate: int = TARGET_SAMPLE_RATE):
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

    Hugs the ambient level of the performance instead of one flat value:
    drops during a quiet ghost-note passage so those notes clear it,
    rises during a loud fill so acoustic ring/bleed doesn't.
    """

    def __init__(self, config: DrumIntelligenceConfig):
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
# Multi-Band Onset Detector
# ========================================================================

class MultiBandOnsetDetector:
    """
    Multi-band onset detection. Low/kick band always uses flat spectral-
    flux (onset_strength); mid/high bands use HFC + adaptive-median
    thresholding when config.hfc_enabled, falling back to spectral-flux
    only if HFC genuinely couldn't run (audio too short, curve failed).
    """

    def __init__(self, config: DrumIntelligenceConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate
        self._band_filter = SimpleBandFilter()
        self._hfc = HFCOnsetDetector(config, sample_rate)
        self._median_threshold = AdaptiveMedianThreshold(config)

    def detect_onsets(self, audio: np.ndarray) -> Dict[str, List[float]]:
        """Detect onsets per frequency band."""
        if not LIBROSA_AVAILABLE:
            return {'low': [], 'mid': [], 'high': []}

        if self.config.sanitize_input and np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            low_audio = self._band_filter.filter_band(
                audio, self.sample_rate, self.config.low_band_hz[0], self.config.low_band_hz[1]
            )
            mid_audio = self._band_filter.filter_band(
                audio, self.sample_rate, self.config.mid_band_hz[0], self.config.mid_band_hz[1]
            )
            high_audio = self._band_filter.filter_band(
                audio, self.sample_rate, self.config.high_band_hz[0], self.config.high_band_hz[1]
            )

            return {
                'low': self._detect_onsets_in_band(low_audio, band='low'),
                'mid': self._detect_onsets_in_band(mid_audio, band='mid'),
                'high': self._detect_onsets_in_band(high_audio, band='high'),
            }
        except Exception:
            return {'low': [], 'mid': [], 'high': []}

    def _detect_onsets_in_band(self, audio: np.ndarray, band: str = 'low') -> List[float]:
        """Detect onsets in a single frequency band."""
        if not LIBROSA_AVAILABLE or len(audio) == 0:
            return []

        if band in ('mid', 'high') and self.config.hfc_enabled:
            hfc_onsets = self._detect_onsets_hfc(audio)
            if hfc_onsets is not None:
                return hfc_onsets
            # else: HFC genuinely couldn't run - fall through below

        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        if np.all(audio == 0):
            return []

        try:
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=self.sample_rate,
                hop_length=256
            )

            if len(onset_env) == 0:
                return []

            # A global percentile or absolute-floor threshold can't
            # reliably separate real transients from quiet background
            # noise/bleed here: onset_strength operates on a
            # log-compressed spectrogram, so quiet noise can produce
            # frame-to-frame flux comparable in scale to a genuine
            # transient (verified directly - a pure noise floor's mean
            # flux was only ~4x smaller than a real transient's peak, not
            # the orders-of-magnitude gap a track-wide threshold needs).
            # librosa's own adaptive local-average peak-picking (a
            # candidate must exceed ITS OWN neighborhood's average by
            # onset_delta, not some global statistic) is the standard,
            # hyperparameter-tuned approach for exactly this problem.
            onset_times_sec = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=256,
                units="time",
                delta=self.config.onset_delta,
                wait=self.config.onset_wait_frames,
            )
            return [float(t) * 1000.0 for t in onset_times_sec]
        except Exception:
            return []

    def _detect_onsets_hfc(self, audio: np.ndarray) -> Optional[List[float]]:
        """
        HFC + adaptive-median onset detection, for mid/high bands only.

        Returns None (not []) when HFC genuinely couldn't run (audio too
        short, curve computation failed) so the caller falls back to the
        standard onset-strength path - but returns [] when HFC ran fine
        and legitimately found no onsets in this band, so a quiet/silent
        band doesn't get double-checked by both methods.
        """
        curve = self._hfc.compute_hfc_curve(audio)
        if len(curve) == 0:
            return None

        if self.config.median_threshold_enabled:
            peak_frames = self._median_threshold.find_peaks_adaptive(curve)
        elif SCIPY_AVAILABLE:
            peak_frames = [int(p) for p in find_peaks(curve)[0]]
        else:
            peak_frames = []

        hop = self.config.hfc_hop_length
        return [float(f) * hop / self.sample_rate * 1000.0 for f in peak_frames]
