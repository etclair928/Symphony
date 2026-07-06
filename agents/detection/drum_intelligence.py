# =================================================================
# MODULE: agents/detection/drum_intelligence.py
# DESCRIPTION: Complete drum and percussion intelligence for Grimlock 5.0.
#
# VERSION: 5.6.1 (FIXED: Input sanitization, no inline filtering)
# UPDATED: 2026-05-15
#
# CRITICAL FIXES:
#   1. Input sanitization (removes inf/nan before processing)
#   2. Pre-computed multi-band filtering (moved to FeatureBundle)
#   3. No scipy.signal.filtfilt in detection path
#   4. Proper array validation before librosa calls
# =================================================================

import time
import gc
import hashlib
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from collections import defaultdict, deque

# Core imports - ONLY from bedrock
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    DrumType, DrumEvent, DrumKitMapping, WitnessVote, DecisionType
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    DRUM_CONFIDENCE_MIN,
    DRUM_GHOST_NOTE_VELOCITY_MAX,
    DRUM_FLAM_DETECTION_WINDOW_MS,
    DRUM_ROLL_DETECTION_MIN_HITS,
    DRUM_ROLL_MAX_INTERVAL_MS,
    SPICE_MODEL_SIZE,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    DetectionAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)

# Optional imports with safe fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy.signal import find_peaks, medfilt
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SPICE_AVAILABLE = False
try:
    from spice import SPICE, SPICEConfig
    SPICE_AVAILABLE = True
except ImportError:
    pass


# ========================================================================
# Enums and Configuration
# ========================================================================

class DrumArticulation(str, Enum):
    NORMAL = "normal"
    GHOST = "ghost"
    FLAM = "flam"
    DRAG = "drag"
    ROLL = "roll"
    RIMSHOT = "rimshot"
    CROSS_STICK = "cross_stick"
    ACCENT = "accent"


class DrumMicrotiming(str, Enum):
    ON_TOP = "on_top"
    LAID_BACK = "laid_back"
    POCKET = "pocket"
    RUSHING = "rushing"
    DRAGGING = "dragging"


class DetectionPass(str, Enum):
    RAW = "raw"
    CLEANED = "cleaned"
    CLASSIFIED = "classified"
    ARTICULATED = "articulated"
    PATTERN = "pattern"
    FORENSIC = "forensic"
    EXPORT = "export"


@dataclass(frozen=True)
class ConfidenceComponents:
    """Forensic confidence breakdown - immutable."""
    transient_strength: float = 0.0
    spectral_match: float = 0.0
    temporal_consistency: float = 0.0
    model_confidence: float = 0.0
    ensemble_agreement: float = 0.0

    def total(self) -> float:
        return float(np.mean([
            self.transient_strength,
            self.spectral_match,
            self.temporal_consistency,
            self.model_confidence,
            self.ensemble_agreement
        ]))


@dataclass(frozen=True)
class ForensicDrumEvent:
    """Immutable forensic drum event with complete provenance."""
    drum_type: DrumType
    start_ms: float
    end_ms: float
    velocity: int
    confidence: ConfidenceComponents
    articulation: DrumArticulation = DrumArticulation.NORMAL
    is_rimshot: bool = False
    is_cross_stick: bool = False
    flam_pair_id: Optional[str] = None
    spectral_centroid_hz: float = 0.0
    spectral_rolloff_hz: float = 0.0
    transient_sharpness: float = 0.0
    microtiming_offset_ms: float = 0.0
    microtiming_category: Optional[DrumMicrotiming] = None
    detection_pass: DetectionPass = DetectionPass.RAW
    source_model: str = "unknown"
    reasoning_chain: Tuple[str, ...] = field(default_factory=tuple)
    witness_votes: Tuple[WitnessVote, ...] = field(default_factory=tuple)

    def to_note_event(self, kit_mapping: DrumKitMapping) -> NoteEvent:
        """Convert to standard NoteEvent for pipeline."""
        midi_note = kit_mapping.drum_to_midi.get(self.drum_type, 36)
        return NoteEvent(
            pitch=midi_note,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            velocity=self.velocity,
            confidence=self.confidence.total(),
            zero_crossing_rate=self.transient_sharpness,
            source=SourceType.DRUM_INTELLIGENCE,
            reasoning_chain=list(self.reasoning_chain),
            consensus_votes=list(self.witness_votes)
        )


@dataclass
class DrumDetectionConfig:
    """Configuration for drum detection."""
    spice_model_size: str = SPICE_MODEL_SIZE
    spice_hop_length_ms: int = 10
    spice_onset_threshold: float = 0.5
    spice_adaptive_threshold: bool = True
    min_confidence: float = DRUM_CONFIDENCE_MIN
    ghost_note_max_velocity: int = DRUM_GHOST_NOTE_VELOCITY_MAX
    flam_window_ms: int = DRUM_FLAM_DETECTION_WINDOW_MS
    roll_min_hits: int = DRUM_ROLL_DETECTION_MIN_HITS
    roll_max_interval_ms: int = DRUM_ROLL_MAX_INTERVAL_MS
    low_band_hz: Tuple[float, float] = (20, 150)
    mid_band_hz: Tuple[float, float] = (150, 4000)
    high_band_hz: Tuple[float, float] = (4000, 20000)
    max_retained_events: int = 10000
    max_retained_patterns: int = 100
    tempo_cache_ttl_seconds: int = 3600
    silence_threshold_db: float = -40.0
    min_energy_window_ms: float = 100.0
    detect_patterns: bool = True
    pattern_hash_bits: int = 16
    use_ensemble: bool = True
    ensemble_agreement_threshold: float = 0.6
    # NEW: Sanitize input before processing
    sanitize_input: bool = True

    # Onset detection tuning (MultiBandOnsetDetector). A pure percentile
    # threshold always finds "peaks" regardless of whether a band has
    # any real transient content: onset_strength operates on a
    # log-compressed spectrogram, so quiet background noise/bleed can
    # produce frame-to-frame flux comparable in scale to a genuine
    # transient - neither a global percentile nor a floor relative to
    # the band's own max can reliably tell them apart. onset_delta/
    # onset_wait_frames instead drive librosa's own adaptive
    # local-average peak-picking (librosa.onset.onset_detect), the
    # standard, hyperparameter-optimized approach for exactly this
    # problem - a candidate must exceed its LOCAL neighborhood's average
    # by onset_delta, not some track-wide statistic.
    onset_delta: float = 0.12
    onset_wait_frames: int = 10
    # Same-drum-type dedup window - widened from a very tight 12ms
    # (well within STFT frame-to-frame jitter/leakage range, so it
    # barely caught anything) to 20ms, comfortably below the fastest
    # physically-realistic same-drum repeat rate.
    dedup_window_ms: float = 20.0


# ========================================================================
# Safe Resampler
# ========================================================================

class SafeResampler:
    """Safe polyphase resampling - no aliasing."""

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample using polyphase filter."""
        if orig_sr == target_sr:
            return audio

        # Sanitize first
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
# Simple Band Filter (using FFT, not IIR)
# ========================================================================

class SimpleBandFilter:
    """Simple FFT-based band filter - deterministic, no state."""

    @staticmethod
    def filter_band(audio: np.ndarray, sample_rate: int,
                    low_hz: float, high_hz: float) -> np.ndarray:
        """FFT-based bandpass filter - no recursive state."""
        if not SCIPY_AVAILABLE:
            return audio

        # Sanitize
        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        n = len(audio)
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
        fft = np.fft.rfft(audio)

        # Apply band mask
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        fft_filtered = fft * mask

        return np.fft.irfft(fft_filtered, n=n)


# ========================================================================
# Multi-Band Onset Detector (No inline filtering)
# ========================================================================

class MultiBandOnsetDetector:
    """
    Multi-band onset detection - uses pre-computed or simple FFT bands.
    """

    def __init__(self, config: DrumDetectionConfig, sample_rate: int = TARGET_SAMPLE_RATE,
                coprocessor: Optional[Any] = None):
        self.config = config
        self.sample_rate = sample_rate
        self._band_filter = SimpleBandFilter()
        # Optional DrumCoprocessor (agents/detection/drum_coprocessor.py).
        # When present, mid/high bands route through its HFC + adaptive-
        # median onset detection instead of flat spectral-flux energy -
        # low/kick band deliberately keeps the existing approach, since
        # HFC's bin-index weighting would suppress exactly the low-
        # frequency content that defines a kick.
        self._coprocessor = coprocessor

    def detect_onsets(self, audio: np.ndarray) -> Dict[str, List[float]]:
        """Detect onsets per frequency band."""
        if not LIBROSA_AVAILABLE:
            return {'low': [], 'mid': [], 'high': []}

        # Sanitize input
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

            low_onsets = self._detect_onsets_in_band(low_audio, band='low')
            mid_onsets = self._detect_onsets_in_band(mid_audio, band='mid')
            high_onsets = self._detect_onsets_in_band(high_audio, band='high')

            return {
                'low': low_onsets,
                'mid': mid_onsets,
                'high': high_onsets
            }
        except Exception as e:
            return {'low': [], 'mid': [], 'high': []}

    def _detect_onsets_in_band(self, audio: np.ndarray, band: str = 'low') -> List[float]:
        """Detect onsets in a single frequency band."""
        if not LIBROSA_AVAILABLE or len(audio) == 0:
            return []

        if band in ('mid', 'high') and self._coprocessor is not None and self._coprocessor.config.hfc_enabled:
            try:
                hfc_onsets = self._coprocessor.detect_onsets_hfc(audio)
                if hfc_onsets is not None:
                    return hfc_onsets
            except Exception:
                pass  # fall through to the standard path below

        # Validate audio before passing to librosa
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
            # flux was only ~4x smaller than a real transient's peak,
            # not the orders-of-magnitude gap a track-wide threshold
            # needs to discriminate on). librosa's own adaptive
            # local-average peak-picking (a candidate must exceed ITS
            # OWN neighborhood's average by onset_delta, not some
            # global statistic) is the standard, hyperparameter-tuned
            # approach for exactly this problem.
            onset_times_sec = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=256,
                units="time",
                delta=self.config.onset_delta,
                wait=self.config.onset_wait_frames,
            )
            return [float(t) * 1000.0 for t in onset_times_sec]
        except Exception as e:
            return []


# ========================================================================
# Spectral Classifier
# ========================================================================

class SpectralDrumClassifier:
    """Spectral-based drum classification for fallback mode."""

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def classify(self, audio_slice: np.ndarray, onset_time_ms: float) -> DrumType:
        """Classify drum type from audio slice."""
        if not LIBROSA_AVAILABLE or len(audio_slice) < 512:
            return DrumType.PERCUSSION

        # Sanitize
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
            # buckets (median 3242Hz), which is exactly why measuring the
            # actual GM pitch distribution downstream found 0 kicks and
            # 85% cymbal-type hits despite kicks clearly being present in
            # the audio. Checking low-frequency content FIRST, independent
            # of whatever else is happening in the rest of the spectrum,
            # catches these masked kicks without needing to guess whether
            # a hit was "pure" kick or kick-plus-something-else.
            spectrum = np.abs(np.fft.rfft(audio_slice)) ** 2
            freqs = np.fft.rfftfreq(len(audio_slice), 1.0 / self.sample_rate)
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
                # stroke) has no branch here at all before this - its
                # bright, short "click" legitimately sits in this same
                # centroid range as hi-hat/ride, so every cross-stick hit
                # was previously forced into HI_HAT_CLOSED or RIDE with no
                # other option, regardless of how the boundaries below were
                # tuned. Verified directly on a real second-line snare
                # performance (heavy cross-stick use): mid-band onsets
                # clustered at centroid 1200-2500Hz (median ~1470Hz), crest
                # 3-8.35 (median ~4.3) - all previously landing as RIDE.
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
    """SPICE wrapper with adaptive threshold and forensic output."""

    def __init__(self, config: DrumDetectionConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate
        self._model = None
        self._model_load_time_ms = 0.0
        self._available = False
        self._init_model()

    def _init_model(self):
        """Lazy load SPICE model."""
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

        # Sanitize
        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        threshold = self.config.spice_onset_threshold
        if self.config.spice_adaptive_threshold:
            # Denser passages and faster tempos mean more overlapping
            # transients, so weak/spurious activations are MORE likely to
            # get mistaken for real onsets there - require more evidence
            # (raise the threshold), not less. This used to subtract,
            # making detection more permissive exactly where false
            # positives are most likely - directly contributing to
            # over-detected, too-busy drum parts in dense sections.
            threshold += density * 0.1
            threshold += ((tempo - 60) / 180) * 0.05
            threshold = max(0.2, min(0.8, threshold))

        resampler = SafeResampler()
        audio_22050 = resampler.resample(audio, self.sample_rate, 22050)

        try:
            activations = self._model.predict(audio_22050, threshold=threshold)
            return self._convert_activations(activations)
        except Exception:
            return []

    def _convert_activations(self, activations: Dict[str, np.ndarray]) -> List[ForensicDrumEvent]:
        """Convert SPICE activations to immutable forensic events."""
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
# Ensemble Detector
# ========================================================================

class EnsembleDrumDetector:
    """Ensemble detector combining SPICE, onset, and spectral classification."""

    def __init__(self, config: DrumDetectionConfig, sample_rate: int = TARGET_SAMPLE_RATE,
                coprocessor: Optional[Any] = None):
        self.config = config
        self.sample_rate = sample_rate
        self._spice = AdaptiveSPICEWrapper(config, sample_rate)
        # coprocessor (agents/detection/drum_coprocessor.py's DrumCoprocessor)
        # is optional and created lazily here if the caller doesn't supply
        # one, so EnsembleDrumDetector still works standalone.
        if coprocessor is None:
            from agents.detection.drum_coprocessor import create_drum_coprocessor
            coprocessor = create_drum_coprocessor(sample_rate)
        self._coprocessor = coprocessor
        self._onset = MultiBandOnsetDetector(config, sample_rate, coprocessor=self._coprocessor)
        self._spectral = SpectralDrumClassifier(sample_rate)

    def detect(self, audio: np.ndarray, density: float = 0.5, tempo: float = 120.0) -> List[ForensicDrumEvent]:
        """Detect drums using ensemble voting."""
        # Sanitize input
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
        if self.config.use_ensemble and self._coprocessor.config.nmf_enabled:
            try:
                band_ranges = {'low': self.config.low_band_hz, 'mid': self.config.mid_band_hz,
                              'high': self.config.high_band_hz}
                polyphony_events, contentions = self._coprocessor.decompose_polyphony(
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

    def _resolve_contentions(self, contentions: List[Tuple[ForensicDrumEvent, List[Any]]],
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
        Convert onset detections to forensic events.

        Previously guessed drum_type purely from which frequency band the
        onset was detected in (low->KICK, mid->SNARE, high->HI_HAT_CLOSED),
        with no other outcome possible - a tom, whichever band its
        fundamental happened to land in, could only ever come out as a
        kick or a snare, never a tom. SPICE (the ML-based classifier vote)
        isn't installed in this environment (SPICE_AVAILABLE=False), so
        this onset-band path is the ONLY vote source in practice, and
        SpectralDrumClassifier - built specifically to do real centroid/
        crest-factor classification - was instantiated in __init__ but
        never actually called anywhere. Now used here to classify each
        onset's own audio slice for real; the band guess is kept only as
        a fallback for slices too short to classify.

        Classifies from the SAME band-filtered audio the onset was found
        in, not the raw full mix. Verified on real full-kit recordings
        (drummer playing one featured piece over a continuing hi-hat/
        cymbal groove): classifying the raw mix let simultaneous cymbal
        bleed dominate the spectral centroid regardless of which band
        actually triggered the onset, so almost everything - even loud
        tom/snare hits - read as hi-hat/ride/crash. Band-filtering first
        removes exactly the bleed MultiBandOnsetDetector already isolated
        the onset away from.
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

                confidence = ConfidenceComponents(
                    transient_strength=0.7,
                    model_confidence=0.5,
                    temporal_consistency=0.6,
                    spectral_match=spectral_match,
                    ensemble_agreement=0.0
                )

                # A flat +150ms for every drum type regardless of how
                # fast it's actually being played was the real cause of
                # Scribe's phase-consistency gate flagging ~50% of
                # consecutive same-type drum hits as "incoherent overlap":
                # a hi-hat pattern at 8th/16th notes routinely has hits
                # under 150ms apart, so nearly every hit's artificially
                # long end_ms extended past the NEXT same-type hit's
                # start_ms regardless of real timing. Reuses
                # AdaptiveSPICEWrapper's per-type duration table via the
                # self._spice instance this class already holds - that
                # table was already correct but only ever wired into the
                # SPICE detection path (SPICE isn't installed in this
                # environment, so this onset-detector path is the one
                # that actually runs). Additionally caps it against the
                # next onset in this same band so a slow-decaying type
                # (crash=800ms) still can't overlap a genuinely fast
                # next hit.
                estimated_end = time_ms + self._spice._estimate_duration(drum_type)
                if onset_idx + 1 < len(sorted_onset_times):
                    # Cap exactly AT the next onset (a true zero gap),
                    # not a hair before it - leaving even a 1ms buffer
                    # here would land squarely in
                    # Scribe._check_phase_consistency's "0 < gap_ms < 5"
                    # near-zero-gap anomaly band and get flagged as
                    # incoherent on every single capped note, which is
                    # exactly backwards: a hit whose estimated decay is
                    # capped by the next hit's onset is behaving
                    # perfectly normally, not anomalously.
                    estimated_end = min(estimated_end, sorted_onset_times[onset_idx + 1])
                estimated_end = max(estimated_end, time_ms + 10.0)

                events.append(ForensicDrumEvent(
                    drum_type=drum_type,
                    start_ms=time_ms,
                    end_ms=estimated_end,
                    velocity=70,
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
        Previously grouped ALL votes into a 25ms time bucket regardless of
        drum_type and kept only the single highest-confidence event per
        bucket - meaning even if NMF successfully split a genuine
        simultaneous kick+tom into two separate events, this step would
        still collapse them back into one, discarding whichever had lower
        confidence. Grouping by (time_bucket, drum_type) instead lets
        different-type votes at the same instant coexist (real polyphony),
        while same-type votes from different sources still compete/merge
        into one event exactly as before (unchanged agreement-ratio
        behavior below).
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


# ========================================================================
# PatternHashDetector, MicrotimingAnalyzer, SilenceGater
# ========================================================================

class PatternHashDetector:
    """Hash-based pattern detection - O(n) complexity."""

    def __init__(self, config: DrumDetectionConfig, tempo: float = 120.0):
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


class MicrotimingAnalyzer:
    """Subdivision-aware microtiming analysis."""

    def __init__(self, config: DrumDetectionConfig, tempo: float = 120.0):
        self.config = config
        self.tempo = tempo
        self._grids = self._build_grids()

    def _build_grids(self) -> Dict[str, List[float]]:
        beat_ms = 60000 / self.tempo
        return {
            'quarter': [i * beat_ms for i in range(64)],
            'eighth': [i * beat_ms / 2 for i in range(128)],
            'triplet': [i * beat_ms / 3 for i in range(192)],
            'sixteenth': [i * beat_ms / 4 for i in range(256)]
        }

    def analyze(self, events: List[ForensicDrumEvent]) -> List[ForensicDrumEvent]:
        updated_events = []

        for event in events:
            best_grid = None
            best_deviation = float('inf')

            for grid_name, grid_times in self._grids.items():
                if not grid_times:
                    continue
                nearest = min(grid_times, key=lambda t: abs(t - event.start_ms))
                deviation = event.start_ms - nearest

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


class SilenceGater:
    """Detects and skips silent regions for efficiency."""

    def __init__(self, config: DrumDetectionConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate

    def get_active_regions(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        if not LIBROSA_AVAILABLE:
            return [(0, len(audio))]

        # Sanitize
        if np.any(~np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            hop_length = 512
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

            threshold_linear = 10 ** (self.config.silence_threshold_db / 20)
            active = rms > threshold_linear

            window_ms = self.config.min_energy_window_ms
            min_active_frames = int(window_ms * self.sample_rate / 1000 / hop_length)

            regions = []
            in_region = False
            start = 0

            for i, is_active in enumerate(active):
                if is_active and not in_region:
                    in_region = True
                    start = i * hop_length
                elif not is_active and in_region:
                    if (i * hop_length - start) > min_active_frames * hop_length:
                        regions.append((start, i * hop_length))
                    in_region = False

            if in_region:
                regions.append((start, len(audio)))

            return regions
        except Exception:
            return [(0, len(audio))]


# ========================================================================
# Main Drum Intelligence Agent
# ========================================================================

class DrumIntelligence:
    """Complete drum intelligence for Grimlock 5.0."""

    def __init__(
            self,
            config: Optional[DrumDetectionConfig] = None,
            drum_kit_mapping: Optional[DrumKitMapping] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        self._name = "drum_intelligence"
        self._source_type = SourceType.DRUM_INTELLIGENCE
        self._config = config or DrumDetectionConfig()
        self._kit_mapping = drum_kit_mapping or DrumKitMapping.gm_standard()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback
        self._detection_threshold = Confidence.LOW.value

        from agents.detection.drum_coprocessor import create_drum_coprocessor
        self._coprocessor = create_drum_coprocessor(TARGET_SAMPLE_RATE)
        self._ensemble = EnsembleDrumDetector(self._config, TARGET_SAMPLE_RATE, coprocessor=self._coprocessor)
        self._silence_gater = SilenceGater(self._config, TARGET_SAMPLE_RATE)
        self._resampler = SafeResampler()

        self._last_events: List[ForensicDrumEvent] = []
        self._last_patterns: List[Dict[str, Any]] = []
        self._detected_tempo: float = 120.0
        self._last_execution_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0
        self._tempo_cache: Dict[str, Tuple[float, float]] = {}
        self._history: deque = deque(maxlen=self._config.max_retained_events)

        self._log_status("DrumIntelligence initialized")

    def _sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Sanitize audio buffer - remove inf/nan."""
        if not self._config.sanitize_input:
            return audio

        if np.any(~np.isfinite(audio)):
            self._log_status("Found non-finite values in audio, sanitizing", "warn")
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        return audio

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def name(self) -> str:
        return self._name

    def run(self, audio_buffer: np.ndarray, context: AudioContext,
            tempo_bpm: Optional[float] = None) -> StageResult:
        """Process audio and return StageResult with NoteEvents."""
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting drum detection")

        try:
            events = self.detect(audio_buffer, context, tempo_bpm)

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory

            self._last_execution_time_ms = execution_time_ms

            if self._get_current_memory_mb() > STAGGERED_GC_TRIGGER_MB:
                self._force_cleanup()

            if self._music_box:
                # self._last_events holds the full ForensicDrumEvent
                # objects - each with a real ConfidenceComponents
                # breakdown, articulation, and (for NMF/contention-
                # touched hits) a reasoning_chain tag - all of which
                # previously only fed the single aggregate event_count
                # below, then got discarded.
                articulation_counts: Dict[str, int] = {}
                nmf_recovered = 0
                contention_resolved = 0
                confidence_components_sum = {
                    "transient_strength": 0.0, "spectral_match": 0.0,
                    "temporal_consistency": 0.0, "model_confidence": 0.0,
                    "ensemble_agreement": 0.0,
                }
                for e in self._last_events:
                    articulation_counts[e.articulation.value] = articulation_counts.get(e.articulation.value, 0) + 1
                    for tag in (e.reasoning_chain or ()):
                        if "NMF polyphony split" in tag:
                            nmf_recovered += 1
                        elif "Contention resolved" in tag:
                            contention_resolved += 1
                    for field_name in confidence_components_sum:
                        confidence_components_sum[field_name] += getattr(e.confidence, field_name)
                n = max(1, len(self._last_events))
                mean_confidence_components = {k: v / n for k, v in confidence_components_sum.items()}

                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type=DecisionType.DRUM_TESTIMONY,
                    before_state={"audio_shape": list(audio_buffer.shape)},
                    after_state={
                        "event_count": len(events),
                        "pattern_count": len(self._last_patterns),
                        "tempo": self._detected_tempo,
                        "articulation_counts": articulation_counts,
                        "nmf_recovered_count": nmf_recovered,
                        "contention_resolved_count": contention_resolved,
                        "mean_confidence_components": mean_confidence_components,
                    },
                    reasoning=f"Detected {len(events)} drum events",
                    reversible=True
                )

            self._update_progress(1.0, f"Complete: {len(events)} events")

            return StageResult(
                stage_name=self._name,
                success=len(events) > 0,
                events=events,
                metadata={
                    "event_count": len(events),
                    "pattern_count": len(self._last_patterns),
                    "tempo": self._detected_tempo,
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Detection failed: {e}", "error")
            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def detect(self, audio_buffer: np.ndarray, context: AudioContext,
               tempo_bpm: Optional[float] = None) -> List[NoteEvent]:
        """Detect drums using 5.0 pipeline."""
        if audio_buffer is None or len(audio_buffer) == 0:
            return []

        # CRITICAL: Sanitize input before any processing
        audio_buffer = self._sanitize_audio(audio_buffer)

        if audio_buffer.dtype != np.float32:
            audio_buffer = audio_buffer.astype(np.float32)

        # AudioContext carries no tempo field at all (by design - it's a
        # deterministic identity/format anchor, not an analysis result),
        # so without an explicit tempo_bpm from the caller this silently
        # fell back to a hardcoded 120 BPM guess for EVERY song regardless
        # of its real tempo. That wrong tempo fed straight into
        # MicrotimingAnalyzer's beat grids and PatternHashDetector's
        # sixteenth-note grid spacing, making both effectively meaningless
        # for anything not close to 120 BPM. The pipeline already knows
        # the real detected tempo by the time drum_detection runs (it runs
        # after tempo_analysis) - it just never passed it in.
        tempo = tempo_bpm if tempo_bpm and tempo_bpm > 0 else self._get_cached_tempo(context)
        self._detected_tempo = tempo

        density = self._estimate_rolling_density(audio_buffer)

        self._log_status(f"Tempo: {tempo:.1f} BPM, Density: {density:.2f}")

        self._update_progress(0.2, "Ensemble detection")
        raw_events = self._ensemble.detect(audio_buffer, density, tempo)

        self._update_progress(0.4, "Cleaning events")
        cleaned_events = self._deduplicate_events(raw_events, window_ms=self._config.dedup_window_ms)
        cleaned_events.sort(key=lambda e: (e.start_ms, e.velocity))

        self._update_progress(0.6, "Classifying articulations")
        classified_events = self._classify_articulations(cleaned_events, audio_buffer)

        self._update_progress(0.7, "Detecting patterns")
        # Pattern annotation is enrichment metadata only (pattern_count in
        # StageResult.metadata) - it never feeds the returned NoteEvents.
        # A bug in here (there was one: indexing an already-resolved
        # ForensicDrumEvent as if it were still an index) previously
        # propagated up through run()'s try/except and silently zeroed out
        # every real drum detection for the whole track. Isolate it so a
        # future bug in this specific step can only ever cost pattern
        # metadata, never the actual drum notes.
        try:
            pattern_detector = PatternHashDetector(self._config, tempo)
            self._last_patterns = pattern_detector.detect_patterns(classified_events)
        except Exception as e:
            self._log_status(f"Pattern detection failed (non-fatal): {e}", "warn")
            self._last_patterns = []

        self._update_progress(0.8, "Analyzing microtiming")
        microtiming_analyzer = MicrotimingAnalyzer(self._config, tempo)
        timed_events = microtiming_analyzer.analyze(classified_events)

        self._update_progress(0.85, "Adding forensic data")
        forensic_events = self._add_forensic_data(timed_events)

        # DrumDetectionConfig.min_confidence existed but was never actually
        # applied anywhere in this pipeline - every detected blip became a
        # NoteEvent regardless of confidence, with no floor at all. That's
        # a direct contributor to "drums too busy": low-confidence noise
        # from onset detection had no gate to stop it becoming a note.
        before_filter = len(forensic_events)
        forensic_events = [e for e in forensic_events
                           if e.confidence.total() >= self._config.min_confidence]
        if before_filter != len(forensic_events):
            self._log_status(
                f"Confidence filter: {before_filter - len(forensic_events)} "
                f"of {before_filter} events below {self._config.min_confidence} removed")

        self._update_progress(0.95, "Converting to NoteEvents")
        note_events = self._to_note_events(forensic_events)

        self._last_events = forensic_events[-self._config.max_retained_events:]

        return note_events

    def _deduplicate_events(self, events: List[ForensicDrumEvent], window_ms: float = 12.0) -> List[ForensicDrumEvent]:
        if not events:
            return []

        events.sort(key=lambda e: e.start_ms)
        result = []

        for event in events:
            if not result:
                result.append(event)
            else:
                last = result[-1]
                if event.drum_type == last.drum_type and abs(event.start_ms - last.start_ms) < window_ms:
                    if event.confidence.total() > last.confidence.total():
                        result[-1] = event
                else:
                    result.append(event)

        return result

    def _classify_articulations(self, events: List[ForensicDrumEvent],
                               audio: Optional[np.ndarray] = None) -> List[ForensicDrumEvent]:
        updated_events = []

        for event in events:
            articulation = DrumArticulation.NORMAL

            if event.velocity <= self._config.ghost_note_max_velocity:
                articulation = DrumArticulation.GHOST

            updated = replace(
                event,
                articulation=articulation,
                reasoning_chain=event.reasoning_chain + (f"Articulation: {articulation.value}",)
            )
            updated_events.append(updated)

        # Flam: two same-type hits close together (a grace note just before
        # the main stroke). flam_pair_id was already being set here, but
        # the event's articulation field itself was never actually updated
        # to FLAM - it silently stayed NORMAL/GHOST from the loop above.
        # Now also requires a real energy dip between the two onsets (when
        # audio is available) - timing proximity alone can't distinguish a
        # genuine flam from a fast, evenly-loud double stroke that just
        # happens to land in the same window.
        for i in range(len(updated_events) - 1):
            e1 = updated_events[i]
            e2 = updated_events[i + 1]

            if e1.drum_type == e2.drum_type:
                interval = e2.start_ms - e1.start_ms
                if 20 <= interval <= self._config.flam_window_ms:
                    has_dip = True
                    if audio is not None:
                        try:
                            has_dip = self._coprocessor.articulation.has_energy_dip(
                                audio, e1.start_ms, e2.start_ms)
                        except Exception:
                            has_dip = True
                    if has_dip:
                        flam_id = f"flam_{i}_{int(e1.start_ms)}"
                        updated_events[i] = replace(
                            updated_events[i], flam_pair_id=flam_id, articulation=DrumArticulation.FLAM)
                        updated_events[i + 1] = replace(
                            updated_events[i + 1], flam_pair_id=flam_id, articulation=DrumArticulation.FLAM)

        # Roll: roll_min_hits+ consecutive same-type hits, each within
        # roll_max_interval_ms of the previous one. This config already
        # existed (roll_min_hits/roll_max_interval_ms) but nothing ever
        # used it - DrumArticulation.ROLL was never assigned anywhere,
        # so a real press/buzz roll was previously indistinguishable from
        # a run of separately-articulated single hits.
        i = 0
        while i < len(updated_events):
            run = [i]
            j = i + 1
            while j < len(updated_events):
                prev = updated_events[run[-1]]
                curr = updated_events[j]
                if (curr.drum_type == prev.drum_type
                        and 0 < (curr.start_ms - prev.start_ms) <= self._config.roll_max_interval_ms):
                    run.append(j)
                    j += 1
                else:
                    break

            if len(run) >= self._config.roll_min_hits:
                for idx in run:
                    e = updated_events[idx]
                    updated_events[idx] = replace(
                        e, articulation=DrumArticulation.ROLL,
                        reasoning_chain=e.reasoning_chain + (f"Roll: {len(run)}-hit run",)
                    )
                i = run[-1] + 1
            else:
                i += 1

        # Buzz/press roll: the discrete-onset-run detection above can only
        # catch a roll the onset detector manages to split into
        # roll_min_hits+ separate clean hits - a real buzz roll often
        # blurs into one continuous texture that never decomposes that
        # way at all. Track zero-crossing-rate variance over a rolling
        # window instead: a sustained high-ZCR/low-variance region gets
        # flagged as a roll even when it produced few or no discrete
        # onsets of its own.
        if audio is not None:
            try:
                buzz_regions = self._coprocessor.articulation.detect_buzz_roll_regions(audio)
            except Exception:
                buzz_regions = []
            for start_ms, end_ms in buzz_regions:
                for idx, e in enumerate(updated_events):
                    if start_ms <= e.start_ms <= end_ms and e.articulation != DrumArticulation.ROLL:
                        updated_events[idx] = replace(
                            e, articulation=DrumArticulation.ROLL,
                            reasoning_chain=e.reasoning_chain + (
                                f"Buzz roll region: {start_ms:.0f}-{end_ms:.0f}ms",)
                        )

        return updated_events

    def _add_forensic_data(self, events: List[ForensicDrumEvent]) -> List[ForensicDrumEvent]:
        updated_events = []
        for event in events:
            updated = replace(
                event,
                spectral_centroid_hz=0.0,
                spectral_rolloff_hz=0.0,
                transient_sharpness=event.confidence.transient_strength,
                detection_pass=DetectionPass.FORENSIC
            )
            updated_events.append(updated)

        return updated_events

    def _to_note_events(self, events: List[ForensicDrumEvent]) -> List[NoteEvent]:
        return [e.to_note_event(self._kit_mapping) for e in events]

    def _get_cached_tempo(self, context: AudioContext) -> float:
        # FIXED: Use 'file_hash' not 'file_hash_sha256'
        cache_key = context.file_hash or context.file_path

        if cache_key in self._tempo_cache:
            tempo, timestamp = self._tempo_cache[cache_key]
            if time.time() - timestamp < self._config.tempo_cache_ttl_seconds:
                return tempo

        tempo = 120.0
        self._tempo_cache[cache_key] = (tempo, time.time())
        return tempo

    def _estimate_rolling_density(self, audio: np.ndarray) -> float:
        if not LIBROSA_AVAILABLE:
            return 0.5

        audio = self._sanitize_audio(audio)

        try:
            hop_length = 512
            onset_env = librosa.onset.onset_strength(y=audio, sr=TARGET_SAMPLE_RATE, hop_length=hop_length)

            window_frames = int(2 * TARGET_SAMPLE_RATE / hop_length)
            if len(onset_env) <= window_frames:
                return 0.5

            densities = []
            for i in range(0, len(onset_env) - window_frames, window_frames // 2):
                window = onset_env[i:i + window_frames]
                density = np.mean(window > np.percentile(window, 75))
                densities.append(density)

            return np.mean(densities) if densities else 0.5
        except Exception:
            return 0.5

    def _force_cleanup(self):
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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

    def validate(self, gate: ValidationGate) -> ValidationResult:
        if gate == ValidationGate.SILENCE_DETECTOR:
            is_valid = len(self._last_events) > 0
            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.EXCESS_SILENCE,
                events_validated=len(self._last_events),
                events_rejected=0
            )
        elif gate == ValidationGate.CONFIDENCE_THRESHOLD:
            if not self._last_events:
                return ValidationResult(is_valid=False, gate_used=gate, reason=VetoReason.CONFIDENCE_TOO_LOW)

            avg_confidence = np.mean([e.confidence.total() for e in self._last_events])
            is_valid = avg_confidence >= self._detection_threshold

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                confidence_before=avg_confidence,
                confidence_after=avg_confidence if is_valid else 0.0
            )
        else:
            return ValidationResult(is_valid=True, gate_used=gate)

    def get_confidence(self) -> Confidence:
        if not self._last_events:
            return Confidence.HALLUCINATION
        avg_confidence = np.mean([e.confidence.total() for e in self._last_events])
        return Confidence.from_float(avg_confidence)

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        if len(self._last_events) < 2:
            return (VetoReason.EXCESS_SILENCE, f"Only {len(self._last_events)} events detected")
        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        return SchoenbergResult(
            verdict=SchoenbergVerdict.PERCUSSION,
            zero_crossing_rate=0.0,
            spectral_flatness=0.7,
            reason="Percussion events by design"
        )

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

        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": before - after,
            "triggered_by": self._name
        }

    @property
    def detection_threshold(self) -> Confidence:
        return Confidence.LOW

    def get_raw_confidence(self) -> float:
        if not self._last_events:
            return 0.0
        return np.mean([e.confidence.total() for e in self._last_events])


def create_drum_intelligence(
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> DrumIntelligence:
    config = DrumDetectionConfig()
    return DrumIntelligence(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )