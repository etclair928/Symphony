# =================================================================
# MODULE: agents/detection/drum_intelligence.py
# DESCRIPTION: Complete drum and percussion intelligence for Grimlock 5.0.
#
# VERSION: 5.1 (FIXED: Input sanitization, no inline filtering)
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
    DrumType, DrumEvent, DrumKitMapping, WitnessVote
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

    def __init__(self, config: DrumDetectionConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate
        self._band_filter = SimpleBandFilter()

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

            low_onsets = self._detect_onsets_in_band(low_audio)
            mid_onsets = self._detect_onsets_in_band(mid_audio)
            high_onsets = self._detect_onsets_in_band(high_audio)

            return {
                'low': low_onsets,
                'mid': mid_onsets,
                'high': high_onsets
            }
        except Exception as e:
            return {'low': [], 'mid': [], 'high': []}

    def _detect_onsets_in_band(self, audio: np.ndarray) -> List[float]:
        """Detect onsets in a single frequency band."""
        if not LIBROSA_AVAILABLE or len(audio) == 0:
            return []

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

            peaks, _ = find_peaks(
                onset_env,
                height=np.percentile(onset_env, 75),
                distance=10
            )

            hop_ms = 256 / self.sample_rate * 1000
            return [p * hop_ms for p in peaks]
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

            if spectral_centroid < 150:
                return DrumType.KICK
            elif spectral_centroid < 800:
                if crest_factor > 5:
                    return DrumType.SNARE
                else:
                    return DrumType.TOM_MID
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

    def __init__(self, config: DrumDetectionConfig, sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config
        self.sample_rate = sample_rate
        self._spice = AdaptiveSPICEWrapper(config, sample_rate)
        self._onset = MultiBandOnsetDetector(config, sample_rate)
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

        onset_events = self._onset_to_events(self._onset.detect_onsets(audio))
        votes.append(onset_events)

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

    def _onset_to_events(self, onsets: Dict[str, List[float]]) -> List[ForensicDrumEvent]:
        """Convert onset detections to forensic events."""
        events = []
        band_mapping = {
            'low': DrumType.KICK,
            'mid': DrumType.SNARE,
            'high': DrumType.HI_HAT_CLOSED
        }

        for band, onset_times in onsets.items():
            drum_type = band_mapping.get(band, DrumType.PERCUSSION)

            for time_ms in onset_times:
                confidence = ConfidenceComponents(
                    transient_strength=0.7,
                    model_confidence=0.5,
                    temporal_consistency=0.6,
                    spectral_match=0.5,
                    ensemble_agreement=0.0
                )

                events.append(ForensicDrumEvent(
                    drum_type=drum_type,
                    start_ms=time_ms,
                    end_ms=time_ms + 150,
                    velocity=70,
                    confidence=confidence,
                    source_model="onset_detector",
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
        all_events = []
        for vote in votes:
            all_events.extend(vote)

        groups = defaultdict(list)
        for event in all_events:
            time_bucket = int(event.start_ms / 25) * 25
            groups[time_bucket].append(event)

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

        self._ensemble = EnsembleDrumDetector(self._config, TARGET_SAMPLE_RATE)
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
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="drum_detection",
                    before_state={"audio_shape": list(audio_buffer.shape)},
                    after_state={
                        "event_count": len(events),
                        "pattern_count": len(self._last_patterns),
                        "tempo": self._detected_tempo
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
        cleaned_events = self._deduplicate_events(raw_events)
        cleaned_events.sort(key=lambda e: (e.start_ms, e.velocity))

        self._update_progress(0.6, "Classifying articulations")
        classified_events = self._classify_articulations(cleaned_events)

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

    def _classify_articulations(self, events: List[ForensicDrumEvent]) -> List[ForensicDrumEvent]:
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

        for i in range(len(updated_events) - 1):
            e1 = updated_events[i]
            e2 = updated_events[i + 1]

            if e1.drum_type == e2.drum_type:
                interval = e2.start_ms - e1.start_ms
                if 20 <= interval <= self._config.flam_window_ms:
                    flam_id = f"flam_{i}_{int(e1.start_ms)}"
                    updated_events[i] = replace(e1, flam_pair_id=flam_id)
                    updated_events[i + 1] = replace(e2, flam_pair_id=flam_id)

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