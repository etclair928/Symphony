# =================================================================
# MODULE: agents/detection/rhythm_engine.py
# VERSION: 5.6.1
# DESCRIPTION: Rhythm detection engine for onset, beat, and pattern detection.
# Based on 4.7's rhythm_engine.py but refactored for 5.0.
#
# Changes for 5.0:
# - Removed confidence routing (routing is forbidden)
# - Added MusicBox integration for forensic logging
# - Added groove-aware onset detection
# - Added multi-resolution onset detection
# - Simplified to single responsibility (just detect rhythm)
# - Compliant with protocols.py and order_types.py
# - FIXED: numpy array to float conversion errors
#
# Authored by: DeepSeek - Refactored from 4.7 for 5.0 (2026-05-15)
# =================================================================

import warnings
import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import deque

# Try to import optional dependencies
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available. Rhythm engine functionality limited.")

try:
    from scipy import signal
    from scipy.signal import find_peaks, correlate

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Rhythm engine functionality limited.")

# Core imports - ONLY from order_types, constants, protocols
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    OnsetEvent, RhythmPattern, Confidence, VetoReason,
    ValidationGate, ValidationResult, SchoenbergResult,
    SchoenbergVerdict, UNKNOWN_HASH_PLACEHOLDER
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    LIBROSA_ONSET_HOP_LENGTH,
    LIBROSA_ONSET_BACKTRACK,
    LIBROSA_ONSET_THRESHOLD,
    MIN_CONFIDENCE_TO_PASS,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    DetectionAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol
)


# ========================================================================
# Helper Functions
# ========================================================================

def _to_float(value: Union[float, np.ndarray, None]) -> float:
    """
    Safely convert numpy array or scalar to float.

    CRITICAL FIX: Handles the TypeError when find_peaks returns 1-element arrays.
    """
    if value is None:
        return 0.0
    if np.isscalar(value):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value.item() if value.size == 1 else value[0])
    if hasattr(value, '__len__') and len(value) > 0:
        return float(value[0])
    return float(value)


def _to_int(value: Union[int, np.ndarray, None]) -> int:
    """Safely convert numpy array or scalar to int."""
    if value is None:
        return 0
    if np.isscalar(value):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        return int(value.item() if value.size == 1 else value[0])
    if hasattr(value, '__len__') and len(value) > 0:
        return int(value[0])
    return int(value)


# ========================================================================
# Configuration Dataclasses
# ========================================================================

@dataclass
class OnsetDetectionConfig:
    """Configuration for onset detection."""

    # Spectral flux parameters
    hop_length: int = 512  # Hop length in samples
    n_fft: int = 2048  # FFT window size
    win_length: int = 2048  # Window length
    window_type: str = 'hann'  # Window function

    # Onset detection
    onset_threshold: float = 0.5  # Threshold for onset detection (fallback path only)
    min_distance_ms: float = 10  # Minimum distance between onsets (ms)
    backtrack: bool = True  # Backtrack to peak of onset
    pre_avg: int = 3  # Frames for pre-average
    post_avg: int = 3  # Frames for post-average

    # librosa.onset.onset_detect's local-average peak-picking delta - a
    # candidate onset must exceed ITS OWN neighborhood's average by this
    # much, not some track-wide statistic. Same value already validated
    # for this exact purpose in drum_intelligence.py's DrumDetectionConfig.
    onset_delta: float = 0.12

    # Multi-resolution
    use_multi_resolution: bool = True  # Detect onsets at multiple time scales
    resolutions_ms: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class RhythmEngineState:
    """Internal state for rhythm engine (minimal, passed forward only)."""
    last_onsets: List[OnsetEvent] = field(default_factory=list)
    last_patterns: List[RhythmPattern] = field(default_factory=list)
    last_tempo_bpm: float = 120.0
    last_tempo_confidence: float = 0.0
    last_execution_time_ms: float = 0.0
    last_memory_delta_mb: float = 0.0


# ========================================================================
# Onset Detector (Pure function, no state)
# ========================================================================

class OnsetDetector:
    """
    Multi-resolution onset detector with groove-aware thresholding.

    Based on 4.7's onset detection but enhanced for 5.0 with:
    - Mel-scaled spectral flux (perceptually weighted)
    - Adaptive threshold based on groove density
    - Multi-resolution detection for different instruments
    - FIXED: numpy array to float conversion

    Law of Unidirectional Integrity: No external dependencies.
    """

    def __init__(self, config: Optional[OnsetDetectionConfig] = None,
                 sample_rate: int = TARGET_SAMPLE_RATE):
        self.config = config or OnsetDetectionConfig()
        self.sample_rate = sample_rate

    def detect_onsets(self, audio: np.ndarray) -> List[OnsetEvent]:
        """
        Detect onsets in audio.

        Returns:
            List of OnsetEvent objects with timestamps and strengths
        """
        if not LIBROSA_AVAILABLE or not SCIPY_AVAILABLE:
            return self._detect_onsets_fallback(audio)

        onsets = []

        if self.config.use_multi_resolution:
            # Detect at multiple resolutions and merge
            all_onsets = []
            for res_ms in self.config.resolutions_ms:
                hop = max(64, int(res_ms * self.sample_rate / 1000))
                res_onsets = self._detect_onsets_at_resolution(audio, hop)
                all_onsets.extend(res_onsets)

            # Merge close onsets (within 20ms)
            onsets = self._merge_onsets(all_onsets, merge_window_ms=20)
        else:
            onsets = self._detect_onsets_at_resolution(audio, self.config.hop_length)

        return onsets

    def _detect_onsets_at_resolution(self, audio: np.ndarray, hop_length: int) -> List[OnsetEvent]:
        """
        Detect onsets at a specific time resolution.

        Previously computed onset_strength as np.mean(spectral_flux, axis=0)
        (a mel-band average), then globally max-normalized it and ran
        find_peaks against a single track-wide "adaptive" threshold. That's
        the same class of bug already found and fixed in
        drum_intelligence.py: one loud transient dominates the
        normalization and squashes quieter-but-real hits below threshold,
        and _compute_adaptive_threshold's own density heuristic made dense
        passages LESS sensitive, not more. Verified directly on real drum
        audio: this drastically under-detected (13 onsets vs.
        DrumIntelligence's 201 hits on the identical 30s stem). Replaced
        with librosa.onset.onset_detect's local-average peak-picking - a
        candidate must exceed ITS OWN neighborhood's average, not a
        global statistic - the same standard fix already applied there.
        """
        try:
            onset_env = librosa.onset.onset_strength(
                y=audio, sr=self.sample_rate, hop_length=hop_length
            )
        except Exception:
            return []

        if len(onset_env) == 0:
            return []

        min_distance_samples = int(self.config.min_distance_ms * self.sample_rate / (hop_length * 1000))
        min_distance_samples = max(1, min_distance_samples)

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=self.config.backtrack,
            delta=self.config.onset_delta,
            wait=min_distance_samples,
        )

        # Normalize strength against the 95th percentile, not the single
        # loudest frame in the whole clip. Verified directly: a lone
        # exceptionally loud transient can sit far above the rest of the
        # envelope's real dynamic range (e.g. max=9.78 vs. p95=1.72 on a
        # real 30s drum stem), so dividing by the true max crushed every
        # other genuine hit down to a near-zero strength - which then
        # zeroed out every onset's note confidence downstream regardless
        # of tempo confidence. A high percentile is far more robust to
        # that one outlier while still anchoring "strength 1.0" to
        # genuinely loud hits (values above it just clip to 1.0).
        reference_env = float(np.percentile(onset_env, 95)) if len(onset_env) else 0.0

        onsets = []
        for frame in onset_frames:
            frame = int(frame)
            timestamp_ms = (frame * hop_length) / self.sample_rate * 1000

            env_val = _to_float(onset_env[frame]) if frame < len(onset_env) else 0.0
            env_val = max(0.0, env_val)
            strength = max(0.0, min(1.0, env_val / (reference_env + 1e-8))) if reference_env > 0 else 0.0

            onsets.append(OnsetEvent(
                timestamp_ms=timestamp_ms,
                strength=strength,
                width_ms=10.0,
                spectral_flux=env_val
            ))

        return onsets

    def _detect_onsets_fallback(self, audio: np.ndarray) -> List[OnsetEvent]:
        """Fallback onset detection when librosa/scipy unavailable."""
        # Simple energy-based onset detection
        frame_size = int(0.025 * self.sample_rate)  # 25ms frames
        hop_size = int(0.010 * self.sample_rate)  # 10ms hop

        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy.append(np.sum(frame ** 2))

        if not energy:
            return []

        energy = np.array(energy)
        energy_diff = np.diff(energy)
        energy_diff = np.maximum(energy_diff, 0)

        # Find peaks
        if len(energy_diff) == 0:
            return []

        threshold = np.mean(energy_diff) + np.std(energy_diff)

        if not SCIPY_AVAILABLE:
            # Manual peak detection without scipy
            peaks = []
            for i in range(1, len(energy_diff) - 1):
                if energy_diff[i] > threshold and energy_diff[i] > energy_diff[i - 1] and energy_diff[i] > energy_diff[
                    i + 1]:
                    peaks.append(i)
        else:
            peaks, _ = find_peaks(energy_diff, height=threshold)

        onsets = []
        for peak in peaks:
            timestamp_ms = (peak * hop_size) / self.sample_rate * 1000
            strength = _to_float(energy_diff[peak]) if peak < len(energy_diff) else 0.0
            onsets.append(OnsetEvent(
                timestamp_ms=timestamp_ms,
                strength=min(1.0, strength / (threshold + 1e-8)),
                width_ms=10.0,
                spectral_flux=0.0
            ))

        return onsets

    def _merge_onsets(self, onsets: List[OnsetEvent], merge_window_ms: float) -> List[OnsetEvent]:
        """Merge onsets that are very close in time."""
        if not onsets:
            return []

        # Sort by timestamp
        onsets.sort(key=lambda o: o.timestamp_ms)

        merged = []
        current = onsets[0]

        for next_onset in onsets[1:]:
            if next_onset.timestamp_ms - current.timestamp_ms <= merge_window_ms:
                # Merge: keep stronger one
                if next_onset.strength > current.strength:
                    current = next_onset
            else:
                merged.append(current)
                current = next_onset

        merged.append(current)
        return merged

    def compute_onset_density(self, onsets: List[OnsetEvent]) -> float:
        """Compute average onset density (onsets per second)."""
        if not onsets or len(onsets) < 2:
            return 0.0

        duration_ms = onsets[-1].timestamp_ms - onsets[0].timestamp_ms
        if duration_ms <= 0:
            return 0.0

        return len(onsets) / (duration_ms / 1000)


# ========================================================================
# Pattern Detector (Pure function, no state)
# ========================================================================

class PatternDetector:
    """
    Detects repetitive rhythmic patterns.

    Based on 4.7's pattern detection using autocorrelation.
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def find_patterns(self, onsets: List[OnsetEvent]) -> List[RhythmPattern]:
        """
        Find repeating patterns in onsets.

        Returns:
            List of detected rhythm patterns
        """
        if len(onsets) < 4 or not SCIPY_AVAILABLE:
            return []

        # Convert onsets to time array
        times = np.array([o.timestamp_ms for o in onsets])
        if len(times) < 2:
            return []

        intervals = np.diff(times)

        if len(intervals) < 2:
            return []

        # Find dominant interval via autocorrelation
        autocorr = correlate(intervals, intervals, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        max_autocorr = np.max(autocorr)
        if max_autocorr <= 0:
            return []

        # Find peaks
        peaks, _ = find_peaks(autocorr, height=max_autocorr * 0.3)

        patterns = []
        for peak in peaks[:3]:  # Top 3 patterns
            peak_idx = int(peak)
            if peak_idx >= len(intervals):
                pattern_interval = intervals[-1] if len(intervals) > 0 else 500.0
            else:
                pattern_interval = intervals[peak_idx]

            # Find pattern occurrences
            occurrences = []
            for i in range(len(times) - peak_idx):
                if i < len(intervals) and peak_idx < len(intervals):
                    if abs(intervals[i] - pattern_interval) < pattern_interval * 0.1:
                        occurrences.append(i)

            if len(occurrences) >= 2:
                # Compute swing ratio
                swing_ratio = self._detect_swing(intervals, pattern_interval)

                # Convert autocorr value to confidence (0.0-1.0)
                confidence_val = _to_float(autocorr[peak_idx] / max_autocorr) if max_autocorr > 0 else 0.5
                confidence_val = max(0.0, min(1.0, confidence_val))

                # Get pattern onsets
                pattern_onsets = [onsets[i] for i in occurrences[:4] if i < len(onsets)]

                patterns.append(RhythmPattern(
                    pattern_id=f"pattern_{len(patterns)}",
                    pattern_interval_ms=float(pattern_interval),
                    confidence=confidence_val,
                    onsets_in_pattern=pattern_onsets,
                    is_swing=swing_ratio > 1.15,
                    swing_ratio=float(swing_ratio)
                ))

        return patterns

    def _detect_swing(self, intervals: np.ndarray, base_interval: float) -> float:
        """Detect swing ratio (longer/shorter alternation)."""
        if len(intervals) < 4:
            return 1.0

        # Look for alternating pattern: long, short, long, short
        pattern = []
        for interval in intervals[:16]:  # Look at first 16 intervals
            if interval > base_interval * 1.1:
                pattern.append('L')
            elif interval < base_interval * 0.9:
                pattern.append('S')
            else:
                pattern.append('M')

        # Check for LS alternating pattern
        pattern_str = ''.join(pattern)
        if 'LSLS' in pattern_str or 'SLSL' in pattern_str:
            # Compute swing ratio
            longs = [intervals[i] for i, p in enumerate(pattern) if p == 'L' and i < len(intervals)]
            shorts = [intervals[i] for i, p in enumerate(pattern) if p == 'S' and i < len(intervals)]
            if longs and shorts:
                return float(np.mean(longs) / np.mean(shorts))

        return 1.0


# ========================================================================
# Tempo Estimator (Pure function, no state)
# ========================================================================

class TempoEstimator:
    """
    Tempo estimation from onsets and (when available) raw audio.

    Based on 4.7's tempo tracking.
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def estimate_tempo(self, onsets: List[OnsetEvent],
                        audio: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Estimate tempo from onsets (and audio, when available).

        Returns:
            Tuple of (tempo_bpm, confidence) where confidence is 0.0-1.0

        Previously averaged raw inter-onset intervals directly into a BPM
        figure with only gross-outlier trimming. That treats every onset
        as if it landed on the beat, so a real kit pattern (kick/snare/
        hihat at mixed 8th/16th-note subdivisions) produces a meaningless
        average - and its own variance-based confidence collapsed toward
        zero for the same reason, since high interval variance looks like
        "noise" even when it's a legitimate mixed-subdivision groove.
        Verified on real drum audio: this produced 40 BPM (should be far
        higher) with near-zero confidence, which then crushed every
        onset's note confidence downstream in _convert_onsets_to_notes.

        Uses librosa.beat.beat_track (autocorrelation-based - the same
        approach agents/analysis/tempo_intelligence.py uses for the
        pipeline's authoritative tempo) when raw audio is available, since
        it correctly handles subdivisions instead of just averaging IOIs.
        Falls back to the IOI-based estimate only if audio isn't provided
        or beat tracking fails outright.
        """
        if audio is not None and LIBROSA_AVAILABLE and len(audio) > 0:
            try:
                tempo, beat_frames = librosa.beat.beat_track(
                    y=audio, sr=self.sample_rate, hop_length=512
                )
                # librosa >=0.10 returns tempo as a 0-d/1-element ndarray.
                tempo = float(np.asarray(tempo).reshape(-1)[0])
                tempo = max(40.0, min(240.0, tempo))

                if len(beat_frames) > 4:
                    beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate, hop_length=512)
                    beat_intervals = np.diff(beat_times)
                    if len(beat_intervals) > 0:
                        consistency = 1.0 - min(1.0, float(np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-8)))
                        confidence = min(0.9, 0.5 + consistency * 0.4)
                    else:
                        confidence = 0.5
                else:
                    confidence = 0.3

                return tempo, confidence
            except Exception:
                pass  # fall through to the IOI-based estimate below

        if len(onsets) < 4:
            return 120.0, 0.0

        # Compute inter-onset intervals
        times = np.array([o.timestamp_ms for o in onsets])
        intervals = np.diff(times)

        if len(intervals) < 2:
            return 120.0, 0.0

        # Remove outliers
        median_interval = np.median(intervals)
        valid_intervals = intervals[intervals < median_interval * 1.5]
        valid_intervals = valid_intervals[valid_intervals > median_interval * 0.5]

        if len(valid_intervals) < 2:
            return 120.0, 0.0

        # Average interval in seconds
        avg_interval_sec = np.mean(valid_intervals) / 1000

        # Convert to BPM
        tempo_bpm = 60.0 / avg_interval_sec if avg_interval_sec > 0 else 120.0

        # Compute confidence (inverse of variance)
        variance = np.var(valid_intervals)
        confidence = min(1.0, 1.0 / (1.0 + variance / 1000))

        # Clamp tempo to reasonable range
        tempo_bpm = max(40.0, min(240.0, tempo_bpm))

        return tempo_bpm, confidence


# ========================================================================
# Rhythm Engine (Main Agent)
# ========================================================================

class RhythmEngine:
    """
    Rhythm detection engine for Grimlock 5.0.

    Implements:
        - DetectionAgentProtocol (detect method)
        - ScribeValidatable (validate, get_confidence, get_veto_status)
        - MemoryManagedProtocol (release_buffer, get_memory_footprint_mb)

    Based on 4.7's rhythm_engine but simplified:
        - Detects onsets (attacks)
        - Detects rhythmic patterns
        - Estimates tempo
        - Converts to NoteEvents for unified pipeline

    Law of Relational Physics: Measures distance between events,
    not distance to grid.
    Law of Unidirectional Integrity: No calls to other agents.
    """

    def __init__(
            self,
            onset_config: Optional[OnsetDetectionConfig] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Args:
            onset_config: Onset detection configuration
            music_box: MusicBox for forensic logging
            progress_callback: Optional callback for progress updates
        """
        self._name = "rhythm_engine"
        self._source_type = SourceType.RHYTHM
        self._music_box = music_box
        self._progress_callback = progress_callback

        # Configuration
        self.onset_config = onset_config or OnsetDetectionConfig()
        self._detection_threshold = Confidence.LOW.value  # 0.25

        # Components
        self._onset_detector = OnsetDetector(self.onset_config)
        self._pattern_detector = PatternDetector()
        self._tempo_estimator = TempoEstimator()

        # State (minimal, only for getters)
        self._state = RhythmEngineState()

        # Memory tracking
        self._allocated_buffers: Dict[str, int] = {}
        self._last_memory_mb: float = 0.0

    # ========================================================================
    # AgentProtocol Implementation
    # ========================================================================

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def name(self) -> str:
        return self._name

    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """
        Process audio and return StageResult with NoteEvents.

        This is the primary entry point for the pipeline.
        """
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        # Reset state for this run
        self._state = RhythmEngineState()

        try:
            # Detect rhythm and convert to notes
            events = self.detect(audio_buffer, context)

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory

            self._state.last_execution_time_ms = execution_time_ms
            self._state.last_memory_delta_mb = memory_delta_mb

            # Log to MusicBox
            if self._music_box:
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="rhythm_detection",
                    before_state={"audio_shape": list(audio_buffer.shape)},
                    after_state={
                        "event_count": len(events),
                        "onset_count": len(self._state.last_onsets),
                        "pattern_count": len(self._state.last_patterns),
                        "tempo_bpm": self._state.last_tempo_bpm
                    },
                    reasoning=f"Detected {len(events)} rhythm events at {self._state.last_tempo_bpm:.1f} BPM",
                    reversible=True
                )

            return StageResult(
                stage_name=self._name,
                success=len(events) > 0,
                events=events,
                metadata={
                    "onset_count": len(self._state.last_onsets),
                    "pattern_count": len(self._state.last_patterns),
                    "tempo_bpm": self._state.last_tempo_bpm,
                    "tempo_confidence": self._state.last_tempo_confidence,
                    "groove_field": self.compute_groove_field()
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            import traceback
            traceback.print_exc()

            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=execution_time_ms,
                memory_delta_mb=self._get_current_memory_mb() - start_memory
            )

    # ========================================================================
    # DetectionAgentProtocol Implementation
    # ========================================================================

    def detect(self, stem_buffer: np.ndarray, context: AudioContext) -> List[NoteEvent]:
        """
        Detect rhythm and convert to NoteEvents.

        Drums are represented as MIDI notes for unified pipeline.
        """
        self._update_progress(0.0, "Starting rhythm detection")

        # Validate input
        if stem_buffer is None or len(stem_buffer) == 0:
            return []

        # Ensure float32
        if stem_buffer.dtype != np.float32:
            stem_buffer = stem_buffer.astype(np.float32)

        # Ensure 1D (mono)
        if stem_buffer.ndim == 2:
            stem_buffer = np.mean(stem_buffer, axis=1)

        # Step 1: Detect onsets
        self._update_progress(0.2, "Detecting onsets")
        onsets = self._onset_detector.detect_onsets(stem_buffer)
        self._state.last_onsets = onsets

        # Step 2: Find patterns
        self._update_progress(0.5, "Finding rhythmic patterns")
        patterns = self._pattern_detector.find_patterns(onsets)
        self._state.last_patterns = patterns

        # Step 3: Estimate tempo
        self._update_progress(0.7, "Estimating tempo")
        tempo, tempo_confidence = self._tempo_estimator.estimate_tempo(onsets, audio=stem_buffer)
        self._state.last_tempo_bpm = tempo
        self._state.last_tempo_confidence = tempo_confidence

        # Step 4: Convert to NoteEvents (kick drum mapping)
        self._update_progress(0.9, "Converting to notes")
        events = self._convert_onsets_to_notes(onsets, tempo, tempo_confidence)

        # Step 5: Filter and post-process
        events = self._post_process(events)

        self._update_progress(1.0, "Rhythm detection complete")

        return events

    @property
    def detection_threshold(self) -> Confidence:
        """Minimum confidence to emit a note."""
        return Confidence.LOW

    def get_raw_confidence(self) -> float:
        """Return raw float for the last detection."""
        return self._state.last_tempo_confidence

    # ========================================================================
    # ScribeValidatable Implementation
    # ========================================================================

    def validate(self, gate: ValidationGate) -> ValidationResult:
        """
        Run validation gate on this agent's output.

        Gates supported:
            - SILENCE_DETECTOR: Check for too much silence
            - CONFIDENCE_THRESHOLD: Check average confidence
            - TEMPO_REASONABLENESS: Check tempo range
        """
        events = self._state.last_onsets

        if gate == ValidationGate.SILENCE_DETECTOR:
            # Check if we have enough onsets
            if len(events) < 2:
                return ValidationResult(
                    is_valid=False,
                    gate_used=gate,
                    reason=VetoReason.EXCESS_SILENCE,
                    detail=f"Only {len(events)} onsets detected",
                    events_validated=0,
                    events_rejected=len(events)
                )
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                events_validated=len(events),
                events_rejected=0
            )

        elif gate == ValidationGate.CONFIDENCE_THRESHOLD:
            # Check average confidence
            if not events:
                return ValidationResult(
                    is_valid=False,
                    gate_used=gate,
                    reason=VetoReason.CONFIDENCE_TOO_LOW,
                    detail="No onsets detected",
                    confidence_before=0.0,
                    confidence_after=0.0
                )

            avg_strength = np.mean([o.strength for o in events])
            is_valid = avg_strength >= MIN_CONFIDENCE_TO_PASS

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                detail=f"Average onset strength: {avg_strength:.2f}",
                events_validated=len(events),
                events_rejected=0 if is_valid else len(events),
                confidence_before=avg_strength,
                confidence_after=avg_strength if is_valid else 0.0
            )

        elif gate == ValidationGate.TEMPO_REASONABLENESS:
            # Check tempo range
            tempo = self._state.last_tempo_bpm
            is_valid = 40 <= tempo <= 240

            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.TEMPO_OUTLIER,
                detail=f"Tempo: {tempo:.1f} BPM",
                events_validated=len(events),
                events_rejected=0
            )

        else:
            # Unsupported gate - pass by default
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                detail=f"Gate {gate.value} not supported by RhythmEngine",
                events_validated=len(events)
            )

    def get_confidence(self) -> Confidence:
        """Return overall confidence of last output."""
        if not self._state.last_onsets:
            return Confidence.HALLUCINATION

        avg_strength = np.mean([o.strength for o in self._state.last_onsets])
        combined = self._state.last_tempo_confidence * avg_strength
        return Confidence.from_float(combined)

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        """Return (reason, detail) if currently vetoed, else None."""
        if len(self._state.last_onsets) < 2:
            return (VetoReason.EXCESS_SILENCE, f"Only {len(self._state.last_onsets)} onsets detected")

        if self._state.last_tempo_bpm < 40 or self._state.last_tempo_bpm > 240:
            return (VetoReason.TEMPO_OUTLIER, f"Tempo {self._state.last_tempo_bpm:.1f} BPM out of range")

        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        """
        Run harmonic series check.

        For rhythm engine, onsets are percussive by nature.
        Returns PERCUSSION verdict since rhythm hits are not tonal.
        """
        if len(self._state.last_onsets) < 2:
            return SchoenbergResult(
                verdict=SchoenbergVerdict.NOISE,
                zero_crossing_rate=0.0,
                spectral_flatness=0.7,
                reason="Insufficient onsets for rhythm analysis"
            )

        return SchoenbergResult(
            verdict=SchoenbergVerdict.PERCUSSION,
            zero_crossing_rate=0.0,
            spectral_flatness=0.8,
            reason="Rhythm engine produces percussive events by design"
        )

    # ========================================================================
    # MemoryManagedProtocol Implementation
    # ========================================================================

    def release_buffer(self, buffer_name: str) -> None:
        """Explicit deletion request for audio buffers."""
        if buffer_name in self._allocated_buffers:
            del self._allocated_buffers[buffer_name]
            gc.collect()

    def get_memory_footprint_mb(self) -> float:
        """Current memory usage in MB."""
        return self._get_current_memory_mb()

    def can_release(self, buffer_name: str) -> bool:
        """Return True if this buffer is safe to delete."""
        # Rhythm engine buffers are only needed during detection
        return True

    def staggered_gc(self) -> Dict[str, Any]:
        """Run garbage collection and return stats."""
        before = self._get_current_memory_mb()
        gc.collect()
        after = self._get_current_memory_mb()

        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": before - after,
            "triggered_by": self._name
        }

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _convert_onsets_to_notes(
            self,
            onsets: List[OnsetEvent],
            tempo: float,
            tempo_confidence: float
    ) -> List[NoteEvent]:
        """
        Convert onsets to NoteEvents.

        Onsets become kick drum notes (MIDI note 36).
        """
        events = []

        for onset in onsets:
            # Confidence based mostly on real onset strength, with tempo
            # agreement only a minor corroborating factor - "is this a
            # real transient" and "do we trust the tempo/grid" are
            # different questions. The old 0.5/0.5 split let a low (or
            # legitimately uncertain) tempo_confidence crush a genuinely
            # strong onset down near zero, which - combined with the
            # tempo estimator's own confidence collapsing on real
            # multi-subdivision drum patterns - silently zeroed out every
            # onset once real onset detection actually started finding
            # them (see TempoEstimator.estimate_tempo).
            confidence = onset.strength * (0.7 + tempo_confidence * 0.3)
            confidence = max(0.0, min(1.0, confidence))

            # Simple duration: 100ms default, adjusted by tempo
            duration_ms = max(50.0, min(200.0, 60000.0 / max(tempo, 40.0) / 8.0))

            event = NoteEvent(
                pitch=36,  # Kick drum (GM mapping)
                start_ms=onset.timestamp_ms,
                end_ms=onset.timestamp_ms + duration_ms,
                velocity=int(min(127, onset.strength * 127)),
                confidence=confidence,
                zero_crossing_rate=0.0,  # Will be filled by Scribe
                source=SourceType.RHYTHM,
                reasoning_chain=[f"Detected onset with strength {onset.strength:.2f}"]
            )

            events.append(event)

        return events

    def _post_process(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """
        Post-process events:
        - Remove events below confidence threshold
        - Merge very close events
        - Apply density limit
        """
        # Filter by confidence
        events = [e for e in events if e.confidence >= self._detection_threshold]

        # Merge events within 20ms of each other - but capped, so a run of
        # closely-spaced low-confidence bleed onsets (each within 20ms of
        # the PREVIOUS one, individually legitimate merge candidates) can't
        # chain-extend one note indefinitely. These are single kick-drum
        # hits (see detect() above), not sustained notes, so a real kick
        # is never legitimately hundreds of ms long - 300ms is already
        # generous even at a slow tempo.
        MAX_MERGED_DURATION_MS = 300.0
        if len(events) > 1:
            merged = []
            events.sort(key=lambda e: e.start_ms)
            current = events[0]
            chain_start_ms = current.start_ms

            for next_event in events[1:]:
                candidate_end = max(current.end_ms, next_event.end_ms)
                if (next_event.start_ms - current.end_ms <= 20
                        and candidate_end - chain_start_ms <= MAX_MERGED_DURATION_MS):
                    # Merge: keep longer duration, higher confidence
                    current.end_ms = candidate_end
                    current.confidence = max(current.confidence, next_event.confidence)
                    current.velocity = max(current.velocity, next_event.velocity)
                else:
                    merged.append(current)
                    current = next_event
                    chain_start_ms = current.start_ms
            merged.append(current)
            events = merged

        return events

    def _update_progress(self, progress: float, message: str):
        """Update progress via callback."""
        if self._progress_callback:
            self._progress_callback(progress, message)

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    # ========================================================================
    # Public Getter Methods
    # ========================================================================

    def get_onsets(self) -> List[OnsetEvent]:
        """Get last detected onsets."""
        return self._state.last_onsets

    def get_patterns(self) -> List[RhythmPattern]:
        """Get last detected patterns."""
        return self._state.last_patterns

    def get_tempo(self) -> float:
        """Get last estimated tempo."""
        return self._state.last_tempo_bpm

    def compute_groove_field(self) -> Dict[str, Any]:
        """
        Compute groove field metrics from onsets.

        Law of Relational Physics: Measures inter-onset relationships.
        """
        if len(self._state.last_onsets) < 4:
            return {"has_groove": False}

        times = [o.timestamp_ms for o in self._state.last_onsets]
        intervals = np.diff(times)

        if len(intervals) < 2:
            return {"has_groove": False}

        # Detect swing (alternating long/short)
        even_intervals = intervals[0::2] if len(intervals) > 1 else []
        odd_intervals = intervals[1::2] if len(intervals) > 2 else []

        is_swing = False
        swing_ratio = 1.0

        if len(even_intervals) > 0 and len(odd_intervals) > 0:
            avg_even = np.mean(even_intervals)
            avg_odd = np.mean(odd_intervals)
            if min(avg_even, avg_odd) > 0:
                swing_ratio = max(avg_even, avg_odd) / min(avg_even, avg_odd)
                is_swing = swing_ratio > 1.15

        # Detect Dilla pocket (micro-timing variations)
        interval_std = np.std(intervals) if len(intervals) > 0 else 0
        avg_interval = np.mean(intervals) if len(intervals) > 0 else 0
        is_dilla = interval_std / avg_interval > 0.15 if avg_interval > 0 else False

        return {
            "has_groove": is_swing or is_dilla,
            "is_swing": is_swing,
            "swing_ratio": float(swing_ratio),
            "is_dilla_pocket": is_dilla,
            "interval_variation": float(interval_std / avg_interval) if avg_interval > 0 else 0,
            "average_interval_ms": float(avg_interval),
            "tempo_bpm": self._state.last_tempo_bpm
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get rhythm detection statistics."""
        return {
            "name": self.name,
            "source_type": self.source_type.value,
            "last_onset_count": len(self._state.last_onsets),
            "last_pattern_count": len(self._state.last_patterns),
            "last_tempo_bpm": self._state.last_tempo_bpm,
            "last_tempo_confidence": self._state.last_tempo_confidence,
            "last_execution_time_ms": self._state.last_execution_time_ms,
            "last_memory_delta_mb": self._state.last_memory_delta_mb,
            "groove_field": self.compute_groove_field()
        }


# ========================================================================
# Convenience Functions
# ========================================================================

def create_rhythm_engine(
        onset_threshold: float = 0.5,
        use_multi_resolution: bool = True,
        music_box: Optional[MusicBoxProtocol] = None
) -> RhythmEngine:
    """
    Convenience function to create a rhythm engine.

    Args:
        onset_threshold: Threshold for onset detection (0-1)
        use_multi_resolution: Use multi-resolution detection
        music_box: MusicBox for logging

    Returns:
        Configured RhythmEngine instance
    """
    config = OnsetDetectionConfig(
        onset_threshold=onset_threshold,
        use_multi_resolution=use_multi_resolution
    )

    return RhythmEngine(onset_config=config, music_box=music_box)


def quick_rhythm_test(audio_path: str) -> Dict[str, Any]:
    """
    Quick test function for rhythm engine.

    Args:
        audio_path: Path to test audio file

    Returns:
        Dictionary with test results
    """
    if not LIBROSA_AVAILABLE:
        return {"error": "librosa not available"}

    import librosa

    print(f"Testing Rhythm Engine on {audio_path}")

    # Load audio
    audio, sr = librosa.load(audio_path, mono=True, sr=TARGET_SAMPLE_RATE)
    print(f"Audio loaded: {len(audio) / sr:.1f}s")

    # Create context
    context = AudioContext(
        file_path=audio_path,
        original_sample_rate=sr,
        working_sample_rate=sr,
        duration_seconds=len(audio) / sr,
        num_channels=1,
        file_hash=UNKNOWN_HASH_PLACEHOLDER,
        memory_mb=0.0
    )

    # Create engine
    engine = create_rhythm_engine()

    # Detect
    result = engine.run(audio, context)

    # Get stats
    stats = engine.get_statistics()

    print(f"  Success: {result.success}")
    print(f"  Events: {len(result.events)}")
    print(f"  Detected onsets: {stats['last_onset_count']}")
    print(f"  Tempo: {stats['last_tempo_bpm']:.1f} BPM")
    print(f"  Patterns: {stats['last_pattern_count']}")
    print(f"  Groove: swing={stats['groove_field']['is_swing']}, dilla={stats['groove_field']['is_dilla_pocket']}")

    return stats


# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        quick_rhythm_test(sys.argv[1])
    else:
        print("Usage: python rhythm_engine.py <audio_file>")