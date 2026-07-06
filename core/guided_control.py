# =================================================================
# MODULE: core/guided_control.py
# DESCRIPTION: Guided Mode Controller for Grimlock 5.0
#
# When the user knows their song's parameters (tempo, key, time signature, genre),
# this module bypasses unreliable automatic detection and provides:
#   - Direct tempo/beat grid generation (skip Madmom)
#   - Key-constrained pitch detection (faster, fewer hallucinations)
#   - Time signature for groove field (skip complex detection)
#   - Genre hints for model selection
#
# DEPENDS ON: order_types.py (Confidence, SourceType, TimeSignatureCandidate, etc.)
#             constants.py (MIN_TEMPO_BPM, MAX_TEMPO_BPM, etc.)
#             protocols.py (AgentFactoryProtocol, etc.)
#
# This is the heart of the "human knows best" philosophy.
#
# VERSION: 5.6.1 (refactored)
# UPDATED: 2026-05-12
# =================================================================

from __future__ import annotations

import math
import bisect
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union, Set
from enum import Enum

# Import ONLY from core modules - NO CIRCULAR IMPORTS
from core.order_types import (
    Confidence, SourceType, NoteEvent, PulseField, BeatGrid,
    TempoMap, TempoEvent, TimeSignatureCandidate, ValidationGate,
    VetoReason, StageResult, AudioContext, QuantizationStrategy,
    DecisionType
)
from core.constants import (
    MIN_TEMPO_BPM, MAX_TEMPO_BPM, DEFAULT_TEMPO_BPM,
    MIN_CONFIDENCE_TO_PASS, RITORNELLO_MAX_SNAP_MS,
    HEARTBEAT_NOTE_PITCH, HEARTBEAT_NOTE_VELOCITY, HEARTBEAT_NOTE_DURATION_MS,
    PROFESSIONAL_FIDELITY_SR
)

# Import from feature_bundle for spectral evidence (optional)
from core.feature_bundle import EvidenceType, FeatureBundle


# =================================================================
# PART 1: GUIDED MODE ENUMS (Extends order_types)
# =================================================================

class GenreType(str, Enum):
    """Genre hints for model selection and processing."""
    JAZZ = "jazz"
    ROCK = "rock"
    CLASSICAL = "classical"
    ELECTRONIC = "electronic"
    HIP_HOP = "hip_hop"
    POP = "pop"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "GenreType":
        """Safe conversion from string to GenreType."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN


class TimeSignature(str, Enum):
    """Common time signatures. Extends order_types.TimeSignatureCandidate."""
    TWO_FOUR = "2/4"
    THREE_FOUR = "3/4"
    FOUR_FOUR = "4/4"
    SIX_EIGHT = "6/8"
    FIVE_FOUR = "5/4"
    SEVEN_EIGHT = "7/8"
    TWELVE_EIGHT = "12/8"

    @classmethod
    def from_numbers(cls, numerator: int, denominator: int) -> Optional["TimeSignature"]:
        """Convert numerator/denominator to TimeSignature enum."""
        sig_str = f"{numerator}/{denominator}"
        for sig in cls:
            if sig.value == sig_str:
                return sig
        return None

    @property
    def numerator(self) -> int:
        return int(self.value.split('/')[0])

    @property
    def denominator(self) -> int:
        return int(self.value.split('/')[1])

    @property
    def beats_per_measure(self) -> int:
        return self.numerator

    @property
    def beat_unit(self) -> int:
        return self.denominator

    def to_candidate(self, confidence: Confidence = Confidence.PERFECT) -> TimeSignatureCandidate:
        """Convert to TimeSignatureCandidate for pipeline compatibility."""
        return TimeSignatureCandidate(
            numerator=self.numerator,
            denominator=self.denominator,
            confidence=confidence.value,
            sample_segment_start_ms=0.0,
            sample_segment_end_ms=0.0
        )


# =================================================================
# PART 2: GUIDED PARAMETERS DATA STRUCTURE
# =================================================================

@dataclass
class GuidedParams:
    """
    User-provided parameters that override automatic detection.

    All fields are optional. If None, the pipeline will auto-detect.
    This is the primary interface for guided mode.

    LAW: Guided parameters are suggestions, not hard overrides.
          The pipeline may still veto results that violate confidence thresholds.
    """
    # Tempo and timing
    tempo_bpm: Optional[float] = None
    time_signature: Optional[Union[str, TimeSignature]] = None

    # Harmonic context
    key_signature: Optional[str] = None  # e.g., "C major", "C# minor"
    root_note: Optional[int] = None  # MIDI note number for key root (0-11)
    scale_type: Optional[str] = None  # "major", "minor", "dorian", etc.

    # Genre and style
    genre: Optional[Union[str, GenreType]] = None

    # Advanced overrides
    swing_ratio: Optional[float] = None  # 0.5 = straight, 0.67 = triplet
    custom_beat_grid_ms: Optional[List[float]] = None
    constant_tempo: bool = True
    polyphonic: bool = True

    # Analysis scope
    analyze_bass: bool = True
    analyze_drums: bool = True
    analyze_harmony: bool = True
    analyze_groove: bool = True

    def __post_init__(self):
        """Normalize and validate parameters."""
        # Normalize time signature
        if isinstance(self.time_signature, str):
            try:
                self.time_signature = TimeSignature(self.time_signature)
            except ValueError:
                if '/' in self.time_signature:
                    parts = self.time_signature.split('/')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        self.time_signature = TimeSignature.from_numbers(
                            int(parts[0]), int(parts[1])
                        )

        # Normalize genre
        if isinstance(self.genre, str):
            self.genre = GenreType.from_string(self.genre)

        # Validate tempo range
        if self.tempo_bpm is not None:
            if self.tempo_bpm < MIN_TEMPO_BPM or self.tempo_bpm > MAX_TEMPO_BPM:
                self.tempo_bpm = DEFAULT_TEMPO_BPM

        # Validate swing ratio
        if self.swing_ratio is not None:
            if self.swing_ratio < 0.5 or self.swing_ratio > 0.85:
                self.swing_ratio = 0.5

    @property
    def has_tempo(self) -> bool:
        return self.tempo_bpm is not None and self.tempo_bpm > 0

    @property
    def has_time_signature(self) -> bool:
        return self.time_signature is not None

    @property
    def has_key(self) -> bool:
        return self.key_signature is not None or self.root_note is not None

    @property
    def has_genre(self) -> bool:
        return self.genre is not None and self.genre != GenreType.UNKNOWN

    @property
    def has_swing(self) -> bool:
        return self.swing_ratio is not None

    @property
    def has_custom_beat_grid(self) -> bool:
        return self.custom_beat_grid_ms is not None and len(self.custom_beat_grid_ms) > 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuidedParams":
        """Create GuidedParams from dictionary (e.g., from API request)."""
        time_sig = data.get('time_signature')
        if isinstance(time_sig, dict):
            # Handle nested dict format
            time_sig = TimeSignature.from_numbers(
                time_sig.get('numerator', 4),
                time_sig.get('denominator', 4)
            )

        return cls(
            tempo_bpm=data.get('tempo_bpm'),
            time_signature=time_sig,
            key_signature=data.get('key_signature'),
            root_note=data.get('root_note'),
            scale_type=data.get('scale_type'),
            genre=data.get('genre'),
            swing_ratio=data.get('swing_ratio'),
            custom_beat_grid_ms=data.get('custom_beat_grid_ms'),
            constant_tempo=data.get('constant_tempo', True),
            polyphonic=data.get('polyphonic', True),
            analyze_bass=data.get('analyze_bass', True),
            analyze_drums=data.get('analyze_drums', True),
            analyze_harmony=data.get('analyze_harmony', True),
            analyze_groove=data.get('analyze_groove', True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tempo_bpm': self.tempo_bpm,
            'time_signature': self.time_signature.value if self.time_signature else None,
            'key_signature': self.key_signature,
            'root_note': self.root_note,
            'scale_type': self.scale_type,
            'genre': self.genre.value if self.genre else None,
            'swing_ratio': self.swing_ratio,
            'custom_beat_grid_ms': self.custom_beat_grid_ms,
            'constant_tempo': self.constant_tempo,
            'polyphonic': self.polyphonic,
            'analyze_bass': self.analyze_bass,
            'analyze_drums': self.analyze_drums,
            'analyze_harmony': self.analyze_harmony,
            'analyze_groove': self.analyze_groove,
        }

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate parameters. Returns (is_valid, error_message)."""
        if self.tempo_bpm is not None and (self.tempo_bpm < MIN_TEMPO_BPM or self.tempo_bpm > MAX_TEMPO_BPM):
            return False, f"Tempo must be between {MIN_TEMPO_BPM} and {MAX_TEMPO_BPM} BPM"

        if self.swing_ratio is not None and (self.swing_ratio < 0.5 or self.swing_ratio > 0.85):
            return False, "Swing ratio should be between 0.5 (straight) and 0.85 (heavy swing)"

        return True, None


# =================================================================
# PART 3: BEAT GRID GENERATOR (Bypasses Madmom)
# =================================================================

@dataclass
class GuidedBeatGrid(BeatGrid):
    """
    Extended BeatGrid with guided mode metadata.
    Inherits from order_types.BeatGrid for pipeline compatibility.
    """
    source: SourceType = SourceType.GUIDED
    confidence: Confidence = Confidence.HIGH
    is_guided: bool = True
    guided_tempo_bpm: float = 0.0
    guided_time_signature: TimeSignature = TimeSignature.FOUR_FOUR

    def to_pulse_field(self) -> PulseField:
        """
        Convert to PulseField for downstream quantization.
        Overrides parent to set confidence from guided mode.
        """
        pulse_field = super().to_pulse_field(self.confidence)
        # Mark as guided in metadata
        pulse_field.__dict__['_guided'] = True
        return pulse_field

    def snap_to_grid(self, time_ms: float, tolerance_ms: float = 50) -> Tuple[float, bool]:
        """
        Snap a timestamp to the nearest beat if within tolerance.
        Returns (snapped_time, was_snapped).
        """
        closest, _ = self.get_closest_beat(time_ms)
        if abs(time_ms - closest) <= tolerance_ms:
            return closest, True
        return time_ms, False


class GuidedBeatGridGenerator:
    """
    Generate beat grid from guided tempo without running Madmom.
    This bypasses the brittle RNNBeatProcessor and sample rate issues.
    """

    def __init__(self):
        self._grid_cache: Dict[Tuple[float, str, float], GuidedBeatGrid] = {}

    def generate(
            self,
            tempo_bpm: float,
            time_signature: TimeSignature,
            duration_seconds: float,
            start_offset_ms: float = 0,
            swing_ratio: Optional[float] = None,
            confidence: Confidence = Confidence.HIGH
    ) -> GuidedBeatGrid:
        """
        Generate a beat grid from tempo and time signature.

        Args:
            tempo_bpm: Tempo in beats per minute
            time_signature: Time signature (e.g., 4/4, 3/4)
            duration_seconds: Total duration of audio in seconds
            start_offset_ms: Offset for first beat (ms)
            swing_ratio: Optional swing feel (0.5=straight, 0.67=triplet)
            confidence: Confidence level for this grid

        Returns:
            GuidedBeatGrid object with beat timestamps
        """
        # Check cache
        cache_key = (tempo_bpm, time_signature.value, duration_seconds)
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        # Calculate beat duration in ms
        beat_duration_ms = (60.0 / tempo_bpm) * 1000.0
        measure_duration_ms = beat_duration_ms * time_signature.beats_per_measure

        total_duration_ms = duration_seconds * 1000.0

        # Generate beat times
        beat_times: List[float] = []
        downbeat_times: List[float] = []

        current_beat = 0
        current_time_ms = start_offset_ms

        while current_time_ms <= total_duration_ms:
            beat_times.append(current_time_ms)

            # Check if this is a downbeat (first beat of measure)
            if current_beat % time_signature.beats_per_measure == 0:
                downbeat_times.append(current_time_ms)

            # Apply swing if specified (alternate long/short)
            if swing_ratio is not None and swing_ratio != 0.5:
                if current_beat % 2 == 1:  # Off-beats get stretched
                    current_time_ms += beat_duration_ms * swing_ratio
                else:
                    current_time_ms += beat_duration_ms * (2 - swing_ratio)
            else:
                current_time_ms += beat_duration_ms

            current_beat += 1

        grid = GuidedBeatGrid(
            beat_times_ms=beat_times,
            downbeat_times_ms=downbeat_times,
            measure_duration_ms=measure_duration_ms,
            beat_duration_ms=beat_duration_ms,
            tempo_bpm=tempo_bpm,
            time_signature=time_signature.value,
            source=SourceType.GUIDED,
            confidence=confidence,
            is_guided=True,
            guided_tempo_bpm=tempo_bpm,
            guided_time_signature=time_signature
        )

        self._grid_cache[cache_key] = grid
        return grid


# =================================================================
# PART 4: KEY-CONSTRAINED PITCH DETECTION
# =================================================================

# Note to MIDI mapping
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NUMBERS = {name: i for i, name in enumerate(NOTE_NAMES)}

# Scale definitions (intervals from root, 0-11)
SCALES: Dict[str, List[int]] = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    'blues': [0, 3, 5, 6, 7, 10],
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],
}


class KeyConstraint:
    """
    Key-based constraint for pitch detection.
    Filters note candidates to only those in the specified key.
    """

    def __init__(self, key_signature: str):
        """
        Initialize key constraint from key signature string.
        Examples: "C major", "C# minor", "Eb major", "F# minor"
        """
        self.key_string = key_signature
        self.root_note, self.scale_type = self._parse_key(key_signature)
        self.allowed_pitches: Set[int] = self._build_allowed_pitches()

    def _parse_key(self, key: str) -> Tuple[int, str]:
        """Parse key string to (root_midi, scale_type)."""
        parts = key.strip().lower().split()
        if len(parts) < 2:
            return 0, 'major'

        root_str = parts[0]
        scale_type = parts[1] if parts[1] in SCALES else 'major'

        # Handle flats
        flat_map = {
            'eb': 3, 'ab': 8, 'bb': 10, 'db': 1, 'gb': 6, 'cb': 11
        }
        if root_str in flat_map:
            root_midi = flat_map[root_str]
        elif root_str.endswith('#'):
            root_name = root_str[0].upper() + '#'
            root_midi = NOTE_NUMBERS.get(root_name, 0)
        else:
            root_name = root_str[0].upper() + (root_str[1:] if len(root_str) > 1 else '')
            root_midi = NOTE_NUMBERS.get(root_name, 0)

        return root_midi, scale_type

    def _build_allowed_pitches(self) -> Set[int]:
        """Build set of allowed MIDI note numbers (0-127) within this key."""
        intervals = SCALES.get(self.scale_type, SCALES['major'])
        allowed = set()

        # Generate allowed notes across all octaves
        for octave in range(0, 11):
            root_pitch = self.root_note + (octave * 12)
            for interval in intervals:
                pitch = root_pitch + interval
                if 0 <= pitch <= 127:
                    allowed.add(pitch)

        # Add relative minor if major, relative major if minor
        if 'major' in self.scale_type:
            relative_minor_root = (self.root_note - 3) % 12
            for octave in range(0, 11):
                root_pitch = relative_minor_root + (octave * 12)
                for interval in SCALES['minor']:
                    pitch = root_pitch + interval
                    if 0 <= pitch <= 127:
                        allowed.add(pitch)
        elif 'minor' in self.scale_type:
            relative_major_root = (self.root_note + 3) % 12
            for octave in range(0, 11):
                root_pitch = relative_major_root + (octave * 12)
                for interval in SCALES['major']:
                    pitch = root_pitch + interval
                    if 0 <= pitch <= 127:
                        allowed.add(pitch)

        return allowed

    def filter_notes(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """Filter notes to only those allowed in this key."""
        return [note for note in notes if note.pitch in self.allowed_pitches]

    def adjust_confidence(self, note: NoteEvent) -> float:
        """Return adjusted confidence based on key membership."""
        if note.pitch in self.allowed_pitches:
            # Boost for key-conformant notes
            return min(1.0, note.confidence * 1.1)
        else:
            # Penalize notes outside the key
            return note.confidence * 0.7

    def get_scale_notes(self, octave: int = 4) -> List[int]:
        """Get list of allowed MIDI notes in the specified octave."""
        root_pitch = self.root_note + (octave * 12)
        intervals = SCALES.get(self.scale_type, SCALES['major'])
        return [root_pitch + interval for interval in intervals]

    def get_confidence_for_pitch(self, pitch: int) -> Confidence:
        """Return Confidence enum for a pitch based on key membership."""
        if pitch in self.allowed_pitches:
            return Confidence.MEDIUM
        return Confidence.LOW


# =================================================================
# PART 5: GUIDED MODE CONTROLLER
# =================================================================

class GuidedModeController:
    """
    Main controller for guided mode functionality.
    Coordinates all bypass mechanisms: tempo, key, time signature.
    """

    def __init__(self):
        self.beat_generator = GuidedBeatGridGenerator()
        self._active_key_constraint: Optional[KeyConstraint] = None

    def should_bypass_tempo_detection(self, params: GuidedParams) -> bool:
        """Determine if we should skip tempo detection."""
        return params.has_tempo or params.has_custom_beat_grid

    def should_bypass_key_detection(self, params: GuidedParams) -> bool:
        """Determine if we should skip automatic key detection."""
        return params.has_key

    def should_bypass_time_signature_detection(self, params: GuidedParams) -> bool:
        """Determine if we should skip automatic time signature detection."""
        return params.has_time_signature

    def get_beat_grid(
            self,
            params: GuidedParams,
            duration_seconds: float
    ) -> Optional[GuidedBeatGrid]:
        """
        Generate beat grid from guided parameters.
        Returns None if not enough information.
        """
        if not params.has_tempo:
            return None

        time_sig = params.time_signature if params.has_time_signature else TimeSignature.FOUR_FOUR

        return self.beat_generator.generate(
            tempo_bpm=params.tempo_bpm,
            time_signature=time_sig,
            duration_seconds=duration_seconds,
            swing_ratio=params.swing_ratio,
            confidence=Confidence.HIGH if params.constant_tempo else Confidence.MEDIUM
        )

    def get_tempo_map(self, params: GuidedParams, duration_seconds: float) -> Optional[TempoMap]:
        """
        Generate TempoMap from guided parameters.
        Returns None if no tempo provided.
        """
        if not params.has_tempo:
            return None

        tempo_events = []
        if not params.constant_tempo:
            # For non-constant tempo with guided mode, we still need detection
            # But we can bootstrap with the guided tempo
            pass

        return TempoMap(
            initial_tempo_bpm=params.tempo_bpm,
            tempo_events=tempo_events,
            confidence=Confidence.HIGH if params.constant_tempo else Confidence.MEDIUM
        )

    def get_key_constraint(self, params: GuidedParams) -> Optional[KeyConstraint]:
        """Get key constraint for filtering notes."""
        if not params.has_key:
            return None
        return KeyConstraint(params.key_signature)

    def filter_notes_by_key(self, notes: List[NoteEvent], params: GuidedParams) -> List[NoteEvent]:
        """Filter notes to only those in the guided key."""
        if not params.has_key:
            return notes
        constraint = KeyConstraint(params.key_signature)
        return constraint.filter_notes(notes)

    def get_genre_settings(self, params: GuidedParams) -> Dict[str, Any]:
        """Get processing settings based on genre hint."""
        if not params.has_genre:
            return {}

        genre_configs = {
            GenreType.JAZZ: {
                'swing_default': 0.66,
                'complex_timing': True,
                'constant_tempo': False,
                'preferred_detectors': ['basic_pitch', 'madmom'],
                'preferred_sample_rate': PROFESSIONAL_FIDELITY_SR,  # 44100 for jazz
            },
            GenreType.ROCK: {
                'swing_default': 0.5,
                'complex_timing': False,
                'constant_tempo': True,
                'preferred_detectors': ['basic_pitch'],
                'preferred_sample_rate': 44100,
            },
            GenreType.CLASSICAL: {
                'swing_default': 0.5,
                'complex_timing': True,
                'constant_tempo': False,
                'preferred_detectors': ['basic_pitch'],
                'preferred_sample_rate': 44100,
            },
            GenreType.ELECTRONIC: {
                'swing_default': 0.5,
                'complex_timing': False,
                'constant_tempo': True,
                'preferred_detectors': ['madmom'],
                'preferred_sample_rate': 44100,
            },
            GenreType.HIP_HOP: {
                'swing_default': 0.55,
                'complex_timing': False,
                'constant_tempo': True,
                'preferred_detectors': ['basic_pitch'],
                'preferred_sample_rate': 44100,
            },
            GenreType.POP: {
                'swing_default': 0.5,
                'complex_timing': False,
                'constant_tempo': True,
                'preferred_detectors': ['basic_pitch'],
                'preferred_sample_rate': 44100,
            },
        }

        return genre_configs.get(params.genre, {})

    def create_pipeline_overrides(self, params: GuidedParams) -> Dict[str, Any]:
        """Create pipeline configuration overrides from guided params."""
        overrides = {
            'skip_tempo_detection': self.should_bypass_tempo_detection(params),
            'skip_key_detection': self.should_bypass_key_detection(params),
            'skip_time_signature_detection': self.should_bypass_time_signature_detection(params),
            'use_guided_beat_grid': params.has_tempo,
            'use_guided_key_constraint': params.has_key,
            'guided_tempo_bpm': params.tempo_bpm if params.has_tempo else None,
            'guided_key_signature': params.key_signature if params.has_key else None,
            'guided_time_signature': params.time_signature.value if params.time_signature else None,
            'constant_tempo_assumption': params.constant_tempo,
            'analyze_bass': params.analyze_bass,
            'analyze_drums': params.analyze_drums,
            'analyze_groove': params.analyze_groove,
        }

        # Add genre-specific settings
        genre_settings = self.get_genre_settings(params)
        overrides.update(genre_settings)

        return overrides

    def get_heartbeat_notes(self, context: AudioContext) -> List[NoteEvent]:
        """
        Generate heartbeat notes for empty results (Law 3 compliance).
        Returns a single barely-audible note at C4.
        """
        heartbeat = NoteEvent(
            pitch=HEARTBEAT_NOTE_PITCH,
            start_ms=0.0,
            end_ms=HEARTBEAT_NOTE_DURATION_MS,
            velocity=HEARTBEAT_NOTE_VELOCITY,
            confidence=Confidence.HALLUCINATION.value,  # Not a real note
            zero_crossing_rate=0.0,
            source=SourceType.GUIDED,
            reasoning_chain=["Heartbeat generated for empty transcription"]
        )
        return [heartbeat]

    def log_guided_usage(self, params: GuidedParams) -> List[str]:
        """Generate log messages about which guided features are active."""
        messages = []

        if params.has_tempo:
            messages.append(f"Guided Mode: tempo = {params.tempo_bpm} BPM (skipping Madmom)")
        if params.has_time_signature:
            sig = params.time_signature.value if hasattr(params.time_signature, 'value') else params.time_signature
            messages.append(f"Guided Mode: time signature = {sig}")
        if params.has_key:
            messages.append(f"Guided Mode: key = {params.key_signature} (constraining pitch detection)")
        if params.has_swing:
            messages.append(f"Guided Mode: swing ratio = {params.swing_ratio}")
        if params.has_genre:
            genre = params.genre.value if hasattr(params.genre, 'value') else params.genre
            messages.append(f"Guided Mode: genre = {genre}")

        if not messages:
            messages.append("Guided Mode: no overrides active (using auto-detection)")

        return messages


# =================================================================
# PART 6: HELPER FUNCTIONS
# =================================================================

def create_guided_params_from_form(
        tempo_bpm: Optional[float] = None,
        time_signature_num: Optional[int] = None,
        time_signature_den: Optional[int] = None,
        key_signature: Optional[str] = None,
        genre: Optional[str] = None,
        swing_ratio: Optional[float] = None
) -> GuidedParams:
    """Create GuidedParams from web form inputs."""
    time_sig = None
    if time_signature_num and time_signature_den:
        time_sig = TimeSignature.from_numbers(time_signature_num, time_signature_den)

    return GuidedParams(
        tempo_bpm=tempo_bpm,
        time_signature=time_sig,
        key_signature=key_signature,
        genre=genre,
        swing_ratio=swing_ratio
    )


def get_scale_notes(key_signature: str, octave: int = 4) -> List[int]:
    """Get the list of MIDI notes in the specified key."""
    constraint = KeyConstraint(key_signature)
    return constraint.get_scale_notes(octave)


def validate_guided_params(params: GuidedParams) -> Tuple[bool, Optional[str]]:
    """Validate guided parameters. Returns (is_valid, error_message)."""
    return params.validate()


# =================================================================
# PART 7: DEPRECATION WARNING SHIM
# =================================================================

def __getattr__(name: str):
    """
    Provide deprecation warnings for legacy names.
    """
    legacy_renames = {
        "TARGET_SAMPLE_RATE_GUIDED": "PROFESSIONAL_FIDELITY_SR",
    }

    if name in legacy_renames:
        warnings.warn(
            f"'{name}' is deprecated in Grimlock 5.0. "
            f"Use '{legacy_renames[name]}' from constants instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from core.constants import PROFESSIONAL_FIDELITY_SR
        return PROFESSIONAL_FIDELITY_SR

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")