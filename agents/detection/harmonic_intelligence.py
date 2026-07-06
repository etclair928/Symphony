# =================================================================
# MODULE: agents/detection/harmonic_intelligence.py
# DESCRIPTION: Harmonic/tonal detection as evidence interpretation.
#
# VERSION: 5.6.1 (FIXED: O(1) dictionary lookup, sanitization)
# UPDATED: 2026-05-15
#
# CRITICAL FIXES:
#   1. Dictionary-based partial lookup (O(1) not O(n²))
#   2. Input sanitization for audio slices
#   3. Proper feature bundle validation
# =================================================================

import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from functools import lru_cache

from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    PartialTrack, HarmonicSeries, EvidenceType, EvidenceLease,
    MemoryPriority, FeatureBundle
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    MIN_PITCH_MIDI,
    MAX_PITCH_MIDI,
    MIN_FREQUENCY_HZ,
    MAX_FREQUENCY_HZ,
    TONAL_CORRELATION_THRESHOLD,
    HARMONIC_SERIES_MATCH_THRESHOLD,
    MAX_INHARMONICITY,
    STAGGERED_GC_TRIGGER_MB,
    SPECTRAL_FLATNESS_TONAL_THRESHOLD
)
from core.protocols import (
    DetectionAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)


@dataclass
class HarmonicValidationResult:
    """Result of harmonic validation with witness-ready data."""
    is_valid: bool
    confidence: float
    partials_found: List[PartialTrack]
    inharmonicity: float
    zcr_profile: Dict[str, float]
    veto_reason: Optional[VetoReason] = None


# ========================================================================
# HarmonicValidator (Cent-based, Musical Tolerances)
# ========================================================================

class HarmonicValidator:
    """
    Harmonic series validator with musical cent-based tolerance.
    Pure function - no state. Consumes FeatureBundle evidence.
    """

    def __init__(self, config: 'HarmonicConfig',
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def _sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Sanitize audio slice."""
        if np.any(~np.isfinite(audio)):
            return np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return audio

    def validate(self, note: NoteEvent, features: FeatureBundle) -> HarmonicValidationResult:
        """Validate a candidate note against harmonic evidence."""
        harmonic_evidence = features.get_harmonic_evidence(note)

        if not harmonic_evidence.get('partials'):
            return HarmonicValidationResult(
                is_valid=False,
                confidence=0.0,
                partials_found=[],
                inharmonicity=1.0,
                zcr_profile={},
                veto_reason=VetoReason.HARMONIC_SERIES_FAILURE
            )

        fundamental_freq_hz = note.fundamental_freq_hz or self._pitch_to_hz(note.pitch)
        partials = self._detect_partials_from_evidence(harmonic_evidence, fundamental_freq_hz)
        inharmonicity = self._calculate_weighted_inharmonicity(partials, fundamental_freq_hz)
        zcr_profile = self._analyze_zcr_profile(note, features)

        is_valid = len(partials) >= self.config.min_partials_required
        is_valid = is_valid and inharmonicity < MAX_INHARMONICITY

        veto_reason = None
        if not is_valid:
            veto_reason = VetoReason.HARMONIC_SERIES_FAILURE
        elif zcr_profile.get('verdict') == 'noise':
            veto_reason = VetoReason.HALLUCINATED_NOTE

        confidence = len(partials) / self.config.max_harmonics
        confidence *= (1 - inharmonicity)
        confidence *= (1 - zcr_profile.get('noise_ratio', 0))

        # Corroborate against polyphonic_peaks (real local-maxima detection
        # on the raw magnitude spectrogram, independent of the CQT-based
        # partial detection above). Soft signal, not a hard veto - some
        # legitimate sustained/soft-onset tones have a smooth spectral
        # envelope with no sharp local maximum, so failing to find one
        # nudges confidence down rather than invalidating the note.
        if not features.is_fundamental_prominent_peak(note):
            confidence *= 0.85

        return HarmonicValidationResult(
            is_valid=is_valid,
            confidence=min(1.0, max(0.0, confidence)),
            partials_found=partials,
            inharmonicity=inharmonicity,
            zcr_profile=zcr_profile,
            veto_reason=veto_reason
        )

    def _detect_partials_from_evidence(self, harmonic_evidence: Dict[str, Any],
                                       fundamental_freq_hz: float) -> List[PartialTrack]:
        """
        Detect partials using pre-computed harmonic_network.
        O(1) dictionary lookup - FIXED from O(n²) nested loops.

        Takes the already-resolved fundamental_freq_hz (real value or
        pitch-derived fallback) rather than reading note.fundamental_freq_hz
        directly - this used to bypass the fallback validate() otherwise
        applies, so a note with no fundamental_freq_hz set would always
        compute target_freq = 0 * harmonic = 0 and never match anything.
        """
        partials = []
        partial_data = harmonic_evidence.get('partials', [])

        if not partial_data:
            return partials

        # CRITICAL FIX: Convert to dictionary for O(1) lookup
        partial_dict = {}
        for p_data in partial_data:
            partial_num = p_data.get('partial_number')
            if partial_num is not None:
                partial_dict[partial_num] = p_data

        for harmonic in range(1, self.config.max_harmonics + 1):
            target_freq = fundamental_freq_hz * harmonic

            # O(1) dictionary lookup instead of O(n) linear scan
            matching_partial = partial_dict.get(harmonic)

            if matching_partial:
                detected_freq = matching_partial.get('frequency_hz', 0)

                if detected_freq > 0 and target_freq > 0:
                    cent_deviation = 1200 * np.log2(detected_freq / target_freq)

                    if harmonic <= 3:
                        tolerance_cents = 30.0
                    elif harmonic <= 5:
                        tolerance_cents = 50.0
                    else:
                        tolerance_cents = 70.0

                    if abs(cent_deviation) <= tolerance_cents:
                        confidence = 1.0 - (abs(cent_deviation) / tolerance_cents)
                        confidence *= (0.7 ** (harmonic - 1))

                        partials.append(PartialTrack(
                            partial_number=harmonic,
                            frequency_hz=detected_freq,
                            amplitude_db=matching_partial.get('amplitude_db', -60),
                            confidence=float(confidence)
                        ))

        return partials

    def _calculate_weighted_inharmonicity(self, partials: List[PartialTrack],
                                          fundamental_freq_hz: float) -> float:
        """
        Weighted inharmonicity - power partials matter more.

        Compares each partial's actual measured frequency against where
        it SHOULD sit for a perfect harmonic series (partial_number *
        fundamental) - not against itself. The previous version computed
        `expected = partial_number` then `deviation = |freq - expected *
        (freq / partial_number)|`, which algebraically always reduces to
        `|freq - freq| == 0` regardless of any real data, so this always
        reported perfectly harmonic (0.0) and the inharmonicity gate
        never rejected anything and never penalized confidence.
        """
        if not partials or fundamental_freq_hz <= 0:
            return 1.0

        weighted_sum = 0.0
        total_weight = 0.0

        for p in partials:
            if p.partial_number in [2, 3]:
                weight = 3.0
            elif p.partial_number >= 7:
                weight = 0.5
            else:
                weight = 1.0

            expected_freq_hz = p.partial_number * fundamental_freq_hz
            deviation = abs(p.frequency_hz - expected_freq_hz)
            inharmonicity_ratio = deviation / expected_freq_hz if expected_freq_hz > 0 else 0.0

            weighted_sum += weight * inharmonicity_ratio
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 1.0

    @staticmethod
    def _pitch_to_hz(pitch: int) -> float:
        """MIDI pitch -> frequency, used when a note has no fundamental_freq_hz of its own."""
        return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

    def _analyze_zcr_profile(self, note: NoteEvent, features: FeatureBundle) -> Dict[str, float]:
        """Analyze ZCR profile over note duration."""
        onset_idx = np.searchsorted(features.time_ms, note.start_ms)
        offset_idx = np.searchsorted(features.time_ms, note.end_ms)

        if offset_idx <= onset_idx:
            return {"verdict": "insufficient", "noise_ratio": 1.0}

        note_zcr = features.zcr[onset_idx:offset_idx]

        if len(note_zcr) < 3:
            return {"verdict": "insufficient", "noise_ratio": 0.5}

        onset_zcr = note_zcr[0]
        sustained_zcr = np.mean(note_zcr[1:])
        zcr_ratio = onset_zcr / (sustained_zcr + 1e-8)

        if onset_zcr > 0.3 and sustained_zcr < 0.2:
            verdict = "slap_bass"
            noise_ratio = 0.1
        elif onset_zcr > 0.3 and sustained_zcr > 0.3:
            verdict = "noise"
            noise_ratio = 0.9
        else:
            verdict = "tonal"
            noise_ratio = 0.0

        return {
            "onset_zcr": float(onset_zcr),
            "sustained_zcr": float(sustained_zcr),
            "zcr_ratio": float(zcr_ratio),
            "verdict": verdict,
            "noise_ratio": noise_ratio
        }


# ========================================================================
# Key Detector (Jazz-Aware, Dynamic Penalties)
# ========================================================================

class KeyDetector:
    """Key detection with jazz-aware dynamic penalties."""

    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    KEYS_MAJOR = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
    KEYS_MINOR = ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'Bbm', 'Fm', 'Cm', 'Gm', 'Dm']

    def __init__(self, config: 'HarmonicConfig',
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self._key_profile_cache = {}

    NOTE_TO_PITCH_CLASS = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
        'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    }

    def detect_key(self, chroma: np.ndarray) -> Tuple[str, float]:
        """
        Detect key from chromagram evidence.

        A relative major/minor pair (e.g. C major / A minor) shares the
        exact same 7 pitch classes, so a correlation match this close
        can genuinely be a coin flip on the raw profile alone - this is
        the single most common failure mode plain Krumhansl-Schmuckler
        correlation runs into. Two real, corroborating signals now help
        disambiguate rather than trusting the top correlation blindly:

        - Weighting the piece's opening and closing sections more
          heavily than the middle when averaging chroma - both are
          strongly correlated with the true tonal center in tonal
          music, whereas naive full-track averaging can wash that
          emphasis out entirely.
        - When the top two candidates are a relative major/minor pair
          with a correlation gap too small to be meaningful, checking
          for the minor key's raised leading tone (from harmonic/
          melodic minor - real minor-key music emphasizes it for
          cadential motion) against its natural (Aeolian) 7th, a
          distinction the shared 7-note profile can't make on its own.
        """
        if chroma is None or chroma.size == 0:
            return "Cm", 0.0

        avg_chroma = self._weighted_chroma_mean(chroma)

        if np.sum(avg_chroma) > 0:
            avg_chroma = avg_chroma / np.sum(avg_chroma)

        candidates: List[Tuple[float, str, bool, int]] = []
        for key_idx, key in enumerate(self.KEYS_MAJOR):
            correlation = self._correlate_with_key(avg_chroma, self.MAJOR_PROFILE, key_idx)
            candidates.append((correlation, key, False, key_idx))
        for key_idx, key in enumerate(self.KEYS_MINOR):
            correlation = self._correlate_with_key(avg_chroma, self.MINOR_PROFILE, key_idx)
            candidates.append((correlation, key, True, key_idx))

        candidates.sort(key=lambda c: -c[0])
        best_correlation, best_key, best_is_minor, best_idx = candidates[0]

        # KEYS_MAJOR[i] and KEYS_MINOR[i] are always a relative pair
        # (same key_idx = same 7 natural notes) - check the runner-up
        # for exactly this ambiguity before trusting the raw winner.
        for correlation, key, is_minor, key_idx in candidates[1:]:
            if key_idx != best_idx or is_minor == best_is_minor:
                continue
            if abs(correlation - best_correlation) < 0.05:
                major_key, minor_key = (best_key, key) if not best_is_minor else (key, best_key)
                resolved_key, resolved_is_minor = self._break_relative_tie(
                    avg_chroma, major_key, minor_key)
                if resolved_is_minor != best_is_minor:
                    # Tie-break flipped the decision to the runner-up.
                    best_key = resolved_key
                    best_correlation = correlation
            break

        confidence = max(0.0, min(1.0, (best_correlation + 0.5) / 1.5))

        return best_key, confidence

    def _weighted_chroma_mean(self, chroma: np.ndarray) -> np.ndarray:
        """
        Average chroma frames with extra weight on the piece's opening
        and closing sections, both strongly correlated with the true
        tonal center in tonal music - a big part of why naive full-track
        averaging can mistake a piece for its relative major/minor.
        """
        n_frames = chroma.shape[1] if chroma.ndim > 1 else 1
        if n_frames < 10:
            return np.mean(chroma, axis=1) if chroma.ndim > 1 else chroma

        weights = np.ones(n_frames)
        edge = max(1, int(n_frames * 0.15))
        weights[:edge] *= 2.0
        weights[-edge:] *= 2.0

        weighted = chroma * weights[np.newaxis, :]
        return np.sum(weighted, axis=1) / np.sum(weights)

    def _break_relative_tie(
            self, avg_chroma: np.ndarray, major_key: str, minor_key: str
    ) -> Tuple[str, bool]:
        """
        major_key and minor_key are a relative pair (identical 7 natural
        notes). Check the minor key's raised leading tone (harmonic/
        melodic minor's distinctive 7th) against its natural (Aeolian)
        7th - meaningfully more of the former is real evidence of actual
        minor-key harmonic usage (dominant/cadential motion), which nothing
        in the shared natural-scale profile could otherwise reveal.
        """
        minor_tonic_pc = self.NOTE_TO_PITCH_CLASS.get(minor_key.rstrip('m'))
        if minor_tonic_pc is None:
            return major_key, False

        raised_leading_tone_pc = (minor_tonic_pc + 11) % 12
        natural_seventh_pc = (minor_tonic_pc + 10) % 12

        raised = avg_chroma[raised_leading_tone_pc]
        natural = avg_chroma[natural_seventh_pc]

        if raised > natural * 1.15:
            return minor_key, True
        return major_key, False

    def _correlate_with_key(self, chroma: np.ndarray, profile: np.ndarray,
                            key_idx: int) -> float:
        """Pearson correlation with rotated key profile."""
        rotated_profile = np.roll(profile, -key_idx)

        chroma_mean = np.mean(chroma)
        profile_mean = np.mean(rotated_profile)

        numerator = np.sum((chroma - chroma_mean) * (rotated_profile - profile_mean))
        denominator = np.sqrt(np.sum((chroma - chroma_mean) ** 2) *
                              np.sum((rotated_profile - profile_mean) ** 2))

        return numerator / denominator if denominator > 0 else 0.0

    def get_key_weight(self, note: NoteEvent, key: str, key_confidence: float) -> float:
        """Get confidence weight for a note based on key."""
        if key_confidence < self.config.key_confidence_threshold:
            return 1.0

        pitch_class = note.pitch % 12
        is_in_key = self._is_pitch_in_key(pitch_class, key)

        if is_in_key:
            return 1.05
        else:
            if key_confidence > 0.8:
                return 0.90
            elif key_confidence > 0.65:
                return 0.95
            else:
                return 0.98

    def _is_pitch_in_key(self, pitch_class: int, key: str) -> bool:
        if key not in self._key_profile_cache:
            self._key_profile_cache[key] = self._build_key_profile(key)
        return pitch_class in self._key_profile_cache[key]

    def _build_key_profile(self, key: str) -> set:
        major_profiles = {
            'C': {0, 2, 4, 5, 7, 9, 11},
            'G': {7, 9, 11, 0, 2, 4, 6},
            'D': {2, 4, 6, 7, 9, 11, 1},
            'A': {9, 11, 1, 2, 4, 6, 8},
            'E': {4, 6, 8, 9, 11, 1, 3},
            'B': {11, 1, 3, 4, 6, 8, 10},
            'F#': {6, 8, 10, 11, 1, 3, 5},
            'Db': {1, 3, 5, 6, 8, 10, 0},
            'Ab': {8, 10, 0, 1, 3, 5, 7},
            'Eb': {3, 5, 7, 8, 10, 0, 2},
            'Bb': {10, 0, 2, 3, 5, 7, 9},
            'F': {5, 7, 9, 10, 0, 2, 4},
        }

        minor_relative = {
            'Am': 'C', 'Em': 'G', 'Bm': 'D', 'F#m': 'A', 'C#m': 'E', 'G#m': 'B',
            'D#m': 'F#', 'Bbm': 'Db', 'Fm': 'Ab', 'Cm': 'Eb', 'Gm': 'Bb', 'Dm': 'F'
        }

        if key in major_profiles:
            return major_profiles[key]
        elif key in minor_relative:
            relative_major = minor_relative[key]
            major_set = major_profiles.get(relative_major, {0, 2, 4, 5, 7, 9, 11})
            return {(pc - 3) % 12 for pc in major_set}

        return {0, 2, 4, 5, 7, 9, 11}


# ========================================================================
# Immutable Note Transformer
# ========================================================================

class NoteTransformer:
    """Immutable note transformations with forensic tracking."""

    @staticmethod
    def apply_key_context(note: NoteEvent, key: str, key_confidence: float,
                          key_weight: float) -> NoteEvent:
        new_confidence = note.confidence * key_weight

        reasoning = list(note.reasoning_chain) if note.reasoning_chain else []
        reasoning.append(f"Key context: {key} (conf={key_confidence:.2f}, weight={key_weight:.2f})")

        return replace(
            note,
            confidence=min(1.0, new_confidence),
            reasoning_chain=reasoning
        )

    @staticmethod
    def apply_harmonic_evidence(note: NoteEvent, validation: HarmonicValidationResult) -> NoteEvent:
        new_confidence = note.confidence * validation.confidence

        reasoning = list(note.reasoning_chain) if note.reasoning_chain else []
        reasoning.append(f"Validator: {len(validation.partials_found)} partials, "
                         f"conf={validation.confidence:.2f}")

        return replace(
            note,
            confidence=min(1.0, new_confidence),
            harmonic_series_match_ratio=validation.confidence,
            reasoning_chain=reasoning
        )

    @staticmethod
    def create_testimony(note: NoteEvent, validation: HarmonicValidationResult,
                         key: str, key_confidence: float) -> WitnessTestimony:
        return WitnessTestimony(
            witness_id=SourceType.TONAL,
            testimony_time_ms=note.start_ms,
            events=[note] if validation.is_valid else [],
            confidence=validation.confidence * key_confidence,
            metadata={
                "partials_found": len(validation.partials_found),
                "inharmonicity": validation.inharmonicity,
                "zcr_profile": validation.zcr_profile,
                "detected_key": key,
                "key_confidence": key_confidence,
                "veto_reason": validation.veto_reason.value if validation.veto_reason else None
            }
        )


# ========================================================================
# Harmonic Intelligence Agent
# ========================================================================

@dataclass
class HarmonicConfig:
    """Configuration for harmonic intelligence."""
    hop_length: int = 512
    n_fft: int = 4096
    min_freq_hz: float = MIN_FREQUENCY_HZ
    max_freq_hz: float = MAX_FREQUENCY_HZ
    max_harmonics: int = 8
    min_partials_required: int = 2
    detect_key: bool = True
    key_confidence_threshold: float = 0.5
    min_note_duration_ms: float = 40.0
    max_gap_ms: float = 50.0
    cleanup_after_detection: bool = True


class HarmonicIntelligence:
    """Harmonic/tonal detection agent for Grimlock 5.0."""

    def __init__(
            self,
            config: Optional[HarmonicConfig] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        self._name = "harmonic_intelligence"
        self._source_type = SourceType.TONAL
        self._config = config or HarmonicConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._progress_callback = progress_callback
        self._detection_threshold = Confidence.LOW.value

        self._validator = HarmonicValidator(self._config, status_reporter)
        self._key_detector = KeyDetector(self._config, status_reporter)
        self._transformer = NoteTransformer()

        self._last_detected_key: Optional[str] = None
        self._last_key_confidence: float = 0.0
        self._last_validation_count: int = 0
        self._last_execution_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0
        self._active_leases: Dict[str, EvidenceLease] = {}

        self._log_status("HarmonicIntelligence initialized", "info")

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def name(self) -> str:
        return self._name

    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """Process audio using SHARED FeatureBundle evidence."""
        start_time = time.time()
        start_memory = self._get_current_memory_mb()

        self._update_progress(0.0, "Starting harmonic detection")
        self._log_status("Running harmonic intelligence", "info")

        try:
            if not hasattr(context, 'feature_bundle') or context.feature_bundle is None:
                self._log_status("No FeatureBundle in context - cannot proceed", "error")
                return StageResult(
                    stage_name=self._name,
                    success=False,
                    events=[],
                    metadata={"error": "Missing FeatureBundle"},
                    veto_reason=VetoReason.EMPTY_RESULT,
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            features: FeatureBundle = context.feature_bundle

            self._acquire_evidence_leases(features)

            self._update_progress(0.2, "Detecting key")
            if self._config.detect_key:
                self._last_detected_key, self._last_key_confidence = self._key_detector.detect_key(
                    features.chroma
                )
                self._log_status(f"Detected key: {self._last_detected_key} "
                                 f"(confidence {self._last_key_confidence:.2f})", "info")
            else:
                self._last_detected_key = None
                self._last_key_confidence = 0.0

            candidates = self._get_candidates_from_context(context)

            self._update_progress(0.4, f"Validating {len(candidates)} candidates")

            testimonies = []
            validated_count = 0

            for idx, candidate in enumerate(candidates):
                if idx % 100 == 0:
                    progress = 0.4 + (0.4 * idx / max(len(candidates), 1))
                    self._update_progress(progress, f"Validating note {idx + 1}/{len(candidates)}")

                validation = self._validator.validate(candidate, features)

                key_weight = self._key_detector.get_key_weight(
                    candidate, self._last_detected_key, self._last_key_confidence
                )

                note = self._transformer.apply_key_context(
                    candidate, self._last_detected_key, self._last_key_confidence, key_weight
                )
                note = self._transformer.apply_harmonic_evidence(note, validation)

                testimony = self._transformer.create_testimony(
                    note, validation, self._last_detected_key, self._last_key_confidence
                )
                testimonies.append(testimony)

                if validation.is_valid:
                    validated_count += 1

            self._update_progress(0.85, f"Validated {validated_count} of {len(candidates)} notes")
            self._last_validation_count = validated_count

            events = []
            for testimony in testimonies:
                events.extend(testimony.events)

            events = self._post_process_events(events)

            self._release_evidence_leases(features)

            if self._config.cleanup_after_detection:
                self._force_cleanup(features)

            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_current_memory_mb() - start_memory
            self._last_execution_time_ms = execution_time_ms

            if self._music_box:
                self._music_box.log_decision(
                    stage_name=self._name,
                    decision_type="harmonic_validation",
                    before_state={"candidate_count": len(candidates)},
                    after_state={
                        "event_count": len(events),
                        "validated_count": validated_count,
                        "detected_key": self._last_detected_key,
                        "key_confidence": self._last_key_confidence
                    },
                    reasoning=f"Validated {validated_count} of {len(candidates)} harmonic events",
                    reversible=True
                )

            self._update_progress(1.0, f"Complete: {len(events)} validated notes")

            return StageResult(
                stage_name=self._name,
                success=len(events) > 0,
                events=events,
                metadata={
                    "event_count": len(events),
                    "validated_count": validated_count,
                    "candidate_count": len(candidates),
                    "detected_key": self._last_detected_key,
                    "key_confidence": self._last_key_confidence,
                    "execution_time_ms": execution_time_ms,
                    "memory_delta_mb": memory_delta_mb,
                    "total_memory_freed_mb": self._total_memory_freed_mb
                },
                execution_time_ms=execution_time_ms,
                memory_delta_mb=memory_delta_mb
            )

        except Exception as e:
            self._log_status(f"Detection failed: {e}", "error")
            if self._status_reporter:
                self._status_reporter.error(self._name, str(e))

            return StageResult(
                stage_name=self._name,
                success=False,
                events=[],
                metadata={"error": str(e)},
                veto_reason=VetoReason.EMPTY_RESULT,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def _get_candidates_from_context(self, context: AudioContext) -> List[NoteEvent]:
        if hasattr(context, 'pitch_candidates') and context.pitch_candidates:
            return context.pitch_candidates
        return []

    def _acquire_evidence_leases(self, features: FeatureBundle):
        needed_evidence = [
            EvidenceType.STFT, EvidenceType.CHROMA, EvidenceType.HARMONIC_NETWORK
        ]
        for et in needed_evidence:
            if et not in features.evidence_leases:
                lease = EvidenceLease(
                    evidence_type=et,
                    owner_stage=self._name,
                    acquired_at=time.time(),
                    expires_at=time.time() + 300,
                    priority=MemoryPriority.PERSISTENT
                )
                self._active_leases[et.name] = lease

    def _release_evidence_leases(self, features: FeatureBundle):
        self._active_leases.clear()

    def _post_process_events(self, events: List[NoteEvent]) -> List[NoteEvent]:
        if not events:
            return events

        events = [e for e in events if e.duration_ms() >= self._config.min_note_duration_ms]
        events = [e for e in events if e.confidence >= self._detection_threshold]
        events = [e for e in events if MIN_PITCH_MIDI <= e.pitch <= MAX_PITCH_MIDI]
        events = self._merge_same_pitch_notes(events)

        return events

    def _merge_same_pitch_notes(self, events: List[NoteEvent]) -> List[NoteEvent]:
        if len(events) < 2:
            return events

        events.sort(key=lambda e: e.start_ms)
        merged = []
        current = events[0]

        for next_event in events[1:]:
            if next_event.pitch == current.pitch:
                gap = next_event.start_ms - current.end_ms
                if gap < self._config.max_gap_ms:
                    current = replace(
                        current,
                        end_ms=max(current.end_ms, next_event.end_ms),
                        confidence=max(current.confidence, next_event.confidence),
                        velocity=max(current.velocity, next_event.velocity)
                    )
                else:
                    merged.append(current)
                    current = next_event
            else:
                merged.append(current)
                current = next_event

        merged.append(current)
        return merged

    def _force_cleanup(self, features: FeatureBundle):
        if features.memory_mb > STAGGERED_GC_TRIGGER_MB:
            features.release(EvidenceType.CQT)
            features.release(EvidenceType.CHROMA)

        before = self._get_current_memory_mb()
        gc.collect()
        after = self._get_current_memory_mb()
        self._total_memory_freed_mb += max(0, before - after)

    def _update_progress(self, progress: float, message: str):
        if self._progress_callback:
            self._progress_callback(progress, message)
        if self._status_reporter:
            self._status_reporter.progress(self._name, progress, message)

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info(self._name, message)
            elif level == "warn":
                self._status_reporter.warn(self._name, message)
            elif level == "error":
                self._status_reporter.error(self._name, message)

    def _get_current_memory_mb(self) -> float:
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def validate(self, gate: ValidationGate) -> ValidationResult:
        if gate == ValidationGate.SCHOENBERG_MIRROR:
            if self._last_validation_count == 0:
                return ValidationResult(
                    is_valid=False,
                    gate_used=gate,
                    reason=VetoReason.HARMONIC_SERIES_FAILURE,
                    detail="No harmonic series validated",
                    events_validated=0,
                    events_rejected=0
                )
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                events_validated=self._last_validation_count,
                events_rejected=0
            )
        elif gate == ValidationGate.CONFIDENCE_THRESHOLD:
            is_valid = self._last_key_confidence >= self._config.key_confidence_threshold
            return ValidationResult(
                is_valid=is_valid,
                gate_used=gate,
                reason=None if is_valid else VetoReason.CONFIDENCE_TOO_LOW,
                detail=f"Key confidence: {self._last_key_confidence:.2f}",
                confidence_before=self._last_key_confidence,
                confidence_after=self._last_key_confidence if is_valid else 0.0
            )
        else:
            return ValidationResult(
                is_valid=True,
                gate_used=gate,
                detail=f"Gate {gate.value} not fully supported",
                events_validated=self._last_validation_count
            )

    def get_confidence(self) -> Confidence:
        return Confidence.from_float(self._last_key_confidence)

    def get_veto_status(self) -> Optional[Tuple[VetoReason, str]]:
        if self._last_validation_count == 0:
            return (VetoReason.HARMONIC_SERIES_FAILURE, "No harmonic series validated")
        if self._last_key_confidence < self._config.key_confidence_threshold:
            return (VetoReason.CONFIDENCE_TOO_LOW,
                    f"Key confidence {self._last_key_confidence:.2f}")
        return None

    def apply_schoenberg_mirror(self) -> SchoenbergResult:
        if self._last_validation_count == 0:
            return SchoenbergResult(
                verdict=SchoenbergVerdict.NOISE,
                zero_crossing_rate=0.0,
                spectral_flatness=0.7,
                reason="No harmonic series detected"
            )
        return SchoenbergResult(
            verdict=SchoenbergVerdict.TONAL,
            zero_crossing_rate=0.0,
            spectral_flatness=SPECTRAL_FLATNESS_TONAL_THRESHOLD,
            reason=f"Validated {self._last_validation_count} notes with harmonic series"
        )

    @property
    def detection_threshold(self) -> Confidence:
        return Confidence.LOW

    def get_raw_confidence(self) -> float:
        return self._last_key_confidence

    def release_buffer(self, buffer_name: str) -> None:
        pass

    def get_memory_footprint_mb(self) -> float:
        return self._get_current_memory_mb()

    def can_release(self, buffer_name: str) -> bool:
        return True

    def staggered_gc(self) -> Dict[str, Any]:
        before = self._get_current_memory_mb()
        gc.collect()
        after = self._get_current_memory_mb()
        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": before - after,
            "triggered_by": self._name,
            "total_freed_mb": self._total_memory_freed_mb
        }

    def get_detected_key(self) -> Optional[str]:
        return self._last_detected_key

    def get_key_confidence(self) -> float:
        return self._last_key_confidence

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type.value,
            "detected_key": self._last_detected_key,
            "key_confidence": self._last_key_confidence,
            "validation_count": self._last_validation_count,
            "last_execution_time_ms": self._last_execution_time_ms,
            "total_memory_freed_mb": self._total_memory_freed_mb
        }


def create_harmonic_intelligence(
        detect_key: bool = True,
        min_partials_required: int = 2,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> HarmonicIntelligence:
    config = HarmonicConfig(
        detect_key=detect_key,
        min_partials_required=min_partials_required
    )
    return HarmonicIntelligence(
        config=config,
        status_reporter=status_reporter,
        music_box=music_box
    )