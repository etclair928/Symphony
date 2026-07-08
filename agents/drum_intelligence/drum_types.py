# =================================================================
# MODULE: agents/drum_intelligence/drum_types.py
# VERSION: 6.0-draft
# DESCRIPTION: Shared enums, event/config dataclasses for the drum
# intelligence package. No logic lives here - just the vocabulary every
# other module in this package shares.
#
# REPLACES: agents/detection/drum_intelligence.py's DrumArticulation,
# DrumMicrotiming, DetectionPass, ConfidenceComponents, ForensicDrumEvent,
# DrumDetectionConfig, and agents/detection/drum_coprocessor.py's
# CoprocessorConfig, DrumTypeVote.
#
# DrumDetectionConfig and CoprocessorConfig are merged into one
# DrumIntelligenceConfig here. The coprocessor was explicitly built as a
# bolt-on "second, MIR-technique-based analytical layer... runs ALONGSIDE"
# the original classifier (its own module docstring's words) - a
# reasonable way to add NMF/HFC without disturbing what already worked at
# the time. Now that this package restructures NMF/HFC as native parts of
# onset_detection.py/classification.py rather than a second bolted-on
# system, there's no reason for their configuration to live in a
# separate dataclass either.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional

import numpy as np

from core.order_types import NoteEvent, SourceType, DrumType, DrumKitMapping, WitnessVote
from core.constants import (
    TARGET_SAMPLE_RATE,
    DRUM_CONFIDENCE_MIN,
    DRUM_GHOST_NOTE_VELOCITY_MAX,
    DRUM_FLAM_DETECTION_WINDOW_MS,
    DRUM_ROLL_DETECTION_MIN_HITS,
    DRUM_ROLL_MAX_INTERVAL_MS,
    SPICE_MODEL_SIZE,
)


# ========================================================================
# Enums
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
    INTERPRETED = "interpreted"
    FORENSIC = "forensic"
    EXPORT = "export"


# ========================================================================
# Event + confidence dataclasses
# ========================================================================

@dataclass(frozen=True)
class ConfidenceComponents:
    """Forensic confidence breakdown - immutable."""
    transient_strength: float = 0.0
    spectral_match: float = 0.0
    temporal_consistency: float = 0.0
    model_confidence: float = 0.0
    ensemble_agreement: float = 0.0
    # New: how well this hit's metrical position (from interpretation.py)
    # matches what's typical for its drum type - 0.5 (neutral) when no
    # metrical context was available to judge against, not "unknown/0".
    metrical_fit: float = 0.5

    def total(self) -> float:
        return float(np.mean([
            self.transient_strength,
            self.spectral_match,
            self.temporal_consistency,
            self.model_confidence,
            self.ensemble_agreement,
            self.metrical_fit,
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
    # Which position-in-bar (0-indexed, mod the real detected meter
    # numerator) this hit landed on, and the phase-aligned confidence that
    # went with it - populated by interpretation.py's metrical evidence,
    # None when no reliable meter/downbeat-phase was available to judge
    # against. This is the actual per-event home for task #83's finding
    # (kick-on-1, cross-stick-on-4), not just a diagnostic side-script.
    metrical_position: Optional[int] = None
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


@dataclass(frozen=True)
class DrumTypeVote:
    """One source's vote about what type a single hit is - consumed by
    consensus.py's contention resolution when two sources disagree about
    the same physical event."""
    source: str
    drum_type: DrumType
    confidence: float
    weight: float = 1.0


# ========================================================================
# Unified configuration
# ========================================================================

@dataclass
class DrumIntelligenceConfig:
    """Configuration for the whole drum intelligence package - one config,
    not two. Every threshold here has a real reason documented at the
    call site that actually uses it (see onset_detection.py/
    classification.py/articulation.py/interpretation.py docstrings)."""

    # --- General / SPICE (classification.py) ---
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
    sanitize_input: bool = True

    # --- Onset detection tuning (onset_detection.py) ---
    onset_delta: float = 0.12
    onset_wait_frames: int = 10
    dedup_window_ms: float = 20.0

    # --- NMF polyphony recovery (classification.py) ---
    nmf_enabled: bool = True
    nmf_max_components: int = 4
    nmf_iterations: int = 60
    nmf_polyphony_window_ms: float = 15.0
    nmf_min_bootstrap_examples: int = 2
    nmf_template_slice_ms: float = 40.0
    nmf_activation_threshold: float = 0.15
    nmf_min_energy_ratio: float = 0.3

    # --- HFC onset detection, mid/high bands only (onset_detection.py) ---
    hfc_enabled: bool = True
    hfc_n_fft: int = 1024
    hfc_hop_length: int = 256

    # --- Adaptive median-filtered thresholding (onset_detection.py) ---
    median_threshold_enabled: bool = True
    median_window_frames: int = 31
    median_alpha: float = 1.25
    median_beta: float = 0.02

    # --- Envelope-based articulation refinement (articulation.py) ---
    envelope_flam_gap_required: bool = True
    flam_dip_ratio_threshold: float = 0.6
    buzz_zcr_window_ms: float = 80.0
    buzz_zcr_variance_threshold: float = 0.002
    buzz_zcr_percentile: float = 60.0

    # --- Metrical-position interpretation (interpretation.py, NEW) ---
    # Whether classification confidence should be adjusted at all by
    # position-in-bar evidence - kept toggleable since it's new and
    # depends on a real detected meter being available at all.
    metrical_evidence_enabled: bool = True
    # How much a strong metrical-position match/mismatch can shift
    # confidence - deliberately a soft nudge (see interpretation.py),
    # never a hard override of what classification actually measured.
    metrical_fit_weight: float = 0.15

    def __post_init__(self):
        assert self.nmf_max_components >= 2, "nmf_max_components must allow at least 2 simultaneous types"
        assert self.nmf_min_bootstrap_examples >= 1
        assert self.median_window_frames >= 3, "median_window_frames must be at least 3"
        if self.median_window_frames % 2 == 0:
            self.median_window_frames += 1
