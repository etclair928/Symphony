# =================================================================
# MODULE: agents/quantization/temporal_lattice.py
# VERSION: 5.6.1
#
# BUG 5 FIX:
# PROBLEM: phase_delta_ms from ReverseGeoCrypt/GrooveField is absolute
#          cyclic phase (e.g., -1211ms at 152 BPM = ~3 beats early)
#          but resolve() was applying it as direct grid offset.
#
# FIX: Normalize phase delta to [-beat_duration/2, +beat_duration/2]
#      before applying any grid calculations.
# =================================================================

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, replace
from enum import Enum

from core.order_types import (
    NoteEvent, GrooveField, TempoMap, PulseField, Confidence,
    SourceType, QuantizationStrategy
)
from core.constants import (
    MIN_TEMPO_BPM,
    MAX_TEMPO_BPM,
    RITORNELLO_MAX_SNAP_MS,
    PRESERVE_ORIGINAL_MS_ALWAYS
)


# ========================================================================
# Enums and Types
# ========================================================================

class WitnessPriority(str, Enum):
    """Which witness has authority for which timing aspect."""
    TEMPO_ANCHOR = "tempo_anchor"
    GEOMETRY = "geometry"
    FEEL = "feel"
    POCKET = "pocket"


class LatticeConfidence(str, Enum):
    """Overall confidence in the assembled lattice."""
    LOCKED = "locked"
    STABLE = "stable"
    DEGRADED = "degraded"
    AMBIGUOUS = "ambiguous"
    FALLBACK = "fallback"


class QuantizationMode(str, Enum):
    """Overall quantization behavior mode."""
    PRESERVE_HUMAN = "preserve_human"  # Keep original unless egregious
    GROOVE_AWARE = "groove_aware"  # Apply groove but respect bounds
    RIGID = "rigid"  # Full quantization to grid


@dataclass
class WitnessTestimony:
    """Structured testimony from a single witness."""
    source: SourceType
    value: float
    confidence: float
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ========================================================================
# Temporal Lattice - The Final Authority
# ========================================================================

@dataclass
class TemporalLattice:
    """
    The 'Final Witness' for timing.

    Translates forensic observations (Lattice, Pulse, Phase, Groove)
    into a playable, human-aligned grid that the Quantizer can execute.
    """

    # ====================================================================
    # 1. The Physical Geometry (from ReverseGeoCrypt)
    # ====================================================================
    period_ms: float = 0.0
    lattice_confidence: float = 0.0
    # How many equal parts each beat divides into - 4 for simple/binary
    # meter (straight 16th notes), 3 for compound/ternary meter (6/8,
    # 9/8, 12/8, or a triplet-eighth shuffle feel). Previously always
    # hardcoded to 4 everywhere period_ms was derived from bpm, so
    # compound-meter material had every note forced onto a straight
    # 16th-note grid regardless of what TimeSignatureDetector found.
    subdivisions_per_beat: int = 4

    # ====================================================================
    # 2. The Feel (from GrooveField)
    # ====================================================================
    phase_delta_ms: float = 0.0  # RAW cyclic phase from analysis
    normalized_phase_ms: float = 0.0  # NORMALIZED to [-beat/2, +beat/2]
    swing_ratio: float = 0.5
    groove_confidence: float = 0.0
    mode: QuantizationMode = QuantizationMode.GROOVE_AWARE

    # ====================================================================
    # 3. The Anchor (from TempoIntelligence / PulseAnalysis)
    # ====================================================================
    bpm: float = 120.0
    tempo_confidence: float = 0.0
    phase_anchor_ms: float = 0.0

    # ====================================================================
    # 4. The Human Tolerance (The 'Pocket')
    # ====================================================================
    microtiming_window_ms: float = 15.0
    quantize_strength: float = 0.35
    preserve_ghost_notes: bool = True

    # ====================================================================
    # 5. Safety Bounds
    # ====================================================================
    max_quantization_shift_ms: float = 80.0  # Never shift more than this

    # ====================================================================
    # 6. State
    # ====================================================================
    overall_confidence: LatticeConfidence = LatticeConfidence.AMBIGUOUS
    _beat_cache: Dict[int, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Calculate the spooky geometry once on instantiation."""
        # subdivisions_per_beat must be a sane, positive small integer -
        # only 3 (compound/ternary) and 4 (simple/binary) are meaningful
        # musically; guard against garbage input from evidence dicts.
        if self.subdivisions_per_beat not in (2, 3, 4, 6):
            self.subdivisions_per_beat = 4

        # If period is missing but we have BPM, derive the finest grid
        # subdivision (16th note for simple meter, triplet-eighth for
        # compound meter - see subdivisions_per_beat above)
        if self.period_ms == 0 and self.bpm > 0:
            self.period_ms = 60000.0 / self.bpm / self.subdivisions_per_beat

        # Clamp period to reasonable range (20ms = 3000 BPM, 500ms = 30 BPM)
        self.period_ms = max(20.0, min(500.0, self.period_ms))

        # Clamp swing ratio
        self.swing_ratio = max(0.5, min(0.85, self.swing_ratio))

        # NORMALIZE THE PHASE DELTA - THIS IS THE CRITICAL FIX
        # Convert absolute cyclic phase (e.g., -1211ms) to musical offset
        beat_duration_ms = self.period_ms * self.subdivisions_per_beat  # one full beat
        if beat_duration_ms > 0:
            # Wrap to [-beat/2, +beat/2] range
            half_beat = beat_duration_ms / 2.0
            self.normalized_phase_ms = (
                                               (self.phase_delta_ms + half_beat) % beat_duration_ms
                                       ) - half_beat
        else:
            self.normalized_phase_ms = 0.0

        # Determine overall confidence
        self.overall_confidence = self._calculate_confidence()

        # Pre-calculate beat positions for performance
        self._precompute_beats()

    def _calculate_confidence(self) -> LatticeConfidence:
        """Calculate overall confidence based on witness agreement."""
        avg_confidence = (
                self.lattice_confidence * 0.3 +
                self.groove_confidence * 0.3 +
                self.tempo_confidence * 0.4
        )

        if avg_confidence >= 0.85:
            return LatticeConfidence.LOCKED
        elif avg_confidence >= 0.65:
            return LatticeConfidence.STABLE
        elif avg_confidence >= 0.40:
            return LatticeConfidence.DEGRADED
        elif avg_confidence >= 0.20:
            return LatticeConfidence.AMBIGUOUS
        else:
            return LatticeConfidence.FALLBACK

    def _precompute_beats(self, max_beats: int = 256):
        """Pre-calculate beat positions for fast lookup."""
        if self.period_ms <= 0:
            return

        for i in range(-max_beats, max_beats + 1):
            self._beat_cache[i] = i * self.period_ms

    def get_beat_time(self, beat_index: int) -> float:
        """Get the absolute time for a beat index."""
        return self._beat_cache.get(beat_index, beat_index * self.period_ms)

    def get_nearest_beat_index(self, time_ms: float) -> int:
        """Find the nearest beat index to a given time."""
        if self.period_ms <= 0:
            return 0
        return int(round(time_ms / self.period_ms))

    def _is_offbeat_eighth(self, time_ms: float) -> bool:
        """
        Is this note landing on the off-beat eighth note (the "and" of
        the beat) - the position swing feel actually applies to?

        This used to compare a note's distance from the nearest 16th-note
        grid line against self.period_ms / 4 - but period_ms IS the
        16th-note length here (see __post_init__), so that threshold was
        really 1/4 of a 16th note, a 64th-note-scale window. Almost every
        note that wasn't hit dead-on the 16th grid got flagged "offbeat"
        and had swing-offset applied in resolve() as a result, adding
        timing jitter to notes that were never near the actual eighth-note
        upbeat position swing is meant to affect.

        Checks proximity to the halfway point of the enclosing quarter
        note beat instead, with a half-16th-note tolerance on either side -
        enough to absorb natural onset-detection jitter around the true
        off-beat eighth without also catching the adjacent 16th-note
        positions on either side of it.
        In compound/ternary meter (subdivisions_per_beat != 4) the grid
        itself already lands on the true triplet/compound subdivisions -
        there's no separate "off-beat eighth" concept to additionally
        swing-push the way binary meter has, so this returns False and
        resolve() applies no extra swing offset (the grid spacing alone
        already carries the compound feel).
        """
        if self.subdivisions_per_beat != 4:
            return False
        beat_duration_ms = self.period_ms * self.subdivisions_per_beat
        if beat_duration_ms <= 0:
            return False
        beat_position = time_ms % beat_duration_ms
        half_beat = beat_duration_ms / 2.0
        return abs(beat_position - half_beat) < self.period_ms / 2.0

    def _get_pocket_width_ms(self) -> float:
        """
        Calculate dynamic pocket width.

        The phase delta widens the pocket (expressive intent)
        but doesn't shift the grid (preserves timeline).
        """
        # Normalized phase adds to pocket (musical expression)
        phase_contribution = abs(self.normalized_phase_ms) * 0.15

        return self.microtiming_window_ms + phase_contribution

    # ====================================================================
    # The Judge Function (CORRECTED)
    # ====================================================================

    def resolve(self, raw_ms: float, is_offbeat: bool = False,
                velocity: int = 80, confidence: float = 0.5) -> float:
        """
        The 'Judge' function - CORRECTED VERSION.

        Takes a raw timestamp and re-aligns it to the human lattice.

        CRITICAL FIX:
            - Normalized phase delta widens pocket, doesn't shift grid
            - Max shift bounds prevent timeline collapse
            - No negative timestamps
        """
        # Ghost notes preserve raw timing
        if self.preserve_ghost_notes and velocity < 30:
            return raw_ms

        # Full preserve mode
        if self.mode == QuantizationMode.PRESERVE_HUMAN:
            nearest_idx = self.get_nearest_beat_index(raw_ms)
            nearest_time = self.get_beat_time(nearest_idx)
            if abs(raw_ms - nearest_time) > RITORNELLO_MAX_SNAP_MS:
                direction = 1 if nearest_time > raw_ms else -1
                return max(0.0, raw_ms + direction * (RITORNELLO_MAX_SNAP_MS * 0.2))
            return raw_ms

        # STEP A: FIND TARGET GRID POINT
        lattice_count = self.get_nearest_beat_index(raw_ms)
        target_ms = self.get_beat_time(lattice_count)

        # STEP B: APPLY SWING (only if offbeat and significant)
        if is_offbeat and self.swing_ratio > 0.5:
            swing_offset = (self.swing_ratio - 0.5) * self.period_ms
            target_ms += swing_offset

        # STEP C: PHASE DELTA IS NOT APPLIED DIRECTLY
        # The normalized phase widens the pocket (expressive intent)
        # but NEVER shifts the grid (prevents negative timestamps)
        pocket = self._get_pocket_width_ms()

        # STEP D: POCKET CHECK - preserve human nuance
        if abs(raw_ms - target_ms) <= pocket:
            return raw_ms

        # STEP E: CONFIDENCE-ADJUSTED NUDGE
        effective_strength = self.quantize_strength * (1.0 - confidence)
        effective_strength = np.clip(effective_strength, 0.0, 1.0)

        # STEP F: LERP (Linear Interpolation)
        new_time = (1 - effective_strength) * raw_ms + (effective_strength * target_ms)

        # STEP G: ENFORCE SAFETY BOUNDS
        delta = new_time - raw_ms

        # Never shift more than max_quantization_shift_ms
        if abs(delta) > self.max_quantization_shift_ms:
            delta = np.sign(delta) * self.max_quantization_shift_ms
            new_time = raw_ms + delta

        # Never shift earlier than original by more than RITORNELLO_MAX_SNAP_MS
        if delta < 0 and abs(delta) > RITORNELLO_MAX_SNAP_MS:
            new_time = raw_ms - RITORNELLO_MAX_SNAP_MS

        # CRITICAL: Never produce negative timestamp
        new_time = max(0.0, new_time)

        # Also respect NoteEvent validation tolerance
        if new_time < raw_ms - 5.0:
            new_time = raw_ms - 5.0

        return float(new_time)

    # ====================================================================
    # Batch Processing
    # ====================================================================

    def quantize_notes(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """
        Apply the lattice to a list of notes.

        This is the main entry point for the Quantizer.
        """
        quantized = []

        for note in notes:
            is_offbeat = self._is_offbeat_eighth(note.start_ms)

            # Resolve the timing
            snapped_start = self.resolve(
                raw_ms=note.start_ms,
                is_offbeat=is_offbeat,
                velocity=note.velocity,
                confidence=note.confidence
            )

            # Apply the same offset to end time (preserve duration)
            delta = snapped_start - note.start_ms
            snapped_end = max(snapped_start + 10.0, note.end_ms + delta)  # Minimum 10ms duration

            # Create quantized note. Previously built via NoteEvent(...)
            # naming only 7 identity fields + the 5 new snap-related ones -
            # every OTHER field on the note (harmonic_series_match_ratio,
            # fundamental_freq_hz, reasoning_chain, consensus_votes,
            # veto_attempts, midi_channel, phrase_id, ...) silently reset
            # to its default for every note that passes through
            # quantization, i.e. every pitched note in the pipeline.
            # That's why upstream evidence like harmonic_validation's
            # per-note confidence never survived to reach Scribe's final
            # gate, which depends on exactly that field once no raw audio
            # is available. replace() carries every other field forward
            # unchanged, only actually overriding the ones quantization
            # itself computed.
            quantized_note = replace(
                note,
                snapped_start_ms=snapped_start,
                snapped_end_ms=snapped_end,
                snap_reason=f"temporal_lattice: period={self.period_ms:.1f}ms, swing={self.swing_ratio:.2f}, phase_norm={self.normalized_phase_ms:.1f}ms",
                snap_confidence_delta=abs(delta) * 0.001,
                quantization_strategy=QuantizationStrategy.GROOVE_AWARE
            )
            quantized.append(quantized_note)

        return quantized

    # ====================================================================
    # Factory Methods
    # ====================================================================

    @classmethod
    def assemble_from_witnesses(cls, evidence: Dict[str, Any]) -> 'TemporalLattice':
        """
        Factory method to gather the 'Shouts' from the Evidence Tracker
        and build the unified Lattice.
        """
        # Extract values with defaults
        period_ms = evidence.get('reverse_geo_lattice_period', 0.0)
        lattice_conf = evidence.get('reverse_geo_confidence', 0.3)

        # Real measured swing ratio (from PhaseDeltaAnalyzer's bass-vs-kick
        # model or OnsetSwingAnalyzer's kick-independent fallback) takes
        # priority - it's the actual measured value, not a 3-4 bucket
        # approximation. Falls back to the groove_type bucketing only when
        # nothing real was measured (evidence's default is the neutral 0.5).
        measured_swing_ratio = evidence.get('swing_ratio', 0.5)
        if abs(measured_swing_ratio - 0.5) > 1e-6:
            swing_ratio = measured_swing_ratio
        else:
            groove_type = evidence.get('groove_type', 'straight')
            if groove_type == 'wide_swing':
                swing_ratio = 0.72
            elif groove_type == 'dilla_pocket':
                swing_ratio = 0.65
            elif groove_type == 'laid_back':
                swing_ratio = 0.55
            else:
                swing_ratio = 0.5

        # RAW phase delta - will be normalized in __post_init__
        raw_phase_delta = evidence.get('groove_phase_delta', 0.0)
        groove_conf = evidence.get('groove_confidence', 0.3)

        bpm = evidence.get('pulse_bpm', 120.0)
        tempo_conf = evidence.get('pulse_confidence', 0.3)
        phase_anchor = evidence.get('pulse_phase_anchor_ms', 0.0)

        # Optional overrides
        microtiming_window = evidence.get('microtiming_window_ms', 15.0)
        quantize_strength = evidence.get('quantize_strength', 0.35)
        max_shift = evidence.get('max_quantization_shift_ms', 80.0)

        # Mode from evidence
        mode_str = evidence.get('quantization_mode', 'groove_aware')
        try:
            mode = QuantizationMode(mode_str)
        except ValueError:
            mode = QuantizationMode.GROOVE_AWARE

        # Compound meter (6/8, 9/8, 12/8 - denominator 8 with a numerator
        # divisible by 3) divides each beat into 3, not 4 - forcing that
        # onto a straight 16th-note grid was the single biggest source of
        # mis-timed notes in compound-meter material (verified: every
        # note in a real 3/4-detected section got quantized against a
        # grid built for a meter it wasn't in).
        time_sig_num = evidence.get('time_signature_numerator', 4)
        time_sig_den = evidence.get('time_signature_denominator', 4)
        is_compound = time_sig_den == 8 and time_sig_num % 3 == 0 and time_sig_num >= 6
        subdivisions_per_beat = 3 if is_compound else 4

        return cls(
            period_ms=period_ms,
            lattice_confidence=lattice_conf,
            phase_delta_ms=raw_phase_delta,  # Raw cyclic phase
            swing_ratio=swing_ratio,
            groove_confidence=groove_conf,
            bpm=bpm,
            tempo_confidence=tempo_conf,
            phase_anchor_ms=phase_anchor,
            microtiming_window_ms=microtiming_window,
            quantize_strength=quantize_strength,
            max_quantization_shift_ms=max_shift,
            mode=mode,
            subdivisions_per_beat=subdivisions_per_beat,
        )

    @classmethod
    def from_tempo_intelligence(cls, tempo_map: TempoMap,
                                confidence: Confidence = Confidence.MEDIUM) -> 'TemporalLattice':
        """Create a lattice primarily from TempoIntelligence."""
        return cls(
            bpm=tempo_map.initial_tempo_bpm,
            tempo_confidence=confidence.value_f,
            overall_confidence=LatticeConfidence.DEGRADED
        )

    @classmethod
    def from_pulse_field(cls, pulse_field: PulseField) -> 'TemporalLattice':
        """Create a lattice from PulseField analysis."""
        period_ms = pulse_field.beat_grid_ms[1] - pulse_field.beat_grid_ms[0] if len(
            pulse_field.beat_grid_ms) > 1 else 500.0
        period_ms = period_ms / 4

        return cls(
            period_ms=period_ms,
            bpm=pulse_field.tempo_bpm,
            tempo_confidence=pulse_field.confidence.value_f,
            phase_anchor_ms=pulse_field.phase_shift_ms,
            overall_confidence=LatticeConfidence.STABLE
        )

    @classmethod
    def from_groove_field(cls, groove: GrooveField, bpm: float = 120.0) -> 'TemporalLattice':
        """Create a lattice primarily from GrooveField analysis."""
        if groove.is_wide_swing:
            swing_ratio = 0.72
        elif groove.is_dilla_pocket:
            swing_ratio = 0.65
        else:
            swing_ratio = 0.5

        return cls(
            phase_delta_ms=groove.bass_kick_phase_delta_ms,  # Raw cyclic phase
            swing_ratio=swing_ratio,
            groove_confidence=groove.confidence.value_f,
            bpm=bpm,
            overall_confidence=LatticeConfidence.STABLE
        )

    @classmethod
    def from_reverse_geo_crypt(cls, period_ms: float, confidence: float = 0.5,
                               bpm: float = 120.0, phase_delta_ms: float = 0.0) -> 'TemporalLattice':
        """Create a lattice primarily from ReverseGeoCrypt analysis."""
        return cls(
            period_ms=period_ms,
            lattice_confidence=confidence,
            phase_delta_ms=phase_delta_ms,
            bpm=bpm,
            overall_confidence=LatticeConfidence.DEGRADED
        )

    @classmethod
    def default(cls, bpm: float = 120.0) -> 'TemporalLattice':
        """Create a default lattice when no witnesses are available."""
        return cls(
            period_ms=60000.0 / bpm / 4.0,
            bpm=bpm,
            overall_confidence=LatticeConfidence.FALLBACK
        )

    # ====================================================================
    # Utility Methods
    # ====================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lattice to dictionary for JSON export."""
        return {
            "period_ms": self.period_ms,
            "lattice_confidence": self.lattice_confidence,
            "phase_delta_ms_raw": self.phase_delta_ms,
            "phase_delta_ms_normalized": self.normalized_phase_ms,
            "swing_ratio": self.swing_ratio,
            "groove_confidence": self.groove_confidence,
            "bpm": self.bpm,
            "tempo_confidence": self.tempo_confidence,
            "phase_anchor_ms": self.phase_anchor_ms,
            "microtiming_window_ms": self.microtiming_window_ms,
            "quantize_strength": self.quantize_strength,
            "max_quantization_shift_ms": self.max_quantization_shift_ms,
            "preserve_ghost_notes": self.preserve_ghost_notes,
            "mode": self.mode.value,
            "overall_confidence": self.overall_confidence.value
        }

    def get_summary(self) -> str:
        """Get human-readable summary of the lattice."""
        return (
            f"TemporalLattice: {self.bpm:.1f}BPM, "
            f"period={self.period_ms:.1f}ms, "
            f"swing={self.swing_ratio:.2f}, "
            f"phase_raw={self.phase_delta_ms:.0f}ms→norm={self.normalized_phase_ms:.1f}ms, "
            f"pocket={self.microtiming_window_ms:.0f}ms, "
            f"strength={self.quantize_strength:.0%}, "
            f"max_shift={self.max_quantization_shift_ms:.0f}ms, "
            f"mode={self.mode.value}, "
            f"confidence={self.overall_confidence.value}"
        )

    def __repr__(self) -> str:
        return self.get_summary()


# ========================================================================
# Lattice Quantizer
# ========================================================================

class TemporalLatticeQuantizer:
    """Quantizer that uses the TemporalLattice for timing decisions."""

    def __init__(self, lattice: TemporalLattice):
        self.lattice = lattice

    def quantize(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """Quantize notes using the lattice."""
        return self.lattice.quantize_notes(notes)

    def quantize_note(self, note: NoteEvent) -> NoteEvent:
        """Quantize a single note."""
        return self.lattice.quantize_notes([note])[0]

    def get_lattice(self) -> TemporalLattice:
        """Get the underlying lattice."""
        return self.lattice

    def update_from_evidence(self, evidence: Dict[str, Any]) -> 'TemporalLatticeQuantizer':
        """Update the lattice from new evidence and return self."""
        self.lattice = TemporalLattice.assemble_from_witnesses(evidence)
        return self


# ========================================================================
# Convenience Functions
# ========================================================================

def create_lattice_from_pipeline_results(
        tempo_bpm: Optional[float] = None,
        tempo_confidence: float = 0.5,
        groove_phase_delta: float = 0.0,
        groove_type: str = "straight",
        lattice_period_ms: float = 0.0,
        microtiming_window_ms: float = 15.0,
        quantize_strength: float = 0.35,
        max_shift_ms: float = 80.0
) -> TemporalLattice:
    """Create a lattice directly from pipeline results."""
    evidence = {
        'reverse_geo_lattice_period': lattice_period_ms,
        'groove_phase_delta': groove_phase_delta,
        'groove_type': groove_type,
        'pulse_bpm': tempo_bpm or 120.0,
        'pulse_confidence': tempo_confidence,
        'microtiming_window_ms': microtiming_window_ms,
        'quantize_strength': quantize_strength,
        'max_quantization_shift_ms': max_shift_ms
    }
    return TemporalLattice.assemble_from_witnesses(evidence)


def quick_lattice_test():
    """Quick test function for TemporalLattice - now with phase normalization."""
    print("\n" + "=" * 60)
    print("Temporal Lattice Test (with Phase Normalization)")
    print("=" * 60)

    # Test 1: Raw phase delta -1211ms (3 beats early) at 74.4 BPM
    print("\n--- TEST 1: Raw phase -1211ms, 74.4 BPM ---")
    evidence_1 = {
        'reverse_geo_lattice_period': 405.1,
        'reverse_geo_confidence': 0.85,
        'groove_phase_delta': -1211.0,  # This will be normalized
        'groove_type': 'wide_swing',
        'groove_confidence': 0.78,
        'pulse_bpm': 74.4,
        'pulse_confidence': 0.82,
        'microtiming_window_ms': 15.0,
        'quantize_strength': 0.35,
        'max_quantization_shift_ms': 80.0
    }

    lattice_1 = TemporalLattice.assemble_from_witnesses(evidence_1)
    print(f"\n{lattice_1.get_summary()}")
    print(f"  Raw phase delta: {lattice_1.phase_delta_ms:.1f}ms")
    print(f"  Normalized phase: {lattice_1.normalized_phase_ms:.1f}ms")
    print(f"  Pocket width: {lattice_1._get_pocket_width_ms():.1f}ms")

    # Test 2: Early note with large phase delta (should NOT go negative)
    print("\n--- TEST 2: Early note safety check ---")
    early_notes = [0.0, 40.0, 100.0, 200.0]

    from core.order_types import NoteEvent, SourceType

    for raw in early_notes:
        note = NoteEvent(
            pitch=60,
            start_ms=raw,
            end_ms=raw + 100,
            velocity=80,
            confidence=0.8,
            zero_crossing_rate=0.0,
            source=SourceType.PITCH
        )

        snapped = lattice_1.resolve(raw, is_offbeat=False, velocity=80, confidence=0.8)
        delta = snapped - raw

        print(
            f"  raw={raw:6.1f}ms → snapped={snapped:6.1f}ms (Δ={delta:+6.1f}ms) {'✓' if snapped >= 0 else '✗ NEGATIVE!'}")

    # Test 3: Phase normalization at different tempos
    print("\n--- TEST 3: Phase normalization by tempo ---")
    test_cases = [
        (74.4, -1211.0, "3 beats early"),
        (120.0, -1000.0, "5 beats early at 120 BPM"),
        (152.0, -1211.0, "~3 beats early at 152 BPM"),
        (90.0, 650.0, "Positive phase"),
    ]

    for bpm, phase, desc in test_cases:
        period_ms = 60000.0 / bpm / 4.0
        lattice = TemporalLattice(
            period_ms=period_ms,
            phase_delta_ms=phase,
            bpm=bpm
        )
        quarter_note = period_ms * 4
        print(f"\n  {desc}: {bpm:.1f}BPM, period={period_ms:.1f}ms, quarter={quarter_note:.1f}ms")
        print(f"    Raw phase: {phase:6.1f}ms → Normalized: {lattice.normalized_phase_ms:+6.1f}ms")
        print(f"    (This is a {abs(lattice.normalized_phase_ms) / quarter_note * 100:.1f}% quarter note shift)")

    # Test 4: Boundary enforcement
    print("\n--- TEST 4: Max shift boundary enforcement ---")
    lattice_strict = TemporalLattice(
        period_ms=125.0,  # 120 BPM 16th
        max_quantization_shift_ms=50.0,
        quantize_strength=1.0  # Full quantize
    )

    far_notes = [0.0, 500.0, 1000.0]
    for raw in far_notes:
        snapped = lattice_strict.resolve(raw, confidence=0.0)
        delta = snapped - raw
        print(f"  raw={raw:6.1f}ms → snapped={snapped:6.1f}ms (Δ={delta:+6.1f}ms)")
        print(f"    Shift limited to {lattice_strict.max_quantization_shift_ms:.1f}ms ✓")

    print("\n" + "=" * 60)
    print("Lattice test complete - no negative timestamps produced.")
    print("=" * 60)


if __name__ == "__main__":
    quick_lattice_test()