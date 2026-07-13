# =================================================================
# MODULE: epistemic/musical_findings_map.py
# DESCRIPTION: Cross-stage musical intelligence accumulator.
#
# VERSION: 6.0
# UPDATED: 2026-07-12
#
# PHILOSOPHY:
#     Every stage that discovers something musical writes here.
#     Every stage that needs musical context reads here.
#     Nothing is lost between stages.
#
#     v6.0: this used to copy tempo/key/groove/lattice values out of the
#     frozen *Testimony objects each stage already builds (core/testimony.py)
#     into a second, separate set of mutable scalar fields - two sources of
#     truth for the same number, and the copy lost whatever curve/provenance
#     data the Testimony object had that the scalar didn't. The Testimony
#     objects are now the real backing store; the old scalar names survive
#     as read/write properties over them so every existing call site keeps
#     working unchanged.
#
# FLOW:
#     tempo_analysis      -> writes tempo_testimony (+ tempo_contention if
#                             EpistemicCouncil.resolve_tempo_contention ran)
#     tonal_detection     -> writes tonal_testimony
#     harmonic_validation -> writes harmonic_testimony
#     groove_analysis     -> writes groove_testimony
#     reverse_geo_crypt   -> writes lattice_testimony
#     quantization        -> reads everything above via
#                             to_temporal_lattice_evidence()
#     epistemic_council   -> reads and writes the pre-flight constraint maps
# =================================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Any

from core.testimony import (
    TempoTestimony, GrooveTestimony, HarmonicValidationTestimony,
    TonalTestimony, LatticeTestimony, QuaverTestimony, StructuralTestimony,
)


def _default_tempo_testimony() -> TempoTestimony:
    return TempoTestimony(tempo_bpm=120.0, confidence=0.0, source="default", beat_times_ms=[])


def _default_groove_testimony() -> GrooveTestimony:
    return GrooveTestimony(phase_delta_ms=0.0, is_wide_swing=False, is_dilla_pocket=False, confidence=0.0)


def _default_harmonic_testimony() -> HarmonicValidationTestimony:
    return HarmonicValidationTestimony(validated_notes=[], vetoed_count=0, detected_key=None,
                                        key_confidence=0.0, method="default")


def _default_tonal_testimony() -> TonalTestimony:
    return TonalTestimony(chord_map=[], detected_key=None, key_confidence=0.0, frame_count=0)


def _default_lattice_testimony() -> LatticeTestimony:
    return LatticeTestimony(period_ms=0.0, confidence=0.0, events_analyzed=0, ratio_family="unknown")


@dataclass
class MusicalFindingsMap:
    """
    Shared musical state that accumulates across all pipeline stages.

    This is a mutable accumulator - stages write to it freely. The
    tempo/harmony/groove/lattice fields are the real *Testimony objects
    each stage already builds (core/testimony.py); the scalar names below
    (tempo_bpm, detected_key, ...) are backward-compatible properties over
    them, not a second copy.
    """

    # ------------------------------------------------------------------
    # Real backing store: the Testimony objects each stage builds
    # ------------------------------------------------------------------
    tempo_testimony: TempoTestimony = field(default_factory=_default_tempo_testimony)
    groove_testimony: GrooveTestimony = field(default_factory=_default_groove_testimony)
    harmonic_testimony: HarmonicValidationTestimony = field(default_factory=_default_harmonic_testimony)
    tonal_testimony: TonalTestimony = field(default_factory=_default_tonal_testimony)
    lattice_testimony: LatticeTestimony = field(default_factory=_default_lattice_testimony)

    # Tempo-witness disagreement, as already computed by
    # EpistemicCouncil.resolve_tempo_contention() and logged to music_box -
    # exposed here too so it's inspectable without digging through the
    # forensic log. None when only one tempo witness was available (no
    # contention to record).
    tempo_contention: Optional[Dict[str, Any]] = None

    time_signature_numerator: int = 4
    time_signature_denominator: int = 4

    # ------------------------------------------------------------------
    # Fields with no Testimony equivalent - confirmed genuine orphans
    # (GrooveTestimony has no swing_ratio/groove_type field at all), not
    # duplication needing a fix.
    # ------------------------------------------------------------------
    swing_ratio: float = 0.5                 # 0.5=straight, 0.72=wide swing
    groove_type: str = "straight"            # "wide_swing", "dilla_pocket", "straight"

    # ------------------------------------------------------------------
    # Instrument Presence (from timbre_analysis, epistemic_council) -
    # normalized float fractions; TimbreTestimony.instrument_counts holds
    # raw int counts, a different shape, so this stays its own field
    # rather than a derived property.
    # ------------------------------------------------------------------
    instrument_families: Dict[str, float] = field(default_factory=dict)
    # e.g. {"piano": 0.85, "strings": 0.60, "brass": 0.30}

    # ------------------------------------------------------------------
    # Epistemic Council Constraint Maps (from Pre-Flight models).
    # Written by Librosa/SPICE before heavier models run. Actually wiring
    # these into a "skip dead zones" pass for Basic Pitch/Omnizart is a
    # real, separate feature addition (would touch EpistemicCouncil's
    # detection call sites) - deliberately left unwired here, not a bug in
    # this map.
    # ------------------------------------------------------------------
    librosa_cqt_heatmap: Optional[np.ndarray] = None
    active_time_windows_ms: List[tuple] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Testimony objects with no dedicated Pipeline attribute - these used
    # to be written as undeclared dynamic attributes (only worked because
    # this dataclass wasn't slotted); now real, schema-enforced fields.
    # ------------------------------------------------------------------
    quaver_testimonies: List[QuaverTestimony] = field(default_factory=list)
    structural_testimony: Optional[StructuralTestimony] = None
    reanalysis_performed: bool = False
    anechoic_profile: str = ""
    anechoic_region_count: int = 0

    # ------------------------------------------------------------------
    # Audit Trail
    # ------------------------------------------------------------------
    stages_written: List[str] = field(default_factory=list)
    # Record which stages contributed to avoid re-processing

    def record_stage(self, stage_name: str) -> None:
        if stage_name not in self.stages_written:
            self.stages_written.append(stage_name)

    # ------------------------------------------------------------------
    # Backward-compatible scalar properties over the Testimony objects.
    # Getters read through; setters use dataclasses.replace() (this
    # codebase's standing convention for updating frozen dataclasses)
    # rather than mutating the Testimony object in place.
    # ------------------------------------------------------------------

    @property
    def tempo_bpm(self) -> float:
        return self.tempo_testimony.tempo_bpm

    @tempo_bpm.setter
    def tempo_bpm(self, value: float) -> None:
        self.tempo_testimony = replace(self.tempo_testimony, tempo_bpm=value)

    @property
    def tempo_confidence(self) -> float:
        return self.tempo_testimony.confidence

    @tempo_confidence.setter
    def tempo_confidence(self, value: float) -> None:
        self.tempo_testimony = replace(self.tempo_testimony, confidence=value)

    @property
    def tempo_source(self) -> str:
        return self.tempo_testimony.source

    @tempo_source.setter
    def tempo_source(self, value: str) -> None:
        self.tempo_testimony = replace(self.tempo_testimony, source=value)

    @property
    def beat_times_ms(self) -> np.ndarray:
        return np.asarray(self.tempo_testimony.beat_times_ms, dtype=np.float64)

    @beat_times_ms.setter
    def beat_times_ms(self, value: Any) -> None:
        self.tempo_testimony = replace(self.tempo_testimony, beat_times_ms=list(value))

    @property
    def detected_key(self) -> Optional[str]:
        return self.harmonic_testimony.detected_key

    @detected_key.setter
    def detected_key(self, value: Optional[str]) -> None:
        self.harmonic_testimony = replace(self.harmonic_testimony, detected_key=value)

    @property
    def key_confidence(self) -> float:
        return self.harmonic_testimony.key_confidence

    @key_confidence.setter
    def key_confidence(self, value: float) -> None:
        self.harmonic_testimony = replace(self.harmonic_testimony, key_confidence=value)

    @property
    def chord_map(self) -> List[Dict[str, Any]]:
        return self.tonal_testimony.chord_map

    @chord_map.setter
    def chord_map(self, value: List[Dict[str, Any]]) -> None:
        self.tonal_testimony = replace(self.tonal_testimony, chord_map=value)

    @property
    def phase_delta_ms(self) -> float:
        return self.groove_testimony.phase_delta_ms

    @phase_delta_ms.setter
    def phase_delta_ms(self, value: float) -> None:
        self.groove_testimony = replace(self.groove_testimony, phase_delta_ms=value)

    @property
    def groove_confidence(self) -> float:
        return self.groove_testimony.confidence

    @groove_confidence.setter
    def groove_confidence(self, value: float) -> None:
        self.groove_testimony = replace(self.groove_testimony, confidence=value)

    @property
    def lattice_period_ms(self) -> float:
        return self.lattice_testimony.period_ms

    @lattice_period_ms.setter
    def lattice_period_ms(self, value: float) -> None:
        self.lattice_testimony = replace(self.lattice_testimony, period_ms=value)

    @property
    def lattice_confidence(self) -> float:
        return self.lattice_testimony.confidence

    @lattice_confidence.setter
    def lattice_confidence(self, value: float) -> None:
        self.lattice_testimony = replace(self.lattice_testimony, confidence=value)

    # ------------------------------------------------------------------
    # Export for TemporalLattice
    # ------------------------------------------------------------------

    def blended_confidence(self) -> float:
        """
        Weighted blend of the three song-level confidences this map holds.
        Pulled out as its own method (rather than inlined in
        to_temporal_lattice_evidence()) so anything else that needs "how
        confident is the ensemble overall" has one place to ask, instead of
        re-deriving the same weights a second time.
        """
        return (
            self.lattice_confidence * 0.3
            + self.groove_confidence * 0.3
            + self.tempo_confidence * 0.4
        )

    def to_temporal_lattice_evidence(self) -> Dict[str, Any]:
        """
        Export this map as the evidence dict consumed by
        TemporalLattice.assemble_from_witnesses().
        """
        blended_confidence = self.blended_confidence()
        # Lower confidence -> gentler snapping (lower quantize_strength) and
        # more timing tolerance (wider microtiming_window_ms). Matches this
        # codebase's established convention (Ritornello's activation-level
        # gating, TemporalLattice.resolve()'s own per-note confidence
        # scaling): when uncertain, disturb the audio less. Previously these
        # were hardcoded literals (0.35/15.0) unrelated to the confidence
        # values exported right alongside them.
        quantize_strength = float(np.clip(0.15 + 0.35 * blended_confidence, 0.15, 0.5))
        microtiming_window_ms = float(np.clip(25.0 - 15.0 * blended_confidence, 10.0, 25.0))

        beat_times = self.beat_times_ms
        return {
            "pulse_bpm":                 self.tempo_bpm,
            "pulse_confidence":          self.tempo_confidence,
            "pulse_phase_anchor_ms":     float(beat_times[0])
                                         if len(beat_times) > 0 else 0.0,
            "groove_phase_delta":        self.phase_delta_ms,
            "groove_type":               self.groove_type,
            "groove_confidence":         self.groove_confidence,
            "swing_ratio":               self.swing_ratio,
            "time_signature_numerator":   self.time_signature_numerator,
            "time_signature_denominator": self.time_signature_denominator,
            "reverse_geo_lattice_period": self.lattice_period_ms,
            "reverse_geo_confidence":    self.lattice_confidence,
            "microtiming_window_ms":     microtiming_window_ms,
            "quantize_strength":         quantize_strength,
            "mode":                      "groove_aware",
        }

    def __repr__(self) -> str:
        return (
            f"MusicalFindingsMap("
            f"bpm={self.tempo_bpm:.1f}, "
            f"key={self.detected_key or 'unknown'}, "
            f"swing={self.swing_ratio:.2f}, "
            f"groove={self.groove_type}, "
            f"lattice={self.lattice_period_ms:.1f}ms, "
            f"stages={self.stages_written})"
        )
