# =================================================================
# MODULE: epistemic/musical_findings_map.py
# DESCRIPTION: Cross-stage musical intelligence accumulator.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-18
#
# PHILOSOPHY:
#     Right now every stage computes information, stores it locally,
#     and dies. This module is the bridge that keeps it alive.
#
#     Every stage that discovers something musical writes here.
#     Every stage that needs musical context reads here.
#     Nothing is lost between stages.
#
# FLOW:
#     tempo_analysis      → writes bpm, beat_times
#     tonal_detection     → writes chord_map
#     harmonic_validation → writes detected_key
#     groove_analysis     → writes swing_ratio, phase_delta
#     reverse_geo_crypt   → writes lattice_period
#     quantization        → reads everything above
#     voice_continuity    → reads instrument_families, chord_map
#     epistemic_council   → reads and writes constraint maps
# =================================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class MusicalFindingsMap:
    """
    Shared musical state that accumulates across all pipeline stages.

    This is a mutable accumulator — stages write to it freely.
    It is NOT a message bus or event system. It is simply a
    well-typed container for musical intelligence that would
    otherwise die at the end of each stage.
    """

    # ------------------------------------------------------------------
    # Tempo (from tempo_analysis)
    # ------------------------------------------------------------------
    tempo_bpm: float = 120.0
    tempo_confidence: float = 0.0
    tempo_source: str = "default"            # "librosa", "guided", "default"
    beat_times_ms: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    time_signature_numerator: int = 4
    time_signature_denominator: int = 4

    # ------------------------------------------------------------------
    # Harmony (from harmonic_validation, tonal_detection)
    # ------------------------------------------------------------------
    detected_key: Optional[str] = None       # e.g. "Bb major", "D minor"
    key_confidence: float = 0.0
    chord_map: List[Dict[str, Any]] = field(default_factory=list)
    # chord_map entries: {"start_ms": float, "chord": str, "confidence": float}

    # ------------------------------------------------------------------
    # Groove (from groove_analysis)
    # ------------------------------------------------------------------
    swing_ratio: float = 0.5                 # 0.5=straight, 0.72=wide swing
    phase_delta_ms: float = 0.0             # Bass-kick timing offset (clamped ±100ms)
    groove_type: str = "straight"            # "wide_swing", "dilla_pocket", "straight"
    groove_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Lattice (from reverse_geo_crypt)
    # ------------------------------------------------------------------
    lattice_period_ms: float = 0.0           # Physical grid spacing
    lattice_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Instrument Presence (from timbre_analysis, epistemic_council)
    # ------------------------------------------------------------------
    instrument_families: Dict[str, float] = field(default_factory=dict)
    # e.g. {"piano": 0.85, "strings": 0.60, "brass": 0.30}

    # ------------------------------------------------------------------
    # Epistemic Council Constraint Maps (from Pre-Flight models)
    # Written by Librosa and SPICE before heavier models run.
    # Read by Basic Pitch and Omnizart to skip dead zones.
    # ------------------------------------------------------------------
    librosa_cqt_heatmap: Optional[np.ndarray] = None
    # Shape: (n_pitches, n_frames), CQT energy per pitch per time
    # Regions with near-zero energy = skip in heavier models

    spice_melody_notes: List[Any] = field(default_factory=list)
    # NoteEvent list from SPICE — the melodic thread
    # Notes found by both SPICE and Basic Pitch → confidence 0.99

    active_time_windows_ms: List[tuple] = field(default_factory=list)
    # List of (start_ms, end_ms) windows with significant activity
    # Computed from CQT heatmap — used to skip silence in heavier models

    # ------------------------------------------------------------------
    # Audit Trail
    # ------------------------------------------------------------------
    stages_written: List[str] = field(default_factory=list)
    # Record which stages contributed to avoid re-processing

    def record_stage(self, stage_name: str) -> None:
        if stage_name not in self.stages_written:
            self.stages_written.append(stage_name)

    def has_stage(self, stage_name: str) -> bool:
        return stage_name in self.stages_written

    # ------------------------------------------------------------------
    # Convenience Readers
    # ------------------------------------------------------------------

    def get_chord_at(self, time_ms: float) -> Optional[Dict[str, Any]]:
        """Return the most recently detected chord before time_ms."""
        candidates = [c for c in self.chord_map if c["start_ms"] <= time_ms]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c["start_ms"])

    def has_active_content_at(self, time_ms: float, tolerance_ms: float = 50.0) -> bool:
        """Return True if any active window covers this timestamp."""
        if not self.active_time_windows_ms:
            return True   # No constraint → assume active everywhere
        return any(
            (start - tolerance_ms) <= time_ms <= (end + tolerance_ms)
            for start, end in self.active_time_windows_ms
        )

    def compute_active_windows_from_heatmap(
        self,
        hop_length: int = 512,
        sample_rate: int = 16000,
        energy_threshold: float = 0.02,
        min_window_ms: float = 100.0,
    ) -> None:
        """
        Populate active_time_windows_ms from the CQT heatmap.
        Regions where max energy across all pitches exceeds threshold
        are marked active. Silence regions are skipped by heavier models.
        """
        if self.librosa_cqt_heatmap is None:
            return

        # Energy per frame: max across all pitch bins
        frame_energy = np.max(np.abs(self.librosa_cqt_heatmap), axis=0)
        frame_times_ms = (
            np.arange(len(frame_energy)) * hop_length / sample_rate * 1000.0
        )
        active_mask = frame_energy > (np.max(frame_energy) * energy_threshold)

        # Group consecutive active frames into windows
        windows = []
        in_window = False
        win_start = 0.0

        for i, (t, is_active) in enumerate(zip(frame_times_ms, active_mask)):
            if is_active and not in_window:
                win_start = t
                in_window = True
            elif not is_active and in_window:
                win_end = t
                if (win_end - win_start) >= min_window_ms:
                    windows.append((win_start, win_end))
                in_window = False

        if in_window and len(frame_times_ms) > 0:
            win_end = float(frame_times_ms[-1])
            if (win_end - win_start) >= min_window_ms:
                windows.append((win_start, win_end))

        self.active_time_windows_ms = windows

    def to_temporal_lattice_evidence(self) -> Dict[str, Any]:
        """
        Export this map as the evidence dict consumed by
        TemporalLattice.assemble_from_witnesses().
        """
        return {
            "pulse_bpm":                 self.tempo_bpm,
            "pulse_confidence":          self.tempo_confidence,
            "pulse_phase_anchor_ms":     float(self.beat_times_ms[0])
                                         if len(self.beat_times_ms) > 0 else 0.0,
            "groove_phase_delta":        self.phase_delta_ms,
            "groove_type":               self.groove_type,
            "groove_confidence":         self.groove_confidence,
            # Real numeric ratio (set by groove_analysis, either from
            # PhaseDeltaAnalyzer's bass-vs-kick model or OnsetSwingAnalyzer's
            # kick-independent fallback) - previously computed but never
            # exposed here, so assemble_from_witnesses always re-derived a
            # coarse 3-bucket (0.72/0.65/0.5) approximation from groove_type
            # instead of using the actual measured ratio.
            "swing_ratio":               self.swing_ratio,
            "time_signature_numerator":   self.time_signature_numerator,
            "time_signature_denominator": self.time_signature_denominator,
            "reverse_geo_lattice_period": self.lattice_period_ms,
            "reverse_geo_confidence":    self.lattice_confidence,
            "microtiming_window_ms":     15.0,
            "quantize_strength":         0.35,
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