# =================================================================
# MODULE: agents/quantization/ritornello.py
# DESCRIPTION: Musical coherence repair pass - v6.0 restoration.
#
# VERSION: 6.0
# UPDATED: 2026-07-12
#
# WHAT THIS IS NOT (anymore):
#     Ritornello is NOT a quantizer. Snap-to-grid timing belongs to
#     TemporalLattice (agents/quantization/temporal_lattice.py), which is
#     what actually runs as the pipeline's "quantization" stage. Same-
#     pitch overlap/adjacency merging belongs to VelocityMerge
#     (agents/quantization/velocity_merge.py), which runs immediately
#     before this stage. Both of those already existed and already did
#     their jobs well - this file used to duplicate a chunk of that work
#     under its own name (multi-stage coarse/medium/fine/micro grid
#     snapping) while never actually being wired into the real pipeline
#     at all. That's gone.
#
# WHAT THIS IS (restored to its real pre-5.0 intent):
#     A post-quantization, post-merge MUSICAL COHERENCE REPAIR pass.
#     VelocityMerge and TemporalLattice both operate note-by-note (or
#     pair-by-pair); this stage looks at the repaired note list as a
#     whole and fixes what neither of them can:
#       - Pass 2: Sustain Recovery - a note's own acoustic energy often
#         keeps ringing well past whatever timestamp onset detection
#         guessed as its end. Extend it, non-destructively.
#       - Pass 3: Micro-Note Purge - very short, low-confidence notes
#         are usually detection noise, UNLESS they have real harmonic
#         evidence behind them (harmonic_series_match_ratio, from
#         Schoenberg Mirror) - a signal the pre-5.0 version never had.
#       - Pass 4: Tie Reconstruction - two same-pitch notes whose gap
#         straddles a beat line are usually one sustained note split by
#         a detection artifact at the barline, not two real notes.
#       - Pass 6: Pitch-Wobble Collapse - the deliberate complement to
#         VelocityMerge's hard pitch-equality gate: repeated near-(not
#         exact-)pitch fluctuations are vibrato or tracking noise, not a
#         new note. VelocityMerge is right to refuse these (protects
#         real chromatic neighbors); this pass is what catches them
#         instead.
#       - Pass 7: Trouble Map - flags measures with low average
#         confidence or high note-per-beat fragmentation.
#       - CorrectionScribe: for measures flagged by the Trouble Map, goes
#         back to the REAL isolated stem audio for that time window,
#         optionally pitch-shifts it up an octave to escape a muddy low
#         end, re-runs detection with Basic Pitch (not librosa - Basic
#         Pitch is the trusted engine for this), transposes back down,
#         and injects corrected notes with boosted confidence as PROOF
#         of the repair. "Repair and prove," not just "clean up."
#
#     Passes 1 (Fragment Collapse) and 5 (Phrase Grammar) from the
#     pre-5.0 design are deliberately NOT reimplemented here - see the
#     rationale comment below, right before the pass classes.
#
# NON-DESTRUCTIVE AUDIT LAW (unchanged from every version of this file):
#     start_ms/end_ms are NEVER overwritten. snapped_start_ms/
#     snapped_end_ms shadow them. The original is always there.
#
# DROPPED FROM THE PRE-5.0 VERSION:
#     The `_N` wrapper/adapter class for lineage tracking - foreign to
#     this codebase's convention (dataclasses.replace() or direct
#     mutation on the real NoteEvent), and the exact shape of bug this
#     project has repeatedly found and fixed elsewhere (a wrapper or
#     partial-field constructor silently dropping fields a later stage
#     needed). RevisionEvent + reasoning_chain tags replace it entirely.
#     `async def revise()` - the pipeline is fully synchronous
#     everywhere; this is now a plain `def repair()`.
#
# Authored by: Claude - v6.0 restoration (2026-07-12)
# Based on the pre-5.0 (4.7-era) Ritornello's real design intent.
# =================================================================

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.order_types import NoteEvent, SourceType, BeatGrid, TimeSignatureCandidate
from core.testimony import RitornelloTestimony
from core.basic_pitch_utils import run_basic_pitch_on_slice
from core.constants import (
    RITORNELLO_ACTIVATION_LIGHT_THRESHOLD,
    RITORNELLO_ACTIVATION_FULL_THRESHOLD,
    RITORNELLO_ACTIVATION_DEEP_THRESHOLD,
    SUSTAIN_RECOVERY_RMS_FLOOR_RATIO,
    SUSTAIN_RECOVERY_MAX_EXTEND_MS,
    SUSTAIN_RECOVERY_FRAME_MS,
    SUSTAIN_RECOVERY_FFT_SIZE,
    SUSTAIN_RECOVERY_NEXT_ONSET_MARGIN_MS,
    MICRO_NOTE_MIN_DURATION_MS,
    MICRO_NOTE_PURGE_CONFIDENCE_CEILING,
    MICRO_NOTE_HARMONIC_JUSTIFICATION_RATIO,
    MICRO_NOTE_PURGE_NO_EVIDENCE_CONFIDENCE_FLOOR,
    TIE_RECONSTRUCTION_BEAT_PROXIMITY_MS,
    TIE_RECONSTRUCTION_MAX_GAP_MS,
    PITCH_WOBBLE_MAX_SEMITONES,
    PITCH_WOBBLE_MIN_SEMITONES,
    PITCH_WOBBLE_MAX_GAP_MS,
    PITCH_WOBBLE_MAX_MEMBER_DURATION_MS,
    TROUBLE_MAP_MIN_CONFIDENCE,
    TROUBLE_MAP_MAX_FRAGMENTATION_NOTES_PER_BEAT,
    CORRECTION_SCRIBE_OCTAVE_SHIFT_UP,
    CORRECTION_SCRIBE_OCTAVE_SHIFT_SEMITONES,
    CORRECTION_SCRIBE_BASIC_PITCH_ONSET_THRESHOLD,
    CORRECTION_SCRIBE_BASIC_PITCH_FRAME_THRESHOLD,
    CORRECTION_SCRIBE_CONFIDENCE_BOOST,
    CORRECTION_SCRIBE_MAX_MEASURES_PER_RUN,
    CORRECTION_SCRIBE_MEASURE_PADDING_MS,
)
from core.protocols import MusicBoxProtocol, StatusReporterProtocol


# ========================================================================
# Shared helper: read a note's stem tag
# ========================================================================

def _stem_tag(note: NoteEvent) -> str:
    """
    Reads the "stem:X" reasoning_chain tag pipeline.py's stem-tagging
    stages already attach (bass/other/vocals/master/drums). Duplicated
    locally rather than imported from orchestration.pipeline - this
    module has no agent/orchestration imports, matching the "no agent
    imports, pure transformer" rule every quantization agent in this
    package already follows. Falls back to "master" (the full mix) when
    a note has no stem tag, same fallback Schoenberg Mirror's own
    isolated-stem routing uses.
    """
    for tag in reversed(note.reasoning_chain or []):
        if isinstance(tag, str) and tag.startswith("stem:"):
            return tag.split(":", 1)[1]
    return "master"


# ========================================================================
# Kept dataclasses (per explicit product direction), rebuilt for this
# codebase's real data model
# ========================================================================

@dataclass(frozen=True)
class MeasureMap:
    """
    Thin, non-authoritative wrapper over an existing BeatGrid for
    measure/beat lookups - does NOT duplicate beat detection. Built FROM
    the pipeline's own already-computed BeatGrid + TimeSignatureCandidate,
    not from madmom directly (madmom is one tempo witness among several,
    unrelated to how the beat grid itself gets built).
    """
    measure_boundaries_ms: List[float]   # one entry per downbeat
    beats_per_measure: int
    beat_duration_ms: float
    tempo_bpm: float

    @classmethod
    def from_beat_grid(cls, beat_grid: BeatGrid,
                       time_signature: Optional[TimeSignatureCandidate] = None) -> "MeasureMap":
        if beat_grid is None or not beat_grid.downbeat_times_ms:
            raise ValueError("BeatGrid has no downbeat data to build a MeasureMap from")
        beats_per_measure = time_signature.numerator if time_signature else 4
        return cls(
            measure_boundaries_ms=list(beat_grid.downbeat_times_ms),
            beats_per_measure=max(1, beats_per_measure),
            beat_duration_ms=max(1e-6, beat_grid.beat_duration_ms),
            tempo_bpm=beat_grid.tempo_bpm,
        )

    def measure_index_at(self, time_ms: float) -> int:
        idx = bisect.bisect_right(self.measure_boundaries_ms, time_ms) - 1
        return max(0, idx)

    def measure_bounds(self, measure_index: int) -> Tuple[float, float]:
        if not self.measure_boundaries_ms:
            return 0.0, 0.0
        measure_index = max(0, min(measure_index, len(self.measure_boundaries_ms) - 1))
        start = self.measure_boundaries_ms[measure_index]
        if measure_index + 1 < len(self.measure_boundaries_ms):
            end = self.measure_boundaries_ms[measure_index + 1]
        else:
            end = start + self.beats_per_measure * self.beat_duration_ms
        return start, end

    def nearest_beat_ms(self, time_ms: float) -> float:
        if not self.measure_boundaries_ms:
            return time_ms
        idx = self.measure_index_at(time_ms)
        measure_start, _ = self.measure_bounds(idx)
        beats_from_start = round((time_ms - measure_start) / self.beat_duration_ms)
        return measure_start + beats_from_start * self.beat_duration_ms

    def is_near_beat(self, time_ms: float, tolerance_ms: float) -> bool:
        return abs(self.nearest_beat_ms(time_ms) - time_ms) <= tolerance_ms


@dataclass(frozen=True)
class RevisionEvent:
    """
    Lightweight per-note change record - replaces the old wrapper-class
    lineage system entirely. This is the audit trail: what pass touched
    which note, why, and what changed.
    """
    pass_name: str        # "sustain_recovery" | "micro_note_purge" |
                           # "tie_reconstruction" | "pitch_wobble_collapse" |
                           # "correction_scribe"
    action: str            # "extended" | "purged" | "tied" | "collapsed" | "injected"
    note_pitch: int
    note_start_ms: float    # original, untouched start_ms - the audit key
    confidence_before: float
    confidence_after: float
    reason: str
    measure_index: Optional[int] = None


@dataclass(frozen=True)
class TroubleMeasure:
    """Diagnostics - a measure flagged as low-confidence and/or fragmented."""
    measure_number: int
    start_ms: float
    end_ms: float
    avg_confidence: float
    reasons: List[str] = field(default_factory=list)
    corrected_count: int = 0   # filled in via replace() after CorrectionScribe runs


@dataclass
class RitornelloConfig:
    """Activation-level thresholds (light/full/deep) plus one knob per pass."""

    activation_light_threshold: float = RITORNELLO_ACTIVATION_LIGHT_THRESHOLD
    activation_full_threshold: float = RITORNELLO_ACTIVATION_FULL_THRESHOLD
    activation_deep_threshold: float = RITORNELLO_ACTIVATION_DEEP_THRESHOLD

    enable_sustain_recovery: bool = True
    enable_micro_note_purge: bool = True
    enable_tie_reconstruction: bool = True
    enable_pitch_wobble_collapse: bool = True
    enable_trouble_map: bool = True
    enable_correction_scribe: bool = True   # only actually invoked at "deep" activation

    sustain_recovery_rms_floor_ratio: float = SUSTAIN_RECOVERY_RMS_FLOOR_RATIO
    sustain_recovery_max_extend_ms: float = SUSTAIN_RECOVERY_MAX_EXTEND_MS
    sustain_recovery_frame_ms: float = SUSTAIN_RECOVERY_FRAME_MS
    sustain_recovery_fft_size: int = SUSTAIN_RECOVERY_FFT_SIZE
    sustain_recovery_next_onset_margin_ms: float = SUSTAIN_RECOVERY_NEXT_ONSET_MARGIN_MS

    micro_note_min_duration_ms: float = MICRO_NOTE_MIN_DURATION_MS
    micro_note_purge_confidence_ceiling: float = MICRO_NOTE_PURGE_CONFIDENCE_CEILING
    micro_note_harmonic_justification_ratio: float = MICRO_NOTE_HARMONIC_JUSTIFICATION_RATIO
    micro_note_purge_no_evidence_floor: float = MICRO_NOTE_PURGE_NO_EVIDENCE_CONFIDENCE_FLOOR

    tie_reconstruction_beat_proximity_ms: float = TIE_RECONSTRUCTION_BEAT_PROXIMITY_MS
    tie_reconstruction_max_gap_ms: float = TIE_RECONSTRUCTION_MAX_GAP_MS

    pitch_wobble_max_semitones: float = PITCH_WOBBLE_MAX_SEMITONES
    pitch_wobble_min_semitones: float = PITCH_WOBBLE_MIN_SEMITONES
    pitch_wobble_max_gap_ms: float = PITCH_WOBBLE_MAX_GAP_MS
    pitch_wobble_max_member_duration_ms: float = PITCH_WOBBLE_MAX_MEMBER_DURATION_MS

    trouble_map_min_confidence: float = TROUBLE_MAP_MIN_CONFIDENCE
    trouble_map_max_fragmentation_notes_per_beat: float = TROUBLE_MAP_MAX_FRAGMENTATION_NOTES_PER_BEAT

    correction_scribe_octave_shift_up: bool = CORRECTION_SCRIBE_OCTAVE_SHIFT_UP
    correction_scribe_octave_shift_semitones: int = CORRECTION_SCRIBE_OCTAVE_SHIFT_SEMITONES
    correction_scribe_onset_threshold: float = CORRECTION_SCRIBE_BASIC_PITCH_ONSET_THRESHOLD
    correction_scribe_frame_threshold: float = CORRECTION_SCRIBE_BASIC_PITCH_FRAME_THRESHOLD
    correction_scribe_confidence_boost: float = CORRECTION_SCRIBE_CONFIDENCE_BOOST
    correction_scribe_max_measures_per_run: int = CORRECTION_SCRIBE_MAX_MEASURES_PER_RUN
    correction_scribe_measure_padding_ms: float = CORRECTION_SCRIBE_MEASURE_PADDING_MS


# ========================================================================
# WHY PASSES 1 AND 5 ARE NOT REIMPLEMENTED HERE
# ========================================================================
# Pass 1 (Fragment Collapse - merge same-pitch notes within a gap):
#   agents/quantization/velocity_merge.py's OverlapDetector.build_merge_groups
#   already does this exactly: a hard pitch-equality gate
#   (_pitches_compatible) plus gap-based adjacency merging. It runs
#   immediately before this stage. Reimplementing it here would duplicate
#   that pass on already-merged output.
#
# Pass 5 (Phrase Grammar - phrase-level cleanup):
#   epistemic/phrase_intelligence.py's PhraseIntelligence is the closest
#   existing analog, but it computes a song-level structural verse/
#   chorus/bridge trajectory, not per-phrase note-grouping grammar at the
#   note level - a weaker redundancy claim than Pass 1's. Skipped per
#   explicit product direction, not because PhraseIntelligence provably
#   replaces it. If real note-grammar gaps show up later, that's a
#   genuine candidate for a NEW pass, not evidence this one was wrong to
#   skip.
# ========================================================================


class SustainRecovery:
    """
    Pass 2: extend notes still acoustically ringing past their nominal end.

    Pitch-aware since the burden-of-proof realignment: the "still
    ringing?" question is answered by energy in narrow bands around the
    note's OWN fundamental (and 2nd harmonic), never broadband RMS.
    Broadband asks "is anything still playing in this note's stem", which
    in a real mix - or worse, a stem carrying separation bleed - is
    almost always yes. Verified on real audio (Hopeful.mp3, full-mix
    harness): the broadband version extended nearly every note toward the
    400ms cap, stretching bass median duration 4.8x and pushing
    same-register overlap from 55% to 81%. The detection model's original
    end boundary is only overridden when THIS pitch demonstrably kept
    sounding.

    Second guard, same principle: never extend into the next detected
    onset of the same pitch. That onset is a re-articulation the
    detection model explicitly found; running a "repair" over it deletes
    real rhythmic information.
    """

    def __init__(self, config: RitornelloConfig):
        self.config = config

    @staticmethod
    def _pitch_hz(pitch_midi: int) -> float:
        return 440.0 * 2.0 ** ((pitch_midi - 69) / 12.0)

    def _band_level(self, frame: np.ndarray, sample_rate: int, f0_hz: float,
                    fft_cache: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        RMS-equivalent level of the energy within +/-1 semitone of the
        note's fundamental and 2nd harmonic. Kept in amplitude units (not
        energy) so sustain_recovery_rms_floor_ratio retains its original
        "fraction of the onset level" meaning.

        fft_cache holds (hanning window, rfft bin frequencies) per frame
        length - recomputing those per call dominated runtime when this
        pass faces thousands of notes (verified: pushed the ritornello
        stage past its watchdog on a 6485-note run).
        """
        n = len(frame)
        cached = fft_cache.get(n)
        if cached is None:
            cached = (np.hanning(n), np.fft.rfftfreq(n, 1.0 / sample_rate))
            fft_cache[n] = cached
        window, freqs = cached

        spec = np.abs(np.fft.rfft(frame * window))

        band_energy = 0.0
        for harmonic in (1, 2):
            target = f0_hz * harmonic
            if target >= sample_rate / 2:
                break
            lo = target * 2.0 ** (-1.0 / 12.0)
            hi = target * 2.0 ** (1.0 / 12.0)
            lo_i = int(np.searchsorted(freqs, lo))
            hi_i = int(np.searchsorted(freqs, hi))
            if hi_i <= lo_i:
                # Bass fundamentals sit below one bin's width at this FFT
                # size - fall back to the nearest bin so low notes still
                # get a real (if coarse) narrowband reading.
                lo_i = int(np.argmin(np.abs(freqs - target)))
                hi_i = lo_i + 1
            lo_i = max(0, lo_i - 1)
            hi_i = min(len(spec), hi_i + 1)
            band = spec[lo_i:hi_i]
            band_energy += float(np.dot(band, band))

        return float(np.sqrt(band_energy / max(1, n)))

    def recover(
            self, notes: List[NoteEvent], stem_audio_by_tag: Dict[str, np.ndarray],
            sample_rate: int,
    ) -> Tuple[List[NoteEvent], List[RevisionEvent]]:
        if sample_rate <= 0:
            return notes, []

        revisions: List[RevisionEvent] = []
        updated: List[NoteEvent] = []
        hop_samples = max(64, int(self.config.sustain_recovery_frame_ms * sample_rate / 1000))
        fft_samples = max(hop_samples, int(self.config.sustain_recovery_fft_size))
        max_extend_samples = int(self.config.sustain_recovery_max_extend_ms * sample_rate / 1000)
        margin_ms = self.config.sustain_recovery_next_onset_margin_ms

        # Next same-pitch onset per note: the hard ceiling no extension
        # may cross (that onset is a re-articulation the detector found).
        onsets_by_pitch: Dict[int, List[float]] = defaultdict(list)
        for n in notes:
            onsets_by_pitch[n.pitch].append(n.get_active_start_ms())
        for onsets in onsets_by_pitch.values():
            onsets.sort()

        # (hanning window, rfft bin freqs) per frame length, shared across
        # every note this call - see _band_level.
        fft_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for note in notes:
            audio = stem_audio_by_tag.get(_stem_tag(note))
            if audio is None:
                updated.append(note)
                continue

            f0_hz = self._pitch_hz(note.pitch)
            onset_start = int(note.get_active_start_ms() * sample_rate / 1000)
            onset_end = int(note.get_active_end_ms() * sample_rate / 1000)
            onset_slice = audio[onset_start:onset_start + fft_samples]
            if len(onset_slice) < 256:
                updated.append(note)
                continue

            onset_level = self._band_level(onset_slice, sample_rate, f0_hz, fft_cache)
            if onset_level <= 1e-8:
                updated.append(note)
                continue

            same_pitch_onsets = onsets_by_pitch[note.pitch]
            next_idx = bisect.bisect_right(same_pitch_onsets, note.get_active_start_ms())
            next_onset_ms = (same_pitch_onsets[next_idx]
                             if next_idx < len(same_pitch_onsets) else float("inf"))
            extension_ceiling_ms = next_onset_ms - margin_ms

            floor = onset_level * self.config.sustain_recovery_rms_floor_ratio
            extend_samples = 0
            pos = onset_end
            while extend_samples < max_extend_samples:
                candidate_end_ms = note.get_active_end_ms() + (
                    (extend_samples + hop_samples) * 1000.0 / sample_rate)
                if candidate_end_ms > extension_ceiling_ms:
                    break
                frame = audio[pos:pos + fft_samples]
                if len(frame) < fft_samples // 2:
                    break
                if self._band_level(frame, sample_rate, f0_hz, fft_cache) < floor:
                    break
                pos += hop_samples
                extend_samples += hop_samples

            if extend_samples <= 0:
                updated.append(note)
                continue

            new_end_ms = note.get_active_end_ms() + (extend_samples * 1000.0 / sample_rate)
            revisions.append(RevisionEvent(
                pass_name="sustain_recovery", action="extended",
                note_pitch=note.pitch, note_start_ms=note.start_ms,
                confidence_before=note.confidence, confidence_after=note.confidence,
                reason=f"energy at {f0_hz:.0f}Hz stayed above floor for "
                       f"{extend_samples / sample_rate * 1000:.0f}ms past the note's nominal end",
            ))
            updated.append(replace(
                note, snapped_end_ms=new_end_ms,
                reasoning_chain=list(note.reasoning_chain or []) + ["ritornello:sustain_recovery:extended"],
            ))

        return updated, revisions


class MicroNotePurge:
    """Pass 3: delete very short, low-confidence notes unless harmonically justified."""

    def __init__(self, config: RitornelloConfig):
        self.config = config

    def purge(self, notes: List[NoteEvent]) -> Tuple[List[NoteEvent], List[RevisionEvent]]:
        revisions: List[RevisionEvent] = []
        kept: List[NoteEvent] = []

        for note in notes:
            is_short = note.active_duration_ms() < self.config.micro_note_min_duration_ms

            if note.harmonic_series_match_ratio is None:
                # Harmonic validation never scored this note - absence of
                # evidence is not evidence of hallucination. The detection
                # model already vouched for the note's existence, so
                # overriding it requires hallucination-tier confidence,
                # not merely "below the normal ceiling".
                should_purge = (is_short
                                and note.confidence < self.config.micro_note_purge_no_evidence_floor)
                reason = (f"duration={note.active_duration_ms():.0f}ms "
                          f"confidence={note.confidence:.2f} below no-evidence floor "
                          f"{self.config.micro_note_purge_no_evidence_floor:.2f}")
            else:
                is_low_conf = note.confidence < self.config.micro_note_purge_confidence_ceiling
                harmonic_ok = (note.harmonic_series_match_ratio
                               >= self.config.micro_note_harmonic_justification_ratio)
                should_purge = is_short and is_low_conf and not harmonic_ok
                reason = (f"duration={note.active_duration_ms():.0f}ms "
                          f"confidence={note.confidence:.2f}, harmonic ratio "
                          f"{note.harmonic_series_match_ratio:.2f} below justification threshold")

            if should_purge:
                revisions.append(RevisionEvent(
                    pass_name="micro_note_purge", action="purged",
                    note_pitch=note.pitch, note_start_ms=note.start_ms,
                    confidence_before=note.confidence, confidence_after=0.0,
                    reason=reason,
                ))
                continue

            kept.append(note)

        return kept, revisions


class TieReconstruction:
    """Pass 4: tie same-pitch notes whose gap straddles a beat line."""

    def __init__(self, config: RitornelloConfig):
        self.config = config

    def reconstruct(
            self, notes: List[NoteEvent], measure_map: Optional[MeasureMap],
    ) -> Tuple[List[NoteEvent], List[RevisionEvent]]:
        if measure_map is None or len(notes) < 2:
            return notes, []

        revisions: List[RevisionEvent] = []
        sorted_notes = sorted(notes, key=lambda n: n.get_active_start_ms())
        result: List[NoteEvent] = []
        i = 0
        while i < len(sorted_notes):
            note = sorted_notes[i]
            if i + 1 < len(sorted_notes):
                nxt = sorted_notes[i + 1]
                gap = nxt.get_active_start_ms() - note.get_active_end_ms()
                # nxt.confidence <= note.confidence: a genuinely
                # re-articulated note carries its own full-strength onset;
                # a detection artifact splitting ONE held note at the
                # barline leaves a weaker second fragment (decaying
                # sustain). Only the decay pattern justifies overriding
                # an onset the detection model explicitly reported.
                if (note.pitch == nxt.pitch
                        and 0 <= gap <= self.config.tie_reconstruction_max_gap_ms
                        and nxt.confidence <= note.confidence):
                    boundary_ms = note.get_active_end_ms() + gap / 2.0
                    if measure_map.is_near_beat(boundary_ms, self.config.tie_reconstruction_beat_proximity_ms):
                        merged_conf = max(note.confidence, nxt.confidence)
                        merged = replace(
                            note, snapped_end_ms=nxt.get_active_end_ms(),
                            confidence=merged_conf,
                            reasoning_chain=list(note.reasoning_chain or []) + ["ritornello:tie_reconstruction:tied"],
                        )
                        revisions.append(RevisionEvent(
                            pass_name="tie_reconstruction", action="tied",
                            note_pitch=note.pitch, note_start_ms=note.start_ms,
                            confidence_before=note.confidence, confidence_after=merged_conf,
                            reason=f"gap {gap:.1f}ms straddles beat at {boundary_ms:.0f}ms",
                            measure_index=measure_map.measure_index_at(boundary_ms),
                        ))
                        result.append(merged)
                        i += 2
                        continue
            result.append(note)
            i += 1

        return result, revisions


class PitchWobbleCollapse:
    """
    Pass 6: collapse repeated NEAR-(not exact-)pitch fluctuations - the
    deliberate complement to VelocityMerge's hard pitch-equality gate,
    which is right to refuse these (protects real chromatic neighbors)
    but leaves genuine vibrato/tracking-noise wobble untouched.
    """

    def __init__(self, config: RitornelloConfig):
        self.config = config

    def collapse(self, notes: List[NoteEvent]) -> Tuple[List[NoteEvent], List[RevisionEvent]]:
        if len(notes) < 2:
            return notes, []

        revisions: List[RevisionEvent] = []
        sorted_notes = sorted(notes, key=lambda n: n.get_active_start_ms())
        max_member_ms = self.config.pitch_wobble_max_member_duration_ms
        result: List[NoteEvent] = []
        i = 0
        while i < len(sorted_notes):
            group = [sorted_notes[i]]
            j = i + 1
            # A note longer than max_member_ms is a deliberately-played
            # note, not tracking noise - real chromatic passing tones and
            # grace-note figures are exactly the near-pitch/small-gap
            # pattern this pass targets, and duration is the
            # discriminator. A long seed never starts a group; a long
            # neighbor never joins one.
            if group[0].active_duration_ms() < max_member_ms:
                while j < len(sorted_notes):
                    prev = group[-1]
                    curr = sorted_notes[j]
                    gap = curr.get_active_start_ms() - prev.get_active_end_ms()
                    semis = abs(curr.pitch - prev.pitch)
                    if (0 <= gap <= self.config.pitch_wobble_max_gap_ms
                            and self.config.pitch_wobble_min_semitones <= semis <= self.config.pitch_wobble_max_semitones
                            and curr.active_duration_ms() < max_member_ms):
                        group.append(curr)
                        j += 1
                    else:
                        break

            if len(group) >= 2:
                best = max(group, key=lambda n: n.confidence)
                span_start = min(n.get_active_start_ms() for n in group)
                span_end = max(n.get_active_end_ms() for n in group)
                avg_conf = sum(n.confidence for n in group) / len(group)
                # Base the replace() on group[0] (chronologically first),
                # not `best` - NoteEvent.__post_init__ requires
                # snapped_start_ms >= start_ms - 5ms, and `best` (the
                # highest-confidence member) isn't necessarily the
                # earliest one, so anchoring on it could set the span's
                # snapped_start_ms BEFORE its own original start_ms and
                # violate that invariant. group[0]'s own start_ms is
                # always <= span_start by construction (sorted_notes is
                # time-ordered), so this is always safe. `best`'s pitch
                # is still used - it's the group's most confident PITCH
                # reading, independent of which member started first.
                collapsed = replace(
                    group[0], pitch=best.pitch,
                    snapped_start_ms=span_start, snapped_end_ms=span_end,
                    confidence=avg_conf,
                    reasoning_chain=list(group[0].reasoning_chain or []) + [
                        f"ritornello:pitch_wobble_collapse:collapsed_{len(group)}_notes"],
                )
                revisions.append(RevisionEvent(
                    pass_name="pitch_wobble_collapse", action="collapsed",
                    note_pitch=best.pitch, note_start_ms=group[0].start_ms,
                    confidence_before=best.confidence, confidence_after=avg_conf,
                    reason=f"{len(group)} notes within {self.config.pitch_wobble_max_semitones} "
                           "semitones collapsed",
                ))
                result.append(collapsed)
                i = j
            else:
                result.append(group[0])
                i += 1

        return result, revisions


class TroubleMap:
    """Pass 7: flag measures with low average confidence or high fragmentation."""

    def __init__(self, config: RitornelloConfig):
        self.config = config

    def build(self, notes: List[NoteEvent], measure_map: Optional[MeasureMap]) -> List[TroubleMeasure]:
        if measure_map is None or not notes:
            return []

        by_measure: Dict[int, List[NoteEvent]] = defaultdict(list)
        for note in notes:
            idx = measure_map.measure_index_at(note.get_active_start_ms())
            by_measure[idx].append(note)

        trouble: List[TroubleMeasure] = []
        for idx, group in sorted(by_measure.items()):
            avg_conf = sum(n.confidence for n in group) / len(group)
            start_ms, end_ms = measure_map.measure_bounds(idx)
            duration_beats = max(1e-6, (end_ms - start_ms) / measure_map.beat_duration_ms)
            notes_per_beat = len(group) / duration_beats

            reasons = []
            if avg_conf < self.config.trouble_map_min_confidence:
                reasons.append(f"low_confidence:{avg_conf:.2f}")
            if notes_per_beat > self.config.trouble_map_max_fragmentation_notes_per_beat:
                reasons.append(f"fragmented:{notes_per_beat:.1f}_notes_per_beat")

            if reasons:
                trouble.append(TroubleMeasure(
                    measure_number=idx, start_ms=start_ms, end_ms=end_ms,
                    avg_confidence=avg_conf, reasons=reasons,
                ))

        return trouble


class CorrectionScribe:
    """
    'Repair and prove' - for each flagged TroubleMeasure, re-analyzes the
    REAL isolated stem audio for that time window using Basic Pitch (not
    librosa - librosa is used here only as a DSP utility for the optional
    octave shift, never as the detection engine), transposes back down if
    shifted, and injects corrected notes with boosted confidence as proof
    of the repair.
    """

    def __init__(self, config: RitornelloConfig, music_box: Optional[MusicBoxProtocol] = None):
        self.config = config
        self._music_box = music_box

    def correct(
            self, trouble_measures: List[TroubleMeasure], notes: List[NoteEvent],
            stem_audio_by_tag: Dict[str, np.ndarray], sample_rate: int,
    ) -> Tuple[List[NoteEvent], List[TroubleMeasure], List[RevisionEvent]]:
        revisions: List[RevisionEvent] = []
        updated_notes = list(notes)
        updated_trouble = list(trouble_measures)

        for ti, trouble in enumerate(trouble_measures[:self.config.correction_scribe_max_measures_per_run]):
            measure_notes = [n for n in notes
                             if trouble.start_ms <= n.get_active_start_ms() < trouble.end_ms]
            tag = self._dominant_stem(measure_notes)
            audio = stem_audio_by_tag.get(tag)
            if audio is None:
                continue

            pad = self.config.correction_scribe_measure_padding_ms
            window_start_ms = max(0.0, trouble.start_ms - pad)
            window_end_ms = trouble.end_ms + pad
            start_sample = int(window_start_ms * sample_rate / 1000)
            end_sample = min(len(audio), int(window_end_ms * sample_rate / 1000))
            if end_sample - start_sample < int(sample_rate * 0.1):
                continue

            segment = audio[start_sample:end_sample].astype(np.float32)
            shift_applied = 0
            analysis_segment = segment
            if self.config.correction_scribe_octave_shift_up:
                try:
                    import librosa
                    analysis_segment = librosa.effects.pitch_shift(
                        segment, sr=sample_rate,
                        n_steps=self.config.correction_scribe_octave_shift_semitones)
                    shift_applied = self.config.correction_scribe_octave_shift_semitones
                except Exception:
                    analysis_segment = segment
                    shift_applied = 0

            try:
                raw_notes = run_basic_pitch_on_slice(
                    analysis_segment, sample_rate,
                    onset_threshold=self.config.correction_scribe_onset_threshold,
                    frame_threshold=self.config.correction_scribe_frame_threshold,
                    min_note_duration_ms=self.config.micro_note_min_duration_ms,
                    min_frequency_hz=27.5, max_frequency_hz=4186.0,
                )
            except Exception:
                continue

            if not raw_notes:
                continue

            new_notes: List[NoteEvent] = []
            for start_s, end_s, pitch_midi, amplitude in raw_notes:
                pitch = pitch_midi - shift_applied
                confidence = min(1.0, amplitude + self.config.correction_scribe_confidence_boost)
                new_note = NoteEvent(
                    pitch=pitch,
                    start_ms=window_start_ms + start_s * 1000.0,
                    end_ms=window_start_ms + end_s * 1000.0,
                    velocity=int(min(127, amplitude * 127)),
                    confidence=confidence,
                    zero_crossing_rate=0.0,
                    source=SourceType.RITORNELLO,
                    reasoning_chain=[f"stem:{tag}", "correction_scribe:basic_pitch",
                                     f"octave_shift={shift_applied}"],
                )
                new_notes.append(new_note)
                revisions.append(RevisionEvent(
                    pass_name="correction_scribe", action="injected",
                    note_pitch=pitch, note_start_ms=new_note.start_ms,
                    confidence_before=0.0, confidence_after=confidence,
                    reason="surgically re-analyzed from isolated stem audio via Basic Pitch",
                    measure_index=trouble.measure_number,
                ))

            updated_notes = [n for n in updated_notes
                             if not (trouble.start_ms <= n.get_active_start_ms() < trouble.end_ms)]
            updated_notes.extend(new_notes)
            updated_trouble[ti] = replace(trouble, corrected_count=len(new_notes))

            if self._music_box is not None:
                try:
                    self._music_box.log_decision(
                        stage_name="ritornello",
                        decision_type="correction_scribe_repair",
                        before_state={"measure": trouble.measure_number,
                                     "prior_note_count": len(measure_notes)},
                        after_state={"corrected_note_count": len(new_notes), "stem": tag,
                                    "octave_shift": shift_applied},
                        reasoning=f"CorrectionScribe re-analyzed measure {trouble.measure_number} "
                                 f"via Basic Pitch ({tag} stem)",
                        reversible=True,
                    )
                except Exception:
                    pass

        return sorted(updated_notes, key=lambda n: n.get_active_start_ms()), updated_trouble, revisions

    @staticmethod
    def _dominant_stem(measure_notes: List[NoteEvent]) -> str:
        counts: Dict[str, int] = defaultdict(int)
        for n in measure_notes:
            counts[_stem_tag(n)] += 1
        if not counts:
            return "master"
        return max(counts, key=counts.get)


class Ritornello:
    """
    Top-level entry point - orchestrates all real passes plus
    CorrectionScribe, gated by activation level.
    """

    def __init__(self, config: Optional[RitornelloConfig] = None,
                 music_box: Optional[MusicBoxProtocol] = None,
                 status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config or RitornelloConfig()
        self._music_box = music_box
        self._status_reporter = status_reporter
        self._sustain_recovery = SustainRecovery(self.config)
        self._micro_note_purge = MicroNotePurge(self.config)
        self._tie_reconstruction = TieReconstruction(self.config)
        self._pitch_wobble_collapse = PitchWobbleCollapse(self.config)
        self._trouble_map = TroubleMap(self.config)
        self._correction_scribe = CorrectionScribe(self.config, music_box)

    def _activation_level(self, grid_confidence: float) -> str:
        if grid_confidence < self.config.activation_light_threshold:
            return "off"
        if grid_confidence < self.config.activation_full_threshold:
            return "light"
        if grid_confidence < self.config.activation_deep_threshold:
            return "full"
        return "deep"

    def repair(
            self,
            notes: List[NoteEvent],
            beat_grid: Optional[BeatGrid],
            time_signature: Optional[TimeSignatureCandidate],
            stem_audio_by_tag: Dict[str, np.ndarray],
            sample_rate: int,
            grid_confidence: float,
    ) -> Tuple[List[NoteEvent], RitornelloTestimony]:
        original_count = len(notes)
        if not notes:
            return [], RitornelloTestimony([], 0, [], [], 0, "skipped")

        level = self._activation_level(grid_confidence)
        if level == "off":
            return list(notes), RitornelloTestimony(list(notes), original_count, [], [], 0, "off")

        measure_map: Optional[MeasureMap] = None
        if beat_grid is not None:
            try:
                measure_map = MeasureMap.from_beat_grid(beat_grid, time_signature)
            except Exception:
                measure_map = None

        current = list(notes)
        all_revisions: List[RevisionEvent] = []

        if level in ("full", "deep"):
            if self.config.enable_sustain_recovery:
                current, revs = self._sustain_recovery.recover(current, stem_audio_by_tag, sample_rate)
                all_revisions.extend(revs)
            if self.config.enable_micro_note_purge:
                current, revs = self._micro_note_purge.purge(current)
                all_revisions.extend(revs)
            if self.config.enable_tie_reconstruction:
                current, revs = self._tie_reconstruction.reconstruct(current, measure_map)
                all_revisions.extend(revs)
            if self.config.enable_pitch_wobble_collapse:
                current, revs = self._pitch_wobble_collapse.collapse(current)
                all_revisions.extend(revs)

        trouble_measures: List[TroubleMeasure] = []
        if self.config.enable_trouble_map:
            trouble_measures = self._trouble_map.build(current, measure_map)

        corrected_count = 0
        if level == "deep" and self.config.enable_correction_scribe and trouble_measures:
            current, trouble_measures, revs = self._correction_scribe.correct(
                trouble_measures, current, stem_audio_by_tag, sample_rate)
            all_revisions.extend(revs)
            corrected_count = sum(t.corrected_count for t in trouble_measures)

        if self._music_box is not None:
            for rev in all_revisions:
                try:
                    self._music_box.log_decision(
                        stage_name="ritornello",
                        decision_type=f"ritornello_{rev.pass_name}",
                        before_state={"pitch": rev.note_pitch, "start_ms": rev.note_start_ms,
                                     "confidence": rev.confidence_before},
                        after_state={"confidence": rev.confidence_after, "reason": rev.reason},
                        reasoning=rev.reason,
                        reversible=True,
                    )
                except Exception:
                    pass

        testimony = RitornelloTestimony(
            repaired_notes=current, original_count=original_count,
            revision_events=all_revisions, trouble_measures=trouble_measures,
            corrected_by_scribe_count=corrected_count, activation_level=level,
        )
        return current, testimony


# ========================================================================
# Self-test (replaces the old quick_quantize_test - matches
# temporal_lattice.py's own quick_lattice_test() convention)
# ========================================================================

if __name__ == "__main__":
    def _make_note(pitch: int, start_ms: float, end_ms: float, confidence: float = 0.7,
                   harmonic_ratio: Optional[float] = None,
                   stem: Optional[str] = None) -> NoteEvent:
        return NoteEvent(
            pitch=pitch, start_ms=start_ms, end_ms=end_ms, velocity=80,
            confidence=confidence, zero_crossing_rate=0.0, source=SourceType.PITCH,
            harmonic_series_match_ratio=harmonic_ratio,
            reasoning_chain=[f"stem:{stem}"] if stem else [],
        )

    # ---- SustainRecovery in isolation: the pitch-aware regression test ----
    # The broadband bug this guards against: "is anything still playing in
    # this note's stem" is nearly always yes in a real mix, so every note
    # got extended toward the cap regardless of whether ITS pitch kept
    # sounding.
    sr = 22050
    t = np.arange(int(sr * 1.5)) / sr

    # bass stem: pure 220Hz (A3, MIDI 57) tone ringing 0-800ms
    bass_audio = np.where(t < 0.8, 0.5 * np.sin(2 * np.pi * 220.0 * t), 0.0).astype(np.float32)
    # other stem: 329.63Hz (E4, MIDI 64) tone for 100-400ms, then a LOUD
    # DIFFERENT-pitch (220Hz) tone from 400ms on - broadband would extend
    # the E4 note into it; narrowband must not.
    other_audio = np.where(
        (t >= 0.1) & (t < 0.4), 0.5 * np.sin(2 * np.pi * 329.63 * t),
        np.where(t >= 0.4, 0.5 * np.sin(2 * np.pi * 220.0 * t), 0.0),
    ).astype(np.float32)

    sustain_notes = [
        # Rings past its detected end -> extended, but never past the next
        # same-pitch onset at 700ms (a re-articulation the detector found)
        _make_note(57, 100.0, 400.0, stem="bass"),
        _make_note(57, 700.0, 790.0, stem="bass"),
        # Its own pitch STOPS at 400ms; only other-pitch energy follows ->
        # must NOT be extended (this was the broadband bug)
        _make_note(64, 100.0, 400.0, stem="other"),
    ]
    sustain_cfg = RitornelloConfig()
    recovered, sustain_revs = SustainRecovery(sustain_cfg).recover(
        sustain_notes, {"bass": bass_audio, "other": other_audio}, sr)

    a3 = next(n for n in recovered if n.pitch == 57 and n.start_ms == 100.0)
    e4 = next(n for n in recovered if n.pitch == 64)
    print(f"A3 extended to {a3.get_active_end_ms():.0f}ms "
          f"(ceiling: next same-pitch onset 700ms - {sustain_cfg.sustain_recovery_next_onset_margin_ms:.0f}ms margin)")
    print(f"E4 end unchanged at {e4.get_active_end_ms():.0f}ms despite loud other-pitch audio after it")
    assert a3.get_active_end_ms() > 400.0, "ringing note should be extended"
    assert a3.get_active_end_ms() <= 700.0 - sustain_cfg.sustain_recovery_next_onset_margin_ms, \
        "extension must stop short of the next same-pitch onset"
    assert e4.get_active_end_ms() == 400.0, \
        "note whose own pitch stopped must NOT be extended by other-pitch energy (broadband bug)"

    # ---- Full repair() pass-interaction test ----
    beat_grid = BeatGrid(
        beat_times_ms=[i * 500.0 for i in range(16)],
        downbeat_times_ms=[0.0, 2000.0, 4000.0, 6000.0],
        measure_duration_ms=2000.0, beat_duration_ms=500.0, tempo_bpm=120.0,
        time_signature="4/4",
    )
    time_sig = TimeSignatureCandidate(numerator=4, denominator=4, confidence=0.9,
                                      sample_segment_start_ms=0.0, sample_segment_end_ms=8000.0)

    notes = [
        # Tie candidate: same pitch, 10ms gap straddling the beat at 2000ms,
        # second fragment weaker (decay pattern) -> tied
        _make_note(60, 1500.0, 1995.0),
        _make_note(60, 2005.0, 2500.0, confidence=0.5),
        # Micro-note with real harmonic evidence AGAINST it -> purged
        _make_note(72, 3000.0, 3030.0, confidence=0.2, harmonic_ratio=0.1),
        # Micro-note with NO harmonic evidence at all, confidence above the
        # no-evidence floor -> SURVIVES (absence of evidence is not
        # evidence of hallucination; the detector vouched for it)
        _make_note(71, 3050.0, 3080.0, confidence=0.2),
        # Micro-note spared by harmonic evidence FOR it
        _make_note(74, 3100.0, 3130.0, confidence=0.2, harmonic_ratio=0.9),
        # Pitch wobble: three short near-pitch notes close together -> collapsed
        _make_note(65, 5000.0, 5100.0, confidence=0.6),
        _make_note(66, 5110.0, 5200.0, confidence=0.8),
        _make_note(65, 5210.0, 5300.0, confidence=0.5),
        # NOT wobble: near-pitch neighbors but each a deliberately-held
        # note (> member-duration cap) - a real chromatic line -> untouched
        _make_note(58, 6000.0, 6250.0, confidence=0.6),
        _make_note(59, 6260.0, 6510.0, confidence=0.6),
    ]

    config = RitornelloConfig(enable_correction_scribe=False)
    ritornello = Ritornello(config=config)
    repaired, testimony = ritornello.repair(
        notes=notes, beat_grid=beat_grid, time_signature=time_sig,
        stem_audio_by_tag={}, sample_rate=22050, grid_confidence=0.9,
    )

    print(f"Activation: {testimony.activation_level}")
    print(f"{testimony.original_count} -> {len(repaired)} notes")
    print(f"Revisions: {len(testimony.revision_events)}")
    for rev in testimony.revision_events:
        print(f"  {rev.pass_name}: {rev.action} pitch={rev.note_pitch} - {rev.reason}")

    assert testimony.activation_level == "deep"
    tie_revs = [r for r in testimony.revision_events if r.pass_name == "tie_reconstruction"]
    assert len(tie_revs) == 1, f"expected 1 tie, got {len(tie_revs)}"
    purge_revs = [r for r in testimony.revision_events if r.pass_name == "micro_note_purge"]
    assert len(purge_revs) == 1, f"expected 1 purge, got {len(purge_revs)}"
    assert purge_revs[0].note_pitch == 72, "only the evidence-backed purge should fire"
    assert any(n.pitch == 71 for n in repaired), \
        "micro-note with NO harmonic evidence must survive (burden of proof)"
    assert any(n.pitch == 74 for n in repaired), "harmonically-justified micro-note should survive"
    wobble_revs = [r for r in testimony.revision_events if r.pass_name == "pitch_wobble_collapse"]
    assert len(wobble_revs) == 1, f"expected 1 wobble collapse, got {len(wobble_revs)}"
    assert any(n.pitch == 58 for n in repaired) and any(n.pitch == 59 for n in repaired), \
        "held chromatic neighbors must NOT be collapsed as wobble"
    print("\nAll assertions passed.")
