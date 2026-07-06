# =================================================================
# MODULE: agents/analysis/voice_continuity.py
# DESCRIPTION: Voice continuity and polyphonic voice separation.
#
# VERSION: 5.6.1 (FIXED: VoiceContinuityResult field mismatch)
# UPDATED: 2026-05-25
#
# CRITICAL FIXES in 5.2.0:
#   1. VoiceContinuityResult now uses correct field names:
#      - voice_crossings (not crossings)
#      - confidence (not statistics/execution_time_ms)
#   2. Removed invalid 'crossings' parameter
#   3. Fixed empty-notes early return
#   4. Added proper confidence calculation
#
# WHAT CHANGED IN 5.1:
#   1. MAX_VOICE_GAP_MS was 50ms → changed default to 500ms
#   2. MAX_OCTAVE_JUMP_SEMITONES was 14 → changed default to 24
#   3. Added PersistentWitnessLog — time-indexed memory of timbral witnesses
#   4. Added ClusterAwareVoiceSeparator — uses cluster_id from SpectralMasker
#
# PHILOSOPHY:
#     An instrument doesn't teleport.
#     If we witnessed a cello at 1200ms and see a cello-shaped
#     timbral profile at 1500ms, they are the same cello until
#     proven otherwise. The burden of proof is on fragmentation,
#     not on continuity.
# =================================================================

import time
import gc
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass, field, replace
from enum import Enum
from collections import defaultdict, deque

from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    Voice, VoiceRole, VoiceContinuityResult, DecisionType
)
from core.constants import (
    TARGET_SAMPLE_RATE,
    MIN_PITCH_MIDI,
    MAX_PITCH_MIDI,
    MIN_NOTE_DURATION_MS,
    STAGGERED_GC_TRIGGER_MB
)
from core.protocols import (
    AnalysisAgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol
)

logger = logging.getLogger(__name__)


# ========================================================================
# Enums
# ========================================================================

class VoiceSeparationStrategy(str, Enum):
    PITCH_PROXIMITY = "pitch_proximity"
    CLUSTER_AWARE = "cluster_aware"
    SPECTRAL = "spectral"
    HARMONIC = "harmonic"
    HYBRID = "hybrid"


class VoiceCrossingType(str, Enum):
    SIMPLE_CROSSING = "simple_crossing"
    CHAINED_CROSSING = "chained_crossing"
    HIDDEN_CROSSING = "hidden_crossing"


# ========================================================================
# Instrument Identity
# ========================================================================

def _get_instrument_family(note: NoteEvent) -> str:
    """
    Read the instrument family timbre_analysis tagged this note with
    ("family:brass" etc. in reasoning_chain - pipeline.py's
    _tag_notes_with_instrument_family is the only thing that writes this).
    Returns "unknown" if the note was never classified (timbre_analysis
    only classifies a bounded sample of notes per song, not all of them).
    """
    if note.reasoning_chain:
        for tag in note.reasoning_chain:
            if isinstance(tag, str) and tag.startswith('family:'):
                return tag.split(':', 1)[1]
    return "unknown"


# Standard playable/written MIDI pitch ranges per broad instrument family -
# lowest fundamental an instrument in the family can produce, up to the
# highest note standard orchestral/big-band writing calls for (sounding
# pitch, not written/transposed pitch, matching NoteEvent.pitch elsewhere
# in this codebase). Family-level only, matching the classifier's own
# granularity: the true range is the union across that family's
# constituent instruments (e.g. STRINGS spans double bass's low E1 up
# through violin's orchestral top), so this is intentionally loose - it
# catches genuinely implausible outliers (a "brass" note below any real
# brass instrument's range) without pretending to know which specific
# instrument it is. "synth"/"unknown" are intentionally absent: synths
# have no physical constraint, and an unclassified note has no basis for
# one either.
FAMILY_PITCH_RANGE_MIDI: Dict[str, Tuple[int, int]] = {
    "strings": (28, 100),    # E1 (double bass) - E7 (violin, orchestral top)
    "brass": (26, 88),       # D1 (tuba) - E6 (lead trumpet, big-band top)
    "woodwind": (34, 96),    # Bb1 (bassoon) - C7 (piccolo)
    "piano": (21, 108),      # A0 - C8 (full piano range)
    "plucked": (24, 96),     # C1 (harp) - C7 (guitar/harp standard top)
    "voice": (36, 88),       # C2 (bass extreme) - E6 (soprano extreme)
    "percussion": (36, 108),  # C2 (timpani) - C8 (glockenspiel, pitched percussion only)
}


# ========================================================================
# Configuration
# ========================================================================

@dataclass
class VoiceConfig:
    """
    Configuration for voice continuity analysis.

    KEY DEFAULTS (jazz-tuned):
        max_voice_gap_ms:         500ms  (was 50ms — too aggressive for jazz)
        max_octave_jump_semitones: 24    (was 14 — jazz comping leaps further)
        max_voices:                8     (typical jazz small ensemble)
    """

    max_voice_gap_ms: float = 500.0
    max_octave_jump_semitones: int = 24
    max_voices: int = 8
    min_note_duration_ms: float = MIN_NOTE_DURATION_MS

    # Cost function weights
    pitch_weight: float = 0.65
    time_weight: float = 0.35

    # Voice crossing detection
    crossing_window_ms: float = 100.0
    crossing_tolerance_semitones: int = 2

    # Register ranges for role assignment (MIDI note numbers)
    bass_range: Tuple[int, int] = (28, 52)  # E1 – E3
    tenor_range: Tuple[int, int] = (45, 69)  # A2 – A4
    alto_range: Tuple[int, int] = (53, 77)  # F3 – F5
    soprano_range: Tuple[int, int] = (60, 84)  # C4 – C6

    # Persistent Witness Log
    witness_lookback_ms: float = 2000.0
    witness_similarity_threshold: float = 0.78

    # Cluster awareness
    prefer_cluster_grouping: bool = True

    # Instrument identity: an additive cost penalty (added on top of the
    # normal pitch/time cost) when a candidate note's classified family
    # conflicts with the voice's own last-known family. Deliberately large
    # enough (> the 0.85 new-voice cutoff, on top of any base cost >= 0)
    # to reliably force a new voice rather than let it slide through on a
    # good pitch/time match - a trumpet note should not quietly become
    # part of an established violin voice just because it landed close in
    # pitch and time. Still only applies when BOTH notes have a confident
    # (non-"unknown") classification, so it only engages once there's
    # real corroborating evidence, not on a guess.
    family_mismatch_penalty: float = 1.0
    check_instrument_range: bool = True

    # Performance
    max_notes_per_voice: int = 10000
    detect_crossings: bool = True
    repair_octave_jumps: bool = True
    cleanup_after_analysis: bool = True


# ========================================================================
# Persistent Witness Log
# ========================================================================

@dataclass
class WitnessEntry:
    """A single timbral observation logged to the PersistentWitnessLog."""
    timestamp_ms: float
    cluster_id: int
    voice_id: str
    instrument_family: str
    pitch_range: Tuple[int, int]
    spectral_centroid: float
    embedding_snapshot: Optional[np.ndarray]
    confidence: float


class PersistentWitnessLog:
    """Time-indexed memory of timbral witnesses."""

    def __init__(self, max_entries: int = 2000):
        self._entries: List[WitnessEntry] = []
        self.max_entries = max_entries

    def add(self, entry: WitnessEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def query_near(self, timestamp_ms: float, window_ms: float = 500.0) -> List[WitnessEntry]:
        lo = timestamp_ms - window_ms
        hi = timestamp_ms + window_ms
        return [e for e in self._entries if lo <= e.timestamp_ms <= hi]

    def query_lookback(self, timestamp_ms: float, lookback_ms: float) -> List[WitnessEntry]:
        lo = timestamp_ms - lookback_ms
        return [e for e in self._entries if lo <= e.timestamp_ms <= timestamp_ms]

    def find_best_match(self, embedding: np.ndarray, timestamp_ms: float,
                        lookback_ms: float = 2000.0,
                        similarity_threshold: float = 0.78) -> Optional[WitnessEntry]:
        candidates = self.query_lookback(timestamp_ms, lookback_ms)
        best_entry: Optional[WitnessEntry] = None
        best_similarity = -1.0

        for entry in candidates:
            if entry.embedding_snapshot is None:
                continue
            sim = self._cosine_similarity(embedding, entry.embedding_snapshot)
            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        if best_similarity >= similarity_threshold:
            return best_entry
        return None

    def find_by_cluster(self, cluster_id: int, timestamp_ms: float,
                        lookback_ms: float = 2000.0) -> Optional[WitnessEntry]:
        candidates = [e for e in self.query_lookback(timestamp_ms, lookback_ms)
                      if e.cluster_id == cluster_id]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.timestamp_ms)

    def get_active_cluster_voices(self, timestamp_ms: float,
                                  window_ms: float = 1000.0) -> Dict[int, str]:
        recent = self.query_near(timestamp_ms, window_ms)
        cluster_voices: Dict[int, str] = {}
        for entry in sorted(recent, key=lambda e: e.timestamp_ms):
            if entry.cluster_id >= 0:
                cluster_voices[entry.cluster_id] = entry.voice_id
        return cluster_voices

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def __len__(self) -> int:
        return len(self._entries)


# ========================================================================
# Cluster-Aware Voice Separator
# ========================================================================

class ClusterAwareVoiceSeparator:
    """Separates voices using cluster_id tags from SpectralMasker."""

    def __init__(self, config: VoiceConfig, witness_log: Optional[PersistentWitnessLog] = None):
        self.config = config
        self.witness_log = witness_log or PersistentWitnessLog()
        self.last_assignment_costs: List[float] = []

    def separate_voices(self, notes: List[NoteEvent]) -> List[Voice]:
        self.last_assignment_costs = []

        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n.start_ms)

        clustered = [n for n in sorted_notes if self._get_cluster_id(n) >= 0]
        unclustered = [n for n in sorted_notes if self._get_cluster_id(n) < 0]

        voices: Dict[str, List[NoteEvent]] = defaultdict(list)
        cluster_to_voice: Dict[int, str] = {}
        next_voice_id = 0

        def new_voice_id() -> str:
            nonlocal next_voice_id
            vid = f"voice_{next_voice_id}"
            next_voice_id += 1
            return vid

        # Process clustered notes
        for note in clustered:
            cid = self._get_cluster_id(note)

            if cid in cluster_to_voice:
                vid = cluster_to_voice[cid]
            else:
                past = self.witness_log.find_by_cluster(cid, note.start_ms, self.config.witness_lookback_ms)
                if past:
                    vid = past.voice_id
                    cluster_to_voice[cid] = vid
                else:
                    vid = new_voice_id()
                    cluster_to_voice[cid] = vid

            voices[vid].append(note)
            self.last_assignment_costs.append(0.0)  # evidence-based (real cluster identity), not guessed
            self.witness_log.add(WitnessEntry(
                timestamp_ms=note.start_ms, cluster_id=cid, voice_id=vid,
                instrument_family=_get_instrument_family(note), pitch_range=(note.pitch, note.pitch),
                spectral_centroid=0.0, embedding_snapshot=None, confidence=note.confidence,
            ))

        # Process unclustered notes
        for note in unclustered:
            best_vid: Optional[str] = None
            best_cost = float('inf')

            for vid, note_list in voices.items():
                if not note_list:
                    continue
                last_note = max(note_list, key=lambda n: n.start_ms)
                time_gap = note.start_ms - last_note.end_ms

                # Reject genuine overlap: a voice is monophonic, so a note
                # starting before the voice's last note ended cannot join it.
                if time_gap < 0 or time_gap > self.config.max_voice_gap_ms:
                    continue

                pitch_gap = abs(note.pitch - last_note.pitch)
                pitch_cost = min(1.0, pitch_gap / self.config.max_octave_jump_semitones)
                time_cost = min(1.0, time_gap / self.config.max_voice_gap_ms)
                cost = (pitch_cost * self.config.pitch_weight + time_cost * self.config.time_weight)

                if (_get_instrument_family(note) != "unknown"
                        and _get_instrument_family(last_note) != "unknown"
                        and _get_instrument_family(note) != _get_instrument_family(last_note)):
                    cost += self.config.family_mismatch_penalty

                if cost < best_cost:
                    best_cost = cost
                    best_vid = vid

            if best_vid and best_cost < 0.85:
                voices[best_vid].append(note)
                self.last_assignment_costs.append(best_cost)
            else:
                # Only every recorded a cost for notes that successfully
                # joined a voice - a note forced to start a fresh voice
                # (the WORSE outcome: either every existing voice was a
                # bad fit, or none were even viable) silently contributed
                # nothing, biasing mean_assignment_cost toward only the
                # good assignments. best_cost stays float('inf') when no
                # voice existed to compare against at all (e.g. the very
                # first note) - that's not a real assignment decision, so
                # nothing is recorded in that specific case; a real but
                # too-costly candidate (best_cost finite, just >= 0.85)
                # does get recorded, clamped to 1.0 since cost can exceed
                # that once family_mismatch_penalty is added.
                if best_cost != float('inf'):
                    self.last_assignment_costs.append(min(best_cost, 1.0))
                vid = new_voice_id()
                voices[vid] = [note]

        # Enforce max_voices limit
        if len(voices) > self.config.max_voices:
            voices = self._merge_excess_voices(voices)

        # Convert to Voice objects
        result = []
        for vid, note_list in voices.items():
            note_list.sort(key=lambda n: n.start_ms)
            if note_list:
                result.append(self._build_voice(vid, note_list))

        return result

    def _get_cluster_id(self, note: NoteEvent) -> int:
        if hasattr(note, 'metadata') and isinstance(note.metadata, dict):
            return int(note.metadata.get('cluster_id', -1))
        if note.reasoning_chain:
            for tag in note.reasoning_chain:
                if isinstance(tag, str) and tag.startswith('cluster:'):
                    try:
                        return int(tag.split(':')[1])
                    except (IndexError, ValueError):
                        pass
        return -1

    def _merge_excess_voices(self, voices: Dict[str, List[NoteEvent]]) -> Dict[str, List[NoteEvent]]:
        while len(voices) > self.config.max_voices:
            smallest_vid = min(voices, key=lambda v: len(voices[v]))
            smallest_notes = voices.pop(smallest_vid)

            if not smallest_notes or not voices:
                break

            small_avg = float(np.mean([n.pitch for n in smallest_notes]))
            best_vid = min(voices.keys(), key=lambda v: abs(np.mean([n.pitch for n in voices[v]]) - small_avg))
            voices[best_vid].extend(smallest_notes)
            voices[best_vid].sort(key=lambda n: n.start_ms)

        return voices

    def _build_voice(self, voice_id: str, notes: List[NoteEvent]) -> Voice:
        pitches = [n.pitch for n in notes]
        return Voice(
            voice_id=voice_id,
            role=VoiceRole.INNER_VOICE,
            notes=notes,
            start_time_ms=notes[0].start_ms,
            end_time_ms=notes[-1].end_ms,
            average_pitch=float(np.mean(pitches)),
            pitch_range_semitones=max(pitches) - min(pitches) if len(pitches) > 1 else 0,
            is_monophonic=True,
        )


# ========================================================================
# Pitch Proximity Voice Separator
# ========================================================================

class PitchProximitySeparator:
    """Voice separation based on pitch proximity and temporal continuity."""

    def __init__(self, config: VoiceConfig, status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter
        self.last_assignment_costs: List[float] = []

    def separate_voices(self, notes: List[NoteEvent]) -> List[Voice]:
        self.last_assignment_costs = []

        if len(notes) < 2:
            return [self._single_note_voice(i, n) for i, n in enumerate(notes)]

        sorted_notes = sorted(notes, key=lambda n: (n.start_ms, n.pitch))
        active_voices: List[Tuple[int, NoteEvent, float]] = []
        voices: Dict[int, List[NoteEvent]] = defaultdict(list)
        next_id = 0

        for note in sorted_notes:
            best_idx = -1
            best_cost = float('inf')

            for idx, (vid, last_note, end_time) in enumerate(active_voices):
                time_gap = note.start_ms - end_time
                # A voice is monophonic by construction (is_monophonic=True
                # below) — a note that starts before the voice's last note
                # ended is a genuine simultaneous overlap and can never
                # belong to the same voice, no matter how close in pitch.
                if time_gap < 0 or time_gap > self.config.max_voice_gap_ms:
                    continue

                pitch_gap = abs(note.pitch - last_note.pitch)
                pitch_cost = min(1.0, pitch_gap / self.config.max_octave_jump_semitones)
                time_cost = min(1.0, time_gap / self.config.max_voice_gap_ms)
                cost = (pitch_cost * self.config.pitch_weight + time_cost * self.config.time_weight)

                # Instrument identity: a voice should stay the same
                # instrument. If both notes have a confident (non-unknown)
                # family classification and they disagree, penalize this
                # join heavily - a trumpet note shouldn't quietly become
                # part of an established violin voice just because the
                # pitch/timing happened to line up.
                if (_get_instrument_family(note) != "unknown"
                        and _get_instrument_family(last_note) != "unknown"
                        and _get_instrument_family(note) != _get_instrument_family(last_note)):
                    cost += self.config.family_mismatch_penalty

                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx

            if best_idx >= 0 and best_cost < 0.85:
                vid, _, _ = active_voices[best_idx]
                voices[vid].append(note)
                active_voices[best_idx] = (vid, note, note.end_ms)
                self.last_assignment_costs.append(best_cost)
            else:
                # See the matching comment in ClusterAwareVoiceSeparator -
                # forcing a new voice is the worse outcome and previously
                # recorded no cost at all, biasing mean_assignment_cost
                # toward only the successful joins.
                if best_cost != float('inf'):
                    self.last_assignment_costs.append(min(best_cost, 1.0))
                voices[next_id].append(note)
                active_voices.append((next_id, note, note.end_ms))
                next_id += 1

            active_voices = [(v, l, e) for v, l, e in active_voices
                             if note.start_ms - e <= self.config.max_voice_gap_ms]

        result = []
        for vid, note_list in voices.items():
            note_list.sort(key=lambda n: n.start_ms)
            if note_list:
                pitches = [n.pitch for n in note_list]
                result.append(Voice(
                    voice_id=f"voice_{vid}",
                    role=VoiceRole.INNER_VOICE,
                    notes=note_list,
                    start_time_ms=note_list[0].start_ms,
                    end_time_ms=note_list[-1].end_ms,
                    average_pitch=float(np.mean(pitches)),
                    pitch_range_semitones=max(pitches) - min(pitches),
                    is_monophonic=True,
                ))

        result.sort(key=lambda v: v.average_pitch)
        return result

    def _single_note_voice(self, i: int, note: NoteEvent) -> Voice:
        return Voice(
            voice_id=f"voice_{i}",
            role=VoiceRole.INNER_VOICE,
            notes=[note],
            start_time_ms=note.start_ms,
            end_time_ms=note.end_ms,
            average_pitch=float(note.pitch),
            pitch_range_semitones=0,
            is_monophonic=True,
        )


# ========================================================================
# Voice Crossing Detector
# ========================================================================

class VoiceCrossingDetector:
    """Detects when two voices cross pitch registers."""

    def __init__(self, config: VoiceConfig, status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def detect_crossings(self, voices: List[Voice]) -> List[Tuple[Voice, Voice, float]]:
        if len(voices) < 2:
            return []

        crossings = []
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                v_upper = voices[i] if voices[i].average_pitch > voices[j].average_pitch else voices[j]
                v_lower = voices[j] if voices[i].average_pitch > voices[j].average_pitch else voices[i]

                overlap_start = max(v_upper.start_time_ms, v_lower.start_time_ms)
                overlap_end = min(v_upper.end_time_ms, v_lower.end_time_ms)

                if overlap_start >= overlap_end:
                    continue

                t = overlap_start
                while t < overlap_end:
                    upper_pitch = self._pitch_at(v_upper, t)
                    lower_pitch = self._pitch_at(v_lower, t)
                    if upper_pitch and lower_pitch:
                        if upper_pitch <= lower_pitch + self.config.crossing_tolerance_semitones:
                            crossings.append((v_upper, v_lower, t))
                            break
                    t += self.config.crossing_window_ms

        return crossings

    def _pitch_at(self, voice: Voice, time_ms: float) -> Optional[int]:
        for note in voice.notes:
            if note.start_ms <= time_ms <= note.end_ms:
                return note.pitch
        return None


# ========================================================================
# Voice Role Assigner
# ========================================================================

class VoiceRoleAssigner:
    """Assigns musical roles to voices."""

    def __init__(self, config: VoiceConfig, status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def assign_roles(self, voices: List[Voice]) -> List[Voice]:
        if not voices:
            return voices

        sorted_voices = sorted(voices, key=lambda v: v.average_pitch)
        updated = []

        for i, voice in enumerate(sorted_voices):
            role = self._determine_role(voice, i, len(sorted_voices))
            updated.append(replace(voice, role=role))

        return updated

    def _determine_role(self, voice: Voice, index: int, total: int) -> VoiceRole:
        p = voice.average_pitch
        lo_b, hi_b = self.config.bass_range
        lo_t, hi_t = self.config.tenor_range
        lo_a, hi_a = self.config.alto_range
        lo_s, hi_s = self.config.soprano_range

        if lo_b <= p <= hi_b:
            return VoiceRole.BASS
        if lo_t <= p <= hi_t and total >= 3:
            return VoiceRole.TENOR
        if lo_a <= p <= hi_a and total >= 3:
            return VoiceRole.ALTO
        if lo_s <= p <= hi_s:
            return VoiceRole.SOPRANO

        if index == 0:
            return VoiceRole.BASS
        if index == total - 1:
            return VoiceRole.MELODY
        return VoiceRole.INNER_VOICE


# ========================================================================
# Octave Jump Repair
# ========================================================================

class OctaveJumpRepair:
    """Detects and flags large octave jumps in melodic lines."""

    def __init__(self, config: VoiceConfig, status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def repair_voice(self, voice: Voice) -> Voice:
        if not self.config.repair_octave_jumps or len(voice.notes) < 2:
            return voice

        repaired = []
        prev = None

        for note in voice.notes:
            if prev:
                jump = abs(note.pitch - prev.pitch)
                if jump > self.config.max_octave_jump_semitones:
                    flagged = replace(note, reasoning_chain=[
                        *(note.reasoning_chain or []),
                        f"Large interval flagged: {jump} semitones from previous note"
                    ])
                    repaired.append(flagged)
                    prev = flagged
                    continue
            repaired.append(note)
            prev = note

        return replace(voice, notes=repaired)


# ========================================================================
# Instrument Range Validator
# ========================================================================

class InstrumentRangeValidator:
    """
    Soft plausibility check: does a voice's own dominant instrument family
    physically span the pitch of each of its own notes? Flags (does not
    remove) notes that fall outside FAMILY_PITCH_RANGE_MIDI for the
    voice's dominant classified family - e.g. a "brass"-tagged note below
    any real brass instrument's lowest fundamental.
    """

    def __init__(self, config: VoiceConfig, status_reporter: Optional[StatusReporterProtocol] = None):
        self.config = config
        self.status_reporter = status_reporter

    def validate_voice(self, voice: Voice) -> Voice:
        if not self.config.check_instrument_range or not voice.notes:
            return voice

        family = self._dominant_family(voice)
        if family is None:
            return voice

        pitch_range = FAMILY_PITCH_RANGE_MIDI.get(family)
        if pitch_range is None:
            return voice
        lo, hi = pitch_range

        flagged = []
        for note in voice.notes:
            if note.pitch < lo or note.pitch > hi:
                flagged.append(replace(
                    note,
                    confidence=min(1.0, max(0.0, note.confidence * 0.7)),
                    reasoning_chain=[
                        *(note.reasoning_chain or []),
                        f"Range implausible: pitch {note.pitch} outside {family} range ({lo}-{hi})"
                    ]
                ))
            else:
                flagged.append(note)

        return replace(voice, notes=flagged)

    def _dominant_family(self, voice: Voice) -> Optional[str]:
        counts: Dict[str, int] = {}
        for note in voice.notes:
            family = _get_instrument_family(note)
            if family != "unknown":
                counts[family] = counts.get(family, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)


# ========================================================================
# VoiceContinuity — Main Agent (FIXED)
# ========================================================================

class VoiceContinuity:
    """
    Main voice continuity agent for Grimlock 5.2.

    FIXED: VoiceContinuityResult now uses correct field names.
    """

    def __init__(self, config: Optional[VoiceConfig] = None,
                 status_reporter: Optional[StatusReporterProtocol] = None,
                 music_box: Optional[MusicBoxProtocol] = None):
        self.config = config or VoiceConfig()
        self._status_reporter = status_reporter
        self._music_box = music_box

        self._proximity_separator = PitchProximitySeparator(self.config, status_reporter)
        self._crossing_detector = VoiceCrossingDetector(self.config, status_reporter)
        self._role_assigner = VoiceRoleAssigner(self.config, status_reporter)
        self._octave_repair = OctaveJumpRepair(self.config, status_reporter)
        self._range_validator = InstrumentRangeValidator(self.config, status_reporter)
        self._witness_log = PersistentWitnessLog()

    def analyze(self, notes: List[NoteEvent],
                witness_log: Optional[PersistentWitnessLog] = None,
                audio_context: Optional[AudioContext] = None) -> VoiceContinuityResult:
        """
        Separate polyphonic notes into coherent voice streams.

        Returns:
            VoiceContinuityResult with voices, voice_crossings, octave_jumps, confidence
        """
        start_time = time.time()

        # FIX: Empty notes early return with correct field names
        if not notes:
            return VoiceContinuityResult(
                voices=[],
                voice_crossings=[],
                octave_jumps=[],
                confidence=Confidence.LOW
            )

        active_log = witness_log or self._witness_log

        # Filter out notes that are too short
        valid_notes = [n for n in notes if (n.end_ms - n.start_ms) >= self.config.min_note_duration_ms]
        filtered_count = len(notes) - len(valid_notes)
        if filtered_count > 0:
            logger.debug(f"VoiceContinuity: filtered {filtered_count} sub-threshold notes")

        # Choose separator
        has_cluster_tags = any(self._get_cluster_id(n) >= 0 for n in valid_notes)

        if has_cluster_tags and self.config.prefer_cluster_grouping:
            logger.debug("VoiceContinuity: using ClusterAwareVoiceSeparator")
            separator = ClusterAwareVoiceSeparator(self.config, active_log)
            voices = separator.separate_voices(valid_notes)
        else:
            logger.debug("VoiceContinuity: using PitchProximitySeparator")
            separator = self._proximity_separator
            voices = separator.separate_voices(valid_notes)

        logger.debug(f"VoiceContinuity: {len(voices)} voices from {len(valid_notes)} notes")

        # Detect voice crossings
        voice_crossings = []
        if self.config.detect_crossings and len(voices) > 1:
            voice_crossings = self._crossing_detector.detect_crossings(voices)
            if voice_crossings:
                logger.debug(f"VoiceContinuity: {len(voice_crossings)} voice crossings detected")

        # Assign musical roles
        voices = self._role_assigner.assign_roles(voices)

        # Flag large octave jumps within each voice. OctaveJumpRepair was
        # instantiated but never actually invoked here - the scan below for
        # "Large interval flagged" tags always found nothing, since
        # repair_voice() (the only thing that writes that tag) never ran.
        voices = [self._octave_repair.repair_voice(v) for v in voices]

        # Flag notes whose pitch falls outside their own voice's dominant
        # classified instrument family's real playable range - keeps a
        # voice's identity locked instead of letting range-implausible
        # notes pass through silently.
        voices = [self._range_validator.validate_voice(v) for v in voices]

        octave_jumps = []
        if self.config.repair_octave_jumps:
            for voice in voices:
                for note in voice.notes:
                    if note.reasoning_chain:
                        for rc in note.reasoning_chain:
                            if "Large interval flagged" in rc:
                                octave_jumps.append((voice.voice_id, note.start_ms, note.pitch))

        # Update witness log
        for voice in voices:
            for note in voice.notes:
                self._witness_log.add(WitnessEntry(
                    timestamp_ms=note.start_ms,
                    cluster_id=self._get_cluster_id(note),
                    voice_id=voice.voice_id,
                    instrument_family=_get_instrument_family(note),
                    pitch_range=(note.pitch, note.pitch),
                    spectral_centroid=0.0,
                    embedding_snapshot=None,
                    confidence=note.confidence,
                ))

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Calculate confidence based on voice count, crossings, and the
        # real per-note assignment cost quality from the separator (0.0 =
        # confident/evidence-based join, up to the 0.85 new-voice cutoff =
        # marginal join). A separation built from many marginal, forced
        # assignments should not report HIGH confidence just because the
        # resulting voice/crossing counts happen to look tidy.
        assignment_costs = getattr(separator, 'last_assignment_costs', [])
        mean_assignment_cost = float(np.mean(assignment_costs)) if assignment_costs else 0.0

        if len(voices) == 0:
            confidence = Confidence.HALLUCINATION
        elif len(voice_crossings) > len(voices) or mean_assignment_cost >= 0.6:
            confidence = Confidence.LOW
        elif len(voices) <= 4 and mean_assignment_cost < 0.3:
            confidence = Confidence.HIGH
        else:
            confidence = Confidence.MEDIUM

        logger.debug(
            f"VoiceContinuity: complete in {elapsed_ms:.0f}ms — {len(voices)} voices, {len(voice_crossings)} crossings")

        if self._music_box is not None:
            range_flags = sum(
                1 for v in voices for n in v.notes
                if n.reasoning_chain and any("Range implausible" in t for t in n.reasoning_chain)
            )
            self._music_box.log_decision(
                stage_name="voice_continuity",
                decision_type=DecisionType.ANALYSIS_EVIDENCE,
                before_state={"note_count": len(valid_notes)},
                after_state={
                    "voice_count": len(voices),
                    "crossing_count": len(voice_crossings),
                    "octave_jump_count": len(octave_jumps),
                    "range_violation_count": range_flags,
                    "mean_assignment_cost": mean_assignment_cost,
                    "confidence": confidence.value if hasattr(confidence, "value") else str(confidence),
                    "elapsed_ms": elapsed_ms,
                },
                reasoning=f"{len(voices)} voices from {len(valid_notes)} notes, "
                          f"mean assignment cost {mean_assignment_cost:.2f}",
                reversible=True,
            )

        if self.config.cleanup_after_analysis:
            gc.collect()

        # FIX: Return with correct field names - NO 'crossings' parameter
        return VoiceContinuityResult(
            voices=voices,
            voice_crossings=voice_crossings,
            octave_jumps=octave_jumps,
            confidence=confidence
        )

    def get_witness_log(self) -> PersistentWitnessLog:
        return self._witness_log

    def _get_cluster_id(self, note: NoteEvent) -> int:
        if hasattr(note, 'metadata') and isinstance(note.metadata, dict):
            return int(note.metadata.get('cluster_id', -1))
        if note.reasoning_chain:
            for tag in note.reasoning_chain:
                if isinstance(tag, str) and tag.startswith('cluster:'):
                    try:
                        return int(tag.split(':')[1])
                    except (IndexError, ValueError):
                        pass
        return -1

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": "VoiceContinuity",
            "version": "5.2.0",
            "config": {
                "max_voice_gap_ms": self.config.max_voice_gap_ms,
                "max_octave_jump_semitones": self.config.max_octave_jump_semitones,
                "max_voices": self.config.max_voices,
                "prefer_cluster_grouping": self.config.prefer_cluster_grouping,
            },
            "witness_log_size": len(self._witness_log),
        }


# ========================================================================
# Factory
# ========================================================================

def create_voice_continuity(config: Optional[VoiceConfig] = None,
                            status_reporter: Optional[StatusReporterProtocol] = None,
                            music_box: Optional[MusicBoxProtocol] = None) -> VoiceContinuity:
    return VoiceContinuity(config=config or VoiceConfig(),
                           status_reporter=status_reporter,
                           music_box=music_box)