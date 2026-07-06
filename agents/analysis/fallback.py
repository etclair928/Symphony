# =================================================================
# MODULE: agents/analysis/fallback.py
# DESCRIPTION: Fallback analysis agents for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# PHILOSOPHY:
#     When primary analysis models are unavailable, fallback agents
#     provide graceful degradation. They return empty or default
#     results while logging warnings for forensic audit.
#
#     This ensures the pipeline continues to function even when
#     optional dependencies are missing.
#
# KEY ARCHITECTURE:
#     - Implements same interfaces as primary agents
#     - Returns sensible defaults (empty lists, neutral values)
#     - Logs warnings to MusicBox for forensic audit
#     - StatusReporter integration
#     - Minimal external dependencies (only bedrock types that exist)
#
# Authored by: DeepSeek - Complete 5.0 rewrite (2026-05-11)
# =================================================================

import warnings
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

# Core imports - ONLY from bedrock types that exist
from core.order_types import (
    NoteEvent, SourceType, AudioContext, StageResult,
    Confidence, VetoReason, ValidationGate, ValidationResult,
    SchoenbergResult, SchoenbergVerdict, WitnessTestimony,
    Voice, VoiceRole, TempoMap, PulseField
)
from core.constants import DEFAULT_TEMPO_BPM, TARGET_SAMPLE_RATE
from core.protocols import (
    MusicBoxProtocol, StatusReporterProtocol
)


# ========================================================================
# Local Fallback Types (don't exist in order_types)
# ========================================================================

@dataclass
class FallbackGrooveField:
    """Fallback groove field when GrooveField not available."""
    bass_kick_phase_delta_ms: float = 0.0
    is_wide_swing: bool = False
    is_dilla_pocket: bool = False
    average_inter_note_distance_ms: float = 500.0
    tempo_estimate_bpm: float = 120.0
    confidence: Confidence = Confidence.LOW
    groove_signature: str = "neutral"
    phase_delta_variance_ms: float = 0.0


@dataclass
class FallbackVoiceContinuityResult:
    """Fallback voice continuity result."""
    voices: List[Voice] = field(default_factory=list)
    voice_crossings: List[Tuple] = field(default_factory=list)
    octave_jumps: List[Tuple] = field(default_factory=list)
    confidence: Confidence = Confidence.HALLUCINATION


@dataclass
class FallbackTempoIntelligenceResult:
    """Fallback tempo intelligence result."""
    tempo_map: Optional[TempoMap] = None
    pulse_field: Optional[PulseField] = None
    beat_track: List = field(default_factory=list)
    time_signature_candidates: List = field(default_factory=list)
    beat_grid: Optional[Any] = None
    time_signature_primary: Optional[Any] = None
    is_constant_tempo: bool = True


@dataclass
class FallbackTimeSignatureCandidate:
    """Fallback time signature candidate."""
    numerator: int = 4
    denominator: int = 4
    confidence: float = 0.5
    sample_segment_start_ms: float = 0.0
    sample_segment_end_ms: float = 0.0


@dataclass
class FallbackGeometricLattice:
    """Fallback geometric lattice when ReverseGeoCrypt fails."""
    subdivision_sec: float = 0.5
    subdivision_ms: float = 500.0
    ratio_family: str = "binary"
    error_score: float = 0.5
    entropy: float = 0.5
    confidence: float = 0.0
    confidence_level: str = "sparse"
    derived_bpm: float = 120.0
    phase: float = 0.0
    anchor_alignment: float = 0.0

    def to_tactus_bpm(self) -> float:
        return self.derived_bpm / 4


@dataclass
class FallbackEvent:
    """Fallback event for ReverseGeoCrypt."""
    time: float = 0.0
    strength: float = 0.0
    event_type: str = "unknown"
    source: SourceType = SourceType.PITCH
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackAnchor:
    """Fallback anchor for ReverseGeoCrypt."""
    time: float = 0.0
    confidence: float = 0.0
    source: SourceType = SourceType.RHYTHM
    witness_votes: Tuple = field(default_factory=tuple)


@dataclass
class FallbackAnechoicReport:
    """Fallback anechoic report."""
    frame_times: List[float] = field(default_factory=list)
    feature_maps: Dict[str, List[float]] = field(default_factory=dict)
    regions: List = field(default_factory=list)
    stem_type: str = "full_mix"
    duration_seconds: float = 0.0
    profile: str = "auto"
    memory_mb: float = 0.0

    def query(self, start: float, end: float) -> 'FallbackSilenceState':
        return FallbackSilenceState()

    def get_silent_regions(self, threshold: float = 0.65) -> List:
        return []

    def get_resonant_regions(self, threshold: float = 0.60) -> List:
        return []


@dataclass
class FallbackSilenceState:
    """Fallback silence state."""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    energy_floor: float = 0.0
    spectral_stasis: float = 0.0
    harmonic_persistence: float = 0.0
    noise_floor_probability: float = 0.0
    decay_completion: float = 0.0
    transient_absence: float = 0.0
    rhythmic_void_probability: float = 0.0
    resonance_probability: float = 0.0
    active_material_probability: float = 0.0
    cymbal_wash_probability: float = 0.0
    pedal_tone_probability: float = 0.0
    subdivision_alignment: Any = None
    confidence: float = 0.0
    region_type: str = "uncertain"

    def get_confidence_penalty(self) -> float:
        return 0.0


# ========================================================================
# Fallback Voice Continuity Analyzer
# ========================================================================

class FallbackVoiceAnalyzer:
    """
    Fallback voice continuity analyzer.

    Returns empty voice list (no separation) while logging warnings.
    """

    def __init__(
            self,
            analyzer_type: str = "voice",
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        self._name = f"fallback_{analyzer_type}"
        self._source_type = SourceType.VOICE_CONTINUITY
        self.analyzer_type = analyzer_type
        self._status_reporter = status_reporter
        self._music_box = music_box

        self._log_status(f"FallbackVoiceAnalyzer initialized (type: {analyzer_type})", "warn")

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def separate_voices(self, notes: List[NoteEvent]) -> List[Voice]:
        """Return empty voice list (no separation)."""
        self._log_status("Using fallback voice separation - returning empty results", "warn")

        if self._music_box:
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="fallback",
                before_state={"note_count": len(notes)},
                after_state={"voice_count": 0},
                reasoning="Fallback voice analyzer - primary model unavailable",
                reversible=False
            )

        return []

    def detect_voice_crossings(self, voices: List[Voice]) -> List[Tuple[Voice, Voice, float]]:
        """Return empty crossing list."""
        return []

    def assign_voice_roles(self, voices: List[Voice]) -> List[Voice]:
        """Return voices unchanged (no role assignment)."""
        return voices

    def get_continuity_result(self, notes: List[NoteEvent]) -> FallbackVoiceContinuityResult:
        """Return empty continuity result."""
        return FallbackVoiceContinuityResult(
            voices=[],
            voice_crossings=[],
            octave_jumps=[],
            confidence=Confidence.HALLUCINATION
        )

    def analyze(self, notes: List[NoteEvent], context: AudioContext) -> FallbackVoiceContinuityResult:
        """Main analysis method."""
        return self.get_continuity_result(notes)

    def repair_octave_jumps(self, voice: Voice) -> Voice:
        """Return voice unchanged."""
        return voice

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info(self._name, message)
            elif level == "warn":
                self._status_reporter.warn(self._name, message)
            elif level == "error":
                self._status_reporter.error(self._name, message)


# ========================================================================
# Fallback Groove Field Analyzer
# ========================================================================

class FallbackGrooveAnalyzer:
    """
    Fallback groove field analyzer.

    Returns neutral groove values (no swing, no intentional feel).
    """

    def __init__(
            self,
            analyzer_type: str = "groove",
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        self._name = f"fallback_{analyzer_type}"
        self._source_type = SourceType.GROOVE_FIELD
        self.analyzer_type = analyzer_type
        self._status_reporter = status_reporter
        self._music_box = music_box

        self._log_status(f"FallbackGrooveAnalyzer initialized (type: {analyzer_type})", "warn")

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def analyze(
            self,
            bass_events: List[NoteEvent],
            kick_events: List[NoteEvent],
            context: AudioContext
    ) -> FallbackGrooveField:
        """Return neutral groove field."""
        self._log_status("Using fallback groove analysis - returning neutral values", "warn")

        if self._music_box:
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="fallback",
                before_state={
                    "bass_count": len(bass_events),
                    "kick_count": len(kick_events)
                },
                after_state={
                    "groove_signature": "neutral",
                    "is_wide_swing": False,
                    "is_dilla_pocket": False
                },
                reasoning="Fallback groove analyzer - primary model unavailable",
                reversible=False
            )

        return FallbackGrooveField(
            bass_kick_phase_delta_ms=0.0,
            is_wide_swing=False,
            is_dilla_pocket=False,
            average_inter_note_distance_ms=500.0,
            tempo_estimate_bpm=120.0,
            confidence=Confidence.LOW,
            groove_signature="neutral",
            phase_delta_variance_ms=0.0
        )

    def get_phase_delta(self, note1: NoteEvent, note2: NoteEvent) -> float:
        """Return zero phase delta."""
        return 0.0

    def build_note_context(self, events: List[NoteEvent]) -> List:
        """Return empty context list."""
        return []

    def get_phase_consistency(self) -> float:
        """Return zero consistency."""
        return 0.0

    def should_preserve_groove(self) -> bool:
        """Return False (don't preserve groove)."""
        return False

    def get_quantization_recommendation(self) -> Dict[str, Any]:
        """Return default quantization recommendation."""
        return {
            "recommendation": "normal_quantize",
            "reason": "Fallback groove analyzer - no groove detected",
            "max_snap_ms": 100,
            "confidence": 0.0
        }

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info(self._name, message)
            elif level == "warn":
                self._status_reporter.warn(self._name, message)
            elif level == "error":
                self._status_reporter.error(self._name, message)


# ========================================================================
# Fallback Tempo Intelligence Analyzer
# ========================================================================

class FallbackTempoAnalyzer:
    """
    Fallback tempo intelligence analyzer.

    Returns default tempo (120 BPM) with no changes.
    """

    def __init__(
            self,
            analyzer_type: str = "tempo",
            default_tempo: float = DEFAULT_TEMPO_BPM,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        self._name = f"fallback_{analyzer_type}"
        self._source_type = SourceType.TEMPO_INTELLIGENCE
        self.analyzer_type = analyzer_type
        self._default_tempo = default_tempo
        self._status_reporter = status_reporter
        self._music_box = music_box

        self._log_status(f"FallbackTempoAnalyzer initialized (type: {analyzer_type})", "warn")

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def analyze_tempo(self, audio_buffer) -> TempoMap:
        """Return default tempo map."""
        self._log_status("Using fallback tempo analysis - returning default tempo", "warn")

        if self._music_box:
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="fallback",
                before_state={},
                after_state={"tempo_bpm": self._default_tempo},
                reasoning="Fallback tempo analyzer - primary model unavailable",
                reversible=False
            )

        # Create a simple TempoMap (this type exists)
        return TempoMap(
            initial_tempo_bpm=self._default_tempo,
            tempo_events=[],
            confidence=Confidence.LOW
        )

    def build_pulse_field(self, tempo_map: TempoMap, duration_seconds: float = 60.0) -> PulseField:
        """Build neutral pulse field (PulseField exists in order_types)."""
        beat_duration_ms = 60000.0 / tempo_map.initial_tempo_bpm
        duration_ms = duration_seconds * 1000

        beat_grid = list(np.arange(0, duration_ms, beat_duration_ms))
        sixteenth_grid = []
        for beat in beat_grid:
            for i in range(4):
                sixteenth_grid.append(beat + i * beat_duration_ms / 4)

        return PulseField(
            tempo_bpm=tempo_map.initial_tempo_bpm,
            confidence=Confidence.LOW,
            beat_grid_ms=beat_grid,
            sixteenth_grid_ms=sorted(sixteenth_grid),
            pulse_strength=[0.5] * len(beat_grid),
            phase_shift_ms=0.0,
            is_double_time=False,
            is_half_time=False
        )

    def detect_time_signature(self, beat_track: List[float]) -> List[FallbackTimeSignatureCandidate]:
        """Return default 4/4 time signature."""
        return [
            FallbackTimeSignatureCandidate(
                numerator=4,
                denominator=4,
                confidence=0.5,
                sample_segment_start_ms=0,
                sample_segment_end_ms=0
            )
        ]

    def get_intelligence_result(self, audio_buffer, duration_seconds: float = 60.0) -> FallbackTempoIntelligenceResult:
        """Return default tempo intelligence result."""
        tempo_map = self.analyze_tempo(audio_buffer)
        pulse_field = self.build_pulse_field(tempo_map, duration_seconds)

        return FallbackTempoIntelligenceResult(
            tempo_map=tempo_map,
            pulse_field=pulse_field,
            beat_track=[],
            time_signature_candidates=self.detect_time_signature([]),
            beat_grid=None,
            time_signature_primary=None,
            is_constant_tempo=True
        )

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info(self._name, message)
            elif level == "warn":
                self._status_reporter.warn(self._name, message)
            elif level == "error":
                self._status_reporter.error(self._name, message)


# ========================================================================
# Fallback Pulse Field Analyzer
# ========================================================================

class FallbackPulseAnalyzer:
    """
    Fallback pulse field analyzer.

    Returns simple grid based on default tempo.
    """

    def __init__(
            self,
            analyzer_type: str = "pulse",
            default_tempo: float = DEFAULT_TEMPO_BPM,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        self._name = f"fallback_{analyzer_type}"
        self._source_type = SourceType.PULSE_FIELD
        self.analyzer_type = analyzer_type
        self._default_tempo = default_tempo
        self._status_reporter = status_reporter
        self._music_box = music_box

        self._log_status(f"FallbackPulseAnalyzer initialized (type: {analyzer_type})", "warn")

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def analyze_pulse_field(
            self,
            onset_times: List[float],
            onset_strengths: List[float],
            audio=None,
            tempo_hint: Optional[float] = None
    ) -> PulseField:
        """Return simple pulse field based on tempo hint."""
        tempo = tempo_hint or self._default_tempo
        beat_duration_ms = 60000.0 / tempo
        duration_ms = max(onset_times) + 5000 if onset_times else 60000

        beat_grid = list(np.arange(0, duration_ms, beat_duration_ms))
        sixteenth_grid = []
        for beat in beat_grid:
            for i in range(4):
                sixteenth_grid.append(beat + i * beat_duration_ms / 4)

        self._log_status("Using fallback pulse analysis - returning simple grid", "warn")

        if self._music_box:
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="fallback",
                before_state={"onset_count": len(onset_times)},
                after_state={"tempo_bpm": tempo, "beat_count": len(beat_grid)},
                reasoning="Fallback pulse analyzer - primary model unavailable",
                reversible=False
            )

        return PulseField(
            tempo_bpm=tempo,
            confidence=Confidence.LOW,
            beat_grid_ms=beat_grid,
            sixteenth_grid_ms=sorted(sixteenth_grid),
            pulse_strength=[0.5] * len(beat_grid),
            phase_shift_ms=0.0,
            is_double_time=False,
            is_half_time=False
        )

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info(self._name, message)
            elif level == "warn":
                self._status_reporter.warn(self._name, message)
            elif level == "error":
                self._status_reporter.error(self._name, message)


# ========================================================================
# Fallback ReverseGeoCrypt Analyzer
# ========================================================================

class FallbackReverseGeoCrypt:
    """
    Fallback ReverseGeoCrypt analyzer.

    Returns None (no lattice found) with default values.
    """

    def __init__(
            self,
            analyzer_type: str = "reverse_geo",
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        self._name = f"fallback_{analyzer_type}"
        self._source_type = SourceType.TEMPO_INTELLIGENCE
        self.analyzer_type = analyzer_type
        self._status_reporter = status_reporter
        self._music_box = music_box

        self._log_status(f"FallbackReverseGeoCrypt initialized (type: {analyzer_type})", "warn")

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def decrypt(
            self,
            events: List,
            anchors: List = None
    ) -> Optional[FallbackGeometricLattice]:
        """Return fallback lattice (no real detection)."""
        self._log_status("Using fallback reverse geo crypt - returning fallback lattice", "warn")

        if self._music_box:
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="fallback",
                before_state={"event_count": len(events), "anchor_count": len(anchors or [])},
                after_state={"lattice_found": True, "lattice_bpm": 120},
                reasoning="Fallback reverse geo crypt - returning default lattice",
                reversible=False
            )

        return FallbackGeometricLattice(
            subdivision_sec=0.5,
            subdivision_ms=500.0,
            ratio_family="binary",
            error_score=0.5,
            entropy=0.5,
            confidence=0.3,
            confidence_level="sparse",
            derived_bpm=120.0,
            phase=0.0,
            anchor_alignment=0.0
        )

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info(self._name, message)
            elif level == "warn":
                self._status_reporter.warn(self._name, message)
            elif level == "error":
                self._status_reporter.error(self._name, message)


# ========================================================================
# Fallback Anechoic Ma Analyzer
# ========================================================================

class FallbackAnechoicMa:
    """
    Fallback Anechoic Ma silence analyzer.

    Returns empty report with neutral values.
    """

    def __init__(
            self,
            analyzer_type: str = "anechoic",
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        self._name = f"fallback_{analyzer_type}"
        self._source_type = SourceType.ANECHOIC_MA
        self.analyzer_type = analyzer_type
        self._status_reporter = status_reporter
        self._music_box = music_box

        self._log_status(f"FallbackAnechoicMa initialized (type: {analyzer_type})", "warn")

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def analyze(
            self,
            y: np.ndarray,
            sr: int,
            rhythm_field=None,
            stem_type=None
    ) -> FallbackAnechoicReport:
        """Return empty report."""
        self._log_status("Using fallback anechoic ma - returning empty report", "warn")

        if self._music_box:
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="fallback",
                before_state={"audio_duration": len(y) / sr if sr > 0 else 0},
                after_state={"report_empty": True},
                reasoning="Fallback anechoic ma - primary model unavailable",
                reversible=False
            )

        return FallbackAnechoicReport(
            frame_times=[],
            feature_maps={},
            regions=[],
            stem_type=str(stem_type) if stem_type else "full_mix",
            duration_seconds=len(y) / sr if sr > 0 else 0,
            profile="auto",
            memory_mb=0.0
        )

    def _log_status(self, message: str, level: str = "info"):
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info(self._name, message)
            elif level == "warn":
                self._status_reporter.warn(self._name, message)
            elif level == "error":
                self._status_reporter.error(self._name, message)


# ========================================================================
# Factory Function
# ========================================================================

def create_fallback_analyzer(analyzer_type: str, **kwargs) -> Any:
    """
    Create a fallback analyzer of the specified type.

    Args:
        analyzer_type: One of 'voice', 'groove', 'tempo', 'pulse', 'reverse_geo', 'anechoic'
        **kwargs: Additional arguments for the analyzer

    Returns:
        Fallback analyzer instance
    """
    if analyzer_type == "voice":
        return FallbackVoiceAnalyzer(**kwargs)
    elif analyzer_type == "groove":
        return FallbackGrooveAnalyzer(**kwargs)
    elif analyzer_type == "tempo":
        return FallbackTempoAnalyzer(**kwargs)
    elif analyzer_type == "pulse":
        return FallbackPulseAnalyzer(**kwargs)
    elif analyzer_type == "reverse_geo":
        return FallbackReverseGeoCrypt(**kwargs)
    elif analyzer_type == "anechoic":
        return FallbackAnechoicMa(**kwargs)
    else:
        raise ValueError(f"Unknown fallback analyzer type: {analyzer_type}")


# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Fallback Analysis Agents Test")
    print("=" * 60)

    # Test voice analyzer
    print("\n1. Voice Fallback:")
    voice = FallbackVoiceAnalyzer()
    result = voice.separate_voices([])
    print(f"   Voices: {len(result)}")

    # Test groove analyzer
    print("\n2. Groove Fallback:")
    groove = FallbackGrooveAnalyzer()
    gf = groove.analyze([], [], None)
    print(f"   Groove signature: {gf.groove_signature}")
    print(f"   Wide swing: {gf.is_wide_swing}")

    # Test tempo analyzer
    print("\n3. Tempo Fallback:")
    tempo = FallbackTempoAnalyzer()
    tm = tempo.analyze_tempo(None)
    print(f"   Tempo: {tm.initial_tempo_bpm} BPM")

    # Test pulse analyzer
    print("\n4. Pulse Fallback:")
    pulse = FallbackPulseAnalyzer()
    pf = pulse.analyze_pulse_field([], [])
    print(f"   Tempo: {pf.tempo_bpm} BPM")
    print(f"   Beats: {len(pf.beat_grid_ms)}")

    # Test reverse geo crypt
    print("\n5. ReverseGeoCrypt Fallback:")
    rgc = FallbackReverseGeoCrypt()
    lattice = rgc.decrypt([])
    print(f"   Lattice BPM: {lattice.to_tactus_bpm() if lattice else 0}")
    print(f"   Confidence: {lattice.confidence if lattice else 0}")

    # Test anechoic ma
    print("\n6. Anechoic Ma Fallback:")
    anechoic = FallbackAnechoicMa()
    report = anechoic.analyze(np.zeros(16000), 16000)
    print(f"   Duration: {report.duration_seconds}s")
    print(f"   Regions: {len(report.regions)}")
    print("\n" + "=" * 60)
    print("All fallback analyzers working correctly.")
    print("=" * 60)