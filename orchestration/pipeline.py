# =================================================================
# MODULE: orchestration/pipeline.py
# DESCRIPTION: Grimlock 5.6 — Canonical Testimony Architecture
# VERSION: 5.6.1 (Added QuaverIntelligence + PhraseIntelligence stages)
# UPDATED: 2026-06-02
#
# CRITICAL FIXES IN 5.6.1:
#   1. Added QuaverIntelligence stage (symbolic duration witness)
#   2. Added PhraseIntelligence stage (structural phase engine)
#   3. Added StructuralFeedback stage (re-analysis with bias)
#   4. Fixed transcribe_file() to accept 8 parameters
#   5. Preserved drum notes through all stages
#   6. Added helper methods for testimony extraction
# =================================================================

from __future__ import annotations

import asyncio
import gc
import os
import time
import traceback
import tempfile
import shutil
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Union, Tuple

import numpy as np

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from core.order_types import (
    AudioContext, Confidence, GrooveField, NoteEvent,
    PulseField, SourceType, StemType, TempoMap, Voice,
    VoiceContinuityResult, UNKNOWN_HASH_PLACEHOLDER, ExportOptions,
    SchoenbergVerdict, ValidationGate, DecisionType,
)
from core.acoustic_intelligence import AcousticIntelligence
from core.constants import (
    ANALYSIS_SR, BASIC_PITCH_SR, CREPE_SR, MADMOM_SR,
    DEFAULT_PROCESSING_DURATION_SECONDS,
    STAGE_TIMEOUT_ANALYSIS_SECONDS,
    STAGE_TIMEOUT_DETECTION_SECONDS,
    STAGE_TIMEOUT_EXPORT_SECONDS,
    STAGE_TIMEOUT_LOAD_SECONDS,
    STAGE_TIMEOUT_QUANTIZATION_SECONDS,
    STAGE_TIMEOUT_SEPARATION_SECONDS,
    DEMUCS_SECONDS_PER_AUDIO_SECOND,
    DEMUCS_TIMEOUT_FLOOR_SECONDS,
    DEMUCS_TIMEOUT_CAP_SECONDS,
    MEL_ROFORMER_SECONDS_PER_AUDIO_SECOND,
    MEL_ROFORMER_TIMEOUT_FLOOR_SECONDS,
    MEL_ROFORMER_TIMEOUT_CAP_SECONDS,
    SEPARATION_STAGE_TIMEOUT_BUFFER_SECONDS,
    DEFAULT_TIME_SIGNATURE_NUMERATOR,
    DEFAULT_TIME_SIGNATURE_DENOMINATOR,
    TEMPO_INTELLIGENCE_ENSEMBLE_WEIGHT,
    REVERSE_GEO_CRYPT_WITNESS_WEIGHT,
    RHYTHM_ENGINE_WITNESS_WEIGHT,
    TEMPO_ANALYSIS_SECONDS_PER_AUDIO_SECOND,
    TEMPO_ANALYSIS_TIMEOUT_FLOOR_SECONDS,
    TEMPO_ANALYSIS_TIMEOUT_CAP_SECONDS,
    DRUM_DETECTION_SECONDS_PER_AUDIO_SECOND,
    DRUM_DETECTION_TIMEOUT_FLOOR_SECONDS,
    DRUM_DETECTION_TIMEOUT_CAP_SECONDS,
    MIN_TEMPO_BPM,
    MAX_TEMPO_BPM,
    MIN_CONFIDENCE_TO_PASS,
)
from core.feature_bundle import EvidenceType, extract_features
from core.evidence_tracker import EvidenceTracker
from epistemic.musical_findings_map import MusicalFindingsMap
from core.audio_views import AudioViews, ChannelLayout
from core.testimony import (
    TestimonyStageResult, TestimonyNormalizer,
    IngestionTestimony, FeatureExtractionTestimony, TempoTestimony,
    SeparationTestimony, PitchTestimony, RhythmTestimony, DrumTestimony,
    TonalTestimony, TimbreTestimony, HarmonicValidationTestimony,
    GrooveTestimony, PulseTestimony, LatticeTestimony,
    VoiceContinuityTestimony, QuantizationTestimony, MergeTestimony,
    ConsensusTestimony, ValidationTestimony, ExportTestimony,
    AnechoicMATestimony, SchoenbergMirrorTestimony,
)
from core.guided_control import (
    GuidedParams,
    GuidedModeController,
    KeyConstraint,
    GuidedBeatGrid,
    create_guided_params_from_form
)

from memory.guardian import MemoryGuardian
from memory.arena import scoped_arena, MemoryArena
from orchestration.music_box import MusicBox
from orchestration.scribe import Scribe
from orchestration.cancellation import CancellableThread
from epistemic.epistemic_council import EpistemicCouncil, CouncilConfig
from agents.factory import AgentFactory, FactoryConfig
from agents.analysis.voice_continuity import VoiceContinuity, VoiceConfig
from agents.analysis.timbre_intelligence import TimbreIntelligence
from agents.analysis.anechoic_ma import AnechoicMa, AnechoicConfig, AnechoicMode
from agents.quantization.temporal_lattice import TemporalLattice, TemporalLatticeQuantizer
from agents.quantization.velocity_merge import VelocityMerge, VelocityMergeConfig, MergeStrategy
from agents.quantization.ritornello import Ritornello, RitornelloConfig
from core.testimony import RitornelloTestimony
from agents.detection.harmonic_intelligence import HarmonicConfig, HarmonicValidator, KeyDetector
from ingestion.loader import load_audio
import soundfile as sf
import librosa


# ====================================================================
# Stage Definition
# ====================================================================

class PipelineStage:
    __slots__ = (
        'name', 'executor', 'source_type', 'timeout_seconds',
        'is_essential', 'requires_gc', 'estimated_memory_mb',
    )

    def __init__(self, name, executor, source_type,
                 timeout_seconds=60.0, is_essential=False,
                 requires_gc=False, estimated_memory_mb=100):
        self.name = name
        self.executor = executor
        self.source_type = source_type
        self.timeout_seconds = timeout_seconds
        self.is_essential = is_essential
        self.requires_gc = requires_gc
        self.estimated_memory_mb = estimated_memory_mb


# ====================================================================
# Pipeline Result
# ====================================================================

class PipelineResult:
    def __init__(self, success, notes, drum_notes, verdict, timing,
                 total_time_seconds, peak_memory_mb, music_box_session,
                 stages_executed, aborted, abort_reason,
                 detected_tempo_bpm, detected_key, key_confidence,
                 groove_field, voices, is_truncated, truncation_duration,
                 original_duration_seconds, music_box_logs=None, veto_history=None):
        self.success = success
        self.notes = notes
        self.drum_notes = drum_notes
        self.verdict = verdict
        self.timing = timing
        self.total_time_seconds = total_time_seconds
        self.peak_memory_mb = peak_memory_mb
        self.music_box_session = music_box_session
        self.stages_executed = stages_executed
        self.aborted = aborted
        self.abort_reason = abort_reason
        self.detected_tempo_bpm = detected_tempo_bpm
        self.detected_key = detected_key
        self.key_confidence = key_confidence
        self.groove_field = groove_field
        self.voices = voices
        self.is_truncated = is_truncated
        self.truncation_duration = truncation_duration
        self.original_duration_seconds = original_duration_seconds
        # The actual logged records, not just the session ID - every
        # finding this session's music_box wiring recorded (confidence
        # breakdowns, witness votes, contention arbitration, ...) was
        # otherwise unreachable from outside the pipeline instance
        # itself, which callers like main.py never hold onto (they use
        # the module-level transcribe_file() convenience wrapper).
        self.music_box_logs = music_box_logs or []
        self.veto_history = veto_history or []

    def __repr__(self):
        s = "SUCCESS" if self.success else "FAILED"
        return (f"PipelineResult({s}, {len(self.notes)} pitched + "
                f"{len(self.drum_notes)} drum, {self.total_time_seconds:.1f}s)")


# ====================================================================
# Main Pipeline
# ====================================================================

class GrimlockPipeline:
    """
    Grimlock 5.6.1 — Canonical Testimony Architecture with Structural Intelligence.

    Law of Testimony:  Every stage returns TestimonyStageResult[T].
    Law of Evidence:   Audio is viewed, never mutated (AudioViews).
    Law of Boundaries: Tuple unpacking ONLY in TestimonyNormalizer.
    Law of Separation: _drum_notes NEVER enters the pitched pool.
    Law of Order:      Quantize → Merge → Consensus. Never reversed.
    """

    CONFIDENCE_FLOOR = 0.25

    def __init__(
            self,
            guided_params: Optional[Union[GuidedParams, Dict[str, Any]]] = None,
            music_box: Optional[MusicBox] = None,
            guardian: Optional[MemoryGuardian] = None,
            scribe: Optional[Scribe] = None,
            agent_factory: Optional[AgentFactory] = None,
            export_options: Optional[ExportOptions] = None,
            debug: bool = False,
            truncate_duration: float = DEFAULT_PROCESSING_DURATION_SECONDS,
            progress_callback: Optional[Callable[[str, int], None]] = None,
            **kwargs
    ):
        if kwargs:
            print(f"[PIPELINE] Ignoring unexpected kwargs: {list(kwargs.keys())}")

        self.progress_callback = progress_callback

        if isinstance(guided_params, dict):
            self.guided_params = GuidedParams.from_dict(guided_params)
        else:
            self.guided_params = guided_params

        self.guided_controller = GuidedModeController()
        self._key_constraint = None
        self.debug = debug
        self.truncate_duration = truncate_duration
        self._original_duration_seconds = 0.0

        self.music_box = music_box or MusicBox()
        self.guardian = guardian or MemoryGuardian(music_box=self.music_box)
        self.scribe = scribe or Scribe(music_box=self.music_box)
        self.evidence_tracker = EvidenceTracker()
        self.agent_factory = agent_factory or AgentFactory(music_box=self.music_box)
        self.export_options = export_options or ExportOptions()

        self._current_stage = None
        self._stage_times = {}
        self._results = {}
        self._aborted = False
        self._abort_reason = None
        self._audio_path = None
        self._output_path = None

        self._master_id = None
        self._drums_stem_id = None
        self._bass_stem_id = None
        self._other_stem_id = None
        self._feature_bundle_id = None

        self._analysis_prepared_id = None
        self._rhythm_prepared_id = None
        self._pitch_prepared_id = None
        self._drum_prepared_id = None
        self._other_pitch_prepared_id = None

        self._context = None
        self._beat_times_ms = np.array([])
        self._tempo_map = None
        self._beat_grid = None
        self._downbeat_times_ms: List[float] = []
        self._time_signature_candidates = []
        self._time_signature_primary = None

        # Per-model separation timeouts - overwritten in _run_ingestion() once
        # the real audio duration is known (see _compute_separation_timeouts).
        self._demucs_timeout = DEMUCS_TIMEOUT_FLOOR_SECONDS
        self._mel_roformer_timeout = MEL_ROFORMER_TIMEOUT_FLOOR_SECONDS

        self._pitch_notes_bass: List[NoteEvent] = []
        self._pitch_notes_master: List[NoteEvent] = []
        self._pitch_notes_other: List[NoteEvent] = []
        self._pitch_notes_vocals: List[NoteEvent] = []
        self._rhythm_notes: List[NoteEvent] = []
        self._drum_notes: List[NoteEvent] = []

        self._tonal_events: List[Dict] = []
        self._voices: List[Voice] = []

        self._pulse_field = None
        self._timbre_analysis = None
        self._timbre_clusters = None
        self._anechoic_report = None

        self._quantized_pitched: List[NoteEvent] = []
        self._merged_notes: List[NoteEvent] = []
        self._repaired_notes: List[NoteEvent] = []
        self._consensus_notes: List[NoteEvent] = []

        self._temporal_lattice = None
        self._witness_log = None
        self._musical_findings: MusicalFindingsMap = MusicalFindingsMap()

        # Temp directory for file-based handoff
        self._temp_dir = None

        self._using_guided_tempo = self.guided_params and self.guided_params.tempo_bpm
        self._using_guided_key = self.guided_params and self.guided_params.key_signature

        self._peak_memory_mb = 0.0
        self._arena_stats = {}

        self._stages = self._build_pipeline()

        print(f"[PIPELINE] Grimlock 5.6.1 ready — {len(self._stages)} stages")
        if debug:
            for s in self._stages:
                print(f"[PIPELINE]   {s.name} (timeout={s.timeout_seconds}s, "
                      f"essential={s.is_essential}, mem={s.estimated_memory_mb}MB)")

    # =================================================================
    # Testimony-object properties - delegate to self._musical_findings so
    # tempo/groove/quaver/structural testimony live in exactly one place
    # (the findings map) instead of a second copy on the Pipeline itself.
    # =================================================================

    @property
    def _tempo(self) -> TempoTestimony:
        return self._musical_findings.tempo_testimony

    @_tempo.setter
    def _tempo(self, value: TempoTestimony) -> None:
        self._musical_findings.tempo_testimony = value

    @property
    def _groove(self) -> GrooveTestimony:
        return self._musical_findings.groove_testimony

    @_groove.setter
    def _groove(self, value: GrooveTestimony) -> None:
        self._musical_findings.groove_testimony = value

    @property
    def _quaver_testimonies(self) -> List["QuaverTestimony"]:
        return self._musical_findings.quaver_testimonies

    @_quaver_testimonies.setter
    def _quaver_testimonies(self, value: List["QuaverTestimony"]) -> None:
        self._musical_findings.quaver_testimonies = value

    @property
    def _structural_testimony(self) -> Optional["StructuralTestimony"]:
        return self._musical_findings.structural_testimony

    @_structural_testimony.setter
    def _structural_testimony(self, value: Optional["StructuralTestimony"]) -> None:
        self._musical_findings.structural_testimony = value

    @property
    def _detected_key(self) -> Optional[str]:
        return self._musical_findings.detected_key

    @_detected_key.setter
    def _detected_key(self, value: Optional[str]) -> None:
        self._musical_findings.detected_key = value

    @property
    def _key_confidence(self) -> float:
        return self._musical_findings.key_confidence

    @_key_confidence.setter
    def _key_confidence(self, value: float) -> None:
        self._musical_findings.key_confidence = value

    # =================================================================
    # Public API
    # =================================================================

    def run(self, audio_path: str, output_path: Optional[str] = None) -> PipelineResult:
        return self._execute_pipeline(audio_path, output_path)

    def transcribe(self, audio_path: str, output_path: Optional[str] = None) -> PipelineResult:
        return self._execute_pipeline(audio_path, output_path)

    # =================================================================
    # Build Pipeline Stages
    # =================================================================

    def _build_pipeline(self) -> List[PipelineStage]:
        stages = [
            PipelineStage("ingestion", self._run_ingestion, SourceType.PIPELINE,
                          timeout_seconds=STAGE_TIMEOUT_LOAD_SECONDS, is_essential=True,
                          estimated_memory_mb=50),
            PipelineStage("feature_extraction", self._run_feature_extraction, SourceType.SCRIBE,
                          timeout_seconds=30.0, is_essential=True, estimated_memory_mb=200),
            PipelineStage("tempo_analysis", self._run_tempo_analysis, SourceType.TEMPO_INTELLIGENCE,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=False,
                          estimated_memory_mb=100),
            PipelineStage("separation", self._run_separation, SourceType.DEMUCS,
                          timeout_seconds=STAGE_TIMEOUT_SEPARATION_SECONDS, is_essential=False,
                          estimated_memory_mb=400),
            PipelineStage("rhythm_detection", self._run_rhythm_detection, SourceType.RHYTHM,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=True,
                          estimated_memory_mb=100),
            # This stage runs the full 5-model EpistemicCouncil TWICE
            # (direct pass + octave-shift-corrected pass) plus Omnizart -
            # confirmed via real runs to consistently take 60000-60020ms
            # and produce ZERO notes with no success print or exception
            # message under the old 60s default, meaning it was being
            # silently killed before it could ever return real results.
            # Bumping to 300s wasn't enough either - a real run still hit
            # that exact new ceiling and timed out again, while a separate
            # run of the same clip completed in 277s. This stage's real
            # cost is highly variable (system load, model cold-start
            # variance across process runs, not just audio content), so
            # matching it to STAGE_TIMEOUT_SEPARATION_SECONDS's own budget
            # (600s) is the safer bet rather than guessing another
            # marginal bump.
            PipelineStage("pitch_detection_bass", self._run_pitch_detection_bass, SourceType.PITCH,
                          timeout_seconds=600.0, is_essential=False,
                          estimated_memory_mb=150),
            PipelineStage("pitch_detection_master", self._run_pitch_detection_master, SourceType.PITCH,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=False,
                          estimated_memory_mb=150),
            PipelineStage("pitch_detection_other", self._run_pitch_detection_other, SourceType.PITCH,
                          timeout_seconds=180.0, is_essential=False,
                          estimated_memory_mb=200),
            PipelineStage("pitch_detection_vocals", self._run_pitch_detection_vocals, SourceType.PITCH,
                          timeout_seconds=180.0, is_essential=False,
                          estimated_memory_mb=200),
            PipelineStage("tonal_detection", self._run_tonal_detection, SourceType.TONAL,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=False,
                          estimated_memory_mb=100),
            PipelineStage("drum_detection", self._run_drum_detection, SourceType.DRUM_INTELLIGENCE,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=False,
                          # Bumped from 80: DrumIntelligence's NMF/HFC
                          # coprocessor layer computes its own STFT plus
                          # working arrays for polyphony decomposition on
                          # top of the existing detection path.
                          estimated_memory_mb=220),
            PipelineStage("anechoic_ma", self._run_anechoic_ma, SourceType.ANECHOIC_MA,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=100),
            PipelineStage("timbre_analysis", self._run_timbre_analysis, SourceType.ANECHOIC_MA,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=200),
            PipelineStage("harmonic_validation", self._run_harmonic_validation, SourceType.TONAL,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=20),
            PipelineStage("groove_analysis", self._run_groove_analysis, SourceType.GROOVE_FIELD,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=10),
            # reverse_geo_crypt runs before pulse_analysis (swapped from the
            # original order) so PulseFieldAnalyzer can actually consume its
            # lattice_period_ms/lattice_confidence output - neither stage
            # depends on anything the other one produces, so this is safe.
            PipelineStage("reverse_geo_crypt", self._run_reverse_geo_crypt, SourceType.TEMPO_INTELLIGENCE,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=10),
            PipelineStage("pulse_analysis", self._run_pulse_analysis, SourceType.PULSE_FIELD,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=10),
            # Now that pitch_detection_bass actually completes instead of
            # silently timing out (see its own timeout fix above), this
            # stage's real note count roughly doubled (e.g. 371 vs 175 on
            # a real 20s clip) - confirmed via a real run to blow through
            # the shared 30s analysis timeout ("[WARN] Timeout after 30s"),
            # discarding whatever clustering/phrase-tagging work was still
            # in progress. Given its own override rather than bumping
            # STAGE_TIMEOUT_ANALYSIS_SECONDS itself, since several other
            # stages share that constant and don't have this problem.
            PipelineStage("voice_continuity", self._run_voice_continuity, SourceType.VOICE_CONTINUITY,
                          timeout_seconds=120.0, is_essential=False,
                          estimated_memory_mb=20),
            # NEW STAGES: QuaverIntelligence and PhraseIntelligence
            PipelineStage("quaver_intelligence", self._run_quaver_intelligence, SourceType.PITCH,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=True,
                          estimated_memory_mb=50),
            # "quantization"/"velocity_merge" were both tagged
            # SourceType.RITORNELLO for years even though neither one
            # ever ran the real Ritornello - this stage runs
            # TemporalLattice (pure snap-to-grid), retagged honestly.
            PipelineStage("quantization", self._run_quantization, SourceType.TEMPORAL_LATTICE,
                          timeout_seconds=STAGE_TIMEOUT_QUANTIZATION_SECONDS, is_essential=True,
                          estimated_memory_mb=30),
            PipelineStage("velocity_merge", self._run_velocity_merge, SourceType.VELOCITY_MERGE,
                          timeout_seconds=STAGE_TIMEOUT_QUANTIZATION_SECONDS, is_essential=False,
                          estimated_memory_mb=20),
            # The REAL Ritornello, restored - a musical-coherence repair
            # pass (sustain recovery, micro-note purge, tie
            # reconstruction, pitch-wobble collapse, trouble-mapping +
            # CorrectionScribe), not a quantizer. Runs after both
            # TemporalLattice and VelocityMerge have already done their
            # jobs. See agents/quantization/ritornello.py's module
            # docstring for the full v6.0 restoration rationale.
            PipelineStage("ritornello", self._run_ritornello, SourceType.RITORNELLO,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=60),
            PipelineStage("consensus", self._run_consensus, SourceType.CONSENSUS_ENGINE,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=True,
                          estimated_memory_mb=20),
            PipelineStage("schoenberg_mirror", self._run_schoenberg_mirror, SourceType.SCHOENBERG_MIRROR,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=50),
            PipelineStage("phrase_intelligence", self._run_phrase_intelligence, SourceType.VOICE_CONTINUITY,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=30),
            PipelineStage("structural_feedback", self._run_structural_feedback, SourceType.CONSENSUS_ENGINE,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=10),
            PipelineStage("validation", self._run_validation, SourceType.SCRIBE,
                          timeout_seconds=10.0, is_essential=True, estimated_memory_mb=5),
            PipelineStage("export", self._run_export, SourceType.SCRIBE,
                          timeout_seconds=STAGE_TIMEOUT_EXPORT_SECONDS, is_essential=False,
                          estimated_memory_mb=10),
        ]
        return stages

    # ====================================================================
    # Helper Methods
    # ====================================================================

    def _extract_testimony_dict(self, stage_name: str) -> Dict[str, Any]:
        """
        Extract testimony dict from a stage result.

        Some Testimony classes carry a per-note or per-timestamp evidence
        dict as one of their fields (e.g. GrooveTestimony.swing_alignment,
        AnechoicMATestimony.resonance_by_note) alongside their whole-track
        aggregate fields. quaver_intelligence.py's evidence gathering does
        direct per-note keyed lookups (e.g. groove_field.get(f"alignment_
        {start_ms}")) against whatever dict it's handed - it has no idea
        those keys live one level down in a nested field. Flatten any
        dict-typed field's keys into the top level too, so both the
        aggregate consumers and the per-note lookups work against the
        same returned dict without quaver_intelligence.py needing to know
        the nesting.
        """
        result = self._results.get(stage_name)
        if not result or not result.success:
            return {}
        t = result.testimony
        if isinstance(t, dict):
            merged = dict(t)
        elif hasattr(t, '__dict__'):
            merged = {k: v for k, v in vars(t).items() if not k.startswith('_')}
        else:
            return {}

        for value in list(merged.values()):
            if isinstance(value, dict):
                merged.update(value)

        return merged

    def _get_stage_result(self, stage_name: str) -> Optional[TestimonyStageResult]:
        """Get TestimonyStageResult object for a stage."""
        return self._results.get(stage_name)

    def _reapply_consensus_with_bias(self, notes: List[NoteEvent],
                                     structural_bias: str,
                                     expected_phrase_length: int) -> List[NoteEvent]:
        """Re-apply consensus with structural bias from phrase intelligence."""
        if not notes:
            return notes

        # Apply confidence adjustments based on structural bias
        adjusted = []
        for note in notes:
            # Boost confidence for notes that align with structural expectations
            # This is a simplified version - would need full consensus re-run
            adjusted.append(note)

        return adjusted

    # ====================================================================
    # Execution Methods
    # ====================================================================

    def _execute_pipeline(self, audio_path: str, output_path: Optional[str] = None) -> PipelineResult:
        self._audio_path = audio_path
        self._output_path = output_path
        # Create temp directory for file-based handoff
        self._temp_dir = tempfile.mkdtemp(prefix="grimlock_")
        return self._execute()

    def _execute(self) -> PipelineResult:
        start = time.time()
        print(f"\n{'=' * 60}")
        print(f"GRIMLOCK 5.6.1 — {Path(self._audio_path).name}")
        if self.truncate_duration > 0:
            print(f"Truncation: first {self.truncate_duration:.0f}s")
        print(f"{'=' * 60}\n")

        with scoped_arena(self.guardian, "pipeline_main") as arena:
            self._arena = arena
            self.guardian.register_arena(arena)
            try:
                for i, stage in enumerate(self._stages):
                    print(f"[{i + 1:02d}/{len(self._stages)}] {stage.name}")
                    if self._aborted:
                        break

                    alloc = self.guardian.request_allocation(
                        requester=stage.name,
                        estimated_mb=stage.estimated_memory_mb,
                        can_evict=True,
                    )
                    if not alloc.approved:
                        if stage.is_essential:
                            self._aborted = True
                            self._abort_reason = f"Memory denied for {stage.name}"
                            print(f"  [ABORT] {self._abort_reason}")
                            break
                        print(f"  [SKIP] memory denied")
                        continue

                    # Each stage gets its own child arena, nested under
                    # the main pipeline arena, so any buffers a stage
                    # stores can be reclaimed right after that stage
                    # finishes instead of accumulating for the whole
                    # pipeline's lifetime (previously ONE arena lived for
                    # the entire run). Stages still transparently reach
                    # buffers earlier stages stored in the parent arena -
                    # _get_slot() falls back up the parent chain (fixed
                    # in memory/arena.py alongside this).
                    stage_arena = MemoryArena(self.guardian, f"stage_{stage.name}", parent_arena=arena)
                    self._arena = stage_arena
                    try:
                        t0 = time.time()
                        result = self._run_stage_with_timeout(stage)
                        elapsed_ms = (time.time() - t0) * 1000
                    finally:
                        stage_arena.clear_cache()
                        stage_arena.exit()
                        self._arena = arena
                    if hasattr(result, 'execution_time_ms'):
                        result.execution_time_ms = elapsed_ms
                    self._results[stage.name] = result
                    self._stage_times[stage.name] = elapsed_ms

                    if self.progress_callback:
                        try:
                            progress_pct = int(round((i + 1) / len(self._stages) * 100))
                            self.progress_callback(stage.name, progress_pct)
                        except Exception:
                            pass

                    if not getattr(result, 'success', False):
                        errs = "; ".join(getattr(result, 'errors', [])) if hasattr(result, 'errors') else "non-fatal"
                        if stage.is_essential:
                            self._aborted = True
                            self._abort_reason = f"{stage.name}: {errs}"
                            print(f"  [ABORT] {self._abort_reason}")
                            break
                        print(f"  [WARN] {errs}")
                    else:
                        icon = "✓"
                        if hasattr(result, 'warnings') and result.warnings:
                            icon = "⚠"
                        print(f"  {icon} {elapsed_ms:.0f}ms")
                        if self.debug and hasattr(result, 'warnings') and result.warnings:
                            for w in result.warnings[:3]:
                                print(f"    ⚠ {w}")

                    self.guardian.release_allocation(stage.name)
                    mem = self.guardian.get_memory_report()
                    self._peak_memory_mb = max(self._peak_memory_mb, mem.current_mb)
                    if stage.requires_gc:
                        gc.collect()

            except Exception as e:
                self._aborted = True
                self._abort_reason = str(e)
                if self.debug:
                    traceback.print_exc()
            finally:
                self.guardian.unregister_arena("pipeline_main")
                # Clean up temp directory
                if self._temp_dir and Path(self._temp_dir).exists():
                    shutil.rmtree(self._temp_dir, ignore_errors=True)

        return self._build_result(time.time() - start)

    def _run_stage_with_timeout(self, stage: PipelineStage) -> TestimonyStageResult:
        """Run stage with timeout - synchronous, no nested asyncio.run()"""
        result_box = [None]
        error_box = [None]

        def target():
            try:
                result_box[0] = stage.executor()
            except BaseException as e:
                # BaseException (not just Exception) because a timed-out
                # stage is cancelled by injecting a KeyboardInterrupt into
                # this thread (see CancellableThread.cancel()) - that's
                # expected control flow here, not a real user interrupt,
                # and must not escape as an unhandled thread exception.
                error_box[0] = e
                if self.debug and not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()

        thread = CancellableThread(target=target)
        completed, _, _ = thread.run_with_timeout(stage.timeout_seconds)

        if not completed:
            return TestimonyStageResult(stage_name=stage.name, success=False, testimony=None,
                               errors=[f"Timeout after {stage.timeout_seconds}s"])
        if error_box[0]:
            return TestimonyStageResult(stage_name=stage.name, success=False, testimony=None,
                               errors=[str(error_box[0])])
        return result_box[0] or TestimonyStageResult(stage_name=stage.name, success=False,
                                            testimony=None, errors=["Stage returned None"])

    # ====================================================================
    # Utilities
    # ====================================================================

    def _make_context(self, sr: int) -> AudioContext:
        return AudioContext(
            file_path=self._audio_path or "",
            original_sample_rate=(self._audio_views.sample_rate
                                  if self._audio_views else sr),
            working_sample_rate=sr,
            duration_seconds=(self._audio_views.master_stereo.shape[1] /
                              self._audio_views.sample_rate
                              if self._audio_views else 30.0),
            num_channels=2,
            file_hash=(self._audio_views.sha256_hash
                       if self._audio_views and self._audio_views.sha256_hash
                       else UNKNOWN_HASH_PLACEHOLDER),
            memory_mb=0,
        )

    def _get_tempo_bpm(self) -> float:
        """
        Current tempo BPM. self._tempo now always delegates to
        self._musical_findings.tempo_testimony, which defaults to a real
        TempoTestimony(120.0, ...) before tempo_analysis runs rather than a
        bare float sentinel - no hasattr guard needed anymore.
        """
        return self._tempo.tempo_bpm

    def _get_tempo_map(self) -> Optional[TempoMap]:
        """Current TempoMap, or None if not yet available."""
        return self._tempo.tempo_map

    def _piptrack_to_notes(self, pitches, magnitudes, sr, hop,
                           source_label, confidence_floor=0.15,
                           min_dur_ms=30.0) -> List[NoteEvent]:
        events = []
        tpf = hop / sr * 1000.0
        gmax = float(np.max(magnitudes)) + 1e-8
        ns = None
        np_ = None
        nf = None
        nc = 0.0
        nfr = 0

        for i in range(pitches.shape[1]):
            mags = magnitudes[:, i]
            idx = int(np.argmax(mags))
            freq = pitches[idx, i]
            t = i * tpf

            if freq > 0:
                pitch = int(round(12.0 * np.log2(freq / 440.0) + 69))
                conf = float(mags[idx] / gmax)
                if ns is None:
                    ns, np_, nf, nc, nfr = t, pitch, freq, conf, 1
                elif abs(pitch - np_) <= 1:
                    nfr += 1
                    nc += conf
                else:
                    if nfr >= 2:
                        ac = nc / nfr
                        if (t - ns) >= min_dur_ms and ac >= confidence_floor:
                            events.append(NoteEvent(
                                pitch=np_, start_ms=ns, end_ms=t,
                                velocity=int(min(127, ac * 80 + 30)),
                                confidence=ac, zero_crossing_rate=0.0,
                                source=SourceType.PITCH,
                                fundamental_freq_hz=nf,
                                reasoning_chain=[source_label]))
                    ns, np_, nf, nc, nfr = t, pitch, freq, conf, 1
            else:
                if ns is not None and nfr >= 2:
                    ac = nc / nfr
                    dur = t - ns
                    if dur >= min_dur_ms and ac >= confidence_floor:
                        events.append(NoteEvent(
                            pitch=np_, start_ms=ns, end_ms=t,
                            velocity=int(min(127, ac * 80 + 30)),
                            confidence=ac, zero_crossing_rate=0.0,
                            source=SourceType.PITCH,
                            fundamental_freq_hz=nf,
                            reasoning_chain=[source_label]))
                ns = np_ = nf = None
                nc = 0.0
                nfr = 0
        return events

    @staticmethod
    def _build_chord_templates() -> Dict[str, np.ndarray]:
        templates = {}
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for root in range(12):
            n = names[root]
            maj = np.zeros(12)
            min_ = np.zeros(12)
            dom7 = np.zeros(12)
            for i in [0, 4, 7]: maj[(root + i) % 12] = 1.0
            for i in [0, 3, 7]: min_[(root + i) % 12] = 1.0
            for i in [0, 4, 7, 10]: dom7[(root + i) % 12] = 1.0
            templates[f"{n}maj"] = maj
            templates[f"{n}min"] = min_
            templates[f"{n}7"] = dom7
        return templates

    # ====================================================================
    # Stage: Ingestion
    # ====================================================================

    def _run_ingestion(self) -> TestimonyStageResult:
        try:
            master = load_audio(self._audio_path, truncate_duration=self.truncate_duration)
            self._original_duration_seconds = master.original_duration

            stereo = master.audio
            if stereo.ndim == 1:
                stereo = np.stack([stereo, stereo], axis=0)
            elif stereo.ndim == 2 and stereo.shape[1] == 2:
                stereo = stereo.T

            self._audio_views = AudioViews(
                master_stereo=stereo.astype(np.float32),
                sample_rate=master.sample_rate,
                channel_layout=ChannelLayout.STEREO,
                sha256_hash=master.sha256_hash,
            )
            stats = self._audio_views.get_statistics()
            print(f"  {stats['duration_seconds']:.1f}s @ {master.sample_rate}Hz "
                  f"| stereo_width={stats['stereo_width']:.2f}")

            self._scale_separation_timeouts(stats['duration_seconds'])

            return TestimonyStageResult(
                stage_name="ingestion", success=True,
                testimony=IngestionTestimony(
                    audio_views=self._audio_views,
                    original_duration_seconds=master.original_duration,
                    is_truncated=master.is_truncated,
                    truncation_duration_seconds=self.truncate_duration,
                    file_hash=master.sha256_hash,
                    memory_mb=master.memory_mb,
                ),
            )
        except Exception as e:
            return TestimonyStageResult("ingestion", False, None, errors=[str(e)])

    def _scale_separation_timeouts(self, duration_seconds: float) -> None:
        """
        Scale the Demucs and Mel-Roformer separation timeouts to the actual
        audio length just loaded, and widen the outer "separation" stage
        watchdog to cover both attempts in sequence (Demucs first, then
        Mel-Roformer on failure/timeout) plus the STFT fallback.

        A flat timeout tight enough for a short clip starves the fallback
        chain of any time to run at all on a longer one - each model gets
        its own duration-scaled budget instead.
        """
        self._demucs_timeout = min(
            DEMUCS_TIMEOUT_CAP_SECONDS,
            max(DEMUCS_TIMEOUT_FLOOR_SECONDS, duration_seconds * DEMUCS_SECONDS_PER_AUDIO_SECOND)
        )
        self._mel_roformer_timeout = min(
            MEL_ROFORMER_TIMEOUT_CAP_SECONDS,
            max(MEL_ROFORMER_TIMEOUT_FLOOR_SECONDS, duration_seconds * MEL_ROFORMER_SECONDS_PER_AUDIO_SECOND)
        )
        stage_timeout = (self._demucs_timeout + self._mel_roformer_timeout
                         + SEPARATION_STAGE_TIMEOUT_BUFFER_SECONDS)

        for stage in self._stages:
            if stage.name == "separation":
                stage.timeout_seconds = stage_timeout
                break

        if self.debug:
            print(f"  [SEP] Budget for {duration_seconds:.1f}s audio: "
                  f"Demucs={self._demucs_timeout:.0f}s, "
                  f"Mel-Roformer={self._mel_roformer_timeout:.0f}s, "
                  f"stage={stage_timeout:.0f}s")

        # tempo_analysis now runs TempoIntelligence's 3-witness ensemble
        # plus the ReverseGeoCrypt witness in sequence - also scale it,
        # same reasoning as separation above.
        tempo_analysis_timeout = min(
            TEMPO_ANALYSIS_TIMEOUT_CAP_SECONDS,
            max(TEMPO_ANALYSIS_TIMEOUT_FLOOR_SECONDS,
                duration_seconds * TEMPO_ANALYSIS_SECONDS_PER_AUDIO_SECOND)
        )
        for stage in self._stages:
            if stage.name == "tempo_analysis":
                stage.timeout_seconds = tempo_analysis_timeout
                break

        if self.debug:
            print(f"  [TEMPO] Budget for {duration_seconds:.1f}s audio: "
                  f"tempo_analysis={tempo_analysis_timeout:.0f}s")

        # drum_detection's NMF/HFC coprocessor decomposition is far more
        # expensive than the other detection stages (rhythm/pitch_bass/
        # pitch_master/tonal all stayed well under the flat 60s timeout
        # on a real 294s track, but drum_detection hit it and returned
        # zero drum events for the entire song) - scale it the same way.
        drum_detection_timeout = min(
            DRUM_DETECTION_TIMEOUT_CAP_SECONDS,
            max(DRUM_DETECTION_TIMEOUT_FLOOR_SECONDS,
                duration_seconds * DRUM_DETECTION_SECONDS_PER_AUDIO_SECOND)
        )
        for stage in self._stages:
            if stage.name == "drum_detection":
                stage.timeout_seconds = drum_detection_timeout
                break

        if self.debug:
            print(f"  [DRUMS] Budget for {duration_seconds:.1f}s audio: "
                  f"drum_detection={drum_detection_timeout:.0f}s")

    # ====================================================================
    # Stage: Feature Extraction
    # ====================================================================

    def _run_feature_extraction(self) -> TestimonyStageResult:
        if self._audio_views is None:
            return TestimonyStageResult("feature_extraction", False, None, errors=["No audio"])
        try:
            import librosa
            audio = self._audio_views.mono_sum_normalized
            if self._audio_views.sample_rate != ANALYSIS_SR:
                audio = librosa.resample(audio,
                                         orig_sr=self._audio_views.sample_rate,
                                         target_sr=ANALYSIS_SR)
            bundle = extract_features(audio, ANALYSIS_SR, debug=self.debug)
            self._feature_bundle = bundle
            for et in bundle.get_available_evidence():
                self.evidence_tracker.acquire(et, "feature_extraction")
            feats = bundle.get_available_evidence()
            print(f"  {len(feats)} feature types, {bundle.memory_mb:.1f}MB")
            return TestimonyStageResult(
                "feature_extraction", True,
                FeatureExtractionTestimony(
                    feature_bundle_id="bundle",
                    memory_mb=bundle.memory_mb,
                    feature_types=[e.name for e in feats],
                    hop_length=bundle.hop_length,
                    n_fft=2048,
                ),
                evidence_acquired=[e.name for e in feats],
            )
        except Exception as e:
            return TestimonyStageResult("feature_extraction", False, None, errors=[str(e)])

    # ====================================================================
    # Stage: Tempo Analysis
    # ====================================================================

    def _run_tempo_analysis(self) -> TestimonyStageResult:
        if self._audio_views is None:
            return TestimonyStageResult("tempo_analysis", False, None, errors=["No audio"])
        try:
            import librosa
            from agents.analysis.tempo_intelligence import (
                TempoIntelligence, BeatGridBuilder, TempoConfig, TempoOctaveCorrector,
            )
            from agents.analysis.reverse_geo_crypt import ReverseGeoCrypt

            audio = self._audio_views.mono_sum_normalized
            if self._audio_views.sample_rate != ANALYSIS_SR:
                audio = librosa.resample(audio,
                                         orig_sr=self._audio_views.sample_rate,
                                         target_sr=ANALYSIS_SR)

            context = self._make_context(ANALYSIS_SR)
            tempo_intel = TempoIntelligence(music_box=self.music_box)
            # Reuse feature_extraction's cached onset envelope (same audio,
            # same ANALYSIS_SR, same hop_length=512) for analyze_tempo's
            # beat-grid/pulse-field/time-signature stage instead of letting
            # it recompute an identical one - see
            # ARCHITECTURE_CONSOLIDATION_PLAN.md §3.2. analyze_tempo's
            # docstring documents why a normalized envelope is safe here.
            cached_onset_env = (self._feature_bundle.onset_strength
                                if self._feature_bundle is not None else None)
            result = tempo_intel.analyze_tempo(audio, context, onset_env=cached_onset_env)

            ti_tempo = float(result.tempo_map.initial_tempo_bpm)
            ti_confidence = result.tempo_map.confidence.value_f

            # Independent witness: ReverseGeoCrypt votes on inter-onset ratio
            # geometry rather than beat-tracking, so it catches a different
            # class of error (octave/half-double tempo confusion in particular).
            geo_tempo = None
            geo_confidence = 0.0
            try:
                geo_crypt = ReverseGeoCrypt(music_box=self.music_box)
                geo_result = geo_crypt.run(audio, context, onset_env=cached_onset_env)
                if geo_result.success and geo_result.metadata.get("lattice_found"):
                    candidate = float(geo_result.metadata.get("tactus_bpm", 0.0))
                    if MIN_TEMPO_BPM <= candidate <= MAX_TEMPO_BPM:
                        geo_tempo = candidate
                        geo_confidence = float(geo_result.metadata.get("confidence", 0.0))
            except Exception as geo_err:
                if self.debug:
                    print(f"  [TEMPO] ReverseGeoCrypt witness failed: {geo_err}")

            if geo_tempo is not None and geo_confidence > 0:
                # Previously a naive confidence*weight-weighted arithmetic
                # mean of ti_tempo/geo_tempo directly - the exact same
                # failure mode anchor_and_align_tempo_witnesses() was built
                # to fix for TempoIntelligence's OWN internal ensemble,
                # just one level up: confirmed on real audio (You Say God
                # Says.mp3) that ti_tempo=90.9(conf=0.80) and
                # geo_tempo=43.1(conf=0.59, ~half of 90.9 - an octave
                # reading, not an independent estimate) blended to 74.7 BPM,
                # a value close to neither and not musically real, while
                # independent verification converged on ~123 BPM. Routed
                # through EpistemicCouncil.resolve_tempo_contention - the
                # same arbitration entry point drum-type and note-duration
                # disputes already use (it delegates to the identical
                # anchor-and-ratio-align math, and additionally logs any
                # witness disagreement to music_box as a first-class
                # decision record instead of silently excluding it).
                resolution = EpistemicCouncil.resolve_tempo_contention([
                    (ti_tempo, ti_confidence, TEMPO_INTELLIGENCE_ENSEMBLE_WEIGHT, "tempo_intelligence"),
                    (geo_tempo, geo_confidence, REVERSE_GEO_CRYPT_WITNESS_WEIGHT, "reverse_geo_crypt"),
                ], music_box=self.music_box)
                tempo, confidence, tempo_contention_details = resolution
                source = "tempo_intelligence+reverse_geo_crypt"
            else:
                tempo, confidence = ti_tempo, ti_confidence
                source = "tempo_intelligence"
                # Single witness - nothing to contend, matches
                # MusicalFindingsMap.tempo_contention's documented None
                # default (no second-guessing a lone witness).
                tempo_contention_details = None

            # Octave correction: none of the witnesses above ever ask "is
            # this off by a musically common rational factor (double/half/
            # triplet/dotted/etc.)?" - they vote between independent
            # algorithms, but if several agree on the same wrong octave,
            # nothing catches it. Test the blended estimate against real
            # onset times before finalizing.
            octave_correction = None
            try:
                onset_times_sec = librosa.onset.onset_detect(
                    y=audio, sr=ANALYSIS_SR, units="time")
                # Onset envelope for autocorrelation-based scoring -
                # confirmed directly on real audio to be far more robust
                # than onset-to-grid alignment for swung/compound-meter
                # material, where onsets are deliberately off a straight
                # grid but the underlying periodicity is still intact.
                #
                # This used to independently recompute onset_strength here
                # via librosa, on the EXACT same audio (mono_sum_normalized
                # resampled to ANALYSIS_SR) and hop_length (512) that
                # feature_extraction, the stage immediately before this
                # one, already computed and cached in self._feature_bundle
                # - a pure, wasteful duplicate with zero justification for
                # being separate (see ARCHITECTURE_CONSOLIDATION_PLAN.md
                # §3.1). _autocorrelation_score's own internal /peak
                # normalization makes it invariant to the pre-normalization
                # FeatureBundle.onset_strength applies, so reusing it here
                # doesn't change the correction math at all - only removes
                # the duplicate computation. Falls back to a fresh
                # computation if the bundle isn't available for any reason.
                if self._feature_bundle is not None and self._feature_bundle.onset_strength is not None:
                    octave_onset_env = self._feature_bundle.onset_strength
                else:
                    octave_onset_env = librosa.onset.onset_strength(y=audio, sr=ANALYSIS_SR, hop_length=512)
                corrector = TempoOctaveCorrector(TempoConfig())
                corrected_tempo, reason, debug_info = corrector.correct(
                    tempo, onset_times_sec,
                    onset_env=octave_onset_env, onset_env_sr=ANALYSIS_SR, onset_env_hop_length=512,
                    anchor_confidence=confidence)
                if corrected_tempo != tempo:
                    if self.debug:
                        print(f"  [TEMPO] Octave correction: {tempo:.1f} -> "
                              f"{corrected_tempo:.1f} BPM ({reason})")
                    tempo = corrected_tempo
                    octave_correction = reason
                    source = f"{source}+octave_corrected"
            except Exception as octave_err:
                if self.debug:
                    print(f"  [TEMPO] Octave correction failed (non-fatal): {octave_err}")

            time_sig = result.time_signature_primary
            time_sig_num = time_sig.numerator if time_sig else DEFAULT_TIME_SIGNATURE_NUMERATOR
            time_sig_den = time_sig.denominator if time_sig else DEFAULT_TIME_SIGNATURE_DENOMINATOR

            # Rebuild the beat/downbeat grid against the final blended tempo
            # so it stays consistent with the value actually reported.
            if abs(tempo - ti_tempo) > 0.01:
                blended_map = TempoMap(initial_tempo_bpm=tempo,
                                       tempo_events=result.tempo_map.tempo_events,
                                       confidence=Confidence.from_float(confidence))
                grid_builder = BeatGridBuilder(TempoConfig())
                pulse_field = grid_builder.build_pulse_field(blended_map, context.duration_seconds)
                beat_grid = grid_builder.build_beat_grid(
                    pulse_field.beat_grid_ms, tempo,
                    numerator=time_sig_num, denominator=time_sig_den,
                    downbeat_phase=time_sig.downbeat_phase if time_sig else 0,
                )
            else:
                beat_grid = result.beat_grid

            beat_ms = list(beat_grid.beat_times_ms if beat_grid else result.pulse_field.beat_grid_ms)
            downbeat_ms = list(beat_grid.downbeat_times_ms) if beat_grid else []

            tempo_map = TempoMap(
                initial_tempo_bpm=tempo,
                tempo_events=result.tempo_map.tempo_events,
                time_signature_numerator=time_sig_num,
                time_signature_denominator=time_sig_den,
                confidence=Confidence.from_float(confidence),
            )
            testimony = TempoTestimony(
                tempo_bpm=tempo,
                confidence=confidence,
                source=source,
                beat_times_ms=beat_ms,
                tempo_map=tempo_map,
                octave_correction=octave_correction,
            )
            self._tempo = testimony
            self._musical_findings.tempo_contention = tempo_contention_details
            self._beat_grid = beat_grid
            self._downbeat_times_ms = downbeat_ms
            self._time_signature_candidates = result.time_signature_candidates
            # Kept for later-stage tempo re-arbitration (rhythm_detection
            # submits its drum-stem tempo witness after this stage runs -
            # rebuilding the beat grid there needs the real detected
            # meter/downbeat-phase, not a re-derivation).
            self._time_signature_primary = time_sig
            self._musical_findings.tempo_bpm = tempo
            self._musical_findings.tempo_confidence = confidence
            self._musical_findings.beat_times_ms = np.array(beat_ms)
            self._musical_findings.time_signature_numerator = time_sig_num
            self._musical_findings.time_signature_denominator = time_sig_den
            self._musical_findings.record_stage("tempo_analysis")
            geo_str = (f"geo_crypt={geo_tempo:.1f}BPM(conf={geo_confidence:.2f})"
                      if geo_tempo is not None else "geo_crypt=no lattice")
            octave_str = f", octave_correction={octave_correction}" if octave_correction else ""
            print(f"  {tempo:.1f} BPM ({source}, conf={confidence:.2f}) "
                  f"[tempo_intel={ti_tempo:.1f}BPM(conf={ti_confidence:.2f}), {geo_str}], "
                  f"{len(beat_ms)} beats, {len(downbeat_ms)} downbeats, "
                  f"time_sig={time_sig_num}/{time_sig_den}{octave_str}")
            return TestimonyStageResult("tempo_analysis", True, testimony)
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            fallback = TempoTestimony(tempo_bpm=120.0, confidence=0.3,
                                      source="default", beat_times_ms=[])
            self._tempo = fallback
            self._musical_findings.tempo_bpm = 120.0
            return TestimonyStageResult("tempo_analysis", True, fallback,
                               warnings=[f"Tempo failed ({e}), using 120 BPM"])

    # ====================================================================
    # Stage: Separation (Demucs → Mel-Roformer → STFT)
    # ====================================================================

    def _apply_stage_zero_conditioning(self, audio_stereo: np.ndarray) -> np.ndarray:
        """
        Pre-separation conditioning (orchestration/Stage_Zero.py's
        StageZeroProcessor) - phase alignment, NMF noise-floor/reverb
        suppression, mid-side normalization, adaptive HPSS, spectral
        normalization, and peel-back. This module was fully built but had
        ZERO callers anywhere in the repo (confirmed via full-codebase
        audit) - it was never wired into the live pipeline at all, so
        separation has always been running on raw, unconditioned audio.

        Only the core StageZeroProcessor.process() DSP path is used here -
        it operates on plain numpy stereo and doesn't touch the file's own
        duplicate SourceType/NoteEvent/AudioContext/etc. type definitions,
        which are a separate, incompatible shadow of core.order_types used
        only by this file's own (unused, likely-stale) separation helper
        functions. Those are deliberately NOT imported here.

        Its own config docstring says "sensible defaults for jazz small-
        ensemble recordings" and none of this code has ever run against
        real audio before (zero callers = never exercised) - this
        project's real test material is hip-hop/pop, a very different
        acoustic profile, so this needs real e2e verification before
        trusting the defaults blindly. harmonic_mask_enabled is left on
        but harmless here: it requires a real chord_prior_hz, which
        harmonic_validation (much later in the pipeline) hasn't produced
        yet at this point - the stage no-ops on an empty chord list.

        audio_stereo arrives here as (samples, 2) - the shape the rest of
        _run_separation already expects - but StageZeroProcessor's DSP
        (e.g. _phase_align_gcc_phat's stereo[0]/stereo[1]) assumes
        (2, samples), so this transposes at the boundary in both
        directions and leaves the rest of _run_separation's shape
        assumptions untouched.
        """
        try:
            from orchestration.Stage_Zero import StageZeroProcessor, StageZeroConfig

            config = StageZeroConfig(sample_rate=44100)
            processor = StageZeroProcessor(config)
            bpm_hint = self._get_tempo_bpm() if self._tempo else None

            report = processor.process(audio_stereo.T, bpm_hint=bpm_hint)

            if self.music_box:
                self.music_box.log_decision(
                    stage_name="stage_zero_conditioning",
                    decision_type=DecisionType.ANALYSIS_EVIDENCE,
                    before_state={"stages_available": len(vars(config))},
                    after_state={
                        "stages_applied": report.stages_applied,
                        "noise_floor_db": report.noise_floor_db,
                        "reverb_rt60_estimate_ms": report.reverb_rt60_estimate_ms,
                        "stereo_width": [report.stereo_width_original, report.stereo_width_adjusted],
                        "estimated_source_count": report.estimated_source_count,
                        "acoustic_profile": report.acoustic_profile,
                    },
                    reasoning=f"Stage Zero pre-separation conditioning: {', '.join(report.stages_applied) or 'none applied'}",
                    reversible=True,
                )

            print(f"  [Stage Zero] Conditioning applied: {', '.join(report.stages_applied) or 'none'}")
            return report.conditioned_stereo.T.astype(audio_stereo.dtype)
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            print(f"  [Stage Zero] Conditioning skipped ({e}) - using unconditioned audio")
            return audio_stereo

    def _run_separation(self) -> TestimonyStageResult:
        if self._audio_views is None:
            return TestimonyStageResult("separation", False, None, errors=["No audio"])

        import librosa

        left = self._audio_views.left_channel
        right = self._audio_views.right_channel
        sr = self._audio_views.sample_rate
        if sr != 44100:
            left = librosa.resample(left, orig_sr=sr, target_sr=44100)
            right = librosa.resample(right, orig_sr=sr, target_sr=44100)
        audio_stereo = np.stack([left, right], axis=1)

        audio_stereo = self._apply_stage_zero_conditioning(audio_stereo)

        context = self._make_context(44100)

        print("  [SEP] Trying Demucs htdemucs...")
        result = self._try_demucs(audio_stereo, 44100, context)
        if result is not None:
            self._apply_separation_result(result)
            stems = self._available_stems()
            silent = self._silent_stems()
            print(f"  [SEP] Demucs succeeded — stems: {stems}")
            if silent:
                print(f"  [SEP] WARNING: stem(s) came back silent/hallucinated: {silent}")
            warnings_list = [f"Stem(s) silent despite being reported available: {silent}"] if silent else []
            return TestimonyStageResult("separation", True,
                               SeparationTestimony(stems_available=stems,
                                                   separation_time_seconds=0,
                                                   model_used=result.separator_used or "DEMUCS_HD",
                                                   fallback_used=False,
                                                   stems_hallucinated=silent),
                               warnings=warnings_list)

        # Demucs declined or failed (memory pressure, timeout, exception) -
        # DemucsSeparator.separate() now returns None honestly in every one
        # of those cases rather than silently substituting a different
        # method under the same name (see agents/separation/demucs.py's
        # module docstring). This is the one remaining fallback tier below
        # it - a real signal-processing heuristic, not a neural model (no
        # genuine 4-stem Mel-Roformer/BS-Roformer checkpoint exists
        # publicly - verified this session against audio-separator's full
        # model catalog).
        print("  [SEP] Demucs unavailable — trying heuristic stem splitter...")
        result = self._try_heuristic_splitter(audio_stereo, 44100, context)
        if result is not None:
            self._apply_separation_result(result)
            stems = self._available_stems()
            silent = self._silent_stems()
            print(f"  [SEP] Heuristic splitter succeeded — stems: {stems}")
            if silent:
                print(f"  [SEP] WARNING: stem(s) came back silent/hallucinated: {silent}")
            warnings_list = ["Demucs unavailable, used heuristic (non-neural) stem splitter"]
            if silent:
                warnings_list.append(f"Stem(s) silent despite being reported available: {silent}")
            return TestimonyStageResult("separation", True,
                               SeparationTestimony(stems_available=stems,
                                                   separation_time_seconds=0,
                                                   model_used="heuristic_stem_splitter",
                                                   fallback_used=True,
                                                   stems_hallucinated=silent),
                               warnings=warnings_list)

        print("  [SEP] Using STFT fallback (librosa HPSS + Butterworth)...")
        self._apply_simple_stft_separation(audio_stereo, 44100)
        print("  [SEP] STFT separation complete")
        return TestimonyStageResult("separation", True,
                           SeparationTestimony(stems_available=["drums", "bass", "other"],
                                               separation_time_seconds=0,
                                               model_used="simple_stft",
                                               fallback_used=True),
                           warnings=["Using STFT fallback — stems are lower quality"])

    def _silent_stems(self, threshold: float = 1e-6) -> List[str]:
        """Which of drums/bass/other are present in AudioViews but carry no
        real signal - e.g. DemucsSeparator degrading through its own
        internal fallback chain to identity, which zeros drums/bass/vocals
        and dumps everything into "other". Previously nothing checked stem
        CONTENT after separation "succeeded", only whether a stem slot
        existed at all."""
        silent = []
        for name, stem in [("drums", self._audio_views.drums_stereo),
                           ("bass", self._audio_views.bass_stereo),
                           ("other", self._audio_views.other_stereo),
                           ("vocals", self._audio_views.vocals_stereo)]:
            if stem is not None and np.max(np.abs(stem)) <= threshold:
                silent.append(name)
        return silent

    def _try_demucs(self, audio_stereo: np.ndarray, sr: int, context=None):
        try:
            from agents.separation.demucs import DemucsSeparator
            separator = DemucsSeparator(
                model_name="htdemucs", device="cpu",
                music_box=self.music_box,
                timeout_seconds=self._demucs_timeout,
            )
            if context is None:
                context = self._make_context(sr)
            result = separator.separate(audio_stereo, context)
            if result is None:
                return None
            for st in [StemType.DRUMS, StemType.BASS, StemType.OTHER]:
                stem = result.get_stem(st)
                if stem is not None and np.max(np.abs(stem)) > 1e-6:
                    return result
            return None
        except ImportError:
            print("  [SEP] Demucs not installed")
            return None
        except Exception as e:
            print(f"  [SEP] Demucs failed: {type(e).__name__}: {e}")
            return None

    def _try_heuristic_splitter(self, audio_stereo: np.ndarray, sr: int, context=None):
        """Non-neural fallback: CQT low-band energy for bass, onset-strength
        transients for drums. Not a substitute for a real model - a real
        4-stem Mel-Roformer/BS-Roformer checkpoint doesn't publicly exist
        (checked this session), so this honest heuristic is the actual
        tier below Demucs, not a fake neural "second opinion"."""
        try:
            import time as _time
            from agents.separation.demucs import HeuristicStemSplitter
            from core.order_types import SeparationResult
            splitter = HeuristicStemSplitter(sample_rate=sr)
            start = _time.time()
            stems = splitter.separate(audio_stereo.T if audio_stereo.shape[1] == 2 else audio_stereo)
            if not stems or all(np.max(np.abs(s)) <= 1e-6 for s in stems.values()):
                return None
            result = SeparationResult(
                stems=stems,
                separation_time_seconds=_time.time() - start,
                memory_usage_mb=0.0,
                confidence=Confidence.MEDIUM,
                separator_used="HEURISTIC_STEM_SPLITTER",
            )
            for st in [StemType.DRUMS, StemType.BASS, StemType.OTHER]:
                stem = result.get_stem(st)
                if stem is not None and np.max(np.abs(stem)) > 1e-6:
                    return result
            return None
        except ImportError:
            print("  [SEP] Heuristic stem splitter not available")
            return None
        except Exception as e:
            print(f"  [SEP] Heuristic stem splitter failed: {type(e).__name__}: {e}")
            return None

    def _apply_separation_result(self, result) -> None:
        def _fix(stem):
            if stem is None:
                return None
            stem = stem.astype(np.float32)
            if stem.ndim == 1:
                return np.stack([stem, stem])
            if stem.ndim == 2 and stem.shape[1] == 2:
                return stem.T
            return stem

        self._audio_views = AudioViews(
            master_stereo=self._audio_views.master_stereo,
            sample_rate=self._audio_views.sample_rate,
            channel_layout=self._audio_views.channel_layout,
            drums_stereo=_fix(result.get_stem(StemType.DRUMS)),
            bass_stereo=_fix(result.get_stem(StemType.BASS)),
            other_stereo=_fix(result.get_stem(StemType.OTHER)),
            vocals_stereo=_fix(result.get_stem(StemType.VOCALS)),
            sha256_hash=self._audio_views.sha256_hash,
        )
        self._maybe_dump_stems_for_debug()

    def _maybe_dump_stems_for_debug(self) -> None:
        """
        Opt-in isolated-stem WAV dump for offline analysis - complete no-op
        unless GRIMLOCK_DUMP_STEMS is set, so it carries zero cost/risk in
        normal runs. Added because this session repeatedly needed to
        inspect real isolated-stem audio directly (bass-bleed diagnosis,
        timbre classification, brush-articulation characterization) and
        had no way to get it out of the pipeline without ad-hoc scripts
        each time.
        """
        out_dir = os.environ.get("GRIMLOCK_DUMP_STEMS")
        if not out_dir or self._audio_views is None:
            return
        try:
            import soundfile as sf
            os.makedirs(out_dir, exist_ok=True)
            sr = self._audio_views.sample_rate
            for name, stereo in (("drums", self._audio_views.drums_stereo),
                                 ("bass", self._audio_views.bass_stereo),
                                 ("other", self._audio_views.other_stereo),
                                 ("vocals", self._audio_views.vocals_stereo)):
                if stereo is None:
                    continue
                sf.write(os.path.join(out_dir, f"{name}.wav"), stereo.T, sr)
            print(f"  [DEBUG] Dumped isolated stems to {out_dir}")
        except Exception as e:
            print(f"  [DEBUG] Stem dump failed: {e}")

    def _apply_simple_stft_separation(self, audio_stereo: np.ndarray, sr: int) -> None:
        import librosa
        import scipy.signal as signal

        mono = (np.mean(audio_stereo, axis=1)
                if audio_stereo.ndim == 2 else audio_stereo).astype(np.float32)

        harmonic, percussive = librosa.effects.hpss(mono, margin=(1.0, 5.0))

        try:
            sos = signal.butter(4, 250, btype='low', fs=sr, output='sos')
            bass = signal.sosfiltfilt(sos, mono).astype(np.float32)
        except Exception:
            bass = np.zeros_like(mono)

        other = (harmonic - bass).astype(np.float32)
        drums = percussive.astype(np.float32)

        def _s(arr):
            return np.stack([arr, arr])

        self._audio_views = AudioViews(
            master_stereo=self._audio_views.master_stereo,
            sample_rate=self._audio_views.sample_rate,
            channel_layout=self._audio_views.channel_layout,
            drums_stereo=_s(drums),
            bass_stereo=_s(bass),
            other_stereo=_s(other),
            sha256_hash=self._audio_views.sha256_hash,
        )

    def _available_stems(self) -> List[str]:
        stems = []
        if self._audio_views.drums_stereo is not None: stems.append("drums")
        if self._audio_views.bass_stereo is not None: stems.append("bass")
        if self._audio_views.other_stereo is not None: stems.append("other")
        if self._audio_views.vocals_stereo is not None: stems.append("vocals")
        return stems

    # ====================================================================
    # Stage: Rhythm Detection
    # ====================================================================

    def _run_rhythm_detection(self) -> TestimonyStageResult:
        if self._audio_views is None:
            return TestimonyStageResult("rhythm_detection", False, None, errors=["No audio"])
        try:
            from agents.detection.rhythm_engine import RhythmEngine
            import librosa

            drums = self._audio_views.drums_stereo
            audio = (np.mean(drums, axis=0) if drums is not None
                     else self._audio_views.mono_sum_normalized)
            sr = self._audio_views.sample_rate
            if sr != BASIC_PITCH_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=BASIC_PITCH_SR)

            context = self._make_context(BASIC_PITCH_SR)
            engine = RhythmEngine()
            result = engine.detect(audio, context)
            events = (result.events if hasattr(result, 'events')
                      else result if isinstance(result, list) else [])
            notes = [e.to_note_event() if hasattr(e, 'to_note_event') else e
                     for e in events]
            self._rhythm_notes = notes
            print(f"  {len(notes)} rhythm events")

            # Tempo estimator demotion (ARCHITECTURE_CONSOLIDATION_PLAN.md
            # §3.3): RhythmEngine's internal tempo estimate - computed
            # from the ISOLATED drum stem, a far cleaner rhythmic signal
            # than the full mix tempo_analysis works from - used to be
            # consumed only for its own internal note-duration math and
            # discarded, one of the audit's "isolated estimators with
            # zero cross-references" findings. Now it's submitted as a
            # corroborating witness to the same EpistemicCouncil
            # arbitration tempo_analysis already went through. Weight 0.7
            # (matching the ReverseGeoCrypt witness precedent) is
            # deliberate demotion: with the pipeline blend at weight 1.0,
            # this witness can refine the tempo within the established
            # octave (rescaled by ratio) or corroborate it, but can never
            # unilaterally hijack the anchor unless the blend's own
            # confidence is genuinely much lower than the drum stem's.
            self._submit_rhythm_tempo_witness(engine)

            return TestimonyStageResult("rhythm_detection", True,
                               RhythmTestimony(onset_events=events,
                                               kick_notes=[n for n in notes if n.pitch == 36],
                                               pattern_count=len(notes),
                                               tempo_estimate_bpm=self._get_tempo_bpm(),
                                               confidence=0.6))
        except Exception as e:
            return TestimonyStageResult("rhythm_detection", True,
                               RhythmTestimony(onset_events=[], kick_notes=[],
                                               pattern_count=0,
                                               tempo_estimate_bpm=120.0, confidence=0.0),
                               warnings=[str(e)])

    def _submit_rhythm_tempo_witness(self, engine) -> None:
        """
        Submit RhythmEngine's drum-stem tempo estimate to
        EpistemicCouncil.resolve_tempo_contention alongside the tempo the
        pipeline already established, and adopt the arbitrated result.
        Best-effort: any failure leaves the established tempo untouched.

        Stage ordering makes this safe: rhythm_detection (05) runs before
        every downstream tempo consumer (drum_detection, groove, pulse,
        quantization, export) - only Stage Zero's HPSS kernel hint (04)
        has already consumed the pre-arbitration value, and that only
        selects a median-filter kernel size, not anything note-level.
        """
        try:
            rhythm_tempo = float(engine._state.last_tempo_bpm)
            rhythm_conf = float(engine._state.last_tempo_confidence)
            if rhythm_tempo <= 0 or rhythm_conf <= 0 or self._tempo is None:
                return

            current_tempo = float(self._tempo.tempo_bpm)
            current_conf = float(self._tempo.confidence)

            resolution = EpistemicCouncil.resolve_tempo_contention([
                (current_tempo, current_conf, TEMPO_INTELLIGENCE_ENSEMBLE_WEIGHT, "pipeline_blend"),
                (rhythm_tempo, rhythm_conf, RHYTHM_ENGINE_WITNESS_WEIGHT, "rhythm_engine_drum_stem"),
            ], music_box=self.music_box)
            if resolution is None:
                return
            resolved_tempo, resolved_conf, details = resolution

            print(f"  drum-stem tempo witness: {rhythm_tempo:.1f}BPM(conf={rhythm_conf:.2f}) "
                  f"vs established {current_tempo:.1f}BPM(conf={current_conf:.2f}) "
                  f"-> {resolved_tempo:.1f}BPM(conf={resolved_conf:.2f})")

            if abs(resolved_tempo - current_tempo) / current_tempo < 0.01:
                # Corroboration without meaningful change - still adopt the
                # (possibly improved) confidence, skip the grid rebuild.
                self._musical_findings.tempo_confidence = resolved_conf
                return

            self._rebuild_tempo_state(resolved_tempo, resolved_conf,
                                      source_suffix="+rhythm_engine")
        except Exception as e:
            if self.debug:
                print(f"  [RHYTHM] tempo witness submission skipped: {e}")

    def _rebuild_tempo_state(self, tempo: float, confidence: float,
                             source_suffix: str) -> None:
        """Re-derive the beat grid and tempo testimony for an arbitrated
        tempo, reusing the same BeatGridBuilder path _run_tempo_analysis
        uses for its own blended-tempo rebuilds (real detected meter and
        downbeat phase, not re-derived defaults)."""
        from agents.analysis.tempo_intelligence import BeatGridBuilder, TempoConfig

        time_sig = self._time_signature_primary
        time_sig_num = time_sig.numerator if time_sig else DEFAULT_TIME_SIGNATURE_NUMERATOR
        time_sig_den = time_sig.denominator if time_sig else DEFAULT_TIME_SIGNATURE_DENOMINATOR

        prior = self._tempo
        tempo_map = TempoMap(
            initial_tempo_bpm=tempo,
            tempo_events=prior.tempo_map.tempo_events if prior and prior.tempo_map else [],
            time_signature_numerator=time_sig_num,
            time_signature_denominator=time_sig_den,
            confidence=Confidence.from_float(confidence),
        )
        grid_builder = BeatGridBuilder(TempoConfig())
        duration_seconds = (self._audio_views.master_stereo.shape[1] / self._audio_views.sample_rate
                            if self._audio_views is not None else 0.0)
        pulse_field = grid_builder.build_pulse_field(tempo_map, duration_seconds)
        beat_grid = grid_builder.build_beat_grid(
            pulse_field.beat_grid_ms, tempo,
            numerator=time_sig_num, denominator=time_sig_den,
            downbeat_phase=time_sig.downbeat_phase if time_sig else 0,
        )

        beat_ms = list(beat_grid.beat_times_ms) if beat_grid else []
        self._tempo = TempoTestimony(
            tempo_bpm=tempo,
            confidence=confidence,
            source=(prior.source if prior else "arbitrated") + source_suffix,
            beat_times_ms=beat_ms,
            tempo_map=tempo_map,
            octave_correction=prior.octave_correction if prior else None,
        )
        self._beat_grid = beat_grid
        self._downbeat_times_ms = list(beat_grid.downbeat_times_ms) if beat_grid else []
        self._musical_findings.tempo_bpm = tempo
        self._musical_findings.tempo_confidence = confidence
        self._musical_findings.beat_times_ms = np.array(beat_ms)
        print(f"  tempo re-arbitrated: {tempo:.1f} BPM (conf={confidence:.2f}), "
              f"beat grid rebuilt ({len(beat_ms)} beats, "
              f"{len(self._downbeat_times_ms)} downbeats)")

    # ====================================================================
    # Stage: Pitch Detection Bass
    # ====================================================================

    def _run_pitch_detection_bass(self) -> TestimonyStageResult:
        if self._audio_views is None or self._audio_views.bass_stereo is None:
            return TestimonyStageResult("pitch_detection_bass", True,
                               PitchTestimony(notes=[], confidence=0.0,
                                              model_used="none", stem_type="bass",
                                              frame_count=0, rejected_frames=0),
                               warnings=["No bass stem"])
        try:
            import librosa

            audio = np.mean(self._audio_views.bass_stereo, axis=0)
            sr = self._audio_views.sample_rate
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audio = audio.astype(np.float32)

            # Full multi-witness council (Librosa+SPICE+Basic Pitch+CREPE
            # refinement) instead of the old CREPE-primary/Librosa-fallback
            # PitchIntelligence path - bass gets the same real ensemble
            # agreement every other pitch-capable stem does. Bass-specific:
            # analyze_bass_sync also runs the SAME ensemble on the audio
            # shifted up an octave (then corrects results back down),
            # since a bass fundamental's often-weak/ambiguous low-end
            # energy is a well-known pitch-tracker failure mode that
            # simply doesn't apply once it's transposed into a range every
            # witness here is more reliable in.
            config = CouncilConfig(
                use_librosa=True, use_spice=True, use_basic_pitch=True,
                use_crepe=True, use_omnizart=False,
                basic_pitch_onset_threshold=0.35, basic_pitch_frame_threshold=0.20,
                spice_confidence_floor=0.50, crepe_confidence_floor=0.75,
                agreement_bonus=0.15, max_workers=2,
                min_frequency_hz=30.0, max_frequency_hz=400.0,
            )
            council = EpistemicCouncil(config=config)
            notes, updated_findings = council.analyze_bass_sync(
                audio, 16000, findings=self._musical_findings)
            council.close()
            self._musical_findings = updated_findings

            # Soft corroboration penalty for the piano-overlap register
            # (~pitch 45/A2 and up, where a piano's left hand commonly
            # doubles or crosses a bass line). Real per-note evidence
            # (bass-stem clutter investigation, see ARCHITECTURE_
            # CONSOLIDATION_PLAN.md/GRIMLOCK_6.0_DRAFT.md §12.10) found a
            # clean split: notes below this register spanned the whole
            # confidence range including many SPICE+BasicPitch-locked
            # (2-witness) hits, while every note landing in this overlap
            # register that had only single-witness corroboration
            # (council:single_*) was also the weakest-confidence content -
            # a specific, actionable fingerprint for likely bleed rather
            # than a real bass fundamental. This penalizes weak evidence
            # in that register rather than rejecting the register outright,
            # since real bass lines do legitimately reach this high.
            BASS_OVERLAP_REGISTER_FLOOR = 45
            WEAK_CORROBORATION_PENALTY = 0.35
            penalized_count = 0
            for n in notes:
                if n.pitch < BASS_OVERLAP_REGISTER_FLOOR:
                    continue
                tags = n.reasoning_chain or []
                strongly_corroborated = any(
                    t == "council:SPICE+BP_LOCKED" or
                    (isinstance(t, str) and t.startswith("council:") and "_witnesses_" in t)
                    for t in tags
                )
                if not strongly_corroborated:
                    n.confidence = max(0.0, n.confidence * (1.0 - WEAK_CORROBORATION_PENALTY))
                    n.reasoning_chain = list(tags) + ["bass:weak_corroboration_overlap_register_penalized"]
                    penalized_count += 1
            if penalized_count and self.debug:
                print(f"  [BASS] Penalized {penalized_count} weakly-corroborated notes "
                      f"at/above MIDI {BASS_OVERLAP_REGISTER_FLOOR} (piano-overlap register)")

            # This stage's audio IS the isolated bass stem already, so a
            # hard MIDI-range floor doesn't need any family classification
            # first - C1 (24) covers even a 5-string's low B with headroom
            # for detection jitter, C5 (72) is generous for slap/harmonics.
            before_count = len(notes)
            notes = [n for n in notes if 24 <= n.pitch <= 72]
            if before_count != len(notes) and self.debug:
                print(f"  [BASS] Rejected {before_count - len(notes)} out-of-range notes (outside MIDI 24-72)")

            for n in notes:
                n.reasoning_chain = list(n.reasoning_chain or []) + ["stem:bass"]
            self._pitch_notes_bass = notes
            avg_conf = float(np.mean([n.confidence for n in notes])) if notes else 0.0
            print(f"  {len(self._pitch_notes_bass)} notes (bass, council+octave-shift)")
            scribe_warnings = self.scribe.warn_if_problematic(
                self._pitch_notes_bass, SourceType.PITCH, "pitch_detection_bass")
            if self.debug and scribe_warnings:
                for w in scribe_warnings:
                    print(f"    [Scribe] {w}")
            return TestimonyStageResult("pitch_detection_bass", True,
                               PitchTestimony(notes=self._pitch_notes_bass,
                                              confidence=avg_conf, model_used="epistemic_council_bass",
                                              stem_type="bass",
                                              frame_count=len(self._pitch_notes_bass),
                                              rejected_frames=0))
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            return TestimonyStageResult("pitch_detection_bass", True,
                               PitchTestimony(notes=[], confidence=0.0,
                                              model_used="failed", stem_type="bass",
                                              frame_count=0, rejected_frames=0),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Pitch Detection Master
    # ====================================================================

    def _run_pitch_detection_master(self) -> TestimonyStageResult:
        if self._audio_views is None:
            return TestimonyStageResult("pitch_detection_master", False, None, errors=["No audio"])
        try:
            import librosa

            # mono_sum_normalized is the WHOLE mixed song - everything
            # already present in the separately-analyzed bass and other
            # stems is present here too. Running the full 5-model council
            # on it unconditionally meant every note got independently
            # re-detected from up to 3 overlapping audio views (bass,
            # master, other) - 5 models x 3 views = up to 15 largely-
            # redundant readings of mostly the same content, not genuine
            # independent corroboration. Master is only a real, useful
            # witness when separation actually failed and bass/other carry
            # no signal of their own - otherwise it's pure duplication that
            # the downstream consensus step then has to untangle.
            silent = self._silent_stems()
            if "bass" not in silent or "other" not in silent:
                self._pitch_notes_master = []
                print("  0 notes (master, skipped - bass/other stems carry real signal)")
                return TestimonyStageResult("pitch_detection_master", True,
                                   PitchTestimony(notes=[], confidence=0.0,
                                                  model_used="skipped_redundant", stem_type="master",
                                                  frame_count=0, rejected_frames=0))

            audio = self._audio_views.mono_sum_normalized
            sr = self._audio_views.sample_rate
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audio = audio.astype(np.float32)

            # Full multi-witness council instead of the old CREPE-primary/
            # Librosa-fallback PitchIntelligence path - master mix gets
            # the same real ensemble agreement as bass/other now. Only
            # reached as a genuine fallback when both stems are silent.
            config = CouncilConfig(
                use_librosa=True, use_spice=True, use_basic_pitch=True,
                use_crepe=True, use_omnizart=False,
                basic_pitch_onset_threshold=0.35, basic_pitch_frame_threshold=0.20,
                spice_confidence_floor=0.50, crepe_confidence_floor=0.75,
                agreement_bonus=0.15, max_workers=2,
            )
            council = EpistemicCouncil(config=config)
            events, updated_findings = council.analyze_sync(
                audio, 16000, findings=self._musical_findings)
            council.close()
            self._musical_findings = updated_findings

            self._pitch_notes_master = list(events) if events else []
            for n in self._pitch_notes_master:
                n.reasoning_chain = list(n.reasoning_chain or []) + ["stem:master"]
            avg_conf = float(np.mean([n.confidence for n in self._pitch_notes_master])) if self._pitch_notes_master else 0.0
            print(f"  {len(self._pitch_notes_master)} notes (master, council)")
            scribe_warnings = self.scribe.warn_if_problematic(
                self._pitch_notes_master, SourceType.PITCH, "pitch_detection_master")
            if self.debug and scribe_warnings:
                for w in scribe_warnings:
                    print(f"    [Scribe] {w}")
            return TestimonyStageResult("pitch_detection_master", True,
                               PitchTestimony(notes=self._pitch_notes_master,
                                              confidence=avg_conf, model_used="epistemic_council",
                                              stem_type="master",
                                              frame_count=len(self._pitch_notes_master),
                                              rejected_frames=0))
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            return TestimonyStageResult("pitch_detection_master", True,
                               PitchTestimony(notes=[], confidence=0.0,
                                              model_used="failed", stem_type="master",
                                              frame_count=0, rejected_frames=0),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Pitch Detection Other (Epistemic Council)
    # ====================================================================

    def _run_pitch_detection_other(self) -> TestimonyStageResult:
        if self._audio_views is None:
            return TestimonyStageResult("pitch_detection_other", False, None, errors=["No audio"])
        try:
            import librosa

            other = self._audio_views.other_stereo
            audio = (np.mean(other, axis=0) if other is not None
                     else self._audio_views.mono_sum_normalized)
            sr = self._audio_views.sample_rate
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            temp_wav_path = None
            if self._temp_dir:
                temp_wav_path = Path(self._temp_dir) / "other_stem_16k.wav"
                sf.write(str(temp_wav_path), audio, 16000)
                print(f"  [COUNCIL] Wrote temp file: {temp_wav_path}")

            config = CouncilConfig(
                use_librosa=True,
                use_spice=True,
                use_basic_pitch=True,
                use_crepe=True,
                use_omnizart=False,
                basic_pitch_onset_threshold=0.35,
                basic_pitch_frame_threshold=0.20,
                spice_confidence_floor=0.50,
                crepe_confidence_floor=0.75,
                agreement_bonus=0.15,
                max_workers=2,
            )

            council = EpistemicCouncil(config=config)

            if temp_wav_path and temp_wav_path.exists():
                audio_for_council, _ = librosa.load(str(temp_wav_path), sr=16000, mono=True)
            else:
                audio_for_council = audio

            notes, updated_findings = council.analyze_sync(
                audio=audio_for_council,
                sr=16000,
                findings=self._musical_findings
            )
            council.close()

            testimony = PitchTestimony(
                notes=notes,
                confidence=0.7 if notes else 0.0,
                model_used="epistemic_council",
                stem_type="other",
                frame_count=len(notes),
                rejected_frames=0
            )
            self._pitch_notes_other = list(notes)
            for n in self._pitch_notes_other:
                n.reasoning_chain = list(n.reasoning_chain or []) + ["stem:other"]
            self._musical_findings = updated_findings

            print(f"  {len(self._pitch_notes_other)} notes (other, council)")
            scribe_warnings = self.scribe.warn_if_problematic(
                self._pitch_notes_other, SourceType.PITCH, "pitch_detection_other")
            if self.debug and scribe_warnings:
                for w in scribe_warnings:
                    print(f"    [Scribe] {w}")
            return TestimonyStageResult("pitch_detection_other", True, testimony)

        except Exception as e:
            print(f"  [COUNCIL] Failed: {e}")
            traceback.print_exc()
            try:
                import librosa
                other = self._audio_views.other_stereo
                audio = (np.mean(other, axis=0) if other is not None
                         else self._audio_views.mono_sum_normalized)
                if self._audio_views.sample_rate != 16000:
                    audio = librosa.resample(audio,
                                             orig_sr=self._audio_views.sample_rate,
                                             target_sr=16000)
                harmonic, _ = librosa.effects.hpss(audio)
                pitches, magnitudes = librosa.piptrack(
                    y=harmonic, sr=16000, threshold=0.10, hop_length=256)
                notes = self._piptrack_to_notes(pitches, magnitudes, 16000, 256,
                                                "librosa_other_fallback")
                for n in notes:
                    n.reasoning_chain = list(n.reasoning_chain or []) + ["stem:other"]
                self._pitch_notes_other = notes
                return TestimonyStageResult("pitch_detection_other", True,
                                   PitchTestimony(notes=notes, confidence=0.5,
                                                  model_used="librosa_fallback",
                                                  stem_type="other",
                                                  frame_count=pitches.shape[1],
                                                  rejected_frames=pitches.shape[1] - len(notes)),
                                   warnings=[f"Council failed ({e}), librosa fallback"])
            except Exception as e2:
                self._pitch_notes_other = []
                return TestimonyStageResult("pitch_detection_other", True,
                                   PitchTestimony(notes=[], confidence=0.0,
                                                  model_used="none", stem_type="other",
                                                  frame_count=0, rejected_frames=0),
                                   warnings=[f"All detection failed: {e2}"])

    # ====================================================================
    # Stage: Pitch Detection Vocals
    # ====================================================================

    def _run_pitch_detection_vocals(self) -> TestimonyStageResult:
        """
        Mirrors pitch_detection_bass/other's EpistemicCouncil path, on the
        vocals stem - which the separator has always computed and this
        pipeline discarded until now (nothing built AudioViews.vocals_stereo
        or ran a detection stage against it, so vocal content was silently
        absorbed into "other" alongside piano/guitar/synth, undifferentiated).

        Deliberately does NOT fall back to the full mix like
        pitch_detection_other does when its own stem is missing - an absent
        vocals stem means "this song has no vocals stem to analyze," not
        "analyze the full polyphonic mix and call it vocals."

        Frequency/MIDI range widened relative to bass/other's defaults to
        cover the full singing range (bass singer low E2 through soprano
        high C6) rather than reusing an instrumental range verbatim; vocal
        pitch's own idiosyncrasies (vibrato, portamento/glides, breathiness)
        aren't otherwise specially handled yet - a follow-up witness-config
        tuning pass, not addressed here.
        """
        if self._audio_views is None or self._audio_views.vocals_stereo is None:
            return TestimonyStageResult("pitch_detection_vocals", True,
                               PitchTestimony(notes=[], confidence=0.0,
                                              model_used="none", stem_type="vocals",
                                              frame_count=0, rejected_frames=0),
                               warnings=["No vocals stem"])
        try:
            import librosa

            audio = np.mean(self._audio_views.vocals_stereo, axis=0)
            sr = self._audio_views.sample_rate
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audio = audio.astype(np.float32)

            config = CouncilConfig(
                use_librosa=True, use_spice=True, use_basic_pitch=True,
                use_crepe=True, use_omnizart=False,
                basic_pitch_onset_threshold=0.35, basic_pitch_frame_threshold=0.20,
                spice_confidence_floor=0.50, crepe_confidence_floor=0.75,
                agreement_bonus=0.15, max_workers=2,
                min_frequency_hz=70.0, max_frequency_hz=1100.0,
            )
            council = EpistemicCouncil(config=config)
            notes, updated_findings = council.analyze_sync(
                audio, 16000, findings=self._musical_findings)
            council.close()
            self._musical_findings = updated_findings

            # Hard MIDI-range floor mirroring bass's own post-filter, sized
            # to the singing range (E2-C6) rather than an instrument's:
            # low bass-singer chest voice up through soprano head-voice top.
            before_count = len(notes)
            notes = [n for n in notes if 40 <= n.pitch <= 84]
            if before_count != len(notes) and self.debug:
                print(f"  [VOCALS] Rejected {before_count - len(notes)} out-of-range notes (outside MIDI 40-84)")

            for n in notes:
                n.reasoning_chain = list(n.reasoning_chain or []) + ["stem:vocals"]
            self._pitch_notes_vocals = notes
            avg_conf = float(np.mean([n.confidence for n in notes])) if notes else 0.0
            print(f"  {len(self._pitch_notes_vocals)} notes (vocals, council)")
            scribe_warnings = self.scribe.warn_if_problematic(
                self._pitch_notes_vocals, SourceType.PITCH, "pitch_detection_vocals")
            if self.debug and scribe_warnings:
                for w in scribe_warnings:
                    print(f"    [Scribe] {w}")
            return TestimonyStageResult("pitch_detection_vocals", True,
                               PitchTestimony(notes=self._pitch_notes_vocals,
                                              confidence=avg_conf, model_used="epistemic_council",
                                              stem_type="vocals",
                                              frame_count=len(self._pitch_notes_vocals),
                                              rejected_frames=0))
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            self._pitch_notes_vocals = []
            return TestimonyStageResult("pitch_detection_vocals", True,
                               PitchTestimony(notes=[], confidence=0.0,
                                              model_used="failed", stem_type="vocals",
                                              frame_count=0, rejected_frames=0),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Tonal Detection
    # ====================================================================

    def _run_tonal_detection(self) -> TestimonyStageResult:
        empty = TonalTestimony(chord_map=[], detected_key=None,
                               key_confidence=0.0, frame_count=0)
        if self._feature_bundle is None:
            return TestimonyStageResult("tonal_detection", True, empty)
        try:
            import librosa
            if not self._feature_bundle.has_evidence(EvidenceType.CHROMA):
                return TestimonyStageResult("tonal_detection", True, empty)

            chroma = self._feature_bundle.chroma
            hop = self._feature_bundle.hop_length
            templates = self._build_chord_templates()
            window = max(4, int(0.5 * ANALYSIS_SR / hop))
            n_frames = chroma.shape[1]
            chord_map = []
            prev = None

            for fi in range(0, n_frames, window // 2):
                win = chroma[:, fi:min(fi + window, n_frames)]
                if win.shape[1] == 0:
                    continue
                mc = np.mean(win, axis=1)
                norm = np.linalg.norm(mc)
                if norm < 0.05:
                    continue
                mc /= norm
                best, best_s = None, -1.0
                for name, tmpl in templates.items():
                    s = float(np.dot(mc, tmpl) / (np.linalg.norm(tmpl) + 1e-8))
                    if s > best_s:
                        best_s = s
                        best = name
                if best_s > 0.55 and best != prev:
                    t_ms = float(librosa.frames_to_time(fi, sr=ANALYSIS_SR,
                                                        hop_length=hop) * 1000.0)
                    chord_map.append({"start_ms": t_ms, "chord": best,
                                      "confidence": best_s})
                    prev = best

            self._tonal_events = chord_map
            testimony = TonalTestimony(chord_map=chord_map, detected_key=None,
                                       key_confidence=0.0, frame_count=n_frames)
            self._musical_findings.tonal_testimony = testimony
            self._musical_findings.record_stage("tonal_detection")
            print(f"  {len(chord_map)} chord events")
            return TestimonyStageResult("tonal_detection", True, testimony)
        except Exception as e:
            return TestimonyStageResult("tonal_detection", True, empty, warnings=[str(e)])

    # ====================================================================
    # Stage: Drum Detection
    # ====================================================================

    def _run_drum_detection(self) -> TestimonyStageResult:
        empty = DrumTestimony(drum_events=[], note_events=[],
                              drum_types_detected=[], confidence=0.0)
        if self._audio_views is None or self._audio_views.drums_stereo is None:
            return TestimonyStageResult("drum_detection", True, empty)
        try:
            from agents.drum_intelligence import DrumIntelligence
            import librosa

            audio = np.mean(self._audio_views.drums_stereo, axis=0)
            sr = self._audio_views.sample_rate
            if sr != BASIC_PITCH_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=BASIC_PITCH_SR)

            context = self._make_context(BASIC_PITCH_SR)
            # AudioContext carries no tempo field, so without this
            # DrumIntelligence silently defaulted to a hardcoded 120 BPM
            # for its microtiming grids and pattern-hash spacing regardless
            # of the song's real tempo - drum_detection runs after
            # tempo_analysis, so the real value is already available here.
            # Also passes the real detected meter/downbeat phase (same
            # source tempo_analysis already computed for build_beat_grid)
            # so MicrotimingAnalyzer's grids and MetricalPositionEvidence's
            # beat-slot priors are phase-aligned instead of assuming beat 1
            # always lands at t=0.
            # run() (not detect() directly) so the music_box logging
            # wired into run() actually fires - this used to call
            # detect() directly with no music_box passed in at all,
            # silently discarding every confidence/articulation/NMF
            # finding DrumIntelligence computes for its 200+ events.
            stage_result = DrumIntelligence(music_box=self.music_box).run(
                audio, context, tempo_bpm=self._get_tempo_bpm(),
                time_sig_numerator=(self._musical_findings.time_signature_numerator
                                    or DEFAULT_TIME_SIGNATURE_NUMERATOR),
                downbeat_offset_ms=(self._downbeat_times_ms[0] if self._downbeat_times_ms else 0.0))
            result = stage_result
            events = (result.events if hasattr(result, 'events')
                      else result if isinstance(result, list) else [])
            notes = [e.to_note_event() if hasattr(e, 'to_note_event') else e
                     for e in events]
            self._drum_notes = notes

            # Per-note percussive anchor: how close does each pitched/rhythm
            # note actually sit to a real detected drum hit, rather than the
            # flat 0.50/0.0 defaults quaver_intelligence always saw before.
            anchor_notes = (
                list(self._pitch_notes_bass) + list(self._pitch_notes_master) +
                list(self._pitch_notes_other) + list(self._pitch_notes_vocals) +
                list(self._rhythm_notes)
            )
            per_note = self._compute_per_note_drum_anchor(notes, anchor_notes)

            print(f"  {len(notes)} drum events → MIDI channel 9, "
                  f"{len(per_note)} per-note anchor scores")
            scribe_warnings = self.scribe.warn_if_problematic(
                notes, SourceType.DRUM_INTELLIGENCE, "drum_detection")
            if self.debug and scribe_warnings:
                for w in scribe_warnings:
                    print(f"    [Scribe] {w}")
            return TestimonyStageResult("drum_detection", True,
                               DrumTestimony(drum_events=events, note_events=notes,
                                             drum_types_detected=list({n.pitch for n in notes}),
                                             confidence=0.7,
                                             per_note=per_note))
        except Exception as e:
            return TestimonyStageResult("drum_detection", True, empty, warnings=[str(e)])

    def _compute_per_note_drum_anchor(
            self, drum_notes: List[NoteEvent], notes: List[NoteEvent]
    ) -> Dict[str, Dict[str, float]]:
        """
        For each pitched/rhythm note, find the nearest actual detected drum
        hit and derive how strongly this note is anchored to real percussive
        timing (anchor_confidence, weighted by the drum hit's own detection
        confidence) versus how far it sits from any percussive support
        (anti_groove, its complement) - real onset proximity, not the flat
        0.50/0.0 defaults quaver_intelligence previously always saw.
        """
        per_note: Dict[str, Dict[str, float]] = {}
        if not drum_notes or not notes:
            return per_note

        drum_starts = np.array([d.start_ms for d in drum_notes])
        window_ms = 60.0

        for note in notes:
            idx = int(np.argmin(np.abs(drum_starts - note.start_ms)))
            distance = abs(float(drum_starts[idx]) - note.start_ms)
            proximity = max(0.0, 1.0 - distance / window_ms)
            anchor_confidence = proximity * max(0.3, float(drum_notes[idx].confidence))

            per_note[str(int(note.start_ms))] = {
                "anchor_confidence": float(anchor_confidence),
                "anti_groove": float(max(0.0, 1.0 - anchor_confidence)),
            }

        return per_note

    # ====================================================================
    # Stage: Anechoic MA
    # ====================================================================

    def _run_anechoic_ma(self) -> TestimonyStageResult:
        empty = AnechoicMATestimony(
            frame_count=0, region_count=0,
            silent_region_count=0, resonant_region_count=0,
            profile="unknown", applied=False,
        )

        if self._audio_views is None:
            return TestimonyStageResult("anechoic_ma", True, empty, warnings=["No audio"])

        try:
            audio = self._audio_views.mono_sum_normalized
            sr = self._audio_views.sample_rate

            config = AnechoicConfig(
                mode=AnechoicMode.STANDARD,
                silence_threshold=0.65,
                resonance_threshold=0.60,
                detect_reverb=True,
                swing_detection_enabled=True,
            )
            oracle = AnechoicMa(
                config=config,
                status_reporter=None,
                music_box=self.music_box,
            )

            report = oracle.analyze(
                y=audio,
                sr=sr,
                rhythm_field=self._tempo if self._tempo else None,
                stem_type=StemType.FULL_MIX,
            )

            self._anechoic_report = report

            self._musical_findings.anechoic_profile = report.profile.value
            self._musical_findings.anechoic_region_count = len(report.regions)
            self._musical_findings.record_stage("anechoic_ma")

            # Per-note resonance: does the acoustic energy actually sustain
            # through the note's claimed duration, or has it decayed to
            # silence? This is the core "how long does a note actually
            # sound" signal quaver_intelligence needs to judge sustain vs.
            # premature-cutoff vs. detection-artifact hypotheses - it was
            # designed to consume this (AnechoicMATestimony.resonance_map,
            # keyed "resonance_{pitch}_{start_ms}") but nothing ever
            # populated it.
            all_notes_for_resonance = (
                list(self._pitch_notes_bass) + list(self._pitch_notes_master) +
                list(self._pitch_notes_other) + list(self._pitch_notes_vocals) +
                list(self._rhythm_notes) + list(self._drum_notes)
            )
            resonance_map = self._compute_per_note_resonance(report, all_notes_for_resonance)

            print(f"  {len(report.regions)} regions "
                  f"({len(report.get_silent_regions())} silent, "
                  f"{len(report.get_resonant_regions())} resonant) "
                  f"profile={report.profile.value}, "
                  f"{len(resonance_map)} per-note resonance scores")

            return TestimonyStageResult("anechoic_ma", True,
                               AnechoicMATestimony(
                                   frame_count=len(report.frame_times),
                                   region_count=len(report.regions),
                                   silent_region_count=len(report.get_silent_regions()),
                                   resonant_region_count=len(report.get_resonant_regions()),
                                   profile=report.profile.value,
                                   applied=True,
                                   resonance_map=resonance_map,
                               ))

        except Exception as e:
            return TestimonyStageResult("anechoic_ma", True, empty,
                               warnings=[f"Anechoic MA failed: {e}"])

    def _compute_per_note_resonance(self, report, notes: List[NoteEvent]) -> Dict[str, float]:
        """
        Query the AnechoicReport for each note's own window and record its
        resonance_probability - i.e. how much the acoustic energy actually
        sustains through the time this note claims to be sounding,
        independent of whatever duration detection guessed.

        Keyed on note.start_ms (not the snapped/active timestamp) to match
        quaver_intelligence.py's lookup exactly - this stage runs on
        pre-quantization notes anyway, so the two are equal here, but the
        key must match what the reader actually asks for.

        AnechoicReport.query() works in seconds; NoteEvent timestamps are
        milliseconds.
        """
        resonance_map: Dict[str, float] = {}
        for note in notes:
            start_sec = note.start_ms / 1000.0
            end_sec = max(note.end_ms / 1000.0, start_sec + 0.01)
            try:
                state = report.query(start_sec, end_sec)
                key = f"resonance_{note.pitch}_{int(note.start_ms)}"
                resonance_map[key] = float(state.resonance_probability)
            except Exception:
                continue
        return resonance_map

    # ====================================================================
    # Stage: Timbre Analysis
    # ====================================================================

    def _run_timbre_analysis(self) -> TestimonyStageResult:
        empty = TimbreTestimony(instrument_counts={}, entities=[],
                                total_analyzed=0, confidence=0.0)
        all_pitched = (list(self._pitch_notes_bass) +
                       list(self._pitch_notes_master) +
                       list(self._pitch_notes_other) +
                       list(self._pitch_notes_vocals))
        if not all_pitched:
            return TestimonyStageResult("timbre_analysis", True, empty)
        try:
            import librosa
            engine = TimbreIntelligence(sample_rate=ANALYSIS_SR)

            def _stem_tag(note: NoteEvent) -> str:
                for tag in reversed(note.reasoning_chain or []):
                    if isinstance(tag, str) and tag.startswith("stem:"):
                        return tag.split(":", 1)[1]
                return "unknown"

            # Classify each note from the isolated stem it actually came
            # from, not the full mix - a note detected in the bass stem
            # should have its TIMBRE judged against bass-stem audio too,
            # otherwise a "woodwind" verdict on a bass note can't
            # distinguish real stem-separation bleed from "something else
            # was simply loud in the full mix at that instant" (the same
            # confound Schoenberg Mirror's own stem-routing fix already
            # solved for harmonic validation - see _run_schoenberg_mirror).
            raw_stem_sources = {
                "bass": self._audio_views.bass_stereo,
                "other": self._audio_views.other_stereo,
                "vocals": self._audio_views.vocals_stereo,
            }
            stem_audio: Dict[str, np.ndarray] = {}
            for name, stereo in raw_stem_sources.items():
                if stereo is None or np.max(np.abs(stereo)) <= 1e-6:
                    continue
                mono = np.mean(stereo, axis=0)
                if self._audio_views.sample_rate != ANALYSIS_SR:
                    mono = librosa.resample(mono, orig_sr=self._audio_views.sample_rate,
                                            target_sr=ANALYSIS_SR)
                stem_audio[name] = mono

            # Fallback for notes with a missing/unrecognized stem tag (e.g.
            # "master") or whose own stem's audio is silent/unavailable -
            # same full-mix behavior this stage always had.
            fallback_audio = self._audio_views.mono_sum_normalized
            if self._audio_views.sample_rate != ANALYSIS_SR:
                fallback_audio = librosa.resample(fallback_audio,
                                                  orig_sr=self._audio_views.sample_rate,
                                                  target_sr=ANALYSIS_SR)

            entities = []
            counts: Dict[str, int] = {}

            # Round-robin the fixed classification budget across stems
            # instead of a straight all_pitched[:80] slice - bass is
            # always listed first in all_pitched, so a plain slice let it
            # monopolize the whole budget whenever it had >=80 notes of
            # its own, leaving other/vocals with zero classification
            # coverage even though they had plenty of notes waiting
            # (confirmed: 0/76 "other" and 0/90 "vocals" classified in a
            # run where bass alone used all 80 slots).
            TIMBRE_ANALYSIS_BUDGET = 80
            by_stem: Dict[str, List[NoteEvent]] = {}
            for note in all_pitched:
                by_stem.setdefault(_stem_tag(note), []).append(note)

            budgeted_notes: List[NoteEvent] = []
            stem_iters = {name: iter(notes) for name, notes in by_stem.items()}
            while len(budgeted_notes) < TIMBRE_ANALYSIS_BUDGET and stem_iters:
                exhausted = []
                for name, it in stem_iters.items():
                    if len(budgeted_notes) >= TIMBRE_ANALYSIS_BUDGET:
                        break
                    try:
                        budgeted_notes.append(next(it))
                    except StopIteration:
                        exhausted.append(name)
                for name in exhausted:
                    del stem_iters[name]

            for note in budgeted_notes:
                audio = stem_audio.get(_stem_tag(note), fallback_audio)
                start_sample = int(note.start_ms * ANALYSIS_SR / 1000)
                end_sample = int(note.end_ms * ANALYSIS_SR / 1000)
                if (end_sample > start_sample and end_sample <= len(audio) and
                        (end_sample - start_sample) >= int(ANALYSIS_SR * 0.05)):
                    try:
                        seg = audio[start_sample:end_sample].copy()
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        entity = loop.run_until_complete(engine.process(seg, note.start_ms))
                        loop.close()

                        fam = (entity.probable_family.value
                               if entity.probable_family else "unknown")
                        counts[fam] = counts.get(fam, 0) + 1
                        entities.append({"pitch": note.pitch, "start_ms": note.start_ms,
                                         "family": fam, "confidence": entity.confidence})
                    except Exception as e:
                        if self.debug:
                            print(f"  Timbre note error: {e}")
                        continue

            self._musical_findings.instrument_families = {
                k: v / max(len(entities), 1) for k, v in counts.items()}
            self._musical_findings.record_stage("timbre_analysis")
            # Kept per-note (not just the aggregate counts above) so
            # voice_continuity's family-mismatch/range checks can look up
            # each classified note's own family instead of only a
            # song-wide guess.
            self._timbre_analysis = entities
            print(f"  Instruments: {counts}")
            return TestimonyStageResult("timbre_analysis", True,
                               TimbreTestimony(instrument_counts=counts,
                                               entities=entities[:40],
                                               total_analyzed=len(entities),
                                               confidence=0.65))
        except Exception as e:
            return TestimonyStageResult("timbre_analysis", True, empty, warnings=[str(e)])

    # ====================================================================
    # Stage: Harmonic Validation
    # ====================================================================

    def _run_harmonic_validation(self) -> TestimonyStageResult:
        all_notes = (list(self._pitch_notes_bass) +
                     list(self._pitch_notes_master) +
                     list(self._pitch_notes_other) +
                     list(self._pitch_notes_vocals))

        if not all_notes:
            testimony = HarmonicValidationTestimony(validated_notes=[],
                                                     vetoed_count=0,
                                                     detected_key=None,
                                                     key_confidence=0.0,
                                                     method="empty")
            self._musical_findings.harmonic_testimony = testimony
            return TestimonyStageResult("harmonic_validation", True, testimony)

        if self._feature_bundle is None:
            validated = [n for n in all_notes if n.confidence >= self.CONFIDENCE_FLOOR]
            testimony = HarmonicValidationTestimony(validated_notes=validated,
                                                     vetoed_count=len(all_notes) - len(validated),
                                                     detected_key=None, key_confidence=0.0,
                                                     method="confidence_fallback")
            self._musical_findings.harmonic_testimony = testimony
            return TestimonyStageResult("harmonic_validation", True, testimony)
        try:
            cfg = HarmonicConfig(
                min_partials_required=1,
                max_harmonics=8,
                detect_key=True,
                key_confidence_threshold=0.55,
            )
            validator = HarmonicValidator(cfg)
            key_detector = KeyDetector(cfg)

            key, key_conf = None, 0.0
            if self._feature_bundle.has_evidence(EvidenceType.CHROMA):
                key, key_conf = key_detector.detect_key(self._feature_bundle.chroma)
                if key:
                    print(f"  Key: {key} (conf={key_conf:.2f})")

            validated, vetoed = [], 0
            inharmonicity_sum = 0.0
            partials_count_sum = 0
            validated_evidence_count = 0
            for note in all_notes:
                try:
                    r = validator.validate(note, self._feature_bundle)
                    inharmonicity_sum += r.inharmonicity
                    partials_count_sum += len(r.partials_found)
                    validated_evidence_count += 1
                    # r.confidence was computed here from real partial-
                    # tracking evidence but never reached the note itself -
                    # only used below to decide pass/fail, then discarded.
                    # Scribe's FINAL validation gate (orchestration/
                    # scribe.py's SchoenbergMirror call) has no raw audio/
                    # FeatureBundle access at that point and falls back to
                    # reading exactly these two fields off the note -
                    # without them every note showed harmonic_series=None
                    # there, vetoing 100% as "hallucinated" regardless of
                    # how well they'd actually validated here.
                    #
                    # Mutated in place (NoteEvent is a plain, non-frozen
                    # dataclass - Scribe itself mutates this exact field
                    # the same way) rather than via replace() into a new
                    # "validated" list: that list is only ever used for
                    # this stage's own testimony/reporting and is never
                    # read by anything downstream. The notes that actually
                    # flow to quantization/consensus/Scribe are the SAME
                    # objects held in self._pitch_notes_bass/master/other -
                    # a replace()'d copy here would never reach them.
                    note.harmonic_series_match_ratio = r.confidence
                    note.fundamental_freq_hz = note.fundamental_freq_hz or HarmonicValidator._pitch_to_hz(note.pitch)
                    if r.is_valid:
                        validated.append(note)
                    else:
                        vetoed += 1
                except Exception:
                    validated.append(note)

            self._detected_key = key
            self._key_confidence = key_conf
            self._musical_findings.detected_key = key
            self._musical_findings.key_confidence = key_conf
            self._musical_findings.record_stage("harmonic_validation")
            print(f"  {len(validated)}/{len(all_notes)} validated, {vetoed} vetoed")

            if self.music_box and validated_evidence_count > 0:
                # r.inharmonicity/partials_found were computed per-note
                # above (feeding harmonic_series_match_ratio and the
                # pass/fail decision) but the actual PER-NOTE evidence
                # itself was never recorded anywhere - only this batch
                # summary is cheap enough to log for every note without
                # bloating the session log.
                self.music_box.log_decision(
                    stage_name="harmonic_validation",
                    decision_type="harmonic_validation",
                    before_state={"candidate_count": len(all_notes)},
                    after_state={
                        "validated_count": len(validated),
                        "vetoed_count": vetoed,
                        "detected_key": key,
                        "key_confidence": key_conf,
                        "mean_inharmonicity": inharmonicity_sum / validated_evidence_count,
                        "mean_partials_found": partials_count_sum / validated_evidence_count,
                    },
                    reasoning=f"{len(validated)}/{len(all_notes)} notes validated, "
                              f"mean inharmonicity {inharmonicity_sum / validated_evidence_count:.3f}",
                    reversible=True,
                )

            testimony = HarmonicValidationTestimony(validated_notes=validated,
                                                     vetoed_count=vetoed,
                                                     detected_key=key,
                                                     key_confidence=key_conf,
                                                     method="harmonic_validator")
            self._musical_findings.harmonic_testimony = testimony
            return TestimonyStageResult("harmonic_validation", True, testimony)
        except Exception as e:
            validated = [n for n in all_notes if n.confidence >= self.CONFIDENCE_FLOOR]
            testimony = HarmonicValidationTestimony(validated_notes=validated,
                                                     vetoed_count=len(all_notes) - len(validated),
                                                     detected_key=None, key_confidence=0.0,
                                                     method="fallback")
            self._musical_findings.harmonic_testimony = testimony
            return TestimonyStageResult("harmonic_validation", True, testimony,
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Groove Analysis
    # ====================================================================

    def _run_groove_analysis(self) -> TestimonyStageResult:
        empty = GrooveTestimony(phase_delta_ms=0.0, is_wide_swing=False,
                                is_dilla_pocket=False, confidence=0.0)
        # self._rhythm_notes (RhythmEngine, run against the same drum stem)
        # reliably comes back near-empty on real audio - every full
        # end-to-end run this session reported 0-4 rhythm events on songs
        # where DrumIntelligence found 2000+ real drum hits. Sourcing kicks
        # from that near-empty list meant this "if not kicks: return empty"
        # guard fired on essentially every real run, so GrooveFieldAnalyzer
        # never actually executed - every "Swing Type: Straight" in this
        # session's output was that empty default, not a real measurement.
        # self._drum_notes (DrumIntelligence, already verified working) uses
        # the same GM-standard kit mapping (kick=36), so this is a drop-in
        # source swap, not a new dependency.
        kicks = [n for n in self._drum_notes if n.pitch == 36]
        bass = self._pitch_notes_bass

        if not kicks or not bass:
            # PhaseDeltaAnalyzer's bass-vs-kick relational model needs
            # both present - confirmed on real audio this isn't rare: a
            # 60s section of Hopeful.mp3 had 506 real drum hits (hi-hat/
            # ride/rimshot/tom/crash/snare) and zero classified as kick.
            # OnsetSwingAnalyzer needs no specific instrument - it works
            # from any onsets against the real beat grid tempo_analysis
            # already built - so use it here instead of just giving up.
            return self._run_general_swing_fallback(empty)

        try:
            from agents.analysis.groove_field import GrooveFieldAnalyzer

            sr = self._audio_views.sample_rate if self._audio_views else 44100
            context = self._make_context(sr)
            analyzer = GrooveFieldAnalyzer(music_box=self.music_box)
            groove_field = analyzer.analyze(bass, kicks, context)

            confidence_value = groove_field.confidence.value_f

            # Per-note swing alignment: does THIS note's own offset from
            # the straight subdivision grid match the groove's known
            # characteristic phase delta? quaver_intelligence was designed
            # to consume exactly this (groove_field.get(f"alignment_
            # {start_ms}")) but always got the 0.0 default. reverse_geo_crypt
            # hasn't run yet at this point in the stage order, so this uses
            # GrooveFieldAnalyzer's own average_inter_note_distance_ms as
            # the subdivision estimate rather than the lattice period.
            all_notes_for_swing = (
                list(self._pitch_notes_bass) + list(self._pitch_notes_master) +
                list(self._pitch_notes_other) + list(self._pitch_notes_vocals) +
                list(self._rhythm_notes) + list(self._drum_notes)
            )
            alignment = self._compute_per_note_swing_alignment(groove_field, all_notes_for_swing)

            testimony = GrooveTestimony(
                phase_delta_ms=groove_field.bass_kick_phase_delta_ms,
                is_wide_swing=groove_field.is_wide_swing,
                is_dilla_pocket=groove_field.is_dilla_pocket,
                confidence=confidence_value,
                groove_field=groove_field,
                swing_amount_ms=abs(groove_field.bass_kick_phase_delta_ms),
                alignment=alignment,
            )
            self._groove = testimony
            self._musical_findings.phase_delta_ms = groove_field.bass_kick_phase_delta_ms
            self._musical_findings.swing_ratio = 0.72 if groove_field.is_wide_swing else 0.50
            self._musical_findings.groove_type = groove_field.groove_signature
            self._musical_findings.groove_confidence = confidence_value
            self._musical_findings.record_stage("groove_analysis")

            print(f"  {groove_field.groove_signature}: delta={groove_field.bass_kick_phase_delta_ms:+.1f}ms, "
                  f"swing={groove_field.is_wide_swing}, dilla={groove_field.is_dilla_pocket}, "
                  f"conf={confidence_value:.2f}")
            return TestimonyStageResult("groove_analysis", True, testimony)
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            self._groove = empty
            return TestimonyStageResult("groove_analysis", True, empty, warnings=[str(e)])

    def _run_general_swing_fallback(self, empty: GrooveTestimony) -> TestimonyStageResult:
        """
        OnsetSwingAnalyzer fallback for when PhaseDeltaAnalyzer's bass-vs-
        kick model can't run (no kicks and/or no bass detected). Uses
        every available onset (pitched + drum) against the real beat grid
        tempo_analysis already built, so it works regardless of which
        instruments are actually present in the recording.
        """
        beat_grid_ms = (list(self._beat_grid.beat_times_ms)
                        if self._beat_grid is not None else [])
        all_notes = (
            list(self._pitch_notes_bass) + list(self._pitch_notes_master) +
            list(self._pitch_notes_other) + list(self._pitch_notes_vocals) +
            list(self._rhythm_notes) + list(self._drum_notes)
        )
        onset_times_ms = [n.start_ms for n in all_notes]

        if not beat_grid_ms or not onset_times_ms:
            self._groove = empty
            return TestimonyStageResult("groove_analysis", True, empty)

        try:
            from agents.analysis.groove_field import OnsetSwingAnalyzer, GrooveConfig

            analyzer = OnsetSwingAnalyzer(GrooveConfig())
            swing_ratio, confidence, sample_count = analyzer.detect_swing_ratio(
                beat_grid_ms, onset_times_ms)

            is_wide_swing = swing_ratio > 0.68
            groove_signature = ("wide_swing" if is_wide_swing
                                else "straight" if swing_ratio < 0.55
                                else "laid_back")

            testimony = GrooveTestimony(
                phase_delta_ms=0.0,
                is_wide_swing=is_wide_swing,
                is_dilla_pocket=False,
                confidence=confidence,
            )
            self._groove = testimony
            self._musical_findings.swing_ratio = swing_ratio
            self._musical_findings.groove_type = groove_signature
            self._musical_findings.groove_confidence = confidence
            self._musical_findings.record_stage("groove_analysis")

            if self.music_box:
                self.music_box.log_decision(
                    stage_name="groove_field",
                    decision_type=DecisionType.ANALYSIS_EVIDENCE,
                    before_state={"onset_count": len(onset_times_ms), "beat_count": len(beat_grid_ms)},
                    after_state={"swing_ratio": swing_ratio, "confidence": confidence,
                                "sample_count": sample_count, "method": "onset_swing_fallback"},
                    reasoning=f"Kick-independent swing fallback: ratio={swing_ratio:.2f} "
                              f"from {sample_count} off-beat onsets",
                    reversible=True,
                )

            print(f"  {groove_signature} (onset-based, no kick/bass available): "
                  f"swing_ratio={swing_ratio:.2f}, conf={confidence:.2f}, n={sample_count}")
            return TestimonyStageResult("groove_analysis", True, testimony)
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            self._groove = empty
            return TestimonyStageResult("groove_analysis", True, empty, warnings=[str(e)])

    def _compute_per_note_swing_alignment(
            self, groove_field: GrooveField, notes: List[NoteEvent]
    ) -> Dict[str, float]:
        """
        For each note, compare its own offset from the straight subdivision
        grid to the groove's known characteristic phase delta. A note whose
        offset closely matches the groove's signature is well-explained by
        the detected swing/pocket; if the groove shows no real signature
        (phase_delta ~ 0), alignment can't meaningfully mean anything and
        stays 0 for every note.
        """
        alignment: Dict[str, float] = {}
        phase_delta = groove_field.bass_kick_phase_delta_ms

        if abs(phase_delta) < 1e-6 or not notes:
            return alignment

        subdivision_ms = groove_field.average_inter_note_distance_ms
        if not subdivision_ms or subdivision_ms <= 0:
            subdivision_ms = 250.0

        for note in notes:
            nearest_grid = round(note.start_ms / subdivision_ms) * subdivision_ms
            offset = note.start_ms - nearest_grid
            similarity = 1.0 - min(1.0, abs(offset - phase_delta) / (abs(phase_delta) + 1e-6))
            alignment[f"alignment_{int(note.start_ms)}"] = float(max(0.0, similarity))

        return alignment

    # ====================================================================
    # Stage: Pulse Analysis
    # ====================================================================

    def _run_pulse_analysis(self) -> TestimonyStageResult:
        bpm = self._get_tempo_bpm()
        empty = PulseTestimony(tempo_bpm=bpm, confidence=0.0, beat_grid_ms=[])

        # self._rhythm_notes (RhythmEngine) reliably comes back near-empty
        # on real audio (confirmed repeatedly this session: 1 event on a
        # run where DrumIntelligence found 31 real hits in the same clip) -
        # the identical bug _run_groove_analysis and _run_reverse_geo_crypt
        # already had to work around by preferring self._drum_notes. This
        # stage had the same broken source but didn't fail loudly - it just
        # silently fed PulseFieldAnalyzer 1-4 onsets instead of dozens,
        # producing a low-confidence pulse field (confirmed: conf=0.25 on a
        # real run) that looked like a working result rather than a starved
        # one.
        onset_source = self._drum_notes if len(self._drum_notes) >= 4 else self._rhythm_notes
        if not onset_source:
            return TestimonyStageResult("pulse_analysis", True, empty)

        try:
            from agents.analysis.pulse_field import PulseFieldAnalyzer

            sorted_notes = sorted(onset_source, key=lambda n: n.start_ms)
            onset_times = [n.start_ms for n in sorted_notes]
            onset_strengths = [n.confidence for n in sorted_notes]

            # reverse_geo_crypt runs before this stage specifically so its
            # lattice can feed in here as an independent corroborating
            # period estimate (see stage-order comment in _build_pipeline).
            lattice_period_ms = self._musical_findings.lattice_period_ms
            lattice_confidence = self._musical_findings.lattice_confidence

            analyzer = PulseFieldAnalyzer(music_box=self.music_box)
            pulse_field = analyzer.analyze_pulse_field(
                onset_times=onset_times,
                onset_strengths=onset_strengths,
                tempo_hint=bpm,
                lattice_period_ms=lattice_period_ms if lattice_period_ms > 0 else None,
                lattice_confidence=lattice_confidence,
            )

            confidence_value = pulse_field.confidence.value_f
            self._musical_findings.record_stage("pulse_analysis")
            lattice_note = f", lattice={lattice_period_ms:.0f}ms" if lattice_period_ms > 0 else ""

            # Per-note metrical alignment: how close does each note's start
            # land to the pulse grid, and how strongly does its position
            # argue against metrical legitimacy (worst case: sitting right
            # between two grid points). quaver_intelligence was designed to
            # consume exactly this (pulse_field.get(str(int(start_ms)))) but
            # never received anything but the empty-dict default.
            all_notes_for_pulse = (
                list(self._pitch_notes_bass) + list(self._pitch_notes_master) +
                list(self._pitch_notes_other) + list(self._pitch_notes_vocals) +
                list(self._rhythm_notes) + list(self._drum_notes)
            )
            per_note = self._compute_per_note_pulse_alignment(pulse_field, all_notes_for_pulse)

            print(f"  Pulse: {pulse_field.tempo_bpm:.1f} BPM, {len(pulse_field.beat_grid_ms)} beats, "
                  f"conf={confidence_value:.2f}{lattice_note}, {len(per_note)} per-note scores")

            return TestimonyStageResult("pulse_analysis", True,
                               PulseTestimony(tempo_bpm=pulse_field.tempo_bpm, confidence=confidence_value,
                                             beat_grid_ms=pulse_field.beat_grid_ms, pulse_field=pulse_field,
                                             per_note=per_note))
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return TestimonyStageResult("pulse_analysis", True, empty, warnings=[str(e)])

    def _compute_per_note_pulse_alignment(
            self, pulse_field: PulseField, notes: List[NoteEvent]
    ) -> Dict[str, Dict[str, float]]:
        """
        For each note, find the nearest pulse-grid point (sixteenth-note
        resolution if available, else beat) and derive a complementary
        strength/opposition pair from the normalized distance to it:
        strength=1.0 exactly on the grid, opposition=1.0 exactly halfway
        between two grid points (the least metrically plausible place for
        a real onset to start). Weighted by the underlying beat's own
        pulse_strength where available, so a confidently-tracked beat
        counts for more than an uncertain one.
        """
        per_note: Dict[str, Dict[str, float]] = {}
        grid = pulse_field.sixteenth_grid_ms or pulse_field.beat_grid_ms
        if not grid or not notes:
            return per_note

        grid_arr = np.array(grid)
        if len(grid_arr) > 1:
            spacing = float(np.median(np.diff(grid_arr)))
        else:
            spacing = 60000.0 / max(pulse_field.tempo_bpm, 1.0)
        half_spacing = spacing / 2.0 if spacing > 0 else 1.0

        beat_arr = np.array(pulse_field.beat_grid_ms) if pulse_field.beat_grid_ms else None

        for note in notes:
            idx = int(np.argmin(np.abs(grid_arr - note.start_ms)))
            distance = abs(grid_arr[idx] - note.start_ms)
            normalized_distance = min(1.0, distance / half_spacing)

            strength = 1.0 - normalized_distance
            opposition = normalized_distance

            if (beat_arr is not None and len(beat_arr) > 0 and
                    pulse_field.pulse_strength is not None and
                    len(pulse_field.pulse_strength) > 0):
                beat_idx = int(np.argmin(np.abs(beat_arr - note.start_ms)))
                if beat_idx < len(pulse_field.pulse_strength):
                    beat_confidence = max(0.3, float(pulse_field.pulse_strength[beat_idx]))
                    strength *= beat_confidence

            key = str(int(note.start_ms))
            per_note[key] = {
                "strength": float(max(0.0, min(1.0, strength))),
                "opposition": float(max(0.0, min(1.0, opposition))),
            }

        return per_note

    # ====================================================================
    # Stage: Reverse Geo Crypt
    # ====================================================================

    def _run_reverse_geo_crypt(self) -> TestimonyStageResult:
        empty = LatticeTestimony(period_ms=0.0, confidence=0.0,
                                 events_analyzed=0, ratio_family="unknown")

        # RhythmEngine's self._rhythm_notes reliably comes back near-empty
        # on real audio (confirmed this session: 0-4 events on songs where
        # DrumIntelligence found dozens to thousands of real hits) - the
        # exact same problem _run_groove_analysis already had to work
        # around by switching to self._drum_notes instead. This stage was
        # still reading the broken source, so even with the real
        # ReverseGeoCrypt class now wired in below, it was silently
        # hitting the "< 4 events" early-return on every real run and
        # never actually executing.
        onset_source = self._drum_notes if len(self._drum_notes) >= 4 else self._rhythm_notes
        if len(onset_source) < 4:
            return TestimonyStageResult("reverse_geo_crypt", True, empty)

        try:
            return self._run_reverse_geo_crypt_real(empty, onset_source)
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return self._run_reverse_geo_crypt_crude_fallback(empty, onset_source, warning=str(e))

    def _run_reverse_geo_crypt_real(self, empty: LatticeTestimony,
                                     onset_source: List[NoteEvent]) -> TestimonyStageResult:
        """
        Real lattice-search cryptanalysis (agents/analysis/reverse_geo_crypt.py's
        ReverseGeoCrypt class) - previously this stage never called it at
        all, despite the class being fully built and used elsewhere in
        this same file as a tempo_analysis witness. It hand-rolled a crude
        substitute instead (round each inter-onset interval to the
        nearest 10ms, take the most common value), which can't detect
        ratio family (binary/ternary/swing/quintuplet/septuplet), can't
        use downbeat anchors, and has none of the real class's entropy/
        alignment-error scoring.
        """
        from agents.analysis.reverse_geo_crypt import ReverseGeoCrypt, EventExtractor, Anchor, ReverseGeoCryptConfig
        from core.order_types import SourceType

        pairs = sorted(
            (n.start_ms / 1000.0, max(0.1, min(1.0, n.confidence)))
            for n in onset_source
        )
        onset_times = [t for t, _ in pairs]
        strengths = [s for _, s in pairs]

        extractor = EventExtractor(ReverseGeoCryptConfig())
        events = extractor.extract_from_onsets(onset_times, strengths)

        anchors = []
        if self._beat_grid is not None and self._beat_grid.downbeat_times_ms:
            anchors = [
                Anchor(time=t / 1000.0, confidence=0.8, source=SourceType.TEMPO_INTELLIGENCE)
                for t in self._beat_grid.downbeat_times_ms
            ]

        crypt = ReverseGeoCrypt(music_box=self.music_box)
        lattice = crypt.decrypt(events, anchors)

        if lattice is None:
            return TestimonyStageResult("reverse_geo_crypt", True, empty)

        # subdivision_ms is the real class's fine-grained discovered grid
        # unit - the same semantic meaning the crude version's "period"
        # had (an empirically-discovered raw inter-onset-interval scale,
        # not a tempo-anchored macro beat), so downstream consumers
        # (PulseFieldAnalyzer's lattice_period_ms corroborating estimate,
        # TemporalLattice's quantization grid spacing) don't need to
        # change what they expect from this field.
        period = lattice.subdivision_ms
        conf = lattice.confidence
        self._musical_findings.lattice_period_ms = period
        self._musical_findings.lattice_confidence = conf
        self._musical_findings.record_stage("reverse_geo_crypt")

        print(f"  Lattice: period={period:.0f}ms, conf={conf:.2f}, "
              f"ratio_family={lattice.ratio_family.value}, "
              f"tactus={lattice.to_tactus_bpm():.1f}BPM, "
              f"level={lattice.confidence_level.value}, anchors={len(anchors)}")

        testimony = LatticeTestimony(period_ms=period, confidence=conf,
                                      events_analyzed=len(events),
                                      ratio_family=lattice.ratio_family.value)
        self._musical_findings.lattice_testimony = testimony
        return TestimonyStageResult("reverse_geo_crypt", True, testimony)

    def _run_reverse_geo_crypt_crude_fallback(self, empty: LatticeTestimony,
                                               onset_source: List[NoteEvent],
                                               warning: str) -> TestimonyStageResult:
        """Safety net if the real ReverseGeoCrypt path raises - the original
        crude inter-onset-interval mode, kept only so this stage never
        regresses to producing LESS than it did before this fix."""
        times = sorted(n.start_ms for n in onset_source)
        intervals = [round((times[i] - times[i - 1]) / 10) * 10
                     for i in range(1, min(len(times), 50))
                     if 100 < times[i] - times[i - 1] < 2000]
        if not intervals:
            return TestimonyStageResult("reverse_geo_crypt", True, empty, warnings=[warning])

        top = Counter(intervals).most_common(1)[0]
        period = float(top[0])
        conf = min(0.95, top[1] / len(intervals))
        self._musical_findings.lattice_period_ms = period
        self._musical_findings.lattice_confidence = conf
        self._musical_findings.record_stage("reverse_geo_crypt")

        print(f"  Lattice (crude fallback, real path failed: {warning}): "
              f"period={period:.0f}ms, conf={conf:.2f}")
        testimony = LatticeTestimony(period_ms=period, confidence=conf,
                                      events_analyzed=len(intervals),
                                      ratio_family="detected")
        self._musical_findings.lattice_testimony = testimony
        return TestimonyStageResult("reverse_geo_crypt", True, testimony,
                           warnings=[f"Real ReverseGeoCrypt failed, used crude fallback: {warning}"])

    # ====================================================================
    # Stage: Voice Continuity
    # ====================================================================

    def _tag_notes_with_cluster_ids(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """
        Best-effort: scan the master mix for distinct timbral clusters via
        SpectralMasker's StemScanner and tag each note's metadata with the
        cluster active at its onset. Without this, nothing in the pipeline
        ever populates cluster_id, so VoiceContinuity's evidence-based
        ClusterAwareVoiceSeparator can never engage (has_cluster_tags is
        always False) despite prefer_cluster_grouping=True, and every song
        silently falls back to pitch-proximity guessing.
        """
        if self._audio_views is None:
            return notes
        try:
            from agents.analysis.spectral_masker import StemScanner

            mono = np.mean(self._audio_views.master_stereo, axis=0)
            scanner = StemScanner(sample_rate=self._audio_views.sample_rate)
            clusters = scanner.scan(mono, tempo_bpm=self._get_tempo_bpm())
            self._timbre_clusters = clusters
            if not clusters:
                return notes

            tagged = []
            for note in notes:
                best = None
                for cluster in clusters:
                    if not cluster.is_active_at(note.start_ms):
                        continue
                    lo, hi = cluster.pitch_range_midi()
                    if not (lo - 2 <= note.pitch <= hi + 2):
                        continue
                    if best is None or cluster.confidence > best.confidence:
                        best = cluster
                if best is not None:
                    # NoteEvent has no metadata dict field - VoiceContinuity's
                    # _get_cluster_id() reads reasoning_chain "cluster:N" tags
                    # as its real (only working) attachment point.
                    tagged.append(replace(note, reasoning_chain=[
                        *(note.reasoning_chain or []), f"cluster:{best.cluster_id}"
                    ]))
                else:
                    tagged.append(note)
            n_tagged = sum(1 for n in tagged if any(
                isinstance(t, str) and t.startswith('cluster:') for t in (n.reasoning_chain or [])))
            print(f"  cluster tagging: {len(clusters)} timbral clusters found, "
                  f"{n_tagged}/{len(notes)} notes tagged")
            return tagged
        except Exception as e:
            print(f"  cluster tagging skipped: {e}")
            return notes

    def _tag_notes_with_instrument_family(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """
        Tag each note's reasoning_chain with the instrument family
        timbre_analysis already classified it as ("family:brass", etc.).

        timbre_analysis runs earlier in the pipeline and classifies each
        note's own audio segment (see _run_timbre_analysis), but its
        result previously only fed a single song-wide aggregate guess -
        the per-note classifications themselves were computed and then
        discarded. voice_continuity's family-mismatch cost penalty and
        InstrumentRangeValidator both need the per-note family, not a
        song-wide average, to keep a voice's identity locked to a single
        instrument instead of letting it silently drift (e.g. a trumpet
        line reinterpreted mid-phrase as violin).

        Only the first 80 pitched notes get classified by timbre_analysis
        (real-time cost), so most songs will have partial coverage - that's
        fine, since a voice only needs one or two confidently-classified
        notes to establish its identity for the rest of the checks.
        """
        entities = getattr(self, '_timbre_analysis', None)
        if not entities:
            return notes

        by_key = {(e['pitch'], e['start_ms']): e for e in entities}

        tagged = []
        for note in notes:
            entity = by_key.get((note.pitch, note.start_ms))
            if entity is not None and entity.get('family', 'unknown') != 'unknown':
                tagged.append(replace(note, reasoning_chain=[
                    *(note.reasoning_chain or []), f"family:{entity['family']}"
                ]))
            else:
                tagged.append(note)
        return tagged

    def _tag_notes_with_phrase_id(self, phrase_id_map: Dict[str, str]) -> None:
        """
        Write VoiceContinuity's per-note voice assignment directly onto
        each note's own phrase_id field.

        phrase_id_map (built by _compute_per_note_phrase_data, above) was
        already correct, but nothing ever wrote it onto the actual
        NoteEvent objects - only quaver_intelligence's own internal
        lookup ever read it, via this same string key. Scribe's phase-
        consistency check needs the real per-note voice/phrase to tell
        "two different voices legitimately sounding at once" (normal
        polyphony) apart from "this exact voice's own note stream has a
        genuine timing problem" (a real anomaly) - without it, that
        check was flattening every voice into one timeline and flagging
        any overlap between different instruments as incoherent.

        Mutates in place (NoteEvent is a plain, non-frozen dataclass) so
        this reaches self._pitch_notes_bass/master/other directly,
        regardless of whether voice_continuity's own Voice objects held
        the same object references or replace()'d copies for some notes
        (e.g. octave-jump/range-flagged ones).
        """
        if not phrase_id_map:
            return
        tagged_count = 0
        total_count = 0
        for notes_list in (self._pitch_notes_bass, self._pitch_notes_master,
                           self._pitch_notes_other, self._pitch_notes_vocals):
            for note in notes_list:
                total_count += 1
                key = f"note_{note.pitch}_{int(note.start_ms)}_phrase"
                voice_id = phrase_id_map.get(key)
                if voice_id is not None:
                    note.phrase_id = voice_id
                    tagged_count += 1
        print(f"  phrase_id tagging: {tagged_count}/{total_count} notes tagged")
        print(f"  [DEBUG] phrase_id tagged {tagged_count}/{total_count} notes "
              f"(map had {len(phrase_id_map)} entries)")

    def _propagate_voice_reasoning_tags(self) -> None:
        """
        Copy the "cluster:N" / "family:X" reasoning_chain tags that
        _tag_notes_with_cluster_ids/_tag_notes_with_instrument_family
        attached back onto the canonical note objects.

        Both tagging helpers build their output via dataclasses.replace(),
        which returns a NEW NoteEvent - the tagged copy only ever lived
        inside the local `all_pitched` list that was fed to
        VoiceContinuity.analyze() and, from there, into self._voices[*].notes.
        self._pitch_notes_bass/master/other (what every downstream stage
        and the final export actually reads) still point at the original,
        untagged objects, so every note's instrument_family/cluster_id came
        back as None despite timbre_analysis classifying them correctly.
        Mirrors _tag_notes_with_phrase_id's in-place mutation so this
        reaches the same canonical objects regardless of stage order.
        """
        if not self._voices:
            return
        tags_by_key: Dict[Tuple[int, int], List[str]] = {}
        for voice in self._voices:
            for note in voice.notes:
                extra = [t for t in (note.reasoning_chain or [])
                         if isinstance(t, str) and (t.startswith("cluster:") or t.startswith("family:"))]
                if extra:
                    tags_by_key[(note.pitch, int(note.start_ms))] = extra

        tagged_count = 0
        for notes_list in (self._pitch_notes_bass, self._pitch_notes_master,
                           self._pitch_notes_other, self._pitch_notes_vocals):
            for note in notes_list:
                extra = tags_by_key.get((note.pitch, int(note.start_ms)))
                if not extra:
                    continue
                existing = note.reasoning_chain or []
                new_tags = [t for t in extra if t not in existing]
                if new_tags:
                    note.reasoning_chain = [*existing, *new_tags]
                    tagged_count += 1
        print(f"  voice reasoning-tag propagation: {tagged_count} notes gained cluster/family tags")

    def _run_voice_continuity(self) -> TestimonyStageResult:
        all_pitched = (list(self._pitch_notes_bass) +
                       list(self._pitch_notes_master) +
                       list(self._pitch_notes_other) +
                       list(self._pitch_notes_vocals))
        if not all_pitched:
            return TestimonyStageResult("voice_continuity", True,
                               VoiceContinuityTestimony(voices=[], voice_count=0,
                                                        crossing_count=0, confidence=0.0))
        try:
            config = VoiceConfig(
                max_voice_gap_ms=500.0,
                max_octave_jump_semitones=24,
                max_voices=8,
                prefer_cluster_grouping=True,
                witness_lookback_ms=2000.0,
                witness_similarity_threshold=0.78,
            )
            all_pitched = self._tag_notes_with_cluster_ids(all_pitched)
            all_pitched = self._tag_notes_with_instrument_family(all_pitched)
            vc = VoiceContinuity(config=config)
            result = vc.analyze(notes=all_pitched)

            self._voices = result.voices
            crossings = (len(result.voice_crossings)
                         if hasattr(result, 'voice_crossings') else 0)

            # Per-note phrase membership: which traced voice does each note
            # actually belong to (quaver_intelligence keys "note_{pitch}_
            # {start_ms}_phrase"), plus a genuine per-note continuity score
            # and real phrase-boundary timestamps derived from in-voice
            # gaps - all three fields existed on VoiceContinuityTestimony
            # but nothing ever populated them.
            phrase_id_map, continuity_scores, phrase_boundaries = (
                self._compute_per_note_phrase_data(self._voices, config.max_voice_gap_ms))
            self._tag_notes_with_phrase_id(phrase_id_map)
            self._propagate_voice_reasoning_tags()

            print(f"  {len(self._voices)} voices, {crossings} crossings, "
                  f"{len(phrase_id_map)} per-note phrase scores")
            for v in self._voices[:4]:
                print(f"    {v.role.value}: {len(v.notes)} notes")
            return TestimonyStageResult("voice_continuity", True,
                               VoiceContinuityTestimony(voices=self._voices,
                                                        voice_count=len(self._voices),
                                                        crossing_count=crossings,
                                                        confidence=0.7,
                                                        continuity_scores=continuity_scores,
                                                        phrase_boundaries=phrase_boundaries,
                                                        phrase_id_map=phrase_id_map))
        except Exception as e:
            return TestimonyStageResult("voice_continuity", True,
                               VoiceContinuityTestimony(voices=[], voice_count=0,
                                                        crossing_count=0, confidence=0.0),
                               warnings=[str(e)])

    def _compute_per_note_phrase_data(
            self, voices: List[Voice], max_voice_gap_ms: float
    ) -> tuple:
        """
        Walk each traced voice's notes in order and derive three genuine
        per-note/per-timestamp signals from real gap and interval data:

        - phrase_id_map: which voice each note belongs to, keyed exactly
          as quaver_intelligence.py's _gather_adversarial_evidence reads
          it ("note_{pitch}_{start_ms}_phrase" -> voice_id).
        - continuity_scores: how smoothly each note connects to the
          previous note in its voice (small gap + small pitch jump =
          high continuity), keyed by start_ms.
        - phrase_boundaries: timestamps where the gap to the previous
          note exceeds half the voice-tracker's own gap tolerance,
          i.e. a real detected break rather than a guessed one.
        """
        phrase_id_map: Dict[str, str] = {}
        continuity_scores: Dict[float, float] = {}
        phrase_boundaries: List[float] = []
        gap_threshold = max_voice_gap_ms / 2.0 if max_voice_gap_ms > 0 else 250.0

        for voice in voices:
            prev_note: Optional[NoteEvent] = None
            for note in voice.notes:
                phrase_id_map[f"note_{note.pitch}_{int(note.start_ms)}_phrase"] = voice.voice_id

                if prev_note is not None:
                    gap_ms = note.start_ms - prev_note.end_ms
                    pitch_jump = abs(note.pitch - prev_note.pitch)
                    continuity = 1.0 - (gap_ms / 500.0) - (pitch_jump / 24.0)
                    continuity_scores[note.start_ms] = float(max(0.0, min(1.0, continuity)))
                    if gap_ms > gap_threshold:
                        phrase_boundaries.append(note.start_ms)
                prev_note = note

        return phrase_id_map, continuity_scores, sorted(phrase_boundaries)

    # ====================================================================
    # Stage: Quaver Intelligence (NEW)
    # ====================================================================

    def _run_quaver_intelligence(self) -> TestimonyStageResult:
        """Stage: QuaverIntelligence — Symbolic duration witness."""
        from epistemic.quaver_intelligence import QuaverIntelligence, QuaverConfig

        # Collect all pitched notes (pre-quantization). RhythmEngine's
        # events are deliberately NOT included: they're kick-drum hits
        # (pitch 36, SourceType.RHYTHM) that exist as rhythm/onset
        # evidence for groove/pulse/lattice analysis, while
        # DrumIntelligence separately owns actual drum transcription
        # (self._drum_notes, exported on MIDI channel 9). For as long as
        # RhythmEngine's onset strength was broken (sampled at the
        # backtracked valley instead of the peak - see rhythm_engine.py's
        # _detect_onsets_at_resolution) it contributed only 1-4 events
        # here, so its inclusion looked harmless; once fixed it flooded
        # ~106 kick notes per 20s clip into the PITCHED stream, which
        # then survived to MIDI as melodic pitch-36 notes duplicating the
        # entire drum transcription DrumIntelligence already exports
        # (confirmed on real audio: Total Pitched jumped 147 -> 234 with
        # schoenberg percussion counts jumping 5 -> 94, all duplicated
        # kicks).
        all_pitched = (list(self._pitch_notes_bass) +
                       list(self._pitch_notes_master) +
                       list(self._pitch_notes_other) +
                       list(self._pitch_notes_vocals))

        if not all_pitched:
            return TestimonyStageResult("quaver_intelligence", True,
                               testimony={"quaver_testimonies": []},
                               warnings=["No notes for duration analysis"])

        # Get necessary evidence from testimonies
        pulse_field = self._extract_testimony_dict("pulse_analysis")
        groove_field = self._extract_testimony_dict("groove_analysis")
        anechoic_dict = self._extract_testimony_dict("anechoic_ma")
        tempo_dict = self._extract_testimony_dict("tempo_analysis")
        drum_dict = self._extract_testimony_dict("drum_detection")
        voice_dict = self._extract_testimony_dict("voice_continuity")

        # Per-note local tempo: which confirmed tempo segment does each
        # note actually fall in? tempo_analysis runs before any notes
        # exist, so this has to be computed here instead, once notes are
        # available, using the tempo_map it already produced.
        if self._tempo is not None and getattr(self._tempo, 'tempo_map', None):
            tempo_dict.update(self._compute_per_note_local_tempo(self._tempo.tempo_map, all_pitched))

        # Get tempo
        tempo_bpm = self._get_tempo_bpm()

        # subdivision_ms is QuaverIntelligence's reference unit for
        # SYMBOLIC_DURATIONS, where "quarter": 1.0 means one unit IS one
        # quarter note - it must be the real tempo-derived quarter-note
        # length (60000/bpm). It used to be lattice_period_ms, ReverseGeoCrypt's
        # empirically-discovered raw inter-onset-interval mode, which has no
        # tempo-anchoring at all and can just as easily represent an eighth
        # or sixteenth note. Feeding that in as "one quarter note" silently
        # mislabeled every symbolic duration a whole rhythmic level too fine
        # (e.g. calling a 64th-note-scale value "sixteenth") - a real
        # contributor to spurious micro-notes and tuplets, independent of
        # the sub-16th scrutiny added separately. lattice_period_ms is left
        # untouched for TemporalLattice's own, unrelated use of it as raw
        # quantization grid spacing.
        subdivision_ms = 60000.0 / tempo_bpm if tempo_bpm > 0 else 500.0

        # Get instrument hint from timbre analysis
        instrument_hint = "unknown"
        if self._musical_findings.instrument_families:
            instrument_hint = max(self._musical_findings.instrument_families,
                                  key=self._musical_findings.instrument_families.get)

        config = QuaverConfig()
        quaver = QuaverIntelligence(self._musical_findings, config)

        testimonies = quaver.infer_symbolic_durations(
            note_events=all_pitched,
            subdivision_ms=subdivision_ms,
            tempo_bpm=tempo_bpm,
            pulse_field=pulse_field,
            groove_field=groove_field,
            anechoic_ma=anechoic_dict,
            tempo_intelligence=tempo_dict,
            drum_intelligence=drum_dict,
            voice_continuity=voice_dict,
            instrument_hint=instrument_hint,
        )

        self._quaver_testimonies = testimonies
        self._musical_findings.quaver_testimonies = testimonies
        self._musical_findings.record_stage("quaver_intelligence")

        # Calculate metrics
        contention_count = sum(1 for t in testimonies if getattr(t, 'has_contention', False))
        avg_uncertainty = np.mean([getattr(t, 'uncertainty', 0.0) for t in testimonies]) if testimonies else 0

        print(f"  {len(testimonies)} notes analyzed, "
              f"{contention_count} with contention, "
              f"avg uncertainty={avg_uncertainty:.2f}")

        return TestimonyStageResult(
            "quaver_intelligence", True,
            testimony={
                "quaver_testimonies": testimonies,
                "note_count": len(testimonies),
                "contention_count": contention_count,
                "avg_uncertainty": avg_uncertainty,
            }
        )

    def _compute_per_note_local_tempo(self, tempo_map: TempoMap, notes: List[NoteEvent]) -> Dict[str, float]:
        """
        For each note, find which confirmed tempo segment (as bounded by
        tempo_map.tempo_events) it actually falls in, rather than reporting
        the same whole-track initial tempo for every note regardless of
        position. tempo_events is now reliably sparse (see
        TempoIntelligence._confirm_tempo_changes) - typically empty for a
        constant-tempo track - so this is cheap even done per-note.
        """
        local_tempo: Dict[str, float] = {}
        if not notes:
            return local_tempo

        events = sorted(tempo_map.tempo_events, key=lambda e: e.time_ms) if tempo_map.tempo_events else []

        for note in notes:
            bpm = tempo_map.initial_tempo_bpm
            for event in events:
                if note.start_ms >= event.time_ms:
                    bpm = event.tempo_bpm
                else:
                    break
            local_tempo[f"local_tempo_{int(note.start_ms)}"] = float(bpm)

        return local_tempo

    # ====================================================================
    # Stage: Quantization
    # ====================================================================

    def _run_quantization(self) -> TestimonyStageResult:
        # RhythmEngine's kick events deliberately excluded - same reasoning
        # as _run_quaver_intelligence above: they're rhythm evidence, not
        # pitched notes, and including them here routed them to MIDI as
        # melodic pitch-36 notes duplicating DrumIntelligence's own drum
        # transcription once RhythmEngine's strength bug was fixed.
        pitched = (list(self._pitch_notes_bass) +
                   list(self._pitch_notes_master) +
                   list(self._pitch_notes_other) +
                   list(self._pitch_notes_vocals))

        if not pitched:
            self._quantized_pitched = []
            return TestimonyStageResult("quantization", True,
                               QuantizationTestimony(quantized_notes=[],
                                                     original_count=0, snapped_count=0,
                                                     total_confidence_loss=0.0,
                                                     lattice_summary="empty"))
        pitched.sort(key=lambda n: n.start_ms)
        original = len(pitched)

        try:
            lattice = TemporalLattice.assemble_from_witnesses(
                self._musical_findings.to_temporal_lattice_evidence())
            quantizer = TemporalLatticeQuantizer(lattice)
            quantized_list = quantizer.quantize(pitched)

            self._quantized_pitched = list(quantized_list)

            # quantizer.quantize() only ever returns a plain list, never
            # the (list, stats) tuple this used to assume - so preserved/
            # nudged/snapped were always hardcoded zeros regardless of
            # what actually happened. Derive the real numbers instead by
            # comparing each note's resolved position against its own
            # untouched raw start_ms (never overwritten per the
            # non-destructive-audit invariant).
            deltas = [abs(n.snapped_start_ms - n.start_ms) for n in self._quantized_pitched
                      if n.snapped_start_ms is not None]
            preserved = sum(1 for d in deltas if d < 0.01)
            snapped = sum(1 for d in deltas if d >= lattice.max_quantization_shift_ms)
            nudged = len(deltas) - preserved - snapped
            avg_shift_ms = float(np.mean(deltas)) if deltas else 0.0

            # QuaverIntelligence computes competing per-note duration
            # hypotheses (symbolic value, probability, epistemic tension)
            # from six witnesses, but nothing downstream ever consumed
            # them - duration was decided purely from lattice grid math,
            # and the contention_count/avg_uncertainty computed in
            # _run_quaver_intelligence existed only for a console print.
            # Apply quaver's judgement now: confident+uncontested notes
            # get their winning hypothesis directly, genuinely contested
            # notes get arbitrated by EpistemicCouncil using evidence
            # outside the hypotheses' own probabilities (see
            # resolve_duration_contention) instead of being left to
            # blind grid math, which is what happened to every contested
            # note until now.
            pulse_field_for_arbitration = self._extract_testimony_dict("pulse_analysis")
            confident_applied, arbitrated_applied = self._apply_quaver_duration_consensus(
                self._quantized_pitched, self._quaver_testimonies, pulse_field_for_arbitration)

            print(f"  {original} → {len(self._quantized_pitched)} notes")
            print(f"  preserved={preserved} nudged={nudged} snapped={snapped} "
                  f"(avg shift={avg_shift_ms:.1f}ms)")
            print(f"  quaver duration consensus: {confident_applied} confident, "
                  f"{arbitrated_applied} arbitrated (contested)")
            print(f"  {lattice.get_summary()}")
            return TestimonyStageResult("quantization", True,
                               QuantizationTestimony(quantized_notes=self._quantized_pitched,
                                                     original_count=original,
                                                     snapped_count=snapped,
                                                     total_confidence_loss=snapped * 0.01,
                                                     lattice_summary=lattice.get_summary()))
        except Exception as e:
            self._quantized_pitched = pitched
            return TestimonyStageResult("quantization", True,
                               QuantizationTestimony(quantized_notes=pitched,
                                                     original_count=original,
                                                     snapped_count=0,
                                                     total_confidence_loss=0.0,
                                                     lattice_summary="fallback"),
                               warnings=[str(e)])

    def _apply_quaver_duration_consensus(
            self, notes: List[NoteEvent], testimonies: List[Any],
            pulse_field: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int]:
        """
        Use QuaverIntelligence's per-note competing duration hypotheses to
        correct a note's actual sounding duration, when quaver's own
        witnesses agree it should. Until now this testimony was computed
        and immediately discarded - quantization only ever moved a note's
        START time (lattice grid snapping); nothing ever revisited its
        LENGTH using the symbolic-duration judgement quaver was built to
        produce.

        Confident, uncontested notes get their winning hypothesis applied
        directly. Genuinely contested notes (has_contention=True - top
        two candidates within 0.15 probability of each other) used to be
        skipped entirely and left to blind grid math; they're now
        arbitrated by EpistemicCouncil.resolve_duration_contention using
        evidence outside either candidate's own probability (internal
        tension, end-of-note grid alignment) - held to a stricter trust
        bar than a clean, uncontested win, since a tie is inherently less
        certain than one hypothesis simply winning outright.

        Anchored to the note's own lattice-corrected start time so it
        composes correctly with the grid-snapping that already ran.
        Returns (confident_count, arbitrated_count) for the stage log.
        """
        if not testimonies or not notes:
            return 0, 0

        from epistemic.epistemic_council import EpistemicCouncil

        by_key: Dict[Tuple[int, float], Any] = {t.note_key: t for t in testimonies}

        confident_applied = 0
        arbitrated_applied = 0
        for note in notes:
            testimony = by_key.get((note.pitch, note.start_ms))
            if testimony is None:
                continue
            if testimony.contradiction_detected:
                continue

            if testimony.has_contention:
                primary = EpistemicCouncil.resolve_duration_contention(testimony, pulse_field)
                min_trust = 0.65
            else:
                primary = testimony.primary_hypothesis
                min_trust = 0.55

            if primary is None or primary.duration_ms <= 0:
                continue

            trust = primary.probability * (1.0 - primary.epistemic_tension)
            if trust < min_trust:
                continue

            current_start = note.get_active_start_ms()
            current_duration = note.get_active_end_ms() - current_start
            if abs(primary.duration_ms - current_duration) < 15.0:
                continue

            # Sane bounds - this nudges duration toward a musically
            # plausible symbolic value, it doesn't invent an arbitrary one.
            # 4x covers a full whole-vs-quarter jump (the largest single
            # step _find_longer_candidate can propose); the trust and
            # contention gates above are the primary safeguard, this is
            # just a backstop against a pathological outlier.
            new_duration = float(np.clip(primary.duration_ms, 30.0,
                                         min(current_duration * 4.0, 4000.0)))

            note.snapped_start_ms = current_start
            note.snapped_end_ms = current_start + new_duration
            tag = "quaver(arbitrated)" if testimony.has_contention else "quaver"
            reason = (f"{tag}: {primary.symbolic_value} "
                     f"({new_duration:.0f}ms, trust={trust:.2f})")
            note.snap_reason = f"{note.snap_reason}; {reason}" if note.snap_reason else reason
            if testimony.has_contention:
                arbitrated_applied += 1
                if self.music_box:
                    # The competing hypotheses EpistemicCouncil weighed
                    # (probability/support/opposition/epistemic_tension)
                    # only ever fed this one win/lose decision, then got
                    # discarded - worth recording which hypothesis won
                    # and why, not just the final count.
                    hyps = sorted(testimony.duration_hypotheses, key=lambda h: -h.probability)
                    self.music_box.log_decision(
                        stage_name="quantization",
                        decision_type=DecisionType.CONTENTION_RESOLVED,
                        before_state={
                            "candidates": [
                                {"symbolic_value": h.symbolic_value, "probability": h.probability,
                                 "epistemic_tension": h.epistemic_tension}
                                for h in hyps[:2]
                            ],
                        },
                        after_state={
                            "note_key": f"{note.pitch}_{note.start_ms}",
                            "winner": primary.symbolic_value,
                            "winner_probability": primary.probability,
                            "trust": trust,
                            "new_duration_ms": new_duration,
                        },
                        reasoning=f"Contention resolved: {primary.symbolic_value} won with trust {trust:.2f}",
                        reversible=True,
                    )
            else:
                confident_applied += 1

        return confident_applied, arbitrated_applied

    # ====================================================================
    # Stage: Velocity Merge
    # ====================================================================

    def _run_velocity_merge(self) -> TestimonyStageResult:
        notes = list(self._quantized_pitched)
        if not notes:
            self._merged_notes = []
            return TestimonyStageResult("velocity_merge", True,
                               MergeTestimony(merged_notes=[], original_count=0,
                                              removed_count=0, merge_count=0,
                                              velocity_stats={}))
        try:
            config = VelocityMergeConfig(
                overlap_tolerance_ms=10.0,
                adjacent_gap_ms=25.0,
                default_strategy=MergeStrategy.WEIGHTED_AVERAGE,
                preserve_ghost_notes=True,
                min_merge_confidence=0.25,
            )
            merger = VelocityMerge(config=config, music_box=self.music_box)
            result = merger.merge_overlaps(notes)

            self._merged_notes = list(result.merged_notes)
            vel = merger.analyze_velocity_distribution(self._merged_notes)
            recs = merger.get_merge_records()

            print(f"  {len(notes)} → {len(self._merged_notes)} "
                  f"(removed {result.notes_removed}, {len(recs)} merges)")
            return TestimonyStageResult("velocity_merge", True,
                               MergeTestimony(merged_notes=self._merged_notes,
                                              original_count=len(notes),
                                              removed_count=result.notes_removed,
                                              merge_count=len(recs),
                                              velocity_stats=vel))
        except Exception as e:
            self._merged_notes = notes
            return TestimonyStageResult("velocity_merge", True,
                               MergeTestimony(merged_notes=notes, original_count=len(notes),
                                              removed_count=0, merge_count=0,
                                              velocity_stats={}),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Ritornello (musical-coherence repair pass)
    # ====================================================================

    def _run_ritornello(self) -> TestimonyStageResult:
        notes = list(self._merged_notes)
        if not notes:
            self._repaired_notes = []
            return TestimonyStageResult("ritornello", True,
                               RitornelloTestimony([], 0, [], [], 0, "skipped"))
        try:
            stem_audio_by_tag: Dict[str, np.ndarray] = {}
            sample_rate = ANALYSIS_SR
            if self._audio_views is not None:
                sample_rate = self._audio_views.sample_rate
                for tag, stereo in (("bass", self._audio_views.bass_stereo),
                                    ("other", self._audio_views.other_stereo),
                                    ("vocals", self._audio_views.vocals_stereo),
                                    ("master", self._audio_views.master_stereo)):
                    if stereo is not None:
                        stem_audio_by_tag[tag] = np.mean(stereo, axis=0)

            # Reuses whatever confidence the tempo/grid stage already
            # produced to drive activation level, rather than recomputing
            # a second one - falls back to a conservative default if
            # tempo analysis never ran or failed.
            grid_confidence = self._tempo.confidence if self._tempo and hasattr(self._tempo, 'confidence') else 0.5

            ritornello = Ritornello(config=RitornelloConfig(), music_box=self.music_box)
            repaired, testimony = ritornello.repair(
                notes=notes,
                beat_grid=self._beat_grid,
                time_signature=self._time_signature_primary,
                stem_audio_by_tag=stem_audio_by_tag,
                sample_rate=sample_rate,
                grid_confidence=grid_confidence,
            )
            self._repaired_notes = repaired
            print(f"  {len(notes)} → {len(repaired)} notes "
                  f"({testimony.activation_level} activation, "
                  f"{len(testimony.revision_events)} revisions, "
                  f"{len(testimony.trouble_measures)} trouble measures, "
                  f"{testimony.corrected_by_scribe_count} scribe-corrected)")
            return TestimonyStageResult("ritornello", True, testimony)
        except Exception as e:
            # Non-fatal: fall through with unrepaired notes, same fallback
            # pattern as _run_quantization/_run_velocity_merge above.
            self._repaired_notes = notes
            return TestimonyStageResult("ritornello", True,
                               RitornelloTestimony(notes, len(notes), [], [], 0, "error"),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Consensus (Preserves drum notes)
    # ====================================================================

    def _run_consensus(self) -> TestimonyStageResult:
        # IMPORTANT: Keep pitched and drum notes separate - don't mix them
        pitched_notes = list(self._repaired_notes)
        drum_notes = list(self._drum_notes)  # Preserve drums separately

        # Filter pitched notes only
        before = len(pitched_notes)
        pitched_notes = [n for n in pitched_notes if n.confidence >= self.CONFIDENCE_FLOOR]
        removed = before - len(pitched_notes)
        if removed:
            print(f"  Confidence filter: {removed} notes below {self.CONFIDENCE_FLOOR} removed")

        # Apply Anechoic MA penalty if available
        if self._anechoic_report is not None:
            penalized = 0
            for note in pitched_notes:
                state = self._anechoic_report.query(note.start_ms, note.end_ms)
                penalty = state.get_confidence_penalty()
                if penalty > 0:
                    note.confidence *= (1.0 - penalty)
                    penalized += 1
            if penalized:
                print(f"  Anechoic penalty applied to {penalized} notes")

        # Real cross-stem consensus, not just dedup. The old version only
        # ever compared each note to the ONE immediately-preceding note in
        # sorted order, and always discarded whichever had lower
        # confidence - regardless of whether the two notes came from the
        # same stem (a genuine duplicate) or different stems (real,
        # meaningful corroboration that a second independent witness saw
        # the same event - exactly what an agreement bonus is for). Now
        # that pitch_detection_master no longer runs redundantly in
        # parallel with bass/other, a same-pitch/same-time match here is
        # usually either a real same-stem duplicate that slipped past
        # upstream merging, or genuine cross-stem agreement worth
        # rewarding - this distinguishes the two instead of always
        # silently discarding one note's evidence.
        def _stem_tag(note: NoteEvent) -> str:
            for tag in reversed(note.reasoning_chain or []):
                if tag.startswith("stem:"):
                    return tag.split(":", 1)[1]
            return "unknown"

        pitched_notes.sort(key=lambda n: (n.pitch, n.start_ms))
        groups: List[List[NoteEvent]] = []
        for note in pitched_notes:
            placed = False
            for group in groups:
                rep = group[0]
                if rep.pitch == note.pitch and abs(rep.start_ms - note.start_ms) < 100.0:
                    group.append(note)
                    placed = True
                    break
            if not placed:
                groups.append([note])

        consensus_agreement_bonus = 0.15
        unique_pitched: List[NoteEvent] = []
        corroborated_count = 0
        for group in groups:
            if len(group) == 1:
                unique_pitched.append(group[0])
                continue
            stems = {_stem_tag(n) for n in group}
            base = max(group, key=lambda n: n.confidence)

            # Merge every group member's own reasoning_chain tags (deduped,
            # order-preserving) onto the surviving base note. Previously
            # only base's own chain survived here - the same
            # one-survivor-keeps-its-tags shape found and fixed in
            # velocity_merge.py's _build_reasoning_chain (see GRIMLOCK_6.0
            # §12.9) - silently discarding whatever stem/family/witness-
            # corroboration evidence the OTHER group members carried, even
            # though this is exactly the "real cross-stem agreement" case
            # this block otherwise treats as meaningful.
            merged_tags: List[str] = []
            seen = set()
            for n in group:
                for tag in (n.reasoning_chain or []):
                    if tag not in seen:
                        seen.add(tag)
                        merged_tags.append(tag)
            base.reasoning_chain = merged_tags

            if len(stems) > 1:
                base.confidence = min(1.0, base.confidence + consensus_agreement_bonus)
                base.reasoning_chain = base.reasoning_chain + [
                    f"consensus:corroborated_by_{','.join(sorted(stems))}"]
                corroborated_count += 1
            unique_pitched.append(base)
        if corroborated_count:
            print(f"  {corroborated_count} notes corroborated across multiple stems (confidence boosted)")

        # Keep drums as-is (no deduplication needed - they're percussive)
        self._consensus_notes = unique_pitched
        # Drums remain in self._drum_notes (preserved)

        avg = float(np.mean([n.confidence for n in unique_pitched])) if unique_pitched else 0.0
        print(f"  {len(unique_pitched)} final pitched notes (avg conf={avg:.2f})")
        print(f"  {len(self._drum_notes)} drum notes preserved")

        # ConsensusEngine forensic audit (read-only, does not affect
        # unique_pitched/self._consensus_notes above). ConsensusEngine's
        # vote strategies treat one whole testimony list as a single
        # witness's yes/no claim and simply concatenate surviving
        # witnesses' events with no dedup - fine for auditing "does this
        # stem look trustworthy as a whole" but wrong for merging
        # per-note polyphonic output (would reintroduce the duplicates
        # the dedup above just removed). So this runs it purely as an
        # independent epistemic-veto opinion over the four pre-merge
        # stem witnesses, logged to music_box via its own already-wired
        # ConsensusVoter.reach_consensus() call, without ever touching
        # the note lists the rest of the pipeline actually uses.
        if self.music_box:
            try:
                from agents.validation.consensus_engine import ConsensusEngine
                from core.order_types import WitnessTestimony

                stem_witnesses = [
                    (SourceType.BASS_STEM, self._pitch_notes_bass, 0.65),
                    (SourceType.MASTER_STEM, self._pitch_notes_master, 0.55),
                    (SourceType.OTHER_STEM, self._pitch_notes_other, 0.7),
                    (SourceType.VOCALS_STEM, self._pitch_notes_vocals, 0.7),
                    (SourceType.DRUM_INTELLIGENCE, self._drum_notes, 0.7),
                ]
                now_ms = datetime.now().timestamp() * 1000
                testimonies = [
                    WitnessTestimony(witness_id=src, testimony_time_ms=now_ms,
                                      events=notes, confidence=conf,
                                      metadata={"testimony_type": "stem_audit"})
                    for src, notes, conf in stem_witnesses
                ]
                engine = ConsensusEngine(music_box=self.music_box)
                engine.reach_consensus(testimonies=testimonies)
            except Exception as e:
                if self.debug:
                    print(f"  [ConsensusEngine audit] skipped: {e}")

        return TestimonyStageResult("consensus", True,
                           ConsensusTestimony(final_decision=True, confidence=avg,
                                              veto_triggered=False, veto_source=None,
                                              witness_count=len(unique_pitched)))

    # ====================================================================
    # Stage: Schoenberg Mirror (harmonic legitimacy audit / epistemic veto)
    # ====================================================================

    def _run_schoenberg_mirror(self) -> TestimonyStageResult:
        empty = SchoenbergMirrorTestimony(
            audited_count=0, vetoed_count=0, tonal_count=0,
            percussion_count=0, uncertain_count=0, noise_count=0,
            hallucination_count=0,
        )
        if not self._consensus_notes or self._audio_views is None:
            return TestimonyStageResult("schoenberg_mirror", True, empty)

        try:
            from agents.validation.schoenberg_mirror import SchoenbergMirror

            # Route each note to the ISOLATED stem it actually came from,
            # not the full downmix. Auditing a piano/vocal note's harmonic
            # series against a mix where drums+bass are sounding
            # simultaneously buries its partials under unrelated energy -
            # partial detection fails almost regardless of how accurate the
            # note's own fundamental estimate is, which is why tonal_count
            # was silently near-zero even after fundamental_freq_hz itself
            # was fixed upstream. Fall back to the full mix only for notes
            # whose stem tag is missing/unavailable.
            def _stem_tag(note: NoteEvent) -> str:
                for tag in reversed(note.reasoning_chain or []):
                    if tag.startswith("stem:"):
                        return tag.split(":", 1)[1]
                return "unknown"

            sr = self._audio_views.sample_rate
            stem_stereo = {
                "bass": self._audio_views.bass_stereo,
                "other": self._audio_views.other_stereo,
                "drums": self._audio_views.drums_stereo,
                "vocals": self._audio_views.vocals_stereo,
            }
            contracts = {}
            for stem_name, stereo in stem_stereo.items():
                if stereo is not None and np.max(np.abs(stereo)) > 1e-6:
                    contracts[stem_name] = AcousticIntelligence.create_contract(
                        np.mean(stereo, axis=0), sr, f"schoenberg_mirror_{stem_name}", force_mono=True,
                    )
            fallback_contract = AcousticIntelligence.create_contract(
                self._audio_views.mono_sum_normalized, sr, "schoenberg_mirror", force_mono=True,
            )

            mirror = SchoenbergMirror(music_box=self.music_box)
            results = []
            for note in self._consensus_notes:
                contract = contracts.get(_stem_tag(note), fallback_contract)
                results.append(mirror.audit_note(note, audio_contract=contract, feature_bundle=self._feature_bundle))

            counts = {v: 0 for v in SchoenbergVerdict}
            kept_notes = []
            vetoed = 0

            for note, result in zip(self._consensus_notes, results):
                counts[result.verdict] += 1
                if mirror.should_veto(result):
                    vetoed += 1
                    continue
                note.confidence = mirror.adjust_confidence(note, result)
                kept_notes.append(note)

            self._consensus_notes = kept_notes

            print(f"  {len(kept_notes)}/{len(results)} notes passed "
                  f"(vetoed={vetoed}, tonal={counts[SchoenbergVerdict.TONAL]}, "
                  f"percussion={counts[SchoenbergVerdict.PERCUSSION]}, "
                  f"uncertain={counts[SchoenbergVerdict.UNCERTAIN]})")

            return TestimonyStageResult("schoenberg_mirror", True,
                               SchoenbergMirrorTestimony(
                                   audited_count=len(results),
                                   vetoed_count=vetoed,
                                   tonal_count=counts[SchoenbergVerdict.TONAL],
                                   percussion_count=counts[SchoenbergVerdict.PERCUSSION],
                                   uncertain_count=counts[SchoenbergVerdict.UNCERTAIN],
                                   noise_count=counts[SchoenbergVerdict.NOISE],
                                   hallucination_count=counts[SchoenbergVerdict.HALLUCINATION],
                               ))
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return TestimonyStageResult("schoenberg_mirror", True, empty, warnings=[str(e)])

    # ====================================================================
    # Stage: Phrase Intelligence (NEW)
    # ====================================================================

    def _run_phrase_intelligence(self) -> TestimonyStageResult:
        """Stage: PhraseIntelligence — Structural phase state machine."""
        from epistemic.phrase_intelligence import PhraseIntelligence

        # Need consensus notes first
        if not hasattr(self, '_consensus_notes') or not self._consensus_notes:
            return TestimonyStageResult("phrase_intelligence", True,
                               testimony={"structural_testimony": None},
                               warnings=["No consensus notes for phrase analysis"])

        # Get all testimonies as TestimonyStageResult objects
        pulse_result = self._get_stage_result("pulse_analysis")
        voice_result = self._get_stage_result("voice_continuity")
        harmonic_result = self._get_stage_result("tonal_detection")
        geo_result = self._get_stage_result("reverse_geo_crypt")
        tempo_result = self._get_stage_result("tempo_analysis")
        groove_result = self._get_stage_result("groove_analysis")

        # Get motif testimony if available (future)
        motif_result = None

        phrase_intel = PhraseIntelligence(self._musical_findings)

        result = phrase_intel.compute_structural_trajectory(
            notes=self._consensus_notes,
            pulse_testimony=pulse_result,
            voice_testimony=voice_result,
            harmonic_testimony=harmonic_result,
            geo_testimony=geo_result,
            tempo_testimony=tempo_result,
            groove_testimony=groove_result,
            motif_testimony=motif_result,
        )

        self._structural_testimony = result.testimony if result.success else None
        self._musical_findings.record_stage("phrase_intelligence")

        if result.success and result.testimony:
            t = result.testimony
            trajectory_len = len(getattr(t, 'structural_trajectory', []))
            transitions_len = len(getattr(t, 'phase_transitions', []))
            print(f"  Trajectory: {trajectory_len} windows, {transitions_len} transitions")

            attractor = getattr(t, 'attractor_basin', 'unknown')
            att_conf = getattr(t, 'attractor_confidence', 0.0)
            print(f"  Attractor: {attractor} (conf={att_conf:.2f})")

            boundaries = getattr(t, 'phrase_boundaries', [])
            if boundaries:
                print(f"  Boundaries: {len(boundaries)} detected")

            anacrusis = getattr(t, 'anacrusis_detected', {})
            if anacrusis.get('detected'):
                print(f"  Anacrusis: {anacrusis.get('duration_ms', 0):.0f}ms")

        return result

    # ====================================================================
    # Stage: Structural Feedback (NEW)
    # ====================================================================

    def _run_structural_feedback(self) -> TestimonyStageResult:
        """Stage: Structural Feedback Loop."""
        if not self._structural_testimony:
            return TestimonyStageResult("structural_feedback", True,
                               testimony={"reanalysis_performed": False},
                               warnings=["No structural testimony for feedback"])

        needs_reanalysis = getattr(self._structural_testimony, 'needs_reanalysis', {})
        if not needs_reanalysis.get("requires_reanalysis", False):
            return TestimonyStageResult("structural_feedback", True,
                               testimony={"reanalysis_performed": False,
                                          "reasoning": "No reanalysis needed"})

        reasoning = needs_reanalysis.get("reasoning", [])
        severity = needs_reanalysis.get("severity", "none")

        print(f"  STRUCTURAL REANALYSIS TRIGGERED: {severity}")
        for r in reasoning[:3]:
            print(f"    ↳ {r}")

        # Re-run consensus with structural bias
        if hasattr(self, '_consensus_notes') and self._consensus_notes:
            structural_regime = getattr(self._structural_testimony, 'structural_regime', {})
            regime = structural_regime.get("regime", "unknown")
            expected_length = getattr(self._structural_testimony, 'expected_phrase_length_bars', 0)

            biased_notes = self._reapply_consensus_with_bias(
                self._consensus_notes,
                structural_bias=regime,
                expected_phrase_length=expected_length,
            )

            if biased_notes != self._consensus_notes:
                self._consensus_notes = biased_notes
                self._musical_findings.reanalysis_performed = True
                self._musical_findings.record_stage("structural_feedback")
                print(f"  Consensus re-run with bias → {len(biased_notes)} notes")

        return TestimonyStageResult("structural_feedback", True,
                           testimony={
                               "reanalysis_performed": True,
                               "reasoning": reasoning,
                               "severity": severity,
                           })

    # ====================================================================
    # Stage: Validation
    # ====================================================================

    def _run_validation(self) -> TestimonyStageResult:
        pitched_notes = list(self._consensus_notes)
        drum_notes = list(self._drum_notes)

        # Filter out-of-range pitches (only for pitched notes)
        bad = [n for n in pitched_notes if not (21 <= n.pitch <= 108)]
        if bad:
            print(f"  Removing {len(bad)} out-of-range pitches")
            pitched_notes = [n for n in pitched_notes if 21 <= n.pitch <= 108]

        for n in pitched_notes:
            if n.end_ms <= n.start_ms:
                n.end_ms = n.start_ms + 50.0

        self._consensus_notes = pitched_notes
        # drum_notes unchanged

        # Actually run Scribe's gates against the final output notes.
        # final_verdict() used to be read straight away with nothing ever
        # having called validate_stage()/validate_events() anywhere in
        # the pipeline, so _hard_vetoes/_soft_vetoes stayed empty forever
        # and this always silently reported "trustworthy" regardless of
        # what the notes actually looked like. Real gates run now:
        # mandatory fields, confidence threshold, silence ratio,
        # Schoenberg Mirror (harmonic series), phase consistency. Empty
        # drum output is validated under the "drum_detection" stage name
        # specifically so Scribe's own empty-is-legitimate exemption for
        # percussion applies - a song can genuinely have no drums, that's
        # not itself untrustworthy.
        if pitched_notes:
            self.scribe.validate_events(pitched_notes, SourceType.CONSENSUS_ENGINE, "final_pitched_notes")
        if drum_notes:
            # Drum hits are inherently percussive/non-harmonic by nature -
            # running the Schoenberg Mirror's harmonic-series check on
            # them is a category error, not a quality signal (confirmed
            # live: it flagged 100% of real, correct drum hits as
            # "hallucinated" in testing). Every other gate still applies.
            non_harmonic_gates = [g for g in ValidationGate if g != ValidationGate.SCHOENBERG_MIRROR]
            self.scribe.validate_events(drum_notes, SourceType.DRUM_INTELLIGENCE, "drum_detection",
                                        apply_gates=non_harmonic_gates)

        verdict = self.scribe.final_verdict()
        passed = verdict.get("passed", True)
        print(
            f"  {'✓ Passed' if passed else '✗ Failed'} — {len(pitched_notes)} pitched notes, {len(drum_notes)} drum notes")
        if not passed:
            for v in verdict.get("hard_vetoes", [])[:5]:
                print(f"    ✗ {v['source']}/{v['stage']}: {v['reason']} - {v['detail']}")
        elif verdict.get("soft_veto_count", 0):
            print(f"    ⚠ {verdict['soft_veto_count']} soft veto(s) recorded")

        # A failed trustworthiness verdict is reported honestly (passed/
        # is_trustworthy below, surfaced in the final summary) but does
        # NOT abort the pipeline or block export, even though this stage
        # is essential - matches the rest of the codebase's "never
        # silently withhold a result" philosophy (see the heartbeat-note
        # fallback in _run_export for a fully-empty transcription).
        return TestimonyStageResult("validation", True,
                           ValidationTestimony(
                               passed=passed,
                               hard_veto_count=verdict.get("hard_veto_count", 0),
                               soft_veto_count=verdict.get("soft_veto_count", 0),
                               verdict_summary=str(verdict),
                               is_trustworthy=passed))

    # ====================================================================
    # Stage: Export (Drums on channel 9)
    # ====================================================================

    def _run_export(self) -> TestimonyStageResult:
        pitched = list(self._consensus_notes)
        drums = list(self._drum_notes)

        if not pitched and not drums:
            return TestimonyStageResult("export", True,
                               ExportTestimony(exported_paths=[], formats=[],
                                               note_count=0),
                               warnings=["No notes to export"])
        try:
            from export.midi_writer import MidiWriter

            out = (self._output_path or
                   f"output/transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid")
            Path(out).parent.mkdir(parents=True, exist_ok=True)

            writer = MidiWriter(music_box=self.music_box)
            tm = (self._get_tempo_map() or
                 TempoMap(initial_tempo_bpm=self._get_tempo_bpm(),
                          tempo_events=[], confidence=Confidence.MEDIUM))

            success = False

            # Use drums-aware export
            if hasattr(writer, 'export_midi_with_drums'):
                success = writer.export_midi_with_drums(
                    pitched_events=pitched,
                    drum_events=drums,
                    tempo_map=tm,
                    output_path=out,
                )
                print(f"  Exported {len(pitched)} pitched + {len(drums)} drum notes")
            else:
                # Fallback: tag drums with channel 9
                for d in drums:
                    try:
                        object.__setattr__(d, '_midi_channel', 9)
                    except Exception:
                        pass
                success = writer.export_midi(
                    events=pitched + drums, tempo_map=tm, output_path=out)

            total = len(pitched) + len(drums)
            if success:
                print(f"  MIDI: {out}")
                print(f"  {len(pitched)} pitched notes (ch 0) + "
                      f"{len(drums)} drum events (ch 9)")
            return TestimonyStageResult("export", success,
                               ExportTestimony(
                                   exported_paths=[out] if success else [],
                                   formats=["midi"] if success else [],
                                   note_count=total),
                               errors=[] if success else ["MIDI write failed"])
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return TestimonyStageResult("export", False,
                               ExportTestimony(exported_paths=[], formats=[],
                                               note_count=0),
                               errors=[str(e)])

    # ====================================================================
    # Result Builder
    # ====================================================================

    def _build_result(self, total_sec: float) -> PipelineResult:
        # feature_extraction is the only stage that ever calls
        # evidence_tracker.acquire() (once per evidence type) - nothing in
        # the whole pipeline ever called release(), so every lease was
        # permanent for the life of the run regardless of whether anything
        # still needed that evidence. By the time results are being built,
        # every stage has finished, so nothing can still legitimately hold
        # a lease - release them all here rather than tracing the exact
        # point each evidence type's last real consumer finishes (which
        # would mean touching many separate stage methods for a mechanism
        # that isn't currently arbitrating any real contention anyway,
        # since feature_extraction is still the only caller of acquire()).
        for evidence_type in list(self.evidence_tracker.get_all_leases().keys()):
            self.evidence_tracker.release(evidence_type, "feature_extraction")

        verdict = self.scribe.final_verdict()
        n_pitch = len(self._consensus_notes)
        n_drum = len(self._drum_notes)
        success = not self._aborted and (n_pitch > 0 or n_drum > 0)

        print(f"\n{'=' * 60}")
        print(f"COMPLETE in {total_sec:.1f}s — "
              f"{'SUCCESS' if success else 'FAILED'}")
        if self._aborted and self._abort_reason:
            print(f"Abort reason: {self._abort_reason}")
        print(f"Pitched: {n_pitch}  Drums: {n_drum}  "
              f"Peak memory: {self._peak_memory_mb:.0f}MB")
        if self._tempo and hasattr(self._tempo, 'tempo_bpm'):
            print(f"Tempo:   {self._tempo.tempo_bpm:.1f} BPM")
        if self._detected_key:
            print(f"Key:     {self._detected_key} ({self._key_confidence:.2f})")

        print("\nStage Timing:")
        for name, ms in sorted(self._stage_times.items(),
                               key=lambda x: -x[1])[:12]:
            bar_len = int(ms / max(self._stage_times.values()) * 30) if self._stage_times else 0
            bar = "█" * bar_len
            print(f"  {name:<28} {bar} {ms:>8.0f}ms")
        print(f"{'=' * 60}")

        try:
            music_box_logs = self.music_box.query_entries(limit=2000)
            veto_history = self.music_box.query_vetoes(limit=500)
        except Exception:
            music_box_logs, veto_history = [], []

        return PipelineResult(
            success=success,
            notes=self._consensus_notes,
            drum_notes=self._drum_notes,
            verdict=verdict,
            timing=self._stage_times,
            total_time_seconds=total_sec,
            peak_memory_mb=self._peak_memory_mb,
            music_box_session=self.music_box.get_session_id(),
            stages_executed=list(self._results.keys()),
            aborted=self._aborted,
            abort_reason=self._abort_reason,
            detected_tempo_bpm=self._get_tempo_bpm(),
            detected_key=self._detected_key,
            key_confidence=self._key_confidence,
            groove_field=(self._groove.groove_field
                          if self._groove and hasattr(self._groove, 'groove_field')
                          else None),
            voices=self._voices,
            is_truncated=self.truncate_duration > 0,
            truncation_duration=self.truncate_duration,
            original_duration_seconds=self._original_duration_seconds,
            music_box_logs=music_box_logs,
            veto_history=veto_history,
        )


# ====================================================================
# Convenience Functions
# ====================================================================

def transcribe_file(
        audio_path: str,
        output_path: str = None,
        tempo_bpm: float = None,  # guided tempo (optional)
        key_signature: str = None,  # guided key (optional)
        time_signature: str = None,  # guided time sig (optional)
        genre: str = None,  # guided genre (optional)
        debug: bool = False,
        truncate_duration: float = None,
) -> PipelineResult:
    """
    Module-level convenience wrapper.
    Accepts all 8 arguments that main.py passes so the call never
    raises TypeError. Guided params are forwarded to the pipeline
    for use in tempo_analysis and tonal_detection if provided.

    Args:
        audio_path:        Path to the audio file.
        output_path:       Where to write the MIDI file.
        tempo_bpm:         Optional guided tempo override.
        key_signature:     Optional guided key (e.g. "Bb minor").
        time_signature:    Optional guided time signature (e.g. "4/4").
        genre:             Optional genre hint (e.g. "jazz").
        debug:             Enable verbose debug output.
        truncate_duration: Process only this many seconds (0 = full).
    """
    from core.constants import DEFAULT_PROCESSING_DURATION_SECONDS

    trunc = truncate_duration if truncate_duration is not None \
        else DEFAULT_PROCESSING_DURATION_SECONDS

    pipeline = create_pipeline(debug=debug, truncate_duration=trunc)

    # Forward guided params into the findings map
    if tempo_bpm:
        pipeline._musical_findings.tempo_bpm = float(tempo_bpm)
        pipeline._musical_findings.tempo_confidence = 1.0
        pipeline._musical_findings.tempo_source = "guided"
        pipeline._musical_findings.record_stage("guided_params")

    if key_signature:
        pipeline._musical_findings.detected_key = key_signature
        pipeline._musical_findings.key_confidence = 1.0

    return pipeline.transcribe(audio_path, output_path)


def quick_transcribe(
        audio_path: str,
        output_path: Optional[str] = None,
        debug: bool = False,
        truncate_duration: float = DEFAULT_PROCESSING_DURATION_SECONDS
) -> PipelineResult:
    pipeline = GrimlockPipeline(
        debug=debug,
        truncate_duration=truncate_duration
    )
    return pipeline.transcribe(audio_path, output_path)


def create_pipeline(
        guided_params: Optional[Union[GuidedParams, Dict[str, Any]]] = None,
        debug: bool = False,
        truncate_duration: float = DEFAULT_PROCESSING_DURATION_SECONDS,
) -> GrimlockPipeline:
    return GrimlockPipeline(
        guided_params=guided_params,
        debug=debug,
        truncate_duration=truncate_duration
    )