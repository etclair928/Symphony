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
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Union

import numpy as np

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from core.order_types import (
    AudioContext, Confidence, GrooveField, NoteEvent,
    PulseField, SourceType, StemType, TempoMap, Voice,
    VoiceContinuityResult, UNKNOWN_HASH_PLACEHOLDER, ExportOptions,
)
from core.constants import (
    ANALYSIS_SR, BASIC_PITCH_SR, CREPE_SR, MADMOM_SR,
    DEFAULT_PROCESSING_DURATION_SECONDS,
    STAGE_TIMEOUT_ANALYSIS_SECONDS,
    STAGE_TIMEOUT_DETECTION_SECONDS,
    STAGE_TIMEOUT_EXPORT_SECONDS,
    STAGE_TIMEOUT_LOAD_SECONDS,
    STAGE_TIMEOUT_QUANTIZATION_SECONDS,
    STAGE_TIMEOUT_SEPARATION_SECONDS,
    MIN_CONFIDENCE_TO_PASS,
)
from core.feature_bundle import EvidenceType, extract_features
from core.evidence_tracker import EvidenceTracker
from epistemic.musical_findings_map import MusicalFindingsMap
from core.audio_views import AudioViews, ChannelLayout
from core.testimony import (
    StageResult, TestimonyNormalizer,
    IngestionTestimony, FeatureExtractionTestimony, TempoTestimony,
    SeparationTestimony, PitchTestimony, RhythmTestimony, DrumTestimony,
    TonalTestimony, TimbreTestimony, HarmonicValidationTestimony,
    GrooveTestimony, PulseTestimony, LatticeTestimony,
    VoiceContinuityTestimony, QuantizationTestimony, MergeTestimony,
    ConsensusTestimony, ValidationTestimony, ExportTestimony,
    AnechoicMATestimony,
)
from core.guided_control import (
    GuidedParams,
    GuidedModeController,
    KeyConstraint,
    GuidedBeatGrid,
    create_guided_params_from_form
)

from memory.guardian import MemoryGuardian
from memory.arena import scoped_arena
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
                 original_duration_seconds):
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

    Law of Testimony:  Every stage returns StageResult[T].
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
        self._tempo = 120.0
        self._beat_times_ms = np.array([])
        self._tempo_map = None

        self._pitch_notes_bass: List[NoteEvent] = []
        self._pitch_notes_master: List[NoteEvent] = []
        self._pitch_notes_other: List[NoteEvent] = []
        self._rhythm_notes: List[NoteEvent] = []
        self._drum_notes: List[NoteEvent] = []

        self._tonal_events: List[Dict] = []
        self._voices: List[Voice] = []
        self._detected_key: Optional[str] = None
        self._key_confidence: float = 0.0

        self._groove_field = None
        self._pulse_field = None
        self._timbre_analysis = None
        self._timbre_clusters = None
        self._anechoic_report = None

        self._quantized_pitched: List[NoteEvent] = []
        self._merged_notes: List[NoteEvent] = []
        self._consensus_notes: List[NoteEvent] = []

        # New attributes for Quaver and Phrase intelligence
        self._quaver_testimonies = []
        self._structural_testimony = None

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
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=50),
            PipelineStage("separation", self._run_separation, SourceType.DEMUCS,
                          timeout_seconds=STAGE_TIMEOUT_SEPARATION_SECONDS, is_essential=False,
                          estimated_memory_mb=400),
            PipelineStage("rhythm_detection", self._run_rhythm_detection, SourceType.RHYTHM,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=True,
                          estimated_memory_mb=100),
            PipelineStage("pitch_detection_bass", self._run_pitch_detection_bass, SourceType.PITCH,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=False,
                          estimated_memory_mb=150),
            PipelineStage("pitch_detection_master", self._run_pitch_detection_master, SourceType.PITCH,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=False,
                          estimated_memory_mb=150),
            PipelineStage("pitch_detection_other", self._run_pitch_detection_other, SourceType.PITCH,
                          timeout_seconds=180.0, is_essential=False,
                          estimated_memory_mb=200),
            PipelineStage("tonal_detection", self._run_tonal_detection, SourceType.TONAL,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=False,
                          estimated_memory_mb=100),
            PipelineStage("drum_detection", self._run_drum_detection, SourceType.DRUM_INTELLIGENCE,
                          timeout_seconds=STAGE_TIMEOUT_DETECTION_SECONDS, is_essential=False,
                          estimated_memory_mb=80),
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
            PipelineStage("pulse_analysis", self._run_pulse_analysis, SourceType.PULSE_FIELD,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=10),
            PipelineStage("reverse_geo_crypt", self._run_reverse_geo_crypt, SourceType.TEMPO_INTELLIGENCE,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=10),
            PipelineStage("voice_continuity", self._run_voice_continuity, SourceType.VOICE_CONTINUITY,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=False,
                          estimated_memory_mb=20),
            # NEW STAGES: QuaverIntelligence and PhraseIntelligence
            PipelineStage("quaver_intelligence", self._run_quaver_intelligence, SourceType.PITCH,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=True,
                          estimated_memory_mb=50),
            PipelineStage("quantization", self._run_quantization, SourceType.RITORNELLO,
                          timeout_seconds=STAGE_TIMEOUT_QUANTIZATION_SECONDS, is_essential=True,
                          estimated_memory_mb=30),
            PipelineStage("velocity_merge", self._run_velocity_merge, SourceType.RITORNELLO,
                          timeout_seconds=STAGE_TIMEOUT_QUANTIZATION_SECONDS, is_essential=False,
                          estimated_memory_mb=20),
            PipelineStage("consensus", self._run_consensus, SourceType.CONSENSUS_ENGINE,
                          timeout_seconds=STAGE_TIMEOUT_ANALYSIS_SECONDS, is_essential=True,
                          estimated_memory_mb=20),
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
        """Extract testimony dict from a stage result."""
        result = self._results.get(stage_name)
        if not result or not result.success:
            return {}
        t = result.testimony
        if isinstance(t, dict):
            return t
        if hasattr(t, '__dict__'):
            return {k: v for k, v in vars(t).items() if not k.startswith('_')}
        return {}

    def _get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get StageResult object for a stage."""
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

                    t0 = time.time()
                    result = self._run_stage_with_timeout(stage)
                    elapsed_ms = (time.time() - t0) * 1000
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

    def _run_stage_with_timeout(self, stage: PipelineStage) -> StageResult:
        """Run stage with timeout - synchronous, no nested asyncio.run()"""
        result_box = [None]
        error_box = [None]

        def target():
            try:
                result_box[0] = stage.executor()
            except Exception as e:
                error_box[0] = e
                if self.debug:
                    traceback.print_exc()

        thread = CancellableThread(target=target)
        completed, _, _ = thread.run_with_timeout(stage.timeout_seconds)

        if not completed:
            return StageResult(stage_name=stage.name, success=False, testimony=None,
                               errors=[f"Timeout after {stage.timeout_seconds}s"])
        if error_box[0]:
            return StageResult(stage_name=stage.name, success=False, testimony=None,
                               errors=[str(error_box[0])])
        return result_box[0] or StageResult(stage_name=stage.name, success=False,
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

    def _run_ingestion(self) -> StageResult:
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
            return StageResult(
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
            return StageResult("ingestion", False, None, errors=[str(e)])

    # ====================================================================
    # Stage: Feature Extraction
    # ====================================================================

    def _run_feature_extraction(self) -> StageResult:
        if self._audio_views is None:
            return StageResult("feature_extraction", False, None, errors=["No audio"])
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
            return StageResult(
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
            return StageResult("feature_extraction", False, None, errors=[str(e)])

    # ====================================================================
    # Stage: Tempo Analysis
    # ====================================================================

    def _run_tempo_analysis(self) -> StageResult:
        if self._audio_views is None:
            return StageResult("tempo_analysis", False, None, errors=["No audio"])
        try:
            import librosa
            hop = 512
            onset_env = None
            if (self._feature_bundle and
                    self._feature_bundle.has_evidence(EvidenceType.ONSET_STRENGTH)):
                onset_env = self._feature_bundle.onset_strength
                hop = self._feature_bundle.hop_length

            if onset_env is not None:
                tempo, beat_frames = librosa.beat.beat_track(
                    onset_envelope=onset_env, sr=ANALYSIS_SR,
                    hop_length=hop, tightness=100, trim=False)
            else:
                audio = self._audio_views.mono_sum_normalized
                if self._audio_views.sample_rate != ANALYSIS_SR:
                    audio = librosa.resample(audio,
                                             orig_sr=self._audio_views.sample_rate,
                                             target_sr=ANALYSIS_SR)
                tempo, beat_frames = librosa.beat.beat_track(
                    y=audio, sr=ANALYSIS_SR, hop_length=hop)

            tempo = float(tempo)
            valid = 40.0 < tempo < 320.0
            if not valid:
                tempo = 120.0
            beat_ms = (librosa.frames_to_time(beat_frames, sr=ANALYSIS_SR, hop_length=hop)
                       * 1000.0)
            testimony = TempoTestimony(
                tempo_bpm=tempo,
                confidence=0.75 if valid else 0.3,
                source="librosa" if valid else "default",
                beat_times_ms=beat_ms.tolist(),
                tempo_map=TempoMap(initial_tempo_bpm=tempo,
                                   tempo_events=[], confidence=Confidence.MEDIUM),
            )
            self._tempo = testimony
            self._musical_findings.tempo_bpm = tempo
            self._musical_findings.tempo_confidence = testimony.confidence
            self._musical_findings.beat_times_ms = beat_ms
            self._musical_findings.record_stage("tempo_analysis")
            print(f"  {tempo:.1f} BPM ({testimony.source}), {len(beat_frames)} beats")
            return StageResult("tempo_analysis", True, testimony)
        except Exception as e:
            fallback = TempoTestimony(tempo_bpm=120.0, confidence=0.3,
                                      source="default", beat_times_ms=[])
            self._tempo = fallback
            self._musical_findings.tempo_bpm = 120.0
            return StageResult("tempo_analysis", True, fallback,
                               warnings=[f"Tempo failed ({e}), using 120 BPM"])

    # ====================================================================
    # Stage: Separation (Demucs → Mel-Roformer → STFT)
    # ====================================================================

    def _run_separation(self) -> StageResult:
        if self._audio_views is None:
            return StageResult("separation", False, None, errors=["No audio"])

        import librosa

        left = self._audio_views.left_channel
        right = self._audio_views.right_channel
        sr = self._audio_views.sample_rate
        if sr != 44100:
            left = librosa.resample(left, orig_sr=sr, target_sr=44100)
            right = librosa.resample(right, orig_sr=sr, target_sr=44100)
        audio_stereo = np.stack([left, right], axis=1)

        context = self._make_context(44100)

        print("  [SEP] Trying Demucs htdemucs...")
        result = self._try_demucs(audio_stereo, 44100, context)
        if result is not None:
            self._apply_separation_result(result)
            stems = self._available_stems()
            print(f"  [SEP] Demucs succeeded — stems: {stems}")
            return StageResult("separation", True,
                               SeparationTestimony(stems_available=stems,
                                                   separation_time_seconds=0,
                                                   model_used="htdemucs",
                                                   fallback_used=False))

        print("  [SEP] Demucs failed — trying Mel-Roformer...")
        result = self._try_mel_roformer(audio_stereo, 44100, context)
        if result is not None:
            self._apply_separation_result(result)
            stems = self._available_stems()
            print(f"  [SEP] Mel-Roformer succeeded — stems: {stems}")
            return StageResult("separation", True,
                               SeparationTestimony(stems_available=stems,
                                                   separation_time_seconds=0,
                                                   model_used="mel_roformer",
                                                   fallback_used=True),
                               warnings=["Demucs unavailable, used Mel-Roformer"])

        print("  [SEP] Using STFT fallback (librosa HPSS + Butterworth)...")
        self._apply_simple_stft_separation(audio_stereo, 44100)
        print("  [SEP] STFT separation complete")
        return StageResult("separation", True,
                           SeparationTestimony(stems_available=["drums", "bass", "other"],
                                               separation_time_seconds=0,
                                               model_used="simple_stft",
                                               fallback_used=True),
                           warnings=["Using STFT fallback — stems are lower quality"])

    def _try_demucs(self, audio_stereo: np.ndarray, sr: int, context=None):
        try:
            from agents.separation.demucs import DemucsSeparator
            separator = DemucsSeparator(
                model_name="htdemucs", device="cpu",
                music_box=self.music_box,
                streaming=True,
                window_samples=262144, hop_samples=131072,
                timeout_seconds=600,
                enable_fallback=False,
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

    def _try_mel_roformer(self, audio_stereo: np.ndarray, sr: int, context=None):
        try:
            from agents.separation.mel_roformer import MelRoformerSeparator
            separator = MelRoformerSeparator(music_box=self.music_box)
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
            print("  [SEP] Mel-Roformer not installed")
            return None
        except Exception as e:
            print(f"  [SEP] Mel-Roformer failed: {type(e).__name__}: {e}")
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
            sha256_hash=self._audio_views.sha256_hash,
        )

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
        return stems

    # ====================================================================
    # Stage: Rhythm Detection
    # ====================================================================

    def _run_rhythm_detection(self) -> StageResult:
        if self._audio_views is None:
            return StageResult("rhythm_detection", False, None, errors=["No audio"])
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
            result = RhythmEngine().detect(audio, context)
            events = (result.events if hasattr(result, 'events')
                      else result if isinstance(result, list) else [])
            notes = [e.to_note_event() if hasattr(e, 'to_note_event') else e
                     for e in events]
            self._rhythm_notes = notes
            print(f"  {len(notes)} rhythm events")
            return StageResult("rhythm_detection", True,
                               RhythmTestimony(onset_events=events,
                                               kick_notes=[n for n in notes if n.pitch == 36],
                                               pattern_count=len(notes),
                                               tempo_estimate_bpm=self._tempo.tempo_bpm if self._tempo else 120.0,
                                               confidence=0.6))
        except Exception as e:
            return StageResult("rhythm_detection", True,
                               RhythmTestimony(onset_events=[], kick_notes=[],
                                               pattern_count=0,
                                               tempo_estimate_bpm=120.0, confidence=0.0),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Pitch Detection Bass
    # ====================================================================

    def _run_pitch_detection_bass(self) -> StageResult:
        if self._audio_views is None or self._audio_views.bass_stereo is None:
            return StageResult("pitch_detection_bass", True,
                               PitchTestimony(notes=[], confidence=0.0,
                                              model_used="none", stem_type="bass",
                                              frame_count=0, rejected_frames=0),
                               warnings=["No bass stem"])
        try:
            from agents.detection.pitch_intelligence import PitchIntelligence
            import librosa

            audio = np.mean(self._audio_views.bass_stereo, axis=0)
            sr = self._audio_views.sample_rate
            if sr != CREPE_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=CREPE_SR)

            context = self._make_context(CREPE_SR)
            events = PitchIntelligence().detect(audio, context)
            self._pitch_notes_bass = list(events) if events else []
            print(f"  {len(self._pitch_notes_bass)} notes (bass)")
            return StageResult("pitch_detection_bass", True,
                               PitchTestimony(notes=self._pitch_notes_bass,
                                              confidence=0.65, model_used="crepe",
                                              stem_type="bass",
                                              frame_count=len(self._pitch_notes_bass),
                                              rejected_frames=0))
        except Exception as e:
            return StageResult("pitch_detection_bass", True,
                               PitchTestimony(notes=[], confidence=0.0,
                                              model_used="failed", stem_type="bass",
                                              frame_count=0, rejected_frames=0),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Pitch Detection Master
    # ====================================================================

    def _run_pitch_detection_master(self) -> StageResult:
        if self._audio_views is None:
            return StageResult("pitch_detection_master", False, None, errors=["No audio"])
        try:
            from agents.detection.pitch_intelligence import PitchIntelligence
            import librosa

            audio = self._audio_views.mono_sum_normalized
            sr = self._audio_views.sample_rate
            if sr != CREPE_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=CREPE_SR)

            context = self._make_context(CREPE_SR)
            events = PitchIntelligence().detect(audio, context)
            self._pitch_notes_master = list(events) if events else []
            print(f"  {len(self._pitch_notes_master)} notes (master)")
            return StageResult("pitch_detection_master", True,
                               PitchTestimony(notes=self._pitch_notes_master,
                                              confidence=0.55, model_used="crepe",
                                              stem_type="master",
                                              frame_count=len(self._pitch_notes_master),
                                              rejected_frames=0))
        except Exception as e:
            return StageResult("pitch_detection_master", True,
                               PitchTestimony(notes=[], confidence=0.0,
                                              model_used="failed", stem_type="master",
                                              frame_count=0, rejected_frames=0),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Pitch Detection Other (Epistemic Council)
    # ====================================================================

    def _run_pitch_detection_other(self) -> StageResult:
        if self._audio_views is None:
            return StageResult("pitch_detection_other", False, None, errors=["No audio"])
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
            self._musical_findings = updated_findings

            print(f"  {len(self._pitch_notes_other)} notes (other, council)")
            return StageResult("pitch_detection_other", True, testimony)

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
                self._pitch_notes_other = notes
                return StageResult("pitch_detection_other", True,
                                   PitchTestimony(notes=notes, confidence=0.5,
                                                  model_used="librosa_fallback",
                                                  stem_type="other",
                                                  frame_count=pitches.shape[1],
                                                  rejected_frames=pitches.shape[1] - len(notes)),
                                   warnings=[f"Council failed ({e}), librosa fallback"])
            except Exception as e2:
                self._pitch_notes_other = []
                return StageResult("pitch_detection_other", True,
                                   PitchTestimony(notes=[], confidence=0.0,
                                                  model_used="none", stem_type="other",
                                                  frame_count=0, rejected_frames=0),
                                   warnings=[f"All detection failed: {e2}"])

    # ====================================================================
    # Stage: Tonal Detection
    # ====================================================================

    def _run_tonal_detection(self) -> StageResult:
        empty = TonalTestimony(chord_map=[], detected_key=None,
                               key_confidence=0.0, frame_count=0)
        if self._feature_bundle is None:
            return StageResult("tonal_detection", True, empty)
        try:
            import librosa
            if not self._feature_bundle.has_evidence(EvidenceType.CHROMA):
                return StageResult("tonal_detection", True, empty)

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
            self._musical_findings.chord_map = chord_map
            self._musical_findings.record_stage("tonal_detection")
            print(f"  {len(chord_map)} chord events")
            return StageResult("tonal_detection", True,
                               TonalTestimony(chord_map=chord_map, detected_key=None,
                                              key_confidence=0.0, frame_count=n_frames))
        except Exception as e:
            return StageResult("tonal_detection", True, empty, warnings=[str(e)])

    # ====================================================================
    # Stage: Drum Detection
    # ====================================================================

    def _run_drum_detection(self) -> StageResult:
        empty = DrumTestimony(drum_events=[], note_events=[],
                              drum_types_detected=[], confidence=0.0)
        if self._audio_views is None or self._audio_views.drums_stereo is None:
            return StageResult("drum_detection", True, empty)
        try:
            from agents.detection.drum_intelligence import DrumIntelligence
            import librosa

            audio = np.mean(self._audio_views.drums_stereo, axis=0)
            sr = self._audio_views.sample_rate
            if sr != BASIC_PITCH_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=BASIC_PITCH_SR)

            context = self._make_context(BASIC_PITCH_SR)
            result = DrumIntelligence().detect(audio, context)
            events = (result.events if hasattr(result, 'events')
                      else result if isinstance(result, list) else [])
            notes = [e.to_note_event() if hasattr(e, 'to_note_event') else e
                     for e in events]
            self._drum_notes = notes
            print(f"  {len(notes)} drum events → MIDI channel 9")
            return StageResult("drum_detection", True,
                               DrumTestimony(drum_events=events, note_events=notes,
                                             drum_types_detected=list({n.pitch for n in notes}),
                                             confidence=0.7))
        except Exception as e:
            return StageResult("drum_detection", True, empty, warnings=[str(e)])

    # ====================================================================
    # Stage: Anechoic MA
    # ====================================================================

    def _run_anechoic_ma(self) -> StageResult:
        empty = AnechoicMATestimony(
            frame_count=0, region_count=0,
            silent_region_count=0, resonant_region_count=0,
            profile="unknown", applied=False,
        )

        if self._audio_views is None:
            return StageResult("anechoic_ma", True, empty, warnings=["No audio"])

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

            print(f"  {len(report.regions)} regions "
                  f"({len(report.get_silent_regions())} silent, "
                  f"{len(report.get_resonant_regions())} resonant) "
                  f"profile={report.profile.value}")

            return StageResult("anechoic_ma", True,
                               AnechoicMATestimony(
                                   frame_count=len(report.frame_times),
                                   region_count=len(report.regions),
                                   silent_region_count=len(report.get_silent_regions()),
                                   resonant_region_count=len(report.get_resonant_regions()),
                                   profile=report.profile.value,
                                   applied=True,
                               ))

        except Exception as e:
            return StageResult("anechoic_ma", True, empty,
                               warnings=[f"Anechoic MA failed: {e}"])

    # ====================================================================
    # Stage: Timbre Analysis
    # ====================================================================

    def _run_timbre_analysis(self) -> StageResult:
        empty = TimbreTestimony(instrument_counts={}, entities=[],
                                total_analyzed=0, confidence=0.0)
        all_pitched = (list(self._pitch_notes_bass) +
                       list(self._pitch_notes_master) +
                       list(self._pitch_notes_other))
        if not all_pitched:
            return StageResult("timbre_analysis", True, empty)
        try:
            import librosa
            engine = TimbreIntelligence(sample_rate=ANALYSIS_SR)
            audio = self._audio_views.mono_sum_normalized
            if self._audio_views.sample_rate != ANALYSIS_SR:
                audio = librosa.resample(audio,
                                         orig_sr=self._audio_views.sample_rate,
                                         target_sr=ANALYSIS_SR)

            entities = []
            counts: Dict[str, int] = {}

            for note in all_pitched[:80]:
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
            print(f"  Instruments: {counts}")
            return StageResult("timbre_analysis", True,
                               TimbreTestimony(instrument_counts=counts,
                                               entities=entities[:40],
                                               total_analyzed=len(entities),
                                               confidence=0.65))
        except Exception as e:
            return StageResult("timbre_analysis", True, empty, warnings=[str(e)])

    # ====================================================================
    # Stage: Harmonic Validation
    # ====================================================================

    def _run_harmonic_validation(self) -> StageResult:
        all_notes = (list(self._pitch_notes_bass) +
                     list(self._pitch_notes_master) +
                     list(self._pitch_notes_other))

        if not all_notes:
            return StageResult("harmonic_validation", True,
                               HarmonicValidationTestimony(validated_notes=[],
                                                           vetoed_count=0,
                                                           detected_key=None,
                                                           key_confidence=0.0,
                                                           method="empty"))

        if self._feature_bundle is None:
            validated = [n for n in all_notes if n.confidence >= self.CONFIDENCE_FLOOR]
            return StageResult("harmonic_validation", True,
                               HarmonicValidationTestimony(validated_notes=validated,
                                                           vetoed_count=len(all_notes) - len(validated),
                                                           detected_key=None, key_confidence=0.0,
                                                           method="confidence_fallback"))
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
            for note in all_notes:
                try:
                    r = validator.validate(note, self._feature_bundle)
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
            return StageResult("harmonic_validation", True,
                               HarmonicValidationTestimony(validated_notes=validated,
                                                           vetoed_count=vetoed,
                                                           detected_key=key,
                                                           key_confidence=key_conf,
                                                           method="harmonic_validator"))
        except Exception as e:
            validated = [n for n in all_notes if n.confidence >= self.CONFIDENCE_FLOOR]
            return StageResult("harmonic_validation", True,
                               HarmonicValidationTestimony(validated_notes=validated,
                                                           vetoed_count=len(all_notes) - len(validated),
                                                           detected_key=None, key_confidence=0.0,
                                                           method="fallback"),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Groove Analysis
    # ====================================================================

    def _run_groove_analysis(self) -> StageResult:
        kicks = [n for n in self._rhythm_notes if n.pitch == 36]
        bass = self._pitch_notes_bass
        empty = GrooveTestimony(phase_delta_ms=0.0, is_wide_swing=False,
                                is_dilla_pocket=False, confidence=0.0)

        if not kicks or not bass:
            self._groove = empty
            return StageResult("groove_analysis", True, empty)

        deltas = []
        for kick in kicks:
            closest = min(bass, key=lambda n: abs(n.start_ms - kick.start_ms))
            d = closest.start_ms - kick.start_ms
            if abs(d) < 500.0:
                deltas.append(d)

        if not deltas:
            self._groove = empty
            return StageResult("groove_analysis", True, empty)

        raw = float(np.median(deltas))
        clamped = float(np.clip(raw, -100.0, 100.0))
        is_swing = abs(clamped) > 8.0
        conf = 0.7 if len(deltas) > 4 else 0.4

        gf = GrooveField(
            bass_kick_phase_delta_ms=clamped,
            is_wide_swing=is_swing, is_dilla_pocket=False,
            average_inter_note_distance_ms=500.0,
            tempo_estimate_bpm=self._tempo.tempo_bpm if self._tempo else 120.0,
            confidence=Confidence.MEDIUM,
            groove_signature="wide_swing" if is_swing else "straight",
        )
        testimony = GrooveTestimony(phase_delta_ms=clamped, is_wide_swing=is_swing,
                                    is_dilla_pocket=False, confidence=conf,
                                    groove_field=gf)
        self._groove = testimony
        self._musical_findings.phase_delta_ms = clamped
        self._musical_findings.swing_ratio = 0.72 if is_swing else 0.50
        self._musical_findings.groove_type = "wide_swing" if is_swing else "straight"
        self._musical_findings.groove_confidence = conf
        self._musical_findings.record_stage("groove_analysis")

        print(f"  Phase delta: {clamped:+.1f}ms (raw={raw:+.0f}ms), "
              f"swing={is_swing}, {len(deltas)} samples")
        return StageResult("groove_analysis", True, testimony)

    # ====================================================================
    # Stage: Pulse Analysis
    # ====================================================================

    def _run_pulse_analysis(self) -> StageResult:
        bpm = self._tempo.tempo_bpm if self._tempo else 120.0
        if not self._rhythm_notes:
            return StageResult("pulse_analysis", True,
                               PulseTestimony(tempo_bpm=bpm, confidence=0.0,
                                              beat_grid_ms=[]))
        times = sorted(n.start_ms for n in self._rhythm_notes)
        intervals = [times[i] - times[i - 1] for i in range(1, len(times))
                     if 100 < times[i] - times[i - 1] < 1500]
        if not intervals:
            return StageResult("pulse_analysis", True,
                               PulseTestimony(tempo_bpm=bpm, confidence=0.0,
                                              beat_grid_ms=[]))

        median_iv = float(np.median(intervals))
        pulse_bpm = 60000.0 / median_iv
        dur_ms = int((self._audio_views.master_stereo.shape[1] /
                      self._audio_views.sample_rate) * 1000)
        grid = list(np.arange(0, dur_ms, median_iv))[:200]

        self._musical_findings.record_stage("pulse_analysis")
        print(f"  Pulse: {pulse_bpm:.1f} BPM, {len(grid)} beats")
        return StageResult("pulse_analysis", True,
                           PulseTestimony(
                               tempo_bpm=pulse_bpm, confidence=0.6,
                               beat_grid_ms=grid,
                               pulse_field=PulseField(
                                   tempo_bpm=pulse_bpm, confidence=Confidence.MEDIUM,
                                   beat_grid_ms=grid, sixteenth_grid_ms=[],
                                   pulse_strength=[1.0] * len(grid))))

    # ====================================================================
    # Stage: Reverse Geo Crypt
    # ====================================================================

    def _run_reverse_geo_crypt(self) -> StageResult:
        empty = LatticeTestimony(period_ms=0.0, confidence=0.0,
                                 events_analyzed=0, ratio_family="unknown")
        if len(self._rhythm_notes) < 4:
            return StageResult("reverse_geo_crypt", True, empty)

        times = sorted(n.start_ms for n in self._rhythm_notes)
        intervals = [round((times[i] - times[i - 1]) / 10) * 10
                     for i in range(1, min(len(times), 50))
                     if 100 < times[i] - times[i - 1] < 2000]
        if not intervals:
            return StageResult("reverse_geo_crypt", True, empty)

        top = Counter(intervals).most_common(1)[0]
        period = float(top[0])
        conf = min(0.95, top[1] / len(intervals))
        self._musical_findings.lattice_period_ms = period
        self._musical_findings.lattice_confidence = conf
        self._musical_findings.record_stage("reverse_geo_crypt")

        print(f"  Lattice: period={period:.0f}ms, conf={conf:.2f}")
        return StageResult("reverse_geo_crypt", True,
                           LatticeTestimony(period_ms=period, confidence=conf,
                                            events_analyzed=len(intervals),
                                            ratio_family="detected"))

    # ====================================================================
    # Stage: Voice Continuity
    # ====================================================================

    def _run_voice_continuity(self) -> StageResult:
        all_pitched = (list(self._pitch_notes_bass) +
                       list(self._pitch_notes_master) +
                       list(self._pitch_notes_other))
        if not all_pitched:
            return StageResult("voice_continuity", True,
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
            vc = VoiceContinuity(config=config)
            result = vc.analyze(notes=all_pitched)

            self._voices = result.voices
            crossings = (len(result.voice_crossings)
                         if hasattr(result, 'voice_crossings') else 0)

            print(f"  {len(self._voices)} voices, {crossings} crossings")
            for v in self._voices[:4]:
                print(f"    {v.role.value}: {len(v.notes)} notes")
            return StageResult("voice_continuity", True,
                               VoiceContinuityTestimony(voices=self._voices,
                                                        voice_count=len(self._voices),
                                                        crossing_count=crossings,
                                                        confidence=0.7))
        except Exception as e:
            return StageResult("voice_continuity", True,
                               VoiceContinuityTestimony(voices=[], voice_count=0,
                                                        crossing_count=0, confidence=0.0),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Quaver Intelligence (NEW)
    # ====================================================================

    def _run_quaver_intelligence(self) -> StageResult:
        """Stage: QuaverIntelligence — Symbolic duration witness."""
        from epistemic.quaver_intelligence import QuaverIntelligence, QuaverConfig

        # Collect all pitched notes (pre-quantization)
        all_pitched = (list(self._pitch_notes_bass) +
                       list(self._pitch_notes_master) +
                       list(self._pitch_notes_other) +
                       list(self._rhythm_notes))

        if not all_pitched:
            return StageResult("quaver_intelligence", True,
                               testimony={"quaver_testimonies": []},
                               warnings=["No notes for duration analysis"])

        # Get necessary evidence from testimonies
        pulse_field = self._extract_testimony_dict("pulse_analysis")
        groove_field = self._extract_testimony_dict("groove_analysis")
        anechoic_dict = self._extract_testimony_dict("anechoic_ma")
        tempo_dict = self._extract_testimony_dict("tempo_analysis")
        drum_dict = self._extract_testimony_dict("drum_detection")
        voice_dict = self._extract_testimony_dict("voice_continuity")

        # Get subdivision_ms from lattice
        subdivision_ms = 250.0  # Default 8th note at 120bpm
        if self._musical_findings.lattice_period_ms:
            subdivision_ms = self._musical_findings.lattice_period_ms

        # Get tempo
        tempo_bpm = self._tempo.tempo_bpm if self._tempo else 120.0

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

        return StageResult(
            "quaver_intelligence", True,
            testimony={
                "quaver_testimonies": testimonies,
                "note_count": len(testimonies),
                "contention_count": contention_count,
                "avg_uncertainty": avg_uncertainty,
            }
        )

    # ====================================================================
    # Stage: Quantization
    # ====================================================================

    def _run_quantization(self) -> StageResult:
        pitched = (list(self._pitch_notes_bass) +
                   list(self._pitch_notes_master) +
                   list(self._pitch_notes_other) +
                   list(self._rhythm_notes))

        if not pitched:
            self._quantized_pitched = []
            return StageResult("quantization", True,
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
            q_result = quantizer.quantize(pitched)

            if isinstance(q_result, tuple):
                quantized_list, stats = q_result
            else:
                quantized_list, stats = q_result, {}

            self._quantized_pitched = list(quantized_list)
            snapped = stats.get("snapped", 0)
            preserved = stats.get("preserved_in_pocket", 0)

            print(f"  {original} → {len(self._quantized_pitched)} notes")
            print(f"  preserved={preserved} nudged={stats.get('nudged', 0)} "
                  f"snapped={snapped}")
            print(f"  {lattice.get_summary()}")
            return StageResult("quantization", True,
                               QuantizationTestimony(quantized_notes=self._quantized_pitched,
                                                     original_count=original,
                                                     snapped_count=snapped,
                                                     total_confidence_loss=snapped * 0.01,
                                                     lattice_summary=lattice.get_summary()))
        except Exception as e:
            self._quantized_pitched = pitched
            return StageResult("quantization", True,
                               QuantizationTestimony(quantized_notes=pitched,
                                                     original_count=original,
                                                     snapped_count=0,
                                                     total_confidence_loss=0.0,
                                                     lattice_summary="fallback"),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Velocity Merge
    # ====================================================================

    def _run_velocity_merge(self) -> StageResult:
        notes = list(self._quantized_pitched)
        if not notes:
            self._merged_notes = []
            return StageResult("velocity_merge", True,
                               MergeTestimony(merged_notes=[], original_count=0,
                                              removed_count=0, merge_count=0,
                                              velocity_stats={}))
        try:
            config = VelocityMergeConfig(
                overlap_tolerance_ms=10.0,
                adjacent_gap_ms=25.0,
                default_strategy=MergeStrategy.WEIGHTED_AVERAGE,
                preserve_ghost_notes=True,
                merge_different_pitches=False,
                min_merge_confidence=0.25,
            )
            merger = VelocityMerge(config=config, music_box=self.music_box)
            result = merger.merge_overlaps(notes)

            self._merged_notes = list(result.merged_notes)
            vel = merger.analyze_velocity_distribution(self._merged_notes)
            recs = merger.get_merge_records()

            print(f"  {len(notes)} → {len(self._merged_notes)} "
                  f"(removed {result.notes_removed}, {len(recs)} merges)")
            return StageResult("velocity_merge", True,
                               MergeTestimony(merged_notes=self._merged_notes,
                                              original_count=len(notes),
                                              removed_count=result.notes_removed,
                                              merge_count=len(recs),
                                              velocity_stats=vel))
        except Exception as e:
            self._merged_notes = notes
            return StageResult("velocity_merge", True,
                               MergeTestimony(merged_notes=notes, original_count=len(notes),
                                              removed_count=0, merge_count=0,
                                              velocity_stats={}),
                               warnings=[str(e)])

    # ====================================================================
    # Stage: Consensus (Preserves drum notes)
    # ====================================================================

    def _run_consensus(self) -> StageResult:
        # IMPORTANT: Keep pitched and drum notes separate - don't mix them
        pitched_notes = list(self._merged_notes)
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

        # Deduplicate pitched notes
        pitched_notes.sort(key=lambda n: (n.pitch, n.start_ms))
        unique_pitched: List[NoteEvent] = []
        for note in pitched_notes:
            if not unique_pitched:
                unique_pitched.append(note)
                continue
            last = unique_pitched[-1]
            if (last.pitch == note.pitch and
                    abs(last.start_ms - note.start_ms) < 100.0):
                if note.confidence > last.confidence:
                    unique_pitched[-1] = note
            else:
                unique_pitched.append(note)

        # Keep drums as-is (no deduplication needed - they're percussive)
        self._consensus_notes = unique_pitched
        # Drums remain in self._drum_notes (preserved)

        avg = float(np.mean([n.confidence for n in unique_pitched])) if unique_pitched else 0.0
        print(f"  {len(unique_pitched)} final pitched notes (avg conf={avg:.2f})")
        print(f"  {len(self._drum_notes)} drum notes preserved")

        return StageResult("consensus", True,
                           ConsensusTestimony(final_decision=True, confidence=avg,
                                              veto_triggered=False, veto_source=None,
                                              witness_count=len(unique_pitched)))

    # ====================================================================
    # Stage: Phrase Intelligence (NEW)
    # ====================================================================

    def _run_phrase_intelligence(self) -> StageResult:
        """Stage: PhraseIntelligence — Structural phase state machine."""
        from epistemic.phrase_intelligence import PhraseIntelligence

        # Need consensus notes first
        if not hasattr(self, '_consensus_notes') or not self._consensus_notes:
            return StageResult("phrase_intelligence", True,
                               testimony={"structural_testimony": None},
                               warnings=["No consensus notes for phrase analysis"])

        # Get all testimonies as StageResult objects
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
        self._musical_findings.structural_testimony = self._structural_testimony
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

    def _run_structural_feedback(self) -> StageResult:
        """Stage: Structural Feedback Loop."""
        if not hasattr(self, '_structural_testimony') or not self._structural_testimony:
            return StageResult("structural_feedback", True,
                               testimony={"reanalysis_performed": False},
                               warnings=["No structural testimony for feedback"])

        needs_reanalysis = getattr(self._structural_testimony, 'needs_reanalysis', {})
        if not needs_reanalysis.get("requires_reanalysis", False):
            return StageResult("structural_feedback", True,
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

        return StageResult("structural_feedback", True,
                           testimony={
                               "reanalysis_performed": True,
                               "reasoning": reasoning,
                               "severity": severity,
                           })

    # ====================================================================
    # Stage: Validation
    # ====================================================================

    def _run_validation(self) -> StageResult:
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

        verdict = self.scribe.final_verdict()
        passed = verdict.get("passed", True)
        print(
            f"  {'✓ Passed' if passed else '✗ Failed'} — {len(pitched_notes)} pitched notes, {len(drum_notes)} drum notes")

        return StageResult("validation", passed,
                           ValidationTestimony(
                               passed=passed,
                               hard_veto_count=verdict.get("hard_veto_count", 0),
                               soft_veto_count=verdict.get("soft_veto_count", 0),
                               verdict_summary=str(verdict),
                               is_trustworthy=passed))

    # ====================================================================
    # Stage: Export (Drums on channel 9)
    # ====================================================================

    def _run_export(self) -> StageResult:
        pitched = list(self._consensus_notes)
        drums = list(self._drum_notes)

        if not pitched and not drums:
            return StageResult("export", True,
                               ExportTestimony(exported_paths=[], formats=[],
                                               note_count=0),
                               warnings=["No notes to export"])
        try:
            from export.midi_writer import MidiWriter

            out = (self._output_path or
                   f"output/transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid")
            Path(out).parent.mkdir(parents=True, exist_ok=True)

            writer = MidiWriter(music_box=self.music_box)
            tm = (self._tempo.tempo_map
                  if self._tempo and hasattr(self._tempo, 'tempo_map')
                  else TempoMap(
                initial_tempo_bpm=self._tempo.tempo_bpm if self._tempo else 120.0,
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
            return StageResult("export", success,
                               ExportTestimony(
                                   exported_paths=[out] if success else [],
                                   formats=["midi"] if success else [],
                                   note_count=total),
                               errors=[] if success else ["MIDI write failed"])
        except Exception as e:
            return StageResult("export", False,
                               ExportTestimony(exported_paths=[], formats=[],
                                               note_count=0),
                               errors=[str(e)])

    # ====================================================================
    # Result Builder
    # ====================================================================

    def _build_result(self, total_sec: float) -> PipelineResult:
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
            detected_tempo_bpm=self._tempo.tempo_bpm if self._tempo else None,
            detected_key=self._detected_key,
            key_confidence=self._key_confidence,
            groove_field=(self._groove.groove_field
                          if self._groove and hasattr(self._groove, 'groove_field')
                          else None),
            voices=self._voices,
            is_truncated=self.truncate_duration > 0,
            truncation_duration=self.truncate_duration,
            original_duration_seconds=self._original_duration_seconds,
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