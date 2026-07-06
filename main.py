#!/usr/bin/env python3
# =================================================================
# MODULE: main.py
# DESCRIPTION: Top-level entry point for Grimlock 5.6.1
#
# VERSION: 5.6.1 (Updated for QuaverIntelligence + PhraseIntelligence)
# UPDATED: 2026-06-02
#
# CHANGES IN 5.6.1:
#   1. Fixed _export_results to use correct PipelineResult attributes
#   2. Removed references to non-existent attributes (master_hash, using_guided_tempo, etc.)
#   3. Added proper groove_field property access
#   4. Updated version strings
#   5. Added structural summary output for phrase intelligence results
# =================================================================

import argparse
import sys
import os
import json
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

# Force UTF-8 stdout/stderr so the many Unicode status characters used
# throughout this codebase (→, ✓, ✗, ⚠, █) don't crash on Windows
# consoles, which default to a legacy codepage like cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# NUMBA WARMUP
# ============================================================================

def warmup_numba():
    """Pre-warm numba JIT compilation before pipeline starts."""
    print("[WARMUP] Pre-warming librosa / numba (first run only)...", flush=True)
    os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.numba_cache'))

    try:
        import librosa
        import numpy as np

        test_audio = np.zeros(2048, dtype=np.float32)
        test_sr = 22050

        _ = librosa.feature.chroma_stft(y=test_audio, sr=test_sr)
        _ = librosa.cqt(y=test_audio, sr=test_sr)
        _ = librosa.feature.melspectrogram(y=test_audio, sr=test_sr)
        _ = librosa.onset.onset_detect(y=test_audio, sr=test_sr, backtrack=True)
        _ = librosa.beat.beat_track(y=test_audio, sr=test_sr)

        print("[WARMUP] Complete. Numba cache ready.", flush=True)

    except ImportError as e:
        print(f"[WARMUP] Warning: librosa not available: {e}", flush=True)
    except Exception as e:
        print(f"[WARMUP] Warning: {e}", flush=True)


warmup_numba()

# ============================================================================
# Core Imports (Using core/__init__.py exports)
# ============================================================================

# Core types (bedrock - stdlib only, no numpy)
from core import (
    # Order Types
    Confidence, SourceType, StemType, NoteEvent, ExportOptions,
    AudioContext, GrooveField, PulseField, TempoMap, Voice,
    VoiceContinuityResult, WitnessTestimony, ConsensusPackage,
    ValidationResult, VetoReason, ValidationGate, QuantizationStrategy,
    VoiceRole, AnechoicProfile, StageResult, SeparationResult,
    beat_ms_to_bpm, bpm_to_beat_ms, hz_to_midi, midi_to_hz,
    ms_to_samples, samples_to_ms, validate_pitch, validate_velocity,

    # Constants
    MEMORY_LIMIT_MB, MIN_TEMPO_BPM, MAX_TEMPO_BPM,
    DEFAULT_PROCESSING_DURATION_SECONDS, TRUNCATION_PRESETS,
    validate_truncation_duration, MIN_CONFIDENCE_TO_PASS,
    RITORNELLO_MAX_SNAP_MS, HEARTBEAT_NOTE_PITCH,
    STAGE_TIMEOUT_LOAD_SECONDS, STAGE_TIMEOUT_SEPARATION_SECONDS,
    STAGE_TIMEOUT_DETECTION_SECONDS, STAGE_TIMEOUT_ANALYSIS_SECONDS,
    STAGE_TIMEOUT_QUANTIZATION_SECONDS, STAGE_TIMEOUT_EXPORT_SECONDS,
    ANALYSIS_SR, BASIC_PITCH_SR, CREPE_SR, MADMOM_SR,
)

# Feature Bundle (spectral evidence)
from core import (
    FeatureBundle, EvidenceType, extract_features,
    create_feature_bundle, bundle_for_madmom, bundle_for_pitch,
    bundle_for_analysis, compute_audio_statistics, compute_feature_statistics,
    sanitize_audio, safe_normalize, validate_fft_size,
)

# Acoustic Intelligence (audio truth engine)
from core import (
    AcousticIntelligence, ImmutableAudio, ChannelLayout,
)

# Contracts (WHAT components must do)
from core import (
    AudioContract, DetectionResult, SeparationContractResult,
    AnalysisResult, validate_audio_contract, validate_detection_result,
    validate_stem_contract,
)

# Model Registry (model validation)
from core import (
    CanonicalRegistry, ModelSpecification, ModelDomain,
    get_model_summary, compare_models,
)

# Guided Control
from core import (
    GuidedParams, TimeSignature, GenreType, GuidedModeController,
    create_guided_params_from_form, get_scale_notes, validate_guided_params,
    KeyConstraint, GuidedBeatGrid, GuidedBeatGridGenerator,
)

# Evidence Tracker
from core import EvidenceTracker, EvidenceLeaseRecord

# Protocols (legacy interfaces - for agent compatibility)
from core import (
    AgentProtocol, MemoryManagedProtocol, ScribeValidatable,
    MusicBoxProtocol, StatusReporterProtocol,
)

# ============================================================================
# Orchestration Imports
# ============================================================================

from orchestration.pipeline import (
    GrimlockPipeline, PipelineResult, create_pipeline, transcribe_file,
)
from orchestration.music_box import MusicBox, LogLevel, create_music_box
from orchestration.scribe import Scribe, create_scribe

# ============================================================================
# Memory Management Imports
# ============================================================================

from memory.guardian import MemoryGuardian, create_memory_guardian
from memory.arena import MemoryArena, BufferID, BorrowScope, scoped_arena

# ============================================================================
# Agents
# ============================================================================

from agents.factory import AgentFactory, FactoryConfig

# ============================================================================
# Export Imports
# ============================================================================

from export.midi_writer import MidiWriter, export_to_midi
from export.json_writer import JsonWriter, export_to_json

# ============================================================================
# Ingestion Imports
# ============================================================================

from ingestion.loader import (
    AudioLoader, load_audio, get_audio_info, LoadedAudio,
)
from ingestion.downsampler import DownsamplingEngine, get_required_sample_rate


# ============================================================================
# Helper Functions
# ============================================================================

def get_preferred_sample_rate_for_genre(genre: Optional[str] = None) -> int:
    if not genre:
        return 44100
    genre_lower = genre.lower()
    high_fidelity = ["jazz", "classical", "acoustic", "orchestral", "vocal"]
    if any(g in genre_lower for g in high_fidelity):
        return 44100
    return 22050


def get_sample_rate_for_models(model_ids: List[str]) -> int:
    if not model_ids:
        return 44100
    rates = []
    for model_id in model_ids:
        if CanonicalRegistry.has_model(model_id):
            spec = CanonicalRegistry.get_spec(model_id)
            rates.append(spec.preferred_sr)
    if not rates:
        return 44100
    counter = Counter(rates)
    return counter.most_common(1)[0][0]


def format_duration(seconds: float) -> str:
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_truncation_duration(value: str) -> float:
    if value.lower() in TRUNCATION_PRESETS:
        return TRUNCATION_PRESETS[value.lower()]
    try:
        duration = float(value)
        return validate_truncation_duration(duration)
    except ValueError:
        raise ValueError(
            f"Invalid truncation duration: {value}. "
            f"Use numeric seconds or preset: {list(TRUNCATION_PRESETS.keys())}"
        )


def get_confidence_level(confidence: float) -> str:
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    elif confidence >= 0.25:
        return "LOW"
    else:
        return "HALLUCINATION"


def safe_get_groove(result: PipelineResult) -> Optional[Any]:
    """Safely get groove_field from PipelineResult."""
    if hasattr(result, 'groove_field') and result.groove_field:
        return result.groove_field
    if hasattr(result, 'groove') and result.groove:
        return result.groove
    return None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CLIConfig:
    input_paths: List[str] = field(default_factory=list)
    output_dir: str = "output"
    output_name: Optional[str] = None

    export_midi: bool = True
    export_json: bool = False
    json_pretty: bool = True
    separate_tracks: bool = False

    guided_tempo: Optional[float] = None
    guided_key: Optional[str] = None
    guided_time_signature: Optional[str] = None
    guided_genre: Optional[str] = None

    target_sample_rate: int = 44100
    force_mono: bool = True
    octave_restore: bool = True
    heartbeat_notes: bool = True
    truncate_duration: float = DEFAULT_PROCESSING_DURATION_SECONDS

    skip_separation: bool = False
    skip_quantization: bool = False
    fast_mode: bool = False

    music_box_path: Optional[str] = None
    verbose: bool = False
    quiet: bool = False
    debug: bool = False

    batch_recursive: bool = False
    batch_extensions: List[str] = field(
        default_factory=lambda: ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff']
    )

    config_file: Optional[str] = None
    device: str = "auto"
    max_memory_mb: int = MEMORY_LIMIT_MB
    max_workers: int = 4

    use_mel_roformer: bool = True
    use_demucs_fallback: bool = True
    mel_roformer_quality: str = "balanced"
    mel_roformer_stereo_mode: str = "mid_side"
    mel_roformer_chunked: bool = True

    skip_pre_scan: bool = False

    @property
    def has_guided_params(self) -> bool:
        return any([self.guided_tempo, self.guided_key, self.guided_time_signature, self.guided_genre])

    @property
    def is_truncated(self) -> bool:
        return self.truncate_duration > 0

    def to_guided_params(self) -> Optional[GuidedParams]:
        if not self.has_guided_params:
            return None
        return GuidedParams(
            tempo_bpm=self.guided_tempo,
            time_signature=self.guided_time_signature,
            key_signature=self.guided_key,
            genre=self.guided_genre,
            constant_tempo=True,
        )


# ============================================================================
# Progress Reporter
# ============================================================================

class ProgressReporter:
    def __init__(self, quiet: bool = False, verbose: bool = False):
        self.quiet = quiet
        self.verbose = verbose
        self._last_progress = 0
        self._current_stage = ""

    def __call__(self, progress: float, stage: str):
        if self.quiet:
            return
        if int(progress) != int(self._last_progress) or stage != self._current_stage:
            self._last_progress = progress
            self._current_stage = stage
            bar_length = 30
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r  [{bar}] {progress:.0f}% - {stage}", end='', flush=True)

    def finish(self):
        if not self.quiet:
            print()


# ============================================================================
# Grimlock CLI
# ============================================================================

class GrimlockCLI:
    def __init__(self):
        self.start_time: Optional[float] = None
        self.config = CLIConfig()
        self._current_file: Optional[str] = None
        self._progress_reporter: Optional[ProgressReporter] = None
        self._acoustic_profile: Optional[Dict[str, Any]] = None

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def run(self, args: Optional[List[str]] = None) -> int:
        self.start_time = time.time()

        try:
            parser = self._create_parser()
            parsed_args = parser.parse_args(args)
            self.config = self._load_config(parsed_args)

            if not self.config.input_paths:
                parser.print_help()
                return 1

            self._progress_reporter = ProgressReporter(
                quiet=self.config.quiet, verbose=self.config.verbose
            )

            if self.config.verbose and not self.config.quiet:
                self._print_config_summary()

            if len(self.config.input_paths) == 1 and not self.config.batch_recursive:
                success = self._process_single_file(self.config.input_paths[0])
                self._progress_reporter.finish()
                return 0 if success else 1
            else:
                results = self._process_batch(self.config.input_paths)
                success_count = sum(1 for r in results if r)
                total_count = len(results)

                self._progress_reporter.finish()
                print(f"\n{'=' * 60}")
                print(f"Batch processing complete: {success_count}/{total_count} successful")
                print(f"{'=' * 60}")

                if success_count == total_count:
                    return 0
                elif success_count > 0:
                    return 2
                else:
                    return 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            return 130
        except Exception as e:
            print(f"\nFatal error: {e}", file=sys.stderr)
            if self.config.verbose or self.config.debug:
                import traceback
                traceback.print_exc()
            return 1

    # ============================================================================
    # Model Registry Validation
    # ============================================================================

    def _validate_model_registry(self) -> bool:
        all_valid = True

        if self.config.use_mel_roformer:
            if not CanonicalRegistry.has_model("mel_roformer"):
                print(f"Warning: Mel-Roformer not registered", file=sys.stderr)
                all_valid = False

        if self.config.use_demucs_fallback:
            if not CanonicalRegistry.has_model("demucs"):
                print(f"Warning: Demucs not registered", file=sys.stderr)
                all_valid = False

        return all_valid

    # ============================================================================
    # Pre-Scan Analysis
    # ============================================================================

    def _pre_scan_audio(self, file_path: str, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        if not self.config.verbose or self.config.quiet or self.config.skip_pre_scan:
            return {}

        print("[PRE-SCAN] Analyzing audio characteristics...")
        results = {}

        # Simple audio validation
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            print(f"  ✗ Invalid audio: contains NaN or Inf")
            results['is_valid'] = False
            return results

        results['is_valid'] = True
        results['duration'] = len(audio) / sample_rate
        results['rms'] = float(np.sqrt(np.mean(audio ** 2)))
        results['peak'] = float(np.max(np.abs(audio)))
        results['silence_ratio'] = float(np.mean(audio ** 2 < 0.0001))

        print(f"  ✓ Audio: {results['duration']:.1f}s, {sample_rate}Hz")
        print(f"  RMS: {results['rms']:.4f}, Peak: {results['peak']:.2f}")
        print(f"  Silence: {results['silence_ratio']:.1%}")

        # FeatureBundle pre-scan (10 seconds only)
        try:
            import librosa
            duration = min(10.0, len(audio) / sample_rate)
            samples = int(duration * sample_rate)
            audio_segment = audio[:samples]

            bundle = extract_features(audio_segment, sample_rate, debug=False)

            if bundle.has_evidence(EvidenceType.CHROMA):
                chroma = bundle.chroma
                chroma_mean = np.mean(chroma, axis=1)
                key_idx = int(np.argmax(chroma_mean))
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                results['estimated_key'] = key_names[key_idx]
                print(f"  Pre-scan key estimate: {results['estimated_key']}")

            if bundle.has_evidence(EvidenceType.ONSET_STRENGTH):
                onset = bundle.onset_strength
                density = float(np.mean(onset > np.percentile(onset, 75)))
                results['density'] = density
                print(f"  Note density: {'High' if density > 0.3 else 'Low'} ({density:.2f})")

        except Exception as e:
            if self.config.debug:
                print(f"  Pre-scan warning: {e}")

        print("[PRE-SCAN] Complete")
        return results

    # ============================================================================
    # Simple Contract Validation
    # ============================================================================

    def _validate_transcription_contract(self, result: PipelineResult, audio_duration: float) -> Tuple[bool, List[str]]:
        issues = []

        if result.notes:
            out_of_range = [n for n in result.notes if not (21 <= n.pitch <= 108)]
            if out_of_range:
                issues.append(f"{len(out_of_range)} notes outside MIDI range (21-108)")

            zero_duration = [n for n in result.notes if n.end_ms <= n.start_ms]
            if zero_duration:
                issues.append(f"{len(zero_duration)} notes with zero or negative duration")

            low_conf = [n for n in result.notes if n.confidence < MIN_CONFIDENCE_TO_PASS]
            if low_conf:
                issues.append(f"{len(low_conf)} notes below confidence threshold ({MIN_CONFIDENCE_TO_PASS})")

        if result.detected_tempo_bpm:
            if result.detected_tempo_bpm < MIN_TEMPO_BPM or result.detected_tempo_bpm > MAX_TEMPO_BPM:
                issues.append(
                    f"Tempo {result.detected_tempo_bpm:.1f} BPM outside range ({MIN_TEMPO_BPM}-{MAX_TEMPO_BPM})")

        is_valid = len(issues) == 0

        if issues and self.config.verbose:
            print(f"  Contract validation: {'PASS' if is_valid else 'FAIL'}")
            for issue in issues[:3]:
                print(f"    • {issue}")

        return is_valid, issues

    # ============================================================================
    # Argument Parser
    # ============================================================================

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="grimlock",
            description="Grimlock 5.6.1 - Professional Audio Transcription with Structural Intelligence",
        )

        parser.add_argument("input", nargs="+", help="Input audio file(s)")
        parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
        parser.add_argument("--output-name", help="Custom output filename")

        guided_group = parser.add_argument_group("Guided Mode")
        guided_group.add_argument("--tempo", type=float, dest="guided_tempo", help="Override tempo detection")
        guided_group.add_argument("--key", type=str, dest="guided_key", help="Override key detection")
        guided_group.add_argument("--time-sig", type=str, dest="guided_time_signature", help="Override time signature")
        guided_group.add_argument("--genre", type=str, dest="guided_genre",
                                  choices=["jazz", "rock", "classical", "electronic", "pop", "hip_hop"])

        export_group = parser.add_argument_group("Export Options")
        export_group.add_argument("--midi", action="store_true", default=True, help="Export MIDI")
        export_group.add_argument("--no-midi", action="store_false", dest="midi", help="Disable MIDI")
        export_group.add_argument("--json", action="store_true", help="Export JSON")
        export_group.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
        export_group.add_argument("--separate-tracks", action="store_true", help="Separate MIDI tracks")

        proc_group = parser.add_argument_group("Processing Options")
        proc_group.add_argument("--sample-rate", type=int, default=44100, help="Target sample rate")
        proc_group.add_argument("--duration", type=float, default=DEFAULT_PROCESSING_DURATION_SECONDS,
                                help="Duration to process")
        proc_group.add_argument("--full-song", action="store_true", help="Process entire song")
        proc_group.add_argument("--no-mono", action="store_true", help="Disable mono downmixing")
        proc_group.add_argument("--no-octave-restore", action="store_true", help="Disable octave restoration")
        proc_group.add_argument("--no-heartbeat", action="store_true", help="Disable heartbeat note")
        proc_group.add_argument("--skip-pre-scan", action="store_true", help="Skip pre-scan analysis")

        pipe_group = parser.add_argument_group("Pipeline Options")
        pipe_group.add_argument("--fast", action="store_true", help="Fast mode")
        pipe_group.add_argument("--skip-separation", action="store_true", help="Skip source separation")
        pipe_group.add_argument("--skip-quantization", action="store_true", help="Skip quantization")

        batch_group = parser.add_argument_group("Batch Processing")
        batch_group.add_argument("--batch", action="store_true", help="Process recursively")
        batch_group.add_argument("--extensions", default=".wav,.mp3,.flac,.m4a,.ogg,.aiff", help="File extensions")

        fore_group = parser.add_argument_group("Forensic Options")
        fore_group.add_argument("--music-box", dest="music_box_path", help="MusicBox log path")
        fore_group.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
        fore_group.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
        fore_group.add_argument("--debug", action="store_true", help="Debug mode")

        adv_group = parser.add_argument_group("Advanced Options")
        adv_group.add_argument("--config", help="JSON config file")
        adv_group.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        adv_group.add_argument("--max-memory", type=int, default=MEMORY_LIMIT_MB)
        adv_group.add_argument("--max-workers", type=int, default=4)

        sep_group = parser.add_argument_group("Separation Options")
        sep_group.add_argument("--separator", choices=["mel_roformer", "demucs", "auto"], default="auto")
        sep_group.add_argument("--mel-quality", choices=["fast", "balanced", "high"], default="balanced")
        sep_group.add_argument("--mel-stereo", choices=["mid_side", "stereo", "mono_fallback"], default="mid_side")
        sep_group.add_argument("--no-mel-chunked", action="store_true", help="Disable chunked inference")
        sep_group.add_argument("--no-demucs-fallback", action="store_true", help="Disable Demucs fallback")

        parser.add_argument("--version", action="version", version="Grimlock 5.6.1")

        return parser

    # ============================================================================
    # Configuration Loading
    # ============================================================================

    def _load_config(self, args) -> CLIConfig:
        config = CLIConfig()

        if args.config:
            try:
                with open(args.config, 'r') as f:
                    file_config = json.load(f)
                    for key, value in file_config.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}", file=sys.stderr)

        config.input_paths = args.input
        config.output_dir = args.output_dir
        config.output_name = args.output_name

        config.guided_tempo = args.guided_tempo
        config.guided_key = args.guided_key
        config.guided_time_signature = args.guided_time_signature
        config.guided_genre = args.guided_genre

        config.export_midi = args.midi
        config.export_json = args.json
        config.json_pretty = args.pretty
        config.separate_tracks = args.separate_tracks

        config.target_sample_rate = args.sample_rate
        config.force_mono = not args.no_mono
        config.octave_restore = not args.no_octave_restore
        config.heartbeat_notes = not args.no_heartbeat
        config.skip_pre_scan = args.skip_pre_scan

        if args.full_song:
            config.truncate_duration = 0.0
        else:
            config.truncate_duration = args.duration

        config.skip_separation = args.skip_separation
        config.skip_quantization = args.skip_quantization
        config.fast_mode = args.fast

        if args.separator == "auto":
            config.use_mel_roformer = True
        elif args.separator == "mel_roformer":
            config.use_mel_roformer = True
        elif args.separator == "demucs":
            config.use_mel_roformer = False

        config.use_demucs_fallback = not args.no_demucs_fallback
        config.mel_roformer_quality = args.mel_quality
        config.mel_roformer_stereo_mode = args.mel_stereo
        config.mel_roformer_chunked = not args.no_mel_chunked

        if config.fast_mode:
            config.mel_roformer_quality = "fast"
            config.mel_roformer_chunked = True

        config.batch_recursive = args.batch
        if args.extensions:
            config.batch_extensions = [e.strip() for e in args.extensions.split(',')]

        config.music_box_path = args.music_box_path
        config.verbose = args.verbose
        config.quiet = args.quiet
        config.debug = args.debug

        config.config_file = args.config
        config.device = args.device
        config.max_memory_mb = args.max_memory
        config.max_workers = args.max_workers

        return config

    def _print_config_summary(self):
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"  Separator:      {'Mel-Roformer' if self.config.use_mel_roformer else 'Demucs'}")
        print(f"  Mel quality:    {self.config.mel_roformer_quality}")
        print(f"  Mel stereo:     {self.config.mel_roformer_stereo_mode}")
        print(f"  Mel chunked:    {self.config.mel_roformer_chunked}")
        print(f"  Demucs fallback: {self.config.use_demucs_fallback}")
        print(f"  Device:         {self.config.device}")
        print(f"  Max memory:     {self.config.max_memory_mb} MB")
        print(f"  Pre-scan:       {'Enabled' if not self.config.skip_pre_scan else 'Disabled'}")
        print("=" * 60)

    # ============================================================================
    # File Processing
    # ============================================================================

    def _process_single_file(self, file_path: str) -> bool:
        file_path = Path(file_path)
        self._current_file = str(file_path)

        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            return False

        if not self._validate_audio_file(file_path):
            return False

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = self.config.output_name or file_path.stem

        if not self.config.quiet:
            self._print_header(file_path)

        guided_params = self.config.to_guided_params()

        pre_scan_results = {}
        try:
            loader = AudioLoader(truncate_duration=self.config.truncate_duration)
            master = loader.load(str(file_path))

            if not self.config.skip_pre_scan:
                pre_scan_results = self._pre_scan_audio(str(file_path), master.audio, master.sample_rate)

                if not pre_scan_results.get('is_valid', True):
                    print(f"Error: Invalid audio file", file=sys.stderr)
                    return False

        except Exception as e:
            if self.config.debug:
                print(f"Pre-scan warning: {e}")
            pre_scan_results = {'is_valid': True}

        try:
            result = transcribe_file(
                str(file_path),
                str(output_dir / f"{base_name}.mid"),
                guided_params.tempo_bpm if guided_params else None,
                guided_params.key_signature if guided_params else None,
                guided_params.time_signature if guided_params else None,
                guided_params.genre if guided_params else None,
                self.config.debug,
                self.config.truncate_duration,
            )

            audio_duration = pre_scan_results.get('duration', 30.0)
            contract_valid, contract_issues = self._validate_transcription_contract(result, audio_duration)

            export_success = self._export_results(result, output_dir, base_name)
            midi_path = output_dir / f"{base_name}.mid"
            midi_success = midi_path.exists() if self.config.export_midi else True

            if not self.config.quiet:
                self._print_detailed_summary(result, export_success, pre_scan_results, contract_valid, midi_path)

            return result.success and export_success and midi_success

        except Exception as e:
            print(f"\nError processing {file_path.name}: {e}", file=sys.stderr)
            if self.config.verbose or self.config.debug:
                import traceback
                traceback.print_exc()
            return False

    def _process_batch(self, input_paths: List[str]) -> List[bool]:
        all_files = []

        for input_path in input_paths:
            path = Path(input_path)
            if path.is_file():
                if path.suffix.lower() in self.config.batch_extensions:
                    all_files.append(path)
            elif path.is_dir() and self.config.batch_recursive:
                for ext in self.config.batch_extensions:
                    all_files.extend(path.rglob(f"*{ext}"))
            elif path.is_dir():
                print(f"Warning: {path} is a directory. Use --batch to process recursively.", file=sys.stderr)

        if not all_files:
            print("No audio files found.", file=sys.stderr)
            return []

        print(f"\nFound {len(all_files)} audio files to process")
        print(f"{'=' * 60}\n")

        results = []
        for i, file_path in enumerate(all_files, 1):
            print(f"[{i}/{len(all_files)}] Processing: {file_path.name}")
            success = self._process_single_file(str(file_path))
            results.append(success)
            print()

        return results

    # ============================================================================
    # Export Methods (FIXED for PipelineResult 5.6.1)
    # ============================================================================

    def _export_results(
            self,
            result: 'PipelineResult',
            output_dir: 'Path',
            base_name: str,
    ) -> bool:
        """
        Export additional outputs beyond the MIDI.
        (MIDI itself is written inside the pipeline's export stage.)

        FIXED: Uses only attributes that exist on PipelineResult 5.6.1.
        """
        export_success = True

        # ---- JSON export (only if --json flag is set) ----
        if self.config.export_json:
            json_path = output_dir / f"{base_name}.json"
            try:
                from export.json_writer import JSONEncoder

                # Get groove field safely
                groove_field = getattr(result, 'groove_field', None)
                if groove_field is None:
                    groove_field = getattr(result, 'groove', None)

                # Get drum notes safely
                drum_notes = getattr(result, 'drum_notes', [])

                # Build audit data from the new PipelineResult shape.
                # Use getattr() with defaults for any missing attributes
                # so this never raises AttributeError.
                audit_data = {
                    "version": "5.6.1",
                    "timestamp": datetime.now().isoformat(),
                    "audio": {
                        "file": base_name,
                        "duration_seconds": result.total_time_seconds,
                        "is_truncated": getattr(result, 'is_truncated', False),
                        "truncation_duration": getattr(result, 'truncation_duration', 0),
                    },
                    "pipeline": {
                        "success": result.success,
                        "aborted": result.aborted,
                        "abort_reason": result.abort_reason,
                        "peak_memory_mb": result.peak_memory_mb,
                        "total_time_s": result.total_time_seconds,
                        "stages_executed": result.stages_executed,
                    },
                    "guided_mode": {
                        "active": bool(self.config.guided_tempo or self.config.guided_key),
                        "tempo_override": self.config.guided_tempo is not None,
                        "key_override": self.config.guided_key is not None,
                    },
                    "detection": {
                        "tempo_bpm": result.detected_tempo_bpm,
                        "key": result.detected_key,
                        "key_confidence": getattr(result, 'key_confidence', 0.0),
                        "total_notes": len(result.notes),
                        "drum_notes": len(drum_notes),
                        "voices": len(getattr(result, 'voices', [])),
                    },
                    "groove": {
                        "groove_type": (groove_field.groove_signature
                                        if groove_field and hasattr(groove_field, 'groove_signature') else None),
                        "phase_delta_ms": (groove_field.bass_kick_phase_delta_ms
                                           if groove_field and hasattr(groove_field,
                                                                       'bass_kick_phase_delta_ms') else 0.0),
                        "is_swing": (groove_field.is_wide_swing
                                     if groove_field and hasattr(groove_field, 'is_wide_swing') else False),
                    },
                    "structural": {
                        "attractor_basin": getattr(result, 'structural_attractor', 'unknown'),
                        "phrase_count": getattr(result, 'phrase_count', 0),
                        "has_anacrusis": getattr(result, 'has_anacrusis', False),
                    },
                    "timing": result.timing,
                    "music_box_logs": getattr(result, 'music_box_logs', []),
                    "veto_history": getattr(result, 'veto_history', []),
                    "notes": [
                        {
                            "pitch": n.pitch,
                            "start_ms": n.start_ms,
                            "end_ms": n.end_ms,
                            "duration_ms": n.duration_ms(),
                            "velocity": n.velocity,
                            "confidence": n.confidence,
                            "source": n.source.value if hasattr(n.source, 'value') else str(n.source),
                            "snapped": n.is_snapped() if hasattr(n, 'is_snapped') else False,
                            "phrase_id": n.phrase_id,
                        }
                        for n in result.notes[:5000]  # Limit for performance
                    ],
                }

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(audit_data, f, indent=2, cls=JSONEncoder)

                if not self.config.quiet:
                    print(f"  ✓ JSON exported: {json_path}")

            except Exception as e:
                print(f"  [WARN] JSON export error: {e}")
                if self.config.debug:
                    import traceback
                    traceback.print_exc()
                export_success = False

        return export_success

    # ============================================================================
    # UI Methods
    # ============================================================================

    def _validate_audio_file(self, file_path: Path) -> bool:
        try:
            info = get_audio_info(file_path)

            if not info.is_supported:
                print(f"Error: Unsupported format: {file_path.suffix}", file=sys.stderr)
                return False

            if info.duration_seconds > 3600:
                print(f"Warning: File duration > 1 hour. This may take a while.", file=sys.stderr)

            if info.file_size_bytes / (1024 * 1024) > 500:
                print(f"Warning: File size > 500MB. This may take a while.", file=sys.stderr)

            return True

        except Exception as e:
            print(f"Error validating audio file: {e}", file=sys.stderr)
            return False

    def _print_header(self, file_path: Path):
        print()
        print(f"{'=' * 60}")
        print(f"GRIMLOCK 5.6.1 - Transcription with Structural Intelligence")
        print(f"{'=' * 60}")
        print(f"File:     {file_path.name}")
        print(f"Output:   {self.config.output_dir}")
        print(f"MIDI:     {'Yes' if self.config.export_midi else 'No'}")
        print(f"JSON:     {'Yes' if self.config.export_json else 'No'}")
        print(f"Mono:     {'Yes' if self.config.force_mono else 'No'}")
        print(f"Sample rate: {self.config.target_sample_rate} Hz")
        print(f"Separator: {'Mel-Roformer' if self.config.use_mel_roformer else 'Demucs'}")

        if self.config.mel_roformer_quality and self.config.use_mel_roformer:
            print(f"Mel quality: {self.config.mel_roformer_quality}")

        if self.config.truncate_duration > 0:
            print(f"Duration: First {self.config.truncate_duration:.0f} seconds")
        else:
            print(f"Duration: Full song")

        if self.config.debug:
            print(f"Debug:    Yes")

        if self.config.guided_tempo:
            print(f"Guided tempo: {self.config.guided_tempo} BPM")
        if self.config.guided_key:
            print(f"Guided key: {self.config.guided_key}")
        if self.config.guided_time_signature:
            print(f"Guided time signature: {self.config.guided_time_signature}")

        print(f"{'=' * 60}")
        print(f"Processing...")

    def _print_detailed_summary(self, result: PipelineResult, export_success: bool,
                                pre_scan_results: Dict[str, Any] = None,
                                contract_valid: bool = True,
                                midi_path: Optional[Path] = None):
        elapsed = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 70)
        print("TRANSCRIPTION SUMMARY")
        print("=" * 70)

        if result.success and export_success:
            print("✓ STATUS: SUCCESS")
        elif result.success and not export_success:
            print("⚠ STATUS: PARTIAL (transcription OK, export failed)")
        else:
            print("✗ STATUS: FAILED")

        if pre_scan_results and pre_scan_results.get('is_valid'):
            print(f"\n🔍 PRE-SCAN ANALYSIS")
            if pre_scan_results.get('estimated_key'):
                print(f"   Pre-scan key: {pre_scan_results['estimated_key']}")
            if pre_scan_results.get('density'):
                density_text = "High" if pre_scan_results['density'] > 0.3 else "Low"
                print(f"   Note density: {density_text}")

        separator_used = getattr(result, 'separator_used', 'mel_roformer')
        print(f"\n🔊 SEPARATION")
        print(f"   Separator: {separator_used.upper()}")

        print(f"\n📊 STATISTICS")
        print(f"   Duration: {format_duration(result.original_duration_seconds)}")
        print(f"   Processing Time: {format_duration(result.total_time_seconds)}")
        print(f"   Peak Memory: {result.peak_memory_mb:.1f} MB")
        print(f"   Total Pitched: {len(result.notes)}")
        print(f"   Total Drums: {len(getattr(result, 'drum_notes', []))}")

        if result.notes:
            rhythm_count = len([n for n in result.notes if n.source == SourceType.RHYTHM])
            pitch_count = len([n for n in result.notes if n.source == SourceType.PITCH])
            other_count = len([n for n in result.notes if n.source not in [SourceType.RHYTHM, SourceType.PITCH]])

            print(f"\n🎵 DETECTION BREAKDOWN")
            print(f"   Rhythm Events:  {rhythm_count}")
            print(f"   Pitch Events:   {pitch_count}")
            print(f"   Other Events:   {other_count}")

            avg_conf = sum(n.confidence for n in result.notes) / len(result.notes)
            high_conf = sum(1 for n in result.notes if n.confidence >= 0.8)
            med_conf = sum(1 for n in result.notes if 0.5 <= n.confidence < 0.8)
            low_conf = sum(1 for n in result.notes if n.confidence < 0.5)

            print(f"\n🎯 CONFIDENCE")
            print(f"   Average: {avg_conf:.2f}")
            print(f"   High (≥0.8):   {high_conf} notes")
            print(f"   Medium (0.5-0.8): {med_conf} notes")
            print(f"   Low (<0.5):    {low_conf} notes")

        if result.detected_tempo_bpm:
            print(f"\n🎚️ TEMPO & GROOVE")
            print(f"   Tempo: {result.detected_tempo_bpm:.1f} BPM")

            groove = safe_get_groove(result)
            if groove:
                groove_sig = getattr(groove, 'groove_signature', 'detected')
                is_wide = getattr(groove, 'is_wide_swing', False)
                is_dilla = getattr(groove, 'is_dilla_pocket', False)
                phase_delta = getattr(groove, 'bass_kick_phase_delta_ms', None)

                print(f"   Groove: {groove_sig if groove_sig else 'detected'}")
                # Used to collapse every non-WIDE_SWING result (including
                # DILLA_POCKET, LAID_BACK, FLUID - all real swing-adjacent
                # feels GrooveFieldAnalyzer already distinguishes) down to
                # the single word "Straight", hiding real detected feel.
                if is_wide:
                    swing_label = "Wide Swing"
                elif is_dilla:
                    swing_label = "Dilla Pocket"
                else:
                    swing_label = groove_sig.replace("_", " ").title() if groove_sig else "Straight"
                print(f"   Swing Type: {swing_label}")
                if phase_delta:
                    print(f"   Phase Delta: {phase_delta:.1f}ms")

        if result.detected_key:
            print(f"\n🎼 KEY DETECTION")
            print(f"   Detected Key: {result.detected_key} (confidence={getattr(result, 'key_confidence', 0.0):.2f})")

        # Phrase Intelligence Summary (NEW)
        structural_attractor = getattr(result, 'structural_attractor', None)
        if structural_attractor:
            print(f"\n🏗️ STRUCTURAL INTELLIGENCE")
            print(f"   Attractor Basin: {structural_attractor}")
            phrase_count = getattr(result, 'phrase_count', 0)
            if phrase_count:
                print(f"   Phrases Detected: {phrase_count}")
            if getattr(result, 'has_anacrusis', False):
                print(f"   Anacrusis (pickup): Yes")

        if not contract_valid:
            print(f"\n⚠️ CONTRACT ISSUES")
            print(f"   Some validation checks failed (see above)")

        if result.notes and hasattr(result.notes[0], 'is_snapped'):
            # is_snapped() is true for every note the quantizer ever ran on,
            # even when it preserved the raw onset exactly (in-pocket) - only
            # count real, musically meaningful displacement here.
            shift_deltas = [abs(n.snapped_start_ms - n.start_ms) for n in result.notes if n.is_snapped()]
            moved = sum(1 for d in shift_deltas if d >= 1.0)
            avg_shift = sum(shift_deltas) / len(shift_deltas) if shift_deltas else 0.0
            print(f"\n📐 QUANTIZATION")
            print(f"   Meaningfully shifted from raw onset: {moved}/{len(result.notes)} "
                  f"({moved / len(result.notes) * 100:.1f}%), avg shift={avg_shift:.1f}ms")

        voices = getattr(result, 'voices', [])
        if voices:
            print(f"\n🎤 VOICE SEPARATION")
            print(f"   Voices detected: {len(voices)}")
            for voice in voices[:3]:
                role_name = voice.role.value if hasattr(voice.role, 'value') else str(voice.role)
                print(f"   - {role_name}: {len(voice.notes)} notes, {voice.pitch_range_semitones:.0f} semitone range")

        # Scribe.final_verdict() returns hard_vetoes/soft_vetoes/advisories,
        # not a flat 'vetoes' key - this used to always read an empty
        # default regardless of what Scribe actually found, silently
        # showing "No vetoes" even when real ones existed.
        verdict = result.verdict or {}
        hard_vetoes = verdict.get('hard_vetoes', [])
        soft_vetoes = verdict.get('soft_vetoes', [])
        if hard_vetoes:
            print(f"\n❌ VALIDATION — UNTRUSTWORTHY ({len(hard_vetoes)} hard veto(s))")
            for v in hard_vetoes[:5]:
                print(f"   • {v.get('source', '?')}/{v.get('stage', '?')}: "
                      f"{v.get('reason', '?')} - {v.get('detail', '')}")
        elif soft_vetoes:
            print(f"\n⚠️ VALIDATION — trustworthy, with {len(soft_vetoes)} warning(s)")
            for v in soft_vetoes[:5]:
                print(f"   • {v.get('source', '?')}/{v.get('stage', '?')}: "
                      f"{v.get('reason', '?')} - {v.get('detail', '')}")
        else:
            print(f"\n✅ VALIDATION")
            print(f"   No vetoes - transcription is trustworthy")

        if getattr(result, 'is_truncated', False) and getattr(result, 'truncation_duration', 0) > 0:
            print(f"\n✂️ TRUNCATION")
            print(f"   Processed: first {result.truncation_duration:.0f} seconds only")

        if self.config.verbose and result.timing:
            print(f"\n⏱️ STAGE TIMING")
            sorted_timing = sorted(result.timing.items(), key=lambda x: x[1], reverse=True)
            for stage, duration_ms in sorted_timing[:10]:
                if duration_ms > 0:
                    print(f"   {stage:20s}: {duration_ms:8.1f} ms")

        print(f"\n💾 EXPORT")
        if self.config.export_midi:
            # MIDI export happens deep inside pipeline.transcribe()'s own
            # "export" stage, not in _export_results() - export_success
            # only reflects the JSON path below, so it can't tell us
            # whether the MIDI write actually happened. Check the file
            # itself instead of assuming success.
            midi_success = bool(midi_path) and midi_path.exists()
            if midi_success:
                print(f"   MIDI: ✓ Exported")
            else:
                print(f"   MIDI: ✗ Failed")
        if self.config.export_json:
            if export_success:
                print(f"   JSON: ✓ Exported")
            else:
                print(f"   JSON: ✗ Failed")

        print("=" * 70)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    cli = GrimlockCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()