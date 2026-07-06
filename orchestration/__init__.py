# =================================================================
# MODULE: orchestration/__init__.py
# DESCRIPTION: Orchestration module exports for Grimlock 5.4
#
# VERSION: 5.6.1
# UPDATED: 2026-05-21
#
# PHILOSOPHY:
#     The orchestrator is the only module allowed to touch every
#     piece of the puzzle. Agents never call the pipeline.
#
#     Law of Unidirectional Integrity: "Data flows forward;
#     dependencies never look back."
#
#     Multi-Proxy Pattern: Master audio is source of truth.
#     Stages request proxies at required sample rates.
#     Arena owns all buffers; pipeline holds only handles.
#
# COMPONENTS:
#     - GrimlockPipeline: Main orchestrator with arena ownership
#     - MusicBox: Forensic flight recorder (append-only logger)
#     - Scribe: Epistemic Veto authority (validation gates)
#     - StageContext: Immutable stage packets (StageInput/StageOutput)
#     - Cancellation: Real thread cancellation for timeouts
#
# EPISTEMIC INTEGRATIONS (NEW):
#     - EpistemicCouncil: Multi-model concurrent pitch detection
#     - MusicalFindingsMap: Cross-stage shared state
#
# STAGE FLOW (21 stages, optimized):
#     1. Ingestion
#     2. Feature Extraction
#     3. Tempo Analysis → writes to MusicalFindingsMap
#     4. Separation (Demucs)
#     5. Rhythm Detection
#     6. Pitch Detection (Bass)
#     7. Pitch Detection (Master)
#     8. Pitch Detection (Other) → uses EpistemicCouncil
#     9. Tonal Detection → writes chord_map to MusicalFindingsMap
#     10. Drum Detection
#     11. Timbre Analysis → writes instrument_families
#     12. Harmonic Validation → writes key signature
#     13. Groove Analysis → writes phase_delta, swing_ratio
#     14. Pulse Analysis
#     15. Reverse Geo Crypt → writes lattice_period_ms
#     16. Voice Continuity
#     17. Quantization → reads from MusicalFindingsMap
#     18. Velocity Merge
#     19. Consensus
#     20. Validation
#     21. Export
#
# USAGE:
#     from orchestration import GrimlockPipeline, MusicBox, Scribe
#
#     pipeline = GrimlockPipeline(
#         guided_params={"tempo_bpm": 90, "key_signature": "C major"},
#         music_box=MusicBox(),
#         scribe=Scribe(),
#         debug=True
#     )
#     result = pipeline.transcribe("song.wav", "output.mid")
# =================================================================

from typing import Dict, Any, Optional, List
from pathlib import Path

# ============================================================================
# Version Information
# ============================================================================

__version__ = "5.6.1"
__author__ = "DeepSeek"
__all__: List[str] = []

# ============================================================================
# Pipeline Components
# ============================================================================

try:
    from orchestration.pipeline import (
        GrimlockPipeline,
        PipelineStage,
        PipelineResult,
        create_pipeline,
        transcribe_file
    )
    __all__.extend([
        "GrimlockPipeline",
        "PipelineStage",
        "PipelineResult",
        "create_pipeline",
        "transcribe_file"
    ])
    _PIPELINE_OK = True
except ImportError as e:
    _PIPELINE_OK = False
    _PIPELINE_ERROR = str(e)

# ============================================================================
# Stage Context (Immutable Packets)
# ============================================================================

try:
    from orchestration.stage_context import (
        StageInput,
        StageOutput,
        StageStatus,
        AudioBuffer,
    )
    __all__.extend([
        "StageInput",
        "StageOutput",
        "StageStatus",
        "AudioBuffer",
    ])
    _STAGE_CONTEXT_OK = True
except ImportError as e:
    _STAGE_CONTEXT_OK = False
    _STAGE_CONTEXT_ERROR = str(e)

# ============================================================================
# Cancellation (Real Thread Cancellation)
# ============================================================================

try:
    from orchestration.cancellation import (
        CancellableThread,
    )
    __all__.append("CancellableThread")
    _CANCELLATION_OK = True
except ImportError as e:
    _CANCELLATION_OK = False
    _CANCELLATION_ERROR = str(e)

# ============================================================================
# Music Box (Forensic Logging)
# ============================================================================

try:
    from orchestration.music_box import (
        MusicBox,
        MusicBoxConfig,
        LogLevel,
        DecisionType,
        LogEntry,
        SessionSummary,
        create_music_box,
    )
    __all__.extend([
        "MusicBox",
        "MusicBoxConfig",
        "LogLevel",
        "DecisionType",
        "LogEntry",
        "SessionSummary",
        "create_music_box",
    ])
    _MUSIC_BOX_OK = True
except ImportError as e:
    _MUSIC_BOX_OK = False
    _MUSIC_BOX_ERROR = str(e)

# ============================================================================
# Scribe (Epistemic Veto Authority)
# ============================================================================

try:
    from orchestration.scribe import (
        Scribe,
        ScribeConfig,
        ValidationGate,
        VetoRecord,
        VetoSeverity,
        ValidationReport,
        RetryRecommendation,
        create_scribe,
    )
    __all__.extend([
        "Scribe",
        "ScribeConfig",
        "ValidationGate",
        "VetoRecord",
        "VetoSeverity",
        "ValidationReport",
        "RetryRecommendation",
        "create_scribe",
    ])
    _SCRIBE_OK = True
except ImportError as e:
    _SCRIBE_OK = False
    _SCRIBE_ERROR = str(e)

# ============================================================================
# Epistemic Integrations (Re-exported for convenience)
# ============================================================================

try:
    from epistemic.epistemic_council import (
        EpistemicCouncil,
        CouncilConfig,
        ModelResult,
        ConsensusNote,
    )
    __all__.extend([
        "EpistemicCouncil",
        "CouncilConfig",
        "ModelResult",
        "ConsensusNote",
    ])
    _EPISTEMIC_COUNCIL_OK = True
except ImportError as e:
    _EPISTEMIC_COUNCIL_OK = False
    _EPISTEMIC_COUNCIL_ERROR = str(e)

try:
    from epistemic.musical_findings_map import (
        MusicalFindingsMap,
        TemporalLatticeEvidence,
    )
    __all__.extend([
        "MusicalFindingsMap",
        "TemporalLatticeEvidence",
    ])
    _MUSICAL_FINDINGS_OK = True
except ImportError as e:
    _MUSICAL_FINDINGS_OK = False
    _MUSICAL_FINDINGS_ERROR = str(e)


# ============================================================================
# Module Docstring
# ============================================================================

__doc__ = """
Grimlock 5.4 Orchestration Module
=================================

The orchestrator is the only module allowed to touch every piece of the puzzle.
Agents never call the pipeline. This enforces Law of Unidirectional Integrity.

Multi-Proxy Pattern:
-------------------
Master audio is the SOURCE OF TRUTH (LoadedAudio with SHA-256 hash).
Stages request proxies at their required sample rates:
    - Demucs/Madmom → master.at_44k
    - Basic Pitch → master.at_22k  
    - CREPE/SPICE → master.at_16k

Downsampling is local, cached, and never modifies the master.

Arena Ownership:
---------------
- Arena owns all buffers (audio, stems, features)
- Pipeline stores only BufferID handles
- Agents borrow via arena.borrow() context manager
- Guardian enforces memory policy and eviction

Real Cancellation:
-----------------
- Timeout actually stops work (CancellableThread)
- No zombie threads or leaked GPU memory
- KeyboardInterrupt propagates properly

Components:
-----------
GrimlockPipeline:
    Main orchestrator - linear assembly line with guided mode support.
    Features: arena ownership, multi-rate proxies, staggered GC, 
              MusicBox logging, Scribe validation, guided overrides,
              real cancellation, retry recommendations.

MusicBox:
    Forensic flight recorder - append-only logger for every decision.
    Features: session isolation, JSONL format, automatic rotation, 
              safe DecisionType handling, memory event tracking.

Scribe:
    Epistemic Veto authority - validates every stage output.
    Features: multiple validation gates, hard/soft vetoes, 
              Schoenberg Mirror, retry recommendations,
              degraded mode eligibility.

StageContext:
    Immutable stage packets for data flow.
    Features: StageInput (context + handles), StageOutput (typed results),
              StageStatus (SUCCESS/DEGRADED/FAILED/EMPTY_BUT_VALID).

EPISTEMIC INTEGRATIONS (NEW in 5.4):
-----------------------------------
EpistemicCouncil:
    Multi-model concurrent pitch detection for the "other" stem.
    Runs Librosa, SPICE, Basic Pitch, CREPE, Omnizart in parallel.
    Uses MusicalFindingsMap for cross-stage state sharing.
    Pre-flight proxy strategy reduces wall time by 60-70%.

MusicalFindingsMap:
    Cross-stage shared state for musical intelligence.
    Collects tempo, key, groove, chord, instrument data from all stages.
    Provides unified evidence dict to TemporalLattice for quantization.

Stage Flow (with MusicalFindingsMap integration):
-------------------------------------------------
1. Ingestion → loads master audio
2. Feature Extraction → spectral features
3. Tempo Analysis → writes to MusicalFindingsMap
4. Separation (Demucs) → drums, bass, other stems
5. Rhythm Detection
6. Pitch Detection (Bass)
7. Pitch Detection (Master)
8. Pitch Detection (Other) → EpistemicCouncil reads MusicalFindingsMap
9. Tonal Detection → writes chord_map
10. Drum Detection
11. Timbre Analysis → writes instrument_families
12. Harmonic Validation → writes key signature
13. Groove Analysis → writes phase_delta, swing_ratio
14. Pulse Analysis
15. Reverse Geo Crypt → writes lattice_period_ms
16. Voice Continuity
17. Quantization → reads MusicalFindingsMap via .to_temporal_lattice_evidence()
18. Velocity Merge
19. Consensus
20. Validation
21. Export

Validation Gates (Scribe):
-------------------------
1. Empty Result Check
2. Mandatory Fields Check
3. Schoenberg Mirror (Harmonic Series)
4. Silence Ratio Check
5. Confidence Threshold Check
6. Phase Consistency Check (Relational Physics)
7. Hallucination Ratio Check

Retry Recommendations:
---------------------
- NONE: No retry needed
- WITH_FALLBACK: Use fallback model
- WITH_DEGRADED: Run in degraded mode
- WITH_DIFFERENT_MODEL: Try alternative model
- AFTER_MEMORY_CLEANUP: Free memory first

5 Strategies Integration:
------------------------
Strategy 1 (Streaming Windows): Pipeline supports chunked audio
Strategy 2 (Disk-Backed): MusicBox logs persist to disk
Strategy 3 (mmap for Audit): MusicBox supports memory-mapped queries
Strategy 4 (GC Discipline): Staggered GC between stages
Strategy 5 (Sparse Representations): Only event metadata passed
"""


# ============================================================================
# Module Initialization Check
# ============================================================================

def _check_imports() -> Dict[str, bool]:
    """Verify all orchestration components are importable."""
    return {
        "pipeline": _PIPELINE_OK,
        "stage_context": _STAGE_CONTEXT_OK,
        "cancellation": _CANCELLATION_OK,
        "music_box": _MUSIC_BOX_OK,
        "scribe": _SCRIBE_OK,
        "epistemic_council": _EPISTEMIC_COUNCIL_OK,
        "musical_findings": _MUSICAL_FINDINGS_OK,
    }


_IMPORT_STATUS = _check_imports()
_IMPORTS_OK = all(_IMPORT_STATUS.values())


def _report_import_issues() -> None:
    """Report any import issues for debugging."""
    if _IMPORTS_OK:
        return

    import warnings
    issues = []
    for name, ok in _IMPORT_STATUS.items():
        if not ok:
            error_var = f"_{name.upper()}_ERROR"
            error_msg = globals().get(error_var, "unknown error")
            issues.append(f"{name}: {error_msg}")

    if issues:
        warnings.warn(
            f"Some orchestration components failed to import:\n  " + "\n  ".join(issues),
            ImportWarning
        )


# Run import check on module load
_report_import_issues()


# ============================================================================
# Version Info
# ============================================================================

version_info = {
    "module": "orchestration",
    "version": __version__,
    "imports_ok": _IMPORTS_OK,
    "import_status": _IMPORT_STATUS,
    "components": ["pipeline", "music_box", "scribe", "stage_context", "cancellation"],
    "epistemic_components": ["epistemic_council", "musical_findings_map"],
    "features": [
        "multi_proxy_pattern",
        "arena_ownership",
        "guided_mode",
        "forensic_logging",
        "epistemic_veto",
        "real_cancellation",
        "retry_recommendations",
        "groove_aware_quantization",
        "spectral_masking",
        "harmonic_validation",
        "epistemic_council",
        "musical_findings_map",
    ],
    "stage_count": 21,
}


# ============================================================================
# Lazy Imports for Heavy Components
# ============================================================================

def get_pipeline(debug: bool = False, **kwargs) -> 'GrimlockPipeline':
    """Lazy factory for GrimlockPipeline."""
    if not _PIPELINE_OK:
        raise ImportError(f"Cannot create pipeline: {_PIPELINE_ERROR}")
    from orchestration.pipeline import GrimlockPipeline
    return GrimlockPipeline(debug=debug, **kwargs)


def get_music_box(log_path: str = None, **kwargs) -> 'MusicBox':
    """Lazy factory for MusicBox."""
    if not _MUSIC_BOX_OK:
        raise ImportError(f"Cannot create music box: {_MUSIC_BOX_ERROR}")
    from orchestration.music_box import create_music_box
    return create_music_box(log_path=log_path, **kwargs)


def get_scribe(music_box: 'MusicBox' = None, **kwargs) -> 'Scribe':
    """Lazy factory for Scribe."""
    if not _SCRIBE_OK:
        raise ImportError(f"Cannot create scribe: {_SCRIBE_ERROR}")
    from orchestration.scribe import create_scribe
    return create_scribe(music_box=music_box, **kwargs)


def get_epistemic_council(config: Optional['CouncilConfig'] = None) -> 'EpistemicCouncil':
    """Lazy factory for EpistemicCouncil."""
    if not _EPISTEMIC_COUNCIL_OK:
        raise ImportError(f"Cannot create EpistemicCouncil: {_EPISTEMIC_COUNCIL_ERROR}")
    from epistemic.epistemic_council import EpistemicCouncil, CouncilConfig
    return EpistemicCouncil(config=config or CouncilConfig())


def get_musical_findings_map() -> 'MusicalFindingsMap':
    """Lazy factory for MusicalFindingsMap."""
    if not _MUSICAL_FINDINGS_OK:
        raise ImportError(f"Cannot create MusicalFindingsMap: {_MUSICAL_FINDINGS_ERROR}")
    from epistemic.musical_findings_map import MusicalFindingsMap
    return MusicalFindingsMap()


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_transcribe(
    audio_path: str,
    output_path: str = None,
    guided_tempo: float = None,
    guided_key: str = None,
    debug: bool = False
) -> 'PipelineResult':
    """
    Quick transcription with minimal configuration.

    Args:
        audio_path: Path to audio file
        output_path: Optional output path
        guided_tempo: Optional tempo override
        guided_key: Optional key override
        debug: Enable debug output

    Returns:
        PipelineResult
    """
    if not _PIPELINE_OK:
        raise ImportError(f"Cannot transcribe: {_PIPELINE_ERROR}")

    return transcribe_file(
        audio_path=audio_path,
        output_path=output_path,
        tempo_bpm=guided_tempo,
        key_signature=guided_key,
        debug=debug
    )


def get_pipeline_info() -> Dict[str, Any]:
    """Get information about the pipeline configuration."""
    info = {
        "version": __version__,
        "components": version_info["components"],
        "epistemic_components": version_info["epistemic_components"],
        "features": version_info["features"],
        "stage_count": version_info["stage_count"],
        "imports_ok": _IMPORTS_OK,
        "import_status": _IMPORT_STATUS,
    }

    # Try to get resampling info
    try:
        from ingestion.downsampler import DownsamplingEngine
        info["resampling_available"] = DownsamplingEngine.is_available()
        info["quality_presets"] = list(DownsamplingEngine.get_quality_presets().keys())
    except ImportError:
        info["resampling_available"] = False
        info["quality_presets"] = []

    # Model sample rates
    info["model_rates"] = {
        "crepe": 16000,
        "basic_pitch": 22050,
        "demucs": 44100,
        "madmom": 44100,
        "spice": 16000,
        "omnizart": 44100,
    }

    return info


def get_stage_names() -> List[str]:
    """Get list of all pipeline stage names."""
    if not _PIPELINE_OK:
        return []

    try:
        from orchestration.pipeline import create_pipeline
        pipeline = create_pipeline(debug=False)
        if hasattr(pipeline, '_stages'):
            return [s.name for s in pipeline._stages]
    except Exception:
        pass
    return []


def get_stage_count() -> int:
    """Get the total number of pipeline stages."""
    return len(get_stage_names()) or version_info["stage_count"]


def get_epistemic_features() -> Dict[str, Any]:
    """Get information about epistemic integrations."""
    return {
        "epistemic_council_available": _EPISTEMIC_COUNCIL_OK,
        "musical_findings_available": _MUSICAL_FINDINGS_OK,
        "features": [
            "multi_model_concurrency",
            "cross_state_sharing",
            "pre_flight_proxy",
            "temporal_lattice_evidence",
        ],
    }


# ============================================================================
# Standalone Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRIMLOCK 5.4 - ORCHESTRATION MODULE")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")

    print("\nImport Status:")
    for name, ok in _IMPORT_STATUS.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    print(f"\nFeatures: {', '.join(version_info['features'])}")

    print("\nAvailable Components:")
    print("  ✓ GrimlockPipeline: Main orchestrator (arena ownership)")
    print("  ✓ StageInput/StageOutput: Immutable stage packets")
    print("  ✓ CancellableThread: Real thread cancellation")
    print("  ✓ MusicBox: Forensic flight recorder")
    print("  ✓ Scribe: Epistemic Veto authority")

    if _EPISTEMIC_COUNCIL_OK:
        print("  ✓ EpistemicCouncil: Multi-model concurrent pitch detection")
    if _MUSICAL_FINDINGS_OK:
        print("  ✓ MusicalFindingsMap: Cross-stage shared state")

    print("\n" + "=" * 70)
    print("Quick Test:")
    print("=" * 70)

    # Test MusicBox
    print("\n1. MusicBox:")
    if _MUSIC_BOX_OK:
        mb = get_music_box()
        mb.log_decision(
            stage_name="test",
            decision_type="test",
            before_state={},
            after_state={"test": True},
            reasoning="Test log entry",
            reversible=True
        )
        stats = mb.get_stats()
        print(f"   Session: {stats['session_id']}")
        print(f"   Buffer size: {stats['buffer_size']}")
        print(f"   Error count: {stats['error_count']}")
    else:
        print("   ✗ MusicBox not available")

    # Test Scribe
    print("\n2. Scribe:")
    if _SCRIBE_OK:
        scribe = get_scribe()
        print(f"   Scribe initialized")
        print(f"   Retry recommendations: {[r.value for r in RetryRecommendation]}")
    else:
        print("   ✗ Scribe not available")

    # Test StageStatus
    print("\n3. StageStatus:")
    if _STAGE_CONTEXT_OK:
        from orchestration.stage_context import StageStatus
        print(f"   Status values: {[s.value for s in StageStatus]}")
    else:
        print("   ✗ StageContext not available")

    # Test MusicalFindingsMap
    print("\n4. MusicalFindingsMap:")
    if _MUSICAL_FINDINGS_OK:
        findings = get_musical_findings_map()
        findings.tempo_bpm = 120.0
        findings.swing_ratio = 0.65
        findings.record_stage("test")
        evidence = findings.to_temporal_lattice_evidence()
        print(f"   Tempo: {findings.tempo_bpm} BPM")
        print(f"   Swing: {findings.swing_ratio}")
        print(f"   Evidence keys: {list(evidence.keys())}")
    else:
        print("   ✗ MusicalFindingsMap not available")

    # Test EpistemicCouncil
    print("\n5. EpistemicCouncil:")
    if _EPISTEMIC_COUNCIL_OK:
        council = get_epistemic_council()
        print(f"   Council created")
        print(f"   Available models: {council.get_available_models() if hasattr(council, 'get_available_models') else 'unknown'}")
    else:
        print("   ✗ EpistemicCouncil not available")

    # Test Pipeline factory
    print("\n6. Pipeline Factory:")
    if _PIPELINE_OK:
        pipeline = get_pipeline(debug=False)
        print(f"   Pipeline created")
        stages = get_stage_names()
        print(f"   Stages: {len(stages)}")
        print(f"   Stage count: {get_stage_count()}")
    else:
        print("   ✗ Pipeline not available")

    # Test info
    print("\n7. Pipeline Info:")
    info = get_pipeline_info()
    print(f"   Version: {info['version']}")
    print(f"   Stage count: {info['stage_count']}")
    print(f"   Resampling available: {info['resampling_available']}")
    if info.get('quality_presets'):
        print(f"   Quality presets: {info['quality_presets']}")
    print(f"   Features: {', '.join(info['features'][:5])}...")

    # Test epistemic features
    print("\n8. Epistemic Features:")
    ep_info = get_epistemic_features()
    for key, value in ep_info.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print(f"Orchestration module ready (v{__version__}).")
    print("=" * 70)