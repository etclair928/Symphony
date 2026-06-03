# =================================================================
# MODULE: epistemic/__init__.py
# DESCRIPTION: Epistemic module exports for Grimlock 5.6.1
#
# VERSION: 5.6.1 (Added QuaverIntelligence + PhraseIntelligence)
# UPDATED: 2026-06-02
#
# PHILOSOPHY:
#     "Epistemic" means "relating to knowledge or its validation."
#
#     This module houses the intelligence that doesn't just process
#     audio — it reasons about what it hears. It asks:
#         "Do I actually know this?"
#         "What is the confidence of that knowledge?"
#         "How do multiple witnesses agree or disagree?"
#
#     The Epistemic Council is the Socratic method applied to pitch.
#     Multiple models debate, and consensus emerges.
#
#     The Musical Findings Map is the shared memory of the ensemble.
#     Every stage writes its conclusions; every stage reads the truth.
#
#     QuaverIntelligence is the symbolic duration witness. It doesn't
#     quantize — it testifies with competing duration hypotheses.
#
#     PhraseIntelligence is the structural phase engine. It doesn't
#     detect form — it tracks structural trajectory over time.
#
# COMPONENTS (NEW in 5.6.1):
#     - EpistemicCouncil: Multi-model concurrent pitch detection
#     - MusicalFindingsMap: Cross-stage shared state for quantization
#     - QuaverIntelligence: Symbolic duration witness (NEW)
#     - PhraseIntelligence: Structural phase engine (NEW)
#
# INTEGRATION WITH PIPELINE:
#     The pipeline imports from epistemic/ and uses:
#         - EpistemicCouncil in _run_pitch_detection_other
#         - MusicalFindingsMap in __init__ and _build_temporal_lattice
#         - QuaverIntelligence in _run_quaver_intelligence
#         - PhraseIntelligence in _run_phrase_intelligence
#
# KEY CONCEPTS:
#     1. Multi-Proxy Consensus: Multiple models vote on each note
#     2. Pre-Flight Proxy: Fast models constrain slow models
#     3. Cross-Stage Memory: Findings persist across pipeline stages
#     4. Temporal Lattice Evidence: Unified dict for quantization
#     5. Symbolic Duration Witness: Competing duration hypotheses (NEW)
#     6. Structural Phase Engine: Trajectory through phase space (NEW)
#
# USAGE:
#     from epistemic import EpistemicCouncil, MusicalFindingsMap
#     from epistemic import QuaverIntelligence, PhraseIntelligence
#
#     # Shared state across stages
#     findings = MusicalFindingsMap()
#     findings.tempo_bpm = 120.0
#     findings.swing_ratio = 0.65
#
#     # Multi-model pitch detection
#     council = EpistemicCouncil()
#     notes = await council.analyze(audio, sr=16000, findings=findings)
#
#     # Symbolic duration witness
#     quaver = QuaverIntelligence(findings)
#     testimonies = quaver.infer_symbolic_durations(...)
#
#     # Structural phase engine
#     phrase = PhraseIntelligence(findings)
#     trajectory = phrase.compute_structural_trajectory(...)
# =================================================================

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# ============================================================================
# Version Information
# ============================================================================

__version__ = "5.6.1"
__author__ = "DeepSeek"
__all__: List[str] = []

# ============================================================================
# Epistemic Council (Multi-Model Concurrent Pitch Detection)
# ============================================================================

try:
    from epistemic.epistemic_council import (
        EpistemicCouncil,
        CouncilConfig,
        ModelResult,
        ConsensusNote,
        Vote,
        Witness,
        Verdict,
    )

    __all__.extend([
        "EpistemicCouncil",
        "CouncilConfig",
        "ModelResult",
        "ConsensusNote",
        "Vote",
        "Witness",
        "Verdict",
    ])
    _EPISTEMIC_COUNCIL_OK = True
except ImportError as e:
    _EPISTEMIC_COUNCIL_OK = False
    _EPISTEMIC_COUNCIL_ERROR = str(e)

# ============================================================================
# Musical Findings Map (Cross-Stage Shared State)
# ============================================================================

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
# Quaver Intelligence (Symbolic Duration Witness) - NEW in 5.6.1
# ============================================================================

try:
    from agents.quantization.quaver_intelligence import (
        QuaverIntelligence,
        QuaverConfig,
        DurationHypothesis,
        WitnessContribution,
        QuaverTestimony,
    )

    __all__.extend([
        "QuaverIntelligence",
        "QuaverConfig",
        "DurationHypothesis",
        "WitnessContribution",
        "QuaverTestimony",
    ])
    _QUAVER_INTELLIGENCE_OK = True
except ImportError as e:
    _QUAVER_INTELLIGENCE_OK = False
    _QUAVER_INTELLIGENCE_ERROR = str(e)

# ============================================================================
# Phrase Intelligence (Structural Phase Engine) - NEW in 5.6.1
# ============================================================================

try:
    from agents.analysis.phrase_intelligence import (
        PhraseIntelligence,
        StructuralInvariantVector,
        StructuralWindow,
        PhaseTransition,
        StructuralTestimony,
    )

    __all__.extend([
        "PhraseIntelligence",
        "StructuralInvariantVector",
        "StructuralWindow",
        "PhaseTransition",
        "StructuralTestimony",
    ])
    _PHRASE_INTELLIGENCE_OK = True
except ImportError as e:
    _PHRASE_INTELLIGENCE_OK = False
    _PHRASE_INTELLIGENCE_ERROR = str(e)

# ============================================================================
# Module Docstring
# ============================================================================

__doc__ = """
Epistemic Module — Knowledge Validation for Grimlock 5.6.1
=========================================================

"Epistemic" means "relating to knowledge or its validation."

This module houses the intelligence that doesn't just process audio —
it reasons about what it hears. It asks:
    - "Do I actually know this pitch?"
    - "What is the confidence of that knowledge?"
    - "How do multiple witnesses agree or disagree?"
    - "What is the true duration of this note?"
    - "What is the structural phase of this passage?"


EpistemicCouncil: Multi-Model Concurrent Pitch Detection
--------------------------------------------------------
The Council is the Socratic method applied to pitch detection.
Multiple models debate, and consensus emerges.

Architecture:
    Stage 1 (Fast Models): Librosa + SPICE (~1s)
        → Produces constraint map + active time windows

    Stage 2 (Heavy Models): Basic Pitch + Omnizart
        → Guided by Stage 1 constraints (reduces compute by 60%)

    Stage 3 (Refinement): CREPE
        → High-confidence refinement of detected notes

    Stage 4 (Consensus): Weighted voting + agreement bonus
        → Produces final ConsensusNote list

Models Supported:
    - Librosa (free, fast, moderate accuracy)
    - SPICE (Spectral Pitch, good for monophonic)
    - Basic Pitch (Google, good polyphonic)
    - CREPE (high accuracy, slower)
    - Omnizart (Chinese music focus, optional)


MusicalFindingsMap: Cross-Stage Shared State
--------------------------------------------
The Musical Findings Map is the shared memory of the ensemble.
Every stage writes its conclusions; every stage reads the truth.

What it stores:
    - Tempo (BPM, confidence, source, beat_times_ms)
    - Key Signature (detected key, confidence)
    - Groove (phase delta, swing ratio, groove type)
    - Lattice (period_ms, confidence)
    - Instruments (family distribution)
    - Chords (chord_map for tonal detection)
    - Active Windows (time ranges with activity)
    - Quaver Testimonies (duration hypotheses) - NEW
    - Structural Testimony (phase trajectory) - NEW

Lifecycle:
    1. TempoAnalysis writes tempo_bpm, beat_times_ms
    2. TonalDetection writes chord_map
    3. GrooveAnalysis writes phase_delta_ms, swing_ratio
    4. ReverseGeoCrypt writes lattice_period_ms
    5. HarmonicValidation writes detected_key, key_confidence
    6. TimbreAnalysis writes instrument_families
    7. QuaverIntelligence writes quaver_testimonies (NEW)
    8. PhraseIntelligence writes structural_testimony (NEW)
    9. Quantization reads via .to_temporal_lattice_evidence()


QuaverIntelligence: Symbolic Duration Witness (NEW in 5.6.1)
-----------------------------------------------------------
QuaverIntelligence is NOT a quantizer. It is an EPISTEMIC WITNESS.

It generates COMPETING DURATION HYPOTHESES with:
    - Supporting evidence from other witnesses
    - Opposing evidence from other witnesses
    - Epistemic tension scores
    - Uncertainty metrics

It does NOT:
    - Mutate notes
    - Decide which duration is correct
    - Enforce grids

It TESTIFIES. The EpistemicCouncil arbitrates.
The ConsensusEngine stabilizes truth.

Key Concepts:
    - DurationHypothesis: One candidate symbolic duration
    - WitnessContribution: One agent's support/opposition
    - QuaverTestimony: Complete testimony for one note
    - Epistemic Tension: High when support ≈ opposition

Inputs (consumes testimony from):
    - PulseFieldAnalyzer (beat phase, downbeat confidence)
    - GrooveField (swing amount, deviation)
    - AnechoicMa (resonance, sustain collapse)
    - TempoIntelligence (local tempo)
    - DrumIntelligence (percussive anchor)
    - VoiceContinuity (phrase grouping)
    - ReverseGeoCrypt (lattice period)

Output:
    - List[QuaverTestimony] → consumed by EpistemicCouncil


PhraseIntelligence: Structural Phase Engine (NEW in 5.6.1)
---------------------------------------------------------
PhraseIntelligence is NOT a structural classifier. It is a
STRUCTURAL PHASE ENGINE that tracks trajectory through time.

It computes:
    - Structural invariants (orthogonal dimensions)
    - Phase transitions (THEME → DEVELOPMENT → RECAPITULATION)
    - Attractor basins (genre EMERGENCE, not detection)
    - Boundary strength (multi-witness agreement)
    - Re-analysis requests (structural contradictions)

Key Concepts:
    - StructuralPhase: Discrete state (THEME, DEVELOPMENT, SOLO, etc.)
    - StructuralInvariantVector: 6 orthogonal dimensions
    - AttractorBasin: Where trajectory converges (POP, JAZZ, CLASSICAL)
    - PhaseTransition: Change between structural phases
    - StructuralTestimony: Complete trajectory with re-analysis flags

Inputs (consumes testimony from):
    - PulseFieldAnalyzer (downbeat confidence, stability)
    - VoiceContinuity (phrase boundaries, continuity)
    - HarmonicIntelligence (progressions, cadences)
    - ReverseGeoCrypt (bar structure, lattice)
    - TempoIntelligence (tempo curve)
    - GrooveField (swing intensity)

Output:
    - StructuralTestimony → consumed by ConsensusEngine + Scribe


Why This Matters:
----------------
Without MusicalFindingsMap, TemporalLattice had to re-extract
evidence from stage results every time. With the map:
    - Single source of truth
    - No duplicate extraction logic
    - Easy debugging (findings.stages_written shows provenance)

Without QuaverIntelligence, duration was treated as a single
value, not a contested hypothesis. With Quaver:
    - Multiple competing duration possibilities
    - Support AND opposition from witnesses
    - Epistemic tension reveals uncertainty

Without PhraseIntelligence, structure was a snapshot, not a
trajectory. With Phrase:
    - Phase transitions over time
    - Attractor basin convergence (genre emergence)
    - Structural re-analysis when contradictions detected


Integration with Pipeline Stages (5.6.1):
----------------------------------------
Stage 3 (Tempo Analysis)           → writes to findings
Stage 8 (Pitch Other)              → reads active windows from findings
Stage 9 (Tonal Detection)          → writes chord_map
Stage 11 (Timbre Analysis)         → writes instrument_families
Stage 12 (Harmonic Validation)     → writes detected_key
Stage 13 (Groove Analysis)         → writes phase_delta, swing_ratio
Stage 15 (Reverse Geo Crypt)       → writes lattice_period_ms
Stage 16 (Voice Continuity)        → writes phrase boundaries
Stage 17 (Quaver Intelligence)     → writes quaver_testimonies (NEW)
Stage 18 (Quantization)            → reads via .to_temporal_lattice_evidence()
Stage 20 (Consensus)               → consumes QuaverTestimony
Stage 21 (Phrase Intelligence)     → writes structural_testimony (NEW)
Stage 22 (Structural Feedback)     → reads requires_reanalysis flag
Stage 23 (Validation)              → validates structural phase
Stage 24 (Export)                  → includes structural summary


Key Data Structures (NEW):
-------------------------
QuaverConfig:
    Configuration for QuaverIntelligence
    - tempo_relative_tolerance, tuplet_tolerance_factor
    - swing_threshold_ms, sustain_resonance_threshold
    - artifact_opposition_threshold

DurationHypothesis:
    One competing duration candidate
    - symbolic_value ("quarter", "eighth", "triplet", etc.)
    - duration_ms, probability
    - support_score, opposition_score
    - epistemic_tension, uncertainty
    - supporting_witnesses, opposing_witnesses

QuaverTestimony:
    Complete duration testimony for one note
    - note_key, start_ms, raw_duration_ms
    - duration_hypotheses (list)
    - phrase_context, instrument_profile
    - uncertainty, contradiction_detected

StructuralInvariantVector:
    6 orthogonal structural dimensions
    - tonal_stability, temporal_cyclicity
    - transformational_depth, event_density
    - constraint_strength, surface_entropy

StructuralPhase:
    Discrete structural states
    - THEME, DEVELOPMENT, TRANSITION
    - RECAPITULATION, CODA, INTRO
    - SOLO, LOOP, AMBIGUOUS

AttractorBasin:
    Genre emergence (not detection)
    - POP, JAZZ, CLASSICAL, AMBIENT, BINARY

StructuralTestimony:
    Complete structural trajectory
    - structural_trajectory (list of windows)
    - phase_transitions (list)
    - attractor_basin, attractor_confidence
    - needs_reanalysis (with severity)
    - phrase_boundaries, anacrusis_detected


Philosophy:
----------
The Epistemic module embodies the 5.6.1 principle:
    "Don't just process. Reason. And when uncertain, testify."

The Epistemic Council replaces "run one model, hope it's right"
with "run multiple models, let them vote."

The Musical Findings Map replaces "extract same data 5 times"
with "write once, read many."

QuaverIntelligence replaces "snap to grid, hope it's right"
with "generate competing hypotheses, let Council arbitrate."

PhraseIntelligence replaces "detect form, output label"
with "track phase trajectory, let genre emerge."

Together, they make Grimlock not just a transcriber,
but a musical intelligence that knows what it knows
— and knows what it doesn't know.
"""


# ============================================================================
# Module Initialization Check
# ============================================================================

def _check_imports() -> Dict[str, bool]:
    """Verify all epistemic components are importable."""
    return {
        "epistemic_council": _EPISTEMIC_COUNCIL_OK,
        "musical_findings": _MUSICAL_FINDINGS_OK,
        "quaver_intelligence": _QUAVER_INTELLIGENCE_OK,
        "phrase_intelligence": _PHRASE_INTELLIGENCE_OK,
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
            f"Some epistemic components failed to import:\n  " + "\n  ".join(issues),
            ImportWarning
        )


# Run import check on module load
_report_import_issues()

# ============================================================================
# Version Info
# ============================================================================

version_info = {
    "module": "epistemic",
    "version": __version__,
    "imports_ok": _IMPORTS_OK,
    "import_status": _IMPORT_STATUS,
    "components": [
        "epistemic_council",
        "musical_findings",
        "quaver_intelligence",
        "phrase_intelligence",
    ],
    "features": [
        "multi_model_concurrency",
        "pre_flight_proxy",
        "consensus_voting",
        "agreement_bonus",
        "cross_stage_state",
        "temporal_lattice_evidence",
        "stage_provenance",
        "symbolic_duration_witness",
        "competing_hypotheses",
        "epistemic_tension",
        "structural_phase_engine",
        "phase_trajectory",
        "attractor_basin_emergence",
        "structural_reanalysis",
    ],
    "models": ["librosa", "spice", "basic_pitch", "crepe", "omnizart"],
    "witnesses": ["quaver", "phrase"],
}


# ============================================================================
# Lazy Imports for Heavy Components
# ============================================================================

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


def get_default_council_config() -> 'CouncilConfig':
    """Get default configuration for EpistemicCouncil."""
    if not _EPISTEMIC_COUNCIL_OK:
        raise ImportError(f"Cannot create CouncilConfig: {_EPISTEMIC_COUNCIL_ERROR}")
    from epistemic.epistemic_council import CouncilConfig
    return CouncilConfig()


# ============================================================================
# Quaver Intelligence Lazy Factories (NEW)
# ============================================================================

def get_quaver_intelligence(
        findings_map: 'MusicalFindingsMap',
        config: Optional['QuaverConfig'] = None
) -> 'QuaverIntelligence':
    """Lazy factory for QuaverIntelligence."""
    if not _QUAVER_INTELLIGENCE_OK:
        raise ImportError(f"Cannot create QuaverIntelligence: {_QUAVER_INTELLIGENCE_ERROR}")
    from agents.quantization.quaver_intelligence import QuaverIntelligence, QuaverConfig
    return QuaverIntelligence(findings_map, config or QuaverConfig())


def get_default_quaver_config() -> 'QuaverConfig':
    """Get default configuration for QuaverIntelligence."""
    if not _QUAVER_INTELLIGENCE_OK:
        raise ImportError(f"Cannot create QuaverConfig: {_QUAVER_INTELLIGENCE_ERROR}")
    from agents.quantization.quaver_intelligence import QuaverConfig
    return QuaverConfig()


# ============================================================================
# Phrase Intelligence Lazy Factories (NEW)
# ============================================================================

def get_phrase_intelligence(
        findings_map: 'MusicalFindingsMap'
) -> 'PhraseIntelligence':
    """Lazy factory for PhraseIntelligence."""
    if not _PHRASE_INTELLIGENCE_OK:
        raise ImportError(f"Cannot create PhraseIntelligence: {_PHRASE_INTELLIGENCE_ERROR}")
    from agents.analysis.phrase_intelligence import PhraseIntelligence
    return PhraseIntelligence(findings_map)


def get_default_structural_config() -> Dict[str, Any]:
    """Get default configuration for PhraseIntelligence."""
    return {
        "window_size_ms": 2000.0,
        "window_overlap": 0.5,
        "boundary_rest_threshold_bars": 0.75,
        "boundary_pitch_jump_threshold": 7,
    }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_epistemic_pipeline(
        audio: 'np.ndarray',
        sr: int = 16000,
        use_all_models: bool = True,
        **kwargs
) -> List['ConsensusNote']:
    """
    Convenience function: run EpistemicCouncil on audio and return notes.

    Args:
        audio: Audio array (float32, mono)
        sr: Sample rate (must be 16000 for most models)
        use_all_models: If True, enable all available models
        **kwargs: Additional CouncilConfig parameters

    Returns:
        List of ConsensusNote objects
    """
    import asyncio

    config_kwargs = {}
    if use_all_models:
        config_kwargs.update({
            "use_librosa": True,
            "use_spice": True,
            "use_basic_pitch": True,
            "use_crepe": True,
            "use_omnizart": False,  # Heavy, optional
        })
    config_kwargs.update(kwargs)

    config = get_default_council_config()
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    council = get_epistemic_council(config)
    findings = get_musical_findings_map()

    # Run async in sync context
    try:
        loop = asyncio.get_running_loop()
        # Already in async context - can't run
        raise RuntimeError("Use async version in async context")
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        notes = asyncio.run(council.analyze(audio, sr, findings))

    council.close()
    return notes


def get_findings_summary(findings: 'MusicalFindingsMap') -> Dict[str, Any]:
    """
    Get a human-readable summary of MusicalFindingsMap.

    Args:
        findings: MusicalFindingsMap instance

    Returns:
        Dictionary with summary statistics
    """
    return {
        "tempo": {
            "bpm": findings.tempo_bpm,
            "confidence": findings.tempo_confidence,
            "source": findings.tempo_source,
        },
        "key": {
            "signature": findings.detected_key,
            "confidence": findings.key_confidence,
        },
        "groove": {
            "phase_delta_ms": findings.phase_delta_ms,
            "swing_ratio": findings.swing_ratio,
            "type": findings.groove_type,
        },
        "lattice": {
            "period_ms": findings.lattice_period_ms,
            "confidence": findings.lattice_confidence,
        },
        "instruments": findings.instrument_families,
        "chord_count": len(findings.chord_map),
        "stages_contributed": findings.stages_written,
        "has_active_windows": len(findings.active_time_windows_ms) > 0,
        "has_quaver_testimonies": hasattr(findings, 'quaver_testimonies') and bool(findings.quaver_testimonies),
        "has_structural_testimony": hasattr(findings, 'structural_testimony') and findings.structural_testimony is not None,
    }


def is_epistemic_available() -> Dict[str, bool]:
    """
    Check which epistemic components are available.

    Returns:
        Dictionary with availability flags
    """
    return {
        "epistemic_council": _EPISTEMIC_COUNCIL_OK,
        "musical_findings": _MUSICAL_FINDINGS_OK,
        "quaver_intelligence": _QUAVER_INTELLIGENCE_OK,
        "phrase_intelligence": _PHRASE_INTELLIGENCE_OK,
        "full_epistemic": _IMPORTS_OK,
    }


# ============================================================================
# Standalone Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRIMLOCK 5.6.1 - EPISTEMIC MODULE")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")

    print("\nImport Status:")
    for name, ok in _IMPORT_STATUS.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    print(f"\nFeatures: {', '.join(version_info['features'])}")
    print(f"Models: {', '.join(version_info['models'])}")
    print(f"Witnesses: {', '.join(version_info['witnesses'])}")

    print("\nAvailable Components:")
    if _EPISTEMIC_COUNCIL_OK:
        print("  ✓ EpistemicCouncil: Multi-model concurrent pitch detection")
        print("    - Librosa, SPICE, Basic Pitch, CREPE, Omnizart")
        print("    - Pre-flight proxy strategy")
        print("    - Consensus voting with agreement bonus")
    else:
        print(f"  ✗ EpistemicCouncil: {_EPISTEMIC_COUNCIL_ERROR}")

    if _MUSICAL_FINDINGS_OK:
        print("  ✓ MusicalFindingsMap: Cross-stage shared state")
        print("    - Tempo, key, groove, lattice storage")
        print("    - Stage provenance tracking")
        print("    - TemporalLattice evidence export")
    else:
        print(f"  ✗ MusicalFindingsMap: {_MUSICAL_FINDINGS_ERROR}")

    if _QUAVER_INTELLIGENCE_OK:
        print("  ✓ QuaverIntelligence: Symbolic duration witness")
        print("    - Competing duration hypotheses")
        print("    - Support + opposition from witnesses")
        print("    - Epistemic tension scoring")
    else:
        print(f"  ✗ QuaverIntelligence: {_QUAVER_INTELLIGENCE_ERROR}")

    if _PHRASE_INTELLIGENCE_OK:
        print("  ✓ PhraseIntelligence: Structural phase engine")
        print("    - 6-dimensional invariant vectors")
        print("    - Phase state machine (THEME → DEVELOPMENT → RECAP)")
        print("    - Attractor basin emergence")
        print("    - Structural re-analysis triggers")
    else:
        print(f"  ✗ PhraseIntelligence: {_PHRASE_INTELLIGENCE_ERROR}")

    print("\n" + "=" * 70)
    print("Quick Test:")
    print("=" * 70)

    # Test MusicalFindingsMap
    print("\n1. MusicalFindingsMap:")
    if _MUSICAL_FINDINGS_OK:
        findings = get_musical_findings_map()

        # Write some data
        findings.tempo_bpm = 120.0
        findings.tempo_confidence = 0.85
        findings.tempo_source = "test"
        findings.swing_ratio = 0.65
        findings.groove_type = "dilla_pocket"
        findings.record_stage("test_stage")

        # Read back
        print(f"   Tempo: {findings.tempo_bpm} BPM (conf={findings.tempo_confidence})")
        print(f"   Swing: {findings.swing_ratio}")
        print(f"   Groove: {findings.groove_type}")
        print(f"   Stages written: {findings.stages_written}")

        # Test evidence conversion
        evidence = findings.to_temporal_lattice_evidence()
        print(f"   Evidence keys: {list(evidence.keys())}")

        # Test summary
        summary = get_findings_summary(findings)
        print(f"   Summary keys: {list(summary.keys())}")
    else:
        print("   ✗ MusicalFindingsMap not available")

    # Test EpistemicCouncil configuration
    print("\n2. EpistemicCouncil Config:")
    if _EPISTEMIC_COUNCIL_OK:
        config = get_default_council_config()
        print(f"   use_librosa: {config.use_librosa}")
        print(f"   use_spice: {config.use_spice}")
        print(f"   use_basic_pitch: {config.use_basic_pitch}")
        print(f"   use_crepe: {config.use_crepe}")
        print(f"   use_omnizart: {config.use_omnizart}")
        print(f"   max_workers: {config.max_workers}")
        print(f"   agreement_bonus: {config.agreement_bonus}")

        # Create council (no audio, just test creation)
        council = get_epistemic_council(config)
        print(f"   Council created successfully")
        council.close()
    else:
        print("   ✗ EpistemicCouncil not available")

    # Test QuaverIntelligence config
    print("\n3. QuaverIntelligence Config:")
    if _QUAVER_INTELLIGENCE_OK:
        config = get_default_quaver_config()
        print(f"   tempo_relative_tolerance: {config.tempo_relative_tolerance}")
        print(f"   tuplet_tolerance_factor: {config.tuplet_tolerance_factor}")
        print(f"   swing_threshold_ms: {config.swing_threshold_ms}")
        print(f"   sustain_resonance_threshold: {config.sustain_resonance_threshold}")
        print(f"   artifact_opposition_threshold: {config.artifact_opposition_threshold}")
        print(f"   jazz_mode: {config.jazz_mode}")
    else:
        print("   ✗ QuaverIntelligence not available")

    # Test PhraseIntelligence config
    print("\n4. PhraseIntelligence Config:")
    if _PHRASE_INTELLIGENCE_OK:
        config = get_default_structural_config()
        print(f"   window_size_ms: {config['window_size_ms']}")
        print(f"   window_overlap: {config['window_overlap']}")
        print(f"   boundary_rest_threshold_bars: {config['boundary_rest_threshold_bars']}")
        print(f"   boundary_pitch_jump_threshold: {config['boundary_pitch_jump_threshold']}")
    else:
        print("   ✗ PhraseIntelligence not available")

    # Test availability check
    print("\n5. Availability Check:")
    avail = is_epistemic_available()
    for name, available in avail.items():
        status = "✓" if available else "✗"
        print(f"   {status} {name}")

    # Test version info
    print("\n6. Version Info:")
    print(f"   Module: {version_info['module']}")
    print(f"   Version: {version_info['version']}")
    print(f"   Components: {', '.join(version_info['components'])}")

    print("\n" + "=" * 70)
    print(f"Epistemic module ready (v{__version__}).")
    print("=" * 70)
