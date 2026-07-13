# =================================================================
# MODULE: agents/quantization/__init__.py
# DESCRIPTION: Quantization agents module.
#
# VERSION: 6.0
# UPDATED: 2026-07-12
#
# EXPORTS:
#   - TemporalLattice: snap-to-grid quantization (the pipeline's real
#     "quantization" stage)
#   - VelocityMerge: overlapping/adjacent note merging (same-pitch,
#     gap-based)
#   - Ritornello: musical-coherence REPAIR pass that runs after both of
#     those (sustain recovery, micro-note purge, tie reconstruction,
#     pitch-wobble collapse, trouble-mapping + CorrectionScribe) - NOT a
#     quantizer. See agents/quantization/ritornello.py's module docstring
#     for the full v6.0 restoration rationale.
#
# LAW 2 COMPLIANCE:
#   This module only imports quantization agents - it does NOT import
#   other modules that could create circular dependencies. Agents are
#   independent.
#
# NON-DESTRUCTIVE AUDIT LAW:
#   Original timestamps (start_ms, end_ms) are NEVER overwritten.
#   Quantized/repaired values go to snapped_start_ms, snapped_end_ms.
#
# NOTE: orchestration/pipeline.py imports TemporalLattice/VelocityMerge/
# Ritornello directly from their own submodules, never through this
# package __init__ - the registry/factory helpers this file used to carry
# (create_quantization_pipeline, run_quantization_pipeline,
# synthesize_timing_lattice, get_quantization_agent, ...) were built
# around the pre-6.0 Ritornello's different API (a StageResult-producing
# .run(audio_buffer, context) agent) and were never on any real execution
# path. Removed rather than patched to avoid carrying convenience
# scaffolding nothing actually calls.
# =================================================================

from typing import Dict, List, Any, Optional

# Import main agent classes
from agents.quantization.ritornello import (
    Ritornello, RitornelloConfig, MeasureMap, RevisionEvent, TroubleMeasure,
    SustainRecovery, MicroNotePurge, TieReconstruction, PitchWobbleCollapse,
    TroubleMap, CorrectionScribe,
)
from agents.quantization.temporal_lattice import (
    TemporalLattice, TemporalLatticeQuantizer, WitnessTestimony,
    WitnessPriority, LatticeConfidence, create_lattice_from_pipeline_results,
    quick_lattice_test,
)
from agents.quantization.velocity_merge import (
    VelocityMerge, VelocityMergeConfig, create_velocity_merge,
)

# Import enums and types
from core.order_types import (
    NoteEvent,
    QuantizationStrategy,
    QuantizationStage,
    MergeStrategy,
    MergeCandidate,
    VelocityMergeResult,
    AudioContext,
    GrooveField,
    TempoMap,
    PulseField,
)

__version__ = "6.0"
__author__ = "Claude"

__all__ = [
    # Main agent classes
    "Ritornello",
    "TemporalLattice",
    "TemporalLatticeQuantizer",
    "VelocityMerge",

    # Ritornello (repair pass) supporting types
    "RitornelloConfig",
    "MeasureMap",
    "RevisionEvent",
    "TroubleMeasure",
    "SustainRecovery",
    "MicroNotePurge",
    "TieReconstruction",
    "PitchWobbleCollapse",
    "TroubleMap",
    "CorrectionScribe",

    # TemporalLattice types
    "WitnessTestimony",
    "WitnessPriority",
    "LatticeConfidence",
    "create_lattice_from_pipeline_results",
    "quick_lattice_test",

    # VelocityMerge
    "VelocityMergeConfig",
    "create_velocity_merge",

    # Enums and types
    "QuantizationStrategy",
    "QuantizationStage",
    "MergeStrategy",
    "MergeCandidate",
    "VelocityMergeResult",
    "GrooveField",
    "TempoMap",
    "PulseField",
]


# ========================================================================
# Module Initialization Check
# ========================================================================

def _check_imports() -> bool:
    """Verify all quantization agents are importable."""
    missing = []

    try:
        from agents.quantization.ritornello import Ritornello
    except ImportError as e:
        missing.append(f"ritornello: {e}")

    try:
        from agents.quantization.temporal_lattice import TemporalLattice
    except ImportError as e:
        missing.append(f"temporal_lattice: {e}")

    try:
        from agents.quantization.velocity_merge import VelocityMerge
    except ImportError as e:
        missing.append(f"velocity_merge: {e}")

    if missing:
        import warnings
        warnings.warn(f"Some quantization agents failed to import: {missing}", ImportWarning)
        return False

    return True


_IMPORTS_OK = _check_imports()

version_info = {
    "module": "agents.quantization",
    "version": __version__,
    "imports_ok": _IMPORTS_OK,
}


# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    print(f"Version: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")
    print()
    print("TemporalLattice self-test:")
    quick_lattice_test()
    print()
    print("Ritornello self-test: run `python -m agents.quantization.ritornello` directly.")
