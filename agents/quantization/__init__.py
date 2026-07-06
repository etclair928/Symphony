# =================================================================
# MODULE: agents/quantization/__init__.py
# DESCRIPTION: Quantization agents module for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-19
#
# EXPORTS:
#   - Ritornello: Non-destructive quantization (preserves original timestamps)
#   - TemporalLattice: Final Witness for timing (synthesizes forensic evidence)
#   - VelocityMerge: Overlapping and adjacent note merging
#
# LAW 2 COMPLIANCE:
#   This module only imports quantization agents - it does NOT import other
#   modules that could create circular dependencies. Agents are independent.
#
# NON-DESTRUCTIVE AUDIT LAW:
#   Original timestamps (start_ms, end_ms) are NEVER overwritten.
#   Quantized values go to snapped_start_ms, snapped_end_ms.
#
# PHILOSOPHY:
#   "The Temporal Lattice is the final authority on timing. It takes
#    the chaotic, forensic evidence from Analysis Agents and turns it
#    into a set of musical rules that the Quantizer can execute."
#
# USAGE:
#   from agents.quantization import (
#       Ritornello, TemporalLattice, VelocityMerge,
#       RitornelloConfig, TemporalLatticeConfig, VelocityMergeConfig,
#       QuantizationStrategy, MergeStrategy, LatticeConfidence,
#       create_ritornello, create_temporal_lattice, create_velocity_merge,
#       create_lattice_from_pipeline_results
#   )
# =================================================================

from typing import Dict, List, Any, Optional, Union, Tuple

# Import main agent classes
from agents.quantization.ritornello import Ritornello, RitornelloConfig
from agents.quantization.temporal_lattice import (
    TemporalLattice, TemporalLatticeQuantizer, WitnessTestimony,
    WitnessPriority, LatticeConfidence
)
from agents.quantization.velocity_merge import VelocityMerge, VelocityMergeConfig

# Import enums and types
from core.order_types import (
    NoteEvent,
    QuantizationStrategy,
    QuantizationStage,
    MergeStrategy,
    MergeCandidate,
    MultiStageQuantization,
    QuantizationPass,
    VelocityMergeResult,
    AudioContext,
    GrooveField,
    TempoMap,
    PulseField
)

# Import convenience factory functions
from agents.quantization.ritornello import create_ritornello
from agents.quantization.temporal_lattice import (
    create_lattice_from_pipeline_results, quick_lattice_test
)
from agents.quantization.velocity_merge import create_velocity_merge

# Version identifier for this module group
__version__ = "5.6.1"

# Module author and metadata
__author__ = "DeepSeek"

__all__ = [
    # Main agent classes
    "Ritornello",
    "TemporalLattice",
    "TemporalLatticeQuantizer",
    "VelocityMerge",

    # Configuration classes
    "RitornelloConfig",
    "VelocityMergeConfig",

    # TemporalLattice types
    "WitnessTestimony",
    "WitnessPriority",
    "LatticeConfidence",

    # Enums and types
    "QuantizationStrategy",
    "QuantizationStage",
    "MergeStrategy",
    "MergeCandidate",
    "MultiStageQuantization",
    "QuantizationPass",
    "VelocityMergeResult",
    "GrooveField",
    "TempoMap",
    "PulseField",

    # Factory functions
    "create_ritornello",
    "create_temporal_lattice",
    "create_velocity_merge",
    "create_lattice_from_pipeline_results",

    # Utility functions
    "quick_lattice_test",
]

# Module-level docstring with quick reference
__doc__ = """
Grimlock 5.0 Quantization Agents
=================================

This module provides quantization agents for the Grimlock 5.0 pipeline.
Each agent is independent, follows Law 2 (no circular imports), and
produces immutable transformations with complete forensic audit trails.

Available Agents:
-----------------
Ritornello (Non-Destructive Quantizer):
    Quantizes note timing while preserving original timestamps.

    Philosophy: "The original signal is the only source of truth;
                 all transformations must be reversible."

    Features:
    - Multi-stage quantization (coarse → medium → fine → micro)
    - Groove-aware snapping (preserves intentional feel)
    - Pulse field integration (probabilistic grid)
    - Forensic audit trail (every snap recorded)
    - Confidence penalties based on snap distance

    Usage:
        quantizer = create_ritornello()
        quantized = quantizer.quantize(notes, groove_field=groove)

TemporalLattice (Final Witness for Timing):
    Translates forensic observations from Analysis Agents into a 
    playable, human-aligned grid for quantization.

    Philosophy: "Any qualified witness can declare a falsehood, but
                 the Temporal Lattice synthesizes truth from all testimony."

    Features:
    - Synthesizes evidence from multiple forensic witnesses
    - Physical geometry from ReverseGeoCrypt (grid spacing)
    - The "Feel" from GrooveField (swing, phase delta)
    - The "Anchor" from TempoIntelligence (where is beat 0?)
    - The "Pocket" (microtiming window preserving human nuance)
    - Confidence-weighted nudge (LERP between raw and grid)
    - Ghost note preservation (velocity < 30 kept raw)

    Key Concepts:
    1. Physical Geometry - The grid spacing from ReverseGeoCrypt
    2. The Feel - Swing ratio and phase delta from GrooveField
    3. The Anchor - Tempo and phase anchor from TempoIntelligence
    4. The Pocket - Microtiming window where human nuance is preserved
    5. The Nudge - Quantization strength (LERP between raw and grid)

    Usage:
        # Assemble from all witnesses
        lattice = TemporalLattice.assemble_from_witnesses(evidence)

        # Or use individual witnesses
        lattice = TemporalLattice.from_tempo_intelligence(tempo_map)
        lattice = TemporalLattice.from_pulse_field(pulse_field)
        lattice = TemporalLattice.from_groove_field(groove, bpm)
        lattice = TemporalLattice.from_reverse_geo_crypt(period_ms, confidence)

        # Apply to notes
        quantized = lattice.quantize_notes(notes)

        # Or use the wrapper
        quantizer = TemporalLatticeQuantizer(lattice)
        result = quantizer.quantize(notes)

    Resolution Algorithm (6 Steps):
    1. Find nearest lattice point
    2. Apply swing bias to offbeats
    3. Add global phase delta (Dilla shift)
    4. Check if in "pocket" (preserve if within window)
    5. Apply confidence-adjusted nudge
    6. Linear interpolation between raw and target

VelocityMerge:
    Merges overlapping or adjacent notes intelligently.

    Philosophy: "Overlapping notes are usually detection errors or
                 voice stealing. Merging improves readability."

    Features:
    - Multiple merge strategies (weighted average, max velocity, etc.)
    - Overlap and adjacency detection
    - Forensic audit trail of all merges
    - Ghost note preservation
    - Pitch-based grouping option

    Usage:
        merger = create_velocity_merge()
        result = merger.merge_overlaps(notes)

Law 2 Compliance:
----------------
Agents do NOT import other agents. All communication is through:
- NoteEvent objects (immutable, with snapped_* fields)
- StageResult (passed forward by Orchestrator)
- WitnessTestimony (testified to Scribe)

Non-Destructive Audit Law:
-------------------------
Original timestamps (start_ms, end_ms) are NEVER overwritten.
Quantized values go to snapped_start_ms, snapped_end_ms.
The original is always there for forensic verification.

TemporalLattice Integration with Analysis Agents:
------------------------------------------------
The lattice expects evidence from the following forensic witnesses:

    evidence = {
        'reverse_geo_lattice_period': 405.1,  # From ReverseGeoCrypt
        'reverse_geo_confidence': 0.85,
        'groove_phase_delta': -1346.0,       # From GrooveField
        'groove_type': 'wide_swing',          # or 'dilla_pocket', 'laid_back'
        'groove_confidence': 0.78,
        'pulse_bpm': 74.4,                   # From TempoIntelligence
        'pulse_confidence': 0.82,
        'pulse_phase_anchor_ms': 0.0
    }

5 Strategies Integration:
------------------------
1. Streaming Windows: Agents process event batches
2. Disk-Backed: Result caching available
3. mmap for Audit: Forensic records support random access
4. GC Discipline: staggered_gc() between stages
5. Sparse Representations: Events only, no audio

For detailed documentation, see individual module docstrings.
"""


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


# Run import check on module load
_IMPORTS_OK = _check_imports()

# ========================================================================
# Agent Registry for Factory Pattern
# ========================================================================

QUANTIZATION_AGENT_REGISTRY = {
    "ritornello": {
        "class": Ritornello,
        "factory": create_ritornello,
        "config_class": RitornelloConfig,
        "description": "Non-destructive quantization with groove awareness"
    },
    "temporal_lattice": {
        "class": TemporalLattice,
        "factory": lambda **kwargs: create_lattice_from_pipeline_results(**kwargs),
        "config_class": None,  # No dedicated config class; uses parameters
        "description": "Final Witness for timing - synthesizes forensic evidence"
    },
    "velocity_merge": {
        "class": VelocityMerge,
        "factory": create_velocity_merge,
        "config_class": VelocityMergeConfig,
        "description": "Overlapping and adjacent note merging"
    }
}


def get_quantization_agent(agent_name: str, **kwargs):
    """
    Factory method to get a quantization agent by name.

    Args:
        agent_name: One of 'ritornello', 'temporal_lattice', 'velocity_merge'
        **kwargs: Configuration parameters for the agent

    Returns:
        Configured agent instance

    Example:
        ritornello = get_quantization_agent('ritornello', max_snap_ms=50)
        lattice = get_quantization_agent('temporal_lattice', tempo_bpm=120.0)
        merger = get_quantization_agent('velocity_merge', overlap_tolerance_ms=15)
    """
    if agent_name not in QUANTIZATION_AGENT_REGISTRY:
        raise ValueError(f"Unknown quantization agent: {agent_name}. "
                         f"Choose from {list(QUANTIZATION_AGENT_REGISTRY.keys())}")

    factory = QUANTIZATION_AGENT_REGISTRY[agent_name]["factory"]
    return factory(**kwargs)


def list_quantization_agents() -> Dict[str, Dict[str, Any]]:
    """List all available quantization agents with metadata."""
    return {
        name: {
            "class": info["class"].__name__,
            "description": info["description"],
            "config": info["config_class"].__name__ if info["config_class"] else "N/A"
        }
        for name, info in QUANTIZATION_AGENT_REGISTRY.items()
    }


def create_temporal_lattice(
        tempo_bpm: Optional[float] = None,
        tempo_confidence: float = 0.5,
        groove_phase_delta: float = 0.0,
        groove_type: str = "straight",
        lattice_period_ms: float = 0.0,
        microtiming_window_ms: float = 15.0,
        quantize_strength: float = 0.35,
        preserve_ghost_notes: bool = True
) -> TemporalLattice:
    """
    Create a TemporalLattice with the given parameters.

    This is a convenience wrapper around create_lattice_from_pipeline_results.

    Args:
        tempo_bpm: Base tempo in beats per minute
        tempo_confidence: Confidence in tempo detection (0-1)
        groove_phase_delta: Global phase shift in milliseconds
        groove_type: 'straight', 'wide_swing', 'dilla_pocket', 'laid_back'
        lattice_period_ms: Grid period in milliseconds (auto-calculated from tempo if 0)
        microtiming_window_ms: Pocket size for preserving raw timing
        quantize_strength: How hard to nudge (0=raw, 1=rigid)
        preserve_ghost_notes: Keep low-velocity notes raw

    Returns:
        Configured TemporalLattice instance

    Example:
        lattice = create_temporal_lattice(
            tempo_bpm=74.4,
            tempo_confidence=0.82,
            groove_phase_delta=-1346.0,
            groove_type='wide_swing',
            microtiming_window_ms=15.0
        )
    """
    return create_lattice_from_pipeline_results(
        tempo_bpm=tempo_bpm,
        tempo_confidence=tempo_confidence,
        groove_phase_delta=groove_phase_delta,
        groove_type=groove_type,
        lattice_period_ms=lattice_period_ms,
        microtiming_window_ms=microtiming_window_ms,
        quantize_strength=quantize_strength
    )


# ========================================================================
# Integration Helpers (Cross-Agent Coordination)
# ========================================================================

def create_quantization_pipeline(
        ritornello: Optional[Ritornello] = None,
        temporal_lattice: Optional[TemporalLattice] = None,
        velocity_merge: Optional[VelocityMerge] = None,
        strategy: QuantizationStrategy = QuantizationStrategy.MULTI_STAGE,
        merge_strategy: MergeStrategy = MergeStrategy.WEIGHTED_AVERAGE
) -> Dict[str, Any]:
    """
    Create a coordinated quantization pipeline.

    This helper wires together quantization agents in the recommended order:
    1. VelocityMerge first (clean up overlapping notes)
    2. TemporalLattice second (synthesize timing evidence)
    3. Ritornello third (apply quantization using the lattice)

    The TemporalLattice serves as the "Final Witness" - it synthesizes
    forensic evidence from analysis agents before quantization.

    Args:
        ritornello: Optional Ritornello instance
        temporal_lattice: Optional TemporalLattice instance (creates default if None)
        velocity_merge: Optional VelocityMerge instance
        strategy: Quantization strategy for Ritornello
        merge_strategy: Merge strategy for VelocityMerge

    Returns:
        Dictionary containing configured agents in execution order
    """
    return {
        "velocity_merge": velocity_merge or create_velocity_merge(),
        "temporal_lattice": temporal_lattice or create_temporal_lattice(),
        "ritornello": ritornello or create_ritornello(),
        "execution_order": ["velocity_merge", "temporal_lattice", "ritornello"],
        "strategy": strategy,
        "merge_strategy": merge_strategy
    }


def run_quantization_pipeline(pipeline: Dict[str, Any], notes: List[NoteEvent],
                              context: Optional[AudioContext] = None,
                              evidence: Optional[Dict[str, Any]] = None) -> VelocityMergeResult:
    """
    Run the full quantization pipeline with forensic evidence synthesis.

    Args:
        pipeline: Pipeline from create_quantization_pipeline()
        notes: List of NoteEvent objects
        context: Optional context with additional parameters
        evidence: Optional forensic evidence from analysis agents for TemporalLattice

    Returns:
        Final VelocityMergeResult after all quantization
    """
    current_notes = notes

    # Step 1: Merge overlapping notes
    merger = pipeline["velocity_merge"]
    merge_result = merger.merge_overlaps(current_notes)
    current_notes = merge_result.merged_notes

    # Step 2: Update TemporalLattice with evidence if provided
    lattice = pipeline["temporal_lattice"]
    if evidence and isinstance(lattice, TemporalLattice):
        lattice = TemporalLattice.assemble_from_witnesses(evidence)

    # Step 3: Apply lattice quantization
    quantized_notes = lattice.quantize_notes(current_notes)

    # Step 4: Apply Ritornello (if needed for additional passes)
    ritornello = pipeline["ritornello"]

    # Build mock context if none provided
    if context is None:
        from core.order_types import AudioContext
        context = AudioContext(
            file_path="",
            original_sample_rate=16000,
            duration_seconds=10,
            num_channels_original=1,
            file_hash_sha256=None
        )
        context.events = quantized_notes

    result = ritornello.run(np.array([]), context)

    # Return result with merge history preserved
    return VelocityMergeResult(
        original_notes=merge_result.original_notes,
        merged_notes=result.events,
        merges_performed=merge_result.merges_performed,
        notes_removed=merge_result.notes_removed
    )


def synthesize_timing_lattice(evidence: Dict[str, Any]) -> TemporalLattice:
    """
    Synthesize a TemporalLattice from forensic evidence.

    This is the primary way to create a lattice from analysis agent output.

    Args:
        evidence: Dictionary containing testimony from analysis agents:
            - reverse_geo_lattice_period: Period in ms from ReverseGeoCrypt
            - reverse_geo_confidence: Confidence (0-1)
            - groove_phase_delta: Global phase shift in ms from GrooveField
            - groove_type: 'straight', 'wide_swing', 'dilla_pocket', 'laid_back'
            - groove_confidence: Confidence (0-1)
            - pulse_bpm: Tempo in BPM from TempoIntelligence
            - pulse_confidence: Confidence (0-1)
            - pulse_phase_anchor_ms: Beat 0 position in ms

    Returns:
        TemporalLattice ready for quantization

    Example:
        evidence = {
            'reverse_geo_lattice_period': 405.1,
            'reverse_geo_confidence': 0.85,
            'groove_phase_delta': -1346.0,
            'groove_type': 'wide_swing',
            'groove_confidence': 0.78,
            'pulse_bpm': 74.4,
            'pulse_confidence': 0.82,
            'pulse_phase_anchor_ms': 0.0
        }
        lattice = synthesize_timing_lattice(evidence)
        quantized = lattice.quantize_notes(my_notes)
    """
    return TemporalLattice.assemble_from_witnesses(evidence)


# ========================================================================
# Module Metadata
# ========================================================================

# Export version info
version_info = {
    "module": "agents.quantization",
    "version": __version__,
    "agents": list(QUANTIZATION_AGENT_REGISTRY.keys()),
    "imports_ok": _IMPORTS_OK
}

# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    import numpy as np

    print("\n" + "=" * 70)
    print("GRIMLOCK 5.0 - QUANTIZATION AGENTS")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")

    print("\nAvailable Agents:")
    for name, info in list_quantization_agents().items():
        print(f"  {name:16} - {info['description']}")

    print("\n" + "=" * 70)
    print("TemporalLattice Test (Jazz Trio Example)")
    print("=" * 70)

    # Run the lattice test
    quick_lattice_test()

    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("=" * 70)

    print("""
# Import specific agents
from agents.quantization import (
    Ritornello, TemporalLattice, VelocityMerge,
    create_temporal_lattice, synthesize_timing_lattice
)

# Create from forensic evidence
evidence = {
    'reverse_geo_lattice_period': 405.1,
    'reverse_geo_confidence': 0.85,
    'groove_phase_delta': -1346.0,
    'groove_type': 'wide_swing',
    'groove_confidence': 0.78,
    'pulse_bpm': 74.4,
    'pulse_confidence': 0.82
}
lattice = synthesize_timing_lattice(evidence)

# Apply to notes
quantized_notes = lattice.quantize_notes(my_notes)

# Create from parameters
lattice = create_temporal_lattice(
    tempo_bpm=120.0,
    groove_type='dilla_pocket',
    quantize_strength=0.35
)

# Full quantization pipeline with evidence
from agents.quantization import create_quantization_pipeline, run_quantization_pipeline

pipeline = create_quantization_pipeline()
result = run_quantization_pipeline(pipeline, my_notes, evidence=evidence)

# Dynamic agent creation
from agents.quantization import get_quantization_agent

lattice = get_quantization_agent('temporal_lattice', tempo_bpm=74.4, groove_type='wide_swing')
    """)

    print("\n" + "=" * 70)
    print("NON-DESTRUCTIVE AUDIT LAW")
    print("=" * 70)
    print("""
Original timestamps are NEVER overwritten:
    - start_ms / end_ms     ← Original (preserved forever)
    - snapped_start_ms      ← Quantized (if snapped)
    - snapped_end_ms        ← Quantized (if snapped)

Every transformation is reversible. The MusicBox logs all decisions.
    """)

    print("=" * 70)
    print("TEMPORAL LATTICE PHILOSOPHY")
    print("=" * 70)
    print("""
"The Temporal Lattice is the final authority on timing. It takes
 the chaotic, forensic evidence from Analysis Agents and turns it
 into a set of musical rules that the Quantizer can execute."

"Any qualified witness can declare a falsehood, but the Temporal
 Lattice synthesizes truth from all testimony."
    """)

    print("=" * 70)