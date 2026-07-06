# =================================================================
# MODULE: agents/validation/__init__.py
# DESCRIPTION: Validation agents module for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# EXPORTS:
#   - ConsensusEngine: Multi-witness epistemic veto system
#
# LAW 2 COMPLIANCE:
#   This module only imports validation agents - it does NOT import other
#   modules that could create circular dependencies. Agents are independent.
#
# LAW OF EPISTEMIC VETO:
#   "No single witness can declare a truth; any qualified witness
#    can declare a falsehood."
#
#   The ConsensusEngine implements this law as the final authority
#   for resolving conflicts between multiple agents (witnesses).
#
# USAGE:
#   from agents.validation import (
#       ConsensusEngine, ConsensusConfig,
#       ConsensusStrategy, ConsensusOutcome,
#       WitnessStatus, ConsensusRound,
#       create_consensus_engine
#   )
# =================================================================

from typing import Dict, List, Any, Optional, Union, Tuple

# Import main agent classes
from agents.validation.consensus_engine import (
    ConsensusEngine,
    ConsensusConfig,
    ConsensusStrategy,
    ConsensusOutcome,
    WitnessStatus,
    ConsensusRound,
    WitnessRegistration
)

# Import core types needed for type hints
from core.order_types import NoteEvent, AudioContext, ConsensusPackage, SourceType

# Import convenience factory functions
from agents.validation.consensus_engine import create_consensus_engine

# Version identifier for this module group
__version__ = "5.6.1"

# Module author and metadata
__author__ = "DeepSeek"

__all__ = [
    # Main agent classes
    "ConsensusEngine",

    # Configuration classes
    "ConsensusConfig",

    # Enums and types
    "ConsensusStrategy",
    "ConsensusOutcome",
    "WitnessStatus",

    # Data classes
    "ConsensusRound",
    "WitnessRegistration",

    # Factory functions
    "create_consensus_engine",
]

# Module-level docstring with quick reference
__doc__ = """
Grimlock 5.0 Validation Agents
===============================

This module provides validation agents for the Grimlock 5.0 pipeline.
Each agent is independent, follows Law 2 (no circular imports), and
implements the Epistemic Veto system.

Available Agents:
-----------------
ConsensusEngine:
    Multi-witness epistemic veto system.

    Philosophy: "No single witness can declare a truth; any qualified
                 witness can declare a falsehood."

    This engine acts as the final authority for resolving conflicts
    between multiple agents (witnesses). It implements the epistemic
    veto rule: if any qualified witness declares a falsehood, the
    consensus is false.

    Features:
    - Multi-witness testimony collection
    - Confidence-weighted consensus
    - Epistemic veto (any veto = false)
    - Witness qualification (confidence threshold)
    - Forensic audit of consensus decisions
    - Multiple consensus strategies:
        * ANY_VETO_WINS (default) - Epistemic veto
        * UNANIMOUS_REQUIRED - All witnesses must agree
        * MAJORITY_VOTE - Simple majority wins
        * WEIGHTED_BY_CONFIDENCE - Weighted by witness confidence

    Usage:
        engine = create_consensus_engine()
        engine.register_witness(SourceType.PITCH, pitch_agent)
        engine.register_witness(SourceType.TONAL, tonal_agent)
        consensus = engine.reach_consensus(target_note=note)

        if consensus.veto_triggered_by:
            print(f"Veto by {consensus.veto_triggered_by}: {consensus.veto_reason}")

Law 2 Compliance:
----------------
Agents do NOT import other agents. All communication is through:
- WitnessTestimony (collected from agents)
- ConsensusPackage (returned to Orchestrator)
- WitnessRegistration (registration records)

Epistemic Veto Law:
------------------
Any qualified witness can declare a falsehood.
The ConsensusEngine would rather miss a ghost note than hallucinate a false one.

Qualified witness criteria:
- Confidence >= LOW threshold (0.25)
- Not blacklisted
- Has valid testimony

5 Strategies Integration:
------------------------
1. Streaming Windows: Process testimonies in batches
2. Disk-Backed: Consensus history can be persisted
3. mmap for Audit: Forensic rounds support random access
4. GC Discipline: staggered_gc() between consensus rounds
5. Sparse Representations: Only testimony metadata, not raw audio

For detailed documentation, see individual module docstrings.
"""


# ========================================================================
# Module Initialization Check
# ========================================================================

def _check_imports() -> bool:
    """Verify all validation agents are importable."""
    missing = []

    try:
        from agents.validation.consensus_engine import ConsensusEngine
    except ImportError as e:
        missing.append(f"consensus_engine: {e}")

    if missing:
        import warnings
        warnings.warn(f"Some validation agents failed to import: {missing}", ImportWarning)
        return False

    return True


# Run import check on module load
_IMPORTS_OK = _check_imports()

# ========================================================================
# Agent Registry for Factory Pattern
# ========================================================================

VALIDATION_AGENT_REGISTRY = {
    "consensus": {
        "class": ConsensusEngine,
        "factory": create_consensus_engine,
        "config_class": ConsensusConfig,
        "description": "Multi-witness epistemic veto system"
    }
}


def get_validation_agent(agent_name: str, **kwargs):
    """
    Factory method to get a validation agent by name.

    Args:
        agent_name: One of 'consensus'
        **kwargs: Configuration parameters for the agent

    Returns:
        Configured agent instance

    Example:
        consensus = get_validation_agent('consensus', strategy=ConsensusStrategy.ANY_VETO_WINS)
    """
    if agent_name not in VALIDATION_AGENT_REGISTRY:
        raise ValueError(f"Unknown validation agent: {agent_name}. "
                         f"Choose from {list(VALIDATION_AGENT_REGISTRY.keys())}")

    factory = VALIDATION_AGENT_REGISTRY[agent_name]["factory"]
    return factory(**kwargs)


def list_validation_agents() -> Dict[str, Dict[str, Any]]:
    """List all available validation agents with metadata."""
    return {
        name: {
            "class": info["class"].__name__,
            "description": info["description"],
            "config": info["config_class"].__name__
        }
        for name, info in VALIDATION_AGENT_REGISTRY.items()
    }


# ========================================================================
# Integration Helpers
# ========================================================================

def create_validation_pipeline(
        consensus_engine: Optional[ConsensusEngine] = None,
        strategy: ConsensusStrategy = ConsensusStrategy.ANY_VETO_WINS
) -> Dict[str, Any]:
    """
    Create a coordinated validation pipeline.

    Currently only includes ConsensusEngine, but designed for expansion.

    Args:
        consensus_engine: Optional ConsensusEngine instance
        strategy: Consensus strategy

    Returns:
        Dictionary containing configured agents in execution order
    """
    return {
        "consensus": consensus_engine or create_consensus_engine(strategy=strategy),
        "execution_order": ["consensus"]
    }


def run_validation_pipeline(pipeline: Dict[str, Any],
                            target_note: Optional[NoteEvent] = None,
                            audio_context: Optional[AudioContext] = None,
                            witnesses: Optional[List[SourceType]] = None) -> ConsensusPackage:
    """
    Run the full validation pipeline.

    Args:
        pipeline: Pipeline from create_validation_pipeline()
        target_note: The note being evaluated
        audio_context: Optional audio context
        witnesses: Optional subset of witnesses to consult

    Returns:
        ConsensusPackage with final decision
    """
    consensus_engine = pipeline["consensus"]
    return consensus_engine.reach_consensus(
        target_note=target_note,
        audio_context=audio_context,
        witnesses=witnesses
    )


# ========================================================================
# Module Metadata
# ========================================================================

# Export version info
version_info = {
    "module": "agents.validation",
    "version": __version__,
    "agents": list(VALIDATION_AGENT_REGISTRY.keys()),
    "imports_ok": _IMPORTS_OK
}

# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRIMLOCK 5.0 - VALIDATION AGENTS")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")

    print("\nAvailable Agents:")
    for name, info in list_validation_agents().items():
        print(f"  {name:12} - {info['description']}")

    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("=" * 70)

    print("""
# Import specific agent
from agents.validation import ConsensusEngine

# Or use factory function
from agents.validation import create_consensus_engine

# Dynamic agent creation
from agents.validation import get_validation_agent, list_validation_agents

agents = list_validation_agents()
consensus = get_validation_agent('consensus', strategy=ConsensusStrategy.ANY_VETO_WINS)

# Register witnesses
for source, agent in witnesses.items():
    consensus.register_witness(source, agent)

# Reach consensus on a note
result = consensus.reach_consensus_on_note(my_note)

# Check for veto
if result.veto_triggered_by:
    print(f"Veto by {result.veto_triggered_by}: {result.veto_reason}")

# Full validation pipeline
from agents.validation import create_validation_pipeline, run_validation_pipeline

pipeline = create_validation_pipeline()
consensus_result = run_validation_pipeline(pipeline, target_note=my_note)
    """)

    print("\n" + "=" * 70)
    print("LAW OF EPISTEMIC VETO")
    print("=" * 70)
    print("""
"No single witness can declare a truth; any qualified witness
 can declare a falsehood."

This is the cornerstone of Grimlock's truth-seeking architecture:
- Witnesses testify (provide evidence)
- Consensus Engine adjudicates (applies veto rule)
- Veto overrides all other votes

The engine would rather miss a ghost note than hallucinate a false one.
    """)

    print("=" * 70)