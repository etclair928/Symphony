# =================================================================
# MODULE: agents/__init__.py
# DESCRIPTION: Agents module for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# EXPORTS:
#   - Base classes (BaseAgent, BaseDetectionAgent, etc.)
#   - Factory (AgentFactory, get_default_factory)
#   - Detection agents (RhythmEngine, PitchIntelligence, etc.)
#   - Analysis agents (GrooveFieldAnalyzer, TempoIntelligence, etc.)
#   - Quantization agents (Ritornello, VelocityMerge)
#   - Separation agents (DemucsSeparator, BSRoformerSeparator)
#   - Validation agents (ConsensusEngine)
#
# LAW 2 COMPLIANCE:
#   This module only imports from submodules - it does NOT import
#   other modules that could create circular dependencies.
#
# USAGE:
#   from agents import (
#       BaseAgent, AgentFactory,
#       RhythmEngine, PitchIntelligence,
#       GrooveFieldAnalyzer, TempoIntelligence,
#       Ritornello, VelocityMerge,
#       ConsensusEngine
#   )
# =================================================================

import importlib
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# ========================================================================
# Version and Metadata
# ========================================================================

__version__ = "5.6.1"
__author__ = "DeepSeek"

# ========================================================================
# Base Classes
# ========================================================================

from agents.base import (
    BaseAgent,
    BaseDetectionAgent,
    BaseAnalysisAgent,
    BaseQuantizationAgent,
    BaseValidationAgent,
    MemoryManagedMixin,
    FallbackDetector,
    FallbackAnalyzer,
    FallbackQuantizer,
    create_fallback_agent
)

# ========================================================================
# Factory
# ========================================================================

from agents.factory import (
    AgentFactory,
    FactoryConfig,
    get_default_factory,
    reset_default_factory,
    create_agent
)

# ========================================================================
# Detection Agents
# ========================================================================

from agents.detection import (
    # Rhythm
    RhythmEngine,
    OnsetDetectionConfig,
    create_rhythm_engine,
    # Pitch
    PitchIntelligence,
    PitchModelType,
    PitchDetectionConfig,
    create_pitch_intelligence,
    # Harmonic/Tonal
    HarmonicIntelligence,
    HarmonicConfig,
    create_harmonic_intelligence,
    # Drum
    DrumIntelligence,
    DrumDetectionConfig,
    create_drum_intelligence
)

# ========================================================================
# Analysis Agents
# ========================================================================

from agents.analysis import (
    # Groove Field
    GrooveFieldAnalyzer,
    GrooveConfig,
    create_groove_analyzer,
    # Voice Continuity
    VoiceContinuity,
    VoiceConfig,
    create_voice_continuity,
    # Tempo Intelligence
    TempoIntelligence,
    TempoConfig,
    create_tempo_intelligence,
    # Pulse Field
    PulseFieldAnalyzer,
    PulseFieldConfig,
    PulseStability,
    create_pulse_field_analyzer,
    # ReverseGeoCrypt
    ReverseGeoCrypt,
    ReverseGeoCryptConfig,
    GeometricLattice,
    Event,
    Anchor,
    EventType,
    create_reverse_geo_crypt,
    # Anechoic Ma
    AnechoicMa,
    AnechoicConfig,
    AnechoicMode,
    create_anechoic_ma,
    analyze_silence
)

# ========================================================================
# Quantization Agents
# ========================================================================

from agents.quantization import (
    # Ritornello
    Ritornello,
    RitornelloConfig,
    create_ritornello,
    # Velocity Merge
    VelocityMerge,
    VelocityMergeConfig,
    create_velocity_merge
)

# ========================================================================
# Separation Agents
# ========================================================================

from agents.separation import (
    DemucsSeparator,
    BSRoformerSeparator,
    HybridSeparator,
    FallbackSeparator
)

# ========================================================================
# Validation Agents
# ========================================================================

from agents.validation import (
    ConsensusEngine,
    ConsensusConfig,
    ConsensusStrategy,
    ConsensusOutcome,
    WitnessStatus,
    ConsensusRound,
    create_consensus_engine
)

# ========================================================================
# Module Docstring
# ========================================================================

__doc__ = """
Grimlock 5.0 Agents Module
===========================

This module contains all agents for the Grimlock 5.0 pipeline.
Agents are organized by responsibility:

Submodules:
-----------
detection/   - Audio-to-notes detection (rhythm, pitch, drums)
analysis/    - Musical analysis (groove, tempo, voice, pulse)
quantization/- Non-destructive timing quantization
separation/  - Stem separation (Demucs, BS_Roformer)
validation/  - Consensus and veto systems

Key Classes:
------------
BaseAgent           - Abstract base for all agents
AgentFactory        - Creates agents (breaks circular imports)

Detection:
  RhythmEngine      - Onset, beat, and pattern detection
  PitchIntelligence - Multi-model pitch detection (Omnizart, Basic Pitch, etc.)
  HarmonicIntelligence - Tonal/harmonic detection with key analysis
  DrumIntelligence  - Complete drum/percussion detection

Analysis:
  GrooveFieldAnalyzer - Relational physics swing detection
  VoiceContinuity     - Voice tracking in polyphonic texture
  TempoIntelligence   - Tempo detection with change tracking
  PulseFieldAnalyzer  - Probabilistic pulse grid
  ReverseGeoCrypt     - Tempo-free geometric lattice detection
  AnechoicMa          - Silence/resonance detection

Quantization:
  Ritornello        - Non-destructive quantization with groove awareness
  VelocityMerge     - Overlapping/adjacent note merging

Separation:
  DemucsSeparator   - Hybrid Transformer Demucs
  BSRoformerSeparator - BS_Roformer for drums/bass
  HybridSeparator   - Ensemble separation

Validation:
  ConsensusEngine   - Multi-witness epistemic veto system

Law 2 Compliance:
----------------
Agents do NOT import other agents. Communication is through:
- Shared FeatureBundle (immutable spectral evidence)
- StageResult (passed forward by Orchestrator)
- WitnessTestimony (testified to Scribe)

For detailed documentation, see individual submodule docstrings.
"""


# ========================================================================
# Submodule Import Check
# ========================================================================

def _check_submodules() -> Dict[str, bool]:
    """Check which submodules are available."""
    status: Dict[str, bool] = {}

    submodules = [
        "base",
        "factory",
        "detection",
        "analysis",
        "quantization",
        "separation",
        "validation"
    ]

    for submodule in submodules:
        try:
            importlib.import_module(f"agents.{submodule}")
            status[submodule] = True
        except ImportError as e:
            status[submodule] = False
            warnings.warn(f"Failed to import agents.{submodule}: {e}", ImportWarning)

    return status


# Lazy import for submodule check
_import_status: Optional[Dict[str, bool]] = None


def get_import_status() -> Dict[str, bool]:
    """Get the import status of all submodules."""
    global _import_status
    if _import_status is None:
        _import_status = _check_submodules()
    return _import_status


# ========================================================================
# Convenience Registry
# ========================================================================

# Map of agent types to their factory methods
AGENT_REGISTRY: Dict[str, Callable] = {
    # Detection
    "rhythm": lambda factory, **kw: factory.create_detector("rhythm"),
    "pitch": lambda factory, **kw: factory.create_detector("pitch"),
    "tonal": lambda factory, **kw: factory.create_detector("tonal"),
    "drum": lambda factory, **kw: factory.create_detector("drum"),

    # Analysis
    "groove": lambda factory, **kw: factory.create_analyzer("groove"),
    "voice": lambda factory, **kw: factory.create_analyzer("voice"),
    "tempo": lambda factory, **kw: factory.create_analyzer("tempo"),
    "pulse": lambda factory, **kw: factory.create_analyzer("pulse"),
    "reverse_geo": lambda factory, **kw: factory.create_analyzer("reverse_geo"),
    "anechoic": lambda factory, **kw: factory.create_analyzer("anechoic"),

    # Quantization
    "ritornello": lambda factory, **kw: factory.create_quantizer("ritornello"),
    "velocity_merge": lambda factory, **kw: factory.create_quantizer("velocity_merge"),

    # Separation
    "demucs": lambda factory, **kw: factory.create_separator("demucs"),
    "roformer": lambda factory, **kw: factory.create_separator("roformer"),
    "hybrid": lambda factory, **kw: factory.create_separator("hybrid"),

    # Validation
    "consensus": lambda factory, **kw: factory.create_validation_agent("consensus"),
    "scribe": lambda factory, **kw: factory.create_validation_agent("scribe"),
}


def create_agent_by_name(agent_name: str, config: dict = None, **kwargs):
    """
    Create an agent by its registered name.

    Args:
        agent_name: Name of the agent (e.g., 'pitch', 'tempo', 'ritornello')
        config: Configuration dictionary
        **kwargs: Additional arguments

    Returns:
        Agent instance
    """
    from agents.factory import get_default_factory

    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent name: {agent_name}. "
                         f"Available: {list(AGENT_REGISTRY.keys())}")

    factory = get_default_factory(config)
    return AGENT_REGISTRY[agent_name](factory, **kwargs)


def list_available_agents() -> Dict[str, List[str]]:
    """List all available agents by category."""
    return {
        "detection": ["rhythm", "pitch", "tonal", "drum"],
        "analysis": ["groove", "voice", "tempo", "pulse", "reverse_geo", "anechoic"],
        "quantization": ["ritornello", "velocity_merge"],
        "separation": ["demucs", "roformer", "hybrid"],
        "validation": ["consensus", "scribe"]
    }


# ========================================================================
# Aggregated Statistics
# ========================================================================

def get_agents_info() -> Dict[str, Any]:
    """Get aggregated information about all agents."""
    return {
        "version": __version__,
        "submodules": get_import_status(),
        "available_agents": list_available_agents(),
        "agent_count": sum(len(v) for v in list_available_agents().values())
    }


# ========================================================================
# Module Exports
# ========================================================================

__all__ = [
    # Version
    "__version__",

    # Base classes
    "BaseAgent",
    "BaseDetectionAgent",
    "BaseAnalysisAgent",
    "BaseQuantizationAgent",
    "BaseValidationAgent",
    "MemoryManagedMixin",
    "FallbackDetector",
    "FallbackAnalyzer",
    "FallbackQuantizer",
    "create_fallback_agent",

    # Factory
    "AgentFactory",
    "FactoryConfig",
    "get_default_factory",
    "reset_default_factory",
    "create_agent",

    # Detection
    "RhythmEngine",
    "OnsetDetectionConfig",
    "create_rhythm_engine",
    "PitchIntelligence",
    "PitchModelType",
    "PitchDetectionConfig",
    "create_pitch_intelligence",
    "HarmonicIntelligence",
    "HarmonicConfig",
    "create_harmonic_intelligence",
    "DrumIntelligence",
    "DrumDetectionConfig",
    "create_drum_intelligence",

    # Analysis
    "GrooveFieldAnalyzer",
    "GrooveConfig",
    "create_groove_analyzer",
    "VoiceContinuity",
    "VoiceConfig",
    "create_voice_continuity",
    "TempoIntelligence",
    "TempoConfig",
    "create_tempo_intelligence",
    "PulseFieldAnalyzer",
    "PulseFieldConfig",
    "PulseStability",
    "create_pulse_field_analyzer",
    "ReverseGeoCrypt",
    "ReverseGeoCryptConfig",
    "GeometricLattice",
    "Event",
    "Anchor",
    "EventType",
    "create_reverse_geo_crypt",
    "AnechoicMa",
    "AnechoicConfig",
    "AnechoicMode",
    "create_anechoic_ma",
    "analyze_silence",

    # Quantization
    "Ritornello",
    "RitornelloConfig",
    "create_ritornello",
    "VelocityMerge",
    "VelocityMergeConfig",
    "create_velocity_merge",

    # Separation
    "DemucsSeparator",
    "BSRoformerSeparator",
    "HybridSeparator",
    "FallbackSeparator",

    # Validation
    "ConsensusEngine",
    "ConsensusConfig",
    "ConsensusStrategy",
    "ConsensusOutcome",
    "WitnessStatus",
    "ConsensusRound",
    "create_consensus_engine",

    # Convenience
    "create_agent_by_name",
    "list_available_agents",
    "get_agents_info",
]

# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRIMLOCK 5.0 - AGENTS MODULE")
    print("=" * 70)

    print(f"\nVersion: {__version__}")

    print("\nSubmodule Import Status:")
    for submodule, available in get_import_status().items():
        status = "✓" if available else "✗"
        print(f"  {status} {submodule}")

    print("\nAvailable Agents by Category:")
    for category, agents in list_available_agents().items():
        print(f"  {category}: {', '.join(agents)}")

    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("=" * 70)

    print("""
# Import specific agent
from agents import RhythmEngine, PitchIntelligence, GrooveFieldAnalyzer

# Use factory for dependency injection
from agents import AgentFactory, get_default_factory

factory = get_default_factory()
rhythm = factory.create_detector("rhythm")
pitch = factory.create_detector("pitch")

# Create by name
from agents import create_agent_by_name

tempo = create_agent_by_name("tempo")
groove = create_agent_by_name("groove")

# Direct instantiation
from agents import create_rhythm_engine, create_pitch_intelligence

rhythm = create_rhythm_engine(onset_threshold=0.6)
pitch = create_pitch_intelligence(model="omnizart")
    """)

    print("=" * 70)