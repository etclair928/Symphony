# =================================================================
# MODULE: agents/detection/__init__.py
# DESCRIPTION: Detection agents module initialization for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# EXPORTS:
#   - RhythmEngine: Onset, beat, and pattern detection
#   - PitchIntelligence: Multi-model pitch detection (Omnizart, Basic Pitch, CREPE, SPICE, Librosa)
#   - HarmonicIntelligence: Harmonic series validation and key detection
#   - DrumIntelligence: Complete drum/percussion detection with ensemble voting
#
# LAW 2 COMPLIANCE:
#   This module only imports agents - it does NOT import other modules
#   that could create circular dependencies. Agents are independent.
#
# USAGE:
#   from agents.detection import (
#       RhythmEngine, PitchIntelligence, HarmonicIntelligence, DrumIntelligence,
#       create_rhythm_engine, create_pitch_intelligence,
#       create_harmonic_intelligence, create_drum_intelligence
#   )
# =================================================================

from typing import Dict, List, Any, Optional

# Import main agent classes
from agents.detection.rhythm_engine import RhythmEngine
from agents.detection.pitch_intelligence import PitchIntelligence, PitchModelType, PitchDetectionConfig
from agents.detection.harmonic_intelligence import HarmonicIntelligence, HarmonicConfig
from agents.detection.drum_intelligence import DrumIntelligence, DrumDetectionConfig

# Import convenience factory functions
from agents.detection.rhythm_engine import create_rhythm_engine
from agents.detection.pitch_intelligence import create_pitch_intelligence
from agents.detection.harmonic_intelligence import create_harmonic_intelligence
from agents.detection.drum_intelligence import create_drum_intelligence

# Import types for type hints and configuration
from agents.detection.rhythm_engine import OnsetDetectionConfig, RhythmEngineState
from agents.detection.pitch_intelligence import PitchModelType, PitchDetectionConfig
from agents.detection.harmonic_intelligence import HarmonicConfig, HarmonicValidationResult, EvidenceType
from agents.detection.drum_intelligence import (
    DrumDetectionConfig, DrumArticulation, DrumMicrotiming,
    DetectionPass, ConfidenceComponents, ForensicDrumEvent
)

# Import FeatureBundle from core.order_types (not from pitch_intelligence)
from core.order_types import FeatureBundle

# Version identifier for this module group
__version__ = "5.6.1"

# Module author and metadata
__author__ = "DeepSeek"
__all__ = [
    # Main agent classes
    "RhythmEngine",
    "PitchIntelligence",
    "HarmonicIntelligence",
    "DrumIntelligence",

    # Factory functions
    "create_rhythm_engine",
    "create_pitch_intelligence",
    "create_harmonic_intelligence",
    "create_drum_intelligence",

    # Configuration classes
    "OnsetDetectionConfig",
    "PitchDetectionConfig",
    "HarmonicConfig",
    "DrumDetectionConfig",

    # Enums and types
    "PitchModelType",
    "DrumArticulation",
    "DrumMicrotiming",
    "DetectionPass",
    "ConfidenceComponents",
    "ForensicDrumEvent",
    "HarmonicValidationResult",
    "EvidenceType",
    "RhythmEngineState",
    "FeatureBundle"
]

# Module-level docstring with quick reference
__doc__ = """
Grimlock 5.0 Detection Agents
==============================

This module provides all detection agents for the Grimlock 5.0 pipeline.
Each agent is independent, follows Law 2 (no circular imports), and
produces immutable WitnessTestimony for the Scribe.

Available Agents:
-----------------
RhythmEngine:
    Detects rhythmic onsets, patterns, and tempo. Converts to kick drum
    NoteEvents for unified pipeline processing.

    Usage:
        engine = RhythmEngine()
        result = engine.run(audio_buffer, context)

PitchIntelligence:
    Multi-model pitch detection supporting:
    - OMNIZART: Polyphonic, highest accuracy (primary)
    - BASIC_PITCH: Spotify's model, good polyphonic
    - CREPE: Monophonic, high frequency accuracy
    - SPICE: Lightweight, fast
    - LIBROSA: Fallback, always available

    Usage:
        engine = create_pitch_intelligence(model=PitchModelType.OMNIZART)
        result = engine.run(audio_buffer, context)

HarmonicIntelligence:
    Harmonic validation and key detection. Consumes FeatureBundle evidence
    and produces WitnessTestimony about harmonic legitimacy.

    Features:
    - Cent-based harmonic tolerance (musical, not linear)
    - Jazz-aware key detection (dynamic penalties for blue notes)
    - ZCR profile analysis (distinguishes slap bass from noise)
    - Weighted inharmonicity (power partials weigh more)

    Usage:
        engine = create_harmonic_intelligence()
        result = engine.run(audio_buffer, context)

DrumIntelligence:
    Complete drum/percussion detection with ensemble voting.

    Features:
    - Ensemble detection (SPICE + onset + spectral)
    - Multi-band onset detection (lows/mids/highs)
    - Hash-based pattern detection (O(n), not O(n²))
    - Subdivision-aware microtiming (quarter/eighth/triplet/sixteenth)
    - Forensic confidence breakdown
    - Immutable transformations throughout

    Usage:
        engine = create_drum_intelligence()
        result = engine.run(audio_buffer, context)

Law 2 Compliance:
----------------
Agents do NOT import other agents. All communication is through:
- Shared FeatureBundle (immutable spectral evidence)
- StageResult (passed forward by Orchestrator)
- WitnessTestimony (testified to Scribe)

Memory Management:
-----------------
All agents implement MemoryManagedProtocol with:
- staggered_gc() for controlled cleanup
- release_buffer() for explicit deallocation
- Bounded history (max_retained_events/patterns)

For detailed documentation, see individual module docstrings.
"""


# Module initialization check (ensures all agents can be imported)
def _check_imports():
    """Verify all detection agents are importable."""
    missing = []

    try:
        from agents.detection.rhythm_engine import RhythmEngine
    except ImportError as e:
        missing.append(f"rhythm_engine: {e}")

    try:
        from agents.detection.pitch_intelligence import PitchIntelligence
    except ImportError as e:
        missing.append(f"pitch_intelligence: {e}")

    try:
        from agents.detection.harmonic_intelligence import HarmonicIntelligence
    except ImportError as e:
        missing.append(f"harmonic_intelligence: {e}")

    try:
        from agents.detection.drum_intelligence import DrumIntelligence
    except ImportError as e:
        missing.append(f"drum_intelligence: {e}")

    if missing:
        import warnings
        warnings.warn(f"Some detection agents failed to import: {missing}", ImportWarning)
        return False

    return True


# Run import check on module load
_IMPORTS_OK = _check_imports()

# Agent registry for factory pattern (optional)
AGENT_REGISTRY = {
    "rhythm": {
        "class": RhythmEngine,
        "factory": create_rhythm_engine,
        "config_class": OnsetDetectionConfig
    },
    "pitch": {
        "class": PitchIntelligence,
        "factory": create_pitch_intelligence,
        "config_class": PitchDetectionConfig
    },
    "harmonic": {
        "class": HarmonicIntelligence,
        "factory": create_harmonic_intelligence,
        "config_class": HarmonicConfig
    },
    "drum": {
        "class": DrumIntelligence,
        "factory": create_drum_intelligence,
        "config_class": DrumDetectionConfig
    }
}


def get_agent(agent_name: str, **kwargs):
    """
    Factory method to get a detection agent by name.

    Args:
        agent_name: One of 'rhythm', 'pitch', 'harmonic', 'drum'
        **kwargs: Configuration parameters for the agent

    Returns:
        Configured agent instance

    Example:
        rhythm_agent = get_agent('rhythm', progress_callback=my_callback)
        pitch_agent = get_agent('pitch', model='omnizart')
    """
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {agent_name}. Choose from {list(AGENT_REGISTRY.keys())}")

    factory = AGENT_REGISTRY[agent_name]["factory"]
    return factory(**kwargs)


def list_agents() -> Dict[str, Dict[str, Any]]:
    """List all available detection agents with metadata."""
    return {
        name: {
            "class": info["class"].__name__,
            "description": info["class"].__doc__.split("\n")[1] if info["class"].__doc__ else "No description",
            "config": info["config_class"].__name__
        }
        for name, info in AGENT_REGISTRY.items()
    }