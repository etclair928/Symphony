# =================================================================
# MODULE: agents/separation/__init__.py
# DESCRIPTION: Separation agents - stem extraction.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-11
#
# EXPORTS:
#   - DemucsSeparator: Facebook's Demucs model for stem separation
#   - BSRoformerSeparator: BS-Roformer for music separation
#   - MelRoformerSeparator: Next-gen mel-spectrogram separator
#   - HybridSeparator: Ensemble of multiple separators
#   - FallbackSeparator: Simple fallback that returns original audio as 'other' stem
#
# LAW 2 COMPLIANCE:
#   This module only imports separation agents - it does NOT import other
#   modules that could create circular dependencies. Agents are independent.
#
# USAGE:
#   from agents.separation import (
#       DemucsSeparator, BSRoformerSeparator, MelRoformerSeparator,
#       HybridSeparator, FallbackSeparator,
#       MelRoformerConfig, HybridConfig
#   )
# =================================================================

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Import main separator classes
from agents.separation.demucs import DemucsSeparator
from agents.separation.roformer import BSRoformerSeparator
from agents.separation.mel_roformer import MelRoformerSeparator, MelRoformerConfig
from agents.separation.hybrid import HybridSeparator, HybridConfig

# Version identifier for this module group
__version__ = "5.6.1"

# Module author and metadata
__author__ = "DeepSeek"

__all__ = [
    "DemucsSeparator",
    "BSRoformerSeparator",
    "MelRoformerSeparator",
    "MelRoformerConfig",
    "HybridSeparator",
    "HybridConfig",
    "FallbackSeparator",
]

# Module-level docstring with quick reference
__doc__ = """
Grimlock 5.0 Separation Agents
===============================

This module provides stem separation agents for the Grimlock 5.0 pipeline.
Each agent is independent, follows Law 2 (no circular imports), and
produces SeparationResult objects for downstream processing.

Available Agents:
-----------------
DemucsSeparator:
    Facebook's Demucs model (Hybrid Transformer Demucs v4).
    Best overall quality for full music separation.

    Features:
    - 4-stem separation (drums, bass, vocals, other)
    - GPU acceleration when available
    - Segment-based processing for long tracks
    - Memory-efficient with staggered GC

    Usage:
        separator = DemucsSeparator()
        result = separator.separate(audio, context)

BSRoformerSeparator:
    BS-Roformer for music separation.
    Excellent for bass and drum separation.

    Features:
    - Band-split Roformer architecture
    - Optimized for bass frequencies
    - Faster than Demucs on CPU

    Usage:
        separator = BSRoformerSeparator()
        result = separator.separate(audio, context)

MelRoformerSeparator:
    Next-generation mel-spectrogram separator.
    State-of-the-art for vocal separation.

    Features:
    - Mel-spectrogram input representation
    - Lightweight transformer architecture
    - Configurable mel bands

    Usage:
        separator = MelRoformerSeparator(config)
        result = separator.separate(audio, context)

HybridSeparator:
    Ensemble of multiple separators.
    Combines strengths of each model.

    Features:
    - Weighted ensemble voting
    - Model-specific confidence scoring
    - Automatic weight selection based on stem type

    Usage:
        separator = HybridSeparator(demucs, roformer, mel_roformer)
        result = separator.separate(audio, context)

FallbackSeparator:
    Simple fallback separator when no models are available.
    Returns original audio as 'other' stem, silence for others.

    Features:
    - Always available (no dependencies)
    - Zero memory footprint
    - Fastest possible separation (just copy)

    Usage:
        separator = FallbackSeparator()
        result = separator.separate(audio, context)

Law 2 Compliance:
----------------
Agents do NOT import other agents. All communication is through:
- AudioContext (shared pipeline state)
- SeparationResult (passed forward by Orchestrator)

Memory Management:
-----------------
All agents implement MemoryManagedProtocol with:
- staggered_gc() for controlled cleanup
- release_stem() for explicit stem deallocation
- Buffer caching with size limits

For detailed documentation, see individual module docstrings.
"""


# ========================================================================
# FALLBACK SEPARATOR (Simple implementation)
# ========================================================================

class FallbackSeparator:
    """
    Simple fallback separator that returns original audio as 'other' stem.

    This is a minimal implementation that requires no external dependencies.
    It always works, making it the ultimate fallback when other separators fail.
    """

    def __init__(self, music_box=None):
        """
        Args:
            music_box: Optional forensic logger (ignored, but kept for API compatibility)
        """
        self._name = "fallback_separator"
        self._music_box = music_box

    @property
    def name(self) -> str:
        return self._name

    @property
    def available_stems(self) -> List:
        """Which stems this separator can produce."""
        from core.order_types import StemType
        return [StemType.OTHER]

    @property
    def separation_quality(self) -> str:
        """Quality level - fallback is lowest quality."""
        return "fallback"

    def separate(self, audio_buffer: np.ndarray, context) -> object:
        """
        Separate audio - returns original as 'other' stem, silence for others.

        Args:
            audio_buffer: Input audio buffer
            context: AudioContext (unused)

        Returns:
            SeparationResult with original audio as 'other' stem
        """
        from core.order_types import StemType, SeparationResult, Confidence

        # Ensure float32
        if audio_buffer.dtype != np.float32:
            audio_buffer = audio_buffer.astype(np.float32)

        # Ensure stereo (2 channels)
        if audio_buffer.ndim == 1:
            audio_buffer = np.stack([audio_buffer, audio_buffer], axis=0)

        # Create silence for other stems
        silence = np.zeros_like(audio_buffer)

        stems = {
            StemType.DRUMS: silence.copy(),
            StemType.BASS: silence.copy(),
            StemType.VOCALS: silence.copy(),
            StemType.OTHER: audio_buffer.copy(),
        }

        return SeparationResult(
            stems=stems,
            separation_time_seconds=0.0,
            memory_usage_mb=0.0,
            confidence=Confidence.MEDIUM
        )

    def run(self, audio_buffer: np.ndarray, context) -> object:
        """Run method for AgentProtocol compatibility."""
        return self.separate(audio_buffer, context)

    def release(self):
        """Release resources."""
        pass


# ========================================================================
# Module Initialization Check
# ========================================================================

def _check_imports() -> bool:
    """Verify all separation agents are importable."""
    missing = []

    try:
        from agents.separation.demucs import DemucsSeparator
    except ImportError as e:
        missing.append(f"demucs: {e}")

    try:
        from agents.separation.roformer import BSRoformerSeparator
    except ImportError as e:
        missing.append(f"roformer: {e}")

    try:
        from agents.separation.mel_roformer import MelRoformerSeparator
    except ImportError as e:
        missing.append(f"mel_roformer: {e}")

    try:
        from agents.separation.hybrid import HybridSeparator
    except ImportError as e:
        missing.append(f"hybrid: {e}")

    try:
        # FallbackSeparator is defined in this file, so it should always work
        pass
    except ImportError as e:
        missing.append(f"fallback: {e}")

    if missing:
        import warnings
        warnings.warn(f"Some separation agents failed to import: {missing}", ImportWarning)
        return False

    return True


# Run import check on module load
_IMPORTS_OK = _check_imports()

# ========================================================================
# Agent Registry for Factory Pattern
# ========================================================================

SEPARATION_AGENT_REGISTRY = {
    "demucs": {
        "class": DemucsSeparator,
        "description": "Facebook's Demucs model - best overall quality"
    },
    "roformer": {
        "class": BSRoformerSeparator,
        "description": "BS-Roformer - excellent for bass and drums"
    },
    "mel_roformer": {
        "class": MelRoformerSeparator,
        "description": "Mel-Roformer - state-of-the-art for vocals",
        "config_class": MelRoformerConfig
    },
    "hybrid": {
        "class": HybridSeparator,
        "description": "Ensemble of multiple separators",
        "config_class": HybridConfig
    },
    "fallback": {
        "class": FallbackSeparator,
        "description": "Simple fallback - original audio as 'other' stem"
    }
}


def get_separation_agent(agent_name: str, **kwargs):
    """
    Factory method to get a separation agent by name.

    Args:
        agent_name: One of 'demucs', 'roformer', 'mel_roformer', 'hybrid', 'fallback'
        **kwargs: Configuration parameters for the agent

    Returns:
        Configured agent instance

    Example:
        demucs = get_separation_agent('demucs', device='cuda')
        hybrid = get_separation_agent('hybrid', demucs_weight=0.5, roformer_weight=0.5)
        fallback = get_separation_agent('fallback')
    """
    if agent_name not in SEPARATION_AGENT_REGISTRY:
        raise ValueError(f"Unknown separation agent: {agent_name}. "
                         f"Choose from {list(SEPARATION_AGENT_REGISTRY.keys())}")

    info = SEPARATION_AGENT_REGISTRY[agent_name]
    agent_class = info["class"]
    return agent_class(**kwargs)


def list_separation_agents() -> Dict[str, Dict[str, Any]]:
    """List all available separation agents with metadata."""
    return {
        name: {
            "class": info["class"].__name__,
            "description": info["description"],
            "has_config": "config_class" in info
        }
        for name, info in SEPARATION_AGENT_REGISTRY.items()
    }


# ========================================================================
# Integration Helpers (Cross-Agent Coordination)
# ========================================================================

def create_hybrid_from_components(
        demucs_weight: float = 0.4,
        roformer_weight: float = 0.3,
        mel_roformer_weight: float = 0.3,
        **kwargs
) -> HybridSeparator:
    """
    Create a HybridSeparator with individual component weights.

    Args:
        demucs_weight: Weight for Demucs model (0.0-1.0)
        roformer_weight: Weight for BS-Roformer model (0.0-1.0)
        mel_roformer_weight: Weight for Mel-Roformer model (0.0-1.0)
        **kwargs: Additional HybridConfig parameters

    Returns:
        Configured HybridSeparator instance
    """
    config = HybridConfig(
        demucs_weight=demucs_weight,
        roformer_weight=roformer_weight,
        mel_roformer_weight=mel_roformer_weight,
        **kwargs
    )
    return HybridSeparator(config=config)


def create_fallback_separator(music_box=None) -> FallbackSeparator:
    """
    Create a fallback separator.

    Args:
        music_box: Optional forensic logger

    Returns:
        Configured FallbackSeparator instance
    """
    return FallbackSeparator(music_box=music_box)


# ========================================================================
# Module Metadata
# ========================================================================

# Export version info
version_info = {
    "module": "agents.separation",
    "version": __version__,
    "agents": list(SEPARATION_AGENT_REGISTRY.keys()),
    "imports_ok": _IMPORTS_OK
}

# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRIMLOCK 5.0 - SEPARATION AGENTS")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")

    print("\nAvailable Agents:")
    for name, info in list_separation_agents().items():
        print(f"  {name:14} - {info['description']}")

    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("=" * 70)

    print("""
# Import specific agents
from agents.separation import DemucsSeparator, BSRoformerSeparator, HybridSeparator, FallbackSeparator

# Direct instantiation
demucs = DemucsSeparator(device='cuda')
result = demucs.separate(audio, context)

# Fallback when nothing else works
fallback = FallbackSeparator()
result = fallback.separate(audio, context)

# Dynamic agent creation
from agents.separation import get_separation_agent, list_separation_agents

agents = list_separation_agents()
hybrid = get_separation_agent('hybrid', demucs_weight=0.5, roformer_weight=0.5)

# Create hybrid from components
from agents.separation import create_hybrid_from_components

hybrid = create_hybrid_from_components(
    demucs_weight=0.4,
    roformer_weight=0.4,
    mel_roformer_weight=0.2
)
    """)

    print("\n" + "=" * 70)
    print("STEM TYPES")
    print("=" * 70)
    print("""
Available stem types (from core.order_types.StemType):
    - FULL_MIX      : Original unseparated audio
    - DRUMS         : Drum stem (kick, snare, cymbals)
    - BASS          : Bass instrument stem
    - VOCALS        : Vocal stem
    - OTHER         : Remaining instruments
    - PERCUSSION    : Auxiliary percussion
    - RHYTHM_SECTION: Combined rhythm section (bass + drums)
    """)

    print("=" * 70)