# =================================================================
# MODULE: agents/analysis/__init__.py
# DESCRIPTION: Analysis agents module for Grimlock 5.0.
#
# VERSION: 5.6.1
# UPDATED: 2026-05-19
#
# EXPORTS:
#   - AnechoicMa: Silence/resonance detector (frame-level evidence)
#   - GrooveFieldAnalyzer: Relational physics for swing detection
#   - PulseFieldAnalyzer: Probabilistic pulse grid detection
#   - ReverseGeoCrypt: Rhythmic cryptanalysis (tempo-free lattice detection)
#   - SpectralMasker: Intelligent spectral separation and masking
#   - TempoIntelligence: Tempo and pulse field detection
#   - VoiceContinuity: Voice tracking across polyphonic texture
#
# LAW 2 COMPLIANCE:
#   This module only imports analysis agents - it does NOT import other
#   modules that could create circular dependencies. Agents are independent.
#
# USAGE:
#   from agents.analysis import (
#       AnechoicMa, GrooveFieldAnalyzer, PulseFieldAnalyzer,
#       ReverseGeoCrypt, SpectralMasker, TempoIntelligence, VoiceContinuity,
#       create_anechoic_ma, create_groove_analyzer, create_pulse_field_analyzer,
#       create_reverse_geo_crypt, create_spectral_masker, create_tempo_intelligence,
#       create_voice_continuity
#   )
# =================================================================

from typing import Dict, List, Any, Optional, Union, Callable, Tuple

# Import main agent classes
from agents.analysis.anechoic_ma import AnechoicMa, AnechoicConfig, AnechoicMode
from agents.analysis.groove_field import GrooveFieldAnalyzer, GrooveConfig
from agents.analysis.pulse_field import PulseFieldAnalyzer, PulseFieldConfig, PulseStability
from agents.analysis.reverse_geo_crypt import ReverseGeoCrypt, ReverseGeoCryptConfig, Event, Anchor, GeometricLattice, \
    EventType

# Import spectral_masker with fallback for missing exports
try:
    from agents.analysis.spectral_masker import (
        SpectralMasker, SpectralMaskerConfig, MaskingMode,
        SeparationStrategy, SpectralProfile, MaskingResult,
        StemScanner, HarmonicCluster
    )
except ImportError as e:
    # If specific exports are missing, try importing just the main class
    try:
        from agents.analysis.spectral_masker import SpectralMasker

        # Define placeholder types if they don't exist
        SpectralMaskerConfig = None
        MaskingMode = None
        SeparationStrategy = None
        SpectralProfile = None
        MaskingResult = None
        StemScanner = None
        HarmonicCluster = None
        print(f"Warning: Partial import of spectral_masker: {e}")
    except ImportError:
        # Module doesn't exist at all
        SpectralMasker = None
        SpectralMaskerConfig = None
        MaskingMode = None
        SeparationStrategy = None
        SpectralProfile = None
        MaskingResult = None
        StemScanner = None
        HarmonicCluster = None
        print(f"Warning: spectral_masker module not found: {e}")

from agents.analysis.tempo_intelligence import TempoIntelligence, TempoConfig
from agents.analysis.voice_continuity import VoiceContinuity, VoiceConfig

# Import convenience factory functions
from agents.analysis.anechoic_ma import create_anechoic_ma, analyze_silence
from agents.analysis.groove_field import create_groove_analyzer
from agents.analysis.pulse_field import create_pulse_field_analyzer
from agents.analysis.reverse_geo_crypt import create_reverse_geo_crypt

# Import spectral_masker factory with fallback
try:
    from agents.analysis.spectral_masker import create_spectral_masker, create_stem_scanner
except ImportError:
    create_spectral_masker = None
    create_stem_scanner = None
    print("Warning: spectral_masker factory functions not available")

from agents.analysis.tempo_intelligence import create_tempo_intelligence
from agents.analysis.voice_continuity import create_voice_continuity

# Version identifier for this module group
__version__ = "5.6.1"

# Module author and metadata
__author__ = "DeepSeek"

# Build __all__ dynamically to avoid missing imports
__all__ = [
    # Main agent classes (only if they exist)
    "AnechoicMa",
    "GrooveFieldAnalyzer",
    "PulseFieldAnalyzer",
    "ReverseGeoCrypt",
    "TempoIntelligence",
    "VoiceContinuity",
]

# Add SpectralMasker to __all__ only if it exists
if SpectralMasker is not None:
    __all__.append("SpectralMasker")

# Add configuration classes
__all__.extend([
    "AnechoicConfig",
    "GrooveConfig",
    "PulseFieldConfig",
    "ReverseGeoCryptConfig",
    "TempoConfig",
    "VoiceConfig",
])

# Add SpectralMaskerConfig only if it exists
if SpectralMaskerConfig is not None:
    __all__.append("SpectralMaskerConfig")

# Add enums and types
__all__.extend([
    "AnechoicMode",
    "PulseStability",
    "Event",
    "Anchor",
    "GeometricLattice",
    "EventType",
])

# Add SpectralMasker types only if they exist
if MaskingMode is not None:
    __all__.extend(["MaskingMode", "SeparationStrategy", "SpectralProfile", "MaskingResult"])

# Add factory functions (only if they exist)
__all__.extend([
    "create_anechoic_ma",
    "create_groove_analyzer",
    "create_pulse_field_analyzer",
    "create_reverse_geo_crypt",
    "create_tempo_intelligence",
    "create_voice_continuity",
    "analyze_silence",
])

if create_spectral_masker is not None:
    __all__.extend(["create_spectral_masker", "mask_stems"])

# Module-level docstring with quick reference
__doc__ = """
Grimlock 5.0 Analysis Agents
=============================

This module provides all analysis agents for the Grimlock 5.0 pipeline.
Each agent is independent, follows Law 2 (no circular imports), and
produces immutable evidence for the Scribe.

Available Agents:
-----------------
AnechoicMa (Silence Oracle):
    Frame-level silence, resonance, and activity detection.
    Produces evidence about whether a time region is genuinely silent,
    resonant (cymbal wash / pedal tone), or musically active.

    Features:
    - Stem-type adaptive thresholds (drums, bass, piano, vocals)
    - Swing-aware subdivision alignment
    - Resonance detection (cymbal wash, pedal tone)
    - Vectorized rolling percentile (40x speedup)

    Usage:
        oracle = create_anechoic_ma()
        report = oracle.analyze(audio, sr, stem_type=StemType.DRUMS)
        state = report.query(note.start, note.end)

GrooveFieldAnalyzer:
    Relational Physics implementation. Measures phase delta between
    bass and kick to detect intentional "wide swing" or "Dilla pocket"
    rather than timing errors.

    Philosophy: "A note's value is defined by its distance to its
    neighbor, not its distance to the grid."

    Features:
    - Phase delta analysis between bass and kick events
    - Swing type detection (wide swing, Dilla pocket, laid back, rushed)
    - Groove confidence scoring with forensic breakdown
    - Quantization recommendations based on groove strength

    Usage:
        analyzer = create_groove_analyzer()
        groove = analyzer.analyze(bass_events, kick_events, context)

PulseFieldAnalyzer:
    Probabilistic pulse grid detection. Represents the underlying pulse
    grid as a probability field rather than a single tempo value.

    Features:
    - Multi-hypothesis pulse tracking (polymeter support)
    - Phase coherence detection with circular statistics
    - Pulse strength annealing over time
    - Integration with ReverseGeoCrypt for lattice-based pulse
    - Integration with TempoIntelligence for tempo map

    Usage:
        analyzer = create_pulse_field_analyzer()
        pulse_field = analyzer.analyze_pulse_field(onset_times, onset_strengths)

ReverseGeoCrypt (Rhythmic Cryptanalysis):
    Tempo-free rhythm detection. Doesn't ask "what is the tempo?"
    Asks "what geometric lattice explains these events?"

    Philosophy: Tempo is the OUTPUT of finding the lattice that
    minimizes structural entropy.

    Features:
    - Ratio-based analysis (tempo-free)
    - Geometric lattice detection
    - Anchor-based phase locking
    - Works with any event source (onsets, chords, ML predictions)

    Usage:
        crypt = create_reverse_geo_crypt()
        lattice = crypt.decrypt(events, anchors=downbeats)
        macro_bpm = lattice.to_tactus_bpm()

SpectralMasker:
    Intelligent spectral separation and masking. Isolates musical
    layers for targeted analysis without full source separation.

    Philosophy: "Don't separate stems when you can mask intelligently."

    Features:
    - Harmonic-percussive separation for rhythm analysis
    - Frequency band masking for instrument isolation
    - Adaptive thresholding based on spectral profile
    - Mask persistence across time frames (temporal coherence)
    - Integration with AnechoicMa for silence-aware masking
    - Spectrogram inpainting for masked regions

    Usage:
        masker = create_spectral_masker()
        mask = masker.create_mask(spectrogram, strategy=SeparationStrategy.HARMONIC)
        isolated = masker.apply_mask(spectrogram, mask)

TempoIntelligence:
    Tempo detection with multiple witnesses and change tracking.

    Features:
    - Multiple tempo witnesses (librosa, madmom, spectral)
    - Ensemble voting for consensus
    - Tempo change detection (accelerando, ritardando)
    - PulseField generation for quantization
    - Time signature detection

    Usage:
        engine = create_tempo_intelligence()
        result = engine.analyze_tempo(audio, context)

VoiceContinuity:
    Voice tracking across polyphonic texture. Traces monophonic
    voices, detects voice crossings, and assigns voice roles.

    Features:
    - Voice separation strategies (pitch proximity, spectral, harmonic)
    - Voice crossing detection with tolerance
    - Octave jump repair with musical context
    - Role assignment (bass, tenor, alto, soprano, melody)

    Usage:
        analyzer = create_voice_continuity()
        result = analyzer.separate_voices(notes)

Law 2 Compliance:
----------------
Agents do NOT import other agents. All communication is through:
- Shared FeatureBundle (immutable spectral evidence)
- StageResult (passed forward by Orchestrator)
- WitnessTestimony (testified to Scribe)

5 Strategies Integration:
------------------------
1. Streaming Windows: All agents support chunked processing
2. Disk-Backed: Caching available for expensive computations
3. mmap for Audit: Random access for forensic verification
4. GC Discipline: staggered_gc() between stages
5. Sparse Representations: Minimal memory footprint

For detailed documentation, see individual module docstrings.
"""


# ========================================================================
# Module Initialization Check
# ========================================================================

def _check_imports() -> bool:
    """Verify all analysis agents are importable."""
    missing = []

    try:
        from agents.analysis.anechoic_ma import AnechoicMa
    except ImportError as e:
        missing.append(f"anechoic_ma: {e}")

    try:
        from agents.analysis.groove_field import GrooveFieldAnalyzer
    except ImportError as e:
        missing.append(f"groove_field: {e}")

    try:
        from agents.analysis.pulse_field import PulseFieldAnalyzer
    except ImportError as e:
        missing.append(f"pulse_field: {e}")

    try:
        from agents.analysis.reverse_geo_crypt import ReverseGeoCrypt
    except ImportError as e:
        missing.append(f"reverse_geo_crypt: {e}")

    try:
        from agents.analysis.tempo_intelligence import TempoIntelligence
    except ImportError as e:
        missing.append(f"tempo_intelligence: {e}")

    try:
        from agents.analysis.voice_continuity import VoiceContinuity
    except ImportError as e:
        missing.append(f"voice_continuity: {e}")

    # SpectralMasker is optional - don't mark as missing if it fails
    try:
        from agents.analysis.spectral_masker import SpectralMasker
    except ImportError:
        pass  # Optional agent

    if missing:
        import warnings
        warnings.warn(f"Some analysis agents failed to import: {missing}", ImportWarning)
        return False

    return True


# Run import check on module load
_IMPORTS_OK = _check_imports()

# ========================================================================
# Agent Registry for Factory Pattern
# ========================================================================

ANALYSIS_AGENT_REGISTRY = {
    "anechoic": {
        "class": AnechoicMa,
        "factory": create_anechoic_ma,
        "config_class": AnechoicConfig,
        "description": "Silence/resonance detector (frame-level evidence)"
    },
    "groove": {
        "class": GrooveFieldAnalyzer,
        "factory": create_groove_analyzer,
        "config_class": GrooveConfig,
        "description": "Relational physics for swing detection"
    },
    "pulse": {
        "class": PulseFieldAnalyzer,
        "factory": create_pulse_field_analyzer,
        "config_class": PulseFieldConfig,
        "description": "Probabilistic pulse grid detection"
    },
    "reverse_geo": {
        "class": ReverseGeoCrypt,
        "factory": create_reverse_geo_crypt,
        "config_class": ReverseGeoCryptConfig,
        "description": "Tempo-free rhythmic cryptanalysis"
    },
    "tempo": {
        "class": TempoIntelligence,
        "factory": create_tempo_intelligence,
        "config_class": TempoConfig,
        "description": "Tempo and pulse field detection"
    },
    "voice": {
        "class": VoiceContinuity,
        "factory": create_voice_continuity,
        "config_class": VoiceConfig,
        "description": "Voice tracking across polyphonic texture"
    }
}

# Add SpectralMasker to registry only if it exists
if SpectralMasker is not None:
    ANALYSIS_AGENT_REGISTRY["spectral_masker"] = {
        "class": SpectralMasker,
        "factory": create_spectral_masker if create_spectral_masker else lambda **kwargs: None,
        "config_class": SpectralMaskerConfig,
        "description": "Intelligent spectral separation and masking"
    }


def get_analysis_agent(agent_name: str, **kwargs):
    """
    Factory method to get an analysis agent by name.

    Args:
        agent_name: One of 'anechoic', 'groove', 'pulse', 'reverse_geo',
                   'spectral_masker', 'tempo', 'voice'
        **kwargs: Configuration parameters for the agent

    Returns:
        Configured agent instance

    Example:
        groove_agent = get_analysis_agent('groove', swing_threshold_ms=10.0)
        tempo_agent = get_analysis_agent('tempo', min_tempo=40, max_tempo=240)
    """
    if agent_name not in ANALYSIS_AGENT_REGISTRY:
        raise ValueError(f"Unknown analysis agent: {agent_name}. "
                         f"Choose from {list(ANALYSIS_AGENT_REGISTRY.keys())}")

    factory = ANALYSIS_AGENT_REGISTRY[agent_name]["factory"]
    if factory is None:
        raise ValueError(f"Agent '{agent_name}' is not available (factory function missing)")
    return factory(**kwargs)


def list_analysis_agents() -> Dict[str, Dict[str, Any]]:
    """List all available analysis agents with metadata."""
    return {
        name: {
            "class": info["class"].__name__ if info["class"] else "Not Available",
            "description": info["description"],
            "config": info["config_class"].__name__ if info["config_class"] else "N/A"
        }
        for name, info in ANALYSIS_AGENT_REGISTRY.items()
    }


# ========================================================================
# Integration Helpers (Cross-Agent Coordination)
# ========================================================================

def create_rhythm_pipeline(
        tempo_engine: Optional[TempoIntelligence] = None,
        pulse_analyzer: Optional[PulseFieldAnalyzer] = None,
        reverse_geo: Optional[ReverseGeoCrypt] = None,
        groove_analyzer: Optional[GrooveFieldAnalyzer] = None
) -> Dict[str, Any]:
    """
    Create a coordinated rhythm analysis pipeline.

    This helper wires together the rhythm-related analysis agents
    without violating Law 2 (they remain independent, just orchestrated).

    Returns:
        Dictionary containing all configured agents
    """
    return {
        "tempo": tempo_engine or create_tempo_intelligence(),
        "pulse": pulse_analyzer or create_pulse_field_analyzer(),
        "reverse_geo": reverse_geo or create_reverse_geo_crypt(),
        "groove": groove_analyzer or create_groove_analyzer()
    }


def create_silence_aware_pipeline(
        anechoic: Optional[AnechoicMa] = None,
        voice_analyzer: Optional[VoiceContinuity] = None
) -> Dict[str, Any]:
    """
    Create a coordinated silence-aware analysis pipeline.

    AnechoicMa provides silence evidence; VoiceContinuity uses it.

    Returns:
        Dictionary containing all configured agents
    """
    return {
        "anechoic": anechoic or create_anechoic_ma(),
        "voice": voice_analyzer or create_voice_continuity()
    }


def create_spectral_isolation_pipeline(
        spectral_masker: Optional[SpectralMasker] = None,
        anechoic: Optional[AnechoicMa] = None
) -> Dict[str, Any]:
    """
    Create a coordinated spectral isolation pipeline.

    SpectralMasker isolates frequency regions; AnechoicMa provides
    silence context for adaptive thresholding.

    Returns:
        Dictionary containing all configured agents
    """
    result = {
        "anechoic": anechoic or create_anechoic_ma()
    }

    # Only add spectral_masker if available
    if spectral_masker is not None or create_spectral_masker is not None:
        result["spectral_masker"] = spectral_masker or create_spectral_masker()
    else:
        result["spectral_masker"] = None
        result["warning"] = "SpectralMasker not available"

    return result


# ========================================================================
# Module Metadata
# ========================================================================

# Export version info
version_info = {
    "module": "agents.analysis",
    "version": __version__,
    "agents": list(ANALYSIS_AGENT_REGISTRY.keys()),
    "imports_ok": _IMPORTS_OK
}

# ========================================================================
# Standalone Test
# ========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRIMLOCK 5.0 - ANALYSIS AGENTS")
    print("=" * 70)

    print(f"\nVersion: {__version__}")
    print(f"Imports OK: {_IMPORTS_OK}")

    print("\nAvailable Agents:")
    for name, info in list_analysis_agents().items():
        print(f"  {name:16} - {info['description']}")

    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("=" * 70)

    print("""
# Import specific agents
from agents.analysis import GrooveFieldAnalyzer, TempoIntelligence, VoiceContinuity

# Or use factory functions
from agents.analysis import create_groove_analyzer, create_tempo_intelligence

# Dynamic agent creation
from agents.analysis import get_analysis_agent, list_analysis_agents

agents = list_analysis_agents()
groove = get_analysis_agent('groove', swing_threshold_ms=12.0)

# Coordinated pipeline
from agents.analysis import create_rhythm_pipeline
rhythm_pipeline = create_rhythm_pipeline()
    """)

    print("\n" + "=" * 70)