# =================================================================
# MODULE: agents/drum_intelligence/__init__.py
# VERSION: 6.0-draft
# DESCRIPTION: Drum and percussion intelligence package - public API.
#
# REPLACES: agents/detection/drum_intelligence.py + agents/detection/
# drum_coprocessor.py (both deleted; see module docstrings in this
# package's other files for what moved where and why).
# =================================================================

from .drum_types import (
    DrumArticulation, DrumMicrotiming, DetectionPass,
    ConfidenceComponents, ForensicDrumEvent, DrumTypeVote,
    DrumIntelligenceConfig,
)
from .onset_detection import (
    SafeResampler, SimpleBandFilter, HFCOnsetDetector,
    AdaptiveMedianThreshold, MultiBandOnsetDetector,
)
from .classification import (
    SpectralDrumClassifier, AdaptiveSPICEWrapper, NMFDrumDecomposer,
)
from .articulation import ArticulationRefiner, ArticulationClassifier
from .interpretation import (
    PatternHashDetector, MicrotimingAnalyzer, MetricalPositionEvidence,
)
from .consensus import PolyphonyResolver, EnsembleDrumDetector
from .orchestrator import DrumIntelligence, SilenceGater, create_drum_intelligence

__version__ = "6.0-draft"

__all__ = [
    # Orchestrator (primary entry point)
    "DrumIntelligence",
    "create_drum_intelligence",
    "SilenceGater",

    # Config
    "DrumIntelligenceConfig",

    # Enums and event types
    "DrumArticulation",
    "DrumMicrotiming",
    "DetectionPass",
    "ConfidenceComponents",
    "ForensicDrumEvent",
    "DrumTypeVote",

    # Onset detection
    "SafeResampler",
    "SimpleBandFilter",
    "HFCOnsetDetector",
    "AdaptiveMedianThreshold",
    "MultiBandOnsetDetector",

    # Classification
    "SpectralDrumClassifier",
    "AdaptiveSPICEWrapper",
    "NMFDrumDecomposer",

    # Articulation
    "ArticulationRefiner",
    "ArticulationClassifier",

    # Interpretation
    "PatternHashDetector",
    "MicrotimingAnalyzer",
    "MetricalPositionEvidence",

    # Consensus
    "PolyphonyResolver",
    "EnsembleDrumDetector",
]
