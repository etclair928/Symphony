# =================================================================
# MODULE: agents/factory.py
# DESCRIPTION: Agent Factory - breaks circular imports by centralizing
# agent creation. Pipeline imports factory, factory creates agents.
#
# VERSION: 5.6.1 (Updated for Pipeline Integration)
# UPDATED: 2026-05-13
#
# LAW OF UNIDIRECTIONAL INTEGRITY:
#     Agents never import Pipeline. Factory is the only module that
#     knows how to create every agent type.
#
# KEY ARCHITECTURE:
#     - Lazy loading (agents imported only when needed)
#     - Agent caching for reuse across multiple runs
#     - Configuration injection with defaults
#     - Fallback agents when primary unavailable
#     - StageOutput contract compliance
#     - Memory arena integration
#
# NEW IN THIS VERSION:
#     - All agents return StageOutput (not raw results)
#     - Accept BufferID handles instead of raw arrays
#     - Memory-aware with borrow patterns
#     - FeatureBundle support for spectral evidence
#
# Authored by: DeepSeek - Updated for Pipeline Integration (2026-05-13)
# =================================================================

import importlib
import warnings
from typing import Dict, Any, Optional, Type, Union, List, Callable
from functools import lru_cache
from dataclasses import dataclass, field

from core.protocols import (
    AgentProtocol, SeparationAgentProtocol, DetectionAgentProtocol,
    AnalysisAgentProtocol, MusicBoxProtocol, StatusReporterProtocol
)
from core.order_types import SourceType, ExportOptions, Confidence
from orchestration.stage_context import StageOutput, StageStatus, StageInput


# ========================================================================
# Configuration Classes
# ========================================================================

@dataclass
class FactoryConfig:
    """Configuration for Agent Factory."""

    # Separation
    demucs_model: str = "htdemucs"
    demucs_segment_seconds: int = 10
    roformer_model: str = "bs_roformer_16k"
    hybrid_demucs_weight: float = 0.6
    hybrid_roformer_weight: float = 0.4
    hybrid_mel_roformer_weight: float = 0.3

    # Detection
    rhythm_onset_threshold: float = 0.5
    rhythm_hop_length: int = 512
    pitch_model: str = "crepe"  # Changed from omnizart to crepe (more reliable)
    pitch_hop_ms: int = 32
    pitch_min_freq: float = 50.0
    pitch_max_freq: float = 2000.0
    tonal_threshold: float = 0.6
    tonal_max_harmonics: int = 8
    drum_model_size: str = "medium"
    drum_onset_threshold: float = 0.5

    # Analysis
    groove_swing_threshold: float = 8.0
    groove_dilla_variance: float = 4.0
    voice_max_gap_ms: int = 50
    voice_max_octave_jump: int = 14
    tempo_min: float = 40.0
    tempo_max: float = 240.0
    tempo_default: float = 120.0
    pulse_resolution_ms: int = 10

    # Quantization
    ritornello_max_snap_ms: int = 100
    ritornello_preserve_original: bool = True
    velocity_overlap_tolerance: int = 10
    velocity_merge_strategy: str = "weighted_average"

    # Validation
    consensus_strategy: str = "any_veto_wins"
    consensus_min_witnesses: int = 2

    # Export
    midi_ppqn: int = 480
    midi_tempo: int = 500000
    octave_restoration: bool = True
    json_pretty: bool = True
    json_include_forensic: bool = True

    # Performance
    use_gpu: bool = True
    device: str = "auto"
    lazy_loading: bool = True
    cache_agents: bool = True

    # Memory
    enable_arena_borrow: bool = True
    default_borrow_ttl_seconds: float = 300.0

    # Reporting
    enable_status_reporter: bool = True


# ========================================================================
# Agent Factory
# ========================================================================

class AgentFactory:
    """
    Agent Factory - creates all agents for Grimlock 5.0.

    All agents now return StageOutput and accept StageInput for
    proper pipeline integration.
    """

    def __init__(
            self,
            config: Optional[Union[Dict[str, Any], FactoryConfig]] = None,
            status_reporter: Optional[StatusReporterProtocol] = None,
            music_box: Optional[MusicBoxProtocol] = None
    ):
        # Normalize config
        if isinstance(config, dict):
            self._config = FactoryConfig(**config)
        elif isinstance(config, FactoryConfig):
            self._config = config
        else:
            self._config = FactoryConfig()

        self._status_reporter = status_reporter
        self._music_box = music_box
        self._cache: Dict[str, Any] = {}
        self._fallback_used: Dict[str, bool] = {}

        self._log_status("AgentFactory initialized")

    # ========================================================================
    # Separation Agents
    # ========================================================================

    def create_separator(self, separator_type: str = "demucs") -> SeparationAgentProtocol:
        """Create a separation agent."""
        cache_key = f"separator_{separator_type}"

        if self._config.cache_agents and cache_key in self._cache:
            return self._cache[cache_key]

        if separator_type == "demucs":
            agent = self._create_demucs()
        elif separator_type == "roformer":
            agent = self._create_roformer()
        elif separator_type == "hybrid":
            agent = self._create_hybrid_separator()
        else:
            raise ValueError(f"Unknown separator type: {separator_type}")

        if self._config.cache_agents:
            self._cache[cache_key] = agent

        return agent

    def _create_demucs(self) -> SeparationAgentProtocol:
        """Create Demucs separation agent."""
        try:
            from agents.separation.demucs import DemucsSeparator

            agent = DemucsSeparator(
                model_name=self._config.demucs_model,
                device=self._config.device,
                music_box=self._music_box
            )
            self._log_status("Demucs separator created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import Demucs: {e}. Using fallback.")
            self._fallback_used["demucs"] = True
            return self._create_fallback_separator()

    def _create_roformer(self) -> SeparationAgentProtocol:
        """Create BS_Roformer separation agent."""
        try:
            from agents.separation.roformer import BSRoformerSeparator

            agent = BSRoformerSeparator(
                model_size=self._config.roformer_model,
                hop_length=320,
                music_box=self._music_box
            )
            self._log_status("BS_Roformer separator created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import BS_Roformer: {e}. Using Demucs fallback.")
            return self._create_demucs()

    def _create_hybrid_separator(self) -> SeparationAgentProtocol:
        """Create hybrid separator."""
        try:
            from agents.separation.hybrid import HybridSeparator, HybridConfig

            config = HybridConfig(
                priority_order=["mel_roformer", "demucs"],
                fallback_to_silence=True,
                cleanup_after_separation=True
            )
            agent = HybridSeparator(
                config=config,
                music_box=self._music_box
            )
            self._log_status("Hybrid separator created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import HybridSeparator: {e}. Using Demucs.")
            return self._create_demucs()

    def _create_fallback_separator(self) -> SeparationAgentProtocol:
        """Create fallback separator."""
        from agents.separation import FallbackSeparator
        return FallbackSeparator(music_box=self._music_box)

    # ========================================================================
    # Detection Agents
    # ========================================================================

    def create_detector(self, detector_type: str) -> DetectionAgentProtocol:
        """Create a detection agent."""
        cache_key = f"detector_{detector_type}"

        if self._config.cache_agents and cache_key in self._cache:
            return self._cache[cache_key]

        if detector_type == "rhythm":
            agent = self._create_rhythm_detector()
        elif detector_type == "pitch":
            agent = self._create_pitch_detector()
        elif detector_type == "tonal":
            agent = self._create_tonal_detector()
        elif detector_type == "drum":
            agent = self._create_drum_detector()
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

        if self._config.cache_agents:
            self._cache[cache_key] = agent

        return agent

    def _create_rhythm_detector(self) -> DetectionAgentProtocol:
        """Create rhythm detection agent."""
        try:
            from agents.detection.rhythm_engine import RhythmEngine, OnsetDetectionConfig

            onset_config = OnsetDetectionConfig(
                onset_threshold=self._config.rhythm_onset_threshold,
                hop_length=self._config.rhythm_hop_length,
            )
            agent = RhythmEngine(
                onset_config=onset_config,
                music_box=self._music_box
            )
            self._log_status("RhythmEngine created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import RhythmEngine: {e}")
            self._fallback_used["rhythm"] = True
            return self._create_fallback_detector("rhythm")

    def _create_pitch_detector(self) -> DetectionAgentProtocol:
        """Create pitch detection agent (CREPE by default)."""
        try:
            from agents.detection.pitch_intelligence import PitchIntelligence, PitchModelType

            model_map = {
                "omnizart": PitchModelType.OMNIZART,
                "basic_pitch": PitchModelType.BASIC_PITCH,
                "crepe": PitchModelType.CREPE,
                "spice": PitchModelType.SPICE,
                "librosa": PitchModelType.LIBROSA
            }
            model = model_map.get(self._config.pitch_model, PitchModelType.CREPE)

            agent = PitchIntelligence(
                config={
                    "primary_model": model,
                    "hop_ms": self._config.pitch_hop_ms,
                    "min_freq_hz": self._config.pitch_min_freq,
                    "max_freq_hz": self._config.pitch_max_freq
                },
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status(f"PitchIntelligence created (model: {self._config.pitch_model})")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import PitchIntelligence: {e}")
            self._fallback_used["pitch"] = True
            return self._create_fallback_detector("pitch")

    def _create_tonal_detector(self) -> DetectionAgentProtocol:
        """Create tonal detection agent."""
        try:
            from agents.detection.harmonic_intelligence import HarmonicIntelligence, HarmonicConfig

            config = HarmonicConfig(
                max_harmonics=self._config.tonal_max_harmonics,
                detect_key=True
            )
            agent = HarmonicIntelligence(
                config=config,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("HarmonicIntelligence created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import HarmonicIntelligence: {e}")
            self._fallback_used["tonal"] = True
            return self._create_fallback_detector("tonal")

    def _create_drum_detector(self) -> DetectionAgentProtocol:
        """Create drum detection agent."""
        try:
            from agents.detection.drum_intelligence import DrumIntelligence, DrumDetectionConfig

            config = DrumDetectionConfig(
                model_size=self._config.drum_model_size,
                onset_threshold=self._config.drum_onset_threshold,
                use_gpu=self._config.use_gpu
            )
            agent = DrumIntelligence(
                config=config,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("DrumIntelligence created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import DrumIntelligence: {e}")
            self._fallback_used["drum"] = True
            return self._create_fallback_detector("drum")

    def _create_fallback_detector(self, detector_type: str) -> DetectionAgentProtocol:
        """Create fallback detector."""
        from agents.base import FallbackDetector
        return FallbackDetector(
            detector_type=detector_type,
            status_reporter=self._status_reporter,
            music_box=self._music_box
        )

    # ========================================================================
    # Analysis Agents
    # ========================================================================

    def create_analyzer(self, analyzer_type: str):
        """Create an analysis agent."""
        cache_key = f"analyzer_{analyzer_type}"

        if self._config.cache_agents and cache_key in self._cache:
            return self._cache[cache_key]

        if analyzer_type == "groove":
            agent = self._create_groove_analyzer()
        elif analyzer_type == "voice":
            agent = self._create_voice_analyzer()
        elif analyzer_type == "tempo":
            agent = self._create_tempo_analyzer()
        elif analyzer_type == "pulse":
            agent = self._create_pulse_analyzer()
        elif analyzer_type == "reverse_geo":
            agent = self._create_reverse_geo_crypt()
        elif analyzer_type == "anechoic":
            agent = self._create_anechoic_ma()
        elif analyzer_type == "harmonic":
            agent = self._create_harmonic_analyzer()
        else:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")

        if self._config.cache_agents:
            self._cache[cache_key] = agent

        return agent

    def _create_groove_analyzer(self):
        """Create Groove Field analyzer."""
        try:
            from agents.analysis.groove_field import GrooveFieldAnalyzer

            agent = GrooveFieldAnalyzer(
                swing_threshold_ms=self._config.groove_swing_threshold,
                dilla_variance_ms=self._config.groove_dilla_variance,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("GrooveFieldAnalyzer created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import GrooveFieldAnalyzer: {e}")
            self._fallback_used["groove"] = True
            return self._create_fallback_analyzer("groove")

    def _create_voice_analyzer(self):
        """Create Voice Continuity analyzer."""
        try:
            from agents.analysis.voice_continuity import VoiceContinuity

            agent = VoiceContinuity(
                max_gap_ms=self._config.voice_max_gap_ms,
                max_octave_jump=self._config.voice_max_octave_jump,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("VoiceContinuity created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import VoiceContinuity: {e}")
            self._fallback_used["voice"] = True
            return self._create_fallback_analyzer("voice")

    def _create_tempo_analyzer(self):
        """Create Tempo Intelligence analyzer."""
        try:
            from agents.analysis.tempo_intelligence import TempoIntelligence

            agent = TempoIntelligence(
                min_tempo=self._config.tempo_min,
                max_tempo=self._config.tempo_max,
                default_tempo=self._config.tempo_default,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("TempoIntelligence created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import TempoIntelligence: {e}")
            self._fallback_used["tempo"] = True
            return self._create_fallback_analyzer("tempo")

    def _create_pulse_analyzer(self):
        """Create Pulse Field analyzer."""
        try:
            from agents.analysis.pulse_field import PulseFieldAnalyzer

            agent = PulseFieldAnalyzer(
                resolution_ms=self._config.pulse_resolution_ms,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("PulseFieldAnalyzer created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import PulseFieldAnalyzer: {e}")
            self._fallback_used["pulse"] = True
            return self._create_fallback_analyzer("pulse")

    def _create_reverse_geo_crypt(self):
        """Create ReverseGeoCrypt analyzer."""
        try:
            from agents.analysis.reverse_geo_crypt import ReverseGeoCrypt

            agent = ReverseGeoCrypt(
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("ReverseGeoCrypt created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import ReverseGeoCrypt: {e}")
            self._fallback_used["reverse_geo"] = True
            return self._create_fallback_analyzer("reverse_geo")

    def _create_anechoic_ma(self):
        """Create Anechoic Ma analyzer."""
        try:
            from agents.analysis.anechoic_ma import AnechoicMa

            agent = AnechoicMa(
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("AnechoicMa created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import AnechoicMa: {e}")
            self._fallback_used["anechoic"] = True
            return self._create_fallback_analyzer("anechoic")

    def _create_harmonic_analyzer(self):
        """Create Harmonic analyzer for validation."""
        try:
            from agents.detection.harmonic_intelligence import HarmonicIntelligence, HarmonicConfig

            config = HarmonicConfig(
                max_harmonics=self._config.tonal_max_harmonics,
                detect_key=True
            )
            agent = HarmonicIntelligence(
                config=config,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("Harmonic analyzer created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import Harmonic analyzer: {e}")
            return self._create_fallback_analyzer("harmonic")

    def _create_fallback_analyzer(self, analyzer_type: str):
        """Create fallback analyzer."""
        from agents.base import FallbackAnalyzer
        return FallbackAnalyzer(
            analyzer_type=analyzer_type,
            status_reporter=self._status_reporter,
            music_box=self._music_box
        )

    # ========================================================================
    # Quantization Agents
    # ========================================================================

    def create_quantizer(self, quantizer_type: str = "ritornello"):
        """Create a quantization agent."""
        cache_key = f"quantizer_{quantizer_type}"

        if self._config.cache_agents and cache_key in self._cache:
            return self._cache[cache_key]

        if quantizer_type == "ritornello":
            agent = self._create_ritornello()
        elif quantizer_type == "velocity_merge":
            agent = self._create_velocity_merge()
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")

        if self._config.cache_agents:
            self._cache[cache_key] = agent

        return agent

    def _create_ritornello(self):
        """Create Ritornello quantizer."""
        try:
            from agents.quantization.ritornello import Ritornello

            agent = Ritornello(
                max_snap_ms=self._config.ritornello_max_snap_ms,
                min_confidence=Confidence.MEDIUM.value,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("Ritornello quantizer created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import Ritornello: {e}")
            self._fallback_used["ritornello"] = True
            return self._create_fallback_quantizer()

    def _create_velocity_merge(self):
        """Create Velocity Merge agent."""
        try:
            from agents.quantization.velocity_merge import VelocityMerge

            agent = VelocityMerge(
                overlap_tolerance_ms=self._config.velocity_overlap_tolerance,
                strategy=self._config.velocity_merge_strategy,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("VelocityMerge created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import VelocityMerge: {e}")
            self._fallback_used["velocity_merge"] = True
            return self._create_fallback_quantizer()

    def _create_fallback_quantizer(self):
        """Create fallback quantizer."""
        from agents.base import FallbackQuantizer
        return FallbackQuantizer(
            status_reporter=self._status_reporter,
            music_box=self._music_box
        )

    # ========================================================================
    # Validation Agents
    # ========================================================================

    def create_validation_agent(self, agent_type: str = "consensus"):
        """Create a validation agent."""
        cache_key = f"validation_{agent_type}"

        if self._config.cache_agents and cache_key in self._cache:
            return self._cache[cache_key]

        if agent_type == "consensus":
            agent = self._create_consensus_engine()
        else:
            raise ValueError(f"Unknown validation agent type: {agent_type}")

        if self._config.cache_agents:
            self._cache[cache_key] = agent

        return agent

    def _create_consensus_engine(self):
        """Create Consensus Engine."""
        try:
            from agents.validation.consensus_engine import ConsensusEngine, ConsensusEngineConfig

            config = ConsensusEngineConfig(
                strategy=self._config.consensus_strategy,
                min_qualified_witnesses=self._config.consensus_min_witnesses
            )
            agent = ConsensusEngine(
                config=config,
                status_reporter=self._status_reporter,
                music_box=self._music_box
            )
            self._log_status("ConsensusEngine created")
            return agent

        except ImportError as e:
            warnings.warn(f"Failed to import ConsensusEngine: {e}")
            self._fallback_used["consensus"] = True
            return self._create_fallback_validation_agent()

    def _create_fallback_validation_agent(self):
        """Create fallback validation agent."""

        class FallbackConsensus:
            def __init__(self, status_reporter=None, music_box=None):
                self.status_reporter = status_reporter
                self.music_box = music_box

            def reach_consensus(self, target_note=None, audio_context=None, witnesses=None):
                return type('ConsensusPackage', (), {
                    'final_decision': True,
                    'consensus_confidence': 0.5,
                    'veto_triggered_by': None,
                    'veto_reason': None,
                    'get_qualified_witnesses': lambda: []
                })()

            def register_witness(self, source, events):
                pass

            def __call__(self, stage_input: StageInput) -> StageOutput:
                return StageOutput(
                    stage_name="consensus",
                    success=True,
                    status=StageStatus.SUCCESS,
                    metadata={"consensus": self.reach_consensus()}
                )

        self._log_status("Using fallback consensus engine", "warn")
        return FallbackConsensus(
            status_reporter=self._status_reporter,
            music_box=self._music_box
        )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def clear_cache(self):
        """Clear all cached agents."""
        self._cache.clear()
        self._log_status("Agent cache cleared")

    def get_cached_agents(self) -> Dict[str, str]:
        """Return list of cached agent types."""
        return {key: type(agent).__name__ for key, agent in self._cache.items()}

    def has_fallback_used(self, agent_type: str) -> bool:
        """Check if a fallback agent was used."""
        return self._fallback_used.get(agent_type, False)

    def get_fallbacks_used(self) -> Dict[str, bool]:
        """Get all fallback usage status."""
        return self._fallback_used.copy()

    def warmup(self, agent_types: List[str]):
        """Preload agents to avoid lazy loading delays."""
        self._log_status(f"Warming up agents: {agent_types}")

        for agent_type in agent_types:
            try:
                if agent_type in ["demucs", "roformer", "hybrid"]:
                    self.create_separator(agent_type)
                elif agent_type in ["rhythm", "pitch", "tonal", "drum"]:
                    self.create_detector(agent_type)
                elif agent_type in ["groove", "voice", "tempo", "pulse", "reverse_geo", "anechoic", "harmonic"]:
                    self.create_analyzer(agent_type)
                elif agent_type in ["ritornello", "velocity_merge"]:
                    self.create_quantizer(agent_type)
                elif agent_type in ["consensus"]:
                    self.create_validation_agent(agent_type)
                else:
                    self._log_status(f"Unknown agent type for warmup: {agent_type}", "warn")
            except Exception as e:
                self._log_status(f"Failed to warmup {agent_type}: {e}", "warn")

        self._log_status("Warmup complete")

    def get_config(self) -> FactoryConfig:
        """Get the factory configuration."""
        return self._config

    def _log_status(self, message: str, level: str = "info"):
        """Log status message."""
        if self._status_reporter:
            if level == "info":
                self._status_reporter.info("AgentFactory", message)
            elif level == "warn":
                self._status_reporter.warn("AgentFactory", message)
            elif level == "error":
                self._status_reporter.error("AgentFactory", message)
        elif self._music_box and level == "error":
            # Fallback logging to music box if no status reporter
            self._music_box.log_error("AgentFactory", Exception(message), {})


# ========================================================================
# Singleton Factory Instance
# ========================================================================

_default_factory: Optional[AgentFactory] = None


def get_default_factory(
        config: Optional[Union[Dict[str, Any], FactoryConfig]] = None,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None
) -> AgentFactory:
    """Get or create the default agent factory instance."""
    global _default_factory
    if _default_factory is None or config is not None:
        _default_factory = AgentFactory(config, status_reporter, music_box)
    return _default_factory


def reset_default_factory():
    """Reset the default factory instance."""
    global _default_factory
    _default_factory = None


# ========================================================================
# Convenience Functions
# ========================================================================

def create_agent(agent_type: str, agent_name: str, **kwargs):
    """Convenience function to create a single agent."""
    config = FactoryConfig(**kwargs)
    factory = AgentFactory(config)

    if agent_type == "separator":
        return factory.create_separator(agent_name)
    elif agent_type == "detector":
        return factory.create_detector(agent_name)
    elif agent_type == "analyzer":
        return factory.create_analyzer(agent_name)
    elif agent_type == "quantizer":
        return factory.create_quantizer(agent_name)
    elif agent_type == "validation":
        return factory.create_validation_agent(agent_name)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ========================================================================
# Stage Agent Wrapper (For Pipeline Integration)
# ========================================================================

class StageAgentWrapper:
    """
    Wrapper that converts any agent to accept StageInput and return StageOutput.

    This allows existing agents to work with the new pipeline without modification.
    """

    def __init__(self, agent, stage_name: str, source_type: SourceType):
        self._agent = agent
        self._stage_name = stage_name
        self._source_type = source_type

    def __call__(self, stage_input: StageInput) -> StageOutput:
        """Execute the agent with StageInput/StageOutput contract."""
        import time
        start_time = time.time()

        try:
            # Extract input buffer if provided
            input_buffer_id = None
            if "input" in stage_input.inputs:
                input_buffer_id = stage_input.inputs["input"]

            # Call the agent (may accept different signatures)
            if hasattr(self._agent, 'run_with_handles'):
                result = self._agent.run_with_handles(input_buffer_id, stage_input)
            elif hasattr(self._agent, 'run'):
                # Need to borrow buffer from arena
                # This assumes the arena is accessible via stage_input
                result = self._agent.run(input_buffer_id, stage_input.context)
            else:
                result = self._agent.process(stage_input)

            elapsed_ms = (time.time() - start_time) * 1000

            return StageOutput(
                stage_name=self._stage_name,
                success=True,
                status=StageStatus.SUCCESS,
                produced_events=getattr(result, 'events', []),
                metadata=getattr(result, 'metadata', {}),
                execution_time_ms=elapsed_ms
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return StageOutput(
                stage_name=self._stage_name,
                success=False,
                status=StageStatus.FAILED,
                error=str(e),
                execution_time_ms=elapsed_ms
            )


def wrap_agent_for_pipeline(agent, stage_name: str, source_type: SourceType) -> StageAgentWrapper:
    """Wrap an agent to conform to the StageInput/StageOutput contract."""
    return StageAgentWrapper(agent, stage_name, source_type)