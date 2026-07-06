# =================================================================
# MODULE: agents/separation/hybrid.py
# VERSION: 5.6.1
# DESCRIPTION: Hybrid separation agent for Grimlock 5.6.1.
#
# Law of Unidirectional Integrity: Data flows forward.
# This agent does NOT make routing decisions. It attempts separation
# with available models in priority order. If all fail, it returns
# zeros (silence) rather than hallucinating or routing to nowhere.
#
# Law of Resource Survival: Memory is managed through the guardian.
# Models are loaded on demand and can be released after use.
#
# Law of Scribe's Truth: Confidence must be earned. This agent
# reports separation quality so the Scribe can veto if needed.
#
# This is a TOOL, not a decision-maker. No conditional logic
# about content. No routing to different pipelines.
# =================================================================

import time
import warnings
import gc
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.order_types import (
    StemType, AudioContext, SeparationResult, Confidence, SourceType,
    StageResult, VetoReason
)
from core.constants import TARGET_SAMPLE_RATE, GC_COLLECT_AFTER_EACH_STAGE
from core.protocols import SeparationAgentProtocol, MemoryManagedProtocol, MusicBoxProtocol


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class HybridConfig:
    """
    Configuration for Hybrid Separator agent.

    Attributes:
        priority_order: List of model names in priority order
        fallback_to_silence: If True, return silence when all models fail
        cleanup_after_separation: If True, release models after use
        mel_roformer_config: Optional config for Mel-RoFormer
        demucs_config: Optional config for Demucs
    """
    priority_order: List[str] = field(default_factory=lambda: ["mel_roformer", "demucs"])
    fallback_to_silence: bool = True
    cleanup_after_separation: bool = True
    mel_roformer_config: Optional[Dict[str, Any]] = None
    demucs_config: Optional[Dict[str, Any]] = None
    max_attempt_time_seconds: float = 300.0  # 5 minutes max per model


class ModelPriority(str, Enum):
    """Priority order for attempting models. Simple, linear, no branching."""
    FIRST = "mel_roformer"
    SECOND = "demucs"
    THIRD = "fallback"


@dataclass
class SeparationAttempt:
    """Record of a single separation attempt."""
    model_name: str
    success: bool
    time_seconds: float
    error: Optional[str] = None


class HybridSeparator(SeparationAgentProtocol, MemoryManagedProtocol):
    """
    Hybrid separator for Grimlock 5.0.

    Simple linear fallback chain:
    1. Try Mel-RoFormer (best quality, multi-stem)
    2. Try Demucs (fallback if Mel-RoFormer unavailable)
    3. Return zeros (silence) - no hallucination

    No content detection. No routing. No split-brain.
    """

    def __init__(
            self,
            config: Optional[HybridConfig] = None,
            music_box: Optional[MusicBoxProtocol] = None,
            memory_guardian: Optional[Any] = None,
            **kwargs
    ):
        """
        Args:
            config: HybridConfig instance (overrides kwargs)
            music_box: Optional forensic logger
            memory_guardian: Optional memory manager
            **kwargs: Individual config parameters
        """
        # Merge config
        if config is None:
            config = HybridConfig(**kwargs)
        else:
            # Override config with kwargs if provided
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self._config = config
        self._name = "hybrid_separator"
        self._source_type = SourceType.HYBRID_SEPARATOR
        self._music_box = music_box
        self._memory_guardian = memory_guardian
        self._attempts: List[SeparationAttempt] = []
        self._model_instances: Dict[str, Any] = {}
        self._last_separation_time_ms: float = 0.0
        self._total_memory_freed_mb: float = 0.0

    # ========================================================================
    # AgentProtocol Implementation
    # ========================================================================

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def run(self, audio_buffer: np.ndarray, context: AudioContext) -> StageResult:
        """Main entry point for pipeline."""
        start_time = time.time()
        result = self.separate(audio_buffer, context)
        execution_time_ms = (time.time() - start_time) * 1000

        # Determine success based on whether we got any non-silence stems
        has_audio = False
        for stem in result.stems.values():
            if stem is not None and np.max(np.abs(stem)) > 1e-6:
                has_audio = True
                break

        return StageResult(
            stage_name=self._name,
            success=has_audio,
            events=[],
            metadata={
                "attempts": [{"model": a.model_name, "success": a.success} for a in self._attempts],
                "successful_model": self._get_successful_model(),
                "separation_time_seconds": result.separation_time_seconds,
                "stems_returned": list(result.stems.keys())
            },
            veto_reason=None if has_audio else VetoReason.EMPTY_RESULT,
            execution_time_ms=execution_time_ms,
            memory_delta_mb=result.memory_usage_mb
        )

    # ========================================================================
    # SeparationAgentProtocol Implementation
    # ========================================================================

    @property
    def available_stems(self) -> List[StemType]:
        """Which stems this separator can produce."""
        return [StemType.DRUMS, StemType.BASS, StemType.OTHER, StemType.VOCALS]

    @property
    def separation_quality(self) -> str:
        """Quality level - hybrid aims for best available."""
        return "high"

    def separate(
            self,
            audio_buffer: np.ndarray,
            context: AudioContext
    ) -> SeparationResult:
        """
        Separate audio into stems using linear fallback chain.

        No branching. No conditional routing. Just try models in order.
        """
        start_time = time.time()
        self._attempts.clear()

        # Ensure audio is float32
        if audio_buffer.dtype != np.float32:
            audio_buffer = audio_buffer.astype(np.float32)

        # Linear fallback chain (no branching logic)
        stems = None

        # Try models in priority order
        for model_name in self._config.priority_order:
            if stems is not None:
                break

            attempt_start = time.time()

            if model_name == "mel_roformer":
                stems = self._try_mel_roformer(audio_buffer, context)
            elif model_name == "demucs":
                stems = self._try_demucs(audio_buffer, context)
            else:
                continue

            attempt_time = time.time() - attempt_start

            self._attempts.append(SeparationAttempt(
                model_name=model_name,
                success=stems is not None,
                time_seconds=attempt_time
            ))

            # Check timeout
            if attempt_time > self._config.max_attempt_time_seconds:
                self._log_warning(f"{model_name} exceeded timeout ({attempt_time:.1f}s)")

        # If all attempts failed, return silence (not hallucination)
        if stems is None and self._config.fallback_to_silence:
            stems = self._create_silence_stems(audio_buffer)
            self._attempts.append(SeparationAttempt(
                model_name="silence_fallback",
                success=True,
                time_seconds=0.0,
                error="All separation models failed - returning silence"
            ))

        total_time = time.time() - start_time
        self._last_separation_time_ms = total_time * 1000

        # Log to MusicBox if available
        if self._music_box:
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="separation_testimony",
                before_state={"audio_shape": list(audio_buffer.shape)},
                after_state={
                    "attempts": [{"model": a.model_name, "success": a.success} for a in self._attempts],
                    "successful_model": self._get_successful_model(),
                    "separation_time_seconds": total_time,
                    "stems_returned": list(stems.keys())
                },
                reasoning=f"Separation using {self._get_successful_model()} in {total_time:.1f}s",
                reversible=False
            )

        # Create result with confidence based on success
        successful_model = self._get_successful_model()
        confidence = Confidence.HIGH if successful_model != "silence_fallback" else Confidence.LOW

        # Clean up model instances to free memory
        if self._config.cleanup_after_separation:
            for model in self._model_instances.values():
                del model
            self._model_instances.clear()

        # Memory cleanup
        self._maybe_gc()

        return SeparationResult(
            stems=stems,
            separation_time_seconds=total_time,
            memory_usage_mb=self.get_memory_footprint_mb(),
            confidence=confidence
        )

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _try_mel_roformer(
            self,
            audio: np.ndarray,
            context: AudioContext
    ) -> Optional[Dict[StemType, np.ndarray]]:
        """Attempt separation with Mel-RoFormer."""
        try:
            from agents.separation.mel_roformer import MelRoformerSeparator, MelRoformerConfig

            # Create config from dict if provided
            if self._config.mel_roformer_config:
                mel_config = MelRoformerConfig(**self._config.mel_roformer_config)
            else:
                mel_config = MelRoformerConfig(
                    n_mels=128,
                    sample_rate=context.original_sample_rate or TARGET_SAMPLE_RATE,
                    device="auto"
                )

            separator = MelRoformerSeparator(config=mel_config, music_box=self._music_box)

            result = separator.separate(audio, context)

            if result and hasattr(result, 'stems') and result.stems:
                # Cache instance for cleanup
                self._model_instances["mel_roformer"] = separator
                return result.stems
            elif isinstance(result, dict) and result:
                return result
            return None

        except ImportError:
            if self._music_box:
                self._log_warning("Mel-RoFormer not installed")
            return None
        except Exception as e:
            if self._music_box:
                self._log_error("mel_roformer", str(e))
            return None

    def _try_demucs(
            self,
            audio: np.ndarray,
            context: AudioContext
    ) -> Optional[Dict[StemType, np.ndarray]]:
        """Attempt separation with Demucs."""
        try:
            from agents.separation.demucs import DemucsSeparator

            # Create config from dict if provided
            if self._config.demucs_config:
                separator = DemucsSeparator(
                    music_box=self._music_box,
                    **self._config.demucs_config
                )
            else:
                separator = DemucsSeparator(music_box=self._music_box)

            result = separator.separate(audio, context)

            if result and hasattr(result, 'stems') and result.stems:
                # Cache instance for cleanup
                self._model_instances["demucs"] = separator
                return result.stems
            elif isinstance(result, dict) and result:
                return result
            return None

        except ImportError:
            if self._music_box:
                self._log_warning("Demucs not installed")
            return None
        except Exception as e:
            if self._music_box:
                self._log_error("demucs", str(e))
            return None

    def _create_silence_stems(
            self,
            audio: np.ndarray
    ) -> Dict[StemType, np.ndarray]:
        """
        Create silence stems when all separation attempts fail.

        Law of Scribe's Truth: Better to return silence than to hallucinate.
        The Scribe will veto empty results rather than letting heartbeat notes through.
        """
        # Ensure stereo (2 channels)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)

        silence = np.zeros_like(audio)

        return {
            StemType.DRUMS: silence.copy(),
            StemType.BASS: silence.copy(),
            StemType.VOCALS: silence.copy(),
            StemType.OTHER: silence.copy()
        }

    def _get_successful_model(self) -> str:
        """Get name of first successful model."""
        for attempt in self._attempts:
            if attempt.success:
                return attempt.model_name
        return "none"

    def get_last_attempts(self) -> List[SeparationAttempt]:
        """Get the attempts from the last separation."""
        return self._attempts.copy()

    def _log_warning(self, message: str):
        """Log warning via MusicBox."""
        if self._music_box and hasattr(self._music_box, 'log_decision'):
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="warning",
                before_state={},
                after_state={},
                reasoning=message,
                reversible=False
            )
        else:
            warnings.warn(message)

    def _log_error(self, model: str, error: str):
        """Log error via MusicBox."""
        if self._music_box and hasattr(self._music_box, 'log_decision'):
            self._music_box.log_decision(
                stage_name=self._name,
                decision_type="error",
                before_state={"model": model},
                after_state={},
                reasoning=f"{model} failed: {error}",
                reversible=False
            )

    # ========================================================================
    # MemoryManagedProtocol Implementation
    # ========================================================================

    def release_buffer(self, buffer_name: str) -> None:
        """Release a buffer by name."""
        if hasattr(self, buffer_name):
            delattr(self, buffer_name)

    def get_memory_footprint_mb(self) -> float:
        """Current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def can_release(self, buffer_name: str) -> bool:
        """Return True if this buffer is safe to delete."""
        return buffer_name.startswith("stem_") or buffer_name in self._model_instances

    def staggered_gc(self) -> Dict[str, Any]:
        """Run garbage collection and return stats."""
        before = self.get_memory_footprint_mb()
        gc.collect()
        after = self.get_memory_footprint_mb()
        freed = before - after
        self._total_memory_freed_mb += max(0, freed)

        return {
            "memory_before_mb": before,
            "memory_after_mb": after,
            "freed_mb": freed,
            "total_freed_mb": self._total_memory_freed_mb,
            "triggered_by": self._name,
            "timestamp": time.time()
        }

    def _maybe_gc(self):
        """Conditional garbage collection."""
        if GC_COLLECT_AFTER_EACH_STAGE:
            self.staggered_gc()

    def estimate_memory_mb(self, duration_seconds: float) -> float:
        """
        Estimate memory usage for separation.

        Law of Resource Survival: Report expected memory so pipeline can decide.
        """
        # Mel-RoFormer: ~1-2GB, Demucs: ~2-4GB
        return 2048.0  # Conservative estimate


# =================================================================
# Factory functions
# =================================================================

def create_hybrid_separator(
        priority_order: Optional[List[str]] = None,
        fallback_to_silence: bool = True,
        cleanup_after_separation: bool = True,
        music_box: Optional[MusicBoxProtocol] = None,
        memory_guardian: Optional[Any] = None
) -> HybridSeparator:
    """
    Create a configured hybrid separator.

    Args:
        priority_order: List of model names in priority order
        fallback_to_silence: If True, return silence when all models fail
        cleanup_after_separation: If True, release models after use
        music_box: MusicBox instance for logging
        memory_guardian: MemoryGuardian instance for resource management

    Returns:
        Configured HybridSeparator instance
    """
    config = HybridConfig(
        priority_order=priority_order or ["mel_roformer", "demucs"],
        fallback_to_silence=fallback_to_silence,
        cleanup_after_separation=cleanup_after_separation
    )
    return HybridSeparator(
        config=config,
        music_box=music_box,
        memory_guardian=memory_guardian
    )


def create_hybrid_from_config(
        config: HybridConfig,
        music_box: Optional[MusicBoxProtocol] = None,
        memory_guardian: Optional[Any] = None
) -> HybridSeparator:
    """Create a hybrid separator from a config object."""
    return HybridSeparator(
        config=config,
        music_box=music_box,
        memory_guardian=memory_guardian
    )