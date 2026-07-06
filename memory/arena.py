# =================================================================
# MODULE: memory/arena.py
# DESCRIPTION: Scoped memory arena with ENFORCED ownership.
#
# VERSION: 5.6.1 (FIXED: AcousticIntelligence.create_contract parameter)
# UPDATED: 2026-05-16
#
# CRITICAL FIX:
#   - Changed create_contract parameter from 'raw_audio' to 'audio'
#     (line ~785 in store() legacy fallback)
#
# All other functionality unchanged from v5.3
# =================================================================

from __future__ import annotations

import gc
import hashlib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, Iterator, List, Optional,
    Set, Tuple, Union,
)

import numpy as np

from core.constants import (
    ANALYSIS_SR,
    BASIC_PITCH_SR,
    BUFFER_LIFECYCLE_STAGES,
    CREPE_SR,
    MAX_CONCURRENT_BUFFERS,
    MEMORY_LIMIT_MB,
    MEMORY_WARNING_THRESHOLD_MB,
    STAGGERED_GC_TRIGGER_MB,
    TARGET_SAMPLE_RATE,
)
from core.order_types import StemType
from core.protocols import MusicBoxProtocol, StatusReporterProtocol

# Canonical audio authority — all transformations go through here
from core.acoustic_intelligence import (
    AcousticIntelligence,
    AudioContract,
    ChannelLayout,
    ImmutableAudio,
)


# ============================================================================
# ENUMS
# ============================================================================

class BufferState(str, Enum):
    ACTIVE      = "active"
    BORROWED    = "borrowed"
    PINNED      = "pinned"
    POOLED      = "pooled"
    RELEASED    = "released"
    TRANSFERRED = "transferred"
    MMAP_BACKED = "mmap"
    PREPARED    = "prepared"


class ArenaPriority(str, Enum):
    CRITICAL = "critical"
    NORMAL   = "normal"
    DEFERRED = "deferred"


class BorrowScope(str, Enum):
    TEMPORARY  = "temporary"
    STAGE      = "stage"
    PERSISTENT = "persistent"


class MemoryTier(Enum):
    TIER_0    = 0   # Critical — never evict (master audio, LoadedAudio)
    TIER_1    = 1   # High — evict last (stems, FeatureBundle)
    TIER_2    = 2   # Normal (derived analysis buffers)
    TIER_3    = 3   # Low — evict first (temporary scratch)
    SPILLABLE = 4   # Can be spilled to disk


class EvictionStrategy(Enum):
    NONE         = "none"
    AGGRESSIVE   = "aggressive"
    BALANCED     = "balanced"
    CONSERVATIVE = "conservative"


# ============================================================================
# STEM WRAPPER (F2 — fixes stem sample rate tracking bug)
# ============================================================================

@dataclass
class StemWithRate:
    """
    Stem audio wrapper that carries sample rate metadata.

    USAGE in pipeline._run_separation():
        raw_stem = demucs_output[stem_name]          # numpy array
        stem = StemWithRate(
            audio=raw_stem,
            sample_rate=44100,                       # always known post-Demucs
            stem_type=StemType.DRUMS,
            model_name="htdemucs"
        )
        stem_id = arena.store("drums_stem", stem)    # SR preserved in contract

    LAW: "Stems must carry their sample rate. No exceptions."
    """
    audio: np.ndarray
    sample_rate: int
    stem_type: StemType
    model_name: str = "demucs"
    _cached_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.audio.dtype != np.float32:
            self.audio = self.audio.astype(np.float32)
        if not self.audio.flags.c_contiguous:
            self.audio = np.ascontiguousarray(self.audio)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.audio.shape

    @property
    def nbytes(self) -> int:
        return self.audio.nbytes

    @property
    def sha256_hash(self) -> str:
        """Lazy SHA-256 for forensic tracking."""
        if self._cached_hash is None:
            self._cached_hash = hashlib.sha256(self.audio.tobytes()).hexdigest()
        return self._cached_hash

    def to_contract(self, source: str = "stem_wrapper") -> AudioContract:
        """Convert to canonical AudioContract for arena storage."""
        return AcousticIntelligence.create_contract(
            audio=self.audio,  # FIXED: was 'raw_audio', now 'audio'
            sample_rate=self.sample_rate,
            source=f"{source}_{self.stem_type.value}",
            force_mono=False,
        )

    def __repr__(self) -> str:
        return (
            f"StemWithRate("
            f"type={self.stem_type.value}, "
            f"sr={self.sample_rate}Hz, "
            f"shape={self.shape}, "
            f"model={self.model_name})"
        )


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class BufferID:
    """Immutable identifier for an arena buffer."""
    arena_name: str
    buffer_name: str
    unique_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __str__(self) -> str:
        return f"{self.arena_name}:{self.buffer_name}:{self.unique_id}"

    @property
    def short(self) -> str:
        return f"{self.buffer_name}:{self.unique_id}"


@dataclass(frozen=True)
class BufferStats:
    """Immutable snapshot of a buffer's stats."""
    id: BufferID
    size_bytes: int
    size_mb: float
    state: BufferState
    borrow_count: int
    created_at: datetime
    last_accessed: datetime
    owner_stage: Optional[str]
    borrow_scope: BorrowScope
    is_audio_contract: bool
    is_loaded_audio: bool
    tier: MemoryTier
    sample_rate: Optional[int]


@dataclass
class BorrowRecord:
    """Record of one active borrow."""
    buffer_id: BufferID
    agent_name: str
    stage: str
    borrowed_at: datetime
    scope: BorrowScope
    expires_at: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class BufferSlot:
    """
    A single buffer slot in the arena.

    contract holds whatever was stored — AudioContract, LoadedAudio, or
    FeatureBundle. Use _is_audio_contract() and _is_loaded_audio_slot()
    to check before accessing type-specific attributes.
    """
    id: BufferID
    contract: Any           # AudioContract | LoadedAudio | FeatureBundle
    size_bytes: int
    is_loaded_audio_obj: bool = False   # True when contract is a LoadedAudio
    state: BufferState = BufferState.ACTIVE
    borrow_count: int = 0
    borrow_records: List[BorrowRecord] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    owner_stage: Optional[str] = None
    priority: ArenaPriority = ArenaPriority.NORMAL
    borrow_scope: BorrowScope = BorrowScope.TEMPORARY
    is_feature_bundle: bool = False
    tier: MemoryTier = MemoryTier.TIER_2
    prepared_from: Optional[BufferID] = None
    preparation_params: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_accessed = datetime.now()

    @property
    def sample_rate(self) -> Optional[int]:
        """
        Sample rate — safe for all stored types.
        Returns None for FeatureBundle (use bundle.sample_rate directly).
        """
        try:
            return self.contract.sample_rate
        except AttributeError:
            return None

    @property
    def samples(self) -> Optional[np.ndarray]:
        """
        Raw samples — only valid for AudioContract slots.
        Returns None for LoadedAudio and FeatureBundle.
        Do not use this in hot paths; borrow via borrow_contract() instead.
        """
        if self.is_loaded_audio_obj or self.is_feature_bundle:
            return None
        try:
            return self.contract.samples
        except AttributeError:
            return None

    def get_size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def get_stats(self) -> BufferStats:
        return BufferStats(
            id=self.id,
            size_bytes=self.size_bytes,
            size_mb=self.get_size_mb(),
            state=self.state,
            borrow_count=self.borrow_count,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            owner_stage=self.owner_stage,
            borrow_scope=self.borrow_scope,
            is_audio_contract=(
                not self.is_loaded_audio_obj and
                not self.is_feature_bundle
            ),
            is_loaded_audio=self.is_loaded_audio_obj,
            tier=self.tier,
            sample_rate=self.sample_rate,
        )


@dataclass
class EvictableBuffer:
    name: str
    arena: "MemoryArena"
    size_bytes: int
    tier: MemoryTier
    is_pinned: bool
    borrow_count: int
    last_accessed: datetime

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass
class ArenaStats:
    name: str
    buffer_count: int
    borrowed_count: int
    pinned_count: int
    pooled_count: int
    prepared_count: int
    loaded_audio_count: int
    total_bytes: int
    total_mb: float
    allocated_mb: float
    freed_mb: float
    peak_mb: float
    age_seconds: float
    pool_hits: int
    pool_misses: int
    transfer_in_count: int
    transfer_out_count: int
    tier_breakdown: Dict[str, int]
    cache_hits: int = 0
    cache_misses: int = 0


# ============================================================================
# MEMORY ARENA
# ============================================================================

class MemoryArena:
    """
    Scoped memory arena with enforced ownership.

    Stores AudioContracts, LoadedAudio objects, and FeatureBundles.
    All audio transformations (resampling, mono conversion) are delegated
    to AcousticIntelligence. The arena is pure storage and borrowing.

    STORAGE HIERARCHY:
        LoadedAudio  → stored as-is via _store_loaded_audio()
                       proxy methods (at_rate, at_44k, at_22k) preserved
        StemWithRate → converted to AudioContract via .to_contract()
                       sample_rate embedded in the contract
        AudioContract → stored via store_contract()
        FeatureBundle → stored via store_feature_bundle()
        numpy array  → converted to AudioContract (legacy, warns)

    MATERIALIZATION:
        prepare_audio()              → explicit, scheduled, Guardian-approved
        prepare_for_analysis()       → convenience (ANALYSIS_SR, mono)
        prepare_for_pitch_detection() → convenience (CREPE_SR, mono)
        prepare_for_rhythm_detection() → convenience (BASIC_PITCH_SR, mono)

    BORROW:
        borrow_contract()  → primary: returns stored object (AudioContract or LoadedAudio)
        borrow_mono()      → convenience: samples from a pre-materialized mono buffer
        borrow_stereo()    → convenience: samples from a pre-materialized stereo buffer
        borrow()           → legacy: raw samples (deprecated, use borrow_contract)
    """

    def __init__(
        self,
        guardian: Any,
        name: str,
        parent_arena: Optional["MemoryArena"] = None,
        status_reporter: Optional[StatusReporterProtocol] = None,
        music_box: Optional[MusicBoxProtocol] = None,
        max_size_mb: float = MEMORY_LIMIT_MB,
    ) -> None:
        self.guardian = guardian
        self.name = name
        self.parent = parent_arena
        self._status_reporter = status_reporter
        self._music_box = music_box
        self._max_size_mb = max_size_mb

        self._buffers: Dict[str, BufferSlot] = {}
        self._pinned_buffers: Dict[str, BufferSlot] = {}
        self._pool: Dict[str, List[BufferSlot]] = {}
        self._prepared_cache: Dict[str, BufferID] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._active_borrows: Dict[str, List[BufferID]] = {}

        self._created_at = datetime.now()
        self._exited = False
        self._total_bytes_allocated = 0
        self._total_bytes_freed = 0
        self._peak_bytes = 0
        self._pool_hits = 0
        self._pool_misses = 0
        self._transfer_in_count = 0
        self._transfer_out_count = 0
        self._tier_counts: Dict[MemoryTier, int] = {t: 0 for t in MemoryTier}

        if guardian:
            guardian.register_arena(self)
            guardian._current_stage = name

        self._log_status(f"MemoryArena '{name}' created (v5.3.1)")

    # ========================================================================
    # TYPE DETECTION HELPERS (F7)
    # ========================================================================

    @staticmethod
    def _is_loaded_audio(buffer: Any) -> bool:
        """
        Detect a LoadedAudio object by its characteristic proxy methods.

        LoadedAudio has at_rate(), sha256_hash, and sample_rate.

        NOTE: Removed the verify_integrity check because LoadedAudio
        actually has verify_integrity() method (for forensic verification).
        The detection now relies on at_rate() presence as the primary
        differentiator from AudioContract.
        """
        return (
            hasattr(buffer, "at_rate")
            and hasattr(buffer, "sha256_hash")
            and hasattr(buffer, "sample_rate")
        )

    @staticmethod
    def _is_audio_contract(buffer: Any) -> bool:
        """Detect a canonical AudioContract."""
        return (
            hasattr(buffer, "samples")
            and hasattr(buffer, "sample_rate")
            and hasattr(buffer, "verify_integrity")
        )

    @staticmethod
    def _is_stem_with_rate(buffer: Any) -> bool:
        """Detect a StemWithRate wrapper."""
        return isinstance(buffer, StemWithRate)

    @staticmethod
    def _is_feature_bundle(buffer: Any) -> bool:
        """Detect a FeatureBundle by its characteristic methods."""
        return (
            hasattr(buffer, "memory_mb")
            and hasattr(buffer, "has_evidence")
            and hasattr(buffer, "release")
        )

    # ========================================================================
    # SIZE AND SAMPLE RATE HELPERS (F5, F6)
    # ========================================================================

    @staticmethod
    def _get_buffer_size(buffer: Any) -> int:
        """
        Get buffer size in bytes — handles all stored types.

        Priority:
            AudioContract → samples.nbytes (most accurate)
            LoadedAudio   → audio.nbytes (raw array size)
            StemWithRate  → audio.nbytes
            FeatureBundle → memory_mb property
            list/other    → conservative estimate
        """
        # AudioContract
        if hasattr(buffer, "samples") and hasattr(buffer.samples, "nbytes"):
            return buffer.samples.nbytes
        # LoadedAudio or StemWithRate: has .audio numpy array
        if hasattr(buffer, "audio") and hasattr(buffer.audio, "nbytes"):
            return buffer.audio.nbytes
        # FeatureBundle: reports its own size
        if hasattr(buffer, "memory_mb"):
            return int(buffer.memory_mb * 1024 * 1024)
        # Raw numpy array
        if hasattr(buffer, "nbytes"):
            return buffer.nbytes
        # List of floats etc.
        if isinstance(buffer, list):
            return len(buffer) * 8
        # Unknown — minimum non-zero
        return 1024

    @staticmethod
    def _get_buffer_sample_rate(buffer: Any) -> Optional[int]:
        """
        Extract sample rate from any stored buffer type.

        Priority (most to least reliable):
            1. AudioContract — canonical, verified
            2. LoadedAudio   — carries original SR, has proxy methods
            3. StemWithRate  — explicit SR field
            4. FeatureBundle — SR at which features were computed
            5. numpy array with .sample_rate attr — best effort
            6. None          — unknown, caller must handle

        This replaces the old _ensure_sample_rate() pattern where
        the arena silently skipped resampling when SR was unknown.
        Now callers know when SR is missing and can error explicitly.
        """
        # AudioContract (canonical)
        if (hasattr(buffer, "sample_rate")
                and hasattr(buffer, "samples")
                and hasattr(buffer, "verify_integrity")):
            return int(buffer.sample_rate)

        # LoadedAudio (proxy object from ingestion)
        if (hasattr(buffer, "sample_rate")
                and hasattr(buffer, "at_rate")):
            return int(buffer.sample_rate)

        # StemWithRate
        if isinstance(buffer, StemWithRate):
            return int(buffer.sample_rate)

        # FeatureBundle
        if hasattr(buffer, "sample_rate") and hasattr(buffer, "has_evidence"):
            return int(buffer.sample_rate)

        # Generic fallback — raw numpy with attached SR
        if hasattr(buffer, "sample_rate"):
            sr = buffer.sample_rate
            if isinstance(sr, int) and sr > 0:
                return sr

        return None

    def _get_contract_size_bytes(self, contract: AudioContract) -> int:
        """Get size of a canonical AudioContract in bytes."""
        return contract.samples.nbytes

    # ========================================================================
    # VALIDATION (F10, F13, F14)
    # ========================================================================

    def _validate_contract(
        self, contract: AudioContract, context: str = "unknown"
    ) -> None:
        """
        Validate a canonical AudioContract via AcousticIntelligence.

        NOT called for LoadedAudio — it has its own validation path.
        NOT called for FeatureBundle.
        """
        AcousticIntelligence.validate_contract(
            contract, context=f"arena_{self.name}_{context}"
        )

    def _sanitize_audio(
        self, audio: np.ndarray, context: str = "unknown"
    ) -> np.ndarray:
        """
        Sanitize raw audio — delegates to AcousticIntelligence (F13).
        Handles NaN, inf, and out-of-range values.
        """
        return AcousticIntelligence.sanitize_audio(
            audio, context=f"arena_{self.name}_{context}"
        )

    def _validate_audio(
        self,
        audio: np.ndarray,
        context: str = "unknown",
        expected_ndim: Optional[int] = None,
        expected_dtype: Optional[np.dtype] = None,
    ) -> None:
        """
        Validate raw audio via AcousticIntelligence (F14).
        Used when a legacy raw array is passed to store().
        """
        try:
            temp = AcousticIntelligence.create_contract(
                audio=audio,  # FIXED: was 'raw_audio', now 'audio'
                sample_rate=44100,
                source=f"arena_validation_{context}",
                force_mono=False,
            )
            AcousticIntelligence.validate_contract(
                temp, context=f"arena_{self.name}_{context}"
            )
        except Exception as exc:
            raise ValueError(
                f"Audio validation failed in {context}: {exc}"
            ) from exc

        if expected_ndim is not None and audio.ndim != expected_ndim:
            raise ValueError(
                f"Expected ndim={expected_ndim}, got {audio.ndim} in {context}"
            )
        if expected_dtype is not None and audio.dtype != expected_dtype:
            raise ValueError(
                f"Expected dtype={expected_dtype}, got {audio.dtype} in {context}"
            )

    # ========================================================================
    # STORAGE — PRIMARY API (F1, F3, F4)
    # ========================================================================

    def store(
        self,
        name: str,
        buffer: Any,
        stem_type: Any = None,
        pin: bool = False,
        priority: ArenaPriority = ArenaPriority.NORMAL,
        borrow_scope: BorrowScope = BorrowScope.TEMPORARY,
        tier: MemoryTier = MemoryTier.TIER_2,
    ) -> BufferID:
        """
        Store any supported buffer type in the arena.

        DETECTION ORDER (most specific first):
            1. LoadedAudio   → _store_loaded_audio() — preserves proxy methods
            2. AudioContract → store_contract() — canonical path
            3. StemWithRate  → to_contract() then store_contract()
            4. FeatureBundle → store_feature_bundle()
            5. numpy array   → create contract via AcousticIntelligence (warns)

        CRITICAL: LoadedAudio MUST be detected before AudioContract.
        LoadedAudio has sample_rate but NOT verify_integrity. If we check
        AudioContract first using a broad duck-type, we might misidentify it.
        The _is_loaded_audio() check is authoritative.

        LAW: "All audio eventually becomes AudioContract.
               All stems carry sample rate. LoadedAudio is sacred."
        """
        if self._exited:
            raise RuntimeError(f"Arena '{self.name}' has already exited")

        # ── 1. LoadedAudio ────────────────────────────────────────────────
        # MUST be first. LoadedAudio proxy methods (at_rate, at_22k, at_44k)
        # are lost the moment we extract .audio from it.
        if self._is_loaded_audio(buffer):
            return self._store_loaded_audio(
                name, buffer, pin, priority, borrow_scope, tier
            )

        # ── 2. AudioContract ──────────────────────────────────────────────
        if self._is_audio_contract(buffer):
            return self.store_contract(
                name, buffer, pin, priority, borrow_scope, tier
            )

        # ── 3. StemWithRate ───────────────────────────────────────────────
        if self._is_stem_with_rate(buffer):
            self._log_status(
                f"Converting StemWithRate({buffer.stem_type.value} "
                f"@{buffer.sample_rate}Hz) to AudioContract for storage"
            )
            contract = buffer.to_contract(source=f"arena_store_{name}")
            return self.store_contract(
                name, contract, pin, priority, borrow_scope, tier
            )

        # ── 4. FeatureBundle ──────────────────────────────────────────────
        if self._is_feature_bundle(buffer):
            return self.store_feature_bundle(name, buffer, pin, priority, tier)

        # ── 5. Legacy: raw numpy array ────────────────────────────────────
        # Warn loudly. Raw arrays have no sample rate — we have to guess.
        sr = self._get_buffer_sample_rate(buffer)
        if sr is None:
            sr = TARGET_SAMPLE_RATE
            self._log_status(
                f"[WARN] store('{name}'): raw buffer with no sample_rate. "
                f"Assuming {sr}Hz. This is the stem sample rate bug. "
                f"Wrap stems in StemWithRate before storing.",
                "warn",
            )
        else:
            self._log_status(
                f"[WARN] store('{name}'): raw buffer detected. "
                f"Creating AudioContract automatically at {sr}Hz. "
                f"Consider wrapping in StemWithRate or AudioContract explicitly.",
                "warn",
            )

        # FIXED: Changed 'raw_audio' to 'audio' to match AcousticIntelligence signature
        contract = AcousticIntelligence.create_contract(
            audio=buffer,  # ← THE FIX: was 'raw_audio', now 'audio'
            sample_rate=sr,
            source=f"arena_legacy_store_{name}",
            force_mono=False,
        )
        return self.store_contract(name, contract, pin, priority, borrow_scope, tier)

    def store_contract(
        self,
        name: str,
        contract: AudioContract,
        pin: bool = False,
        priority: ArenaPriority = ArenaPriority.NORMAL,
        borrow_scope: BorrowScope = BorrowScope.TEMPORARY,
        tier: MemoryTier = MemoryTier.TIER_2,
    ) -> BufferID:
        """
        Store a canonical AudioContract. This is the primary storage path
        for all audio that has been properly wrapped.
        """
        if self._exited:
            raise RuntimeError(f"Arena '{self.name}' has already exited")

        self._validate_contract(contract, context=f"store_{name}")

        buffer_id = BufferID(arena_name=self.name, buffer_name=name)

        existing = self._get_slot(name)
        if existing is not None:
            existing.borrow_count += 1
            existing.touch()
            return existing.id

        size_bytes = self._get_contract_size_bytes(contract)

        slot = BufferSlot(
            id=buffer_id,
            contract=contract,
            size_bytes=size_bytes,
            is_loaded_audio_obj=False,
            state=BufferState.PINNED if pin else BufferState.ACTIVE,
            priority=priority,
            borrow_scope=borrow_scope,
            is_feature_bundle=False,
            tier=tier,
        )

        self._register_slot(name, slot, pin)
        self._log_event(
            "store_contract", name, size_bytes,
            {
                "tier": tier.name,
                "sample_rate": contract.sample_rate,
                "layout": str(contract.channel_layout),
            },
        )
        return buffer_id

    def _store_loaded_audio(
        self,
        name: str,
        loaded_audio: Any,
        pin: bool = False,
        priority: ArenaPriority = ArenaPriority.NORMAL,
        borrow_scope: BorrowScope = BorrowScope.STAGE,
        tier: MemoryTier = MemoryTier.TIER_0,
    ) -> BufferID:
        """
        Store a LoadedAudio object directly — preserving its proxy methods.

        WHY A SEPARATE METHOD:
            LoadedAudio has at_rate(), at_44k, at_22k, at_16k proxy methods.
            If we call store_contract() instead, we'd have to extract
            contract = loaded_audio.contract, which gives us an AudioContract
            but loses the proxy methods. Feature_extraction then can't call
            loaded_audio.at_rate(22050) and has to materialize manually.

            By storing LoadedAudio as-is, callers get the full proxy object
            from borrow_contract() and can call at_rate() freely.

        F4: This is the fix. The slot.is_loaded_audio_obj=True flag tells
        every downstream method that this slot stores a LoadedAudio, not
        an AudioContract.
        """
        if self._exited:
            raise RuntimeError(f"Arena '{self.name}' has already exited")

        buffer_id = BufferID(arena_name=self.name, buffer_name=name)

        existing = self._get_slot(name)
        if existing is not None:
            existing.borrow_count += 1
            existing.touch()
            return existing.id

        size_bytes = self._get_buffer_size(loaded_audio)

        slot = BufferSlot(
            id=buffer_id,
            contract=loaded_audio,      # Store the whole LoadedAudio object
            size_bytes=size_bytes,
            is_loaded_audio_obj=True,   # Flag: this is NOT an AudioContract
            state=BufferState.PINNED if pin else BufferState.ACTIVE,
            priority=priority,
            borrow_scope=borrow_scope,
            is_feature_bundle=False,
            tier=tier,
        )

        self._register_slot(name, slot, pin)
        self._log_event(
            "store_loaded_audio", name, size_bytes,
            {
                "tier": tier.name,
                "has_at_rate": hasattr(loaded_audio, "at_rate"),
                "sample_rate": getattr(loaded_audio, "sample_rate", "unknown"),
            },
        )
        return buffer_id

    def store_feature_bundle(
        self,
        name: str,
        bundle: Any,
        pin: bool = True,
        priority: ArenaPriority = ArenaPriority.NORMAL,
        tier: MemoryTier = MemoryTier.TIER_1,
    ) -> BufferID:
        """Store a FeatureBundle in the arena."""
        if self._exited:
            raise RuntimeError(f"Arena '{self.name}' has already exited")

        buffer_id = BufferID(arena_name=self.name, buffer_name=name)

        existing = self._get_slot(name)
        if existing is not None:
            existing.borrow_count += 1
            existing.touch()
            return existing.id

        size_bytes = self._get_buffer_size(bundle)

        slot = BufferSlot(
            id=buffer_id,
            contract=bundle,
            size_bytes=size_bytes,
            is_loaded_audio_obj=False,
            state=BufferState.PINNED if pin else BufferState.ACTIVE,
            priority=priority,
            borrow_scope=BorrowScope.STAGE,
            is_feature_bundle=True,
            tier=tier,
        )

        self._register_slot(name, slot, pin)
        self._log_event(
            "store_feature_bundle", name, size_bytes,
            {"tier": tier.name, "bundle_mb": getattr(bundle, "memory_mb", 0)},
        )
        return buffer_id

    def _register_slot(
        self, name: str, slot: BufferSlot, pin: bool
    ) -> None:
        """Register a slot and update accounting."""
        if pin:
            self._pinned_buffers[name] = slot
        else:
            self._buffers[name] = slot

        self._total_bytes_allocated += slot.size_bytes
        self._peak_bytes = max(
            self._peak_bytes,
            self._total_bytes_allocated - self._total_bytes_freed,
        )
        self._tier_counts[slot.tier] = self._tier_counts.get(slot.tier, 0) + 1

        if self.guardian:
            self.guardian._global_buffer_count += 1
            self.guardian._global_buffer_bytes += slot.size_bytes

        self._check_pressure()

    # ========================================================================
    # MATERIALIZATION — EXPLICIT HEAVY WORK (separate from borrow)
    # ========================================================================

    def prepare_audio(
        self,
        source_id: Union[BufferID, str],
        target_sr: Optional[int] = None,
        mono: bool = True,
        cache_key: Optional[str] = None,
        agent_name: str = "preparer",
    ) -> BufferID:
        """
        Explicitly materialize audio at a target sample rate and layout.

        This is WHERE heavy DSP work happens. It is NEVER called inside
        borrow_contract(). Callers must call prepare_audio() BEFORE borrowing
        when they need a specific format.

        LAW: "Materialize once. Borrow many times."
        LAW: "prepare_audio() is schedulable. borrow_contract() is O(1)."

        Returns a BufferID for the prepared buffer, which can be borrowed
        cheaply via borrow_mono() or borrow_contract() afterwards.
        """
        if cache_key is None:
            cache_key = self._generate_cache_key(source_id, target_sr, mono)

        if cache_key in self._prepared_cache:
            self._cache_hits += 1
            self._log_event("cache_hit", cache_key, 0, {})
            return self._prepared_cache[cache_key]

        self._cache_misses += 1

        if self.guardian:
            estimated_mb = self._estimate_preparation_memory(
                source_id, target_sr, mono
            )
            alloc = self.guardian.request_allocation(agent_name, estimated_mb, can_evict=True)
            if not alloc.approved:
                raise RuntimeError(
                    f"Guardian denied allocation for prepare_audio: need {estimated_mb}MB"
                )

        with self.borrow_contract(source_id, agent_name) as source:
            current = source

            # Handle LoadedAudio proxy — use its at_rate() method
            if self._is_loaded_audio(current):
                if target_sr is not None:
                    current = current.at_rate(target_sr, mono=mono)
                elif mono and getattr(current, "num_channels", 2) > 1:
                    current = current.to_mono()
            else:
                # Standard AudioContract path
                if mono and current.channel_layout != ChannelLayout.MONO:
                    current = current.to_mono()

                if target_sr is not None and target_sr != current.sample_rate:
                    current = AcousticIntelligence.resample(
                        contract=current,
                        target_sample_rate=target_sr,
                        quality="high",
                    )
                    self._log_event(
                        "materialize_resample", str(source_id), 0,
                        {"from_sr": source.sample_rate, "to_sr": target_sr},
                    )

        prepared_name = f"prepared_{cache_key}"
        prepared_id = self.store_contract(
            name=prepared_name,
            contract=current,
            pin=True,
            tier=MemoryTier.TIER_2,
            borrow_scope=BorrowScope.STAGE,
        )

        slot = self._get_slot(prepared_name)
        if slot:
            slot.state = BufferState.PREPARED
            slot.prepared_from = (
                source_id if isinstance(source_id, BufferID)
                else self._resolve_buffer(source_id).id
            )
            slot.preparation_params = {"target_sr": target_sr, "mono": mono}

        self._prepared_cache[cache_key] = prepared_id

        if self.guardian:
            self.guardian.release_allocation(agent_name)

        return prepared_id

    def prepare_for_analysis(self, source_id: Union[BufferID, str]) -> BufferID:
        """Convenience: ANALYSIS_SR mono for general analysis agents."""
        return self.prepare_audio(
            source_id, target_sr=ANALYSIS_SR, mono=True,
            cache_key=f"analysis_{ANALYSIS_SR}_mono"
        )

    def prepare_for_pitch_detection(
        self, source_id: Union[BufferID, str]
    ) -> BufferID:
        """Convenience: CREPE_SR mono for Basic Pitch / CREPE."""
        return self.prepare_audio(
            source_id, target_sr=CREPE_SR, mono=True,
            cache_key=f"pitch_{CREPE_SR}_mono"
        )

    def prepare_for_rhythm_detection(
        self, source_id: Union[BufferID, str]
    ) -> BufferID:
        """Convenience: BASIC_PITCH_SR mono for rhythm detection."""
        return self.prepare_audio(
            source_id, target_sr=BASIC_PITCH_SR, mono=True,
            cache_key=f"rhythm_{BASIC_PITCH_SR}_mono"
        )

    def prepare_for_separation(
        self, source_id: Union[BufferID, str]
    ) -> BufferID:
        """Convenience: TARGET_SAMPLE_RATE stereo for Demucs / BSRoformer."""
        return self.prepare_audio(
            source_id, target_sr=TARGET_SAMPLE_RATE, mono=False,
            cache_key=f"separation_{TARGET_SAMPLE_RATE}_stereo"
        )

    def clear_cache(self) -> None:
        """
        Clear the materialization cache (F12 — Gemini audit).

        Forces rematerialization on the next prepare_audio() call.
        Use when a source buffer has been replaced and cached variants
        are stale. Also call after memory pressure events.
        """
        count = len(self._prepared_cache)
        self._prepared_cache.clear()
        self._log_status(f"Cleared prepared buffer cache ({count} entries)")

    def _generate_cache_key(
        self,
        source_id: Union[BufferID, str],
        target_sr: Optional[int],
        mono: bool,
    ) -> str:
        return "_".join([
            str(source_id),
            f"sr_{target_sr or 'orig'}",
            "mono" if mono else "stereo",
        ])

    def _estimate_preparation_memory(
        self,
        source_id: Union[BufferID, str],
        target_sr: Optional[int],
        mono: bool,
    ) -> int:
        """Estimate MB needed for materialization (F11 — uses _get_buffer_sample_rate)."""
        slot = self._resolve_buffer(source_id)
        if slot is None:
            return 100

        original_mb = slot.get_size_mb()
        original_sr = self._get_buffer_sample_rate(slot.contract)

        if target_sr is not None and original_sr and original_sr > 0:
            ratio = target_sr / original_sr
            estimated_mb = original_mb * ratio
        else:
            estimated_mb = original_mb

        # Stereo → mono halves the size
        num_channels = getattr(slot.contract, "num_channels", None)
        if mono and num_channels and num_channels > 1:
            estimated_mb *= 0.5

        return int(estimated_mb * 1.2) + 10

    # ========================================================================
    # BORROW API (F8, F9)
    # ========================================================================

    @contextmanager
    def borrow_contract(
        self,
        buffer_id: Union[BufferID, str],
        agent_name: str,
        scope: Optional[BorrowScope] = None,
        read_only: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> Iterator[Any]:
        """
        Borrow the stored object — AudioContract OR LoadedAudio.

        Return type is Any (F8) because we store LoadedAudio directly.
        Callers can check type via _is_loaded_audio() or isinstance.

        This is O(1). No heavy work. Just ref-count and yield.
        If you need audio at a specific sample rate, call prepare_audio()
        BEFORE borrowing, then borrow the prepared buffer.
        """
        slot = self._resolve_buffer(buffer_id)
        if slot is None:
            raise KeyError(
                f"Buffer '{buffer_id}' not found in arena '{self.name}'. "
                f"Available: {list(self._buffers) + list(self._pinned_buffers)}"
            )

        if not read_only:
            raise RuntimeError(
                "Stored objects are immutable. Cannot borrow for writing."
            )

        borrow_scope = scope or slot.borrow_scope
        record = BorrowRecord(
            buffer_id=slot.id,
            agent_name=agent_name,
            stage=(
                self.guardian._current_stage
                if self.guardian else "unknown"
            ),
            borrowed_at=datetime.now(),
            scope=borrow_scope,
            expires_at=(
                datetime.now() + ttl_seconds
                if ttl_seconds else None
            ),
        )

        slot.borrow_records.append(record)
        slot.borrow_count += 1
        slot.state = BufferState.BORROWED
        slot.touch()

        self._active_borrows.setdefault(agent_name, []).append(slot.id)

        if self.guardian:
            self.guardian.record_borrow(slot.id, agent_name)

        self._log_event(
            "borrow", slot.id.short, slot.size_bytes,
            {
                "agent": agent_name,
                "is_loaded_audio": slot.is_loaded_audio_obj,
                "sample_rate": slot.sample_rate,
            },
        )

        try:
            yield slot.contract   # LoadedAudio OR AudioContract OR FeatureBundle
        finally:
            self._release_borrow(slot, agent_name, record)

    @contextmanager
    def borrow_mono(
        self,
        buffer_id: Union[BufferID, str],
        agent_name: str,
        read_only: bool = True,
    ) -> Iterator[np.ndarray]:
        """
        Borrow a pre-materialized mono buffer's samples.

        REQUIRES: Buffer must already be mono (call prepare_audio() first).
        Returns: np.ndarray of shape (n_samples,)
        Raises: RuntimeError if buffer is not mono.

        This is a cheap operation — no resampling, no conversion.
        """
        with self.borrow_contract(buffer_id, agent_name, read_only=read_only) as obj:
            # Handle LoadedAudio
            if self._is_loaded_audio(obj):
                mono_audio = obj.to_mono() if hasattr(obj, "to_mono") else obj
                audio_arr = getattr(mono_audio, "audio", getattr(mono_audio, "samples", None))
                if audio_arr is None:
                    raise RuntimeError(
                        f"LoadedAudio buffer '{buffer_id}' has no accessible samples"
                    )
                if audio_arr.ndim > 1:
                    raise RuntimeError(
                        f"Buffer '{buffer_id}' is not mono. "
                        f"Call prepare_audio(mono=True) first."
                    )
                yield audio_arr
                return

            # Standard AudioContract path
            if obj.channel_layout != ChannelLayout.MONO:
                raise RuntimeError(
                    f"Buffer '{buffer_id}' is not mono "
                    f"(layout={obj.channel_layout}). "
                    f"Call prepare_audio(mono=True) first."
                )
            yield obj.samples

    @contextmanager
    def borrow_stereo(
        self,
        buffer_id: Union[BufferID, str],
        agent_name: str,
        read_only: bool = True,
    ) -> Iterator[np.ndarray]:
        """
        Borrow a pre-materialized stereo buffer's samples.

        REQUIRES: Buffer must already be stereo.
        Returns: np.ndarray of shape (n_samples, 2)
        """
        with self.borrow_contract(buffer_id, agent_name, read_only=read_only) as obj:
            if self._is_loaded_audio(obj):
                audio_arr = getattr(obj, "audio", None)
                if audio_arr is None:
                    raise RuntimeError(f"LoadedAudio '{buffer_id}' has no .audio")
                if audio_arr.ndim != 2 or audio_arr.shape[1] != 2:
                    raise RuntimeError(
                        f"Buffer '{buffer_id}' is not stereo. shape={audio_arr.shape}"
                    )
                yield audio_arr
                return

            if obj.channel_layout != ChannelLayout.STEREO:
                raise RuntimeError(
                    f"Buffer '{buffer_id}' is not stereo "
                    f"(layout={obj.channel_layout}). "
                    f"Call prepare_for_separation() first."
                )
            yield obj.samples

    @contextmanager
    def borrow_feature_bundle(
        self,
        buffer_id: Union[BufferID, str],
        agent_name: str,
    ) -> Iterator[Any]:
        """Borrow a FeatureBundle with type validation."""
        with self.borrow_contract(buffer_id, agent_name) as bundle:
            if not self._is_feature_bundle(bundle):
                raise TypeError(
                    f"Buffer '{buffer_id}' is not a FeatureBundle. "
                    f"Type: {type(bundle).__name__}"
                )
            yield bundle

    @contextmanager
    def borrow(
        self,
        buffer_id: Union[BufferID, str],
        agent_name: str,
        scope: Optional[BorrowScope] = None,
        read_only: bool = True,
        ttl_seconds: Optional[float] = None,
    ) -> Iterator[np.ndarray]:
        """
        Legacy borrow — returns raw samples array.

        Deprecated: prefer borrow_contract() for AudioContracts and
        borrow_mono() / borrow_stereo() for pre-materialized audio.
        Kept for backward compatibility with existing agent code.
        """
        with self.borrow_contract(
            buffer_id, agent_name, scope, read_only, ttl_seconds
        ) as obj:
            if self._is_loaded_audio(obj):
                yield getattr(obj, "audio", None)
            elif self._is_feature_bundle(obj):
                yield obj
            else:
                yield obj.samples

    # ========================================================================
    # BUFFER MANAGEMENT (internal)
    # ========================================================================

    def _get_slot(self, name: str) -> Optional[BufferSlot]:
        # parent_arena was stored (see __init__) but never actually
        # consulted here - a child arena could never see anything the
        # parent stored, despite that being the whole documented point of
        # nested_arenas() (confirmed by direct test: child.borrow_mono()
        # on a buffer the parent stored raised KeyError). Falling back up
        # the parent chain is what makes that documented behavior real.
        slot = self._buffers.get(name) or self._pinned_buffers.get(name)
        if slot is None and self.parent is not None:
            return self.parent._get_slot(name)
        return slot

    def _resolve_buffer(
        self, buffer_id: Union[BufferID, str]
    ) -> Optional[BufferSlot]:
        if isinstance(buffer_id, BufferID):
            name = buffer_id.buffer_name
        else:
            parts = str(buffer_id).split(":")
            name = parts[1] if len(parts) >= 2 else parts[0]
        return self._get_slot(name)

    def _release_borrow(
        self, slot: BufferSlot, agent_name: str, record: BorrowRecord
    ) -> None:
        borrows = self._active_borrows.get(agent_name, [])
        self._active_borrows[agent_name] = [
            bid for bid in borrows if bid != slot.id
        ]

        slot.borrow_count = max(0, slot.borrow_count - 1)
        try:
            slot.borrow_records.remove(record)
        except ValueError:
            pass

        if slot.borrow_count == 0:
            slot.state = BufferState.ACTIVE

        slot.touch()

        if self.guardian:
            self.guardian.record_release(slot.id, agent_name)

        self._log_event(
            "release_borrow", slot.id.short, slot.size_bytes,
            {"agent": agent_name},
        )

    def revoke_borrows_by_agent(
        self, agent_name: str, force: bool = False
    ) -> int:
        """Revoke all active borrows held by an agent (e.g. zombie thread cleanup)."""
        buffer_ids = list(self._active_borrows.get(agent_name, []))
        revoked = sum(
            1 for bid in buffer_ids if self.revoke_borrow(bid, force=force)
        )
        self._log_event(
            "revoke_by_agent", agent_name, 0, {"count": revoked, "force": force}
        )
        return revoked

    def revoke_borrow(
        self, buffer_id: Union[BufferID, str], force: bool = False
    ) -> bool:
        """Revoke a specific borrow (expired or force-cleanup)."""
        slot = self._resolve_buffer(buffer_id)
        if not slot or slot.borrow_count == 0:
            return False

        if not force:
            if not all(r.is_expired for r in slot.borrow_records):
                return False

        for record in slot.borrow_records:
            agent = record.agent_name
            if agent in self._active_borrows:
                self._active_borrows[agent] = [
                    bid for bid in self._active_borrows[agent]
                    if bid != slot.id
                ]
            if self.guardian:
                self.guardian.record_release(slot.id, agent)

        slot.borrow_records.clear()
        slot.borrow_count = 0
        slot.state = BufferState.ACTIVE
        self._log_event("revoke", slot.id.short, slot.size_bytes, {"force": force})
        return True

    # ========================================================================
    # LIFECYCLE — PIN / UNPIN / RELEASE
    # ========================================================================

    def pin(self, buffer_id: Union[BufferID, str]) -> bool:
        slot = self._resolve_buffer(buffer_id)
        if slot is None:
            return False
        name = slot.id.buffer_name
        if name in self._buffers:
            s = self._buffers.pop(name)
            s.state = BufferState.PINNED
            self._pinned_buffers[name] = s
            self._log_event("pin", name, s.size_bytes)
            return True
        return False

    def unpin(self, buffer_id: Union[BufferID, str]) -> bool:
        slot = self._resolve_buffer(buffer_id)
        if slot is None:
            return False
        name = slot.id.buffer_name
        if name in self._pinned_buffers:
            s = self._pinned_buffers.pop(name)
            s.state = BufferState.ACTIVE
            self._buffers[name] = s
            self._log_event("unpin", name, s.size_bytes)
            return True
        return False

    def release(
        self, buffer_id: Union[BufferID, str], force: bool = False
    ) -> bool:
        """Release a buffer and free its memory."""
        slot = self._resolve_buffer(buffer_id)
        if slot is None:
            return False

        if not force and slot.borrow_count > 0:
            self._log_event(
                "release_blocked", slot.id.short, slot.size_bytes,
                {"borrow_count": slot.borrow_count},
            )
            return False

        if force and slot.borrow_count > 0:
            self.revoke_borrow(slot.id, force=True)

        name = slot.id.buffer_name

        # Remove from cache if it was a prepared buffer
        stale_keys = [
            k for k, v in self._prepared_cache.items() if v == slot.id
        ]
        for k in stale_keys:
            del self._prepared_cache[k]

        removed = self._buffers.pop(name, None) or self._pinned_buffers.pop(name, None)
        if removed is None:
            return False

        self._total_bytes_freed += slot.size_bytes
        self._tier_counts[slot.tier] = max(
            0, self._tier_counts.get(slot.tier, 0) - 1
        )
        slot.state = BufferState.RELEASED

        if self.guardian:
            self.guardian._global_buffer_count -= 1
            self.guardian._global_buffer_bytes -= slot.size_bytes

        self._add_to_pool(slot)
        self._log_event("release", name, slot.size_bytes, {"force": force})
        del slot.contract
        del slot
        return True

    def release_all(self, keep_pinned: bool = True) -> Dict[str, Any]:
        """Release all non-borrowed buffers."""
        freed_bytes = 0
        freed_count = 0
        kept = []

        self._prepared_cache.clear()

        for name in list(self._buffers):
            slot = self._buffers[name]
            if slot.borrow_count == 0:
                freed_bytes += slot.size_bytes
                freed_count += 1
                self._total_bytes_freed += slot.size_bytes
                self._tier_counts[slot.tier] = max(
                    0, self._tier_counts.get(slot.tier, 0) - 1
                )
                self._add_to_pool(slot)
                del slot.contract
                del self._buffers[name]

        if keep_pinned:
            kept = list(self._pinned_buffers)
        else:
            for name in list(self._pinned_buffers):
                slot = self._pinned_buffers[name]
                if slot.borrow_count == 0:
                    freed_bytes += slot.size_bytes
                    freed_count += 1
                    self._total_bytes_freed += slot.size_bytes
                    self._tier_counts[slot.tier] = max(
                        0, self._tier_counts.get(slot.tier, 0) - 1
                    )
                    del slot.contract
                    del self._pinned_buffers[name]

        return {
            "arena": self.name,
            "freed_bytes": freed_bytes,
            "freed_mb": freed_bytes / (1024 * 1024),
            "freed_count": freed_count,
            "kept_buffers": kept,
            "peak_mb": self._peak_bytes / (1024 * 1024),
        }

    def _add_to_pool(self, slot: BufferSlot) -> None:
        size_mb = slot.get_size_mb()
        if 1 < size_mb < 100:
            key = "feature_bundle" if slot.is_feature_bundle else "audio_contract"
            pool = self._pool.setdefault(key, [])
            if len(pool) < 5:
                slot.state = BufferState.POOLED
                pool.append(slot)

    # ========================================================================
    # EVICTION
    # ========================================================================

    def get_evictable_buffers(
        self,
        strategy: EvictionStrategy = EvictionStrategy.BALANCED,
    ) -> List[EvictableBuffer]:
        eligible: Set[MemoryTier] = {
            EvictionStrategy.NONE:         set(),
            EvictionStrategy.CONSERVATIVE: {MemoryTier.SPILLABLE},
            EvictionStrategy.BALANCED:     {MemoryTier.TIER_2, MemoryTier.TIER_3, MemoryTier.SPILLABLE},
            EvictionStrategy.AGGRESSIVE:   {MemoryTier.TIER_1, MemoryTier.TIER_2, MemoryTier.TIER_3, MemoryTier.SPILLABLE},
        }.get(strategy, set())

        evictable = [
            EvictableBuffer(
                name=name, arena=self,
                size_bytes=slot.size_bytes, tier=slot.tier,
                is_pinned=False, borrow_count=slot.borrow_count,
                last_accessed=slot.last_accessed,
            )
            for name, slot in self._buffers.items()
            if slot.state != BufferState.PINNED
            and slot.tier in eligible
            and slot.borrow_count == 0
        ]

        if strategy == EvictionStrategy.AGGRESSIVE:
            evictable += [
                EvictableBuffer(
                    name=name, arena=self,
                    size_bytes=slot.size_bytes, tier=slot.tier,
                    is_pinned=True, borrow_count=slot.borrow_count,
                    last_accessed=slot.last_accessed,
                )
                for name, slot in self._pinned_buffers.items()
                if slot.tier in eligible and slot.borrow_count == 0
            ]

        evictable.sort(key=lambda x: (x.tier.value, x.last_accessed))
        return evictable

    def release_evictable(
        self, strategy: EvictionStrategy = EvictionStrategy.BALANCED
    ) -> int:
        evictable = self.get_evictable_buffers(strategy)
        freed = sum(buf.size_bytes for buf in evictable)
        for buf in evictable:
            self.release(buf.name, force=True)
        if freed > 0:
            gc.collect()
        self._log_status(
            f"Evicted {len(evictable)} buffers, "
            f"freed {freed / (1024 * 1024):.1f}MB"
        )
        return freed

    def get_freeable_memory_mb(self) -> float:
        return sum(
            s.size_bytes for s in self._buffers.values()
            if s.borrow_count == 0 and s.tier.value >= 2
        ) / (1024 * 1024)

    # ========================================================================
    # MEMORY PRESSURE
    # ========================================================================

    def _check_pressure(self) -> None:
        current_mb = self._get_current_memory_mb()
        if current_mb > MEMORY_WARNING_THRESHOLD_MB:
            self._log_status(
                f"Memory pressure: {current_mb:.1f}MB > "
                f"{MEMORY_WARNING_THRESHOLD_MB}MB — evicting",
                "warn",
            )
            self._evict_pooled_buffers()
            self._revoke_expired_borrows()

    def _get_current_memory_mb(self) -> float:
        total = sum(
            s.size_bytes
            for s in list(self._buffers.values()) + list(self._pinned_buffers.values())
        )
        return total / (1024 * 1024)

    def _evict_pooled_buffers(self) -> None:
        count = 0
        for pool in self._pool.values():
            for slot in pool:
                del slot.contract
                count += 1
            pool.clear()
        if count:
            self._log_event("pool_evict", "all", 0, {"count": count})

    def _revoke_expired_borrows(self) -> None:
        revoked = 0
        for slot in list(self._buffers.values()) + list(self._pinned_buffers.values()):
            if any(r.is_expired for r in slot.borrow_records):
                self.revoke_borrow(slot.id, force=False)
                revoked += 1
        if revoked:
            self._log_event("revoke_expired", "all", 0, {"count": revoked})

    # ========================================================================
    # ZERO-COPY TRANSFER
    # ========================================================================

    def transfer_to(
        self,
        buffer_id: Union[BufferID, str],
        target_arena: "MemoryArena",
        new_name: Optional[str] = None,
        transfer_borrows: bool = False,
    ) -> Optional[BufferID]:
        """Zero-copy transfer of a buffer to another arena."""
        slot = self._resolve_buffer(buffer_id)
        if slot is None:
            return None

        if not transfer_borrows and slot.borrow_count > 0:
            return None

        name = slot.id.buffer_name
        target_name = new_name or name

        if slot.is_loaded_audio_obj:
            new_id = target_arena._store_loaded_audio(
                target_name, slot.contract,
                pin=(slot.state == BufferState.PINNED),
                priority=slot.priority,
                borrow_scope=slot.borrow_scope,
                tier=slot.tier,
            )
        elif slot.is_feature_bundle:
            new_id = target_arena.store_feature_bundle(
                target_name, slot.contract,
                pin=(slot.state == BufferState.PINNED),
                priority=slot.priority,
                tier=slot.tier,
            )
        else:
            new_id = target_arena.store_contract(
                target_name, slot.contract,
                pin=(slot.state == BufferState.PINNED),
                priority=slot.priority,
                borrow_scope=slot.borrow_scope,
                tier=slot.tier,
            )

        if transfer_borrows:
            for record in slot.borrow_records:
                target_arena._active_borrows.setdefault(
                    record.agent_name, []
                ).append(new_id)

        self._buffers.pop(name, None)
        self._pinned_buffers.pop(name, None)
        slot.state = BufferState.TRANSFERRED
        self._transfer_out_count += 1
        target_arena._transfer_in_count += 1

        # The store_contract()/_store_loaded_audio()/store_feature_bundle()
        # call above already incremented target_arena.guardian's global
        # counts as a normal side effect of registering the new buffer
        # there - but removing the buffer from THIS arena's local dict
        # above bypasses release() (the only other place these counts are
        # decremented), so without this the source guardian's global
        # count/bytes silently drift upward forever across repeated
        # transfers.
        if self.guardian:
            self.guardian._global_buffer_count -= 1
            self.guardian._global_buffer_bytes -= slot.size_bytes

        self._log_event(
            "transfer",
            f"{self.name}:{name} → {target_arena.name}:{target_name}",
            slot.size_bytes,
        )
        return new_id

    # ========================================================================
    # QUERY
    # ========================================================================

    def __contains__(self, buffer_id: Union[BufferID, str]) -> bool:
        return self._resolve_buffer(buffer_id) is not None

    def __len__(self) -> int:
        return len(self._buffers) + len(self._pinned_buffers)

    def list_buffers(self) -> List[BufferStats]:
        return [
            slot.get_stats()
            for slot in list(self._buffers.values()) + list(self._pinned_buffers.values())
        ]

    def get_borrowers(self, buffer_id: Union[BufferID, str]) -> List[str]:
        slot = self._resolve_buffer(buffer_id)
        return [r.agent_name for r in slot.borrow_records] if slot else []

    def get_buffer_by_tier(self, tier: MemoryTier) -> List[str]:
        return [
            name for name, slot in {**self._buffers, **self._pinned_buffers}.items()
            if slot.tier == tier
        ]

    def get_stats(self) -> ArenaStats:
        all_slots = list(self._buffers.values()) + list(self._pinned_buffers.values())
        total_bytes = sum(s.size_bytes for s in all_slots)
        return ArenaStats(
            name=self.name,
            buffer_count=len(all_slots),
            borrowed_count=sum(1 for s in all_slots if s.state == BufferState.BORROWED),
            pinned_count=len(self._pinned_buffers),
            pooled_count=sum(len(v) for v in self._pool.values()),
            prepared_count=sum(1 for s in all_slots if s.state == BufferState.PREPARED),
            loaded_audio_count=sum(1 for s in all_slots if s.is_loaded_audio_obj),
            total_bytes=total_bytes,
            total_mb=total_bytes / (1024 * 1024),
            allocated_mb=self._total_bytes_allocated / (1024 * 1024),
            freed_mb=self._total_bytes_freed / (1024 * 1024),
            peak_mb=self._peak_bytes / (1024 * 1024),
            age_seconds=(datetime.now() - self._created_at).total_seconds(),
            pool_hits=self._pool_hits,
            pool_misses=self._pool_misses,
            transfer_in_count=self._transfer_in_count,
            transfer_out_count=self._transfer_out_count,
            tier_breakdown={t.name: c for t, c in self._tier_counts.items()},
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
        )

    # ========================================================================
    # ARENA LIFECYCLE
    # ========================================================================

    def exit(self, keep_pinned: bool = True) -> Dict[str, Any]:
        if self._exited:
            return {"already_exited": True}

        self._exited = True
        self._prepared_cache.clear()
        result = self.release_all(keep_pinned=keep_pinned)

        if not keep_pinned:
            for name in list(self._pinned_buffers):
                slot = self._pinned_buffers[name]
                if slot.borrow_count == 0:
                    self._total_bytes_freed += slot.size_bytes
                    del slot.contract
                    del self._pinned_buffers[name]

        self._log_event("arena_exit", self.name, result["freed_bytes"], result)

        if result["freed_bytes"] > STAGGERED_GC_TRIGGER_MB * 1024 * 1024:
            if self.guardian:
                self.guardian.staggered_gc(self.name)

        self._log_status(
            f"Arena '{self.name}' exited. Freed: {result['freed_mb']:.1f}MB"
        )
        return result

    def __enter__(self) -> "MemoryArena":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type:
            self._log_event(
                "arena_exception", str(exc_type), 0,
                {"exception": str(exc_val)},
            )
        self.exit(keep_pinned=True)
        return False   # Never suppress exceptions

    # ========================================================================
    # LOGGING
    # ========================================================================

    def _log_event(
        self,
        event_type: str,
        name: str,
        size_bytes: int,
        extra: Optional[Dict] = None,
    ) -> None:
        if self._music_box:
            entry = {
                "arena": self.name,
                "event": event_type,
                "buffer": name,
                "size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 * 1024),
            }
            if extra:
                entry.update(extra)
            self._music_box.log_decision(
                stage_name=f"arena_{self.name}",
                decision_type=event_type,
                before_state={},
                after_state=entry,
                reasoning=f"Buffer {event_type}: {name}",
                reversible=False,
            )

    def _log_status(self, message: str, level: str = "info") -> None:
        if self._status_reporter:
            fn = {
                "info": self._status_reporter.info,
                "warn": self._status_reporter.warn,
                "error": self._status_reporter.error,
            }.get(level, self._status_reporter.info)
            fn(f"Arena[{self.name}]", message)


# ============================================================================
# CONVENIENCE CONTEXT MANAGERS
# ============================================================================

@contextmanager
def scoped_arena(guardian: Any, name: str, **kwargs) -> Iterator[MemoryArena]:
    """Standard scoped arena. Exits and frees memory on context exit."""
    arena = MemoryArena(guardian, name, **kwargs)
    try:
        yield arena
    finally:
        arena.exit()


@contextmanager
def nested_arenas(
    guardian: Any, parent_name: str, child_name: str, **kwargs
) -> Iterator[Tuple[MemoryArena, MemoryArena]]:
    """Create parent + child arena pair. Child exits before parent."""
    parent = MemoryArena(guardian, parent_name, **kwargs)
    child = MemoryArena(guardian, child_name, parent_arena=parent, **kwargs)
    try:
        yield parent, child
    finally:
        child.exit()
        parent.exit()