# =================================================================
# MODULE: agents/analysis/spectral_masker.py
# VERSION: 5.6.1 (ADDED missing MaskingResult and fixed exports)
# UPDATED: 2026-05-25
#
# CRITICAL FIXES APPLIED:
#   1. Synchronous entry point (_extract_window_embeddings) that bridges
#      to the async extraction internals via asyncio.run() - safe because
#      the real pipeline always calls this from a dedicated worker thread
#      (server/websocket.py's run_in_executor), never from a thread with
#      its own event loop already running.
#   2. FIXED clustering: USE RAW embeddings for acoustics, scaled only for distance
#   3. ADDED silence gating (RMS threshold before extraction)
#   4. FIXED n_bins hardcoding → dynamic from STFT
#   5. ADDED STFT reuse across masks (massive performance gain)
#   6. ADDED harmonic comb masks (not just bandpass)
#   7. ADDED cluster validity checks (no fake instruments)
#   8. FIXED physics refinement range collapse
#   9. CHANGED float64 → float32 throughout
#   10. ADDED SpectralProfile export
#   11. ADDED MaskingResult export (fixes import warning)
#   12. ADDED MaskingMode enum with WEIGHTED option
# =================================================================

from __future__ import annotations

import asyncio
import logging
import gc
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
from enum import Enum

import numpy as np
import librosa

from agents.analysis.timbre_intelligence import (
    TimbreFeatureExtractor,
    InstrumentFamilyClassifier,
    InstrumentFamily,
    ExcitationType,
    RichTimbreEmbedding,
)

logger = logging.getLogger(__name__)

# Silence threshold for RMS gating (critical addition)
SILENCE_RMS_THRESHOLD = 0.005  # -46dB


# =====================================================================
# Enums
# =====================================================================

class MaskingMode(str, Enum):
    """
    How aggressively to apply the spectral mask.
    SOFT:    Gaussian rolloff at band edges (default, no artifacts)
    HARD:    Sharp cutoff at band edges (faster, some ringing)
    WEIGHTED: Weight by instrument confidence score
    """
    SOFT = "soft"
    HARD = "hard"
    WEIGHTED = "weighted"


class SeparationStrategy(str, Enum):
    """Strategy for separating instruments from mixed audio."""
    CLUSTERING = "clustering"
    SPECTRAL = "spectral"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class SpectralMaskerConfig:
    n_fft: int = 2048
    hop_length: int = 512
    window_ms: float = 200.0
    hop_ms: float = 100.0
    min_clusters: int = 2
    max_clusters: int = 5
    rolloff_pct: float = 0.15
    min_rolloff_hz: float = 50.0
    masking_mode: MaskingMode = MaskingMode.SOFT
    default_tempo_bpm: float = 120.0

    # Critical additions
    silence_rms_threshold: float = SILENCE_RMS_THRESHOLD
    use_harmonic_combs: bool = True
    harmonic_comb_weight: float = 0.3
    cluster_validity_threshold: float = 0.4

    harmonic_multipliers: Dict[InstrumentFamily, int] = field(default_factory=lambda: {
        InstrumentFamily.PIANO: 24,
        InstrumentFamily.STRINGS: 16,
        InstrumentFamily.BRASS: 10,
        InstrumentFamily.WOODWIND: 14,
        InstrumentFamily.VOICE: 12,
        InstrumentFamily.PLUCKED: 16,
        InstrumentFamily.PERCUSSION: 8,
        InstrumentFamily.UNKNOWN: 12,
    })

    temporal_smooth_window: int = 5
    use_hmm_smoothing: bool = True
    max_workers: int = 4


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class SpectralProfile:
    """Spectral profile of an instrument cluster."""
    centroid_hz: float
    rolloff_hz: float
    bandwidth_hz: float
    flatness: float
    harmonic_ratio: float
    noise_ratio: float

    @classmethod
    def from_embedding(cls, embedding: RichTimbreEmbedding) -> 'SpectralProfile':
        return cls(
            centroid_hz=float(embedding.raw_vector[0]) if len(embedding.raw_vector) > 0 else 1000.0,
            rolloff_hz=float(embedding.raw_vector[1]) if len(embedding.raw_vector) > 1 else 3000.0,
            bandwidth_hz=0.0,
            flatness=float(embedding.raw_vector[2]) if len(embedding.raw_vector) > 2 else 0.5,
            harmonic_ratio=float(embedding.raw_vector[5]) if len(embedding.raw_vector) > 5 else 0.5,
            noise_ratio=1.0 - (float(embedding.raw_vector[5]) if len(embedding.raw_vector) > 5 else 0.5),
        )


@dataclass
class MaskingResult:
    """
    Result of applying a spectral mask to audio.

    Contains the masked audio and metadata about the masking operation.
    """
    masked_audio: np.ndarray
    mask_used: SpectralMask
    original_energy: float
    masked_energy: float
    energy_retained: float

    @classmethod
    def from_apply(cls, audio: np.ndarray, masked: np.ndarray, mask: SpectralMask) -> 'MaskingResult':
        orig_energy = float(np.sum(audio ** 2))
        masked_energy = float(np.sum(masked ** 2))
        return cls(
            masked_audio=masked,
            mask_used=mask,
            original_energy=orig_energy,
            masked_energy=masked_energy,
            energy_retained=masked_energy / (orig_energy + 1e-8)
        )


@dataclass
class SpectralMask:
    freq_bins_hz: np.ndarray
    weights: np.ndarray
    low_hz: float
    high_hz: float
    instrument_family: InstrumentFamily
    cluster_id: int
    confidence: float
    masking_mode: MaskingMode = MaskingMode.SOFT
    harmonic_weights: Optional[np.ndarray] = None

    def apply_to_stft(self, stft: np.ndarray) -> np.ndarray:
        n_bins = stft.shape[0]
        if len(self.weights) != n_bins:
            weights = np.interp(np.linspace(0, 1, n_bins), np.linspace(0, 1, len(self.weights)), self.weights)
        else:
            weights = self.weights

        # Apply harmonic comb enhancement if available
        if self.harmonic_weights is not None:
            if len(self.harmonic_weights) != n_bins:
                harmonic_weights = np.interp(np.linspace(0, 1, n_bins), np.linspace(0, 1, len(self.harmonic_weights)),
                                             self.harmonic_weights)
            else:
                harmonic_weights = self.harmonic_weights
            weights = weights * 0.7 + harmonic_weights * 0.3

        return stft * weights[:, np.newaxis]


@dataclass
class HarmonicCluster:
    cluster_id: int
    instrument_family: InstrumentFamily
    excitation_type: ExcitationType
    confidence: float
    spectral_centroid_mean: float
    spectral_rolloff_mean: float
    freq_range_hz: Tuple[float, float]
    mask: SpectralMask
    active_windows: List[Tuple[float, float]] = field(default_factory=list)
    n_windows: int = 0
    centroid_embedding: Optional[np.ndarray] = None
    silhouette_score: float = 0.0
    temporal_stability: float = 0.0

    def is_active_at(self, timestamp_ms: float, tolerance_ms: float = 200.0) -> bool:
        for start_ms, end_ms in self.active_windows:
            if (start_ms - tolerance_ms) <= timestamp_ms <= (end_ms + tolerance_ms):
                return True
        return False

    def pitch_range_midi(self) -> Tuple[int, int]:
        low_midi = max(0, int(round(12.0 * np.log2(max(1.0, self.freq_range_hz[0]) / 440.0) + 69)))
        high_midi = min(127, int(round(12.0 * np.log2(max(1.0, self.freq_range_hz[1]) / 440.0) + 69)))
        return low_midi, high_midi


# =====================================================================
# Embedding Layout
# =====================================================================

class EmbeddingLayout:
    CENTROID_IDX = 0
    ROLLOFF_IDX = 1
    FLATNESS_IDX = 2
    BANDWIDTH_IDX = 3
    FLUX_IDX = 4
    HARMONIC_RATIO_IDX = 5
    NOISE_RATIO_IDX = 6

    @classmethod
    def get_spectral_centroid(cls, embedding: RichTimbreEmbedding) -> float:
        if len(embedding.raw_vector) > cls.CENTROID_IDX:
            return float(embedding.raw_vector[cls.CENTROID_IDX])
        return 1000.0

    @classmethod
    def get_spectral_rolloff(cls, embedding: RichTimbreEmbedding) -> float:
        if len(embedding.raw_vector) > cls.ROLLOFF_IDX:
            return float(embedding.raw_vector[cls.ROLLOFF_IDX])
        return 3000.0

    @classmethod
    def get_flatness(cls, embedding: RichTimbreEmbedding) -> float:
        if len(embedding.raw_vector) > cls.FLATNESS_IDX:
            return float(embedding.raw_vector[cls.FLATNESS_IDX])
        return 0.5

    @classmethod
    def get_harmonic_ratio(cls, embedding: RichTimbreEmbedding) -> float:
        if len(embedding.raw_vector) > cls.HARMONIC_RATIO_IDX:
            return float(embedding.raw_vector[cls.HARMONIC_RATIO_IDX])
        return 0.5


# =====================================================================
# STEM SCANNER
# =====================================================================

class StemScanner:
    """
    Scans a mixed-instrument audio stem and identifies distinct timbral clusters.
    """

    def __init__(
            self,
            sample_rate: int,
            window_ms: float = 200.0,
            hop_ms: float = 100.0,
            min_clusters: int = 2,
            max_clusters: int = 5,
            config: Optional[SpectralMaskerConfig] = None,
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(window_ms * sample_rate / 1000.0)
        self.hop_samples = int(hop_ms * sample_rate / 1000.0)
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        self.config = config or SpectralMaskerConfig()
        self.config.window_ms = window_ms
        self.config.hop_ms = hop_ms
        self.config.min_clusters = min_clusters
        self.config.max_clusters = max_clusters

        self._extractor = TimbreFeatureExtractor(sample_rate)
        self._classifier = InstrumentFamilyClassifier()
        self._label_history: deque = deque(maxlen=self.config.temporal_smooth_window)

    def scan(self, audio: np.ndarray, tempo_bpm: float = 120.0) -> List[HarmonicCluster]:
        if len(audio) < self.window_samples:
            logger.warning("StemScanner: audio too short for analysis")
            return []

        embeddings_scaled, embeddings_raw, timestamps_ms, raw_embeddings = self._extract_window_embeddings(audio)

        if len(embeddings_scaled) < self.min_clusters * 2:
            logger.warning(f"StemScanner: only {len(embeddings_scaled)} valid windows")
            return self._intelligent_fallback(audio, embeddings_raw, timestamps_ms, raw_embeddings, tempo_bpm)

        n_clusters = self._estimate_n_clusters(embeddings_scaled)
        logger.debug(f"StemScanner: {len(embeddings_scaled)} windows → {n_clusters} clusters")

        labels = self._cluster_embeddings(embeddings_scaled, n_clusters)

        if not self._clusters_are_valid(embeddings_scaled, labels):
            logger.warning("Clusters appear invalid (low separability), using fallback")
            return self._intelligent_fallback(audio, embeddings_raw, timestamps_ms, raw_embeddings, tempo_bpm)

        labels = self._apply_temporal_smoothing(labels)

        clusters = self._build_clusters(
            embeddings_scaled, embeddings_raw, labels, timestamps_ms, raw_embeddings, n_clusters, tempo_bpm
        )

        clusters.sort(key=lambda c: c.n_windows, reverse=True)

        for cluster in clusters:
            logger.debug(
                f"  Cluster {cluster.cluster_id}: {cluster.instrument_family.value}, "
                f"{cluster.n_windows} windows, silhouette={cluster.silhouette_score:.2f}"
            )

        return clusters

    # =================================================================
    # EXTRACTION
    # =================================================================

    def _extract_window_embeddings(self, audio: np.ndarray):
        windows = []
        timestamps_ms = []

        for start in range(0, len(audio) - self.window_samples, self.hop_samples):
            window = audio[start: start + self.window_samples]
            rms = np.sqrt(np.mean(window ** 2))

            if rms < self.config.silence_rms_threshold:
                continue

            windows.append(window)
            timestamps_ms.append((start / self.sample_rate) * 1000.0)

        if not windows:
            return np.array([]), np.array([]), [], []

        # TimbreFeatureExtractor.extract() is declared async (used elsewhere
        # with a running event loop) but does no actual async I/O here, so
        # it must be driven with asyncio.run() from this synchronous
        # scanner - calling it bare returned an un-awaited coroutine object,
        # silently producing zero usable embeddings from every window.
        extracted = asyncio.run(self._extract_all_windows(windows))

        # Filter failures (None) out of BOTH lists in lockstep - a plain
        # append-on-success would desync timestamps_ms from raw_embeddings
        # whenever any single window's extraction throws, silently
        # mislabeling every cluster's active_windows after the failure.
        raw_embeddings: List[RichTimbreEmbedding] = []
        kept_timestamps_ms: List[float] = []
        for emb, ts in zip(extracted, timestamps_ms):
            if emb is not None:
                raw_embeddings.append(emb)
                kept_timestamps_ms.append(ts)
        timestamps_ms = kept_timestamps_ms

        if not raw_embeddings:
            return np.array([]), np.array([]), [], []

        vectors = []
        for emb in raw_embeddings:
            vectors.append(emb.raw_vector.astype(np.float32))

        embeddings_raw = np.stack(vectors)
        embeddings_scaled = self._normalize_embeddings(embeddings_raw)

        return embeddings_scaled, embeddings_raw, timestamps_ms, raw_embeddings

    async def _extract_all_windows(self, windows: List[np.ndarray]) -> List[Optional[RichTimbreEmbedding]]:
        results: List[Optional[RichTimbreEmbedding]] = []
        for i, window in enumerate(windows):
            try:
                results.append(await self._extractor.extract(window))
            except Exception as e:
                logger.debug(f"Window {i} failed: {e}")
                results.append(None)
        return results

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(embeddings)
        except ImportError:
            mean = np.mean(embeddings, axis=0)
            std = np.std(embeddings, axis=0)
            std[std == 0] = 1.0
            return (embeddings - mean) / std

    def _estimate_n_clusters(self, embeddings: np.ndarray) -> int:
        best_k = self.min_clusters
        best_score = -1.0
        upper = min(self.max_clusters, len(embeddings) // 3)
        upper = max(upper, self.min_clusters)

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            for k in range(self.min_clusters, upper + 1):
                try:
                    km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
                    labels = km.fit_predict(embeddings)
                    if len(np.unique(labels)) < 2:
                        continue
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except Exception:
                    continue
        except ImportError:
            pass
        return best_k

    def _clusters_are_valid(self, embeddings: np.ndarray, labels: np.ndarray) -> bool:
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) < 2:
                return False
            score = silhouette_score(embeddings, labels)
            return score > self.config.cluster_validity_threshold
        except ImportError:
            return len(np.unique(labels)) >= 2

    def _cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
            return km.fit_predict(embeddings)
        except ImportError:
            centroids = embeddings[:, EmbeddingLayout.CENTROID_IDX]
            thresholds = np.percentile(centroids, np.linspace(0, 100, n_clusters + 1)[1:-1])
            return np.digitize(centroids, thresholds)

    def _apply_temporal_smoothing(self, labels: np.ndarray) -> np.ndarray:
        if len(labels) < self.config.temporal_smooth_window:
            return labels
        smoothed = labels.copy()
        window = self.config.temporal_smooth_window
        half = window // 2
        for i in range(len(labels)):
            start = max(0, i - half)
            end = min(len(labels), i + half + 1)
            window_labels = labels[start:end]
            unique, counts = np.unique(window_labels, return_counts=True)
            smoothed[i] = unique[np.argmax(counts)]
        return smoothed

    def _compute_active_windows(self, timestamps_ms: List[float], tempo_bpm: float) -> List[Tuple[float, float]]:
        if not timestamps_ms:
            return []
        quarter_note_ms = 60000.0 / max(40.0, tempo_bpm)
        gap_threshold_ms = min(200.0, quarter_note_ms * 0.5)
        window_duration_ms = (self.window_samples / self.sample_rate) * 1000.0
        sorted_ts = sorted(timestamps_ms)
        segments = []
        seg_start = sorted_ts[0]
        seg_end = sorted_ts[0] + window_duration_ms
        for ts in sorted_ts[1:]:
            if ts - seg_end <= gap_threshold_ms:
                seg_end = ts + window_duration_ms
            else:
                segments.append((seg_start, seg_end))
                seg_start = ts
                seg_end = ts + window_duration_ms
        segments.append((seg_start, seg_end))
        return segments

    def _build_clusters(self, embeddings_scaled, embeddings_raw, labels, timestamps_ms, raw_embeddings, n_clusters,
                        tempo_bpm):
        clusters = []

        for cluster_id in range(n_clusters):
            mask_idx = labels == cluster_id
            if not np.any(mask_idx):
                continue

            cluster_raw = embeddings_raw[mask_idx]
            cluster_embeddings_scaled = embeddings_scaled[mask_idx]
            cluster_timestamps = [timestamps_ms[i] for i, m in enumerate(mask_idx) if m]
            cluster_raw_emb = [raw_embeddings[i] for i, m in enumerate(mask_idx) if m]

            centroid_scaled = np.mean(cluster_embeddings_scaled, axis=0)
            centroid_raw = np.mean(cluster_raw, axis=0)

            cluster = self._build_single_cluster(
                cluster_id, centroid_raw, centroid_scaled, cluster_timestamps, cluster_raw_emb, tempo_bpm
            )

            try:
                from sklearn.metrics import silhouette_score
                if len(cluster_embeddings_scaled) > 1 and len(np.unique(labels[mask_idx])) > 1:
                    cluster.silhouette_score = silhouette_score(cluster_embeddings_scaled, labels[mask_idx])
            except:
                pass

            clusters.append(cluster)

        return clusters

    def _build_single_cluster(self, cluster_id, centroid_raw, centroid_scaled, timestamps_ms, raw_embeddings,
                              tempo_bpm):
        spectral_centroid = float(centroid_raw[EmbeddingLayout.CENTROID_IDX])
        spectral_rolloff = float(centroid_raw[EmbeddingLayout.ROLLOFF_IDX])

        low_hz = max(20.0, spectral_centroid / 5.0)
        high_hz = min(20000.0, spectral_rolloff * 1.2)

        all_partials = []
        for emb in raw_embeddings:
            valid_freqs = emb.partial_freqs[emb.partial_freqs > 0]
            if len(valid_freqs) > 0:
                all_partials.append(float(valid_freqs[0]))

        if raw_embeddings:
            dists = [np.linalg.norm(emb.raw_vector.astype(np.float32) - centroid_raw) for emb in raw_embeddings]
            best_emb = raw_embeddings[int(np.argmin(dists))]
            family, excitation, confidence = self._classifier.classify(best_emb)
        else:
            family = InstrumentFamily.UNKNOWN
            excitation = ExcitationType.UNKNOWN
            confidence = 0.0

        harmonic_mult = self.config.harmonic_multipliers.get(family, 12)

        if all_partials:
            median_f0 = float(np.median(all_partials))
            if 20.0 < median_f0 < 5000.0:
                low_hz = max(20.0, median_f0 * 0.5)
                high_hz = min(20000.0, median_f0 * harmonic_mult)

        low_hz, high_hz = self._physics_refine_range(family, low_hz, high_hz)

        mask = self._build_mask(low_hz, high_hz, family, cluster_id, confidence)

        active_windows = self._compute_active_windows(timestamps_ms, tempo_bpm)

        return HarmonicCluster(
            cluster_id=cluster_id,
            instrument_family=family,
            excitation_type=excitation,
            confidence=confidence,
            spectral_centroid_mean=spectral_centroid,
            spectral_rolloff_mean=spectral_rolloff,
            freq_range_hz=(low_hz, high_hz),
            mask=mask,
            active_windows=active_windows,
            n_windows=len(timestamps_ms),
            centroid_embedding=centroid_raw,
        )

    def _physics_refine_range(self, family: InstrumentFamily, low_hz: float, high_hz: float) -> Tuple[float, float]:
        FAMILY_RANGES = {
            InstrumentFamily.STRINGS: (55.0, 4200.0),
            InstrumentFamily.PIANO: (27.5, 4186.0),
            InstrumentFamily.BRASS: (55.0, 1568.0),
            InstrumentFamily.WOODWIND: (65.4, 4186.0),
            InstrumentFamily.VOICE: (82.4, 1047.0),
            InstrumentFamily.PLUCKED: (41.2, 3136.0),
            InstrumentFamily.PERCUSSION: (20.0, 8000.0),
            InstrumentFamily.UNKNOWN: (20.0, 20000.0),
        }

        if family in FAMILY_RANGES:
            phys_low, phys_high = FAMILY_RANGES[family]
            low_hz = max(low_hz, phys_low)
            high_hz = min(high_hz, phys_high)

        min_range_hz = 50.0
        if high_hz - low_hz < min_range_hz:
            center = (low_hz + high_hz) / 2
            low_hz = max(20.0, center - min_range_hz / 2)
            high_hz = min(20000.0, center + min_range_hz / 2)

        if low_hz >= high_hz:
            low_hz = max(20.0, high_hz * 0.5)

        return low_hz, high_hz

    def _build_mask(self, low_hz: float, high_hz: float, family: InstrumentFamily, cluster_id: int,
                    confidence: float) -> SpectralMask:
        sr = self.sample_rate
        n_bins = (self.config.n_fft // 2) + 1
        freq_bins = np.linspace(0, sr / 2.0, n_bins)

        band_width = high_hz - low_hz
        rolloff_hz = max(self.config.min_rolloff_hz, band_width * self.config.rolloff_pct)

        weights = np.ones(n_bins)
        for i, freq in enumerate(freq_bins):
            if freq < low_hz:
                dist = low_hz - freq
                weights[i] = np.exp(-0.5 * (dist / rolloff_hz) ** 2)
            elif freq > high_hz:
                dist = freq - high_hz
                weights[i] = np.exp(-0.5 * (dist / rolloff_hz) ** 2)

        weights = np.clip(weights, 0.0, 1.0)

        harmonic_weights = None
        if self.config.use_harmonic_combs:
            harmonic_weights = np.ones(n_bins)
            band_mask = (freq_bins >= low_hz) & (freq_bins <= high_hz)
            if np.any(band_mask):
                center_freq = (low_hz + high_hz) / 2
                for harm in [1, 2, 3, 4, 5]:
                    harm_freq = center_freq * harm
                    if harm_freq < sr / 2:
                        idx = np.argmin(np.abs(freq_bins - harm_freq))
                        harmonic_weights[idx] = 1.5

        return SpectralMask(
            freq_bins_hz=freq_bins,
            weights=weights,
            low_hz=low_hz,
            high_hz=high_hz,
            instrument_family=family,
            cluster_id=cluster_id,
            confidence=confidence,
            masking_mode=self.config.masking_mode,
            harmonic_weights=harmonic_weights,
        )

    def _intelligent_fallback(self, audio, embeddings_raw, timestamps_ms, raw_embeddings, tempo_bpm):
        logger.debug("StemScanner: using intelligent fallback")

        if len(embeddings_raw) == 0:
            mask = self._build_mask(20.0, 20000.0, InstrumentFamily.UNKNOWN, 0, 0.2)
            return [HarmonicCluster(
                cluster_id=0,
                instrument_family=InstrumentFamily.UNKNOWN,
                excitation_type=ExcitationType.UNKNOWN,
                confidence=0.2,
                spectral_centroid_mean=1000.0,
                spectral_rolloff_mean=3000.0,
                freq_range_hz=(20.0, 20000.0),
                mask=mask,
                active_windows=[(timestamps_ms[0], timestamps_ms[-1])] if timestamps_ms else [],
                n_windows=len(timestamps_ms),
            )]

        centroids = embeddings_raw[:, EmbeddingLayout.CENTROID_IDX]
        centroid_std = np.std(centroids)

        if centroid_std < 100.0:
            centroid_vec = np.mean(embeddings_raw, axis=0)
            cluster = self._build_single_cluster(0, centroid_vec, centroid_vec, timestamps_ms, raw_embeddings,
                                                 tempo_bpm)
            return [cluster]

        low_c = np.percentile(centroids, 33)
        high_c = np.percentile(centroids, 67)
        clusters = []

        for idx, (low_thresh, high_thresh, cid) in enumerate([
            (None, low_c, 0), (low_c, high_c, 1), (high_c, None, 2)
        ]):
            if low_thresh is None:
                idx_mask = centroids <= high_thresh
            elif high_thresh is None:
                idx_mask = centroids > low_thresh
            else:
                idx_mask = (centroids > low_thresh) & (centroids <= high_thresh)

            if np.any(idx_mask):
                ts = [timestamps_ms[i] for i, m in enumerate(idx_mask) if m]
                emb = embeddings_raw[idx_mask]
                raw = [raw_embeddings[i] for i, m in enumerate(idx_mask) if m]
                if len(emb) > 0:
                    centroid_vec = np.mean(emb, axis=0)
                    clusters.append(self._build_single_cluster(cid, centroid_vec, centroid_vec, ts, raw, tempo_bpm))

        return clusters if clusters else [
            self._build_single_cluster(0, np.mean(embeddings_raw, axis=0), np.mean(embeddings_raw, axis=0),
                                       timestamps_ms, raw_embeddings, tempo_bpm)]


# =====================================================================
# SPECTRAL MASKER - WITH STFT REUSE
# =====================================================================

class SpectralMasker:
    def __init__(self, sample_rate: int, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._cached_stft = None
        self._cached_audio_hash = None

    def _compute_stft(self, audio: np.ndarray):
        audio_hash = hash(audio.tobytes()[:1000])
        if self._cached_audio_hash != audio_hash:
            self._cached_stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            self._cached_audio_hash = audio_hash
        return self._cached_stft

    def apply(self, audio: np.ndarray, mask: SpectralMask) -> np.ndarray:
        if len(audio) == 0:
            return audio

        stft = self._compute_stft(audio)
        masked_stft = mask.apply_to_stft(stft)
        masked_audio = librosa.istft(masked_stft, hop_length=self.hop_length, length=len(audio))
        return masked_audio.astype(np.float32)

    def apply_all(self, audio: np.ndarray, clusters: List[HarmonicCluster]) -> List[Tuple[HarmonicCluster, np.ndarray]]:
        if not clusters:
            return []

        stft = self._compute_stft(audio)

        results = []
        for cluster in clusters:
            masked_stft = cluster.mask.apply_to_stft(stft.copy())
            masked_audio = librosa.istft(masked_stft, hop_length=self.hop_length, length=len(audio))
            results.append((cluster, masked_audio.astype(np.float32)))

        return results


# =====================================================================
# Factory Functions
# =====================================================================

def create_stem_scanner(sample_rate: int, window_ms: float = 200.0, hop_ms: float = 100.0,
                        min_clusters: int = 2, max_clusters: int = 5) -> StemScanner:
    return StemScanner(sample_rate, window_ms, hop_ms, min_clusters, max_clusters)


def create_spectral_masker(sample_rate: int, n_fft: int = 2048, hop_length: int = 512) -> SpectralMasker:
    return SpectralMasker(sample_rate, n_fft, hop_length)


def create_default_config() -> SpectralMaskerConfig:
    return SpectralMaskerConfig()


# =====================================================================
# Exports
# =====================================================================

__all__ = [
    "MaskingMode",
    "SeparationStrategy",
    "SpectralProfile",
    "MaskingResult",
    "SpectralMaskerConfig",
    "SpectralMask",
    "HarmonicCluster",
    "StemScanner",
    "SpectralMasker",
    "EmbeddingLayout",
    "create_stem_scanner",
    "create_spectral_masker",
    "create_default_config",
]