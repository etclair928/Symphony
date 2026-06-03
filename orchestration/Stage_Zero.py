#!/usr/bin/env python3
"""
=================================================================
MODULE: Stage_Zero.py
DESCRIPTION: Pre-Processing before Stem Separation on Audio
             Merges Stage_Zero conditioning + Pipeline Stages 1-4
VERSION: 2.0
UPDATED: 2026-05-25

THEORY: Cocktail Party Problem for Music
   - Multi-microphone approaches (spatial filtering via phase alignment)
   - Independent Component Analysis (spectral unmixing)
   - Computational Auditory Scene Analysis (CASA)
   - Harmonic constraint modeling (music theory priors)
   - Iterative "peel-back" subtraction (Bass → Soprano → Inner voices)

PIPELINE (in order):
   1. Ingestion      — Load audio, truncate, compute hash
   2. Phase Align    — GCC-PHAT sub-sample stereo correction (spatial)
   3. Feature Extract — Chroma, onset, spectral features (CASA)
   4. Tempo Analysis — BPM detection for adaptive processing
   5. Noise Floor    — NMF spectral floor profiling
   6. Reverb Suppress — Spectral subtraction of reverb tail
   7. Mid-Side       — Stereo width normalization
   8. Tonal/Transient — Adaptive HPSS (BPM-aware)
   9. Harmonic Mask  — Wiener soft mask from chord priors (music theory)
  10. Spectral Norm  — Per-band level equalization
  11. Source Count   — Estimate how many instruments are present
  12. Peel-Back      — Iterative STFT-domain voice subtraction
  13. Separation     — Demucs → Mel-Roformer → STFT fallback
  14. Export (opt)   — Save conditioned audio for downstream models

OUTPUT: Conditioned audio ready for Demucs/Roformer
=================================================================
"""

from __future__ import annotations

import asyncio
import gc
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any, Union

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================================
# Constants
# =====================================================================

DEFAULT_PROCESSING_DURATION_SECONDS = 120.0
ANALYSIS_SR = 22050
BASIC_PITCH_SR = 16000
CREPE_SR = 16000
MADMOM_SR = 22050

STAGE_TIMEOUT_LOAD_SECONDS = 30.0
STAGE_TIMEOUT_ANALYSIS_SECONDS = 60.0
STAGE_TIMEOUT_SEPARATION_SECONDS = 600.0


# =====================================================================
# Order Types (simplified from core)
# =====================================================================

class SourceType:
    PIPELINE = "pipeline"
    SCRIBE = "scribe"
    TEMPO_INTELLIGENCE = "tempo_intelligence"
    DEMUCS = "demucs"
    RHYTHM = "rhythm"
    PITCH = "pitch"
    TONAL = "tonal"
    DRUM_INTELLIGENCE = "drum_intelligence"
    ANECHOIC_MA = "anechoic_ma"


class StemType:
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    FULL_MIX = "full_mix"
    VOCALS = "vocals"


class Confidence:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class NoteEvent:
    pitch: int
    start_ms: float
    end_ms: float
    velocity: int = 64
    confidence: float = 1.0
    zero_crossing_rate: float = 0.0
    source: str = "unknown"
    fundamental_freq_hz: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.reasoning_chain is None:
            self.reasoning_chain = []


@dataclass
class AudioContext:
    file_path: str
    original_sample_rate: int
    working_sample_rate: int
    duration_seconds: float
    num_channels: int = 2
    file_hash: str = ""
    memory_mb: float = 0.0


@dataclass
class TempoMap:
    initial_tempo_bpm: float
    tempo_events: List[Dict] = field(default_factory=list)
    confidence: str = Confidence.MEDIUM


# =====================================================================
# Stage Zero Configuration
# =====================================================================

@dataclass
class StageZeroConfig:
    """
    All tunable parameters for Stage Zero processing.
    Sensible defaults for jazz small-ensemble recordings.
    """
    sample_rate: int = 44100
    n_fft: int = 4096
    hop_length: int = 512
    window: str = "hann"

    # Phase alignment (spatial filtering)
    phase_align_enabled: bool = True
    phase_max_lag_ms: float = 5.0

    # Noise floor (NMF)
    noise_nmf_enabled: bool = True
    noise_n_components: int = 4
    noise_suppress_db: float = 12.0

    # Reverb suppression
    reverb_suppress_enabled: bool = True
    reverb_tail_ms: float = 80.0
    reverb_suppress_ratio: float = 0.4

    # Mid-Side (stereo width)
    mid_side_enabled: bool = True
    mid_side_target_width: float = 0.65

    # Tonal/Transient (HPSS)
    hpss_enabled: bool = True
    hpss_bpm: float = 0.0
    hpss_kernel_min: int = 11
    hpss_kernel_max: int = 31
    hpss_tonal_weight: float = 0.85

    # Harmonic masking (music theory prior)
    harmonic_mask_enabled: bool = True
    harmonic_mask_chord: List[float] = field(default_factory=list)
    harmonic_mask_n_harmonics: int = 8
    harmonic_mask_width_cents: float = 50.0
    harmonic_mask_floor: float = 0.10

    # Spectral normalization
    spectral_norm_enabled: bool = True
    spectral_norm_n_bands: int = 24
    spectral_norm_target_db: float = -18.0

    # Source count estimation
    source_count_enabled: bool = True
    source_count_min: int = 1
    source_count_max: int = 8

    # Peel-back (iterative subtraction)
    peel_enabled: bool = True
    peel_n_passes: int = 3
    peel_subtract_alpha: float = 0.70

    # Separation model
    separation_model: str = "demucs"  # demucs, mel_roformer, auto
    enable_fallback: bool = True


# =====================================================================
# Stage Zero Report
# =====================================================================

@dataclass
class StageZeroReport:
    """Complete record of what Stage Zero found and did."""
    conditioned_stereo: np.ndarray
    original_stereo: np.ndarray
    sample_rate: int

    # Decompositions
    tonal_mono: Optional[np.ndarray] = None
    transient_mono: Optional[np.ndarray] = None
    mid_channel: Optional[np.ndarray] = None
    side_channel: Optional[np.ndarray] = None

    # Peeled stems
    peeled_bass: Optional[np.ndarray] = None
    peeled_soprano: Optional[np.ndarray] = None
    peeled_inner_voices: Optional[np.ndarray] = None

    # Separated stems (from final model)
    drums_stem: Optional[np.ndarray] = None
    bass_stem: Optional[np.ndarray] = None
    other_stem: Optional[np.ndarray] = None

    # Diagnostics
    phase_lag_samples: int = 0
    stereo_width_original: float = 0.0
    stereo_width_adjusted: float = 0.0
    estimated_source_count: int = 0
    noise_floor_db: float = 0.0
    reverb_rt60_estimate_ms: float = 0.0
    acoustic_profile: str = "unknown"
    detected_tempo_bpm: float = 120.0

    # Spectral diagnostics
    noise_mask: Optional[np.ndarray] = None
    harmonic_mask: Optional[np.ndarray] = None

    # What was applied
    stages_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    separation_model_used: str = ""

    @property
    def conditioned_mono(self) -> np.ndarray:
        return np.mean(self.conditioned_stereo, axis=0)

    def get_summary(self) -> str:
        lines = [
            "=" * 60,
            "STAGE ZERO REPORT",
            "=" * 60,
            f"Tempo:           {self.detected_tempo_bpm:.1f} BPM",
            f"Phase lag:       {self.phase_lag_samples} samples",
            f"Stereo width:    {self.stereo_width_original:.2f} → {self.stereo_width_adjusted:.2f}",
            f"Noise floor:     {self.noise_floor_db:.1f} dB",
            f"Est. sources:    {self.estimated_source_count}",
            f"RT60 estimate:   {self.reverb_rt60_estimate_ms:.0f} ms",
            f"Acoustic profile: {self.acoustic_profile}",
            f"Separation model: {self.separation_model_used}",
            f"Stages applied:  {', '.join(self.stages_applied)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =====================================================================
# Load Audio (simplified from ingestion.loader)
# =====================================================================

@dataclass
class LoadedAudio:
    audio: np.ndarray
    sample_rate: int
    original_duration: float
    is_truncated: bool
    sha256_hash: str
    memory_mb: float


def load_audio(
    file_path: str,
    target_sr: int = 44100,
    truncate_duration: float = DEFAULT_PROCESSING_DURATION_SECONDS,
) -> LoadedAudio:
    """Load audio file with truncation support."""
    try:
        import soundfile as sf
        import librosa
        import hashlib

        # Read file
        audio, orig_sr = sf.read(file_path, dtype='float32')

        # Handle mono/stereo
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        elif audio.ndim == 2 and audio.shape[1] == 2:
            audio = audio.T  # (channels, samples)
        elif audio.ndim == 2 and audio.shape[0] == 2:
            pass  # Already correct shape

        original_duration = audio.shape[1] / orig_sr

        # Resample if needed
        if orig_sr != target_sr:
            audio_resampled = []
            for ch in range(audio.shape[0]):
                ch_resampled = librosa.resample(
                    audio[ch], orig_sr=orig_sr, target_sr=target_sr
                )
                audio_resampled.append(ch_resampled)
            audio = np.array(audio_resampled)
            sr = target_sr
        else:
            sr = orig_sr

        # Truncate if requested
        is_truncated = False
        if truncate_duration > 0:
            max_samples = int(truncate_duration * sr)
            if audio.shape[1] > max_samples:
                audio = audio[:, :max_samples]
                is_truncated = True
                original_duration = truncate_duration

        # Simple hash (first 1MB)
        hash_input = audio[:, :min(44100 * 10, audio.shape[1])].tobytes()
        file_hash = hashlib.sha256(hash_input).hexdigest()[:16]

        memory_mb = audio.nbytes / (1024 * 1024)

        return LoadedAudio(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            original_duration=original_duration,
            is_truncated=is_truncated,
            sha256_hash=file_hash,
            memory_mb=memory_mb,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}")


# =====================================================================
# Core StageZeroProcessor (Condensed from original)
# =====================================================================

class StageZeroProcessor:
    """Pre-separation audio conditioner with all Cocktail Party techniques."""

    def __init__(self, config: Optional[StageZeroConfig] = None):
        self.config = config or StageZeroConfig()
        self._librosa = None
        self._scipy = None
        self._ensure_deps()

    def _ensure_deps(self):
        try:
            import librosa
            self._librosa = librosa
        except ImportError:
            raise ImportError("Stage Zero requires librosa: pip install librosa")
        try:
            import scipy.signal as signal
            self._signal = signal
        except ImportError:
            raise ImportError("Stage Zero requires scipy: pip install scipy")

    def process(
        self,
        stereo_audio: np.ndarray,
        chord_prior_hz: Optional[List[float]] = None,
        bpm_hint: Optional[float] = None,
    ) -> StageZeroReport:
        """Run full conditioning pipeline."""
        cfg = self.config
        audio = self._normalize_shape(stereo_audio)
        sr = cfg.sample_rate

        report = StageZeroReport(
            conditioned_stereo=audio.copy(),
            original_stereo=audio.copy(),
            sample_rate=sr,
        )
        work = audio.copy().astype(np.float32)

        print(f"\n[Stage Zero] Processing {work.shape[1]/sr:.1f}s stereo @ {sr}Hz")

        # Stage 1: Phase Alignment (spatial filtering)
        if cfg.phase_align_enabled and work.shape[0] == 2:
            work, lag = self._phase_align_gcc_phat(work)
            report.phase_lag_samples = lag
            report.stages_applied.append("phase_align")
            if lag != 0:
                print(f"  [phase] Lag corrected: {lag:+d} samples ({lag/sr*1000:.2f}ms)")

        # Stage 2: Noise Floor (NMF)
        noise_mask = None
        if cfg.noise_nmf_enabled:
            noise_mask, noise_db, reverb_ms = self._estimate_noise_nmf(work)
            report.noise_floor_db = noise_db
            report.reverb_rt60_estimate_ms = reverb_ms
            report.noise_mask = noise_mask
            report.stages_applied.append("noise_nmf")
            print(f"  [noise] Floor: {noise_db:.1f}dB, RT60: {reverb_ms:.0f}ms")

        # Stage 3: Reverb Suppression
        if cfg.reverb_suppress_enabled and noise_mask is not None:
            work = self._suppress_reverb(work, noise_mask)
            report.stages_applied.append("reverb_suppress")

        # Stage 4: Mid-Side Processing
        if cfg.mid_side_enabled:
            work, width_orig, width_adj = self._process_mid_side(work)
            mid = (work[0] + work[1]) / 2.0
            side = (work[0] - work[1]) / 2.0
            report.mid_channel = mid.copy()
            report.side_channel = side.copy()
            report.stereo_width_original = width_orig
            report.stereo_width_adjusted = width_adj
            report.stages_applied.append("mid_side")
            print(f"  [mid-side] Width: {width_orig:.2f} → {width_adj:.2f}")

        # Stage 5: Tonal/Transient (HPSS)
        if cfg.hpss_enabled:
            mono = np.mean(work, axis=0)
            bpm = bpm_hint or cfg.hpss_bpm or self._estimate_bpm_fast(mono)
            report.detected_tempo_bpm = bpm
            tonal, transient = self._decompose_hpss_adaptive(mono, bpm)
            report.tonal_mono = tonal.copy()
            report.transient_mono = transient.copy()
            report.stages_applied.append("hpss_adaptive")
            print(f"  [hpss] BPM={bpm:.1f}, tonal_rms={self._rms(tonal):.3f}")

        # Stage 6: Harmonic Masking (music theory prior)
        chord = chord_prior_hz or cfg.harmonic_mask_chord
        if cfg.harmonic_mask_enabled and chord:
            work_mono = np.mean(work, axis=0)
            work_mono, harmonic_mask = self._soft_harmonic_mask(work_mono, chord)
            for ch in range(2):
                work[ch] = self._apply_stft_mask(work[ch], harmonic_mask)
            report.harmonic_mask = harmonic_mask
            report.stages_applied.append("harmonic_mask")
            print(f"  [harm] Mask applied ({len(chord)} chord tones)")

        # Stage 7: Spectral Normalization
        if cfg.spectral_norm_enabled:
            work = self._spectral_normalize(work)
            report.stages_applied.append("spectral_norm")

        # Stage 8: Source Count Estimation
        if cfg.source_count_enabled:
            mono = np.mean(work, axis=0)
            n_sources, profile = self._estimate_source_count(mono)
            report.estimated_source_count = n_sources
            report.acoustic_profile = profile
            report.stages_applied.append("source_count")
            print(f"  [sources] ~{n_sources}, profile: {profile}")

        # Stage 9: Peel-Back (iterative subtraction)
        if cfg.peel_enabled:
            bass, soprano, inner = self._iterative_peel_back(np.mean(work, axis=0))
            report.peeled_bass = bass.copy()
            report.peeled_soprano = soprano.copy()
            report.peeled_inner_voices = inner.copy()
            report.stages_applied.append("peel_back")
            print(f"  [peel] bass_rms={self._rms(bass):.3f}, soprano_rms={self._rms(soprano):.3f}")

        # Final normalization
        work = self._normalize_peak(work)
        report.conditioned_stereo = work.astype(np.float32)

        print(f"[Stage Zero] Complete: {' → '.join(report.stages_applied)}")
        return report

    # -----------------------------------------------------------------
    # Stage 1: Phase Alignment - GCC-PHAT
    # -----------------------------------------------------------------
    def _phase_align_gcc_phat(self, stereo: np.ndarray) -> Tuple[np.ndarray, int]:
        cfg = self.config
        L, R = stereo[0], stereo[1]
        n = len(L)
        max_lag = int(cfg.phase_max_lag_ms * cfg.sample_rate / 1000.0)

        nfft = 2 * n
        L_fft = np.fft.rfft(L, n=nfft)
        R_fft = np.fft.rfft(R, n=nfft)

        XPS = L_fft * np.conj(R_fft)
        denom = np.abs(XPS) + 1e-10
        XPS_phat = XPS / denom
        gcc = np.fft.irfft(XPS_phat, n=nfft)

        search_range = np.concatenate([gcc[:max_lag], gcc[nfft - max_lag:]])
        lags = np.concatenate([np.arange(0, max_lag), np.arange(-max_lag, 0)])
        peak_idx = np.argmax(np.abs(search_range))
        lag = int(lags[peak_idx])

        if lag == 0:
            return stereo, 0

        aligned = stereo.copy()
        if lag > 0:
            aligned[1] = np.roll(R, -lag)
            aligned[1][-lag:] = 0
        else:
            aligned[0] = np.roll(L, -lag)
            aligned[0][lag:] = 0
        return aligned, lag

    # -----------------------------------------------------------------
    # Stage 2: Noise Floor - NMF
    # -----------------------------------------------------------------
    def _estimate_noise_nmf(self, stereo: np.ndarray) -> Tuple[np.ndarray, float, float]:
        cfg, lb = self.config, self._librosa
        mono = np.mean(stereo, axis=0)
        D = lb.stft(mono, n_fft=cfg.n_fft, hop_length=cfg.hop_length, window=cfg.window)
        mag = np.abs(D)
        eps = 1e-10

        try:
            from sklearn.decomposition import NMF
            nmf = NMF(n_components=cfg.noise_n_components, max_iter=100, random_state=42)
            W = nmf.fit_transform(mag)
            H = nmf.components_
            temporal_var = np.var(H, axis=1)
            noise_idx = temporal_var < np.percentile(temporal_var, 40)
            W_noise = W.copy()
            W_noise[:, ~noise_idx] = 0
            noise_estimate = W_noise @ H
            signal_power = np.maximum(mag - noise_estimate, 0) ** 2 + eps
            noise_power = noise_estimate ** 2 + eps
            mask = signal_power / (signal_power + noise_power)
        except ImportError:
            noise_estimate = np.percentile(mag, 15, axis=1, keepdims=True)
            noise_estimate = np.broadcast_to(noise_estimate, mag.shape)
            mask = np.clip((mag - noise_estimate) / (mag + eps), 0.0, 1.0)

        noise_rms = float(np.sqrt(np.mean(np.mean(mag * (1 - mask), axis=0) ** 2)))
        noise_db = float(20 * np.log10(noise_rms + eps))

        rms_env = lb.feature.rms(y=mono, hop_length=cfg.hop_length)[0]
        reverb_ms = self._estimate_rt60(rms_env, cfg.sample_rate, cfg.hop_length)
        return mask.astype(np.float32), noise_db, reverb_ms

    def _estimate_rt60(self, rms_env: np.ndarray, sr: int, hop_length: int) -> float:
        time_per_frame = hop_length / sr * 1000.0
        eps = 1e-10
        log_rms = 20 * np.log10(rms_env + eps)
        best_rt60 = 0.0
        for start in range(0, len(log_rms) - 10, 5):
            segment = log_rms[start:start + 50]
            if len(segment) < 5:
                break
            if np.max(segment) - np.min(segment) < 3.0:
                continue
            x = np.arange(len(segment))
            slope, _ = np.polyfit(x, segment, 1)
            if slope < -0.05:
                frames_to_60db = abs(-60.0 / slope)
                rt60_ms = frames_to_60db * time_per_frame
                if 20 < rt60_ms < 5000:
                    best_rt60 = max(best_rt60, rt60_ms)
        return float(best_rt60)

    # -----------------------------------------------------------------
    # Stage 3: Reverb Suppression
    # -----------------------------------------------------------------
    def _suppress_reverb(self, stereo: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
        cfg, lb = self.config, self._librosa
        out = np.zeros_like(stereo)
        floor = cfg.reverb_suppress_ratio
        for ch in range(stereo.shape[0]):
            D = lb.stft(stereo[ch], n_fft=cfg.n_fft, hop_length=cfg.hop_length, window=cfg.window)
            mag, phase = np.abs(D), np.angle(D)
            mask = noise_mask
            if mask.shape[1] != D.shape[1]:
                from scipy.ndimage import zoom
                mask = zoom(mask, (1.0, D.shape[1] / mask.shape[1]), order=1)
                mask = np.clip(mask, 0.0, 1.0)
            gate = floor + (1.0 - floor) * mask
            mag_g = mag * gate
            D_out = mag_g * np.exp(1j * phase)
            out[ch] = lb.istft(D_out, hop_length=cfg.hop_length, window=cfg.window, length=stereo.shape[1])
        return out.astype(np.float32)

    # -----------------------------------------------------------------
    # Stage 4: Mid-Side Processing
    # -----------------------------------------------------------------
    def _process_mid_side(self, stereo: np.ndarray) -> Tuple[np.ndarray, float, float]:
        cfg = self.config
        eps = 1e-10
        L, R = stereo[0].astype(np.float64), stereo[1].astype(np.float64)
        mid = (L + R) / 2.0
        side = (L - R) / 2.0
        mid_rms = float(np.sqrt(np.mean(mid ** 2)) + eps)
        side_rms = float(np.sqrt(np.mean(side ** 2)) + eps)
        width_original = float(np.clip(side_rms / (mid_rms + side_rms), 0, 1))

        if width_original > 0.01:
            scale = cfg.mid_side_target_width / width_original
            scale = np.clip(scale, 0.3 / max(width_original, 0.01), 1.5)
        else:
            scale = 1.0

        side_adjusted = side * scale
        side_adj_rms = float(np.sqrt(np.mean(side_adjusted ** 2)) + eps)
        width_adjusted = float(np.clip(side_adj_rms / (mid_rms + side_adj_rms), 0, 1))
        L_out = (mid + side_adjusted).astype(np.float32)
        R_out = (mid - side_adjusted).astype(np.float32)
        return np.stack([L_out, R_out]), width_original, width_adjusted

    # -----------------------------------------------------------------
    # Stage 5: Adaptive HPSS
    # -----------------------------------------------------------------
    def _estimate_bpm_fast(self, mono: np.ndarray) -> float:
        try:
            lb, cfg = self._librosa, self.config
            onset_env = lb.onset.onset_strength(y=mono, sr=cfg.sample_rate, hop_length=cfg.hop_length)
            tempo, _ = lb.beat.beat_track(onset_envelope=onset_env, sr=cfg.sample_rate, hop_length=cfg.hop_length)
            return float(tempo) if 40 < tempo < 320 else 120.0
        except Exception:
            return 120.0

    def _decompose_hpss_adaptive(self, mono: np.ndarray, bpm: float) -> Tuple[np.ndarray, np.ndarray]:
        cfg, lb = self.config, self._librosa
        fps = cfg.sample_rate / cfg.hop_length
        beat_frames = int(fps * 60.0 / max(bpm, 40.0))
        t_kernel = int(np.clip(beat_frames, cfg.hpss_kernel_min, cfg.hpss_kernel_max))
        f_kernel = 31
        if t_kernel % 2 == 0:
            t_kernel += 1
        if f_kernel % 2 == 0:
            f_kernel += 1

        D = lb.stft(mono, n_fft=cfg.n_fft, hop_length=cfg.hop_length, window=cfg.window)
        mag, phase = np.abs(D), np.angle(D)

        from scipy.ndimage import median_filter
        tonal_mag = median_filter(mag, size=(1, t_kernel))
        transient_mag = median_filter(mag, size=(f_kernel, 1))

        eps = 1e-8
        total = tonal_mag + transient_mag + eps
        mask_tonal = tonal_mag / total
        mask_transient = transient_mag / total

        tonal = lb.istft(mask_tonal * mag * np.exp(1j * phase), hop_length=cfg.hop_length, window=cfg.window, length=len(mono))
        transient = lb.istft(mask_transient * mag * np.exp(1j * phase), hop_length=cfg.hop_length, window=cfg.window, length=len(mono))
        return tonal.astype(np.float32), transient.astype(np.float32)

    # -----------------------------------------------------------------
    # Stage 6: Harmonic Masking
    # -----------------------------------------------------------------
    def _soft_harmonic_mask(self, mono: np.ndarray, chord_hz: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        cfg, lb = self.config, self._librosa
        eps = 1e-10
        D = lb.stft(mono, n_fft=cfg.n_fft, hop_length=cfg.hop_length, window=cfg.window)
        mag, phase = np.abs(D), np.angle(D)
        freqs = lb.fft_frequencies(sr=cfg.sample_rate, n_fft=cfg.n_fft)

        gain = np.full(len(freqs), cfg.harmonic_mask_floor, dtype=np.float32)
        width_cents = cfg.harmonic_mask_width_cents

        for f0_hz in chord_hz:
            if f0_hz <= 0:
                continue
            for harmonic_n in range(1, cfg.harmonic_mask_n_harmonics + 1):
                target_hz = f0_hz * harmonic_n
                if target_hz > cfg.sample_rate / 2:
                    break
                with np.errstate(divide='ignore', invalid='ignore'):
                    cents_dist = np.where(freqs > 0, 1200.0 * np.log2(freqs / target_hz), np.inf)
                bell = np.exp(-0.5 * (cents_dist / (width_cents / 2.0)) ** 2)
                gain = np.maximum(gain, bell.astype(np.float32))

        mask = gain[:, np.newaxis]
        D_masked = (mag * mask) * np.exp(1j * phase)
        out = lb.istft(D_masked, hop_length=cfg.hop_length, window=cfg.window, length=len(mono))
        return out.astype(np.float32), mask.astype(np.float32)

    def _apply_stft_mask(self, mono: np.ndarray, mask: np.ndarray) -> np.ndarray:
        cfg, lb = self.config, self._librosa
        D = lb.stft(mono, n_fft=cfg.n_fft, hop_length=cfg.hop_length, window=cfg.window)
        mag, phase = np.abs(D), np.angle(D)
        m = mask
        if m.shape[1] != D.shape[1]:
            from scipy.ndimage import zoom
            m = zoom(m, (1.0, D.shape[1] / m.shape[1]), order=1)
        D_out = mag * np.clip(m, 0, 1) * np.exp(1j * phase)
        return lb.istft(D_out, hop_length=cfg.hop_length, window=cfg.window, length=len(mono)).astype(np.float32)

    # -----------------------------------------------------------------
    # Stage 7: Spectral Normalization
    # -----------------------------------------------------------------
    def _spectral_normalize(self, stereo: np.ndarray) -> np.ndarray:
        cfg, lb = self.config, self._librosa
        eps = 1e-10
        bark_edges_hz = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000]
        target_linear = 10 ** (cfg.spectral_norm_target_db / 20.0)
        out = np.zeros_like(stereo)

        for ch in range(stereo.shape[0]):
            D = lb.stft(stereo[ch], n_fft=cfg.n_fft, hop_length=cfg.hop_length, window=cfg.window)
            mag, phase = np.abs(D), np.angle(D)
            freqs = lb.fft_frequencies(sr=cfg.sample_rate, n_fft=cfg.n_fft)
            gain = np.ones(len(freqs), dtype=np.float32)

            for band_idx in range(len(bark_edges_hz) - 1):
                lo_hz, hi_hz = bark_edges_hz[band_idx], bark_edges_hz[band_idx + 1]
                bins = np.where((freqs >= lo_hz) & (freqs < hi_hz))[0]
                if len(bins) == 0:
                    continue
                band_rms = float(np.sqrt(np.mean(mag[bins, :] ** 2)) + eps)
                if band_rms > eps:
                    raw_gain = target_linear / band_rms
                    raw_gain = np.clip(raw_gain, 0.25, 4.0)
                    smooth_gain = 0.6 * raw_gain + 0.4 * 1.0
                    gain[bins] = smooth_gain

            D_normalized = mag * gain[:, np.newaxis] * np.exp(1j * phase)
            out[ch] = lb.istft(D_normalized, hop_length=cfg.hop_length, window=cfg.window, length=stereo.shape[1])
        return self._normalize_peak(out).astype(np.float32)

    # -----------------------------------------------------------------
    # Stage 8: Source Count Estimation
    # -----------------------------------------------------------------
    def _estimate_source_count(self, mono: np.ndarray) -> Tuple[int, str]:
        cfg, lb = self.config, self._librosa
        eps = 1e-10
        D = lb.stft(mono, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        mag = np.abs(D)
        step = max(1, mag.shape[1] // 200)
        mag_sub = mag[:, ::step]

        try:
            _, sv, _ = np.linalg.svd(mag_sub, full_matrices=False)
            sv_norm = sv / (sv[0] + eps)
            threshold = 0.08
            n_sources = int(np.sum(sv_norm > threshold))
            n_sources = int(np.clip(n_sources, cfg.source_count_min, cfg.source_count_max))
        except np.linalg.LinAlgError:
            n_sources = 3

        flatness = float(np.mean(lb.feature.spectral_flatness(S=mag, n_fft=cfg.n_fft)[0]))
        rms_env = lb.feature.rms(y=mono, hop_length=cfg.hop_length)[0]
        rms_var = float(np.std(rms_env))

        if flatness < 0.04 and rms_var < 0.03:
            profile = "studio"
        elif flatness < 0.08 and rms_var < 0.08:
            profile = "live_room"
        elif flatness < 0.15:
            profile = "concert_hall"
        else:
            profile = "outdoor"

        return n_sources, profile

    # -----------------------------------------------------------------
    # Stage 9: Iterative Peel-Back
    # -----------------------------------------------------------------
    def _iterative_peel_back(self, mono: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg, lb = self.config, self._librosa
        eps = 1e-10
        D = lb.stft(mono, n_fft=cfg.n_fft, hop_length=cfg.hop_length, window=cfg.window)
        mag, phase = np.abs(D), np.angle(D)
        freqs = lb.fft_frequencies(sr=cfg.sample_rate, n_fft=cfg.n_fft)
        residual_mag = mag.copy()

        registers = [
            ("bass", 40.0, 350.0),
            ("soprano", 500.0, 3500.0),
            ("inner", 0.0, cfg.sample_rate / 2),
        ]
        peeled = []

        for name, lo_hz, hi_hz in registers:
            freq_mask = (freqs >= lo_hz) & (freqs <= hi_hz)
            if not np.any(freq_mask) or np.sum(residual_mag) < eps:
                peeled.append(np.zeros(len(mono), dtype=np.float32))
                continue

            gain = np.full(len(freqs), 0.05, dtype=np.float64)
            gain[freq_mask] = 1.0
            transitions = np.where(np.diff(freq_mask.astype(int)) != 0)[0]
            rolloff_bins = max(3, int(len(freqs) * 0.02))
            for t in transitions:
                lo = max(0, t - rolloff_bins)
                hi = min(len(freqs), t + rolloff_bins)
                ramp = np.linspace(0.05, 1.0, hi - lo)
                if freq_mask[t] if t < len(freq_mask) else False:
                    gain[lo:hi] = ramp
                else:
                    gain[lo:hi] = ramp[::-1]

            masked_mag = residual_mag * gain[:, np.newaxis]
            alpha = cfg.peel_subtract_alpha
            residual_mag = np.maximum(residual_mag - alpha * masked_mag, 0.0)
            voice_D = masked_mag * np.exp(1j * phase)
            voice = lb.istft(voice_D, hop_length=cfg.hop_length, window=cfg.window, length=len(mono))
            voice = voice.astype(np.float32)
            voice /= (np.max(np.abs(voice)) + eps)
            peeled.append(voice)

        return peeled[0], peeled[1], peeled[2]

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------
    def _normalize_shape(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            return np.stack([audio, audio])
        if audio.ndim == 2:
            if audio.shape[0] == 2:
                return audio
            if audio.shape[1] == 2:
                return audio.T
            if audio.shape[0] == 1:
                return np.stack([audio[0], audio[0]])
        raise ValueError(f"Cannot interpret audio shape {audio.shape}")

    def _normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) + 1e-10
        return (audio / peak).astype(np.float32)

    @staticmethod
    def _rms(audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-10)


# =====================================================================
# Stem Separation Models (Stage 13)
# =====================================================================

class StemSeparationResult:
    def __init__(self, drums: Optional[np.ndarray], bass: Optional[np.ndarray], other: Optional[np.ndarray]):
        self.drums = drums
        self.bass = bass
        self.other = other

    def get_stem(self, stem_type: str) -> Optional[np.ndarray]:
        if stem_type == StemType.DRUMS:
            return self.drums
        if stem_type == StemType.BASS:
            return self.bass
        if stem_type == StemType.OTHER:
            return self.other
        return None


def try_demucs_separation(
    audio_stereo: np.ndarray,
    sample_rate: int,
    timeout_seconds: float = 300.0,
) -> Optional[StemSeparationResult]:
    """Try Demucs separation (htdemucs model)."""
    try:
        from demucs import separate
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            # Ensure shape (samples, channels) for soundfile
            audio_to_write = audio_stereo.T if audio_stereo.shape[0] == 2 else audio_stereo
            sf.write(tmp.name, audio_to_write, sample_rate)

            # Run Demucs
            import subprocess
            out_dir = tempfile.mkdtemp()
            cmd = [
                'demucs', '--two-stems', 'vocals',
                '-o', out_dir, tmp.name
            ]
            subprocess.run(cmd, capture_output=True, timeout=timeout_seconds)

            # Load stems
            import os
            base_name = Path(tmp.name).stem
            stems_dir = Path(out_dir) / 'htdemucs' / base_name
            drums_path = stems_dir / 'drums.wav'
            bass_path = stems_dir / 'bass.wav'
            other_path = stems_dir / 'other.wav'

            drums = sf.read(drums_path)[0].T if drums_path.exists() else None
            bass = sf.read(bass_path)[0].T if bass_path.exists() else None
            other = sf.read(other_path)[0].T if other_path.exists() else None

            # Cleanup
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)

            return StemSeparationResult(drums, bass, other)
    except Exception as e:
        print(f"  [demucs] Failed: {e}")
        return None


def try_mel_roformer_separation(
    audio_stereo: np.ndarray,
    sample_rate: int,
) -> Optional[StemSeparationResult]:
    """Try Mel-Roformer separation."""
    try:
        from mel_roformer import MelRoformer

        # Ensure shape (samples, channels)
        audio_input = audio_stereo.T if audio_stereo.shape[0] == 2 else audio_stereo

        model = MelRoformer.from_pretrained("MYSELF/mel_roformer")
        stems = model.separate(audio_input, sample_rate)

        drums = stems.get('drums')
        bass = stems.get('bass')
        other = stems.get('other')

        # Convert to (channels, samples)
        if drums is not None and drums.ndim == 2:
            drums = drums.T
        if bass is not None and bass.ndim == 2:
            bass = bass.T
        if other is not None and other.ndim == 2:
            other = other.T

        return StemSeparationResult(drums, bass, other)
    except Exception as e:
        print(f"  [mel_roformer] Failed: {e}")
        return None


def fallback_stft_separation(
    audio_stereo: np.ndarray,
    sample_rate: int,
) -> StemSeparationResult:
    """Simple STFT-based separation using HPSS + filtering."""
    import librosa

    mono = np.mean(audio_stereo, axis=0) if audio_stereo.ndim == 2 else audio_stereo
    harmonic, percussive = librosa.effects.hpss(mono, margin=(1.0, 5.0))

    # Simple bass extraction: low-pass filter at 250Hz
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, 250, btype='low', fs=sample_rate, output='sos')
        bass = sosfiltfilt(sos, mono).astype(np.float32)
    except Exception:
        bass = np.zeros_like(mono)

    other = (harmonic - bass).astype(np.float32)
    drums = percussive.astype(np.float32)

    def to_stereo(arr):
        return np.stack([arr, arr])

    return StemSeparationResult(
        drums=to_stereo(drums),
        bass=to_stereo(bass),
        other=to_stereo(other),
    )


# =====================================================================
# Main Stage Zero Entry Point
# =====================================================================

def run_stage_zero(
    audio_path: str,
    output_path: Optional[str] = None,
    chord_prior_hz: Optional[List[float]] = None,
    bpm_hint: Optional[float] = None,
    tempo_override: Optional[float] = None,
    key_override: Optional[str] = None,
    config: Optional[StageZeroConfig] = None,
    debug: bool = False,
    truncate_duration: float = DEFAULT_PROCESSING_DURATION_SECONDS,
) -> StageZeroReport:
    """
    Main entry point: Load audio, condition, separate, return report.

    This is the complete Stage Zero pipeline that combines:
    - Ingestion (load, truncate, hash)
    - Pre-processing (phase align, noise reduction, HPSS, etc.)
    - Stem separation (Demucs → Mel-Roformer → STFT fallback)

    Args:
        audio_path: Path to input audio file
        output_path: Optional path to save conditioned audio
        chord_prior_hz: List of fundamental frequencies (Hz) for harmonic masking
        bpm_hint: Optional tempo hint (overrides auto-detection)
        tempo_override: Force a specific tempo (bypasses detection)
        key_override: Force a specific key (e.g., "C major")
        config: Optional StageZeroConfig
        debug: Enable debug output
        truncate_duration: Max duration to process (seconds)

    Returns:
        StageZeroReport with conditioned audio, separated stems, and diagnostics
    """
    print("\n" + "=" * 70)
    print("STAGE ZERO — Pre-Processing Before Stem Separation")
    print(f"File: {Path(audio_path).name}")
    print("=" * 70)

    start_time = time.time()

    # -----------------------------------------------------------------
    # Stage 1: Ingestion
    # -----------------------------------------------------------------
    print("\n[1/4] Loading audio...")
    loaded = load_audio(audio_path, target_sr=44100, truncate_duration=truncate_duration)
    print(f"      Duration: {loaded.original_duration:.1f}s @ {loaded.sample_rate}Hz")
    print(f"      Channels: {loaded.audio.shape[0]}, Truncated: {loaded.is_truncated}")

    # -----------------------------------------------------------------
    # Stages 2-9: Pre-processing (Phase Align, Noise, HPSS, etc.)
    # -----------------------------------------------------------------
    print("\n[2/4] Pre-processing (conditioning)...")
    cfg = config or StageZeroConfig(sample_rate=loaded.sample_rate)

    # Override tempo if provided
    effective_bpm = bpm_hint or tempo_override
    if tempo_override:
        print(f"      Tempo override: {tempo_override} BPM")

    processor = StageZeroProcessor(config=cfg)
    report = processor.process(
        stereo_audio=loaded.audio,
        chord_prior_hz=chord_prior_hz,
        bpm_hint=effective_bpm,
    )

    # -----------------------------------------------------------------
    # Stages 10-12: Separation
    # -----------------------------------------------------------------
    print("\n[3/4] Stem separation...")
    conditioned = report.conditioned_stereo
    sr = report.sample_rate

    separation_result = None

    # Try Demucs first
    if cfg.separation_model in ["demucs", "auto"]:
        separation_result = try_demucs_separation(conditioned, sr)
        if separation_result and any([
            separation_result.drums is not None,
            separation_result.bass is not None,
            separation_result.other is not None,
        ]):
            report.separation_model_used = "demucs"
            print("      ✓ Demucs separation successful")

    # Try Mel-Roformer if Demucs failed
    if (separation_result is None or not any([
        separation_result.drums is not None,
        separation_result.bass is not None,
    ])) and cfg.separation_model in ["mel_roformer", "auto"]:
        separation_result = try_mel_roformer_separation(conditioned, sr)
        if separation_result and any([
            separation_result.drums is not None,
            separation_result.bass is not None,
        ]):
            report.separation_model_used = "mel_roformer"
            print("      ✓ Mel-Roformer separation successful")

    # Fallback to STFT
    if (separation_result is None or not any([
        separation_result.drums is not None,
        separation_result.bass is not None,
    ])) and cfg.enable_fallback:
        separation_result = fallback_stft_separation(conditioned, sr)
        report.separation_model_used = "stft_fallback"
        report.warnings.append("Using STFT fallback — lower quality stems")
        print("      ⚠ STFT fallback used (lower quality)")

    if separation_result:
        report.drums_stem = separation_result.drums
        report.bass_stem = separation_result.bass
        report.other_stem = separation_result.other

    # -----------------------------------------------------------------
    # Stage 14: Export (optional)
    # -----------------------------------------------------------------
    if output_path:
        print(f"\n[4/4] Exporting conditioned audio to {output_path}...")
        try:
            import soundfile as sf
            # Ensure shape (samples, channels) for export
            audio_to_export = report.conditioned_stereo.T
            sf.write(output_path, audio_to_export, sr)
            print(f"      ✓ Exported")
        except Exception as e:
            print(f"      ✗ Export failed: {e}")
            report.warnings.append(f"Export failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"STAGE ZERO COMPLETE in {elapsed:.1f}s")
    print(report.get_summary())
    print(f"{'=' * 70}\n")

    return report


# =====================================================================
# Command-line Entry Point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage Zero — Pre-processing before stem separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Stage_Zero.py input.wav -o conditioned.wav
  python Stage_Zero.py input.wav --tempo 120 --key "C major"
  python Stage_Zero.py input.wav --chord 261.63 329.63 392.00
  python Stage_Zero.py input.wav --no-phase --no-hpss
        """
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("-o", "--output", help="Output path for conditioned audio (optional)")
    parser.add_argument("--tempo", type=float, help="Override tempo (BPM)")
    parser.add_argument("--key", help="Override key (e.g., 'C major')")
    parser.add_argument("--chord", nargs="+", type=float, help="Chord tone frequencies (Hz) for harmonic mask")
    parser.add_argument("--truncate", type=float, default=DEFAULT_PROCESSING_DURATION_SECONDS,
                        help=f"Truncate duration in seconds (default: {DEFAULT_PROCESSING_DURATION_SECONDS})")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-phase", action="store_true", help="Disable phase alignment")
    parser.add_argument("--no-noise", action="store_true", help="Disable noise reduction")
    parser.add_argument("--no-hpss", action="store_true", help="Disable tonal/transient split")
    parser.add_argument("--no-harmonic", action="store_true", help="Disable harmonic masking")
    parser.add_argument("--no-peel", action="store_true", help="Disable peel-back")
    parser.add_argument("--model", choices=["demucs", "mel_roformer", "auto", "stft"],
                        default="auto", help="Separation model to use")

    args = parser.parse_args()

    # Build config
    config = StageZeroConfig(
        phase_align_enabled=not args.no_phase,
        noise_nmf_enabled=not args.no_noise,
        hpss_enabled=not args.no_hpss,
        harmonic_mask_enabled=not args.no_harmonic,
        peel_enabled=not args.no_peel,
        separation_model=args.model,
    )

    # Convert key to chord frequencies if provided
    chord_hz = None
    if args.chord:
        chord_hz = args.chord
    elif args.key:
        # Simple key to frequency mapping (concert pitch A4=440Hz)
        key_to_hz = {
            'C': 261.63, 'C#': 277.18, 'Db': 277.18, 'D': 293.66,
            'D#': 311.13, 'Eb': 311.13, 'E': 329.63, 'F': 349.23,
            'F#': 369.99, 'Gb': 369.99, 'G': 392.00, 'G#': 415.30,
            'Ab': 415.30, 'A': 440.00, 'A#': 466.16, 'Bb': 466.16,
            'B': 493.88
        }
        # Parse "C major" -> 'C'
        root = args.key.split()[0] if args.key.split() else args.key
        if root in key_to_hz:
            chord_hz = [key_to_hz[root]]
            print(f"Using key {args.key} → chord root {chord_hz[0]:.2f}Hz")

    # Run
    report = run_stage_zero(
        audio_path=args.input,
        output_path=args.output,
        chord_prior_hz=chord_hz,
        bpm_hint=args.tempo,
        tempo_override=args.tempo,
        key_override=args.key,
        config=config,
        debug=args.debug,
        truncate_duration=args.truncate,
    )