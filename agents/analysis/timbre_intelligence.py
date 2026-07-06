# =====================================================================
# MODULE: agents/analysis/timbre_intelligence.py
# VERSION: 5.6.1
# DESCRIPTION:
#     Acoustic identity and timbre analysis engine for Grimlock 5.0.
#     Physics-informed, acoustically rigorous instrument identification.
#
# PHILOSOPHY:
#     Timbre is not a label. Timbre is the persistent acoustic behavior
#     of a resonant physical system. Every instrument is a nonlinear
#     system with its own characteristic way of converting mechanical
#     energy into air pressure waves. The physics cannot lie —
#     but reproductions try very hard.
#
# =====================================================================
#
# ACOUSTIC PRINCIPLES IN USE:
#
#   Harmonic Series (ideal, no stiffness):
#       f_n = n * f0
#
#   String Inharmonicity (Fletcher, 1964):
#       f_n = n * f0 * sqrt(1 + B * n^2)
#       where B = (pi^3 * E * d^4) / (64 * T * L^2)
#       E = Young's modulus, d = diameter, T = tension, L = length
#       Consequence: piano partials are ALWAYS sharp of ideal harmonics.
#       A MIDI pitch-shift breaks this law. The math catches it.
#
#   Bore Geometry (Fourier acoustics):
#       Cylindrical closed bore (clarinet) → odd harmonics only (1,3,5,7)
#       Conical bore (saxophone, oboe) → full series (1,2,3,4,5...)
#       Open bore (flute) → full series with weak fundamental
#       This is not a style choice. It is a boundary condition solution.
#
#   Tristimulus Values (Pollard & Jansson, 1982):
#       T1 = A1 / sum(An)          — energy in the fundamental
#       T2 = (A2+A3+A4) / sum(An) — energy in 2nd–4th harmonics
#       T3 = sum(A5..An) / sum(An) — energy in 5th harmonic and above
#       Perceptually meaningful groupings of harmonic energy.
#
#   Spectral Irregularity (Krimphoff et al., 1994):
#       IRR = sum|An - avg(An-1, An, An+1)| / sum(An)
#       Measures how jagged the harmonic amplitude envelope is.
#       Oboe, distorted guitar → high. Flute → low.
#
#   Bark Critical Bands (Zwicker, 1961):
#       24 frequency bands modeling the ear's critical bandwidth.
#       Bands are ~100 Hz wide at low frequencies, ~3500 Hz at high.
#       Captures perceptual timbral character beyond MFCC.
#
#   Vibrato Jitter (naturalness indicator):
#       Real performer vibrato: 5–15% rate irregularity (biological)
#       LFO-based vibrato: < 1% irregularity (metronomic)
#       This is the most reliable single-note live vs. synthetic test.
#
# =====================================================================

from __future__ import annotations

import uuid
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import librosa
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# =====================================================================
# PHYSICAL CONSTANTS
# =====================================================================

# Bark scale band edges in Hz (Zwicker, 1961)
# 24 critical bands that model human auditory frequency resolution.
BARK_EDGES_HZ = [
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
    1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
    4400, 5300, 6400, 7700, 9500, 12000, 15500
]
N_BARK_BANDS = 24

# Number of harmonics (partials) to track per note.
# 16 gives excellent coverage for most instruments up to 15-20 kHz.
N_PARTIALS = 16

# MFCC count — 13 is standard for instrument timbre.
N_MFCC = 13

# Vibrato rate range considered musical (Hz).
# Faster than 8 Hz is typically flutter or tremolo, not vibrato.
VIBRATO_RATE_MIN_HZ = 3.0
VIBRATO_RATE_MAX_HZ = 8.0


# =====================================================================
# ENUMS
# =====================================================================

class ExcitationType(str, Enum):
    """
    How energy enters the instrument.
    This is the root cause of characteristic noise signatures.
    """
    BOWED   = "bowed"    # Helmholtz slip-stick motion → rosin noise
    PLUCKED = "plucked"  # Impulsive onset → sharp transient
    STRUCK  = "struck"   # Hammer/mallet → transient + decay
    BREATH  = "breath"   # Turbulent airflow → breath noise
    REED    = "reed"     # Reed buzz → odd-harmonic noise coupling
    LIPS    = "lips"     # Brass cup mouthpiece → lip buzz
    VOCAL   = "vocal"    # Vocal folds + formant filtering
    SYNTH   = "synth"    # Electronic — may be any of the above simulated
    UNKNOWN = "unknown"


class InstrumentFamily(str, Enum):
    STRINGS    = "strings"     # Bowed strings (violin, viola, cello, bass)
    PLUCKED    = "plucked"     # Guitar, harp, banjo, pizzicato
    PIANO      = "piano"       # Piano, harpsichord (struck/plucked strings)
    BRASS      = "brass"       # Trumpet, trombone, French horn, tuba
    WOODWIND   = "woodwind"    # Flute, clarinet, saxophone, oboe, bassoon
    PERCUSSION = "percussion"  # Drums, mallet instruments
    VOICE      = "voice"       # Vocal
    SYNTH      = "synth"       # Synthesizer / electronic
    UNKNOWN    = "unknown"


class SynthesisOrigin(str, Enum):
    """
    Estimated origin of the sound source.
    This is a probabilistic inference, not a certainty.
    """
    LIVE_ACOUSTIC   = "live_acoustic"    # Confidence: live instrument
    SAMPLE_LIBRARY  = "sample_library"   # Likely a sampled library
    SYNTHESIS_PATCH = "synthesis_patch"  # Likely a synthesis patch
    AMBIGUOUS       = "ambiguous"        # Cannot determine


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class RichTimbreEmbedding:
    """
    Multi-dimensional acoustic identity vector.
    Each field has a physical or perceptual meaning.

    This replaces the older TimbreEmbedding with physics-informed
    features. The raw_vector concatenates all scalar features into
    a single comparable vector for entity tracking.
    """

    # --- Spectral Features (librosa standard) ---
    spectral_centroid: float      # Brightness indicator (Hz)
    spectral_rolloff: float       # Frequency below which 85% of energy lives
    spectral_flatness: float      # Tonality vs. noise (0=pure tone, 1=noise)
    spectral_bandwidth: float     # Spread of energy around centroid
    spectral_flux: float          # Frame-to-frame spectral change rate

    # --- Harmonic / Noise Separation ---
    harmonic_ratio: float         # Fraction of energy in harmonic component
    noise_ratio: float            # Fraction of energy in percussive/noise

    # --- Inharmonicity ---
    inharmonicity_simple: float   # Basic deviation from ideal harmonics
    inharmonicity_B: float        # Physics-based Fletcher B coefficient
                                  # B=0 → perfect harmonics (flute/sine)
                                  # B>0 → stiff string (piano, guitar)
                                  # B wrong for pitch → synthetic pitch-shift

    # --- Partial (Harmonic) Analysis ---
    partial_freqs: np.ndarray     # Actual frequency of each partial (Hz)
    partial_amps: np.ndarray      # Amplitude of each partial
    n_active_partials: int        # How many partials were detected

    # --- Tristimulus (Pollard & Jansson 1982) ---
    tristimulus_T1: float         # Energy fraction: fundamental
    tristimulus_T2: float         # Energy fraction: 2nd–4th harmonics
    tristimulus_T3: float         # Energy fraction: 5th+ harmonics
    # High T1 = flute-like (strong fundamental)
    # High T3 = bright/reedy (oboe, distorted guitar)

    # --- Bore / Geometry Indicator ---
    even_odd_ratio: float         # Even harmonic energy / total harmonic energy
    # Near 0 = clarinet-like (odd harmonics dominate, cylindrical closed bore)
    # Near 0.5 = flute/saxophone/most instruments (full harmonic series)

    # --- Spectral Irregularity (Krimphoff 1994) ---
    spectral_irregularity: float  # Jaggedness of the harmonic envelope
    # Low = smooth (flute, pure synth)
    # High = rough/bright (oboe, distorted)

    # --- Envelope ---
    attack_time_ms: float         # Time from onset to 90% of peak (ms)
    decay_rate: float             # Average amplitude slope after attack
    attack_entropy: float         # Shannon entropy of attack transient spectrum
    # Low entropy = simple/synthetic onset
    # High entropy = complex physical transient

    # --- Vibrato ---
    vibrato_rate_hz: float        # Dominant pitch modulation rate
    vibrato_depth: float          # Pitch modulation depth (semitone spread)
    vibrato_jitter: float         # Irregularity coefficient of vibrato rate
    # Near 0 = metronomic (LFO — synthetic flag)
    # 0.05–0.15 = natural performer range

    # --- Perceptual Filterbank ---
    mfcc_vector: np.ndarray       # 13 Mel-frequency cepstral coefficients
    bark_energies: np.ndarray     # 24 Bark critical band energy fractions

    # --- Composite Vector (for entity tracking) ---
    raw_vector: np.ndarray        # All scalars + MFCC + Bark concatenated


@dataclass
class MIDILikelihoodReport:
    """
    Probabilistic assessment of whether a sound is live or synthetic.

    This is a PROBABILISTIC inference, not ground truth.
    High-quality sample libraries approach the ambiguous zone.
    The most reliable indicators are vibrato jitter and, across
    multiple notes, inharmonicity consistency.
    """
    overall_score: float          # 0.0 = clearly live, 1.0 = clearly synthetic
    origin_estimate: SynthesisOrigin
    evidence: Dict[str, float]    # Per-indicator scores
    confidence: float             # How confident we are in the estimate


@dataclass
class AcousticEntity:
    """
    A persistent acoustic object tracked through time.
    Represents a single instrument or voice in a polyphonic mixture.
    """
    entity_id: str
    start_ms: float
    last_seen_ms: float

    embedding_history: List[RichTimbreEmbedding] = field(default_factory=list)

    probable_family: InstrumentFamily = InstrumentFamily.UNKNOWN
    probable_excitation: ExcitationType = ExcitationType.UNKNOWN

    confidence: float = 0.0
    continuity_score: float = 0.0

    midi_likelihood: Optional[MIDILikelihoodReport] = None

    active: bool = True
    metadata: Dict = field(default_factory=dict)

    def average_embedding_field(self, field_name: str) -> float:
        """Average a scalar field across all stored embeddings."""
        values = [
            getattr(e, field_name)
            for e in self.embedding_history
            if hasattr(e, field_name)
        ]
        return float(np.mean(values)) if values else 0.0


# =====================================================================
# PARTIAL ANALYZER
#
# This is the physics heart of the system.
# Real instrument tones are built from harmonic series with
# specific amplitude patterns that obey physical law.
# =====================================================================

class PartialAnalyzer:
    """
    Tracks and analyzes individual harmonics (partials) of a tone.

    A partial is the nth component of the harmonic series.
    Partial 1 = fundamental (f0)
    Partial 2 = second harmonic (2 * f0)
    Partial n = nth harmonic (n * f0)

    The relative amplitudes of these partials, their exact frequencies
    relative to the ideal harmonic series, and how they evolve over
    time are the primary determinants of timbre.
    """

    def __init__(self, sample_rate: int, n_fft: int = 4096):
        self.sample_rate = sample_rate
        self.n_fft = n_fft

    def find_fundamental(self, audio: np.ndarray) -> float:
        """
        Estimate f0 (fundamental frequency) using the YIN algorithm.

        YIN (de Cheveigné & Kawahara, 2002) is a time-domain
        autocorrelation method with cumulative mean normalized
        difference — robust for monophonic pitched sounds.

        Returns 0.0 if no clear pitch is found.
        """
        if len(audio) < 2048:
            return 0.0

        try:
            f0_series = librosa.yin(
                audio,
                fmin=librosa.note_to_hz('C2'),   # ~65 Hz — below most bass
                fmax=librosa.note_to_hz('C7'),   # ~2093 Hz — above most instruments
                sr=self.sample_rate
            )
            # YIN returns per-frame estimates. Take median of confident estimates.
            valid = f0_series[f0_series > 0]
            if len(valid) == 0:
                return 0.0
            return float(np.median(valid))

        except Exception:
            return 0.0

    def track_partials(
        self,
        audio: np.ndarray,
        f0: float,
        n_partials: int = N_PARTIALS,
        tolerance: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the actual frequency and amplitude of each partial.

        For each expected partial at n*f0, we search within a
        tolerance window (±15% of f0) for a spectral peak.
        This accommodates real instrument inharmonicity and
        slight pitch instability.

        Args:
            audio: Audio signal
            f0: Fundamental frequency in Hz
            n_partials: Number of partials to find
            tolerance: Search window as fraction of f0

        Returns:
            partial_freqs: Array of actual partial frequencies (Hz)
            partial_amps: Array of partial amplitudes (normalized)
        """
        if f0 <= 0:
            return np.zeros(n_partials), np.zeros(n_partials)

        # High-resolution FFT for accurate frequency measurement
        spectrum = np.abs(np.fft.rfft(audio, n=self.n_fft))
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

        partial_freqs = np.zeros(n_partials)
        partial_amps = np.zeros(n_partials)

        search_window_hz = f0 * tolerance

        for n in range(1, n_partials + 1):
            target_hz = n * f0

            # Stop if target is above Nyquist
            if target_hz > self.sample_rate / 2:
                break

            # Search window around expected partial
            low_hz = target_hz - search_window_hz
            high_hz = target_hz + search_window_hz

            low_bin = int(low_hz * self.n_fft / self.sample_rate)
            high_bin = int(high_hz * self.n_fft / self.sample_rate)

            low_bin = max(0, low_bin)
            high_bin = min(len(spectrum) - 1, high_bin)

            if low_bin >= high_bin:
                continue

            window_spectrum = spectrum[low_bin:high_bin]
            peak_local_idx = np.argmax(window_spectrum)
            peak_global_idx = peak_local_idx + low_bin

            partial_freqs[n - 1] = freqs[peak_global_idx]
            partial_amps[n - 1] = spectrum[peak_global_idx]

        # Normalize amplitudes to range [0, 1] relative to loudest partial
        max_amp = np.max(partial_amps)
        if max_amp > 0:
            partial_amps = partial_amps / max_amp

        return partial_freqs, partial_amps

    def estimate_B_coefficient(
        self,
        partial_freqs: np.ndarray,
        f0: float
    ) -> float:
        """
        Estimate the string inharmonicity B coefficient.

        Physical model (Fletcher, 1964):
            f_n = n * f0 * sqrt(1 + B * n^2)

        For small B (which is almost always true for instruments):
            f_n / (n * f0) ≈ 1 + 0.5 * B * n^2

        Rearranging:
            (f_n / (n * f0) - 1) ≈ 0.5 * B * n^2

        We estimate B via weighted median regression across all
        detected partials.

        Physical interpretation:
            B = 0:           Perfectly harmonic (flute, ideal string)
            B ~ 0.0001:      Bass piano strings (very long, thin)
            B ~ 0.001–0.005: Treble piano strings (shorter, stiffer)
            B ~ 0.01+:       Guitar/banjo high strings
            B wrong for note's pitch register → synthetic pitch-shift detected

        Returns:
            B coefficient. Always >= 0 for real physical strings.
        """
        if f0 <= 0:
            return 0.0

        valid_mask = partial_freqs > 0
        n_valid = np.sum(valid_mask)

        if n_valid < 3:
            return 0.0

        # Partial numbers (1-indexed)
        ns = np.where(valid_mask)[0] + 1
        f_observed = partial_freqs[valid_mask]
        f_ideal = ns * f0

        # Normalized deviation from ideal
        # This should equal 0.5 * B * n^2 for small B
        deviation = (f_observed / f_ideal) - 1.0

        n_sq = ns.astype(float) ** 2

        # Avoid division by zero, use weighted median for robustness
        # against measurement noise on individual partials
        B_per_partial = 2.0 * deviation / (n_sq + 1e-10)

        # Weight by partial number (higher partials show B more clearly)
        B_estimate = float(np.median(B_per_partial))

        # B must be non-negative (string physics)
        # Negative values indicate measurement noise or non-string source
        return float(np.clip(B_estimate, 0.0, 0.05))

    def compute_tristimulus(
        self,
        partial_amps: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute tristimulus values T1, T2, T3.

        Tristimulus (Pollard & Jansson, 1982) groups harmonic energy
        into three perceptually meaningful regions:

            T1: Fundamental alone
                High = fluty, hollow, fundamental-dominant
                Low  = bright, reedy, or percussive

            T2: 2nd, 3rd, and 4th harmonics
                The 'body' of the instrument's tone color

            T3: 5th harmonic and above
                High = bright, nasal, reedy (oboe, distorted guitar)
                Low  = warm, mellow (flute, recorder)

        The T1–T2–T3 space forms a 2D simplex (triangle) that
        maps instrument families to distinct regions.
        """
        total = np.sum(partial_amps) + 1e-10

        T1 = float(partial_amps[0] / total) if len(partial_amps) > 0 else 0.0

        if len(partial_amps) >= 4:
            T2 = float(np.sum(partial_amps[1:4]) / total)
        elif len(partial_amps) > 1:
            T2 = float(np.sum(partial_amps[1:]) / total)
        else:
            T2 = 0.0

        T3 = float(np.sum(partial_amps[4:]) / total) if len(partial_amps) > 4 else 0.0

        return T1, T2, T3

    def compute_even_odd_ratio(
        self,
        partial_amps: np.ndarray
    ) -> float:
        """
        Compute the ratio of even-harmonic energy to total harmonic energy.

        This is a bore geometry discriminator based on acoustic physics:

        CYLINDRICAL CLOSED BORE (clarinet):
            Closed end = pressure antinode, zero particle velocity.
            Only standing waves with odd multiples fit.
            Result: Partials 1, 3, 5, 7, 9... (even harmonics nearly absent)
            even_odd_ratio → near 0.0

        CONICAL BORE or OPEN BORE (saxophone, oboe, flute, all brass):
            Full harmonic series is supported.
            even_odd_ratio → near 0.4–0.5

        This is one of the most physically reliable instrument
        discriminators. A synth patch 'sounding like a clarinet'
        with a balanced even/odd ratio is physically implausible.

        Note: partial_amps index 0 = partial 1 (odd), index 1 = partial 2 (even)...
        """
        if len(partial_amps) < 2:
            return 0.5

        odd_amps  = partial_amps[0::2]   # partials 1, 3, 5, 7, ...
        even_amps = partial_amps[1::2]   # partials 2, 4, 6, 8, ...

        odd_energy  = float(np.sum(odd_amps ** 2))
        even_energy = float(np.sum(even_amps ** 2))
        total       = odd_energy + even_energy + 1e-10

        return float(even_energy / total)

    def compute_spectral_irregularity(
        self,
        partial_amps: np.ndarray
    ) -> float:
        """
        Compute Krimphoff spectral irregularity.

        Measures how jagged the harmonic amplitude envelope is.
        Each partial's amplitude is compared to the local average
        of its neighbors.

            IRR = sum(|A_n - avg(A_{n-1}, A_n, A_{n+1})|) / sum(A_n)

        Reference: Krimphoff, McAdams, Winsberg (1994)
        'Caractérisation du timbre des sons complexes'

        Low irregularity:  flute, recorder, pure synthesizer tones
        High irregularity: oboe, distorted guitar, voice (formant peaks)
        """
        amps = partial_amps[partial_amps > 0]
        if len(amps) < 3:
            return 0.0

        irregularity_sum = 0.0
        for i in range(1, len(amps) - 1):
            local_avg = (amps[i - 1] + amps[i] + amps[i + 1]) / 3.0
            irregularity_sum += abs(amps[i] - local_avg)

        return float(irregularity_sum / (np.sum(amps) + 1e-10))


# =====================================================================
# TRANSIENT ANALYZER
#
# The attack transient is the most information-dense moment in a note.
# Real instruments have physically chaotic, complex onsets.
# Synthesis often has simpler, repeatable onsets.
# =====================================================================

class TransientAnalyzer:
    """
    Analyzes the attack transient of a note.

    The onset of a note is where the excitation mechanism is most
    visible: the bow's first contact, the pick's impact, the reed's
    first vibration, the hammer's strike.

    Real transients are:
    - High in entropy (complex, non-periodic noise burst)
    - Different every time (stochastic physical process)
    - Phase-coherent with the resonator's response

    Synthesized transients are often:
    - Lower entropy (simple ramp or envelope)
    - Identical every time (sample playback)
    - Applied as a static amplitude envelope
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def extract(
        self,
        audio: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Extract attack time, decay rate, and attack entropy.

        Returns:
            attack_time_ms: Time from start to 90% of peak amplitude
            decay_rate: Average amplitude slope after the peak
            attack_entropy: Normalized Shannon entropy of attack spectrum
                            (0 = maximally simple, 1 = maximally complex)
        """
        envelope = np.abs(audio)
        peak = np.max(envelope)

        if peak <= 0 or len(audio) < 64:
            return 0.0, 0.0, 0.0

        # --- Attack time ---
        attack_threshold = peak * 0.90
        attack_idx = int(np.argmax(envelope >= attack_threshold))
        attack_ms = (attack_idx / self.sample_rate) * 1000.0

        # --- Decay rate ---
        # Mean slope of the amplitude envelope after the peak
        # Negative = decaying (normal), positive = re-energized (bowed sustain)
        if attack_idx < len(envelope) - 1:
            post_attack = envelope[attack_idx:]
            decay_rate = float(np.mean(np.diff(post_attack.astype(float))))
        else:
            decay_rate = 0.0

        # --- Attack entropy ---
        # We analyze the first 50ms (or the attack region, whichever is smaller)
        attack_window_samples = min(
            int(0.050 * self.sample_rate),
            attack_idx + 1,
            len(audio)
        )
        attack_audio = audio[:max(attack_window_samples, 64)]

        spectrum = np.abs(np.fft.rfft(attack_audio)) ** 2
        total = np.sum(spectrum)

        if total > 0:
            p = spectrum / total
            # Shannon entropy (nats), avoiding log(0)
            entropy = -np.sum(p * np.log(p + 1e-12))
            # Normalize: max possible entropy = log(N bins)
            max_entropy = np.log(len(p) + 1e-12)
            attack_entropy = float(entropy / max_entropy)
        else:
            attack_entropy = 0.0

        return float(attack_ms), float(decay_rate), float(attack_entropy)


# =====================================================================
# VIBRATO ANALYZER
#
# Vibrato is pitch modulation, typically 4–7 Hz in musical performance.
# The JITTER of vibrato rate is the key live-vs-synthetic discriminator.
# =====================================================================

class VibratoAnalyzer:
    """
    Analyzes vibrato characteristics including rate, depth, and jitter.

    JITTER is the critical metric for live vs. synthetic detection:

        Real performers (violin, cello, voice, wind):
            Vibrato is generated by physical oscillation of the
            diaphragm, bow arm, or embouchure. These are biological
            systems with inherent variability.
            Rate jitter: 5%–15% (coefficient of variation)
            Depth jitter: 10%–20%

        LFO-based synthesis / MIDI vibrato:
            Generated by a low-frequency oscillator. Metronomic.
            Rate jitter: < 1%
            Depth jitter: < 1%

    This is measurable from a single note of sufficient length.
    """

    HOP_LENGTH = 512  # librosa default hop for pitch tracking

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def analyze(
        self,
        audio: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Analyze vibrato from a pitched audio segment.

        Returns:
            vibrato_rate_hz: Dominant vibrato rate in Hz (0 if none)
            vibrato_depth: RMS pitch deviation in semitones (approx)
            vibrato_jitter: Coefficient of variation of inter-peak intervals
                            (0.0 = perfectly regular / synthetic,
                             0.10 = typical live performer)
        """
        if len(audio) < self.sample_rate // 2:
            # Need at least 0.5 seconds for meaningful vibrato analysis
            return 0.0, 0.0, 0.0

        try:
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.HOP_LENGTH
            )

            # Extract pitch track: for each frame, take the bin with max energy
            pitch_track = []
            for frame in range(pitches.shape[1]):
                idx = magnitudes[:, frame].argmax()
                pitch = pitches[idx, frame]
                if pitch > 0:
                    pitch_track.append(pitch)

            if len(pitch_track) < 20:
                return 0.0, 0.0, 0.0

            pitch_track = np.array(pitch_track)

            # Convert to semitones for perceptually meaningful depth
            # Semitone = 12 * log2(f / f_ref)
            # Deviation = 12 * log2(f / median(f))
            median_pitch = np.median(pitch_track[pitch_track > 0])
            if median_pitch <= 0:
                return 0.0, 0.0, 0.0

            # Pitch modulation signal (centered at 0)
            modulation = pitch_track - median_pitch

            # Vibrato depth: std of modulation (in Hz — rough semitone approx below)
            vibrato_depth_hz = float(np.std(modulation))
            # Convert to semitone approximation
            vibrato_depth_semitones = 12.0 * np.log2(
                (median_pitch + vibrato_depth_hz) / median_pitch + 1e-8
            )

            # Vibrato rate: dominant frequency of the modulation signal
            # Use autocorrelation to find the period
            autocorr = np.correlate(modulation, modulation, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            # Suppress lag-0 peak, search for next peak (= vibrato period)
            autocorr[0] = 0

            peaks, peak_props = find_peaks(
                autocorr,
                height=np.max(autocorr) * 0.3,
                distance=5
            )

            if len(peaks) == 0:
                return 0.0, float(vibrato_depth_semitones), 0.0

            # Period in frames → convert to Hz
            vibrato_period_frames = float(peaks[0])
            frames_per_second = self.sample_rate / self.HOP_LENGTH
            vibrato_rate_hz = frames_per_second / (vibrato_period_frames + 1e-8)

            # Sanity check: only accept musically plausible vibrato rates
            if not (VIBRATO_RATE_MIN_HZ <= vibrato_rate_hz <= VIBRATO_RATE_MAX_HZ):
                vibrato_rate_hz = 0.0

            # ---------------------------------------------------------
            # JITTER ANALYSIS
            # Find all peaks in the modulation signal and measure the
            # variation in spacing between consecutive peaks.
            # Regular spacing (jitter ≈ 0) → LFO / synthetic
            # Irregular spacing (jitter > 0.05) → live performer
            # ---------------------------------------------------------
            modulation_peaks, _ = find_peaks(modulation, distance=3)

            if len(modulation_peaks) >= 3:
                inter_peak_intervals = np.diff(modulation_peaks.astype(float))
                mean_interval = np.mean(inter_peak_intervals)

                if mean_interval > 0:
                    jitter = float(
                        np.std(inter_peak_intervals) / mean_interval
                    )
                else:
                    jitter = 0.0
            else:
                jitter = 0.0

            return (
                float(vibrato_rate_hz),
                float(vibrato_depth_semitones),
                float(jitter)
            )

        except Exception as e:
            logger.debug("Vibrato analysis failed: %s", e)
            return 0.0, 0.0, 0.0


# =====================================================================
# BARK BAND ANALYZER
#
# The Bark scale models human auditory frequency resolution.
# This gives a perceptual spectral fingerprint that complements MFCC.
# =====================================================================

class BarkAnalyzer:
    """
    Computes energy in each of the 24 Bark critical bands.

    The Bark scale (Zwicker, 1961) models the frequency resolution
    of the human auditory system. Critical bands are:
    - Narrow at low frequencies (~100 Hz wide at 0–1 kHz)
    - Wide at high frequencies (~3500 Hz wide at 10–15 kHz)

    This reflects the fact that the cochlea has lower frequency
    resolution at high frequencies.

    The 24 Bark bands provide an instrument-independent perceptual
    fingerprint. Two instruments with similar MFCC but different
    Bark patterns may be acoustically different in ways that matter
    to a listener.
    """

    def __init__(self, sample_rate: int, n_fft: int = 4096):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        # Pre-compute frequency bin → Bark band mapping
        self._freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """
        Returns normalized energy in each of the 24 Bark bands.

        Returns:
            np.ndarray of shape (24,), summing to 1.0
        """
        spectrum_power = np.abs(np.fft.rfft(audio, n=self.n_fft)) ** 2

        bark_energies = np.zeros(N_BARK_BANDS)

        for i in range(N_BARK_BANDS):
            low_hz = BARK_EDGES_HZ[i]
            high_hz = BARK_EDGES_HZ[i + 1]

            mask = (self._freqs >= low_hz) & (self._freqs < high_hz)
            bark_energies[i] = float(np.sum(spectrum_power[mask]))

        total = np.sum(bark_energies) + 1e-10
        return bark_energies / total


# =====================================================================
# MIDI / SYNTHETIC LIKELIHOOD ESTIMATOR
#
# The philosophical question made practical:
# Can we hear the difference between a real instrument and a reproduction?
#
# ANSWER: Yes, probabilistically, and here is the evidence framework.
# =====================================================================

class MIDILikelihoodEstimator:
    """
    Estimates the probability that a sound is from a MIDI font,
    sample library, or synthesis patch rather than a live instrument.

    ================================================================
    THE DEEP QUESTION:
    Can we distinguish real instruments from their reproductions?

    Yes. Here is why the physics supports this:

    1. VIBRATO JITTER (most reliable, single-note)
       Real vibrato is generated by biological oscillation (diaphragm,
       bow arm, embouchure). It has inherent rate irregularity of
       5–15%. LFO-generated vibrato is metronomic (< 1% jitter).
       This single test is often decisive.

    2. STRING INHARMONICITY CONSISTENCY (most reliable, multi-note)
       A piano string's B coefficient is physically determined by
       its stiffness, tension, and length. It cannot be different
       for the same string. When a MIDI font pitch-shifts a sample
       up or down, the B coefficient changes in a physically
       impossible way. Detectable across multiple notes of the same
       instrument, even from moderate-quality libraries.

    3. ATTACK TRANSIENT ENTROPY
       Physical onset events are chaotic (high spectral entropy).
       Synthesis envelopes are often simpler. High-quality samples
       have complex attacks, but they replay identically every time
       — detectable only across multiple notes (not implemented here).

    4. NOISE FLOOR SIGNATURE
       Real instruments have physically coupled excitation noise:
       - Flute: turbulent breath noise shaped by embouchure cavity
       - Violin: rosin stick-slip noise coupled to Helmholtz motion
       - Piano: hammer felt impact noise
       A too-clean noise floor is a synthetic flag.

    5. SPECTRAL IRREGULARITY SIGNATURE
       Real instruments have characteristic harmonic jaggedness.
       Wavetable synthesis can produce overly smooth or overly
       regular harmonic patterns not found in physical instruments.

    6. EVEN/ODD HARMONIC CONSISTENCY
       A clarinet that doesn't suppress even harmonics is physically
       impossible (cylindrical closed bore law). Synthesis patches
       that try to "sound like a clarinet" without modeling bore
       geometry will fail this test.

    THE HONEST CEILING:
    High-quality multisampled libraries (Vienna Symphonic Library,
    Spitfire, East West) with round-robin sampling, articulation
    layers, and microtuning approach the ambiguous zone.
    This system will rate them as "ambiguous" or low-likelihood.
    That is the correct and honest answer.
    ================================================================
    """

    def estimate(
        self,
        embedding: RichTimbreEmbedding
    ) -> MIDILikelihoodReport:
        """
        Compute a probabilistic MIDI/synthetic likelihood score.

        Returns a MIDILikelihoodReport with overall score,
        per-indicator evidence, and a SynthesisOrigin classification.
        """
        evidence: Dict[str, float] = {}

        # ---------------------------------------------------------
        # Indicator 1: Vibrato Jitter
        # The single most reliable single-note live/synthetic test.
        # ---------------------------------------------------------
        if embedding.vibrato_depth > 0.1:
            # Only meaningful if vibrato is actually present
            if embedding.vibrato_jitter < 0.02:
                evidence['vibrato_too_regular'] = 0.85   # LFO-like: strong synthetic flag
            elif embedding.vibrato_jitter < 0.05:
                evidence['vibrato_too_regular'] = 0.45   # Possibly synthetic
            else:
                evidence['vibrato_too_regular'] = 0.05   # Natural jitter: live indicator
        else:
            # No vibrato — can't use this indicator
            evidence['vibrato_too_regular'] = 0.25   # Neutral

        # ---------------------------------------------------------
        # Indicator 2: Attack Transient Entropy
        # Low entropy onset = simple synthesis envelope
        # High entropy onset = complex physical transient
        # ---------------------------------------------------------
        if embedding.attack_entropy < 0.40:
            evidence['low_attack_entropy'] = 0.70
        elif embedding.attack_entropy < 0.60:
            evidence['low_attack_entropy'] = 0.35
        else:
            evidence['low_attack_entropy'] = 0.05

        # ---------------------------------------------------------
        # Indicator 3: B Coefficient for Non-Wind Instruments
        # Near-zero B + spectrally harmonic = either:
        #   (a) flute/voice (valid physical case), or
        #   (b) pure synthesis (no stiffness model)
        # This is ambiguous on its own; B is most powerful multi-note.
        # ---------------------------------------------------------
        if embedding.inharmonicity_B < 0.00005 and embedding.harmonic_ratio > 0.85:
            # Perfect harmonicity with high harmonic ratio:
            # Unlikely for string instruments → mild synthetic flag
            evidence['suspicious_harmonicity'] = 0.35
        else:
            evidence['suspicious_harmonicity'] = 0.10

        # ---------------------------------------------------------
        # Indicator 4: Noise Floor Character
        # Physically-coupled excitation noise is characteristic.
        # An unnaturally clean sound is a mild synthetic indicator.
        # ---------------------------------------------------------
        if embedding.noise_ratio < 0.002:
            evidence['unnaturally_clean'] = 0.55
        elif embedding.noise_ratio < 0.01:
            evidence['unnaturally_clean'] = 0.25
        else:
            evidence['unnaturally_clean'] = 0.05

        # ---------------------------------------------------------
        # Indicator 5: Spectral Irregularity
        # Very low irregularity = overly smooth harmonic envelope
        # (possible wavetable or additive synthesis artifact)
        # ---------------------------------------------------------
        if embedding.spectral_irregularity < 0.005:
            evidence['too_smooth_spectrum'] = 0.45
        elif embedding.spectral_irregularity < 0.02:
            evidence['too_smooth_spectrum'] = 0.20
        else:
            evidence['too_smooth_spectrum'] = 0.05

        # ---------------------------------------------------------
        # Indicator 6: Temporal Flux
        # Real instruments have consistent spectral evolution.
        # Very low flux in sustain = looped sample artifact.
        # ---------------------------------------------------------
        if embedding.spectral_flux < 0.01:
            evidence['static_spectrum'] = 0.40
        elif embedding.spectral_flux < 0.10:
            evidence['static_spectrum'] = 0.15
        else:
            evidence['static_spectrum'] = 0.05

        # ---------------------------------------------------------
        # Weighted combination
        # Vibrato jitter gets the highest weight (most reliable).
        # ---------------------------------------------------------
        weights = {
            'vibrato_too_regular':   0.30,
            'low_attack_entropy':    0.20,
            'suspicious_harmonicity': 0.10,
            'unnaturally_clean':     0.15,
            'too_smooth_spectrum':   0.12,
            'static_spectrum':       0.13,
        }

        overall_score = float(np.clip(
            sum(weights[k] * v for k, v in evidence.items()),
            0.0, 1.0
        ))

        # ---------------------------------------------------------
        # Classification
        # ---------------------------------------------------------
        if overall_score < 0.25:
            origin = SynthesisOrigin.LIVE_ACOUSTIC
            confidence = 1.0 - overall_score
        elif overall_score < 0.45:
            origin = SynthesisOrigin.AMBIGUOUS
            confidence = 0.5
        elif overall_score < 0.70:
            origin = SynthesisOrigin.SAMPLE_LIBRARY
            confidence = overall_score
        else:
            origin = SynthesisOrigin.SYNTHESIS_PATCH
            confidence = overall_score

        return MIDILikelihoodReport(
            overall_score=overall_score,
            origin_estimate=origin,
            evidence=evidence,
            confidence=float(confidence)
        )


# =====================================================================
# TIMBRE FEATURE EXTRACTOR
#
# Assembles all sub-analyzers into a single RichTimbreEmbedding.
# =====================================================================

class TimbreFeatureExtractor:
    """
    Main feature extraction pipeline.

    Coordinates all sub-analyzers and assembles a RichTimbreEmbedding
    from a single audio segment.

    For best results:
    - Audio should be a monophonic stem (use Demucs separation first)
    - Minimum length: ~0.5 seconds for vibrato analysis
    - Sample rate: any (converted internally as needed)
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.partial_analyzer = PartialAnalyzer(sample_rate)
        self.transient_analyzer = TransientAnalyzer(sample_rate)
        self.vibrato_analyzer = VibratoAnalyzer(sample_rate)
        self.bark_analyzer = BarkAnalyzer(sample_rate)

    async def extract(self, audio: np.ndarray) -> RichTimbreEmbedding:
        """
        Extract a RichTimbreEmbedding from an audio segment.

        This is the main entry point. Run after Demucs stem separation
        for best single-instrument results.

        Args:
            audio: Mono audio array, float32, normalized to [-1, 1]

        Returns:
            RichTimbreEmbedding with all timbral features
        """
        if len(audio) == 0:
            raise ValueError("Cannot extract features from empty audio.")

        # ---- Standard Spectral Features ----
        centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        ))
        rolloff = float(np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        ))
        flatness = float(np.mean(
            librosa.feature.spectral_flatness(y=audio)
        ))
        bandwidth = float(np.mean(
            librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        ))

        # ---- Spectral Flux ----
        stft = np.abs(librosa.stft(audio))
        if stft.shape[1] > 1:
            flux = float(np.mean(np.sum(np.diff(stft, axis=1) ** 2, axis=0)))
        else:
            flux = 0.0

        # ---- MFCC ----
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=N_MFCC
        )
        mfcc_vector = np.mean(mfcc, axis=1)

        # ---- Harmonic / Percussive Separation ----
        harmonic, percussive = librosa.effects.hpss(audio)
        h_energy = float(np.sum(harmonic ** 2))
        p_energy = float(np.sum(percussive ** 2))
        total_energy = h_energy + p_energy + 1e-10

        harmonic_ratio = h_energy / total_energy
        noise_ratio = p_energy / total_energy

        # ---- Bark Energies ----
        bark_energies = self.bark_analyzer.compute(audio)

        # ---- Fundamental Frequency ----
        f0 = self.partial_analyzer.find_fundamental(audio)

        # ---- Partial Tracking ----
        if f0 > 0:
            partial_freqs, partial_amps = self.partial_analyzer.track_partials(
                audio, f0, n_partials=N_PARTIALS
            )
            n_active = int(np.sum(partial_amps > 0.01))
        else:
            partial_freqs = np.zeros(N_PARTIALS)
            partial_amps = np.zeros(N_PARTIALS)
            n_active = 0

        # ---- Inharmonicity ----
        inharmonicity_B = self.partial_analyzer.estimate_B_coefficient(
            partial_freqs, f0
        )
        # Simple inharmonicity fallback (for compatibility and no-f0 cases)
        inharmonicity_simple = self._simple_inharmonicity(audio)

        # ---- Tristimulus ----
        T1, T2, T3 = self.partial_analyzer.compute_tristimulus(partial_amps)

        # ---- Even/Odd Ratio ----
        even_odd_ratio = self.partial_analyzer.compute_even_odd_ratio(partial_amps)

        # ---- Spectral Irregularity ----
        spectral_irregularity = self.partial_analyzer.compute_spectral_irregularity(
            partial_amps
        )

        # ---- Transient Analysis ----
        attack_ms, decay_rate, attack_entropy = self.transient_analyzer.extract(audio)

        # ---- Vibrato Analysis ----
        vibrato_rate, vibrato_depth, vibrato_jitter = self.vibrato_analyzer.analyze(audio)

        # ---- Assemble Raw Vector ----
        # All scalar features concatenated for cosine similarity tracking.
        # Order matters — keep consistent across all calls.
        scalar_features = np.array([
            centroid,
            rolloff,
            flatness,
            bandwidth,
            flux,
            harmonic_ratio,
            noise_ratio,
            inharmonicity_simple,
            inharmonicity_B * 1000.0,  # Scale up — B is tiny
            T1,
            T2,
            T3,
            even_odd_ratio,
            spectral_irregularity,
            attack_ms / 1000.0,        # Normalize to seconds
            decay_rate,
            attack_entropy,
            vibrato_rate,
            vibrato_depth,
            vibrato_jitter,
        ], dtype=np.float32)

        # Normalize partial amps to fixed N_PARTIALS length for the vector
        raw_vector = np.concatenate([
            scalar_features,
            mfcc_vector.astype(np.float32),
            bark_energies.astype(np.float32),
            partial_amps[:8].astype(np.float32)   # First 8 partials only
        ])

        return RichTimbreEmbedding(
            spectral_centroid=centroid,
            spectral_rolloff=rolloff,
            spectral_flatness=flatness,
            spectral_bandwidth=bandwidth,
            spectral_flux=flux,
            harmonic_ratio=harmonic_ratio,
            noise_ratio=noise_ratio,
            inharmonicity_simple=inharmonicity_simple,
            inharmonicity_B=inharmonicity_B,
            partial_freqs=partial_freqs,
            partial_amps=partial_amps,
            n_active_partials=n_active,
            tristimulus_T1=T1,
            tristimulus_T2=T2,
            tristimulus_T3=T3,
            even_odd_ratio=even_odd_ratio,
            spectral_irregularity=spectral_irregularity,
            attack_time_ms=attack_ms,
            decay_rate=decay_rate,
            attack_entropy=attack_entropy,
            vibrato_rate_hz=vibrato_rate,
            vibrato_depth=vibrato_depth,
            vibrato_jitter=vibrato_jitter,
            mfcc_vector=mfcc_vector,
            bark_energies=bark_energies,
            raw_vector=raw_vector
        )

    def _simple_inharmonicity(self, audio: np.ndarray) -> float:
        """
        Fallback inharmonicity estimate (no f0 required).
        Measures how much spectral peaks deviate from a harmonic series
        anchored at the lowest detected peak.
        """
        spectrum = np.abs(np.fft.rfft(audio))
        peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)

        if len(peaks) < 2:
            return 0.0

        peak_freqs = peaks * self.sample_rate / len(audio)
        fundamental = peak_freqs[0]
        if fundamental <= 0:
            return 0.0

        expected = np.arange(1, len(peak_freqs) + 1) * fundamental
        deviation = np.mean(np.abs(peak_freqs[:len(expected)] - expected))

        return float(deviation)


# =====================================================================
# ACOUSTIC ENTITY TRACKER
# =====================================================================

class AcousticEntityTracker:
    """
    Tracks persistent acoustic entities through time using
    cosine similarity of timbral embedding vectors.

    An acoustic entity represents a single sound source (instrument
    or voice) that persists across multiple time frames.

    Continuity is determined by whether successive timbral embeddings
    are similar enough to belong to the same source.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_history: int = 32
    ):
        self.entities: List[AcousticEntity] = []
        self.similarity_threshold = similarity_threshold
        self.max_history = max_history  # Limit memory per entity

    def update(
        self,
        embedding: RichTimbreEmbedding,
        timestamp_ms: float
    ) -> AcousticEntity:
        """
        Update tracker with a new embedding.

        If the embedding matches an existing entity (cosine similarity
        above threshold), it is assigned to that entity and updates it.
        Otherwise, a new entity is created.
        """
        best_entity = None
        best_similarity = -1.0

        for entity in self.entities:
            if not entity.active:
                continue
            if not entity.embedding_history:
                continue

            # Compare against most recent embedding for this entity
            previous = entity.embedding_history[-1]
            similarity = self._cosine_similarity(
                previous.raw_vector,
                embedding.raw_vector
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_entity = entity

        # ---- Continuity Match ----
        if best_entity is not None and best_similarity >= self.similarity_threshold:
            best_entity.embedding_history.append(embedding)

            # Trim history to max_history to avoid unbounded memory growth
            if len(best_entity.embedding_history) > self.max_history:
                best_entity.embedding_history = (
                    best_entity.embedding_history[-self.max_history:]
                )

            best_entity.last_seen_ms = timestamp_ms
            best_entity.continuity_score = best_similarity

            # Confidence grows with continuity (up to 0.95)
            best_entity.confidence = min(0.95, best_entity.confidence + 0.03)

            return best_entity

        # ---- New Entity ----
        entity = AcousticEntity(
            entity_id=str(uuid.uuid4()),
            start_ms=timestamp_ms,
            last_seen_ms=timestamp_ms,
            embedding_history=[embedding],
            confidence=0.40
        )
        self.entities.append(entity)
        return entity

    def mark_inactive(self, entity_id: str) -> None:
        """Mark an entity as no longer active (note has ended)."""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                entity.active = False

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """
        Cosine similarity between two timbral vectors.
        Returns value in [-1, 1]. Higher = more similar timbre.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))


# =====================================================================
# INSTRUMENT FAMILY CLASSIFIER
#
# Physics-informed rule-based classification.
# Designed to be replaced by or supplemented with ML later,
# but these rules encode real acoustic physics, not guesses.
# =====================================================================

class InstrumentFamilyClassifier:
    """
    Classifies instrument family from timbral embedding using
    physics-informed heuristic rules.

    Each rule is anchored in acoustic physics:

    PIANO:
        - Struck string: fast attack (< 20ms), then pure decay
        - Full harmonic series: even_odd_ratio ~ 0.4-0.5
        - Non-zero inharmonicity_B (stiff bass/treble strings)
        - High initial harmonic ratio, fading to percussive

    BOWED STRINGS (violin, viola, cello):
        - Bow excitation: slow attack (40–200ms), sustained
        - Vibrato: present (4–7 Hz), with natural jitter
        - Near-perfect harmonics (Helmholtz motion): B near 0
        - Even and odd harmonics present

    CLARINET:
        - Reed: moderate noise in attack
        - Odd-harmonic dominance: even_odd_ratio < 0.20
        - Cylindrical closed bore physics (the key discriminator)
        - Mid-range spectral centroid

    FLUTE:
        - Breath noise: high noise_ratio in attack
        - Weak fundamental: low T1
        - Very smooth spectral irregularity
        - Near-zero inharmonicity_B (open cylindrical bore)

    BRASS (trumpet, trombone, French horn):
        - Bright: high spectral_centroid
        - High harmonic ratio in sustain
        - Fast attack, bright T3
        - Even and odd harmonics present (conical or flared bore)

    SAXOPHONE:
        - Conical bore: even harmonics present (unlike clarinet)
        - Reed excitation: similar noise to clarinet
        - Warmer than clarinet: lower spectral_centroid

    VOICE:
        - Formant structure visible in Bark bands
        - Vibrato present with natural jitter
        - High spectral_flatness variability
    """

    def classify(
        self,
        embedding: RichTimbreEmbedding
    ) -> Tuple[InstrumentFamily, ExcitationType, float]:
        """
        Classify instrument family and excitation type.

        Returns:
            family: InstrumentFamily enum
            excitation: ExcitationType enum
            confidence: Classification confidence [0, 1]
        """

        scores: Dict[InstrumentFamily, float] = {
            f: 0.0 for f in InstrumentFamily
        }

        # ---- PIANO ----
        piano_score = 0.0
        if embedding.attack_time_ms < 20.0:
            piano_score += 0.35
        if embedding.harmonic_ratio > 0.75:
            piano_score += 0.20
        if embedding.inharmonicity_B > 0.0001:
            piano_score += 0.25   # String stiffness = likely piano/guitar
        if 0.30 < embedding.even_odd_ratio < 0.55:
            piano_score += 0.10  # Full harmonic series
        if embedding.vibrato_depth < 0.5:
            piano_score += 0.10  # Piano does not vibrato
        scores[InstrumentFamily.PIANO] = piano_score

        # ---- BOWED STRINGS ----
        strings_score = 0.0
        if embedding.attack_time_ms > 40.0:
            strings_score += 0.25   # Bowed attack is slow
        if embedding.vibrato_depth > 0.3:
            strings_score += 0.20
        if VIBRATO_RATE_MIN_HZ < embedding.vibrato_rate_hz < VIBRATO_RATE_MAX_HZ:
            strings_score += 0.15
        if embedding.vibrato_jitter > 0.05:
            strings_score += 0.10   # Natural vibrato jitter
        if embedding.harmonic_ratio > 0.70:
            strings_score += 0.15
        if embedding.inharmonicity_B < 0.0005:
            strings_score += 0.15   # Near-perfect harmonics (Helmholtz motion)
        scores[InstrumentFamily.STRINGS] = strings_score

        # ---- CLARINET (cylindrical closed bore) ----
        clarinet_score = 0.0
        if embedding.even_odd_ratio < 0.18:
            clarinet_score += 0.55   # The key physical discriminator
        if 0.10 < embedding.noise_ratio < 0.40:
            clarinet_score += 0.15   # Reed noise
        if embedding.spectral_centroid < 3000.0:
            clarinet_score += 0.10
        if embedding.tristimulus_T2 > 0.30:
            clarinet_score += 0.10   # Odd-harmonic body energy
        if embedding.attack_time_ms < 50.0:
            clarinet_score += 0.10
        scores[InstrumentFamily.WOODWIND] = clarinet_score  # (shared with flute etc.)

        # ---- FLUTE ----
        flute_score = 0.0
        if embedding.noise_ratio > 0.30:
            flute_score += 0.30   # Prominent breath noise
        if embedding.tristimulus_T1 < 0.20:
            flute_score += 0.20   # Weak fundamental
        if embedding.spectral_irregularity < 0.05:
            flute_score += 0.15   # Very smooth harmonic envelope
        if embedding.even_odd_ratio > 0.35:
            flute_score += 0.15   # Open bore: full harmonic series
        if embedding.inharmonicity_B < 0.0001:
            flute_score += 0.20   # Near-perfectly harmonic (open cylindrical)
        # Flute vs clarinet: if flute_score > clarinet_score, use flute.
        # For now, both go under WOODWIND. Multi-class ML will refine this.
        if flute_score > scores[InstrumentFamily.WOODWIND]:
            scores[InstrumentFamily.WOODWIND] = flute_score

        # ---- BRASS ----
        brass_score = 0.0
        if embedding.spectral_centroid > 1800.0:
            brass_score += 0.25
        if embedding.harmonic_ratio > 0.80:
            brass_score += 0.20
        if embedding.tristimulus_T3 > 0.25:
            brass_score += 0.25   # Bright upper partials (conical bore)
        if embedding.attack_time_ms < 30.0:
            brass_score += 0.15
        if embedding.even_odd_ratio > 0.35:
            brass_score += 0.15   # Full harmonic series (conical/flared bore)
        scores[InstrumentFamily.BRASS] = brass_score

        # ---- VOICE ----
        voice_score = 0.0
        if embedding.vibrato_depth > 0.3 and embedding.vibrato_jitter > 0.08:
            voice_score += 0.30
        if embedding.spectral_flatness > 0.05:
            voice_score += 0.20   # Formant-shaped spectrum
        if embedding.noise_ratio > 0.15:
            voice_score += 0.15   # Breath noise
        if embedding.attack_time_ms > 60.0:
            voice_score += 0.15   # Slow, controlled onset
        scores[InstrumentFamily.VOICE] = voice_score

        # ---- PERCUSSION ----
        perc_score = 0.0
        if embedding.attack_time_ms < 10.0:
            perc_score += 0.30
        if embedding.noise_ratio > 0.50:
            perc_score += 0.25
        if embedding.harmonic_ratio < 0.40:
            perc_score += 0.25
        if embedding.inharmonicity_B < 0.0001 and embedding.harmonic_ratio < 0.5:
            perc_score += 0.20   # Noise-dominated, no clear harmonic structure
        scores[InstrumentFamily.PERCUSSION] = perc_score

        # ---- SELECT WINNER ----
        best_family = max(scores, key=lambda f: scores[f])
        best_score = scores[best_family]

        # If best score is too low, return UNKNOWN
        if best_score < 0.30:
            best_family = InstrumentFamily.UNKNOWN

        # ---- EXCITATION TYPE ----
        excitation = self._infer_excitation(embedding, best_family)

        return best_family, excitation, float(min(best_score, 0.90))

    def _infer_excitation(
        self,
        embedding: RichTimbreEmbedding,
        family: InstrumentFamily
    ) -> ExcitationType:
        """
        Infer the excitation mechanism from family and features.

        This is a secondary classifier using the family classification
        as context. The excitation type determines characteristic
        noise signatures.
        """
        if family == InstrumentFamily.STRINGS:
            if embedding.attack_time_ms > 40:
                return ExcitationType.BOWED
            else:
                return ExcitationType.PLUCKED

        if family == InstrumentFamily.PIANO:
            return ExcitationType.STRUCK

        if family == InstrumentFamily.BRASS:
            return ExcitationType.LIPS

        if family == InstrumentFamily.WOODWIND:
            if embedding.noise_ratio > 0.30:
                return ExcitationType.BREATH   # Flute-like
            else:
                return ExcitationType.REED     # Clarinet/saxophone/oboe-like

        if family == InstrumentFamily.VOICE:
            return ExcitationType.VOCAL

        if family == InstrumentFamily.PERCUSSION:
            return ExcitationType.STRUCK

        return ExcitationType.UNKNOWN


# =====================================================================
# MAIN ENGINE
# =====================================================================

class TimbreIntelligence:
    """
    Main timbre analysis engine for Grimlock 5.0.

    Orchestrates all sub-systems:
        1. TimbreFeatureExtractor → produces RichTimbreEmbedding
        2. AcousticEntityTracker → maintains persistent acoustic identity
        3. InstrumentFamilyClassifier → labels instrument family
        4. MIDILikelihoodEstimator → assesses live vs. synthetic origin

    Usage:
        engine = TimbreIntelligence(sample_rate=44100)
        entity = await engine.process(audio_segment, timestamp_ms=1500.0)

        print(entity.probable_family)       # e.g. InstrumentFamily.STRINGS
        print(entity.probable_excitation)   # e.g. ExcitationType.BOWED
        print(entity.midi_likelihood.origin_estimate)
        print(entity.midi_likelihood.evidence)
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

        self.extractor = TimbreFeatureExtractor(sample_rate)
        self.tracker = AcousticEntityTracker()
        self.classifier = InstrumentFamilyClassifier()
        self.midi_estimator = MIDILikelihoodEstimator()

    async def process(
        self,
        audio: np.ndarray,
        timestamp_ms: float
    ) -> AcousticEntity:
        """
        Process a single audio segment and return the acoustic entity.

        For best results, pass a single stem from Demucs separation.
        The entity is updated if it matches a previously-seen entity,
        or created fresh if it's new.

        Args:
            audio: Mono audio, float32, normalized to [-1, 1]
            timestamp_ms: Position in the full track (milliseconds)

        Returns:
            AcousticEntity with family, excitation, and MIDI likelihood
        """
        if len(audio) == 0:
            raise ValueError("process() received empty audio segment.")

        # Step 1: Extract rich timbral features
        embedding = await self.extractor.extract(audio)

        # Step 2: Track entity continuity through time
        entity = self.tracker.update(embedding, timestamp_ms)

        # Step 3: Classify instrument family and excitation
        family, excitation, confidence = self.classifier.classify(embedding)

        entity.probable_family = family
        entity.probable_excitation = excitation
        entity.confidence = max(entity.confidence, confidence)

        # Step 4: Assess MIDI/synthetic likelihood
        entity.midi_likelihood = self.midi_estimator.estimate(embedding)

        return entity

    def get_all_entities(self) -> List[AcousticEntity]:
        """Return all tracked acoustic entities."""
        return self.tracker.entities

    def get_active_entities(self) -> List[AcousticEntity]:
        """Return only currently active acoustic entities."""
        return [e for e in self.tracker.entities if e.active]