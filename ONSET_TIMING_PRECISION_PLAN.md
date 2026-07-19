# Onset & Timing Precision Plan — Trusted Rhythm from Untrusted Witnesses

**Status: PROPOSED, amended after two rounds of review.** This document
scopes out separable initiatives aimed at one goal: more trustworthy note-
**onset** placement (exact start-time precision) and low-frequency track
stability, as distinct from the tempo/beat-grid/quantization machinery this
project already has. It checks each initiative for real, concrete tie-in
opportunities to `AnechoicMa`, `Ritornello`, `PulseField`,
`HarmonicIntelligence`, and `SchoenbergMirror` — per explicit request, these
are call-outs of *possibility*, not mandatory scope. Nothing here should be
built in one pass; see §5 for phasing.

This document exists because a conversation about standalone CREPE/SPICE/
Basic Pitch/librosa quality comparisons surfaced a real question: none of
this project's pitch-detection models were ever designed to place onsets
precisely, and the codebase currently does very little to compensate for
that at the *individual-note* level (as opposed to the *tempo/beat-grid*
level, which is well covered). All code-location and behavior claims below
were verified against the current source, not assumed. A separate, unrelated
proposal (per-frame chord/interval structural classification for
`chord_map`) was reviewed alongside this plan and deliberately **not**
folded in here — it's a different problem (harmonic labeling, not onset
timing) and has its own real issues (see the session notes if revisiting);
it should get its own document if pursued.

---

## 1. The Problem, Stated Plainly

Every pitch-detection witness in this project has a documented, structural
weakness at onset placement and, for Basic Pitch specifically, at low-end
frequency tracking:

- **Basic Pitch**: has a real onset posteriorgram, but its own note-creation
  step (inside `basic_pitch.inference.predict()`, upstream of this codebase)
  is a greedy, frame-local threshold-and-walk heuristic — it decides "is this
  a boundary?" one frame at a time, with no global consistency check. Its
  frame hop (~11.6ms, `FFT_HOP=256` @ 22050Hz) is also a hard resolution
  ceiling no amount of downstream cleverness can beat *without* looking at
  the raw waveform. Separately, at low fundamental frequencies it's prone to
  octave confusion (locking onto a strong 2nd/3rd harmonic instead of a weak
  true fundamental) and intermittent note dropping — a real, already
  partially-addressed problem, see §1a below.
- **SPICE / CREPE / librosa**: none of the three have an onset concept in
  their training/design at all (confirmed: SPICE is self-supervised on
  pitch-invariance proxy tasks, CREPE and librosa's pyin/piptrack are pure
  pitch-curve trackers). Note segmentation from any of them is a heuristic
  *this project* bolts on after the fact (grouping consecutive same-pitch
  frames) — not a capability those models have.

None of that is news by itself — it's *why* this project's rhythm
intelligence (`RhythmEngine`, `TemporalLattice`, `GrooveField`, `PulseField`,
`ReverseGeoCrypt`) exists as a separate subsystem rather than trusting any
pitch model's own timestamps for tempo/beat/meter. That separation is
correct and should stay. But it means individual **note onset** precision —
not tempo, not beat position, the actual moment a note begins — still rests
entirely on Basic Pitch's own greedy frame-local decision, with only
after-the-fact repair (`Ritornello`) available to fix it once it's already
wrong.

### 1a. Important: the low-end octave problem is *not* untouched today

Task #107 (already in production, `epistemic_council.py`'s
`analyze_bass_sync`) already runs a direct-pitch pass *and* an octave-shift-
up/analyze/correct-down pass on bass audio, treating agreement as strong
evidence and disagreement as a real, visible flag. This is the same
"shift audio into a register the model handles better" trick proposed
independently in outside research reviewed alongside this plan — useful
confirmation it's a recognized, sound technique, not something this project
invented in isolation. **Any new low-end mechanism in §2.3 must say
explicitly how it relates to this existing pass** — replace it, run
alongside it, or (the option recommended below) become the actual
resolution mechanism for the disagreement case #107 already detects and
flags today but doesn't independently resolve. Proposing a new mechanism
as if this gap were unaddressed would repeat the exact "two independent
implementations of the same question, never cross-checked" anti-pattern
`ARCHITECTURE_CONSOLIDATION_PLAN.md` was written to stop.

Four concrete, confirmed gaps drive this plan:

1. **A real, currently-live correctness bug** in `EpistemicCouncil.analyze_sync`'s
   consensus grouping (`epistemic/epistemic_council.py` ~line 1002-1005):
   a consensus note's final `start_ms`/`end_ms` is a *confidence*-weighted
   average across every witness in its agreement group, including SPICE and
   librosa. Their confidence reflects pitch-certainty, not onset-placement
   precision — so a well-timed Basic Pitch onset grouped with a badly-timed
   SPICE vote can get its timing measurably dragged toward SPICE's worse
   estimate. CREPE is *already* correctly exempted from this (its refinement
   pass, `_run_crepe_refinement`, only ever touches `pitch`/`confidence`,
   confirmed by direct read — never `start_ms`/`end_ms`). SPICE and librosa
   get no such exemption today.
2. **An already-built, already-validated onset-refinement technique that's
   never been pointed at pitch.** `agents/detection/rhythm_engine.py`
   already backtracks a coarse peak-picked onset to the nearest real
   high-resolution energy transient via `librosa.onset.onset_backtrack` —
   this is only ever applied to `RhythmEngine`'s own onset detector (used
   mainly for drums/tempo), never to a Basic-Pitch-derived note onset.
3. **An already-installed, already-trusted model whose onset-specific tools
   are never imported.** `madmom` is a real dependency in this codebase
   (`agents/analysis/tempo_intelligence.py` imports
   `RNNBeatProcessor`/`DBNBeatTrackingProcessor` — a Dynamic Bayesian
   Network with a Viterbi decode, used as one tempo-vote witness). Confirmed
   via grep: `madmom.features.onsets.RNNOnsetProcessor`/`CNNOnsetProcessor`
   and its actual SuperFlux implementation (`SpectralOnsetProcessor`) —
   modules trained/built specifically for onset detection, not beat
   tracking — are never imported anywhere in this project.
4. **No mechanism currently adjudicates disagreement in the existing bass
   octave-shift pass** (see §1a) — #107 flags disagreement between its
   direct and shifted passes but has no independent way to resolve which one
   is right beyond raw agreement/disagreement.

---

## 2. Initiatives

### 2.0 — Fix the consensus `start_ms`/`end_ms` blend (foundational, do first)

**Scope**: in `EpistemicCouncil.analyze_sync`'s witness-grouping step, stop
letting SPICE/librosa votes contribute to a group's final `start_ms`/
`end_ms`. Only Basic Pitch (and Omnizart, if ever re-enabled) have a real
onset signal; demote SPICE/librosa to the same pitch-only-corroboration role
CREPE already correctly has. If a group has no Basic Pitch vote at all
(pure SPICE/librosa agreement), fall back to today's blended behavior —
there's nothing better to prefer in that case.

**Tie-ins**: none required — this sits upstream of everything else in this
plan. Doing it first means `Ritornello`, `AnechoicMa`'s confidence penalties,
and `PulseField`'s per-note alignment scoring all start from more trustworthy
boundaries, which amplifies the value of every later initiative rather than
competing with them.

**Effort**: tiny — a ~15-20 line change plus a regression test asserting a
synthetic SPICE-only-disagreement group no longer moves a Basic-Pitch-backed
note's timing.

---

### 2.1 — Onset refinement: backtracking + sub-band energy gating

Three separable pieces, not one — split out since they differ meaningfully
in size and in which notes they actually affect.

**2.1a — Backtrack Basic Pitch's own onsets (small).** For each note Basic
Pitch (post-§2.0) proposes, compute a high-resolution onset-strength
envelope in a small window around its coarse `start_ms` and backtrack to
the nearest true energy transient — the exact technique `rhythm_engine.py`
already uses and has validated, just never pointed at a pitched note's
boundary before.

**2.1b — Monophonic segmentation assist for CREPE/SPICE (small, narrower
value than it sounds).** Use an independent onset detector to inject hard
note boundaries into CREPE/SPICE's otherwise onset-blind pitch curves,
instead of the current naive "group consecutive same-pitch frames"
heuristic. **Be clear-eyed about what this actually buys, though**: once
§2.0 lands, SPICE/librosa can no longer set a consensus note's final timing
at all, so this mostly doesn't fix "onset placement" in the sense this
document is otherwise about. What it *does* fix: (a) corroboration
eligibility — bad SPICE segmentation can miss the 50ms agreement window and
cost a real Basic Pitch note its confidence bonus, and (b) the case where
Basic Pitch misses a note entirely and SPICE/librosa's own timing is all
that exists for it. Worth doing for those two reasons specifically, not as
a general onset-precision fix.

**2.1c — Sub-band onset gating for the low end (medium).** For low-register
content specifically, isolate the target band via a CQT filter bank and
gate raw pitch-confidence by an independent sub-band onset/energy signal:
a candidate pitch's score is suppressed unless real temporal energy backs
it up. As a product (not a geometric mean — `sqrt(P·O)` would be the actual
geometric mean; decide which behavior you want and name it correctly, since
a plain product suppresses harder at the extremes than a true geometric
mean does):

```
FusedScore(t, note) = P(t, note) × O(t, note)
```

Clarify before implementing whether `O` is binary (hard gate) or continuous
(soft discount) — the two behave very differently and "product" describes
either, so the ambiguity needs resolving in code, not left implicit.

**Tie-in — `AnechoicMa` (real, concrete, recommended, applies to all three
sub-parts)**: `AnechoicMa` already computes exactly the frame-level evidence
needed to keep any of this honest. `AnechoicReport.query(start, end) ->
SilenceState` exposes `transient_absence`, `resonance_probability`, and
`decay_completion` per window (confirmed via direct read,
`agents/analysis/anechoic_ma.py` ~line 115-230) — today those numbers are
only used to compute a post-hoc *confidence penalty* on notes that land in
silent/resonant regions. This is the first place they could inform a
*placement* decision instead: only accept a backtrack correction or a
sub-band-gated pitch if the target frame shows low `transient_absence` (a
real attack is actually there), and refuse to snap an onset into a region
scored as high-resonance/still-decaying — the "real onset vs. resonance
tail" disambiguation a naive energy search can't do alone.

**Tie-in — `PulseField` (optional, diagnostic only)**: log how far a refined
onset sits from `PulseField`'s existing `beat_grid_ms` before and after
refinement, as forensic testimony, not as a constraint. Systematic movement
*away* from every plausible metrical position is a signal worth surfacing
(pitched onsets legitimately fall off the grid for swing/rubato — don't
hard-gate on this).

**Tie-in — `Ritornello` (structural, recommended)**: implement all three
sub-parts as a new Ritornello repair pass (`OnsetPrecision`), inheriting the
existing burden-of-proof philosophy, `RevisionEvent` forensic trail, and
confidence-penalty-on-disagreement idiom already established for
`SustainRecovery` — no new arbitration pattern needs inventing.

**Effort**: 2.1a/2.1b small each; 2.1c medium on its own (real DSP work, a
CQT filter bank, and its own validation). Don't let "small-medium" hide
that 2.1c is a real, separate lift from the other two.

---

### 2.2 — Wire in madmom's dedicated onset processors as an independent witness

**Scope**: add a `_run_madmom_onset(audio, sr) -> List[float]` using
`madmom.features.onsets.RNNOnsetProcessor`/`CNNOnsetProcessor` (madmom also
has a correct, canonical SuperFlux implementation via
`SpectralOnsetProcessor` — source SuperFlux from there rather than from
`librosa.onset.onset_strength()`, whose default parameters give plain
spectral flux, not SuperFlux, despite sounding similar). Reuse
`tempo_intelligence.py`'s existing thread-local BLAS isolation pattern for
madmom rather than reinventing it.

**Tie-in — `PulseField` (real, concrete — corrected)**: feed madmom's onset
activations as an *additional evidence input to PulseField's existing
per-note alignment scoring*, not into `beat_grid_ms` itself — `beat_grid_ms`
is PulseField's detected output, not something onset evidence gets injected
into. An onset time falling close to the existing beat grid is real
corroborating evidence it's a genuine, metrically-meaningful attack. This
makes the note-scoring relationship two-way for the first time: today notes
get scored *against* the grid after the fact, but no onset evidence talks
back to where a boundary gets placed.

**Tie-in — `Ritornello`**: the same `OnsetPrecision` pass from §2.1 becomes
a genuine multi-way corroboration point — Basic Pitch's own onset
posteriorgram, the backtrack-refined transient (§2.1a), and madmom's
independent activation — resolved by confidence-weighted blend with a
disagreement penalty, the same pattern already built and tested for
posteriorgram-weighted `SustainRecovery`. No new arbitration design needed.

**Effort**: medium. The model and its decode logic already exist and are
already a trusted dependency; the work is integration, thread isolation,
and — like every other model in this project — validating its actual
quality on this project's own material before trusting it (do not assume a
pretrained onset model performs well here without a real check, per this
project's established habit of re-verifying every witness rather than
taking vendor claims at face value).

---

### 2.3 — Low-end-focused sequence decoder over Basic Pitch's raw posteriorgrams

**Scope**: the biggest, most novel piece, and — per §1a — **must be framed
as resolving/extending task #107's existing bass octave-shift pass, not
competing with it.** Two structurally different sub-decoders, split out
because they really are separate efforts:

**Track A — monophonic bass line (single-path HMM, medium-large).** States
= individual low-register MIDI notes plus a silence state. Pull the actual
note range from the already-built, already-validated family-level MIDI
range table (`InstrumentRangeValidator`, task #31) rather than hand-picking
a new range — keeps this decoder's definition of "bass register" consistent
with what the rest of the pipeline already uses, instead of a third
independently-chosen definition of the same thing.

- *Octave-jump penalty*: high sustain bias (`i → i`), moderate stepwise
  transitions (`i → i±1/±2`), heavy penalty on `±12`/`±24` semitone jumps
  in a single frame unless backed by a real onset spike.
- *Onset-gated silence→note transitions (from external review, genuinely
  useful addition)*: gate the `silence → note` transition specifically by
  the model's **onset** posteriorgram channel, distinct from using the
  **note/frame** channel for ongoing sustain evidence — pairing each
  posteriorgram channel with the specific transition it should govern,
  rather than lumping both into one generic emission score. This directly
  addresses low-end background rumble getting transcribed as ghost notes.
- *Harmonic emission boosting*: rather than a from-scratch formula, reuse
  or directly extend `agents/validation/schoenberg_mirror.py`'s existing
  `harmonic_product_spectrum()` (confirmed present, `~line 400`,
  `HPS_NUM_HARMONICS` already tuned) for "is N's harmonic series actually
  present" — this is the same underlying question a hand-rolled
  `Score(N) + α·Score(N+12) + β·Score(N+19)` formula would be answering
  independently and, per this project's own documented history, risks
  drifting out of sync with the existing tuned implementation if built
  twice.
- **How this relates to task #107**: recommend Track A run as the
  resolution mechanism for cases #107's dual-pass already flags as
  disagreeing, rather than as an independent from-scratch octave-corrector.
  #107 gives you two independent measurements and a disagreement flag today
  with no principled way to pick a winner; Track A's sequence decode is a
  good candidate for actually being that arbiter.

**Track B — polyphonic low-end (medium, needs a design fix before
building).** As proposed (independent per-pitch-bin binary trackers *plus*
a coupled transition penalty between adjacent-semitone bins to suppress
CQT-bin-bleed artifacts) **contains a real internal contradiction**:
"independent" and "coupled" can't both be true — the moment bin N's
transition cost depends on bin N±1's state, the decode is no longer exact
independent per-bin Viterbi (which is cheap), it's a joint decode over
coupled neighborhoods (which is exactly the state-space blowup this design
says it's trying to avoid, just at smaller scale). Resolve by keeping the
per-bin decode genuinely independent and exact, and moving adjacent-
semitone cleanup into a second, separate pass — which already exists:
`Ritornello`'s `PitchWobbleCollapse` already collapses near-duplicate notes
within ~1 semitone. Reuse that pattern instead of baking coupling into the
transition matrix.

This requires one small prerequisite regardless of which track: extend
`run_basic_pitch_with_posteriorgram` (`core/basic_pitch_utils.py`) — which
currently only captures the **note** posteriorgram — to also return the
**onset** channel.

**An untested, flagged alternative worth a real experiment before adopting
(from external review)**: hard low-pass filtering the audio to ~250Hz
before a dedicated bass-only Basic Pitch pass, to strip high-frequency
content that might confuse the model. Plausible, but genuinely risky, not
just a nitpick: HCQT-based models often need harmonic content *above* the
fundamental to correctly identify a weak low fundamental in the first
place — a brick-wall cutoff at 250Hz could strip exactly the 2nd/3rd
harmonic evidence needed to fix octave confusion, making it *worse*. Treat
as a real experiment to run and measure, not something to build on faith.

**Tie-in — `AnechoicMa` (the deepest, most valuable tie-in in this whole
plan)**: rather than a post-hoc filter (§2.1's use), bake `SilenceState`
directly into either track's transition costs — raise the cost of
transitioning into `onset` during a frame scored high-resonance/low-
transient; lower the cost of `sustain → silent` during a decay-complete
frame. Unifies Basic Pitch's own confidence and `AnechoicMa`'s acoustic
modeling into one decode instead of two independently-tuned sequential
filters.

**Tie-in — `PulseField` (optional, keep the weight low)**: a soft
multiplicative discount near `beat_grid_ms`/subdivision positions is
possible but should stay low-weight and clearly optional — baking meter
into a pre-quantization pitch-boundary decision risks re-implementing what
`TemporalLattice` already owns downstream. Don't blur that line without a
clear reason.

**Tie-in — `HarmonicIntelligence` (soft corroboration only)**:
`HarmonicValidator.validate(note, features)` already produces a per-note
harmonic-series/key-fit judgment. A genuine chord-context shift co-occurring
with a candidate boundary is mild evidence *for* a real new onset; an
unchanged harmonic role is mild evidence *against*. Low-weight tie-breaker
only — coarser-grained and less direct than `AnechoicMa`'s acoustic
evidence for this specific decision.

**Tie-in — `Ritornello` (important framing, not just mechanics)**: this
decoder's output should **not** silently replace what Basic Pitch's own
note-creation gives Ritornello — doing so would quietly undo the "burden of
proof on Grimlock, not the detector" principle just re-established (commit
`84e7b8f`). Treat the decoded boundary as a *second, independent
measurement*, blended against Basic Pitch's own boundary the same way
`SustainRecovery` already blends posteriorgram vs. energy evidence.

**Tie-in — `SchoenbergMirror` (recommended safety net, with a prerequisite
fix)**: route any decoded boundary's resulting note through
`SchoenbergMirror.audit_note` as a downstream plausibility gate — flag or
veto a decode whose resulting note produces an implausible harmonic-series
match. **Prerequisite**: this session's own harness run already showed
`schoenberg_mirror` timing out at its current 30s stage limit under real
note volumes (confirmed, `[WARN] Timeout after 30s`). Fix the timeout before
routing more responsibility through it, not after.

**Effort**: large, and larger than a single line item — Track A and Track B
are genuinely separate decoders with separate validation needs; don't scope
them as one deliverable.

---

## 3. Recommended Sequencing

1. **§2.0** (tiny) — fix the consensus blend bug. Independent, high-leverage,
   should happen regardless of what else does.
2. **§2.1a** (small) — backtrack Basic Pitch's own onsets + `AnechoicMa`
   transient guard, as a new `Ritornello` pass.
3. **§2.1b** (small) — CREPE/SPICE segmentation assist, for the two narrower
   reasons named above (corroboration eligibility, Basic-Pitch-miss
   fallback) — not a general onset-precision fix, scope expectations
   accordingly.
4. **§2.2** (medium) — madmom onset processors, folded into the same pass as
   a third witness.
5. **§2.1c** (medium) — CQT sub-band gating for the low end specifically.
   Sequenced after §2.2 since it's the most novel DSP work of the "cheap"
   initiatives and benefits from the other witnesses already being in place
   to cross-check against.
6. **§2.3** (large — gate on measured need). Run the existing before/after
   comparison methodology (median duration, percent-overlap — the same
   metrics used to validate the `SustainRecovery` narrowband-energy fix)
   after 1-5 land, and only build Track A/Track B if low-end octave
   confusion and general onset precision are *still* measurably lacking.
   They may not be.

## 4. How to Measure Any of This Actually Worked

Reuse the methodology already established and trusted in this project:

- The `basic_pitch_only_harness.py` isolation-subclass pattern to isolate
  each initiative's effect independently.
- Median note duration and percent-overlap against a raw, unrepaired
  baseline — the same metrics already used to validate the bass-clutter
  narrowband-energy fix.
- Ear-verification against the standalone Basic Pitch/CREPE/SPICE/librosa
  MIDI comparisons already in `output/`.

## 5. Open Questions

- Does capturing the onset posteriorgram channel (needed for §2.3)
  meaningfully increase `run_basic_pitch_with_posteriorgram`'s memory/
  runtime cost at the note-volume scale seen in the harness (7000+ raw
  notes)?
- Is madmom's onset-specific model (§2.2) actually good on this project's
  material, the way its beat-tracking DBN already proved out? Unverified —
  needs the same bare-model honesty check every other witness has already
  gone through.
- **Does Track A (§2.3) replace, run alongside, or arbitrate task #107's
  existing bass octave-shift disagreement flag?** This needs an answer
  before Track A is scoped further — it changes what Track A even is.
- Is the 250Hz low-pass pre-filter (§2.3, flagged alternative) net positive
  or net negative for octave-confusion specifically, given it may strip the
  harmonic evidence needed to resolve it? Needs a real before/after test,
  not an assumption either way.
