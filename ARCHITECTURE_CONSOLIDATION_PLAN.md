# Architecture Consolidation Plan — Redundant Subsystems

**Status: IN PROGRESS.** This is a development/planning document, not a pure
changelog — it exists so the reasoning behind decisions (what's fixed, what's
deliberately deferred and why) doesn't have to be reconstructed from a chat
transcript later. Real, verified fixes have landed under this plan (see §2);
real, more severe problems have also been *found* as a direct consequence of
that work and are tracked here rather than rushed (see §3). Nothing in this
document should be implemented as a single big-bang change — see §5 for why,
with concrete evidence from this project's own history.

This is a narrower, more urgent sibling to `GRIMLOCK_6.0_DRAFT.md` (the full
6.0 rewrite brainstorm). That document is about redesigning the architecture.
This one is about a specific defect *in the current, running 5.6.1
architecture*: the same handful of questions get answered independently, more
than once, by code that was never told the other answers exist.

---

## 1. The Problem, Stated Plainly

Over many rounds of fixes, this project accumulated several different pieces
of code that each try to answer the *same* underlying question — "where are
the onsets," "what's the tempo," "how sure am I," "what sample rate do I
need" — without knowing any of the others exist. Nothing is calling a shared
function and getting one authoritative answer; instead there are 3-9 separate
implementations per question, each computed from scratch, each with its own
tuning, and — this is the actual bug — **none of them check their answer
against any of the others.**

That absence of cross-checking is the real cost. Duplicate code by itself is
just wasted CPU. The dangerous part is what happens when two independent
implementations *disagree*: whichever one happens to be imported by the
stage that runs last quietly wins, with no flag raised that a disagreement
even occurred. This already happened, concretely:

- `_run_reverse_geo_crypt`'s real lattice search reported **"quintuplet"
  ratio family, tactus ≈ 48.8 BPM** for a 20s clip of *Hopeful.mp3*.
  `_run_tempo_analysis`'s own consensus, on the *same audio, same run*,
  reported **132.2 BPM**. Nothing reconciled these — the pipeline picked
  132.2 BPM because that's what `tempo_analysis` happens to write to
  `self._tempo`, not because anything established it was more correct.
- On *You Say God Says.mp3*, independent verification (librosa's own beat
  tracker plus raw autocorrelation, bypassing this project's code entirely)
  converged strongly on **~123 BPM**, while the pipeline's final answer was
  **74.7 BPM** — the *weakest* candidate in an independent re-check, not the
  strongest.
- `pulse_analysis` was quietly reporting **0.25 confidence** — plausible-
  looking, not obviously broken — while actually running on 1-4 real onsets
  instead of the dozens available from a working detector elsewhere in the
  same pipeline run.

None of these failed loudly. Each looked like a normal, working result. That's
what "too many ideas not working together" actually costs: not slower
runtimes, but silently-contradictory or silently-starved answers with no
mechanism to even notice, let alone resolve in favor of whichever answer is
actually right.

---

## 2. What's Been Done (fixed and verified)

Six real problems have been fixed under this plan so far, each verified with
real audio (not just a compile check) before being considered done. All are
committed and pushed.

### 2.1 Duplicate `LatticeConfidence` enum

**Problem:** `agents/quantization/temporal_lattice.py` and
`agents/analysis/reverse_geo_crypt.py` each independently defined an enum
named `LatticeConfidence`, with different members, with no shared import — a
real collision risk if any future code imported the wrong one and got
confusing attribute errors or silently-wrong comparisons.

**Fix:** Renamed `reverse_geo_crypt.py`'s copy to `GeoCryptConfidence`.
Mechanical, verified via compile-check and unaffected e2e behavior.

### 2.2 `EvidenceTracker.release()` never called

**Problem:** `orchestration/pipeline.py` only ever called
`EvidenceTracker.acquire()`, never `.release()` — so every lease taken out
during `feature_extraction` was permanent for the life of a run, regardless
of the reference-counting machinery the class exists to provide. Not a
crash, just a slow, silent leak of the exact bookkeeping meant to prevent
that class of bug.

**Fix:** Wired proper release into `_build_result()`, at the point where the
pipeline result is finalized and those buffers are genuinely no longer
needed.

### 2.3 Tempo witness-reconciliation used a naive weighted average

**Problem:** `_run_tempo_analysis`'s outer blend between
`TempoIntelligence`'s own result and `ReverseGeoCrypt`'s tactus estimate was
a plain confidence-weighted arithmetic mean of two raw BPM values, with no
check for whether the two witnesses even agreed on the same octave. This is
the exact failure mode `EnsembleTempoDetector` (this project's *internal*
multi-witness tempo ensemble) already had a proven fix for — anchor on the
highest-trust witness, only fold in other witnesses whose tempo relates to
it by a common musical ratio (rescaling them into the anchor's octave
first), and exclude anything that doesn't relate at all. The outer blend
between `TempoIntelligence` and `ReverseGeoCrypt` just never got the same
treatment.

Confirmed on the real numbers behind the *You Say God Says.mp3* bug from §1:
`tempo_intelligence` reported 90.9 BPM at 0.80 confidence; `reverse_geo_crypt`
reported 43.1 BPM at 0.59 confidence — and 43.1 is almost exactly half of
90.9 (ratio 0.474 ≈ 0.5), a classic half-tempo octave reading, not an
independent estimate of a different tempo. The naive average of these two
numbers is 74.7 BPM — a value close to neither witness and not a musically
real tempo at all.

**Fix:** Extracted the anchor-and-align reconciliation logic already proven
correct inside `EnsembleTempoDetector` into a standalone, shared function
(`anchor_and_align_tempo_witnesses()` in `agents/analysis/tempo_intelligence.py`),
and used it for *both* the internal ensemble (which used to duplicate this
logic inline) and the outer `TempoIntelligence`/`ReverseGeoCrypt` blend
(which never had it at all). One function, two call sites, instead of one
correct implementation and one broken duplicate.

**Verified:** Isolated math test on the exact real bug numbers: 74.7 BPM
(broken) → 89.3 BPM (fixed) — now correctly recognizing 43.1 as a rescalable
half-tempo reading of 90.9 rather than an unrelated value to be averaged in
directly. Confirmed via a full e2e pipeline run on the real song, and
regression-checked clean on *Hopeful.mp3* (133.9 BPM, byte-identical to the
pre-fix baseline, since that song's two witnesses already agreed on octave
and the fix is a no-op when they do).

### 2.4 Onset detection: 2 of 9 duplicate/independent sites resolved

The redundant-subsystems audit that produced this document found 9 separate
places in the codebase that independently compute an onset envelope from raw
audio, with zero cross-referencing between them (full list in §3.2 below).
Two have been investigated and fixed so far — and the investigation itself
turned out to be more valuable than the fix, in both cases.

**`orchestration/pipeline.py`'s tempo-analysis octave-correction step.**
Confirmed this one was a *pure*, zero-value duplicate: it independently
recomputed `librosa.onset.onset_strength()` on the exact same audio
(`mono_sum_normalized`, resampled to `ANALYSIS_SR`) with the exact same
`hop_length=512` that `feature_extraction` — the pipeline stage immediately
before it — had already computed and cached in `self._feature_bundle`. Fixed
by reusing the cached value. Proven mathematically equivalent
(`TempoOctaveCorrector`'s autocorrelation scoring normalizes by its own peak
internally, so it's invariant to `FeatureBundle`'s pre-normalization) and
confirmed byte-identical tempo output on *Hopeful.mp3* before and after
(133.9 BPM both times) — a pure removal of wasted computation with zero
behavior change.

**`agents/analysis/pulse_field.py`'s onset extraction.** This one was
actually dead code in practice — `pipeline.py` always supplies
`onset_times`/`onset_strengths` directly, so `PulseFieldAnalyzer`'s own
internal onset-extraction fallback never executes. But tracing *why*
uncovered a real, separate bug: the onset data `pipeline.py` *does* supply
was built from `self._rhythm_notes` (`RhythmEngine`'s output) — the same
source already proven, in an earlier part of this session, to reliably come
back near-empty on real audio (1 event on a run where a working detector
elsewhere in the same pipeline, `DrumIntelligence`, found 31 real hits in
the identical clip). `groove_analysis` and `reverse_geo_crypt` had both
already needed to route around this same broken source. `pulse_analysis`
hadn't — it was silently running on 1-4 onsets instead of dozens, and
producing a low-confidence-but-plausible-looking result (0.25) instead of
an obvious failure. Fixed the same way: prefer `self._drum_notes` when it
has enough events. Confidence went from 0.25 to 0.80 on the same real audio,
with no change to downstream note counts.

A full sweep of every other `self._rhythm_notes` use in `pipeline.py`
confirmed the remaining call sites all combine it into a larger aggregate
alongside real sources (bass/master/other pitch detections) — a near-empty
contribution there is harmless, not the same bug. No further fixes needed in
`pipeline.py` itself for this specific pattern.

---

## 3. What Problems Still Exist (open, with the fix each one is aiming for)

### 3.1 `TempoOctaveCorrector` structurally cannot correct real octave errors

**This is the most severe problem found so far under this plan — more severe
than the bug that motivated it — and it's not fixed.**

**The problem, explained carefully:** After §2.3's witness-blend fix
produces a reconciled tempo, a separate, pre-existing step
(`TempoOctaveCorrector.correct()`) runs to catch cases where *that combined
estimate itself* is off by a clean musical ratio (double, half, triplet,
etc.), by scoring several ratio-derived candidates against the real audio's
onset autocorrelation and picking whichever scores highest. This is where
the *You Say God Says.mp3* bug from §1 actually lives, underneath the bug
§2.3 fixed: 89.3 BPM (the now-correct, reconciled value) gets run through
this corrector and pulled back down to 74.4 BPM, for a reason unrelated to
the blend fix.

Proven with a synthetic ground-truth test, not just this one song: a
perfectly clean 120 BPM click track, deliberately fed a wrong 60 BPM
starting value, correctly showed strong autocorrelation support for the true
120 BPM candidate (score 0.696) — real, substantial evidence — but the
corrector still returned "no_correction_needed" and kept the wrong 60 BPM,
because the *unmodified input itself* ("exact", 60 BPM) scored even higher
(0.899), and the code treats "the top-scoring candidate is just the
unmodified input" as an unconditional stop, regardless of how strong any
other candidate's own absolute support is.

This isn't a rare edge case — it's closer to the *default* outcome for
genuinely periodic material. A signal's own fundamental period will almost
always self-correlate at least as strongly as any of its integer multiples
or submultiples, by construction (a clean pulse train's autocorrelation
naturally peaks at its own period first). Which means: whenever an upstream
witness already locked onto the wrong octave with strong internal
self-consistency, this corrector — whose entire stated purpose is catching
exactly that — will almost never intervene, because the wrong reading will
usually out-score any correction candidate on pure autocorrelation strength
alone.

**Why it isn't fixed yet:** the honest fix requires answering a genuinely
unsettled question in tempo estimation: when two ratio-related candidates
both have real, substantial evidence, which one should win? Always prefer
the faster? Always the slower? Prefer whichever direction the highest-trust
upstream witness (post-§2.3-reconciliation) already leaned? Something else
entirely? Each of those is a real, defensible design choice, and picking
wrong doesn't just fail to fix this bug — it introduces a *new*, symmetric
bug in the opposite direction (e.g., "always prefer faster" would be wrong
for the real cases where a slow reading genuinely is correct). Deciding this
confidently from two test cases (one synthetic, one real song) isn't enough
evidence. This needs validation across several more real, diverse songs
before a fix should be trusted.

**What I hope to achieve, concretely, once this gets its own focused pass:**
a decision procedure that uses the *already-computed* trust information from
§2.3's reconciliation (which witness was the anchor, how confidently) as an
input to the octave-correction step, instead of letting the correction step
re-decide everything from raw autocorrelation alone with no memory of what
the blend step already established. If the anchor witness was
high-confidence and a same-direction correction would move *away* from it,
that should count against the correction; if multiple independent
correction candidates agree with each other but not with "exact," that's a
stronger signal than one candidate's raw score beating another's. This
needs a small, dedicated multi-song validation pass before landing, not a
guess implemented in one sitting.

### 3.2 Onset detection: 7 of 9 sites not yet investigated

Two sites are fixed (§2.4). The remaining seven, in descending order of how
likely each is to hide the same "silently starved by a broken source" bug
pattern rather than being merely wasteful duplication:

| Site | Status |
|---|---|
| `agents/analysis/anechoic_ma.py:613` | Investigated, left alone — its onset call is one of four features (RMS, flux, flatness, harmonic proxy) computed together and trimmed to a common length for region classification. More tightly coupled than the two fixed sites, and no evidence yet that it's starved the way those were — forcing a consolidation here without the same evidence-first justification would be exactly the kind of unverified change this plan exists to avoid. |
| `orchestration/Stage_Zero.py:623` | Not yet investigated. |
| `agents/separation/demucs.py:207` | Not yet investigated. |
| `agents/detection/rhythm_engine.py:215-227` (`OnsetDetector`) | Not yet investigated — notable because this is the *source* of the broken `self._rhythm_notes` data that caused both fixes in §2.4. Fixing the detector itself, rather than routing every consumer around it, may be the more durable fix — see §6 of `GRIMLOCK_6.0_DRAFT.md`, which already flags `RhythmEngine` for a full rewrite on this exact evidence. |
| `agents/analysis/tempo_intelligence.py:212,1320` | Not yet investigated — these are per-witness onset calls inside `EnsembleTempoDetector`; each witness may legitimately need its own onset representation (different witnesses use different underlying algorithms), so this needs the same "verify before consolidating" check as everywhere else, not an assumption either way. |
| `agents/analysis/reverse_geo_crypt.py:909` | Not yet investigated. Would require either reaching into pipeline-level state from inside an `agents/`-layer class (a Law 2 violation) or adding an optional pre-computed-onset parameter to `ReverseGeoCrypt.run()`'s public signature — a real API design decision, not a same-class substitution, so deliberately not rushed. |
| `agents/drum_intelligence`'s own onset detectors (`HFCOnsetDetector`/`MultiBandOnsetDetector`) | Deliberately last/lowest priority per the original plan — these are real, sophisticated, frequency-band-filtered detectors that drum classification specifically depends on; consolidating them onto a shared broadband onset envelope risks losing exactly the band-specific granularity that makes them work, and should only be attempted once the pattern is well-proven elsewhere. |

**What I hope to achieve:** for each remaining site, first establish (with a
quick, targeted script against real audio, the same way the two fixed sites
were established) whether it's (a) a pure duplicate of an already-cached
computation — safe, mechanical fix, or (b) feeding on a broken/starved
source the way `pulse_analysis` was — a real bug, worth fixing regardless of
consolidation, or (c) genuinely needs its own distinct onset representation
— leave alone, document why. Do not assume any site falls into a category
without checking; two of the last three sites checked turned out to be
category (b), not the "just wasteful" case the original audit assumed.

### 3.3 Tempo estimator demotion — not yet started

`agents/detection/rhythm_engine.py:472` (`estimate_tempo`) and
`agents/analysis/groove_field.py:430` (`estimate_tempo`) both compute their
own independent tempo estimate, confirmed via grep to have zero references
to `TempoIntelligence`, `ReverseGeoCrypt`, or `PulseField` — each is a
fully isolated island. (A third, `agents/drum_intelligence/orchestrator.py:432`'s
`_fallback_tempo`, exists as well; a fourth, in the now-deleted
`agents/analysis/fallback.py`, no longer exists since that file was confirmed
dead code and removed.) The one working exception is `PulseField`, which
genuinely does integrate a tempo hint from `TempoIntelligence`'s `TempoMap`
— that pair already talks to each other and needs no further work.

**The problem this causes:** whatever code currently consumes `RhythmEngine`'s
or `GrooveField`'s own tempo estimate is trusting a number that was never
checked against the project's own best available tempo evidence, and could
silently disagree with it by an octave or more with no flag raised — the
same class of bug as everywhere else in this document, just not yet proven
to have caused a visible symptom the way the others have.

**What I hope to achieve:** per the external-review synthesis in §4, the
target isn't a new consensus mechanism — `epistemic/epistemic_council.py`'s
`EpistemicCouncil` already does exactly this kind of weighted-confidence
contention arbitration for drum-type and note-duration disputes
(`resolve_drum_type_contention`, `resolve_duration_contention`). The concrete
next step is extending it with an analogous `resolve_tempo_contention`, and
making that the one place `TempoIntelligence`, `RhythmEngine`, `GrooveField`,
and `ReverseGeoCrypt` all submit a witness to, rather than each being
consumed directly by whatever currently reads them. "Demote, don't remove"
is the default — nothing gets deleted until it's been running as a
submitted-but-non-authoritative witness long enough to show its
contribution is genuinely never load-bearing, not from a code-reading
guess.

### 3.4 Resampling — blocked on a per-site investigation, not yet started

Four parallel paths exist: the canonical `AcousticIntelligence.resample()`
(`core/acoustic_intelligence.py:493`, used by `ingestion/loader.py`), a
deprecated-but-still-live `DownsamplingEngine._legacy_resample` fallback
(`ingestion/downsampler.py`), a purpose-built `SafeResampler`
(`agents/drum_intelligence/onset_detection.py:46`, used only by
`classification.py`), and 9+ ad hoc raw `librosa.resample()` calls scattered
across `orchestration/pipeline.py`, `agents/separation/{roformer,mel_roformer}.py`,
`orchestration/Stage_Zero.py`, and `epistemic/epistemic_council.py`.

Unlike onset detection or tempo, this one is lower-risk *in principle* —
resampling is pure numerics with no musical-judgment ambiguity, so the
question is "which implementation is more correct/faster," not "which
answer is actually right." The cost of the current sprawl is purely
maintenance surface: four places to fix if a resampling bug is ever found,
with no guarantee a fix to one propagates to the others.

**What actually happened when this got investigated:** before touching any
code, checking whether `AcousticIntelligence.resample()` could safely become
the one canonical implementation surfaced a real, documented incompatibility
— it has a deliberate `"LAW: No upsampling (prevents hallucinated
high-frequency data)"` restriction, and raises `ValueError` outright if
`target_sample_rate > source_sr`. Several of the 9+ ad hoc call sites
plausibly upsample (e.g. converting a lower native file rate up to a
specific model's required input rate) — those would simply break if
retargeted onto this function without modification.

**What I hope to achieve:** the same per-site evidence-gathering discipline
used for onset detection — check each of the 9+ ad hoc sites' actual
resampling direction against real audio files with varying native sample
rates, not assume. Sites that only ever downsample can retarget onto
`AcousticIntelligence.resample()` directly. Sites that may upsample need
either a documented exception (kept as their own implementation, with a
comment explaining why) or `AcousticIntelligence.resample()` itself gaining
an explicit, opt-in upsampling mode with its own anti-hallucination
safeguard, if a strong enough case exists that it should handle both
directions. Not yet started — flagged as lower-value than §3.1/§3.2 in the
meantime, since this is a maintenance-surface question, not a correctness
bug the way the tempo/onset findings were.

### 3.5 Confidence/scoring representations — 5 incompatible types, not yet started

Five separate, mutually-incompatible confidence/scoring representations
exist: `core/order_types.py`'s `Confidence` enum, `core/contracts.py`'s
`ConfidenceComponents`, a *second, differently-shaped* class also named
`ConfidenceComponents` in `agents/drum_intelligence/drum_types.py`,
`agents/analysis/groove_field.py`'s `GrooveConfidenceComponents`/
`GrooveConfidenceScorer`, and a plain string-constant class also named
`Confidence` in `orchestration/Stage_Zero.py`, unrelated to any of the
others. No conversion or adapter exists between any of these — code that
needs to compare confidence across subsystems currently can't without
writing a one-off conversion by hand each time it comes up.

**Why this is real but lower priority:** unlike §3.1-3.4, this hasn't yet
been shown to produce a *wrong musical answer* — it's a maintainability and
consistency problem (harder to reason about, easy to introduce a bug when
converting between representations by hand), not a correctness bug like the
tempo/onset findings were. Worth a dedicated pass once the higher-priority
items above are further along, not before.

**What I hope to achieve:** a single shared confidence/evidence
representation (richer than a bare float or enum — something like
confidence + uncertainty + supporting-observation-count + which
algorithm/witness produced it, per the external-review discussion in §4)
that all five current representations converge toward, with the drum- and
groove-specific variants either folding into it directly or explicitly
staying as documented, justified specializations if they carry information
the shared type genuinely can't (not decided yet — needs the same
verify-before-consolidating discipline as everything else here).

---

## 4. External Review Synthesis

This plan was shared with two outside models (Gemini, ChatGPT) for a second
opinion. Their environment/tooling suggestions (MSYS2/MinGW, Docker/WSL2
micro-services for `vamp`/Omnizart compilation) are out of scope for this
document — Omnizart is already integrated, its real bug already fixed, and
already disabled by config pending further work, not blocked on a compiler.
What follows is what's actually relevant from their architectural feedback:
kept as-is, corrected against what this codebase already has, or rejected,
and why.

### 4.1 The core idea both reviewers converged on, and why it's already half-built

Both independently proposed the same shape: don't force every subsystem onto
one "canonical" implementation — instead, have each subsystem publish its
answer plus a confidence/provenance, and resolve disagreements through a
dedicated arbitration layer, rather than "whoever runs last wins."

That's correct, and it reframes §3.2 above: the goal isn't "`FeatureBundle`
becomes the one true onset algorithm," it's "`FeatureBundle` caches the
reusable *signal* (onset envelope, spectrogram, chroma, tempogram — things
that don't change no matter who's asking), while *interpretations* (a
specific tempo, a specific beat grid) stay owned by whichever subsystem
computed them and get reconciled through consensus, not cached as if they
were settled fact." This distinction — cache the DSP, not the conclusion —
is the single most useful correction from the outside review.

**But**: neither reviewer's proposed "Consensus Router
Engine"/"MetricConsensusBroker" is a new component this project needs to
build. It already exists, twice over:

- `epistemic/epistemic_council.py`'s `EpistemicCouncil` already implements
  exactly the weighted-confidence contention pattern both reviewers
  described — `resolve_drum_type_contention(votes: List[DrumTypeVote])`
  breaks ties between competing interpretations of the same event using
  cross-source corroboration weighted by confidence, and
  `resolve_duration_contention(...)` does the equivalent for note
  durations. Both are real, tested, already wired into the pipeline (drum
  classification, quantization).
- `agents/validation/consensus_engine.py`'s `ConsensusEngine` already
  performs whole-witness-list consensus voting with a qualification/
  disqualification/blacklist lifecycle for witnesses that behave badly
  over a run.
- `core/order_types.py`'s `WitnessTestimony` (`witness_id`, `confidence`,
  `metadata`, plus a `get_veto_vote()` method) is already the
  "provenance-carrying evidence object" both reviewers suggested inventing —
  it just doesn't have a tempo/onset-scalar-metric variant yet, and nothing
  currently submits tempo witnesses to it.

The actual gap is narrower and more specific than "build a consensus layer":
tempo's own ensemble (`EnsembleTempoDetector`) runs its internal witness vote
entirely separate from `EpistemicCouncil`/`ConsensusEngine`, and
`RhythmEngine`/`GrooveField`'s tempo estimates are consumed directly by
whatever needs them, never submitted anywhere for arbitration at all (see
§3.3). Building a second, parallel consensus broker to solve this would
itself be a fourth instance of the exact "too many ideas not working
together" pattern this whole document exists to fix.

### 4.2 Ideas adopted into the plan above

- **Demote, don't remove, is the default** — folded into §3.3.
- **Disagreement as a first-class pipeline event, not just a log line.**
  When submitted tempo witnesses cluster tightly, that's unremarkable. When
  they don't (the *Hopeful.mp3* quintuplet-vs-132-BPM case from §1), that
  should be a distinct, logged decision record once tempo witnesses flow
  through `EpistemicCouncil` (§3.3) — using the same confidence-reduction
  pattern `SchoenbergMirror`/`Scribe` already use elsewhere in this codebase
  for low-certainty verdicts, not a new mechanism.
- **A per-song diagnostic report** showing every witness's raw value next to
  the resolved consensus becomes nearly free once tempo witnesses flow
  through one place (§3.3) — the data already exists in `music_box`'s log.
  Worth building once that plumbing exists.
- **Resampling: let callers request a quality profile** (fast/offline/
  realtime) rather than importing a different implementation by hand — a
  reasonable refinement folded into §3.4's eventual design, once the
  per-site direction audit is done.

### 4.3 Ideas rejected or corrected

- **"Halt downstream tasks" on disagreement.** Rejected — every existing
  veto in this codebase (Scribe, SchoenbergMirror, `WitnessTestimony.get_veto_vote()`)
  reduces confidence or excludes a result; none halts the whole run over one
  contested metric. A tempo disagreement should lower confidence and log
  prominently, matching that convention, not introduce a new failure mode.
- **Pydantic**, suggested for the "state duplication bug" (`getattr`-based
  fallback scanning across differently-named fields for the same value).
  The diagnosis is right; the fix is using this project's own existing
  frozen dataclasses consistently and removing the dual-name fallback
  scanning, not adding a new validation dependency for a problem the
  existing conventions already solve.
- **Evidence versioning** (e.g. `TempoEvidence(version=2)`) — reasonable
  future-proofing with no concrete need yet; a one-line note for whoever
  adds a second tempo algorithm later, not something to build ahead of need.
- **Invariant assertions after every stage** — partially already exists
  (`TempoMap.__post_init__` already raises outside [20, 300] BPM,
  `Confidence`'s own values are already bounded); worth a quick audit of
  which of the 26 pipeline stages lack a check dataclasses elsewhere already
  give for free, not a new initiative.

---

## 5. Why This Is a Separate Document Instead of Just Doing It

Every fix made under this plan that touched previously-dead or
never-cross-validated code surfaced a real, previously-invisible bug the
moment it actually ran — not occasionally, but every single time:

- Wiring in the real `ReverseGeoCrypt` class surfaced that it was reading
  from `RhythmEngine`'s onset list, which reliably comes back near-empty on
  real audio.
- Wiring in `Stage_Zero.py`'s conditioning surfaced a Unicode-crash bug and
  an unconditional peak-normalization bug, in code that had *zero prior
  executions* despite being fully built.
- The Omnizart integration surfaced a case-sensitivity bug that had silently
  zeroed out one of five ensemble models for an entire session, invisible
  because the failure was caught and logged only at `warning` level.
- Fixing the tempo witness-blend (§2.3) surfaced a second, more severe,
  still-unfixed bug in the very next step of the same pipeline stage
  (§3.1) — proof that even a *verified, correct* fix can be standing next
  to a bigger problem that only becomes visible once the first one is
  actually cleared out of the way.
- Investigating `pulse_field.py`'s onset call (§2.4) turned out to be dead
  code in the literal sense (the fallback path never executes) — but
  tracing *why* is what surfaced the real, separate confidence bug.

Every consolidation item in §3 touches multiple call sites across multiple
files. Doing all of them at once — or even fully finishing one of them in a
single pass without stopping to verify — multiplies the chance of hitting
one of these latent bugs while also making it hard to tell which specific
change caused a regression if one appears. The discipline that has worked so
far (one change, real e2e verification against a known baseline, *then* the
next change) is the same discipline this plan is built around. Rushing it
defeats the reason it's being done at all.

---

## 6. Non-Goals

This document does not propose:
- A rewrite of any of the subsystems involved (that's `GRIMLOCK_6.0_DRAFT.md`'s job).
- Deleting any purpose-built variant that turns out to have a real reason to
  differ from a "canonical" version once actually compared (§3.2's
  `anechoic_ma.py`/drum-intelligence entries, §3.4's upsampling case — verify
  before consolidating, don't assume duplication always means redundancy).
- A timeline or sequencing commitment across §3.1-3.5 relative to other
  project work — this is a scoping document, not a schedule.
- A confident, one-sitting fix for §3.1 (`TempoOctaveCorrector`) specifically
  — that item needs its own multi-song validation pass before anything
  lands, not a guess made from two test cases.
