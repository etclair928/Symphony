# Architecture Consolidation Plan — Redundant Subsystems

**Status: PROPOSED, NOT YET IMPLEMENTED.** This is a development/planning
document, not a changelog. It exists so the reasoning behind a deliberately
deferred piece of work doesn't have to be reconstructed later. Nothing in
here should be implemented as a single big-bang change — see §5 for why.

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
even occurred. This already happened, concretely, this session:

- `_run_reverse_geo_crypt`'s real lattice search reported **"quintuplet"
  ratio family, tactus ≈ 48.8 BPM** for a 20s clip of *Hopeful.mp3*.
- `_run_tempo_analysis`'s own consensus, on the *same audio, same run*,
  reported **132.2 BPM**.
- Nothing reconciles these. They're both real, both computed from genuine
  DSP, and they disagree by nearly 3x. The pipeline picks 132.2 BPM because
  that's what `tempo_analysis` happens to write to `self._tempo`, not
  because anything established it was more correct.

Separately, on *You Say God Says.mp3*, independent verification (librosa's
own beat tracker plus raw autocorrelation, bypassing this project's code
entirely) converged strongly on **~123 BPM**, while the pipeline's final
answer was **74.7 BPM** — the *weakest* candidate in an independent
re-check, not the strongest. That bug is still open; see §3.2.

This is what "too many ideas not working together" concretely costs: not
slower runtimes, but silently-contradictory answers with no mechanism to
even notice the contradiction, let alone resolve it in favor of whichever
answer is actually right.

---

## 2. Two Items Already Resolved (context, not part of the remaining plan)

Two smaller instances of this same pattern were fixed directly in the audit
that produced this document, and don't need further work:

- **Duplicate `LatticeConfidence` enum.** `agents/quantization/temporal_lattice.py`
  and `agents/analysis/reverse_geo_crypt.py` each defined their own enum
  named `LatticeConfidence`, with different members, no shared import — a
  real collision risk if anything ever imported the wrong one. Fixed by
  renaming `reverse_geo_crypt.py`'s to `GeoCryptConfidence`.
- **`EvidenceTracker.release()` never called.** `orchestration/pipeline.py`
  only ever called `.acquire()`, so leases were permanent for the life of a
  run regardless of the ref-counting the class exists to do. Wired proper
  release into `_build_result()`.

The three items below are the ones still open, and are each a genuine
architectural consolidation, not a small fix — hence the separate document
instead of folding them into ordinary bug-fix work.

---

## 3. The Three Open Consolidations

### 3.1 Onset Detection — 9 independent call sites

A shared, cached onset-strength computation already exists
(`core/feature_bundle.py:807-824`, `EvidenceType.ONSET_STRENGTH`), but almost
nothing uses it. Instead, each of the following recomputes an onset envelope
from raw audio, with its own hop length and thresholding, with **zero
cross-referencing between any of them**:

| Site | What it does |
|---|---|
| `orchestration/Stage_Zero.py:623` | Own onset call inside pre-separation conditioning |
| `orchestration/pipeline.py:960-967` | Own onset call inside tempo analysis |
| `agents/separation/demucs.py:207` | Own onset call for separation heuristics |
| `agents/drum_intelligence/orchestrator.py:447` + `onset_detection.py` | `HFCOnsetDetector`/`MultiBandOnsetDetector` — real, sophisticated, but self-contained |
| `agents/detection/rhythm_engine.py:215-227` | Its own `OnsetDetector` class |
| `agents/analysis/tempo_intelligence.py:212,1320` | Own onset calls |
| `agents/analysis/reverse_geo_crypt.py:909` | Own onset call (event extraction) |
| `agents/analysis/pulse_field.py:963-972` | Own onset call |
| `agents/analysis/anechoic_ma.py:613` | Own onset call |

**Proposed plan (phased, not big-bang):**

1. Pick `core/feature_bundle.py`'s existing cached `onset_strength` property
   as the one canonical source. It's already built and already the "shared
   bedrock" location by convention — this isn't a new design, just making
   the existing intended path the real one.
2. Change consumers to *accept* onset data as a parameter rather than
   compute it, one call site at a time, starting with the ones that are
   cheapest to verify in isolation (`reverse_geo_crypt.py`, `pulse_field.py`
   — both already proven this session to run cleanly against real audio) and
   ending with the drum-intelligence package's own onset detector, which is
   the most sophisticated and most load-bearing — migrate it last, once the
   pattern is proven safe elsewhere.
3. After each single-site migration: re-run the real e2e verification
   (`test audio/Hopeful.mp3`, `You Say God Says.mp3`) and diff the specific
   stage's output against its pre-migration baseline before moving to the
   next site. Do not migrate two sites in the same change — if a regression
   appears, it must be obvious which migration caused it.
4. Leave `drum_intelligence`'s onset detectors as their own thing rather than
   forcing them onto the shared path if step 3 shows the shared onset
   envelope doesn't have the frequency-band granularity drum classification
   needs (low/mid/high band-filtered onsets, per `consensus.py`) — the goal
   is removing *accidental* duplication, not eliminating every
   purpose-built variant that has a real reason to differ.

### 3.2 Tempo/BPM Estimation — at least 3 more estimators beyond the known 3

Already known and already partly addressed this session: `TempoIntelligence`,
`ReverseGeoCrypt`, and the pipeline's own former crude heuristic (now
replaced — see §2 of the commit history, task work on `reverse_geo_crypt`).
Beyond those three, at least three more independent tempo estimators exist:

- `agents/detection/rhythm_engine.py:472` — `estimate_tempo`, IOI-based
- `agents/analysis/groove_field.py:430` — `estimate_tempo`, from IOI distances
- `agents/drum_intelligence/orchestrator.py:432` — `_fallback_tempo`

(A fourth, in `agents/analysis/fallback.py`, no longer exists — that file was
confirmed dead code and deleted this session, so it's out of scope.)

Confirmed via grep: **`RhythmEngine` and `GrooveField` have zero references**
to `TempoIntelligence`, `ReverseGeoCrypt`, or `PulseField` — each computes its
own tempo estimate in total isolation. The one real exception:
`agents/analysis/pulse_field.py:840-847` genuinely does integrate a tempo
hint from `TempoIntelligence`'s `TempoMap` — that pair already talks to each
other and doesn't need fixing.

**Update — the witness-blend half of this bug is fixed; a second, deeper bug
was found underneath it.**

The immediate cause was confirmed exactly as suspected:
`_run_tempo_analysis`'s outer blend between `tempo_intelligence` and
`reverse_geo_crypt` was a naive confidence*weight-weighted average of the
two raw BPM values, with no check for whether they even agreed on the same
octave — the identical failure mode `EnsembleTempoDetector` already had a
proven fix for internally (anchor on the highest-trust witness, rescale
others that relate to it by a common ratio, exclude the rest). That
anchor-and-align logic has been extracted into a shared function,
`anchor_and_align_tempo_witnesses()` in `tempo_intelligence.py`, used by
both the internal ensemble and this outer blend. Verified on the exact real
numbers from the bug report (`tempo_intel=90.9(conf=0.80)`,
`geo_crypt=43.1(conf=0.59)`): the naive blend gave 74.7 BPM; the shared
reconciliation correctly recognizes 43.1 as a half-tempo reading of 90.9
(ratio 0.474 ≈ 0.5), rescales it, and gives 89.3 BPM — much closer to the
higher-trust witness, as it should be.

**But the visible end-to-end symptom didn't fully resolve, because a
second, more severe bug sits downstream of this blend**, in
`TempoOctaveCorrector.correct()` — the step that runs *after* the blend to
catch cases where the *combined* estimate is itself off by a clean ratio.
Proven with a synthetic ground-truth test: a perfectly clean 120 BPM click
track, deliberately fed a wrong 60 BPM input, correctly showed strong
autocorrelation support for the true 120 BPM candidate (score 0.696) — but
the corrector still refused to correct, returning "no_correction_needed",
because the unmodified 60 BPM ("exact") candidate scored even higher
(0.899) and the code treats "the top-scoring candidate is the unmodified
input" as an unconditional stop, regardless of how strong any other
candidate's absolute support is. This isn't a corner case: for genuinely
periodic material, a signal's own fundamental period will almost always
self-correlate at least as strongly as any of its multiples, so "exact"
winning the raw score comparison is closer to the default outcome than an
edge case. This directly explains why *You Say God Says.mp3* still reports
~74 BPM after the blend fix — 89.3 BPM (correct) gets run through this
second, independently-buggy correction step and pulled back down to 74.4
BPM (a "5/6" ratio candidate that outscored "exact" on this specific song's
full-mix onset envelope, for reasons unrelated to the blend fix).

**Deliberately not fixed in this pass.** Deciding which direction tempo-
octave ambiguity should resolve when multiple ratio candidates score
plausibly (prefer the candidate closer to a typical tempo range? prefer
whichever direction the higher-trust upstream witness leaned? something
else?) is a genuine, unsettled problem in tempo estimation generally, not
a small patch — and getting it wrong risks introducing a new bias in the
opposite direction (e.g., a fix that always prefers faster candidates would
itself be wrong whenever a slow reading really is correct). This needs its
own focused investigation with validation across several real songs, not a
fix made confidently from two test cases. Tracked as open work, separate
from the witness-blend fix above, which is complete and verified.

**Proposed plan:**

1. Fix the immediate bug first, in isolation: instrument
   `_run_tempo_analysis`'s witness combination step to log each witness's
   raw value/confidence *before* blending (this may already be logged to
   `music_box` — check first), then re-run on *You Say God Says.mp3* to see
   exactly how 90.9 BPM (`tempo_intelligence`'s own witness) and 74.7 BPM
   (the final blended answer) diverge. This is a one-song, one-stage
   debugging pass, not a consolidation — do this before touching the other
   estimators.
2. Once the immediate bug is understood, decide whether `RhythmEngine` and
   `GrooveField`'s independent tempo estimates should be:
   - **Removed**, if they're now fully redundant with `TempoIntelligence`
     once fixed (both currently exist because whatever consumed them
     couldn't get a tempo value from the real system easily at that point in
     the pipeline — check whether that's still true), or
   - **Kept but demoted to corroborating witnesses**, feeding into
     `TempoIntelligence`'s own consensus (matching the `PulseField` pattern
     that already works) rather than being consumed directly by whatever
     currently reads them.
3. Migrate one consumer at a time, same discipline as §3.1: real e2e
   verification after each change, not batched.

### 3.3 Resampling — 4 parallel paths

- **Canonical:** `core/acoustic_intelligence.py:493`,
  `AcousticIntelligence.resample()` — used by `ingestion/loader.py:205,225`.
- **Deprecated but still live:** `ingestion/downsampler.py`'s
  `DownsamplingEngine`, which still has a working `_legacy_resample`
  fallback path despite being marked deprecated.
- **Purpose-built:** `agents/drum_intelligence/onset_detection.py:46`'s
  `SafeResampler` (polyphase `resample_poly` with a librosa fallback), used
  only by `classification.py:216`.
- **Ad hoc:** raw `librosa.resample()` calls bypassing all three of the
  above, at 9+ sites in `orchestration/pipeline.py`, plus
  `agents/separation/roformer.py:469,532`,
  `agents/separation/mel_roformer.py:756,777`,
  `orchestration/Stage_Zero.py:312`, and `epistemic/epistemic_council.py`'s
  own `_lib.resample` calls.

This one is lower-risk to consolidate than §3.1/§3.2 — resampling is a pure
numerical operation with no musical-judgment ambiguity (unlike tempo/onset
detection, there's no question of "which answer is right," only "which
implementation is more correct/faster"), so disagreement-between-methods
isn't the concern here. The concern is purely maintenance surface: four
places to fix if a resampling bug is ever found, and no guarantee a fix to
one propagates to the others.

**Proposed plan:**

1. Confirm `AcousticIntelligence.resample()` and `SafeResampler` produce
   numerically equivalent output on real audio (a same-input diff test, not
   a design discussion) before touching anything — if they differ in a way
   that matters (e.g., `SafeResampler`'s polyphase method vs. whatever
   `AcousticIntelligence` uses), that difference needs to be understood
   before deciding which one becomes canonical.
2. If equivalent: retarget every raw `librosa.resample()` call site onto
   `AcousticIntelligence.resample()`, mechanically, one file at a time,
   compiling and running that file's own existing tests (or a quick smoke
   script) after each file.
3. Finish deleting `ingestion/downsampler.py`'s legacy path once nothing
   depends on it — it's already marked deprecated; this closes the loop
   rather than leaving a half-deprecated file around indefinitely.

### 3.4 Confidence/Scoring Representations — 5 incompatible types (bonus, lower priority)

Not explicitly called out as one of the three "flagged" items, but the same
root problem: `core/order_types.py:90`'s `Confidence` enum,
`core/contracts.py:222`'s `ConfidenceComponents`,
`agents/drum_intelligence/drum_types.py:82`'s *separate* `ConfidenceComponents`
(different fields, same name), `agents/analysis/groove_field.py:98,454`'s
`GrooveConfidenceComponents`/`GrooveConfidenceScorer`, and
`orchestration/Stage_Zero.py:93`'s plain string-constant `Confidence` class
(unrelated to any of the above). No conversion/adapter exists between any of
these — code that needs to compare confidence across subsystems currently
can't, without writing a one-off conversion each time.

This is real technical debt but lower urgency than §3.1-3.3, since it hasn't
yet been shown to produce a wrong *musical* answer the way the tempo
disagreement did — it's a maintenance/consistency issue, not (yet) a
correctness one. Worth a pass once the other three are done, not before.

---

## 4. Why This Is a Separate Document Instead of Just Doing It

Every fix made this session that touched previously-dead or
never-cross-validated code surfaced a real, previously-invisible bug the
moment it actually ran:

- Wiring in the real `ReverseGeoCrypt` class surfaced that it was reading
  from `RhythmEngine`'s onset list, which reliably comes back near-empty on
  real audio.
- Wiring in `Stage_Zero.py`'s conditioning surfaced a Unicode-crash bug and
  an unconditional peak-normalization bug, in code that had *zero prior
  executions* despite being fully built.
- The Omnizart integration surfaced a case-sensitivity bug that had silently
  zeroed out one of five ensemble models for the entire session, invisible
  because the failure was caught and logged only at `warning` level.

Every one of §3.1-3.3 touches multiple call sites across multiple files.
Doing all three at once — or even fully finishing one of them in a single
pass — multiplies the chance of hitting one of these latent bugs while also
making it hard to tell which specific change caused a regression. The
discipline that worked this session (one change, real e2e verification,
compare against a known baseline, *then* the next change) is the same
discipline this plan is built around. Rushing it defeats the reason it's
being done at all.

## 5. External Review Synthesis

This plan was shared with two outside models (Gemini, ChatGPT) for a second
opinion. Their environment/tooling suggestions (MSYS2/MinGW, Docker/WSL2
micro-services for `vamp`/Omnizart compilation) are out of scope for this
document — Omnizart is already integrated, its real bug already fixed, and
already disabled by config pending further work (see commit history), not
blocked on a compiler. What follows is what's actually relevant from their
architectural feedback: kept as-is, corrected against what this codebase
already has, or rejected, and why.

### 6.1 The core idea both reviewers converged on, and why it's already half-built

Both independently proposed the same shape: don't force every subsystem onto
one "canonical" implementation — instead, have each subsystem publish its
answer plus a confidence/provenance, and resolve disagreements through a
dedicated arbitration layer, rather than "whoever runs last wins."

That's correct, and it should reframe §3.1/§3.2 above: the goal isn't
"`FeatureBundle` becomes the one true onset algorithm," it's "`FeatureBundle`
caches the reusable *signal* (onset envelope, spectrogram, chroma, tempogram
— things that don't change no matter who's asking), while *interpretations*
(a specific tempo, a specific beat grid) stay owned by whichever subsystem
computed them and get reconciled through consensus, not cached as if they
were settled fact." This distinction — cache the DSP, not the conclusion —
is the single most useful correction from the outside review, and it
resolves an ambiguity in the original §3.1 wording.

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

The actual gap — confirmed via the redundant-subsystems audit that produced
§3 — is narrower and more specific than "build a consensus layer": **tempo's
own ensemble (`agents/analysis/tempo_intelligence.py`'s
`EnsembleTempoDetector`) runs its internal 3-witness vote entirely separate
from `EpistemicCouncil`/`ConsensusEngine`, and `RhythmEngine`/`GrooveField`'s
tempo estimates are consumed directly by whatever needs them, never
submitted anywhere for arbitration at all.** Building a second, parallel
consensus broker to solve this would itself be a fourth instance of the
exact "too many ideas not working together" pattern this whole document
exists to fix. The corrected plan: extend `EpistemicCouncil` with a
`resolve_tempo_contention(votes)` alongside its existing
`resolve_drum_type_contention`/`resolve_duration_contention`, and make that
the one place `TempoIntelligence`, `RhythmEngine`, `GrooveField`, and
`ReverseGeoCrypt` all submit a witness to, rather than each independently
deciding whose tempo estimate wins.

### 6.2 Ideas worth adopting

- **Demote, don't remove, is the default — not just an option.** §3.2
  originally offered "removed" or "demoted to witness" as two options to
  weigh. Both reviewers pushed harder on this than the original draft did,
  and they're right: nothing gets deleted until it's been running as a
  submitted-but-non-authoritative witness for long enough to show its
  contribution really is fully redundant. Demotion is the default; removal
  is only what happens after real evidence (not a code-reading guess) shows
  a witness never wins arbitration and never breaks a tie.
- **Disagreement as a first-class pipeline event, not just a log line.**
  When submitted tempo witnesses cluster tightly (121, 122, 123 BPM), that's
  unremarkable. When they don't (123, 48, 127), that's exactly the
  *Hopeful.mp3* quintuplet-vs-132-BPM situation from §1 — and right now
  nothing distinguishes "witnesses agree" from "witnesses wildly disagree
  and we picked one anyway" in the pipeline's own output. Once tempo
  witnesses flow through `EpistemicCouncil`, a disagreement check (are the
  submitted values clustered, or split into groups with no clear majority?)
  should get logged to `music_box` as its own decision record, distinct from
  the resolved value, using the confidence-reduction pattern
  `SchoenbergMirror`/`Scribe` already use elsewhere in this codebase for
  low-certainty verdicts — not a new mechanism, the existing soft-veto
  pattern applied to a metric it doesn't currently cover.
- **A per-song diagnostic report showing every witness's raw value next to
  the resolved consensus** is worth building once tempo witnesses actually
  flow through one place — it becomes nearly free at that point (the data
  already exists in `music_box`'s log), and it directly answers "why did the
  pipeline pick this tempo" without re-running a debugging session by hand,
  the way §1's investigation had to.
- **Resampling: let callers request a quality profile** (fast/offline/
  realtime) rather than each call site importing a different implementation
  by hand — a reasonable refinement to §3.3, though still lower priority
  than §3.1/§3.2 since resampling correctness isn't in question here, only
  which of the 4 existing paths callers happen to have picked.

### 6.3 Ideas rejected or corrected

- **"Halt downstream tasks" on disagreement.** One reviewer's "Epistemic
  Veto Contract" proposed stopping the pipeline outright when witnesses
  disagree past a threshold. That's a harder failure mode than anything
  else in this pipeline uses. Every existing veto in this codebase (Scribe,
  SchoenbergMirror, the `WitnessTestimony.get_veto_vote()` pattern above)
  reduces confidence or excludes a specific result — it never halts the
  whole run over one contested metric. A tempo disagreement should lower
  confidence and get logged prominently, matching that existing convention,
  not introduce a new stop-the-pipeline failure mode this codebase has
  deliberately avoided everywhere else.
- **Pydantic.** Suggested as the fix for the "state duplication bug"
  (`getattr(rhythm_field, 'tempo', None) or getattr(rhythm_field, 'bpm',
  None)`-style fallback scanning). The underlying diagnosis is correct and
  worth fixing, but the fix is using this project's own existing frozen
  dataclasses consistently (e.g. `TempoMap`, `WitnessTestimony`) and
  removing the `getattr`-fallback call sites that read the same value two
  different ways, not introducing a new validation library/dependency for a
  problem the project's existing dataclass conventions already solve.
- **Evidence versioning (`TempoEvidence(version=2)`).** Reasonable
  future-proofing in the abstract, but there's no concrete case in this
  codebase yet where an algorithm change needs to be distinguished from a
  data-shape change by consumers — worth a one-line note for whoever
  eventually adds a second tempo algorithm to `TempoIntelligence`, not
  something to build ahead of that need.
- **Invariant assertions after every stage** (sample rate matches,
  confidence in [0,1], tempo in a plausible range). Partially already
  exists — `TempoMap.__post_init__` already raises outside [20, 300] BPM,
  `Confidence`'s own values are already bounded — so this isn't a new
  pattern to introduce, it's a check for which of the 26 pipeline stages
  are missing a bounds-check `TempoMap`/`Confidence`-style dataclasses
  already give for free elsewhere, worth a quick audit pass but not its own
  initiative.

---

## 6. Non-Goals

This document does not propose:
- A rewrite of any of the subsystems involved (that's `GRIMLOCK_6.0_DRAFT.md`'s job).
- Deleting any purpose-built variant that turns out to have a real reason to
  differ from the "canonical" version once actually compared (§3.1 step 4,
  §3.3 step 1 — verify before consolidating, don't assume duplication always
  means redundancy).
- A timeline or sequencing commitment across §3.1/3.2/3.3 relative to other
  project work — this is a scoping document, not a schedule.
