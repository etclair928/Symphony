# Grimlock 6.0 — Design Decisions

**This is the bible.** Captured from the 2026-07-14/15 teardown/brainstorm
session. These are the load-bearing commitments for the 6.0 rebuild, now
being built fresh in `Jazz` (Symphony remains the reference/resource; nothing
here obligates copying Symphony's code, only its lessons).

**Relationship to `docs/archive/GRIMLOCK_6.0_DRAFT_SUPERSEDED.md`:** that
earlier brainstorm doc is archived, not deleted — it has real content worth
mining (a "Detection vs Interpretation" cross-cutting principle, a
TimbreIntelligence/Instrument-Registry proposal, a tracked known-bugs list).
But its own Core Thesis ("pitch detection is comparatively solved, weight
effort toward Rhythm_Engine instead") is superseded by measured evidence: §0
below shows detection genuinely IS fine (88% faithful), and the actual
disease was the repair/rhythm-adjacent machinery (quantization, velocity
merge, consensus - much of what that draft clusters under Rhythm_Engine)
mutating and deleting notes it had no evidence to touch. Both claims can be
true at once - notation/rhythm work is real and still needed - but the fix
isn't "spend more effort there," it's the immutability wall in §2. Where the
two docs conflict, THIS one wins.

---

## 0. Why 6.0 — the evidence that forced it

Measured on the Grey on Grey "other" stem (60s clip), exact-pitch match
within 300ms against raw Basic Pitch as the reference:

| Stage | Notes | Fidelity to raw Basic Pitch |
|---|---|---|
| Raw Basic Pitch (the floor) | 1707 | — |
| EpistemicCouncil output (voting + CREPE) | 1712 | **88%** |
| Final exported track | 477 | **22%** |

**Detection is not the problem.** The council barely moves Basic Pitch
(88% faithful) — meaning all the multi-witness voting adds cost for almost
nothing. The downstream repair/consensus/quantization stack then takes 88%
fidelity down to **22%** while deleting 72% of the notes. That collapse is
the entire justification for the rebuild.

CREPE refinement specifically: 88% (with) vs 88% (without) — near-zero
benefit, 274 note pitches overwritten for one point of net change. Not the
villain, but it does not earn its cost on polyphonic stems.

---

## 1. The Law (thesis)

- **Basic Pitch is both baseline AND floor.** It *is* the transcription,
  not a candidate to be argued with.
- **Amplify, don't fight.** No layer exists to "correct" the models. Layers
  exist only to add what a model genuinely can't see, and only when proven.
- **Burden of proof.** Nothing changes or deletes a detected note's pitch
  or existence without positive audio evidence that overrides it. The
  default is trust, not trial.

---

## 2. Architectural principles

1. **Immutable detection floor** — detected notes are frozen.
2. **Annotation, not mutation** — downstream passes attach opinions as
   metadata; they never rewrite pitch or delete notes. A single `Compositor`
   reads notes + annotations and emits output in one auditable place.
3. **Detection walled off from presentation** — quantization / velocity /
   timing can never delete a note or change a pitch.
4. **Every pass earns its place** — measured against the floor with a
   first-class harness, or it does not ship.
5. **Fail loud** — no silent timeout / partial / empty result that still
   reports SUCCESS.
6. **One source of truth per fact** — no scalar copies, no parallel stores.
7. **Guided means guided** — user-supplied tempo/key/meter is a hard lock
   nothing re-arbitrates.

---

## 3. Pitch layer

- **Pitch = Basic Pitch per stem.** No council, no voting, no median-blend.
- **CREPE = bass specialist only**, run as its own read (not a mid-stream
  refiner). Drop SPICE and librosa from the pitch path entirely.

---

## 4. Separation

- Attack the "other" junk-drawer at the **audio layer first**: the 6-stem
  Demucs model (`htdemucs_6s`) natively splits **guitar and piano** out of
  "other." (Local copy currently fails a download-hash check — needs a clean
  re-download.)
- Clean audio in = clean notes out. Prefer separating instruments upstream
  over sorting notes downstream.

---

## 5. Instrument Attribution — timbre + voice-continuity, COUPLED

The failure to fix: today timbre → voice is a one-way hand-off; the loop
never closes, so instrument-identity flicker (the "Klangio problem" — one
sax line labeled Sax / Violin / Sax across consecutive measures) is
structurally unfixable.

**Voice continuity is a CONSOLIDATOR (a coherence prior), not a splitter.**
Its job is to prevent (a) identity flicker and (b) over-fragmentation of one
musical line into many staves/tracks.

6.0 makes them **one subsystem, three phases:**

1. **Fingerprint** — per-note timbre votes. Noisy, physical, not trusted
   individually.
2. **Stream** — melodic continuity groups notes into coherent lines
   (pitch-contiguity + temporal proximity + contour continuation). Grounded
   in real MIR theory (auditory stream formation / voice-leading rules).
3. **Resolve** — each *line* gets ONE instrument identity from the
   **aggregate** of its notes' timbre votes, written back to every note. The
   line is the unit of identity, so flicker dies by construction.

- Bias **conservative on splitting**; strong timbre evidence can force a
  split.
- Conflict case = a line whose timbre votes are **bimodal** → a candidate
  split point → evidence-gated decision (real instrument handoff vs
  classifier error). This is the one place real intelligence is spent.
- **Annotation-only:** relabels / groups / routes notes to tracks; never
  touches pitch or deletes notes.

---

## 6. Keep vs rip out

**Keep:** source separation, the DrumIntelligence package (rewritten and
verified this session), MIDI export/engraving, the test/harness
infrastructure, Basic Pitch integration + model caching.

**Rip out:** EpistemicCouncil multi-witness voting, CREPE refinement, the
octave-shift passes (both already disabled), the consensus median-pitch
blend, and the repair-pass chain that gets a vote on a note's pitch.

---

## 7. System architecture (6.0 layer layout)

### Core (bedrock)
Pure types + ambient services. **No ML models, no decision logic.**
- **Canonical types** — one type per concept: one immutable `Note`, one
  `Confidence`, one tempo/meter representation, one stem/source identifier.
- **Ambient services** (callable from anywhere, decide nothing):
  - **Music_Box** — append-only forensic ledger. Passive logger; records
    every hypothesis / veto / transformation for perfect backward
    traceability. It records; it does not arbitrate.
  - **MusicalFindingsMap** — the blackboard: shared state where agents
    publish conclusions. Passive container, single source of truth.
- **Translators/adapters live ONLY at the edges** (external model output →
  canonical type; canonical type → MIDI). No internal module-to-module
  converters — that is where field-loss bugs breed. If you find yourself
  writing a converter between two *internal* modules, the two types should
  have been one. Unify, don't translate.

### Ingestion (Input)
- Pre-scan / file integrity: SHA-256 at ingestion; detect corruption and
  clipping artifacts.
- Tag integrity issues into metadata **and surface them loudly to the user.**
  Do NOT silently reshape downstream engine behavior from a flag — keep any
  behavior change explicit and few. (Invisible flag-driven coupling is the
  same pattern that made 5.x hard to reason about.)

### Audio Engine
Highly optimized DSP domain. **No ML models live here.**
- **Canonical audio proxy** — immutable, read-only, strictly-typed audio
  slices (standard frames/segments). Agents get views, never mutable copies.
- **Decoder boundary** — wraps FFmpeg/SoundFile; decodes once to a single
  canonical array per track. Everything downstream is a view into it.
- **Unified resampler** — one authoritative resampler. Model wrappers NEVER
  call `librosa.resample` themselves; they ask the Audio Engine for "the
  16kHz view" of the track. Kills scattered per-stage resampling and the
  temp-WAV round-trips of 5.x.
- **Cached transform layer** — STFT / CQT / Mel / onset-envelope / chroma
  computed once, served from a byte-budgeted LRU cache. **FeatureBundle's
  contents live HERE** (they are cached derived audio data).
- **Vectorized DSP transforms** — the high-speed ops (rolling-percentile
  filters, etc.).
- **Build order:** ship the *API boundary* (ask-for-a-view / transform /
  resample) backed by a plain LRU cache first. Defer zero-allocation
  mmap / virtual-array optimization until profiling proves memory is the
  bottleneck — it slots in later behind the same interface.

### Model Registry
- One home for all model wrappers (Demucs, Madmom, Basic Pitch, CREPE, ...).
- Lazy-load; cache across calls; **owns model lifetime** — unloads weights
  under memory pressure. Most real memory management actually lives here.

### Separation Engine
- 6-stem Demucs (`htdemucs_6s`) to pull **guitar and piano** out of the
  "other" junk-drawer at the audio layer. Optional second-pass separation on
  the residual "other" later.

### Pitch Engine
- Basic Pitch per stem = the transcription. CREPE bass-only, as its own read.
  No council, no voting, no median-blend, no mid-stream refinement.

### Instrument Attribution (post-pitch subsystem — its own home)
- Timbre + voice-continuity **coupled** (see §5): fingerprint → stream →
  resolve. Annotation-only; the line is the unit of instrument identity.
  Give it an explicit home so it doesn't get orphaned the way
  TimbreIntelligence effectively was in 5.x.

### Rhythm Engine
- Tempo / meter / groove analysis, plus the **DrumIntelligence** package
  (keep the verified 5.x drum package — it's rhythm + drum-type
  classification and belongs here).

### Epistemic layer (scoped DOWN — a referee, not a gate)
- A small **scalar-contradiction referee**, NOT a layer that every musical
  statement passes through.
- Consensus arbitrates genuine scalar disputes **only**: tempo (120 vs 60),
  key, meter (3/4 vs 4/4) — weighted voting, circular statistics,
  ratio-folding.
- **Notes flow through untouched by default.** Basic Pitch already decided a
  note exists — there is no "dispute" to arbitrate on a note. The veto is
  rare and requires positive contradicting evidence; it does not casually
  delete notes.
- Music_Box + MusicalFindingsMap are ambient Core services, NOT part of this
  arbiter.

### Memory (a policy, not a place)
- **No "Memory Engine" stage.** Memory = ownership + budgeted cache + lazy
  model lifetime, distributed to the three owners of the big objects:
  1. Audio Engine owns decoded audio (one copy, views out, dropped when done).
  2. The LRU transform cache holds features (evict by byte budget).
  3. Model Registry owns weights (lazy load, unload under pressure).
- "Memory management" is a byte-budget + eviction *policy* these owners
  consult — not a manual GC manager (which just fights Python's runtime).

### Web
- Presentation/API surface over the pipeline. Unchanged in principle.

### Output → Scribe Engraver
- The single **Compositor / commit step**: reads the immutable notes + all
  attached annotations and emits the MIDI in ONE auditable place. This is
  where the MIDI transcription lives.

### Orchestration Conductor
- **6.0: correct, deterministic, LINEAR.** Get it right first.
- **Deferred to 6.1+:** parallel DAG execution — and only for the stages
  profiling proves are worth parallelizing.
- **Explicitly NOT building: feedback / auto-re-analysis loops.** When
  rhythm and pitch disagree about meter, SURFACE the contradiction (log it;
  in guided mode ask the user, or take the higher-confidence reading and
  record that we did) — never an adaptive loop that can oscillate and
  destroy the traceability Music_Box exists to provide.

---

## 8. Deliberately rejected (recorded so we don't re-propose the traps)

Measured evidence and this session's failures argue against each of these,
however rigorous they *feel*:

- **"Nothing becomes truth without passing through the Epistemic Engine"** —
  the gate-everything model. It is what took fidelity 88% → 22%.
- **Note-deleting Veto Authority as a routine gate** — a loaded gun pointed
  at the transcription; it flagged 46% of real notes as hallucinated.
- **Zero-allocation mmap engine as a first build** — premature optimization
  of the subtlest-bug part before allocation is proven to be the bottleneck.
- **A "Memory Engine" that manages GC / buckets** — fights Python's runtime;
  memory belongs to the object owners.
- **Parallel DAG executor as a 6.0 foundation** — non-determinism before
  correctness; 5.x's failures were sequential plumbing, not concurrency.
- **Feedback / auto-re-analysis loops** — oscillation risk; destroys the
  "I can explain every note" property.
- **Silent "raise noise tolerance downstream" from a metadata flag** —
  invisible coupling.

---

## 9. Core canonical-type audit (2026-07-15, concrete evidence from Symphony)

Real collisions found before writing Jazz's Core types, so the new types are
informed by what actually broke, not guessed at:

- **`Note` (Symphony's `NoteEvent`) is not actually immutable** - a plain
  mutable dataclass every stage appends to / mutates in place ("NoteEvent is
  a plain, non-frozen dataclass" justifies in-place tagging repeatedly in
  Symphony's history). A genuine dead shadow duplicate also exists in
  `Stage_Zero.py` (`source: str` vs the real one's `source: SourceType`
  enum) - confirmed unused, never imported, but a landmine shape if it ever
  had been. **6.0: `Note` is frozen.** `reasoning_chain`/annotations move OFF
  the note entirely into a separate append-only side-store keyed by note
  identity - even the audit trail shouldn't live on a mutable-by-convention
  object. `midi_channel` (export-specific: channel 9 = drums) is dropped from
  the type; it's derived at render time, not baked into the detection layer.
- **"Which stem" was never a real field - it was string-mining.** Seven
  separate call sites in Symphony's pipeline re-derive stem identity by
  grepping `reasoning_chain` for a substring like `"stem:bass"` via a
  `_stem_tag()` helper. Meanwhile `SourceType` conflates "which stem"
  (`BASS_STEM`, `MASTER_STEM`, ...) with "which algorithm" (`RHYTHM`,
  `DEMUCS`, `RITORNELLO`, ...) in one enum, while a separate `StemType` also
  exists with overlapping `BASS`/`OTHER`/`VOCALS` members. Three ways to say
  the same thing, and the canonical one isn't even a field. **6.0:** `Note`
  gets a real `stem: StemType` field. `SourceType` (or its 6.0 equivalent)
  is narrowed to ONLY algorithmic provenance; stem identity lives only in
  `StemType`.
- **Tempo/meter genuinely duplicated three ways.** `tempo_bpm` is stored
  independently in `TempoTestimony`, in `TempoTestimony.tempo_map.
  initial_tempo_bpm`, AND in `BeatGrid.tempo_bpm` - three copies of one
  number that can (and did) drift out of sync. `confidence` is a raw float
  in `TempoTestimony` but the `Confidence` enum in `TempoMap` for the same
  concept. Time signature is `numerator`/`denominator` ints in `TempoMap`
  but a formatted string `"4/4"` in `BeatGrid` - an unforced translation
  boundary. **6.0: one `TempoMeter` type** - tempo_bpm, confidence,
  time_signature (ints, never a string form), beat/downbeat times, in one
  place.
- **Separation is genuinely two objects for one fact** (`SeparationResult`
  with `.separator_used` vs `SeparationTestimony` with `.model_used` - the
  exact bug that shipped earlier this session). **6.0: one `Separation`
  type** - references the stem audio views (Audio Engine owns the actual
  audio) plus the metadata (model used, confidence, timing) in one place.
- **Already resolved, no 6.0 action needed:** the `ConfidenceComponents` /
  `Confidence` naming collisions (two different `ConfidenceComponents`
  classes; a `Confidence` in `Stage_Zero.py` colliding in name with the real
  enum) were renamed to `DrumConfidenceComponents`/`StageZeroConfidence`
  earlier this session - ARCHITECTURE_CONSOLIDATION_PLAN.md §3.5, done.

---

## 10. Where we're heading (2026-07-17, from the 4.7 → 5.6.1 → 6.0 open-problems triptych)

Three documents now formalize the whole lineage's failures, read against real
source (`Grimlock_OpenProblems_Archive/`: the 4.7 bug audit + pathology map,
the 5.6.1 accretion pathologies, the 6.0 domain walls). They converge on a
single finding that should steer 6.0 from here.

### 10.1 The one inherited reflex (name it, so we can catch it)

Every failure from 4.7 onward is a tactic of **one reflex**:

> **When unsure, emit a confident-looking guess and keep going.**

The tell is always the same: **the system never crashes — it fails silently.**
4.7 returned `key = "C"` because a default pre-empted detection; it read
`drums = 0.0` (an *absence*) as a low-confidence *signal* and fired a false
"impossible consensus"; it used `getattr(note, "pitch", 60)` so a malformed
object yields middle C instead of an error. 5.x shipped 74.7 BPM — the
*weakest* candidate on independent re-check — with nothing raised. The reflex
is not a line of code you can delete; it is a habit that rides along with every
capability we port forward. **Most of §2's laws are already explicit refusals
of it** (frozen `Note`, annotate-don't-decide, one-type-per-concept,
surface-loudly, no invisible flag coupling). That the laws line up one-for-one
against the reflex is the strongest evidence the diagnosis is right.

**Forward law (elevate to §2 status): prefer a loud "contested / I don't know"
over a plausible default.** A stage that cannot answer must emit an explicit
low-confidence / contested annotation, never a silent fallback value. If a
default is unavoidable, it is logged as a default, not laundered into the data.

### 10.2 Two debts are still unpaid — these are the direction

The scorecard: canonical types (paid), frozen Note + annotation store (paid),
the Epistemic referee (first real attempt at closing the reconciliation loop
4.7 left open). Two strands are **not** retired, and they are where 6.0 should
point next:

1. **Magic-threshold-as-arbiter.** The `note_support` over-detection filter's
   cutoff sits essentially at the mode of the feature distribution — which is
   4.7's `min_confidence = 0.35` wearing a lab coat. A hard scalar cutoff is a
   decision disguised as a constant, and it is fragile exactly where it is
   placed (highest-density region → maximal label instability under the
   run-to-run variance a still-unseeded Demucs injects). **Direction:** no
   filter ships with a magic scalar cutoff. The operating point must be defined
   in **probability space relative to the run's own distribution** (CFAR-style,
   `GRIMLOCK_6.0_OPEN_PROBLEMS.md §IX.1`), with the false-alarm rate chosen, not
   the threshold guessed. This is a *hard gate on new filters*, not a nice-to-have.

2. **Observability + a provably-closed reconciliation loop.** 4.7 built the
   `ConsensusEngine` and never wired it to act; 6.0's referee is the first loop
   that actually closes, but we cannot yet *prove* it closed because the system
   is under-instrumented and non-deterministic. **Direction:** (a) `Window_Pane`
   (built, `core/window_pane.py`) and `Music_Box` must log **alternatives
   considered + runner-up margins**, not just winners — a decision that discards
   its runner-ups is unobservable in the control-theory sense, and that silence
   is what let every past error hide; (b) **determinism lands first** (seed or
   disable Demucs's random shift) so any reconciliation claim is even measurable.

### 10.3 The sequence (leverage-ordered, from the 6.0 open-problems §XI)

Do these in order; the first two are *measurement infrastructure* and everything
after is unverifiable without them (calibrate the instrument before the science):

1. **Determinism** — seed Demucs. Cheapest, unblocks all measurement. **✓ DONE (2026-07-17):** `separation_engine/demucs_engine.py` now seeds random/numpy/torch before `apply_model` (default `seed=0`, `seed=None` opts out). Verified byte-identical stems (seed=0) vs. differing stems (seed=None, PIANO Δ0.107), AND byte-identical full-pipeline MIDI on Hopeful-20s across two runs. A whole-pipeline RNG audit confirmed Demucs was the *sole* non-deterministic source (meter.py's shuffle is already locally seeded), so the pipeline is now fully reproducible.
2. **Injection–recovery harness** — synthesize ground truth (render known MIDI
   into real audio, score recovery) so "better" stops meaning "sounds better to
   me" and starts meaning a measured precision/recall. Build this *before* the
   next filter, so the next filter is the first one we can actually score.
3. **CFAR-style `note_support`** — re-cut §10.2(1) with a probability-space
   operating point; add the damped-oscillator ownership feature if band energy
   under-discriminates.
4. **Conservation / optimal-transport cross-stem assignment** — the structural
   attack on bleed: stop asking "is this note real?" (unanswerable in isolation)
   and ask "which stem owns this note?" under the closure constraint
   Σ stems ≈ mixture. Replaces per-note verdicts with joint allocation.
5. **The MusicXML/MEI fork** — clean notation is unreachable on a MIDI target by
   construction (`§V` quotient proof); decide the format explicitly or scope the
   deliverable to "a good MIDI a human finishes."
6. **Factor-graph unification** — horizon only. Revisit if the remaining errors
   look like *coordination failures between subsystems* rather than evidence
   failures. Not a 6.0 work item; the annotation store is already its tractable
   first-order form.

### 10.4 The standing goal this serves

The north star is unchanged: **it working and beating Klangio matters more than
how it looks.** Klangio's moat is its backend, not its piano roll — and ours is
the part that's real. The path above is how the backend earns the front end:
determinism and injection–recovery give us a Klangio head-to-head that is a
*measured* result, not an impression; the probability-space and OT work attack
the over-detection that a side-by-side would expose; the format fork decides
what "notation" even means for us. Everything here is analysis/measurement or
output-side — none of it violates the standing rule that notation/export and
review never reach back into transcription.
