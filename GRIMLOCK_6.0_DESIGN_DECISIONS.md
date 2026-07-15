# Grimlock 6.0 — Design Decisions

Captured from the 2026-07-14 teardown/brainstorm session. These are the
load-bearing commitments for the 6.0 rebuild of the pitch/attribution path.
Companion to `GRIMLOCK_6.0_DRAFT.md`.

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
