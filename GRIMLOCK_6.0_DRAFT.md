# Grimlock 6.0 — Draft / Workshop Document

**Status: BRAINSTORM ONLY. Nothing here is approved for implementation.**
This is a consolidated capture of a planning discussion. It exists so the discussion
doesn't have to be reconstructed from a chat transcript. Treat every section as
"proposed, not decided" unless marked otherwise.

**Update (later session, still under 5.6.1):** Several items this document
listed as open 6.0 work have since been done directly under 5.6.1, without
waiting for a 6.0 rewrite — see the "DONE" markers throughout §2/§3/§7 and
the new §12 for the full rundown. The short version: Drum_Intelligence's
rewrite (originally called out as "the biggest one") is complete and
verified end-to-end. §7a and §7b's ingestion/Demucs bugs are fixed. A new,
previously-unbuilt pre-separation conditioning stage (`Stage_Zero.py`) is
now wired in. `ReverseGeoCrypt` went from "kept, not rewritten" to "kept,
and now actually wired in" — the pipeline stage bearing its name had never
called it. And a severe bug was found in the *validation* side of pitch
detection (Schoenberg Mirror), which complicates §1's "pitch is
comparatively solved" framing — see §12.1. A companion document,
`ARCHITECTURE_CONSOLIDATION_PLAN.md`, covers a specific defect this
progress surfaced (redundant subsystems that never cross-check each other)
in enough depth that it isn't duplicated here — §5 and §8 below now point
to it instead of re-deriving it.

---

## 1. Core Thesis (read this first — it reorders everything below)

Pitch detection is comparatively solved. Basic Pitch, CREPE, Librosa, etc. are
already good at finding *what note*. The real, unsolved problem is organizing
detected pitches into sensible rhythmic notation — *when* something happens and
*how long* it lasts, rendered as an actual sheet-music-style grid (quarter notes,
eighth notes, dotted rhythms, rests) rather than raw millisecond timestamps.

Consequence: **Rhythm_Engine is the priority, not Pitch_Engine.** Tempo, meter,
downbeat phase, groove/swing, and duration-quantization together build the
*coordinate system* everything else snaps onto. If that coordinate system is
wrong, no amount of clever pitch detection or downstream cleanup produces
readable notation — you're rounding a good pitch onto a broken grid. Nearly
every bug found in the session that produced this document (tempo
octave-blending, a downbeat phase computed and then discarded, a hardcoded 4/4
beat grid, drum classification with no notion of metrical position) was a
failure in this foundation-building step, not in pitch detection.

**Sequencing implication:** 6.0 effort should weight heavily toward
Rhythm_Engine (tempo/meter/downbeat-phase detection + the quantization layer
that turns those into notated durations) before matching effort on
Pitch_Engine.

---

## 2. Top-Level Architecture

Six areas, replacing the current flat `agents/{analysis,detection,quantization,
validation,separation}` split:

- **Pitch_Engine** — vertical: where notes land on the staff
- **Rhythm_Engine** — horizontal: when in time, how long (includes Quantization
  as its final stage, not a separate cross-cutting thing — see §4)
- **Validation** — arbitration/veto layer, sees across everything
- **Separation** — upstream stem production
- **Memory** — arena/guardian, unchanged
- **Orchestration** — pipeline sequencing, Music_Box, cancellation
- **Core** — foundation/bedrock everything else pulls from

### Cluster membership (as discussed; some entries still open — see §8)

**Rhythm_Engine:**
TempoIntelligence (rewrite — still needed, see §12.4: a real witness-
reconciliation bug was found and only diagnosed, not fixed, this later
session), **DrumIntelligence (rewrite — DONE, see §12.2)**, RhythmEngine
(rewrite — still needed, and now with sharper justification: its onset
detector is confirmed to reliably return near-empty results on real audio,
see §12.5), GrooveField, PulseField, TemporalLattice, Anechoic_Ma,
QuaverIntelligence (rewrite), PhraseIntelligence (rewrite, tentative — see
below), VelocityMerge, Ritornello (tentative — never explicitly confirmed,
follows quantization pattern).

- *Anechoic_Ma* placed here because its actual job is finding silence to
  determine note length — its resonance/room-profile scoring exists *in
  service of* telling real silence apart from reverb tail still ringing, not
  as an acoustic-profiling end in itself.
- *PhraseIntelligence* is explicitly a "for now" placement — phrase/trajectory/
  attractor-basin work is about musical form over time, rhythm-adjacent but
  not a pure timing question like Quaver is.
- *VelocityMerge*: confirmed here after clarifying its actual job. See §5 for
  the specific design principle that came out of that discussion.

**Pitch_Engine:**
PitchIntelligence (rewrite), HarmonicIntelligence (rewrite — and its scope
should now explicitly include the *validation* half, not just detection
ensembles; see §12.1), VoiceContinuity, ReverseGeoCrypt + helpers (kept, not
rewritten, **but no longer "not wired" — see §12.6: the pipeline stage
bearing its name never called it until this later session**),
TimbreIntelligence (confirmed — see §11 for the bigger instrument-registry
idea attached to this one; still unbuilt, and now a better next candidate
given family-level classification has been repeatedly verified working in
real end-to-end runs — see §12.7).

**Validation:**
ConsensusEngine, EpistemicCouncil, Schoenberg Mirror (kept, not rewritten —
**but its `_determine_verdict()` decision order and its core harmonic-
partial-detection math both had severe, previously-invisible bugs, now
fixed — see §12.1**), SpectralMasker (confirmed here, not Separation — its
cluster-validity/silhouette-score gating is what earns its place). Scribe's
placement is still open — see §8.

**Separation:**
Demucs, Mel-Roformer, Hybrid, Roformer, fallback.py (**§7b's wrapper bugs
fixed this later session — see below; a new pre-separation conditioning
phase, `Stage_Zero.py`, is now also wired in ahead of this cluster, see
§12.3, and its own cluster placement is an open question**).

**Kept as-is (no rewrite):** Schoenberg Mirror (see caveat above),
Anechoic_Ma, ReverseGeoCrypt + helpers (see caveat above), Memory
(memory/arena.py, memory/guardian.py — real bug already fixed, nested-arena
scoping already wired in this session).

**Still uncategorized:** Ritornello (see above), Scribe (see §8).

---

## 3. The Rewrite List

Confirmed final list, with completion status added:

- ~~Drum_Intelligence~~ — **DONE.** Was called out as "the biggest one" (user's
  own words) when this list was drafted; migrated into a proper
  `agents/drum_intelligence/` package (onset detection, per-family
  classification, articulation, the new interpretation/groove-position
  layer, consensus/voting, a thin orchestrator) and verified end-to-end on
  real audio. This is the single largest scope reduction to 6.0's remaining
  workload — see §12.2.
- Pitch_Intelligence — not started.
- RhythmEngine (the literal `agents/detection/rhythm_engine.py` module —
  confirmed distinct from the "Rhythm_Engine" *cluster* name used throughout
  this doc for the whole rhythm-side architecture) — not started, and now
  with concrete, fresh evidence for *why*: its onset detector reliably
  returns near-empty results on real audio (1 event on a run where
  DrumIntelligence found 31 real hits in the same song) — see §12.5.
- QuaverIntelligence — not started.
- Tempo_Intelligence — partially addressed (the anchor-and-align blending
  fix in §7e predates this update), but a live, still-unfixed witness-
  reconciliation bug was found this later session — see §12.4.
- Harmonic_Intelligence — not started as a full rewrite; a related but
  distinct validation module (Schoenberg Mirror) got major bug fixes this
  later session instead — see §12.1. Harmonic_Intelligence's own rewrite
  should absorb the same "detection is fine, the validation layer had the
  real bug" lesson.
- TimbreIntelligence (elevated from "kept, with a new build attached" to a
  full rewrite, given the scope of §11's instrument-registry work) — not
  started; see §12.7 for why it's now a stronger next candidate.
- PhraseIntelligence — not started.
- AcousticIntelligence (`core/acoustic_intelligence.py` — the canonical
  `AudioContract` authority; this is a Core-layer rewrite, not agent-layer,
  so its blast radius is larger than the others since everything pulls from
  Core) — **the specific bug motivating this entry, §7a's `force_mono=True`
  override, is fixed** (see §7a below), but that was a targeted patch to
  `ingestion/loader.py`, not the broader Core-layer rewrite this list item
  originally scoped. The rewrite itself is still open.

Common thread each rewrite should guarantee, not just inherit by accident:
- Earned confidence (Law 3) that downstream code can actually check, not a
  number that exists but nothing reads.
- Witness aggregation that's octave/ambiguity-aware by construction (see the
  anchor-and-align fix already made to TempoIntelligence this session as a
  worked example of the failure mode to design against).
- Any "which alternative won and by how much" computation gets logged, not
  discarded (ties into §7, Music_Box/Grimlock University).

### 3a. Confirmed New Builds (new capability, not a rewrite of something broken)

- **Per-instrument registry** (TimbreIntelligence + VoiceContinuity) — see
  §11 for the full description. Confirmed for the 6.0 build list: replace
  family-level range checking with a real per-instrument range registry
  (trumpet vs. trombone, violin vs. viola vs. cello, etc., not "brass"/
  "strings" as aggregates), and have VoiceContinuity hold a specific
  instrument identity per voice rather than a family label, using
  accumulated per-voice range statistics alongside TimbreIntelligence's
  classification.

---

## 4. Detection vs. Interpretation (a cross-cutting principle, not a 7th cluster)

Two layers, present inside each engine rather than as a separate module:

- **Witnesses (detection):** CRNN/madmom beat probabilities, librosa onsets,
  spectral drum classification, CREPE/BasicPitch pitch candidates. Always
  probabilistic/candidate-form, never a final decision. Answers "what raw
  signal is here."
- **Interpreters (meaning):** takes Layer 1's candidates and asks musical
  questions — what's the meter and which position is beat 1, does a
  classification make sense given where it sits rhythmically, what's the
  harmonic function of a note in context. Answers "what does the signal mean."

The technique used this session to find a real downbeat phase (chance-
corrected salience: real value vs. a baseline computed from shuffled
permutations of the same accent values) is a general-purpose interpretation
tool, not tempo-specific. It should be a shared interpretation utility any
engine can call, not something reinvented per engine or, worse, computed and
then discarded (which is literally the bug found in
`TimeSignatureDetector._downbeat_salience` this session).

---

## 5. Revised Law 2 — Clusters, Not Total Silos

Original Law 2 ("Unidirectional Integrity — data flows forward, dependencies
never look back") forced every agent into a full silo. In practice, a single
physical event (a kick hit at 1.23s) is simultaneously evidence for multiple
questions — drum type, rhythmic position, meter implication — and a pure
one-way pipe means whichever stage runs first has to guess without the
others' evidence, and later stages can never talk back.

**Proposed 6.0 version:** no direct agent-to-agent imports, ever — but a
cluster (Rhythm_Engine, Pitch_Engine) may share one explicit, typed context
object ("blackboard") that its members read and write. Cross-cluster requests
route through the Validation layer rather than a direct call between
clusters. This generalizes the already-proposed shared `MusicalContext`
(tempo/meter/key computed once, threaded by construction into every stage
that needs it) as the Rhythm cluster's blackboard, with an analogous one for
Pitch.

**This is no longer purely theoretical — see `ARCHITECTURE_CONSOLIDATION_PLAN.md`.**
A later-session audit found the concrete, present-day version of exactly
this problem: 9 independent onset-detection call sites, 4+ independent
tempo estimators, and a tempo witness (`ReverseGeoCrypt`'s real lattice
search) that reported "quintuplet, 48.8 BPM" while `tempo_analysis`'s own
consensus reported "132.2 BPM" for the *same run, same audio*, with nothing
to reconcile the two. That document's conclusion directly informs this one:
the "Validation layer" this section proposes routing cross-cluster requests
through already exists in working form —
`epistemic/epistemic_council.py`'s `EpistemicCouncil` already does
weighted-confidence contention arbitration for drum-type and duration
disputes. The concrete next step for 6.0's Rhythm_Engine cluster is
extending that same arbitration to tempo (a `resolve_tempo_contention`
alongside its existing `resolve_drum_type_contention`/
`resolve_duration_contention`), not inventing a new blackboard/broker
mechanism from scratch. Read that document before starting Rhythm_Engine's
rewrite — building a clean cluster on top of subsystems that still silently
disagree with each other just relocates the bug into new code.

**Concrete design principle from the VelocityMerge discussion (worth stating
explicitly for every rewrite):** pitch equality as a merge/grouping gate
should be *exact*, never tolerance-based. The current
`merge_different_pitches` / `pitch_tolerance_semitones=2` config option is a
real landmine — F4 (MIDI 65) to Eb4 (MIDI 63) is exactly 2 semitones, meaning
a real chromatic neighbor note sits right at the current default tolerance
boundary and could get silently absorbed into a duration-merge with a
different note. A detected pitch difference should always mean "new event,"
never "maybe the same note, average it in."

---

## 6. The Laws (found in constants.py's header, inside grimlock_refactor_context.md)

- **Law 1: Resource Survival** — "The math must never exceed the machine."
- **Law 2: Unidirectional Integrity** — "Data flows forward; dependencies
  never look back." (see §5 for proposed revision)
- **Law 3: Scribe's Truth** — "Confidence must be earned, not assumed."

These should graduate from naming convention to enforced mechanism in 6.0.
Concrete proof this matters: the separation bug found this session (see §7)
is a direct Law 3 violation — `_identity_fallback` correctly labels
hallucinated stems `Confidence.HALLUCINATION` internally, but nothing at the
stage boundary reads that confidence before the pipeline reports "succeeded."
The law was respected in the small and ignored in the aggregate. Law 1 (the
memory-pressure early-exit that triggered the same incident) is arguably
*working as intended* — the bug is downstream of it, not in it. Law 2 might
be worth an automated import-graph check in CI rather than a comment
convention.

---

## 7. Known Bugs — Track Separately From 6.0 Timing

These are real, currently-live bugs found during this session's investigation.
Whether they get fixed now under 5.6.1 or wait for 6.0 is an open decision
(see §8) — they're listed here so they don't get lost either way.

### 7a. Ingestion forces mono before separation ever runs (major) — **FIXED, later session**
`ingestion/loader.py:452` hardcodes `force_mono = True` inside
`AudioLoader.load()` — not a parameter, not overridable from
`pipeline.py`'s ingestion call. This gets passed to
`AcousticIntelligence.create_contract(force_mono=True)`, which collapses the
signal to mono before separation, before anything else. Downstream,
`audio_views.py`'s `create_mono_views()` — its own docstring says
"duplicates to stereo" — is what actually builds `AudioViews.left_channel`/
`right_channel`, which `_run_separation` stacks and hands to Demucs. **Every
separation this pipeline has run has fed Demucs two identical copies of a
mono sum, not real stereo.** Professional separation models rely significantly
on inter-channel (panning/phase) cues; none have ever been available. This is
the concrete validation of the long-standing "ingestion causes problems for
the actual models" complaint — and it's worse than suspected.

### 7b. Demucs wrapper actively degrades a working model — **FIXED, later session**
Verified directly: the real `apply_model()` htdemucs call, run in isolation,
completes correctly (~950s for a 4:25 song, all four stems present, high
confidence). The surrounding orchestration in `demucs.py`/`pipeline.py` is
what's unreliable:
- A memory-pressure early-exit bails to a fallback *before even trying*
  Demucs, based on a static RSS threshold with no relationship to whether
  Demucs itself would succeed.
- The Mel-Roformer fallback it redirects to has a real bug (shape mismatch:
  `operands could not be broadcast together with shapes (11684736,) (22822,)`),
  which falls through *again* to `_identity_fallback` (zeros for
  drums/bass/vocals, everything into "other").
- `pipeline.py`'s `_try_demucs()` accepts the whole separation as
  "succeeded" if *any one* of drums/bass/other has signal — never checks each
  stem independently.
- `SeparationTestimony` hardcodes `model_used="htdemucs", fallback_used=False`
  regardless of which of the five fallback levels actually fired.
- `streaming=True, window_samples=262144, hop_samples=131072` are passed into
  `DemucsSeparator.__init__` and stored, but `_run_demucs_with_model` never
  reads any of them — dead configuration that looks load-bearing but isn't.

Recommendation from the discussion: don't replace the model, rewrite the
wrapper. Separately worth considering: pipeline order currently runs
`tempo_analysis` (179s, full untouched mix) *before* `separation` — by the
time Demucs runs, ~193s of STFT/onset/beat-tracking work has already
accumulated memory pressure, which is plausibly what triggers the early-exit
in the first place. Running separation first (mirroring the historical
"isolate Madmom, load it first for the memory" pattern) would both give
Demucs a clean budget *and* let tempo/rhythm detection work from isolated
stems instead of the full polyphonic mix.

**What actually got fixed, later session:** the wrapper was rewritten (not
the model) — `demucs.py`'s honesty issue and the Mel-Roformer shape-mismatch
bug were both fixed, and the nested fallback ladder was restructured into
one clear tier instead of the tangled version described above. A real
MelBand-Roformer checkpoint was also added via `audio-separator`. The
memory-pressure early-exit and stage-ordering question above are real
architectural notes that weren't specifically revisited — worth confirming
they're still accurate before relying on this section for anything beyond
history.

### 7c. Duplicate types in Core — **partially fixed, later session**

`StageResult`, `EvidenceType`, and `EvidenceLease`'s duplicate definitions
(the three specific dupes listed below) were resolved this later session.
The broader pattern flagged at the end of this entry turned out to be
right, and bigger than three types: a full-codebase audit found the same
"same concept, multiple incompatible representations" problem recurring for
Tempo, Groove, Pulse, Onset detection, Resampling, and — worst of the
bunch — **five different, mutually-incompatible confidence/scoring types**
(`Confidence` enum, two differently-shaped classes both named
`ConfidenceComponents`, `GrooveConfidenceComponents`, and a bare
string-constant class also named `Confidence` in the newly-discovered
`Stage_Zero.py`). This is now tracked in its own document,
`ARCHITECTURE_CONSOLIDATION_PLAN.md`, rather than expanded here — see §5 and
§8 below.

- `StageResult` defined twice: `core/order_types.py:982` and
  `core/testimony.py:33` (the second one `Generic[T]`).
- `EvidenceType` defined twice: `core/contracts.py:762` and
  `core/feature_bundle.py:35`, as two different enums.
- `EvidenceLease` defined twice: `core/contracts.py:777` and
  `core/feature_bundle.py:306`.
- `AnechoicMaskProtocol` and `AnechoicMaProtocol` both exist in
  `protocols.py` — close enough in name to warrant confirming they're
  actually meant to be different things.
- Broader pattern: Tempo-related types spread across all four core files
  (`TempoMap`/`TempoEvent`/etc. in order_types, `TempoTestimony` in
  testimony.py, `TempoAnalysisResult` in contracts.py,
  `TempoIntelligenceProtocol` in protocols.py). Same fragmentation for
  Groove and Pulse. This is the actual mechanism behind "everything added
  required rewriting order_types" — there was never a rule for which file a
  new type belonging to an existing concept should go in.

### 7d. main.py Duration field — FIXED this session
Was printing `result.total_time_seconds` (pipeline processing time) under
the label "Duration" instead of the song's actual length. Fixed to show both,
labeled correctly.

### 7e. Tempo ensemble blending — FIXED this session
Naive weighted-arithmetic-mean blending across witnesses that disagree by
octave-related factors produced a musically meaningless result. Replaced with
anchor-and-align (highest-trust witness anchors, others get tested against a
ratio table and rescaled into the anchor's octave before blending).

### 7f. Drum kick misclassification — FIXED this session
`SpectralDrumClassifier` missed kicks masked by simultaneous cymbal/hihat
energy (centroid pushed into cymbal range even with a real kick fundamental
present). Fixed by checking low-frequency energy ratio before the centroid
ladder.

### 7g. Bass pitch range — FIXED this session
`CrepeModel` (primary bass/master detector) had no MIDI range validation at
all, unlike `LibrosaModel`'s fallback path. Added.

### 7h. Beat-grid meter/phase — FIXED this session
`build_beat_grid` hardcoded 4/4 and assumed `beat_times[0]` is always the
downbeat — wrong for any non-4/4 meter and for anacrusis/pickup measures.
Now uses the real detected numerator and the winning downbeat phase from
`TimeSignatureDetector` (which itself now returns that phase instead of
discarding it).

---

## 8. Open Questions — Still Unresolved

1. **Scribe placement** — into Validation alongside ConsensusEngine/
   EpistemicCouncil/Schoenberg Mirror (it's literally "Law 3: Scribe's
   Truth"), or stays a distinct orchestration-level gate?
2. **Ritornello / TemporalLattice** — assumed part of Rhythm_Engine's
   quantization stage, never explicitly confirmed one by one.
3. **Music_Box schema freeze** — does it need versioning/freezing now, given
   Grimlock University depends on it, before upstream pipeline internals
   change further?
4. **Window_Pane** — real sketch recovered from Grimlock 4.7 (see §9). Open:
   should it share an event schema with Music_Box (two sinks: live/ephemeral
   vs. persistent/forensic), does the pipeline move toward async anywhere
   (the original sketch assumed it), and is "JSON replay + webhook" the
   actual end goal or a stepping stone?
5. **Web interface** — broken since the 5.0 jump, never fixed since. Is
   rebuilding it in scope for 6.0, or does it stay parked?
6. **Fix-now vs. wait-for-6.0** — ~~undecided~~ **empirically answered for
   §7a/7b/7c: fix-now worked.** All three were fixed directly under 5.6.1,
   verified with real e2e runs, without waiting for a 6.0 rewrite —
   likewise Drum_Intelligence's full rewrite (§3, originally "the biggest
   one"), the Omnizart integration bug, Schoenberg Mirror's harmonic-math
   bug, and `ReverseGeoCrypt`'s wiring gap (§12). The pattern that made this
   work safely: one change at a time, real e2e verification against a known
   baseline after each, never batching multiple architectural changes
   together (see `ARCHITECTURE_CONSOLIDATION_PLAN.md` §4 for why that
   discipline matters — several of these fixes surfaced a latent bug the
   moment previously-dead code actually ran for the first time). This
   doesn't mean *everything* should be fixed now instead of in 6.0 — it
   means well-scoped, single-issue bugs (even "major" ones) are fair game
   for 5.6.1, and 6.0 should be reserved for genuine structural/cluster
   rewrites (Rhythm_Engine's internals, the blackboard/consensus pattern,
   the instrument registry) that can't be done as an isolated patch.
7. **Sequencing** — given §1's thesis (rhythm/quantization is the real
   bottleneck, pitch is comparatively solved), does Pitch_Engine's rewrite
   wait until Rhythm_Engine + Quantization are solid, or proceed in
   parallel? **Refined, not resolved:** §12.1's Schoenberg Mirror findings
   mean "pitch is comparatively solved" needs a split — pitch/harmonic
   *detection* (Basic Pitch, CREPE, the ensemble) is comparatively solved;
   pitch/harmonic *validation* (is a detected pitch musically legitimate,
   Schoenberg Mirror's whole job) just had a bug severe enough to
   misclassify nearly all tonal content as percussion, undetected across
   many prior sessions. Whatever Pitch_Engine's rewrite scope ends up being,
   it needs to explicitly cover the validation half, not just the detection
   ensemble — that's where the actual bug was.
8. **Onset/tempo/resampling consolidation sequencing** (new, from
   `ARCHITECTURE_CONSOLIDATION_PLAN.md`) — does that consolidation happen
   *before* Rhythm_Engine's 6.0 rewrite (clean up the foundation first), as
   part of it (rewrite and consolidate in the same pass), or after (get the
   cluster boundaries right first, worry about internal duplication once
   the shape is settled)? Leaning toward *before or as part of*, given §5's
   point that building a new cluster on top of still-disagreeing
   subsystems just relocates the bug — but not decided.

---

## 9. Window_Pane (recovered sketch, Grimlock 4.7)

A passive, non-blocking observability layer — event emission
(`emit_witness`, `emit_onset`, `emit_drum_hit`, `emit_note`,
`emit_contradiction`, `emit_tempo_verdict`), a `with pane.stage(...)` context
manager for per-stage timing, background memory sampling thread, optional
fire-and-forget webhook streaming, JSON replay dump on `stop()`.

**Relationship to Music_Box (proposed):** Music_Box = forensic, persistent,
append-only, feeds Grimlock University, needs a stable schema. Window_Pane =
live, ephemeral, streaming, feeds a human watching a run happen right now.
Same underlying event shape, different sink + retention policy, rather than
every stage getting instrumented twice by hand.

**Concerns with the sketch as a direct foundation:**
1. One OS thread per webhook send (`threading.Thread` per event) — fine for
   occasional events, not fine at per-note volume (500-1000+ notes/song).
   Needs a bounded worker pool or queue.
2. Async/sync mismatch — docstring example uses `await` inside
   `with pane.stage(...)`, module imports `asyncio` but never uses it. The
   current Grimlock pipeline is fully synchronous. Needs an explicit
   decision either way.
3. Unbounded growth — `_events`/`_memory_samples` grow for the life of the
   run with no cap. Given Law 1 is a proven real constraint this session, a
   telemetry layer should state its own memory budget.
4. No thread-safety on the shared lists between the sampling thread and the
   main pipeline thread — relies on CPython GIL implementation details
   rather than an explicit design.

---

## 10. Music_Box → "Grimlock University" (the other project)

Clarified purpose: Music_Box isn't just a debug log, it's meant to feed a
separate project ("Grimlock University") that captures model decisions, uses
that data to tune variables, and eventually plugs in machine learning so the
system self-corrects. The manual debugging done this session (user provides
ground truth, I trace code to find where decisions diverged) is the slow,
human-bottlenecked version of that same loop.

**What data to capture (the concrete, scoped answer — not designing the ML
project itself):**

Every event needs: `session_id` + `audio_hash`, `stage` + `timestamp_ms`, the
actual parameter/threshold values active when the decision was made, and
critically — **all alternatives considered and their scores, not just the
winner.** This is the single biggest current gap: e.g. the downbeat-phase
search tests 6 phase hypotheses and only the winning score survives; the
runner-up's margin (how close the second-best phase came) is exactly what
tells you how confident a decision really was, and it's discarded today.
Also: downstream fate — did this decision later get vetoed/overridden by
Scribe or consensus? That link doesn't exist today either.

Per-stage specifics discussed: Separation (which fallback level *actually*
fired, per-stem RMS/confidence, memory pressure at decision time, raw
exception text on failure — this alone would have turned tonight's hour of
Demucs investigation into a 10-second log read); Tempo/meter (every
witness's raw values pre-blending, every octave-ratio candidate's score, all
8 time-signature numerator scores, winning phase + runner-up margin); Drums
(raw features feeding classification, which branch fired, NMF/HFC contention
outcomes); Pitch/Harmonic (each ensemble witness's raw output pre-merge,
octave-correction evidence, harmonic validation's actual numbers not just
pass/fail, range-rejection counts); Quantization (raw vs. quantized time per
note, winning grid hypothesis and its margin).

**The sheet-music extension:** ground truth could eventually be an actual
symbolic score (MusicXML, via `music21`) rather than a text description.
Given that, an alignment/diff layer between Grimlock's transcription and the
reference score becomes possible — per-note matched/missed/hallucinated/
wrong-pitch/wrong-rhythm, aggregated per track. Combined with the
"alternatives considered" logging above, this turns a discarded runner-up
score into an actual loss signal: "the system's top choice was X, the
reference says the true answer was Y, which the system also considered and
scored close behind." That's what makes "continually auto-correct" tractable
rather than aspirational. Sourcing actual sheet music for training songs is
explicitly out of scope for Grimlock (6.0) — that's the other project's
problem.

**A capability catalog** (generated from a parameter registry + this
schema, not hand-maintained) would also directly address "I can't always
remember what these models can do" — a discoverability problem, not a memory
problem, solvable with the same infrastructure.

---

## 10a. Process / Discipline (not architecture, but binding)

- **Nothing lands half-wired.** A new module's PR includes its wiring into
  the pipeline, or it doesn't land yet. No more "built in month 3, discovered
  unused in month 8."
- **Only Claude and DeepSeek implement code.** Other AI tools are for
  research/opinions/talking out ideas, never direct implementation. (The web
  interface breaking during the 5.0 jump and never being fixed since is the
  reason this rule exists — treat it as a real scar, not an abstract
  preference.)

---

## 11. TimbreIntelligence — Instrument Registry (CONFIRMED for 6.0 build — see §3a)

Currently: family-group classification only (`InstrumentFamily` enum —
piano/strings/brass/woodwind/voice/plucked/percussion/unknown — plus
`InstrumentRangeValidator`'s per-*family* MIDI range table, e.g. `STRINGS:
(55.0, 4200.0)Hz` in `spectral_masker.py`'s `FAMILY_RANGES`). This was
explicitly the "quicker method" chosen over a bigger idea (originally
Claude's suggestion, not implemented) discussed while editing VoiceContinuity
and TimbreIntelligence in a prior session. No trace of the bigger version
exists in the codebase — confirmed dead-end search for "registry" across the
repo, and no other session transcripts were available to recover it from
(`list_sessions` returned none). Recovered by description instead.

**The actual idea:**
- A registry of *individual instruments* — trumpet and trombone as distinct
  entries, not both folded into "brass"; violin/viola/cello/double bass as
  distinct entries, not all "strings." Each entry carries its own real
  playable range (e.g. a violin's lowest note is its open G string, G3 — a
  detected note below that within a "violin" voice is a strong signal the
  true instrument is actually viola or cello, not evidence the violin somehow
  played below its physical floor).
- VoiceContinuity should track a specific instrument identity for a line, not
  just a family, and hold it consistently across the voice's lifetime — using
  TimbreIntelligence's classification as supporting evidence, with the
  per-instrument range table as a strong constraint. A note falling outside
  the currently-assigned instrument's true range becomes evidence to revise
  the assignment (either "this note isn't part of this voice" or "this voice
  was misidentified"), the same soft-penalty-as-evidence pattern
  `InstrumentRangeValidator` already applies at the coarser family level —
  just at individual-instrument grain.

**Why this is a bigger lift than family classification, and a practical path
through it:** distinguishing trumpet from trombone (or violin from viola)
by timbre alone is a genuinely harder classification problem than
family-level classification — instruments within a family share most of
their timbral character and differ mainly in register and subtler spectral
detail. Rather than requiring a much stronger per-instrument timbral
classifier from day one, the practical route is combining the *already-
working* coarse family classification with **range statistics accumulated
over a voice's whole tracked lifetime** (which VoiceContinuity already
computes) — e.g. "family=brass, and this voice has never gone below F#3 or
above D6 across the whole song" is strong evidence for trumpet over trombone
or tuba, purely from accumulated range, without needing a much better
timbral model to get there.

**What this needs to exist:** an actual per-instrument range reference table
(standard orchestration/band reference data — well-documented, not a novel
research problem) to replace/extend the current family-level `FAMILY_RANGES`
table. Open questions still worth answering: does this tie into MIDI
program/patch assignment at export time (picking the right General MIDI
instrument, not just internal bookkeeping), and how deep does the registry
go for non-orchestral/electronic instruments (specific synth patches,
electric vs. acoustic guitar, etc.) where there's no fixed physical range to
anchor against.

---

## 12. Later-Session Progress Log (still under 5.6.1, referenced throughout above)

This section exists so the cross-references in §2/§3/§5/§7/§8 above have
somewhere concrete to point to, without turning those sections into a
changelog. Ordered roughly by how much they change 6.0's remaining scope.

### 12.1 Schoenberg Mirror's harmonic-legitimacy bugs (complicates §1's thesis)

Investigating a user complaint ("no piano/melody/vocal instruments showing
up") found that `schoenberg_mirror.py` — the module whose entire job is
judging whether a detected pitch is harmonically legitimate — had two
compounding bugs that together made `tonal` verdicts almost unreachable:

1. `_determine_verdict()` checked ZCR-based percussiveness *before* ever
   consulting harmonic-series evidence, so any tonal note with a hard onset
   (piano hammer, plucked string) got short-circuited into `PERCUSSION`
   regardless of how clean its harmonics were.
2. The deeper bug: `HarmonicSeriesAnalyzer._detect_partials` used raw
   FFT-bin frequencies with no sub-bin interpolation, plus a
   neighbor-prominence gate that ordinary spectral leakage defeats. Tested
   against a synthetic, mathematically ideal 6-harmonic tone: it detected
   only 3 of 8 partials and scored 0.36 confidence against a 0.7 threshold
   — meaning **no signal, however clean, could pass.** Fixed with
   local-max search + parabolic interpolation; the same synthetic tone then
   scored 0.98.

Real-audio effect: `tonal` count went from 0 to a real, substantial
fraction of surviving notes across multiple songs. This is the concrete
evidence behind §8's refined question 7 — pitch *detection* wasn't the
problem, pitch *validation* was, and it was invisible until someone asked
"why does nothing sound tonal" and traced it all the way to FFT
bin-quantization math.

### 12.2 Drum_Intelligence rewrite — complete

The rewrite §3 called "the biggest one" is done: migrated from
`agents/detection/{drum_intelligence.py,drum_coprocessor.py}` into a
package (`agents/drum_intelligence/`: onset detection, per-family
classification, articulation, a new interpretation/groove-position layer,
consensus/voting, a thin orchestrator), verified end-to-end on real audio.
A follow-on bug was found and fixed afterward: the consensus layer's
same-type-hit deduplication used a fixed 25ms bucket that missed
near-duplicates straddling a bucket boundary, producing physically
impossible kick "hits" 23-41ms apart, and gave every drum type the same
short refractory period regardless of natural decay time, causing a
sustained cymbal wash to get chopped into a dense run of separate `CRASH`
events (7 in 2.4 seconds). Fixed with a per-drum-type refractory period
(crash/ride/open-hihat get a much longer floor than kick/snare/closed-hihat).

### 12.3 Stage_Zero.py — a previously-unbuilt pre-separation conditioner, now wired in

Not originally on this document's radar at all: `orchestration/Stage_Zero.py`
is a complete, ~1200-line pre-separation audio conditioner (phase alignment
via GCC-PHAT, NMF noise-floor/reverb-tail suppression, mid-side stereo
normalization, adaptive HPSS, chord-prior harmonic masking, spectral
normalization, source-count estimation, iterative peel-back) that had
**zero callers anywhere in the repo** — fully built, never executed. Now
wired into `_run_separation`, running before Demucs/heuristic/STFT
separation.

Two real bugs surfaced the moment it actually ran for the first time (a
recurring pattern this session — see `ARCHITECTURE_CONSOLIDATION_PLAN.md`
§4): Unicode arrow characters crashing on Windows console encoding, and an
unconditional final peak-normalization that forced every output to exactly
1.0 peak regardless of the source's natural level, discarding all headroom
and shifting the same amplitude-dependent thresholds (crest factor,
low-frequency energy ratio) real detectors downstream key off of. Both
fixed.

**Open, not yet resolved:** its own config docstring says "sensible
defaults for jazz small-ensemble recordings" — this project's real test
material is hip-hop/pop, a very different acoustic profile. A/B testing on
real audio showed roughly doubled runtime with mixed, inconclusive quality
deltas (some metrics up, some down, all within normal run-to-run variance).
Not yet decided: keep it enabled by default, gate it behind a flag pending
more testing, or fold its more universally-safe pieces (phase alignment is
genre-agnostic; harmonic masking and peel-back are the more
genre-specific-feeling ones) into the `AcousticIntelligence` Core rewrite
instead of leaving it as a separate bolt-on stage. Also open: which cluster
it belongs to (see §2's Separation entry).

### 12.4 TempoIntelligence — a live, unfixed witness-reconciliation bug

On *You Say God Says.mp3*, independent verification (librosa's own beat
tracker plus raw autocorrelation, bypassing this project's code) converged
strongly on **~123 BPM**. The pipeline's final answer was **74.7 BPM** — the
*weakest* candidate in the independent re-check, not the strongest, even
though `tempo_analysis`'s own internal witness (`tempo_intelligence`) had
separately reported 90.9 BPM, much closer to the independently-verified
value. This has been diagnosed, not fixed — see
`ARCHITECTURE_CONSOLIDATION_PLAN.md` §3.2 for the concrete next debugging
step (log each witness's raw value before blending, on this specific song,
before touching anything else).

### 12.5 RhythmEngine's onset detector — confirmed broken on real audio

`agents/detection/rhythm_engine.py`'s onset detector (feeding
`self._rhythm_notes`) reliably returns near-empty results on real audio —
1 event on a *Hopeful.mp3* run where `DrumIntelligence` found 31 real hits
in the same clip; earlier in this project's history, 0-4 events on songs
where `DrumIntelligence` found 2000+. `groove_analysis` had already worked
around this by switching to `DrumIntelligence`'s hit list instead. The
`reverse_geo_crypt` pipeline stage (§12.6) had the identical bug and needed
the identical fix. This is now solid, repeated evidence that
`RhythmEngine`'s own onset detection — not just its tempo estimate (§3) —
needs the rewrite, not just a workaround at each call site.

### 12.6 ReverseGeoCrypt — wired in for the first time, plus a real result

The pipeline stage named `reverse_geo_crypt` never called the actual
`ReverseGeoCrypt` class described in §2's "kept, not rewritten" line — it
ran a hand-rolled substitute instead (round each inter-onset interval to
the nearest 10ms, take the most common value). The real class is fully
built, sophisticated (ratio-family clustering, phase-alignment search,
entropy-based rubato detection), and was already being used elsewhere in
the same file as one `tempo_analysis` witness — just never by the stage
bearing its name. Now wired in with real downbeat anchors (see §12.5 for
why it needed the same onset-source fix `groove_analysis` already needed).
Real result on *Hopeful.mp3*: reported "quintuplet" ratio family,
independently agreeing with `tempo_analysis`'s own separate
"octave_correction=5/4 (quintuplet compression)" finding for the same song
— two independent systems converging on an unusual rhythmic characteristic
is a good sign the real algorithm produces genuine signal. It's also the
source of §12.4's tempo-disagreement discovery — wiring in a previously-dead
real system surfaced a real, previously-invisible bug elsewhere, which is
becoming this session's most consistent pattern (see
`ARCHITECTURE_CONSOLIDATION_PLAN.md` §4).

### 12.7 TimbreIntelligence's family-level classification — repeatedly verified working

Not a bug fix, but relevant to §11's prioritization: the *existing*
family-level classification (`InstrumentFamily` enum — piano/strings/brass/
woodwind/percussion) has now been exercised repeatedly in real end-to-end
runs and produced plausible, real-sounding instrument-family breakdowns
(e.g. `{'strings': 11, 'brass': 12, 'woodwind': 46, 'piano': 6}` on one run).
That's a stronger foundation to build the individual-instrument registry
(§11) on top of than existed when this document was first drafted — the
coarse classification it needs to combine with accumulated range statistics
is proven, not hypothetical.

### 12.8 Omnizart — case-sensitivity bug, contributed zero votes all session

`epistemic_council.py`'s Omnizart integration passed `model_path="pop"`
(lowercase) to select the pop-music checkpoint; the real API only matches
an exact, capitalized dict key (`'Pop'`). Every single call failed
instantly with a caught, warning-level-only exception — Omnizart
contributed zero votes to the pitch-detection ensemble for the entire
session despite being reported as "installed and verified." Fixed the case,
but bare-model testing with the fix applied (isolated stem, full mix, and
the entire untruncated song) still produced zero real notes — likely a
missing `Spleeter` dependency degrading its own feature extraction, or
checkpoint miscalibration for this project's material. Disabled by default
(`CouncilConfig.use_omnizart = False`) since the fix makes it actually run
its full, currently-fruitless inference (~2 min/call) instead of failing
fast — leaving it enabled by default would have reintroduced real runtime
cost for zero benefit. Left as a real, correct bug fix that just doesn't
currently pay for itself on this project's audio.
