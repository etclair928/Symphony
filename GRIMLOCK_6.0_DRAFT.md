# Grimlock 6.0 — Draft / Workshop Document

**Status: BRAINSTORM ONLY. Nothing here is approved for implementation.**
This is a consolidated capture of a planning discussion. It exists so the discussion
doesn't have to be reconstructed from a chat transcript. Treat every section as
"proposed, not decided" unless marked otherwise.

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
TempoIntelligence (rewrite), DrumIntelligence (rewrite), RhythmEngine,
GrooveField, PulseField, TemporalLattice, Anechoic_Ma, QuaverIntelligence
(rewrite), PhraseIntelligence (rewrite, tentative — see below), VelocityMerge,
Ritornello (tentative — never explicitly confirmed, follows quantization
pattern).

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
PitchIntelligence (rewrite), HarmonicIntelligence (rewrite), VoiceContinuity,
ReverseGeoCrypt + helpers (kept, not rewritten), TimbreIntelligence
(confirmed — see §11 for the bigger instrument-registry idea attached to
this one).

**Validation:**
ConsensusEngine, EpistemicCouncil, Schoenberg Mirror (kept, not rewritten),
SpectralMasker (confirmed here, not Separation — its cluster-validity/
silhouette-score gating is what earns its place). Scribe's placement is
still open — see §8.

**Separation:**
Demucs, Mel-Roformer, Hybrid, Roformer, fallback.py.

**Kept as-is (no rewrite):** Schoenberg Mirror, Anechoic_Ma, ReverseGeoCrypt +
helpers, Memory (memory/arena.py, memory/guardian.py — real bug already fixed,
nested-arena scoping already wired in this session).

**Still uncategorized:** Ritornello (see above), Scribe (see §8).

---

## 3. The Rewrite List

Confirmed final list:

Pitch_Intelligence, RhythmEngine (the literal `agents/detection/rhythm_engine.py`
module — confirmed distinct from the "Rhythm_Engine" *cluster* name used
throughout this doc for the whole rhythm-side architecture), Drum_Intelligence
(**the biggest one — user's own words**), QuaverIntelligence, Tempo_Intelligence,
Harmonic_Intelligence, TimbreIntelligence (elevated from "kept, with a new
build attached" to a full rewrite, given the scope of §11's instrument-registry
work), PhraseIntelligence, AcousticIntelligence (`core/acoustic_intelligence.py`
— the canonical `AudioContract` authority; this is a Core-layer rewrite, not
agent-layer, so its blast radius is larger than the others since everything
pulls from Core; it's also the exact module whose correct `force_mono=False`
default gets overridden by `ingestion/loader.py`'s hardcoded `True`, see §7a).

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

### 7a. Ingestion forces mono before separation ever runs (major)
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

### 7b. Demucs wrapper actively degrades a working model
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

### 7c. Duplicate types in Core
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
6. **Fix-now vs. wait-for-6.0** — the bugs in §7a/7b/7c are real and
   currently live in main. Undecided whether they get addressed under 5.6.1
   immediately or as part of the 6.0 rewrite.
7. **Sequencing** — given §1's thesis (rhythm/quantization is the real
   bottleneck, pitch is comparatively solved), does Pitch_Engine's rewrite
   wait until Rhythm_Engine + Quantization are solid, or proceed in
   parallel?

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
