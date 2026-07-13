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

**Update (later still, same document, still 5.6.1):** A user-reported
"bass stem sounds cluttered" complaint turned into a real forensic-evidence
audit (§12.9-12.10) that found the same "computed but silently discarded"
pattern one more time — in *three* separate places this time — plus two
genuinely new findings that change 6.0's scope: TimbreIntelligence
classifies every note from the full mix, not the isolated stem it actually
came from (§12.11), and vocals — already separated out by Demucs/Mel-
Roformer on every single run — are thrown away immediately after
separation and never get their own pitch-detection pass at all (§12.13).
The latter also plausibly explains a related finding, the "other" stem's
notes collapsing into one implausibly wide VoiceContinuity voice (§12.12).

**Update (later session, still under 5.6.1):** Two of §8's open questions
just got resolved directly, without waiting for 6.0. Question 2 (Ritornello's
placement/purpose) is answered: it's been restored to its pre-5.0 design —
not part of quantization at all, but a distinct post-quantization "musical
coherence repair pass" (surgical per-measure re-detection via Basic Pitch,
the `CorrectionScribe` "repair and prove" philosophy) — see §12.14. And
§7c/§3.5's "five incompatible confidence/scoring representations" finding
is closed: the two genuine name collisions were renamed
(`DrumConfidenceComponents`, `StageZeroConfidence`), the rest documented as
deliberate specializations rather than forced into one type — see §12.15,
which also covers `MusicalFindingsMap` being rebuilt to actually be the
single source of truth §5 already assumed it was, instead of a
scalar-copy mirror of testimony objects computed elsewhere.

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
below), VelocityMerge, **Ritornello (placement confirmed — see §12.14: not
part of quantization, a distinct post-quantization repair-pass stage that
runs after VelocityMerge/TemporalLattice)**.

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
ensembles; see §12.1), VoiceContinuity (kept, not rewritten — **but see
§12.12: the "other" stem's notes collapsed into a single voice spanning
nearly 3 octaves in one real run, not yet root-caused**), ReverseGeoCrypt +
helpers (kept, not rewritten, **but no longer "not wired" — see §12.6: the
pipeline stage bearing its name never called it until this later
session**), TimbreIntelligence (confirmed — see §11 for the bigger
instrument-registry idea attached to this one; still unbuilt, and now a
better next candidate given family-level classification has been
repeatedly verified working in real end-to-end runs — see §12.7 — **but
see §12.11: it classifies every note from the full mix, not the isolated
stem the note actually came from, which should be fixed before leaning on
its output for any cross-stem bleed diagnosis**). **New in this round:
Vocals has no pitch-detection stage at all — see §12.13, and §3a for the
confirmed-new-build entry.**

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
- **Vocals pitch-detection stage** — see §12.13 for the full finding.
  Confirmed for the build list: add `vocals_stereo` to `AudioViews` (the
  separator already computes this stem and it is currently discarded), add
  a `_run_pitch_detection_vocals` stage mirroring the existing bass stage
  (same EpistemicCouncil/VoiceContinuity/TimbreIntelligence/consensus
  machinery, notes tagged `stem:vocals`), with its own witness-confidence
  tuning for vocal-specific pitch behavior (vibrato, portamento/glides,
  breathiness) rather than reusing the bass config verbatim. Lyrics
  (word-level transcription) is a deliberately separate, optional bolt-on —
  see §12.13 and §8's new open question on ASR scope.

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
§8 below. **The confidence-type item itself is now closed — see §12.15**:
the two real name collisions were renamed, the rest documented as
deliberate, justified specializations rather than forced into one type.

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

### 7i. instrument_family/reasoning_chain never reaching exported notes — FIXED, later session (three compounding bugs — see §12.9)
A bass-stem-clutter investigation found the exported JSON's `instrument_family`
was `None` on every single note, despite the classification being real and
computed correctly upstream. Three independent bugs stacked on top of each
other: (1) `NoteEvent` has no `instrument_family` attribute at all —
`json_writer.py`'s and `main.py`'s `hasattr()` checks always silently
failed; the real data only ever existed as a `"family:X"` tag inside
`reasoning_chain`. (2) Even reading the tag correctly, it still never
arrived — `pipeline.py`'s cluster/family-tagging helpers built tagged
copies via `dataclasses.replace()` that only ever lived inside
`VoiceContinuity`'s `Voice` objects, never written back to the canonical
`self._pitch_notes_bass/master/other` lists everything downstream actually
reads. (3) The deepest layer: `velocity_merge.py`'s
`_build_reasoning_chain()` rebuilt every merged note's `reasoning_chain`
from scratch (just a generic "Merged N notes..." summary), discarding
every upstream tag — stem, family, cluster, bass-provenance, council-vote —
for every note that passes through a merge, which is most notes. All three
fixed; verified end-to-end that real `instrument_family`/`reasoning_chain`
data now survives to exported notes for the first time. See §12.9 for the
fuller writeup and why this is the same architectural smell as §12.1-12.8.

---

## 8. Open Questions — Still Unresolved

1. **Scribe placement** — into Validation alongside ConsensusEngine/
   EpistemicCouncil/Schoenberg Mirror (it's literally "Law 3: Scribe's
   Truth"), or stays a distinct orchestration-level gate?
2. ~~Ritornello / TemporalLattice — assumed part of Rhythm_Engine's
   quantization stage, never explicitly confirmed one by one.~~ **Resolved,
   later session:** they're siblings, not one stage. `TemporalLattice`
   is the pure snap-to-grid quantizer; `Ritornello` is a distinct
   post-quantization "musical coherence repair pass" (restored to its
   pre-5.0 design intent) that runs after it and `VelocityMerge`, doing
   surgical per-measure re-detection against isolated stem audio rather
   than any further quantizing. See §12.14.
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
9. **Is lyric transcription (ASR) in scope for Grimlock at all?** (new,
   from §12.13) Vocals pitch-detection is a natural extension of work this
   project already does — same `NoteEvent`/consensus/voice machinery, just
   one more stem. Word-level lyric transcription is a categorically
   different problem (speech recognition, not music transcription, a
   different model family entirely) that happens to become *easy* to bolt
   on once vocals are isolated. Worth deciding explicitly whether that's a
   deliberate scope expansion for 6.0, a "someday, clearly separate
   feature" parked next to Window_Pane/the web interface (§8.4-8.5), or
   out of scope for this project entirely — rather than drifting into it
   by default just because the isolated audio will already be sitting
   there.
10. **Does §12.12's "other"-stem voice collapse need its own investigation,
    or does it resolve once vocals are separated out?** (new, from
    §12.12/§12.13) If a meaningful share of "other"'s ~3-octave single-voice
    problem is really an un-separated vocal line sitting on top of
    piano/guitar/pads, wiring in the vocals stem (§12.13) first and
    re-measuring before investing in a dedicated VoiceContinuity fix would
    avoid solving a problem that partially self-resolves.

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

### 12.9 Forensic per-note evidence discarded at three separate merge points — FIXED

A user complaint ("bass stem sounds cluttered, might be piano bleed")
needed real per-note `instrument_family`/`reasoning_chain` data to answer
with evidence instead of speculation. Every exported note had
`instrument_family: None` and many were missing even their `stem:` tag.
Three independent bugs stacked on each other, each hiding the next one
until the previous was fixed:

1. `NoteEvent` has no `instrument_family` attribute at all.
   `export/json_writer.py`'s "complete" serializer and `main.py`'s JSON
   export both gated on `hasattr(note, 'instrument_family')`, which is
   always `False` — the real classification only ever existed as a
   `"family:X"` string inside `reasoning_chain`. Both fixed to parse the
   tag from there.
2. Even reading the tag correctly, it still never arrived on any exported
   note. `pipeline.py`'s `_tag_notes_with_cluster_ids`/
   `_tag_notes_with_instrument_family` build tagged copies via
   `dataclasses.replace()`, but those copies only ever lived inside
   `VoiceContinuity`'s `Voice` objects (`self._voices`) — never written
   back to `self._pitch_notes_bass/master/other`, which is what every
   downstream stage and the final export actually reads. Fixed by adding
   `_propagate_voice_reasoning_tags()`, mirroring how
   `_tag_notes_with_phrase_id` already does this same back-write for
   `phrase_id`.
3. The deepest layer, and the one that made the other two fixes still
   look broken at first: `agents/quantization/velocity_merge.py`'s
   `_build_reasoning_chain()` rebuilt every merged note's
   `reasoning_chain` **from scratch** — just `"Merged N notes using
   {strategy}"` plus a velocity list — discarding every tag any of the
   merged notes carried in (`stem:`, `family:`, `cluster:`, `bass:*`
   provenance, `council:*` witness-agreement). Since most notes pass
   through a merge on their way to export, this alone was enough to wipe
   out the first two fixes' results almost entirely. Fixed to collect and
   dedupe every upstream tag from the whole merge group first, then
   append the merge-summary lines.

All three verified end-to-end on *Hopeful.mp3*: before any of these three
fixes, 52 of 146 exported notes carried no `stem:` tag at all and only 10
carried a real `instrument_family`. After all three, a fresh run showed
**0 of 151 notes missing a stem tag** and 25 carrying a real
`instrument_family` (13 woodwind/4 piano/2 brass/1 percussion on the bass
stem; 4 woodwind/1 brass on "other") — real provenance now survives to
actually-exported notes for the first time. **Not yet checked**: `_run_consensus`'s own
cross-stem grouping (pipeline.py, ~line 3349) has the identical
one-survivor-keeps-its-tags-the-rest-lose-theirs shape when it merges a
same-pitch/same-time group down to a single `base` note — lower
suspected impact (the `base` note's own now-preserved tags plus the added
`consensus:corroborated_by_X` tag still carry real information), but not
independently verified the way the three above were.

**Why this belongs in a 6.0 document, not just a bug list:** this is the
same failure shape as §12.1, §12.3, and §12.6 — real evidence computed
correctly by an upstream stage, silently discarded at the next
merge/replace boundary, invisible until someone asks a specific question
of the data and traces why the answer is empty. Three instances of it
compounding on the exact same field is a strong signal that "does this
`replace()`/rebuild preserve every accumulator field (`reasoning_chain`,
`consensus_votes`, `veto_attempts`, ...) by construction" needs to be a
named invariant any 6.0 rewrite enforces (ties into §6's Law 3 discussion)
rather than something each merge site has to remember to get right on its
own.

### 12.10 Bass-stem clutter — diagnosed with real evidence, filter proposed but not yet implemented

With §12.9's fix in place, real per-note data on *Hopeful.mp3*'s bass stem
showed a clean, specific split rather than uniform noise:

- Notes in MIDI register 30-37 (Bb1-C#3): several carry
  `council:SPICE+BP_LOCKED` (both independent pitch witnesses agree) with
  confidence 0.75-1.00. This is the real bass line.
- Notes in MIDI register 48-51 (C3-D#3) — the exact register where a
  piano's left hand overlaps a bass line — were, without exception,
  single-witness (`council:single_librosa`) or weak 2-witness
  corroboration, **never** SPICE+BasicPitch-locked, with confidence as low
  as 0.21-0.47. One of these same notes was also the one TimbreIntelligence
  classified as "woodwind" rather than anything bass-like (caveat: see
  §12.11 on why that specific data point is weaker evidence than it looks).

This is a concrete, actionable fingerprint: weak pitch-detection
corroboration clusters specifically in the piano-overlap register, not
randomly across the whole bass range. **Proposed, not yet implemented:**
gate bass-stem note acceptance above ~pitch 45 on 2+-witness agreement
(ideally the SPICE+BasicPitch lock tier) rather than accepting a single
`librosa`-only witness at face value in that specific overlap-prone zone.

### 12.11 TimbreIntelligence classifies every note from the full mix, not its own isolated stem

`pipeline.py`'s `_run_timbre_analysis` feeds every note's classification
window from `self._audio_views.mono_sum_normalized` — the whole song —
regardless of which stem the note's pitch actually came from. A
"woodwind" verdict on a bass-stem note therefore can't cleanly distinguish
"this is real stem-separation bleed" from "something else was simply
loud in the full mix at that exact instant, and the classifier was never
given the isolated audio needed to tell the difference." This directly
weakens §12.10's bleed diagnosis (the one classified note in the suspect
cluster came back "woodwind," but that verdict is confounded by full-mix
scope) and, more broadly, undermines any future use of TimbreIntelligence
for cross-stem bleed detection or the §11 instrument-registry work.

**The fix already exists as a pattern in this same codebase**: §12.1's
Schoenberg Mirror fix already solved the identical problem for harmonic
validation — routing each note's audit to the isolated stem its own
`stem:` reasoning_chain tag names, falling back to the full mix only when
the tag is missing. TimbreIntelligence should get the same treatment.
Not yet implemented.

### 12.12 "Other"-stem notes collapse into one implausibly wide VoiceContinuity voice

On the same *Hopeful.mp3* run, all 66 "other"-stem notes (piano/guitar/
synth content — whatever isn't bass or drums) landed in a single
VoiceContinuity voice spanning MIDI pitch 42-79 — nearly 3 octaves —
versus bass's much cleaner split (one dominant 21-note voice plus a
handful of legitimate single-note outlier segments elsewhere in time).
Average confidence for the "other" voice was also notably lower (0.43)
than bass's dominant voice (0.59). Not yet root-caused: could be genuine
wide/register-jumping content in a real polyphonic accompaniment stem, or
a sign VoiceContinuity's clustering isn't splitting multiple truly
independent simultaneous lines within "other" the way it should. See
§12.13 and §8's new open question 10 — since "other" currently has to
absorb an entire un-separated vocal line on top of whatever instruments
are actually in it, wiring vocals out first and re-measuring is the
cheaper next step before assuming this needs its own dedicated fix.

**Reproduced on a second, independent run** (after §12.9's velocity_merge
fix landed, so a different note count overall): "other" again put every
single note — 75 of 75 — into one voice, this time spanning pitch 38-79.
The same run also showed bass's own dominant voice widen noticeably (34-70,
versus 34-51 on the first run) — the two runs aren't a controlled A/B
(raw note counts differed, 76 vs. 28 for bass), so this isn't strong
evidence on its own, but it's a reason to widen the suspicion beyond
"other" alone: worth checking whether VoiceContinuity's collapse tendency
shows up on bass too under the right conditions, not just on the stem
that happens to be the most polyphonic.

### 12.13 Vocals are separated by the model on every run and then thrown away — no vocal pitch or lyric pipeline exists

Investigating where a real vocal-line detector should live surfaced a
concrete, present-day gap: Demucs/Mel-Roformer's standard separation
output already includes a real 4th stem —
`agents/separation/demucs.py`'s own `STEM_ORDER = ["drums", "bass",
"other", "vocals"]`, and `StemType.VOCALS` already exists in
`core/order_types.py`. But `pipeline.py`'s `_apply_separation_result`
(~line 1317) only ever pulls `StemType.DRUMS`/`BASS`/`OTHER` off the
separator's result, and `AudioViews` has no `vocals_stereo` field at all
to put a fourth one in. **The vocals stem is computed by the separator on
every single run and discarded immediately after, every time.** Whatever
vocal content exists today gets bundled into "other" alongside piano/
guitar/synth, completely undifferentiated — plausibly a real contributor
to §12.12's finding.

**Confirmed for the build list** (§3a): add `vocals_stereo` to
`AudioViews` (trivial — the array already exists in the separator's
result, just needs to stop being dropped), add a
`_run_pitch_detection_vocals` stage that mirrors the existing bass stage
exactly — same `EpistemicCouncil`/`VoiceContinuity`/`TimbreIntelligence`/
consensus machinery, notes tagged `"stem:vocals"` — rather than building
anything new. The one real difference from bass: vocal pitch behaves
differently from instrumental pitch (vibrato, portamento/glides,
non-tempered bends, breathiness), so the witness config (confidence
floors, which models get weighted highest) likely needs its own tuning
pass rather than reusing bass's config verbatim. Structurally, though,
this is a sibling stage, not a parallel system.

**Lyrics are a separate problem, deliberately not folded into this.**
Word-level lyric transcription is automatic speech recognition, not music
transcription — a different model family (Whisper/faster-whisper, or
whisperX for word-level alignment) with no real relationship to
`NoteEvent`/pitch detection. Recommended shape if/when this is built: an
independent, optional stage that reads the same isolated vocal audio
`vocals_stereo` would provide and produces its own word-timeline output
(word + start_ms + end_ms), exported alongside the note data but not
merged into it. See §8's new open question 9 on whether this should be in
scope for 6.0 at all, versus parked next to Window_Pane/the web interface
as a deliberately separate feature.

### 12.14 Ritornello restored to its pre-5.0 design intent — a repair pass, not a quantizer

The version of `agents/quantization/ritornello.py` this document's §2 had
marked "tentative/never confirmed" turned out to be a scope drift: 5.0-era
Ritornello had quietly become a second quantizer duplicating
`TemporalLattice`'s job, imported by `agents/quantization/__init__.py` but
never actually called anywhere in `orchestration/pipeline.py` — the same
"computed but silently discarded" shape as §12.1/§12.3/§12.6/§12.9. The
user supplied the real pre-5.0 source (Grimlock 4.7) showing Ritornello's
original job: not quantization at all, but a **post-quantization musical-
coherence repair pass** — sustain recovery, micro-note purging, tie
reconstruction, pitch-wobble collapse, and trouble-measure flagging, plus a
`CorrectionScribe` that surgically re-analyzes flagged measures against the
isolated stem audio (optionally pitch-shifted up an octave to escape muddy
low-end) using an aggressive-threshold re-detection pass, injecting
corrected notes with a confidence boost as "proof of repair."

Fully rewritten under that original intent, explicitly not reimplementing
what `VelocityMerge` (fragment collapse) and `PhraseIntelligence` (a weaker
phrase-grammar analog) already own, and — per explicit direction — using
Basic Pitch rather than librosa as `CorrectionScribe`'s re-detection engine
(a shared `core/basic_pitch_utils.py` was extracted so `EpistemicCouncil`
and Ritornello can't drift on how to call it correctly, the same class of
bug fixed once already in §12.1's neighborhood). Wired in as its own real
pipeline stage between `velocity_merge` and `consensus` (28 stages now, was
27), with the two stages that had been mistagged with Ritornello's
`SourceType` for years (`quantization`, `velocity_merge`) retagged to their
actual owners. Verified end-to-end on real audio (*Hopeful.mp3*,
*Measured_Impact.mp3*): real, non-zero revision/trouble-measure counts,
genuine `ritornello:sustain_recovery:extended`-tagged notes surviving to
final export, and real `CorrectionScribe`-corrected notes with correct
octave-shift/stem provenance tags on an isolated real audio slice.

### 12.15 MusicalFindingsMap rebuilt as a real single source of truth, plus §3.5 closed

An external audit correctly diagnosed `MusicalFindingsMap` as "just a
mutable bag of scalar values" with no real provenance, but its proposed fix
(a brand-new `MusicalEvidence`/witness-testify/consensus architecture) would
have been a *seventh* parallel confidence/evidence system in a codebase that
already has six — several of which (the ~30 frozen per-stage `*Testimony`
dataclasses in `core/testimony.py`, `EpistemicCouncil.resolve_tempo_
contention`/`resolve_drum_type_contention`/`resolve_duration_contention`,
`ConsensusVoter`'s weighted-confidence voting) already do most of what was
being proposed, just not applied to the findings map itself. The real
disease, verified against live code: the same tempo/key/groove/lattice
values get computed once as a clean, frozen `Testimony` object and then
manually, lossily copied into the map's separate mutable scalars —
`Pipeline` *also* separately held a third copy of the same objects
(`self._tempo`, `self._groove`, etc.) — and 5 pipeline stages wrote
attributes the map never declared as fields at all, silently working only
because the dataclass wasn't slotted.

Fixed by making the map hold the real `Testimony` objects directly, with
the old scalar names (`tempo_bpm`, `detected_key`, ...) surviving as
read/write properties over them (zero call-site changes needed at the ~13
existing write sites beyond assigning the richer object alongside), and
collapsing `Pipeline`'s separate third copy into properties reading through
the map. The 5 previously-undeclared attributes are now real, schema-
enforced fields. `to_temporal_lattice_evidence()`'s two hardcoded constants
(`quantize_strength=0.35`, `microtiming_window_ms=15.0`) now genuinely
derive from a blended confidence, which also gives `TemporalLattice`'s
previously-computed-but-never-used `overall_confidence`/`LatticeConfidence`
enum an actual purpose for the first time.

Separately, but closed in the same pass: §3.5/§7c's "five incompatible
confidence/scoring representations" finding turned out to be two genuine
name collisions (`ConfidenceComponents` defined twice with different
shapes; `Confidence` defined twice with unrelated meanings) plus three
legitimately different, purpose-built representations that don't need
unifying. Resolved with two pure renames (`DrumConfidenceComponents`,
`StageZeroConfidence` — zero behavior change, since drum classification's
weighting was already tuned and verified) rather than a forced shared base
class with no real polymorphic consumer to justify it. Both pieces
verified end-to-end on real audio (*Hopeful.mp3*, drum regression on
*Copper_and_Skin.mp3*) and cross-checked against two earlier same-session
runs — every deterministic structural value (tempo, time signature, key,
lattice period/ratio-family) held bit-identical across all three runs,
confirming neither change altered pipeline behavior anywhere it shouldn't.
