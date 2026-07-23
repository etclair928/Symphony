# Symphony (5.6.1) — The Open Problems

*A formal statement of the problems in the mature, accreted 5.x codebase — the companion, from the other side, to Jazz's `GRIMLOCK_6.0_OPEN_PROBLEMS.md`. Written for intellectual curiosity: to see clearly what species of problem Symphony actually has, in prose, math, and CS terms.*

*The central thesis: Symphony carries **two orthogonal classes of problem stacked on top of each other**, and confusing them is itself a trap. Class A is the domain — the inverse-physics walls that are true of audio-to-notation itself, and which Symphony shares identically with Jazz and with any transcription engine ever written. Class B is the accretion — the pathologies of a system grown by many rounds of local fixes, none of which are about music at all. **Jazz was built to escape Class B, not Class A.** Every architectural law in Jazz's design bible is the negation of a specific Class B pathology named below. So this document doubles as the formal answer to "why does Jazz exist."*

---

## 0. Preliminaries

Symphony is the same operator chain as Jazz — separation $D$, transcription $T$, analysis $A$, export $E$ — but realized as an **accreted** system: ~40 modules across five clusters (`agents/`, `core/`, `epistemic/`, `orchestration/`, plus `server/`, `export/`, `ingestion/`), built over many rounds where each round answered a local question with a local piece of code. The defining structural fact, stated in `ARCHITECTURE_CONSOLIDATION_PLAN.md` §1:

> the same handful of questions get answered independently, more than once, by code that was never told the other answers exist.

Hold that sentence. It is the seed of most of Class B.

---

## PART A — The Inherited Domain Walls (shared with Jazz)

These are not re-derived here; they are formalized in `Jazz/GRIMLOCK_6.0_OPEN_PROBLEMS.md` §I–VI and hold **identically** for Symphony, because they are properties of the problem, not the code:

- **A.I Source-separation leakage** — bleed as an underdetermined inverse problem ($S \gg C$); ~16.5% of notes are honestly-transcribed leakage.
- **A.II Over-detection** — unsupervised legitimacy classification with no labels.
- **A.III Pipeline stochasticity** — Demucs non-determinism makes the whole chain a random function.
- **A.IV Rhythm quantization** — an ill-posed inverse problem needing a readability prior.
- **A.V The MIDI ceiling** — $\rho:\mathcal{N}\to\mathcal{M}$ is non-injective; clean notation is unreachable on a MIDI target by construction.
- **A.VI The evaluation gap** — no ground truth; every automatic metric is a surrogate.

Symphony has an `export/` path and a `server/` (web) surface Jazz lacks, so it faces A.V *sooner* (it actually tried to ship notation), but the wall is the same. **Nothing in Part A is what makes Symphony hard to work on.** That is Part B.

---

## PART B — The Accretion Pathologies (Symphony's distinctive problems)

Seven pathologies, each with the same shape: a piece of local reasoning that was individually sensible, composed into a global property that is not. For each: the math, the CS name, the concrete Symphony evidence, and the Jazz law that negates it.

### B.1 — The Reconciliation Problem (the central one)

**Prose.** Many independent pieces of code estimate the *same* latent quantity — where the onsets are, what the tempo is, how confident to be, what sample rate is needed — and none of them reconcile. Whichever writes to the shared field *last in execution order* silently wins. No disagreement is ever detected.

**Math.** Let $\theta$ be a latent quantity (say tempo). Symphony contains estimators $f_1,\dots,f_N$ producing $\hat\theta_1,\dots,\hat\theta_N$. A correct system emits a principled fusion $g(\hat\theta_1,\dots,\hat\theta_N)$. Symphony instead emits

$$ \hat\theta_{\text{out}} = \hat\theta_{\sigma(N)}, \qquad \sigma = \text{execution order}, $$

i.e. **the arbiter is `argmax` over *write time*, a quantity with no relationship to correctness.** Worse, no disagreement test

$$ \exists\, i,j : |\hat\theta_i - \hat\theta_j| > \tau \;\Rightarrow\; \text{raise} $$

is ever evaluated. Disagreement is not resolved wrongly; it is *not observed*.

**Evidence (same audio, same run).** `reverse_geo_crypt` reported quintuplet / tactus ≈ **48.8 BPM** for a *Hopeful* clip while `tempo_analysis` reported **132.2 BPM**; the pipeline emitted 132.2 only because that is what got written to `self._tempo` last. On *You Say God Says* the pipeline emitted **74.7 BPM** — the *weakest* candidate under independent re-check (~123 BPM). `pulse_analysis` reported a plausible **0.25 confidence** while starved on 1–4 onsets instead of the dozens available elsewhere in the same run.

**CS name.** A missing **sensor-fusion layer**; execution order acting as an implicit, arbitrary arbiter. Duplicate computation wastes CPU; the real cost is **silent contradiction with no detector**.

**Jazz's negation.** The Epistemic referee + the single `MusicalFindingsMap` blackboard: every estimator publishes to one place, and reconciliation is an explicit, testable operator — not a side effect of import order.

### B.2 — Destructive Mutation / No Single Source of Truth

**Prose.** State is mutated in place as it flows. Passes overwrite and drop fields on the note objects, so the original detection is unrecoverable and information only ever *decreases* down the pipeline.

**Math.** Let $I(s_k)$ be the information the pipeline state at stage $k$ retains about the true note. Under destructive update, each field-dropping pass is a projection $\pi_k$ with $|\text{img}\,\pi_k| < |\text{dom}\,\pi_k|$, so

$$ I(s_0) \ge I(s_1) \ge \dots \ge I(s_K), \quad\text{strict at every lossy pass.} $$

Monotone non-increasing information with no recovery path. An immutable core with append-only annotations inverts the inequality: $I$ is non-**decreasing** (annotations only add), and $s_0$ is always still there.

**Evidence.** `velocity_merge` constructing merged notes that discard non-listed fields; `temporal_lattice.quantize_notes` discarding every `NoteEvent` field it didn't explicitly copy; consensus tag-loss; `harmonic_series_match_ratio` / `phrase_id` / `instrument_family` repeatedly "never reaching exported notes" — all the same bug class: a pass rebuilds a note and silently drops what it didn't know to keep.

**CS name.** Absence of an immutable-core model → destructive updates → irreversible information loss **and** provenance loss (you cannot ask "what did the detector originally say?").

**Jazz's negation.** The frozen `Note` + `AnnotationStore` keyed by `Note.id`. Passes never mutate; they annotate. Information is monotone *non-decreasing* by construction — the load-bearing §2.2 law.

### B.3 — Invisible Flag-Driven Coupling

**Prose.** Behavior changes are triggered by flags set far away, invisibly reshaping what downstream stages do. Reasoning about any stage requires knowing hidden mode-state established elsewhere.

**Math.** The output should be $y = P(x)$ for input $x$. In practice it is $y = P(x; m)$ where $m$ is hidden global mode-state. The *effective* control-flow graph has edges

$$ (\text{flag-setter}) \longrightarrow (\text{distant consumer}) $$

that are visible at **neither** endpoint's call site. Local reasoning is therefore unsound: no stage can be understood from its own text.

**Evidence.** The design bible names this directly — *"Invisible flag-driven coupling is the same pattern that made 5.x hard to reason about"* — and mandates the fix (surface integrity issues in metadata, do **not** silently reshape engine behavior from a flag).

**CS name.** Hidden global state / action-at-a-distance; **non-locality of reasoning**. The reachability of a behavior is not decidable from the local source.

**Jazz's negation.** Keep behavior changes explicit and *few*; tag issues loudly into metadata rather than branching engine internals on an invisible flag.

### B.4 — Manual Lifecycle & Memory Bookkeeping

**Prose.** Feature buffers are managed by hand — acquire/release leases, arenas, guardians, transfer accounting. Manual resource management reintroduces the entire C-era leak/double-free bug family, now for audio feature data, under a hard memory constraint ("Law 1").

**Math.** A lease is reference-counted: $r(t) = (\text{acquires up to }t) - (\text{releases up to }t)$, and a buffer is freeable iff $r(t)=0$. If `release()` is never invoked,

$$ r(t) = \#\text{acquire}(t) \;\; \Rightarrow\;\; r(t)\ \text{monotone} \uparrow \;\Rightarrow\; \text{never freed.} $$

The reference-counting machinery exists but is not *driven* — the defining shape of a manual-memory leak.

**Evidence.** `EvidenceTracker.acquire()` called, `.release()` never called → every feature-extraction lease permanent for the run. `transfer_to()` guardian accounting gap in `arena.py`; unbounded merge chains in `_post_process`. Each a manual-lifecycle bug, none a crash — a silent leak of exactly the bookkeeping meant to prevent leaks.

**CS name.** Manual memory management without RAII/GC discipline; reference counting that isn't actually closed-loop.

**Jazz's negation.** A byte-budgeted LRU cache behind the Audio Engine's API boundary; defer hand-tuned arena/mmap optimization until profiling proves it's needed. Let the boundary own the lifecycle, not every call site.

### B.5 — Type Collisions and Boundary Field-Loss

**Prose.** The same concept exists as several different types, and code converts between the internal copies — each converter a place a field can be dropped.

**Math.** With $M$ internal representations of one concept, the system admits up to

$$ \binom{M}{2} = \frac{M(M-1)}{2} $$

converter edges, each a projection that can be lossy. Field-loss bugs are **exactly** these edges losing coordinates. The count grows quadratically in $M$; the only value of $M$ with zero lossy internal edges is $M=1$.

**Evidence.** Duplicate `LatticeConfidence` (two enums, different members, no shared import); duplicate `StageResult` / `EvidenceType` / `EvidenceLease` in Core; multiple bespoke `Confidence` classes. The field-loss bugs of B.2 are the downstream symptom of B.5's redundant types.

**CS name.** Absence of a canonical schema; the **"N adapters" anti-pattern**; field-loss as lossy projection at every internal boundary.

**Jazz's negation.** "One type per concept. Translators/adapters live **only** at the edges (external model → canonical type; canonical type → MIDI). If you're writing a converter between two *internal* modules, the two types should have been one. **Unify, don't translate.**" That edict sets $M=1$ internally, driving the quadratic converter surface to zero.

### B.6 — The Observability Gap (the meta-pathology)

**Prose.** Every failure in B.1–B.5 failed **silently**. The reason nobody noticed 74.7-vs-123 BPM, or a starved 0.25 confidence, is that nothing was watching. An accreted system survives its own pathologies only if they are *visible*, and Symphony's were not.

**Math.** Borrow observability from control theory: a system is observable if internal state $x$ can be reconstructed from the output $y=h(x)$ over time. Silent overwrite makes $h$ **non-injective on the fields that matter** — many distinct internal histories (a healthy 132 BPM vs. an unreconciled 48.8-vs-132 collision) map to the same terminal output. The internal state is therefore **unrecoverable from the output**: unobservable. The specific missing signals are the *alternatives considered* and the *runner-up margin* — the downbeat search tests 6 phase hypotheses and only the winner's score survives, discarding exactly the margin that tells you how confident the decision really was.

**CS name.** A missing **observability layer**; the discarded-runner-up problem. This is the connective tissue: B.1–B.5 are individually survivable **iff** instrumented, and they weren't.

**Jazz's negation.** `Music_Box` (forensic, persistent, records every hypothesis + veto + transformation) and its live sibling `Window_Pane` (built, verified) — plus the "surface loudly" edict. The design goal that alternatives-considered and runner-up margins get *logged* is precisely the fix for $h$ being non-injective.

### B.7 — Stranded Capability (the inverse of dead code)

**Prose.** Symphony is full of built, capable subsystems that are simply **not on the execution path** — unwired, not dead. The risk is the mirror image of dead code: not "safe to delete" but "value stranded off the call graph, invisibly not running."

**Math.** Two graphs over the same modules: the **module graph** (what imports what) and the **call graph** (what actually runs). A node can satisfy $v \in \text{ModuleGraph}$ while $v \notin \text{reachable}(\text{CallGraph}, \text{entry})$. Symphony has many such $v$: present, tested, wireable — unreached.

**Evidence.** The standing project rule itself: *"unused-looking agents/core/epistemic classes are unwired, not dead; wire them in via the factory, don't delete."* `SpectralDrumClassifier` was dead-code-until-wired; `reverse_geo_crypt` reached a broken onset source; whole analysis agents sit behind `agents/factory.py` awaiting wiring.

**CS name.** A **reachability** gap between module graph and call graph. Manual wiring (`factory.py`) is the mechanism, and manual wiring that must be remembered is itself an accretion risk.

**Jazz's negation.** Explicit, centralized wiring in the Conductor with per-stage verification — reachability made a checked property, not a thing you hope someone remembered.

---

## PART C — Synthesis: the entropy of accretion

Stack the seven. Individually each is a reasonable local decision — compute the tempo here, merge these notes there, set this flag, manage this buffer, add this type, skip the logging, leave this agent for later. Composed, they produce a system with a single emergent property:

$$ \textbf{complexity outran observability.} $$

Symphony's real open problem is not any one bug in B.1–B.7 — it is that the **rate of silent, un-reconciled interaction grew faster than the system's ability to see itself.** Each fix (and many have genuinely landed — six under the consolidation plan alone) tends to *reveal* another silent disagreement rather than close the account. That is the signature of the classic **second-system / software-entropy** problem: local order purchased at the cost of global legibility.

Two honest counter-currents, because the story is not "rewrite always wins":

1. **Some debt is payable in place.** The 5.6.1 record shows *fix-now worked* for a real set of problems (§7a/b/c, the full DrumIntelligence rewrite, the Omnizart bug, Schoenberg's harmonic-math bug, ReverseGeoCrypt's wiring) — fixed under 5.6.1, verified on real audio, without waiting for a rewrite. The discipline that made it safe: **one change at a time, real e2e verification against a known baseline, never batching architectural changes.** Not everything requires Jazz.

2. **The rewrite is a bet, not a proof.** Jazz is the wager that re-deriving from structural laws (immutability, single source of truth, unify-don't-translate, surface-loudly, explicit wiring) is cheaper than continuing to service Symphony's compounding interaction cost. The bet is *reasonable* — each Jazz law negates a named Symphony pathology above — but it is not free, and it re-pays Part A's domain walls from scratch. The open question Symphony poses to Jazz is exactly: **was the interaction cost high enough to justify paying the domain walls twice?**

The most compact way to state Symphony's condition: it is a system where **the physics (Part A) was never the thing that hurt.** What hurt was that many individually-correct local decisions composed into a whole that could no longer watch itself think. Jazz is the hypothesis that you fix that with laws, not fixes — and Symphony, still running, still transcribing, is the control group that keeps that hypothesis honest.

---

## PART D — Audit snapshot (2026-07-22)

*Parts A–C are the standing diagnosis. This part is a dated, evidence-backed reading of the code as it actually stands today, so the document tracks reality rather than drifting into a fixed essay. It records three things: what has been paid down since the diagnosis was written, the concrete live problems a fresh audit surfaces, and one structural change to the domain-wall picture.*

**Method.** Static audit of the 77 tracked `*.py` files at HEAD `8939ac5`: import smoke of the core modules (all clean), the committed test (`test_musicxml_writer.py`, passes), an AST sweep of exception handlers, and a targeted re-read of the sites Parts B and A cite. No pipeline re-run was needed for the structural findings; the notation-surface findings rest on the two real end-to-end runs done this session (Hopeful, Measured_Impact).

### D.1 — What has been paid down (the diagnosis is partly stale, in Symphony's favour)

- **B.1 reconciliation now has a real disagreement operator — for tempo.** `EpistemicCouncil.resolve_tempo_contention` (wired at `pipeline._run_tempo_analysis`) anchors on the highest confidence·weight witness, ratio-aligns the rest, **and logs the disagreement to `Music_Box` as a first-class decision record** instead of letting execution order silently arbitrate. The exact 48.8-vs-132 collision Part B.1 opens with is the case this closes. This is the single most load-bearing pathology in Part C, and it is no longer unobserved — for tempo.
- **B.6 observability got a concrete new instrument: the export duration gate** (`core/duration_check.py`, wired into `_run_export`). It compares the rendered MIDI's playback length to the processed audio span and **surfaces a mismatch loudly** (prints the ratio, logs a `duration_gate_failed` decision) rather than shipping a broken timeline in silence — the precise reflex Part C names. It is truncation-aware and, notably, its own outer guard was hardened this session so the anti-silent-failure check cannot itself fail silently.
- **B.5 type collisions** named in the diagnosis (`LatticeConfidence`, `StageResult`/`EvidenceType`/`EvidenceLease`, the bespoke `Confidence` classes) were unified or renamed under the consolidation plan and the MusicalFindingsMap refactor. One collision remains and is now **visible on a new surface** — see D.2.

### D.2 — Concrete live problems (the "bugs and problems" this audit surfaces)

Ranked by how much they hurt the current north star (a readable score), not by code severity:

1. **Meter mis-detection — the barlines are wrong on the page.** On the reference material the detector settles on 4/4 where the true meter is 6/4 (compound), and the meter stage *does* flag contention — so this is an accuracy wall, not a silent one. But every downstream barline, beam group, and measure count inherits the error, and it is the single most visible notation defect. (Task #255, still open.) This is a Part A.IV/domain problem, not accretion.
2. **Instrument identity: the score was grouping on the noisiest available signal — now fixed by using the stem instead (the Jazz lesson, measured).** The exporter first grouped notation parts by the per-note *timbre* family (`family:X`/`voice_family:X`). On real material that signal is badly unreliable: a Hopeful run (56 pitched notes) had **43% of notes with no family tag at all**, and the rest were labelled with orchestral families (brass/strings/woodwind) for a guitar/piano/vocal song — so grouping by them produced phantom *Brass/Synth/Woodwinds* parts and **no Bass staff**, with correctly-separated bass notes (tagged `stem:bass`) scattered into "Strings". The key measurement: every note *already carries its stem of origin* (`stem:bass`/`stem:vocals`/`stem:other`), which is ground truth from the separator, not an inference. Switching the exporter's grouping key to stem-first (timbre only as a fallback for untagged notes) turns that same run into three correct parts — **Bass / Other / Vocals** — verified on the real notes. This is the single most direct thing Jazz teaches Symphony's notation layer: *the stem is the instrument identity; do not re-derive it per-note from a noisy timbre classifier on a merged mix.* The residual (the two coexisting `InstrumentFamily` enums; the timbre classifier's own low-register bias #141) still wants paying down, but it no longer drives the page.
3. **Voice tracking (data association) — Symphony is actually AHEAD of Jazz here; the exporter just wasn't using it.** The "776 near-empty voices for 2183 notes" result was **Jazz's** uncapped greedy streamer (`stream_into_lines`), *not* Symphony — an earlier draft of this section wrongly imported it. Symphony's `agents/analysis/voice_continuity.py` is the more mature implementation: it is **capped** (`max_voices=8`, with `_merge_excess_voices`), **timbre-aware** (a `family_mismatch_penalty` on the assignment cost), and **cluster-aware** (SpectralMasker `cluster_id` via a witness log), and it writes its per-note assignment onto each note as `phrase_id`. Measured on the Hopeful run it produced 8 voices, not a shatter. The real gap was that the MusicXML exporter ignored `phrase_id` and re-derived voices by raw overlap — but a measurement showed those 8 voices each mix 5–7 families (they are time-overlap streams, not clean lines), so wiring `phrase_id` as-is would not help. Symphony's genuine risk is the *opposite* of Jazz's: **under**-segmentation (voice_continuity collapsing "other" into one wide voice, #142), not shatter. The proper unlock remains Part A's data-association problem (MTT/JPDA-shaped), upstream of any exporter — but on the accreted-vs-rewrite question this is a clean case where the accreted system carries the better mechanism.
4. **The note-onset tempo witness reads ~19% high, systematically.** Measured on two songs (Hopeful 173.6/146, No Pasarán 158.7/133 — both ×1.19): it reports a subdivision as the tactus. The B.1 referee now *excludes* it when its fold does not align to the anchor, so the output is correct — but the bias is **masked, not fixed at source** (task #271). A masked-but-unfixed witness is a latent trap the moment the anchor is itself weak.
5. **The B.6 debt is bounded but structurally intact.** An AST sweep finds **48 of 390 exception handlers silently swallow** (body is `pass` only), plus one bare `except:` (`spectral_masker.py:513`, an optional-metric guard). The worst silent overwrites named in Part B are gone, but "defensive swallow" is still the ambient idiom — each of the 48 is a place a real error becomes invisible.

6. **The duration gate caught a real timeline bug — a sample-rate mislabel that compressed every 48 kHz transcription by ~8% (FOUND + FIXED, 2026-07-23).** `_run_separation` resamples the audio to 44100 (its working rate) and gets 44100-rate stems, but `_apply_separation_result` stored them in a rebuilt `AudioViews` still labeled with the *master's* rate. For a 48 kHz source, every downstream stage then read `sr = audio_views.sample_rate = 48000` and treated the 44100 stems as if 48000, compressing the whole timeline by exactly 44100/48000 = 0.9187 — a 60 s clip's notes ending at ~55 s. The gate flagged it as `MIDI_SHORT ≈ 0.92×` on Hopeful, No Pasarán, *and* Grey on Grey (three different songs, identical ratio — the content-independent constancy that a scaling bug, not tail under-detection, produces). The fix: resample each stem back to the master rate in `_apply_separation_result` so the one `AudioViews.sample_rate` is true for master and stems alike. Verified: Hopeful 60 s went `0.92× MIDI_SHORT → 1.006× PASS`. This is exactly the class of silent timeline regression the gate (D.1) was built to catch, and the first time it earned its keep on a real defect.

   *A recorded reasoning error, kept as a lesson:* an earlier draft of this section called the same `MIDI_SHORT` a "non-problem — content-dependent tail under-detection," on the strength of two data points (one song short, one — Measured_Impact — at `1.003×`). That was wrong: the outlier was almost certainly *sample-rate*-dependent (a 44.1 kHz source has no resample and no mismatch), not content-dependent, and two points were too few to tell a constant ratio from a variable one. The gate was right; the analyst was hasty. Three points and a constant 0.92 are what exposed it.

### D.3 — The domain-wall picture changed: A.V is now *partially* scaled

Part A.V (the MIDI ceiling — clean notation unreachable on a MIDI target) was stated as a wall Symphony hits "sooner" because it tried to ship notation. Symphony now emits **MusicXML** (`export/musicxml_writer.py`, opt-in `--musicxml`, emitting both a faithful and a legibility-policy view), which structurally provides what MIDI cannot: independent voices as separate staves, a key signature, rests, and tie-capable durations, with validity guaranteed against the MuseScore voice-numbering trap. Verified valid and re-parseable on two real songs. So the format half of A.V is no longer a wall — which **sharpens** the diagnosis: every remaining notation defect (D.2 items 1–3) is now provably an *upstream* problem (meter, instrument identity, voice association), not a limitation of the output target. The residual true ceiling (enharmonic spelling, beaming nuance) is a smaller thing than it looked when MIDI was the only exit.

---

*Extension note: the same lens applied to Grimlock 4.7/4.8 would ask a narrower, earlier question — not "how did the accretion compound" (that's the 5.x story) but "which of these seven pathologies were already present at 4.7, and which were introduced by the very fixes that made 4.7 more capable?" That is a genuinely different and answerable historical question, and a natural next document.*
