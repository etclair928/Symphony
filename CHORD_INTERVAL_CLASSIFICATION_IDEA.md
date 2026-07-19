# Chord & Interval Structural Classification — Parked Idea

**Status: PARKED, not scoped, not started.** Saved for later workshopping.
This is a different problem than `ONSET_TIMING_PRECISION_PLAN.md` (harmonic
*labeling* of simultaneous notes, not onset *timing*) and was deliberately
kept out of that document for that reason. Nothing here has been decided —
this is the raw idea plus the issues found reviewing it, so a future session
doesn't have to reconstruct either from scratch.

---

## 1. The Original Idea

Polyphonic pitch trackers (Basic Pitch, Omnizart) output a flat cloud of
concurrent note testimonies with no structural interpretation — no sense of
"this is a clean dyad," "this is a triad," "this is incidental low-end
cluster noise." The proposal was a sanitization/classification stage sitting
between raw voice extraction and `tonal_testimony.chord_map`, with three
pieces:

1. **Frequency-ratio/interval gating** — classify simultaneous pitch pairs
   by their semitone/cents distance (P4 ≈ 500¢, P5 ≈ 700¢, sixths, sevenths,
   sub-300¢ clusters) rather than trusting raw MIDI bins alone.
2. **Low-end cluster de-noising** — below a crossover frequency (proposed:
   150Hz), collapse multiple simultaneous low pitches down to "the dominant
   bass voice plus the strongest interval," discarding the rest as mud.
3. **Structural template matching** — test the surviving pitch set against
   a fixed table (clean dyad, major/minor triad, diminished/augmented,
   dense cluster) rather than an open-ended classifier guessing at a chord
   from an unstructured pitch bucket.

A concrete sketch (`PolyphonicHarmonicSanitizer`) was provided: sort a
frame's `{frequency: energy}` pairs by energy, apply the low-end collapse,
cap at ~5 surviving voices, convert to semitone intervals from the lowest
surviving pitch, and match against a small dict of triad/dyad signatures
(`{3,7}` → minor triad, `{4,7}` → major triad, etc.), falling back to
`"dense_cluster"`/`"polyphonic_complex"` for anything else. The open
question attached to it: should the template matcher be extended to catch
jazz voicings (rootless 9ths/13ths)?

## 2. What's Actually Wrong With It (found during review)

- **Root-finding bug, not a nitpick.** The sketch takes `frequencies[0]`
  after a numeric sort — i.e. "the lowest pitch present" — as the chord
  root, and computes every interval relative to that. Lowest-sounding pitch
  and harmonic root are different things whenever a chord isn't in root
  position. A first-inversion C major triad voiced E-G-C would compute
  intervals from E (E→G is a minor third, E→C is a minor sixth) and fail to
  match the major-triad template `{4,7}` at all, despite unambiguously being
  a C major chord. Real root-finding needs to test each candidate note as
  root and see which produces a valid template match — not assume
  lowest-pitch-present = root.
- **The pitch precision claim is contradicted by its own implementation.**
  The prose sells "precise harmonic distance via frequency ratios/cents" as
  the improvement over crude fixed MIDI bins; the actual code immediately
  does `round(12 * np.log2(f / root))` — rounds to the nearest semitone,
  throwing away exactly the cents-level precision just claimed as the
  advance. Either keep real cents-level comparison through the matching
  step, or drop the framing — as written it's inconsistent.
- **The 150Hz "keep only the dominant bass tone" filter should probably be
  rejected outright, not just reworked.** It unconditionally discards every
  sub-150Hz pitch except the single loudest one, on the assumption that
  simultaneous low pitches are inherently mud. This directly contradicts
  something already checked empirically this project: the "bass gets
  cluttered by bleed from other instruments" hypothesis was tested against
  real audio (isolated Demucs bass stem vs. "other" stem, matched
  timestamps) and was **not** confirmed — 13% exact pitch/time matches,
  49.5% same-pitch-class matches, consistent with ordinary harmonic
  coherence (bass and other instruments legitimately sharing chord tones),
  not clutter. A hard filter built on an assumption already tested and not
  confirmed would delete real, legitimate polyphony (a bass note plus a
  genuine simultaneous low piano/organ chord tone is completely normal).
  This is exactly the kind of confident-sounding, unvalidated aggressive
  filter this project's burden-of-proof principle (established for
  `Ritornello`'s repair passes) exists to catch before it ships.
- **API/data-model mismatch.** The sketch operates on frame-level
  `{frequency: energy}` dicts. This codebase's actual harmonic/tonal
  analysis (`HarmonicValidator.validate(note, features)`,
  `KeyDetector`) operates on discrete `NoteEvent` objects with
  `start_ms`/`end_ms`/`pitch`, not continuous per-frame frequency-energy
  maps. It's presented as something you can "plug directly between raw
  voice extraction and `tonal_testimony` writing," but it doesn't match how
  this project's data actually flows — real integration would need either a
  new frame-level extraction step feeding it, or reworking it to operate on
  groups of already-detected simultaneous `NoteEvent`s instead.

## 3. What's Worth Keeping

- The core goal — enriching `chord_map` with named interval/chord-quality
  tags instead of leaving simultaneous notes as an uninterpreted bucket —
  is a reasonable, real gap.
- The template-matching pattern itself (`unique_intervals.issuperset({...})`
  against a small dict of known chord-tone signatures) is clean, simple,
  easily testable, and easily extensible to sevenths/ninths/thirteenths —
  **once** root-finding is fixed. Extending the template table before fixing
  root detection would just mean the extended templates also silently
  misfire on anything not in root position.
- The dyad-labeling table (P4/P5/sixth/seventh/cluster by semitone count)
  is a fine, simple starting point on its own.

## 4. If This Gets Picked Up Later

Rough shape of what an actually-sound version would need, for whoever
scopes this next:

1. Real root-finding (try each candidate pitch as root, pick whichever
   produces the best template match — not "lowest pitch wins").
2. Decide whether cents-level precision matters enough to keep through
   matching, or explicitly commit to semitone-rounding and drop the cents
   framing.
3. Drop or heavily gate the low-end collapse filter behind real evidence
   for this specific song/stem, given the existing negative result on the
   clutter hypothesis — don't ship a blanket rule.
4. Decide the actual integration point given this codebase's `NoteEvent`-
   based data model, not a frame-level dict — likely: group already-detected
   simultaneous notes by time-window, then classify the group.
5. Extend the template table to real extended jazz voicings only after 1-4
   are settled.
