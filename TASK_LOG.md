# Symphony/Grimlock — Task Log

A running numbered record of every discrete task completed across this project's
sessions. Generated from the live task tracker; kept in ID order (earlier tasks
often set up context later ones depend on). Status is current as of the last
update below.

---

1. Commit tempo false-positive fix
2. Update `_extract_testimony_dict` to flatten per-note sub-dicts
3. Wire real per-note resonance from AnechoicMa
4. Wire per-note pulse alignment from PulseField
5. Wire per-note swing alignment from GrooveField
6. Wire per-note local tempo from TempoIntelligence
7. Wire real drum_anchor/anti_groove from DrumIntelligence
8. Wire real per-note phrase membership from VoiceContinuity
9. Verify quaver_intelligence now uses real per-note evidence + full pipeline test
10. Full codebase logic review for musical transcription accuracy
11. Wire QuaverIntelligence duration hypotheses into quantization decisions
12. Fix MIDI export hardcoded 120 BPM tick conversion
13. Fix offbeat detection threshold 4x too tight in TemporalLattice
14. Wire real Scribe validation into `_run_validation`
15. Fix main.py vetoes display key mismatch
16. Add soft Scribe checks to major detection stages
17. Fix harmonic_intelligence.py inharmonicity tautology
18. Tune drum onset detection to reduce false positives ("too busy")
19. Expand EpistemicCouncil to arbitrate contested duration hypotheses
20. Fix fallback.py missing numpy import
21. Populate ReverseGeoCrypt's `sixteenth_grid_ms`
22. Fix `detect_tempo_changes` re-running full ensemble per window
23. Improve key-signature detection (relative major/minor confusion)
24. Fix PitchProximitySeparator temporal-overlap cost bug
25. Fix VoiceContinuity confidence calc to use real assignment cost
26. Wire SpectralMasker `cluster_id` onto `NoteEvent.metadata` in pipeline
27. Wire polyphonic_peaks into harmonic_intelligence.py
28. Decide fate of `FeatureBundle.voice_separation` stub
29. Store per-note timbre entities and tag notes with instrument family
30. Add family-mismatch soft cost penalty to voice separators
31. Build family-level MIDI range table and InstrumentRangeValidator
32. Fix quick_rhythm_test's stale AudioContext construction
33. Run DrumIntelligence against isolated drum stem
34. Run RhythmEngine against isolated drum stem
35. Test SpectralDrumClassifier centroid boundaries on cymbal/snare samples
36. Test MultiBandOnsetDetector on isolated tom hits
37. Wire SpectralDrumClassifier into EnsembleDrumDetector (currently dead code)
38. Fix FLAM articulation bug and implement ROLL detection
39. Add RIMSHOT/cross-stick branch to SpectralDrumClassifier
40. Create drum_coprocessor.py module skeleton + CoprocessorConfig
41. Implement HFCOnsetDetector + AdaptiveMedianThreshold
42. Implement NMFDrumDecomposer (bootstrap templates + manual multiplicative-update NMF)
43. Implement envelope-based flam/buzz-roll refinements
44. Add `EpistemicCouncil.resolve_drum_type_contention`
45. Fix `_find_consensus` grouping key to `(time_bucket, drum_type)`
46. Wire coprocessor into EnsembleDrumDetector.detect() and route HFC into MultiBandOnsetDetector
47. Bump `drum_detection` stage `estimated_memory_mb` in pipeline.py
48. Verify: regression, synthetic polyphony test, real-audio tuning, contention-vs-polyphony separation
49. Fix false-positive polyphony candidates from single-hit broadband bleed
50. Run full end-to-end pipeline test on Hopeful.mp3 (60s)
51. Fix `harmonic_series_match_ratio` never propagated to NoteEvent
52. Fix harmonic_validation testimony being disconnected from downstream notes
53. Fix `TemporalLattice.quantize_notes` discarding all non-listed NoteEvent fields
54. Fix VelocityMerge merged-note construction discarding fields too
55. Fix Basic Pitch `predict()` call crashing on every invocation
56. Wire `phrase_id` from `phrase_id_map` onto actual NoteEvent objects
57. Fix `Scribe._check_phase_consistency` treating polyphony as an anomaly
58. Fix drum event `end_ms` using flat 150ms instead of per-type duration table
59. Fix `transfer_to()` guardian accounting gap in arena.py
60. Delete 3 confirmed-dead deprecated guardian.py methods
61. Wire `nested_arenas()`/`clear_cache()` into pipeline main stage loop
62. Run version sweep to 5.6.1 across ~60 files
63. Add `query_vetoes()`/`query_entries()` to MusicBox
64. Wire `log_decision` into 8 currently-silent music_box-holding classes
65. Expand music_box logging in 4 already-active classes
66. Verify all three initiatives end-to-end
67. Fix `MusicBox.flush()` silently destroying buffer when no log_path configured
68. Fix `log_decision` flooding session log with fake errors for free-form decision_type strings
69. Wire `music_box_logs`/`veto_history` through PipelineResult into main.py JSON export
70. Replace synthetic beat grid with real detected beat positions
71. Build general swing/shuffle detector independent of kick presence
72. Add ternary/compound-meter subdivision support to TemporalLattice
73. Fix `rhythm_engine.py` `_post_process` unbounded merge chain
74. Fix stale asyncio header comment in spectral_masker.py
75. Delete orphaned basic_pitch_test.py
76. Fix VoiceContinuity: new-voice creation never records assignment cost
77. Fix midi_writer.py polyphony force-off sending wrong pitch
78. Fix velocity_merge.py unnormalized duration term in weighted-average merge
79. Fix velocity_merge.py `gap==0` excluded from both overlap and adjacent detection
80. Fix SpectralDrumClassifier misclassifying kicks as cymbals
81. Expose downbeat phase from `TimeSignatureDetector._downbeat_salience`
82. Fix `build_beat_grid` hardcoded 4/4 + wrong `beat_times[0]`-is-downbeat assumption
83. Wire real metrical-position evidence into DrumIntelligence classification
84. Fix main.py Duration field showing processing time not song length
85. Investigate bass+drums stems both silent on You Say God Says.mp3
86. Fix ingestion `force_mono` hardcoding (§7a)
87. Fix Demucs wrapper honesty + Mel-Roformer shape-mismatch bug (§7b)
88. Fix duplicate Core types: `StageResult`, `EvidenceType`, `EvidenceLease` (§7c)
89. Remove VelocityMerge fuzzy pitch-tolerance merge option
90. Add audio-separator dependency, wire real MelBand-Roformer checkpoint
91. Restructure demucs.py: collapse redundant nested fallback into one clear ladder
92. Read full drum_intelligence.py + drum_coprocessor.py before restructuring
93. Design new `agents/drum_intelligence/` package structure
94. Migrate onset detection + shared types to new package
95. Migrate classification (per-family experts) to new package
96. Migrate articulation refinement to new package
97. Build new interpretation.py — the missing groove/metrical-position layer
98. Migrate consensus/voting to new package
99. Build thin orchestrator + update all import sites
100. Verify rewritten DrumIntelligence end-to-end on real audio
101. Investigate/fix tempo lock at 44.9 BPM on You Say God Says.mp3
102. Fix reverse_geo_crypt out-of-range tempo crash path
103. Investigate flat 0.25 separation confidence for all stems
104. Repo-wide TODO/FIXME/XXX sweep for quick real fixes
105. Check consensus `other_stem` empty-result veto is expected behavior
106. Wire real SPICE model via tensorflow_hub into EpistemicCouncil
107. Build bass octave-shift-up/analyze/correct-down pitch pass
108. Wire EpistemicCouncil into bass and master pitch stages
109. Assess Omnizart install feasibility (isolated venv likely needed)
110. Review harmonic_intelligence.py for compatibility with richer pitch evidence
111. Cache Basic Pitch / CREPE model loads across EpistemicCouncil calls
112. Install patched local omnizart with vamp made optional
113. Stop running master (full mix) as redundant parallel pitch source
114. Replace crude pitch/100ms dedup with real cross-stem consensus
115. Fix DrumIntelligence hardcoded 44100Hz sample rate bug
116. Fix `pitch_detection_bass` silently timing out at 60s
117. Fix schoenberg_mirror ZCR-before-harmonics verdict ordering
118. Route schoenberg_mirror audits to isolated stem audio
119. Fix HarmonicSeriesAnalyzer partial-detection bugs
120. Independent bare-model analysis of 'other' stem vs Grimlock
121. Fix Omnizart `model_path` case bug + disable by default
122. Fix drum near-duplicate/refractory-period bug + investigate wiring
123. Investigate tempo/lattice bugs and Groove/Pulse/ReverseGeoCrypt/AnechoicMa wiring
124. Full-codebase audit: false wiring, dead code, redundant subsystems
125. Wire Stage_Zero pre-separation conditioning into pipeline
126. Fix duplicate `LatticeConfidence` enum + EvidenceTracker cleanup
127. Fix reverse_geo_crypt reading broken RhythmEngine onset source
128. Execute ARCHITECTURE_CONSOLIDATION_PLAN.md
129. Document TempoOctaveCorrector structural bug (deferred)
130. Finish remaining 7 onset-detection sites per consolidation plan §3.2
131. Fix TempoOctaveCorrector structural bug + tempo estimator demotion
132. Diagnose bass-stem clutter using TimbreIntelligence/VoiceContinuity data
133. Add `vocals_stereo` to AudioViews + pull VOCALS stem from separator
134. Add `_run_pitch_detection_vocals` stage mirroring bass
135. Thread vocals notes through `all_pitched` aggregations
136. Route TimbreIntelligence classification to isolated stem audio
137. Implement bass-note witness-corroboration filter
138. Verify vocals wiring + timbre isolation + bass filter e2e *(in progress)*

---

*Last generated from the live task tracker. Items are added as new work is
scoped and marked complete as they land — this file is a log, not a plan;
see `GRIMLOCK_6.0_DRAFT.md` and `ARCHITECTURE_CONSOLIDATION_PLAN.md` for
forward-looking scope and open problems.*
