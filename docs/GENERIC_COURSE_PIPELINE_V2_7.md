# Generic Course Architecture V2.7 — Figma 02.27

This is an **incremental, offline** implementation for educational courses of different subjects. It is **not** the complete dubbing product.

## What this increment implements

- Audits canonical versus placeholder course identity (`pilot_course`, `Course pilot`) without guessing an official title. Generated fingerprint IDs are **source-instance-scoped** and do not authorize sharing a glossary across different lessons.
- Reuses extraction and domain evidence from the existing V2 pipeline, now stripping trailing sentence punctuation from candidate terms. Nothing is prepopulated with Andrew Ng or neural-network vocabulary.
- Attaches unit IDs, original `source_refs` and text evidence to candidate terms; no invented terms.
- Validates a proposed `source_term` against real source text, matching course identity and requiring a `sense` and `preferred_ar`. **This is structural validation, not model or expert semantic validation.**
- Requires an explicit, auditable human decision (`reviewer`, `decision_id`, `approved_at`, `confirmed_course_id`) to store a newly approved term. Does not approve anything automatically.
- Imports *only* explicitly approved same-course historical records. Blocks cross-course entries and re-use of a term not present in the current lesson.
- Runs conservative literal consistency QA on approved Arabic renderings; potential mismatches require human review. A successful literal match **does not prove semantic equivalence**.
- Produces impact lists of units requiring renewed QA when a previously approved Arabic rendering is changed or removed.
- Returns a separately versioned, derived report without touching `source_segments`, timing, translation, `final_dub_ar`, or original files.

## Current pilot

`translation_master_v2_5_editorial_candidate.json`: 17 units, 320 aligned words. The original course has only generic identity `pilot_course` / `Course pilot`; V2.7 does not misrepresent it as a verified course. Domain inference suggests “neural network” from five unit references but labels this **unvalidated**.

## How to run without repeating old steps

Open the **existing** GitHub Colab notebook (`notebooks/ArabicDubbing_CourseTranslation_V2_6_DynamicGlossary_QA_Colab.ipynb`), pull fresh code via section 1, run section 3 to locate `MASTER_PATH`, then run the newly inserted **5A — Generic Course Glossary V2.7** cell. Sections 4 and 5 have previously saved derived outputs: the new cell re-reads the master and enriches it **in memory**, so it need not regenerate previously saved V2.6 outputs. No translation, ASR, TTS or API is called.

Change `EXPLICIT_MASTER_PATH` and optional `COURSE_METADATA_OVERRIDES` in section 3 to handle other courses. A known title/ID must be grounded in user-supplied course metadata, not inferred as a certainty from speech.

## Next engineering work (not yet done)

- Multi-lesson, user-confirmed canonical course identity and versioned glossary persistence across lessons.
- Independent context/meaning validation with a user-approved provider or qualified human; avoid automatic semantic approval.
- Glossary-driven re-QA of changed translations using original English and full context.
- TTS pronunciation/timing review and actual voice-to-video synchronization, mixing and publishing.

## Verification

`python -m unittest discover -s tests -v` runs deterministic offline tests; `tests/test_course_glossary_v27.py` covers placeholders, different course fields, source evidence, validation gates, approved memory, cross-course isolation, literal QA, impact analysis and non-mutation. No paid API, GPU or TTS is required.