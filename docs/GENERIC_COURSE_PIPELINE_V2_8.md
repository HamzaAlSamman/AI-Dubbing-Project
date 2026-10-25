# مشروع الدبلجة: Multi-Lesson Pipeline V2.8 — implementation and engineering gates

## What V2.8 implements now (offline)

1. **Course Registry:** Supports many lessons under one *explicitly confirmed* canonical course ID and title, or isolated source-only mode if unidentified. Never treats `pilot_course` or an auto-generated fingerprint as an official course ID. Rejects conflicting non-placeholder source IDs and duplicate lesson IDs. TUs are always namespaced `lesson_id/tu_0001`, while the original IDs remain unchanged.
2. **Evidence-first Terminology Review Queue:** Extracts candidate terms from each lesson's source, combines per-course candidates (including isolated variables such as `M` and `y`, and connector phrases such as `for loop` / `for loops`), and retains real source unit IDs, word references and excerpts. It provides questions for sense disambiguation. It never fabricates a translation or approved technical meaning. Low-frequency terms still need model/human review, so coverage cannot be assumed complete.
3. **Approved Course Memory:** Writes an independent, versioned JSON snapshot only when explicit human-confirmed metadata and audited term-approval records are provided. Enforces course scope, unique decision IDs, and rejection of contradictory approvals. Previous approved records can be reused on new lessons even if the term isn't spoken in the newest lesson, provided their prior historical audit evidence is preserved. This is audit metadata, not cryptographic proof of identity.
4. **Cross-lesson QA:** Checks approved Arabic renderings against source occurrences, without treating literal string matches as semantic proof. If the same English surface has multiple approved senses, it requires sense disambiguation unless the reviewer supplied a scoped mapping (`sense_unit_ids`). Outputs human review flags; never auto-approves text.
5. **Audio Timeline Audit:** Reads an **already saved** TTS manifest. Compares generated clip durations with the source TU spans and flags likely overlaps or excessive speed changes. The reported speed ratio is diagnostic only, not permission to time-stretch or mix the output. No generated video and no audio synthesis.

## What is intentionally NOT done

- No new ASR, alignment, Translation Unit Builder, LLM translation, human approval, API request, GPU run, TTS, media synthesis or source mutation.
- Model-based contextual meaning validation is not complete. That is an opt-in later phase, subject to user provider/terms/cost selection.
- Full multi-lesson source metadata registration and real source audio/video synchronization are not proven on multiple authentic courses. Synthetic tests verify logic but not domain correctness.
- Model/human phrase review, pronunciations and final edit permissions still require explicit decisions. `final_dub_ar` remains unapproved.

## Current real-file pilot

`translation_master_v2_5_editorial_candidate.json` has 17 units and 320 word references. The original `course.id=pilot_course`, `course.title=Course pilot` are placeholders; source-only mode prevents sharing approvals across lessons. The previous Gemini TTS pilot has 3 actual saved WAV clips; the offline auditor can read the existing manifest but **does not verify the audio bytes**.

## How to run in EXISTING Colab notebook

The existing `notebooks/ArabicDubbing_CourseTranslation_V2_6_DynamicGlossary_QA_Colab.ipynb` is extended, not replaced. After sections **1** (Drive/GitHub) and **3** (select real V2.5 Master), run **5A (V2.7)** or the new **5B (V2.8)** directly. Section 5B uses `CANONICAL_COURSE=None` until you explicitly provide a confirmed course ID/title and reviewer, and `EXTRA_LESSON_MASTERS=[]` until additional lesson masters exist. A deterministic report is saved to Drive without overwriting existing artifacts. Then run **7B** to audit the old three-clip TTS manifest *if it exists*. Do **not** rerun V2.6 source enrichment or plan creation just to get these reports.

After approval decisions exist, create a **separately versioned approval file** rather than modifying source, and pass its records as `EXPLICIT_APPROVED_TERMS`; the notebook will never infer approvals from machine QA or old drafts.

## QA commands

`python -m unittest discover -s tests -v`

Includes course identity, colliding TU IDs across lessons, metadata integrity, cross-course isolation, conflicting decisions, approval versioning, ambiguous senses, source immutability, and read-only real V2.5 + historic TTS-manifest integration. Run the earlier `test_free_backends.py` also when the full GitHub checkout is in the same directory; no API calls occur in these tests.

## Next milestones in scope

- **V2.9:** Review interface and auditable term approval/amendment workflow; contextual semantic validation with an optional provider and explicit prompt/model provenance.
- **V3.0:** QA gates for text length, pronunciation, timings and audio content, plus safe per-unit source-aligned mixing on an isolated video copy (after human approval). Full-course continuity across genuine lessons.
- **Operational:** A private UI, jobs/checkpointing and Colab MCP execution access must be separately validated. A GitHub connection does not prove access to the Colab GPU session.