# Translation model-first V2.9 — baseline and gates

## Goal

Use an actual contextual translation model first, not a preset neural-network vocabulary or n-gram glossary as the translator. The course's extracted terms are unapproved hints; human-approved course-specific terminology is binding only for the exact matching course ID. The model derives new English→Arabic candidates, which stay independent from the existing translated `natural_ar` and original WhisperX words and timestamps.

## Candidate model lineup — selection, not final quality approval

- **Primary candidate:** `gemini-3.8-flash` (long context and structured output) — Free Tier eligible subject to account and rate limits. This model is a *candidate* until it wins a blind assessment.
- **Economic comparison:** `gemini-3.5-flash-lite` — Free Tier eligible, optimized for high-volume translation and simple processing.
- **Future no-per-call-API alternative:** Google `translategemma-4b-it` on Colab GPU; requires model access, sufficient GPU RAM, setup and quality benchmarking, not wired to the current notebook.
- `gemini-2.5-flash` no longer selected by default on newly configured pilots. Google recommends 3.5 Flash-Lite / 3.8 Flash for new projects.

Provider documentation: https://ai.google.dev/gemini-api/docs/models , https://ai.google.dev/gemini-api/docs/pricing , https://huggingface.co/google/translategemma-4b-it

**Billing and data governance:** Free Tier has per-project/model quota. When billing is enabled calls may be billable. The free tier may use content for service improvement. Only send course material to an external model if its sharing and rights are authorized. No auto-fallback and no retries on HTTP 429. Do not commit `GEMINI_API_KEY` to GitHub.

## Evaluation procedure

1. Load existing `translation_master_v2_5_editorial_candidate.json` via the current Colab notebook; reuse V2.6 `prepared` file rather than regenerate old translations.
2. Run new section `6A`: by default prints **six planned calls**, zero actual. It uses units `tu_0004`, `tu_0007`, and `tu_0015`, representing technical symbols, bidirectional propagation, and numerical labelling.
3. After explicit permission, check project **Free Tier** and rights; turn on all three runtime flags in the 6A cell. Then six **one-pass** model requests (one per unit/model) return `accurate_ar`, `natural_ar`, terminology suggestions and uncertainties into **new separate JSON files**. Source, aligned words and timestamps stay intact; no TTS.
4. Compare side-by-side with the existing V2.5 Arabic. Human evaluation criteria: semantic adequacy, terminology in context, numbers/variables/code, negation, easy spoken Arabic, unnatural additions, omissions, and later the duration of TTS output. Flag `tu_0007` and `tu_0015` original alignment warnings independent of the translation scores.
5. Establish the winning translation model only after this review, expand pilot to a second non-AI field, and *then* enable the full course glossary/QA implementation.

No unit will receive `final_dub_ar` or `approved` from this automated evaluation. Generated terminology always remains `candidate`. The tool refuses overwriting matching prior results and rejects fabricated source terms.

## Code organization

- `arabic_dubbing_v2/translation_model_v29.py`: plan, prompt, backend adapter contract, explicit provider gate, and output structure.
- `tests/test_translation_model_v29.py`: offline mocks for authorizations, no mutation, duplicates, non-source terms, overwrite prevention, request budgets.
- Existing Colab notebook, new section `6A`; no new notebook is created.

## Validation scope

- Local offline tests use a synthetic multi-domain project and the existing Master with 17 units. No real model-quality comparison was run here. Provider error/availability, runtime GPU, linguistic quality, and audio quality remain unverified.