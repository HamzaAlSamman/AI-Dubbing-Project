# V2.10 — General educational course translation, source-first, batched

The previous pilot's 17 Arabic translations are **not used as inputs** to new-course translation. This is an independent pathway which requires only a Master JSON with `translation_units[].id`, English text in `translation_units[].source.text`, and any existing timing/ref metadata. No Arabic translation fields are needed.

Workflow: aligned source Master → context-bound unit batches → one structured Gemini response per batch → per-ID validation → independent candidate JSON per batch → bilingual QA, dynamic glossary review, human approval → separate TTS stage. No ASR/translation/TTS happens automatically by importing code or executing the plan.

**Models and quotas.** Initial candidate for bulk translation: `gemini-3.5-flash-lite`, with observed project caps 15 RPM/500 RPD and 250K TPM; use `gemini-3.8-flash` for a *separate*, deliberately selected QA/test run (observed 5 RPM/20 RPD). These are observed user-screenshot quotas, not immutable global allowances. All clients and keys sharing the same project contribute to usage. Batch multiple units per *synchronous request*; do **not** confuse this with Google's separately billed Batch API service.

Packing defaults: 15 source units / request, max 14k input characters (rough guard only; actual token budgets can differ), 1 neighbor each side. Model returns an ordered `results` list; IDs, source material, word refs and timestamps are not altered. Long TUs fail clearly rather than being truncated. A verified cache stops duplicate API requests after session interruption; any conflicting cache blocks execution. Enforce max new requests, optional remaining daily budget, minimum local delay between requests. A 429 or any provider error stops execution; no automatic retries or provider switches. Store per-batch candidate files under a source fingerprint-specific directory, never replacing an existing file.

For **500 source TUs**, 20 units/request yields **25 planned requests**, subject to input/output token constraints and provider quality. No guarantee of exact runtime or request success. Gemini TTS shown at 3 RPM/10 RPD is not viable for a full course with one unit per request: handle speech separately using longer allowed synthesis segments where the model supports it, or a properly licensed local TTS model (not chosen by this translation module).

## Notebook use

In the *same* existing Colab notebook, after section 1 mounts Drive and pulls GitHub, use **6B — Generic Course Translation**. Set `NEW_COURSE_MASTER_PATH` explicitly. Leave `PREVIEW_ONLY=True` initially. New course files do not default to Andrew Ng/pilot. Sections 3–6A remain for legacy experiment and are not prerequisites of 6B. Do not run all notebook cells blindly.

The free Gemini API tier may use submitted content to improve products. Check course rights, privacy, account billing, and limits before enabling. Translation candidates are not editorial, terminological, dubbing, or audio approvals. Existing sources and original IDs/word timing survive untouched.

Tests cover diverse subjects, missing Arabic translation, scoped course glossary, source fingerprint sensitivity, batch splitting, preview gates, caching/reuse, denial of corrupt cached output, source immutability, rate pacing and quotas. Tests do not constitute a live Colab GPU or provider reliability test.