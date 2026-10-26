# V2.13 — Duration Matching: TTS → exact-duration WAV → timeline

This is an **offline** stage that runs *after* a TTS model has generated a WAV. It does **not** select/generate the speech model, call an API, or change the course source data.

## Behavior

- Accepts a source Master (with translation units, source time spans), a TTS manifest with a unit ID and SHA256 for an existing WAV, and a directory of WAV files. Cross-checks that the original audio and source Master belong to the same run where a master SHA is supplied.
- Reports the generated speech length, the target source length, the required speed ratio, and potential source-unit overlap.
- For a modest overrun (no more than ~20%), renders a new WAV via **FFmpeg `rubberband`** tempo processing (pitch-preserving); pads a few trailing silent samples to reach the exact sample-count target. Does not trim spoken audio.
- For a modest underrun, pads the end with silence, never a cut. Overlong clips such as 2.64s vs 1.322s are **blocked** and must go through shorter text/retake TTS later; this module never silently rewrites source text or invokes the model.
- Results are **timing-matched preview candidates**, not guaranteed comprehensible or perceptually natural. Acoustic content QA, actual mixing, pauses and speaker overlaps are separate gates. No automatic human approval is required for the agent path, but failures quarantine the candidate.
- Saves new uniquely named WAV, immutable fit manifest and a separate source-binding report. Leaves original WAV, source transcript, alignments, IDs and timestamps unchanged.

## Usage in existing Colab

In the existing notebook, run **section 1** for Drive + code, then **section 7C**; fill in 4 paths/IDs. `DURATION_FIT_EXECUTE=False` plans without rendering. Set it to True only when you explicitly want offline duration fitting of an existing TTS WAV.

## Historical pilot from 12 October 2026

| Unit | Target | TTS recording | Decision (20% maximum speed adjustment) |
| --- | ---: | ---: | --- |
| tu_0001 | 1.322s | 2.64s | BLOCK: requires shorter Arabic text and new TTS |
| tu_0004 | 11.487s | 13.36s | Moderate stretch candidate |
| tu_0016 | 4.881s | 5.24s | Moderate stretch candidate |

The old WAVs are in Google Drive and are not available as local files in this development session. The module has been tested using generated synthetic WAVs; **it has not processed those three historical recordings on Colab**.

## Development checks

Run `python -m unittest discover -s tests -v`. Regression tests include WAV duration matching to sample precision, silence padding, rejection of excessive time compression, source/manifest SHA mismatch, source-timeline overlap and non-destructive behavior.