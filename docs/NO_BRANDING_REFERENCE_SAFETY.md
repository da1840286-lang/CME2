# CME2 `no_branding` safety contract

## Scope

`no_branding` is a fail-closed policy across reference preparation, prompt compilation and generated-frame QA. A missing check or an unverified image is never a successful delivery. `target_brand_allowed` remains a separate policy and is not changed by this patch.

## Existing layers retained

1. **Prompt / generation policy.** `cme/branding.py::branding_prompt_instruction` is included by `cme/compiler.py`. In `no_branding`, it forbids logos, brand names, model names, labels, readable lettering and pseudo-text anywhere, including on the product. Product references may guide silhouette, proportions, colours, materials, controls and construction; marked surfaces must be plain and unmarked.
2. **Generated-frame QA.** `cme/qa.py` keeps OCR and a required `system_no_text_policy` visual Judge check. If OCR finds no confident words, the OCR row may be `NOT_RUN` and non-required, but the visual Judge row remains required. If that Judge cannot return a valid result, the check is `NOT_RUN` and required; `cme/receipt.py::overall` therefore returns `UNVERIFIED`, not `PASS`. The reserve Judge is invoked only after the primary Judge exhausts transient retries.
3. **Reference preparation.** The reference sent to image generation must first pass OCR cleanup safeguards and an independent visual Judge check. Any ambiguous, incomplete, oversized, rejected or unverified preparation is stopped before the image provider is called.

## Reference-cleaning changes

- Rotated OCR detections with confidence below 40 are discarded. Detections from 40 through 54 remain diagnostic only and cannot authorize a mask. A rotated detection may authorize masking alone at confidence 60+, or from confidence 55 when a separate angle/variant independently corroborates it.
- Polygon groups are joined only by overlap with the fixed strongest-detection anchor. The algorithm no longer grows a union bounding box and uses that expanded rectangle to chain-merge unrelated OCR hits.
- The old rectangular reconstruction path is removed from the repair decision. `_letter_mask` must produce a coherent glyph mask; repair is confined to that mask plus a small repair margin.
- Conservative initial ceilings reject a candidate instead of repairing it when its OCR candidate region exceeds 15% of the image, its verified glyph mask exceeds 8%, or its changed-pixel area exceeds 10%.
- Pixels outside the exact repair support must remain bit-identical. Any outside-mask pixel drift rejects the candidate. The report records outside-mask MAE, similarity and changed-pixel count.
- Any failed or ambiguous case reports `REFERENCE_CLEANING_FAILED` and writes **no image** to the `target_product_unbranded_*` path. Stale restored candidates and sidecars are removed before processing. The untouched original remains available only at its original OfferCard path. A candidate that fails the independent visual Judge is deleted as well. Failed results are never cached as successful preparations.
- `cme/run.py` records a distinct blocked status and raises `ReferenceCleaningError`; `cme/smoke.py` writes `failure_class=REFERENCE_CLEANING_FAILED`. The image generator is not called.

## Calibration status — do not overgeneralize

The current ceilings are intentionally conservative and are not claimed to be calibrated for every product-photo class. The local regression set currently includes two synthetic positive text-label examples and the real SEESE product-reference failure. The positive examples produced small glyph masks (about 2.0% and 2.1% of the image); the SEESE OCR candidate covered more than 27% and its candidate glyph mask was about 22.9%. The SEESE photo is therefore rejected without altering a pixel.

These values are a fail-closed starting point for this regression, not a production-wide statistical threshold. Broaden calibration with representative real photographs (dark labels, embossed marks, curved surfaces, textured plastics, reflective metal, small labels and multiple logos) before relaxing any ceiling. A rejected reference requires a clean photo or explicit human review; the code does not infer that an uncertain OCR region is a logo.

## Judge reserve

`openrouter_free_judge_reserve` uses `dots-studio/dots-3-note-preview:free`. This was chosen over Nemotron because it is a separate model family/provider and avoids correlating the reserve with the Nemotron Brain and Analyst. OpenRouter currently lists text+image input and JSON response formatting. It is a free preview endpoint and has **not** been live-verified for CME Judge prompts, image transport, accuracy or quota. OpenRouter lists the model as scheduled to go away on 2026-12-31 and says it is not available through EU in-region routing. It is therefore a temporary reserve candidate, not a production guarantee. The primary remains `google/gemma-4-31b-it:free`; the reserve is used only after primary transient retries fail. If both fail, the result remains `UNVERIFIED` and generation/delivery cannot be reported as PASS.

## Regression: supplied SEESE reference

`tests/fixtures/seese_product_reference.jpg` is the product image from the failed FRAME run. The test requires either a safely cleaned image or a fail-closed result. With the current OCR output, the expected result is `REFERENCE_CLEANING_FAILED` with byte-for-byte equivalent RGB pixels in the prepared image; FRAME must make zero image-generation provider calls and write a blocked receipt with the distinct failure class.

## Local verification

- Full suite: **661 passed, 6 skipped, 142 subtests passed** (Pillow `getdata()` deprecation warnings remain in the reference-mask tests).
- `python -m compileall -q cme tests`: PASS.
- `adapters.json` and `settings.example.json` JSON parsing: PASS.
- `git diff --check`: PASS.
- The supplied SEESE fixture currently returns `REFERENCE_CLEANING_FAILED` with `AMBIGUOUS_LOW_CONFIDENCE_OCR`; no `target_product_unbranded_*` PNG is written, and the original source remains byte-for-byte untouched. The image-generation provider is not called in that case.
- The reserve Judge route is covered by a mocked HTTP-429 regression test. No live Judge request was made in this local validation.
