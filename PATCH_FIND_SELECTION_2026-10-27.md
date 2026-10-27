# FIND Selection integration — 27 October 2026

This branch integrates the source-selection audit corrections into the current CME2 code path, preserving the existing Target-Fit evidence fix in `cme/brain.py`. It intentionally keeps current branch interfaces rather than blindly replacing working modules with older files whose internal APIs differ.

## Integrated behavior

- Product-head anchors are resolved from product phrases (`tire inflator for cars` → `inflator`), not simply the last token. Use-context terms are checked separately.
- When source copy names a product but omits target use context, the match is explicitly marked `LOW` confidence rather than presented as a confident contextual match.
- Hashtags are not enough by themselves: `#carvacuum` needs independent source-body product/promotion evidence; verb-only tags such as `#vacuuming` are not product-noun evidence.
- Mechanism-source ranking uses `attention_rank_v1`: fixed absolute weights, coverage, period-aware recency, and engagement-rate shrinkage for small view counts. The result is an attention proxy, never commercial proof.
- Cross-run query memory refreshes the use timestamp. FIND prioritizes never-used queries and then least-recently-used queries within the existing country/offer/mode scope and 30-day TTL.
- Candidate output and Radar UI expose source confidence/context separately from `source_gate_eligible` and `target_fit_status`. The legacy `transfer_eligible` value remains a deprecated source-gate alias only.

## Kept intact

The branch's current `source_contract_run_id` artifact loader, cross-host redirect credential stripping, explicit Analyze retry, `imageio-ffmpeg` dependency, current workflow, and short `target_evidence` regression fix are preserved.

## Scope

The live discovery adapter remains TikTok SocialCrawl only. This change does not claim live YouTube/Instagram discovery or prove that an attention rank predicts sales. Validation status is determined by the GitHub Actions run for this branch.
