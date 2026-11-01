# CME2 — BRAIN retry staircase (local implementation)

**Base:** branch `cme2-clean-2026-10-31`, commit `05c564eda692630b2c1da7cd70450695b4f537b7` (GitHub Actions run `41324289662`).

## Implemented

1. The initial BRAIN response is still the full contract. A repair is narrowed immediately to one failed atom or one explicitly mapped structural requirement.
2. Repair routes are explicit: retry 1 uses the primary model; retry 2 uses the configured reserve; retry 3 uses the reserve again with an 80-character compact text target. The existing budget remains one initial call plus three retries. All repair schemas retain the original 160-character provider-side text allowance so structured-output providers cannot truncate at the 100/80-character target.
3. Every retry schema restricts the number and identity/role/shot of accepted atoms. Returned atoms outside the active allow-list are ignored and recorded only by ID and `text_sha256`.
4. All authorized returned atoms still pass through `parse_atoms()` and the full existing validators: claim, language, text quality, no-added-text, vertical, shot, composition, frame-budget, mechanism-link and transfer-decision checks. The dangling-word check remains in `atom_text_quality_errors()`; it was not moved into `parse_atoms()`.
5. Structural repairs are isolated. New suffix IDs are allowed only for roles explicitly required by an active structural key. Mechanism shot-order repairs are restricted to the mechanism role and required shot. Unmapped structural failures with no safe allow-list fail closed without another provider call.
6. Each accepted atom records the adapter/model and the round/strategy that produced it. The mapping is included in `brain_trace.json` and `atoms_manifest.json`. Reserve-call failures also retain the attempted adapter/model in `call.info`.
7. The OpenRouter reserve adapter is added as `openrouter_free_brain_reserve` (`google/gemma-4-26b-a4b-it:free`) using JSON-object mode. It is marked `verified: false`; the repository settings explicitly allow the first unverified call. The primary remains unchanged.

## Not changed

- Target-Fit Gate and Source Ledger rules are not bypassed or weakened.
- Atom code length limit remains 120; provider schemas retain 160-character headroom. The 100/80-character targets remain prompt guidance, not schema truncation boundaries. The quality validator still requires complete, well-formed text.
- No retry budget increase.
- No real provider call was made during local tests.
- No GitHub branch, commit, workflow or artifact was modified.

## Local verification

Command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
```

Result after the schema-headroom correction: **654 passed, 6 skipped, 142 subtests passed**. Ten Pillow `Image.getdata()` deprecation warnings are from pre-existing branding tests. `python -m compileall -q cme tests` and JSON parsing for `adapters.json` / `settings.example.json` also pass.

The real run `41324289662` was replayed in offline tests using its exact invalid BRAIN response fixture; successful correction and fail-closed exhaustion were both tested with deterministic mocked providers. The reserve model itself has not yet been live-verified with the user's API key.
