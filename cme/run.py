"""Produce one shot: gates -> compile -> bind receipt -> send exact text -> QA -> verdict.

Every attempt leaves a record, including blocked ones and provider failures.
"""
from __future__ import annotations
import json
from pathlib import Path
from .compiler import BudgetError, compile_image_prompt
from .branding import BRANDING_POLICIES, NO_BRANDING, SWEEP_ANGLES, prepare_unbranded_reference, reference_lettering_rule, resolve_branding_contract
from .reference_cache import ReferenceVerificationCache
from .qa import check_reference_unbranded, PRODUCT_IDENTITY_CHECK_ID, SYSTEM_NO_TEXT_ID
from .atoms import check_claims, check_language
from .receipt import assert_exact_payload, attach_qa, build_receipt, overall
from .state import accept_shot, record_attempt
from .vertical import GateError, check_vertical, offer_gate


def _write(out: Path, name: str, data: dict) -> None:
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    (out / "receipts" / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


class AttemptLimit(Exception):
    pass


def _blocked(state, out, shot, attempt, status, detail, branding_contract=None):
    record_attempt(state, shot, "", status, detail)
    payload = {"run_id": state.run_id, "shot": shot, "attempt": attempt, "overall": status, "error": detail,
               "commercial_status": "NOT_PRODUCED"}
    if branding_contract:
        payload.update(branding_contract)
    _write(out, f"blocked_{state.run_id}_s{shot:02d}_a{attempt:02d}", payload)


def _file_sha256(path):
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def qa_retry_feedback(state, receipts_dir, shot: int, branding_policy=NO_BRANDING,
                      approved_brand_marks=()) -> tuple[str, ...]:
    """Build correction directives only from the immediately preceding failed frame's required QA checks."""
    previous = next((
        item for item in reversed(getattr(state, "attempts", []))
        if item.get("shot") == shot and item.get("receipt_id")
    ), None)
    if not previous or str(previous.get("status", "")).upper() != "FAILED_QA":
        return ()

    receipt_id = str(previous.get("receipt_id") or "").strip()
    if (not receipt_id.startswith("drec_") or len(receipt_id) != 29
            or any(char not in "0123456789abcdef" for char in receipt_id[5:])):
        return ()
    try:
        receipt = json.loads((Path(receipts_dir) / f"{receipt_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if receipt.get("receipt_id") != receipt_id or receipt.get("overall") != "FAILED_QA":
        return ()

    # A failed MUST-atom row can make the receipt FAILED_QA even when its own check is
    # required=False; its failure still needs to inform the next generated prompt.
    failed_must_rows = {
        str(row.get("atom_id")) for row in receipt.get("rows", [])
        if isinstance(row, dict) and row.get("priority") == "MUST"
        and str(row.get("status", "")).upper() == "FAILED_QA"
    }
    all_failed = [
        check for check in receipt.get("checks", [])
        if isinstance(check, dict) and str(check.get("status", "")).upper() == "FAIL"
        and (check.get("required") is True or str(check.get("check_id")) in failed_must_rows)
    ]
    failed = sorted(all_failed, key=lambda check: check.get("required") is not True)[:4]
    feedback = []
    for check in failed:
        check_id = str(check.get("check_id") or "required_qa_check")
        detail = " ".join(str(check.get("detail") or "").split())[:240]
        evidence = f"Previous QA finding: {detail}. " if detail else "Previous required QA check failed. "
        if check_id == SYSTEM_NO_TEXT_ID:
            if branding_policy == NO_BRANDING:
                correction = (
                    "Render all product surfaces as plain, unmarked material. No logos, lettering, wordmarks, "
                    "badges, icons, logo-like colored symbols, or pseudo-text on the product or anywhere in the frame; "
                    "do not replace a removed mark with another symbol."
                )
            else:
                approved = ", ".join(str(mark) for mark in approved_brand_marks if str(mark).strip())
                approved_clause = (f"Preserve only these approved target-brand marks on the actual product: {approved}. "
                                   if approved else
                                   "Preserve only target-product branding already visible in the original reference and consistent with the OfferCard name/model. ")
                correction = (
                    approved_clause +
                    "Remove all source-ad, competitor, unrelated or invented branding, and all added captions, "
                    "subtitles, UI, watermarks, promotional text and pseudo-lettering. Do not erase or redesign an approved target mark."
                )
        elif check_id == PRODUCT_IDENTITY_CHECK_ID:
            if branding_policy == NO_BRANDING:
                marking_rule = "Keep all product markings blank when the no-branding policy applies."
            else:
                approved = ", ".join(str(mark) for mark in approved_brand_marks if str(mark).strip())
                marking_rule = (f"Preserve only these approved target-brand marks: {approved}."
                                if approved else
                                "Preserve only target-product markings already visible in the original reference and consistent with the OfferCard name/model.")
            correction = (
                "Treat the supplied target-product reference as the controlling specification. Match its exact "
                "silhouette, proportions, materials, colors, and visible functional parts; do not substitute a generic "
                "product or redesign the body. Preserve any missing parts identified in the QA finding. " + marking_rule
            )
        elif check.get("required") is not True:
            correction = (
                "This required scene element was not clearly visible in the previous frame. Make it unmistakable "
                "and large enough to judge on sight, while preserving all other passing MUST elements and "
                "system-owned policies."
            )
        else:
            correction = (
                "Correct the visible defect described by this required QA finding while preserving all other "
                "passing MUST elements and system-owned policies."
            )
        feedback.append(f"{check_id}. {evidence}Required correction: {correction}")
    return tuple(feedback)


def _image_input_manifest(previous_refs, product_photos, generation_product_photos=None,
                          preparation_reports=(), branding_policy=None) -> list[dict]:
    """Audit exact generator inputs and original target-reference identity."""
    if branding_policy not in BRANDING_POLICIES:
        raise ValueError("image input manifest requires the canonical OfferCard branding_policy")
    out = []
    for raw_path in previous_refs:
        path = Path(raw_path)
        out.append({"index": len(out)+1, "role": "previous_generated_frame", "filename": path.name,
                    "sha256": _file_sha256(path), "available": path.is_file()})
    actual_refs = list(generation_product_photos or product_photos)
    reports = list(preparation_reports or ())
    for index, (raw_source, raw_actual) in enumerate(zip(product_photos, actual_refs)):
        source, actual = Path(raw_source), Path(raw_actual)
        role = "target_product_photo_unbranded_reference" if branding_policy == NO_BRANDING else "target_product_photo"
        row = {"index": len(out)+1, "role": role, "filename": actual.name,
               "sha256": _file_sha256(actual), "available": actual.is_file()}
        if source.resolve() != actual.resolve():
            row["source_filename"] = source.name
            row["source_sha256"] = _file_sha256(source)
        if index < len(reports):
            row["preparation"] = _preparation_record(reports[index])
        out.append(row)
    return out


def _preparation_record(item: dict) -> dict:
    """What the receipt keeps about reference preparation: outcome, evidence and the exact inputs."""
    record = {"status": item.get("status"), "masked_regions": item.get("masked_regions", 0),
              "rejected_regions": item.get("rejected_regions", 0), "residual_regions": item.get("residual_regions", 0),
              "detected_words": list(item.get("detected_words") or [])[:12],
              "source_sha256": item.get("source_sha256"), "prepared_sha256": item.get("prepared_sha256"),
              "cached": bool(item.get("cached"))}
    if item.get("verification"):
        record["verification"] = dict(item["verification"])
    return record


def _commercial_status(provider_info: dict) -> str:
    """Anything produced by a free or never-confirmed adapter is a test artifact."""
    if provider_info and provider_info.get("tier") == "paid" and provider_info.get("verified"):
        return "CANDIDATE"
    return "TEST_ONLY"


def produce_shot(*, offer, atoms, shot, contract_id, state, out_dir, image_provider, qa_checks,
                 attempt=1, reserve=0, refs=(), vertical=None, max_attempts=0, reference_judge=None):
    """image_provider(prompt, refs) -> png bytes. qa_checks(path) -> list[CheckResult]."""
    out = Path(out_dir)
    branding_contract = resolve_branding_contract(
        offer.branding_policy, offer.approved_brand_marks, offer.name, offer.product_type
    )
    branding_policy = branding_contract["branding_policy"]
    approved_brand_marks = branding_contract["approved_brand_marks"]
    state.bind(offer.offer_id, contract_id)
    if max_attempts and sum(1 for x in state.attempts if x.get("shot") == shot and x.get("receipt_id")) >= max_attempts:
        _blocked(state, out, shot, attempt, "BLOCKED_ATTEMPTS", f"{max_attempts} attempts already used for shot {shot}", branding_contract=branding_contract)
        raise AttemptLimit(f"attempt limit {max_attempts} reached for shot {shot} in this offer/contract")
    # Defence in depth: all entry paths, including hand-supplied atoms, get the same contract checks.
    claim_errors = {k: list(v) for k, v in check_claims(atoms, offer).items()}
    for aid, msgs in check_language(atoms).items():
        claim_errors.setdefault(aid, []).extend(msgs)
    if claim_errors:
        reasons = [f"{aid}: {msg}" for aid, msgs in claim_errors.items() for msg in msgs]
        _blocked(state, out, shot, attempt, "BLOCKED_CLAIMS", "; ".join(reasons), branding_contract=branding_contract)
        raise GateError(reasons)
    if not any(a.shot == shot and a.channel == "image" and a.priority == "MUST" for a in atoms):
        reason = f"shot {shot} has no MUST image atom"
        _blocked(state, out, shot, attempt, "BLOCKED_GATE", reason, branding_contract=branding_contract)
        raise GateError([reason])
    ad_requirements = ()
    if vertical is not None:                       # gates run before any paid call
        gate = offer_gate(offer, vertical)
        reasons = list(gate.blocked) + [f"{k}: {v[0]}" for k, v in check_vertical(atoms, offer, vertical).items()]
        if reasons:
            _blocked(state, out, shot, attempt, "BLOCKED_GATE", "; ".join(reasons), branding_contract=branding_contract)
            raise GateError(reasons)
        ad_requirements = gate.requirements
    retry_feedback = qa_retry_feedback(
        state, out / "receipts", shot, branding_policy=branding_policy,
        approved_brand_marks=approved_brand_marks
    )
    if retry_feedback:
        print(f"QA retry feedback: correcting {len(retry_feedback)} required failure(s) from the previous attempt.")
    try:
        compiled = compile_image_prompt(
            atoms, shot, reserve=reserve, branding_policy=branding_policy,
            approved_brand_marks=approved_brand_marks,
            retry_feedback=retry_feedback,
        )
    except BudgetError as e:
        _blocked(state, out, shot, attempt, "BLOCKED_BUDGET", str(e), branding_contract=branding_contract)
        raise
    provider_product_refs = list(offer.photos)
    preparation_reports = []
    if branding_policy == NO_BRANDING:
        verdict_cache = ReferenceVerificationCache.from_env(reference_lettering_rule(), len(SWEEP_ANGLES))
        for index, source in enumerate(offer.photos, start=1):
            destination = out / "reference_inputs" / f"target_product_unbranded_{index:02d}.png"
            cache_restored = bool(verdict_cache and verdict_cache.restore(source, destination))
            report = prepare_unbranded_reference(source, destination)
            # A cached file may be restored but rejected by the preparation-version check. Treat it as a miss
            # in that case so the newly prepared PASS is written back under the current algorithm version.
            cache_hit = bool(cache_restored and report.get("cached") is True)
            if cache_hit and report.get("verification"):
                report["verification_cache"] = "HIT"
            preparation_reports.append(report)
            if report.get("status") in {"OCR_UNAVAILABLE", "OCR_FAILED"}:
                detail = (
                    f"no_branding cannot safely prepare target reference {Path(source).name}: "
                    f"{report.get('status')} ({report.get('error_type', 'unknown error')}); "
                    "install a working Tesseract OCR engine before generation"
                )
                _blocked(state, out, shot, attempt, "BLOCKED_BRANDING_REFERENCE", detail, branding_contract=branding_contract)
                raise GateError([detail])
            if report.get("status") == "MASK_INCOMPLETE":
                detail = (f"no_branding could not remove lettering from target reference {Path(source).name}: "
                          f"{report.get('residual_regions', 0)} region(s) still read as text after masking "
                          f"({', '.join(report.get('detected_words', [])[:6])}); supply a clean product photo")
                _blocked(state, out, shot, attempt, "BLOCKED_BRANDING_REFERENCE", detail, branding_contract=branding_contract)
                raise GateError([detail])
            # An empty OCR result is not proof of a clean image: only an independent visual check confirms it.
            verdict = report.get("verification") or check_reference_unbranded(report["path"], reference_judge, shot)
            if hasattr(verdict, "status"):
                report["verification"] = {"status": verdict.status, "evidence": str(verdict.detail)[:200]}
            if report["verification"]["status"] != "PASS":
                why = ("lettering is still visible: " if report["verification"]["status"] == "FAIL"
                       else "the prepared reference could not be confirmed: ")
                detail = (f"no_branding blocked generation for target reference {Path(source).name}; {why}"
                          f"{report['verification']['evidence']} (preparation: {report.get('status')}, "
                          f"{report.get('masked_regions', 0)} region(s) masked). Supply a clean product photo "
                          "or use target_brand_allowed.")
                _blocked(state, out, shot, attempt, "BLOCKED_BRANDING_REFERENCE", detail, branding_contract=branding_contract)
                raise GateError([detail])
            if report["verification"]["status"] == "PASS" and Path(report["path"]).with_suffix(".prep.json").is_file():
                sidecar = Path(report["path"]).with_suffix(".prep.json")
                try:
                    cached = json.loads(sidecar.read_text(encoding="utf-8"))
                    cached["verification"] = report["verification"]
                    sidecar.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")
                except (OSError, ValueError):
                    pass
            if verdict_cache and report["verification"]["status"] == "PASS" and not cache_hit:
                verdict_cache.store(source, destination)
            provider_product_refs[index-1] = report["path"]

    receipt = build_receipt(contract_id, state.run_id, shot, attempt, atoms, compiled.text)
    assert_exact_payload(receipt, compiled.text)
    image_path = out / "shots" / f"shot_{shot:02d}_a{attempt:02d}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    provider_refs = list(refs) + provider_product_refs
    base = {"lost": compiled.lost_line(), "ad_requirements": list(ad_requirements),
            "provider": {}, "commercial_status": "TEST_ONLY",
            **branding_contract,
            "qa_retry_feedback": list(retry_feedback),
            "reference_preparation": [_preparation_record(item) for item in preparation_reports],
            "image_inputs": _image_input_manifest(
                refs, offer.photos, provider_product_refs, preparation_reports, branding_policy
            )}
    try:
        image_path.write_bytes(image_provider(receipt.sent_prompt, provider_refs))
        base["provider"] = dict(getattr(image_provider, "info", {}) or {})
        base["commercial_status"] = _commercial_status(base["provider"])
        receipt = attach_qa(receipt, atoms, qa_checks(image_path))
    except Exception as e:                          # keep the diagnostics, then let the caller see the error
        base["provider"] = dict(getattr(image_provider, "info", {}) or base.get("provider", {}))
        status = f"ERROR_{type(e).__name__}"
        record_attempt(state, shot, receipt.receipt_id, status, str(e))
        _write(out, receipt.receipt_id, {**receipt.to_dict(), **base, "overall": status, "error": str(e)})
        raise
    verdict = overall(receipt)
    record_attempt(state, shot, receipt.receipt_id, verdict)
    if verdict == "PASS":
        accept_shot(state, shot, image_path, receipt.receipt_id)
    _write(out, receipt.receipt_id, {**receipt.to_dict(), **base, "image_path": str(image_path)})
    return receipt, compiled, verdict
