"""Produce one shot: gates -> compile -> bind receipt -> send exact text -> QA -> verdict.

Every attempt leaves a record, including blocked ones and provider failures.
"""
from __future__ import annotations
import json
from pathlib import Path
from .compiler import BudgetError, compile_image_prompt
from .branding import NO_BRANDING, prepare_unbranded_reference, resolved_brand_marks
from .qa import check_reference_unbranded
from .atoms import check_claims, check_language
from .receipt import assert_exact_payload, attach_qa, build_receipt, overall
from .state import accept_shot, record_attempt
from .vertical import GateError, check_vertical, offer_gate


def _write(out: Path, name: str, data: dict) -> None:
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    (out / "receipts" / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


class AttemptLimit(Exception):
    pass


def _blocked(state, out, shot, attempt, status, detail):
    record_attempt(state, shot, "", status, detail)
    _write(out, f"blocked_{state.run_id}_s{shot:02d}_a{attempt:02d}",
           {"run_id": state.run_id, "shot": shot, "attempt": attempt, "overall": status, "error": detail,
            "commercial_status": "NOT_PRODUCED"})


def _file_sha256(path):
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _image_input_manifest(previous_refs, product_photos, generation_product_photos=None,
                          preparation_reports=(), branding_policy="target_brand_allowed") -> list[dict]:
    """Audit exact generator inputs and original target-reference identity."""
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
    state.bind(offer.offer_id, contract_id)
    if max_attempts and sum(1 for x in state.attempts if x.get("shot") == shot and x.get("receipt_id")) >= max_attempts:
        _blocked(state, out, shot, attempt, "BLOCKED_ATTEMPTS", f"{max_attempts} attempts already used for shot {shot}")
        raise AttemptLimit(f"attempt limit {max_attempts} reached for shot {shot} in this offer/contract")
    # Defence in depth: all entry paths, including hand-supplied atoms, get the same contract checks.
    claim_errors = {k: list(v) for k, v in check_claims(atoms, offer).items()}
    for aid, msgs in check_language(atoms).items():
        claim_errors.setdefault(aid, []).extend(msgs)
    if claim_errors:
        reasons = [f"{aid}: {msg}" for aid, msgs in claim_errors.items() for msg in msgs]
        _blocked(state, out, shot, attempt, "BLOCKED_CLAIMS", "; ".join(reasons))
        raise GateError(reasons)
    if not any(a.shot == shot and a.channel == "image" and a.priority == "MUST" for a in atoms):
        reason = f"shot {shot} has no MUST image atom"
        _blocked(state, out, shot, attempt, "BLOCKED_GATE", reason)
        raise GateError([reason])
    ad_requirements = ()
    if vertical is not None:                       # gates run before any paid call
        gate = offer_gate(offer, vertical)
        reasons = list(gate.blocked) + [f"{k}: {v[0]}" for k, v in check_vertical(atoms, offer, vertical).items()]
        if reasons:
            _blocked(state, out, shot, attempt, "BLOCKED_GATE", "; ".join(reasons))
            raise GateError(reasons)
        ad_requirements = gate.requirements
    try:
        compiled = compile_image_prompt(
            atoms, shot, reserve=reserve, branding_policy=offer.branding_policy,
            approved_brand_marks=resolved_brand_marks(offer.branding_policy, offer.approved_brand_marks,
                                                      offer.name, offer.product_type),
        )
    except BudgetError as e:
        _blocked(state, out, shot, attempt, "BLOCKED_BUDGET", str(e))
        raise
    provider_product_refs = list(offer.photos)
    preparation_reports = []
    if offer.branding_policy == NO_BRANDING:
        for index, source in enumerate(offer.photos, start=1):
            destination = out / "reference_inputs" / f"target_product_unbranded_{index:02d}.png"
            report = prepare_unbranded_reference(source, destination)
            preparation_reports.append(report)
            if report.get("status") in {"OCR_UNAVAILABLE", "OCR_FAILED"}:
                detail = (
                    f"no_branding cannot safely prepare target reference {Path(source).name}: "
                    f"{report.get('status')} ({report.get('error_type', 'unknown error')}); "
                    "install a working Tesseract OCR engine before generation"
                )
                _blocked(state, out, shot, attempt, "BLOCKED_BRANDING_REFERENCE", detail)
                raise GateError([detail])
            if report.get("status") == "MASK_INCOMPLETE":
                detail = (f"no_branding could not remove lettering from target reference {Path(source).name}: "
                          f"{report.get('residual_regions', 0)} region(s) still read as text after masking "
                          f"({', '.join(report.get('detected_words', [])[:6])}); supply a clean product photo")
                _blocked(state, out, shot, attempt, "BLOCKED_BRANDING_REFERENCE", detail)
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
                _blocked(state, out, shot, attempt, "BLOCKED_BRANDING_REFERENCE", detail)
                raise GateError([detail])
            if report["verification"]["status"] == "PASS" and Path(report["path"]).with_suffix(".prep.json").is_file():
                sidecar = Path(report["path"]).with_suffix(".prep.json")
                try:
                    cached = json.loads(sidecar.read_text(encoding="utf-8"))
                    cached["verification"] = report["verification"]
                    sidecar.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")
                except (OSError, ValueError):
                    pass
            provider_product_refs[index-1] = report["path"]

    receipt = build_receipt(contract_id, state.run_id, shot, attempt, atoms, compiled.text)
    assert_exact_payload(receipt, compiled.text)
    image_path = out / "shots" / f"shot_{shot:02d}_a{attempt:02d}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    provider_refs = list(refs) + provider_product_refs
    base = {"lost": compiled.lost_line(), "ad_requirements": list(ad_requirements),
            "provider": {}, "commercial_status": "TEST_ONLY",
            "branding_policy": offer.branding_policy,
            "reference_preparation": [_preparation_record(item) for item in preparation_reports],
            "image_inputs": _image_input_manifest(
                refs, offer.photos, provider_product_refs, preparation_reports, offer.branding_policy
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
