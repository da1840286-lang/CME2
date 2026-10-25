"""First live test: one Brain call and/or one frame with honest QA. Everything is marked TEST_ONLY.

    python -m cme.smoke brain --offer examples/offer_example.json --settings settings.json --out out
    python -m cme.smoke frame --offer examples/offer_example.json --settings settings.json --out out
    python -m cme.smoke all   --offer examples/offer_example.json --settings settings.json --out out

Keys come from the environment: OPENROUTER_API_KEY, HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from .atoms import check_claims, check_language, parse_atoms
from .branding import allowed_brand_tokens, resolved_brand_marks
from .brain import (BrainError, TransferRejected, atom_budget_errors, build_source_transfer_context,
                    derive_mechanism_links, missing_mechanism_links, reserved_system_atom_errors,
                    validate_transfer_decision, visual_structure_errors, request_atoms, request_transfer_fit)
from .compiler import BudgetError
from .offer import OfferError, load_offer
from .providers import CallLog, InfraError, ProviderCallError, ProviderPolicyError
from .qa import brand_tokens, make_qa_checks, validate_product_references
from .registry import SettingsError, load_catalog, load_settings, make_brain_call, make_image_provider, make_judge
from .run import AttemptLimit, produce_shot
from .state import StateMismatch, load_state, new_run, save_state
from .vertical import GateError, check_vertical, load_vertical

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRIEF = "One vertical 9:16 ad frame: the product used in a realistic home scene, product clearly visible."


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def offer_fingerprint(offer) -> tuple[str, list]:
    photo_hashes = []
    for photo in offer.photos:
        path = Path(photo)
        if not path.exists():
            raise OfferError(f"product photo missing before contract binding: {photo}")
        photo_hashes.append({"name": path.name, "sha256": _sha256_bytes(path.read_bytes())})
    payload = {"offer_id": offer.offer_id, "name": offer.name, "product_type": offer.product_type,
               "verified_claims": list(offer.verified_claims), "forbidden_claims": list(offer.forbidden_claims),
               "photo_hashes": photo_hashes, "platform": offer.platform, "geo": offer.geo,
               "product_category": offer.product_category, "confirmations": list(offer.confirmations),
               "branding_policy": getattr(offer, "branding_policy", "target_brand_allowed"),
               "approved_brand_marks": list(getattr(offer, "approved_brand_marks", ()))}
    return _sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")), photo_hashes


def contract_id_for(atoms_json: list, offer) -> str:
    fingerprint, _ = offer_fingerprint(offer)
    payload = {"atoms": atoms_json, "offer_fingerprint": fingerprint}
    return "contract-" + _sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))[:16]


def validate_atoms_manifest(atoms_path, manifest_path, atoms_json, offer, shot=None):
    path, manifest_file = Path(atoms_path), Path(manifest_path)
    if not manifest_file.exists():
        raise BrainError({"#manifest": ["atoms_manifest.json is missing; rerun Brain for this offer"]})
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BrainError({"#manifest": [f"cannot read atoms manifest: {e}"]}) from e
    if manifest.get("schema_version") != 1:
        raise BrainError({"#manifest": ["unsupported atoms manifest schema_version"]})
    fingerprint, photo_hashes = offer_fingerprint(offer)
    expected = {
        "offer_id": offer.offer_id,
        "atoms_sha256": _sha256_bytes(path.read_bytes()),
        "offer_fingerprint": fingerprint,
        "photo_hashes": photo_hashes,
        "contract_id": contract_id_for(atoms_json, offer),
    }
    errors = {k: [f"manifest mismatch for {k}"] for k, v in expected.items() if manifest.get(k) != v}
    requested = manifest.get("requested_shots")
    if type(requested) is not int or requested < 1:
        errors["#requested_shots"] = ["manifest requested_shots must be a positive integer"]
    elif shot is not None and shot > requested:
        errors["#shot"] = [f"shot {shot} is outside the Brain contract's requested shot range"]
    if type(requested) is int and any(type(a.get("shot")) is not int or a["shot"] > requested
                                      for a in atoms_json if isinstance(a, dict)):
        errors["#atoms"] = ["atoms contain a shot outside the Brain contract's requested range"]
    decision = manifest.get("transfer_decision")
    if manifest.get("source_contract_sha256"):
        if not isinstance(decision, dict) or decision.get("transferable") is not True:
            errors["#transfer_decision"] = ["source-bound manifest must contain transfer_decision.transferable=true"]
        decision_errors = validate_transfer_decision(decision, atoms_json)
        if decision_errors:
            errors["#transfer_decision"] = decision_errors
    elif decision is not None:
        decision_errors = validate_transfer_decision(decision, atoms_json)
        if decision_errors:
            errors["#transfer_decision"] = decision_errors
    links = manifest.get("mechanism_links") or []
    lost_links = missing_mechanism_links(atoms_json, links)
    if lost_links:
        errors["#mechanism"] = [f"mechanism link '{n}' is no longer carried by a MUST atom" for n in lost_links]
    errors.update(reserved_system_atom_errors(atoms_json))
    if type(requested) is int and requested >= 1:
        errors.update(visual_structure_errors(atoms_json, requested, links))
        errors.update(atom_budget_errors(atoms_json, requested))
    if errors:
        raise BrainError(errors)
    return manifest


def atoms_summary(atoms) -> str:
    rows = {}
    for a in atoms:
        rows.setdefault((a.shot, a.channel), {"MUST": 0, "SHOULD": 0, "MAY": 0})[a.priority] += 1
    return "\n".join(f"  shot {s} / {c}: " + ", ".join(f"{k}={v}" for k, v in d.items()) for (s, c), d in sorted(rows.items()))


def write_run_result(out: Path, command: str, exit_code: int, failure_class: str, message: str, verdict=None) -> None:
    """Write diagnostic classification without changing the workflow's established exit codes."""
    payload = {"command": command, "exit_code": exit_code, "failure_class": failure_class, "status": "OK" if exit_code == 0 else "ERROR", "message": str(message or "")[:1000]}
    if verdict: payload["verdict"] = verdict
    try: (Path(out) / "run_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc: print(f"Не удалось записать run_result.json: {exc}")


def image_seed_for_attempt(base_seed: int, attempt: int, contract_id: str) -> int:
    """Use a reproducible per-contract seed, then advance it for each retry of that contract."""
    if type(base_seed) is not int:
        raise SettingsError("image adapter seed must be an integer")
    if type(attempt) is not int or attempt < 1:
        raise SettingsError("frame attempt must be a positive integer")
    suffix = str(contract_id).rsplit("-", 1)[-1]
    try:
        contract_offset = int(suffix[-8:], 16) % 1_000_000
    except ValueError as exc:
        raise SettingsError("frame contract_id must end with a hexadecimal digest") from exc
    # Add one so a new contract never reuses the old catalog seed by default.
    return base_seed + contract_offset + 1 + attempt - 1


def run_brain(args, offer, settings, catalog, vertical, log, run_id, ctx, out):
    atoms_path, manifest_path = out / "atoms.json", out / "atoms_manifest.json"
    source_transfer = None
    if args.source_contract:
        source_path = Path(args.source_contract)
        try:
            source_payload = json.loads(source_path.read_text(encoding="utf-8"))
            source_transfer = build_source_transfer_context(source_payload)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            # Reject malformed context before touching a previously accepted atom contract.
            raise BrainError({"#source_contract": [f"cannot load supported source-transfer JSON {source_path}: {e}"]}) from e
    # A failed Brain call must never leave an older contract looking current.
    manifest_path.unlink(missing_ok=True)
    atoms_path.unlink(missing_ok=True)
    call = make_brain_call(settings, catalog, log, run_id, **ctx)
    target_fit_gate = None
    # Every gate and Brain round is kept in brain_trace.json, also when the run dies in the middle: a transient
    # provider error after two paid rounds must not erase why those rounds were not accepted.
    trace = {"run_id": run_id, "gate_rounds": [], "brain_rounds": []}
    trace_path = out / "brain_trace.json"
    trace_path.unlink(missing_ok=True)
    try:
        if source_transfer is not None:
            # Gate before atom generation: an incompatible source should be rejected without
            # spending a second model call on a production contract. Preserve the decision as an audit artifact.
            target_fit_gate = request_transfer_fit(offer, args.brief, source_transfer, call, trace_out=trace["gate_rounds"])
            (out / "target_fit_gate.json").write_text(
                json.dumps(target_fit_gate, ensure_ascii=False, indent=2), encoding="utf-8")
            if target_fit_gate.get("transferable") is not True:
                raise TransferRejected(target_fit_gate)
            source_transfer = {**source_transfer, "target_fit_gate": target_fit_gate}
        transfer_decision = {}
        atoms = request_atoms(offer, args.brief, args.shots, call, vertical=vertical,
                              source_transfer=source_transfer, decision_out=transfer_decision,
                              rounds_out=trace["brain_rounds"])
    finally:
        try:
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            print(f"Не удалось записать brain_trace.json: {exc}")
    data = [a.to_dict() for a in atoms]
    atoms_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    fingerprint, photo_hashes = offer_fingerprint(offer)
    manifest = {"schema_version": 1, "run_id": run_id, "offer_id": offer.offer_id,
                "atoms_sha256": _sha256_bytes(atoms_path.read_bytes()), "offer_fingerprint": fingerprint,
                "photo_hashes": photo_hashes, "contract_id": contract_id_for(data, offer),
                "requested_shots": args.shots,
                "mechanism_links": [l["link"] for l in derive_mechanism_links(source_transfer)],
                "target_fit_gate": target_fit_gate,
                "brief_sha256": _sha256_bytes(args.brief.encode("utf-8")),
                "source_contract_sha256": _sha256_bytes(Path(args.source_contract).read_bytes()) if args.source_contract else None,
                "model_id": call.info.get("model_id"), "responded_model": call.info.get("responded_model")}
    if transfer_decision:
        manifest["transfer_decision"] = transfer_decision
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Brain: {len(atoms)} атомов, модель {call.info.get('model_id')} (ответила: {call.info.get('responded_model') or 'не сообщено'})")
    print(atoms_summary(atoms))
    return data


def run_frame(args, offer, settings, catalog, vertical, log, run_id, ctx, out, atoms_data, atom_provenance="external_unbound"):
    parsed = parse_atoms(atoms_data)
    if parsed.errors:
        raise BrainError(parsed.errors)
    atoms = parsed.atoms
    claim_errors = check_claims(atoms, offer)
    vertical_errors = check_vertical(atoms, offer, vertical)
    merged_errors = {k: list(v) for k, v in claim_errors.items()}
    for aid, reasons in check_language(atoms).items():
        merged_errors.setdefault(aid, []).extend(reasons)
    for aid, reasons in vertical_errors.items():
        merged_errors.setdefault(aid, []).extend(reasons)
    if merged_errors:
        raise BrainError(merged_errors)
    if not any(a.shot == args.shot and a.channel == "image" and a.priority == "MUST" for a in atoms):
        raise BrainError({f"#shot_{args.shot}": ["selected shot needs at least one MUST image atom"]})
    contract_id = contract_id_for(atoms_data, offer)
    prev_path = Path(args.previous_state) if args.previous_state else out / "state.json"
    previous = load_state(prev_path) if prev_path.exists() else None
    state = new_run(run_id, previous, offer.offer_id, contract_id)
    for note in state.carry_notes:
        print("перенос:", note)
    # The normal CLI (empty ctx) must never spend provider quota on the known solid-colour sample placeholder.
    # Tests inject a transport/key in ctx and use synthetic fixtures; they exercise the QA logic separately.
    if not ctx:
        reference_issues = validate_product_references(offer.photos)
        if reference_issues:
            raise OfferError("Product-photo preflight blocked generation: " + "; ".join(reference_issues))
    image_seed = None
    image_spec = catalog.get(settings.image_adapter)
    if image_spec and image_spec.kind == "image" and image_spec.provider == "hf_space":
        # A retry is a new, reproducible candidate, not the same fixed-seed image again.
        image_seed = image_seed_for_attempt(image_spec.params.get("seed", 0), args.attempt, contract_id)
    provider = make_image_provider(settings, catalog, log, run_id, image_seed=image_seed, **ctx)
    judge = make_judge(settings, catalog, log, run_id, **ctx) if settings.judge_adapter else None
    transfer_decision = None
    manifest_path = out / "atoms_manifest.json"
    if atom_provenance == "verified_manifest" and manifest_path.is_file():
        try:
            transfer_decision = json.loads(manifest_path.read_text(encoding="utf-8")).get("transfer_decision")
        except (OSError, ValueError):
            transfer_decision = None
    approved_marks = resolved_brand_marks(offer.branding_policy, offer.approved_brand_marks, offer.name, offer.product_type)
    qa = make_qa_checks(
        atoms, args.shot, judge_fn=judge, ocr_required=not args.no_ocr_required,
        verified_claims=offer.verified_claims, product_references=offer.photos,
        allowed_text=allowed_brand_tokens(
            offer.branding_policy, approved_marks, offer.name, offer.product_type
        ),
        transfer_decision=transfer_decision, branding_policy=offer.branding_policy,
        approved_brand_marks=approved_marks,
    )
    refs = [state.shots[args.shot - 1]["image_path"]] if (args.shot - 1) in state.shots else []
    try:
        receipt, compiled, verdict = produce_shot(
            offer=offer, atoms=atoms, shot=args.shot, contract_id=contract_id, state=state, out_dir=out,
            image_provider=provider, qa_checks=qa, attempt=args.attempt, refs=refs, vertical=vertical,
            max_attempts=settings.max_attempts_per_shot, reference_judge=judge)
    finally:
        save_state(state, out / "state.json")
    receipt_path = out / "receipts" / f"{receipt.receipt_id}.json"
    rep = json.loads(receipt_path.read_text(encoding="utf-8"))
    rep["atom_provenance"] = atom_provenance
    if atom_provenance != "verified_manifest":
        rep["commercial_status"] = "TEST_ONLY"
    receipt_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Кадр {args.shot}: вердикт {verdict} | статус {rep['commercial_status']} | {compiled.lost_line()}")
    print(f"  изображение: {rep['image_path']}\n  receipt: {out / 'receipts' / (receipt.receipt_id + '.json')}")
    for c in receipt.checks:
        print(f"  проверка {c['check_id']}: {c['status']} ({c['source']}) {c['detail']}")
    if rep["ad_requirements"]:
        print("  требования к рекламе:", ", ".join(rep["ad_requirements"]))
    return verdict


def main(argv=None, ctx=None) -> int:
    ap = argparse.ArgumentParser(prog="cme.smoke")
    ap.add_argument("command", choices=["brain", "frame", "all"])
    ap.add_argument("--offer", default=str(ROOT / "examples" / "offer_example.json"))
    ap.add_argument("--catalog", default=str(ROOT / "adapters.json"))
    ap.add_argument("--settings", default=str(ROOT / "settings.json"))
    ap.add_argument("--vertical", default=str(ROOT / "verticals" / "physical_goods_us.json"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--brief", default=DEFAULT_BRIEF)
    ap.add_argument("--source-contract", help="Optional old CME source/transfer/rebuild JSON (or saved Radar analysis) to atomize")
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--atoms")
    ap.add_argument("--previous-state")
    ap.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    ap.add_argument("--no-ocr-required", action="store_true",
                    help="record the no-text check as not required (use only when no OCR engine is installed)")
    args = ap.parse_args(argv)
    ctx = ctx or {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        offer = load_offer(args.offer)
        settings, catalog = load_settings(args.settings), load_catalog(args.catalog)
        vertical = load_vertical(args.vertical)
        log = CallLog(out / "calls.jsonl")
        data = None
        atom_provenance = "verified_manifest"
        if args.command in ("brain", "all"):
            data = run_brain(args, offer, settings, catalog, vertical, log, args.run_id, ctx, out)
        if args.command in ("frame", "all"):
            if data is None:
                src = Path(args.atoms) if args.atoms else out / "atoms.json"
                data = json.loads(src.read_text(encoding="utf-8"))
                if src.resolve() == (out / "atoms.json").resolve():
                    validate_atoms_manifest(src, out / "atoms_manifest.json", data, offer, shot=args.shot)
                    atom_provenance = "verified_manifest"
                else:
                    atom_provenance = "external_unbound"
            else:
                validate_atoms_manifest(out / "atoms.json", out / "atoms_manifest.json", data, offer, shot=args.shot)
            # `all` must honor the full Brain contract. Generate sequentially so each accepted frame
            # becomes the visual reference for the next shot; stop rather than chain a failed/unverified frame.
            shot_numbers = range(1, args.shots + 1) if args.command == "all" else (args.shot,)
            for shot_number in shot_numbers:
                args.shot = shot_number
                frame_run_id = args.run_id if shot_number == 1 else f"{args.run_id}-s{shot_number:02d}"
                # Retry only rejected/undelivered frames; UNVERIFIED is a checker failure, not a frame failure.
                tries = min(2, max(1, int(settings.max_attempts_per_shot or 1))) if args.command == "all" else 1
                for attempt_number in range(1, tries + 1):
                    args.attempt = attempt_number
                    verdict = run_frame(args, offer, settings, catalog, vertical, log, frame_run_id, ctx, out, data, atom_provenance)
                    if verdict not in ("FAILED_QA", "NOT_DELIVERED") or attempt_number == tries:
                        break
                    print(f"Кадр {shot_number}: попытка {attempt_number} не прошла QA ({verdict}); новая попытка с другим seed.")
                if verdict == "UNVERIFIED":
                    write_run_result(out, args.command, 6, "QA_UNVERIFIED", "Required QA checks did not produce a verifiable verdict.", verdict)
                    return 6
                if verdict in ("FAILED_QA", "NOT_DELIVERED"):
                    failure_class = "NOT_DELIVERED" if verdict == "NOT_DELIVERED" else "VISUAL_QA_FAILED"
                    write_run_result(out, args.command, 7, failure_class, "A required MUST condition was not delivered or did not pass QA.", verdict)
                    return 7
    except FileNotFoundError as e:
        message = f"Не найден файл: {e.filename}. Скопируйте settings.example.json в settings.json и поправьте его."
        print(message); write_run_result(out,args.command,3,"SETUP_OR_FILE_ERROR",message); return 3
    except SettingsError as e:
        message=f"Ошибка настроек: {e}"; print(message); write_run_result(out,args.command,3,"SETTINGS_ERROR",message); return 3
    except InfraError as e:
        message=f"Сбой инфраструктуры (квота, лимит, сеть): {e}\nЗапись попытки сохранена в {out}. Повторите позже или объявите fallback."
        print(message); write_run_result(out,args.command,2,"INFRASTRUCTURE_ERROR",message); return 2
    except (GateError, BudgetError, AttemptLimit, ProviderPolicyError, StateMismatch, OfferError) as e:
        message=f"Остановлено до платного вызова или политикой: {e}"; print(message); write_run_result(out,args.command,3,"CONTRACT_OR_PREFLIGHT_BLOCKED",message); return 3
    except TransferRejected as e:
        message = f"TRANSFER_REJECTED: source mechanism cannot be transferred to this target product: {e.decision.get('reason', '')}"
        print(message)
        write_run_result(out, args.command, 4, "TRANSFER_REJECTED", message)
        return 4
    except BrainError as e:
        message=f"Brain не вернул валидные атомы: {e.errors}"; print(message)
        error_text=json.dumps(e.errors,ensure_ascii=False)
        failure_class=("VISUAL_CONTRACT_INVALID" if any(t in error_text for t in ("COMPOSITION_REQUIRED","MECHANISM_SHOT_ORDER","SYSTEM_OWNED_QA","transfer_decision")) else "BRAIN_OUTPUT_INVALID")
        write_run_result(out,args.command,4,failure_class,message)
        return 4
    except ProviderCallError as e:
        message=f"Провайдер ответил ошибкой: {e}"; print(message); write_run_result(out,args.command,5,"PROVIDER_RESPONSE_ERROR",message); return 5
    if args.command == "brain":
        write_run_result(out,args.command,0,"BRAIN_CONTRACT_READY","Brain contract and manifest passed validation.")
    else:
        write_run_result(out,args.command,0,"DELIVERED_PASS","All requested frames passed the configured delivery and QA gates.","PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
