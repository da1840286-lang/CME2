"""One receipt per attempt and per scope. One place computes delivery status.

Scopes keep operations apart: an image receipt covers image and qa atoms of one shot, a video
receipt covers motion atoms. The receipt binds the exact text sent to the provider.
A requirement counts as delivered only when its text is in that text; it counts as
verified only when a check of the produced result passed. MUST requirements need VERIFIED.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, replace
from .atoms import norm

DELIVERED, NOT_DELIVERED, VERIFIED, FAILED_QA, UNVERIFIED = (
    "DELIVERED", "NOT_DELIVERED", "VERIFIED", "FAILED_QA", "UNVERIFIED")
SCOPE_CHANNELS = {"image": ("image", "qa"), "video": ("motion",)}


class ReceiptError(Exception):
    pass


class PayloadMismatch(ReceiptError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def delivery_status(atom, payload: str, qa: dict) -> str:
    """The single definition of 'was this requirement delivered/verified'.

    qa: check_id -> status (PASS / FAIL / NOT_RUN). Presence in a prompt never verifies anything.
    """
    qa_state = qa.get(atom.id)
    if atom.channel != "qa" and norm(atom.text) not in norm(payload):
        return NOT_DELIVERED
    if qa_state == "PASS":
        return VERIFIED
    if qa_state == "FAIL":
        return FAILED_QA
    return UNVERIFIED if atom.channel == "qa" else DELIVERED


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    contract_id: str
    run_id: str
    scope: str
    shot: int
    attempt: int
    sent_prompt: str
    prompt_sha256: str
    rows: tuple = ()
    checks: tuple = ()  # dicts: check_id, status, required, source, detail

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id, "contract_id": self.contract_id, "run_id": self.run_id,
            "scope": self.scope, "shot": self.shot, "attempt": self.attempt,
            "sent_prompt": self.sent_prompt, "prompt_sha256": self.prompt_sha256,
            "rows": list(self.rows), "checks": list(self.checks), "overall": overall(self),
        }


def _rows(atoms, scope, shot, payload, qa):
    channels = SCOPE_CHANNELS[scope]
    return tuple(
        {"atom_id": a.id, "priority": a.priority, "channel": a.channel,
         "status": delivery_status(a, payload, qa)}
        for a in atoms if a.channel in channels and (scope == "video" or a.shot == shot))


def _build(scope, contract_id, run_id, shot, attempt, atoms, payload) -> Receipt:
    if not contract_id or not run_id:
        raise ReceiptError("contract_id and run_id are required (no silent defaults)")
    sha = sha256_text(payload)
    rid = "drec_" + sha256_text(f"{contract_id}|{run_id}|{scope}|{shot}|{attempt}|{sha}")[:24]
    return Receipt(rid, contract_id, run_id, scope, shot, attempt, payload, sha,
                   _rows(atoms, scope, shot, payload, {}))


def build_receipt(contract_id, run_id, shot, attempt, atoms, image_prompt) -> Receipt:
    return _build("image", contract_id, run_id, shot, attempt, atoms, image_prompt)


def build_video_receipt(contract_id, run_id, attempt, atoms, motion_prompt) -> Receipt:
    return _build("video", contract_id, run_id, 0, attempt, atoms, motion_prompt)


def assert_exact_payload(receipt: Receipt, payload_text: str) -> None:
    if sha256_text(payload_text) != receipt.prompt_sha256:
        raise PayloadMismatch(
            "payload differs from the receipt-bound prompt; compile again instead of appending text")


def attach_qa(receipt: Receipt, atoms, results) -> Receipt:
    """Enrich the same receipt (identity unchanged) with QA results."""
    checks = tuple({"check_id": r.check_id, "status": r.status, "required": r.required,
                    "source": r.source, "detail": r.detail,
                     "pixels": bool(getattr(r, "pixels", False))} for r in results)
    qa = {c["check_id"]: c["status"] for c in checks}
    return replace(receipt, checks=checks,
                   rows=_rows(atoms, receipt.scope, receipt.shot, receipt.sent_prompt, qa))


def overall(receipt: Receipt) -> str:
    rows, checks = receipt.rows, receipt.checks
    must = [r for r in rows if r["priority"] == "MUST"]
    if any(r["status"] == NOT_DELIVERED for r in must):
        return NOT_DELIVERED
    if any(c["required"] and c["status"] == "FAIL" for c in checks) or any(r["status"] == FAILED_QA for r in must):
        return FAILED_QA
    # file integrity or a claim matching the offer card says nothing about the frame: only pixel checks count
    code_pass = any(c["source"] == "code" and c["status"] == "PASS" and c.get("pixels") for c in checks)
    if (not code_pass
            or any(c["required"] and c["status"] == "NOT_RUN" for c in checks)
            or any(r["status"] != VERIFIED for r in must)):
        return UNVERIFIED
    return "PASS"
