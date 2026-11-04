"""Run state carried between runs. Every record is tagged with the run, offer and contract that made it."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


ATTEMPT_CONSUMING_STATUSES = frozenset({"PASS", "FAILED_QA", "UNVERIFIED", "NOT_DELIVERED"})


def attempt_consumed(record: dict) -> bool:
    """Whether a recorded frame operation produced an image candidate.

    New records carry an explicit `generated` flag. Legacy records are classified from
    their final QA status; provider/infrastructure errors do not consume an image attempt.
    """
    if not isinstance(record, dict):
        return False
    generated = record.get("generated")
    if type(generated) is bool:
        return generated
    return str(record.get("status") or "").upper() in ATTEMPT_CONSUMING_STATUSES


def count_attempts(records, shot: int | None = None) -> int:
    """Count generated frame candidates, optionally for one shot."""
    return sum(1 for record in (records or ())
               if isinstance(record, dict)
               and (shot is None or record.get("shot") == shot)
               and attempt_consumed(record))


class StateMismatch(Exception):
    pass


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class RunState:
    run_id: str
    previous_run_id: str = ""
    offer_id: str = ""
    contract_id: str = ""
    shots: dict = field(default_factory=dict)      # shot -> {image_path, sha256, receipt_id, run_id, offer_id, contract_id}
    attempts: list = field(default_factory=list)   # every attempt carries its run_id
    carry_notes: list = field(default_factory=list)

    def bind(self, offer_id: str, contract_id: str) -> None:
        """Pin the state to one offer and contract; mixing them is an error, not a silent reuse."""
        if self.offer_id and self.offer_id != offer_id:
            raise StateMismatch(f"state belongs to offer {self.offer_id!r}, got {offer_id!r}")
        if self.contract_id and self.contract_id != contract_id:
            raise StateMismatch(f"state belongs to contract {self.contract_id!r}, got {contract_id!r}")
        self.offer_id, self.contract_id = offer_id, contract_id

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "previous_run_id": self.previous_run_id, "offer_id": self.offer_id,
                "contract_id": self.contract_id, "shots": {str(k): v for k, v in self.shots.items()},
                "attempts": self.attempts, "carry_notes": self.carry_notes}


def new_run(run_id: str, previous: "RunState | None" = None, offer_id: str = "", contract_id: str = "") -> RunState:
    st = RunState(run_id, previous.run_id if previous else "", offer_id, contract_id)
    same_contract = bool(offer_id and contract_id and previous and
                         previous.offer_id == offer_id and previous.contract_id == contract_id)
    if same_contract:
        # Quota protection must survive a fresh CLI run for the same exact offer/contract.
        st.attempts = [dict(a) for a in previous.attempts]
    if previous:
        # Accepted images may still be carried for legacy unbound states, but attempts only
        # carry across an exact, non-empty offer/contract match above.
        for shot, rec in previous.shots.items():
            p = Path(rec["image_path"])
            if rec.get("offer_id", "") != offer_id or rec.get("contract_id", "") != contract_id:
                st.carry_notes.append(f"shot {shot} not carried: made for another offer or contract")
            elif not p.exists() or file_sha256(p) != rec["sha256"]:
                st.carry_notes.append(f"shot {shot} not carried: file missing or hash changed")
            else:
                st.shots[shot] = dict(rec)
        if previous.attempts and not same_contract:
            st.carry_notes.append("attempt history not carried: offer or contract changed")
    return st


def record_attempt(state: RunState, shot: int, receipt_id: str, status: str, error: str = "",
                   generated: bool | None = None) -> None:
    if generated is None:
        generated = str(status or "").upper() in ATTEMPT_CONSUMING_STATUSES
    state.attempts.append({"run_id": state.run_id, "shot": shot, "receipt_id": receipt_id,
                           "status": status, "error": error, "generated": bool(generated)})


def accept_shot(state: RunState, shot: int, image_path, receipt_id: str) -> None:
    state.shots[shot] = {"image_path": str(image_path), "sha256": file_sha256(image_path),
                         "receipt_id": receipt_id, "run_id": state.run_id,
                         "offer_id": state.offer_id, "contract_id": state.contract_id}


def this_run_attempts(state: RunState) -> list:
    return [a for a in state.attempts if a["run_id"] == state.run_id]


def save_state(state: RunState, path) -> None:
    Path(path).write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


def load_state(path) -> RunState:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunState(d["run_id"], d.get("previous_run_id", ""), d.get("offer_id", ""), d.get("contract_id", ""),
                    {int(k): v for k, v in d.get("shots", {}).items()},
                    d.get("attempts", []), d.get("carry_notes", []))
