"""Per-run candidate ledger: one row for every FIND candidate, whatever happened to it.

The ledger is the data that lets an operator pick a score threshold later (after 15-20 runs) instead of guessing:
score, scorer action, relevance, rank, WEAK_SOURCE flag, analysis status, Target-Fit Gate verdict and final QA.
It reads facts only; it never changes selection.
"""
from __future__ import annotations

from .select import SourceThreshold, block_reason, rank_transfer_candidates, weak_source_reasons

# Failure classes that can only be reached after the Target-Fit Gate approved the transfer.
_GATE_PASSED_CLASSES = {"DELIVERED_PASS", "QA_UNVERIFIED", "NOT_DELIVERED", "VISUAL_QA_FAILED"}


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build_ledger(candidates, threshold: SourceThreshold | None = None, *, attempts=(), rejections=(),
                 selected_id: str = "", failure_class: str = "", verdicts=()) -> list:
    ranked, _ = rank_transfer_candidates(candidates, threshold)
    rank_of = {str(item.get("id")): index for index, item in enumerate(ranked, start=1)}
    last_analysis = {}
    for item in attempts or []:
        if isinstance(item, dict) and item.get("candidate_id"):
            last_analysis[str(item["candidate_id"])] = str(item.get("status") or "UNKNOWN")
    rejected = {str(item.get("candidate_id")): str(item.get("reason") or "") for item in rejections or []
                if isinstance(item, dict)}
    qa = ", ".join(sorted({str(v.get("overall")) for v in verdicts or [] if isinstance(v, dict) and v.get("overall")}))
    rows = []
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "")
        score = item.get("source_score") if isinstance(item.get("source_score"), dict) else {}
        check = item.get("source_check") if isinstance(item.get("source_check"), dict) else {}
        block = block_reason(item)
        weak = weak_source_reasons(item, threshold) if not block else []
        if cid in rejected:
            gate, gate_reason = "REJECTED", rejected[cid]
        elif cid == selected_id and failure_class in _GATE_PASSED_CLASSES:
            gate, gate_reason = "ACCEPTED", ""
        elif cid == selected_id:
            gate, gate_reason = "UNKNOWN", f"run ended with {failure_class or 'no result'} before a gate verdict was recorded"
        else:
            gate, gate_reason = "NOT_REACHED", ""
        rows.append({
            "candidate_id": cid, "brand": item.get("brand_name") or item.get("brand"), "title": item.get("title"),
            "final_score": _num(score.get("final_score")), "scorer_action": score.get("action"),
            "relevance": check.get("classification"),
            "eligible": not block and cid in rank_of, "block_reason": block or None,
            "rank": rank_of.get(cid), "weak_source": bool(weak), "weak_reasons": weak,
            "analysis_status": last_analysis.get(cid, "NOT_TRIED"),
            "gate_verdict": gate, "gate_reason": gate_reason,
            "qa_verdict": qa if cid == selected_id and qa else None,
            "selected": cid == selected_id, "outcome": failure_class if cid == selected_id else None,
        })
    return rows
