"""Ranked source loop: retry eligible source analysis and fall through only on transfer rejection.

Analysis and production are injected. The module contains no GitHub Actions or provider dependencies.
An explicit operator selection can disable fallback so a manual choice is never silently replaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RETRYABLE_ANALYSIS = frozenset({"NO_VIDEO", "VIDEO_UNAVAILABLE", "API_ERROR", "INVALID_OUTPUT"})
TRANSFER_REJECTED = "TRANSFER_REJECTED"


@dataclass
class LoopResult:
    selected: dict | None = None
    payload: object = None
    report: dict | None = None
    attempts: list = field(default_factory=list)
    rejections: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    last_payload: object = None


def analyze_with_invalid_output_retry(analyze, *, max_retries=1) -> list:
    """Retry the same analysis once only when the model output is invalid."""
    try:
        retries = max(0, int(max_retries))
    except (TypeError, ValueError):
        raise ValueError("max_retries must be a non-negative integer")

    attempts = []
    for _ in range(retries + 1):
        try:
            attempt = analyze()
        except Exception as exc:
            attempt = {"status": "INTERNAL_ERROR", "detail": f"{type(exc).__name__}: {exc}"}
        if not isinstance(attempt, dict):
            attempt = {"status": "UNKNOWN", "detail": "invalid analysis result"}
        attempts.append(attempt)
        if str(attempt.get("status") or "").upper() != "INVALID_OUTPUT":
            break
    return attempts


def run_source_loop(pool, *, limit, analyze, produce, skip_reason=None,
                    allow_transfer_fallback=True, retryable=RETRYABLE_ANALYSIS, skipped=None) -> LoopResult:
    """Try candidates in rank order; only TRANSFER_REJECTED may trigger production fallback."""
    result = LoopResult(skipped=skipped if skipped is not None else [])
    try:
        max_attempts = max(0, int(limit))
    except (TypeError, ValueError):
        raise ValueError("limit must be a non-negative integer")
    tried = 0
    for source in pool:
        if tried >= max_attempts:
            break
        candidate_id = str(source.get("id") or "").strip()
        if not candidate_id:
            result.skipped.append("candidate without a stable ID")
            continue
        reason = skip_reason(source) if skip_reason else ""
        if reason:
            result.skipped.append(f"{candidate_id}: {reason}")
            continue
        tried += 1
        attempt, payload = analyze(source)
        attempt = attempt if isinstance(attempt, dict) else {"status": "UNKNOWN", "detail": "invalid analysis result"}
        result.attempts.append(attempt)
        result.last_payload = payload
        if attempt.get("status") != "OK":
            if attempt.get("status") not in retryable:
                break
            continue
        report = produce(source, payload)
        report = report if isinstance(report, dict) else {"failure_class": "INVALID_PRODUCTION_REPORT"}
        result.selected, result.payload, result.report = source, payload, report
        if report.get("failure_class") == TRANSFER_REJECTED:
            result.rejections.append({
                "candidate_id": candidate_id,
                "reason": str(report.get("failure_detail") or report.get("detail") or "")[:300],
            })
            if allow_transfer_fallback and tried < max_attempts:
                continue
        break
    return result
