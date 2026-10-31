"""Canonical Analyze -> Brain source contract for separately dispatched workflows."""
from __future__ import annotations

import hashlib
import json
from typing import Any


_SCHEMA = "cme.radar.source_contract.v1"


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _publication_timestamp(candidate: dict) -> Any:
    direct = candidate.get("published_at")
    if direct not in (None, ""):
        return direct
    legacy = _mapping(candidate.get("legacy_record"))
    direct = legacy.get("published_at")
    if direct not in (None, ""):
        return direct
    raw = _mapping(legacy.get("raw"))
    post = _mapping(raw.get("post"))
    for key in ("published_at", "create_time", "createTime", "created_at", "createdAt", "timestamp"):
        if post.get(key) not in (None, ""):
            return post[key]
    return None


def build_source_contract(analysis: dict, selected_candidate: dict | None = None) -> dict:
    """Build a complete, self-contained handoff contract from the exact Analyze artifact.

    The full `result` is deliberately kept at the contract root because Brain's existing
    `build_source_transfer_context` consumes that shape. Provider metrics and their per-field
    provenance are preserved separately from commercial/economic verification status.
    """
    if not isinstance(analysis, dict):
        raise ValueError("Analyze output must be a JSON object")
    if str(analysis.get("status") or "").upper() != "OK":
        raise ValueError("Only a successful Analyze output can become a source contract")
    result = analysis.get("result")
    if not isinstance(result, dict):
        raise ValueError("Analyze output is missing its structured result")
    required_sections = ("mechanism_hypothesis", "transferability")
    missing = [name for name in required_sections if not isinstance(result.get(name), (dict, str))]
    if missing:
        raise ValueError("Analyze result is missing required transfer sections: " + ", ".join(missing))

    candidate = _mapping(selected_candidate)
    legacy = _mapping(candidate.get("legacy_record"))
    source_selection = _mapping(candidate.get("source_selection"))
    performance = _mapping(candidate.get("performance_evidence"))
    commercial = _mapping(candidate.get("commercial_evidence"))
    source = _mapping(candidate.get("source_check"))
    lane = str(legacy.get("discovery_source") or "").lower()
    if "sponsored_search" in lane:
        source_lane = "SPONSORED_SEARCH"
    elif "ads_top" in lane:
        source_lane = "ADS_TOP"
    else:
        source_lane = "UNKNOWN"

    payload = dict(analysis)
    payload.update({
        "contract_schema": _SCHEMA,
        "review_status": (
            "PATTERN_EXTRACTED_COMMERCIAL_PERFORMANCE_CONFIRMED"
            if commercial.get("status") == "CONFIRMED"
            else "PATTERN_EXTRACTED_COMMERCIAL_PERFORMANCE_UNVERIFIED"
        ),
        "source_scope": str(analysis.get("source_evidence_scope") or
                             "sampled frames only; mechanism and transferability remain hypotheses"),
        "source_provenance": {
            "candidate_id": str(candidate.get("id") or analysis.get("candidate_id") or ""),
            "platform": candidate.get("platform"),
            "source_url": candidate.get("source_url"),
            "published_at": _publication_timestamp(candidate),
            "discovery_source": legacy.get("discovery_source"),
            "source_lane": source_lane,
            "discovery_query": legacy.get("discovery_query"),
            "country": candidate.get("country") or legacy.get("discovery_country"),
            "period_days": candidate.get("period_days"),
            "source_role": source_selection.get("role"),
            "source_role_reason": source_selection.get("role_reason"),
            "source_check": source,
            "source_selection": source_selection,
            "analysis_eligibility": _mapping(candidate.get("analysis_eligibility")),
        },
        "source_performance": {
            "status": "CONFIRMED" if commercial.get("status") == "CONFIRMED" else "UNVERIFIED",
            "performance_evidence": performance,
            "commercial_evidence": commercial,
            "metrics": _mapping(candidate.get("metrics")),
            "retention_evidence": _mapping(candidate.get("retention_evidence")),
            "source_score": _mapping(candidate.get("source_score")),
            "is_selection_gate": False,
        },
        "analysis_integrity": {
            "result_sha256": hashlib.sha256(
                json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "section_kinds_preserved": {
                key: str(value.get("kind") or "UNKNOWN").upper()
                for key, value in result.items()
                if isinstance(value, dict) and "kind" in value
            },
        },
    })
    return payload
