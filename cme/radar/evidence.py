"""Evidence labels for source discovery; not a profitability prediction.

These labels describe the evidence present in a source record. They must not change FIND
ranking, source eligibility, Selection, Target-Fit, or the ALL workflow.
"""
from __future__ import annotations

from typing import Any

_NUMERIC_SIGNALS = (
    "ctr", "cvr", "play_2s_rate", "play_6s_rate", "views", "likes", "shares", "comments",
)
_ECONOMIC_METRICS = (
    "roas", "cpa", "net_profit", "profit", "revenue_per_1000_views", "epc",
)


def _present_number(record: dict[str, Any], field: str) -> bool:
    value = record.get(field)
    if value in (None, ""):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number >= 0


def _confirmed_economic_validation(record: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Require an explicit numeric campaign outcome, success criterion, and verification provenance."""
    raw = record.get("economic_validation")
    if not isinstance(raw, dict):
        return False, {}
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    valid_metrics = {}
    for name in _ECONOMIC_METRICS:
        value = metrics.get(name)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        valid_metrics[name] = parsed
    verified = bool(
        str(raw.get("status") or "").upper() == "CONFIRMED"
        and raw.get("target_met") is True
        and valid_metrics
        and str(raw.get("success_criteria") or "").strip()
        and str(raw.get("verification_source") or "").strip()
        and str(raw.get("verified_at") or "").strip()
        and str(raw.get("verified_by") or "").strip()
    )
    if not verified:
        return False, {}
    return True, {
        "verification_source": str(raw.get("verification_source"))[:160],
        "verified_at": str(raw.get("verified_at"))[:80],
        "success_criteria": str(raw.get("success_criteria"))[:240],
        "metrics": valid_metrics,
    }


def classify_commercial_evidence(
    record: dict[str, Any] | None,
    source_score: dict[str, Any] | None,
    performance_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Label evidence strength without turning proxies or heuristic gate results into sales proof."""
    record = record if isinstance(record, dict) else {}
    source_score = source_score if isinstance(source_score, dict) else {}
    performance_evidence = performance_evidence if isinstance(performance_evidence, dict) else {}

    available = [field for field in _NUMERIC_SIGNALS if _present_number(record, field)]
    commercial_signals: list[str] = []
    for field in ("ctr", "cvr"):
        if field in available:
            commercial_signals.append(f"source_metric:{field}")
    if str(record.get("objective") or "").strip():
        commercial_signals.append("ad_objective_present")
    if str(record.get("landing_page_url") or record.get("landing_page") or "").strip():
        commercial_signals.append("landing_page_present")
    if str(record.get("discovery_source") or "").lower() == "ads_top":
        commercial_signals.append("ads_top_performance_board")
    if performance_evidence.get("basis") == "ADS_TOP_PERFORMANCE_BOARD":
        commercial_signals.append("ads_top_performance_board")
    commercial_signals = list(dict.fromkeys(commercial_signals))

    economically_validated, validation = _confirmed_economic_validation(record)
    if economically_validated:
        level = "ECONOMICALLY_VALIDATED"
        explanation = "A structured campaign outcome, verification provenance, and a stated success criterion are present."
        missing: list[str] = []
    elif commercial_signals:
        level = "COMMERCIAL_PROXY"
        explanation = "Commercial/ad-delivery signals exist, but they do not establish profit or economic efficiency."
        missing = [
            "confirmed campaign outcome (spend plus conversions/revenue or profit)",
            "defined economic target or break-even margin",
            "verifiable outcome report and validation provenance",
        ]
    elif available or performance_evidence.get("basis") == "ENGAGEMENT_PROXY":
        level = "ATTENTION_ONLY"
        explanation = "Only attention/playback/engagement evidence is available; this is not proof of sales."
        missing = [
            "source-side commercial signal (for example CTR/CVR or campaign objective/landing page)",
            "confirmed campaign outcome and economic target",
        ]
    else:
        level = "NO_EVIDENCE"
        explanation = "No usable performance or commercial signal was found in the normalized source record."
        missing = ["performance evidence", "commercial evidence", "verified economic outcome"]

    signals = list(commercial_signals)
    if level != "ECONOMICALLY_VALIDATED":
        signals.extend(f"attention_metric:{field}" for field in available if field not in {"ctr", "cvr"})
        basis = performance_evidence.get("basis")
        if basis:
            signals.append(f"performance_basis:{basis}")
    return {
        "level": level,
        "commercial_result_confirmed": economically_validated,
        "available_metrics": available,
        "signals": list(dict.fromkeys(signals)),
        "missing": missing,
        "explanation": explanation,
        "validation": validation,
        "money_gate_is_profit_proof": False,
        "source_score_is_profit_probability": False,
    }
