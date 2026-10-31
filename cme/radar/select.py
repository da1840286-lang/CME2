"""Canonical source admission for pattern extraction and downstream ALL handoff.

PATTERN_SOURCE_V1 decides whether a FIND result may be inspected for its creative
mechanism. Scorer action, money-gate labels and performance proxies are audit
evidence, not admission criteria. Target-Fit remains the later transferability gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .fit import ADJACENT_MECHANISM, TARGET_PRODUCT

AUTO_ALLOWED_CLASSES = ("DIRECT", "EXPANDED")
VIDEO_DURATION_RANGES = ("all", "up_to_40", "40_to_60", "60_to_90", "over_90")
PATTERN_SOURCE_CONTRACT = "PATTERN_SOURCE_V1"


@dataclass(frozen=True)
class SourceThreshold:
    """Legacy score threshold retained as a diagnostic-only WEAK_SOURCE flag.

    Historical settings remain readable. Even mode='enforce' no longer creates
    a second gate or changes FIND order.
    """
    min_final_score: float | None = None
    allowed_actions: tuple = ()
    mode: str = "flag"

    @property
    def active(self) -> bool:
        return self.min_final_score is not None or bool(self.allowed_actions)

    @classmethod
    def from_settings(cls, settings) -> "SourceThreshold":
        return cls(min_final_score=getattr(settings, "source_min_final_score", None),
                   allowed_actions=tuple(getattr(settings, "source_allowed_actions", ()) or ()),
                   mode=getattr(settings, "source_threshold_mode", "flag") or "flag")


def weak_source_reasons(candidate: dict, threshold: SourceThreshold | None) -> list:
    """Return score/action observations only; never used to determine eligibility."""
    if threshold is None or not threshold.active or not isinstance(candidate, dict):
        return []
    score = candidate.get("source_score") or {}
    score = score if isinstance(score, dict) else {}
    reasons = []
    if threshold.min_final_score is not None:
        value = score.get("final_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reasons.append(f"final_score {value!r} is missing or not a number (diagnostic bar {threshold.min_final_score:g})")
        elif value < threshold.min_final_score:
            reasons.append(f"final_score {value:g} is below the diagnostic bar {threshold.min_final_score:g}")
    if threshold.allowed_actions:
        action = str(score.get("action") or "").upper()
        if action not in threshold.allowed_actions:
            reasons.append(f"scorer action {action or 'MISSING'} is not in diagnostic set {', '.join(threshold.allowed_actions)}")
    return reasons


def video_duration_filter_reason(record: dict, selected_range: str = "all") -> str | None:
    """Shared implementation of the operator-selected FIND duration filter."""
    selected_range = str(selected_range or "all").strip().lower()
    if selected_range not in VIDEO_DURATION_RANGES:
        raise ValueError(f"video_duration_range must be one of {VIDEO_DURATION_RANGES}")
    if selected_range == "all":
        return None
    record = record if isinstance(record, dict) else {}
    raw = record.get("duration_seconds")
    if isinstance(raw, bool) or raw in (None, "", []):
        return "VIDEO_DURATION_UNKNOWN"
    try:
        duration = float(raw)
    except (TypeError, ValueError, OverflowError):
        return "VIDEO_DURATION_UNKNOWN"
    if not math.isfinite(duration) or duration <= 0:
        return "VIDEO_DURATION_UNKNOWN"
    matches = {
        "up_to_40": duration <= 40,
        "40_to_60": 40 < duration <= 60,
        "60_to_90": 60 < duration <= 90,
        "over_90": duration > 90,
    }[selected_range]
    return None if matches else "VIDEO_DURATION_OUT_OF_RANGE"


def effective_source_classification(candidate: dict) -> str:
    """Resolve a verified FIND product role against the legacy relevance label.

    Kept for diagnostics and reports only. It is not a pattern-admission score.
    """
    check = candidate.get("source_check") or {}
    classification = str(check.get("classification") or "").upper()
    if check.get("excluded") or classification == "HARD_EXCLUDE":
        return classification or "HARD_EXCLUDE"
    role_info = candidate.get("source_selection")
    if not isinstance(role_info, dict):
        return classification
    role = str(role_info.get("role") or "").upper()
    promoted = role_info.get("promotion_evidence") is True
    target_promoted = role_info.get("target_promotion_evidence") is True
    if role == TARGET_PRODUCT and promoted and target_promoted:
        return "DIRECT"
    if role == ADJACENT_MECHANISM and promoted:
        return "EXPANDED"
    return classification


def pattern_source_reason(candidate: dict) -> str:
    """Return the canonical PATTERN_SOURCE_V1 admission decision.

    The contract checks media, source exclusion, FIND's verified role/promotion,
    product-query match, attention coverage/floor when FIND records them, and the
    operator's duration filter. It intentionally ignores commercial score/action,
    Money Gate pass/hard-block flags, and all heuristic profitability proxies.
    Older records without source_selection require a verified DIRECT/EXPANDED label.
    """
    if not isinstance(candidate, dict):
        return "candidate record is missing or invalid"
    if not candidate.get("video_url"):
        return "no playable video URL"

    check = candidate.get("source_check") or {}
    if not isinstance(check, dict):
        check = {}
    classification = str(check.get("classification") or "").upper()
    if check.get("excluded") or classification == "HARD_EXCLUDE":
        return "hard-excluded by source verification"

    selection = candidate.get("source_selection")
    request = candidate.get("request") or {}
    if not isinstance(request, dict):
        request = {}
    mode = str(
        (selection.get("mode") if isinstance(selection, dict) else None)
        or request.get("selection_mode") or "product"
    ).strip().lower()
    if mode not in {"product", "mechanism"}:
        return f"unsupported source-selection mode {mode!r}"
    if isinstance(selection, dict):
        role = str(selection.get("role") or "").strip().upper()
        promoted = selection.get("promotion_evidence") is True
        if mode == "product":
            if role != TARGET_PRODUCT:
                return f"source role {role or 'MISSING'} is not TARGET_PRODUCT"
            if not promoted:
                return "target product lacks independent promotion evidence"
            # Same pair of conditions as FIND's final Product Gate (promotion AND target-noun promotion).
            if "target_promotion_evidence" in selection and selection.get("target_promotion_evidence") is not True:
                return "target noun lacks promotion evidence"
        elif mode == "mechanism":
            if role != ADJACENT_MECHANISM:
                return f"source role {role or 'MISSING'} is not ADJACENT_MECHANISM"
            if not promoted:
                return "adjacent product lacks independent promotion evidence"
        else:
            return f"unsupported source-selection mode {mode!r}"

        minimum_coverage = request.get("minimum_attention_coverage")
        if minimum_coverage not in (None, ""):
            attention = selection.get("attention") or {}
            coverage = attention.get("coverage") if isinstance(attention, dict) else None
            if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
                return "FIND attention coverage is missing"
            if coverage < float(minimum_coverage):
                return f"attention coverage {coverage:g} is below the FIND minimum {float(minimum_coverage):g}"

        attention_floor = request.get("attention_floor")
        if attention_floor not in (None, ""):
            attention = selection.get("attention") or {}
            score = attention.get("score") if isinstance(attention, dict) else None
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                return "FIND attention score is missing for the configured floor"
            if score < float(attention_floor):
                return f"attention score {score:g} is below the configured FIND floor {float(attention_floor):g}"
    elif classification not in AUTO_ALLOWED_CLASSES:
        return f"legacy source relevance {classification or 'MISSING'} is not verified"

    legacy_record = candidate.get("legacy_record") or {}
    query_match = candidate.get("query_product_match")
    if not isinstance(query_match, dict) and isinstance(legacy_record, dict):
        query_match = legacy_record.get("query_product_match")
    if mode == "mechanism":
        if not isinstance(query_match, dict):
            return "mechanism source requires an explicit positive product-query match"
        if query_match.get("matches") is False:
            return str(query_match.get("reason") or "QUERY_PRODUCT_MISMATCH")
        if query_match.get("matches") is not True:
            return "mechanism source requires an explicit positive product-query match"
        if not str(query_match.get("query_anchor") or "").strip():
            return "mechanism source query match has no verified product anchor"
    # Product mode can contain legacy query_match=None, because its source-role gate independently
    # verifies target-product promotion. An explicit mismatch is always rejected.
    elif isinstance(query_match, dict) and query_match.get("matches") is False:
        return str(query_match.get("reason") or "source does not match the requested product query")

    duration_range = str(request.get("video_duration_range") or "all").strip().lower()
    duration_reason = video_duration_filter_reason(
        {"duration_seconds": candidate.get("duration_seconds")}, duration_range
    )
    if duration_reason == "VIDEO_DURATION_UNKNOWN":
        return f"selected duration range {duration_range} requires a known video duration"
    if duration_reason == "VIDEO_DURATION_OUT_OF_RANGE":
        return f"video duration does not match selected range {duration_range}"

    return ""


def block_reason(candidate: dict, allowed_classes=AUTO_ALLOWED_CLASSES, min_final_score=None) -> str:
    """Backward-compatible name for PATTERN_SOURCE_V1; legacy score arguments are ignored."""
    return pattern_source_reason(candidate)


def select_candidate(candidates, allowed_classes=AUTO_ALLOWED_CLASSES, min_final_score=None):
    """Backward-compatible selector using PATTERN_SOURCE_V1 in FIND order."""
    return select_pattern_candidate(candidates)


def select_pattern_candidate(candidates):
    """Choose the first candidate passing PATTERN_SOURCE_V1, preserving input order."""
    skipped = []
    for item in candidates or []:
        reason = pattern_source_reason(item)
        if not reason:
            return item, skipped
        candidate_id = item.get("id") if isinstance(item, dict) else "<invalid>"
        skipped.append(f"{candidate_id}: {reason}")
    return None, skipped


def rank_pattern_candidates(candidates, threshold: SourceThreshold | None = None):
    """Filter only on PATTERN_SOURCE_V1; preserve FIND order exactly.

    Threshold is accepted for backward compatibility and is diagnostic only.
    Scores, scorer actions, commercial gates and relevance tie-breaks cannot
    reorder or remove candidates before ANALYZE.
    """
    eligible, skipped = [], []
    for item in candidates or []:
        reason = pattern_source_reason(item)
        if reason:
            candidate_id = item.get("id") if isinstance(item, dict) else "<invalid>"
            skipped.append(f"{candidate_id}: {reason}")
        else:
            eligible.append(item)
    return eligible, skipped


def rank_transfer_candidates(candidates, threshold: SourceThreshold | None = None):
    """Deprecated compatibility alias; automatic analysis is not commercial approval."""
    return rank_pattern_candidates(candidates, threshold)


def select_transfer_candidate(candidates, threshold: SourceThreshold | None = None):
    """Deprecated compatibility alias for the first PATTERN_SOURCE_V1 candidate."""
    ordered, skipped = rank_pattern_candidates(candidates, threshold)
    return (ordered[0] if ordered else None), skipped


def analysis_review_reason(candidate: dict, allowed_classes=None) -> str:
    """Backward-compatible alias for the canonical pattern-source contract."""
    return pattern_source_reason(candidate)


def select_analysis_candidate(candidates, allowed_classes=None):
    """Backward-compatible alias; selects pattern sources, not commercially approved ads."""
    return select_pattern_candidate(candidates)
