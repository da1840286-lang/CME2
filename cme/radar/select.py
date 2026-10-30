"""Source selection for automatic transfer production.

Automatic transfer candidates must pass the strict source-commercial gate. Separate pattern-extraction
helpers remain available for explicitly selected/manual analysis flows.
"""
from __future__ import annotations

from dataclasses import dataclass

from .fit import ADJACENT_MECHANISM, TARGET_PRODUCT

AUTO_ALLOWED_CLASSES = ("DIRECT", "EXPANDED")


@dataclass(frozen=True)
class SourceThreshold:
    """Optional strength bar for automatic sources. OFF by default: no number is hard-wired.

    Hard conditions in block_reason (video, scorer not REJECT, no hard-exclude, ...) always apply.
    This bar is only about how strong a *usable* source looks. In "flag" mode a source below the bar is kept
    and marked WEAK_SOURCE (and ranked after the qualified ones); in "enforce" mode it is dropped, and when
    nothing is left the run stops honestly with NO_QUALIFIED_SOURCE instead of producing a weak creative.
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
    """Reasons a usable source is below the configured bar. Empty when no bar is set or the source clears it."""
    if threshold is None or not threshold.active or not isinstance(candidate, dict):
        return []
    score = candidate.get("source_score") or {}
    score = score if isinstance(score, dict) else {}
    reasons = []
    if threshold.min_final_score is not None:
        value = score.get("final_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reasons.append(f"final_score {value!r} is missing or not a number (bar {threshold.min_final_score:g})")
        elif value < threshold.min_final_score:
            reasons.append(f"final_score {value:g} is below the bar {threshold.min_final_score:g}")
    if threshold.allowed_actions:
        action = str(score.get("action") or "").upper()
        if action not in threshold.allowed_actions:
            reasons.append(f"scorer action {action or 'MISSING'} is not in {', '.join(threshold.allowed_actions)}")
    return reasons


def effective_source_classification(candidate: dict) -> str:
    """Resolve a verified FIND product role against the legacy relevance label.

    The legacy label remains unchanged unless FIND supplies positive source-body promotion
    evidence. A verified target product maps to DIRECT; a verified adjacent product maps to
    EXPANDED. Missing or incomplete role evidence never upgrades an UNVERIFIED source.
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


def block_reason(candidate: dict, allowed_classes=AUTO_ALLOWED_CLASSES, min_final_score=None) -> str:
    """Empty string when the candidate passes the legacy strict source-commercial gate."""
    if not candidate.get("video_url"):
        return "no video URL"

    check = candidate.get("source_check") or {}
    classification = str(check.get("classification") or "").upper()
    if check.get("excluded") or classification == "HARD_EXCLUDE":
        return "hard-excluded by the legacy source check"
    classification = effective_source_classification(candidate)
    if allowed_classes is not None and classification not in allowed_classes:
        return f"source relevance {classification or 'MISSING'} is not one of {', '.join(allowed_classes)}"

    score = candidate.get("source_score") or {}
    if not isinstance(score, dict):
        score = {}
    action = str(score.get("action") or "").upper()
    tags_raw = score.get("tags") or []
    if isinstance(tags_raw, str):
        tags = {tags_raw}
    elif isinstance(tags_raw, (list, tuple, set)):
        tags = {str(tag) for tag in tags_raw}
    else:
        tags = set()

    components = score.get("components") or {}
    components = components if isinstance(components, dict) else {}
    money_gate = components.get("money_signal_gate") or {}
    money_gate = money_gate if isinstance(money_gate, dict) else {}
    duration = components.get("duration") or {}
    duration = duration if isinstance(duration, dict) else {}

    source_value_type = str(score.get("source_value_type") or "").upper()
    try:
        mechanism_score = float(score.get("mechanism_source_score") or 0)
    except (TypeError, ValueError):
        mechanism_score = 0.0
    mechanism_pass = (
        source_value_type in {"MECHANISM_SOURCE", "BOTH"} and mechanism_score >= 70.0
    ) or "long_source_allowed_for_mechanism" in tags

    # A scorer's explicit rejection is terminal. Mechanism-source is a separate path only
    # when the scorer itself did not reject the candidate.
    if action == "REJECT":
        return "legacy scorer action is REJECT"

    money_hard_block = money_gate.get("hard_block") is True or "money_signal_hard_block" in tags
    if money_hard_block and not mechanism_pass:
        return "money-signal hard block; no mechanism-source exception"

    duration_hard_block = duration.get("hard_block") is True or "duration_over_60_hard_block" in tags
    if duration_hard_block and not mechanism_pass:
        return "duration hard block; no mechanism-source exception"

    money_pass = money_gate.get("pass") is True or "money_signal_pass" in tags
    if not money_pass and not mechanism_pass:
        return "no explicit money-gate pass or mechanism-source score >= 70"

    final_score = score.get("final_score")
    if isinstance(final_score, (int, float)) and not isinstance(final_score, bool):
        if final_score <= 0 and not mechanism_pass:
            return f"final_score {final_score!r} is zero or below"
        if min_final_score is not None and final_score < min_final_score:
            return f"final_score {final_score!r} is below the required {min_final_score}"
    elif min_final_score is not None:
        return f"final_score {final_score!r} is below the required {min_final_score}"

    return ""


def select_candidate(candidates, allowed_classes=AUTO_ALLOWED_CLASSES, min_final_score=None):
    """Return the first candidate passing the legacy strict source-commercial gate."""
    skipped = []
    for item in candidates or []:
        reason = block_reason(item, allowed_classes, min_final_score)
        if not reason:
            return item, skipped
        skipped.append(f"{item.get('id')}: {reason}")
    return None, skipped



def pattern_source_reason(candidate: dict) -> str:
    """Return a blocker only when a source cannot safely be used to extract a pattern.

    Deliberately does not check scorer action, money signals, score floors, duration,
    or relevance labels. Those fields remain evidence/ranking metadata; they do not prove
    whether a new target-native creative can be made from the source's pattern.
    """
    if not isinstance(candidate, dict):
        return "candidate record is missing or invalid"
    if not candidate.get("video_url"):
        return "no video URL"

    check = candidate.get("source_check") or {}
    if not isinstance(check, dict):
        check = {}
    classification = str(check.get("classification") or "").upper()
    if check.get("excluded") or classification == "HARD_EXCLUDE":
        return "hard-excluded by the legacy source check"

    return ""


def select_pattern_candidate(candidates):
    """Choose the first playable, non-hard-excluded source in the existing FIND order."""
    skipped = []
    for item in candidates or []:
        reason = pattern_source_reason(item)
        if not reason:
            return item, skipped
        candidate_id = item.get("id") if isinstance(item, dict) else "<invalid>"
        skipped.append(f"{candidate_id}: {reason}")
    return None, skipped


RELEVANCE_TIEBREAK = {"DIRECT": 2, "EXPANDED": 1}


def _score(item) -> float:
    score = (item.get("source_score") or {}) if isinstance(item, dict) else {}
    try:
        return float(score.get("final_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rank_transfer_candidates(candidates, threshold: SourceThreshold | None = None):
    """Return only commercially eligible transfer sources, ordered by score and relevance.

    With an active threshold, sources that clear it rank first. Weak sources follow (flag mode) or are
    dropped with a WEAK_SOURCE reason (enforce mode). Without a threshold the order is unchanged.
    """
    usable, skipped = [], []
    for position, item in enumerate(candidates or []):
        reason = block_reason(item)
        if reason:
            candidate_id = item.get("id") if isinstance(item, dict) else "<invalid>"
            skipped.append(f"{candidate_id}: {reason}")
            continue
        weak = weak_source_reasons(item, threshold)
        if weak and threshold is not None and threshold.mode == "enforce":
            skipped.append(f"{item.get('id')}: WEAK_SOURCE: " + "; ".join(weak))
            continue
        relevance = RELEVANCE_TIEBREAK.get(effective_source_classification(item), 0)
        usable.append((1 if weak else 0, -_score(item), -relevance, position, item))
    usable.sort(key=lambda row: row[:4])
    return [row[4] for row in usable], skipped

def select_transfer_candidate(candidates, threshold: SourceThreshold | None = None):
    """Select the highest-ranked candidate passing the strict commercial eligibility gate."""
    ordered, skipped = rank_transfer_candidates(candidates, threshold)
    return (ordered[0] if ordered else None), skipped


def analysis_review_reason(candidate: dict, allowed_classes=None) -> str:
    """Backward-compatible alias for the former analysis-only gate.

    allowed_classes is accepted for older callers but intentionally does not gate
    pattern extraction. Relevance labels and commercial scores remain visible as evidence.
    """
    return pattern_source_reason(candidate)


def select_analysis_candidate(candidates, allowed_classes=None):
    """Backward-compatible alias; selects pattern sources, not commercially approved ads."""
    return select_pattern_candidate(candidates)
