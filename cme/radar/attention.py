"""Deterministic attention proxy for source selection, never a profitability model."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

VERSION = "attention_absolute_v1"
WEIGHTS = {"views": 0.25, "likes": 0.15, "comments": 0.10, "shares": 0.10, "play_2s_rate": 0.15, "play_6s_rate": 0.25}
COUNT_CAPS = {"views": 1_000_000, "likes": 100_000, "comments": 10_000, "shares": 10_000}


def _number(value):
    if isinstance(value, dict):
        value = value.get("value")
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _date(value):
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        stamp = float(value)
        if stamp > 10_000_000_000:
            stamp /= 1000.0
        try:
            return datetime.fromtimestamp(stamp, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return None


def _published_at(record):
    direct = record.get("published_at")
    if direct not in (None, ""):
        return direct
    raw = record.get("raw")
    post = raw.get("post") if isinstance(raw, dict) else None
    if not isinstance(post, dict):
        return None
    for key in ("published_at", "create_time", "createTime", "created_at", "createdAt", "timestamp"):
        if post.get(key) not in (None, ""):
            return post[key]
    return None


def attention_absolute_v1(record: dict, period_days: int = 30, now=None) -> dict:
    """Fixed absolute weights: missing values keep their weight; coverage exposes missing data."""
    record = record if isinstance(record, dict) else {}
    try:
        requested_period = int(period_days or 30)
    except (TypeError, ValueError):
        requested_period = 30
    period_days = requested_period if requested_period in (7, 30, 180) else 30
    now_dt = now or datetime.now(timezone.utc)
    if not isinstance(now_dt, datetime):
        now_dt = _date(now_dt) or datetime.now(timezone.utc)
    elif now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    else:
        now_dt = now_dt.astimezone(timezone.utc)

    values = {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    for field, weight in WEIGHTS.items():
        raw = record.get(field)
        if raw in (None, ""):
            raw = metrics.get(field)
        value = _number(raw)
        if value is None:
            continue
        if field in COUNT_CAPS:
            value = min(1.0, math.log1p(value) / math.log1p(COUNT_CAPS[field]))
        else:
            if value > 1.0:
                value /= 100.0
            value = min(1.0, value)
        values[field] = max(0.0, value)

    base = sum(WEIGHTS[field] * values[field] for field in values)
    coverage = sum(WEIGHTS[field] for field in values)
    published = _date(_published_at(record))
    age_days = None if published is None else max(0.0, (now_dt - published).total_seconds() / 86400.0)
    recency = 1.0 if age_days is None else max(0.5, min(1.0, 0.8 ** (age_days / period_days)))
    return {
        "version": VERSION, "score": round(base * recency * 100.0, 2),
        "base_score": round(base * 100.0, 2), "coverage": round(coverage, 3),
        "available_fields": list(values), "components": {key: round(value, 4) for key, value in values.items()},
        "period_days": period_days, "age_days": None if age_days is None else round(age_days, 2),
        "recency_multiplier": round(recency, 4),
        "recency_status": "KNOWN" if age_days is not None else "UNKNOWN_NO_PENALTY",
        "is_commercial_proof": False,
        "warning": "Attention proxy only; not sales, conversion, profit, or causal-mechanism evidence.",
    }


# Independent, fixed-weight rank for the mechanism lane when the source has no conversion evidence.
# This deliberately scores attention/reach signals only; it must never be surfaced as money proof.
MECHANISM_WEIGHTS = {
    "reach": 0.30, "amplification": 0.20, "save_rate": 0.20,
    "share_rate": 0.15, "like_rate": 0.10, "recency": 0.05,
}
MECHANISM_RATE_CAPS = {"save_rate": 0.004, "share_rate": 0.001, "like_rate": 0.02}
MECHANISM_REACH_SHRINK = 50_000.0


def attention_rank(record: dict, period_days: int = 30, now=None) -> dict:
    """Absolute mechanism-source attention score with no within-batch percentile or missing-field renormalization."""
    record = record if isinstance(record, dict) else {}
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    post = raw.get("post") if isinstance(raw.get("post"), dict) else {}
    ext = post.get("ext") if isinstance(post.get("ext"), dict) else {}
    engagement = post.get("engagement") if isinstance(post.get("engagement"), dict) else {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}

    def pick(*values):
        for value in values:
            parsed = _number(value)
            if parsed is not None:
                return parsed
        return None

    views = pick(record.get("views"), metrics.get("views"), engagement.get("views"))
    if views is None or views <= 0:
        return {
            "version": "attention_rank_v1", "score": None, "rank": None, "coverage": 0.0,
            "components": {}, "available_fields": [], "missing_components": list(MECHANISM_WEIGHTS),
            "is_commercial_proof": False,
            "warning": "Attention proxy only; no usable view count. Not sales, conversion or profit evidence.",
        }

    likes = pick(record.get("likes"), metrics.get("likes"), engagement.get("likes"))
    shares = pick(record.get("shares"), metrics.get("shares"), engagement.get("shares"))
    saves = pick(record.get("saves"), metrics.get("saves"), engagement.get("saves"))
    followers = pick(record.get("author_followers"), record.get("followers"), ext.get("author_followers"),
                     post.get("author_followers"))
    try:
        period = int(period_days or 30)
    except (TypeError, ValueError):
        period = 30
    period = period if period in (7, 30, 180) else 30

    def clamp(value):
        return max(0.0, min(100.0, value))

    components = {"reach": clamp((math.log10(max(views, 1.0)) - 3.0) / 3.5 * 100.0)}
    if followers is not None and followers > 0:
        components["amplification"] = clamp(math.log10(max(views / followers, 1.0)) / 3.0 * 100.0)

    # Shrink engagement rates toward zero for small view counts so a tiny sample cannot dominate.
    confidence = views / (views + MECHANISM_REACH_SHRINK)
    for name, count in (("save_rate", saves), ("share_rate", shares), ("like_rate", likes)):
        if count is not None:
            components[name] = clamp(count / views / MECHANISM_RATE_CAPS[name] * 100.0) * confidence

    published = _date(_published_at(record))
    now_dt = _date(now) if now is not None and not isinstance(now, datetime) else now
    now_dt = now_dt or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    else:
        now_dt = now_dt.astimezone(timezone.utc)
    age_days = None if published is None else max(0.0, (now_dt - published).total_seconds() / 86400.0)
    if age_days is not None:
        components["recency"] = clamp(100.0 * (1.0 - age_days / max(1.0, float(period) * 2.0)))

    total_weight = sum(MECHANISM_WEIGHTS.values())
    score = sum(MECHANISM_WEIGHTS[name] * value for name, value in components.items()) / total_weight
    coverage = sum(MECHANISM_WEIGHTS[name] for name in components) / total_weight
    missing = [name for name in MECHANISM_WEIGHTS if name not in components]
    fields_used = sorted({
        field for name in components for field in {
            "reach": ["views"],
            "amplification": ["views", "author_followers"],
            "save_rate": ["views", "saves"],
            "share_rate": ["views", "shares"],
            "like_rate": ["views", "likes"],
            "recency": ["published_at"],
        }[name]
    })
    rounded = round(score, 2)
    return {
        "version": "attention_rank_v1", "score": rounded, "rank": rounded,
        "base_score": round(score, 2), "coverage": round(coverage, 3),
        "components": {name: round(value, 2) for name, value in components.items()},
        "available_fields": fields_used, "missing_components": missing,
        "period_days": period, "age_days": None if age_days is None else round(age_days, 2),
        "is_commercial_proof": False,
        "warning": "Attention proxy only; not sales, conversion, profit, or causal-mechanism evidence.",
    }


MECHANISM_ATTENTION_FLOOR = 25.0
MECHANISM_MIN_COVERAGE = 0.35
