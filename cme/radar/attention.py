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


MECHANISM_ATTENTION_FLOOR = 25.0
MECHANISM_MIN_COVERAGE = 0.35
