"""Non-vendored retry policy for SocialCrawl retention requests.

The legacy discovery adapter remains byte-for-byte intact. Only requests that
explicitly ask for retention data receive the provider-documented longer timeout
and bounded transient retries; other requests retain the legacy behavior.
"""
from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from .legacy.discovery.tiktok_socialcrawl import SocialCrawlError, SocialCrawlTikTokTopAds


RETENTION_REQUEST_TIMEOUT_SECONDS = 112.0
RETENTION_MAX_RETRIES = 3
RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})



_RETENTION_CURVE_MAX_POINTS = 120
_RETENTION_VALUE_KEYS = ("retention", "value", "rate", "ratio", "fraction", "share", "percentage", "percent")
_RETENTION_SECOND_KEYS = ("second", "seconds", "time", "t")


def _finite_retention_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _retention_point_value(point: Any) -> float | None:
    number = _finite_retention_number(point)
    if number is not None:
        return number
    if isinstance(point, dict):
        for key in _RETENTION_VALUE_KEYS:
            if key in point:
                number = _finite_retention_number(point.get(key))
                if number is not None:
                    return number
    return None


def _retention_point_second(point: Any) -> int | None:
    if not isinstance(point, dict):
        return None
    for key in _RETENTION_SECOND_KEYS:
        value = _finite_retention_number(point.get(key))
        if value is not None and value >= 1 and value.is_integer():
            return int(value)
    return None


def normalize_retention_curve(curve: Any, max_points: int = _RETENTION_CURVE_MAX_POINTS) -> tuple[list[float | None] | None, int, str]:
    """Normalize supported curve shapes without shifting seconds when a point is invalid."""
    cap = max(1, int(max_points))
    if curve is None:
        return None, 0, "missing"
    if isinstance(curve, dict):
        for key in ("points", "values", "curve"):
            nested = curve.get(key)
            if isinstance(nested, list):
                points, invalid, shape = normalize_retention_curve(nested, max_points=cap)
                return points, invalid, f"mapping:{key}/{shape}"
        numeric_seconds = []
        for key, value in curve.items():
            try:
                second = int(str(key).strip())
            except (TypeError, ValueError):
                numeric_seconds = []
                break
            if second < 1 or str(second) != str(key).strip():
                numeric_seconds = []
                break
            numeric_seconds.append((second, value))
        if numeric_seconds and len(numeric_seconds) == len(curve):
            highest_raw = max(second for second, _ in numeric_seconds)
            highest = min(highest_raw, cap)
            points: list[float | None] = [None] * highest
            for second, value in numeric_seconds:
                if second <= cap:
                    points[second - 1] = _retention_point_value(value)
            invalid = sum(point is None for point in points) + max(0, highest_raw - cap)
            if any(point is not None for point in points):
                return points, invalid, "mapping:second_keys"
            return None, invalid, "mapping:second_keys:no_valid_points"
        return None, 0, "unsupported_dict"
    if not isinstance(curve, list):
        return None, 0, f"unsupported_{type(curve).__name__}"
    if not curve:
        return None, 0, "empty_list"

    raw_points = curve[:cap]
    explicit_seconds = [_retention_point_second(point) for point in raw_points]
    schema = "list:object" if any(isinstance(point, dict) for point in raw_points) else "list:scalar"
    if any(second is not None for second in explicit_seconds):
        highest_raw = max((second or 0) for second in explicit_seconds)
        highest = min(highest_raw, cap)
        points = [None] * highest
        for point, second in zip(raw_points, explicit_seconds):
            if second is not None and second <= cap:
                points[second - 1] = _retention_point_value(point)
        invalid = (
            sum(second is None for second in explicit_seconds)
            + sum(value is None for value in points)
            + max(0, len(curve) - cap)
        )
        if any(value is not None for value in points):
            return points, invalid, schema + ":explicit_seconds"
        return None, invalid, schema + ":no_valid_points"

    points = [_retention_point_value(point) for point in raw_points]
    invalid = sum(point is None for point in points) + max(0, len(curve) - cap)
    if any(point is not None for point in points):
        return points, invalid, schema
    return None, invalid, schema + ":no_valid_points"


class RetentionRequestError(SocialCrawlError):
    """SocialCrawl request failure with structured, safe-to-persist diagnostics."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class RetryingSocialCrawlTikTokTopAds(SocialCrawlTikTokTopAds):
    """Preserve the vendored adapter; add bounded retry only for retention requests."""

    @staticmethod
    def _raw_ad(item: dict[str, Any]) -> dict[str, Any]:
        post = item.get("post", item)
        if not isinstance(post, dict):
            return {}
        ext = post.get("ext") if isinstance(post.get("ext"), dict) else {}
        ad = ext.get("ad") if isinstance(ext.get("ad"), dict) else {}
        return ad

    def _unwrap_item(self, item: dict[str, Any]) -> dict[str, Any]:
        normalized = super()._unwrap_item(item)
        ad = self._raw_ad(item)
        curve, invalid_points, shape = normalize_retention_curve(ad.get("retention"))
        normalized["retention_curve"] = curve
        normalized["retention_curve_invalid_points"] = invalid_points
        provenance = normalized.get("retention_provenance")
        provenance = dict(provenance) if isinstance(provenance, dict) else {}
        if curve is not None:
            provenance["retention_curve"] = "post.ext.ad.retention"
        elif ad.get("retention") is None:
            provenance["retention_curve"] = "post.ext.ad.retention:missing"
        else:
            provenance["retention_curve"] = f"post.ext.ad.retention:{shape}"
        normalized["retention_provenance"] = provenance
        return normalized

    @classmethod
    def _retention_schema_diagnostics(cls, items: list[dict[str, Any]]) -> dict[str, Any]:
        shape_counts: dict[str, int] = {}
        examples: list[dict[str, Any]] = []
        curves_available = 0
        for row in items:
            raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
            ad = cls._raw_ad(raw)
            curve = ad.get("retention")
            points, invalid_points, shape = normalize_retention_curve(curve)
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            if points is not None:
                curves_available += 1
            if len(examples) < 5:
                examples.append({
                    "ad_id": str(row.get("ad_id") or ""),
                    "curve_shape": shape,
                    "raw_curve_type": type(curve).__name__ if curve is not None else "null",
                    "raw_point_count": len(curve) if isinstance(curve, (list, dict)) else None,
                    "parsed_point_count": len(points) if isinstance(points, list) else 0,
                    "invalid_or_missing_points": invalid_points,
                    "first_point_type": type(curve[0]).__name__ if isinstance(curve, list) and curve else None,
                    "first_point_keys": sorted(str(key) for key in curve[0].keys())[:12]
                        if isinstance(curve, list) and curve and isinstance(curve[0], dict) else [],
                })
        return {
            "rows": len(items),
            "rows_with_parsed_curve": curves_available,
            "shape_counts": dict(sorted(shape_counts.items())),
            "examples": examples,
            "note": "Shape diagnostics only; curve values stay attached to their source record.",
        }

    def fetch_board(self, **kwargs) -> dict[str, Any]:
        # Keep the requested board limit. Retention points accompany each ad, but a
        # 2s-ranked board and a 6s-ranked board are different populations and must not
        # be collapsed just because both return the same retention fields.
        retention_requested = bool(kwargs.get("include_retention"))
        result = super().fetch_board(**kwargs)
        if retention_requested:
            diagnostics = result.get("raw_playback_field_diagnostics")
            diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
            diagnostics["retention_curve_schema"] = self._retention_schema_diagnostics(result.get("items") or [])
            result["raw_playback_field_diagnostics"] = diagnostics
        return result

    @staticmethod
    def _retry_delay_seconds(retry_after: Any, retry_index: int) -> float:
        delay = None
        if retry_after not in (None, ""):
            try:
                delay = max(0.0, float(str(retry_after).strip()))
            except (TypeError, ValueError):
                try:
                    retry_at = parsedate_to_datetime(str(retry_after))
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    delay = None
        if delay is None:
            delay = float(2 ** min(max(retry_index, 0), 2))
        # Prevent a bad Retry-After value from holding a runner indefinitely.
        return min(delay, 30.0)

    def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        retention_requested = str(params.get("include") or "").strip().lower() == "retention"
        if not retention_requested:
            return super()._request_json(url, params)
        # Retention uses its own request path, so it must not inherit the cache
        # policy from the preceding keyword-filtered playback-board request.
        self.last_cache_control_requested = None
        if not self.api_key:
            raise SocialCrawlError("SOCIALCRAWL_API_KEY is not configured")

        timeout = max(float(self.timeout), RETENTION_REQUEST_TIMEOUT_SECONDS)
        idem = str(uuid.uuid4())
        headers = {"x-api-key": self.api_key, "Idempotency-Key": idem}
        started = time.monotonic()
        history: list[dict[str, Any]] = []

        def diagnostics(status: int | None = None, error: str | None = None) -> dict[str, Any]:
            result = {
                "provider": "SocialCrawl",
                "endpoint": url,
                "provider_params": dict(params),
                "retention_requested": True,
                "timeout_seconds": timeout,
                "max_retries": RETENTION_MAX_RETRIES,
                "attempt_count": len(history),
                "retry_count": max(0, len(history) - 1),
                "idempotency_key_reused_across_retries": True,
                "last_http_status": status,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "attempts": [dict(item) for item in history],
            }
            if error:
                result["error"] = str(error)[:500]
            return result

        for attempt_index in range(RETENTION_MAX_RETRIES + 1):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
            except requests.exceptions.RequestException as exc:
                history.append({
                    "attempt": attempt_index + 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                })
                if attempt_index < RETENTION_MAX_RETRIES:
                    delay = self._retry_delay_seconds(None, attempt_index)
                    history[-1]["retry_delay_seconds"] = delay
                    time.sleep(delay)
                    continue
                diag = diagnostics(error=f"{type(exc).__name__}: {exc}")
                self.last_request_diagnostics = diag
                raise RetentionRequestError(
                    f"SocialCrawl retention request failed after {len(history)} attempt(s) "
                    f"(timeout={timeout:g}s): {type(exc).__name__}: {exc}",
                    diagnostics=diag,
                ) from exc

            history.append({"attempt": attempt_index + 1, "http_status": response.status_code})
            if response.status_code in RETRYABLE_HTTP_STATUSES and attempt_index < RETENTION_MAX_RETRIES:
                retry_after = getattr(response, "headers", {}).get("Retry-After")
                delay = self._retry_delay_seconds(retry_after, attempt_index)
                history[-1]["retry_delay_seconds"] = delay
                time.sleep(delay)
                continue

            try:
                data = response.json()
            except ValueError as exc:
                diag = diagnostics(status=response.status_code, error=str(exc))
                self.last_request_diagnostics = diag
                raise RetentionRequestError(
                    f"SocialCrawl returned non-JSON HTTP {response.status_code} for retention request",
                    diagnostics=diag,
                ) from exc

            diag = diagnostics(status=response.status_code)
            self.last_request_diagnostics = diag
            if not response.ok or not isinstance(data, dict) or not data.get("success", False):
                err = data.get("error", {}) if isinstance(data, dict) else {}
                err = err if isinstance(err, dict) else {}
                error_type = err.get("type", "HTTP_ERROR")
                message = err.get("message", f"HTTP {response.status_code}")
                diag["provider_error_type"] = error_type
                diag["provider_error_message"] = str(message)[:500]
                self.last_request_diagnostics = diag
                raise RetentionRequestError(f"{error_type}: {message}", diagnostics=diag)
            return data

        diag = diagnostics(error="request ended without a response")
        self.last_request_diagnostics = diag
        raise RetentionRequestError("SocialCrawl retention request ended without a response", diagnostics=diag)
