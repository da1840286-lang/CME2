"""Non-vendored retry policy for SocialCrawl retention requests.

The legacy discovery adapter remains byte-for-byte intact. Only requests that
explicitly ask for retention data receive the provider-documented longer timeout
and bounded transient retries; other requests retain the legacy behavior.
"""
from __future__ import annotations

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


class RetentionRequestError(SocialCrawlError):
    """SocialCrawl request failure with structured, safe-to-persist diagnostics."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class RetryingSocialCrawlTikTokTopAds(SocialCrawlTikTokTopAds):
    """Preserve the vendored adapter; add bounded retry only for retention requests."""

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
