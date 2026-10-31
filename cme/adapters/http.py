"""Shared HTTP helpers for adapters. The transport is injectable so tests never touch the network."""
from __future__ import annotations
import base64
import json
import mimetypes
import re
from pathlib import Path
from ..providers import InfraError, ProviderCallError


def _safe_detail(value: str) -> str:
    """Flatten and redact provider-supplied text before adding it to logs/errors."""
    text = " ".join(str(value).split())
    text = re.sub(r"(?i)\bsk-or-v1-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)\b(api[_ -]?key|authorization)\s*[:=]\s*\S+", r"\1=[REDACTED]", text)
    return text[:240]


def _error_detail(raw: bytes) -> str:
    """Return bounded, redacted error fields; avoid dumping arbitrary response bodies."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return _safe_detail(raw.decode("utf-8", errors="replace"))
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return _safe_detail(raw.decode("utf-8", errors="replace"))

    error = payload["error"]
    parts = []
    code = error.get("code")
    if type(code) is int or (isinstance(code, str) and code.strip()):
        safe_code = re.sub(r"[^A-Za-z0-9_.-]", "", str(code))[:40]
        if safe_code:
            parts.append(f"provider_code={safe_code}")
    message = error.get("message")
    if isinstance(message, str) and message.strip():
        parts.append("provider_message=" + _safe_detail(message))

    metadata = error.get("metadata")
    if isinstance(metadata, dict):
        provider_name = metadata.get("provider_name")
        if isinstance(provider_name, str) and provider_name.strip():
            parts.append("provider=" + _safe_detail(provider_name))
        upstream = metadata.get("raw")
        if isinstance(upstream, dict):
            for key in ("message", "detail", "error"):
                value = upstream.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append("provider_detail=" + _safe_detail(value))
                    break
        elif isinstance(upstream, str) and upstream.strip():
            parts.append("provider_detail=" + _safe_detail(upstream))

    return "; ".join(parts)[:800] or _safe_detail(raw.decode("utf-8", errors="replace"))


def urllib_transport(method, url, headers, body, timeout=180):
    import urllib.error
    import urllib.request
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError) as e:
        raise InfraError(f"network: {e}") from e


def data_uri(path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def check_status(name: str, status: int, raw: bytes) -> None:
    if status == 429 or status >= 500:
        detail = _error_detail(raw)
        suffix = f": {detail}" if detail else ""
        raise InfraError(f"{name} HTTP {status}{suffix}")
    if status >= 400:
        raise ProviderCallError(f"{name} HTTP {status}: {raw[:200]!r}")
