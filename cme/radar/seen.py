"""TTL-scoped cross-run FIND history. Missing/corrupt state fails open."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

VERSION = 1
DEFAULT_TTL_DAYS = 30


def default_path():
    override = os.environ.get("CME_FIND_SEEN_PATH", "").strip()
    return Path(override) if override else Path.home() / ".cache" / "cme" / "find-seen-v1.json"


def build_scope(*, country: str, offer: dict, selection_mode: str, platform: str = "tiktok") -> dict:
    offer = offer if isinstance(offer, dict) else {}
    identity = {
        "target_offer": str(offer.get("target_offer") or offer.get("name") or "").strip().casefold(),
        "product_type": str(offer.get("product_type") or "").strip().casefold(),
        "product_category": str(offer.get("product_category") or "").strip().casefold(),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    scope = {"platform": str(platform or "tiktok").strip().casefold(),
             "country": str(country or "").strip().upper(),
             "offer_fingerprint": fingerprint,
             "selection_mode": str(selection_mode or "product").strip().casefold()}
    scope["scope_id"] = hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    return scope


def source_identity_keys(record: dict | None) -> set[str]:
    record = record if isinstance(record, dict) else {}
    keys = set()
    ad_id, video_id = str(record.get("ad_id") or "").strip(), str(record.get("video_id") or "").strip()
    if ad_id:
        keys.update((ad_id, "ad:" + ad_id))
    if video_id:
        keys.add("video:" + video_id)
    raw_url = str(record.get("source_url") or record.get("url") or "").strip()
    if raw_url:
        try:
            parsed = urlsplit(raw_url)
            canonical = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), (parsed.path or "/").rstrip("/") or "/", "", "")) if parsed.scheme and parsed.netloc else raw_url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        except Exception:
            canonical = raw_url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        if canonical:
            keys.add("url:" + canonical)
    return keys


class SeenLedger:
    def __init__(self, path=None, *, ttl_days: int = DEFAULT_TTL_DAYS, clock=None):
        self.path = Path(path) if path is not None else default_path()
        self.ttl_seconds = max(1, int(ttl_days)) * 86400
        self.clock = clock or time.time

    def _read(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != VERSION or not isinstance(data.get("entries"), dict):
                return {"version": VERSION, "entries": {}}
            return data
        except (OSError, ValueError, TypeError):
            return {"version": VERSION, "entries": {}}

    def _clean(self, data, now):
        cutoff, changed = now - self.ttl_seconds, False
        for scope_id in list(data["entries"]):
            entry = data["entries"].get(scope_id)
            if not isinstance(entry, dict):
                del data["entries"][scope_id]
                changed = True
                continue
            sources = entry.get("sources")
            if not isinstance(sources, dict):
                sources = {}
                entry["sources"] = sources
                changed = True
            for key, stamp in list(sources.items()):
                try:
                    expired = float(stamp) < cutoff
                except (TypeError, ValueError):
                    expired = True
                if expired:
                    del sources[key]
                    changed = True
            queries = entry.get("queries")
            if not isinstance(queries, list):
                entry["queries"] = []
                changed = True
            else:
                fresh = []
                for item in queries:
                    try:
                        keep = float(item.get("seen_at", 0)) >= cutoff
                    except (AttributeError, TypeError, ValueError):
                        keep = False
                    if keep:
                        fresh.append(item)
                if fresh != queries:
                    entry["queries"] = fresh
                    changed = True
            if not entry.get("sources") and not entry.get("queries"):
                del data["entries"][scope_id]
                changed = True
        return data, changed

    def _write(self, data):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                    stream.write("\n")
                os.replace(tmp, self.path)
            finally:
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass
            return True
        except OSError:
            return False

    def seen_keys(self, scope: dict) -> set[str]:
        data, changed = self._clean(self._read(), self.clock())
        entry = data["entries"].get(str(scope.get("scope_id") or ""), {})
        if changed:
            self._write(data)
        return set((entry.get("sources") or {}).keys()) if isinstance(entry, dict) else set()

    def recent_queries(self, scope: dict, limit: int = 50) -> list[str]:
        data, changed = self._clean(self._read(), self.clock())
        entry = data["entries"].get(str(scope.get("scope_id") or ""), {})
        rows = entry.get("queries", []) if isinstance(entry, dict) else []
        result = [str(row.get("query")) for row in rows if isinstance(row, dict) and row.get("query")]
        if changed:
            self._write(data)
        return result[-max(0, int(limit)):]

    def remember(self, scope: dict, records=(), queries=()):
        now = float(self.clock())
        data, _ = self._clean(self._read(), now)
        scope_id = str(scope.get("scope_id") or "")
        if not scope_id:
            return False
        entry = data["entries"].setdefault(scope_id, {"scope": dict(scope), "sources": {}, "queries": []})
        if not isinstance(entry, dict):
            entry = {"scope": dict(scope), "sources": {}, "queries": []}
            data["entries"][scope_id] = entry
        entry["scope"] = dict(scope)
        sources = entry.setdefault("sources", {})
        for record in records or ():
            for key in source_identity_keys(record):
                sources[key] = now
        history = entry.setdefault("queries", [])
        seen = {str(row.get("query") or "").casefold() for row in history if isinstance(row, dict)}
        for query in queries or ():
            query = " ".join(str(query or "").split())
            if query and query.casefold() not in seen:
                history.append({"query": query, "seen_at": now})
                seen.add(query.casefold())
        entry["queries"] = history[-200:]
        return self._write(data)
