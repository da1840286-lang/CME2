"""Cross-run cache of PASS verdicts for prepared no_branding product references.

A prepared reference is a pure function of (source photo, masking sweep) and its visual check is a pure function of
(prepared pixels, judge rule). Re-asking a free judge model on every CI run only adds a way to fail on a provider
quota. The cache therefore stores ONLY a PASS verdict and serves it only when every input still matches:
  - the source photo bytes (sha256),
  - the prepared PNG bytes (sha256) - tampering or corruption is a miss,
  - the exact judge rule text and masking sweep (a stricter rule invalidates old verdicts),
  - an age limit (default 30 days; CME_REFERENCE_CACHE_DAYS).
FAIL, NOT_RUN and any unverified preparation are never cached. Disabled unless CME_REFERENCE_CACHE_DIR is set.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

CACHE_VERSION = 1


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReferenceVerificationCache:
    def __init__(self, directory, rule_text: str, sweep: int, max_age_days: float = 30.0,
                 algorithm_version: str = "", now=time.time):
        self.dir = Path(directory)
        self.rule_sha = hashlib.sha256(str(rule_text).encode("utf-8")).hexdigest()
        self.sweep = int(sweep)
        self.algorithm_version = str(algorithm_version or "")
        self.max_age = float(max_age_days) * 86400.0
        self.now = now

    @classmethod
    def from_env(cls, rule_text: str, sweep: int):
        directory = os.environ.get("CME_REFERENCE_CACHE_DIR", "").strip()
        if not directory:
            return None
        try:
            days = float(os.environ.get("CME_REFERENCE_CACHE_DAYS", "30"))
        except ValueError:
            days = 30.0
        # Cache entries must never outlive a change to the reference-masking algorithm.
        from .branding import REFERENCE_PREPARATION_VERSION
        return cls(directory, rule_text, sweep, days, algorithm_version=REFERENCE_PREPARATION_VERSION)

    def _key(self, source) -> str:
        return hashlib.sha256(
            f"{CACHE_VERSION}|{self.algorithm_version}|{_sha(source)}|{self.rule_sha}|{self.sweep}".encode()
        ).hexdigest()[:40]

    def restore(self, source, destination) -> bool:
        """Copy a still-valid PASS entry next to `destination`; True only when it was restored."""
        try:
            key = self._key(source)
            png, meta_path = self.dir / f"{key}.png", self.dir / f"{key}.prep.json"
            if not (png.is_file() and meta_path.is_file()):
                return False
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            verification = meta.get("verification") or {}
            valid = (
                meta.get("cache_version") == CACHE_VERSION
                and verification.get("status") == "PASS"
                and meta.get("source_sha256") == _sha(source)
                and meta.get("prepared_sha256") == _sha(png)
                and meta.get("rule_sha256") == self.rule_sha
                and meta.get("sweep") == self.sweep
                and (not self.algorithm_version or str(meta.get("algorithm_version") or "") == self.algorithm_version)
                and self.now() - float(meta.get("verified_at", 0)) <= self.max_age
            )
            if not valid:
                return False
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(png, destination)
            restored = {k: v for k, v in meta.items() if k not in ("cache_version", "rule_sha256", "verified_at")}
            restored["prepared_sha256"] = _sha(destination)
            destination.with_suffix(".prep.json").write_text(
                json.dumps(restored, ensure_ascii=False, indent=1), encoding="utf-8")
            return True
        except (OSError, ValueError, TypeError):
            return False

    def store(self, source, destination) -> bool:
        """Persist a PASS-verified preparation. Anything else is refused."""
        try:
            destination = Path(destination)
            sidecar = destination.with_suffix(".prep.json")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if (meta.get("verification") or {}).get("status") != "PASS":
                return False
            if (meta.get("source_sha256") != _sha(source) or meta.get("prepared_sha256") != _sha(destination)
                    or (self.algorithm_version and str(meta.get("algorithm_version") or "") != self.algorithm_version)):
                return False
            key = self._key(source)
            self.dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination, self.dir / f"{key}.png")
            entry = {**meta, "cache_version": CACHE_VERSION, "rule_sha256": self.rule_sha,
                     "verified_at": self.now()}
            (self.dir / f"{key}.prep.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
            return True
        except (OSError, ValueError, TypeError):
            return False
