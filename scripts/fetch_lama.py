#!/usr/bin/env python3
"""Download the LaMa TorchScript weights used by the no_branding reference cleaner and verify their SHA-256.

Default location: ~/.cache/cme/big-lama.pt (override with CME_LAMA_WEIGHTS). Source: GitHub release asset of
enesmsahin/simple-lama-inpainting (big-lama, ~206 MB). Idempotent: an existing file with the right hash is kept.
"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    dest = Path(os.environ.get("CME_LAMA_WEIGHTS") or Path.home() / ".cache" / "cme" / "big-lama.pt")
    if dest.is_file() and sha256(dest) == SHA256:
        print(f"LaMa weights OK: {dest}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {URL} -> {dest}")
    urllib.request.urlretrieve(URL, tmp)
    got = sha256(tmp)
    if got != SHA256:
        tmp.unlink(missing_ok=True)
        print(f"SHA-256 mismatch: {got}", file=sys.stderr)
        return 1
    tmp.replace(dest)
    print(f"LaMa weights OK: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
