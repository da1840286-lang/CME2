"""Paid image adapter over fal's HTTP API. Model path and params come from config; transport is injectable."""
from __future__ import annotations
import json
import os
from ..providers import ProviderCallError
from .http import check_status, data_uri, urllib_transport


def make_fal_image(params: dict, api_key: str | None = None, transport=None):
    """params: model_path, base_url, extra (dict), ref_param, ref_style ('list'|'single'), max_refs."""
    send = transport or urllib_transport
    base = params.get("base_url", "https://fal.run").rstrip("/")

    def generate(prompt: str, refs) -> bytes:
        key = api_key or os.getenv("FAL_KEY")
        if not key:
            raise ProviderCallError("FAL_KEY is not set")
        headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
        body = {"prompt": prompt, **params.get("extra", {})}
        ref_param = params.get("ref_param")
        if ref_param and refs:
            uris = [data_uri(r) for r in list(refs)[: params.get("max_refs", 4)]]
            body[ref_param] = uris[0] if params.get("ref_style") == "single" else uris
        status, raw = send("POST", f"{base}/{params['model_path']}", headers, body)
        check_status("fal", status, raw)
        try:
            url = json.loads(raw)["images"][0]["url"]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ProviderCallError(f"unexpected fal response: {raw[:200]!r}") from e
        status, img = send("GET", url, {}, None)
        if status >= 400:
            raise ProviderCallError(f"image download HTTP {status}")
        return img

    return generate
