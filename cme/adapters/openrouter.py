"""OpenRouter adapters: Brain (structured atoms) and a batch vision judge. Model id is explicit config."""
from __future__ import annotations
import json
import os
import re
from collections import Counter
from pathlib import Path
from ..atoms import ATOM_SCHEMA, TRANSFER_ATOM_SCHEMA, KINDS
from ..providers import InfraError, ProviderCallError
from .http import check_status, data_uri, urllib_transport

URL = "https://openrouter.ai/api/v1/chat/completions"


def _chat(params, api_key, send, messages, response_format=None):
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ProviderCallError("OPENROUTER_API_KEY is not set")
    body = {"model": params["model_id"], "messages": messages,
            "temperature": params.get("temperature", 0.2), "max_tokens": params.get("max_tokens", 4000)}
    if params.get("reasoning") is not None:
        body["reasoning"] = params["reasoning"]
    if response_format:
        body["response_format"] = response_format
    status, raw = send("POST", URL, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body)
    check_status("openrouter", status, raw)
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ProviderCallError(f"non-JSON response: {raw[:200]!r}") from e
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        code = err.get("code") if isinstance(err, dict) else None
        msg = err.get("message", err) if isinstance(err, dict) else err
        if code == 429 or (isinstance(code, int) and code >= 500):
            raise InfraError(f"openrouter error {code}: {msg}")
        raise ProviderCallError(f"openrouter error {code}: {msg}")
    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise ProviderCallError(f"unexpected response shape: {raw[:200]!r}") from e
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    meta = {
        "model": data.get("model"),
        "cost": usage.get("cost"),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "message_has_reasoning": bool(message.get("reasoning") or message.get("reasoning_details")),
        "message_has_refusal": bool(message.get("refusal")),
    }
    return content or "", meta


def extract_json(text: str):
    """Models wrap JSON in fences or prose; take the first object. None when there is none."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        try:
            return json.loads(cand)
        except ValueError:
            continue
    return None


def _safe_response_excerpt(content: str, limit: int = 400) -> str:
    """Return a short model-output sample for diagnostics without exposing common API-key forms."""
    excerpt = str(content or "")
    excerpt = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", excerpt)
    excerpt = re.sub(
        r"(?i)\bsk-(?:or-v1|proj|ant|live)-[A-Za-z0-9_-]{8,}|\bsk-[A-Za-z0-9_-]{20,}",
        "[REDACTED]", excerpt)
    excerpt = re.sub(
        r"(?i)\b(api[_ -]?key|authorization|access[_ -]?token|refresh[_ -]?token|secret)\s*[:=]\s*['\"]?[^,\s;'\"]+",
        r"\1=[REDACTED]", excerpt)
    return re.sub(r"\s+", " ", excerpt).strip()[:limit]


def _format(params, schema=None):
    mode = params.get("response_format_mode", "json_object")
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "atoms", "strict": True, "schema": schema or ATOM_SCHEMA}}
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def make_openrouter_brain(params: dict, api_key=None, transport=None):
    send = transport or urllib_transport

    def call(request: dict):
        if request.get("task") == "target_fit_gate":
            root_shape = '{"atoms": [], "transfer_decision": {...}}'
        else:
            root_shape = '{"atoms": [...], "transfer_decision": {...}}' if request.get("source_transfer") else '{"atoms": [...]}'
        system = ("You are the Brain of an ad production system. Reply with ONE JSON object " + root_shape + " "
                  "and nothing else. Each atom: id, kind, text (max 120 chars), priority (MUST/SHOULD/MAY), "
                  f"channel (image/motion/qa), shot. Allowed kinds: {', '.join(KINDS)}. "
                  + " ".join(request.get("rules", [])) + " JSON schema: " + json.dumps(request["schema"]))
        user = {k: request.get(k) for k in ("task", "offer", "brief", "shots", "only_fix", "source_transfer", "mechanism_links", "required_source_elements") if request.get(k) is not None}
        content, meta = _chat(params, api_key, send,
                              [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                              _format(params, request.get("schema", ATOM_SCHEMA)))
        data = extract_json(content)
        invalid_reason = None
        if not str(content or "").strip():
            invalid_reason = "empty_response"
        elif data is None:
            # Distinguish valid JSON of the wrong root type from text that is not JSON.
            try:
                decoded = json.loads(content)
            except (TypeError, ValueError):
                invalid_reason = "invalid_json"
            else:
                invalid_reason = "json_root_not_object" if not isinstance(decoded, dict) else "missing_atoms_list"
        elif not isinstance(data, dict):
            invalid_reason = "json_root_not_object"
        elif "atoms" not in data:
            invalid_reason = "missing_atoms_list"
        elif not isinstance(data.get("atoms"), list):
            invalid_reason = "atoms_not_list"

        if invalid_reason:
            # Logged by call_with_policy as a bounded diagnostic. Never log raw output:
            # the excerpt is flattened, length-limited, and stripped of common API-key forms.
            meta = {**meta, "diagnostic": {
                "component": "openrouter_brain",
                "reason": invalid_reason,
                "response_excerpt": _safe_response_excerpt(content),
                "response_chars": len(str(content or "")),
                "finish_reason": meta.get("finish_reason"),
                "completion_tokens": meta.get("completion_tokens"),
                "reasoning_tokens": meta.get("reasoning_tokens"),
                "message_has_reasoning": meta.get("message_has_reasoning"),
                "message_has_refusal": meta.get("message_has_refusal"),
            }}
            return {"atoms": [], "_error": f"response rejected: {invalid_reason}"}, meta
        return data, meta

    return call


def make_openrouter_judge(params: dict, api_key=None, transport=None):
    """Judge one generated frame; optional target-product references enable an identity check in the same call."""
    send = transport or urllib_transport
    identity_id = "cme_product_identity_ref"

    def judge(image_path, atoms, reference_images=()):
        refs = [Path(p) for p in list(reference_images or ())[:3]]
        for p in refs:
            if not p.is_file() or p.stat().st_size == 0:
                raise ProviderCallError(f"target-product reference is missing or empty: {p.name}")
        system = ("You verify one generated image against numbered requirements. Be strict: if you cannot clearly see "
                  "the requirement satisfied, answer false. Reply ONLY with JSON "
                  "{\"results\": [{\"id\": \"...\", \"pass\": true|false, \"evidence\": \"<12 words\"}]}.")
        reqs = [{"id": a.id, "requirement": a.text} for a in atoms]
        if refs:
            reqs.append({"id": identity_id, "requirement": (
                "PRODUCT IDENTITY: compare the product visible in the generated frame with the original target-product reference image(s). "
                "Ignore differences in background, lighting, camera, composition, branding and printed/embossed text. "
                "Treat logo/label removal as intentional under the campaign branding policy; do not fail identity merely because those marks are absent. "
                "PASS only if silhouette, construction, color, materials, controls and other non-text physical details are consistent. "
                "FAIL if the product is absent or materially different. If the reference or generated view does not support a confident comparison, "
                "answer false. Evidence must describe visible product details, not prompt text.")})
        user_content = [{"type": "text", "text": json.dumps(reqs, ensure_ascii=False)}]
        user_content.append({"type": "text", "text": "IMAGE A — GENERATED FRAME TO QA. Check every requested atom against this image."})
        user_content.append({"type": "image_url", "image_url": {"url": data_uri(image_path)}})
        for i, ref in enumerate(refs, start=1):
            user_content.append({"type": "text", "text": f"REFERENCE IMAGE {i} — ORIGINAL TARGET-PRODUCT REFERENCE. Use this image only to check product identity; do not copy its background or composition."})
            user_content.append({"type": "image_url", "image_url": {"url": data_uri(ref)}})
        content, meta = _chat(params, api_key, send, [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}],
            {"type": "json_object"} if params.get("response_format_mode", "json_object") != "prompt" else None)
        data = extract_json(content)
        out = {}
        requested_ids = {a.id for a in atoms}
        if refs:
            requested_ids.add(identity_id)
        rows = data.get("results") if isinstance(data, dict) else None
        matching_rows = []
        duplicate_ids = []
        rows_invalid_pass = 0
        rows_missing_evidence = 0
        if isinstance(rows, list):
            ids = [r.get("id") for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)
                   and r.get("id") in requested_ids]
            counts = Counter(ids)
            duplicate_ids = sorted(rid for rid, count in counts.items() if count > 1)
            matching_rows = [r for r in rows if isinstance(r, dict)
                             and isinstance(r.get("id"), str) and r.get("id") in requested_ids]
            for r in rows:
                if not isinstance(r, dict) or not isinstance(r.get("id"), str):
                    continue
                rid, passed, evidence = r["id"], r.get("pass"), r.get("evidence")
                if rid not in requested_ids:
                    continue
                if counts[rid] != 1 or type(passed) is not bool or not isinstance(evidence, str) or not evidence.strip():
                    if type(passed) is not bool: rows_invalid_pass += 1
                    if not isinstance(evidence, str) or not evidence.strip(): rows_missing_evidence += 1
                    continue
                out[rid] = {"pass": passed, "evidence": evidence.strip()[:120]}
        if len(out) != len(requested_ids):
            if not str(content or "").strip(): reason = "empty_response"
            elif data is None: reason = "invalid_json"
            elif not isinstance(data, dict): reason = "json_root_not_object"
            elif not isinstance(rows, list): reason = "missing_results_list"
            elif not rows: reason = "empty_results_list"
            elif not matching_rows: reason = "no_requested_ids_in_results"
            elif not out: reason = "all_matching_rows_rejected"
            else: reason = "partial_results"
            meta = {**meta, "diagnostic": {
                "component": "openrouter_judge", "reason": reason,
                "expected_ids": len(requested_ids), "valid_ids": len(out),
                "matching_rows": len(matching_rows), "duplicate_ids": duplicate_ids[:10],
                "rows_invalid_pass": rows_invalid_pass, "rows_missing_evidence": rows_missing_evidence,
                "missing_ids": sorted(requested_ids - set(out))[:10],
                "response_excerpt": _safe_response_excerpt(content),
                "finish_reason": meta.get("finish_reason"), "completion_tokens": meta.get("completion_tokens"),
                "reasoning_tokens": meta.get("reasoning_tokens"),
                "message_has_reasoning": meta.get("message_has_reasoning"),
                "message_has_refusal": meta.get("message_has_refusal")}}
        return out, meta

    return judge


ANALYST_SYSTEM = (
    "You analyse ONE short product ad from a few still frames taken at known timestamps. You did NOT see the full video. "
    "Reply ONLY with JSON: {\"hook\":S,\"buyer_problem\":S,\"product_intervention\":S,\"result_demo\":S,"
    "\"evidence\":[{\"t_sec\":number,\"frame\":int,\"observation\":str}],\"cta\":S,\"mechanism_hypothesis\":S,"
    "\"transferability\":S,\"limits\":{\"text\":str,\"missing\":[str]},"
    "\"ratings\":{\"problem_clarity\":1-5|null,\"product_demo\":1-5|null,\"proof_visibility\":1-5|null,\"sequence_quality\":1-5|null}} "
    "where S = {\"text\": str, \"kind\": \"OBSERVED\"|\"INFERRED\"}. OBSERVED only for what is visible in the frames; "
    "interpretations are INFERRED. mechanism_hypothesis and transferability are always INFERRED and are hypotheses, not proof "
    "of profitability. evidence t_sec must be one of the given frame timestamps. If something cannot be known from the frames, "
    "say so in limits.missing and use null ratings. Write all text values in English."
)


def make_openrouter_analyst(params: dict, api_key=None, transport=None):
    """analyst(frames, context) -> ({'data': dict|None, 'raw_excerpt': str}, meta). frames: [{'t_sec', 'path'}]."""
    send = transport or urllib_transport

    def analyze(frames, context):
        header = {"ad": context, "frames": [{"index": i, "t_sec": f["t_sec"]} for i, f in enumerate(frames)]}
        content = [{"type": "text", "text": json.dumps(header, ensure_ascii=False)}]
        content += [{"type": "image_url", "image_url": {"url": data_uri(f["path"])}} for f in frames]
        text, meta = _chat(params, api_key, send,
                           [{"role": "system", "content": ANALYST_SYSTEM}, {"role": "user", "content": content}],
                           {"type": "json_object"} if params.get("response_format_mode", "json_object") != "prompt" else None)
        return {"data": extract_json(text), "raw_excerpt": text[:1500]}, meta

    return analyze
