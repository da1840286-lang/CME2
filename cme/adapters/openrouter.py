"""OpenRouter adapters: Brain (structured atoms) and a batch vision judge. Model id is explicit config."""
from __future__ import annotations
import json
import os
import re
from collections import Counter
from pathlib import Path
from ..atoms import ATOM_SCHEMA, TRANSFER_ATOM_SCHEMA, KINDS
from ..judge_protocol import (
    OBSERVE_SYSTEM, VERIFY_SYSTEM, decide, parse_observations, verify_user_payload,
)
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


def _write_judge_trace(image_path, atom_ids, trace, diagnostic):
    """Append one JSON line per observe/verify judge call when configured. Never raises."""
    path_value = os.environ.get("CME_JUDGE_TRACE")
    if not path_value:
        return
    try:
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"image": Path(str(image_path)).name, "atoms": atom_ids, "diagnostic": diagnostic, **trace}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 - tracing must never change a QA verdict
        pass


def make_openrouter_judge(params: dict, api_key=None, transport=None):
    """Judge one generated frame; optional target-product references enable an identity check in the same call."""
    send = transport or urllib_transport
    identity_id = "cme_product_identity_ref"

    def judge_single(image_path, atoms, reference_images=()):
        refs = [Path(p) for p in list(reference_images or ())[:3]]
        for p in refs:
            if not p.is_file() or p.stat().st_size == 0:
                raise ProviderCallError(f"target-product reference is missing or empty: {p.name}")
        system = """You are an independent, conservative visual QA judge. Inspect the actual image pixels; never infer that a requested result happened merely because the prompt says so, the tool is present, or the product is near the problem. If visible evidence is ambiguous, answer false.
For mech_result_* requirements, the after-state must be plainly visible and distinguishable from the baseline; require both a recognizable starting/problem state and a visibly changed result when the atom asks for a change. An adjacent tool or untouched problem material is not proof of success.
For mech_intervention_* requirements, require visible physical contact or an action connecting the target product to the exact problem; mere placement beside the problem is not enough. Check each atom independently.
Reply ONLY with JSON: {"results": [{"id": "...", "pass": true|false, "evidence": "brief visible evidence"}]}."""
        reqs = []
        for a in atoms:
            requirement = a.text
            if str(a.id).startswith("mech_result_"):
                requirement = (
                    "RESULT-STATE GATE: prove the visible after-state and its difference from the baseline. "
                    "FAIL if only the product/tool is visible, if the problem material appears untouched, or if "
                    "the claimed change cannot be clearly seen. " + requirement
                )
            elif str(a.id).startswith("mech_intervention_"):
                requirement = (
                    "ACTIVE-INTERVENTION GATE: prove visible contact/action connecting the target product to the "
                    "specific problem. FAIL if the product is merely nearby or posed beside it. " + requirement
                )
            reqs.append({
                "id": a.id, "kind": getattr(a, "kind", ""),
                "priority": getattr(a, "priority", ""), "channel": getattr(a, "channel", ""),
                "requirement": requirement,
            })
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

    def judge(image_path, atoms, reference_images=()):
        """Default to image-only observation, text-only verification, then a code-side decision.

        judge_mode=single_pass preserves the legacy one-call path for A/B calibration. The identity
        comparison still needs the pixels and therefore remains a separate vision call.
        """
        mode = str(params.get("judge_mode") or "observe_verify").strip().lower()
        if mode == "single_pass":
            return judge_single(image_path, atoms, reference_images)
        refs = list(reference_images or ())[:3]
        # Identity-only checks do not need an inventory/verification pair.
        if not atoms and refs:
            answers, meta = judge_single(image_path, [], refs)
            return answers, {**meta, "judge_mode": "observe_verify",
                             "judge_phases": [{"phase": "identity", "completion_tokens": meta.get("completion_tokens")}],
                             "observation_count": 0}

        pinned = {**params, "temperature": params.get("judge_temperature", 0)}
        json_format = {"type": "json_object"} if params.get("response_format_mode", "json_object") != "prompt" else None
        out, diagnostic, phases = {}, None, []

        observe_content, observe_meta = _chat(pinned, api_key, send, [
            {"role": "system", "content": OBSERVE_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Inspect this image and list only visible facts."},
                {"type": "image_url", "image_url": {"url": data_uri(image_path)}}]}], json_format)
        phases.append({"phase": "observe", "completion_tokens": observe_meta.get("completion_tokens"),
                       "cost": observe_meta.get("cost")})
        observations = parse_observations(extract_json(observe_content))
        meta = {**observe_meta}
        trace = {"observations": observations}
        if observations is not None and not observations and atoms:
            # A well-formed empty inventory is evidence that nothing was visibly confirmed, not a malformed answer.
            out.update({a.id: {"pass": False, "evidence": "inventory reports no visible facts"} for a in atoms})
            trace["decided"] = out
        elif observations is None:
            diagnostic = {"component": "openrouter_judge", "reason": "observation_phase_invalid",
                          "response_excerpt": _safe_response_excerpt(observe_content),
                          "finish_reason": observe_meta.get("finish_reason")}
        elif atoms:
            verify_content, verify_meta = _chat(pinned, api_key, send, [
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": verify_user_payload(observations, atoms)}], json_format)
            phases.append({"phase": "verify", "completion_tokens": verify_meta.get("completion_tokens"),
                           "cost": verify_meta.get("cost")})
            meta = {**verify_meta}
            verify_data = extract_json(verify_content)
            decided = decide(verify_data, atoms, observations)
            trace.update({"verify": verify_data, "decided": decided})
            out.update({k: {"pass": v["pass"], "evidence": v["evidence"]} for k, v in decided.items()})
            missing = sorted({a.id for a in atoms} - set(decided))
            if missing:
                diagnostic = {"component": "openrouter_judge", "reason": "verify_phase_partial",
                              "missing_ids": missing[:10], "response_excerpt": _safe_response_excerpt(verify_content),
                              "finish_reason": verify_meta.get("finish_reason")}

        # Identity compares generated pixels with the target reference, so it remains a vision call.
        if refs:
            identity_rows, identity_meta = judge_single(image_path, [], refs)
            out.update(identity_rows)
            phases.append({"phase": "identity", "completion_tokens": identity_meta.get("completion_tokens"),
                           "cost": identity_meta.get("cost")})
            if identity_meta.get("diagnostic") and not diagnostic:
                diagnostic = identity_meta["diagnostic"]
        _write_judge_trace(image_path, [a.id for a in atoms], trace, diagnostic)
        meta = {**meta, "judge_mode": "observe_verify", "judge_phases": phases,
                "observation_count": len(observations or [])}
        if diagnostic:
            meta["diagnostic"] = diagnostic
        return out, meta

    return judge


ANALYST_SYSTEM = """You analyse ONE short product ad from a few still frames taken at known timestamps. You did NOT see the full video.
Reply ONLY with JSON:
{
  "hook": S, "buyer_problem": S, "product_intervention": S, "result_demo": S,
  "evidence": [{"t_sec": number, "frame": int, "observation": str, "supports": [section_key, ...]}],
  "cta": S, "mechanism_hypothesis": S, "transferability": S,
  "limits": {"text": str, "missing": [str]},
  "ratings": {"problem_clarity": 1-5|null, "product_demo": 1-5|null, "proof_visibility": 1-5|null, "sequence_quality": 1-5|null}
}
Each S is {"text": str, "kind": "OBSERVED"|"INFERRED"}. OBSERVED only for what is visible in the frames; interpretation is INFERRED.
The first frame is taken near the start of the video and the last frame near its end, so the opening and final shot are both included. Compare earlier and later frames of the same object or surface and state explicitly whether anything visibly changed (appeared, disappeared, moved, filled, changed state) or stayed the same. Mark result_demo OBSERVED only when a later frame visibly differs from an earlier one in the way the text claims; if the frames show only the action and not its outcome, say so in the text and mark result_demo INFERRED. Describe the actual content of each frame, not what the ad probably intends; a close-up of a part is not proof of a changed result.
Each evidence row's supports must name the exact section keys it visually supports (hook, buyer_problem, product_intervention, result_demo, cta, mechanism_hypothesis, transferability). Every section marked OBSERVED must have at least one valid evidence row naming that section in supports.
Frame is a zero-based index from the provided frame list; t_sec must equal the timestamp at that index. Do not infer support from a nearby observation.
mechanism_hypothesis and transferability are always INFERRED and are hypotheses, not proof of profitability. If something cannot be known from the frames, say so in limits.missing and use null ratings. Write all text values in English."""


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
