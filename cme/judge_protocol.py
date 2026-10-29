"""Two-phase visual QA protocol: observe first, verify second, decide in code.

Phase one inventories visible facts without showing requirements. Phase two checks the inventory against
requirements without seeing pixels. The code verifies quote provenance and requires role-specific proof
conditions for mechanism-result and intervention atoms before allowing PASS.
"""
from __future__ import annotations

import json
import re
from collections import Counter

MAX_OBSERVATIONS = 24
MIN_OBSERVATIONS = 3
MAX_PREDICATES = 6
MIN_QUOTE_CHARS = 6
MAX_OBSERVATION_CHARS = 200
ASPECTS = {"problem_state", "changed_state", "contact_action", "object_presence", "other"}
ROLE_ASPECTS = {
    "mech_result_": {"problem_state", "changed_state"},
    "mech_intervention_": {"problem_state", "contact_action"},
}

OBSERVE_SYSTEM = """You are a careful visual inspector. You will see ONE image. You are NOT told what the image is supposed to show. Do not guess purposes, brands, or intentions.
List only concrete, visible facts as short observations. Cover, when visible:
- every distinct object: what it is, where it is in the frame (left/center/right, top/middle/bottom), and its state;
- every region whose surface condition differs from its neighbours, including its location, relative size, and borders;
- adjacent objects or regions: clearly touching/overlapping, clearly separate, or unclear;
- any visible text or logo, copied as seen.
Rules: one fact per observation, at most 25 words each, no reasoning, no requirements, no 'probably'. If something is unclear, say 'unclear'. Never state a fact you cannot see.
Reply ONLY with JSON: {"observations": [{"id": "o1", "kind": "object|region|relation|text|other", "text": "..."}]} with 3 to 24 items. If fewer than 3 meaningful facts are visible, say so rather than inventing facts."""

VERIFY_SYSTEM = """You verify requirements against a written inventory of what is visible in an image. You do NOT see the image. The inventory is the only evidence.
For each requirement, split it into the smallest independent conditions (1 to 6). For each condition give:
- "aspect": one of problem_state, changed_state, contact_action, object_presence, other;
- "claim": the condition in short words;
- "observation_ids": inventory ids that directly show it ([] if none);
- "quote": up to 12 words copied EXACTLY from one cited observation ("" if none);
- "holds": true only if the cited observations plainly show it; false if they contradict it, are ambiguous, or do not mention it.
For ids starting mech_result_, you MUST include both problem_state and changed_state conditions. For ids starting mech_intervention_, you MUST include both problem_state and contact_action conditions. For all other ids use the applicable aspect(s), including object_presence when relevant.
Absence of evidence is false. Do not infer an after-state, contact, or effect that the inventory does not state. Nearness is not contact. Do not omit any part of a requirement.
For requirements that claim a change or result, include one condition for the starting/problem state and one for the visibly changed state. For requirements that claim an action on something, include a condition for visible contact or action between the two things.
Reply ONLY with JSON: {"results": [{"id": "<requirement id>", "conditions": [{"aspect": "...", "claim": "...", "observation_ids": ["o1"], "quote": "...", "holds": true}]}]}"""


def _norm(text) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text or "").lower())).strip()


def parse_observations(data) -> list[dict] | None:
    """Return a bounded, de-duplicated inventory or None if phase one is not trustworthy."""
    rows = data.get("observations") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    seen, out = set(), []
    for row in rows[:MAX_OBSERVATIONS]:
        if not isinstance(row, dict):
            continue
        oid, text = row.get("id"), row.get("text")
        if not isinstance(oid, str) or not oid.strip() or not isinstance(text, str) or not text.strip():
            continue
        oid = oid.strip()
        if oid in seen:
            continue
        kind = str(row.get("kind") or "other")[:16]
        if kind not in {"object", "region", "relation", "text", "other"}:
            kind = "other"
        seen.add(oid)
        out.append({"id": oid, "kind": kind, "text": " ".join(text.split())[:MAX_OBSERVATION_CHARS]})
    return out if len(out) >= MIN_OBSERVATIONS else None


def verify_user_payload(observations: list[dict], atoms) -> str:
    reqs = [{"id": a.id, "requirement": a.text} for a in atoms]
    return json.dumps({"inventory": observations, "requirements": reqs}, ensure_ascii=False)


def _condition_supported(condition: dict, by_id: dict) -> tuple[bool, str, str]:
    """Return support, source observation text, and a failure reason. The model is not the final arbiter."""
    claim = str(condition.get("claim") or "").strip()
    if condition.get("holds") is not True:
        return False, "", f"not confirmed: {claim or 'condition'}"
    aspect = condition.get("aspect")
    if aspect not in ASPECTS:
        return False, "", f"invalid proof aspect: {claim or 'condition'}"
    ids = condition.get("observation_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in by_id for i in ids):
        return False, "", f"missing/unknown observation citation: {claim}"
    quote = _norm(condition.get("quote"))
    if len(quote) < MIN_QUOTE_CHARS:
        return False, "", f"no verifiable quote: {claim}"
    cited = [by_id[i] for i in ids]
    for obs in cited:
        if quote in _norm(obs["text"]):
            return True, obs["text"], ""
    return False, "", f"quote not found in cited observation: {claim}"


def decide(data, atoms, observations: list[dict]) -> dict:
    """Return decisions only for unique, structurally valid model answers.

    Missing, duplicate, or malformed answers are omitted so upstream QA treats them as NOT_RUN (never PASS).
    For mechanism results/interventions, required proof aspects are enforced in code, not only requested in a prompt.
    """
    rows = data.get("results") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}
    by_id = {o["id"]: o for o in observations}
    wanted = {a.id for a in atoms}
    counts = Counter(r.get("id") for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)
                     and r.get("id") in wanted)
    out = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        atom_id = row["id"]
        if atom_id not in wanted or counts[atom_id] != 1:
            continue
        conditions = row.get("conditions")
        if not isinstance(conditions, list) or not conditions or len(conditions) > MAX_PREDICATES:
            continue
        if any(not isinstance(c, dict) for c in conditions):
            continue
        aspects = {c.get("aspect") for c in conditions}
        required = next((value for prefix, value in ROLE_ASPECTS.items() if atom_id.startswith(prefix)), set())
        if not required.issubset(aspects):
            missing = sorted(required - aspects)
            out[atom_id] = {"pass": False, "evidence": ("no visual confirmation - missing proof aspect: " + ", ".join(missing))[:120],
                            "conditions": len(conditions)}
            continue
        evidences, failure = [], ""
        for condition in conditions:
            ok, evidence, reason = _condition_supported(condition, by_id)
            if not ok:
                failure = reason
                break
            if evidence not in evidences:
                evidences.append(evidence)
        if failure:
            out[atom_id] = {"pass": False, "evidence": ("no visual confirmation - " + failure)[:120],
                            "conditions": len(conditions)}
        else:
            out[atom_id] = {"pass": True, "evidence": " | ".join(evidences)[:120],
                            "conditions": len(conditions)}
    return out


