"""Atoms: the contract format between Brain and code.

Brain writes atoms, never finished prompts. Code validates, assembles, budgets.
"""
from __future__ import annotations
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict

MAX_TEXT = 120
PRIORITIES = ("MUST", "SHOULD", "MAY")
CHANNELS = ("image", "motion", "qa")
KINDS = ("scene", "product", "action", "composition", "light", "claim", "anchor", "text_rule", "motion", "qa_check")
ID_RE = re.compile(r"[a-z0-9_]{2,40}")

# Strict response schemas: source-transfer calls must also return transfer_decision.
# JSON-schema strict mode requires every declared object property to be required.
_ATOM_ITEM_SCHEMA = {
    "type": "object",
    "required": ["id", "kind", "text", "priority", "channel", "shot"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z0-9_]{2,40}$"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "text": {"type": "string", "minLength": 3, "maxLength": MAX_TEXT},
        "priority": {"type": "string", "enum": list(PRIORITIES)},
        "channel": {"type": "string", "enum": list(CHANNELS)},
        "shot": {"type": "integer", "minimum": 1},
    },
}
_TRANSFER_DECISION_SCHEMA = {
    "type": "object",
    "required": ["transferable", "abstract_mechanism", "preserved", "reason"],
    "additionalProperties": False,
    "properties": {
        "transferable": {"type": "boolean"},
        "abstract_mechanism": {"type": "string", "maxLength": 200},
        "preserved": {"type": "array", "maxItems": 4, "items": {
            "type": "object", "required": ["property", "atom_id"], "additionalProperties": False,
            "properties": {
                "property": {"type": "string", "maxLength": 160},
                "atom_id": {"type": "string", "pattern": "^[a-z0-9_]{2,40}$"},
            },
        }},
        "reason": {"type": "string", "maxLength": 240},
    },
}
ATOM_SCHEMA = {
    "type": "object", "required": ["atoms"], "additionalProperties": False,
    "properties": {"atoms": {"type": "array", "items": _ATOM_ITEM_SCHEMA}},
}
TRANSFER_ATOM_SCHEMA = {
    "type": "object", "required": ["atoms", "transfer_decision"], "additionalProperties": False,
    "properties": {
        "atoms": {"type": "array", "items": _ATOM_ITEM_SCHEMA},
        "transfer_decision": _TRANSFER_DECISION_SCHEMA,
    },
}

@dataclass(frozen=True)
class Atom:
    id: str
    kind: str
    text: str
    priority: str
    channel: str
    shot: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParseResult:
    atoms: list
    errors: dict  # atom id (or "#index") -> list of reasons


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().strip(".,;:!").lower()


def parse_atoms(raw) -> ParseResult:
    atoms, errors = [], {}
    if not isinstance(raw, list):
        return ParseResult([], {"#root": ["atoms must be a list"]})
    # Count only string IDs. Malformed list/dict IDs must not crash Counter.
    counts = Counter(item.get("id") for item in raw
                     if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id"))
    allowed_fields = {"id", "kind", "text", "priority", "channel", "shot"}
    expected_types = {"id": str, "kind": str, "text": str, "priority": str, "channel": str}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors[f"#{i}"] = ["atom must be an object"]
            continue
        raw_id = item.get("id")
        key = raw_id if isinstance(raw_id, str) and raw_id else f"#{i}"
        errs = []
        extra = set(item) - allowed_fields
        if extra:
            errs.append("unknown field(s): " + ", ".join(sorted(map(str, extra))))
        for field, typ in expected_types.items():
            if field not in item or item[field] is None or item[field] == "":
                errs.append(f"missing {field}")
            elif type(item[field]) is not typ:
                errs.append(f"{field} must be a string")
        if "shot" not in item or item.get("shot") is None:
            errs.append("missing shot")
        elif type(item["shot"]) is not int or item["shot"] < 1:
            errs.append("shot must be a positive integer")
        if not errs:
            if not ID_RE.fullmatch(item["id"]):
                errs.append("id must match [a-z0-9_]{2,40}")
            if item["kind"] not in KINDS:
                errs.append(f"kind must be one of {', '.join(KINDS)}: product claims must use kind 'claim'")
            if item["kind"] == "qa_check" and item["channel"] != "qa":
                errs.append("QA_CHECK_CHANNEL: kind='qa_check' must use channel 'qa'; verification criteria must not be image-prompt text")
            if item["priority"] not in PRIORITIES:
                errs.append(f"bad priority {item['priority']}")
            if item["channel"] not in CHANNELS:
                errs.append(f"bad channel {item['channel']}")
            text = item["text"].strip()
            if len(text) > MAX_TEXT:
                errs.append(f"text {len(text)} chars > {MAX_TEXT}: split or shorten")
            if len(text) < 3:
                errs.append("text too short")
            if counts[item["id"]] > 1:
                errs.append(f"duplicate id used {counts[item['id']]} times: every occurrence is rejected")
        if errs:
            errors.setdefault(key, []).extend(errs)
            continue
        atoms.append(Atom(item["id"], item["kind"], item["text"].strip(),
                          item["priority"], item["channel"], item["shot"]))
    return ParseResult(atoms, errors)


def check_claims(atoms, offer) -> dict:
    """Code-enforced product truth: claim atoms carry verified claims verbatim."""
    verified = {norm(c) for c in offer.verified_claims}
    forbidden = [norm(c) for c in offer.forbidden_claims]
    errors = {}
    for a in atoms:
        errs = []
        if a.kind == "claim" and a.channel != "qa":
            errs.append("CLAIM_CHANNEL: verified claims are offer facts; use channel 'qa' and a separate image atom for visible evidence")
        if a.kind == "claim" and norm(a.text) not in verified:
            errs.append("UNVERIFIED_CLAIM: claim atom must be a verified claim verbatim")
        low = norm(a.text)
        for f in forbidden:
            if f and f in low:
                errs.append(f"FORBIDDEN_CLAIM: contains '{f}'")
        if errs:
            errors[a.id] = errs
    return errors


def check_language(atoms) -> dict:
    """Atom text goes straight into an English-trained image prompt: reject non-Latin letters.

    Accented Latin letters (product names) are fine; Cyrillic, CJK, Arabic and the like are not.
    """
    errors = {}
    for a in atoms:
        bad = sorted({ch for ch in a.text if ch.isalpha() and "LATIN" not in unicodedata.name(ch, "")})
        if bad:
            errors[a.id] = [f"NON_ENGLISH_ATOM: atom text must be English; found {''.join(bad[:6])!r}"]
    return errors
