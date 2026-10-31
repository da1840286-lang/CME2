"""Vertical rules live in data (verticals/*.json), never in engine code.

Two layers: claim substantiation patterns (apply to every offer in the vertical) and
category rules (platform blocks, audience requirements, restricted context).
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from .atoms import norm


class GateError(Exception):
    def __init__(self, reasons):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass(frozen=True)
class GateResult:
    blocked: tuple
    requirements: tuple


@dataclass(frozen=True)
class Vertical:
    vertical_id: str
    geo: str
    status: str
    patterns: tuple      # (id, compiled regex)
    category_rules: dict


def load_vertical(path) -> Vertical:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    pats = tuple((p["id"], re.compile(p["regex"], re.IGNORECASE)) for p in d.get("substantiation_patterns", []))
    return Vertical(d["vertical_id"], d["geo"].upper(), d.get("status", ""), pats, d.get("category_rules", {}))


def offer_gate(offer, vertical: Vertical) -> GateResult:
    """Offer-level gate, run before any paid call."""
    blocked, reqs = [], []
    if offer.geo != vertical.geo:
        blocked.append(f"offer geo {offer.geo} is not covered by vertical {vertical.vertical_id} ({vertical.geo})")
    rule = vertical.category_rules.get(offer.product_category)
    if rule:
        if rule.get("blocked_platforms") and not offer.platform:
            blocked.append(f"category {offer.product_category} needs an explicit platform (blocked on: "
                           f"{', '.join(rule['blocked_platforms'])})")
        elif offer.platform in rule.get("blocked_platforms", []):
            blocked.append(f"category {offer.product_category} is not allowed on {offer.platform}")
        for r in rule.get("requirements", []):
            reqs.append(r)
            if r not in offer.confirmations:
                blocked.append(f"requirement {r!r} is not confirmed: verify it, then add it to the offer card 'confirmations'")
    return GateResult(tuple(blocked), tuple(reqs))


def check_vertical(atoms, offer, vertical: Vertical) -> dict:
    """Atom-level checks. Claims matching a pattern must be verbatim verified claims."""
    verified = {norm(c) for c in offer.verified_claims}
    rule = vertical.category_rules.get(offer.product_category, {})
    ctx = [re.compile(p, re.IGNORECASE) for p in rule.get("restricted_context_patterns", [])]
    errors = {}
    for a in atoms:
        errs, low = [], norm(a.text)
        for pid, rx in vertical.patterns:
            if rx.search(a.text) and low not in verified:
                errs.append(f"NEEDS_SUBSTANTIATION[{pid}]: add as a verified claim in the offer card or remove")
        for rx in ctx:
            if rx.search(a.text):
                errs.append(f"RESTRICTED_CONTEXT[{offer.product_category}]: weapon-context language")
        if errs:
            errors[a.id] = errs
    return errors
