"""Offer card: the only place where product facts enter the system.

The engine never knows a concrete product. Everything about the offer is data.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from .branding import BRANDING_POLICIES, TARGET_BRAND_ALLOWED, resolve_branding_contract

MAX_CLAIM_CHARS = 120


class OfferError(ValueError):
    pass


@dataclass(frozen=True)
class OfferCard:
    offer_id: str
    name: str
    product_type: str
    verified_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    photos: tuple[str, ...]
    platform: str = ""
    geo: str = "US"
    product_category: str = ""
    confirmations: tuple[str, ...] = ()
    branding_policy: str = TARGET_BRAND_ALLOWED
    approved_brand_marks: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "offer_id": self.offer_id, "name": self.name, "product_type": self.product_type,
            "verified_claims": list(self.verified_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "photos": [Path(x).name for x in self.photos], "platform": self.platform, "geo": self.geo,
            "product_category": self.product_category,
            "confirmations": list(self.confirmations),
            "branding_policy": self.branding_policy,
            "approved_brand_marks": list(self.approved_brand_marks),
        }


def load_offer(path: str | Path, check_files: bool = True) -> OfferCard:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    errors = []
    for key in ("offer_id", "name", "product_type"):
        if not str(data.get(key, "")).strip():
            errors.append(f"missing {key}")
    claims = [str(c).strip() for c in data.get("verified_claims", []) if str(c).strip()]
    if not claims:
        errors.append("verified_claims must not be empty")
    for c in claims:
        if len(c) > MAX_CLAIM_CHARS:
            errors.append(f"claim longer than {MAX_CLAIM_CHARS} chars: {c[:30]}...")
    policy = str(data.get("branding_policy", TARGET_BRAND_ALLOWED)).strip().lower()
    if policy not in BRANDING_POLICIES:
        errors.append(f"branding_policy must be one of: {', '.join(BRANDING_POLICIES)}")
    raw_marks = data.get("approved_brand_marks", [])
    if not isinstance(raw_marks, list):
        errors.append("approved_brand_marks must be a JSON array of confirmed target-brand names/marks")
        raw_marks = []
    marks = [str(mark).strip() for mark in raw_marks if isinstance(mark, str) and mark.strip()]
    if policy in BRANDING_POLICIES:
        contract = resolve_branding_contract(policy, marks, data.get("name", ""), data.get("product_type", ""))
        policy = contract["branding_policy"]
        marks = contract["approved_brand_marks"]
    photos = [str(x) for x in data.get("photos", [])]
    if not photos:
        errors.append("at least one product photo is required")
    resolved = []
    for ph in photos:
        cand = Path(ph) if Path(ph).is_absolute() else (p.parent / ph)
        if check_files and not cand.exists():
            errors.append(f"photo not found: {ph}")
        resolved.append(str(cand.resolve()))
    photos = resolved
    if errors:
        raise OfferError("; ".join(errors))
    return OfferCard(
        offer_id=str(data["offer_id"]).strip(), name=str(data["name"]).strip(),
        product_type=str(data["product_type"]).strip(),
        verified_claims=tuple(claims),
        forbidden_claims=tuple(str(c).strip() for c in data.get("forbidden_claims", []) if str(c).strip()),
        photos=tuple(photos), platform=str(data.get("platform", "")).strip().lower(),
        geo=str(data.get("geo", "US")).strip().upper() or "US",
        product_category=str(data.get("product_category", "")).strip().lower(),
        confirmations=tuple(str(c).strip() for c in data.get("confirmations", []) if str(c).strip()),
        branding_policy=policy, approved_brand_marks=tuple(marks),
    )
