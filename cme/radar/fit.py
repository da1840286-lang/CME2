"""Source-body fit for FIND. Search keywords are never evidence about the ad itself."""
from __future__ import annotations

import re

TARGET_PRODUCT = "TARGET_PRODUCT"
ADJACENT_MECHANISM = "ADJACENT_MECHANISM"
OFF_TOPIC = "OFF_TOPIC"

_STOP = {
    "a", "an", "and", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    "cordless", "handheld", "portable", "wireless", "rechargeable", "electric", "mini",
    "compact", "lightweight", "powerful", "smart", "new", "best", "top", "small", "large",
    "car", "auto", "automotive", "home", "kitchen", "office", "outdoor", "your", "our",
}
_PRODUCT_NOUNS = (
    "air duster", "vacuum", "blower", "air compressor", "drill", "screwdriver", "power bank",
    "charger", "phone case", "phone stand", "headphones", "earbuds", "speaker", "camera",
    "dash cam", "dashcam", "flashlight", "lamp", "light", "blender", "coffee maker", "air fryer",
    "water bottle", "organizer", "massager", "cleaner", "brush", "shaver", "trimmer", "mop",
    "pressure washer", "sealer", "label maker", "projector", "tripod", "fan", "humidifier",
    "purifier", "tool", "pet hair remover", "lint remover", "car seat", "seat cover",
)
_PROMOTION = re.compile(
    r"\b(?:buy(?:\s+now)?|shop\s+now|order(?:\s+today|\s+now)?|get\s+yours|grab\s+(?:this|yours)|"
    r"available\s+now|add\s+to\s+cart|free\s+shipping|blows?|removes?|cleans?|sucks?|picks?\s+up|"
    r"scrubs?|cuts?|charges?|stores?|organizes?|reduces?|filters?|washes?|polishes?|dries?|"
    r"inflates?|deflates?|works?\s+(?:great|well)|easy\s+to\s+carry|in\s+seconds|one[-\s]handed|"
    r"perfect\s+for|made\s+for|designed\s+for|try\s+it)\b", re.IGNORECASE,
)
_CTA = re.compile(r"\b(?:buy(?:\s+now)?|shop\s+now|order(?:\s+today|\s+now)?|get\s+yours|add\s+to\s+cart|grab\s+this)\b", re.IGNORECASE)
_TARGET_ACTION = re.compile(
    r"\b(?:removes?|sucks?|cleans?|picks?\s+up|delivers?|provides?|features?|offers?|keeps?|"
    r"has|works?|is\s+designed|is\s+made)\b", re.IGNORECASE,
)


def _record_source_text(record: dict) -> str:
    """Read only title and source caption. Deliberately excludes all query/discovery metadata."""
    record = record if isinstance(record, dict) else {}
    raw = record.get("raw")
    post = raw.get("post") if isinstance(raw, dict) else {}
    post = post if isinstance(post, dict) else {}
    content = post.get("content") if isinstance(post.get("content"), dict) else {}
    ext = post.get("ext") if isinstance(post.get("ext"), dict) else {}
    ad = ext.get("ad") if isinstance(ext.get("ad"), dict) else {}
    values = [record.get("title"), content.get("text"), ad.get("title")]
    return " ".join(dict.fromkeys(" ".join(str(value or "").split()) for value in values if str(value or "").strip()))


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(text or "").casefold()) if len(token) > 1 and token not in _STOP}


def _target_identity(product_type: str, product_category: str = "", target_offer: str = "") -> tuple[set[str], str]:
    text = product_type or product_category or target_offer or ""
    tokens = re.findall(r"[a-z0-9]+", str(text).casefold())
    meaningful = [token for token in tokens if len(token) > 1 and token not in _STOP]
    return set(meaningful), (meaningful[-1] if meaningful else "")


def _target_is_promoted(body: str, anchor: str) -> bool:
    """Require product-specific action after the target noun or a nearby direct CTA."""
    if not anchor:
        return False
    for sentence in re.split(r"[\n.!?;]+", body):
        match = re.search(r"\b" + re.escape(anchor) + r"\b", sentence)
        if not match:
            continue
        tail = sentence[match.end():match.end() + 100]
        if _TARGET_ACTION.search(tail):
            return True
        cta = _CTA.search(sentence)
        if cta and abs(cta.start() - match.start()) <= 35:
            return True
    return False


def classify_source_role(record: dict, *, product_type: str = "", product_category: str = "", target_offer: str = "") -> dict:
    """Classify advertised-product role, not causal-mechanism proof or downstream Target-Fit."""
    source_text = _record_source_text(record)
    body = re.sub(r"#[a-z0-9_]+", " ", source_text.casefold())
    body_tokens = _tokens(body)
    target_terms, anchor = _target_identity(product_type, product_category, target_offer)
    promotion_match = _PROMOTION.search(body)
    detected = [noun for noun in _PRODUCT_NOUNS if re.search(r"\b" + re.escape(noun) + r"\b", body)]
    target_match = bool(anchor and re.search(r"\b" + re.escape(anchor) + r"\b", body))
    target_promoted = _target_is_promoted(body, anchor)
    promoted = bool(promotion_match)
    if target_match and target_promoted and promoted:
        role = TARGET_PRODUCT
        reason = "Target noun is tied to product-specific action/CTA in the source body."
    elif detected and promoted:
        role = ADJACENT_MECHANISM
        reason = "A different identifiable physical product is promoted; causal mechanism and transferability still require analysis."
    else:
        role = OFF_TOPIC
        if not source_text:
            reason = "No source-body title/caption; search metadata is not evidence."
        elif detected and not promoted:
            reason = "Product noun appears without independent source-body promotion or demonstration evidence."
        elif target_match and not target_promoted:
            reason = "Target noun appears in context but is not independently shown as the promoted product."
        else:
            reason = "Source body does not establish a promoted target or adjacent product."
    return {
        "role": role, "reason": reason, "source_body_present": bool(source_text),
        "source_body_evidence": source_text[:500], "target_anchor": anchor or None,
        "matched_target_terms": sorted(term for term in target_terms if term in body_tokens),
        "detected_product_nouns": detected, "promotion_evidence": promoted,
        "target_promotion_evidence": target_promoted,
        "promotion_signal": promotion_match.group(0) if promotion_match else None,
    }
