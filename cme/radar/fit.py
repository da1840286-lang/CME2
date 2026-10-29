"""Source-body fit for FIND. Search keywords are never evidence about the ad itself."""
from __future__ import annotations

import re

TARGET_PRODUCT = "TARGET_PRODUCT"
ADJACENT_MECHANISM = "ADJACENT_MECHANISM"
OFF_TOPIC = "OFF_TOPIC"

_MODIFIERS = {
    "a", "an", "and", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    "cordless", "handheld", "portable", "wireless", "rechargeable", "electric", "mini",
    "compact", "lightweight", "powerful", "smart", "new", "best", "top", "small", "large",
    "your", "our", "his", "her", "their",
}
_CONTEXT_ALIASES = {
    "car": {"car", "cars", "auto", "automotive", "vehicle", "vehicles", "dashboard", "trunk", "road"},
    "home": {"home", "household", "house", "indoor", "livingroom", "bedroom"},
    "kitchen": {"kitchen", "cooking", "cookware", "foodprep"},
    "office": {"office", "desk", "workstation"},
    "travel": {"travel", "traveling", "travelling", "roadtrip", "camping"},
}
# Match the product head, not the last word of a phrase ("tire inflator for cars" -> inflator).
_PRODUCT_HEAD_PATTERNS = (
    (re.compile(r"\b(?:vacuum\s+cleaner|car\s+vacuum|handheld\s+vacuum|vacuum)\b", re.I), "vacuum"),
    (re.compile(r"\b(?:tyre|tire)\s+inflator\b", re.I), "inflator"),
    (re.compile(r"\b(?:portable\s+)?air\s+compressor\b", re.I), "compressor"),
    (re.compile(r"\b(?:pressure|power)\s+washer\b", re.I), "washer"),
    (re.compile(r"\bnoise[ -]?cancel(?:ling|ing)?\s+headphones?\b", re.I), "headphones"),
    (re.compile(r"\bwireless\s+earbuds?\b", re.I), "earbuds"),
    (re.compile(r"\bair\s+duster\b", re.I), "duster"),
    (re.compile(r"\bpower\s+bank\b", re.I), "bank"),
    (re.compile(r"\bphone\s+case\b", re.I), "case"),
    (re.compile(r"\bphone\s+stand\b", re.I), "stand"),
    (re.compile(r"\bdash\s*cam\b", re.I), "cam"),
)
_PRODUCT_NOUNS = (
    "air duster", "vacuum", "blower", "air compressor", "drill", "screwdriver", "power bank",
    "charger", "phone case", "phone stand", "headphones", "earbuds", "speaker", "camera",
    "dash cam", "dashcam", "flashlight", "lamp", "light", "blender", "coffee maker", "air fryer",
    "water bottle", "organizer", "massager", "cleaner", "brush", "shaver", "trimmer", "mop",
    "pressure washer", "sealer", "label maker", "projector", "tripod", "fan", "humidifier",
    "purifier", "tool", "pet hair remover", "lint remover", "car seat", "seat cover", "tire inflator",
    "tyre inflator", "air compressor",
)
_PROMOTION = re.compile(
    r"\b(?:buy(?:\s+now)?|shop\s+now|order(?:\s+today|\s+now)?|get\s+yours|grab\s+(?:this|yours)|"
    r"available\s+now|add\s+to\s+cart|free\s+shipping|blows?|removes?|cleans?|sucks?|picks?\s+up|"
    r"scrubs?|cuts?|charges?|stores?|organizes?|reduces?|filters?|washes?|polishes?|dries?|"
    r"inflates?|deflates?|works?\s+(?:great|well)|easy\s+to\s+carry|in\s+seconds|one[-\s]handed|"
    r"perfect\s+for|made\s+for|designed\s+for|try\s+it|game\s*changer|must[-\s]?have|"
    r"discount|sale|deal|introducing|meet\s+the)\b", re.IGNORECASE,
)
_CTA = re.compile(r"\b(?:buy(?:\s+now)?|shop\s+now|order(?:\s+today|\s+now)?|get\s+yours|add\s+to\s+cart|grab\s+this)\b", re.IGNORECASE)
_TARGET_ACTION = re.compile(
    r"\b(?:removes?|sucks?|cleans?|picks?\s+up|delivers?|provides?|features?|offers?|keeps?|"
    r"has|works?|blows?|charges?|inflates?|filters?|washes?|scrubs?|cuts?|is\s+designed|is\s+made)\b",
    re.IGNORECASE,
)
# Rows returned by Sponsored Search / Ads Top are paid-ad inventory by construction. That is a
# property of the post (not of the search query), so it may corroborate promotion when the post's
# own text names the target product noun. It never replaces the noun-in-body requirement.
_PAID_INVENTORY_SOURCES = frozenset({"tiktok_sponsored_search", "sponsored_search", "ads_top"})
_SPEC = re.compile(r"\b\d+(?:\.\d+)?\s?(?:psi|v|w|kpa|pa|mah|mm|cm|oz|ml|l|rpm|lm|gb)\b", re.IGNORECASE)
_PROMO_TAGS = {"tiktokshop", "tiktokmademebuyit", "amazonfinds", "amazonfind", "ad", "sponsored", "shopnow"}


def _record_source_view(record: dict) -> dict:
    """Read only source-body fields; do not treat search queries as evidence."""
    record = record if isinstance(record, dict) else {}
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    post = raw.get("post") if isinstance(raw.get("post"), dict) else {}
    content = post.get("content") if isinstance(post.get("content"), dict) else {}
    ext = post.get("ext") if isinstance(post.get("ext"), dict) else {}
    ad = ext.get("ad") if isinstance(ext.get("ad"), dict) else {}
    values = [record.get("title"), content.get("text"), ad.get("title")]
    pieces = [" ".join(str(value or "").split()) for value in values if str(value or "").strip()]
    text = " ".join(dict.fromkeys(pieces))
    hashtags = []
    raw_hashtags = ext.get("hashtags") or []
    if isinstance(raw_hashtags, (list, tuple, set)):
        for item in raw_hashtags:
            if isinstance(item, dict):
                item = item.get("name") or item.get("title") or item.get("hashtag") or ""
            tag = str(item or "").strip().lstrip("#").casefold().replace("_", "")
            if tag:
                hashtags.append(tag)
    hashtags.extend(tag.replace("_", "") for tag in re.findall(r"#([a-z0-9_]+)", text.casefold()))
    body = re.sub(r"#[a-z0-9_]+", " ", text.casefold())
    tokens = re.findall(r"[a-z0-9]+", body)
    return {"text": text, "body": body, "body_tokens": tokens, "hashtags": list(dict.fromkeys(hashtags))}


def _record_source_text(record: dict) -> str:
    return _record_source_view(record)["text"]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(text or "").casefold()) if len(token) > 1 and token not in _MODIFIERS}


def _product_anchor(text: str) -> str:
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))
    for pattern, anchor in _PRODUCT_HEAD_PATTERNS:
        if pattern.search(normalized):
            return anchor
    tokens = [token for token in re.findall(r"[a-z0-9]+", normalized)
              if len(token) > 1 and token not in _MODIFIERS
              and not any(token in aliases for aliases in _CONTEXT_ALIASES.values())]
    return tokens[-1] if tokens else ""


def _target_identity(product_type: str, product_category: str = "", target_offer: str = "") -> tuple[set[str], str]:
    """Resolve the advertised product noun; usage context such as 'cars' is not a product head."""
    choices = [str(value or "") for value in (product_type, product_category, target_offer) if str(value or "").strip()]
    for text in choices:
        anchor = _product_anchor(text)
        if anchor:
            meaningful = _tokens(text)
            return meaningful, anchor
    return set(), ""


def _target_contexts(product_type: str, product_category: str = "", target_offer: str = "") -> set[str]:
    text = " ".join(str(value or "") for value in (product_type, product_category, target_offer)).casefold()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    return {context for context, aliases in _CONTEXT_ALIASES.items() if tokens & aliases}


def _context_verified(expected: set[str], view: dict) -> bool:
    if not expected:
        return True
    tokens = set(view["body_tokens"])
    tags = set(view["hashtags"])
    for context in expected:
        aliases = _CONTEXT_ALIASES[context]
        if tokens & aliases:
            return True
        if any(any(alias.replace(" ", "") in tag for alias in aliases) for tag in tags):
            return True
    return False


def _anchor_present(anchor: str, view: dict) -> str:
    """Return body/hashtag evidence for a product noun; verb-only #vacuuming does not count."""
    if not anchor or len(anchor) < 3:
        return ""
    forms = {anchor, anchor + "s", anchor + "es"}
    if any(token in forms for token in view["body_tokens"]):
        return "body"
    suffixes = {"cleaner", "cleaners", "machine", "machines", "tool", "tools", "review", "reviews"}
    for tag in view["hashtags"]:
        compact = re.sub(r"[^a-z0-9]", "", str(tag).casefold())
        if compact in forms:
            return "hashtag"
        if anchor == "vacuum" and compact.startswith("vacuum") and compact[len("vacuum"):] in suffixes:
            return "hashtag"
        # Compounds such as #carvacuum and #tireinflator are okay; gerunds are not.
        if compact.endswith(anchor) and compact[:-len(anchor)] and not compact.endswith("ing"):
            return "hashtag"
    return ""


def _promo_signals(record: dict, view: dict) -> list[str]:
    found = []
    if _PROMOTION.search(view["text"]):
        found.append("promo_wording")
    if _SPEC.search(view["text"]):
        found.append("product_spec")
    if any(tag in _PROMO_TAGS for tag in view["hashtags"]):
        found.append("commerce_hashtag")
    if str(record.get("landing_page_url") or record.get("landing_page") or "").strip():
        found.append("landing_page")
    return found


def _target_is_promoted(body: str, anchor: str) -> bool:
    """Require target-noun-linked action/CTA, not a passing mention in unrelated source copy."""
    if not anchor:
        return False
    for sentence in re.split(r"[\n.!?;]+", body):
        match = re.search(r"\b" + re.escape(anchor) + r"\b", sentence)
        if not match:
            continue
        tail = sentence[match.end():match.end() + 120]
        if _TARGET_ACTION.search(tail):
            return True
        cta = _CTA.search(sentence)
        if cta and abs(cta.start() - match.start()) <= 55:
            return True
        promo = _PROMOTION.search(tail)
        if promo and promo.start() <= 90:
            return True
    return False



def _paid_inventory_corroborates(record: dict, view: dict, anchor: str, contexts: set, mention_source: str) -> bool:
    """Paid-ad inventory + target noun in the post's own text + an independent corroborator.

    Corroborators: target use context in the text (e.g. "car" for a car vacuum) or the noun also in the
    post's hashtags. Passing mentions ("vacuum sealed", "before vacuuming") have neither and stay out.
    """
    if not anchor or not mention_source:
        return False
    if str((record or {}).get("discovery_source") or "").strip().lower() not in _PAID_INVENTORY_SOURCES:
        return False
    # Verb use ("then vacuum it away", "to vacuum") names an action, not the advertised product.
    verb_use = re.compile(
        r"\b(?:then|to|will|can|gonna|would|should)\s+" + re.escape(anchor) + r"\b|"
        r"\b" + re.escape(anchor) + r"(?:ed)?\s+(?:it|them|up|away|out)\b", re.IGNORECASE)
    if verb_use.search(view.get("body") or ""):
        return False
    if contexts and _context_verified(contexts, view):
        return True
    return bool(_anchor_present(anchor, {"body_tokens": [], "hashtags": view["hashtags"]}))


def match_source_to_product_query(record: dict, query: str) -> dict:
    """Require the product returned by Sponsored Search to match its own query.

    Query terms define the expected product; they never count as evidence that
    the source itself promotes that product.
    """
    view = _record_source_view(record)
    anchor = _product_anchor(query)
    if not anchor:
        return {
            "matches": False, "query_anchor": None,
            "reason": "QUERY_PRODUCT_UNVERIFIED",
        }
    mention_source = _anchor_present(anchor, view)
    if not mention_source:
        return {
            "matches": False, "query_anchor": anchor,
            "reason": "QUERY_PRODUCT_MISMATCH",
        }
    linked = _target_is_promoted(view["body"], anchor)
    if mention_source == "hashtag" and _promo_signals(record, view):
        linked = True
    if not linked and _paid_inventory_corroborates(record, view, anchor, _target_contexts(query), mention_source):
        linked = True
    if not linked:
        return {
            "matches": False, "query_anchor": anchor,
            "mention_source": mention_source,
            "reason": "QUERY_PRODUCT_NOT_PROMOTED",
        }
    return {
        "matches": True, "query_anchor": anchor,
        "mention_source": mention_source, "reason": "",
    }



def classify_source_role(record: dict, *, product_type: str = "", product_category: str = "", target_offer: str = "") -> dict:
    """Classify the promoted product from source content; no query metadata, mechanism or Target-Fit inference."""
    view = _record_source_view(record)
    source_text = view["text"]
    body = view["body"]
    body_tokens = set(view["body_tokens"])
    target_terms, anchor = _target_identity(product_type, product_category, target_offer)
    target_source = _anchor_present(anchor, view)
    promotion_match = _PROMOTION.search(source_text)
    promo = _promo_signals(record, view)
    detected = [noun for noun in _PRODUCT_NOUNS if re.search(r"\b" + re.escape(noun) + r"\b", body)]
    target_promoted = _target_is_promoted(body, anchor)
    # A hashtag can identify the target product when the source body independently promotes/demonstrates it.
    if target_source == "hashtag" and promo:
        target_promoted = True
    promoted = bool(promotion_match or promo)
    contexts = _target_contexts(product_type, product_category, target_offer)
    context_verified = _context_verified(contexts, view)

    # Paid-inventory corroboration: a product-naming title such as "Handheld Cordless Car Vacuum
    # cleaner!" carries no verb or CTA, yet is exactly what a seller's ad looks like.
    paid_target_ok = False
    if target_source and not (target_promoted and promoted) and _paid_inventory_corroborates(
            record, view, anchor, contexts, target_source):
        paid_target_ok = True
        target_promoted = True
        promo = list(promo) + ["paid_ad_inventory"]
    promoted_for_target = promoted or paid_target_ok

    if target_source and target_promoted and promoted_for_target:
        role = TARGET_PRODUCT
        confidence = "MEDIUM" if context_verified else "LOW"
        reason = "Target product noun is tied to source-body promotion/action." + (
            " Target use context is present." if context_verified else " Target use context is not stated; family-level match only."
        )
    elif detected and promoted:
        role = ADJACENT_MECHANISM
        confidence = "MEDIUM" if len(promo) > 1 else "LOW"
        reason = "A different identifiable physical product is promoted; causal mechanism and transferability remain untested."
    else:
        role, confidence = OFF_TOPIC, "NONE"
        if not source_text:
            reason = "No source-body title/caption; search metadata is not evidence."
        elif target_source and not target_promoted:
            reason = "Target noun appears in context but is not independently shown as the promoted product."
        elif detected and not promoted:
            reason = "Product noun appears without independent source-body promotion or demonstration evidence."
        else:
            reason = "Source body does not establish a promoted target or adjacent product."

    return {
        "role": role, "reason": reason, "confidence": confidence, "context_verified": context_verified,
        "source_body_present": bool(source_text), "source_body_evidence": source_text[:500],
        "target_anchor": anchor or None, "target_mention_source": target_source or None,
        "matched_target_terms": sorted(term for term in target_terms if term in body_tokens),
        "detected_product_nouns": detected, "promotion_evidence": promoted_for_target,
        "target_promotion_evidence": target_promoted, "promo_signals": promo,
        "promotion_signal": promotion_match.group(0) if promotion_match else None,
    }
