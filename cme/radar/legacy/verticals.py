from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
from urllib.parse import urlparse

from .market_taxonomy import ALIASES, TAXONOMIES, MarketTaxonomy, norm_key, resolve_key, term_in_text


@dataclass(frozen=True)
class VerticalProfile:
    name: str
    include: tuple[str, ...]
    strong: tuple[str, ...]
    exclude: tuple[str, ...]
    topical: tuple[str, ...] = ()
    identity: tuple[str, ...] = ()
    adjacent: tuple[str, ...] = ()


def _build_profile(tax: MarketTaxonomy) -> VerticalProfile:
    return VerticalProfile(
        name=tax.name,
        include=tuple(dict.fromkeys(tax.direct_terms + tax.adjacent_terms + tax.profile_include_extra)),
        strong=tax.direct_terms,
        identity=tax.direct_terms,
        adjacent=tax.adjacent_terms,
        topical=tax.profile_topical,
        exclude=tax.hard_exclude_terms,
    )


PROFILES = {key: _build_profile(tax) for key, tax in TAXONOMIES.items()}


def canonical_vertical(vertical: str, *, target_offer: str = "", prelander: str = "") -> str:
    """Resolve operator-facing vertical names/aliases to a configured market profile name.

    ``Software`` is an ambiguous operator label: it resolves to a vertical only when the target
    offer/prelander carries that vertical's intent terms, matched as whole words (never substrings).
    Unknown verticals are returned unchanged so callers can report them as unsupported.
    """
    raw = norm_key(vertical)
    key = resolve_key(raw)
    if key is not None:
        return TAXONOMIES[key].name
    if raw in {"software", "software tools", "ai software"}:
        context = " ".join((str(target_offer or ""), str(prelander or ""))).lower()
        for tax in TAXONOMIES.values():
            if any(term_in_text(context, term) for term in tax.software_alias_intent_terms):
                return tax.name
    return vertical or ""


def _flatten(obj: Any) -> str:
    if isinstance(obj, dict):
        return " ".join(f"{k} {v}" for k, v in obj.items())
    if isinstance(obj, list):
        return " ".join(str(x) for x in obj)
    return str(obj or "")


def _text(record: dict[str, Any]) -> str:
    raw = record.get("raw") or {}
    post = raw.get("post") if isinstance(raw, dict) else {}
    post = post if isinstance(post, dict) else {}
    content = post.get("content") if isinstance(post, dict) else {}
    content = content if isinstance(content, dict) else {}
    ext = post.get("ext") if isinstance(post, dict) else {}
    ext = ext if isinstance(ext, dict) else {}
    ad = ext.get("ad") if isinstance(ext, dict) else {}
    ad = ad if isinstance(ad, dict) else {}
    parts = [
        record.get("title"), record.get("brand"), record.get("brand_name"), record.get("industry"),
        record.get("objective"), record.get("source_url"), record.get("landing_page"), record.get("landing_page_url"),
        content.get("text"), ad.get("title"), ad.get("brand_name"), ad.get("industry"), ad.get("objective"),
        ad.get("landing_page"), ad.get("landing_page_url"), ad.get("url"),
    ]
    return _flatten(parts).lower()


def _term_present(text: str, term: str) -> bool:
    text = " ".join((text or "").lower().split())
    term = " ".join((term or "").lower().split())
    if not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _hits(text: str, terms: tuple[str, ...] | list[str]) -> list[str]:
    return [t for t in terms if _term_present(text, t)]


def _domain(record: dict[str, Any]) -> str:
    for key in ("landing_page_url", "landing_page", "source_url"):
        value = str(record.get(key) or "").strip()
        if value:
            try:
                host = urlparse(value if "://" in value else "https://" + value).netloc.lower()
                if host.startswith("www."):
                    host = host[4:]
                return host
            except Exception:
                pass
    return ""


def _brand(record: dict[str, Any]) -> str:
    return " ".join(str(record.get(key) or "") for key in ("brand", "brand_name")).lower().strip()


def _brand_exact(text: str, brands: tuple[str, ...]) -> list[str]:
    return [b for b in brands if _term_present(text, b)]


def _content_text(record: dict[str, Any]) -> str:
    """Return source-content text without treating metadata/query terms as proof of entity identity."""
    raw = record.get("raw") or {}
    post = raw.get("post") if isinstance(raw, dict) else {}
    post = post if isinstance(post, dict) else {}
    content = post.get("content") if isinstance(post, dict) else {}
    content = content if isinstance(content, dict) else {}
    ext = post.get("ext") if isinstance(post, dict) else {}
    ext = ext if isinstance(ext, dict) else {}
    ad = ext.get("ad") if isinstance(ext, dict) else {}
    ad = ad if isinstance(ad, dict) else {}
    return _flatten([
        record.get("title"), content.get("text"), ad.get("title"),
    ]).lower()


def _ad_brand_text(record: dict[str, Any]) -> str:
    """Return advertiser-provided brand fields only; creator identity is intentionally excluded."""
    raw = record.get("raw") or {}
    post = raw.get("post") if isinstance(raw, dict) else {}
    post = post if isinstance(post, dict) else {}
    ext = post.get("ext") if isinstance(post, dict) else {}
    ext = ext if isinstance(ext, dict) else {}
    ad = ext.get("ad") if isinstance(ext, dict) else {}
    ad = ad if isinstance(ad, dict) else {}
    return _flatten([ad.get("brand_name"), ad.get("brand")]).lower()


def _entity_context_rule(tax: MarketTaxonomy, entity: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    target = " ".join(str(entity or "").lower().split())
    for name, positive, negative in tax.entity_context_rules:
        if " ".join(str(name or "").lower().split()) == target:
            return tuple(positive), tuple(negative)
    return (), ()


def _resolve_entities(record: dict[str, Any], profile: VerticalProfile, tax: MarketTaxonomy) -> dict[str, Any]:
    """Resolve brand/entity mentions without confusing search-query words with identity proof.

    Search criteria are intentionally absent from the decision. Query terms discover candidates;
    authoritative advertiser/domain evidence or source-context corroboration establishes identity.
    """
    content = _content_text(record)
    metadata_brand = _brand(record)
    advertiser_brand = _ad_brand_text(record)
    direct_mentions = _brand_exact(_text(record), tax.direct_brands)
    content_mentions = _brand_exact(content, tax.direct_brands)
    metadata_mentions = _brand_exact(metadata_brand, tax.direct_brands)
    advertiser_mentions = _brand_exact(advertiser_brand, tax.direct_brands)

    positive_by_entity: dict[str, list[str]] = {}
    negative_by_entity: dict[str, list[str]] = {}
    resolved: list[str] = []
    unresolved: list[str] = []
    conflicted: list[str] = []

    for entity in direct_mentions:
        positive_terms, negative_terms = _entity_context_rule(tax, entity)
        positive = _hits(content, positive_terms) if positive_terms else []
        negative = _hits(content, negative_terms) if negative_terms else []
        positive_by_entity[entity] = positive
        negative_by_entity[entity] = negative

        authoritative = entity in advertiser_mentions
        metadata_exact = entity in metadata_mentions
        ambiguous = entity in tax.context_required_brands or bool(positive_terms or negative_terms)

        if authoritative or (entity in direct_mentions and not ambiguous and (metadata_exact or entity in tax.direct_domains)):
            resolved.append(entity)
            continue
        if ambiguous:
            if negative:
                conflicted.append(entity)
                continue
            if positive:
                resolved.append(entity)
                continue
            unresolved.append(entity)
            continue
        # For an ordinary (non-ambiguous) known brand, metadata can still establish identity;
        # a content-only mention needs at least one strong source-market signal.
        if metadata_exact:
            resolved.append(entity)
        elif entity in content_mentions and (_hits(content, profile.strong) or len(_hits(content, profile.topical)) >= 2):
            resolved.append(entity)
        else:
            unresolved.append(entity)

    # Domain/URL evidence is authoritative even when text mentions are absent.
    domain = _domain(record)
    direct_domain_hits = [d for d in tax.direct_domains if domain == d or domain.endswith("." + d)]
    direct_url_hits = [frag for frag in tax.direct_url_fragments if frag in _text(record)]
    identity_source = "advertiser_brand" if advertiser_mentions else "domain/url" if (direct_domain_hits or direct_url_hits) else "contextual_content" if resolved else "unresolved_content"
    return {
        "observed_direct_brand_hits": direct_mentions,
        "content_brand_hits": content_mentions,
        "metadata_brand_hits": metadata_mentions,
        "advertiser_brand_hits": advertiser_mentions,
        "resolved_brand_hits": list(dict.fromkeys(resolved)),
        "unresolved_brand_hits": list(dict.fromkeys(unresolved)),
        "conflicted_brand_hits": list(dict.fromkeys(conflicted)),
        "positive_context_hits": positive_by_entity,
        "negative_context_hits": negative_by_entity,
        "direct_domain_hits": direct_domain_hits,
        "direct_url_hits": direct_url_hits,
        "identity_source": identity_source,
    }


def _industry_conflict(industry: str, tax: MarketTaxonomy) -> bool:
    """Native industry label conflicts with the vertical (weak prior; needs corroboration to reject)."""
    value = " ".join((industry or "").lower().split())
    return any(term_in_text(value, x) for x in tax.conflict_industries)


def source_market_evidence(record: dict[str, Any], vertical: str) -> dict[str, Any]:
    """Return source-only market evidence. Target offer/prelander are never read here."""
    key = resolve_key(vertical)
    profile = PROFILES.get(key) if key else None
    if not profile:
        return {"status": "UNKNOWN", "market_path": {}, "reason": "no_profile"}
    tax = TAXONOMIES[key]

    text = _text(record)
    domain = _domain(record)
    brand = _brand(record)
    industry = str(record.get("industry") or "").strip()
    entity_resolution = _resolve_entities(record, profile, tax)
    direct_brand_hits = list(entity_resolution["resolved_brand_hits"])
    direct_domain_hits = list(entity_resolution["direct_domain_hits"])
    direct_url_hits = list(entity_resolution["direct_url_hits"])
    direct_product_hits = _hits(text, tax.direct_terms)
    adjacent_brand_hits = _brand_exact(text, tax.adjacent_brands)
    adjacent_product_hits = _hits(text, tax.adjacent_terms)
    hard_brand_hits = _brand_exact(brand, tax.hard_exclude_brands)
    hard_domain_hits = [d for d in tax.hard_exclude_domains if domain == d or domain.endswith("." + d)]
    hard_term_hits = _hits(text, tax.hard_exclude_terms)
    industry_conflict = _industry_conflict(industry, tax)
    context_required_brand_hits = [b for b in direct_brand_hits if b in tax.context_required_brands]
    # An explicit brand/domain blocklist hit is absolute. Ambiguous conflict words ("bar", "toy",
    # "travel") or a native-industry label only reject when the source has NO corroborating
    # identity evidence for this vertical (otherwise "search bar" would reject a real Cursor ad).
    corroborated = bool(direct_brand_hits or direct_domain_hits or direct_url_hits or len(set(_hits(text, profile.strong))) >= 2)
    rejected = bool(hard_brand_hits or hard_domain_hits or ((hard_term_hits or industry_conflict) and not corroborated))

    return {
        "status": "REJECT" if rejected else "UNVERIFIED",
        "market_path": {
            "native_industry": industry or "UNKNOWN",
            "entity": direct_brand_hits,
            "entity_observed": entity_resolution["observed_direct_brand_hits"],
            "domain": direct_domain_hits,
            "url": direct_url_hits,
            "product_type": direct_product_hits[:20],
            "use_case": _hits(text, profile.topical)[:20],
            "adjacent_entity": adjacent_brand_hits,
            "adjacent_product": adjacent_product_hits[:20],
        },
        "conflict_evidence": {
            "brand": hard_brand_hits,
            "domain": hard_domain_hits,
            "terms": hard_term_hits[:20],
            "industry": industry_conflict,
        },
        "context_required_brand_hits": context_required_brand_hits,
        "entity_resolution": entity_resolution,
    }


def classify_vertical(record: dict[str, Any], vertical: str, context: str = "", mode: str = "strict", *, target_offer: str = "", prelander: str = "") -> dict[str, Any]:
    vertical = canonical_vertical(vertical, target_offer=target_offer, prelander=prelander)
    key = resolve_key(vertical)
    profile = PROFILES.get(key) if key else None
    if not profile:
        return {
            "status": "UNKNOWN", "classification": "UNVERIFIED", "score": 0.0,
            "hits": [], "strong_hits": [], "identity_hits": [], "topical_hits": [], "adjacent_hits": [],
            "excluded": [], "strict_match": False, "expanded_match": False, "unverified": True,
            "reason": "no_profile", "market_path": {},
        }

    tax = TAXONOMIES[key]
    text = _text(record)
    brand = _brand(record)
    domain = _domain(record)
    industry = str(record.get("industry") or "").lower()
    title = str(record.get("title") or "").lower()

    excluded = _hits(text, profile.exclude)
    strong_hits = _hits(text, profile.strong)
    identity_hits = _hits(text, profile.identity)
    include_hits = _hits(text, profile.include)
    topical_hits = _hits(text, profile.topical)
    adjacent_hits = _hits(text, profile.adjacent)

    hard_brand_hits = _brand_exact(brand, tax.hard_exclude_brands)
    hard_domain_hits = [d for d in tax.hard_exclude_domains if domain == d or domain.endswith("." + d)]
    industry_conflict = _industry_conflict(industry, tax)

    entity_resolution = _resolve_entities(record, profile, tax)
    observed_direct_brand_hits = list(entity_resolution["observed_direct_brand_hits"])
    direct_brand_hits = list(entity_resolution["resolved_brand_hits"])
    direct_domain_hits = list(entity_resolution["direct_domain_hits"])
    direct_url_hits = list(entity_resolution["direct_url_hits"])
    adjacent_brand_hits = _brand_exact(text, tax.adjacent_brands)
    context_required_brand_hits = [b for b in direct_brand_hits if b in tax.context_required_brands]

    title_identity = any(_term_present(title, x) for x in profile.strong)
    software_industry = any(_term_present(industry, x) for x in (
        "software", "saas", "application software", "computer software", "information technology", "cloud computing",
    ))

    # Blocklisted brand/domain is absolute; ambiguous conflict words or industry labels only reject
    # without corroborating identity evidence (see source_market_evidence).
    corroborated = bool(direct_brand_hits or direct_domain_hits or direct_url_hits or len(set(strong_hits)) >= 2)
    hard_exclude = bool(hard_brand_hits or hard_domain_hits or ((industry_conflict or excluded) and not corroborated))
    if hard_exclude:
        status, classification = "REJECT", "HARD_EXCLUDE"
    else:
        # Known but broad AI brands need product/use-case corroboration. This prevents
        # a general Qwen/OpenAI/Google ad from becoming direct only because of the brand.
        broad_brand_needs_context = bool(set(context_required_brand_hits) & set(tax.context_required_brands))
        broad_brand_supported = bool(strong_hits) or len(set(topical_hits)) >= 2
        direct_identity = bool(
            direct_domain_hits or direct_url_hits or direct_brand_hits
            or (observed_direct_brand_hits and not direct_brand_hits and broad_brand_supported)
            or (len(set(identity_hits)) >= 2)
            or (len(set(strong_hits)) >= 2)
            or (len(set(identity_hits)) >= 1 and (title_identity or len(set(topical_hits)) >= 2))
        )
        adjacent_identity = bool(
            not direct_identity and (
                adjacent_brand_hits
                or len(set(adjacent_hits)) >= 2
                or (software_industry and len(set(adjacent_hits)) >= 1 and len(set(topical_hits)) >= 2)
            )
        )
        if direct_identity:
            status, classification = ("HIGH", "DIRECT") if (direct_brand_hits or direct_domain_hits or direct_url_hits or len(set(strong_hits)) >= 2) else ("MEDIUM", "DIRECT")
        elif adjacent_identity:
            status, classification = "LOW", "EXPANDED"
        else:
            status, classification = "UNKNOWN", "UNVERIFIED"

    score = 5.0
    score += min(18.0, len(set(include_hits)) * 2.0)
    score += min(30.0, len(set(identity_hits)) * 10.0)
    score += min(24.0, len(set(strong_hits)) * 8.0)
    score += min(12.0, len(set(adjacent_hits)) * 4.0)
    score += 35.0 if direct_brand_hits or direct_domain_hits or direct_url_hits else 0.0
    score += 10.0 if adjacent_brand_hits else 0.0
    score += 4.0 if software_industry else 0.0
    score += 3.0 if len(set(topical_hits)) >= 3 else 0.0
    score -= min(100.0, 70.0 + 8.0 * len(excluded)) if hard_exclude else 0.0
    # Broad industry labels never create identity by themselves.
    if classification == "UNVERIFIED" and not (direct_brand_hits or direct_domain_hits or direct_url_hits or strong_hits or adjacent_hits):
        score = min(score, 42.0)
    score = max(0.0, min(100.0, score))

    strict_match = classification == "DIRECT"
    expanded_match = classification in {"DIRECT", "EXPANDED"}
    unverified = classification == "UNVERIFIED"
    market = source_market_evidence(record, vertical)
    reason = (
        f"brand={brand} domain={domain} industry={industry} observed_direct_brands={observed_direct_brand_hits} direct_brands={direct_brand_hits} direct_domains={direct_domain_hits} "
        f"identity={identity_hits[:10]} strong={strong_hits[:10]} topical={topical_hits[:10]} adjacent={adjacent_hits[:10]} "
        f"hard_brand={hard_brand_hits} hard_domain={hard_domain_hits} excluded={excluded[:10]} industry_conflict={industry_conflict} "
        f"software_industry={software_industry} context_required_brand_hits={context_required_brand_hits} classification={classification} mode={mode}. "
        "Target context intentionally excluded from source classification."
    )
    return {
        "status": status,
        "classification": classification,
        "score": round(score, 1),
        "hits": include_hits[:20],
        "strong_hits": strong_hits[:20],
        "identity_hits": identity_hits[:20],
        "adjacent_hits": adjacent_hits[:20],
        "topical_hits": topical_hits[:20],
        "excluded": excluded[:20],
        "hard_exclude_brand_hits": hard_brand_hits,
        "hard_exclude_domain_hits": hard_domain_hits,
        "direct_brand_hits": direct_brand_hits,
        "observed_direct_brand_hits": observed_direct_brand_hits,
        "direct_domain_hits": direct_domain_hits,
        "direct_url_hits": direct_url_hits,
        "adjacent_brand_hits": adjacent_brand_hits,
        "context_required_brand_hits": context_required_brand_hits,
        "industry_conflict": industry_conflict,
        "strict_match": strict_match,
        "expanded_match": expanded_match,
        "unverified": unverified,
        "known_brand": bool(direct_brand_hits),
        "entity_resolution": entity_resolution,
        "software_industry": software_industry,
        "market_path": market.get("market_path", {}),
        "conflict_evidence": market.get("conflict_evidence", {}),
        "reason": reason,
    }


def discovery_seed_groups(vertical: str) -> list[list[str]]:
    key = resolve_key(vertical)
    taxonomy = TAXONOMIES.get(key) if key else None
    return [list(group) for group in taxonomy.seed_groups] if taxonomy else []
