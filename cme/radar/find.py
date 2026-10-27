"""FIND adapter. Runs the old engine's real discovery code and maps its output to a compact record.

Nothing here ranks or filters on its own: ranking, classification and source relevance come from the
vendored legacy modules. Unknown values stay None, never zero.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
from pathlib import Path

from .evidence import classify_commercial_evidence
from .attention import attention_absolute_v1, attention_rank, MECHANISM_ATTENTION_FLOOR, MECHANISM_MIN_COVERAGE
from .fit import ADJACENT_MECHANISM, classify_source_role
from .seen import SeenLedger, build_scope

DATA_DIR = Path(__file__).resolve().parent / "data"
# Search-only seeds for the live Sponsored Search lane. Keep separate from the
# legacy CTA/tag corpus, which is retained for byte-identical compatibility.
PRODUCT_SEARCH_TAGS_DIR = DATA_DIR / "product_search_tags"
os.environ.setdefault("CME_VERTICALS_DIR", str(DATA_DIR / "verticals"))  # before the legacy import

VERTICAL = "Physical Products"
METRIC_FIELDS = ("views", "likes", "comments", "shares", "ctr", "cvr", "play_2s_rate", "play_6s_rate",
                 "rank", "cost_score", "like_tier", "estimated_reach", "engagement_rate")
PERIODS = (7, 30, 180)

LEGACY_ERROR = ""
try:
    from .legacy.discover import canonical_source_identity_keys, discover_and_rank
    from .legacy.discovery.tiktok_socialcrawl import SocialCrawlError, SocialCrawlTikTokTopAds
    from .legacy.verticals import classify_vertical
except Exception as e:  # e.g. the 'requests' package is missing
    LEGACY_ERROR = f"{type(e).__name__}: {e}"
    discover_and_rank = canonical_source_identity_keys = SocialCrawlTikTokTopAds = classify_vertical = None

    class SocialCrawlError(Exception):
        pass


class FindUnavailable(Exception):
    def __init__(self, state: str, detail: str):
        self.state, self.detail = state, detail
        super().__init__(f"{state}: {detail}")


def find_status() -> dict:
    if LEGACY_ERROR:
        return {"state": "UNAVAILABLE", "detail": LEGACY_ERROR, "how_to_fix": "pip install -r requirements.txt"}
    if not os.getenv("SOCIALCRAWL_API_KEY"):
        return {"state": "NOT_CONFIGURED", "detail": "SOCIALCRAWL_API_KEY is not set",
                "how_to_fix": "export SOCIALCRAWL_API_KEY=... and restart Creative Radar (or import a saved discovery file)"}
    return {"state": "READY", "detail": "", "how_to_fix": ""}


def _d(decision, name, default=None):
    if decision is None:
        return default
    return decision.get(name, default) if isinstance(decision, dict) else getattr(decision, name, default)


def stable_id(c: dict) -> str:
    if c.get("ad_id"):
        return f"tiktok:ad:{c['ad_id']}"
    if c.get("video_id"):
        return f"tiktok:video:{c['video_id']}"
    basis = "|".join(sorted(c.get("keys") or [])) or json.dumps(c.get("source_url") or c.get("title") or "", sort_keys=True)
    return "tiktok:src:" + hashlib.sha256(basis.encode()).hexdigest()[:16]


def to_candidate(item: dict, request: dict, collected_at: str, origin: str, source_label: str = "") -> dict:
    """item = {'record': old record, 'decision': ArbitrageDecision or its saved dict}."""
    rec = item["record"]
    decision = item.get("decision")
    source_kind = str(rec.get("discovery_source") or "")
    default_source = "socialcrawl:tiktok_sponsored_search" if source_kind == "tiktok_sponsored_search" else "socialcrawl:tiktok_ads_top"
    src = source_label or default_source
    boards = list(rec.get("discovery_boards") or [])
    metrics = {}
    for name in METRIC_FIELDS:
        value = rec.get(name)
        if value is not None and value != "":
            metrics[name] = {"value": value, "source": src, "boards": boards, "collected_at": collected_at}
    rel = rec.get("vertical_relevance") or {}
    brand = rec.get("brand")
    video = rec.get("video_url_540p") or rec.get("video_url") or rec.get("video_url_360p") or ""
    score = None
    if decision is not None:
        score = {k: _d(decision, k) for k in ("final_score", "performance_score", "commercial_score", "funnel_fit_score",
                                              "evidence_score", "longevity_score", "duration_fit_score", "action",
                                              "mechanism_source_score", "source_value_type")}
        score["penalties"] = _d(decision, "penalties")
        score["tags"] = _d(decision, "tags")
        score["components"] = _d(decision, "components") or {}
    why = {"explanation": _d(decision, "explanation"), "action": _d(decision, "action"), "tags": _d(decision, "tags"),
           "scope": rec.get("discovery_scope"), "boards": boards, "keywords": rec.get("discovery_keywords") or [],
           "hits": rec.get("discovery_hits"), "relevance_reason": rel.get("reason")}
    c = {
        "keys": sorted(canonical_source_identity_keys(rec)) if canonical_source_identity_keys else [],
        "origin": origin, "platform": "tiktok",
        "country": rec.get("discovery_country") or request.get("country"),
        "period_days": rec.get("discovery_period") or request.get("period"),
        "ad_id": rec.get("ad_id") or None, "video_id": rec.get("video_id") or None,
        "source_url": rec.get("source_url") or None, "video_url": video or None, "cover_url": rec.get("cover_url") or None,
        "brand": None if brand in (None, "", "UNKNOWN") else brand, "brand_name": rec.get("brand_name") or None,
        "title": rec.get("title") or None, "industry": rec.get("industry") or None,
        "landing_page": rec.get("landing_page") or None, "landing_page_url": rec.get("landing_page_url") or None,
        "duration_seconds": rec.get("duration_seconds"), "duration_source": rec.get("duration_source"),
        "metrics": metrics,
        "source_check": {"status": rel.get("status") or "UNKNOWN", "classification": rel.get("classification"),
                         "score": rel.get("score"), "excluded": rel.get("excluded"), "reason": rel.get("reason"),
                         "product_relevance": rec.get("product_relevance") or {},
                         "scope": rec.get("discovery_scope")},
        "source_score": score,
        "product_relevance": rec.get("product_relevance") or {},
        "performance_evidence": rec.get("performance_evidence") or {},
        "commercial_evidence": classify_commercial_evidence(
            rec, score, rec.get("performance_evidence") or {}
        ),
        "shortlist_eligibility": rec.get("shortlist_eligibility") or {},
        "why": {**why, "product_relevance": rec.get("product_relevance") or {},
                "performance_evidence": rec.get("performance_evidence") or {},
                "selection_reason": rec.get("auto_selection_reason") or ""},
        "discovered_at": collected_at,
        "request": {k: request.get(k) for k in ("country", "period", "order_bys", "keywords", "vertical", "selection_mode")},
        "legacy_record": rec,
    }
    role = rec.get("source_role_info")
    if not isinstance(role, dict):
        role = classify_source_role(
            rec, product_type=request.get("product_type", ""),
            product_category=request.get("product_category", ""),
            target_offer=request.get("target_offer", ""),
        )
    attention = rec.get("attention_score")
    if not isinstance(attention, dict):
        attention = attention_absolute_v1(rec, period_days=c.get("period_days") or request.get("period") or 30)
    hard_excluded = bool(rel.get("excluded")) or str(rel.get("classification") or "").upper() == "HARD_EXCLUDE"
    source_gate_eligible = bool(c.get("video_url")) and not hard_excluded
    c["stage"] = "PATTERN_CANDIDATE" if source_gate_eligible else "SOURCE_REJECTED"
    c["source_gate_eligible"] = source_gate_eligible
    # Deprecated compatibility field: source-gate eligibility only, never Target-Fit approval.
    c["transfer_eligible"] = source_gate_eligible
    c["target_fit_status"] = "NOT_EVALUATED"
    c["source_selection"] = {
        "mode": request.get("selection_mode") or "product",
        "role": role.get("role") or "OFF_TOPIC",
        "role_reason": role.get("reason") or "",
        "source_body_evidence": role.get("source_body_evidence") or "",
        "target_anchor": role.get("target_anchor"),
        "target_mention_source": role.get("target_mention_source"),
        "matched_target_terms": role.get("matched_target_terms") or [],
        "detected_product_nouns": role.get("detected_product_nouns") or [],
        "promotion_evidence": bool(role.get("promotion_evidence")),
        "target_promotion_evidence": bool(role.get("target_promotion_evidence")),
        "promo_signals": role.get("promo_signals") or [],
        "context_verified": role.get("context_verified"),
        "confidence": role.get("confidence") or "UNKNOWN",
        "attention": attention,
        "attention_is_commercial_proof": False,
        "decision": rec.get("selection_decision") or "RANKED_BY_LEGACY_DISCOVERY",
        "reason": rec.get("selection_reason") or "",
    }
    c["id"] = stable_id(c)
    return c


def _rotate_query_terms(terms: list[str], prior_queries: list[str]) -> list[str]:
    """Prefer unseen queries, then least-recently-used queries; preserve caller order for ties."""
    history_position = {str(term).casefold(): index for index, term in enumerate(prior_queries or []) if str(term).strip()}
    indexed = list(enumerate(terms))
    indexed.sort(key=lambda pair: (
        0 if str(pair[1]).casefold() not in history_position else 1,
        history_position.get(str(pair[1]).casefold(), -1),
        pair[0],
    ))
    return [term for _, term in indexed]


def _clean_terms(value) -> list[str]:
    """Normalize comma-separated or list-form source-query inputs without inventing terms."""
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    out, seen = [], set()
    for raw in values:
        term = " ".join(str(raw or "").strip().split())
        key = term.casefold()
        if not term or key == "auto" or key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def _offer_context(params: dict) -> dict[str, str]:
    """Read only the product-identifying fields used by FIND from an optional OfferCard."""
    raw_offer = params.get("offer")
    offer = raw_offer if isinstance(raw_offer, dict) else {}
    fields = {
        "target_offer": params.get("target_offer") or params.get("offer_name") or offer.get("name"),
        "product_type": params.get("product_type") or offer.get("product_type"),
        "product_category": params.get("product_category") or offer.get("product_category"),
        "prelander": params.get("prelander") or "",
        "landing_page": params.get("landing_page") or "",
    }
    return {key: " ".join(str(value or "").strip().split()) for key, value in fields.items()}


_PRODUCT_QUERY_MODIFIERS = {
    "portable", "cordless", "handheld", "rechargeable", "electric", "wireless", "mini", "compact",
    "foldable", "smart", "new", "best", "top", "lightweight", "small", "large", "automatic",
    "for", "with", "without", "the", "a", "an", "and", "or", "of", "in", "to", "by", "on",
    "my", "your", "our", "his", "her", "their", "device", "product", "products", "gadget", "gadgets",
    "tool", "tools", "accessory", "accessories", "cleaning", "shop", "sale", "buy", "ads", "ad",
}
_PRODUCT_CONTEXT_TERMS = {"car", "auto", "automotive", "home", "kitchen", "office", "outdoor"}


def _product_terms(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", str(value or "").casefold())
            if len(word) > 1 and word not in _PRODUCT_QUERY_MODIFIERS}


def _product_seed_vocabulary() -> set[str]:
    """Use the configured product-query corpus for high-confidence category mismatch checks."""
    path = PRODUCT_SEARCH_TAGS_DIR / "physical_products.txt"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return set().union(*(_product_terms(line) for line in lines if line.strip() and not line.lstrip().startswith("#")))


def _query_offer_conflict(keywords: list[str], offer: dict[str, str]) -> dict | None:
    """Catch clear product-query conflicts; ignore generic, ambiguous, and mixed compatible queries."""
    identity = offer.get("product_type") or offer.get("product_category")
    target = _product_terms(identity)
    vocabulary = _product_seed_vocabulary()
    target_specific = (target & vocabulary) - _PRODUCT_CONTEXT_TERMS
    if not target_specific:
        return None

    checked = []
    for keyword in keywords:
        terms = _product_terms(keyword)
        anchors = (terms & vocabulary) - _PRODUCT_CONTEXT_TERMS
        # A generic query (e.g. "kitchen gadget") is not specific enough to reject.
        if not anchors:
            continue
        overlap = anchors & target_specific
        union = anchors | target_specific
        similarity = len(overlap) / len(union) if union else 0.0
        checked.append({"query": keyword, "anchors": sorted(anchors), "similarity": round(similarity, 3)})

    if not checked or any(row["similarity"] >= 0.35 for row in checked):
        return None
    query_list = ", ".join(repr(row["query"]) for row in checked)
    return {
        "status": "INPUT_CONFLICT",
        "query_anchors": sorted({term for row in checked for term in row["anchors"]}),
        "offer_anchors": sorted(target_specific),
        "queries": checked,
        "reason": (f"The explicit product query ({query_list}) appears to target a different product category "
                   f"than offer product_type/product_category={identity!r}. Correct the query or offer before FIND; "
                   "no SocialCrawl requests were sent."),
    }


def _find_diagnostics(stats: dict, candidate_count: int) -> dict:
    """Distinguish absent inventory, unverified relevance, and no qualified auto source."""
    calls = stats.get("calls") or []
    strategy = stats.get("discovery_strategy") or {}
    raw_returned = sum(int(call.get("returned") or 0) for call in calls if isinstance(call, dict))
    direct = int(stats.get("direct_relevant_ads") or 0)
    expanded = int(stats.get("expanded_relevant_ads") or 0)
    target_relevant = int(stats.get("target_relevant_ads") or 0)
    money_ready = int(stats.get("money_ready_source_ads") or 0)
    mechanism_ready = int(stats.get("mechanism_ready_source_ads") or 0)
    analysis_ready = int(stats.get("analysis_ready_source_ads") or 0)
    hard_excluded = int(stats.get("pre_visual_verification_hard_excluded_ads") or 0)
    selection_details = stats.get("source_selection_diagnostics") or {}
    performance_ready = int(stats.get("performance_ready_source_ads") or 0)
    if candidate_count:
        outcome = "CANDIDATES_FOUND"
        explanation = (
            "FIND shortlisted product-relevant sources using the strongest available platform-performance evidence. "
            "Views/likes-only candidates are labeled as engagement proxies, not proof of sales. "
            "Source-commercial/mechanism eligibility and Target-Fit remain separate downstream transfer gates."
        )
    elif raw_returned == 0:
        outcome = "NO_API_ROWS"
        explanation = "SocialCrawl returned no ad rows; check query coverage, credentials, provider response, and available credits."
    elif target_relevant == 0:
        outcome = "NO_TARGET_RELEVANCE"
        explanation = (
            "Rows were returned, but no playable source had confirmed target relevance. Unknown-relevance rows are held "
            "for a separate source check, not admitted from the query keyword alone. "
            "commercial success is not required to extract a pattern in the manual pattern-only path."
        )
    elif performance_ready == 0:
        outcome = "NO_PERFORMANCE_EVIDENCE"
        explanation = (
            "Product-relevant sources were found, but none had a performance-board position or an available "
            "performance/engagement score strong enough to justify the analysis shortlist. This is not a profitability verdict."
        )
    else:
        outcome = "NO_RANKED_CANDIDATE"
        explanation = (
            "Readiness counters show qualified sources, but none survived final queue construction; inspect "
            "deduplication and source-selection diagnostics."
        )
    return {
        "outcome": outcome,
        "explanation": explanation,
        "search_lane": strategy.get("supply_lane") or "unknown",
        "requests_executed": int(stats.get("executed_requests") or len(calls)),
        "raw_ads_returned": raw_returned,
        "unique_ads": int(stats.get("unique_ads") or 0),
        "direct_relevant_ads": direct,
        "expanded_relevant_ads": expanded,
        "target_relevant_ads": target_relevant,
        "product_relevant_ads": int(stats.get("product_relevant_ads") or target_relevant),
        "performance_ready_source_ads": performance_ready,
        "money_ready_source_ads": money_ready,
        "mechanism_ready_source_ads": mechanism_ready,
        "analysis_ready_source_ads": analysis_ready,
        "unverified_ads": int(stats.get("unverified_ads") or 0),
        "hard_excluded_ads": hard_excluded,
        "source_selection_diagnostics": selection_details,
        "search_queries_executed": list(strategy.get("supply_queries_executed") or []),
        "request_sources": [str(call.get("source") or "unknown") for call in calls if isinstance(call, dict)],
    }


def _source_seen(record: dict, keys: set[str]) -> bool:
    if not keys:
        return False
    if canonical_source_identity_keys:
        return bool(canonical_source_identity_keys(record) & keys)
    return bool(set(record.get("keys") or []) & keys)


def _run_mechanism_find(*, request: dict, offer: dict, keywords: list[str], client,
                        ledger: SeenLedger, scope: dict, prior_seen_keys: set[str],
                        stamp: str, country: str, period: int, max_requests: int,
                        max_ads: int, limit_per_board: int) -> dict:
    """Explicit-keyword Sponsored-Search-only adjacent-product selection; no Ads Top lane."""
    date_posted = {7: "this-week", 30: "this-month", 180: "last-6-months"}[period]
    calls, raw_records, qualified, rejected = [], [], [], []
    rows_by_identity, identity_keys_seen = {}, set()
    raw_count = 0
    for query in keywords[:max_requests]:
        try:
            response = client.fetch_sponsored_search(
                query=query, region=country, date_posted=date_posted,
                sort_by="most-liked", relevance="filter", limit=limit_per_board,
            )
        except SocialCrawlError as exc:
            raise FindUnavailable("UNAVAILABLE", str(exc)) from exc
        except OSError as exc:
            raise FindUnavailable("UNAVAILABLE", f"{type(exc).__name__}: {exc}") from exc
        rows = response.get("items") or []
        calls.append({
            "call": len(calls) + 1, "source": "sponsored_search", "order_by": "sponsored_search",
            "keyword": query, "returned": len(rows), "credits_used": response.get("credits_used"),
            "request_id": response.get("request_id"), "provider_params": response.get("params") or {},
        })
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_count += 1
            record = dict(row)
            record.update({
                "discovery_source": "tiktok_sponsored_search", "discovery_query": query,
                "discovery_order_by": "sponsored_search", "discovery_country": country,
                "discovery_period": period, "discovery_vertical": VERTICAL,
            })
            raw_records.append(record)
            keys = canonical_source_identity_keys(record) if canonical_source_identity_keys else set()
            identity = tuple(sorted(keys))
            if not identity:
                identity = (str(record.get("ad_id") or record.get("video_id") or ""),)
            if identity == ("",):
                rejected.append({"ad_id": "", "title": str(record.get("title") or "")[:160],
                                 "reason": "MISSING_SOURCE_IDENTITY"})
                continue
            if keys and (keys & identity_keys_seen):
                rejected.append({"ad_id": str(record.get("ad_id") or record.get("video_id") or ""),
                                 "title": str(record.get("title") or "")[:160], "reason": "DUPLICATE_SOURCE"})
                continue
            rows_by_identity[identity] = record
            identity_keys_seen.update(keys)

    try:
        from datetime import datetime
        parsed_now = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        parsed_now = None

    for record in rows_by_identity.values():
        rel = classify_vertical(
            record, VERTICAL, target_offer=offer.get("target_offer") or "",
            prelander=offer.get("prelander") or "",
        ) if classify_vertical else {}
        record["vertical_relevance"] = rel
        role = classify_source_role(
            record, product_type=offer.get("product_type") or "",
            product_category=offer.get("product_category") or "",
            target_offer=offer.get("target_offer") or "",
        )
        attention = attention_rank(record, period_days=period, now=parsed_now)
        record["source_role_info"], record["attention_score"] = role, attention
        video = record.get("video_url_540p") or record.get("video_url") or record.get("video_url_360p")
        hard_excluded = bool(rel.get("excluded")) or str(rel.get("classification") or "").upper() == "HARD_EXCLUDE"
        if _source_seen(record, prior_seen_keys):
            reason = "SEEN_WITHIN_TTL"
        elif hard_excluded:
            reason = "HARD_EXCLUDED_BY_SOURCE_CHECK"
        elif not video:
            reason = "NO_PLAYABLE_VIDEO"
        elif role.get("role") != ADJACENT_MECHANISM:
            reason = str(role.get("role") or "OFF_TOPIC")
        elif not role.get("promotion_evidence"):
            reason = "NO_INDEPENDENT_PROMOTION_EVIDENCE"
        elif attention["coverage"] < MECHANISM_MIN_COVERAGE:
            reason = "LOW_ATTENTION_COVERAGE"
        elif attention["score"] < MECHANISM_ATTENTION_FLOOR:
            reason = "BELOW_ATTENTION_PROXY_FLOOR"
        else:
            reason = ""
        if reason:
            rejected.append({
                "ad_id": str(record.get("ad_id") or record.get("video_id") or ""),
                "title": str(record.get("title") or "")[:160], "reason": reason,
                "role": role.get("role"), "attention_score": attention.get("score"),
                "attention_coverage": attention.get("coverage"),
            })
            record["selection_decision"], record["selection_reason"] = "REJECTED", reason
            continue
        record["performance_evidence"] = {
            "basis": "ATTENTION_PROXY_ONLY", "score": attention["score"],
            "coverage": attention["coverage"], "is_commercial_proof": False,
        }
        record["selection_decision"] = "QUALIFIED_ADJACENT_MECHANISM"
        qualified.append(record)

    qualified.sort(key=lambda record: (
        -float((record.get("attention_score") or {}).get("score") or 0),
        -float((record.get("attention_score") or {}).get("coverage") or 0),
        str(record.get("ad_id") or record.get("video_id") or ""),
    ))
    selected = qualified[:max_ads]
    for record in selected:
        record["selection_decision"] = "SELECTED_ADJACENT_MECHANISM"
        record["selection_reason"] = (
            "Promoted adjacent product passed the source-body, media, and attention-proxy gates. "
            "Causal mechanism, transferability, and commercial value remain unevaluated."
        )
    for record in qualified[max_ads:]:
        record["selection_decision"] = "QUALIFIED_NOT_SELECTED_LIMIT"
        record["selection_reason"] = "Passed role and attention gates but exceeded max_ads."
    candidates = [to_candidate({"record": record, "decision": None}, request, stamp, "found") for record in selected]
    query_history_before = ledger.recent_queries(scope)
    ledger.remember(scope, raw_records, queries=[call.get("keyword") for call in calls])

    reason_counts = {}
    for row in rejected:
        key = str(row.get("reason") or "UNKNOWN")
        reason_counts[key] = reason_counts.get(key, 0) + 1
    unique_rows = len(rows_by_identity)
    stats = {
        "calls": calls, "planned_requests": len(calls), "executed_requests": len(calls),
        "request_budget": max_requests, "unique_ads": unique_rows, "raw_ads_returned": raw_count,
        "fresh_unique_ads": unique_rows,
        "seen_before_ads": reason_counts.get("SEEN_WITHIN_TTL", 0),
        "product_relevant_ads": sum(1 for row in rows_by_identity.values()
                                    if (row.get("source_role_info") or {}).get("role") == "TARGET_PRODUCT"),
        "adjacent_mechanism_ads": len(qualified), "auto_shortlist_ads": len(selected),
        "performance_lane_reserved": False,
        "discovery_strategy": {
            "selection_mode": "mechanism", "supply_lane": "tiktok_sponsored_search_only",
            "supply_search_enabled": True, "supply_search_budget": len(calls), "performance_budget": 0,
            "performance_calls_used": 0, "supply_queries_planned": keywords[:max_requests],
            "supply_queries_executed": [call["keyword"] for call in calls],
            "request_policy": "Explicit-keyword Sponsored Search only; Ads Top/performance ordering disabled.",
            "attention_proxy_version": "attention_absolute_v1", "attention_is_commercial_proof": False,
            "seen_ttl_days": 30, "prior_query_history": query_history_before,
        },
        "source_selection_diagnostics": {
            "qualified_count": len(qualified), "selected_count": len(selected),
            "rejected_count": len(rejected), "rejected_by_reason": dict(sorted(reason_counts.items())),
            "rejected_examples": rejected[:50], "by_reason": dict(sorted(reason_counts.items())),
            "examples": rejected[:50],
        },
    }
    diagnostics = {
        "outcome": "CANDIDATES_FOUND" if candidates else ("NO_API_ROWS" if raw_count == 0 else "NO_QUALIFIED_ADJACENT_MECHANISM"),
        "explanation": (
            "Adjacent-product sources passed text, media, and fixed-weight attention-proxy checks. This is not commercial proof; "
            "the causal mechanism and independent Target-Fit remain downstream."
            if candidates else
            "No adjacent-product source passed source-body, media, attention-coverage, and attention-floor rules. "
            "Search queries were not used as source-body evidence."
        ),
        "search_lane": "tiktok_sponsored_search_only", "requests_executed": len(calls),
        "raw_ads_returned": raw_count, "unique_ads": unique_rows, "direct_relevant_ads": 0,
        "expanded_relevant_ads": 0, "target_relevant_ads": stats["product_relevant_ads"],
        "adjacent_mechanism_ads": len(qualified), "unverified_ads": 0,
        "hard_excluded_ads": reason_counts.get("HARD_EXCLUDED_BY_SOURCE_CHECK", 0),
        "source_selection_diagnostics": stats["source_selection_diagnostics"],
        "search_queries_executed": [call["keyword"] for call in calls],
        "request_sources": ["sponsored_search" for _ in calls], "performance_calls_used": 0,
        "commercial_evidence_note": "Attention proxy is not commercial proof; Target-Fit remains NOT_EVALUATED.",
    }
    return {"candidates": candidates, "stats": stats, "request": request, "diagnostics": diagnostics}


def run_find(params: dict, client=None, now: str | None = None) -> dict:
    """Run product discovery or the separate explicit-keyword adjacent-mechanism lane."""
    st = find_status()
    if st["state"] == "UNAVAILABLE":
        raise FindUnavailable("UNAVAILABLE", st["detail"])
    country = str(params.get("country") or "US").upper()
    period = int(params.get("period") or 30)
    max_requests = int(params.get("max_requests") or 3)
    max_ads = int(params.get("max_ads") or 20)
    selection_mode = str(params.get("selection_mode") or "product").strip().lower()
    if selection_mode not in {"product", "mechanism"}:
        raise ValueError("selection_mode must be 'product' or 'mechanism'")
    if len(country) != 2 or not country.isalpha():
        raise ValueError("country must be a 2-letter code")
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    if not 1 <= max_requests <= 10 or not 1 <= max_ads <= 50:
        raise ValueError("max_requests must be 1..10 and max_ads 1..50")
    limit_per_board = int(params.get("limit_per_board") or 10)
    if not 1 <= limit_per_board <= 100:
        raise ValueError("limit_per_board must be 1..100")
    order_bys = [str(x) for x in (params.get("order_bys") or ["for_you"])]
    performance_order_bys = _clean_terms(params.get("performance_order_bys"))
    user_keywords = _clean_terms(params.get("keywords"))
    offer = _offer_context(params)
    performance_names = {"ctr", "cvr", "play_2s_rate", "play_6s_rate"}
    if selection_mode == "mechanism":
        if not user_keywords:
            raise ValueError("mechanism selection requires explicit keywords; offer fields are not discovery keywords")
        if any(str(value).strip() in performance_names for value in performance_order_bys + order_bys):
            raise ValueError("mechanism mode disables Ads Top/performance order; choose product mode for performance-board discovery")
    conflict = _query_offer_conflict(user_keywords, offer) if selection_mode == "product" else None
    search_terms = _clean_terms(user_keywords + [offer["target_offer"], offer["product_type"], offer["product_category"]]) if selection_mode == "product" else user_keywords
    request = {
        "country": country, "period": period, "order_bys": order_bys,
        "performance_order_bys": performance_order_bys, "keywords": user_keywords,
        "search_terms": search_terms, "selection_mode": selection_mode,
        "target_offer": offer["target_offer"], "product_type": offer["product_type"],
        "product_category": offer["product_category"], "max_ads": max_ads,
        "max_requests": max_requests, "vertical": VERTICAL,
    }
    if conflict:
        return {
            "candidates": [],
            "stats": {"calls": [], "executed_requests": 0, "unique_ads": 0, "input_validation": conflict,
                      "discovery_strategy": {"supply_lane": "not_started", "supply_queries_executed": [],
                                             "request_policy": "Stopped by explicit query/offer category conflict before provider calls."}},
            "request": request,
            "diagnostics": {
                "outcome": "INPUT_CONFLICT", "explanation": conflict["reason"], "search_lane": "not_started",
                "requests_executed": 0, "raw_ads_returned": 0, "direct_relevant_ads": 0,
                "expanded_relevant_ads": 0, "unverified_ads": 0, "hard_excluded_ads": 0,
                "search_queries_executed": [], "request_sources": [], "input_conflict": conflict,
            },
        }
    client = client or SocialCrawlTikTokTopAds()
    enabled = client.enabled() if callable(client.enabled) else client.enabled
    if not enabled:
        raise FindUnavailable("NOT_CONFIGURED", find_status()["detail"] or "SOCIALCRAWL_API_KEY is not set")
    stamp = now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ledger = SeenLedger()
    scope = build_scope(country=country, offer=offer, selection_mode=selection_mode)
    prior_seen_keys = ledger.seen_keys(scope)
    prior_query_history = ledger.recent_queries(scope)
    if selection_mode == "mechanism":
        user_keywords = _rotate_query_terms(user_keywords, prior_query_history)
        request["keywords"] = list(user_keywords)
        request["search_terms"] = list(user_keywords)
        return _run_mechanism_find(
            request=request, offer=offer, keywords=user_keywords, client=client,
            ledger=ledger, scope=scope, prior_seen_keys=prior_seen_keys, stamp=stamp,
            country=country, period=period, max_requests=max_requests, max_ads=max_ads,
            limit_per_board=limit_per_board,
        )
    search_terms = _rotate_query_terms(search_terms, prior_query_history)
    request["search_terms"] = list(search_terms)
    try:
        ranked = discover_and_rank(
            client=client, country=country, period=period, keywords=search_terms, order_bys=order_bys,
            limit_per_board=limit_per_board, max_ads=max_ads, max_requests=max_requests,
            keyword_boards=max_requests, vertical=VERTICAL, target_offer=offer["target_offer"],
            product_type=offer["product_type"] or offer["product_category"],
            prelander=offer["prelander"], landing_page=offer["landing_page"],
            discovery_mode=str(params.get("discovery_mode") or "auto"),
            exclude_ad_ids=prior_seen_keys,
            search_tags_dir=PRODUCT_SEARCH_TAGS_DIR, supply_search=True,
            performance_order_bys=performance_order_bys)
    except SocialCrawlError as exc:
        raise FindUnavailable("UNAVAILABLE", str(exc)) from exc
    except OSError as exc:
        raise FindUnavailable("UNAVAILABLE", f"{type(exc).__name__}: {exc}") from exc
    candidates = [to_candidate(item, request, stamp, "found") for item in ranked]
    stats = getattr(client, "last_discovery_stats", {}) or {}
    strategy = stats.setdefault("discovery_strategy", {})
    executed_calls = stats.get("calls") or []
    strategy["request_policy"] = (
        "Bounded Sponsored Search plus reserved Ads Top performance queries. FIND shortlists by product-type match "
        "and strongest available performance evidence; money/mechanism and Target-Fit remain downstream transfer gates."
    )
    strategy["supply_search_enabled"] = True
    strategy["supply_lane"] = "tiktok_sponsored_search_plus_ads_top_performance"
    strategy["selection_mode"] = "product"
    strategy["supply_search_calls_used"] = sum(1 for call in executed_calls if call.get("source") == "sponsored_search")
    strategy["performance_calls_used"] = sum(1 for call in executed_calls if call.get("source") == "ads_top")
    diagnostics = _find_diagnostics(stats, len(candidates))
    evidence_counts = {}
    for candidate in candidates:
        evidence = candidate.get("commercial_evidence") or {}
        level = str(evidence.get("level") or "NO_EVIDENCE")
        evidence_counts[level] = evidence_counts.get(level, 0) + 1
    diagnostics["commercial_evidence_counts"] = dict(sorted(evidence_counts.items()))
    diagnostics["economic_validation_note"] = (
        "Commercial evidence labels describe available evidence only. Score, money-gate pass, CTR/CVR, "
        "views and engagement do not by themselves confirm profitability."
    )
    executed_queries = [str(call.get("keyword") or "") for call in executed_calls
                        if isinstance(call, dict) and call.get("keyword")
                        and str(call.get("source") or "") == "sponsored_search"]
    if not executed_queries:
        executed_queries = list(strategy.get("supply_queries_executed") or [])
    ledger.remember(scope, [item.get("record") or {} for item in ranked], queries=executed_queries)
    diagnostics["selection_mode"], diagnostics["seen_ttl_days"] = "product", 30
    diagnostics["prior_query_history"] = ledger.recent_queries(scope)
    return {"candidates": candidates, "stats": stats, "request": request, "diagnostics": diagnostics}

def import_discovery_file(path) -> dict:
    """Import a saved old-engine discovery file ({request, stats, results:[{record, decision}]}).

    Imported candidates are labelled origin='imported'; they are never presented as automatically found.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("unsupported file: expected an old CME discovery JSON with a 'results' list")
    request = data.get("request") or {}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))
    out = []
    for x in data["results"]:
        if isinstance(x, dict) and isinstance(x.get("record"), dict):
            out.append(to_candidate(x, request, stamp, "imported", f"import:{p.name}"))
    if not out:
        raise ValueError("no usable records in the file")
    return {"candidates": out, "stats": data.get("stats") or {}, "request": request}
