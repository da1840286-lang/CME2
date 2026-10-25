"""FIND adapter. Runs the old engine's real discovery code and maps its output to a compact record.

Nothing here ranks or filters on its own: ranking, classification and source relevance come from the
vendored legacy modules. Unknown values stay None, never zero.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
# Search-only seeds for the live Sponsored Search lane. Keep separate from the
# legacy CTA/tag corpus, which is retained for byte-identical compatibility.
PRODUCT_SEARCH_TAGS_DIR = DATA_DIR / "product_search_tags"
os.environ.setdefault("CME_VERTICALS_DIR", str(DATA_DIR / "verticals"))  # before the legacy import

VERTICAL = "Physical Products"
METRIC_FIELDS = ("views", "likes", "comments", "shares", "ctr", "cvr", "rank", "cost_score", "like_tier",
                 "estimated_reach", "engagement_rate")
PERIODS = (7, 30, 180)

LEGACY_ERROR = ""
try:
    from .legacy.discover import canonical_source_identity_keys, discover_and_rank
    from .legacy.discovery.tiktok_socialcrawl import SocialCrawlError, SocialCrawlTikTokTopAds
except Exception as e:  # e.g. the 'requests' package is missing
    LEGACY_ERROR = f"{type(e).__name__}: {e}"
    discover_and_rank = canonical_source_identity_keys = SocialCrawlTikTokTopAds = None

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
                         "scope": rec.get("discovery_scope")},
        "source_score": score, "why": why, "discovered_at": collected_at,
        "request": {k: request.get(k) for k in ("country", "period", "order_bys", "keywords", "vertical")},
        "legacy_record": rec,
    }
    c["id"] = stable_id(c)
    return c


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
    if candidate_count:
        outcome = "CANDIDATES_FOUND"
        explanation = (
            "FIND returned sources admitted by the configured discovery mode. Automatic mode requires confirmed "
            "DIRECT/EXPANDED relevance plus either the source-commercial gate or the independent mechanism-source path; "
            "Target-Fit still gates transfer."
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
    elif analysis_ready == 0:
        outcome = "NO_QUALIFIED_SOURCE"
        explanation = (
            "Target-relevant ads were found, but none passed the source-commercial gate or the independent "
            "mechanism-source path. Weak ads were not used to fill the candidate list."
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
        "money_ready_source_ads": money_ready,
        "mechanism_ready_source_ads": mechanism_ready,
        "analysis_ready_source_ads": analysis_ready,
        "unverified_ads": int(stats.get("unverified_ads") or 0),
        "hard_excluded_ads": hard_excluded,
        "source_selection_diagnostics": selection_details,
        "search_queries_executed": list(strategy.get("supply_queries_executed") or []),
        "request_sources": [str(call.get("source") or "unknown") for call in calls if isinstance(call, dict)],
    }


def run_find(params: dict, client=None, now: str | None = None) -> dict:
    """One manual FIND run. Product-source search first; at most max_requests API calls."""
    st = find_status()
    if st["state"] == "UNAVAILABLE":
        raise FindUnavailable("UNAVAILABLE", st["detail"])
    country = str(params.get("country") or "US").upper()
    period = int(params.get("period") or 30)
    max_requests = int(params.get("max_requests") or 3)
    max_ads = int(params.get("max_ads") or 20)
    if len(country) != 2 or not country.isalpha():
        raise ValueError("country must be a 2-letter code")
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    if not 1 <= max_requests <= 10 or not 1 <= max_ads <= 50:
        raise ValueError("max_requests must be 1..10 and max_ads 1..50")
    client = client or SocialCrawlTikTokTopAds()
    enabled = client.enabled() if callable(client.enabled) else client.enabled  # a property in the old client
    if not enabled:
        raise FindUnavailable("NOT_CONFIGURED", find_status()["detail"] or "SOCIALCRAWL_API_KEY is not set")
    order_bys = [str(x) for x in (params.get("order_bys") or ["for_you"])]
    user_keywords = _clean_terms(params.get("keywords"))
    offer = _offer_context(params)
    # Query exact operator terms first, then offer identity, product type and category.
    # The adaptive ranker can fall back to product-category seeds from product_search_tags;
    # CTA-only tags are intentionally not used as the first discovery lane.
    search_terms = _clean_terms(user_keywords + [offer["target_offer"], offer["product_type"], offer["product_category"]])
    request = {"country": country, "period": period, "order_bys": order_bys, "keywords": user_keywords,
               "search_terms": search_terms, "target_offer": offer["target_offer"],
               "product_type": offer["product_type"], "product_category": offer["product_category"],
               "max_ads": max_ads, "max_requests": max_requests, "vertical": VERTICAL}
    try:
        ranked = discover_and_rank(
            client=client, country=country, period=period, keywords=search_terms, order_bys=order_bys,
            limit_per_board=int(params.get("limit_per_board") or 10), max_ads=max_ads, max_requests=max_requests,
            keyword_boards=max_requests, vertical=VERTICAL, target_offer=offer["target_offer"],
            prelander=offer["prelander"], landing_page=offer["landing_page"],
            discovery_mode=str(params.get("discovery_mode") or "auto"),
            search_tags_dir=PRODUCT_SEARCH_TAGS_DIR, supply_search=True)
    except SocialCrawlError as e:
        raise FindUnavailable("UNAVAILABLE", str(e)) from e
    except OSError as e:  # network errors from requests derive from OSError
        raise FindUnavailable("UNAVAILABLE", f"{type(e).__name__}: {e}") from e
    stamp = now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    candidates = [to_candidate(x, request, stamp, "found") for x in ranked]
    stats = getattr(client, "last_discovery_stats", {}) or {}
    strategy = stats.setdefault("discovery_strategy", {})
    executed_calls = stats.get("calls") or []
    strategy["request_policy"] = (
        "Bounded Sponsored Search plus reserved Ads Top performance queries, even when no prior source passes "
        "the money/mechanism gate; final automatic eligibility and Target-Fit transfer gates remain strict."
    )
    strategy["supply_search_enabled"] = True
    strategy["supply_lane"] = "tiktok_sponsored_search_plus_ads_top_performance"
    strategy["supply_search_calls_used"] = sum(1 for call in executed_calls if call.get("source") == "sponsored_search")
    strategy["performance_calls_used"] = sum(1 for call in executed_calls if call.get("source") == "ads_top")
    diagnostics = _find_diagnostics(stats, len(candidates))
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
