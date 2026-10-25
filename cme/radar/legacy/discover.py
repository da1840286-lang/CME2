from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from .arbitrage import score_tiktok_candidate
from .discovery.tiktok_socialcrawl import SocialCrawlTikTokTopAds


from .verticals import canonical_vertical, discovery_seed_groups
from .market_taxonomy import TAXONOMIES, resolve_key


_SEARCH_TAGS_DEFAULT_DIR = Path("data/search_tags")


def canonical_source_identity_keys(record: dict[str, Any] | None) -> set[str]:
    """Stable ad/video/post identities; remove signed query strings from source URLs."""
    if not isinstance(record, dict):
        return set()
    keys: set[str] = set()
    ad_id = str(record.get("ad_id") or "").strip()
    video_id = str(record.get("video_id") or "").strip()
    if ad_id:
        keys.update({ad_id, f"ad:{ad_id}"})
    if video_id:
        keys.add(f"video:{video_id}")
    raw_url = str(record.get("source_url") or record.get("url") or "").strip()
    if raw_url:
        try:
            parsed = urlsplit(raw_url)
            if parsed.scheme and parsed.netloc:
                canonical = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/", "", ""))
            else:
                canonical = raw_url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        except Exception:
            canonical = raw_url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        if canonical:
            keys.add(f"url:{canonical}")
    return keys


def is_source_seen(record: dict[str, Any] | None, seen_keys: set[str] | None) -> bool:
    known = {str(x) for x in (seen_keys or set()) if x}
    return bool(canonical_source_identity_keys(record) & known)


def vertical_slug(vertical: str) -> str:
    value = " ".join((vertical or "").strip().lower().split())
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "default"


def search_tags_path(vertical: str, search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR) -> Path:
    return Path(search_tags_dir) / f"{vertical_slug(vertical)}.txt"


def load_search_tags(vertical: str, search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR) -> list[str]:
    p = search_tags_path(vertical, search_tags_dir)
    if not p.exists():
        return []
    tags: list[str] = []
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
        if line:
            tags.append(line)
    return list(dict.fromkeys(tags))


def list_search_tag_files(search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR) -> list[dict[str, str]]:
    root = Path(search_tags_dir)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.glob("*.txt")):
        out.append({"slug": p.stem, "filename": p.name, "path": str(p)})
    return out


def _coverage_path() -> Path:
    path = Path.home() / ".creative_engine" / "discovery_coverage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _query_seed_pool(vertical: str, search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR) -> list[str]:
    key = " ".join((vertical or "").lower().split())
    canonical = resolve_key(key) or key
    file_tags = load_search_tags(vertical, search_tags_dir)
    brands: list[str] = []
    for group in discovery_seed_groups(canonical):
        brands.extend(group)

    # Sponsored Search behaves much better with identifiable product/brand entities
    # than with generic semantic queries. Keep ambiguous entities (e.g. `Cursor`, which
    # is also a common UI term) out of the first search positions, but retain them as
    # a later coverage layer. Generic terms remain available as a fallback corpus.
    taxonomy = TAXONOMIES.get(canonical)
    ambiguous_seeds = {x.lower() for x in (taxonomy.ambiguous_seed_terms if taxonomy else ())}
    strong_brands = [x for x in brands if x and x.lower() not in ambiguous_seeds]
    ambiguous_brands = [x for x in brands if x and x.lower() in ambiguous_seeds]
    return list(dict.fromkeys(strong_brands + ambiguous_brands + file_tags))


def _query_signal(keyword: str | None, vertical: str) -> tuple[float, str]:
    """Score a Sponsored Search query by target-market identity strength.

    This score is a routing signal only. It never establishes that an individual ad is
    relevant; Source Verify / visual review remain authoritative for the ad itself.
    """
    value = " ".join((keyword or "").strip().lower().split())
    if not value:
        return 0.0, "NONE"
    taxonomy = TAXONOMIES.get(resolve_key(vertical) or "")
    if not taxonomy:
        return 20.0, "LOW"
    direct_brands = {x.lower() for x in taxonomy.direct_brands}
    ambiguous = {x.lower() for x in taxonomy.ambiguous_seed_terms}
    adjacent_brands = {x.lower() for x in taxonomy.adjacent_brands}
    direct_terms = {x.lower() for x in taxonomy.direct_terms}
    adjacent_terms = {x.lower() for x in taxonomy.adjacent_terms}
    if value in direct_brands:
        return (55.0, "MEDIUM") if value in ambiguous else (100.0, "HIGH")
    if value in adjacent_brands:
        return 45.0, "MEDIUM"
    if value in direct_terms:
        return 30.0, "LOW"
    if value in adjacent_terms:
        return 20.0, "LOW"
    return 10.0, "LOW"


def _automatic_supply_query_pool(
    vertical: str,
    excluded: set[str],
    search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR,
) -> list[str]:
    """Rank automatic seeds by source-market identity strength, preserving tie order."""
    canonical = canonical_vertical(vertical)
    ranked: list[tuple[str, float, int]] = []
    seen = {str(x).lower() for x in excluded if x}
    for source_rank, raw in enumerate(_query_seed_pool(canonical, search_tags_dir)):
        value = " ".join((raw or "").strip().split())
        norm = value.lower()
        if not value or norm == "auto" or norm in seen:
            continue
        seen.add(norm)
        score, _ = _query_signal(value, canonical)
        ranked.append((value, score, source_rank))
    ranked.sort(key=lambda x: (-x[1], x[2]))
    return [value for value, _, _ in ranked]


def _ordered_supply_queries(
    vertical: str,
    explicit: list[str],
    max_count: int,
    search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR,
) -> list[str]:
    """Keep operator keywords fixed; rotate automatic brand seeds across live runs."""
    if max_count <= 0:
        return []
    canonical = canonical_vertical(vertical)
    explicit_ranked: list[tuple[str, float, int]] = []
    explicit_seen: set[str] = set()
    for source_rank, raw in enumerate(explicit):
        value = " ".join((raw or "").strip().split())
        norm = value.lower()
        if not value or norm == "auto" or norm in explicit_seen:
            continue
        explicit_seen.add(norm)
        score, _ = _query_signal(value, canonical)
        explicit_ranked.append((value, score, source_rank))
    # Operator-provided terms still honor source-market identity strength, so a
    # concrete product brand (Replit) precedes ambiguous entities (Cursor).
    explicit_ranked.sort(key=lambda x: (-x[1], x[2]))
    explicit_values = [value for value, _, _ in explicit_ranked]
    automatic = _automatic_supply_query_pool(canonical, explicit_seen, search_tags_dir)
    if not automatic:
        return explicit_values[:max_count]
    path = _coverage_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        payload = {}
    key = " ".join((canonical or "").lower().split()) + "::supply"
    try:
        offset = int(payload.get(key, 0) or 0) % len(automatic)
    except (TypeError, ValueError):
        offset = 0
    rotated = automatic[offset:] + automatic[:offset]
    return (explicit_values + rotated)[:max_count]


def _advance_supply_query_cursor(
    vertical: str,
    keyword: str,
    *,
    explicit: list[str],
    search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR,
) -> None:
    """Advance the saved cursor only after an automatic query actually executes."""
    value = " ".join((keyword or "").strip().split())
    norm = value.lower()
    explicit_norm = {
        " ".join((x or "").strip().split()).lower()
        for x in explicit if x and x.strip() and x.strip().lower() != "auto"
    }
    if not value or norm in explicit_norm:
        return
    canonical = canonical_vertical(vertical)
    automatic = _automatic_supply_query_pool(canonical, explicit_norm, search_tags_dir)
    if not any(candidate.lower() == norm for candidate in automatic):
        return
    path = _coverage_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    key = " ".join((canonical or "").lower().split()) + "::supply"
    try:
        offset = int(payload.get(key, 0) or 0)
    except (TypeError, ValueError):
        offset = 0
    payload[key] = (offset + 1) % len(automatic)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _next_seed_groups(vertical: str, count: int, search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR) -> list[str]:
    if count <= 0:
        return []
    seeds = _query_seed_pool(vertical, search_tags_dir)
    if not seeds:
        return []
    path = _coverage_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        payload = {}
    key = " ".join((vertical or "").lower().split())
    file_tags = load_search_tags(vertical, search_tags_dir)
    file_set = {x.lower() for x in file_tags}
    primary = [x for x in seeds if x.lower() in file_set]
    fallback = [x for x in seeds if x.lower() not in file_set]
    tag_idx = int(payload.get(f"{key}::tags", 0) or 0) % len(primary) if primary else 0
    selected: list[str] = []
    for offset in range(min(count, len(primary))):
        selected.append(primary[(tag_idx + offset) % len(primary)])
    remaining = count - len(selected)
    if remaining and fallback:
        fb_idx = int(payload.get(f"{key}::fallback", 0) or 0) % len(fallback)
        for offset in range(min(remaining, len(fallback))):
            selected.append(fallback[(fb_idx + offset) % len(fallback)])
        payload[f"{key}::fallback"] = (fb_idx + remaining) % len(fallback)
    if primary:
        payload[f"{key}::tags"] = (tag_idx + min(count, len(primary))) % len(primary)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return list(dict.fromkeys(selected))


def vertical_seeds(vertical: str) -> list[str]:
    groups = discovery_seed_groups(vertical)
    return [g[0] for g in groups if g]


def build_discovery_request(
    *,
    country: str,
    period: int,
    keywords: list[str] | None,
    order_bys: list[str],
    industry: str = "",
    ad_language: str = "en",
    limit_per_board: int = 10,
    max_requests: int = 5,
    vertical: str = "",
    discovery_mode: str = "auto",
    search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR,
) -> dict[str, Any]:
    seeds = [x for x in (keywords or []) if x.strip()]
    return {
        "country": country.upper(),
        "period": period,
        "keywords": seeds,
        "order_bys": order_bys,
        "industry": industry,
        "ad_language": ad_language,
        "limit_per_board": min(100, max(1, limit_per_board)),
        "max_requests": max(1, max_requests),
        "vertical": vertical,
        "discovery_mode": discovery_mode,
        "auto_vertical_seeds": vertical_seeds(vertical),
        "search_tags_file": str(search_tags_path(vertical, search_tags_dir)),
        "search_tags": load_search_tags(vertical, search_tags_dir),
    }


def load_seen_ad_ids(results_dir: str | Path = "results", country: str = "", vertical: str = "", *, include_local: bool = True) -> set[str]:
    root = Path(results_dir)
    seen: set[str] = set()
    local_files = ("tiktok_discovery.json", "last_job_discovery.json") if include_local else ()
    for filename in local_files:
        p = root / filename
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in payload.get("results") or []:
            record = item.get("record") if isinstance(item, dict) else None
            if isinstance(record, dict):
                seen.update(canonical_source_identity_keys(record))
    # Cross-version suppression: keep a small per-user index keyed by country + target vertical.
    global_path = Path.home() / ".creative_engine" / "seen_tiktok_ads.json"
    if global_path.exists():
        try:
            payload = json.loads(global_path.read_text(encoding="utf-8"))
            scope = f"{(country or '').upper()}::{(vertical or '').strip().lower()}"
            for ad_id in payload.get(scope, []):
                seen.add(str(ad_id))
        except Exception:
            pass
    return seen


def remember_seen_ad_ids(ranked: list[dict[str, Any]], request: dict[str, Any]) -> None:
    """Persist selected IDs so a new project/version does not rediscover the same ad."""
    path = Path.home() / ".creative_engine" / "seen_tiktok_ads.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[str]] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload = {str(k): [str(x) for x in (v or [])] for k, v in raw.items() if isinstance(v, list)}
        except Exception:
            payload = {}
    scope = f"{str(request.get('country') or '').upper()}::{str(request.get('vertical') or '').strip().lower()}"
    current = set(payload.get(scope, []))
    for item in ranked:
        current.update(canonical_source_identity_keys(item.get("record") or {}))
    payload[scope] = sorted(current)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_item(combined: dict[str, dict[str, Any]], item: dict[str, Any], order_by: str, keyword: str | None) -> bool:
    key = item.get("ad_id") or item.get("video_id") or item.get("source_url") or item.get("video_url")
    if not key:
        return False
    key = str(key)
    existing = combined.get(key)
    query_quality, query_confidence = _query_signal(keyword, str(item.get("discovery_vertical") or "")) if keyword else (0.0, "NONE")
    if order_by == "sponsored_search" and keyword:
        item["discovery_query_quality"] = query_quality
        item["discovery_query_confidence"] = query_confidence
    else:
        item.setdefault("discovery_query_quality", 0.0)
        item.setdefault("discovery_query_confidence", "NONE")
    if existing is None:
        item["discovery_boards"] = [order_by]
        item["discovery_keywords"] = [keyword] if keyword else []
        item["discovery_hits"] = 1
        combined[key] = item
        return True
    existing["discovery_hits"] = int(existing.get("discovery_hits", 1)) + 1
    if order_by not in existing.setdefault("discovery_boards", []):
        existing["discovery_boards"].append(order_by)
    if keyword and keyword not in existing.setdefault("discovery_keywords", []):
        existing["discovery_keywords"].append(keyword)
    for field in (
        "ctr", "cvr", "rank", "cost_score", "like_tier", "views", "likes",
        "comments", "shares", "estimated_reach", "duration_seconds", "video_url",
        "video_url_540p", "video_url_360p", "source_url", "industry", "objective",
    ):
        if existing.get(field) in (None, "") and item.get(field) not in (None, ""):
            existing[field] = item[field]
    if float(existing.get("discovery_query_quality", 0.0) or 0.0) < float(item.get("discovery_query_quality", 0.0) or 0.0):
        existing["discovery_query_quality"] = item.get("discovery_query_quality", 0.0)
        existing["discovery_query_confidence"] = item.get("discovery_query_confidence", "NONE")
    return False


def _has_video_url(record: dict[str, Any]) -> bool:
    """Return True when a discovered record exposes a usable video URL.

    SocialCrawl can place media either in normalized fields or inside the raw
    post payload. Discovery should only count a record as source-ready when a
    downloader has something concrete to fetch.
    """
    direct = record.get("video_url") or record.get("video_url_540p") or record.get("video_url_360p")
    if direct:
        return True
    raw = record.get("raw") or {}
    post = raw.get("post") if isinstance(raw, dict) else {}
    if not isinstance(post, dict):
        return False
    content = post.get("content") or {}
    ext = post.get("ext") or {}
    ad = ext.get("ad") or {}
    media = content.get("media_urls")
    if isinstance(media, list):
        if any(str(x or "").strip() for x in media):
            return True
    elif media:
        return True
    return bool(ad.get("video_url") or ad.get("video_url_540p") or ad.get("video_url_360p"))


def _is_hard_excluded(record: dict[str, Any]) -> bool:
    rel = record.get("vertical_relevance") or {}
    return rel.get("classification") == "HARD_EXCLUDE" or bool(rel.get("excluded"))


def _discovery_ready_count(records: list[dict[str, Any]], vertical: str) -> int:
    """Count raw records with usable media that are not hard-excluded."""
    if not vertical:
        return sum(1 for r in records if _has_video_url(r))
    return sum(1 for r in records if _has_video_url(r) and not _is_hard_excluded(r))


def _source_queue_candidate_count(records: list[dict[str, Any]], vertical: str) -> int:
    """Count candidates eligible to enter the Source Verify queue.

    The queue intentionally includes UNVERIFIED rows. They are not target-approved;
    they are backup source candidates that Gemini may verify after deterministic checks.
    Only hard-excluded records and rows without usable video are kept out.
    """
    return _discovery_ready_count(records, vertical)


def _fetch_plan(
    *,
    client: SocialCrawlTikTokTopAds,
    country: str,
    period: int,
    industry: str,
    vertical: str = "",
    ad_language: str,
    limit_per_board: int,
    plan: Iterable[tuple[str, str | None]],
    combined: dict[str, dict[str, Any]],
    calls: list[dict[str, Any]],
    total_credits: int,
) -> int:
    for call_no, (order_by, keyword) in enumerate(plan, start=len(calls) + 1):
        if order_by == "sponsored_search":
            result = client.fetch_sponsored_search(
                query=keyword or "", region=country,
                date_posted={7: "this-week", 30: "this-month", 180: "last-6-months"}.get(period, "this-month"),
                sort_by="most-liked", relevance="filter",
            )
        else:
            result = client.fetch_board(
                country=country, period=period, order_by=order_by, keyword=keyword,
                industry=industry or None, ad_language=ad_language, limit=limit_per_board,
            )
        credits = result.get("credits_used")
        if isinstance(credits, (int, float)):
            total_credits += int(credits)
        before = len(combined)
        new_items = 0
        for item in result["items"]:
            if not isinstance(item, dict):
                continue
            item["discovery_vertical"] = str(vertical or "")
            item.setdefault("discovery_source", "tiktok_sponsored_search" if order_by == "sponsored_search" else "ads_top")
            item.setdefault("discovery_query", keyword or "")
            item.setdefault("discovery_order_by", order_by)
            if _merge_item(combined, item, order_by, keyword):
                new_items += 1
        calls.append({
            "call": call_no,
            "source": "sponsored_search" if order_by == "sponsored_search" else "ads_top",
            "order_by": order_by,
            "keyword": keyword or "",
            "returned": len(result["items"]),
            "new_unique": new_items,
            "unique_after": len(combined),
            "credits_used": credits,
            "request_id": result.get("request_id"),
        })
    return total_credits


def _classify(records: list[dict[str, Any]], vertical: str, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from .verticals import classify_vertical
    strict_pool: list[dict[str, Any]] = []
    expanded_pool: list[dict[str, Any]] = []
    excluded_pool: list[dict[str, Any]] = []
    unverified_pool: list[dict[str, Any]] = []
    for r in records:
        rel = classify_vertical(r, vertical, mode=mode) if vertical else {
            "status": "UNKNOWN", "classification": "DIRECT", "score": 50.0, "hits": [], "strong_hits": [], "excluded": [],
            "strict_match": True, "expanded_match": True, "unverified": False, "reason": "not_supplied",
        }
        r["vertical_relevance"] = rel
        if not vertical:
            strict_pool.append(r)
            continue
        classification = rel.get("classification")
        if classification == "HARD_EXCLUDE" or rel.get("excluded"):
            excluded_pool.append(r)
        elif classification == "DIRECT" or rel.get("strict_match"):
            strict_pool.append(r)
        elif classification == "EXPANDED" or rel.get("expanded_match"):
            expanded_pool.append(r)
        else:
            unverified_pool.append(r)
    return strict_pool, expanded_pool, excluded_pool, unverified_pool


def _preselection_relevance(item: dict[str, Any]) -> float:
    """Return source-vertical relevance evidence for pre-video ordering.

    UNVERIFIED does not mean approved; the score is only used to order unknowns
    by the amount of source-side thematic evidence available before Gemini.
    """
    record = item.get("record") or {}
    rel = record.get("vertical_relevance") or {}
    try:
        return float(rel.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _rank(pool: list[dict[str, Any]], *, target_offer: str, prelander: str, landing_page: str, vertical: str, population: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not pool:
        return []
    # Keep ranking and the source-ready stop gate on the same performance cohort.
    comparison_population = population if population is not None else pool
    ranked: list[dict[str, Any]] = []
    for r in pool:
        decision = score_tiktok_candidate(
            r, comparison_population, target_offer=target_offer, prelander=prelander,
            landing_page=landing_page, vertical=vertical,
        )
        ranked.append({"record": r, "decision": decision})
    # Source relevance is the primary ordering signal for discovery. Performance
    # remains the tie-breaker within the same relevance level.
    ranked.sort(
        key=lambda x: (
            -_preselection_relevance(x),
            -float(getattr(x.get("decision"), "final_score", 0) or 0),
        )
    )
    return ranked


def _fresh_first(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer unseen ads, then source relevance, then pre-video decision score."""
    return sorted(
        ranked,
        key=lambda x: (
            bool((x.get("record") or {}).get("seen_before")),
            -_preselection_relevance(x),
            -float(getattr(x.get("decision"), "final_score", 0) or 0),
        ),
    )


def _money_ready_item(item: dict[str, Any]) -> bool:
    """Whether a ranked source item is safe to enter the Selection queue."""
    decision = item.get("decision") or {}
    if isinstance(decision, dict):
        components = decision.get("components") or {}
        action = str(decision.get("action") or "")
    else:
        components = getattr(decision, "components", {}) or {}
        action = str(getattr(decision, "action", "") or "")
    gate = components.get("money_signal_gate") or {}
    return bool(gate.get("pass")) and action in {"DEEP", "SHALLOW"}


def _select_money_ready_queue(
    pools: list[list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    """Build an eligible queue across all discovery scopes without changing pool ranking."""
    if limit <= 0:
        return []
    supply_queries: list[str] = []
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Scope priority is deterministic: STRICT -> EXPANDED -> UNVERIFIED -> MECHANISM.
    # Within each pool, preserve the existing ranking.
    for pool in pools:
        for item in pool:
            record = item.get("record") or {}
            cid = str(record.get("ad_id") or "")
            if not cid or cid in seen or not _money_ready_item(item):
                continue
            seen.add(cid)
            selected.append(item)
            if len(selected) >= limit:
                return selected
    return selected


def discover_and_rank(
    *,
    client: SocialCrawlTikTokTopAds,
    country: str,
    period: int,
    keywords: list[str] | None,
    order_bys: list[str],
    industry: str = "",
    ad_language: str = "en",
    limit_per_board: int = 10,
    max_ads: int = 20,
    max_requests: int = 5,
    keyword_boards: int = 1,
    vertical: str = "",
    target_offer: str = "",
    prelander: str = "",
    landing_page: str = "",
    discovery_mode: str = "auto",
    exclude_ad_ids: set[str] | None = None,
    search_tags_dir: str | Path = _SEARCH_TAGS_DEFAULT_DIR,
    supply_search: bool = False,
    source_queue_size: int | None = None,
) -> list[dict[str, Any]]:
    mode = (discovery_mode or "auto").lower().strip()
    if mode not in {"auto", "strict", "expanded", "mechanism-hunt"}:
        raise ValueError("discovery_mode must be auto, strict, expanded or mechanism-hunt")

    exclude_ad_ids = {str(x) for x in (exclude_ad_ids or set()) if x}
    vertical = canonical_vertical(vertical, target_offer=target_offer, prelander=prelander)
    order_list: list[str] = []
    for x in list(order_bys) + ["for_you", "ctr", "cvr", "play_6s_rate", "play_2s_rate"]:
        if x in client.ORDER_BY and x not in order_list:
            order_list.append(x)
    # Target-query rotation uses only the operator-selected surfaces. The built-in
    # fallbacks remain available for broad coverage but must not silently replace the
    # configured order when a target query is being executed.
    target_order_list = [x for x in list(order_bys) if x in client.ORDER_BY]
    if not target_order_list:
        target_order_list = ["for_you"]

    try:
        source_queue_limit = max(int(source_queue_size or max_ads), int(max_ads))
    except (TypeError, ValueError):
        source_queue_limit = int(max_ads)

    combined: dict[str, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    total_credits = 0

    explicit = [x.strip() for x in (keywords or []) if x and x.strip()]
    explicit_idx = 0
    used_keywords: set[str] = set()

    def eligible_records() -> list[dict[str, Any]]:
        rows = list(combined.values())
        for r in rows:
            r["seen_before"] = is_source_seen(r, exclude_ad_ids)
        return rows

    def source_queue_count() -> int:
        records = [r for r in eligible_records() if not r.get("seen_before")]
        # Diagnostic count stays broad. It must not control Discovery termination because
        # raw video inventory can be entirely organic/unverified and later fail the
        # pre-video money gate.
        _classify(records, vertical, mode="strict")
        return _source_queue_candidate_count(records, vertical)

    def target_relevant_count() -> int:
        """Count target-relevant candidates with usable media; this controls Discovery recovery."""
        records = [r for r in eligible_records() if not r.get("seen_before")]
        strict, expanded, _, _ = _classify(records, vertical, mode="strict")
        return sum(
            1
            for record in (strict + expanded)
            if _has_video_url(record) and not _is_hard_excluded(record)
        )

    def source_analysis_ready_counts() -> tuple[int, int]:
        """Count distinct candidates ready for analysis via either supported path.

        Commercial readiness must pass the same Selection gate. The alternate path is
        allowed only when the scorer independently reports mechanism_source_score >= 70
        and keeps the action non-REJECT (including the intentional long-source exception).
        """
        money_count = 0
        mechanism_count = 0
        from .verticals import source_market_evidence

        records = [r for r in eligible_records() if not r.get("seen_before")]
        strict_pool, expanded_pool, _, _ = _classify(records, vertical, mode="strict")
        for record in strict_pool + expanded_pool:
            if not _has_video_url(record):
                continue

            try:
                market = source_market_evidence(record, vertical) if vertical else {}
            except Exception:
                market = {}
            conflict = market.get("conflict_evidence") if isinstance(market, dict) else {}
            if _is_hard_excluded(record) or (
                isinstance(market, dict)
                and market.get("status") == "REJECT"
                and isinstance(conflict, dict)
                and any(bool(conflict.get(k)) for k in ("brand", "domain", "terms", "industry"))
            ):
                continue

            try:
                decision = score_tiktok_candidate(
                    record,
                    records,
                    target_offer=target_offer,
                    prelander=prelander,
                    landing_page=landing_page,
                    vertical=vertical,
                )
            except Exception:
                continue

            item = {"record": record, "decision": decision}
            if _money_ready_item(item):
                money_count += 1
                continue

            action = str(getattr(decision, "action", "") or "").upper()
            source_value_type = str(getattr(decision, "source_value_type", "") or "").upper()
            try:
                mechanism_score = float(getattr(decision, "mechanism_source_score", 0) or 0)
            except (TypeError, ValueError):
                mechanism_score = 0.0
            if (
                action in {"DEEP", "SHALLOW"}
                and source_value_type in {"MECHANISM_SOURCE", "BOTH"}
                and mechanism_score >= 70.0
            ):
                mechanism_count += 1

        return money_count, mechanism_count
    if supply_search and mode != "mechanism-hunt":
        # Live adaptive policy:
        # 1) keep searching target/product queries while no source passes a supported readiness gate,
        # 2) permit Ads Top only after a commercial money-gate pass or an independent mechanism-source score >= 70,
        # 3) if bounded target search produces no qualified source, stop without buying broad Ads Top inventory.
        supply_search_budget = min(max(0, int(keyword_boards)), max_requests)
        supply_queries = _ordered_supply_queries(
            vertical, explicit, max_count=supply_search_budget, search_tags_dir=search_tags_dir,
        )
        query_index = 0
        search_calls = 0
        top_index = 0
        while len(calls) < max_requests:
            money_count, mechanism_count = source_analysis_ready_counts()
            analysis_ready_count = money_count + mechanism_count

            if analysis_ready_count >= source_queue_limit:
                break

            if analysis_ready_count == 0 and search_calls < supply_search_budget and query_index < len(supply_queries):
                kw = supply_queries[query_index]
                query_index += 1
                search_calls += 1
                total_credits = _fetch_plan(
                    client=client, country=country, period=period, industry=industry, vertical=vertical,
                    ad_language=ad_language, limit_per_board=limit_per_board,
                    plan=[("sponsored_search", kw)], combined=combined, calls=calls,
                    total_credits=total_credits,
                )
                _advance_supply_query_cursor(
                    vertical, kw, explicit=explicit, search_tags_dir=search_tags_dir
                )
                continue

            # Target relevance without commercial/mechanism readiness is not enough to
            # justify the expensive Ads Top query. Keep the rejected source in diagnostics.
            if analysis_ready_count == 0:
                break

            # A qualified source exists. Only now may Ads Top be used to fill an incomplete queue.
            if len(calls) >= max_requests:
                break
            order_by = order_list[top_index % len(order_list)]
            total_credits = _fetch_plan(
                client=client, country=country, period=period, industry=industry, vertical=vertical,
                ad_language=ad_language, limit_per_board=limit_per_board,
                plan=[(order_by, None)], combined=combined, calls=calls,
                total_credits=total_credits,
            )
            top_index += 1
            continue
    else:
        # Legacy Ads Top path: keep the old broad/performance behaviour when the
        # dedicated Sponsored Search supply lane is explicitly disabled.
        try:
            target_budget = min(max(0, int(keyword_boards)), max_requests)
        except (TypeError, ValueError):
            target_budget = 0
        fallback_reserve = 1 if target_budget and max_requests - target_budget > 0 else 0
        initial_broad_budget = max(0, max_requests - target_budget - fallback_reserve)
        if max_requests > 1 and initial_broad_budget == 0:
            initial_broad_budget = 1
            target_budget = min(target_budget, max_requests - initial_broad_budget)

        for idx in range(initial_broad_budget):
            order_by = order_list[idx % len(order_list)]
            total_credits = _fetch_plan(
                client=client, country=country, period=period, industry=industry, vertical=vertical,
                ad_language=ad_language, limit_per_board=limit_per_board,
                plan=[(order_by, None)], combined=combined, calls=calls,
                total_credits=total_credits,
            )
            if source_queue_count() >= source_queue_limit and target_relevant_count() > 0:
                break

        if target_relevant_count() == 0 or source_queue_count() < source_queue_limit:
            while len(calls) < max_requests and len(calls) < initial_broad_budget + target_budget:
                kw = None
                while explicit_idx < len(explicit) and kw is None:
                    candidate = explicit[explicit_idx].strip()
                    explicit_idx += 1
                    norm = candidate.lower()
                    if norm and norm not in used_keywords:
                        kw = candidate
                if kw is None and vertical and mode != "mechanism-hunt":
                    rotated = _next_seed_groups(vertical, count=1, search_tags_dir=search_tags_dir)
                    if rotated:
                        candidate = rotated[0].strip()
                        norm = candidate.lower()
                        if norm and norm not in used_keywords:
                            kw = candidate
                if kw is None:
                    break
                used_keywords.add(kw.lower())
                order_by = target_order_list[len(calls) % len(target_order_list)]
                total_credits = _fetch_plan(
                    client=client, country=country, period=period, industry=industry, vertical=vertical,
                    ad_language=ad_language, limit_per_board=limit_per_board,
                    plan=[(order_by, kw)], combined=combined, calls=calls, total_credits=total_credits,
                )
                if source_queue_count() >= source_queue_limit and target_relevant_count() > 0:
                    break

        if source_queue_count() < source_queue_limit and len(calls) < max_requests:
            fallback_index = initial_broad_budget
            while len(calls) < max_requests and source_queue_count() < source_queue_limit:
                total_credits = _fetch_plan(
                    client=client, country=country, period=period, industry=industry, vertical=vertical,
                    ad_language=ad_language, limit_per_board=limit_per_board,
                    plan=[(order_list[fallback_index % len(order_list)], None)],
                    combined=combined, calls=calls, total_credits=total_credits,
                )
                fallback_index += 1

    strict_pool, expanded_pool, excluded_pool, unverified_pool = _classify(eligible_records(), vertical, mode="strict")


    records = eligible_records()
    fresh_records = [r for r in records if not r.get("seen_before")]
    strict_pool, expanded_pool, excluded_pool, unverified_pool = _classify(fresh_records, vertical, mode="strict")

    # Seen ads stay in raw diagnostics but cannot satisfy supply gates, occupy handoff
    # slots, or be sent through external reasoning again.
    direct_pool = strict_pool + expanded_pool
    mechanism_pool = [r for r in fresh_records if r not in direct_pool and r not in unverified_pool and not (r.get("vertical_relevance") or {}).get("excluded")]
    # Discovery hard-exclusions are terminal for the pre-Gemini stage. They remain
    # visible in diagnostics, but can never enter Selection or Download. This is the
    # key protection against obvious category contamination such as perfume/hotel ads.
    raw_verify_pool = [
        r for r in fresh_records
        if not _is_hard_excluded(r) and _has_video_url(r)
    ]
    raw_source_queue = list(raw_verify_pool)

    selected: list[dict[str, Any]] = []
    # Selection is a source queue: a row without an actual video URL cannot be
    # selected, regardless of its metadata score. Keep such rows in raw diagnostics,
    # but do not let them consume the source queue.
    strict_ranked = _fresh_first(_rank([r for r in strict_pool if _has_video_url(r)], target_offer=target_offer, prelander=prelander, landing_page=landing_page, vertical=vertical, population=records))
    expanded_ranked = _fresh_first(_rank([r for r in expanded_pool if _has_video_url(r)], target_offer=target_offer, prelander=prelander, landing_page=landing_page, vertical=vertical, population=records))
    unverified_ranked = _fresh_first(_rank([
        r for r in unverified_pool
        if _has_video_url(r)
        and r.get("discovery_source") == "tiktok_sponsored_search"
        and str(r.get("discovery_query_confidence") or "") == "HIGH"
    ], target_offer=target_offer, prelander=prelander, landing_page=landing_page, vertical=vertical, population=records))
    mechanism_ranked = _fresh_first(_rank([r for r in mechanism_pool if _has_video_url(r)], target_offer=target_offer, prelander=prelander, landing_page=landing_page, vertical=vertical, population=records))

    if mode == "strict":
        selected = strict_ranked[:source_queue_limit]
        fallback_path = "STRICT_ONLY"
    elif mode == "expanded":
        selected = expanded_ranked[:source_queue_limit]
        fallback_path = "STRICT_PLUS_EXPANDED"
    elif mode == "mechanism-hunt":
        selected = mechanism_ranked[:source_queue_limit]
        fallback_path = "MECHANISM_HUNT"
    else:
        # Auto mode is target-specific: only target-relevant pools may enter the live
        # source queue. Mechanism-hunt/raw fallbacks remain diagnostic data and cannot
        # dilute a target queue with broad-market ads.
        auto_pools = [strict_ranked, expanded_ranked, unverified_ranked]
        if mode == "auto":
            selected = _select_money_ready_queue(auto_pools, source_queue_limit)
        else:
            selected = _select_money_ready_queue(
                [strict_ranked, expanded_ranked, unverified_ranked, mechanism_ranked],
                source_queue_limit,
            )
        selected_ids = {str(x["record"].get("ad_id") or "") for x in selected}
        fill_pools = (strict_ranked, expanded_ranked, unverified_ranked) if mode == "auto" else (strict_ranked, expanded_ranked, unverified_ranked, mechanism_ranked)
        for pool in fill_pools:
            for x in pool:
                cid = str(x["record"].get("ad_id") or "")
                if not cid or cid in selected_ids:
                    continue
                selected.append(x)
                selected_ids.add(cid)
                if len(selected) >= source_queue_limit:
                    break
            if len(selected) >= source_queue_limit:
                break

        # In target-specific live mode, a raw/unverified video is not allowed to fill
        # a target-market queue merely because a target source was found elsewhere.
        # That would reintroduce the exact broad-market contamination this adaptive
        # policy is designed to eliminate. Legacy/source-verification fallback remains
        # available only outside auto mode.
        if mode != "auto" and len(selected) < source_queue_limit and raw_source_queue:
            raw_ranked = _fresh_first(_rank(
                raw_source_queue,
                target_offer=target_offer,
                prelander=prelander,
                landing_page=landing_page,
                vertical=vertical,
                population=records,
            ))
            for x in raw_ranked:
                cid = str(x["record"].get("ad_id") or "")
                if not cid or cid in selected_ids:
                    continue
                x["record"]["discovery_scope"] = "SOURCE_VERIFY_REQUIRED"
                x["record"]["source_vertical_status"] = "UNVERIFIED"
                selected.append(x)
                selected_ids.add(cid)
                if len(selected) >= source_queue_limit:
                    break

        if len(strict_ranked) >= max_ads:
            fallback_path = "STRICT"
        elif len(strict_ranked) + len(expanded_ranked) >= max_ads:
            fallback_path = "STRICT_TO_EXPANDED"
        elif unverified_ranked and len(selected) > len(strict_ranked) + len(expanded_ranked):
            fallback_path = "STRICT_TO_EXPANDED_TO_UNVERIFIED_VERIFY"
        elif mechanism_ranked and len(selected) > len(strict_ranked) + len(expanded_ranked):
            fallback_path = "STRICT_TO_EXPANDED_TO_MECHANISM_HUNT"
        else:
            fallback_path = "STRICT_TO_EXPANDED_SHORTFALL"


    # IDs can differ while a platform returns the same underlying post; collapse by the
    # stable source URL/video identity before discovery stats and handoff ranking.
    seen_this_batch: set[str] = set()
    deduped_selected: list[dict[str, Any]] = []
    duplicate_source_records_removed = 0
    for item in selected:
        keys = canonical_source_identity_keys(item.get("record") or {})
        stable = {key for key in keys if key.startswith(("url:", "video:"))}
        compare = stable or {key for key in keys if key.startswith("ad:")}
        if compare & seen_this_batch:
            duplicate_source_records_removed += 1
            continue
        seen_this_batch.update(compare)
        deduped_selected.append(item)
    selected = deduped_selected

    # Target-specific live discovery must fail closed when no target supply was found.
    # A high-performance broad ad is not a valid substitute for a target-market source.
    if mode == "auto" and target_relevant_count() == 0 and not unverified_ranked:
        selected = []
        fallback_path = "NO_TARGET_RELEVANCE"

    for item in selected:
        r = item["record"]
        status = (r.get("vertical_relevance") or {}).get("status", "UNKNOWN")
        if r in strict_pool:
            scope = "STRICT"
        elif r in expanded_pool:
            scope = "EXPANDED"
        elif r in unverified_pool:
            scope = "UNVERIFIED_VERIFY"
        elif r.get("discovery_scope") == "SOURCE_VERIFY_REQUIRED":
            scope = "SOURCE_VERIFY_REQUIRED"
        else:
            scope = "MECHANISM_HUNT"
        r["discovery_scope"] = scope
        r["source_vertical_status"] = status if scope in {"STRICT", "EXPANDED"} else ("UNVERIFIED" if scope == "UNVERIFIED_VERIFY" else "OUT_OF_VERTICAL")

    stats = client.last_discovery_stats or {}
    pre_visual_verification_excluded = [
        {
            "ad_id": str(r.get("ad_id") or r.get("video_id") or ""),
            "industry": str(r.get("industry") or ""),
            "brand": str(r.get("brand") or r.get("brand_name") or ""),
            "reason": ((r.get("vertical_relevance") or {}).get("reason") or "")[:500],
        }
        for r in excluded_pool
    ]

    target_relevant_ads = sum(
        1 for record in (strict_pool + expanded_pool)
        if _has_video_url(record) and not _is_hard_excluded(record)
    )
    money_ready_source_ads, mechanism_ready_source_ads = source_analysis_ready_counts()
    analysis_ready_source_ads = money_ready_source_ads + mechanism_ready_source_ads
    if analysis_ready_source_ads >= source_queue_limit:
        discovery_stop_basis = "analysis_ready_source_queue"
    elif len(calls) >= max_requests and analysis_ready_source_ads == 0:
        discovery_stop_basis = "request_budget_exhausted_no_analysis_ready_source"
    elif analysis_ready_source_ads == 0 and target_relevant_ads > 0:
        discovery_stop_basis = "target_relevance_waiting_for_money_or_mechanism_gate"
    elif len(calls) >= max_requests:
        discovery_stop_basis = "request_budget_exhausted"
    elif target_relevant_ads > 0 and _source_queue_candidate_count(fresh_records, vertical) >= source_queue_limit:
        discovery_stop_basis = "target_relevance_and_queue"
    else:
        discovery_stop_basis = "target_relevance_shortfall"

    stats.update({
        "planned_requests": len(calls),
        "executed_requests": len(calls),
        "calls": calls,
        "total_credits_reported": total_credits,
        "unique_ads": len(records),
        "raw_unique_ads": len(records),
        "max_ads": max_ads,
        "source_queue_size": source_queue_limit,
        "request_budget": max_requests,
        "shortfall": max(0, max_ads - len(fresh_records)),
        "fresh_unique_ads": len(fresh_records),
        "seen_before_ads": len(records) - len(fresh_records),
        "source_ready_ads": _discovery_ready_count(records, vertical),
        "source_queue_eligible_ads": _source_queue_candidate_count(fresh_records, vertical),
        "money_ready_source_ads": money_ready_source_ads,
        "mechanism_ready_source_ads": mechanism_ready_source_ads,
        "analysis_ready_source_ads": analysis_ready_source_ads,
        "target_relevant_ads": target_relevant_ads,
        "target_relevant_shortfall": max(0, source_queue_limit - target_relevant_ads),
        "discovery_stop_basis": discovery_stop_basis,
        "performance_lane_reserved": bool(supply_search and max_requests > min(max(0, int(keyword_boards)), max_requests)),
        "qualified_source_shortfall": max(0, max_ads - _discovery_ready_count(fresh_records, vertical)),
        "strict_relevant_ads": len(strict_pool),
        "expanded_relevant_ads": len(expanded_pool),
        "unverified_ads": len(unverified_pool),
        "mechanism_hunt_ads": len(mechanism_pool),
        "raw_verification_candidates": len(raw_verify_pool),
        "raw_source_queue_candidates": len(raw_source_queue),
        "relevant_unique_ads": len(direct_pool),
        "direct_relevant_ads": len(direct_pool),
        "final_ranked_ads": len(selected),
        "duplicate_source_records_removed": duplicate_source_records_removed,
        "relevance_shortfall": max(0, max_ads - len(direct_pool)),
        "raw_irrelevant_ads": len(excluded_pool),
        "pre_visual_verification_hard_excluded_ads": len(pre_visual_verification_excluded),
        "pre_visual_verification_hard_excluded": pre_visual_verification_excluded,
        "seen_filtered_ads": sum(1 for r in records if r.get("seen_before")),
        "seen_reused_ads": sum(1 for x in selected if (x.get("record") or {}).get("seen_before")),
        "selected_scope_counts": {
            "STRICT": sum(1 for x in selected if x["record"].get("discovery_scope") == "STRICT"),
            "EXPANDED": sum(1 for x in selected if x["record"].get("discovery_scope") == "EXPANDED"),
            "UNVERIFIED_VERIFY": sum(1 for x in selected if x["record"].get("discovery_scope") == "UNVERIFIED_VERIFY"),
            "MECHANISM_HUNT": sum(1 for x in selected if x["record"].get("discovery_scope") == "MECHANISM_HUNT"),
        },
        "discovery_mode": mode,
        "fallback_path": fallback_path,
        "discovery_strategy": {
            "supply_search_enabled": bool(supply_search),
            "supply_search_budget": min(max(0, int(keyword_boards)), max_requests) if supply_search else 0,
            "performance_budget": max(0, max_requests - (min(max(0, int(keyword_boards)), max_requests) if supply_search else 0)) if supply_search else initial_broad_budget,
            "initial_broad_budget": (max(0, max_requests - min(max(0, int(keyword_boards)), max_requests) - (1 if min(max(0, int(keyword_boards)), max_requests) and max_requests - min(max(0, int(keyword_boards)), max_requests) > 0 else 0)) if supply_search else initial_broad_budget),
            "target_keyword_budget": (min(max(0, int(keyword_boards)), max_requests) if supply_search else target_budget),
            "fallback_broad_budget": (0 if supply_search else fallback_reserve),
            "keyword_boards": max(0, int(keyword_boards)),
            "source_queue_size": source_queue_limit,
            "supply_lane": "tiktok_search_sponsored" if supply_search else "ads_top",
            "supply_queries_planned": supply_queries if supply_search and mode != "mechanism-hunt" else [],
            "supply_queries_executed": [c.get("keyword", "") for c in calls if c.get("source") == "sponsored_search"],
            "query_failover_policy": "target_seeds_until_money_or_mechanism_ready -> Ads Top only after source gate",
        },
        "search_tags_file": str(search_tags_path(vertical, search_tags_dir)),
        "search_tags_available": load_search_tags(vertical, search_tags_dir),
        "search_tags_used": [c.get("keyword", "") for c in calls if c.get("keyword")],
    })
    client.last_discovery_stats = stats
    return selected


def save_discovery_json(
    path: str | Path,
    ranked: list[dict[str, Any]],
    request: dict[str, Any],
    stats: dict[str, Any] | None = None,
    *,
    remember_ids: bool = True,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": request,
        "stats": stats or {},
        "results": [
            {
                "record": x["record"],
                "decision": {
                    "final_score": x["decision"].final_score,
                    "performance_score": x["decision"].performance_score,
                    "commercial_score": x["decision"].commercial_score,
                    "funnel_fit_score": x["decision"].funnel_fit_score,
                    "evidence_score": x["decision"].evidence_score,
                    "longevity_score": x["decision"].longevity_score,
                    "duration_fit_score": x["decision"].duration_fit_score,
                    "penalties": x["decision"].penalties,
                    "components": x["decision"].components,
                    "action": x["decision"].action,
                    "tags": x["decision"].tags,
                    "explanation": x["decision"].explanation,
                },
            }
            for x in ranked
        ],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if remember_ids:
        try:
            remember_seen_ad_ids(ranked, request)
        except Exception:
            pass
    return p
