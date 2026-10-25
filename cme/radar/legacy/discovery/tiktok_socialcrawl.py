from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any

import requests


class SocialCrawlError(RuntimeError):
    pass


class SocialCrawlTikTokTopAds:
    """Adapter for SocialCrawl's TikTok Creative Center Top Ads leaderboard."""

    BASE_URL = "https://www.socialcrawl.dev/v1/tiktok/ads/top"
    SEARCH_URL = "https://www.socialcrawl.dev/v1/tiktok/search"
    ORDER_BY = ("for_you", "ctr", "cvr", "play_2s_rate", "play_6s_rate", "impression", "like")

    def __init__(self, api_key: str | None = None, timeout: float = 45.0):
        self.api_key = api_key or os.getenv("SOCIALCRAWL_API_KEY", "")
        self.timeout = timeout
        self.last_discovery_stats: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise SocialCrawlError("SOCIALCRAWL_API_KEY is not configured")
        idem = str(uuid.uuid4())
        r = requests.get(
            url,
            params=params,
            headers={"x-api-key": self.api_key, "Idempotency-Key": idem},
            timeout=self.timeout,
        )
        try:
            data = r.json()
        except ValueError as exc:
            raise SocialCrawlError(f"SocialCrawl returned non-JSON HTTP {r.status_code}") from exc
        if not r.ok or not data.get("success", False):
            err = data.get("error", {}) or {}
            etype = err.get("type", "HTTP_ERROR")
            msg = err.get("message", f"HTTP {r.status_code}")
            raise SocialCrawlError(f"{etype}: {msg}")
        return data

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._request_json(self.BASE_URL, params)

    @staticmethod
    def _unwrap_item(item: dict[str, Any]) -> dict[str, Any]:
        post = item.get("post", item)
        ext = post.get("ext", {}) or {}
        ad = ext.get("ad", {}) or {}
        content = post.get("content", {}) or {}
        engagement = post.get("engagement", {}) or {}
        computed = item.get("computed", {}) or {}
        video = post.get("video", {}) or {}

        media_urls = content.get("media_urls")
        if isinstance(media_urls, list):
            video_url = media_urls[0] if media_urls else ""
        else:
            video_url = media_urls or ""
        video_url = video_url or ad.get("video_url") or video.get("url") or ""
        video_url_540p = ad.get("video_url_540p") or ""
        video_url_360p = ad.get("video_url_360p") or ""

        author = post.get("author", {}) or {}
        # IMPORTANT: only treat fields explicitly named in seconds as duration.
        # A generic `duration` field is ambiguous and may not be in seconds, so it
        # must never silently become a 0.8s / 2.3s fake duration for our hard gate.
        duration_sources = [
            ("post.content.duration_seconds", content.get("duration_seconds")),
            ("post.content.durationSeconds", content.get("durationSeconds")),
            ("post.video.duration_seconds", video.get("duration_seconds")),
            ("post.video.durationSeconds", video.get("durationSeconds")),
            ("post.ext.duration_seconds", ext.get("duration_seconds")),
            ("post.ext.durationSeconds", ext.get("durationSeconds")),
            ("post.ext.ad.duration_seconds", ad.get("duration_seconds")),
            ("post.ext.ad.durationSeconds", ad.get("durationSeconds")),
            ("post.duration_seconds", post.get("duration_seconds")),
            ("post.durationSeconds", post.get("durationSeconds")),
            ("item.duration_seconds", item.get("duration_seconds")),
            ("item.durationSeconds", item.get("durationSeconds")),
        ]
        duration_seconds = None
        duration_source = "unknown"
        duration_raw_seconds = None
        for source_name, value in duration_sources:
            if value in (None, "", 0, "0", 0.0):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            # SocialCrawl documents this field in seconds. Values below 1 second
            # are not credible video durations for our use-case and have appeared
            # in live responses (e.g. 0.014s). Keep them for audit, but do not
            # allow them to influence the short-video policy or score.
            if parsed < 1.0:
                duration_raw_seconds = parsed
                duration_source = f"{source_name}:invalid_lt_1s"
                duration_seconds = None
                break
            if parsed > 0:
                duration_seconds = parsed
                duration_source = source_name
                break

        return {
            "ad_id": str(post.get("id") or ad.get("id") or ""),
            "video_id": str(video.get("id") or ""),
            "source_url": post.get("url") or ad.get("url") or "",
            "video_url": video_url,
            "video_url_540p": video_url_540p,
            "video_url_360p": video_url_360p,
            "cover_url": content.get("thumbnail_url") or ad.get("cover_url") or "",
            "brand": author.get("username") or ad.get("brand_name") or "UNKNOWN",
            "brand_name": author.get("display_name") or ad.get("brand_name") or "",
            "title": content.get("text") or ad.get("title") or "",
            "landing_page": ad.get("landing_page") or ad.get("landing_page_name") or "",
            "landing_page_url": ad.get("landing_page_url") or ad.get("landing_page_link") or "",
            "duration_seconds": duration_seconds,
            "duration_source": duration_source,
            "duration_raw_seconds": duration_raw_seconds,
            "views": engagement.get("views"),
            "likes": engagement.get("likes"),
            "comments": engagement.get("comments"),
            "shares": engagement.get("shares"),
            "estimated_reach": computed.get("estimated_reach"),
            "engagement_rate": computed.get("engagement_rate"),
            "language": computed.get("language"),
            "ctr": ad.get("ctr"),
            "cvr": ad.get("cvr"),
            "rank": ad.get("rank"),
            "cost_score": ad.get("cost_score"),
            "like_tier": ad.get("like_tier"),
            "objective": ad.get("objective"),
            "industry": ad.get("industry"),
            "ad_format": ad.get("ad_format") or post.get("ad_format"),
            "published_at": None,  # Ads Top does not publish a reliable publish date.
            "raw": item,
        }

    def fetch_sponsored_search(
        self,
        *,
        query: str,
        region: str = "US",
        date_posted: str = "this-month",
        sort_by: str = "most-liked",
        relevance: str = "filter",
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Fetch a broad paid-video supply lane from TikTok Search.

        This is intentionally separate from Ads Top: Search is a supply/discovery
        surface and Ads Top is a performance leaderboard. Search rows are labeled
        sponsored by SocialCrawl and carry normal TikTok video media/engagement.
        ``region`` routes the request; it is not treated as a hard market filter.
        """
        q = str(query or "").strip()
        if not q:
            return {"items": [], "credits_used": 0, "request_id": "", "params": {}}
        if date_posted not in {"yesterday", "this-week", "this-month", "last-3-months", "last-6-months", "all-time"}:
            date_posted = "this-month"
        params: dict[str, Any] = {
            "query": q,
            "region": str(region or "US").upper(),
            "date_posted": date_posted,
            "sort_by": sort_by if sort_by in {"relevance", "most-liked", "date-posted"} else "most-liked",
            "label": "sponsored",
            "relevance": relevance if relevance in {"filter", "score"} else "filter",
        }
        # Omitting limit keeps the cheap paged search page (about 30 rows / 1 credit).
        if limit is not None:
            params["limit"] = max(1, min(120, int(limit)))
        payload = self._request_json(self.SEARCH_URL, params)
        data = payload.get("data") or {}
        raw_items = data.get("items") or []
        items = [self._unwrap_item(x) for x in raw_items if isinstance(x, dict)]
        for item in items:
            item["discovery_source"] = "tiktok_sponsored_search"
            item["discovery_query"] = q
            item["discovery_order_by"] = "sponsored_search"
            item["discovery_country"] = str(region or "US").upper()
            item["discovery_period"] = date_posted
        return {
            "items": items,
            "credits_used": payload.get("credits_used"),
            "credits_remaining": payload.get("credits_remaining"),
            "request_id": payload.get("request_id"),
            "params": params,
            "walk": data.get("walk") or {},
        }

    def fetch_board(
        self,
        *,
        country: str = "US",
        period: int = 30,
        order_by: str = "for_you",
        keyword: str | None = None,
        industry: str | None = None,
        ad_language: str | None = "en",
        like_tier: int | None = None,
        ad_format: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        if order_by not in self.ORDER_BY:
            raise ValueError(f"Unsupported order_by={order_by}; choose one of {self.ORDER_BY}")
        if period not in {7, 30, 180}:
            raise ValueError("period must be 7, 30 or 180")
        params: dict[str, Any] = {
            "country": country.upper(),
            "period": period,
            "order_by": order_by,
            "limit": max(1, min(100, int(limit))),
        }
        if keyword:
            params["keyword"] = keyword
        if industry:
            params["industry"] = industry
        if ad_language:
            params["ad_language"] = ad_language
        if like_tier is not None:
            params["like_tier"] = like_tier
        if ad_format:
            params["ad_format"] = ad_format

        payload = self._get(params)
        data = payload.get("data") or {}
        raw_items = data.get("items") or data.get("commercial_contents") or []
        items = [self._unwrap_item(x) for x in raw_items if isinstance(x, dict)]
        for item in items:
            item["discovery_query"] = keyword or ""
            item["discovery_order_by"] = order_by
            item["discovery_country"] = country.upper()
            item["discovery_period"] = period
        return {
            "items": items,
            "credits_used": payload.get("credits_used"),
            "credits_remaining": payload.get("credits_remaining"),
            "request_id": payload.get("request_id"),
            "params": params,
        }

    def discover(
        self,
        *,
        country: str,
        period: int,
        keywords: list[str] | None = None,
        order_bys: list[str] | None = None,
        industry: str | None = None,
        ad_language: str = "en",
        like_tier: int | None = None,
        ad_format: str | None = None,
        limit_per_board: int = 10,
        max_ads: int = 20,
        max_requests: int = 5,
        keyword_boards: int = 1,
        sleep_between: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Discover Top Ads using a bounded hybrid plan.

        The first requests sample TikTok's broad performance boards. Optional keyword
        requests are added only after broad boards, so a vague keyword cannot collapse
        discovery to a single advertiser.
        """
        default_order_bys = ["for_you", "ctr", "cvr", "play_6s_rate", "play_2s_rate", "impression", "like"]
        order_bys = [x for x in (order_bys or []) if x in self.ORDER_BY]
        for fallback in default_order_bys:
            if len(order_bys) >= max_requests:
                break
            if fallback not in order_bys:
                order_bys.append(fallback)
        if not order_bys:
            order_bys = default_order_bys[:max_requests]
        keywords = [x.strip() for x in (keywords or []) if x and x.strip()]

        plan: list[tuple[str, str | None]] = []
        # For a first market test, broad Top Ads boards are more useful than vague
        # keyword searches. Use every available broad performance board first.
        if not keywords:
            for order_by in order_bys:
                plan.append((order_by, None))
                if len(plan) >= max_requests:
                    break
        else:
            # Explicit keywords remain supported, but broad boards get priority.
            broad_count = min(max(3, max_requests - 2), len(order_bys), max_requests)
            for order_by in order_bys[:broad_count]:
                plan.append((order_by, None))
            for keyword in keywords:
                if len(plan) >= max_requests:
                    break
                plan.append(("ctr", keyword))

        combined: dict[str, dict[str, Any]] = {}
        calls: list[dict[str, Any]] = []
        total_credits = 0
        stopped_early = False

        for call_no, (order_by, keyword) in enumerate(plan, 1):
            result = self.fetch_board(
                country=country,
                period=period,
                order_by=order_by,
                keyword=keyword,
                industry=industry,
                ad_language=ad_language,
                like_tier=like_tier,
                ad_format=ad_format,
                limit=limit_per_board,
            )
            credits = result.get("credits_used")
            if isinstance(credits, (int, float)):
                total_credits += int(credits)
            before = len(combined)
            new_items = 0
            for item in result["items"]:
                key = (
                    item.get("ad_id")
                    or item.get("video_id")
                    or item.get("source_url")
                    or item.get("video_url")
                )
                if not key:
                    continue
                existing = combined.get(str(key))
                if existing is None:
                    item["discovery_boards"] = [order_by]
                    item["discovery_keywords"] = [keyword] if keyword else []
                    item["discovery_hits"] = 1
                    combined[str(key)] = item
                    new_items += 1
                else:
                    existing["discovery_hits"] = int(existing.get("discovery_hits", 1)) + 1
                    if order_by not in existing.setdefault("discovery_boards", []):
                        existing["discovery_boards"].append(order_by)
                    if keyword and keyword not in existing.setdefault("discovery_keywords", []):
                        existing["discovery_keywords"].append(keyword)
                    for field in (
                        "ctr", "cvr", "rank", "cost_score", "like_tier", "views", "likes",
                        "comments", "shares", "estimated_reach", "duration_seconds", "video_url",
                        "source_url", "industry", "objective",
                    ):
                        if existing.get(field) in (None, "") and item.get(field) not in (None, ""):
                            existing[field] = item[field]
            calls.append({
                "call": call_no,
                "order_by": order_by,
                "keyword": keyword or "",
                "returned": len(result["items"]),
                "new_unique": new_items,
                "unique_after": len(combined),
                "credits_used": credits,
                "request_id": result.get("request_id"),
            })
            if sleep_between:
                time.sleep(sleep_between)

        # Do not stop after the first board. Sort only after all planned boards are merged.
        records = list(combined.values())
        self.last_discovery_stats = {
            "planned_requests": len(plan),
            "executed_requests": len(calls),
            "calls": calls,
            "total_credits_reported": total_credits,
            "unique_ads": len(records),
            "max_ads": max_ads,
            "request_budget": max_requests,
            "shortfall": max(0, max_ads - len(records)),
            "stopped_by_budget": len(calls) >= max_requests and len(records) < max_ads,
        }
        # Return the complete bounded discovery pool. Final `max_ads` is applied only
        # AFTER vertical qualification/ranking, otherwise relevant ads later in the
        # merged boards can be discarded before the vertical gate ever sees them.
        return records
