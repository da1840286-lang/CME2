"""FIND adapter. Runs the old engine's real discovery code and maps its output to a compact record.

Nothing here ranks or filters on its own: ranking, classification and source relevance come from the
vendored legacy modules. Unknown values stay None, never zero.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from .evidence import classify_commercial_evidence
from .attention import attention_absolute_v1, attention_rank, MECHANISM_ATTENTION_FLOOR, MECHANISM_MIN_COVERAGE
from .fit import ADJACENT_MECHANISM, TARGET_PRODUCT, classify_source_role, match_source_to_product_query
from .seen import SeenLedger, build_scope
from .provider_contract import (
    CONTRACT_AWARE_MODE, contract_summary, plan_uses_contract, ordered_target_queries,
)
from .select import VIDEO_DURATION_RANGES, pattern_source_reason, video_duration_filter_reason

DATA_DIR = Path(__file__).resolve().parent / "data"
# Search-only seeds for the live Sponsored Search lane. Keep separate from the
# legacy CTA/tag corpus, which is retained for byte-identical compatibility.
PRODUCT_SEARCH_TAGS_DIR = DATA_DIR / "product_search_tags"
os.environ.setdefault("CME_VERTICALS_DIR", str(DATA_DIR / "verticals"))  # before the legacy import

VERTICAL = "Physical Products"
METRIC_FIELDS = ("views", "likes", "comments", "shares", "ctr", "cvr", "play_2s_rate", "play_6s_rate",
                 "retention_2s", "retention_6s", "retention_curve",
                 "rank", "cost_score", "like_tier", "estimated_reach", "engagement_rate")
PERIODS = (7, 30, 180)
def _video_duration_filter_reason(record: dict, selected_range: str = "all") -> str | None:
    """Compatibility wrapper around the shared FIND/ALL duration contract."""
    return video_duration_filter_reason(record, selected_range)

LEGACY_ERROR = ""
try:
    from .legacy.discover import _ordered_supply_queries, canonical_source_identity_keys, discover_and_rank
    from .legacy.arbitrage import score_tiktok_candidate
    from .legacy.discovery.tiktok_socialcrawl import SocialCrawlError, SocialCrawlTikTokTopAds
    from .socialcrawl_client import RetryingSocialCrawlTikTokTopAds
    from .legacy.verticals import classify_vertical
except Exception as e:  # e.g. the 'requests' package is missing
    LEGACY_ERROR = f"{type(e).__name__}: {e}"
    discover_and_rank = canonical_source_identity_keys = _ordered_supply_queries = SocialCrawlTikTokTopAds = RetryingSocialCrawlTikTokTopAds = classify_vertical = score_tiktok_candidate = None

    class SocialCrawlError(Exception):
        pass


class FindUnavailable(Exception):
    def __init__(self, state: str, detail: str, diagnostics: dict | None = None):
        self.state, self.detail = state, detail
        self.diagnostics = diagnostics or {}
        super().__init__(f"{state}: {detail}")


def _retention_execution_plan(
    requested_order_bys: list[str], include_retention: bool, requested_max_requests: int,
    performance_plan: str = "contract_aware",
) -> tuple[list[str], int, dict]:
    """Keep distinct TikTok board rankings; each row carries both retention points.

    performance_plan="contract_aware" (default): see provider_contract. Ads Top play-rate boards can
    neither be product-targeted nor expose numeric rates, and include=retention cannot be combined with
    a keyword, so executing them cannot yield a product-relevant ad carrying that metric. The whole
    request budget goes to product-targeted Sponsored Search and the requested metric is reported as
    not obtainable. performance_plan="legacy_ads_top" keeps the old plan for explicit probing.
    """
    requested = list(dict.fromkeys(str(value) for value in requested_order_bys if value))
    effective = list(requested)
    effective_max_requests = int(requested_max_requests)
    if plan_uses_contract(performance_plan) and {"play_2s_rate", "play_6s_rate"} & set(requested):
        return [], effective_max_requests, {
            "mode": CONTRACT_AWARE_MODE,
            "requested_performance_order_bys": requested,
            "effective_performance_order_bys": [],
            "requested_max_requests": effective_max_requests,
            "effective_max_requests": effective_max_requests,
            "retention_calls": 0, "targeted_board_calls": 0,
            "retention_requested_but_not_executed": bool(include_retention),
            "requested_metric_status": "NOT_OBTAINABLE_FOR_PRODUCT_QUERY",
            "provider_contract": contract_summary(),
            "rationale": (
                "play_2s_rate/play_6s_rate are Ads Top sort orders only: rows carry no numeric rate and "
                "keyword-filtered boards returned off-target ads in every audited run. The budget is spent on "
                "product-targeted Sponsored Search (exact product, product type, generic)."
            ),
        }
    strategy = {
        "mode": "operator_requested_order",
        "requested_performance_order_bys": requested,
        "effective_performance_order_bys": effective,
        "requested_max_requests": effective_max_requests,
        "effective_max_requests": effective_max_requests,
        "note": "Retention points are not the exact play-rate fields used to sort Ads Top.",
    }
    both_retention_boards = {"play_2s_rate", "play_6s_rate"}.issubset(set(requested))
    if include_retention and both_retention_boards:
        # The provider forbids combining include=retention with keyword/industry filters.
        # Keep the two target-filtered playback rankings separate and allocate one additional
        # unfiltered call for retention capture; do not silently replace the target boards.
        if effective_max_requests < 5:
            raise ValueError(
                "include_retention with playback_both needs find_requests >= 5 "
                "for two Sponsored Search queries, both targeted Ads Top boards, and one retention call."
            )
        target_boards = [name for name in requested if name in {"play_2s_rate", "play_6s_rate"}]
        strategy = {
            "mode": "targeted_playback_boards_plus_retention",
            "requested_performance_order_bys": requested,
            "effective_performance_order_bys": effective,
            "requested_max_requests": int(requested_max_requests),
            "effective_max_requests": effective_max_requests,
            "targeted_board_order_bys": target_boards,
            "retention_request_order_by": target_boards[0] if target_boards else "play_6s_rate",
            "retention_calls": 1,
            "targeted_board_calls_are_keyword_filtered": True,
            "retention_call_is_unfiltered": True,
            "rationale": (
                "Complete up to two target Sponsored Search queries, one keyword-filtered Ads Top request for each requested "
                "playback ranking, and one unfiltered include=retention request. The bounded request plan does not substitute "
                "unrequested boards; the retention response is merged only by source identity; product relevance and "
                "source-quality gates remain unchanged."
            ),
            "interpretation": (
                "Retention points are not the exact play_2s_rate/play_6s_rate values used to sort Ads Top "
                "and are not proof of conversion, sales, or profit."
            ),
        }
    return effective, effective_max_requests, strategy


def _target_board_keyword(offer: dict, fallback: str = "") -> str:
    """Prefer distinctive target brand/model tokens over generic product-type words."""
    offer = offer if isinstance(offer, dict) else {}
    target = " ".join(str(offer.get("target_offer") or offer.get("name") or "").split())
    product_type = " ".join(str(offer.get("product_type") or offer.get("product_category") or "").split())
    product_tokens = {token.casefold() for token in re.findall(r"[a-z0-9]+", product_type, flags=re.IGNORECASE)}
    target_tokens = re.findall(r"[a-z0-9]+", target, flags=re.IGNORECASE)
    specific_tokens = [token for token in target_tokens if token.casefold() not in product_tokens]
    if specific_tokens:
        return " ".join(specific_tokens)
    return target or product_type or " ".join(str(fallback or "").split())


def _retention_enrichment_policy(calls: list[dict]) -> dict:
    """Only buy unfiltered retention data if targeted boards found target-product ads."""
    target_calls = [
        call for call in (calls or [])
        if isinstance(call, dict)
        and call.get("candidate_lane")
        and call.get("source") == "ads_top"
        and call.get("order_by") in {"play_2s_rate", "play_6s_rate"}
        and bool(str(call.get("keyword") or "").strip())
        and not call.get("retention_requested")
    ]
    if not target_calls:
        return {
            "should_run": False,
            "reason": "NO_TARGETED_PLAYBACK_BOARD_CALLS",
            "targeted_board_calls": 0,
            "target_role_match_count": 0,
        }
    matched = 0
    for call in target_calls:
        try:
            matched += max(0, int(call.get("target_role_match_count", call.get("eligible_rows_returned", 0)) or 0))
        except (TypeError, ValueError):
            continue
    if matched == 0:
        return {
            "should_run": False,
            "reason": "NO_TARGET_PRODUCT_ROWS_ON_TARGETED_PLAYBACK_BOARDS",
            "targeted_board_calls": len(target_calls),
            "target_role_match_count": 0,
        }
    return {
        "should_run": True,
        "reason": "TARGET_PRODUCT_ROWS_FOUND",
        "targeted_board_calls": len(target_calls),
        "target_role_match_count": matched,
    }


class _DiscoveryEvidenceTracker:
    """Observe provider rows without changing the vendored discovery contract or its gates."""

    _LOCAL_FIELDS = frozenset({
        "_client", "calls", "values_by_identity", "positions_by_identity",
        "provenance_by_identity", "values_by_ad_id", "positions_by_ad_id",
        "retention_order_by", "offer_context", "enforce_product_role",
        "product_role_gate", "candidate_provider_ids",
        "unfiltered_retention_identity_tokens", "unfiltered_retention_ad_ids",
        "multi_sponsored_search_queries", "restricted_order_bys", "_sponsored_search_batch_consumed",
    })
    PERFORMANCE_FIELDS = (
        "ctr", "cvr", "play_2s_rate", "play_6s_rate", "views", "likes",
        "comments", "shares", "estimated_reach", "rank",
    )
    RETENTION_FIELDS = ("retention_2s", "retention_6s", "retention_curve")

    def __init__(
        self, client, *, offer=None, multi_sponsored_search_queries=None, restrict_order_bys=None,
    ):
        offer = offer if isinstance(offer, dict) else {}
        context = {
            "target_offer": str(offer.get("target_offer") or offer.get("name") or "").strip(),
            "product_type": str(offer.get("product_type") or "").strip(),
            "product_category": str(offer.get("product_category") or "").strip(),
        }
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "multi_sponsored_search_queries", list(dict.fromkeys(
            str(query).strip() for query in (multi_sponsored_search_queries or []) if str(query).strip()
        )))
        object.__setattr__(self, "restricted_order_bys", list(dict.fromkeys(
            str(value).strip() for value in (restrict_order_bys or []) if str(value).strip()
        )))
        object.__setattr__(self, "_sponsored_search_batch_consumed", False)
        object.__setattr__(self, "offer_context", context)
        object.__setattr__(self, "enforce_product_role", bool(
            context["product_type"] or context["product_category"] or context["target_offer"]))
        object.__setattr__(self, "calls", [])
        object.__setattr__(self, "values_by_identity", {})
        object.__setattr__(self, "positions_by_identity", {})
        object.__setattr__(self, "provenance_by_identity", {})
        object.__setattr__(self, "values_by_ad_id", {})
        object.__setattr__(self, "positions_by_ad_id", {})
        object.__setattr__(self, "retention_order_by", None)
        object.__setattr__(self, "candidate_provider_ids", set())
        object.__setattr__(self, "unfiltered_retention_identity_tokens", set())
        object.__setattr__(self, "unfiltered_retention_ad_ids", set())
        object.__setattr__(self, "product_role_gate", {
            "applied": bool(context["product_type"] or context["product_category"] or context["target_offer"]),
            "raw_rows": 0, "accepted_rows": 0, "rejected_rows": 0,
            "rejected_by_reason": {}, "rejected_examples": [],
        })

    @property
    def ORDER_BY(self):
        """Limit the legacy scheduler to requested performance boards for a dual-board run."""
        available = set(getattr(self._client, "ORDER_BY", set()) or set())
        restricted = set(self.restricted_order_bys or [])
        if not restricted:
            return available
        # The legacy loop builds a general order list before its performance list.
        # Keep its default token valid, but never advertise unrequested metrics.
        if "for_you" in available:
            restricted.add("for_you")
        return restricted

    def __getattr__(self, name):
        return getattr(self._client, name)

    def __setattr__(self, name, value):
        if name in self._LOCAL_FIELDS:
            object.__setattr__(self, name, value)
        else:
            setattr(self._client, name, value)

    @staticmethod
    def _identity_tokens(record):
        if not isinstance(record, dict):
            return set()
        tokens = set()
        try:
            tokens.update(str(value) for value in (canonical_source_identity_keys(record) or set()) if value)
        except Exception:
            pass
        for field in ("ad_id", "video_id"):
            value = str(record.get(field) or "").strip()
            if value:
                tokens.add("record:" + field + ":" + value)
        return tokens

    @classmethod
    def _row_values(cls, row):
        source_signal = row.get("source_signal")
        source_signal = source_signal if isinstance(source_signal, dict) else {}
        values = {}
        for field in cls.PERFORMANCE_FIELDS + cls.RETENTION_FIELDS:
            value = row.get(field)
            if value in (None, ""):
                value = source_signal.get(field)
            if value not in (None, "", []):
                values[field] = value
        invalid_points = row.get("retention_curve_invalid_points")
        if invalid_points not in (None, ""):
            values["retention_curve_invalid_points"] = invalid_points
        return values

    @staticmethod
    def _nonempty_count(rows, field):
        return sum(1 for row in rows if isinstance(row, dict) and row.get(field) not in (None, "", []))

    def _remember(
        self, result, *, source, order_by="", keyword="", requested_keyword=None,
        retention_requested=False, unfiltered_retention=False, candidate_lane=True,
    ):
        """Record raw provider evidence, filtering product discovery before the legacy selector."""
        result = result if isinstance(result, dict) else {}
        raw_rows = [dict(row) for row in (result.get("items") or []) if isinstance(row, dict)]
        rows_for_legacy, eligible_rows, rejected, role_counts = [], 0, [], {}
        for position, row in enumerate(raw_rows, start=1):
            tokens = self._identity_tokens(row)
            ad_id = str(row.get("ad_id") or row.get("video_id") or "").strip()
            if candidate_lane:
                identity = ad_id or (sorted(tokens)[0] if tokens else "")
                if identity:
                    self.candidate_provider_ids.add(identity)

            values = self._row_values(row)
            if not retention_requested:
                for field in self.RETENTION_FIELDS + ("retention_curve_invalid_points", "retention_provenance"):
                    values.pop(field, None)
                    row.pop(field, None)
            for token in tokens:
                bucket = self.values_by_identity.setdefault(token, {})
                for field, value in values.items():
                    if unfiltered_retention and field in self.RETENTION_FIELDS + ("retention_curve_invalid_points",):
                        bucket[field] = value
                    else:
                        bucket.setdefault(field, value)
                positions = self.positions_by_identity.setdefault(token, {})
                if source == "ads_top":
                    if retention_requested:
                        position_key = str(order_by) + "|retention"
                    elif keyword:
                        position_key = str(order_by) + "|keyword:" + str(keyword)
                    else:
                        position_key = str(order_by)
                    positions.setdefault(position_key, position)
                provenance = row.get("retention_provenance")
                if retention_requested and isinstance(provenance, dict) and provenance:
                    provenance_bucket = self.provenance_by_identity.setdefault(token, {})
                    for field, value in provenance.items():
                        if value:
                            if unfiltered_retention:
                                provenance_bucket[str(field)] = value
                            else:
                                provenance_bucket.setdefault(str(field), value)

            if ad_id:
                id_values = self.values_by_ad_id.setdefault(ad_id, {})
                for field, value in values.items():
                    if unfiltered_retention and field in self.RETENTION_FIELDS + ("retention_curve_invalid_points",):
                        id_values[field] = value
                    else:
                        id_values.setdefault(field, value)
                if source == "ads_top":
                    id_positions = self.positions_by_ad_id.setdefault(ad_id, {})
                    if retention_requested:
                        position_key = str(order_by) + "|retention"
                    elif keyword:
                        position_key = str(order_by) + "|keyword:" + str(keyword)
                    else:
                        position_key = str(order_by)
                    id_positions.setdefault(position_key, position)

            if retention_requested and unfiltered_retention and any(
                row.get(field) not in (None, "", []) for field in self.RETENTION_FIELDS
            ):
                self.unfiltered_retention_identity_tokens.update(tokens)
                if ad_id:
                    self.unfiltered_retention_ad_ids.add(ad_id)
                object.__setattr__(self, "retention_order_by", order_by)

            eligible = True
            if candidate_lane and self.enforce_product_role:
                role = classify_source_role(
                    row,
                    product_type=self.offer_context.get("product_type") or self.offer_context.get("product_category") or "",
                    product_category=self.offer_context.get("product_category") or "",
                    target_offer=self.offer_context.get("target_offer") or "",
                )
                row["source_role_info"] = role
                eligible = (
                    role.get("role") == TARGET_PRODUCT
                    and bool(role.get("promotion_evidence"))
                    and bool(role.get("target_promotion_evidence"))
                )
                if not eligible:
                    reason = str(role.get("role") or "OFF_TOPIC").upper()
                    if reason == TARGET_PRODUCT and not role.get("target_promotion_evidence"):
                        reason = "TARGET_PRODUCT_NOT_PROMOTED"
                    elif reason == TARGET_PRODUCT and not role.get("promotion_evidence"):
                        reason = "NO_INDEPENDENT_PROMOTION_EVIDENCE"
                    role_counts[reason] = role_counts.get(reason, 0) + 1
                    rejected.append({
                        "ad_id": ad_id, "title": str(row.get("title") or "")[:180],
                        "role": role.get("role"), "reason": reason,
                        "source": source, "order_by": order_by, "keyword": keyword or "",
                    })
            if eligible:
                eligible_rows += 1
            # Preserve raw rows for the existing discovery engine and its rejection diagnostics.
            # The final adapter gate removes role-ineligible candidates from the actual shortlist.
            rows_for_legacy.append(row)

        if candidate_lane:
            gate = self.product_role_gate
            gate["raw_rows"] += len(raw_rows)
            gate["accepted_rows"] += eligible_rows
            gate["rejected_rows"] += len(raw_rows) - eligible_rows
            for reason, count in role_counts.items():
                gate["rejected_by_reason"][reason] = gate["rejected_by_reason"].get(reason, 0) + count
            gate["rejected_examples"].extend(rejected[:max(0, 12 - len(gate["rejected_examples"]))])

        call = {
            "call": len(self.calls) + 1, "source": source, "order_by": order_by,
            "keyword": keyword or "",
            "requested_keyword": (
                str(requested_keyword or "")
                if requested_keyword is not None and str(requested_keyword or "") != str(keyword or "")
                else ""
            ),
            "returned": len(raw_rows),
            "provider_rows_returned": len(raw_rows),
            "eligible_rows_returned": eligible_rows if candidate_lane else None,
            "role_rejected_rows": len(raw_rows) - eligible_rows if candidate_lane else 0,
            "role_rejected_by_reason": role_counts,
            "candidate_lane": bool(candidate_lane),
            "credits_used": result.get("credits_used"), "request_id": result.get("request_id"),
            "provider_params": result.get("params") or {},
            "returned_ad_ids": [str(row.get("ad_id") or row.get("video_id") or "") for row in raw_rows[:12]],
            "provider_ad_ids": [str(row.get("ad_id") or row.get("video_id") or "") for row in raw_rows
                                if row.get("ad_id") or row.get("video_id")],
            "playback_metrics_returned": {
                field: self._nonempty_count(raw_rows, field) for field in ("play_2s_rate", "play_6s_rate")
            },
            "retention_requested": bool(retention_requested),
            "unfiltered_retention_request": bool(retention_requested and unfiltered_retention),
            "retention_returned": {
                field: self._nonempty_count(raw_rows, field) for field in self.RETENTION_FIELDS
            } if retention_requested else {},
            "raw_playback_field_diagnostics": result.get("raw_playback_field_diagnostics") or {},
            "provider_response_meta": result.get("provider_response_meta") or {},
        }
        if source == "ads_top" and order_by in {"play_2s_rate", "play_6s_rate"} and keyword and candidate_lane and not retention_requested:
            call["target_role_match_count"] = eligible_rows
            call["target_role_match_status"] = (
                "TARGET_ROLE_MATCH_FOUND" if eligible_rows else "NO_TARGET_ROLE_MATCH_IN_RESPONSE"
            )
            if raw_rows and not eligible_rows:
                call["target_query_relevance_warning"] = (
                    "The targeted playback board returned rows, but none passed the target-product source-role gate."
                )
        self.calls.append(call)
        result = dict(result)
        result["items"] = rows_for_legacy
        return result

    def role_gate_diagnostics(self):
        gate = dict(self.product_role_gate)
        gate["rejected_by_reason"] = dict(self.product_role_gate.get("rejected_by_reason") or {})
        gate["rejected_examples"] = list(self.product_role_gate.get("rejected_examples") or [])
        gate["unique_provider_ads"] = len(self.candidate_provider_ids)
        return gate

    def has_unfiltered_retention(self, record):
        tokens = self._identity_tokens(record)
        if tokens & self.unfiltered_retention_identity_tokens:
            return True
        ad_id = str((record or {}).get("ad_id") or (record or {}).get("video_id") or "").strip()
        return bool(ad_id and ad_id in self.unfiltered_retention_ad_ids)

    def fetch_sponsored_search(self, **kwargs):
        """Fetch the scheduled target queries as one legacy discovery batch.

        The legacy controller stops its Sponsored Search lane as soon as any source
        looks analysis-ready. For the dual-playback/retention plan that is too early:
        final product-role and requested-metric gates have not run yet. Batch the two
        bounded target queries before handing rows into the unchanged legacy ranker.
        """
        queries = list(self.multi_sponsored_search_queries or [])
        if not queries or self._sponsored_search_batch_consumed:
            result = self._client.fetch_sponsored_search(**kwargs)
            return self._remember(
                result, source="sponsored_search", order_by="sponsored_search",
                keyword=str(kwargs.get("query") or ""),
            )

        self._sponsored_search_batch_consumed = True
        combined_items, responses = [], []
        total_credits = 0
        for query in queries:
            provider_kwargs = dict(kwargs)
            provider_kwargs["query"] = query
            response = self._client.fetch_sponsored_search(**provider_kwargs)
            response = response if isinstance(response, dict) else {}
            remembered = self._remember(
                response, source="sponsored_search", order_by="sponsored_search", keyword=query,
            )
            for item in remembered.get("items") or []:
                if isinstance(item, dict):
                    item.setdefault("discovery_query", query)
                    combined_items.append(item)
            responses.append(response)
            credits = response.get("credits_used")
            if isinstance(credits, (int, float)):
                total_credits += int(credits)

        return {
            "items": combined_items,
            "credits_used": total_credits,
            "request_id": next((row.get("request_id") for row in reversed(responses) if row.get("request_id")), None),
            "params": {
                "batched_queries": queries,
                "provider_requests": len(responses),
                "request_ids": [row.get("request_id") for row in responses if row.get("request_id")],
            },
            "raw_playback_field_diagnostics": {},
        }

    def fetch_board(self, **kwargs):
        requested_keyword = str(kwargs.get("keyword") or "")
        provider_kwargs = dict(kwargs)
        order_by = str(kwargs.get("order_by") or "")
        retention_requested = bool(kwargs.get("include_retention"))

        # The legacy scheduler feeds its first generic supply term to both playback
        # boards. Prefer the actual target brand/model so the provider filters the
        # requested product, rather than silently repeating a broad category query.
        if (
            order_by in {"play_2s_rate", "play_6s_rate"}
            and order_by in set(self.restricted_order_bys or [])
            and not retention_requested
        ):
            target_keyword = _target_board_keyword(self.offer_context, requested_keyword)
            if target_keyword:
                provider_kwargs["keyword"] = target_keyword

        effective_keyword = str(provider_kwargs.get("keyword") or "")
        result = self._client.fetch_board(**provider_kwargs)
        unfiltered_retention = retention_requested and not provider_kwargs.get("keyword") and not provider_kwargs.get("industry")
        return self._remember(
            result, source="ads_top", order_by=order_by,
            keyword=effective_keyword, requested_keyword=requested_keyword,
            retention_requested=retention_requested, unfiltered_retention=unfiltered_retention,
        )

    def fetch_retention_enrichment(self, **kwargs):
        """Fetch retention separately; this sample enriches identities and never enters the shortlist."""
        result = self._client.fetch_board(**kwargs)
        retention_requested = bool(kwargs.get("include_retention"))
        unfiltered_retention = retention_requested and not kwargs.get("keyword") and not kwargs.get("industry")
        return self._remember(
            result, source="ads_top", order_by=str(kwargs.get("order_by") or ""),
            keyword=str(kwargs.get("keyword") or ""),
            retention_requested=retention_requested, unfiltered_retention=unfiltered_retention,
            candidate_lane=False,
        )

    def evidence_for(self, record):
        values, positions, provenance = {}, {}, {}
        for token in self._identity_tokens(record):
            for field, value in self.values_by_identity.get(token, {}).items():
                values.setdefault(field, value)
            for key, value in self.positions_by_identity.get(token, {}).items():
                positions.setdefault(key, value)
            for field, value in self.provenance_by_identity.get(token, {}).items():
                provenance.setdefault(field, value)
        record_id = str((record or {}).get("ad_id") or (record or {}).get("video_id") or "").strip()
        if record_id:
            for field, value in self.values_by_ad_id.get(record_id, {}).items():
                values.setdefault(field, value)
            for key, value in self.positions_by_ad_id.get(record_id, {}).items():
                positions.setdefault(key, value)
        return values, positions, provenance


def _merge_discovery_evidence(ranked, tracker):
    """Attach observed metrics, board positions and identity-matched retention to shortlisted rows."""
    for item in ranked or []:
        if not isinstance(item, dict):
            continue
        record = item.get("record")
        if not isinstance(record, dict):
            continue
        values, positions, provenance = tracker.evidence_for(record)
        unfiltered_retention = tracker.has_unfiltered_retention(record)
        for field, value in values.items():
            if unfiltered_retention and field in tracker.RETENTION_FIELDS + ("retention_curve_invalid_points",):
                record[field] = value
            elif record.get(field) in (None, ""):
                record[field] = value
        current_positions = record.get("discovery_board_positions")
        current_positions = dict(current_positions) if isinstance(current_positions, dict) else {}
        for key, value in positions.items():
            current_positions.setdefault(key, value)
        if current_positions:
            record["discovery_board_positions"] = current_positions
        current_provenance = record.get("retention_provenance")
        current_provenance = dict(current_provenance) if isinstance(current_provenance, dict) else {}
        for field, value in provenance.items():
            if not value:
                continue
            if values.get(field) not in (None, "", []):
                # A validated value from this retention request supersedes stale/invalid provenance.
                current_provenance[field] = value
            else:
                current_provenance.setdefault(field, value)
        if current_provenance:
            record["retention_provenance"] = current_provenance
        if any(record.get(field) not in (None, "", []) for field in tracker.RETENTION_FIELDS):
            if unfiltered_retention:
                record["retention_collection_order_by"] = tracker.retention_order_by
                record["retention_collected_via_unfiltered_request"] = True
            else:
                record.pop("retention_collection_order_by", None)
                record["retention_collected_via_unfiltered_request"] = False


def _enrich_selection_diagnostics(stats, tracker):
    diagnostic = stats.get("source_selection_diagnostics")
    if not isinstance(diagnostic, dict):
        return
    for row in diagnostic.get("examples") or []:
        if not isinstance(row, dict):
            continue
        ad_id = str(row.get("ad_id") or "").strip()
        if not ad_id:
            continue
        values = tracker.values_by_ad_id.get(ad_id) or {}
        positions = tracker.positions_by_ad_id.get(ad_id) or {}
        current_values = row.get("performance_values")
        current_values = dict(current_values) if isinstance(current_values, dict) else {}
        for field, value in values.items():
            if field in tracker.PERFORMANCE_FIELDS:
                current_values.setdefault(field, value)
        row["performance_values"] = current_values
        current_positions = row.get("discovery_board_positions")
        current_positions = dict(current_positions) if isinstance(current_positions, dict) else {}
        for key, value in positions.items():
            current_positions.setdefault(key, value)
        row["discovery_board_positions"] = current_positions


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
    retention_boards = [board for board in boards if board in {"play_2s_rate", "play_6s_rate"}]
    retention_collection_source = ["include=retention"] if rec.get("retention_collected_via_unfiltered_request") else []
    retention_collection_order_by = rec.get("retention_collection_order_by")
    retention_provenance = rec.get("retention_provenance")
    retention_provenance = retention_provenance if isinstance(retention_provenance, dict) else {}
    retention_metric_names = {"retention_2s", "retention_6s", "retention_curve"}
    for name in METRIC_FIELDS:
        value = rec.get(name)
        if value is not None and value != "":
            field_source = "socialcrawl:tiktok_ads_top" if name in retention_metric_names else src
            field_boards = (
                retention_collection_source or retention_boards or boards
            ) if name in retention_metric_names else boards
            metrics[name] = {"value": value, "source": field_source, "boards": field_boards, "collected_at": collected_at}
            if name in retention_provenance:
                metrics[name]["provenance"] = retention_provenance[name]
    retention_curve = rec.get("retention_curve")
    retention_curve_points = sum(
        1 for point in retention_curve
        if isinstance(point, (int, float)) and not isinstance(point, bool)
    ) if isinstance(retention_curve, list) else 0
    has_retention = any(rec.get(field) not in (None, "", []) for field in ("retention_2s", "retention_6s", "retention_curve"))
    retention_evidence = {
        "status": "AVAILABLE" if has_retention else "NOT_AVAILABLE",
        "source": "socialcrawl:tiktok_ads_top" if has_retention else None,
        "boards": retention_boards,
        "collection_order_by": retention_collection_order_by,
        "collected_via_unfiltered_request": bool(rec.get("retention_collected_via_unfiltered_request")),
        "retention_2s": rec.get("retention_2s"),
        "retention_6s": rec.get("retention_6s"),
        "curve_point_count": retention_curve_points,
        "curve_invalid_or_missing_points": rec.get("retention_curve_invalid_points"),
        "provenance": dict(retention_provenance),
        "is_exact_play_rate": False,
        "commercial_result_confirmed": False,
        "interpretation": (
            "TikTok audience-retention points; not the exact play-rate values used to sort Ads Top "
            "and not evidence of conversion, sales, or profit."
        ),
    }
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
           "hits": rec.get("discovery_hits"), "relevance_reason": rel.get("reason"),
           "discovery_board_positions": rec.get("discovery_board_positions") or {},
           "retention_evidence": retention_evidence}
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
        "retention_evidence": retention_evidence,
        "source_check": {"status": rel.get("status") or "UNKNOWN", "classification": rel.get("classification"),
                         "score": rel.get("score"), "excluded": rel.get("excluded"), "reason": rel.get("reason"),
                         "product_relevance": rec.get("product_relevance") or {},
                         "scope": rec.get("discovery_scope")},
        "source_score": score,
        "product_relevance": rec.get("product_relevance") or {},
        "query_product_match": rec.get("query_product_match") or {},
        "performance_evidence": rec.get("performance_evidence") or {},
        "commercial_evidence": classify_commercial_evidence(
            rec, score, rec.get("performance_evidence") or {}
        ),
        "shortlist_eligibility": rec.get("shortlist_eligibility") or {},
        "why": {**why, "product_relevance": rec.get("product_relevance") or {},
                "performance_evidence": rec.get("performance_evidence") or {},
                "selection_reason": rec.get("auto_selection_reason") or ""},
        "discovered_at": collected_at,
        "request": {k: request.get(k) for k in ("country", "period", "order_bys", "keywords", "vertical", "selection_mode", "video_duration_range", "minimum_attention_coverage", "attention_floor")},
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
    eligibility_reason = pattern_source_reason(c)
    source_gate_eligible = not bool(eligibility_reason)
    c["analysis_eligibility"] = {
        "contract": "PATTERN_SOURCE_V1",
        "eligible": source_gate_eligible,
        "reason": eligibility_reason or None,
    }
    c["stage"] = "PATTERN_CANDIDATE" if source_gate_eligible else "SOURCE_REJECTED"
    c["source_gate_eligible"] = source_gate_eligible
    # Deprecated compatibility field: pattern-source admission only, never Target-Fit approval.
    c["transfer_eligible"] = source_gate_eligible
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
        # Keep one explicit use-case exception: "pet hair" is a contiguous qualifier for a vacuum,
        # although both words occur in the product-query vocabulary as part of "pet hair remover".
        # Any other extra product anchor (e.g. "blender" in "blender car vacuum") is a second product,
        # even when the target's single anchor makes the Jaccard similarity look high.
        extras = anchors - target_specific
        tokens = re.findall(r"[a-z0-9]+", str(keyword or "").casefold())
        extra_positions = [i for i, token in enumerate(tokens) if token in extras]
        contiguous = extra_positions == list(range(extra_positions[0], extra_positions[0] + len(extra_positions))) \
            if extra_positions else True
        extra_phrase = [tokens[i] for i in extra_positions]
        use_case_exception = extras == {"pet", "hair"} and extra_phrase == ["pet", "hair"] and contiguous
        contains_offer = bool(target_specific) and overlap == target_specific and (not extras or use_case_exception)
        # A complete target match is accepted only with no other product anchor or the explicit use-case phrase.
        # Partial matches retain the previous similarity threshold.
        compatible = contains_offer if overlap == target_specific else similarity >= 0.35
        checked.append({"query": keyword, "anchors": sorted(anchors), "similarity": round(similarity, 3),
                        "contains_offer_anchors": contains_offer, "compatible": compatible})

    if not checked or any(row["compatible"] for row in checked):
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
    """Distinguish missing inventory, irrelevant sources, and missing requested performance evidence."""
    calls = stats.get("calls") or []
    strategy = stats.get("discovery_strategy") or {}
    raw_returned = int(stats.get("provider_rows_returned") or 0)
    if "provider_rows_returned" not in stats:
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
    performance_gate = stats.get("requested_performance_evidence_gate") or {}
    role_gate = stats.get("product_source_role_gate") or {}
    if candidate_count:
        outcome = "CANDIDATES_FOUND"
        explanation = (
            "FIND shortlisted sources only after product-role and requested-performance evidence checks. "
            "Attention metrics are not proof of sales; commercial/mechanism and Target-Fit remain downstream gates."
        )
    elif raw_returned == 0:
        outcome = "NO_API_ROWS"
        explanation = "SocialCrawl returned no ad rows; check query coverage, credentials, provider response, and available credits."
    elif target_relevant == 0:
        outcome = "NO_TARGET_RELEVANCE"
        explanation = (
            "Rows were returned, but no playable source had confirmed target relevance. Unknown-relevance rows remain "
            "held for source review; search terms alone are not evidence that the ad promotes the target product. "
            "commercial success is not required to extract a pattern, but target relevance and source-role checks "
            "remain separate admission requirements."
        )
    elif role_gate.get("applied") and int(role_gate.get("raw_rows") or 0) > 0 and int(role_gate.get("accepted_rows") or 0) == 0:
        outcome = "NO_TARGET_PRODUCT_AD"
        explanation = (
            "Discovery returned relevant-looking rows, but none independently showed the target product being promoted. "
            "Incidental mentions, use context, and adjacent products were rejected in product mode."
        )
    elif (performance_gate.get("applied")
          and int(performance_gate.get("considered_candidates") or 0) > 0
          and int(performance_gate.get("accepted_candidates") or 0) == 0):
        outcome = "NO_REQUESTED_PERFORMANCE_EVIDENCE"
        explanation = (
            "Target-product sources were found, but none had a native numeric value or a real position on the requested "
            "performance board(s). FIND did not silently substitute views/likes engagement proxies."
        )
    elif performance_ready == 0:
        outcome = "NO_PERFORMANCE_EVIDENCE"
        explanation = (
            "Product-relevant sources were found, but none had a performance-board position or available performance "
            "evidence sufficient for the requested shortlist. This is not a profitability verdict."
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
        "unique_ads": int(stats.get("provider_unique_ads") or stats.get("unique_ads") or 0),
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


def _numeric_signal(value) -> bool:
    """Accept only finite numeric metrics/positions; missing and boolean values are not evidence."""
    if isinstance(value, bool) or value in (None, "", []):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _has_requested_performance_evidence(candidate: dict, requested_metrics: list[str]) -> bool:
    """Require a requested native metric or a real position on a requested board."""
    if not requested_metrics:
        return True
    candidate = candidate if isinstance(candidate, dict) else {}
    record = candidate.get("legacy_record") if isinstance(candidate.get("legacy_record"), dict) else {}
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    for metric in requested_metrics:
        value = record.get(metric)
        metric_row = metrics.get(metric)
        if isinstance(metric_row, dict):
            value = metric_row.get("value", value)
        if _numeric_signal(value):
            return True
    positions = record.get("discovery_board_positions")
    positions = positions if isinstance(positions, dict) else {}
    why = candidate.get("why") if isinstance(candidate.get("why"), dict) else {}
    why_positions = why.get("discovery_board_positions")
    if isinstance(why_positions, dict):
        positions = {**positions, **why_positions}
    for key, position in positions.items():
        if any(str(key) == metric or str(key).startswith(metric + "|") for metric in requested_metrics):
            if _numeric_signal(position) and float(position) >= 1:
                return True
    performance = candidate.get("performance_evidence")
    performance = performance if isinstance(performance, dict) else {}
    reported_boards = performance.get("performance_boards")
    if isinstance(reported_boards, list) and any(str(board) in requested_metrics for board in reported_boards):
        return True
    return False


def _apply_product_selection_gates(candidates: list[dict], request: dict) -> tuple[list[dict], dict]:
    """Apply target-role and requested-performance gates after the legacy shortlist as defense-in-depth."""
    request = request if isinstance(request, dict) else {}
    context_present = bool(request.get("product_type") or request.get("product_category") or request.get("target_offer"))
    known_metrics = {"ctr", "cvr", "play_2s_rate", "play_6s_rate"}
    requested = list(dict.fromkeys(
        str(metric) for metric in (request.get("requested_performance_order_bys") or [])
        if str(metric) in known_metrics
    ))
    role_rejected, metric_rejected, accepted = [], [], []
    role_eligible_count = 0
    role_gate = {
        "applied": context_present, "raw_candidates": len(candidates or []),
        "role_eligible_candidates": 0, "rejected_candidates": 0,
        "rejected_by_reason": {}, "rejected_examples": [],
    }
    performance_gate = {
        "applied": bool(requested), "requested_metrics": requested,
        "considered_candidates": 0, "accepted_candidates": 0,
        "rejected_candidates": 0, "rejected_examples": [],
        "fallback_to_engagement_proxy": False,
    }
    for candidate in candidates or []:
        candidate = candidate if isinstance(candidate, dict) else {}
        record = candidate.get("legacy_record")
        record = record if isinstance(record, dict) else {}
        if context_present:
            role = classify_source_role(
                record,
                product_type=request.get("product_type") or request.get("product_category") or "",
                product_category=request.get("product_category") or "",
                target_offer=request.get("target_offer") or "",
            )
            record["source_role_info"] = role
            source_selection = candidate.get("source_selection")
            source_selection = dict(source_selection) if isinstance(source_selection, dict) else {}
            source_selection.update({
                "role": role.get("role") or "OFF_TOPIC", "role_reason": role.get("reason") or "",
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
            })
            candidate["source_selection"] = source_selection
            eligible_role = (
                role.get("role") == TARGET_PRODUCT
                and bool(role.get("promotion_evidence"))
                and bool(role.get("target_promotion_evidence"))
            )
            if not eligible_role:
                reason = str(role.get("role") or "OFF_TOPIC").upper()
                if reason == TARGET_PRODUCT and not role.get("target_promotion_evidence"):
                    reason = "TARGET_PRODUCT_NOT_PROMOTED"
                elif reason == TARGET_PRODUCT and not role.get("promotion_evidence"):
                    reason = "NO_INDEPENDENT_PROMOTION_EVIDENCE"
                role_gate["rejected_candidates"] += 1
                role_gate["rejected_by_reason"][reason] = role_gate["rejected_by_reason"].get(reason, 0) + 1
                role_rejected.append({
                    "id": candidate.get("id"), "ad_id": record.get("ad_id"),
                    "title": str(record.get("title") or candidate.get("title") or "")[:180],
                    "role": role.get("role"), "reason": reason,
                })
                continue
        role_eligible_count += 1
        if requested:
            performance_gate["considered_candidates"] += 1
            if not _has_requested_performance_evidence(candidate, requested):
                metric_rejected.append({
                    "id": candidate.get("id"), "ad_id": record.get("ad_id"),
                    "title": str(record.get("title") or candidate.get("title") or "")[:180],
                    "reason": "REQUESTED_PERFORMANCE_EVIDENCE_MISSING",
                    "available_basis": (candidate.get("performance_evidence") or {}).get("basis"),
                    "requested_metrics": list(requested),
                })
                continue
            performance_gate["accepted_candidates"] += 1
        accepted.append(candidate)
    role_gate["role_eligible_candidates"] = role_eligible_count
    role_gate["rejected_examples"] = role_rejected[:12]
    performance_gate["rejected_candidates"] = len(metric_rejected)
    performance_gate["rejected_examples"] = metric_rejected[:12]
    return accepted, {"product_selection": role_gate, "requested_performance": performance_gate}


def _attach_tracker_diagnostics(stats: dict, tracker: _DiscoveryEvidenceTracker) -> dict:
    """Separate raw provider-return counts from the filtered candidate queue."""
    if not isinstance(stats, dict):
        stats = {}
    tracked = [call for call in tracker.calls if call.get("candidate_lane")]
    summary = tracker.role_gate_diagnostics()
    stats["provider_rows_returned"] = int(summary.get("raw_rows") or 0)
    stats["provider_unique_ads"] = int(summary.get("unique_provider_ads") or 0)
    stats["product_source_role_gate"] = summary
    calls = stats.get("calls")
    if not isinstance(calls, list):
        calls = []
        stats["calls"] = calls
    for index, observed in enumerate(tracked):
        if index >= len(calls):
            break
        call = calls[index]
        if not isinstance(call, dict):
            continue
        call["provider_rows_returned"] = int(observed.get("provider_rows_returned") or 0)
        call["eligible_rows_returned"] = int(observed.get("eligible_rows_returned") or 0)
        call["role_rejected_rows"] = int(observed.get("role_rejected_rows") or 0)
        call["role_rejected_by_reason"] = dict(observed.get("role_rejected_by_reason") or {})
        call["playback_metrics_returned"] = dict(observed.get("playback_metrics_returned") or {})
        call["unfiltered_retention_request"] = bool(observed.get("unfiltered_retention_request"))
    return stats


def _source_seen(record: dict, keys: set[str]) -> bool:
    if not keys:
        return False
    if canonical_source_identity_keys:
        return bool(canonical_source_identity_keys(record) & keys)
    return bool(set(record.get("keys") or []) & keys)


def _run_mechanism_find(*, request: dict, offer: dict, keywords: list[str], client,
                        ledger: SeenLedger, scope: dict, prior_seen_keys: set[str],
                        stamp: str, country: str, period: int, max_requests: int,
                        max_ads: int, limit_per_board: int,
                        accept_role: str = ADJACENT_MECHANISM, lane: str = "mechanism",
                        attention_floor: float | None = None,
                        requested_metric_status: dict | None = None,
                        video_duration_range: str = "all") -> dict:
    """Sponsored-Search-only selection; no Ads Top lane, no pool-relative percentile.

    lane="mechanism": adjacent-product sources (explicit keywords).
    lane="target_product": same gates for the target product class (contract-aware product mode).
    """
    target_lane = lane == "target_product"
    # Target lane: attention ranks candidates but does not cut them by default (floor 0.0). Reach is priority
    # metadata, not a quality gate; a fixed floor removed relevant car-vacuum ads at 80-140k views. The
    # coverage gate still requires real engagement fields, and an operator may set attention_floor.
    floor = (0.0 if target_lane else MECHANISM_ATTENTION_FLOOR) if attention_floor is None else float(attention_floor)
    request["minimum_attention_coverage"] = MECHANISM_MIN_COVERAGE
    request["attention_floor"] = floor
    date_posted = {7: "this-week", 30: "this-month", 180: "last-6-months"}[period]
    calls, raw_records, qualified, rejected = [], [], [], []
    rows_by_identity, identity_keys_seen = {}, set()
    raw_count = 0
    duration_gate = {
        "applied": video_duration_range != "all",
        "range": video_duration_range,
        "considered_candidates": 0,
        "accepted_candidates": 0,
        "rejected_unknown_duration": 0,
        "rejected_out_of_range": 0,
    }
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
        query_product_match = match_source_to_product_query(
            record, str(record.get("discovery_query") or "")
        )
        record["query_product_match"] = query_product_match
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
        elif not query_product_match.get("matches"):
            reason = str(query_product_match.get("reason") or "QUERY_PRODUCT_UNVERIFIED")
        elif role.get("role") != accept_role:
            reason = str(role.get("role") or "OFF_TOPIC")
        elif not role.get("promotion_evidence"):
            reason = "NO_INDEPENDENT_PROMOTION_EVIDENCE"
        elif attention["coverage"] < MECHANISM_MIN_COVERAGE:
            reason = "LOW_ATTENTION_COVERAGE"
        elif attention["score"] < floor:
            reason = "BELOW_ATTENTION_PROXY_FLOOR"
        else:
            duration_gate["considered_candidates"] += 1
            reason = _video_duration_filter_reason(record, video_duration_range)
            if reason == "VIDEO_DURATION_UNKNOWN":
                duration_gate["rejected_unknown_duration"] += 1
            elif reason == "VIDEO_DURATION_OUT_OF_RANGE":
                duration_gate["rejected_out_of_range"] += 1
            else:
                duration_gate["accepted_candidates"] += 1
        if reason:
            rejected.append({
                "ad_id": str(record.get("ad_id") or record.get("video_id") or ""),
                "title": str(record.get("title") or "")[:160], "reason": reason,
                "role": role.get("role"), "query_product_anchor": query_product_match.get("query_anchor"),
                "query_product_reason": query_product_match.get("reason"),
                "attention_score": attention.get("score"),
                "attention_coverage": attention.get("coverage"),
            })
            record["selection_decision"], record["selection_reason"] = "REJECTED", reason
            continue
        record["performance_evidence"] = {
            "basis": "ATTENTION_PROXY_ONLY", "score": attention["score"],
            "coverage": attention["coverage"], "is_commercial_proof": False,
        }
        if requested_metric_status:
            record["performance_evidence"].update(requested_metric_status)
            record["performance_evidence"]["requested_metric_confirmed"] = False
        record["selection_decision"] = "QUALIFIED_TARGET_PRODUCT" if target_lane else "QUALIFIED_ADJACENT_MECHANISM"
        qualified.append(record)

    qualified.sort(key=lambda record: (
        -float((record.get("attention_score") or {}).get("score") or 0),
        -float((record.get("attention_score") or {}).get("coverage") or 0),
        str(record.get("ad_id") or record.get("video_id") or ""),
    ))
    # Product-first FIND uses the same pre-video scorer as legacy discover_and_rank.
    # Keep its full provider-result cohort as the comparison population; the attention
    # ranking and all role/query gates above remain unchanged.
    scored_qualified = []
    scoring_failed = 0
    if target_lane and qualified and score_tiktok_candidate is None:
        raise FindUnavailable(
            "SCORER_UNAVAILABLE",
            LEGACY_ERROR or "legacy score_tiktok_candidate could not be imported",
        )
    comparison_population = list(rows_by_identity.values())
    for record in qualified:
        decision = None
        if target_lane:
            try:
                decision = score_tiktok_candidate(
                    record, comparison_population,
                    target_offer=offer.get("target_offer") or "",
                    prelander=offer.get("prelander") or "",
                    landing_page=offer.get("landing_page") or "",
                    vertical=VERTICAL,
                )
            except Exception as exc:
                scoring_failed += 1
                rejected.append({
                    "ad_id": str(record.get("ad_id") or record.get("video_id") or ""),
                    "title": str(record.get("title") or "")[:160],
                    "reason": "LEGACY_SCORING_FAILED",
                    "detail": f"{type(exc).__name__}: {exc}"[:240],
                    "role": (record.get("source_role_info") or {}).get("role"),
                })
                continue
        scored_qualified.append((record, decision))

    selected = scored_qualified[:max_ads]
    for record, _decision in selected:
        record["selection_decision"] = "SELECTED_TARGET_PRODUCT" if target_lane else "SELECTED_ADJACENT_MECHANISM"
        record["selection_reason"] = (
            ("Promoted target-class product" if target_lane else "Promoted adjacent product")
            + " passed the source-body, media, and attention-proxy gates. "
            + ("The existing legacy pre-video source score is attached; attention and score are not proof of sales or profit. "
               if target_lane else "")
            + "Causal mechanism, transferability, and independent Target-Fit remain downstream."
        )
    for record, _decision in scored_qualified[max_ads:]:
        record["selection_decision"] = "QUALIFIED_NOT_SELECTED_LIMIT"
        record["selection_reason"] = "Passed role and attention gates but exceeded max_ads."
    candidates = [
        to_candidate({"record": record, "decision": decision}, request, stamp, "found")
        for record, decision in selected
    ]
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
        "video_duration_filter": duration_gate,
        "performance_lane_reserved": False,
        "discovery_strategy": {
            "selection_mode": "product" if target_lane else "mechanism",
            "supply_lane": "tiktok_sponsored_search_only",
            "supply_search_enabled": True, "supply_search_budget": len(calls), "performance_budget": 0,
            "performance_calls_used": 0, "supply_queries_planned": keywords[:max_requests],
            "supply_queries_executed": [call["keyword"] for call in calls],
            "request_policy": (
                "Contract-aware product plan: product-targeted Sponsored Search only; Ads Top/retention not "
                "executed because the provider cannot return them for a product-targeted ad."
                if target_lane else
                "Explicit-keyword Sponsored Search only; Ads Top/performance ordering disabled."),
            "attention_floor": floor, "requested_metric_status": requested_metric_status or None,
            "attention_proxy_version": "attention_absolute_v1", "attention_is_commercial_proof": False,
            "seen_ttl_days": 30, "prior_query_history": query_history_before,
        },
        "source_selection_diagnostics": {
            "qualified_count": len(qualified), "scored_count": len(scored_qualified),
            "scoring_failed_count": scoring_failed, "selected_count": len(selected),
            "rejected_count": len(rejected), "rejected_by_reason": dict(sorted(reason_counts.items())),
            "rejected_examples": rejected[:50], "by_reason": dict(sorted(reason_counts.items())),
            "examples": rejected[:50],
        },
    }
    diagnostics = {
        "outcome": "CANDIDATES_FOUND" if candidates else (
            "NO_API_ROWS" if raw_count == 0
            else "SCORING_FAILED" if target_lane and qualified and not scored_qualified
            else "NO_TARGET_RELEVANCE" if target_lane else "NO_QUALIFIED_ADJACENT_MECHANISM"),
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
    stats["total_credits_reported"] = sum(
        int(call.get("credits_used") or 0) for call in calls if isinstance(call.get("credits_used"), (int, float)))
    diagnostics["total_credits_reported"] = stats["total_credits_reported"]
    if (not candidates and duration_gate["applied"] and duration_gate["considered_candidates"] > 0
            and duration_gate["accepted_candidates"] == 0):
        diagnostics["outcome"] = "NO_MATCHING_VIDEO_DURATION"
        diagnostics["explanation"] = (
            "Sources passed the existing FIND gates, but none matched the selected video-duration range. "
            "Candidates with unknown duration were excluded from this range."
        )
    diagnostics["video_duration_filter"] = duration_gate
    if requested_metric_status:
        availability = {
            "requested_metrics": list(requested_metric_status.get("requested_metrics") or []),
            "numeric_values_returned": {}, "board_rows_returned": {},
            "status": requested_metric_status.get("requested_metric_status"),
            "provider_contract": contract_summary(),
            "interpretation": (
                "Requested playback metrics were not executed as boards: the provider cannot return them for a "
                "product-targeted ad. Candidates carry labeled attention-proxy evidence only."
            ),
        }
        stats["performance_metric_availability"] = availability
        diagnostics["performance_metric_availability"] = availability
        diagnostics["requested_metric_status"] = requested_metric_status.get("requested_metric_status")
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
    video_duration_range = str(params.get("video_duration_range") or "all").strip().lower()
    if video_duration_range not in VIDEO_DURATION_RANGES:
        raise ValueError(f"video_duration_range must be one of {VIDEO_DURATION_RANGES}")
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
    raw_include_retention = params.get("include_retention", False)
    include_retention = raw_include_retention is True or str(raw_include_retention).strip().lower() in {"1", "true", "yes", "on"}
    if include_retention and not {"play_2s_rate", "play_6s_rate"} & set(performance_order_bys):
        raise ValueError("include_retention needs performance_order_bys with play_2s_rate and/or play_6s_rate")
    requested_max_requests = max_requests
    requested_performance_order_bys = list(performance_order_bys)
    performance_plan = str(params.get("performance_plan") or "contract_aware").strip().lower()
    include_retention_requested = include_retention
    performance_order_bys, max_requests, retention_strategy = _retention_execution_plan(
        requested_performance_order_bys, include_retention, requested_max_requests,
        performance_plan=performance_plan,
    )
    contract_aware = retention_strategy.get("mode") == CONTRACT_AWARE_MODE
    if contract_aware:
        include_retention = False  # not executed: retention cannot be product-targeted
    user_keywords = _clean_terms(params.get("keywords"))
    offer = _offer_context(params)
    performance_names = {"ctr", "cvr", "play_2s_rate", "play_6s_rate"}
    if selection_mode == "mechanism":
        if not user_keywords:
            raise ValueError("mechanism selection requires explicit keywords; offer fields are not discovery keywords")
        if any(str(value).strip() in performance_names
               for value in requested_performance_order_bys + performance_order_bys + order_bys):
            raise ValueError("mechanism mode disables Ads Top/performance order; choose product mode for performance-board discovery")
    conflict = _query_offer_conflict(user_keywords, offer) if selection_mode == "product" else None
    search_terms = _clean_terms(user_keywords + [offer["target_offer"], offer["product_type"], offer["product_category"]]) if selection_mode == "product" else user_keywords
    request = {
        "country": country, "period": period, "order_bys": order_bys,
        "performance_order_bys": performance_order_bys,
        "requested_performance_order_bys": requested_performance_order_bys,
        "include_retention": include_retention,
        "include_retention_requested": include_retention_requested,
        "performance_plan": performance_plan,
        "max_requests": max_requests, "requested_max_requests": requested_max_requests,
        "retention_strategy": retention_strategy,
        "keywords": user_keywords,
        "search_terms": search_terms, "selection_mode": selection_mode,
        "target_offer": offer["target_offer"], "product_type": offer["product_type"],
        "product_category": offer["product_category"], "max_ads": max_ads,
        "video_duration_range": video_duration_range,
        "vertical": VERTICAL,
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
    client = client or RetryingSocialCrawlTikTokTopAds()
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
            limit_per_board=limit_per_board, video_duration_range=video_duration_range,
        )
    if contract_aware:
        queries = ordered_target_queries(
            offer["target_offer"], offer["product_type"], user_keywords, offer["product_category"],
            limit=min(3, requested_max_requests),
            rotate=lambda terms: _rotate_query_terms(terms, prior_query_history),
        )
        request["search_terms"] = list(queries)
        request["targeted_sponsored_search_queries"] = list(queries)
        request["provider_request_plan"] = {
            "target_sponsored_search_queries": len(queries), "targeted_playback_boards": [],
            "unfiltered_retention_calls": 0, "unrequested_performance_fallbacks": False,
            "requested_metric_status": retention_strategy["requested_metric_status"],
        }
        raw_floor = params.get("attention_floor")
        return _run_mechanism_find(
            request=request, offer=offer, keywords=queries, client=client,
            ledger=ledger, scope=scope, prior_seen_keys=prior_seen_keys, stamp=stamp,
            country=country, period=period, max_requests=len(queries), max_ads=max_ads,
            limit_per_board=limit_per_board, accept_role=TARGET_PRODUCT, lane="target_product",
            attention_floor=(None if raw_floor in (None, "") else float(raw_floor)),
            video_duration_range=video_duration_range,
            requested_metric_status={
                "requested_metrics": list(requested_performance_order_bys),
                "requested_metric_status": retention_strategy["requested_metric_status"],
            },
        )
    search_terms = _rotate_query_terms(search_terms, prior_query_history)
    request["search_terms"] = list(search_terms)
    dual_playback_retention = retention_strategy.get("mode") == "targeted_playback_boards_plus_retention"
    targeted_search_queries = (
        _ordered_supply_queries(
            VERTICAL, search_terms, max_count=2, search_tags_dir=PRODUCT_SEARCH_TAGS_DIR,
        )
        if dual_playback_retention and _ordered_supply_queries else []
    )
    if dual_playback_retention:
        request["targeted_sponsored_search_queries"] = list(targeted_search_queries)
        request["provider_request_plan"] = {
            "target_sponsored_search_queries": len(targeted_search_queries),
            "targeted_playback_boards": list(performance_order_bys),
            "targeted_board_keyword": _target_board_keyword(offer, user_keywords[0] if user_keywords else ""),
            "unfiltered_retention_calls_planned": 1,
            "unfiltered_retention_calls": 1,
            "retention_policy": "run only if targeted playback boards contain target-product rows",
            "unrequested_performance_fallbacks": False,
        }
    evidence_tracker = _DiscoveryEvidenceTracker(
        client, offer=offer,
        multi_sponsored_search_queries=targeted_search_queries if dual_playback_retention else None,
        restrict_order_bys=performance_order_bys if dual_playback_retention else None,
    )
    retention_enrichment_execution = {
        "requested": bool(dual_playback_retention),
        "executed": False,
        "reason": "NOT_REQUESTED" if not dual_playback_retention else "PENDING_TARGET_BOARD_RESULTS",
    }
    try:
        ranked = discover_and_rank(
            client=evidence_tracker, country=country, period=period, keywords=search_terms, order_bys=order_bys,
            limit_per_board=limit_per_board, max_ads=max_ads,
            # One internal Sponsored Search slot represents the two provider queries
            # batched above; reserve exactly one internal call for each requested board.
            max_requests=(3 if dual_playback_retention else max_requests),
            keyword_boards=(1 if dual_playback_retention else max_requests),
            # Keep the legacy scheduler from ending the bounded batch early merely
            # because pre-gate scoring marks enough raw sources as analysis-ready.
            source_queue_size=(
                500 if dual_playback_retention else
                min(250, max(50, max_ads * 5)) if video_duration_range != "all" else None
            ),
            vertical=VERTICAL, target_offer=offer["target_offer"],
            product_type=offer["product_type"] or offer["product_category"],
            prelander=offer["prelander"], landing_page=offer["landing_page"],
            discovery_mode=str(params.get("discovery_mode") or "auto"),
            exclude_ad_ids=prior_seen_keys,
            search_tags_dir=PRODUCT_SEARCH_TAGS_DIR, supply_search=True,
            performance_order_bys=performance_order_bys,
            include_retention=(False if dual_playback_retention else include_retention))
        if dual_playback_retention:
            # Retention is enrichment only; don't spend the extra 20+ credits when the
            # targeted performance boards returned no target-product ads to enrich.
            retention_policy = _retention_enrichment_policy(evidence_tracker.calls)
            retention_strategy["retention_calls_planned"] = int(retention_strategy.get("retention_calls") or 0)
            retention_strategy["retention_calls_executed"] = int(bool(retention_policy["should_run"]))
            request_plan = request.get("provider_request_plan") or {}
            request_plan["unfiltered_retention_calls_planned"] = retention_strategy["retention_calls_planned"]
            request_plan["unfiltered_retention_calls"] = int(bool(retention_policy["should_run"]))
            request_plan["retention_execution"] = dict(retention_policy)
            if retention_policy["should_run"]:
                evidence_tracker.fetch_retention_enrichment(
                    country=country,
                    period=period,
                    order_by=retention_strategy["retention_request_order_by"],
                    keyword=None,
                    industry=None,
                    ad_language="en",
                    limit=limit_per_board,
                    include_retention=True,
                )
                retention_strategy["retention_calls"] = 1
                retention_strategy["retention_call_skipped_reason"] = ""
                retention_enrichment_execution = {
                    **retention_policy, "requested": True, "executed": True,
                }
            else:
                retention_strategy["retention_calls"] = 0
                retention_strategy["retention_call_skipped_reason"] = retention_policy["reason"]
                retention_enrichment_execution = {
                    **retention_policy, "requested": True, "executed": False,
                }
    except SocialCrawlError as exc:
        raise FindUnavailable(
            "UNAVAILABLE",
            str(exc),
            diagnostics={
                "provider": "SocialCrawl",
                "error_type": type(exc).__name__,
                "request": getattr(exc, "diagnostics", {}) or getattr(client, "last_request_diagnostics", {}),
                "discovery_stats": getattr(client, "last_discovery_stats", {}) or {},
            },
        ) from exc
    except OSError as exc:
        raise FindUnavailable(
            "UNAVAILABLE",
            f"{type(exc).__name__}: {exc}",
            diagnostics={
                "provider": "SocialCrawl",
                "error_type": type(exc).__name__,
                "request": getattr(client, "last_request_diagnostics", {}),
                "discovery_stats": getattr(client, "last_discovery_stats", {}) or {},
            },
        ) from exc
    stats = getattr(client, "last_discovery_stats", {}) or {}
    if dual_playback_retention:
        # Replace the legacy controller's internal-call view with the actual provider
        # request ledger: the batched search call above represents multiple real requests.
        observed_calls = [call for call in evidence_tracker.calls if call.get("candidate_lane")]
        if observed_calls:
            provider_calls, seen_provider_ids = [], set()
            for observed in observed_calls:
                provider_ids = list(observed.get("provider_ad_ids") or [])
                new_unique = len(set(provider_ids) - seen_provider_ids)
                seen_provider_ids.update(provider_ids)
                provider_calls.append({
                    "call": len(provider_calls) + 1,
                    "source": observed.get("source"),
                    "order_by": observed.get("order_by"),
                    "keyword": observed.get("keyword") or "",
                    "requested_keyword": observed.get("requested_keyword") or "",
                    "target_role_match_count": observed.get("target_role_match_count"),
                    "target_role_match_status": observed.get("target_role_match_status") or "",
                    "target_query_relevance_warning": observed.get("target_query_relevance_warning") or "",
                    "returned": int(observed.get("provider_rows_returned") or 0),
                    "new_unique": new_unique,
                    "unique_after": len(seen_provider_ids),
                    "credits_used": observed.get("credits_used"),
                    "request_id": observed.get("request_id"),
                    "provider_params": observed.get("provider_params") or {},
                    "returned_ad_ids": list(observed.get("returned_ad_ids") or []),
                    "playback_metrics_returned": dict(observed.get("playback_metrics_returned") or {}),
                    "retention_requested": bool(observed.get("retention_requested")),
                    "retention_returned": dict(observed.get("retention_returned") or {}),
                    "raw_playback_field_diagnostics": observed.get("raw_playback_field_diagnostics") or {},
                    "provider_response_meta": observed.get("provider_response_meta") or {},
                })
            stats["calls"] = provider_calls
            calls = list(stats.get("calls") or [])
            stats["planned_requests"] = len(provider_calls)
            stats["executed_requests"] = len(provider_calls)
            stats["request_budget"] = requested_max_requests
            stats["search_queries_executed"] = [
                call.get("keyword", "") for call in observed_calls if call.get("source") == "sponsored_search"
            ]
            stats["request_sources"] = [call.get("source") for call in observed_calls]
            stats["total_credits_reported"] = sum(
                int(call.get("credits_used") or 0) for call in observed_calls
                if isinstance(call.get("credits_used"), (int, float))
            )
            strategy = stats.setdefault("discovery_strategy", {})
            strategy.update({
                "supply_search_budget": len(targeted_search_queries),
                "target_keyword_budget": len(targeted_search_queries),
                "keyword_boards": len(targeted_search_queries),
                "supply_queries_planned": list(targeted_search_queries),
                "supply_queries_executed": [
                    call.get("keyword", "") for call in observed_calls if call.get("source") == "sponsored_search"
                ],
                "performance_budget": len(performance_order_bys),
                "reserved_performance_budget": len(performance_order_bys),
                "fallback_broad_budget": 0,
                "performance_calls_used": sum(1 for call in observed_calls if call.get("source") == "ads_top"),
                "query_failover_policy": (
                    "complete bounded target search batch, then exactly the requested performance boards; "
                    "no unrequested CTR/CVR or engagement fallback boards"
                ),
            })
        else:
            calls = list(stats.get("calls") or [])
        enrichment_calls = [
            call for call in evidence_tracker.calls
            if not call.get("candidate_lane") and call.get("retention_requested")
        ]
        if enrichment_calls:
            retention_call = dict(enrichment_calls[-1])
            retention_call["call"] = len(calls) + 1
            calls.append(retention_call)
            stats["calls"] = calls
            stats["planned_requests"] = len(calls)
            stats["executed_requests"] = len(calls)
            stats["request_budget"] = requested_max_requests
            credits = retention_call.get("credits_used")
            if isinstance(credits, (int, float)):
                stats["total_credits_reported"] = int(stats.get("total_credits_reported") or 0) + int(credits)
    stats = _attach_tracker_diagnostics(stats, evidence_tracker)
    stats["retention_enrichment"] = dict(retention_enrichment_execution)
    _merge_discovery_evidence(ranked, evidence_tracker)
    _enrich_selection_diagnostics(stats, evidence_tracker)
    duration_filter_diagnostics = {
        "applied": video_duration_range != "all",
        "range": video_duration_range,
        "considered_candidates": len(ranked),
        "accepted_candidates": 0,
        "rejected_unknown_duration": 0,
        "rejected_out_of_range": 0,
    }
    if duration_filter_diagnostics["applied"]:
        duration_qualified = []
        for item in ranked:
            reason = _video_duration_filter_reason(item.get("record"), video_duration_range)
            if reason == "VIDEO_DURATION_UNKNOWN":
                duration_filter_diagnostics["rejected_unknown_duration"] += 1
            elif reason == "VIDEO_DURATION_OUT_OF_RANGE":
                duration_filter_diagnostics["rejected_out_of_range"] += 1
            else:
                duration_filter_diagnostics["accepted_candidates"] += 1
                duration_qualified.append(item)
        ranked = duration_qualified
    else:
        duration_filter_diagnostics["accepted_candidates"] = len(ranked)
    candidates_before_final_gates = [to_candidate(item, request, stamp, "found") for item in ranked]
    candidates, final_gates = _apply_product_selection_gates(candidates_before_final_gates, request)
    if dual_playback_retention or duration_filter_diagnostics["applied"]:
        accepted_before_output_limit = len(candidates)
        candidates = candidates[:max_ads]
        final_gates["product_selection"]["accepted_before_output_limit"] = accepted_before_output_limit
        final_gates["product_selection"]["output_limit"] = max_ads
        final_gates["product_selection"]["returned_candidates"] = len(candidates)
        final_gates["product_selection"]["trimmed_after_gates"] = max(
            0, accepted_before_output_limit - len(candidates)
        )
    stats["video_duration_filter"] = duration_filter_diagnostics
    stats["product_selection_gate"] = final_gates["product_selection"]
    stats["requested_performance_evidence_gate"] = final_gates["requested_performance"]
    strategy = stats.setdefault("discovery_strategy", {})
    executed_calls = stats.get("calls") or []
    if retention_strategy.get("mode") == "targeted_playback_boards_plus_retention":
        if retention_enrichment_execution.get("executed"):
            strategy["request_policy"] = (
                "Complete the bounded target Sponsored Search batch and requested target-product playback boards, then "
                "run one separate unfiltered retention enrichment request only because target-product rows were found. "
                "Retention does not replace playback-rate evidence or relax Selection/Target-Fit gates."
            )
        else:
            strategy["request_policy"] = (
                "Complete the bounded target Sponsored Search batch and requested target-product playback boards. "
                "Skip the paid unfiltered retention request when no target-product rows pass the targeted board role gate; "
                "do not substitute engagement proxies or relax Selection/Target-Fit gates."
            )
    else:
        strategy["request_policy"] = (
            "Bounded Sponsored Search plus reserved Ads Top performance queries. FIND shortlists by product-type match "
            "and strongest available performance evidence; money/mechanism and Target-Fit remain downstream transfer gates."
        )
    strategy["supply_search_enabled"] = True
    strategy["supply_lane"] = "tiktok_sponsored_search_plus_ads_top_performance"
    strategy["selection_mode"] = "product"
    strategy["supply_search_calls_used"] = sum(1 for call in executed_calls if call.get("source") == "sponsored_search")
    strategy["performance_calls_used"] = sum(
        1 for call in executed_calls
        if call.get("source") == "ads_top" and not call.get("retention_requested")
    )
    if performance_order_bys:
        playback_board_calls = [
            call for call in (stats.get("calls") or [])
            if isinstance(call, dict)
            and call.get("order_by") in {"play_2s_rate", "play_6s_rate"}
            and not call.get("retention_requested")
        ]
        stats["performance_metric_availability"] = {
            "requested_metrics": list(performance_order_bys),
            "numeric_values_returned": {
                metric: sum(
                    int((call.get("playback_metrics_returned") or {}).get(metric) or 0)
                    for call in playback_board_calls if call.get("order_by") == metric
                )
                for metric in performance_order_bys
            },
            "board_rows_returned": {
                metric: sum(
                    int(call.get("returned") or 0)
                    for call in playback_board_calls if call.get("order_by") == metric
                )
                for metric in performance_order_bys
            },
            "provider_contract_note": (
                "SocialCrawl documents play_2s_rate/play_6s_rate as Ads Top sorting options, "
                "but does not publish those numeric rate values in each ad row. Retention points "
                "are separately returned observations and are not the exact board score."
            ),
            "interpretation": (
                "A positive board row count proves only that the requested board returned rows; "
                "it does not mean the numeric playback-rate fields were returned."
            ),
        }
    diagnostics = _find_diagnostics(stats, len(candidates))
    diagnostics["video_duration_filter"] = duration_filter_diagnostics
    if (duration_filter_diagnostics["applied"] and duration_filter_diagnostics["considered_candidates"] > 0
            and duration_filter_diagnostics["accepted_candidates"] == 0):
        diagnostics["outcome"] = "NO_MATCHING_VIDEO_DURATION"
        diagnostics["explanation"] = (
            "FIND found candidates, but none matched the selected video-duration range. "
            "Records with unknown duration were excluded from the selected range."
        )
    diagnostics["product_source_role_gate"] = stats.get("product_source_role_gate") or {}
    diagnostics["product_selection_gate"] = stats.get("product_selection_gate") or {}
    diagnostics["requested_performance_evidence_gate"] = stats.get("requested_performance_evidence_gate") or {}
    diagnostics["retention_enrichment"] = stats.get("retention_enrichment") or {}
    diagnostics["performance_metric_availability"] = stats.get("performance_metric_availability") or {}
    diagnostics["provider_query_diagnostics"] = [
        {
            "call": call.get("call"),
            "source": call.get("source"),
            "order_by": call.get("order_by"),
            "keyword": call.get("keyword") or "",
            "requested_keyword": call.get("requested_keyword") or "",
            "target_role_match_count": call.get("target_role_match_count"),
            "target_role_match_status": call.get("target_role_match_status") or "",
            "target_query_relevance_warning": call.get("target_query_relevance_warning") or "",
            "returned": call.get("returned", 0),
            "credits_used": call.get("credits_used"),
            "request_id": call.get("request_id"),
            "provider_params": call.get("provider_params") or {},
            "provider_response_meta": call.get("provider_response_meta") or {},
            "playback_metrics_returned": call.get("playback_metrics_returned") or {},
            "retention_requested": bool(call.get("retention_requested")),
            "retention_returned": call.get("retention_returned") or {},
            "raw_playback_field_diagnostics": call.get("raw_playback_field_diagnostics") or {},
        }
        for call in (stats.get("calls") or []) if isinstance(call, dict)
    ]
    retention_calls = [
        call for call in (stats.get("calls") or [])
        if isinstance(call, dict) and call.get("retention_requested")
    ]
    retention_returned = {
        field: sum(int((call.get("retention_returned") or {}).get(field) or 0) for call in retention_calls)
        for field in ("retention_2s", "retention_6s", "retention_curve")
    }
    selected_with_retention = sum(
        1 for candidate in candidates
        if (candidate.get("retention_evidence") or {}).get("status") == "AVAILABLE"
    )
    retention_summary = {
        "strategy": retention_strategy,
        "retention_calls": len(retention_calls),
        "retention_call_details": [
            {
                "order_by": call.get("order_by"),
                "keyword": call.get("keyword") or "",
                "retention_requested": bool(call.get("retention_requested")),
                "returned": call.get("returned", 0),
            }
            for call in retention_calls
        ],
        "targeted_playback_board_calls": [
            {
                "order_by": call.get("order_by"),
                "keyword": call.get("keyword") or "",
                "returned": call.get("returned", 0),
                "retention_requested": bool(call.get("retention_requested")),
            }
            for call in (stats.get("calls") or [])
            if call.get("order_by") in {"play_2s_rate", "play_6s_rate"}
            and call.get("keyword") and not call.get("retention_requested")
        ],
        "returned_rows_with_retention": retention_returned,
        "selected_candidates_with_retention": selected_with_retention,
        "selected_candidates_from_retention_board": (
            0 if retention_strategy.get("mode") == "targeted_playback_boards_plus_retention" else sum(
                1 for candidate in candidates
                if any(board in {"play_2s_rate", "play_6s_rate"} for board in ((candidate.get("why") or {}).get("boards") or []))
            )
        ),
        "curve_schema": [
            ((call.get("raw_playback_field_diagnostics") or {}).get("retention_curve_schema") or {})
            for call in retention_calls
        ],
        "note": (
            "Retention-board rows still must pass product relevance and source-quality gates. "
            "Unrelated high-retention ads must not be admitted solely because they rank on Ads Top."
        ),
    }
    stats["retention_selection"] = retention_summary
    diagnostics["retention_selection"] = retention_summary
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
