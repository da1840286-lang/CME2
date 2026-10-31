"""What the SocialCrawl surfaces can and cannot tell FIND (single source of truth).

Every claim below is backed by the audited runs (FIND 40647469407 .. 40681442750); the flags are
data, so a controlled probe can flip them without touching selection logic.
"""
from __future__ import annotations

CONTRACT_AWARE_MODE = "contract_aware_sponsored_batch"
LEGACY_MODE = "legacy_ads_top"

PROVIDER_CONTRACT = {
    "ads_top": {
        # play_2s_rate / play_6s_rate are accepted as order_by values; rows never carried the numbers
        # (playback_metrics_returned == 0 in every audited call).
        "numeric_play_rates_in_rows": False,
        # A keyword on a play-rate board did not narrow results: the same off-topic ads (skin care,
        # bird feeder, swap meet) came back for "car vacuum" and "cordless handheld car vacuum".
        "keyword_filter_verified_for_play_rate_boards": False,
        # include=retention cannot be combined with keyword/industry (provider error), so retention
        # can only describe an unfiltered board and can never be about the target product.
        "retention_combinable_with_keyword": False,
    },
    "sponsored_search": {
        "keyword_targeting": True,
        "native_performance_fields": ("views", "likes", "comments", "shares"),
        "numeric_play_rates_in_rows": False,
    },
}


def plan_uses_contract(performance_plan: str) -> bool:
    return str(performance_plan or "contract_aware").strip().lower() != LEGACY_MODE


def contract_summary() -> dict:
    ads = PROVIDER_CONTRACT["ads_top"]
    return {
        "ads_top_numeric_play_rates_in_rows": ads["numeric_play_rates_in_rows"],
        "ads_top_keyword_filter_verified_for_play_rate_boards": ads["keyword_filter_verified_for_play_rate_boards"],
        "ads_top_retention_combinable_with_keyword": ads["retention_combinable_with_keyword"],
        "sponsored_search_keyword_targeting": PROVIDER_CONTRACT["sponsored_search"]["keyword_targeting"],
        "consequence": (
            "No single provider surface returns an ad that is simultaneously product-targeted and "
            "ranked/measured on play_2s_rate or play_6s_rate."
        ),
    }


def ordered_target_queries(target_offer: str, product_type: str, keywords, product_category: str = "",
                           *, limit: int = 3, rotate=None) -> list[str]:
    """Deterministic Sponsored Search plan: exact product, then the operator's own keywords, then derived terms.

    The exact product name always leads. Keywords the operator typed come next and are never rotated away or
    displaced by terms derived from the offer (product type, category); only those derived terms rotate across
    runs so repeated runs do not re-buy the same pool. If the operator typed more keywords than the budget
    allows, the overflow is the caller's to report (see request.dropped_operator_keywords).
    """
    def clean(value) -> str:
        return " ".join(str(value or "").split())

    seen, head, operator, derived = set(), [], [], []

    def add(bucket, term):
        term = clean(term)
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            bucket.append(term)

    add(head, target_offer)
    for term in keywords or []:
        add(operator, term)
    for term in (product_type, product_category):
        add(derived, term)
    if callable(rotate):
        derived = list(rotate(derived))
    return (head + operator + derived)[: max(0, int(limit))]
