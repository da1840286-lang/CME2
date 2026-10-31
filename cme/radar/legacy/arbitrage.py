from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .verticals import classify_vertical


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v in (None, "", "null"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _percentile(value: float | None, values: list[float]) -> float | None:
    if value is None or not values:
        return None
    clean = sorted(v for v in values if v is not None)
    # A single observation has no comparison cohort. Treating it as the
    # 100th percentile falsely labels isolated ads as elite performers.
    if len(clean) < 2:
        return None
    less_or_equal = sum(1 for v in clean if v <= value)
    return 100.0 * (less_or_equal - 1) / (len(clean) - 1)


def _performance_signal(record: dict[str, Any], population: list[dict[str, Any]]) -> tuple[float, dict[str, Any], list[str]]:
    s = record.get("source_signal") or {}
    # Use TikTok-native performance metrics first. Engagement is only a fallback.
    metric_weights = {
        "ctr": 0.45,
        "cvr": 0.20,
        "play_6s_rate": 0.15,
        "play_2s_rate": 0.10,
        "estimated_reach": 0.10,
    }
    components: dict[str, Any] = {}
    weighted: list[tuple[float, float]] = []
    fallback_values: list[float] = []
    tags: list[str] = []

    for field, weight in metric_weights.items():
        value = record.get(field, s.get(field))
        try:
            value_f = float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            value_f = None
        if value_f is None:
            continue
        vals = []
        for x in population:
            raw = x.get(field, (x.get("source_signal") or {}).get(field))
            try:
                if raw not in (None, ""):
                    vals.append(float(raw))
            except (TypeError, ValueError):
                pass
        p = _percentile(value_f, vals)
        if p is not None:
            components[field] = round(p, 1)
            weighted.append((p, weight))

    if weighted:
        active_weight = sum(w for _, w in weighted) or 1.0
        score = sum(v * w for v, w in weighted) / active_weight
    else:
        for field in ("likes", "views"):
            value = record.get(field, s.get(field))
            try:
                value_f = float(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                value_f = None
            if value_f is not None:
                vals = []
                for x in population:
                    raw = x.get(field, (x.get("source_signal") or {}).get(field))
                    try:
                        if raw not in (None, ""):
                            vals.append(float(raw))
                    except (TypeError, ValueError):
                        pass
                p = _percentile(value_f, vals)
                if p is not None:
                    components[field] = round(p, 1)
                    fallback_values.append(p)
        score = sum(fallback_values) / len(fallback_values) if fallback_values else 50.0

    rank = _num(record.get("rank"), 0)
    if rank > 0:
        # Rank is secondary corroboration, never the main performance metric.
        limit = max(10, len(population))
        rank_score = max(0.0, 100.0 * (1 - (rank - 1) / max(1, limit - 1)))
        components["board_rank"] = round(rank_score, 1)
        score = score * 0.90 + rank_score * 0.10

    if score >= 92:
        tags.append("elite_performance_signal")
    elif score >= 80:
        tags.append("strong_performance_signal")
    elif score >= 65:
        tags.append("usable_performance_signal")
    else:
        tags.append("weak_performance_signal")
    return max(0.0, min(100.0, score)), components, tags

def _metadata_commercial_score(record: dict[str, Any], target_offer: str, prelander: str, landing_page: str) -> tuple[float, list[str]]:
    s = record.get("source_signal") or {}
    text = " ".join(str(record.get(k, "") or "") for k in ("title", "brand", "brand_name", "industry", "objective")).lower()
    objective = str(record.get("objective", "") or s.get("objective", "")).lower()
    score = 0.0
    tags: list[str] = []
    if record.get("offer"):
        score += 12; tags.append("offer_present")
    commercial_intent = str(record.get("commercial_intent", "") or s.get("commercial_intent", "")).upper()
    if commercial_intent == "HIGH":
        score += 20; tags.append("high_commercial_intent_signal")
    elif commercial_intent == "MEDIUM":
        score += 10; tags.append("medium_commercial_intent_signal")
    if record.get("title"):
        score += 18; tags.append("ad_copy_present")
    if record.get("brand") not in (None, "", "UNKNOWN"):
        score += 10; tags.append("advertiser_identified")
    if objective:
        score += 12; tags.append("objective_present")
        if any(x in objective for x in ("conversion", "conversions", "app", "lead", "sales", "product")):
            score += 15; tags.append("conversion_oriented_objective")
    if record.get("video_url"):
        score += 8; tags.append("video_available")
    if record.get("proof"):
        score += 8; tags.append("proof_signal")
    if record.get("problem"):
        score += 7; tags.append("human_desire_signal")
        tags.append("tension_signal")
    if record.get("transformation"):
        score += 7; tags.append("transformation_signal")
    target = " ".join(x for x in (target_offer, prelander, landing_page) if x).lower()
    if target:
        tokens = {t for t in target.replace("/", " ").replace(",", " ").split() if len(t) >= 4}
        hits = sum(1 for t in tokens if t in text)
        if hits:
            score += min(20.0, hits * 4.0); tags.append("target_context_overlap")
    return min(100.0, score), tags


def _money_signal_gate(
    record: dict[str, Any],
    *,
    performance_score: float,
    commercial_score: float,
) -> tuple[bool, float, list[str], str]:
    """Require source-side evidence that a high-performing creative is commercially actionable.

    Views and engagement alone are never a money signal. This gate runs before deep
    video/LLM analysis so expensive analysis budget is reserved for economically
    credible source candidates. Target-overlap is intentionally ignored.
    """
    raw = record.get("raw") or {}
    computed = raw.get("computed") if isinstance(raw, dict) else {}
    computed = computed if isinstance(computed, dict) else {}
    labels = computed.get("labels") if isinstance(computed, dict) else {}
    labels = labels if isinstance(labels, dict) else {}
    intent = labels.get("intent") if isinstance(labels, dict) else {}
    intent = intent if isinstance(intent, dict) else {}
    sponsored = labels.get("sponsored") if isinstance(labels, dict) else {}
    sponsored = sponsored if isinstance(sponsored, dict) else {}

    objective = str(record.get("objective") or "").strip().lower()
    landing_page_url = str(record.get("landing_page_url") or "").strip()
    ctr = record.get("ctr")
    cvr = record.get("cvr")
    commercial_intent = str(record.get("commercial_intent") or (record.get("source_signal") or {}).get("commercial_intent") or "").upper()
    buyer = bool(intent.get("buyer"))
    try:
        seller = float(intent.get("seller") or 0.0)
    except (TypeError, ValueError):
        seller = 0.0
    intent_label = str(intent.get("label") or "").lower()
    # Sponsored Search results can have no Ads-Top objective, CTR/CVR, or LP.
    # Accept only an explicit high-confidence SocialCrawl sponsor classification
    # that names the sponsor; views alone still never satisfy this gate.
    try:
        sponsored_probability = float(sponsored.get("p") or 0.0)
    except (TypeError, ValueError):
        sponsored_probability = 0.0
    sponsored_brand = str(sponsored.get("brand") or "").strip()
    high_confidence_sponsored = bool(
        sponsored_brand
        and (bool(sponsored.get("disclosed")) or bool(sponsored.get("undisclosed")))
        and sponsored_probability >= 0.70
    )

    score = 0.0
    tags: list[str] = []
    strong_signals = 0
    conversion_objectives = (
        "conversion", "conversions", "lead", "leads", "sales", "sale",
        "purchase", "app", "product", "shop", "shopping",
    )

    if objective:
        score += 20.0
        tags.append("objective_signal")
        if any(token in objective for token in conversion_objectives):
            score += 20.0
            strong_signals += 1
            tags.append("conversion_objective_signal")

    if landing_page_url:
        score += 20.0
        strong_signals += 1
        tags.append("landing_page_signal")

    numeric_signals: list[str] = []
    for name, value in (("ctr", ctr), ("cvr", cvr)):
        try:
            parsed = float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None and parsed > 0:
            numeric_signals.append(name)
    if numeric_signals:
        score += min(30.0, 15.0 * len(numeric_signals))
        strong_signals += len(numeric_signals)
        tags.extend(f"{name}_signal" for name in numeric_signals)

    if commercial_intent == "HIGH":
        score += 20.0
        strong_signals += 1
        tags.append("high_commercial_intent_signal")
    elif commercial_intent == "MEDIUM":
        score += 8.0
        tags.append("medium_commercial_intent_signal")

    if high_confidence_sponsored:
        score += 30.0
        strong_signals += 1
        tags.append("high_confidence_sponsored_signal")

    if buyer:
        score += 20.0
        strong_signals += 1
        tags.append("buyer_intent_signal")
    if seller >= 0.5:
        score += 12.0
        strong_signals += 1
        tags.append("seller_intent_signal")
    elif intent_label in {"lead_generation", "sales", "shopping", "product", "commercial"}:
        score += 8.0
        tags.append("commercial_intent_label")

    if record.get("brand") not in (None, "", "UNKNOWN"):
        score += 5.0
        tags.append("advertiser_identified")

    score = min(100.0, score)
    performance_ok = float(performance_score) >= 65.0
    strong_case = strong_signals >= 1 and score >= 35.0
    metadata_case = bool(
        landing_page_url
        and record.get("brand") not in (None, "", "UNKNOWN")
        and float(performance_score) >= 70.0
    )
    pass_gate = performance_ok and (strong_case or metadata_case or commercial_score >= 70.0)

    # Legacy/imported fixtures keep their historical behavior. Live discovery always
    # stamps discovery_source, so all live source candidates are subject to this gate.
    if not record.get("discovery_source"):
        return True, score, tags + ["money_signal_legacy_unenforced"], "legacy/imported candidate"

    if not performance_ok:
        reason = f"performance below commercial floor: {performance_score:.1f} < 65.0"
    elif not pass_gate:
        reason = "no sufficient source-side money signal before deep analysis"
    else:
        reason = "source-side commercial evidence is sufficient for deep analysis"
    tags.append("money_signal_pass" if pass_gate else "money_signal_hard_block")
    return pass_gate, score, tags, reason


def _duration_value(record: dict[str, Any]) -> float | None:
    candidates = [
        record.get("duration_seconds"),
        (record.get("source_signal") or {}).get("duration_seconds"),
    ]
    raw = record.get("raw") or {}
    if isinstance(raw, dict):
        post = raw.get("post") or {}
        content = post.get("content") or {}
        video = post.get("video") or {}
        ext = post.get("ext") or {}
        ad = ext.get("ad") or {}
        candidates.extend([
            content.get("duration_seconds"),
            content.get("durationSeconds"),
            video.get("duration_seconds"),
            video.get("durationSeconds"),
            ext.get("duration_seconds"),
            ext.get("durationSeconds"),
            ad.get("duration_seconds"),
            ad.get("durationSeconds"),
            post.get("duration_seconds"),
            post.get("durationSeconds"),
            raw.get("duration_seconds"),
            raw.get("durationSeconds"),
        ])
    for value in candidates:
        if value not in (None, ""):
            try:
                parsed = float(value)
                # Treat sub-second API values as unverified metadata, not a real
                # duration. The actual MP4 probe becomes authoritative after download.
                if parsed >= 1.0:
                    return parsed
            except (TypeError, ValueError):
                continue
    return None


def _duration_fit_score(record: dict[str, Any]) -> tuple[float, list[str], bool, float | None]:
    raw = _duration_value(record)
    if raw in (None, ""):
        raw_hint = record.get("duration_raw_seconds")
        if raw_hint in (None, ""):
            # The live adapter may have an old-style record where the bad API value
            # is still stored directly in duration_seconds. Detect that explicitly.
            direct = record.get("duration_seconds")
            try:
                if direct not in (None, "") and 0 < float(direct) < 1.0:
                    raw_hint = direct
            except (TypeError, ValueError):
                pass
        if raw_hint not in (None, ""):
            return 50.0, ["duration_api_value_invalid_requires_probe"], False, None
        return 50.0, ["duration_unknown_requires_verify"], False, None
    duration = _num(raw, -1)
    if duration <= 0:
        return 50.0, ["duration_unknown_requires_verify"], False, None
    if duration > 60:
        return 0.0, ["duration_over_60_hard_block"], True, duration
    if 8 <= duration <= 20:
        return 100.0, ["duration_ideal_8_20s"], False, duration
    if 20 < duration <= 34:
        return 95.0, ["duration_preferred_21_34s"], False, duration
    if 34 < duration <= 45:
        return 80.0, ["duration_acceptable_35_45s"], False, duration
    if 45 < duration <= 60:
        return 65.0, ["duration_soft_46_60s"], False, duration
    return 75.0, ["duration_under_8s"], False, duration


def _longevity_score(record: dict[str, Any]) -> tuple[float, list[str]]:
    s = record.get("source_signal") or {}
    days = _num(record.get("longevity_days", s.get("longevity_days")), 0)
    repeated = bool(record.get("repeated_signal", s.get("repeated_signal", False)))
    variations = _num(record.get("variation_count", s.get("variation_count")), 0)
    boards = len(record.get("discovery_boards") or [])
    hits = _num(record.get("discovery_hits"), 1)
    score = 45.0
    tags: list[str] = []
    if days >= 30:
        score += 20; tags.append("30d_longevity")
    elif days >= 14:
        score += 10; tags.append("14d_longevity")
    if repeated:
        score += 20; tags.append("repeated_signal")
    if variations >= 3:
        score += 10; tags.append("variation_depth")
    if boards >= 2:
        score += 10; tags.append("multi_board_hit")
    elif boards == 1:
        tags.append("single_board_hit")
    if hits >= 2:
        score += 10; tags.append("duplicate_board_confirmation")
    return min(100.0, score), tags


@dataclass
class ArbitrageDecision:
    final_score: float
    performance_score: float
    commercial_score: float
    funnel_fit_score: float
    evidence_score: float
    longevity_score: float
    duration_fit_score: float
    penalties: float
    components: dict[str, Any]
    tags: list[str]
    action: str
    explanation: str
    mechanism_source_score: float = 0.0
    source_value_type: str = "UNKNOWN"


def _funnel_fit(record: dict[str, Any], target_offer: str, prelander: str, landing_page: str) -> tuple[float, list[str]]:
    if not (target_offer or prelander or landing_page):
        return 60.0, ["funnel_context_not_supplied"]
    corpus = " ".join(str(record.get(k, "") or "") for k in ("title", "brand", "industry", "objective")).lower()
    target = " ".join(x for x in (target_offer, prelander, landing_page) if x).lower()
    tokens = {t for t in target.replace("/", " ").replace(",", " ").split() if len(t) >= 4}
    hits = sum(1 for t in tokens if t in corpus)
    if hits >= 4:
        return 90.0, ["strong_metadata_funnel_overlap"]
    if hits >= 2:
        return 78.0, ["moderate_metadata_funnel_overlap"]
    if hits >= 1:
        return 68.0, ["weak_metadata_funnel_overlap"]
    # Unknown at pre-analysis stage is not a reason to destroy a strong Top Ad.
    return 55.0, ["funnel_fit_requires_video_analysis"]


def _mechanism_source_score(record: dict[str, Any], pattern: dict[str, Any] | None = None, *, vertical_relevance: float = 0.0) -> tuple[float, list[str]]:
    """Independent preselection signal for mechanism-source value; not commercial proof."""
    pattern = pattern or {}
    sig = pattern.get("signature") if isinstance(pattern.get("signature"), dict) else pattern
    proof=str(sig.get("proof_type") or "").lower(); hook=str(sig.get("hook_type") or "").lower(); structure=str(sig.get("structure") or "").lower()
    intent=str(sig.get("commercial_intent") or record.get("commercial_intent") or "").upper(); click=str(sig.get("click_intent") or record.get("click_intent") or "").upper()
    score=0.0; tags=[]
    if hook in {"question","problem_reveal","curiosity","result_first","demo_first","social_proof"}: score+=15; tags.append("mechanism_hook_signal")
    if proof in {"demo","testimonial","before_after","authority","data"}: score+=22; tags.append("mechanism_proof_signal")
    if any(x in structure for x in ("problem","proof","transformation","offer")): score+=18; tags.append("mechanism_structure_signal")
    if "problem" in structure and ("proof" in structure or "transformation" in structure): score+=12; tags.append("problem_solution_signal")
    if intent=="HIGH": score+=8; tags.append("commercial_context_signal")
    elif intent=="MEDIUM": score+=4; tags.append("commercial_context_weak")
    if click in {"HIGH","MEDIUM"}: score+=8; tags.append("action_intent_signal")
    if vertical_relevance>=70: score+=10; tags.append("target_context_relevance")
    elif vertical_relevance>=50: score+=5; tags.append("target_context_possible")
    if _num((record.get("source_signal") or {}).get("views"),0)>0: score+=5; tags.append("attention_proxy_only")
    return min(100.0,score),tags


def score_tiktok_candidate(
    record: dict[str, Any],
    population: list[dict[str, Any]],
    *,
    target_offer: str = "",
    prelander: str = "",
    landing_page: str = "",
    vertical: str = "",
    pattern: dict[str, Any] | None = None,
) -> ArbitrageDecision:
    perf, perf_parts, perf_tags = _performance_signal(record, population)
    commercial, commercial_tags = _metadata_commercial_score(record, target_offer, prelander, landing_page)
    funnel, funnel_tags = _funnel_fit(record, target_offer, prelander, landing_page)
    longevity, longevity_tags = _longevity_score(record)
    duration_fit, duration_tags, duration_block, duration_seconds = _duration_fit_score(record)
    vertical_info = classify_vertical(record, vertical) if vertical else {"status": "UNKNOWN", "classification": "DIRECT", "score": 50.0, "hits": [], "excluded": [], "reason": "not_supplied"}
    # UNKNOWN/UNVERIFIED is intentionally neutral pre-video: it is neither approval nor rejection.
    vertical_score = float(vertical_info["score"] if vertical_info.get("classification") != "UNVERIFIED" else 50.0)

    evidence = 0.0
    boards = len(record.get("discovery_boards") or [])
    hits = _num(record.get("discovery_hits"), 1)
    evidence += 35 if boards >= 3 else 25 if boards >= 2 else 15 if boards == 1 else 0
    evidence += 20 if record.get("brand") not in (None, "", "UNKNOWN") else 0
    evidence += 20 if record.get("video_url") else 0
    evidence += 15 if (record.get("source_url") or (record.get("source_signal") or {}).get("source_url")) else 0
    evidence += 10 if (record.get("source_signal") or {}).get("source_kind") else 0
    evidence += 10 if hits >= 2 else 0
    evidence = min(100.0, evidence)

    money_pass, money_score, money_tags, money_reason = _money_signal_gate(
        record, performance_score=perf, commercial_score=commercial
    )
    mechanism_source_score, mechanism_tags = _mechanism_source_score(record, pattern=pattern, vertical_relevance=vertical_score)
    commercial_winner = money_pass and perf >= 65 and commercial >= 55 and vertical_score >= 50
    mechanism_source = mechanism_source_score >= 70
    source_value_type = "BOTH" if commercial_winner and mechanism_source else "COMMERCIAL_WINNER" if commercial_winner else "MECHANISM_SOURCE" if mechanism_source else "NONE"

    penalties = 0.0
    if duration_block:
        penalties += 100.0
    if perf < 35:
        penalties += 12.0
    # No video is a major issue for deep analysis, but not a hard reject if the source URL exists.
    if not record.get("video_url") and not (record.get("source_url") or (record.get("source_signal") or {}).get("source_url")):
        penalties += 20.0

    # Pre-video score: only use evidence that exists before Gemini sees the video.
    # Commercial/funnel remain report-only until Deep Analysis.
    final = (
        perf * 0.45
        + vertical_score * 0.25
        + duration_fit * 0.10
        + evidence * 0.10
        + longevity * 0.10
        - penalties
    )
    final = max(0.0, min(100.0, final))

    classification = vertical_info.get("classification", "DIRECT") if vertical else "DIRECT"
    if classification == "HARD_EXCLUDE" or vertical_info.get("status") == "REJECT":
        action = "REJECT"
    elif not money_pass and record.get("discovery_source") and not mechanism_source:
        # Direct scorer keeps the historical Commercial Winner gate. The live
        # orchestrator may reopen a candidate only after Pattern Scout supplies
        # an independent mechanism-source score.
        action = "REJECT"
    elif classification == "UNVERIFIED" and perf >= 85:
        # High-signal unknowns go to Deep for source verification.
        action = "DEEP"
    elif classification == "UNVERIFIED" and perf >= 70:
        action = "SHALLOW"
    elif final >= 65 and (
        (vertical_info["status"] == "HIGH" and perf >= 65)
        or (vertical_info["status"] == "MEDIUM" and perf >= 85)
        or (not vertical and perf >= 65)
    ):
        action = "DEEP"
    elif classification in {"DIRECT", "EXPANDED"} and vertical_info["status"] in {"HIGH", "MEDIUM", "LOW"} and perf >= 60:
        action = "SHALLOW"
    else:
        # Do not hard-reject a non-excluded unknown merely because source metadata is sparse.
        action = "REJECT" if perf < 55 else "SHALLOW"
    if duration_block and mechanism_source:
        action = "DEEP"
        mechanism_tags.append("long_source_allowed_for_mechanism")
    elif duration_block:
        action = "REJECT"
    tags = list(dict.fromkeys(perf_tags + commercial_tags + funnel_tags + longevity_tags + duration_tags + money_tags + mechanism_tags))
    explanation = (
        f"Performance {perf:.0f}, vertical {vertical_score:.0f} ({vertical_info['status']}), commercial metadata {commercial:.0f}, "
        f"funnel {funnel:.0f}, evidence {evidence:.0f}, duration {duration_fit:.0f}, penalties {penalties:.0f}. "
        f"Pre-video weights: performance 45%, vertical relevance 25%, duration 10%, evidence 10%, longevity 10%. "
        f"Money gate={"PASS" if money_pass else "FAIL"} ({money_score:.0f}): {money_reason}."
    )
    if duration_seconds is None:
        tags.append("duration_requires_local_probe")
    if classification == "UNVERIFIED":
        tags.append("source_vertical_requires_video_verification")
    elif classification == "EXPANDED":
        tags.append("adjacent_vertical_signal")

    components = {
        "performance": perf_parts,
        "vertical": vertical_info,
        "commercial": commercial,
        "money_signal_gate": {"pass": money_pass, "score": money_score, "hard_block": not money_pass, "reason": money_reason, "tags": money_tags},
        "funnel": funnel,
        "evidence": evidence,
        "longevity": longevity,
        "duration": {"seconds": duration_seconds, "fit_score": duration_fit, "hard_block": duration_block, "long_source_exception": bool(duration_block and mechanism_source)},
        "mechanism_source": {"score": mechanism_source_score, "eligible": mechanism_source, "source_value_type": source_value_type, "tags": mechanism_tags},
        "weights": {"performance": 0.45, "vertical": 0.25, "duration": 0.10, "evidence": 0.10, "longevity": 0.10},
    }
    return ArbitrageDecision(final, perf, commercial, funnel, evidence, longevity, duration_fit, penalties, components, tags, action, explanation, mechanism_source_score, source_value_type)
