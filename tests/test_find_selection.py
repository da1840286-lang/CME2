"""Offline acceptance tests for the separate FIND Selection layer."""
from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cme.radar.attention import attention_absolute_v1, attention_rank
from cme.radar.fit import ADJACENT_MECHANISM, OFF_TOPIC, TARGET_PRODUCT, classify_source_role, match_source_to_product_query
from cme.radar.find import run_find
from cme.radar.legacy import discover as legacy_discover
from cme.radar.select import pattern_source_reason
from cme.radar.seen import SeenLedger, build_scope


class FitTests(unittest.TestCase):
    def test_target_product_requires_source_body_promotion(self):
        text = "Cordless handheld car vacuum removes pet hair with powerful suction. Get yours today."
        record = {"title": text, "raw": {"post": {"content": {"text": text}}},
                  "discovery_query": "air duster", "discovery_keywords": ["air duster"]}
        result = classify_source_role(record, product_type="cordless handheld car vacuum")
        self.assertEqual(result["role"], TARGET_PRODUCT)
        self.assertTrue(result["promotion_evidence"])

    def test_real_car_vacuum_sponsored_copy_is_not_lost_on_plural_or_compound_hashtags(self):
        examples = [
            "my car been so messy ngl new vacuum here we go #car #carvacuumcleaner #vacuumcleaner",
            "It Blows AND Vacuums and it's on Black Friday Sale #carvacuumcleaner",
        ]
        for text in examples:
            with self.subTest(text=text):
                result = classify_source_role({"title": text}, product_type="cordless handheld car vacuum")
                self.assertEqual(result["role"], TARGET_PRODUCT)
                self.assertTrue(result["promotion_evidence"])
                self.assertTrue(result["target_promotion_evidence"])

    def test_adjacent_product_is_not_accepted_from_query_alone(self):
        text = "Rechargeable air duster blows dust from car vents in seconds. Get yours today."
        record = {"title": text, "raw": {"post": {"content": {"text": text}}},
                  "discovery_query": "cordless car vacuum", "discovery_keywords": ["cordless car vacuum"]}
        self.assertEqual(classify_source_role(record, product_type="cordless handheld car vacuum")["role"], ADJACENT_MECHANISM)
        query_only = {"title": "Watch until the end",
                      "raw": {"post": {"content": {"text": "Watch until the end"}}},
                      "discovery_query": "air duster", "discovery_keywords": ["air duster"]}
        self.assertEqual(classify_source_role(query_only, product_type="cordless handheld car vacuum")["role"], OFF_TOPIC)

    def test_target_noun_as_context_does_not_override_adjacent_product(self):
        text = "Use your car vacuum carefully. This air duster blows dust from vents. Get yours today."
        record = {"title": text, "raw": {"post": {"content": {"text": text}}}}
        result = classify_source_role(record, product_type="cordless handheld car vacuum")
        self.assertEqual(result["role"], ADJACENT_MECHANISM)

    def test_product_noun_without_promotion_is_not_a_match(self):
        record = {"title": "An air duster on a desk", "raw": {"post": {"content": {"text": "An air duster on a desk"}}}}
        result = classify_source_role(record, product_type="cordless handheld car vacuum")
        self.assertEqual(result["role"], OFF_TOPIC)
        self.assertFalse(result["promotion_evidence"])

    def test_store_account_name_is_not_promotion_evidence_by_itself(self):
        record = {"title": "A vacuum on a desk @Home Store"}
        result = classify_source_role(record, product_type="cordless handheld car vacuum")
        self.assertEqual(result["role"], OFF_TOPIC)
        self.assertFalse(result["promotion_evidence"])


class AttentionTests(unittest.TestCase):
    def test_missing_metrics_do_not_renormalize_fixed_weights(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        views_only = attention_absolute_v1({"views": 1_000_000}, period_days=30, now=now)
        with_likes = attention_absolute_v1({"views": 1_000_000, "likes": 100_000}, period_days=30, now=now)
        self.assertEqual(views_only["score"], 25.0)
        self.assertEqual(views_only["coverage"], 0.25)
        self.assertEqual(with_likes["score"], 40.0)
        self.assertFalse(views_only["is_commercial_proof"])

    def test_recency_depends_on_requested_period_only_when_date_is_known(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        record = {"views": 500_000, "likes": 5_000, "published_at": (now - timedelta(days=30)).isoformat()}
        self.assertLess(attention_absolute_v1(record, period_days=7, now=now)["score"],
                        attention_absolute_v1(record, period_days=180, now=now)["score"])
        unknown = attention_absolute_v1({"views": 500_000}, period_days=7, now=now)
        self.assertEqual(unknown["recency_status"], "UNKNOWN_NO_PENALTY")


class SeenLedgerTests(unittest.TestCase):
    def test_scope_isolated_and_ttl_prunes_old_identity(self):
        now = [1_800_000_000.0]
        with tempfile.TemporaryDirectory() as temp:
            ledger = SeenLedger(Path(temp) / "seen.json", ttl_days=30, clock=lambda: now[0])
            scope = build_scope(country="US", offer={"product_type": "car vacuum"}, selection_mode="mechanism")
            other = build_scope(country="US", offer={"product_type": "air duster"}, selection_mode="mechanism")
            ledger.remember(scope, [{"ad_id": "a1", "video_id": "v1", "source_url": "https://tiktok.com/x/1?sig=abc"}], ["air duster"])
            self.assertIn("video:v1", ledger.seen_keys(scope))
            self.assertEqual(ledger.seen_keys(other), set())
            self.assertEqual(ledger.recent_queries(scope), ["air duster"])
            now[0] += 31 * 86400
            self.assertEqual(ledger.seen_keys(scope), set())
            self.assertEqual(ledger.recent_queries(scope), [])

    def test_missing_or_corrupt_state_fails_open(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seen.json"
            ledger = SeenLedger(path)
            scope = build_scope(country="US", offer={}, selection_mode="product")
            self.assertEqual(ledger.seen_keys(scope), set())
            path.write_text("{broken", encoding="utf-8")
            self.assertEqual(ledger.seen_keys(scope), set())


class FakeClient:
    enabled = True

    def __init__(self, items):
        self.items, self.calls, self.last_discovery_stats = items, [], {}

    def fetch_sponsored_search(self, **kwargs):
        self.calls.append(("sponsored_search", kwargs))
        return {"items": self.items, "credits_used": 1, "request_id": "fixture", "params": kwargs}

    def fetch_board(self, **kwargs):
        self.calls.append(("ads_top", kwargs))
        raise AssertionError("mechanism mode must not call Ads Top")


class MechanismFindTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger_path = Path(self.temp.name) / "seen.json"
        self.target = {"name": "Fanttik car vacuum", "product_type": "cordless handheld car vacuum", "product_category": "vacuum"}

    def item(self, ad_id="a1"):
        text = "Rechargeable air duster blows dust from car vents in seconds. Get yours today."
        return {
            "ad_id": ad_id, "video_id": "v-" + ad_id,
            "source_url": "https://www.tiktok.com/@brand/video/" + ad_id,
            "video_url": "https://cdn.example.test/" + ad_id + ".mp4",
            "title": text, "views": 600_000, "likes": 20_000, "comments": 400, "shares": 600,
            "published_at": "2025-12-20T00:00:00Z",
            "raw": {"post": {"content": {"text": text}, "ext": {"ad": {"title": text}}}}}
    
    def test_missing_keyword_and_playback_order_fail_before_provider_calls(self):
        client = FakeClient([self.item()])
        with self.assertRaisesRegex(ValueError, "explicit keywords"):
            run_find({"selection_mode": "mechanism", "keywords": [], "offer": self.target}, client=client)
        self.assertEqual(client.calls, [])
        with self.assertRaisesRegex(ValueError, "performance order"):
            run_find({"selection_mode": "mechanism", "keywords": ["air duster"], "offer": self.target,
                      "performance_order_bys": ["play_6s_rate"]}, client=client)
        self.assertEqual(client.calls, [])

    def test_mechanism_mode_uses_only_sponsored_search_and_separates_gates(self):
        client = FakeClient([self.item()])
        params = {"selection_mode": "mechanism", "country": "US", "period": 30, "max_requests": 2,
                  "max_ads": 5, "keywords": ["air duster"], "offer": self.target}
        with mock.patch.dict("os.environ", {"CME_FIND_SEEN_PATH": str(self.ledger_path)}):
            result = run_find(params, client=client, now="2026-01-01T00:00:00Z")
        self.assertEqual([call[0] for call in client.calls], ["sponsored_search"])
        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["source_selection"]["role"], ADJACENT_MECHANISM)
        self.assertTrue(candidate["source_gate_eligible"])
        self.assertEqual(pattern_source_reason(candidate), "")
        self.assertTrue(candidate["transfer_eligible"])
        self.assertEqual(candidate["stage"], "PATTERN_CANDIDATE")
        self.assertEqual(candidate["target_fit_status"], "NOT_EVALUATED")
        self.assertFalse(candidate["source_selection"]["attention_is_commercial_proof"])
        self.assertEqual(result["diagnostics"]["performance_calls_used"], 0)
        self.assertEqual(result["request"]["selection_mode"], "mechanism")

    def test_find_applies_seen_ledger_across_runs(self):
        client = FakeClient([self.item("seen-across-runs")])
        params = {"selection_mode": "mechanism", "country": "US", "period": 30, "max_requests": 1,
                  "max_ads": 5, "keywords": ["air duster"], "offer": self.target}
        with mock.patch.dict("os.environ", {"CME_FIND_SEEN_PATH": str(self.ledger_path)}):
            first = run_find(params, client=client, now="2026-01-01T00:00:00Z")
            second = run_find(params, client=client, now="2026-01-01T00:00:01Z")
        self.assertEqual(len(first["candidates"]), 1)
        self.assertEqual(second["candidates"], [])
        diag = second["diagnostics"]["source_selection_diagnostics"]
        self.assertEqual(diag["rejected_by_reason"].get("SEEN_WITHIN_TTL"), 1)
        self.assertEqual(second["diagnostics"]["performance_calls_used"], 0)
        self.assertEqual(len(client.calls), 2)


class ZipAuditIntegrationRegressionTests(unittest.TestCase):
    def test_product_head_does_not_turn_usage_context_into_product_anchor(self):
        text = "Tire inflator inflates a flat tire in seconds. Get yours today."
        result = classify_source_role(
            {"title": text, "raw": {"post": {"content": {"text": text}}}},
            product_type="tire inflator for cars",
        )
        self.assertEqual(result["target_anchor"], "inflator")
        self.assertEqual(result["role"], TARGET_PRODUCT)

    def test_missing_target_use_context_is_explicitly_low_confidence(self):
        text = "This vacuum removes crumbs in seconds. Get yours today."
        result = classify_source_role(
            {"title": text, "raw": {"post": {"content": {"text": text}}}},
            product_type="cordless handheld car vacuum",
        )
        self.assertEqual(result["role"], TARGET_PRODUCT)
        self.assertFalse(result["context_verified"])
        self.assertEqual(result["confidence"], "LOW")
        self.assertIn("context is not stated", result["reason"])

    def test_compound_product_hashtag_can_support_identity(self):
        text = "Quick cleaning for your car. Get yours today. #carvacuum"
        result = classify_source_role(
            {"title": text, "raw": {"post": {"content": {"text": text}}}},
            product_type="cordless handheld car vacuum",
        )
        self.assertEqual(result["role"], TARGET_PRODUCT)
        self.assertEqual(result["target_mention_source"], "hashtag")

    def test_verb_only_product_hashtag_is_not_noun_evidence(self):
        text = "Cleaning dusty seats. Get yours today. #vacuuming"
        result = classify_source_role(
            {"title": text, "raw": {"post": {"content": {"text": text}}}},
            product_type="cordless handheld car vacuum",
        )
        self.assertEqual(result["role"], OFF_TOPIC)
        self.assertFalse(result["target_promotion_evidence"])


class QueryRotationRegressionTests(unittest.TestCase):
    def test_find_prefers_unseen_query_then_least_recently_used(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger_path = Path(temp) / "seen.json"
            client = FakeClient([{
                "ad_id": "rotation-ad", "video_id": "rotation-video",
                "source_url": "https://www.tiktok.com/@brand/video/rotation-ad",
                "video_url": "https://cdn.example.test/rotation.mp4",
                "title": "Rechargeable air duster blows dust from car vents. Get yours today.",
                "views": 600_000, "likes": 20_000, "comments": 400, "shares": 600,
                "published_at": "2025-12-20T00:00:00Z",
                "raw": {"post": {"content": {"text": "Rechargeable air duster blows dust from car vents. Get yours today."}}},
            }])
            params = {
                "selection_mode": "mechanism", "country": "US", "period": 30, "max_requests": 1,
                "max_ads": 5, "keywords": ["air duster", "portable blower"],
                "offer": {"name": "Car vacuum", "product_type": "car vacuum"},
            }
            with mock.patch.dict("os.environ", {"CME_FIND_SEEN_PATH": str(ledger_path)}):
                run_find(params, client=client, now="2026-01-01T00:00:00Z")
                run_find(params, client=client, now="2026-01-01T00:00:01Z")
            calls = [entry[1]["query"] for entry in client.calls]
            self.assertEqual(calls, ["air duster", "portable blower"])


class MechanismAttentionRankRegressionTests(unittest.TestCase):
    def test_mechanism_rank_uses_fixed_weights_and_exposes_missing_coverage(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = attention_rank({"views": 1_000_000}, period_days=30, now=now)
        self.assertAlmostEqual(result["score"], 25.71, delta=0.02)
        self.assertEqual(result["coverage"], 0.3)
        self.assertIn("amplification", result["missing_components"])
        self.assertIn("recency", result["missing_components"])
        self.assertFalse(result["is_commercial_proof"])

    def test_mechanism_rank_shrinks_very_small_sample_engagement_rates(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        small = attention_rank({"views": 1_000, "likes": 1_000, "published_at": now.isoformat()}, now=now)
        large = attention_rank({"views": 100_000, "likes": 100_000, "published_at": now.isoformat()}, now=now)
        self.assertLess(small["components"]["like_rate"], 3.0)
        self.assertGreater(large["components"]["like_rate"], 60.0)
        self.assertFalse(small["is_commercial_proof"])

    def test_mechanism_rank_recency_uses_requested_period(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        record = {"views": 500_000, "likes": 5_000, "shares": 500,
                  "published_at": (now - timedelta(days=30)).isoformat()}
        recent_window = attention_rank(record, period_days=7, now=now)
        broad_window = attention_rank(record, period_days=180, now=now)
        self.assertGreater(broad_window["score"], recent_window["score"])


class RestoredArchiveCoverageTests(unittest.TestCase):
    """Ported FIND selection checks adapted to this branch's current contracts."""

    def setUp(self):
        self.target = {"name": "Fanttik car vacuum", "product_type": "cordless handheld car vacuum",
                       "product_category": "vacuum"}

    def record(self, text, **extra):
        return {
            "title": text,
            "raw": {"post": {"content": {"text": text}, "ext": {"hashtags": []}}},
            **extra,
        }

    def test_legacy_match_rejects_target_mention_when_other_product_is_sold(self):
        text = "My car vacuum sits on the shelf. This portable pump inflates a flat tire in seconds. Order now!"
        result = legacy_discover._product_type_relevance(
            self.record(text), "cordless handheld car vacuum"
        )
        self.assertEqual(result["status"], "NO_MATCH")
        self.assertIn("same sentence", result["reason"])

    def test_legacy_match_does_not_attach_standalone_cta_to_target_mention(self):
        result = legacy_discover._product_type_relevance(
            self.record("My car vacuum sits on the shelf. Order now!"),
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "NO_MATCH")

    def test_legacy_match_accepts_target_product_pitch_in_same_sentence(self):
        text = "My cordless car vacuum removes crumbs in seconds. Get yours today."
        result = legacy_discover._product_type_relevance(
            self.record(text), "cordless handheld car vacuum"
        )
        self.assertEqual(result["status"], "MATCH")

    def test_legacy_match_does_not_treat_query_as_source_evidence(self):
        result = legacy_discover._product_type_relevance(
            {"title": "Watch until the end", "discovery_query": "cordless car vacuum",
             "discovery_keywords": ["cordless car vacuum"]},
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "NO_MATCH")

    def test_legacy_match_requires_product_pitch(self):
        result = legacy_discover._product_type_relevance(
            self.record("A car vacuum on my shelf."),
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "NO_MATCH")

    def test_legacy_match_rejects_verb_vacuuming(self):
        result = legacy_discover._product_type_relevance(
            self.record("Vacuuming my car today, shop now"),
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "NO_MATCH")

    def test_legacy_match_rejects_verb_only_vacuum_hashtag(self):
        result = legacy_discover._product_type_relevance(
            self.record("Cleaning dusty seats. Get yours today. #vacuuming"),
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "NO_MATCH")

    def test_legacy_match_allows_compound_product_hashtag_with_independent_pitch(self):
        result = legacy_discover._product_type_relevance(
            self.record("Quick cleaning for your car. Get yours today. #carvacuum"),
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(result["product_anchor"], "vacuum")

    def test_legacy_match_keeps_product_family_when_context_is_missing(self):
        result = legacy_discover._product_type_relevance(
            self.record("This vacuum removes crumbs in seconds. Get yours today."),
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(result["confidence"], "LOW")
        self.assertFalse(result["context_verified"])

    def test_carvacuum_hashtag_does_not_verify_car_use_context_for_household_vacuum_copy(self):
        text = "This cordless stick vacuum cleans carpets and pet hair. Get yours today. #carvacuum"
        result = legacy_discover._product_type_relevance(
            self.record(text), "cordless handheld car vacuum"
        )
        self.assertEqual(result["status"], "MATCH")
        self.assertFalse(result["context_verified"])
        self.assertEqual(result["confidence"], "LOW")
        self.assertIn("context-like hashtag is not accepted as proof", result["reason"])

    def test_target_context_improves_role_confidence(self):
        contextual = classify_source_role(
            self.record("This car vacuum removes crumbs in seconds. Get yours today."),
            product_type="cordless handheld car vacuum",
        )
        uncontextual = classify_source_role(
            self.record("This vacuum removes crumbs in seconds. Get yours today."),
            product_type="cordless handheld car vacuum",
        )
        self.assertTrue(contextual["context_verified"])
        self.assertFalse(uncontextual["context_verified"])
        self.assertEqual(contextual["confidence"], "MEDIUM")
        self.assertEqual(uncontextual["confidence"], "LOW")

    def test_promoted_non_target_product_is_classified_as_adjacent(self):
        result = classify_source_role(
            self.record("Portable tire inflator inflates a flat tire in seconds. Get yours today."),
            product_type="cordless handheld car vacuum",
        )
        self.assertEqual(result["role"], ADJACENT_MECHANISM)

    def test_query_match_accepts_advertised_query_product(self):
        result = match_source_to_product_query(
            self.record("Portable air duster blows dust from vents in seconds. Get yours today."),
            "air duster",
        )
        self.assertTrue(result["matches"])
        self.assertEqual(result["query_anchor"], "duster")

    def test_query_match_rejects_mismatched_advertised_product(self):
        result = match_source_to_product_query(
            self.record("Portable air compressor inflates tires in seconds. Get yours today."),
            "air duster",
        )
        self.assertFalse(result["matches"])
        self.assertEqual(result["reason"], "QUERY_PRODUCT_MISMATCH")

    def test_query_match_rejects_incidental_product_mention_without_promotion(self):
        result = match_source_to_product_query(
            self.record("The air duster sits on a shelf. This tire inflator inflates tires in seconds."),
            "air duster",
        )
        self.assertFalse(result["matches"])
        self.assertEqual(result["reason"], "QUERY_PRODUCT_NOT_PROMOTED")

    def test_mechanism_find_rejects_compressor_returned_for_air_duster_query(self):
        text = "Portable air compressor inflates tires in seconds. Get yours today."
        item = {
            "ad_id": "compressor-for-duster-query", "video_id": "compressor-video",
            "source_url": "https://www.tiktok.com/@brand/video/compressor-for-duster-query",
            "video_url": "https://cdn.example.test/compressor.mp4",
            "title": text, "views": 600_000, "likes": 20_000, "comments": 400, "shares": 600,
            "published_at": "2025-12-20T00:00:00Z",
            "raw": {"post": {"content": {"text": text}, "ext": {"ad": {"title": text}}}},
        }
        client = FakeClient([item])
        params = {"selection_mode": "mechanism", "country": "US", "period": 30,
                  "max_requests": 1, "max_ads": 5, "keywords": ["air duster"], "offer": self.target}
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.dict("os.environ", {"CME_FIND_SEEN_PATH": str(Path(temp) / "seen.json")}):
                result = run_find(params, client=client, now="2026-01-01T00:00:00Z")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(
            result["diagnostics"]["source_selection_diagnostics"]["rejected_by_reason"].get("QUERY_PRODUCT_MISMATCH"),
            1,
        )

    def test_unknown_selection_mode_fails_before_provider_call(self):
        client = FakeClient([])
        with self.assertRaisesRegex(ValueError, "selection_mode"):
            run_find({"selection_mode": "everything", "keywords": ["air duster"], "offer": self.target}, client=client)
        self.assertEqual(client.calls, [])

    def test_attention_rank_is_deterministic_for_same_source(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        post = {"views": 220_000, "likes": 530, "shares": 120, "saves": 270,
                "author_followers": 18_000, "published_at": now.isoformat()}
        self.assertEqual(attention_rank(post, now=now), attention_rank(dict(post), now=now))

    def test_attention_reach_component_increases_with_views(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        small = attention_rank({"views": 1_000}, now=now)
        large = attention_rank({"views": 1_000_000}, now=now)
        self.assertGreater(large["components"]["reach"], small["components"]["reach"])
        self.assertGreater(large["score"], small["score"])

    def test_attention_rank_is_unknown_without_usable_views(self):
        result = attention_rank({"likes": 100_000}, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertIsNone(result["score"])
        self.assertEqual(result["coverage"], 0.0)
        self.assertFalse(result["is_commercial_proof"])

    def test_seen_ledger_canonicalizes_source_urls(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = SeenLedger(Path(temp) / "seen.json")
            scope = build_scope(country="US", offer={"product_type": "car vacuum"}, selection_mode="product")
            ledger.remember(scope, [{"source_url": "https://EXAMPLE.com/video/123?campaign=x#top"}])
            self.assertIn("url:https://example.com/video/123", ledger.seen_keys(scope))

    def test_seen_ledger_isolates_product_mode_from_mechanism_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = SeenLedger(Path(temp) / "seen.json")
            offer = {"product_type": "car vacuum"}
            product = build_scope(country="US", offer=offer, selection_mode="product")
            mechanism = build_scope(country="US", offer=offer, selection_mode="mechanism")
            ledger.remember(product, [{"ad_id": "same-ad"}])
            self.assertIn("ad:same-ad", ledger.seen_keys(product))
            self.assertEqual(ledger.seen_keys(mechanism), set())

    def test_seen_ledger_refreshes_query_use_timestamp(self):
        now = [1000.0]
        with tempfile.TemporaryDirectory() as temp:
            ledger = SeenLedger(Path(temp) / "seen.json", clock=lambda: now[0])
            scope = build_scope(country="US", offer={"product_type": "car vacuum"}, selection_mode="mechanism")
            ledger.remember(scope, queries=["air duster", "portable blower"])
            now[0] += 5
            ledger.remember(scope, queries=["air duster"])
            self.assertEqual(ledger.recent_queries(scope), ["portable blower", "air duster"])

    def test_failed_provider_call_does_not_record_query_or_source(self):
        class BrokenClient(FakeClient):
            def fetch_sponsored_search(self, **kwargs):
                self.calls.append(("sponsored_search", kwargs))
                raise OSError("synthetic offline fixture")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seen.json"
            client = BrokenClient([])
            params = {"selection_mode": "mechanism", "country": "US", "period": 30,
                      "max_requests": 1, "max_ads": 5, "keywords": ["air duster"], "offer": self.target}
            with mock.patch.dict("os.environ", {"CME_FIND_SEEN_PATH": str(path)}):
                with self.assertRaises(Exception):
                    run_find(params, client=client, now="2026-01-01T00:00:00Z")
            self.assertFalse(path.exists())
            self.assertEqual(len(client.calls), 1)
