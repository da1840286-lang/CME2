"""Offline acceptance tests for the separate FIND Selection layer."""
from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cme.radar.attention import attention_absolute_v1
from cme.radar.fit import ADJACENT_MECHANISM, OFF_TOPIC, TARGET_PRODUCT, classify_source_role
from cme.radar.find import run_find
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
