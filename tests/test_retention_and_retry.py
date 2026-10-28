import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import cme.radar.find  # noqa: F401  sets CME_VERTICALS_DIR before the legacy import
from cme.radar.legacy.discover import _fetch_plan, _merge_item, discover_and_rank
from cme.radar.legacy.discovery.tiktok_socialcrawl import SocialCrawlError, SocialCrawlTikTokTopAds, parse_retention
from cme.radar.socialcrawl_client import RetryingSocialCrawlTikTokTopAds, normalize_retention_curve
from cme.radar.find import _retention_execution_plan, to_candidate
from cme.run import qa_retry_feedback


def board(items=None):
    return {"success": True, "credits_used": 20, "request_id": "r1",
            "data": {"items": items or []}}


class RecordingClient(SocialCrawlTikTokTopAds):
    def __init__(self):
        super().__init__(api_key="test")
        self.sent = []

    def _get(self, params):
        self.sent.append(dict(params))
        return board()


class RequestRetryTests(unittest.TestCase):
    @staticmethod
    def response(status=200, payload=None, headers=None):
        response = Mock()
        response.status_code = status
        response.ok = 200 <= status < 400
        response.headers = headers or {}
        response.json.return_value = payload or {"success": True, "data": {"items": []}}
        return response

    def test_retention_timeout_retries_with_same_idempotency_key(self):
        client = RetryingSocialCrawlTikTokTopAds(api_key="test")
        with patch("cme.radar.socialcrawl_client.requests.get",
                   side_effect=[requests.exceptions.ReadTimeout("slow"), self.response()]) as get, \
             patch("cme.radar.socialcrawl_client.time.sleep"):
            payload = client._request_json(client.BASE_URL, {"include": "retention", "order_by": "play_6s_rate"})
        self.assertTrue(payload["success"])
        self.assertEqual(get.call_count, 2)
        calls = [call.kwargs for call in get.call_args_list]
        self.assertEqual([call["timeout"] for call in calls], [112.0, 112.0])
        self.assertEqual(calls[0]["headers"]["Idempotency-Key"], calls[1]["headers"]["Idempotency-Key"])
        self.assertEqual(client.last_request_diagnostics["attempt_count"], 2)
        self.assertEqual(client.last_request_diagnostics["retry_count"], 1)

    def test_non_retention_request_keeps_45s_timeout_and_no_retry(self):
        client = RetryingSocialCrawlTikTokTopAds(api_key="test")
        with patch("cme.radar.socialcrawl_client.requests.get",
                   side_effect=requests.exceptions.ReadTimeout("slow")) as get:
            with self.assertRaises(requests.exceptions.ReadTimeout):
                client._request_json(client.BASE_URL, {"order_by": "for_you"})
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.kwargs["timeout"], 45.0)

    def test_retention_retries_transient_http_status_and_honors_retry_after(self):
        client = RetryingSocialCrawlTikTokTopAds(api_key="test")
        first = self.response(status=429, payload={"success": False}, headers={"Retry-After": "0"})
        second = self.response()
        with patch("cme.radar.socialcrawl_client.requests.get",
                   side_effect=[first, second]) as get, \
             patch("cme.radar.socialcrawl_client.time.sleep") as sleep:
            client._request_json(client.BASE_URL, {"include": "retention", "order_by": "play_2s_rate"})
        self.assertEqual(get.call_count, 2)
        self.assertEqual(sleep.call_args.args, (0.0,))
        self.assertEqual(
            get.call_args_list[0].kwargs["headers"]["Idempotency-Key"],
            get.call_args_list[1].kwargs["headers"]["Idempotency-Key"],
        )

    def test_exhausted_retention_retries_keep_attempt_diagnostics(self):
        client = RetryingSocialCrawlTikTokTopAds(api_key="test")
        with patch("cme.radar.socialcrawl_client.requests.get",
                   side_effect=requests.exceptions.ReadTimeout("slow")) as get, \
             patch("cme.radar.socialcrawl_client.time.sleep"):
            with self.assertRaises(SocialCrawlError) as raised:
                client._request_json(client.BASE_URL, {"include": "retention", "order_by": "play_6s_rate"})
        self.assertEqual(get.call_count, 4)
        diag = raised.exception.diagnostics
        self.assertEqual(diag["attempt_count"], 4)
        self.assertEqual(diag["retry_count"], 3)
        self.assertTrue(diag["idempotency_key_reused_across_retries"])
        self.assertEqual(len({call.kwargs["headers"]["Idempotency-Key"] for call in get.call_args_list}), 1)


class ParseRetentionTests(unittest.TestCase):
    def test_nested_values_keep_scale_and_provenance(self):
        out = parse_retention({"retention": [1, 0.8, 0.5], "retention_2s": "0.62", "retention_6s": 0})
        self.assertEqual(out["retention_curve"], [1.0, 0.8, 0.5])
        self.assertEqual(out["retention_2s"], 0.62)
        self.assertEqual(out["retention_6s"], 0.0)
        self.assertEqual(out["retention_provenance"]["retention_6s"], "post.ext.ad.retention_6s")

    def test_missing_curve_is_none_not_invented(self):
        out = parse_retention({})
        self.assertIsNone(out["retention_curve"])
        self.assertIsNone(out["retention_2s"])
        self.assertIsNone(out["retention_6s"])

    def test_invalid_points_are_dropped_and_counted(self):
        out = parse_retention({"retention": [0.9, None, "x", -1, True, float("nan"), 0.4],
                               "retention_2s": True, "retention_6s": -0.1})
        self.assertEqual(out["retention_curve"], [0.9, 0.4])
        self.assertEqual(out["retention_curve_invalid_points"], 5)
        self.assertIsNone(out["retention_2s"])
        self.assertIsNone(out["retention_6s"])
        self.assertTrue(out["retention_provenance"]["retention_2s"].endswith(":invalid"))

    def test_unsupported_shape_is_recorded_not_guessed(self):
        out = parse_retention({"retention": {"unknown": 1}})
        self.assertIsNone(out["retention_curve"])
        self.assertIn("unsupported_dict", out["retention_provenance"]["retention_curve"])

    def test_normalizer_reads_post_ext_ad(self):
        item = {"post": {"id": "1", "ext": {"ad": {"retention_2s": 0.7, "retention": [1, 0.7]}}}}
        row = SocialCrawlTikTokTopAds._unwrap_item(item)
        self.assertEqual(row["retention_2s"], 0.7)
        self.assertEqual(row["retention_curve"], [1.0, 0.7])
        self.assertIsNone(row["play_2s_rate"])


class FetchBoardRetentionTests(unittest.TestCase):
    def test_default_request_has_no_include(self):
        client = RecordingClient()
        client.fetch_board(order_by="play_6s_rate")
        self.assertNotIn("include", client.sent[0])

    def test_playback_board_adds_include_retention(self):
        client = RecordingClient()
        client.fetch_board(order_by="play_2s_rate", include_retention=True, ad_language=None)
        self.assertEqual(client.sent[0]["include"], "retention")

    def test_keyword_or_industry_never_mixed_with_retention(self):
        client = RecordingClient()
        with self.assertRaises(ValueError):
            client.fetch_board(order_by="play_6s_rate", include_retention=True, keyword="car vacuum")
        with self.assertRaises(ValueError):
            client.fetch_board(order_by="play_6s_rate", include_retention=True, industry="auto")
        self.assertEqual(client.sent, [])

    def test_non_playback_board_is_refused(self):
        client = RecordingClient()
        with self.assertRaises(ValueError):
            client.fetch_board(order_by="ctr", include_retention=True)


class FakeBoardClient:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def fetch_board(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return {"items": [], "credits_used": 1, "request_id": "x", "params": {}}


class FetchPlanRetentionTests(unittest.TestCase):
    def run_plan(self, plan, retention_boards):
        client, calls = FakeBoardClient(), []
        _fetch_plan(client=client, country="US", period=30, industry="", ad_language="en", limit_per_board=10,
                    plan=plan, combined={}, calls=calls, total_credits=0, retention_boards=retention_boards)
        return client, calls

    def test_only_keywordless_playback_boards_request_retention(self):
        client, calls = self.run_plan(
            [("play_6s_rate", None), ("ctr", None), ("play_2s_rate", "car vacuum")],
            frozenset({"play_2s_rate", "play_6s_rate"}))
        self.assertTrue(client.calls[0].get("include_retention"))
        self.assertIsNone(client.calls[0].get("keyword"))
        self.assertIsNone(client.calls[0].get("industry"))
        self.assertNotIn("include_retention", client.calls[1])
        self.assertNotIn("include_retention", client.calls[2])
        self.assertEqual([c["retention_requested"] for c in calls], [True, False, False])

    def test_off_by_default(self):
        client, calls = self.run_plan([("play_6s_rate", None)], None)
        self.assertNotIn("include_retention", client.calls[0])
        self.assertFalse(calls[0]["retention_requested"])

    def test_target_filtered_boards_and_retention_keep_separate_positions(self):
        base = {
            "ad_id": "same-ad",
            "title": "Cordless car vacuum",
            "source_url": "https://www.tiktok.com/@brand/video/same-ad",
            "video_url": "https://cdn.example/video.mp4",
        }
        responses = [
            {"items": [dict(base)], "credits_used": 1, "request_id": "a", "params": {}},
            {"items": [dict(base)], "credits_used": 1, "request_id": "b", "params": {}},
            {"items": [{**base, "retention_2s": 0.7, "retention_6s": 0.3,
                        "retention_curve": [1.0, 0.7, 0.3]}],
             "credits_used": 20, "request_id": "c", "params": {}},
        ]
        client, calls, combined = FakeBoardClient(responses), [], {}
        _fetch_plan(
            client=client, country="US", period=30, industry="", ad_language="en",
            limit_per_board=10,
            plan=[
                ("play_6s_rate", "car vacuum"),
                ("play_2s_rate", "cordless car vacuum"),
                ("play_6s_rate", None),
            ],
            combined=combined, calls=calls, total_credits=0,
            retention_boards=frozenset({"play_2s_rate", "play_6s_rate"}),
        )
        self.assertEqual(client.calls[0]["keyword"], "car vacuum")
        self.assertNotIn("include_retention", client.calls[0])
        self.assertEqual(client.calls[1]["keyword"], "cordless car vacuum")
        self.assertNotIn("include_retention", client.calls[1])
        self.assertTrue(client.calls[2]["include_retention"])
        self.assertIsNone(client.calls[2]["keyword"])
        self.assertEqual([call["retention_requested"] for call in calls], [False, False, True])
        positions = combined["same-ad"]["discovery_board_positions"]
        self.assertEqual(positions["play_6s_rate|keyword:car vacuum"], 1)
        self.assertEqual(positions["play_2s_rate|keyword:cordless car vacuum"], 1)
        self.assertEqual(positions["play_6s_rate|retention"], 1)

    def test_dual_playback_retention_discovery_uses_filtered_boards_then_retention(self):
        class Client(SocialCrawlTikTokTopAds):
            def __init__(self):
                super().__init__(api_key="test")
                self.calls = []
                self.last_discovery_stats = {}

            def fetch_sponsored_search(self, **kwargs):
                self.calls.append({"source": "sponsored_search", **kwargs})
                return {"items": [], "credits_used": 0, "request_id": "search", "params": {}}

            def fetch_board(self, **kwargs):
                self.calls.append({"source": "ads_top", **kwargs})
                return {"items": [], "credits_used": 0, "request_id": "board", "params": {}}

        client = Client()
        with patch("cme.radar.legacy.discover._ordered_supply_queries",
                   return_value=["car vacuum", "cordless car vacuum"]), \
             patch("cme.radar.legacy.discover._advance_supply_query_cursor"):
            discover_and_rank(
                client=client, country="US", period=30, keywords=["car vacuum"],
                order_bys=["for_you"], industry="", ad_language="en", limit_per_board=10,
                max_ads=3, max_requests=5, keyword_boards=5, vertical="Physical Products",
                target_offer="Fanttik V8 Mate Cordless Car Vacuum",
                product_type="cordless handheld car vacuum", discovery_mode="auto",
                supply_search=True, performance_order_bys=["play_6s_rate", "play_2s_rate"],
                include_retention=True,
            )
        self.assertEqual([call["source"] for call in client.calls],
                         ["sponsored_search", "sponsored_search", "ads_top", "ads_top", "ads_top"])
        board_calls = [call for call in client.calls if call["source"] == "ads_top"]
        self.assertEqual(
            [(call["order_by"], call.get("keyword"), bool(call.get("include_retention")))
             for call in board_calls],
            [
                ("play_6s_rate", "car vacuum", False),
                ("play_2s_rate", "cordless car vacuum", False),
                ("play_6s_rate", None, True),
            ],
        )

    def test_dual_playback_retention_needs_five_requests(self):
        class Client(SocialCrawlTikTokTopAds):
            def __init__(self):
                super().__init__(api_key="test")
                self.last_discovery_stats = {}

        with self.assertRaisesRegex(ValueError, "find_requests >= 5"):
            discover_and_rank(
                client=Client(), country="US", period=30, keywords=["car vacuum"],
                order_bys=["for_you"], industry="", ad_language="en", limit_per_board=10,
                max_ads=3, max_requests=4, keyword_boards=4, vertical="Physical Products",
                target_offer="Fanttik V8 Mate Cordless Car Vacuum",
                product_type="cordless handheld car vacuum", discovery_mode="auto",
                supply_search=True, performance_order_bys=["play_6s_rate", "play_2s_rate"],
                include_retention=True,
            )


class RetentionMergeTests(unittest.TestCase):
    def test_retention_fields_and_provenance_merge_after_ordinary_board(self):
        combined = {}
        _merge_item(combined, {
            "ad_id": "same-ad", "retention_2s": None, "retention_6s": None, "retention_curve": None,
            "retention_curve_invalid_points": 0, "retention_provenance": {},
        }, "ctr", None)
        _merge_item(combined, {
            "ad_id": "same-ad", "retention_2s": 0.7, "retention_6s": 0.3,
            "retention_curve": [1.0, 0.7, 0.3], "retention_curve_invalid_points": 1,
            "retention_provenance": {"retention_2s":"post.ext.ad.retention_2s","retention_6s":"post.ext.ad.retention_6s","retention_curve":"post.ext.ad.retention"},
        }, "play_6s_rate", None, retention_requested=True)
        row = combined["same-ad"]
        self.assertEqual(row["retention_2s"], 0.7)
        self.assertEqual(row["retention_6s"], 0.3)
        self.assertEqual(row["retention_curve"], [1.0, 0.7, 0.3])
        self.assertEqual(row["retention_curve_invalid_points"], 1)
        self.assertEqual(row["retention_provenance"]["retention_curve"], "post.ext.ad.retention")

    def test_empty_or_invalid_duplicate_does_not_erase_valid_curve(self):
        combined = {}
        _merge_item(combined, {
            "ad_id": "same-ad", "retention_2s": 0.7, "retention_6s": 0.3,
            "retention_curve": [1.0, 0.7, 0.3], "retention_curve_invalid_points": 0,
            "retention_provenance": {"retention_2s":"post.ext.ad.retention_2s","retention_6s":"post.ext.ad.retention_6s","retention_curve":"post.ext.ad.retention"},
        }, "play_6s_rate", None)
        _merge_item(combined, {
            "ad_id": "same-ad", "retention_2s": None, "retention_6s": None, "retention_curve": None,
            "retention_curve_invalid_points": 3,
            "retention_provenance": {"retention_2s":"post.ext.ad.retention_2s:invalid","retention_curve":"post.ext.ad.retention:no_valid_points"},
        }, "play_2s_rate", None, retention_requested=True)
        row = combined["same-ad"]
        self.assertEqual(row["retention_2s"], 0.7)
        self.assertEqual(row["retention_curve"], [1.0, 0.7, 0.3])
        self.assertEqual(row["retention_provenance"]["retention_2s"], "post.ext.ad.retention_2s")
        self.assertEqual(row["retention_provenance"]["retention_curve"], "post.ext.ad.retention")


class RetryFeedbackForMustJudgeChecks(unittest.TestCase):
    def test_failed_must_mechanism_check_gets_a_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipts_dir = Path(tmp) / "receipts"
            receipts_dir.mkdir()
            rid = "drec_" + "c" * 24
            (receipts_dir / f"{rid}.json").write_text(json.dumps({
                "receipt_id": rid, "overall": "FAILED_QA",
                "rows": [{"atom_id": "mech_result_1", "priority": "MUST", "channel": "image", "status": "FAILED_QA"},
                         {"atom_id": "comp_1", "priority": "SHOULD", "channel": "image", "status": "FAILED_QA"}],
                "checks": [
                    {"check_id": "mech_result_1", "status": "FAIL", "required": False, "detail": "No cleaned strip visible."},
                    {"check_id": "comp_1", "status": "FAIL", "required": False, "detail": "Layout differs."},
                ]}), encoding="utf-8")
            state = type("S", (), {"attempts": [{"shot": 1, "receipt_id": rid, "status": "FAILED_QA"}]})()
            feedback = qa_retry_feedback(state, receipts_dir, 1)
            self.assertEqual(len(feedback), 1)
            self.assertIn("mech_result_1", feedback[0])
            self.assertIn("No cleaned strip visible", feedback[0])
            self.assertNotIn("comp_1", feedback[0])


class RetentionCurveNormalizationTests(unittest.TestCase):
    def test_numeric_curve_preserves_time_positions_with_invalid_points(self):
        points, invalid, shape = normalize_retention_curve([0.91, None, "bad", 0.42])
        self.assertEqual(points, [0.91, None, None, 0.42])
        self.assertEqual(invalid, 2)
        self.assertEqual(shape, "list:scalar")

    def test_structured_curve_uses_explicit_second_and_keeps_gaps(self):
        points, invalid, shape = normalize_retention_curve([
            {"second": 1, "retention": 0.9},
            {"second": 3, "retention": 0.55},
        ])
        self.assertEqual(points, [0.9, None, 0.55])
        self.assertEqual(shape, "list:object:explicit_seconds")
        self.assertGreaterEqual(invalid, 1)

    def test_second_keyed_mapping_is_ordered_by_second(self):
        points, invalid, shape = normalize_retention_curve({"3": 0.4, "1": 0.9})
        self.assertEqual(points, [0.9, None, 0.4])
        self.assertEqual(shape, "mapping:second_keys")
        self.assertEqual(invalid, 1)

    def test_unknown_curve_shape_is_not_guessed(self):
        points, invalid, shape = normalize_retention_curve({"opaque": "unknown"})
        self.assertIsNone(points)
        self.assertEqual(shape, "unsupported_dict")


    def test_retention_call_respects_requested_board_limit(self):
        response = {"items": [], "credits_used": 20, "request_id": "r1", "params": {},
                    "raw_playback_field_diagnostics": {}}
        client = RetryingSocialCrawlTikTokTopAds(api_key="test")
        with patch.object(SocialCrawlTikTokTopAds, "fetch_board", return_value=response) as base_fetch:
            client.fetch_board(country="US", period=30, order_by="play_6s_rate",
                               include_retention=True, limit=10)
        self.assertEqual(base_fetch.call_args.kwargs["limit"], 10)

class RetentionExecutionPolicyTests(unittest.TestCase):
    def test_both_board_request_preserves_distinct_retention_board_rankings(self):
        order_bys, max_requests, strategy = _retention_execution_plan(
            ["play_6s_rate", "play_2s_rate"], True, 5)
        self.assertEqual(order_bys, ["play_6s_rate", "play_2s_rate"])
        self.assertEqual(max_requests, 5)
        self.assertEqual(strategy["mode"], "targeted_playback_boards_plus_retention")
        self.assertEqual(strategy["effective_max_requests"], 5)
        self.assertEqual(
            strategy["targeted_board_order_bys"], ["play_6s_rate", "play_2s_rate"])
        self.assertEqual(strategy["retention_calls"], 1)
        self.assertTrue(strategy["targeted_board_calls_are_keyword_filtered"])
        self.assertTrue(strategy["retention_call_is_unfiltered"])
        self.assertEqual(strategy["retention_request_order_by"], "play_6s_rate")

    def test_without_retention_does_not_rewrite_requested_board_order(self):
        order_bys, max_requests, strategy = _retention_execution_plan(
            ["play_6s_rate", "play_2s_rate"], False, 5)
        self.assertEqual(order_bys, ["play_6s_rate", "play_2s_rate"])
        self.assertEqual(max_requests, 5)
        self.assertEqual(strategy["mode"], "operator_requested_order")

    def test_dual_retention_policy_requires_two_searches_both_boards_and_retention_slot(self):
        for request_count in (3, 4):
            with self.subTest(request_count=request_count):
                with self.assertRaisesRegex(ValueError, "find_requests >= 5"):
                    _retention_execution_plan(["play_2s_rate", "play_6s_rate"], True, request_count)


class RetentionCandidatePropagationTests(unittest.TestCase):
    def test_retention_metrics_keep_ads_top_provenance_and_not_play_rate_labels(self):
        record = {
            "ad_id": "ad-retention-1",
            "title": "Cordless car vacuum",
            "source_url": "https://www.tiktok.com/@brand/video/ad-retention-1",
            "video_url": "https://cdn.example/video.mp4",
            "discovery_source": "tiktok_sponsored_search",
            "discovery_boards": ["sponsored_search", "play_6s_rate"],
            "retention_2s": 0.71,
            "retention_6s": 0.43,
            "retention_curve": [0.9, 0.71, None, 0.52, 0.43],
            "retention_curve_invalid_points": 1,
            "retention_provenance": {
                "retention_2s": "post.ext.ad.retention_2s",
                "retention_6s": "post.ext.ad.retention_6s",
                "retention_curve": "post.ext.ad.retention",
            },
        }
        candidate = to_candidate({"record": record, "decision": None}, {"country": "US", "period": 30}, "2026-10-28T00:00:00Z", "found")
        self.assertEqual(candidate["metrics"]["retention_2s"]["value"], 0.71)
        self.assertEqual(candidate["metrics"]["retention_2s"]["source"], "socialcrawl:tiktok_ads_top")
        self.assertEqual(candidate["metrics"]["retention_2s"]["boards"], ["play_6s_rate"])
        self.assertNotIn("play_6s_rate", candidate["metrics"])
        self.assertEqual(candidate["retention_evidence"]["status"], "AVAILABLE")
        self.assertFalse(candidate["retention_evidence"]["is_exact_play_rate"])
        self.assertFalse(candidate["retention_evidence"]["commercial_result_confirmed"])


if __name__ == "__main__":
    unittest.main()
