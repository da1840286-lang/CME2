import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import cme.radar.find  # noqa: F401  sets CME_VERTICALS_DIR before the legacy import
from cme.radar.legacy.discover import _fetch_plan, _merge_item
from cme.radar.legacy.discovery.tiktok_socialcrawl import SocialCrawlError, SocialCrawlTikTokTopAds, parse_retention
from cme.radar.socialcrawl_client import RetryingSocialCrawlTikTokTopAds
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
    def __init__(self):
        self.calls = []

    def fetch_board(self, **kwargs):
        self.calls.append(kwargs)
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


if __name__ == "__main__":
    unittest.main()
