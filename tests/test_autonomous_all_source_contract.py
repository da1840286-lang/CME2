"""Offline, data-backed cross-stage tests for Product/Mechanism admission and autonomous ALL.

The Product fixture is derived from GitHub Actions FIND run 41090668585 on commit
b4d843a1b51025e0ababb71ebe68c8f8688156d7. Provider URLs and query-match fields are
removed deliberately: these are real source records exercising the legacy Product
path where query-product match may be absent. No external API is called.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import cme.radar.find as find_module
from cme.radar.find import _apply_product_selection_gates, run_find, to_candidate
from cme.radar.select import pattern_source_reason, rank_pattern_candidates, sync_pattern_source_status
from tests.test_source_loop import _embedded_all_stage_script

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "real_product_find_unknown_query.json"


def product_candidates_from_real_find_fixture():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    request = {
        "country": "US", "period": 180, "selection_mode": "product",
        "video_duration_range": "all", "target_offer": fixture["target_offer"],
        "product_type": "cordless handheld car vacuum", "product_category": "vacuum",
        "requested_performance_order_bys": [],
    }
    candidates = [
        to_candidate({"record": copy.deepcopy(item["legacy_record"])}, request,
                     "2026-10-31T00:00:00Z", "found")
        for item in fixture["candidates"]
    ]
    accepted, diagnostics = _apply_product_selection_gates(candidates, request)
    return accepted, diagnostics, request


_FAKE_AUTONOMOUS_ALL_RUNNER = r'''
import json, sys
from pathlib import Path
import cme.radar.find as find_module
import cme.radar.server as server

PLAN = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
find_module.run_find = lambda params: PLAN["find_result"]

class Store:
    def ingest(self, items, source, state): pass
    def decide(self, cid, state, reason): pass
    def analyses(self, cid):
        return [{"result": {"mechanism_hypothesis": {"text": "offline source mechanism hypothesis"}}}]
    def close(self): pass

class FakeApp:
    def __init__(self, work, **kwargs):
        self.work = Path(work)
        self.store = Store()
        self.calls = []
        self.analysis_calls = {}
        self.calls_path = Path("fake_app_calls.json")
    def _record(self, stage, cid):
        self.calls.append({"stage": stage, "candidate_id": cid})
        self.calls_path.write_text(json.dumps(self.calls, indent=2), encoding="utf-8")
    def do_analyze(self, cid):
        self._record("analyze", cid)
        self.analysis_calls[cid] = self.analysis_calls.get(cid, 0) + 1
        return {"status": "OK", "analysis_id": self.analysis_calls[cid], "detail": "offline deterministic test adapter"}
    def do_produce(self, cid):
        self._record("produce", cid)
        spec = PLAN["produce"][cid]
        run_id = "offline-" + str(len(self.calls))
        folder = self.work / "production" / run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run_result.json").write_text(json.dumps({"failure_class": spec["failure_class"]}), encoding="utf-8")
        return {"run_id": run_id, "status": spec["status"], "exit_code": spec["exit_code"],
                "failure_class": spec["failure_class"], "failure_detail": spec.get("detail", ""),
                "images": [], "verdicts": [{"overall": spec.get("verdict", "")}],
                "offer_name": "Fanttik V8 Mate Cordless Car Vacuum", "provider_calls_made": 0}

server.RadarApp = FakeApp
exec(compile(Path(sys.argv[2]).read_text(encoding="utf-8"), "live-smoke-all", "exec"), {"__name__": "__main__"})
'''


def execute_autonomous_all(find_result, *, offer, produce_plan, selection_mode, keywords):
    """Run the actual embedded ALL workflow with an offline FIND replay and fake stage adapters."""
    plan = {"find_result": find_result, "produce": produce_plan}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "out").mkdir()
        (root / "out" / "live_offer.json").write_text(json.dumps(offer), encoding="utf-8")
        (root / "event.json").write_text(json.dumps({"inputs": {"offer_json": json.dumps(offer)}}), encoding="utf-8")
        (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (root / "workflow.py").write_text(_embedded_all_stage_script(), encoding="utf-8")
        (root / "runner.py").write_text(_FAKE_AUTONOMOUS_ALL_RUNNER, encoding="utf-8")
        (root / "summary.md").write_text("", encoding="utf-8")
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CME_STAGE": "all", "SOURCE_CONTRACT_PROVIDED": "false",
            "CANDIDATE_ID": "", "FIND_RUN_ID": "", "SHOTS": "1",
            "GITHUB_STEP_SUMMARY": str(root / "summary.md"), "GITHUB_RUN_ID": "offline-contract-test",
            "GITHUB_EVENT_PATH": str(root / "event.json"),
            "FIND_COUNTRY": "US", "FIND_PERIOD": "180", "FIND_REQUESTS": "3",
            "FIND_MAX_ADS": "10", "FIND_KEYWORDS": keywords, "FIND_SELECTION_MODE": selection_mode,
            "FIND_VIDEO_DURATION_RANGE": "all", "FIND_PERFORMANCE_ORDER": "default",
            "FIND_INCLUDE_RETENTION": "false",
        }
        completed = subprocess.run([sys.executable, "runner.py", "plan.json", "workflow.py"],
                                   cwd=root, env=env, capture_output=True, text=True, timeout=60)
        live = json.loads((root / "out" / "live_result.json").read_text(encoding="utf-8")) if (root / "out" / "live_result.json").exists() else {}
        attempts = json.loads((root / "out" / "source_analysis_attempts.json").read_text(encoding="utf-8")) if (root / "out" / "source_analysis_attempts.json").exists() else {}
        calls = json.loads((root / "fake_app_calls.json").read_text(encoding="utf-8")) if (root / "fake_app_calls.json").exists() else []
        return completed, live, attempts, calls


class ProductSourceContractTests(unittest.TestCase):
    def test_standard_product_run_find_path_preserves_unknown_query_as_unknown(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        ranked_rows = []
        for item in fixture["candidates"][:3]:
            record = copy.deepcopy(item["legacy_record"])
            record.pop("query_product_match", None)
            ranked_rows.append({"record": record, "decision": {
                "final_score": 78, "performance_score": 65, "commercial_score": 62,
                "action": "REVIEW", "explanation": "offline archived FIND fixture",
            }})

        class NoNetworkClient:
            enabled = True
            last_discovery_stats = {}

        request = {
            "country": "US", "period": 180, "max_requests": 1, "max_ads": 10,
            "keywords": ["car vacuum"],
            "offer": {"name": fixture["target_offer"], "target_offer": fixture["target_offer"],
                      "product_type": "cordless handheld car vacuum", "product_category": "vacuum"},
            "performance_plan": "legacy_ads_top",
        }
        # Exercise run_find's ordinary Product route and its real to_candidate plus
        # Product Gate code. Only the external discovery/ranking adapter is replayed.
        with mock.patch.object(find_module, "discover_and_rank", return_value=ranked_rows):
            result = run_find(request, client=NoNetworkClient(), now="2026-10-31T00:00:00Z")
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(result["diagnostics"]["outcome"], "CANDIDATES_FOUND")
        self.assertEqual(result["stats"]["product_selection_gate"]["rejected_candidates"], 0)
        for candidate in result["candidates"]:
            self.assertIsNone(candidate["query_product_match"])
            self.assertEqual(candidate["stage"], "PATTERN_CANDIDATE")
            self.assertTrue(candidate["analysis_eligibility"]["eligible"])
            self.assertTrue(candidate["source_gate_eligible"])
            self.assertTrue(candidate["transfer_eligible"])

    def test_real_find_product_sources_without_query_match_reach_all_in_find_order(self):
        candidates, diagnostics, _request = product_candidates_from_real_find_fixture()
        self.assertEqual(len(candidates), 4)
        self.assertEqual(diagnostics["product_selection"]["rejected_candidates"], 0)
        ranked, skipped = rank_pattern_candidates(candidates)
        self.assertEqual([c["id"] for c in ranked], [c["id"] for c in candidates])
        self.assertEqual(skipped, [])
        for candidate in ranked:
            self.assertIsNone(candidate["query_product_match"])
            self.assertEqual(candidate["stage"], "PATTERN_CANDIDATE")
            self.assertTrue(candidate["analysis_eligibility"]["eligible"])
            self.assertTrue(candidate["source_gate_eligible"])
            self.assertTrue(candidate["transfer_eligible"])
            self.assertTrue(candidate["source_selection"]["target_promotion_evidence"])

    def test_explicit_product_query_mismatch_is_still_rejected_and_statuses_sync(self):
        candidates, _diagnostics, request = product_candidates_from_real_find_fixture()
        candidate = copy.deepcopy(candidates[0])
        candidate["query_product_match"] = {"matches": False, "reason": "explicit product query mismatch"}
        accepted, diagnostics = _apply_product_selection_gates(
            [candidate], {**request, "requested_performance_order_bys": []})
        self.assertEqual(accepted, [])
        self.assertEqual(diagnostics["product_selection"]["rejected_candidates"], 1)
        self.assertEqual(candidate["stage"], "SOURCE_REJECTED")
        self.assertFalse(candidate["analysis_eligibility"]["eligible"])
        self.assertFalse(candidate["source_gate_eligible"])
        self.assertFalse(candidate["transfer_eligible"])
        self.assertIn("mismatch", candidate["analysis_eligibility"]["reason"])

    def test_product_requires_target_specific_promotion_evidence(self):
        candidates, _diagnostics, _request = product_candidates_from_real_find_fixture()
        candidate = copy.deepcopy(candidates[0])
        candidate["source_selection"]["target_promotion_evidence"] = False
        reason = sync_pattern_source_status(candidate)
        self.assertIn("target-specific promotion evidence", reason)
        self.assertEqual(candidate["stage"], "SOURCE_REJECTED")
        self.assertFalse(candidate["analysis_eligibility"]["eligible"])
        self.assertFalse(candidate["source_gate_eligible"])
        self.assertFalse(candidate["transfer_eligible"])

    def test_product_gate_recomputes_role_and_does_not_trust_stale_positive_receipt(self):
        candidates, _diagnostics, request = product_candidates_from_real_find_fixture()
        candidate = copy.deepcopy(candidates[0])
        candidate["legacy_record"]["title"] = "A car interior cleaning clip with no product offer or CTA"
        candidate["legacy_record"]["raw"] = {"post": {"content": {"text": candidate["legacy_record"]["title"]}}}
        accepted, _diagnostics = _apply_product_selection_gates(
            [candidate], {**request, "requested_performance_order_bys": []})
        self.assertEqual(accepted, [])
        self.assertEqual(candidate["stage"], "SOURCE_REJECTED")
        self.assertFalse(candidate["source_gate_eligible"])
        self.assertFalse(candidate["transfer_eligible"])

    def test_mechanism_requires_explicit_positive_product_query_match(self):
        candidate = {
            "id": "mechanism-no-query-match", "video_url": "https://fixture.invalid/air-duster.mp4",
            "duration_seconds": 24,
            "request": {"selection_mode": "mechanism", "video_duration_range": "all"},
            "source_selection": {
                "mode": "mechanism", "role": "ADJACENT_MECHANISM",
                "promotion_evidence": True, "target_promotion_evidence": False,
            },
        }
        reason = pattern_source_reason(candidate)
        self.assertIn("explicit positive product-query match", reason)
        candidate["query_product_match"] = {"matches": True, "query_anchor": "air duster"}
        self.assertEqual(pattern_source_reason(candidate), "")


class AutonomousAllDataBackedTests(unittest.TestCase):
    OK = {"status": "OK", "exit_code": 0, "failure_class": "DELIVERED_PASS"}
    REJECT = {"status": "ATOM_CONTRACT_FAILED", "exit_code": 4,
              "failure_class": "TRANSFER_REJECTED",
              "detail": "offline test: source mechanism does not transfer to target"}

    def test_all_runs_find_automatically_and_uses_next_real_find_candidate_after_transfer_rejection(self):
        candidates, _diagnostics, _request = product_candidates_from_real_find_fixture()
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        first, second, third = candidates[:3]
        find_result = {
            "candidates": candidates,
            "stats": {
                "calls": [{"returned": len(candidates), "keyword": "car vacuum"}],
                "unique_ads": len(candidates), "direct_relevant_ads": len(candidates),
                "discovery_strategy": {"supply_queries_executed": ["car vacuum"]},
            },
            "diagnostics": {"outcome": "CANDIDATES_FOUND", "requests_executed": 1},
            "request": {"selection_mode": "product", "target_offer": fixture["target_offer"]},
        }
        completed, live, attempts, calls = execute_autonomous_all(
            find_result, offer={"target_offer": fixture["target_offer"],
                                "product_type": "cordless handheld car vacuum", "product_category": "vacuum"},
            produce_plan={first["id"]: self.REJECT, second["id"]: self.OK, third["id"]: self.OK},
            selection_mode="product", keywords="car vacuum")
        self.assertEqual(completed.returncode, 0, completed.stderr[-2500:] + completed.stdout[-1000:])
        self.assertEqual(live["candidate_id"], second["id"])
        self.assertEqual(live["failure_class"], "DELIVERED_PASS")
        self.assertEqual([r["candidate_id"] for r in live["transfer_rejections"]], [first["id"]])
        self.assertEqual(attempts["selection_mode"], "automatic_find_order")
        self.assertEqual(attempts["selected_candidate_id"], second["id"])
        self.assertEqual(calls, [
            {"stage": "analyze", "candidate_id": first["id"]},
            {"stage": "produce", "candidate_id": first["id"]},
            {"stage": "analyze", "candidate_id": second["id"]},
            {"stage": "produce", "candidate_id": second["id"]},
        ])

    def test_mechanism_find_output_is_accepted_by_autonomous_all_and_fallback(self):
        class FakeMechanismClient:
            enabled = True
            ORDER_BY = {"for_you"}
            def __init__(self, rows):
                self.rows = rows
                self.calls = []
                self.last_discovery_stats = {}
            def fetch_sponsored_search(self, **kwargs):
                self.calls.append(("sponsored_search", kwargs))
                return {"items": self.rows, "credits_used": 1, "request_id": "mechanism-fixture"}
            def fetch_board(self, **kwargs):
                raise AssertionError("Mechanism mode must not call Ads Top")

        def source(ad_id):
            text = "Rechargeable air duster blows dust from car vents in seconds. Get yours today."
            return {
                "ad_id": ad_id, "video_id": "video-" + ad_id,
                "source_url": "https://www.tiktok.com/@fixture/video/" + ad_id,
                "video_url": "https://fixture.invalid/" + ad_id + ".mp4",
                "title": text, "views": 600_000, "likes": 20_000,
                "comments": 400, "shares": 600, "published_at": "2025-12-20T00:00:00Z",
                "raw": {"post": {"content": {"text": text}, "ext": {"ad": {"title": text}}}},
            }

        offer = {"target_offer": "Fanttik V8 Mate Cordless Car Vacuum",
                 "product_type": "cordless handheld car vacuum", "product_category": "vacuum"}
        client = FakeMechanismClient([source("air-duster-a"), source("air-duster-b")])
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {
            "CME_FIND_SEEN_PATH": str(Path(temp) / "mechanism-seen.json")
        }):
            find_result = run_find({
                "selection_mode": "mechanism", "country": "US", "period": 30,
                "max_requests": 1, "max_ads": 5, "keywords": ["air duster"], "offer": offer,
            }, client=client, now="2026-10-31T00:00:00Z")
        candidates = find_result["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual([call[0] for call in client.calls], ["sponsored_search"])
        self.assertTrue(all(c["source_selection"]["mode"] == "mechanism" for c in candidates))
        self.assertTrue(all(c["query_product_match"]["matches"] is True for c in candidates))
        ranked, skipped = rank_pattern_candidates(candidates)
        self.assertEqual([c["id"] for c in ranked], [c["id"] for c in candidates])
        self.assertEqual(skipped, [])

        first, second = candidates
        completed, live, attempts, calls = execute_autonomous_all(
            find_result, offer=offer,
            produce_plan={first["id"]: self.REJECT, second["id"]: self.OK},
            selection_mode="mechanism", keywords="air duster")
        self.assertEqual(completed.returncode, 0, completed.stderr[-2500:] + completed.stdout[-1000:])
        self.assertEqual(live["candidate_id"], second["id"])
        self.assertEqual([r["candidate_id"] for r in live["transfer_rejections"]], [first["id"]])
        self.assertEqual(attempts["selection_mode"], "automatic_find_order")
        self.assertEqual(calls, [
            {"stage": "analyze", "candidate_id": first["id"]},
            {"stage": "produce", "candidate_id": first["id"]},
            {"stage": "analyze", "candidate_id": second["id"]},
            {"stage": "produce", "candidate_id": second["id"]},
        ])


if __name__ == "__main__":
    unittest.main()