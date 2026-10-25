"""Transient-error retry and declared reserve model for Brain/Gate/Judge; optional source threshold, WEAK_SOURCE,
NO_QUALIFIED_SOURCE and the per-run candidate ledger. Everything offline."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cme.registry as registry
from cme.providers import CallLog, InfraError, ProviderCallError, ProviderPolicyError
from cme.radar.ledger import build_ledger
from cme.radar.select import SourceThreshold, rank_transfer_candidates, select_transfer_candidate, weak_source_reasons
from cme.registry import AdapterSpec, Settings, SettingsError, load_settings, make_brain_call
from tests.test_brand_selection_audit import cand
from tests.test_source_loop import _FAKE_RADAR_APP, _embedded_all_stage_script

ROOT = Path(__file__).resolve().parents[1]
REQUEST = {"schema": {}, "rules": [], "offer": {}, "brief": "", "shots": 1}


def completion(content, model):
    return json.dumps({"model": model, "choices": [{"message": {"content": content}}], "usage": {"cost": 0}}).encode()


def catalog():
    return {
        "b": AdapterSpec("b", "brain", "openrouter", "vendor/primary:free", "free", {}, verified=True),
        "b2": AdapterSpec("b2", "brain", "openrouter", "vendor/reserve:free", "free", {}, verified=True),
        "j": AdapterSpec("j", "judge", "openrouter", "vendor/judge:free", "free", {}, verified=True),
    }


ERR_502 = (502, json.dumps({"error": {"code": 502, "message": "Bad gateway"}}).encode())


class TransientRetryTests(unittest.TestCase):
    def build(self, directory, transport, **settings):
        log = CallLog(Path(directory) / "calls.jsonl")
        base = {"brain_adapter": "b", "allow_free": True}
        base.update(settings)
        call = make_brain_call(Settings(**base), catalog(), log, "run1", transport=transport, openrouter_key="k")
        return call, Path(directory) / "calls.jsonl"

    def rows(self, path):
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]

    def test_brain_survives_one_transient_502(self):
        sent = []

        def transport(method, url, headers, payload):
            sent.append(payload["model"])
            return ERR_502 if len(sent) == 1 else (200, completion(json.dumps({"atoms": []}), payload["model"]))
        with tempfile.TemporaryDirectory() as d:
            call, log = self.build(d, transport)
            self.assertEqual(call(REQUEST), {"atoms": []})
            self.assertEqual(len(sent), 2)
            rows = self.rows(log)
            self.assertIn("retry", [r.get("status") for r in rows if r.get("component") == "brain_resilience"])
            self.assertEqual(rows[-1]["status"], "ok")

    def test_retries_can_be_switched_off(self):
        sent = []

        def transport(method, url, headers, payload):
            sent.append(1)
            return ERR_502
        with tempfile.TemporaryDirectory() as d:
            call, _ = self.build(d, transport, transient_retries=0)
            with self.assertRaises(InfraError):
                call(REQUEST)
            self.assertEqual(len(sent), 1)

    def test_a_non_transient_error_is_not_retried(self):
        sent = []

        def transport(method, url, headers, payload):
            sent.append(1)
            return 400, b'{"error": {"message": "bad request"}}'
        with tempfile.TemporaryDirectory() as d:
            call, _ = self.build(d, transport)
            with self.assertRaises(ProviderCallError):
                call(REQUEST)
            self.assertEqual(len(sent), 1)

    def test_pause_between_retries_comes_from_settings(self):
        sent = []

        def transport(method, url, headers, payload):
            sent.append(1)
            return ERR_502 if len(sent) == 1 else (200, completion('{"atoms": []}', payload["model"]))
        with tempfile.TemporaryDirectory() as d, patch.object(registry, "_sleep") as sleep:
            call, _ = self.build(d, transport, retry_delay_seconds=2.5)
            call(REQUEST)
            sleep.assert_called_once_with(2.5)

    def test_declared_reserve_model_answers_after_the_primary_is_exhausted(self):
        sent = []

        def transport(method, url, headers, payload):
            sent.append(payload["model"])
            if payload["model"] == "vendor/primary:free":
                return ERR_502
            return 200, completion(json.dumps({"atoms": []}), payload["model"])
        with tempfile.TemporaryDirectory() as d:
            call, log = self.build(d, transport, brain_fallback_adapter="b2")
            self.assertEqual(call(REQUEST), {"atoms": []})
            self.assertEqual(sent, ["vendor/primary:free", "vendor/primary:free", "vendor/reserve:free"])
            self.assertTrue(call.info["fallback_used"])
            self.assertEqual(call.info["model_id"], "vendor/reserve:free")
            self.assertIn("brain_fallback", [r.get("component") for r in self.rows(log)])

    def test_no_reserve_declared_means_no_silent_switch(self):
        sent = []

        def transport(method, url, headers, payload):
            sent.append(payload["model"])
            return ERR_502
        with tempfile.TemporaryDirectory() as d:
            call, _ = self.build(d, transport)
            with self.assertRaises(InfraError):
                call(REQUEST)
            self.assertEqual(set(sent), {"vendor/primary:free"})

    def test_reserve_must_exist_match_the_kind_and_differ_from_the_primary(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("missing", "j", "b"):
                with self.assertRaises(ProviderPolicyError, msg=name):
                    self.build(d, lambda *a: ERR_502, brain_fallback_adapter=name)

    def test_settings_validate_the_new_fields(self):
        def load(**extra):
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "s.json"
                path.write_text(json.dumps(extra), encoding="utf-8")
                return load_settings(path)
        defaults = load()
        self.assertEqual((defaults.transient_retries, defaults.retry_delay_seconds, defaults.source_min_final_score),
                         (1, 0.0, None))
        self.assertEqual(defaults.source_threshold_mode, "flag")
        parsed = load(source_selection={"min_final_score": 60, "allowed_actions": ["deep"], "mode": "enforce"},
                      brain_fallback_adapter="b2", judge_fallback_adapter="j2")
        self.assertEqual((parsed.source_min_final_score, parsed.source_allowed_actions, parsed.source_threshold_mode),
                         (60.0, ("DEEP",), "enforce"))
        self.assertEqual((parsed.brain_fallback_adapter, parsed.judge_fallback_adapter), ("b2", "j2"))
        for bad in ({"transient_retries": 9}, {"transient_retries": True}, {"retry_delay_seconds": -1},
                    {"source_selection": []}, {"source_selection": {"mode": "strict"}},
                    {"source_selection": {"allowed_actions": "DEEP"}}, {"brain_fallback_adapter": 3}):
            with self.assertRaises(SettingsError, msg=str(bad)):
                load(**bad)

    def test_example_settings_keep_the_threshold_off(self):
        settings = load_settings(ROOT / "settings.example.json")
        self.assertFalse(SourceThreshold.from_settings(settings).active)
        self.assertEqual((settings.brain_fallback_adapter, settings.judge_fallback_adapter), ("", ""))


class SourceThresholdTests(unittest.TestCase):
    POOL = [cand("a", 66.0), cand("b", 57.0), cand("c", 63.0, action="WATCH")]

    def ids(self, items):
        return [x["id"] for x in items]

    def test_without_a_bar_nothing_changes(self):
        ordered, skipped = rank_transfer_candidates(self.POOL)
        self.assertEqual(self.ids(ordered), ["a", "c", "b"])
        self.assertEqual(skipped, [])
        self.assertEqual(self.ids(rank_transfer_candidates(self.POOL, SourceThreshold())[0]), ["a", "c", "b"])
        self.assertEqual(weak_source_reasons(self.POOL[1], SourceThreshold()), [])

    def test_flag_mode_keeps_weak_sources_but_ranks_them_last(self):
        bar = SourceThreshold(min_final_score=60.0, allowed_actions=("DEEP",), mode="flag")
        ordered, skipped = rank_transfer_candidates(self.POOL, bar)
        self.assertEqual(self.ids(ordered), ["a", "c", "b"])  # a clears the bar; c and b are weak and follow by score
        self.assertEqual(skipped, [])
        self.assertEqual(weak_source_reasons(self.POOL[0], bar), [])
        self.assertIn("below the bar", weak_source_reasons(self.POOL[1], bar)[0])
        self.assertIn("action WATCH", weak_source_reasons(self.POOL[2], bar)[0])
        self.assertEqual(select_transfer_candidate(self.POOL, bar)[0]["id"], "a")

    def test_flag_mode_puts_a_strong_lower_ranked_source_before_a_weak_higher_scored_one(self):
        pool = [cand("watch", 70.0, action="WATCH"), cand("deep", 62.0)]
        bar = SourceThreshold(allowed_actions=("DEEP",))
        self.assertEqual(self.ids(rank_transfer_candidates(pool, bar)[0]), ["deep", "watch"])

    def test_enforce_mode_drops_weak_sources_with_a_reason(self):
        bar = SourceThreshold(min_final_score=65.0, mode="enforce")
        ordered, skipped = rank_transfer_candidates(self.POOL, bar)
        self.assertEqual(self.ids(ordered), ["a"])
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all("WEAK_SOURCE" in x for x in skipped))

    def test_hard_conditions_still_win_over_any_bar(self):
        rejected = cand("r", 99.0, gate=False, action="REJECT")
        ordered, skipped = rank_transfer_candidates([rejected], SourceThreshold(min_final_score=1.0))
        self.assertEqual(ordered, [])
        self.assertIn("REJECT", skipped[0])


class LedgerTests(unittest.TestCase):
    def test_every_candidate_gets_a_row_with_the_facts_we_need_to_pick_a_bar_later(self):
        pool = [cand("a", 66.0), cand("b", 57.0), cand("x", 40.0, gate=False, action="REJECT")]
        bar = SourceThreshold(min_final_score=60.0)
        rows = build_ledger(
            pool, bar, attempts=[{"candidate_id": "a", "status": "OK"}, {"candidate_id": "b", "status": "OK"}],
            rejections=[{"candidate_id": "a", "reason": "TRANSFER_CORE_DROPPED"}], selected_id="b",
            failure_class="DELIVERED_PASS", verdicts=[{"overall": "PASS"}])
        by_id = {r["candidate_id"]: r for r in rows}
        self.assertEqual(len(rows), 3)
        self.assertEqual((by_id["a"]["rank"], by_id["a"]["weak_source"], by_id["a"]["gate_verdict"]), (1, False, "REJECTED"))
        self.assertEqual(by_id["a"]["gate_reason"], "TRANSFER_CORE_DROPPED")
        self.assertEqual((by_id["b"]["rank"], by_id["b"]["weak_source"]), (2, True))
        self.assertEqual((by_id["b"]["gate_verdict"], by_id["b"]["qa_verdict"], by_id["b"]["outcome"]),
                         ("ACCEPTED", "PASS", "DELIVERED_PASS"))
        self.assertFalse(by_id["x"]["eligible"])
        self.assertIsNone(by_id["x"]["rank"])
        self.assertIn("REJECT", by_id["x"]["block_reason"])
        self.assertEqual(by_id["x"]["gate_verdict"], "NOT_REACHED")
        self.assertEqual(by_id["x"]["analysis_status"], "NOT_TRIED")


class WorkflowThresholdIntegrationTests(unittest.TestCase):
    """Run the embedded Actions script with a fake RadarApp and a settings.json in a temporary ROOT."""
    OK = {"status": "OK", "exit_code": 0, "failure_class": "DELIVERED_PASS"}
    REJECT = {"status": "ATOM_CONTRACT_FAILED", "exit_code": 4, "failure_class": "TRANSFER_REJECTED",
              "detail": "TRANSFER_CORE_DROPPED: hook and mechanism were replaced"}

    def run_workflow(self, plan, scores=(("a", 70), ("b", 69), ("c", 68)), source_selection=None, candidate_id=""):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "out").mkdir()
            (root / "find-input").mkdir()
            items = [{"id": cid, "title": f"title {cid}", "brand": "brand", "video_url": "https://video.example/x.mp4",
                      "source_check": {"classification": "DIRECT"},
                      "source_score": {"final_score": score, "action": "DEEP",
                                       "components": {"money_signal_gate": {"pass": True}}}} for cid, score in scores]
            payload = json.dumps({"candidates": items})
            (root / "out" / "candidates.json").write_text(payload, encoding="utf-8")
            (root / "find-input" / "candidates.json").write_text(payload, encoding="utf-8")
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "workflow.py").write_text(_embedded_all_stage_script(), encoding="utf-8")
            runner = _FAKE_RADAR_APP.replace(
                "server.RadarApp = FakeApp",
                "server.RadarApp = FakeApp\nimport os\nserver.ROOT = Path(os.environ['CME_TEST_ROOT'])")
            (root / "runner.py").write_text(runner, encoding="utf-8")
            (root / "summary.md").write_text("", encoding="utf-8")
            settings_root = root / "cfg"
            settings_root.mkdir()
            if source_selection is not None:
                (settings_root / "settings.json").write_text(json.dumps({"source_selection": source_selection}), encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": str(ROOT), "CANDIDATE_ID": candidate_id, "FIND_RUN_ID": "",
                   "SHOTS": "1", "GITHUB_STEP_SUMMARY": str(root / "summary.md"), "GITHUB_RUN_ID": "test",
                   "CME_TEST_ROOT": str(settings_root)}
            done = subprocess.run([sys.executable, "runner.py", "plan.json", "workflow.py"], cwd=root, env=env,
                                  capture_output=True, text=True, timeout=60)
            read = lambda name: (json.loads((root / "out" / name).read_text(encoding="utf-8"))
                                 if (root / "out" / name).exists() else {})
            return done, read("live_result.json"), read("candidates_ledger.json")

    def test_default_run_writes_a_ledger_and_marks_nothing_weak(self):
        done, live, ledger = self.run_workflow({"analysis": {}, "produce": {"a": self.REJECT, "b": self.OK, "c": self.OK}})
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        self.assertEqual(live["candidate_id"], "b")
        self.assertFalse(live["weak_source"])
        self.assertEqual(live["source_rank"], 2)
        self.assertFalse(ledger["threshold"]["active"])
        rows = {r["candidate_id"]: r for r in ledger["candidates"]}
        self.assertEqual(rows["a"]["gate_verdict"], "REJECTED")
        self.assertIn("TRANSFER_CORE_DROPPED", rows["a"]["gate_reason"])
        self.assertEqual((rows["b"]["gate_verdict"], rows["b"]["outcome"]), ("ACCEPTED", "DELIVERED_PASS"))
        self.assertEqual(rows["c"]["gate_verdict"], "NOT_REACHED")

    def test_flag_mode_marks_a_weak_selected_source_without_blocking_it(self):
        done, live, ledger = self.run_workflow(
            {"analysis": {}, "produce": {"a": self.OK, "b": self.OK, "c": self.OK}},
            source_selection={"min_final_score": 75, "mode": "flag"})
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        self.assertTrue(live["weak_source"])
        self.assertIn("below the bar", live["weak_source_reasons"][0])
        self.assertTrue(all(r["weak_source"] for r in ledger["candidates"]))

    def test_enforce_mode_stops_honestly_when_nothing_qualifies(self):
        done, live, ledger = self.run_workflow(
            {"analysis": {}, "produce": {"a": self.OK, "b": self.OK, "c": self.OK}},
            source_selection={"min_final_score": 75, "mode": "enforce"})
        self.assertEqual(done.returncode, 4)
        self.assertIn("NO_QUALIFIED_SOURCE", done.stderr)
        self.assertEqual(live["failure_class"], "NO_QUALIFIED_SOURCE")
        self.assertEqual(live["provider_calls_made"], 0)
        self.assertEqual(len(live["skipped_candidates"]), 3)
        self.assertTrue(all("WEAK_SOURCE" in x for x in live["skipped_candidates"]))
        self.assertTrue(all(r["rank"] is None for r in ledger["candidates"]))
        self.assertTrue(all(r["analysis_status"] == "NOT_TRIED" for r in ledger["candidates"]))

    def test_no_eligible_source_at_all_is_also_no_qualified_source(self):
        done, live, _ = self.run_workflow({"analysis": {}, "produce": {}}, scores=())
        self.assertEqual(done.returncode, 4)
        self.assertEqual(live["failure_class"], "NO_QUALIFIED_SOURCE")

    def test_manual_candidate_is_flagged_but_never_blocked_by_the_bar(self):
        done, live, _ = self.run_workflow(
            {"analysis": {}, "produce": {"a": self.OK}}, candidate_id="a",
            source_selection={"min_final_score": 99, "mode": "enforce"})
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        self.assertEqual(live["candidate_id"], "a")
        self.assertTrue(live["weak_source"])


if __name__ == "__main__":
    unittest.main()
