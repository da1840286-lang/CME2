import json
import unittest
from pathlib import Path

from cme.brain import atom_budget_errors
from cme.radar.source_loop import analyze_with_invalid_output_retry, run_source_loop


def atom(atom_id, text, shot=1, kind="scene", priority="MUST", channel="image"):
    return {"id": atom_id, "text": text, "shot": shot, "kind": kind, "priority": priority, "channel": channel}


class FrameBudgetTests(unittest.TestCase):
    def test_redundant_requirements_are_rejected_but_mechanism_is_protected(self):
        items = [
            atom("mech_problem_debris", "car seat and floor mat covered with visible crumbs and dust"),
            atom("scene_problem", "car interior with crumbs and dust on seats and floor mats"),
            atom("action_problem_debris", "visible crumbs and dust scattered on car seat and floor mat"),
            atom("product_vacuum", "handheld cordless vacuum, compact black body"),
        ]
        errors = atom_budget_errors(items, 1)
        self.assertEqual(set(errors), {"scene_problem", "action_problem_debris"})
        self.assertIn("FRAME_REDUNDANT", errors["scene_problem"][0])

    def test_one_layout_per_shot(self):
        items = [
            atom("comp_split", "left half dirty seat, right half same seat clean, sharp boundary", kind="scene"),
            atom("composition_side_view", "side view with vacuum body fully visible", kind="composition"),
        ]
        errors = atom_budget_errors(items, 1)
        self.assertEqual(set(errors), {"composition_side_view"})
        self.assertIn("FRAME_LAYOUT_CONFLICT", errors["composition_side_view"][0])

    def test_must_ceiling_is_reported_per_shot(self):
        words = "red blue green amber violet cyan olive pink brown grey teal gold".split()
        items = [atom(f"obj_{i}", f"{words[i]} {words[(i+3)%12]} item number {i} on shelf {words[(i+5)%12]}")
                 for i in range(8)]
        errors = atom_budget_errors(items, 1)
        self.assertIn("#must_budget_1", errors)

    def test_different_shots_have_independent_budgets(self):
        items = [atom(f"s1_{i}", f"alpha{i} beta{i} gamma{i} delta{i} epsilon{i}") for i in range(5)]
        items += [atom(f"s2_{i}", f"zeta{i} eta{i} theta{i} iota{i} kappa{i}", shot=2) for i in range(5)]
        self.assertEqual(atom_budget_errors(items, 2), {})


class SourceLoopTests(unittest.TestCase):
    def run_loop(self, ids, *, statuses=None, reports=None, limit=3, **kwargs):
        statuses, reports = statuses or {}, reports or {}
        order = []

        def analyze(source):
            cid = source["id"]
            order.append(("analyze", cid))
            return {"candidate_id": cid, "status": statuses.get(cid, "OK")}, {"candidate_id": cid}

        def produce(source, payload):
            cid = source["id"]
            order.append(("produce", cid))
            return reports.get(cid, {"failure_class": "DELIVERED_PASS", "exit_code": 0})

        result = run_source_loop([{"id": cid} for cid in ids], limit=limit, analyze=analyze, produce=produce, **kwargs)
        return result, order

    def test_invalid_output_retries_the_same_candidate_once(self):
        calls = []
        outcomes = [
            {"status": "INVALID_OUTPUT", "detail": "invalid JSON"},
            {"status": "OK", "detail": "valid on retry"},
        ]

        def analyze():
            calls.append("candidate-a")
            return outcomes.pop(0)

        attempts = analyze_with_invalid_output_retry(analyze, max_retries=1)
        self.assertEqual(calls, ["candidate-a", "candidate-a"])
        self.assertEqual([a["status"] for a in attempts], ["INVALID_OUTPUT", "OK"])

    def test_non_invalid_analysis_errors_are_not_retried(self):
        calls = []
        attempts = analyze_with_invalid_output_retry(
            lambda: (calls.append("candidate-a") or {"status": "API_ERROR"}),
            max_retries=1,
        )
        self.assertEqual(calls, ["candidate-a"])
        self.assertEqual([a["status"] for a in attempts], ["API_ERROR"])

    def test_transfer_rejection_moves_to_next_ranked_candidate(self):
        result, order = self.run_loop(
            ["a", "b", "c"],
            reports={"a": {"failure_class": "TRANSFER_REJECTED", "failure_detail": "mechanism mismatch"},
                     "b": {"failure_class": "DELIVERED_PASS"}},
        )
        self.assertEqual(result.selected["id"], "b")
        self.assertEqual([r["candidate_id"] for r in result.rejections], ["a"])
        self.assertEqual(order, [("analyze", "a"), ("produce", "a"), ("analyze", "b"), ("produce", "b")])

    def test_all_candidates_rejected_returns_last_rejection(self):
        rejection = {"failure_class": "TRANSFER_REJECTED", "failure_detail": "not transferable"}
        result, _ = self.run_loop(["a", "b", "c"], reports={"a": rejection, "b": rejection, "c": rejection})
        self.assertEqual(result.selected["id"], "c")
        self.assertEqual(result.report["failure_class"], "TRANSFER_REJECTED")
        self.assertEqual(len(result.rejections), 3)

    def test_manual_choice_is_never_replaced(self):
        result, order = self.run_loop(
            ["a", "b"], reports={"a": {"failure_class": "TRANSFER_REJECTED"}, "b": {"failure_class": "DELIVERED_PASS"}},
            allow_transfer_fallback=False,
        )
        self.assertEqual(result.selected["id"], "a")
        self.assertEqual(order, [("analyze", "a"), ("produce", "a")])

    def test_provider_or_visual_qa_failure_does_not_trigger_transfer_fallback(self):
        for failure in ("INFRASTRUCTURE_ERROR", "VISUAL_QA_FAILED", "QA_UNVERIFIED", "BRAIN_OUTPUT_INVALID"):
            result, order = self.run_loop(
                ["a", "b"], reports={"a": {"failure_class": failure}, "b": {"failure_class": "DELIVERED_PASS"}}
            )
            self.assertEqual(result.selected["id"], "a", failure)
            self.assertEqual(order, [("analyze", "a"), ("produce", "a")], failure)

    def test_retryable_analysis_failure_moves_on_but_internal_error_stops(self):
        result, _ = self.run_loop(["a", "b"], statuses={"a": "VIDEO_UNAVAILABLE"})
        self.assertEqual(result.selected["id"], "b")
        result, order = self.run_loop(["a", "b"], statuses={"a": "INTERNAL_ERROR"})
        self.assertIsNone(result.selected)
        self.assertEqual(order, [("analyze", "a")])

    def test_limit_counts_tried_candidates_and_skips_do_not_consume_it(self):
        result, order = self.run_loop(
            ["skip", "a", "b"], limit=2,
            skip_reason=lambda source: "no video" if source["id"] == "skip" else "",
            reports={"a": {"failure_class": "TRANSFER_REJECTED"}, "b": {"failure_class": "DELIVERED_PASS"}},
        )
        self.assertEqual([item for item in order if item[0] == "produce"], [("produce", "a"), ("produce", "b")])
        self.assertEqual(result.skipped, ["skip: no video"])

    def test_workflow_is_wired_to_testable_source_loop(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "live-smoke.yml").read_text(encoding="utf-8")
        self.assertIn("from cme.radar.source_loop import run_source_loop", workflow)
        self.assertIn("loop_result = run_source_loop(", workflow)
        self.assertIn('allow_transfer_fallback=(selection_mode != "manual_from_prior_find")', workflow)


    def test_workflow_rejects_conflicting_all_stage_handoffs_and_mechanism_performance_order(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "live-smoke.yml").read_text(encoding="utf-8")
        self.assertIn('if stage == "all" and selected_candidate and (raw_source or source_run_id):', workflow)
        self.assertIn('if [[ "$FIND_SELECTION_MODE" == "mechanism" && "$FIND_PERFORMANCE_ORDER" != "default" ]]; then', workflow)
        self.assertIn("which creates a source contract", workflow)


if __name__ == "__main__":
    unittest.main()



# Exercise the embedded Actions script with a fake RadarApp. No provider calls are made.
def _embedded_all_stage_script():
    import yaml
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "live-smoke.yml"
    document = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    for step in document["jobs"]["smoke"]["steps"]:
        run = step.get("run") or ""
        if "run_source_loop" not in run:
            continue
        loop_pos = run.rindex("run_source_loop")
        marker_pos = run.rfind("<<'PY'", 0, loop_pos)
        if marker_pos < 0:
            raise AssertionError("could not locate the Python heredoc containing run_source_loop")
        start = run.index("\n", marker_pos) + 1
        end = run.index("\nPY", loop_pos)
        return run[start:end] + "\n"
    raise AssertionError("workflow does not run the testable source loop")


_FAKE_RADAR_APP = r'''
import json, sys
from pathlib import Path
import cme.radar.server as server
PLAN = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

class Store:
    def ingest(self, items, source, state): pass
    def decide(self, cid, state, reason): pass
    def analyses(self, cid): return [{"result": {"mechanism_hypothesis": {"text": "source mechanism"}}}]
    def close(self): pass

class FakeApp:
    def __init__(self, work, **kwargs):
        self.work = Path(work)
        self.store = Store()
        self.calls = 0
    def do_analyze(self, cid):
        return {"status": PLAN.get("analysis", {}).get(cid, "OK"), "analysis_id": 1}
    def do_produce(self, cid):
        self.calls += 1
        spec = PLAN["produce"][cid]
        run_id = f"prod-test-{self.calls}"
        folder = self.work / "production" / run_id
        folder.mkdir(parents=True)
        (folder / "run_result.json").write_text(json.dumps({"failure_class": spec["failure_class"]}), encoding="utf-8")
        return {"run_id": run_id, "status": spec["status"], "exit_code": spec["exit_code"],
                "failure_class": spec["failure_class"], "failure_detail": spec.get("detail", ""),
                "images": [], "verdicts": [{"overall": spec.get("verdict", "")}],
                "offer_name": "Target", "provider_calls_made": 0}
server.RadarApp = FakeApp
exec(compile(Path(sys.argv[2]).read_text(encoding="utf-8"), "workflow", "exec"), {"__name__": "__main__"})
'''


def _candidate(candidate_id, score):
    return {"id": candidate_id, "title": f"title {candidate_id}", "brand": "brand",
            "video_url": "https://video.example/source.mp4",
            "source_check": {"classification": "DIRECT"},
            "source_score": {"final_score": score, "action": "DEEP",
                             "components": {"money_signal_gate": {"pass": True}}}}


class WorkflowFallbackIntegrationTests(unittest.TestCase):
    OK = {"status": "OK", "exit_code": 0, "failure_class": "DELIVERED_PASS"}
    REJECT = {"status": "ATOM_CONTRACT_FAILED", "exit_code": 4, "failure_class": "TRANSFER_REJECTED",
              "detail": "source mechanism cannot be shown by target product"}

    def run_workflow(self, plan, candidate_id="", ids=("a", "b", "c")):
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "out").mkdir()
            candidates = {"candidates": [_candidate(cid, 70-i) for i, cid in enumerate(ids)]}
            (root / "out" / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
            (root / "find-input").mkdir()
            (root / "find-input" / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "workflow.py").write_text(_embedded_all_stage_script(), encoding="utf-8")
            (root / "runner.py").write_text(_FAKE_RADAR_APP, encoding="utf-8")
            (root / "summary.md").write_text("", encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                   "CANDIDATE_ID": candidate_id, "FIND_RUN_ID": "", "SHOTS": "1",
                   "GITHUB_STEP_SUMMARY": str(root / "summary.md"), "GITHUB_RUN_ID": "test"}
            completed = subprocess.run([sys.executable, "runner.py", "plan.json", "workflow.py"],
                                       cwd=root, env=env, capture_output=True, text=True, timeout=60)
            result_path = root / "out" / "live_result.json"
            live = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            attempt_path = root / "out" / "source_analysis_attempts.json"
            attempts = json.loads(attempt_path.read_text(encoding="utf-8")) if attempt_path.exists() else {}
            return completed, live, attempts

    def test_first_transfer_rejection_uses_next_ranked_candidate(self):
        completed, live, attempts = self.run_workflow({
            "analysis": {}, "produce": {"a": self.REJECT, "b": self.OK, "c": self.OK}})
        self.assertEqual(completed.returncode, 0, completed.stderr[-1000:])
        self.assertEqual(live["candidate_id"], "b")
        self.assertEqual(live["failure_class"], "DELIVERED_PASS")
        self.assertEqual([row["candidate_id"] for row in live["transfer_rejections"]], ["a"])
        self.assertEqual(attempts["selected_candidate_id"], "b")

    def test_all_rejected_returns_transfer_rejected_not_a_fake_pass(self):
        completed, live, _ = self.run_workflow({
            "analysis": {}, "produce": {"a": self.REJECT, "b": self.REJECT, "c": self.REJECT}})
        self.assertEqual(completed.returncode, 4)
        self.assertEqual(live["failure_class"], "TRANSFER_REJECTED")
        self.assertEqual(len(live["transfer_rejections"]), 3)

    def test_manual_candidate_is_never_replaced(self):
        completed, live, attempts = self.run_workflow(
            {"analysis": {}, "produce": {"a": self.REJECT, "b": self.OK}}, candidate_id="a")
        self.assertEqual(completed.returncode, 4)
        self.assertEqual(live["candidate_id"], "a")
        self.assertEqual([row["candidate_id"] for row in attempts["attempts"]], ["a"])
        self.assertEqual(attempts["max_candidates"], 1)

    def test_provider_error_is_preserved_and_does_not_trigger_transfer_fallback(self):
        provider = {"status": "INFRA_ERROR", "exit_code": 2, "failure_class": "INFRASTRUCTURE_ERROR",
                    "verdict": "FAILED_QA"}
        completed, live, _ = self.run_workflow({"analysis": {}, "produce": {"a": provider, "b": self.OK}})
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(live["candidate_id"], "a")
        self.assertEqual(live["failure_class"], "INFRASTRUCTURE_ERROR")
