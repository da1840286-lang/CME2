"""Values and contracts must survive every hop: FIND -> select -> store -> Analyze -> Brain -> Frame.

The Analyze output used here is a real one (tests/fixtures/real_analysis_result.json, produced by the live
`all` run 41026869082-successor). Provider traffic is faked; model quality is out of scope.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cme.radar.find as F
from cme.brain import build_source_transfer_context, derive_mechanism_links
from cme.radar.legacy.discovery import tiktok_socialcrawl as T
from cme.radar.select import pattern_source_reason, rank_pattern_candidates
from cme.radar.store import Store
from cme.smoke import main as smoke_main
from tests.helpers import gate_account
from tests.judge_fakes import judge_reply_body
from tests.radar_fixtures import FakeResp, synthetic_post, synthetic_response
import tests.test_firstrun as first_run

atom, completion = first_run.atom, first_run.completion

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "real_analysis_result.json").read_text(encoding="utf-8"))
OFFER = {"name": "Acme Model 9 Cordless Car Vacuum", "product_type": "cordless handheld car vacuum"}
GOOD_TITLES = [
    "Handheld Cordless Car Vacuum cleaner! Shop now #carvacuum #caraccessories",
    "Mini cordless vacuum cleaner for my car link in bio #vacuumcleaner #carcleaning",
]


def _post(ad_id, title, views):
    post = synthetic_post(ad_id, brand="UNKNOWN", title=title, ctr=None, cvr=None, views=views, duration=25)
    post["post"]["author"]["username"] = "UNKNOWN"
    post["post"]["author"]["display_name"] = ""
    post["post"]["ext"]["ad"].update({"ctr": None, "cvr": None, "objective": "", "landing_page_url": ""})
    post["post"]["content"]["text"] = title
    return post


class FindToSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.n = 0

    def find(self, **params):
        self.n += 1
        items = [_post(f"g{i}", title, 900000 - i * 1000) for i, title in enumerate(GOOD_TITLES)]
        items.append(_post("junk", "Twin Turbo F150 for sale", 4000))
        base = {"country": "US", "period": 30, "max_requests": 3, "max_ads": 5, "keywords": ["car vacuum"],
                "discovery_mode": "auto", "selection_mode": "product", "video_duration_range": "all",
                "performance_order_bys": [], "include_retention": False, "offer": OFFER}
        base.update(params)
        with mock.patch.dict(os.environ, {"CME_FIND_SEEN_PATH": os.path.join(self.tmp.name, f"s{self.n}.json")}), \
                mock.patch.object(T.requests, "get", lambda *a, **k: FakeResp(synthetic_response(items))):
            return F.run_find(base, client=T.SocialCrawlTikTokTopAds(api_key="K"))

    def assert_handoff_contract(self, result):
        """Every candidate FIND returns has statuses that agree with each other and with its own source_selection."""
        self.assertTrue(result["candidates"], result["diagnostics"].get("outcome"))
        ids = [c["id"] for c in result["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))
        for c in result["candidates"]:
            reason = pattern_source_reason(c)
            self.assertEqual(c["analysis_eligibility"]["eligible"], not reason, c["id"])
            self.assertEqual(c["analysis_eligibility"]["reason"], reason or None)
            self.assertEqual(c["stage"], "PATTERN_CANDIDATE" if not reason else "SOURCE_REJECTED")
            self.assertEqual(c["source_gate_eligible"], c["transfer_eligible"])
            self.assertEqual(c["source_gate_eligible"], not reason)
            self.assertTrue(c["video_url"])
            self.assertEqual(c["source_selection"]["mode"], c["request"]["selection_mode"])
        # the all-stage reads candidates.json from disk: nothing may change on the way
        again = json.loads(json.dumps(result["candidates"], ensure_ascii=False))
        ranked, skipped = rank_pattern_candidates(again)
        self.assertEqual([c["id"] for c in ranked], [c["id"] for c in result["candidates"] if c["source_gate_eligible"]])
        self.assertEqual(skipped, [])
        self.assertTrue(ranked)

    def test_default_product_path_uses_targeted_sponsored_search_and_hands_candidates_to_all(self):
        result = self.find()
        self.assertEqual(result["request"]["performance_plan_effective"], "contract_aware_sponsored_batch")
        self.assert_handoff_contract(result)
        self.assertTrue(all(c["query_product_match"]["matches"] is True for c in result["candidates"]))
        self.assertNotIn("junk", " ".join(c["id"] for c in result["candidates"]))

    def test_contract_aware_product_path_hands_eligible_candidates_to_all(self):
        result = self.find(performance_order_bys=["play_6s_rate", "play_2s_rate"])
        self.assertEqual(result["request"]["performance_plan_effective"], "contract_aware_sponsored_batch")
        self.assert_handoff_contract(result)
        for c in result["candidates"]:
            self.assertIs(c["query_product_match"]["matches"], True)
            self.assertEqual(c["performance_evidence"]["requested_metric_status"], "NOT_OBTAINABLE_FOR_PRODUCT_QUERY")

    def test_mechanism_path_hands_eligible_candidates_to_all(self):
        result = self.find(selection_mode="mechanism", keywords=["cordless handheld vacuum"],
                           offer={"name": "Reusable silicone food storage bag", "product_type": "kitchen storage"})
        self.assert_handoff_contract(result)
        self.assertTrue(all(c["source_selection"]["role"] == "ADJACENT_MECHANISM" for c in result["candidates"]))

    def test_mechanism_candidate_without_positive_query_match_is_not_admissible(self):
        result = self.find(selection_mode="mechanism", keywords=["cordless handheld vacuum"],
                           offer={"name": "Reusable silicone food storage bag", "product_type": "kitchen storage"})
        candidate = json.loads(json.dumps(result["candidates"][0]))
        candidate.pop("query_product_match", None)
        candidate["legacy_record"].pop("query_product_match", None)
        self.assertIn("explicit positive product-query match", pattern_source_reason(candidate))
        candidate["query_product_match"] = {"matches": False, "query_anchor": "vacuum", "reason": "QUERY_PRODUCT_MISMATCH"}
        self.assertEqual(pattern_source_reason(candidate), "QUERY_PRODUCT_MISMATCH")
        candidate["query_product_match"] = {"matches": True, "query_anchor": "vacuum", "mention_source": "body", "reason": ""}
        self.assertEqual(pattern_source_reason(candidate), "")

    def test_explicit_query_mismatch_still_blocks_admission(self):
        result = self.find(performance_order_bys=["play_6s_rate"])
        candidate = json.loads(json.dumps(result["candidates"][0]))
        candidate["query_product_match"] = {"matches": False, "reason": "QUERY_PRODUCT_MISMATCH"}
        self.assertEqual(pattern_source_reason(candidate), "QUERY_PRODUCT_MISMATCH")

    def test_missing_target_promotion_is_still_blocked_in_product_mode(self):
        candidate = json.loads(json.dumps(self.find()["candidates"][0]))
        candidate["source_selection"]["target_promotion_evidence"] = False
        self.assertEqual(pattern_source_reason(candidate), "target noun lacks promotion evidence")
        candidate["source_selection"]["role"] = "ADJACENT_MECHANISM"
        self.assertIn("not TARGET_PRODUCT", pattern_source_reason(candidate))

    def test_store_keeps_every_field_analyze_and_the_report_read(self):
        result = self.find(performance_order_bys=["play_6s_rate"])
        store = Store(Path(self.tmp.name) / "radar.sqlite")
        store.ingest(result["candidates"], "run-1", "found")
        for original in result["candidates"]:
            saved = store.get(original["id"])
            for key in ("video_url", "source_selection", "analysis_eligibility", "performance_evidence",
                        "commercial_evidence", "query_product_match", "duration_seconds", "request", "title"):
                self.assertEqual(saved[key], original[key], key)


class AnalysisToBrainTests(unittest.TestCase):
    CTX = build_source_transfer_context({"result": FIXTURE["result"]})

    def test_every_analysis_section_and_its_evidence_label_cross_the_boundary(self):
        result = FIXTURE["result"]
        for key, ctx_key in (("hook", "hook"), ("buyer_problem", "buyer_problem"),
                             ("product_intervention", "product_intervention"), ("result_demo", "result_demo"),
                             ("cta", "cta"), ("mechanism_hypothesis", "mechanism_hypothesis"),
                             ("transferability", "transferability_hypothesis")):
            self.assertEqual(self.CTX[ctx_key], result[key]["text"], key)
            self.assertEqual(self.CTX["section_kinds"][key], result[key]["kind"], key)
        self.assertEqual(len(self.CTX["evidence"]), min(8, len(result["evidence"])))
        self.assertEqual(self.CTX["limits"]["text"], result["limits"]["text"])
        self.assertNotIn("video_url", json.dumps(self.CTX))

    def test_unproven_result_is_flagged_instead_of_silently_required_as_observed(self):
        links = {link["link"]: link for link in derive_mechanism_links(self.CTX)}
        self.assertEqual(set(links), {"problem", "intervention", "result"})
        self.assertEqual(links["result"]["source_kind"], "INFERRED")
        self.assertEqual(links["intervention"]["source_kind"], "OBSERVED")
        self.assertIn("do not show a visible", links["result"]["source_statement"])

    def test_old_contracts_without_labels_still_derive_links_unchanged(self):
        ctx = dict(self.CTX)
        ctx.pop("section_kinds")
        links = derive_mechanism_links(ctx)
        self.assertEqual({link["link"] for link in links}, {"problem", "intervention", "result"})
        self.assertTrue(all("source_kind" not in link for link in links))


class BrainToFrameChainTests(unittest.TestCase):
    """smoke `all` with the analysis handoff exactly as server.do_produce writes it."""

    def run_chain(self, tmp):
        evidence_ts = [row["t_sec"] for row in FIXTURE["result"]["evidence"]]
        account = gate_account()
        for row in account["source_elements_preserved"]:
            row["evidence_t_sec"] = [evidence_ts[1]]
        fit = {"transferable": True, "abstract_mechanism": "Show a visible task being completed by hand",
               "target_native_mapping": "Show the bag holding food and one zip closure action",
               "target_evidence": ["Reusable and dishwasher safe"],
               "reason": "The target class can honestly show the same visible task and result", **account}
        atoms = [
            atom("keep_hook_1", "Open on the distinctive attention hook before the product action appears"),
            atom("keep_mechanism_1", "Show the problem, target intervention, and resulting improvement as one visible sequence", kind="action"),
            atom("mech_problem_1", "Loose food pieces are scattered on the counter"),
            atom("mech_intervention_1", "The bag is held open beside the loose pieces", kind="action"),
            atom("mech_result_1", "The pieces are inside the sealed bag", kind="action"),
            atom("comp_1", "Left shows loose pieces, right shows the bag with pieces inside"),
            atom("prod_1", "Silicone bag in the foreground", kind="product"),
        ]
        decision = {"transferable": True, "abstract_mechanism": fit["abstract_mechanism"],
                    "preserved": [{"property": "visible sequence", "atom_id": "keep_mechanism_1"}], "reason": "same visible sequence"}
        self.sent = []

        def transport(method, url, headers, payload):
            self.sent.append(payload)
            judged = judge_reply_body(payload)
            if judged is not None:
                return 200, completion(json.dumps(judged), payload["model"])
            user = json.loads(payload["messages"][1]["content"])
            body = ({"atoms": [], "transfer_decision": fit} if user.get("task") == "target_fit_gate"
                    else {"atoms": atoms, "transfer_decision": decision})
            return 200, completion(json.dumps(body), payload["model"])

        helper = first_run.SmokeEndToEndTests("test_all_path_produces_marked_test_artifacts")
        ctx = helper.make_ctx(tmp=tmp)
        ctx["transport"] = transport
        handoff = {"status": "OK", "candidate_id": "tiktok:ad:1", "analysis_id": 7, "result": FIXTURE["result"],
                   "source_scope": "sampled frames only"}
        source = Path(tmp) / "source_contract.json"
        source.write_text(json.dumps(handoff), encoding="utf-8")
        out = Path(tmp) / "out"
        code = smoke_main(["all", "--settings", helper.settings(tmp), "--out", str(out), "--source-contract",
                           str(source), "--no-ocr-required"], ctx)
        return code, out, source, atoms

    def test_analysis_values_reach_brain_and_atoms_reach_frame_with_matching_contract_ids(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            code, out, source, atoms = self.run_chain(tmp)
            self.assertEqual(code, 0)
            users = [json.loads(p["messages"][1]["content"]) for p in self.sent
                     if isinstance(p["messages"][1]["content"], str)]
            gate_request = next(u for u in users if u.get("task") == "target_fit_gate")
            atom_request = next(u for u in users if u.get("task") != "target_fit_gate")
            for request in (gate_request, atom_request):
                transfer = request["source_transfer"]
                self.assertEqual(transfer["result_demo"], FIXTURE["result"]["result_demo"]["text"])
                self.assertEqual(transfer["section_kinds"]["result_demo"], "INFERRED")
            self.assertEqual({link["link"]: link["source_kind"] for link in atom_request["mechanism_links"]},
                            {"problem": "INFERRED", "intervention": "OBSERVED", "result": "INFERRED"})
            manifest = json.loads((out / "atoms_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_contract_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertTrue(manifest["target_fit_gate"]["transferable"])
            gate = json.loads((out / "target_fit_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["abstract_mechanism"], manifest["transfer_decision"]["abstract_mechanism"])
            state = json.loads((out / "state.json").read_text(encoding="utf-8"))
            receipt = json.loads(next((out / "receipts").glob("drec_*.json")).read_text(encoding="utf-8"))
            self.assertEqual({manifest["contract_id"], state["contract_id"], receipt["contract_id"]}, {manifest["contract_id"]})
            self.assertEqual(receipt["overall"], "PASS")
            self.assertEqual(state["shots"]["1"]["receipt_id"], receipt["receipt_id"])
            atoms_json = json.loads((out / "atoms.json").read_text(encoding="utf-8"))
            must_ids = {a["id"] for a in atoms_json if a["priority"] == "MUST" and a["channel"] == "image"}
            checked = {row["atom_id"] if "atom_id" in row else row.get("id") for row in receipt["rows"]}
            self.assertTrue(must_ids <= checked, must_ids - checked)
            self.assertEqual(manifest["mechanism_links"], ["problem", "intervention", "result"])
            self.assertEqual({a["id"] for a in atoms_json}, {a["id"] for a in atoms})


if __name__ == "__main__":
    unittest.main()
