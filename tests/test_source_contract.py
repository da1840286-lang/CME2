"""Regression coverage for a real Analyze artifact becoming a Brain-ready contract."""
import json
import unittest
from pathlib import Path

from cme.brain import build_source_transfer_context
from cme.radar.source_contract import build_source_contract

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = json.loads((ROOT / "tests/fixtures/real_source_analysis_41241133791.json").read_text(encoding="utf-8"))
CANDIDATE = json.loads((ROOT / "tests/fixtures/real_source_candidate_41240647807.json").read_text(encoding="utf-8"))


class SourceContractHandoffTests(unittest.TestCase):
    def test_real_analyze_artifact_is_self_contained_and_brain_readable(self):
        contract = build_source_contract(ANALYSIS, CANDIDATE)
        self.assertEqual(contract["contract_schema"], "cme.radar.source_contract.v1")
        self.assertEqual(contract["result"], ANALYSIS["result"])
        self.assertEqual(contract["candidate_id"], ANALYSIS["candidate_id"])
        self.assertEqual(contract["source_provenance"]["source_lane"], "SPONSORED_SEARCH")
        self.assertEqual(contract["source_provenance"]["published_at"], "2026-09-25T13:00:00.000Z")
        self.assertEqual(contract["source_performance"]["status"], "UNVERIFIED")
        self.assertIn("views", contract["source_performance"]["metrics"])
        self.assertNotIn("ctr", contract["source_performance"]["metrics"])

        context = build_source_transfer_context(contract)
        result = ANALYSIS["result"]
        self.assertEqual(context["mechanism_hypothesis"], result["mechanism_hypothesis"]["text"])
        self.assertEqual(context["transferability_hypothesis"], result["transferability"]["text"])
        for key in ("hook", "buyer_problem", "product_intervention", "result_demo", "cta",
                    "mechanism_hypothesis", "transferability"):
            self.assertEqual(context["section_kinds"][key], result[key]["kind"], key)
        self.assertEqual(len(context["evidence"]), min(8, len(result["evidence"])))
        self.assertEqual(
            contract["analysis_integrity"]["section_kinds_preserved"]["mechanism_hypothesis"],
            result["mechanism_hypothesis"]["kind"],
        )

    def test_failed_or_incomplete_analysis_cannot_be_promoted_to_contract(self):
        failed = dict(ANALYSIS, status="API_ERROR")
        with self.assertRaisesRegex(ValueError, "successful Analyze"):
            build_source_contract(failed, CANDIDATE)
        incomplete = dict(ANALYSIS, result={"hook": {"text": "only hook", "kind": "OBSERVED"}})
        with self.assertRaisesRegex(ValueError, "required transfer sections"):
            build_source_contract(incomplete, CANDIDATE)

    def test_contract_contains_no_raw_video_or_provider_payload(self):
        contract = build_source_contract(ANALYSIS, CANDIDATE)
        serialized = json.dumps(contract, ensure_ascii=False)
        self.assertNotIn('"video_url"', serialized)
        self.assertNotIn('"raw"', serialized)
        self.assertNotIn('"video_url_540p"', serialized)

    def test_product_and_mechanism_selection_identity_survives_source_contract(self):
        for mode, role in (("product", "TARGET_PRODUCT"), ("mechanism", "ADJACENT_MECHANISM")):
            candidate = json.loads(json.dumps(CANDIDATE))
            candidate["request"] = {**(candidate.get("request") or {}), "selection_mode": mode}
            candidate["source_selection"] = {
                "mode": mode, "role": role, "target_anchor": "vacuum",
                "promotion_evidence": True, "target_promotion_evidence": mode == "product",
                "attention_is_commercial_proof": False,
            }
            contract = build_source_contract(ANALYSIS, candidate)
            provenance = contract["source_provenance"]
            self.assertEqual(contract["contract_schema"], "cme.radar.source_contract.v1")
            self.assertEqual(provenance["source_role"], role)
            self.assertEqual(provenance["source_selection"]["mode"], mode)
            self.assertEqual(provenance["source_selection"]["role"], role)
            self.assertEqual(contract["result"], ANALYSIS["result"])
            self.assertEqual(build_source_transfer_context(contract)["mechanism_hypothesis"],
                             ANALYSIS["result"]["mechanism_hypothesis"]["text"])


if __name__ == "__main__":
    unittest.main()
