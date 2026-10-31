"""Targeted regressions for the six final-build corrections, including a real ALL artifact replay."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from cme.brain import (
    _explicit_product_action_evidence,
    _source_evidence_contradiction,
    build_source_transfer_context,
    build_request,
    missing_mechanism_links,
    request_transfer_fit,
    validate_transfer_fit_decision,
)
from cme.offer import OfferCard, load_offer
from cme.radar.select import pattern_source_reason
from tests.helpers import A

ROOT = Path(__file__).resolve().parents[1]
REAL = json.loads((ROOT / "tests" / "fixtures" / "live_artifact_41206920383.json").read_text(encoding="utf-8"))
REAL_ACTION_REPLAYS = json.loads((ROOT / "tests" / "fixtures" / "real_action_evidence_replays_2026-10-31.json").read_text(encoding="utf-8"))
EX = ROOT / "examples" / "offer_example.json"


class FinalMinimalFixTests(unittest.TestCase):
    def test_real_artifact_clear_silicone_is_not_misread_as_cleaning(self):
        context = build_source_transfer_context(REAL["source_analysis"])
        self.assertEqual(_explicit_product_action_evidence(context), [])
        decision = {
            "transferable": False,
            "reason": "The sampled frames do not show the bag sealing the dish or proving safe oven use.",
            "source_elements_dropped": [
                {"element": "mechanism", "why": "No sampled frame shows the bag sealing the dish."},
                {"element": "intervention", "why": "The bag is placed over the dish but the seal is not demonstrated."},
            ],
        }
        self.assertEqual(_source_evidence_contradiction(decision, context), "")

    def test_action_detector_replays_three_real_artifacts_without_frame_label_false_positives(self):
        expected = {
            "41185128123": [0.5, 8.91],  # bags are filled; food is slid to close
            "41148117074": [],  # Close-up is a camera/framing label, not an action
            "41206920383": [],  # clear silicone bag is descriptive; foil lift is tagged as buyer-problem evidence
        }
        for replay in REAL_ACTION_REPLAYS["runs"]:
            context = {"format": "radar_analysis_hypothesis", "evidence": replay["evidence"]}
            with self.subTest(run_id=replay["run_id"]):
                self.assertEqual(_explicit_product_action_evidence(context), expected[replay["run_id"]])
    def test_real_brain_trace_replay_accepts_individual_offer_quotes_after_gap_retry(self):
        context = build_source_transfer_context(REAL["source_analysis"])
        data = REAL["offer"]
        offer = OfferCard(
            offer_id=data["offer_id"], name=data["name"], product_type=data["product_type"],
            verified_claims=tuple(data["verified_claims"]), forbidden_claims=tuple(data["forbidden_claims"]),
            photos=(), platform=data["platform"].lower(), geo=data["geo"],
            product_category=data["product_category"], confirmations=(),
            branding_policy=data["branding_policy"], approved_brand_marks=tuple(data["approved_brand_marks"]),
        )
        actual_rounds = REAL["brain_trace"]["gate_rounds"]
        first = json.loads(actual_rounds[0]["response_excerpt"])
        second = json.loads(actual_rounds[1]["response_excerpt"])
        # This is the minimal correction requested by the new prompt: one exact quote per field.
        second["transfer_decision"]["target_evidence"] = [
            "Grove Co. Reusable Silicone Food Storage Bag 12 oz",
            "Stand-up base for filling and storage",
            "Secure seal",
        ]
        replies = [first, second]
        captured = []

        def call(request):
            captured.append(request)
            return replies.pop(0)

        result = request_transfer_fit(offer, "Create a target-native frame", context, call)
        self.assertIs(result["transferable"], False)
        self.assertEqual(len(captured), 2)
        self.assertIn("NEGATIVE_TRANSFER_EVIDENCE_GAP", " ".join(captured[1]["previous_errors"]))
        self.assertNotIn("SOURCE_EVIDENCE_CONTRADICTION", " ".join(captured[1]["previous_errors"]))
        prompt_rules = " ".join(captured[0]["rules"])
        self.assertIn("EACH target_evidence array item must be one contiguous verbatim quote from ONE OfferCard field only", prompt_rules)

    def test_combined_multi_field_excerpt_from_real_trace_is_rejected(self):
        data = REAL["offer"]
        offer = OfferCard(
            offer_id=data["offer_id"], name=data["name"], product_type=data["product_type"],
            verified_claims=tuple(data["verified_claims"]), forbidden_claims=tuple(data["forbidden_claims"]),
            photos=(), platform=data["platform"].lower(), geo=data["geo"],
            product_category=data["product_category"], confirmations=(),
            branding_policy=data["branding_policy"], approved_brand_marks=tuple(data["approved_brand_marks"]),
        )
        response = json.loads(REAL["brain_trace"]["gate_rounds"][1]["response_excerpt"])
        errors = validate_transfer_fit_decision(offer, response["transfer_decision"])
        self.assertTrue(any("target_evidence[2]" in error and "one authoritative OfferCard field" in error for error in errors), errors)

    def test_qa_only_mechanism_atom_cannot_satisfy_visual_link(self):
        atom = {"id": "mech_intervention_qa", "kind": "qa_check", "text": "Check mechanism",
                "priority": "MUST", "channel": "qa", "shot": 1}
        self.assertEqual(missing_mechanism_links([atom], ["intervention"]), ["intervention"])

    def test_mechanism_contract_requires_positive_query_match_even_without_source_selection(self):
        candidate = {"video_url": "https://example.test/source.mp4",
                     "source_check": {"classification": "EXPANDED"},
                     "request": {"selection_mode": "mechanism"}}
        self.assertIn("explicit positive product-query match", pattern_source_reason(candidate))
        candidate["query_product_match"] = {"matches": True, "query_anchor": "vacuum", "reason": ""}
        self.assertEqual(pattern_source_reason(candidate), "")
        candidate["query_product_match"]["query_anchor"] = ""
        self.assertIn("no verified product anchor", pattern_source_reason(candidate))

    def test_real_find_candidate_from_all_artifact_passes_product_admission(self):
        candidate = REAL["find_candidate"]
        self.assertEqual(candidate["source_selection"]["role"], "TARGET_PRODUCT")
        self.assertIs(candidate["query_product_match"]["matches"], True)
        self.assertEqual(pattern_source_reason(candidate), "")

        # A mode change cannot make this old Product receipt masquerade as a reviewed Mechanism result.
        mechanism_candidate = json.loads(json.dumps(candidate))
        mechanism_candidate["request"]["selection_mode"] = "mechanism"
        mechanism_candidate["source_selection"]["mode"] = "mechanism"
        mechanism_candidate["source_selection"]["role"] = "ADJACENT_MECHANISM"
        mechanism_candidate["query_product_match"] = None
        self.assertIn("explicit positive product-query match", pattern_source_reason(mechanism_candidate))

    def test_manifest_contract_schema_v2_changes_when_branding_mode_changes(self):
        from cme.smoke import offer_fingerprint
        base = load_offer(EX)
        strict = replace(base, branding_policy="no_branding", approved_brand_marks=())
        branded = replace(base, branding_policy="target_brand_allowed", approved_brand_marks=("ExampleBrand",))
        self.assertNotEqual(offer_fingerprint(strict)[0], offer_fingerprint(branded)[0])
        self.assertEqual(strict.approved_brand_marks, ())


if __name__ == "__main__":
    unittest.main()
