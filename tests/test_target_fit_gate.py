"""Offline regressions for pre-generation target-fit gating and truthful Brain failure status."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cme.brain import (
    BrainError,
    TRANSFER_FIT_SCHEMA,
    build_request,
    request_atoms,
    request_transfer_fit,
    validate_transfer_fit_decision,
)
from cme.offer import load_offer
from cme.smoke import main as smoke_main
from tests.helpers import gate_account

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "offer_example.json"
GOOD_ATOM = {
    "id": "s1_scene", "kind": "scene", "text": "A compact vacuum beside a car seat",
    "priority": "MUST", "channel": "image", "shot": 1,
}


def target_offer():
    base = load_offer(EX)
    return replace(
        base,
        name="Fanttik V8 Mate Cordless Car Vacuum",
        product_type="cordless handheld car vacuum",
        product_category="car cleaning",
        verified_claims=("Cordless car vacuum",),
        confirmations=(),
    )


def source_context():
    return {
        "format": "radar_analysis_hypothesis",
        "source_evidence_scope": "sampled frames only; not full-video verification",
        "buyer_problem": "Phone storage is inconvenient when cables are required",
        "product_intervention": "A magnetic attachment connects an external SSD directly to a phone",
        "result_demo": "The drive attaches without a loose cable",
        "mechanism_hypothesis": "Magnetic phone attachment removes friction from portable storage",
        "transferability_hypothesis": "A hypothesis that needs target-specific verification",
    }


def decision(transferable, evidence=None, account=None):
    return {**(account if account is not None else gate_account()), **{
        "transferable": transferable,
        "abstract_mechanism": "Magnetic coupling connects portable storage directly to a phone",
        "target_native_mapping": "The target must demonstrate the same causal intervention, not just general convenience",
        "target_evidence": evidence or [],
        "reason": ("The source intervention depends on magnetic phone-storage coupling absent from the target"
                   if not transferable else "The target evidence supports the same causal intervention in this offer"),
    }}


class TargetFitGateTests(unittest.TestCase):
    def test_gate_has_a_separate_zero_atom_contract(self):
        self.assertEqual(set(TRANSFER_FIT_SCHEMA["required"]), {"atoms", "transfer_decision"})
        self.assertEqual(TRANSFER_FIT_SCHEMA["properties"]["atoms"]["maxItems"], 0)

    def test_incompatible_source_is_rejected_by_gate_before_any_atom_authoring(self):
        calls = []

        def call(request):
            calls.append(request)
            return {"atoms": [], "transfer_decision": decision(False)}

        result = request_transfer_fit(target_offer(), "Create an original ad", source_context(), call)
        self.assertFalse(result["transferable"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["task"], "target_fit_gate")
        self.assertEqual(calls[0]["schema"]["properties"]["atoms"]["maxItems"], 0)
        self.assertIn("generic desire", " ".join(calls[0]["rules"]))
        self.assertIn("does not require verified suction numbers", " ".join(calls[0]["rules"]))

    def test_normal_visible_vacuum_demo_does_not_require_unverified_optional_specs(self):
        response_decision = {
            "transferable": True,
            "abstract_mechanism": "Show visible crumbs, remove them, then reveal a clean patch",
            "target_native_mapping": "Show crumbs on a car seat and one achievable cleaning pass",
            "target_evidence": ["cordless handheld car vacuum"],
            "reason": "A visible dry-debris demo is normal use and needs no unverified numeric claim",
            **gate_account(),
        }
        calls = []

        def call(request):
            calls.append(request)
            return {"atoms": [], "transfer_decision": response_decision}

        result = request_transfer_fit(
            target_offer(), "Show a visibly cleaner car seat", {
                "format": "radar_analysis_hypothesis",
                "buyer_problem": "Crumbs remain on a car seat",
                "product_intervention": "A cordless handheld vacuum removes visible dry debris",
                "result_demo": "The cleaned patch is visibly clearer than before",
                "mechanism_hypothesis": "A visible action proves a practical cleaning result",
            }, call,
        )
        self.assertTrue(result["transferable"])
        rules = " ".join(calls[0]["rules"])
        self.assertIn("does not require verified suction numbers", rules)
        self.assertIn("unverified unique feature must be rejected", rules)

    def test_transferable_gate_requires_literal_target_offer_evidence(self):
        offer = target_offer()
        valid = decision(True, ["Cordless car vacuum"])
        self.assertEqual(validate_transfer_fit_decision(offer, valid), [])
        invalid = decision(True, ["Magnetic SSD attaches to a phone"])
        self.assertTrue(any("not present in authoritative OfferCard" in x
                            for x in validate_transfer_fit_decision(offer, invalid)))

    def test_invalid_target_fit_response_gets_one_correction_attempt(self):
        calls = []
        trace = []
        responses = [
            {"atoms": [], "transfer_decision": {"transferable": True}},
            {"atoms": [], "transfer_decision": decision(False)},
        ]

        def call(request):
            calls.append(request)
            return responses[len(calls) - 1]

        result = request_transfer_fit(
            target_offer(), "Create an original ad", source_context(), call, trace_out=trace,
        )

        self.assertFalse(result["transferable"], "a corrected valid negative verdict must remain final")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[1]["previous_errors"])
        self.assertIn("previous_errors", calls[1])
        self.assertIn("failed validation", " ".join(calls[1]["rules"]))
        self.assertEqual([row["round"] for row in trace], [1, 2])
        self.assertTrue(trace[0]["validation_errors"])
        self.assertFalse(trace[1].get("validation_errors"))
        self.assertFalse(trace[1]["verdict"]["transferable"])

    def test_invalid_target_fit_response_stops_after_one_retry(self):
        calls = []
        trace = []

        def call(request):
            calls.append(request)
            return {"atoms": [], "transfer_decision": {"transferable": True}}

        with self.assertRaises(BrainError):
            request_transfer_fit(
                target_offer(), "Create an original ad", source_context(), call, trace_out=trace,
            )

        self.assertEqual(len(calls), 2, "Target-Fit must not loop indefinitely on malformed output")
        self.assertEqual(len(trace), 2)
        self.assertTrue(trace[1]["previous_errors"])
        self.assertTrue(trace[1]["validation_errors"])

    def test_gate_fails_closed_on_invalid_or_uncertain_output(self):
        with self.assertRaises(BrainError):
            request_transfer_fit(target_offer(), "brief", source_context(), lambda _request: {"atoms": [], "transfer_decision": {"transferable": True}})
        with self.assertRaisesRegex(BrainError, r"atoms=\[\]"):
            request_transfer_fit(target_offer(), "brief", source_context(), lambda _request: {
                "atoms": [GOOD_ATOM], "transfer_decision": decision(False)})

    def test_atom_authoring_must_preserve_gate_approved_mechanism(self):
        approved = {
            "transferable": True,
            "abstract_mechanism": "Show the messy surface, then prove a visibly cleaner path",
            "target_native_mapping": "Demonstrate one achievable vacuum pass on a car seat",
            "target_evidence": ["Cordless car vacuum"],
            "reason": "The OfferCard supports a target-native visible cleaning demonstration",
        }
        context = {"target_fit_gate": approved}
        req = build_request(target_offer(), "Make one frame", 1, source_transfer=context)
        self.assertIn(approved["abstract_mechanism"], " ".join(req["rules"]))
        wrong = {
            "transferable": True,
            "abstract_mechanism": "Use a dramatic close-up to make the product look premium",
            "preserved": [{"property": "a visible improvement after an action", "atom_id": "s1_scene"}],
            "reason": "A dramatic product close-up looks persuasive to a potential buyer",
        }
        with self.assertRaisesRegex(BrainError, "preserve the independently approved target-fit mechanism"):
            request_atoms(target_offer(), "Make one frame", 1,
                          lambda _request: {"atoms": [GOOD_ATOM], "transfer_decision": wrong},
                          max_retries=0, source_transfer=context)

    def test_smoke_rejects_source_before_second_brain_request(self):
        source_path_data = {"result": {
            "hook": {"text": "Magnetic SSD on a phone", "kind": "OBSERVED"},
            "buyer_problem": {"text": "Portable storage requires cables", "kind": "OBSERVED"},
            "product_intervention": {"text": "Magnetic attachment connects the SSD to a phone", "kind": "OBSERVED"},
            "result_demo": {"text": "The drive attaches without a loose cable", "kind": "OBSERVED"},
            "cta": {"text": "Shop now", "kind": "OBSERVED"},
            "mechanism_hypothesis": {"text": "Magnetic phone attachment removes friction", "kind": "INFERRED"},
            "transferability": {"text": "Could transfer to other products", "kind": "INFERRED"},
        }}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source_path_data), encoding="utf-8")
            out = root / "out"
            calls = []

            class FakeBrainCall:
                info = {"model_id": "test-model", "responded_model": "test-model"}

                def __call__(self, request):
                    calls.append(request)
                    return {"atoms": [], "transfer_decision": decision(False)}

            with patch("cme.smoke.make_brain_call", return_value=FakeBrainCall()):
                code = smoke_main([
                    "brain", "--offer", str(EX), "--settings", str(ROOT / "settings.example.json"),
                    "--source-contract", str(source_path), "--out", str(out), "--shots", "1",
                ])

            self.assertEqual(code, 4)
            self.assertEqual(len(calls), 1, "atom-authoring Brain request must not follow a rejected gate")
            run_result = json.loads((out / "run_result.json").read_text(encoding="utf-8"))
            gate = json.loads((out / "target_fit_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(run_result["failure_class"], "TRANSFER_REJECTED")
            self.assertEqual(run_result["status"], "ERROR")
            self.assertFalse(gate["transferable"])
            self.assertFalse((out / "atoms.json").exists())

    def test_brain_error_cannot_be_overwritten_with_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            with patch("cme.smoke.run_brain", side_effect=BrainError({"#must_budget_1": ["FRAME_BUDGET"]})):
                code = smoke_main([
                    "brain", "--offer", str(EX), "--settings", str(ROOT / "settings.example.json"),
                    "--out", str(out),
                ])
            result = json.loads((out / "run_result.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 4)
            self.assertEqual(result["status"], "ERROR")
            self.assertEqual(result["exit_code"], 4)
            self.assertEqual(result["failure_class"], "BRAIN_OUTPUT_INVALID")
            self.assertNotEqual(result.get("verdict"), "PASS")


if __name__ == "__main__":
    unittest.main()
