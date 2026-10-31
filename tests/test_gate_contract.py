"""Target-Fit Gate contract: element account (preserved/dropped), code-enforced rejection, audit trace.

The acceptance cases replay the failure of run 40192742112: the source's persuasion was a comparison of an old
and a new model, and the transfer ended as a plain demonstration of the target product.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cme.brain import (BrainError, core_preservation_verdict, request_atoms, request_transfer_fit,
                       required_gate_elements, validate_transfer_fit_decision)
from cme.providers import InfraError
from cme.smoke import main as smoke_main
from tests.helpers import gate_account
from tests.test_target_fit_gate import EX, ROOT, GOOD_ATOM, target_offer

COMPARISON_SOURCE = {
    "format": "radar_analysis_hypothesis",
    "source_evidence_scope": "sampled frames only",
    "hook": "Old model and new model are held side by side",
    "buyer_problem": "The old model is weak and leaves debris behind",
    "product_intervention": "The new model is run over the same debris as the old one",
    "result_demo": "The new model clearly picks up what the old model left",
    "mechanism_hypothesis": "Direct comparison of old and new model proves the upgrade",
}


def gate_answer(transferable=True, evidence=("Cordless car vacuum",), **account_kwargs):
    return {
        "atoms": [],
        "transfer_decision": {
            "transferable": transferable,
            "abstract_mechanism": "Compare the old way with the new tool on the same debris",
            "target_native_mapping": "Show the old way of cleaning next to the target vacuum on one car seat",
            "target_evidence": list(evidence),
            "reason": "The target can show a visible comparison against the old way of cleaning",
            **gate_account(**account_kwargs),
        },
    }


class ElementAccountTests(unittest.TestCase):
    def test_required_elements_follow_what_the_source_states(self):
        self.assertEqual(required_gate_elements(COMPARISON_SOURCE),
                         ("hook", "problem", "mechanism", "intervention", "result"))
        no_hook = {k: v for k, v in COMPARISON_SOURCE.items() if k != "hook"}
        self.assertNotIn("hook", required_gate_elements(no_hook))
        self.assertEqual(required_gate_elements({}), ("mechanism",))

    def test_every_required_element_must_be_accounted_for(self):
        decision = gate_answer(preserved=("mechanism", "problem"))["transfer_decision"]
        errors = validate_transfer_fit_decision(target_offer(), decision, COMPARISON_SOURCE)
        text = " ".join(errors)
        for name in ("hook", "intervention", "result"):
            self.assertIn(f"source element '{name}'", text)

    def test_an_element_cannot_be_both_preserved_and_dropped(self):
        decision = gate_answer(preserved=("hook", "mechanism"), dropped=("hook",))["transfer_decision"]
        errors = validate_transfer_fit_decision(target_offer(), decision, {})
        self.assertTrue(any("more than once" in e for e in errors))

    def test_unknown_form_or_element_is_malformed(self):
        decision = gate_answer()["transfer_decision"]
        decision["demo_form_target"] = "vibes"
        decision["source_elements_dropped"] = [{"element": "colour", "why": "does not matter much here"}]
        errors = validate_transfer_fit_decision(target_offer(), decision, {})
        self.assertTrue(any("demo_form_target" in e for e in errors))
        self.assertTrue(any("source_elements_dropped[0].element" in e for e in errors))

    def test_gate_request_carries_required_elements_and_the_anti_default_rule(self):
        seen = []
        request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE,
                             lambda r: seen.append(r) or gate_answer(form_source="comparison", form_target="comparison"))
        self.assertEqual(seen[0]["required_source_elements"], ["hook", "problem", "mechanism", "intervention", "result"])
        rules = " ".join(seen[0]["rules"])
        self.assertIn("ORDINARY DEMO IS NOT A DEFAULT ANSWER", rules)
        self.assertIn("ELEMENT ACCOUNT", rules)

    def test_required_elements_reach_the_model_payload(self):
        from cme.adapters.openrouter import make_openrouter_brain
        sent = []

        def transport(method, url, headers, payload):
            sent.append(json.loads(payload["messages"][1]["content"]))
            body = {"model": payload["model"], "choices": [{"message": {"content": json.dumps(
                gate_answer(form_source="comparison", form_target="comparison"))}}]}
            return 200, json.dumps(body).encode()

        call = make_openrouter_brain({"model_id": "m/x:free"}, api_key="k", transport=transport)
        request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, lambda request: call(request)[0])
        self.assertIn("required_source_elements", sent[0])


class CodeEnforcedRejectionTests(unittest.TestCase):
    def test_dropping_hook_and_mechanism_is_rejected_even_when_model_says_transferable(self):
        answer = gate_answer(preserved=("problem", "intervention", "result"), dropped=("hook", "mechanism"))
        result = request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, lambda r: answer)
        self.assertFalse(result["transferable"])
        self.assertTrue(result["rejected_by_code"])
        self.assertTrue(result["model_transferable"])
        self.assertIn("TRANSFER_MECHANISM_DROPPED", result["reason"])

    def test_preserving_only_the_hook_is_not_a_mechanism_transfer(self):
        answer = gate_answer(preserved=("hook", "problem", "intervention", "result"), dropped=("mechanism",))
        result = request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, lambda r: answer)
        self.assertFalse(result["transferable"])
        self.assertTrue(result["rejected_by_code"])
        self.assertIn("TRANSFER_MECHANISM_DROPPED", result["reason"])

    def test_preserving_the_mechanism_is_required_even_if_the_hook_is_adapted(self):
        answer = gate_answer(preserved=("mechanism", "problem", "intervention", "result"), dropped=("hook",))
        result = request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, lambda r: answer)
        self.assertTrue(result["transferable"])
        self.assertNotIn("rejected_by_code", result)

    def test_comparison_replaced_by_plain_demonstration_is_rejected(self):
        # Hook and mechanism are both claimed as preserved, but the form is now an ordinary demonstration.
        answer = gate_answer(form_source="comparison", form_target="demonstration")
        result = request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, lambda r: answer)
        self.assertFalse(result["transferable"])
        self.assertIn("TRANSFER_FORM_DOWNGRADED", result["reason"])

    def test_comparison_kept_as_comparison_or_before_after_is_accepted(self):
        for target_form in ("comparison", "before_after"):
            answer = gate_answer(form_source="comparison", form_target=target_form)
            result = request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, lambda r, a=answer: a)
            self.assertTrue(result["transferable"], target_form)

    def test_an_ordinary_demo_source_may_stay_an_ordinary_demo(self):
        verdict = core_preservation_verdict(gate_answer(form_source="demonstration", form_target="demonstration")["transfer_decision"])
        self.assertEqual(verdict, "")

    def test_run_40192742112_replay_old_vs_new_model_became_a_vacuum_demo(self):
        """Acceptance: the substitution is rejected; the 'old way vs this vacuum' mapping is accepted."""
        substituted = gate_answer(preserved=("problem", "intervention", "result"), dropped=("hook", "mechanism"),
                                  form_source="comparison", form_target="demonstration")
        substituted["transfer_decision"]["target_native_mapping"] = "A cordless vacuum removes dry crumbs from a car seat"
        rejected = request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, lambda r: substituted)
        self.assertFalse(rejected["transferable"])
        self.assertTrue(rejected["rejected_by_code"])

        mapped = gate_answer(form_source="comparison", form_target="comparison")
        accepted = request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, lambda r: mapped)
        self.assertTrue(accepted["transferable"])
        self.assertEqual(accepted["demo_form_target"], "comparison")


class AuditTraceTests(unittest.TestCase):
    def test_gate_trace_records_response_validation_errors_and_verdict(self):
        trace = []
        bad = {"atoms": [], "transfer_decision": {"transferable": True}}
        with self.assertRaises(BrainError):
            request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, lambda r: bad, trace_out=trace)
        self.assertEqual(trace[0]["phase"], "target_fit_gate")
        self.assertTrue(trace[0]["validation_errors"])
        self.assertIn("transfer_decision", trace[0]["response_excerpt"])

        trace = []
        request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE,
                             lambda r: gate_answer(preserved=("problem", "result", "intervention"), dropped=("hook", "mechanism")),
                             trace_out=trace)
        self.assertTrue(trace[0]["verdict"]["rejected_by_code"])

    def test_gate_trace_keeps_a_provider_failure(self):
        trace = []

        def call(_request):
            raise InfraError("openrouter HTTP 502")
        with self.assertRaises(InfraError):
            request_transfer_fit(target_offer(), "brief", COMPARISON_SOURCE, call, trace_out=trace)
        self.assertIn("InfraError", trace[0]["error"])

    def test_preserved_elements_must_reference_actual_source_frame_timestamps(self):
        source = {**COMPARISON_SOURCE, "evidence": [
            {"t_sec": 1.5, "observation": "Old and new vacuum models appear side by side"},
            {"t_sec": 3.0, "observation": "The newer model picks up remaining crumbs"},
        ]}
        valid = gate_answer(form_source="comparison", form_target="comparison")
        valid["transfer_decision"]["source_elements_preserved"] = [
            {**item, "evidence_t_sec": [1.5 if item["element"] in ("hook", "mechanism") else 3.0]}
            for item in valid["transfer_decision"]["source_elements_preserved"]
        ]
        result = request_transfer_fit(target_offer(), "brief", source, lambda _r: valid)
        self.assertTrue(result["transferable"])

        invalid = gate_answer(form_source="comparison", form_target="comparison")
        invalid["transfer_decision"]["source_elements_preserved"] = [
            {**item, "evidence_t_sec": [99.0]} for item in invalid["transfer_decision"]["source_elements_preserved"]
        ]
        with self.assertRaisesRegex(BrainError, "not a timestamp with a supplied source observation"):
            request_transfer_fit(target_offer(), "brief", source, lambda _r: invalid)

    def test_gate_kept_hook_and_mechanism_require_must_image_atoms(self):
        from cme.brain import missing_keep_atoms
        fit_gate = gate_answer(form_source="comparison", form_target="comparison")["transfer_decision"]
        fit_gate["transferable"] = True
        self.assertEqual(missing_keep_atoms([], fit_gate), ["mechanism", "hook"])
        atoms = [
            {"id": "keep_mechanism_comparison", "priority": "MUST", "channel": "image", "kind": "scene",
             "text": "Show the problem, target intervention, and resulting improvement as one visible sequence"},
            {"id": "keep_hook_comparison", "priority": "MUST", "channel": "image", "kind": "scene",
             "text": "Open on the distinctive attention hook before the product action appears"},
        ]
        self.assertEqual(missing_keep_atoms(atoms, fit_gate), [])

    def test_keep_atom_id_alone_does_not_prove_mechanism_preservation(self):
        from cme.brain import missing_keep_atoms
        fit_gate = gate_answer(form_source="comparison", form_target="comparison")["transfer_decision"]
        fit_gate["transferable"] = True
        unrelated = [
            {"id": "keep_mechanism_comparison", "priority": "MUST", "channel": "image", "kind": "scene",
             "text": "Elegant studio product beauty shot on a seamless white background"},
            {"id": "keep_hook_comparison", "priority": "MUST", "channel": "image", "kind": "scene",
             "text": "Elegant studio product beauty shot on a seamless white background"},
        ]
        self.assertEqual(missing_keep_atoms(unrelated, fit_gate), ["mechanism", "hook"])

    def test_atom_authoring_request_requires_core_keep_atoms(self):
        from cme.brain import build_request
        fit_gate = gate_answer(form_source="comparison", form_target="comparison")["transfer_decision"]
        fit_gate["transferable"] = True
        request = build_request(target_offer(), "brief", 1, source_transfer={"target_fit_gate": fit_gate})
        joined = " ".join(request["rules"])
        self.assertIn("keep_mechanism_", joined)
        self.assertIn("keep_hook_", joined)

    def test_brain_rounds_keep_reasons_and_the_failure_that_ended_the_run(self):
        rounds = []
        calls = []

        def call(request):
            calls.append(request)
            if len(calls) == 3:
                raise InfraError("openrouter HTTP 502")
            return {"atoms": []}  # round 1 and 2: no MUST image atom -> a recorded reason
        with self.assertRaises(InfraError):
            request_atoms(target_offer(), "brief", 1, call, max_retries=2, rounds_out=rounds)
        self.assertEqual([r["round"] for r in rounds], [1, 2, 3])
        self.assertIn("#shot_1", rounds[0]["pending"])
        self.assertIn("MUST image atom", rounds[0]["pending"]["#shot_1"][0])
        self.assertEqual(rounds[1]["only_fix_ids"], ["#shot_1"])
        self.assertIn("InfraError", rounds[2]["error"])
        self.assertIn("#shot_1", rounds[2]["pending_before_failure"])

    def test_smoke_writes_brain_trace_when_the_third_round_dies_of_a_provider_error(self):
        calls = []

        class Flaky:
            info = {"model_id": "test-model", "responded_model": "test-model"}

            def __call__(self, request):
                calls.append(request)
                if len(calls) == 3:
                    raise InfraError("openrouter HTTP 502")
                return {"atoms": []}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            with patch("cme.smoke.make_brain_call", return_value=Flaky()):
                code = smoke_main(["brain", "--offer", str(EX), "--settings", str(ROOT / "settings.example.json"),
                                   "--out", str(out), "--shots", "1"])
            self.assertEqual(code, 2)
            trace = json.loads((out / "brain_trace.json").read_text(encoding="utf-8"))
            self.assertEqual(len(trace["brain_rounds"]), 3)
            self.assertIn("InfraError", trace["brain_rounds"][2]["error"])
            self.assertTrue(trace["brain_rounds"][0]["pending"])
            self.assertFalse((out / "atoms.json").exists())

    def test_smoke_code_rejection_is_transfer_rejected_with_trace(self):
        source = {"result": {
            "hook": {"text": "Old model and new model are held side by side", "kind": "OBSERVED"},
            "buyer_problem": {"text": "The old model leaves debris behind", "kind": "OBSERVED"},
            "product_intervention": {"text": "The new model runs over the same debris", "kind": "OBSERVED"},
            "result_demo": {"text": "The new model picks up what the old one left", "kind": "OBSERVED"},
            "mechanism_hypothesis": {"text": "Side-by-side comparison proves the upgrade", "kind": "INFERRED"},
            "transferability": {"text": "Could transfer", "kind": "INFERRED"},
        }}
        calls = []

        class Brain:
            info = {"model_id": "test-model", "responded_model": "test-model"}

            def __call__(self, request):
                calls.append(request)
                return gate_answer(preserved=("problem", "intervention", "result"), dropped=("hook", "mechanism"),
                                   evidence=("kitchen storage",))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source.json").write_text(json.dumps(source), encoding="utf-8")
            out = root / "out"
            with patch("cme.smoke.make_brain_call", return_value=Brain()):
                code = smoke_main(["brain", "--offer", str(EX), "--settings", str(ROOT / "settings.example.json"),
                                   "--source-contract", str(root / "source.json"), "--out", str(out), "--shots", "1"])
            self.assertEqual(code, 4)
            self.assertEqual(len(calls), 1, "no atom authoring after a code rejection")
            result = json.loads((out / "run_result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["failure_class"], "TRANSFER_REJECTED")
            self.assertIn("TRANSFER_MECHANISM_DROPPED", result["message"])
            gate = json.loads((out / "target_fit_gate.json").read_text(encoding="utf-8"))
            self.assertTrue(gate["rejected_by_code"])
            trace = json.loads((out / "brain_trace.json").read_text(encoding="utf-8"))
            self.assertTrue(trace["gate_rounds"][0]["verdict"]["rejected_by_code"])


if __name__ == "__main__":
    unittest.main()
