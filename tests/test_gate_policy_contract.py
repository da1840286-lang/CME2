"""Target-Fit source-ledger and no-added-text policy regressions."""
from __future__ import annotations

import copy
import gzip
import json
import unittest
from dataclasses import replace
from pathlib import Path

from cme.brain import (
    atom_target_text_policy_errors,
    build_source_transfer_context,
    request_atoms,
    validate_gate_elements,
    validate_transfer_fit_decision,
)
from cme.branding import NO_BRANDING
from tests.helpers import gate_account
from tests.test_brain_text_boundary_recovery import FIXTURE_PATH, pressure_washer_offer
from tests.test_target_fit_gate import target_offer


class GateSourceLedgerAndTextPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(FIXTURE_PATH, "rt", encoding="utf-8") as handle:
            cls.fixture = json.load(handle)
        cls.responses = cls.fixture["responses"]
        cls.source = build_source_transfer_context(cls.fixture["source_contract"])
        cls.offer = target_offer()
        cls.pressure_offer = pressure_washer_offer()

    def _source_with_all_five_elements(self):
        return {
            "hook": "A person holds the product before demonstrating its action",
            "buyer_problem": "The user faces a concrete task that normally requires effort",
            "mechanism_hypothesis": "The product removes effort through its visible intervention",
            "product_intervention": "The product is applied directly to the user's problem",
            "result_demo": "The changed result is visible after the intervention",
        }

    def _decision(self, *, preserved=("hook", "problem", "mechanism", "intervention", "result"),
                  dropped=(), mapping=None):
        return {
            "transferable": True,
            "abstract_mechanism": "The target product visibly performs the same causal intervention",
            "target_native_mapping": mapping or "Show the target product applying the intervention to an outdoor hard surface",
            "target_evidence": ["Cordless car vacuum"],
            "reason": "The target can demonstrate the same visible action and result.",
            **gate_account(preserved=preserved, dropped=dropped),
        }

    def test_empty_dropped_list_is_valid_when_every_source_element_is_preserved(self):
        source = self._source_with_all_five_elements()
        decision = self._decision()
        self.assertEqual(decision["source_elements_dropped"], [])
        self.assertEqual(validate_gate_elements(decision, source), [])

    def test_each_source_element_must_be_accounted_for_exactly_once(self):
        source = self._source_with_all_five_elements()
        decision = self._decision(preserved=("hook", "problem", "mechanism", "intervention"),
                                  dropped=("result",))
        self.assertEqual(validate_gate_elements(decision, source), [])

        decision["source_elements_dropped"].append(
            {"element": "result", "why": "A duplicate accounting entry for the same source result."}
        )
        self.assertTrue(any("appears more than once" in e for e in validate_gate_elements(decision, source)))

        decision = self._decision(preserved=("hook", "problem", "mechanism", "intervention"),
                                  dropped=())
        self.assertTrue(any("source element 'result'" in e for e in validate_gate_elements(decision, source)))

    def test_gate_cannot_invent_source_elements_absent_from_the_source_ledger(self):
        source = self._source_with_all_five_elements()
        source.pop("hook")
        decision = self._decision()
        errors = validate_gate_elements(decision, source)
        self.assertTrue(any("not present in the supplied source contract" in e for e in errors))

    def test_preserved_target_form_must_record_a_target_specific_change(self):
        source = self._source_with_all_five_elements()
        decision = self._decision()
        item = next(x for x in decision["source_elements_preserved"] if x["element"] == "hook")
        item["target_form"] = item["source_form"]
        errors = validate_gate_elements(decision, source)
        self.assertTrue(any("target-specific adaptation" in e for e in errors))

    def test_gate_rejects_text_overlay_in_target_form_but_allows_it_in_source_form(self):
        source = self._source_with_all_five_elements()
        decision = self._decision()
        hook = next(x for x in decision["source_elements_preserved"] if x["element"] == "hook")
        hook["source_form"] = "The source opens with a text overlay about a tough day"
        hook["target_form"] = "Person holds the product outdoors with a text overlay about cleaning without scrubbing"
        errors = validate_gate_elements(decision, source)
        self.assertTrue(any(e.startswith("NO_ADDED_TEXT_POLICY:") for e in errors))

        hook["target_form"] = "Person presents the product outdoors before demonstrating the cleaning action"
        self.assertFalse(any(e.startswith("NO_ADDED_TEXT_POLICY:") for e in validate_gate_elements(decision, source)))

    def test_gate_rejects_overlay_in_target_native_mapping(self):
        decision = self._decision(mapping="Add a text overlay reading clean without scrubbing")
        errors = validate_transfer_fit_decision(self.offer, decision, self._source_with_all_five_elements())
        self.assertTrue(any(e.startswith("NO_ADDED_TEXT_POLICY:") for e in errors))

    def test_explicit_text_prohibition_is_not_mistaken_for_a_text_request(self):
        atoms = [{
            "id": "keep_hook_01", "kind": "scene",
            "text": "Person holds the pressure washer outdoors without any text overlay",
            "priority": "MUST", "channel": "image", "shot": 1,
        }]
        self.assertEqual(atom_target_text_policy_errors(atoms), {})

    def test_no_branding_also_forbids_product_marks_in_target_form_and_atoms(self):
        source = self._source_with_all_five_elements()
        decision = self._decision()
        hook = next(x for x in decision["source_elements_preserved"] if x["element"] == "hook")
        hook["target_form"] = "Person holds the product with its readable logo visible on the casing"
        errors = validate_gate_elements(decision, source, branding_policy=NO_BRANDING)
        self.assertTrue(any(e.startswith("NO_ADDED_TEXT_POLICY:") for e in errors))

        atoms = [{
            "id": "keep_hook_01", "kind": "scene",
            "text": "Show the product with its readable logo visible on the casing",
            "priority": "MUST", "channel": "image", "shot": 1,
        }]
        self.assertTrue(atom_target_text_policy_errors(atoms, NO_BRANDING))

    def test_real_gate_response_with_overlay_is_rejected_but_source_description_is_kept(self):
        response = self.responses["run_41305261765_gate_1"]
        decision = copy.deepcopy(response["transfer_decision"])
        errors = validate_transfer_fit_decision(self.pressure_offer, decision, self.source)
        self.assertTrue(any(e.startswith("NO_ADDED_TEXT_POLICY:") for e in errors))

        hook = next(x for x in decision["source_elements_preserved"] if x["element"] == "hook")
        self.assertIn("text overlay", hook["source_form"].lower())
        hook["target_form"] = (
            "Person presents the cordless pressure washer outdoors before spraying foam onto patio furniture"
        )
        corrected_errors = validate_transfer_fit_decision(self.pressure_offer, decision, self.source)
        self.assertFalse(any(e.startswith("NO_ADDED_TEXT_POLICY:") for e in corrected_errors))

    def test_real_brain_overlay_atom_is_retried_and_replaced_by_visual_hook(self):
        bad = copy.deepcopy(self.responses["run_41305261765_brain_1"])
        good = copy.deepcopy(bad)
        for item in good["atoms"]:
            if item["id"] == "keep_hook_01":
                item["text"] = (
                    "Person presents the cordless pressure washer outdoors before spraying foam onto patio furniture"
                )
        gate = copy.deepcopy(self.responses["run_41305261765_gate_1"]["transfer_decision"])
        hook = next(x for x in gate["source_elements_preserved"] if x["element"] == "hook")
        hook["target_form"] = (
            "Person presents the cordless pressure washer outdoors before spraying foam onto patio furniture"
        )
        source = copy.deepcopy(self.source)
        source["target_fit_gate"] = gate
        responses = [bad, good]
        calls, trace = [], []

        def call(_request):
            calls.append(True)
            return responses[len(calls) - 1]

        atoms = request_atoms(
            self.pressure_offer, "Create a target-native outdoor pressure-washer demonstration.", 1,
            call, source_transfer=source, rounds_out=trace,
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("keep_hook_01", trace[0]["pending"])
        self.assertNotIn("keep_hook_01", trace[1]["pending"])
        self.assertFalse(atom_target_text_policy_errors(atoms, getattr(self.pressure_offer, "branding_policy", None)))


if __name__ == "__main__":
    unittest.main()
