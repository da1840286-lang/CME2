"""Real-response regressions for Brain/Gate text limits and bounded repair cycles."""
from __future__ import annotations

import copy
import gzip
import json
import unittest
from dataclasses import replace
from pathlib import Path

from cme.atoms import ATOM_SCHEMA, TRANSFER_ATOM_SCHEMA, MAX_TEXT, MODEL_ATOM_TEXT_MAX, MODEL_GATE_REASON_MAX, parse_atoms
from cme.brain import (
    BrainError,
    TRANSFER_FIT_SCHEMA,
    build_request,
    atom_text_quality_errors,
    build_source_transfer_context,
    request_atoms,
    request_transfer_fit,
    validate_transfer_decision,
    validate_transfer_fit_decision,
)
from tests.test_target_fit_gate import target_offer


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "brain_text_boundary_real_responses.json.gz"


def pressure_washer_offer():
    return replace(
        target_offer(),
        name="Cordless Pressure Washer",
        product_type="cordless pressure washer",
        product_category="Home Improvement",
        verified_claims=("Cordless pressure washer",),
        confirmations=("The washer sprays detergent foam using its included detergent bottle.",),
        forbidden_claims=(),
    )


class BrainTextBoundaryRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(FIXTURE_PATH, "rt", encoding="utf-8") as handle:
            cls.fixture = json.load(handle)
        cls.responses = cls.fixture["responses"]
        # This historical test fixture represents a human-confirmed foam-sprayer offer.
        # Keep the functional confirmation explicit rather than treating the accessory/photo as proof.
        confirmation = "The washer sprays detergent foam using its included detergent bottle."
        for response in cls.responses.values():
            if not isinstance(response, dict):
                continue
            decision = response.get("transfer_decision")
            if isinstance(decision, dict) and decision.get("transferable") is True:
                evidence = decision.setdefault("target_evidence", [])
                if confirmation not in evidence and len(evidence) < 3:
                    evidence.append(confirmation)
        cls.source_transfer = build_source_transfer_context(cls.fixture["source_contract"])
        cls.offer = pressure_washer_offer()

    def source_with_successful_gate(self):
        source = copy.deepcopy(self.source_transfer)
        source["target_fit_gate"] = copy.deepcopy(
            self.responses["run_41305261765_gate_1"]["transfer_decision"]
        )
        hook = next(
            item for item in source["target_fit_gate"]["source_elements_preserved"]
            if item["element"] == "hook"
        )
        hook["target_form"] = (
            "Person presents the cordless pressure washer outdoors before spraying foam onto patio furniture"
        )
        return source

    def test_provider_schema_has_headroom_but_finished_118_character_atom_is_valid(self):
        self.assertEqual(MAX_TEXT, 120)
        self.assertEqual(MODEL_ATOM_TEXT_MAX, 160)
        self.assertEqual(
            ATOM_SCHEMA["properties"]["atoms"]["items"]["properties"]["text"]["maxLength"],
            160,
        )
        atom = next(
            item for item in self.responses["run_41305261765_brain_1"]["atoms"]
            if item["id"] == "comp_01"
        )
        self.assertGreaterEqual(len(atom["text"]), 101)
        self.assertLessEqual(len(atom["text"]), 120)
        parsed = parse_atoms([atom])
        self.assertEqual(parsed.errors, {})
        self.assertEqual(atom_text_quality_errors(parsed.atoms), {})

        exactly_120 = {
            **atom,
            "text": (
                "A person uses the cordless pressure washer to spray foam across an entire patio surface "
                "in one continuous demonstration."
            ),
        }
        self.assertEqual(len(exactly_120["text"]), 120)
        self.assertEqual(parse_atoms([exactly_120]).errors, {})

    def test_real_120_character_dangling_word_stays_in_second_validator(self):
        atom = next(
            item for item in self.responses["run_41255080474_brain_1"]["atoms"]
            if item["id"].startswith("comp_")
        )
        self.assertEqual(len(atom["text"]), 120)
        self.assertTrue(atom["text"].endswith("with a"))
        parsed = parse_atoms([atom])
        # Contract choice: the parser still accepts dangling words; Brain's quality validator catches them.
        self.assertEqual(parsed.errors, {})
        error = atom_text_quality_errors(parsed.atoms)[atom["id"]][0]
        self.assertTrue(error.startswith("TEXT_REWRITE_REQUIRED:"))
        self.assertIn("at most 100 characters", error)

    def test_real_dangling_separator_uses_same_message_in_parser_and_quality_validator(self):
        atom = next(
            item for item in self.responses["run_41300192141_brain_2"]["atoms"]
            if item["id"].startswith("comp_")
        )
        self.assertEqual(len(atom["text"]), 120)
        self.assertTrue(atom["text"].endswith(","))
        parsed = parse_atoms([atom])
        parser_error = parsed.errors[atom["id"]][0]
        quality_error = atom_text_quality_errors([atom])[atom["id"]][0]
        self.assertEqual(parser_error, quality_error)
        self.assertTrue(parser_error.startswith("TEXT_REWRITE_REQUIRED:"))

    def test_model_text_over_code_limit_is_rejected_with_complete_rewrite_instruction(self):
        over_limit = {
            "id": "comp_over", "kind": "scene",
            "text": "A person demonstrates a cordless pressure washer spraying foam onto an outdoor patio surface in one uninterrupted continuous wide shot.",
            "priority": "MUST", "channel": "image", "shot": 1,
        }
        self.assertGreater(len(over_limit["text"]), MAX_TEXT)
        self.assertLessEqual(len(over_limit["text"]), MODEL_ATOM_TEXT_MAX)
        parsed = parse_atoms([over_limit])
        error = parsed.errors["comp_over"][0]
        self.assertTrue(error.startswith("TEXT_REWRITE_REQUIRED:"))
        self.assertIn("120-character code limit", error)
        self.assertIn("at most 100 characters", error)

    def test_atom_prompt_asks_for_concision_without_rejecting_complete_101_to_120_character_text(self):
        request = build_request(self.offer, "Create one target-native product demonstration.", 1)
        self.assertTrue(any("Aim to keep each atom text at or below 100 characters" in rule for rule in request["rules"]))
        self.assertIn("complete atoms up to the 120-character code limit remain valid", " ".join(request["rules"]).casefold())

    def test_brain_transfer_decision_reason_uses_the_same_schema_headroom_and_code_guard(self):
        self.assertEqual(
            TRANSFER_ATOM_SCHEMA["properties"]["transfer_decision"]["properties"]["reason"]["maxLength"],
            MODEL_GATE_REASON_MAX,
        )
        response = self.responses["run_41305261765_brain_1"]
        decision = copy.deepcopy(response["transfer_decision"])
        self.assertEqual(validate_transfer_decision(decision, response["atoms"]), [])

        decision["reason"] = "x" * 240
        errors = validate_transfer_decision(decision, response["atoms"])
        self.assertTrue(any(error.startswith("TEXT_REWRITE_REQUIRED:") for error in errors))
        self.assertIn("transfer_decision.reason", " ".join(errors))

    def test_gate_schema_has_headroom_and_240_character_reason_is_rejected(self):
        self.assertEqual(
            TRANSFER_FIT_SCHEMA["properties"]["transfer_decision"]["properties"]["reason"]["maxLength"],
            320,
        )
        for key in (
            "run_41255080474_gate_1",
            "run_41295468437_gate_1",
            "run_41300192141_gate_1",
        ):
            decision = self.responses[key]["transfer_decision"]
            self.assertEqual(len(decision["reason"]), 240)
            errors = validate_transfer_fit_decision(self.offer, decision, self.source_transfer)
            self.assertTrue(any(error.startswith("TEXT_REWRITE_REQUIRED:") for error in errors), key)

    def test_gate_retries_truncated_real_reasons_and_accepts_complete_third_response(self):
        first = copy.deepcopy(self.responses["run_41255080474_gate_1"])
        second = copy.deepcopy(self.responses["run_41295468437_gate_1"])
        third = copy.deepcopy(self.responses["run_41305261765_gate_1"])
        third["transfer_decision"]["reason"] = (
            "The target product can demonstrate the same foam-spraying action on outdoor hard surfaces."
        )
        third_hook = next(
            item for item in third["transfer_decision"]["source_elements_preserved"]
            if item["element"] == "hook"
        )
        third_hook["target_form"] = (
            "Person presents the cordless pressure washer outdoors before spraying foam onto patio furniture"
        )
        answers = [first, second, third]
        calls, trace = [], []

        def call(request):
            calls.append(request)
            return answers[len(calls) - 1]

        result = request_transfer_fit(
            self.offer, "Show an outdoor pressure-washer demonstration.", self.source_transfer,
            call, trace_out=trace,
        )
        self.assertTrue(result["transferable"])
        self.assertEqual(len(calls), 3)
        self.assertEqual([row["round_budget"] for row in trace], [3, 3, 3])
        self.assertEqual(calls[0]["schema"]["properties"]["transfer_decision"]["properties"]["reason"]["maxLength"], 320)
        self.assertTrue(trace[0]["validation_errors"][0].startswith("TEXT_REWRITE_REQUIRED:"))
        self.assertTrue(trace[1]["validation_errors"][0].startswith("TEXT_REWRITE_REQUIRED:"))
        self.assertTrue(result["reason"].endswith("."))
        self.assertLess(len(result["reason"]), 240)

    def test_gate_fails_closed_after_exactly_three_truncated_real_responses(self):
        answers = [
            self.responses["run_41255080474_gate_1"],
            self.responses["run_41295468437_gate_1"],
            self.responses["run_41300192141_gate_1"],
        ]
        calls, trace = [], []

        def call(request):
            calls.append(request)
            return answers[len(calls) - 1]

        with self.assertRaises(BrainError) as raised:
            request_transfer_fit(
                self.offer, "Show an outdoor pressure-washer demonstration.", self.source_transfer,
                call, trace_out=trace,
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(trace), 3)
        self.assertIn("#transfer_fit", raised.exception.errors)
        self.assertTrue(any(
            error.startswith("TEXT_REWRITE_REQUIRED:")
            for error in raised.exception.errors["#transfer_fit"]
        ))

    def test_evidence_gap_does_not_silently_add_a_fourth_gate_round(self):
        negative = copy.deepcopy(self.responses["run_41305261765_gate_1"])
        decision = negative["transfer_decision"]
        decision["transferable"] = False
        decision["reason"] = "Essential target compatibility remains unverified."
        decision["source_elements_preserved"] = []
        decision["source_elements_dropped"] = [
            {"element": element, "why": "The target evidence for this source feature is unverified."}
            for element in ("hook", "problem", "mechanism", "intervention", "result")
        ]
        answers = [negative, negative, negative, negative]
        calls, trace = [], []

        def call(request):
            calls.append(request)
            return answers[len(calls) - 1]

        result = request_transfer_fit(
            self.offer, "Show an outdoor pressure-washer demonstration.", self.source_transfer,
            call, trace_out=trace,
        )
        self.assertFalse(result["transferable"])
        self.assertEqual(result["rejection_basis"], "EVIDENCE_GAP")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(trace), 4)  # Three attempts plus the final rejection audit row.

    def test_brain_uses_three_retries_then_accepts_real_fourth_response(self):
        failed = copy.deepcopy(self.responses["run_41295468437_brain_1"])
        repaired = copy.deepcopy(self.responses["run_41305261765_brain_1"])
        # Direct atom repair keeps the exact requested ID; suffix changes are allowed
        # only when an active structural key explicitly requires a new role atom.
        for atom in repaired["atoms"]:
            if atom.get("id") == "comp_01":
                atom["id"] = "comp_1"
        for response in (failed, repaired):
            hook_atom = next(
                atom for atom in response["atoms"] if atom["id"].startswith("keep_hook")
            )
            hook_atom["text"] = (
                "Person presents the cordless pressure washer outdoors before spraying foam onto patio furniture"
            )
        answers = [failed, failed, failed, repaired]
        calls, trace = [], []

        def call(request):
            calls.append(request)
            return answers[len(calls) - 1]

        result = request_atoms(
            self.offer, "Create a target-native outdoor pressure-washer demo.", 1, call,
            source_transfer=self.source_with_successful_gate(), rounds_out=trace,
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(trace), 4)
        self.assertEqual([row["round_budget"] for row in trace], [4, 4, 4, 4])
        self.assertEqual(trace[0]["only_fix_ids"], [])
        self.assertEqual(trace[1]["only_fix_ids"], ["comp_1"])
        self.assertEqual(trace[2]["only_fix_ids"], ["comp_1"])
        self.assertEqual(trace[3]["only_fix_ids"], ["comp_1"])
        self.assertIn("comp_1", {atom.id for atom in result})
        self.assertFalse(any(row.get("pending") for row in trace[-1:]))

    def test_brain_exhausts_all_four_attempts_and_never_returns_partial_contract(self):
        failed = self.responses["run_41295468437_brain_1"]
        calls, trace = [], []

        def call(request):
            calls.append(request)
            return failed

        with self.assertRaises(BrainError) as raised:
            request_atoms(
                self.offer, "Create a target-native outdoor pressure-washer demo.", 1, call,
                source_transfer=self.source_with_successful_gate(), rounds_out=trace,
            )
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(trace), 4)
        self.assertIn("comp_1", raised.exception.errors)
        self.assertNotIn("#composition", raised.exception.errors)
        self.assertTrue(raised.exception.errors["comp_1"][0].startswith("TEXT_REWRITE_REQUIRED:"))


if __name__ == "__main__":
    unittest.main()
