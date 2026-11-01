"""Regressions for recovering an incomplete composition atom without duplicate structural errors."""
from __future__ import annotations

import unittest

from cme.atoms import MAX_TEXT, parse_atoms
from cme.brain import (
    BrainError,
    atom_text_quality_errors,
    build_source_transfer_context,
    request_atoms,
)
from tests.test_target_fit_gate import target_offer


RADAR = {
    "result": {
        "hook": {"text": "A cordless vacuum is used to clean a car interior", "kind": "OBSERVED"},
        "buyer_problem": {"text": "Crumbs remain trapped in a narrow seat crevice", "kind": "OBSERVED"},
        "product_intervention": {"text": "A cordless vacuum nozzle removes crumbs from the crevice", "kind": "OBSERVED"},
        "result_demo": {"text": "The same crevice is visibly clear after vacuuming", "kind": "OBSERVED"},
        "mechanism_hypothesis": {"text": "Visible suction removes debris from tight spaces", "kind": "INFERRED"},
    }
}


def atom(atom_id: str, text: str, kind: str = "scene") -> dict:
    return {
        "id": atom_id,
        "kind": kind,
        "text": text,
        "priority": "MUST",
        "channel": "image",
        "shot": 1,
    }


def brain_response(comp_id: str | None, comp_text: str = "") -> dict:
    atoms = [
        atom("keep_hook_1", "Show a cordless vacuum cleaning crumbs in a car seat."),
        atom("keep_mechanism_1", "Cordless suction removes crumbs from tight spaces.", "action"),
        atom("mech_problem_1", "Crumbs remain visible in a narrow car seat crevice."),
        atom("mech_intervention_1", "The vacuum nozzle contacts and lifts crumbs from the crevice.", "action"),
        atom("mech_result_1", "The same crevice is visibly clear after vacuuming."),
    ]
    if comp_id is not None:
        atoms.append(atom(comp_id, comp_text))
    return {
        "atoms": atoms,
        "transfer_decision": {
            "transferable": True,
            "abstract_mechanism": "Visible cordless suction removes debris from tight spaces",
            "preserved": [
                {"property": "hook", "atom_id": "keep_hook_1"},
                {"property": "mechanism", "atom_id": "keep_mechanism_1"},
                {"property": "problem", "atom_id": "mech_problem_1"},
                {"property": "intervention", "atom_id": "mech_intervention_1"},
                {"property": "result", "atom_id": "mech_result_1"},
            ],
            "reason": "The target product visibly demonstrates the approved cleaning mechanism.",
        },
    }


def source_context() -> dict:
    context = build_source_transfer_context(RADAR)
    # Keep the gate form but omit the element-account arrays: this test isolates
    # composition recovery rather than re-testing the separate five-element gate.
    context["target_fit_gate"] = {
        "transferable": True,
        "abstract_mechanism": "Visible cordless suction removes debris from tight spaces",
        "demo_form_target": "demonstration",
    }
    return context


class CompositionRecoveryTests(unittest.TestCase):
    def test_near_limit_incomplete_atom_gets_specific_short_rewrite_instruction(self):
        text = "One uninterrupted outdoor scene shows cordless pressure washer spraying foam onto hard surface with foam covering and"
        self.assertEqual(len(text), 117)
        errors = atom_text_quality_errors([atom("comp_1", text)])
        message = errors["comp_1"][0]
        self.assertIn(f"{MAX_TEXT}-character limit", message)
        self.assertIn("at most 90 characters", message)
        self.assertIn("Do NOT extend the fragment", message)

    def test_incomplete_composition_retries_only_that_atom_and_accepts_new_suffix(self):
        first = brain_response(
            "comp_1",
            "One uninterrupted outdoor scene shows cordless pressure washer spraying foam onto hard surface with foam covering and",
        )
        corrected = brain_response(
            "comp_2",
            "One close view shows the nozzle lifting crumbs from a narrow seat crevice.",
        )
        # The retry contract asks only for the failed atom. Other accepted atoms
        # must remain held by request_atoms and survive this one-atom response.
        corrected["atoms"] = [item for item in corrected["atoms"] if item["id"] == "comp_2"]
        calls: list[dict] = []
        rounds: list[dict] = []

        def call(request: dict) -> dict:
            calls.append(request)
            return first if len(calls) == 1 else corrected

        result = request_atoms(
            target_offer(),
            "Show a cordless vacuum removing visible crumbs from a car seat crevice.",
            1,
            call,
            source_transfer=source_context(),
            rounds_out=rounds,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(rounds[0]["pending"].keys(), {"comp_1"})
        self.assertEqual(rounds[1]["only_fix_ids"], ["comp_1"])
        self.assertEqual([item["id"] for item in calls[1]["only_fix"]], ["comp_1"])
        self.assertEqual(rounds[1]["pending"], {})
        self.assertIn("comp_2", {item.id for item in result})
        self.assertTrue({
            "keep_hook_1", "keep_mechanism_1", "mech_problem_1",
            "mech_intervention_1", "mech_result_1",
        }.issubset({item.id for item in result}))
        self.assertNotIn("comp_1", {item.id for item in result})
        self.assertEqual(len([item for item in result if item.id.startswith("comp_")]), 1)

    def test_parser_does_not_change_its_existing_dangling_word_behavior(self):
        text = "One continuous patio demo sprays foam with a"
        parsed = parse_atoms([atom("comp_word", text)])
        self.assertEqual(parsed.errors, {})
        self.assertEqual([item.id for item in parsed.atoms], ["comp_word"])

        # The second Brain validator still catches the same tail and owns its repair message.
        message = atom_text_quality_errors(parsed.atoms)["comp_word"][0]
        self.assertIn("ATOM_TEXT_INCOMPLETE", message)
        self.assertIn("dangling article, conjunction, or preposition", message)

    def test_parser_and_brain_use_the_same_short_rewrite_for_near_limit_dangling_endings(self):
        comma_text = (
            "One uninterrupted outdoor scene shows the cordless pressure washer "
            "spraying foam onto patio furniture, siding, or a car,"
        )
        self.assertEqual(len(comma_text), MAX_TEXT)
        raw_atom = atom("comp_1", comma_text)

        parsed = parse_atoms([raw_atom])
        self.assertEqual(parsed.atoms, [])
        parser_message = parsed.errors["comp_1"][0]
        brain_message = atom_text_quality_errors([raw_atom])["comp_1"][0]
        self.assertEqual(parser_message, brain_message)
        self.assertIn("at most 90 characters", parser_message)
        self.assertIn("Do NOT extend the fragment", parser_message)

        conjunction_text = "One uninterrupted outdoor demonstration shows the cordless pressure washer spraying foam across patio furniture and"
        self.assertGreaterEqual(len(conjunction_text), MAX_TEXT - 8)
        conjunction_message = atom_text_quality_errors([atom("comp_2", conjunction_text)])["comp_2"][0]
        self.assertIn("at most 90 characters", conjunction_message)
        self.assertIn("Do NOT extend the fragment", conjunction_message)

    def test_parser_rejection_evicts_previously_accepted_atom_with_the_same_id(self):
        first = brain_response(
            "comp_1",
            "One uninterrupted scene shows the pressure washer spraying foam onto a patio surface.",
        )
        first["atoms"] = [
            {**item, "text": "Keep the target product clearly visible and"}
            if item["id"] == "keep_hook_1" else item
            for item in first["atoms"]
        ]

        malformed = brain_response(
            "comp_1",
            "One uninterrupted outdoor scene shows the cordless pressure washer "
            "spraying foam onto patio furniture, siding, or a car,",
        )
        calls: list[dict] = []
        rounds: list[dict] = []

        def call(request: dict) -> dict:
            calls.append(request)
            return first if len(calls) == 1 else malformed

        with self.assertRaises(BrainError):
            request_atoms(
                target_offer(),
                "Show one uninterrupted target-native pressure-washing demonstration.",
                1,
                call,
                max_retries=1,
                source_transfer=source_context(),
                rounds_out=rounds,
            )

        self.assertEqual(len(calls), 2)
        self.assertIn("comp_1", rounds[1]["pending"])
        self.assertIn("at most 90 characters", rounds[1]["pending"]["comp_1"][0])
        self.assertNotIn("comp_1", rounds[1]["accepted_ids"])
        self.assertNotIn("#composition", rounds[1]["pending"])

    def test_missing_composition_still_raises_composition_required(self):
        response = brain_response(None)
        with self.assertRaises(BrainError) as raised:
            request_atoms(
                target_offer(),
                "Show a cordless vacuum removing visible crumbs from a car seat crevice.",
                1,
                lambda _request: response,
                max_retries=0,
                source_transfer=source_context(),
            )
        self.assertIn("#composition", raised.exception.errors)
        self.assertTrue(
            all(error.startswith("COMPOSITION_REQUIRED")
                for error in raised.exception.errors["#composition"])
        )


if __name__ == "__main__":
    unittest.main()
