"""Regressions for Brain retry reconciliation and transfer-decision element references."""
import unittest

from cme.brain import build_source_transfer_context, request_atoms, validate_transfer_decision
from tests.test_target_fit_gate import target_offer


RADAR = {
    "result": {
        "hook": {"text": "Cordless cleaning in car interiors", "kind": "OBSERVED"},
        "buyer_problem": {"text": "Crumbs remain in seat crevices", "kind": "OBSERVED"},
        "product_intervention": {"text": "A cordless handheld vacuum removes crumbs", "kind": "OBSERVED"},
        "result_demo": {"text": "The crevice is visibly cleaner", "kind": "OBSERVED"},
        "mechanism_hypothesis": {"text": "Visible suction removes debris from tight spaces", "kind": "INFERRED"},
    }
}


def atom(atom_id, text, kind="scene", channel="image", priority="MUST"):
    return {"id": atom_id, "kind": kind, "text": text, "priority": priority, "channel": channel, "shot": 1}


def response(suffix, include_result=True, include_reserved_qa=True):
    atoms = [
        atom(f"keep_hook_{suffix}", "Show the target cordless vacuum in a car interior without a power cord."),
        atom(f"keep_mechanism_{suffix}", "Cordless suction reaches crumbs in tight car seat crevices."),
        atom(f"mech_problem_{suffix}", "Visible crumbs and dust sit in a car seat crevice before cleaning."),
        atom(f"mech_intervention_{suffix}", "The target cordless vacuum nozzle removes crumbs from the seat crevice.", "action"),
        atom(f"comp_{suffix}", "One continuous car surface has a dirty crevice on the left and a clean strip on the right."),
    ]
    if include_result:
        atoms.append(atom(f"mech_result_{suffix}", "The same seat crevice is visibly free of crumbs after vacuuming."))
    if include_reserved_qa:
        atoms.append(atom(f"qa_check_{suffix}", "No text, logos, or brand marks appear in the frame.", "qa_check", "qa"))
    return {
        "atoms": atoms,
        "transfer_decision": {
            "transferable": True,
            "abstract_mechanism": "Visible cordless suction removes debris from tight car spaces",
            "preserved": [
                {"property": "hook", "atom_id": f"keep_hook_{suffix}"},
                {"property": "mechanism", "atom_id": f"keep_mechanism_{suffix}"},
            ],
            "reason": "The target product can visibly demonstrate cordless suction in car crevices.",
        },
    }


class BrainRetryContractTests(unittest.TestCase):
    def test_retry_replaces_old_role_ids_and_ignores_system_owned_branding_qa(self):
        calls = []

        def call(request):
            calls.append(request)
            return response("01", include_result=False) if len(calls) == 1 else response("1")

        atoms = request_atoms(
            target_offer(), "Show the cordless vacuum removing visible crumbs.", 1, call,
            source_transfer=build_source_transfer_context(RADAR),
        )
        ids = {a.id for a in atoms}
        image_must = [a for a in atoms if a.channel == "image" and a.priority == "MUST"]
        self.assertEqual(len(calls), 2)  # only the genuinely missing mechanism result caused a retry
        self.assertIn("keep_hook_1", ids)
        self.assertNotIn("keep_hook_01", ids)
        self.assertIn("keep_mechanism_1", ids)
        self.assertNotIn("keep_mechanism_01", ids)
        self.assertIn("mech_problem_1", ids)
        self.assertNotIn("mech_problem_01", ids)
        self.assertIn("mech_intervention_1", ids)
        self.assertNotIn("mech_intervention_01", ids)
        self.assertIn("mech_result_1", ids)
        self.assertNotIn("mech_result_01", ids)
        self.assertIn("comp_1", ids)
        self.assertNotIn("comp_01", ids)
        self.assertFalse(any(a.kind == "qa_check" for a in atoms))
        self.assertLessEqual(len(image_must), 7)

    def test_canonical_gate_element_property_requires_matching_keep_atom(self):
        atoms = [
            atom("keep_hook_1", "Target vacuum shown in a car interior."),
            atom("keep_mechanism_1", "Cordless suction removes crumbs from tight spaces.", "action"),
        ]
        valid = {
            "transferable": True,
            "abstract_mechanism": "Cordless suction removes debris from tight spaces",
            "preserved": [
                {"property": "hook", "atom_id": "keep_hook_1"},
                {"property": "mechanism", "atom_id": "keep_mechanism_1"},
            ],
            "reason": "The target product visibly demonstrates the same mechanism.",
        }
        self.assertEqual(validate_transfer_decision(valid, atoms), [])
        invalid = {**valid, "preserved": [{"property": "mechanism", "atom_id": "keep_hook_1"}]}
        self.assertTrue(any("keep_mechanism_ prefix" in error
                            for error in validate_transfer_decision(invalid, atoms)))
