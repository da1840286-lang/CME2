"""Regressions for Brain retry reconciliation and transfer-decision element references."""
import copy
import json
import unittest
from pathlib import Path

from cme.brain import (atom_id_matches_role, atom_role, build_request,
                       build_source_transfer_context, request_atoms,
                       validate_transfer_decision)
from cme.offer import OfferCard
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
    def test_transfer_atoms_require_visible_physical_intervention_not_nearby_product_placement(self):
        from cme.offer import load_offer
        from tests.test_target_fit_gate import EX

        req = build_request(
            load_offer(EX), "Show the product actively solving the visible problem.", 1,
            source_transfer=build_source_transfer_context(RADAR),
        )
        rules = " ".join(req["rules"])
        self.assertIn("ACTIVE INTERVENTION EVIDENCE", rules)
        self.assertIn("physically engaging the exact problem material", rules)
        self.assertIn("Do not describe the product as merely resting beside", rules)
        self.assertIn("visible contact point linking the action to the change", rules)

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


    def test_bare_and_suffixed_ids_are_the_same_logical_role(self):
        self.assertEqual(atom_role("keep_hook"), "keep_hook")
        self.assertEqual(atom_role("keep_hook_01"), "keep_hook")
        self.assertEqual(atom_role("mech_intervention"), "mech_intervention")
        self.assertEqual(atom_role("mech_intervention_1"), "mech_intervention")
        self.assertEqual(atom_role("comp_"), "comp")
        self.assertEqual(atom_role("comp_1"), "comp")
        self.assertTrue(atom_id_matches_role("mech_result", "mech_result_"))
        self.assertFalse(atom_id_matches_role("mech_intervention_1", "mech_result"))

    def test_real_failed_brain_run_response_passes_without_repair_round(self):
        fixture_path = Path(__file__).parent / "fixtures" / "brain_failed_run_41252251386.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        source_transfer = build_source_transfer_context({"result": fixture["source_analysis_result"]})
        source_transfer["target_fit_gate"] = fixture["target_fit_gate"]
        offer = OfferCard(
            offer_id="cordless-pressure-washer-us-001",
            name="Cordless Pressure Washer",
            product_type="cordless pressure washer",
            verified_claims=("Cordless pressure washer",),
            forbidden_claims=("guaranteed results", "medical", "cures", "unverified PSI claims",
                              "cleans every surface", "removes all dirt"),
            photos=("target_product_reference.jpg",),
            platform="TikTok Shop", geo="US", product_category="Home Improvement",
            branding_policy="no_branding",
        )
        calls = []
        decision_out = {}
        rounds_out = []
        real_response = fixture["brain_response_round_1"]

        def call(_request):
            calls.append(_request)
            return real_response

        atoms = request_atoms(
            offer, "Show the cordless pressure washer spraying foam onto outdoor hard surfaces.", 1, call,
            source_transfer=source_transfer, decision_out=decision_out, rounds_out=rounds_out,
        )

        self.assertEqual(len(calls), 1, "the actual first-round response already carries every required atom")
        self.assertEqual(len(rounds_out), 1)
        self.assertEqual(rounds_out[0]["pending"], {})
        self.assertEqual(decision_out["abstract_mechanism"], fixture["target_fit_gate"]["abstract_mechanism"])
        self.assertEqual(rounds_out[0]["abstract_mechanism_source"], "target_fit_gate")
        self.assertTrue(rounds_out[0].get("abstract_mechanism_canonicalized"))

        by_id = {item.id: item for item in atoms}
        for atom_id in ("keep_hook", "keep_mechanism", "mech_problem", "mech_intervention",
                        "mech_result", "comp_", "product_vis"):
            self.assertIn(atom_id, by_id)
        self.assertLessEqual(
            sum(a.channel == "image" and a.priority == "MUST" for a in atoms), 7,
            "the fix must reconcile IDs rather than raise the frame budget",
        )
        self.assertFalse(any(a.id.startswith("keep_intervention") or a.id.startswith("keep_result") for a in atoms))

    def test_retry_replaces_bare_real_run_ids_with_suffixed_replacements(self):
        fixture_path = Path(__file__).parent / "fixtures" / "brain_failed_run_41252251386.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        source_transfer = build_source_transfer_context({"result": fixture["source_analysis_result"]})
        source_transfer["target_fit_gate"] = fixture["target_fit_gate"]
        offer = OfferCard(
            offer_id="cordless-pressure-washer-us-001", name="Cordless Pressure Washer",
            product_type="cordless pressure washer", verified_claims=("Cordless pressure washer",),
            forbidden_claims=("guaranteed results", "medical", "cures", "unverified PSI claims",
                              "cleans every surface", "removes all dirt"),
            photos=("target_product_reference.jpg",), platform="TikTok Shop", geo="US",
            product_category="Home Improvement", branding_policy="no_branding",
        )
        first = copy.deepcopy(fixture["brain_response_round_1"])
        first["atoms"] = [a for a in first["atoms"] if a["id"] != "mech_result"]
        second = copy.deepcopy(fixture["brain_response_round_1"])
        id_map = {
            atom["id"]: f"{atom_role(atom['id'])}_1"
            for atom in second["atoms"] if atom_role(atom["id"])
        }
        for atom in second["atoms"]:
            atom["id"] = id_map.get(atom["id"], atom["id"])
        for item in second["transfer_decision"]["preserved"]:
            item["atom_id"] = id_map.get(item["atom_id"], item["atom_id"])
        calls = []
        rounds = []

        def call(_request):
            calls.append(True)
            return first if len(calls) == 1 else second

        atoms = request_atoms(
            offer, "Show the cordless pressure washer spraying foam onto outdoor hard surfaces.", 1, call,
            source_transfer=source_transfer, rounds_out=rounds,
        )
        ids = {a.id for a in atoms}
        self.assertEqual(len(calls), 2)
        self.assertTrue(rounds[0]["pending"])
        self.assertEqual(rounds[1]["pending"], {})
        for role in ("keep_hook", "keep_mechanism", "mech_problem", "mech_intervention", "mech_result", "comp"):
            self.assertIn(f"{role}_1", ids)
            self.assertNotIn(role, ids)
        self.assertLessEqual(sum(a.channel == "image" and a.priority == "MUST" for a in atoms), 7)

    def test_preserved_intervention_and_result_use_existing_mechanism_link_atoms(self):
        atoms = [
            atom("keep_mechanism_1", "Cordless spray coats outdoor surfaces without manual scrubbing.", "action"),
            atom("mech_intervention_1", "Nozzle sprays foam directly onto outdoor hard surfaces.", "action"),
            atom("mech_result_1", "Foam visibly coats the outdoor surface after spraying."),
        ]
        decision = {
            "transferable": True,
            "abstract_mechanism": "A cordless washer applies foam to hard surfaces.",
            "preserved": [
                {"property": "mechanism", "atom_id": "keep_mechanism_1"},
                {"property": "intervention", "atom_id": "mech_intervention_1"},
                {"property": "result", "atom_id": "mech_result_1"},
            ],
            "reason": "The target product visibly demonstrates the approved mechanism.",
        }
        self.assertEqual(validate_transfer_decision(decision, atoms), [])
        wrong = {**decision, "preserved": [
            {"property": "intervention", "atom_id": "mech_result_1"},
        ]}
        self.assertTrue(any("mech_intervention_" in error
                            for error in validate_transfer_decision(wrong, atoms)))

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
