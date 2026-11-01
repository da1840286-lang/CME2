"""Regressions for Brain retry reconciliation and transfer-decision element references."""
import copy
import json
import unittest
from pathlib import Path

from cme.brain import (atom_id_matches_role, atom_role, atom_text_quality_errors,
                       atom_budget_errors, build_request, build_source_transfer_context, composition_form_errors,
                       request_atoms, validate_transfer_decision)
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


def prepare_gate_text_policy_fixture(value):
    """Keep the historical source description, but adapt its target hook to a visual-only form."""
    prepared = copy.deepcopy(value)
    for item in prepared.get("source_elements_preserved", []):
        if isinstance(item, dict) and item.get("element") == "hook":
            item["target_form"] = (
                "Creator presents the target pressure washer outdoors before demonstrating foam application"
            )
    return prepared


def prepare_demo_form_fixture(value):
    """Adapt a historical provider response to its approved demo form and visual-only text policy."""
    prepared = copy.deepcopy(value)
    for item in prepared.get("atoms", []):
        if str(item.get("id") or "").startswith("comp_"):
            item["text"] = (
                "One continuous outdoor demo shows the nozzle spraying foam onto patio furniture "
                "and visibly coating the contacted area."
            )
        elif str(item.get("id") or "").startswith("keep_hook"):
            item["text"] = (
                "Creator presents the target cordless pressure washer outdoors before spraying foam onto a surface"
            )
    decision = prepared.get("transfer_decision") or {}
    preserved = decision.get("preserved") if isinstance(decision.get("preserved"), list) else []
    if not any(isinstance(item, dict) and item.get("property") == "problem" for item in preserved):
        problem = next(
            (item for item in prepared.get("atoms", [])
             if str(item.get("id") or "").startswith("mech_problem")),
            None,
        )
        if problem:
            preserved.append({"property": "problem", "atom_id": problem["id"]})
            decision["preserved"] = preserved
    return prepared


def response(suffix, include_result=True, include_reserved_qa=True):
    atoms = [
        atom(f"keep_hook_{suffix}", "Show the target cordless vacuum in a car interior without a power cord."),
        atom(f"keep_mechanism_{suffix}", "Cordless suction reaches crumbs in tight car seat crevices."),
        atom(f"mech_problem_{suffix}", "Visible crumbs and dust sit in a car seat crevice before cleaning."),
        atom(f"mech_intervention_{suffix}", "The target cordless vacuum nozzle removes crumbs from the seat crevice.", "action"),
        atom(f"comp_{suffix}", "One continuous car-seat demonstration shows the nozzle contacting the crevice as crumbs are lifted."),
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

    def test_real_failed_brain_run_response_is_repaired_by_quality_contract(self):
        fixture_path = Path(__file__).parent / "fixtures" / "brain_failed_run_41252251386.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        source_transfer = build_source_transfer_context({"result": fixture["source_analysis_result"]})
        source_transfer["target_fit_gate"] = prepare_gate_text_policy_fixture(fixture["target_fit_gate"])
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
        original_response = fixture["brain_response_round_1"]
        corrected_response = prepare_demo_form_fixture(original_response)

        def call(_request):
            calls.append(_request)
            return original_response if len(calls) == 1 else corrected_response

        atoms = request_atoms(
            offer, "Show the cordless pressure washer spraying foam onto outdoor hard surfaces.", 1, call,
            source_transfer=source_transfer, decision_out=decision_out, rounds_out=rounds_out,
        )

        self.assertEqual(len(calls), 2, "unfinished composition and incomplete Gate preservation must be retried")
        self.assertEqual(len(rounds_out), 2)
        self.assertTrue(rounds_out[0]["pending"])
        self.assertIn("DEMO_FORM_CONFLICT", json.dumps(rounds_out[0]["pending"]))
        self.assertIn("Gate-preserved source element 'problem'", json.dumps(rounds_out[0]["pending"]))
        self.assertEqual(rounds_out[1]["pending"], {})
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
        source_transfer["target_fit_gate"] = prepare_gate_text_policy_fixture(fixture["target_fit_gate"])
        offer = OfferCard(
            offer_id="cordless-pressure-washer-us-001", name="Cordless Pressure Washer",
            product_type="cordless pressure washer", verified_claims=("Cordless pressure washer",),
            forbidden_claims=("guaranteed results", "medical", "cures", "unverified PSI claims",
                              "cleans every surface", "removes all dirt"),
            photos=("target_product_reference.jpg",), platform="TikTok Shop", geo="US",
            product_category="Home Improvement", branding_policy="no_branding",
        )
        first = prepare_demo_form_fixture(fixture["brain_response_round_1"])
        first["atoms"] = [a for a in first["atoms"] if a["id"] != "mech_result"]
        second = prepare_demo_form_fixture(fixture["brain_response_round_1"])
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

    def test_incomplete_atom_tail_is_rejected(self):
        errors = atom_text_quality_errors([atom("comp_1", "One continuous patio surface, with a")])
        self.assertIn("TEXT_REWRITE_REQUIRED", errors["comp_1"][0])
        self.assertEqual(
            atom_text_quality_errors([atom("comp_1", "One continuous patio demo shows foam coating the sprayed area.")]),
            {},
        )

    def test_demonstration_gate_rejects_unapproved_left_right_contrast(self):
        fit_gate = {"demo_form_target": "demonstration"}
        conflicting = atom(
            "comp_1",
            "One continuous outdoor surface: untreated dirty area on left, foam-coated treated area on right, nozzle at boundary",
        )
        errors = composition_form_errors([conflicting], fit_gate)
        self.assertIn("DEMO_FORM_CONFLICT", errors["comp_1"][0])

    def test_collage_retry_feedback_obeys_the_gate_demo_form(self):
        split = atom("comp_1", "split-screen collage with two separate regions")
        demo_errors = atom_budget_errors([split], 1, fit_gate={"demo_form_target": "demonstration"})
        self.assertIn("FRAME_COLLAGE_WORDING", demo_errors["comp_1"][0])
        self.assertIn("do not introduce left/right zones", demo_errors["comp_1"][0])

        contrast_errors = atom_budget_errors([split], 1, fit_gate={"demo_form_target": "comparison"})
        self.assertIn("active intervention at the visible boundary", contrast_errors["comp_1"][0])

    def test_comparison_gate_allows_an_approved_in_scene_contrast(self):
        fit_gate = {"demo_form_target": "comparison"}
        contrast = atom(
            "comp_1",
            "One continuous outdoor surface: untreated dirty area on left, foam-coated treated area on right, nozzle at boundary",
        )
        self.assertEqual(composition_form_errors([contrast], fit_gate), {})

    def test_brain_retries_an_incomplete_atom_before_accepting_the_contract(self):
        first = response("1", include_reserved_qa=False)
        first["atoms"] = [
            {**item, "text": "One continuous patio demo sprays foam with a"}
            if item["id"] == "comp_1" else item
            for item in first["atoms"]
        ]
        second = response("1", include_reserved_qa=False)
        calls, rounds = [], []

        def call(request):
            calls.append(request)
            return first if len(calls) == 1 else second

        result = request_atoms(
            target_offer(), "Show the cordless vacuum cleaning visible crumbs.", 1, call,
            source_transfer=build_source_transfer_context(RADAR), rounds_out=rounds,
        )
        comp = next(item for item in result if item.id == "comp_1")
        self.assertEqual(len(calls), 2)
        self.assertIn("TEXT_REWRITE_REQUIRED", json.dumps(rounds[0]["pending"]))
        self.assertEqual(rounds[1]["pending"], {})
        self.assertTrue(comp.text.endswith("lifted."))

    def test_brain_retries_composition_that_conflicts_with_gate_demo_form(self):
        source = build_source_transfer_context(RADAR)
        source["target_fit_gate"] = {
            "transferable": True,
            "demo_form_target": "demonstration",
            "source_elements_preserved": [],
            "source_elements_dropped": [],
        }
        first = response("1", include_reserved_qa=False)
        first["atoms"] = [
            {**item, "text": "One continuous surface: untreated dirty area on left, treated clean area on right"}
            if item["id"] == "comp_1" else item
            for item in first["atoms"]
        ]
        second = response("1", include_reserved_qa=False)
        calls, rounds = [], []

        def call(request):
            calls.append(request)
            return first if len(calls) == 1 else second

        result = request_atoms(
            target_offer(), "Show the cordless vacuum cleaning visible crumbs.", 1, call,
            source_transfer=source, rounds_out=rounds,
        )
        comp = next(item for item in result if item.id == "comp_1")
        self.assertEqual(len(calls), 2)
        self.assertIn("DEMO_FORM_CONFLICT", json.dumps(rounds[0]["pending"]))
        self.assertEqual(rounds[1]["pending"], {})
        self.assertNotIn("left", comp.text.lower())
        self.assertNotIn("right", comp.text.lower())

    def test_brain_prompt_uses_gate_demo_form_instead_of_hard_coding_before_after(self):
        from tests.test_target_fit_gate import target_offer

        source = build_source_transfer_context(RADAR)
        source["target_fit_gate"] = {"demo_form_target": "demonstration"}
        rules = " ".join(build_request(target_offer(), "brief", 1, source_transfer=source)["rules"])
        self.assertIn("ONE-SHOT DEMONSTRATION CONTRACT", rules)
        self.assertIn("Do NOT create separate untreated/treated", rules)
        self.assertNotIn("left part untreated", rules)

        source["target_fit_gate"]["demo_form_target"] = "comparison"
        rules = " ".join(build_request(target_offer(), "brief", 1, source_transfer=source)["rules"])
        self.assertIn("ONE-SHOT COMPOSITION CONTRACT", rules)
        self.assertIn("explicitly approved the contrast form", rules)

    def test_brain_preserved_list_must_cover_all_gate_preserved_elements(self):
        fit_gate = {
            "source_elements_preserved": [{"element": name} for name in
                ("hook", "problem", "mechanism", "intervention", "result")],
            "source_elements_dropped": [],
        }
        atoms = [
            atom("keep_hook_1", "A visible hook appears before the target product."),
            atom("keep_mechanism_1", "Cordless suction removes crumbs from tight spaces.", "action"),
            atom("mech_problem_1", "Crumbs remain in a narrow seat crevice."),
            atom("mech_intervention_1", "The nozzle removes crumbs from the crevice.", "action"),
            atom("mech_result_1", "The crevice is visibly clear after vacuuming."),
        ]
        decision = {
            "transferable": True,
            "abstract_mechanism": "Cordless suction removes debris from tight spaces",
            "preserved": [
                {"property": "hook", "atom_id": "keep_hook_1"},
                {"property": "mechanism", "atom_id": "keep_mechanism_1"},
                {"property": "intervention", "atom_id": "mech_intervention_1"},
                {"property": "result", "atom_id": "mech_result_1"},
            ],
            "reason": "The target product visibly demonstrates the same causal mechanism.",
        }
        errors = validate_transfer_decision(decision, atoms, fit_gate=fit_gate)
        self.assertTrue(any("Gate-preserved source element 'problem'" in error for error in errors))

        decision["preserved"].append({"property": "problem", "atom_id": "mech_problem_1"})
        self.assertEqual(validate_transfer_decision(decision, atoms, fit_gate=fit_gate), [])

    def test_transfer_schema_accepts_all_five_gate_elements(self):
        from cme.atoms import TRANSFER_ATOM_SCHEMA
        props = TRANSFER_ATOM_SCHEMA["properties"]["transfer_decision"]["properties"]["preserved"]
        self.assertEqual(props["maxItems"], 5)

