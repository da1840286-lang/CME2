"""Focused regressions for the local Target-Fit / FRAME / Judge patch."""
from __future__ import annotations
from types import SimpleNamespace

from cme.atoms import Atom
from cme.brain import reserved_system_atom_errors, validate_transfer_fit_decision
from cme.judge_protocol import decide, required_aspects
from cme.state import attempt_consumed, count_attempts, record_attempt, new_run
from tests.helpers import gate_account


def _gate_decision(mapping: str, evidence=None, reason="The target function is explicitly confirmed in the OfferCard."):
    account = gate_account(preserved=("problem", "mechanism", "intervention", "result"), dropped=("hook",))
    # Keep source/target forms realistic while preserving the sample's causal mechanism.
    account["source_elements_preserved"][1]["target_form"] = mapping
    account["source_elements_preserved"][2]["target_form"] = mapping
    return {
        **account,
        "transferable": True,
        "abstract_mechanism": "The product applies a visible action to a defined surface",
        "target_native_mapping": mapping,
        "target_evidence": evidence or ["Cordless pressure washer"],
        "reason": reason,
    }


def _offer(claims=("Cordless pressure washer",), confirmations=()):
    return SimpleNamespace(
        name="SEESE Cordless Pressure Washer", product_type="cordless pressure washer",
        product_category="pressure washer", verified_claims=tuple(claims),
        confirmations=tuple(confirmations), branding_policy="target_brand_allowed",
    )


def _obs(rows):
    return {"results": rows}


def _condition(aspect, oid, quote):
    return {"aspect": aspect, "claim": quote, "observation_ids": [oid], "quote": quote, "holds": True}


def test_target_fit_rejects_foam_function_without_functional_offer_evidence():
    d = _gate_decision("Show the pressure washer spraying foam onto bathroom tiles")
    errors = validate_transfer_fit_decision(_offer(), d)
    assert any("TARGET_CAPABILITY_UNVERIFIED" in error for error in errors)


def test_target_fit_accepts_foam_function_when_offer_confirmation_supports_it():
    quote = "Sprays foam when used with the included detergent bottle"
    d = _gate_decision(
        "Show the pressure washer spraying foam onto bathroom tiles",
        evidence=[quote],
    )
    assert validate_transfer_fit_decision(_offer(confirmations=(quote,)), d) == []


def test_target_fit_rejects_plausibility_language_outside_reason():
    d = _gate_decision("The pressure washer plausibly sprays water onto the patio")
    errors = validate_transfer_fit_decision(_offer(), d)
    assert any("TARGET_CAPABILITY_UNCERTAIN" in error for error in errors)


def test_brand_is_not_allowed_inside_visual_atoms():
    atom = Atom("keep_hook_01", "scene", "Show the SEESE pressure washer in use", "MUST", "image", 1)
    errors = reserved_system_atom_errors([atom], ["SEESE"])
    assert any("BRAND_IN_VISUAL_ATOM" in error for messages in errors.values() for error in messages)


def test_mechanism_aspects_are_conditional_and_keep_strict_evidence():
    intervention = Atom("mech_intervention_01", "action", "Nozzle contacts the tile surface", "MUST", "image", 1)
    result = Atom("mech_result_01", "action", "Foam visibly covers the tile surface", "MUST", "image", 1)
    contrast_result = Atom("mech_result_02", "action", "A clean strip appears beside untouched grime", "MUST", "image", 1)
    assert required_aspects(intervention) == {"object_presence", "contact_action"}
    assert required_aspects(result) == {"changed_state"}
    assert required_aspects(contrast_result) == {"changed_state", "problem_state"}

    observations = [
        {"id": "o1", "kind": "object", "text": "A green pressure washer is visible"},
        {"id": "o2", "kind": "relation", "text": "The nozzle touches the tile surface"},
        {"id": "o3", "kind": "region", "text": "Thick white foam covers the tile surface"},
    ]
    passed = decide(_obs([
        {"id": intervention.id, "conditions": [
            _condition("object_presence", "o1", "A green pressure washer is visible"),
            _condition("contact_action", "o2", "The nozzle touches the tile surface"),
        ]},
        {"id": result.id, "conditions": [
            _condition("changed_state", "o3", "Thick white foam covers the tile surface"),
        ]},
    ]), [intervention, result], observations)
    assert passed[intervention.id]["pass"] is True
    assert passed[result.id]["pass"] is True

    missing_contact = decide(_obs([{"id": intervention.id, "conditions": [
        _condition("object_presence", "o1", "A green pressure washer is visible")
    ]}]), [intervention], observations)
    assert "missing proof aspect: contact_action" in missing_contact[intervention.id]["evidence"]


def test_infra_error_does_not_consume_generated_frame_attempt():
    state = new_run("run")
    record_attempt(state, 1, "drec_infra", "ERROR_InfraError", "ZeroGPU quota", generated=False)
    record_attempt(state, 1, "drec_qa", "FAILED_QA", generated=True)
    assert count_attempts(state.attempts, shot=1) == 1
    assert attempt_consumed(state.attempts[0]) is False
    assert attempt_consumed(state.attempts[1]) is True


def test_legacy_infra_error_is_not_counted_but_legacy_qa_failure_is():
    rows = [
        {"shot": 1, "receipt_id": "drec_old_infra", "status": "ERROR_InfraError"},
        {"shot": 1, "receipt_id": "drec_old_qa", "status": "FAILED_QA"},
    ]
    assert count_attempts(rows, shot=1) == 1