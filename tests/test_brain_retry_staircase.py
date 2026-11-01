"""Real BRAIN retry regression: one-atom repair, explicit reserve routing, and provenance."""
from __future__ import annotations

import copy
import gzip
import json
import unittest
from pathlib import Path

from cme.atoms import MAX_TEXT, MODEL_ATOM_TEXT_MAX
from cme.brain import (BrainError, _retry_atom_allowed, _retry_request, _retry_target,
                      build_source_transfer_context, request_atoms)
from tests.test_brain_text_boundary_recovery import pressure_washer_offer

ROOT = Path(__file__).resolve().parent
REAL_RUN = ROOT / "fixtures" / "brain_output_invalid_41324289662.json"
TEXT_FIXTURE = ROOT / "fixtures" / "brain_text_boundary_real_responses.json.gz"


def source_with_gate():
    with gzip.open(TEXT_FIXTURE, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    source = build_source_transfer_context(fixture["source_contract"])
    source["target_fit_gate"] = copy.deepcopy(
        fixture["responses"]["run_41305261765_gate_1"]["transfer_decision"]
    )
    hook = next(item for item in source["target_fit_gate"]["source_elements_preserved"]
                if item["element"] == "hook")
    hook["target_form"] = (
        "Person presents the cordless pressure washer outdoors before spraying foam onto patio furniture"
    )
    return source


def provenance(adapter, model, fallback=False):
    return {
        "primary_adapter": "openrouter_free_brain",
        "primary_model_id": "nvidia/nemotron-3.5-super-vl:free",
        "adapter": adapter,
        "provider": "openrouter",
        "model_id": model,
        "responded_model": model,
        "fallback_used": fallback,
    }


def fixed_comp_response(atom_template):
    atom = copy.deepcopy(atom_template)
    atom["text"] = "Pressure washer sprays foam onto dirty patio furniture, visibly coating its surface."
    return {"atoms": [atom]}


class BrainRetryStaircaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_fixture = json.loads(REAL_RUN.read_text(encoding="utf-8"))
        cls.real_response = cls.real_fixture["response_round_1"]
        cls.comp_atom = next(item for item in cls.real_response["atoms"] if item["id"] == "comp_01")
        cls.source = source_with_gate()
        cls.offer = pressure_washer_offer()

    def make_call(self, primary_responses, fallback_responses):
        state = {"primary": [], "fallback": []}

        def call(request):
            state["primary"].append(copy.deepcopy(request))
            call.info = provenance("openrouter_free_brain", "nvidia/nemotron-3.5-super-vl:free")
            response = primary_responses[min(len(state["primary"]) - 1, len(primary_responses) - 1)]
            return copy.deepcopy(response)

        def call_fallback(request):
            state["fallback"].append(copy.deepcopy(request))
            call.info = provenance(
                "openrouter_free_brain_reserve", "google/gemma-4-26b-a4b-it:free", fallback=True
            )
            response = fallback_responses[min(len(state["fallback"]) - 1, len(fallback_responses) - 1)]
            return copy.deepcopy(response)

        call.info = provenance("openrouter_free_brain", "nvidia/nemotron-3.5-super-vl:free")
        call.call_fallback = call_fallback
        call.has_fallback = True
        return call, state

    def test_all_retry_schemas_keep_provider_headroom_above_code_limit(self):
        target = {"key": "comp_01", "mode": "exact", "id": "comp_01", "role": "comp"}
        expected_targets = {1: 100, 2: 100, 3: 80}

        for retry_number, expected_target in expected_targets.items():
            with self.subTest(retry_number=retry_number):
                request, strategy = _retry_request(
                    self.offer,
                    "Create a target-native outdoor pressure-washer demo.",
                    1,
                    self.source,
                    target,
                    ["TEXT_REWRITE_REQUIRED: shorten the complete sentence"],
                    self.comp_atom["text"],
                    retry_number,
                )
                schema_limit = request["schema"]["properties"]["atoms"]["items"]["properties"]["text"]["maxLength"]
                self.assertEqual(strategy["text_max_chars"], expected_target)
                self.assertGreaterEqual(schema_limit, MAX_TEXT)
                self.assertEqual(schema_limit, MODEL_ATOM_TEXT_MAX)
                self.assertIn(f"no longer than {expected_target} characters", " ".join(request["rules"]))

    def test_run_413242_uses_primary_then_reserve_and_keeps_original_atoms(self):
        original = copy.deepcopy(self.real_response)
        corrected = fixed_comp_response(self.comp_atom)
        call, state = self.make_call(
            [original, original],
            [corrected],
        )
        trace, atom_sources, decision = [], {}, {}

        atoms = request_atoms(
            self.offer,
            "Create a target-native outdoor pressure-washer demo.",
            1,
            call,
            source_transfer=self.source,
            rounds_out=trace,
            decision_out=decision,
            atom_provenance_out=atom_sources,
        )

        self.assertEqual(len(trace), 3)
        self.assertEqual(len(state["primary"]), 2)
        self.assertEqual(len(state["fallback"]), 1)
        self.assertEqual(trace[0]["pending"].keys(), {"comp_01"})
        self.assertEqual(trace[1]["only_fix_ids"], ["comp_01"])
        self.assertEqual(trace[2]["only_fix_ids"], ["comp_01"])
        self.assertEqual(trace[1]["retry_strategy"], "single_atom_primary")
        self.assertEqual(trace[2]["retry_strategy"], "single_atom_reserve")
        self.assertEqual(trace[2]["actual_route"], "fallback")
        self.assertEqual(trace[2]["model_id"], "google/gemma-4-26b-a4b-it:free")
        self.assertTrue(trace[2]["fallback_used"])

        retry_schema = state["primary"][1]["schema"]
        self.assertEqual(retry_schema["properties"]["atoms"]["maxItems"], 1)
        atom_schema = retry_schema["properties"]["atoms"]["items"]["properties"]
        self.assertEqual(atom_schema["id"]["const"], "comp_01")
        self.assertGreaterEqual(atom_schema["text"]["maxLength"], MAX_TEXT)
        self.assertEqual(atom_schema["text"]["maxLength"], MODEL_ATOM_TEXT_MAX)
        self.assertEqual(state["primary"][1]["retry_strategy"]["text_max_chars"], 100)
        self.assertEqual(state["primary"][1]["retry_contract"]["expected_id"], "comp_01")

        result_by_id = {atom.id: atom for atom in atoms}
        first_response_by_id = {item["id"]: item for item in original["atoms"]}
        self.assertEqual(set(result_by_id), set(first_response_by_id))
        for atom_id, before in first_response_by_id.items():
            if atom_id != "comp_01":
                self.assertEqual(result_by_id[atom_id].text, before["text"], atom_id)
                self.assertEqual(atom_sources[atom_id]["model_id"], "nvidia/nemotron-3.5-super-vl:free")
        self.assertEqual(atom_sources["comp_01"]["model_id"], "google/gemma-4-26b-a4b-it:free")
        self.assertTrue(atom_sources["comp_01"]["fallback_used"])
        self.assertEqual(len({row["request_sha256"] for row in trace}), 3)
        self.assertTrue(trace[1]["ignored_rewrites"])
        self.assertTrue(all("text_sha256" in row and "text" not in row for row in trace[1]["ignored_rewrites"]))
        self.assertNotIn("Person holding cordless pressure washer", trace[1]["response_excerpt"])
        self.assertTrue(decision["transferable"])


    def test_narrow_retry_still_runs_the_full_text_policy_validator(self):
        original = copy.deepcopy(self.real_response)
        bad_overlay = copy.deepcopy(self.comp_atom)
        bad_overlay["text"] = "Add bold title text reading CLEAN NOW above the washer."
        call, state = self.make_call([original, {"atoms": [bad_overlay]}], [])
        trace = []

        with self.assertRaises(BrainError) as raised:
            request_atoms(
                self.offer,
                "Create a target-native outdoor pressure-washer demo.",
                1,
                call,
                max_retries=1,
                source_transfer=self.source,
                rounds_out=trace,
            )

        self.assertIn("comp_01", raised.exception.errors)
        self.assertTrue(any(error.startswith("NO_ADDED_TEXT_POLICY:")
                            for error in raised.exception.errors["comp_01"]))
        self.assertEqual(len(state["primary"]), 2)
        self.assertEqual(len(state["fallback"]), 0)
        self.assertEqual(trace[1]["retry_strategy"], "single_atom_primary")
        self.assertIn("NO_ADDED_TEXT_POLICY", json.dumps(trace[1]["pending"]))


    def test_mechanism_shot_repair_authorizes_only_the_matching_link_and_shot(self):
        target = _retry_target(
            {"#mechanism_shot_result": ["result link must be assigned to shot 2"]},
            {"mech_result_01": object()},
        )
        self.assertEqual(target, {
            "key": "#mechanism_shot_result", "mode": "role_shot",
            "role": "mech_result", "shot": 2,
        })
        request, _ = _retry_request(
            self.offer, "Create a two-shot demo.", 2, self.source, target,
            ["result link must be assigned to shot 2"], "The result appears in shot one.", 1,
        )
        atom_schema = request["schema"]["properties"]["atoms"]
        item_props = atom_schema["items"]["properties"]
        self.assertEqual(atom_schema["maxItems"], 1)
        self.assertEqual(item_props["id"]["pattern"], r"^mech_result(?:_[a-z0-9_]+)?$")
        self.assertEqual(item_props["shot"]["const"], 2)
        self.assertTrue(_retry_atom_allowed({"id": "mech_result_02", "shot": 2,
                                             "priority": "MUST", "channel": "image"}, target))
        self.assertFalse(_retry_atom_allowed({"id": "mech_result_02", "shot": 1,
                                              "priority": "MUST", "channel": "image"}, target))
        self.assertFalse(_retry_atom_allowed({"id": "mech_intervention_02", "shot": 2,
                                              "priority": "MUST", "channel": "image"}, target))

    def test_unmapped_frame_budget_error_fails_closed_without_extra_provider_calls(self):
        texts = [
            "A red umbrella rests beside an empty stone fountain in sunlight.",
            "Several copper coins spill across a blue velvet cushion.",
            "A ceramic teapot stands near folded paper maps on a wooden shelf.",
            "Two yellow bicycles lean against a brick wall beside a green gate.",
            "A glass vase holds white tulips on a narrow marble table.",
            "Fresh oranges fill a woven basket beside a striped linen cloth.",
            "A small wooden boat floats across still water near distant reeds.",
            "Blue ceramic tiles form a geometric pattern across the kitchen wall.",
        ]
        response = {"atoms": [
            {"id": f"extra_{index}", "kind": "scene", "text": text,
             "priority": "MUST", "channel": "image", "shot": 1}
            for index, text in enumerate(texts)
        ]}
        calls, trace = [], []

        def call(request):
            calls.append(request)
            return copy.deepcopy(response)

        with self.assertRaises(BrainError) as raised:
            request_atoms(self.offer, "Create a product scene.", 1, call,
                          max_retries=3, rounds_out=trace)

        self.assertIn("#must_budget_1", raised.exception.errors)
        self.assertEqual(len(calls), 1)
        self.assertEqual(trace[-1]["retry_strategy"], "blocked_no_safe_allowlist")
        self.assertEqual(trace[-1]["actual_route"], "not_called")

    def test_repeated_real_bad_atom_uses_three_distinct_retry_requests_then_fails_closed(self):
        original = copy.deepcopy(self.real_response)
        call, state = self.make_call([original, original], [original, original])
        trace = []
        with self.assertRaises(BrainError) as raised:
            request_atoms(
                self.offer,
                "Create a target-native outdoor pressure-washer demo.",
                1,
                call,
                source_transfer=self.source,
                rounds_out=trace,
            )

        self.assertIn("comp_01", raised.exception.errors)
        self.assertEqual(len(trace), 4)
        self.assertEqual(len(state["primary"]), 2)
        self.assertEqual(len(state["fallback"]), 2)
        hashes = [row["request_sha256"] for row in trace]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(
            [row["retry_strategy"] for row in trace],
            ["initial_full_contract", "single_atom_primary", "single_atom_reserve", "single_atom_reserve_compact"],
        )
        self.assertEqual(trace[3]["allowed_atom_ids"], ["comp_01"])
        # The initial contract and narrowed repair schema differ; retry 1/2/3
        # intentionally share the same 160-char provider headroom.
        self.assertEqual(len({row["schema_sha256"] for row in trace}), 2)
        self.assertEqual(trace[3]["request_sha256"], hashes[3])
        self.assertNotIn("#composition", raised.exception.errors)


if __name__ == "__main__":
    unittest.main()
