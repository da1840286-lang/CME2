"""Two-phase judge protocol: structural guarantees and cases that must never yield false PASS."""
import json
import tempfile
import unittest
from pathlib import Path

from cme.adapters.openrouter import make_openrouter_judge
from cme.atoms import Atom
from cme.judge_protocol import decide, parse_observations
from cme.qa import judge_batch_checks

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
       b"\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82")

RESULT = Atom("mech_result_1", "action", "A clear strip is left where the tool passed, beside untouched material", "MUST", "image", 1)
CONTACT = Atom("mech_intervention_1", "action", "The tool touches the edge between the two areas", "MUST", "image", 1)
OBS_FULL = [
    {"id": "o1", "kind": "object", "text": "a grey handheld tool lies on a dark cloth surface, centre-left"},
    {"id": "o2", "kind": "region", "text": "scattered light-coloured grains cover the right half of the cloth"},
    {"id": "o3", "kind": "region", "text": "a clear strip lies where the tool passed beside untouched grains on the cloth"},
    {"id": "o4", "kind": "relation", "text": "the tool tip touches the edge between the clear strip and the grains"},
]
OBS_NO_AFTER = [OBS_FULL[0], OBS_FULL[1],
    {"id": "o3", "kind": "relation", "text": "the tool and the grains are separate, unclear contact"}]


def completion(body, model="m"):
    return json.dumps({"model": model, "choices": [{"message": {"content": json.dumps(body)}}], "usage": {"cost": 0}}).encode()


class DecideTests(unittest.TestCase):
    def verdict(self, conditions, atom=RESULT, observations=OBS_FULL, rows_override=None):
        rows = rows_override if rows_override is not None else [{"id": atom.id, "conditions": conditions}]
        return decide({"results": rows}, [atom], observations)

    def test_confirmed_result_conditions_pass_and_use_inventory_as_evidence(self):
        out = self.verdict([
            {"aspect": "problem_state", "claim": "grains remain", "observation_ids": ["o2"], "quote": "grains cover the right half", "holds": True},
            {"aspect": "changed_state", "claim": "clear strip is visible", "observation_ids": ["o3"], "quote": "clear strip lies where the tool passed beside untouched grains", "holds": True}])
        self.assertTrue(out[RESULT.id]["pass"])
        self.assertIn("scattered light-coloured grains", out[RESULT.id]["evidence"])

    def test_missing_changed_state_cannot_pass_result_atom(self):
        out = self.verdict([
            {"aspect": "problem_state", "claim": "grains remain", "observation_ids": ["o2"], "quote": "grains cover the right half", "holds": True}],
            observations=OBS_NO_AFTER)
        self.assertFalse(out[RESULT.id]["pass"])

    def test_sycophantic_verifier_cannot_invent_an_unseen_after_state(self):
        out = self.verdict([
            {"aspect": "problem_state", "claim": "grains remain", "observation_ids": ["o2"], "quote": "grains cover the right half", "holds": True},
            {"aspect": "changed_state", "claim": "clear strip left", "observation_ids": ["o2"], "quote": "a clear strip is left", "holds": True}],
            observations=OBS_NO_AFTER)
        self.assertFalse(out[RESULT.id]["pass"])

    def test_intervention_requires_contact_action_aspect(self):
        out = self.verdict([
            {"aspect": "problem_state", "claim": "grains visible", "observation_ids": ["o2"], "quote": "grains cover the right half", "holds": True}],
            atom=CONTACT)
        self.assertFalse(out[CONTACT.id]["pass"])

    def test_missing_unknown_or_unquoted_observation_never_passes(self):
        cases = [
            {"aspect": "problem_state", "claim": "x", "observation_ids": [], "quote": "", "holds": True},
            {"aspect": "problem_state", "claim": "x", "observation_ids": ["o99"], "quote": "clear strip is left", "holds": True},
            {"aspect": "changed_state", "claim": "x", "observation_ids": ["o2"], "quote": "a clear strip is left where tool passed", "holds": True},
        ]
        for row in cases:
            out = self.verdict([row], observations=OBS_NO_AFTER)
            self.assertFalse(out[RESULT.id]["pass"])

    def test_one_unsupported_condition_fails_the_whole_atom(self):
        out = self.verdict([
            {"aspect": "problem_state", "claim": "grains", "observation_ids": ["o2"], "quote": "grains cover the right half", "holds": True},
            {"aspect": "changed_state", "claim": "clear area", "observation_ids": [], "quote": "", "holds": False}])
        self.assertFalse(out[RESULT.id]["pass"])

    def test_missing_malformed_or_duplicate_answers_are_not_run(self):
        good = {"aspect": "problem_state", "claim": "grains", "observation_ids": ["o2"], "quote": "grains cover the right half", "holds": True}
        changed = {"aspect": "changed_state", "claim": "clear area", "observation_ids": ["o3"], "quote": "clear strip lies where the tool passed beside untouched grains", "holds": True}
        self.assertEqual(decide({"results": "nope"}, [RESULT], OBS_FULL), {})
        self.assertEqual(decide({"results": [{"id": RESULT.id, "conditions": []}]}, [RESULT], OBS_FULL), {})
        duplicate = [{"id": RESULT.id, "conditions": [good, changed]}, {"id": RESULT.id, "conditions": [good, changed]}]
        self.assertEqual(decide({"results": duplicate}, [RESULT], OBS_FULL), {})

    def test_unexpected_non_dict_condition_is_not_ignored(self):
        out = self.verdict([{"aspect": "problem_state", "claim": "grains", "observation_ids": ["o2"], "quote": "grains cover the right half", "holds": True}, None])
        self.assertEqual(out, {})

    def test_inventory_accepts_empty_and_sparse_lists_as_data(self):
        self.assertEqual(parse_observations({"observations": OBS_FULL[:2]}), OBS_FULL[:2])
        self.assertEqual(parse_observations({"observations": []}), [])
        self.assertIsNone(parse_observations({"observations": "x"}))
        self.assertIsNone(parse_observations({"nothing": 1}))
        dup = OBS_FULL + [dict(OBS_FULL[0], text="duplicate id"), {"id": "", "text": "no id"}]
        self.assertEqual([o["id"] for o in parse_observations({"observations": dup})], ["o1", "o2", "o3", "o4"])

    HOOK = Atom("keep_hook_1", "scene", "Show the BrandX Model-9 as the compact quick solution", "MUST", "image", 1)

    def cond(self, aspect, ids, quote, holds=True, reason=None, claim="c"):
        row = {"aspect": aspect, "claim": claim, "observation_ids": ids, "quote": quote, "holds": holds}
        if reason:
            row["non_visual_reason"] = reason
        return row

    def test_brand_and_benefit_claims_are_excluded_but_presence_is_still_proven(self):
        out = self.verdict([
            self.cond("object_presence", ["o1"], "a grey handheld tool lies", claim="tool visible"),
            self.cond("non_visual", [], "", holds=False, reason="brand_or_model", claim="BrandX Model-9"),
            self.cond("non_visual", [], "", holds=False, reason="purpose_or_benefit", claim="compact quick solution")], atom=self.HOOK)
        self.assertTrue(out[self.HOOK.id]["pass"])
        self.assertIn("not judged: brand_or_model, purpose_or_benefit", out[self.HOOK.id]["evidence"])
        self.assertEqual(out[self.HOOK.id]["excluded"], 2)

    def test_visual_condition_that_fails_still_fails_even_next_to_excluded_claims(self):
        out = self.verdict([
            self.cond("object_presence", ["o1"], "a grey handheld tool lies", holds=False, claim="tool visible"),
            self.cond("non_visual", [], "", holds=False, reason="brand_or_model", claim="BrandX Model-9")], atom=self.HOOK)
        self.assertFalse(out[self.HOOK.id]["pass"])

    def test_requirement_with_only_non_visual_conditions_fails(self):
        out = self.verdict([self.cond("non_visual", [], "", holds=False, reason="power_or_internal", claim="battery powered")], atom=self.HOOK)
        self.assertFalse(out[self.HOOK.id]["pass"])
        self.assertIn("no visually checkable condition", out[self.HOOK.id]["evidence"])

    def test_non_visual_needs_a_valid_reason(self):
        for reason in (None, "because", "state_of_surface"):
            out = self.verdict([
                self.cond("object_presence", ["o1"], "a grey handheld tool lies", claim="tool visible"),
                self.cond("non_visual", [], "", holds=False, reason=reason, claim="model number")], atom=self.HOOK)
            self.assertFalse(out[self.HOOK.id]["pass"], reason)

    def test_non_visual_cannot_hide_obvious_visual_details(self):
        out = self.verdict([
            self.cond("object_presence", ["o1"], "a grey handheld tool lies", claim="tool visible"),
            self.cond("non_visual", [], "", holds=False, reason="time_or_motion", claim="clean strip at right")], atom=self.HOOK)
        self.assertFalse(out[self.HOOK.id]["pass"])

    def test_verifier_cannot_dodge_the_changed_state_by_calling_it_non_visual(self):
        out = self.verdict([
            self.cond("problem_state", ["o2"], "grains cover the right half", claim="starting problem state"),
            self.cond("non_visual", [], "", holds=False, reason="time_or_motion", claim="the result changed")])
        self.assertFalse(out[RESULT.id]["pass"])
        self.assertIn("missing proof aspect: changed_state", out[RESULT.id]["evidence"])


class AdapterProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "f.png"
        self.image.write_bytes(PNG)

    def judge(self, observe_body, verify_body, **params):
        calls = []
        def send(method, url, headers, payload):
            calls.append(payload)
            system = payload["messages"][0]["content"]
            body = observe_body if "careful visual inspector" in system else verify_body
            return 200, completion(body, payload["model"])
        return make_openrouter_judge({"model_id": "vendor/v", **params}, "k", send), calls

    def test_observe_never_sees_requirements_and_verify_never_sees_pixels(self):
        verify = {"results": [{"id": RESULT.id, "conditions": [
            {"aspect": "problem_state", "claim": "grains", "observation_ids": ["o2"], "quote": "grains cover the right half", "holds": True},
            {"aspect": "changed_state", "claim": "clear strip", "observation_ids": ["o3"], "quote": "clear strip lies where the tool passed beside untouched grains", "holds": True}]}]}
        judge, calls = self.judge({"observations": OBS_FULL}, verify)
        out, meta = judge(self.image, [RESULT])
        self.assertNotIn("clear strip is left", json.dumps(calls[0]))
        self.assertNotIn("mech_result_1", json.dumps(calls[0]))
        self.assertNotIn("image_url", json.dumps(calls[1]))
        self.assertEqual(len(calls), 2)
        self.assertTrue(out[RESULT.id]["pass"])
        self.assertEqual(meta["judge_mode"], "observe_verify")

    def test_malformed_observation_inventory_yields_no_answers(self):
        judge, calls = self.judge({"observations": "not a list"}, {"results": []})
        out, meta = judge(self.image, [RESULT, CONTACT])
        self.assertEqual(out, {})
        self.assertEqual(len(calls), 1)
        self.assertEqual(meta["diagnostic"]["reason"], "observation_phase_invalid")
        rows = judge_batch_checks(str(self.image), [RESULT, CONTACT], lambda p, a: out)
        self.assertEqual({r.status for r in rows}, {"NOT_RUN"})
        self.assertTrue(all(r.required for r in rows))

    def test_blank_frame_fails_every_atom_instead_of_not_run(self):
        judge, calls = self.judge({"observations": []}, {"results": []})
        out, _ = judge(self.image, [RESULT, CONTACT])
        self.assertEqual(len(calls), 1)  # empty inventory needs no second call
        self.assertEqual(set(out), {RESULT.id, CONTACT.id})
        self.assertFalse(any(row["pass"] for row in out.values()))
        rows = judge_batch_checks(str(self.image), [RESULT, CONTACT], lambda p, a: out)
        self.assertEqual({r.status for r in rows}, {"FAIL"})

    def test_sparse_inventory_is_verified_and_unconfirmed_conditions_fail(self):
        sparse = [{"id": "o1", "kind": "object", "text": "uniform grey surface, no objects"}]
        no = lambda aspect: {"aspect": aspect, "claim": "x", "observation_ids": [], "quote": "", "holds": False}
        verify = {"results": [
            {"id": RESULT.id, "conditions": [no("problem_state"), no("changed_state")]},
            {"id": CONTACT.id, "conditions": [no("problem_state"), no("contact_action")]}]}
        judge, calls = self.judge({"observations": sparse}, verify)
        out, _ = judge(self.image, [RESULT, CONTACT])
        self.assertEqual(len(calls), 2)
        self.assertFalse(any(row["pass"] for row in out.values()))
        self.assertEqual(set(out), {RESULT.id, CONTACT.id})

    def test_analyst_prompt_uses_neutral_state_change_language(self):
        from cme.adapters.openrouter import ANALYST_SYSTEM
        self.assertIn("appeared, disappeared, moved, filled, changed state", ANALYST_SYSTEM)
        self.assertNotIn("visible mess or material disappeared", ANALYST_SYSTEM)

    def test_trace_file_records_inventory_and_decisions_only_when_enabled(self):
        import os
        from unittest import mock
        verify = {"results": [{"id": RESULT.id, "conditions": [
            {"aspect": "problem_state", "claim": "starting state", "observation_ids": ["o2"],
             "quote": "grains cover the right half", "holds": True},
            {"aspect": "changed_state", "claim": "result state", "observation_ids": ["o3"],
             "quote": "clear strip lies where the tool passed beside untouched grains", "holds": True}]}]}
        judge, _ = self.judge({"observations": OBS_FULL}, verify)
        judge(self.image, [RESULT])
        self.assertFalse((Path(self.temp.name) / "trace.jsonl").exists())
        trace = Path(self.temp.name) / "trace.jsonl"
        with mock.patch.dict(os.environ, {"CME_JUDGE_TRACE": str(trace)}):
            judge(self.image, [RESULT])
        record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["atoms"], [RESULT.id])
        self.assertEqual(len(record["observations"]), 4)
        self.assertTrue(record["decided"][RESULT.id]["pass"])

    def test_partial_verify_leaves_missing_atom_not_run(self):
        verify = {"results": [{"id": RESULT.id, "conditions": [
            {"aspect": "problem_state", "claim": "grains", "observation_ids": ["o2"], "quote": "grains cover the right half", "holds": True},
            {"aspect": "changed_state", "claim": "clear strip", "observation_ids": ["o3"], "quote": "clear strip lies where the tool passed beside untouched grains", "holds": True}]}]}
        judge, _ = self.judge({"observations": OBS_FULL}, verify)
        out, meta = judge(self.image, [RESULT, CONTACT])
        self.assertIn(RESULT.id, out)
        self.assertNotIn(CONTACT.id, out)
        self.assertEqual(meta["diagnostic"]["reason"], "verify_phase_partial")

    def test_temperature_is_zero_for_both_phases(self):
        judge, calls = self.judge({"observations": OBS_FULL}, {"results": []}, temperature=0.9)
        judge(self.image, [RESULT])
        self.assertEqual([c["temperature"] for c in calls], [0, 0])

    def test_single_pass_keeps_legacy_behaviour(self):
        judge, calls = self.judge({}, {"results": [{"id": RESULT.id, "pass": True, "evidence": "seen"}]}, judge_mode="single_pass")
        out, _ = judge(self.image, [RESULT])
        self.assertEqual(len(calls), 1)
        self.assertEqual(out, {RESULT.id: {"pass": True, "evidence": "seen"}})


if __name__ == "__main__":
    unittest.main()


