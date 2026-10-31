"""Focused offline regression for old-CME transfer context -> current stable prompt atoms."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from cme.brain import build_request, build_source_transfer_context, request_atoms
from cme.adapters.openrouter import make_openrouter_brain
from cme.offer import load_offer
from tests.helpers import gate_account
from cme.smoke import main as smoke_main

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "offer_example.json"
GOOD = {"id": "keep_mechanism_s1_scene", "kind": "scene", "text": "Show the problem, target intervention, and resulting improvement as one visible sequence", "priority": "MUST", "channel": "image", "shot": 1}
DECISION = {"transferable": True, "abstract_mechanism": "Show a clear problem and visible improvement",
            "preserved": [{"property": "the visible problem changes into a visibly improved result", "atom_id": "keep_mechanism_s1_scene"}],
            "reason": "The target product can honestly demonstrate the same cause and visible result."}


class SourceTransferBridgeTests(unittest.TestCase):
    def old_contract(self):
        return {
            "schema_version": "2", "mode": "CHATGPT_EXTERNAL_REVIEW", "ad_id": "source-1",
            "source_review": {"source_url": "https://source.invalid/ad", "source_verification": {
                "status": "PASS", "evidence": ["Observed a familiar task and a visible result"]}},
            "mechanism": {"name": "Demonstrate-Then-Prove", "canonical_slug": "demonstrate-then-prove",
                          "human_desire": "resolve a familiar task", "contraindications": ["do not overclaim"]},
            "mechanism_representation": {"abstraction_statement": "Show a recognizable problem, a bounded action, and an inspectable result.",
                                         "causal_chain": ["problem", "action", "result"],
                                         "source_surface_to_discard": ["source brand", "source UI"],
                                         "abstraction_status": "PASS"},
            "transfer": {"rationale": "Keep the causal idea, replace the product surface",
                         "preserve_rules": ["visible problem", "visible result"],
                         "change_rules": ["use target-native choreography"],
                         "assessment": {"verdict": "CONDITIONAL", "source_specific_components": ["source brand"],
                                        "replacement_requirements": ["remove source brand"],
                                        "causal_mapping": {"trigger": "problem", "intervention": "action", "proof": "result"}},
                         "angles": [{"angle": "unused alternate", "body_copy": "not needed"}]},
            "rebuild_plan": {"target_offer": "stale product name", "strategy": "show problem, action, result",
                             "creative_direction": {"angle": "show one problem, one intervention, and inspectable proof"},
                             "mechanism_contract": {"must_preserve": ["problem", "result"], "must_change": ["props"],
                                                    "do_not_claim": ["guarantees"]},
                             "visual_strategy": {"first_frame_strategy": "clear problem"},
                             "shots": [{"id": 1, "purpose": "HOOK", "scene_description": "An original scene for target product",
                                        "action": "Perform one visible action", "proof": "Show the resulting state",
                                        "state_before": "before", "state_after": "after", "mechanism_slots": ["trigger"]}]},
            "image_creative_contract": {"visual_concept": "Target-native product demo", "hook_frame_purpose": "show problem",
                                        "shots": [{"shot_id": 1, "image_prompt": "Original target-product shot",
                                                   "action": "one action", "proof": "visible result"}]},
        }

    def test_old_contract_is_reduced_without_source_url_or_unused_angle_scripts(self):
        ctx = build_source_transfer_context(self.old_contract())
        self.assertEqual(ctx["format"], "legacy_cme_transfer_contract")
        self.assertEqual(ctx["mechanism"]["canonical_slug"], "demonstrate-then-prove")
        self.assertIn("abstraction_statement", ctx["mechanism_representation"])
        self.assertEqual(ctx["mechanism_representation"]["abstraction_status"], "PASS")
        self.assertEqual(ctx["transfer"]["assessment"]["verdict"], "CONDITIONAL")
        self.assertEqual(ctx["target_shots"][0]["state_before"], "before")
        self.assertIn("creative_direction", ctx["rebuild_plan"])
        self.assertEqual(ctx["image_creative_shots"][0]["action"], "one action")
        self.assertNotIn("image_prompt", json.dumps(ctx))
        self.assertNotIn("generation_prompt", json.dumps(ctx))
        self.assertNotIn("motion_prompt", json.dumps(ctx))
        self.assertNotIn("source_url", json.dumps(ctx))
        self.assertNotIn("unused alternate", json.dumps(ctx))
        self.assertIn("target_offer", ctx["rebuild_plan"])  # kept for context; current OfferCard overrides it

    def test_required_source_verification_must_pass_before_transfer(self):
        contract = self.old_contract()
        contract["source_review"]["source_verification"]["status"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "source_verification.status must be PASS"):
            build_source_transfer_context(contract)

    def test_radar_analysis_is_labeled_as_a_hypothesis(self):
        ctx = build_source_transfer_context({"result": {
            "hook": {"text": "Visible mismatch", "kind": "OBSERVED"},
            "mechanism_hypothesis": {"text": "Show the problem then its fix", "kind": "INFERRED"},
            "transferability": {"text": "Could suit another product", "kind": "INFERRED"},
            "limits": {"text": "Sampled frames only", "missing": ["audio"]},
        }})
        self.assertEqual(ctx["format"], "radar_analysis_hypothesis")
        self.assertEqual(ctx["mechanism_hypothesis"], "Show the problem then its fix")
        self.assertIn("not full-video verification", ctx["source_evidence_scope"])

    def test_invalid_source_json_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported source contract"):
            build_source_transfer_context({"hello": "world"})

    def test_request_keeps_transfer_context_and_requires_decision_schema(self):
        from cme.atoms import ATOM_SCHEMA, TRANSFER_ATOM_SCHEMA
        ctx = build_source_transfer_context(self.old_contract())
        offer = load_offer(EX)
        self.assertIs(build_request(offer, "Make an ad", 1)["schema"], ATOM_SCHEMA)
        req = build_request(offer, "Make an ad", 1, source_transfer=ctx)
        self.assertIs(req["schema"], TRANSFER_ATOM_SCHEMA)
        self.assertTrue(any("TRANSFER DECISION" in rule for rule in req["rules"]))
        self.assertEqual(req["source_transfer"], ctx)
        self.assertTrue(any("not the source ad's exact scene" in rule for rule in req["rules"]))
        calls = []
        def call(payload):
            calls.append(payload)
            return ({"atoms": [GOOD, dict(GOOD)], "transfer_decision": DECISION} if len(calls) == 1
                    else {"atoms": [GOOD], "transfer_decision": DECISION})
        atoms = request_atoms(offer, "Make an ad", 1, call, max_retries=1, source_transfer=ctx)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["source_transfer"], ctx)
        self.assertEqual(calls[1]["source_transfer"], ctx)

    def test_smoke_brain_loads_source_contract_and_binds_its_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = json.loads((ROOT / "settings.example.json").read_text(encoding="utf-8"))
            settings_path = root / "settings.json"
            settings_path.write_text(json.dumps(settings), encoding="utf-8")
            source_path = root / "old_contract.json"
            source_path.write_text(json.dumps(self.old_contract()), encoding="utf-8")
            sent = []
            fit_decision = {"transferable": True,
                            "abstract_mechanism": DECISION["abstract_mechanism"],
                            "target_native_mapping": "Show the target product solving a recognizable task with visible proof",
                            "target_evidence": ["kitchen storage"],
                            "reason": "The target offer can demonstrate a problem and inspectable improvement",
                            **gate_account(preserved=("mechanism", "problem", "intervention", "result"), dropped=("hook",))}
            def transport(method, url, headers, payload):
                sent.append(payload)
                user_payload = json.loads(payload["messages"][1]["content"])
                answer = ({"atoms": [], "transfer_decision": fit_decision}
                          if user_payload.get("task") == "target_fit_gate"
                          else {"atoms": [GOOD], "transfer_decision": DECISION})
                response = {"model": payload["model"], "choices": [{"message": {"content": json.dumps(answer)}}]}
                return 200, json.dumps(response).encode()
            out = root / "out"
            code = smoke_main(["brain", "--settings", str(settings_path), "--out", str(out),
                               "--source-contract", str(source_path), "--shots", "1"],
                              {"transport": transport, "openrouter_key": "test-key"})
            self.assertEqual(code, 0)
            user = json.loads(sent[0]["messages"][1]["content"])
            self.assertEqual(user["source_transfer"]["mechanism"]["canonical_slug"], "demonstrate-then-prove")
            manifest = json.loads((out / "atoms_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_contract_sha256"], hashlib.sha256(source_path.read_bytes()).hexdigest())
            self.assertTrue(manifest["target_fit_gate"]["transferable"])
            self.assertTrue((out / "target_fit_gate.json").is_file())

    def test_openrouter_sends_optional_source_transfer_as_a_distinct_payload_field(self):
        ctx = build_source_transfer_context(self.old_contract())
        seen = []
        def send(method, url, headers, payload):
            seen.append(payload)
            body = {"model": payload["model"], "choices": [{"message": {"content": json.dumps({"atoms": [GOOD], "transfer_decision": DECISION})}}]}
            return 200, json.dumps(body).encode()
        brain = make_openrouter_brain({"model_id": "vendor/test", "response_format_mode": "json_object"}, "key", send)
        brain(build_request(load_offer(EX), "brief", 1, source_transfer=ctx))
        user = json.loads(seen[0]["messages"][1]["content"])
        self.assertEqual(user["source_transfer"], ctx)
        self.assertNotIn("source_url", json.dumps(user["source_transfer"]))


if __name__ == "__main__":
    unittest.main()
