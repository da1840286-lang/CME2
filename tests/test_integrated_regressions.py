"""Integrated regressions for cross-stage contracts found in live artifacts."""
import json
import tempfile
import unittest
from pathlib import Path

from cme.adapters.openrouter import make_openrouter_judge
from cme.atoms import Atom, parse_atoms
from cme.brain import _source_evidence_contradiction
from cme.qa import judge_batch_checks
from cme.radar.analyze import validate_analysis
from cme.radar.find import _DiscoveryEvidenceTracker, _apply_product_selection_gates
from cme.radar.fit import TARGET_PRODUCT
from tests.helpers import png_bytes

ROOT = Path(__file__).resolve().parents[1]
TARGET = {
    "target_offer": "Fanttik V8 Mate Cordless Car Vacuum",
    "product_type": "cordless handheld car vacuum",
    "product_category": "vacuum",
}
OFF_TOPIC = {
    "ad_id": "wrong-blower-1", "video_id": "wrong-blower-video",
    "title": "Use the blower mode to push dust out of tight AC vents first, then vacuum it away cleanly #carvacuumcleaner",
    "source_url": "https://www.tiktok.com/@brand/video/wrong-blower-1",
    "video_url": "https://cdn.example.test/wrong-blower.mp4",
    "raw": {"post": {"content": {"text": "Use the blower mode to push dust out of tight AC vents first, then vacuum it away cleanly #carvacuumcleaner"}}},
}
TARGET_AD = {
    "ad_id": "target-vacuum-1", "video_id": "target-vacuum-video",
    "title": "Mud and dirt tracked in on rainy days? Grab this compact vacuum for quick cleanups #carvacuumcleaner",
    "source_url": "https://www.tiktok.com/@brand/video/target-vacuum-1",
    "video_url": "https://cdn.example.test/target-vacuum.mp4",
    "raw": {"post": {"content": {"text": "Mud and dirt tracked in on rainy days? Grab this compact vacuum for quick cleanups #carvacuumcleaner"}}},
}


class FakeSearchClient:
    enabled = True

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def fetch_sponsored_search(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": self.rows, "credits_used": 1, "params": kwargs}


class IntegratedFindGateTests(unittest.TestCase):
    def candidate(self, record):
        return {
            "id": "candidate-" + record["ad_id"], "title": record["title"],
            "legacy_record": dict(record),
            "metrics": {"views": {"value": 700000}, "likes": {"value": 20000}},
            "performance_evidence": {"basis": "ENGAGEMENT_PROXY", "performance_boards": []},
            "source_selection": {}, "stage": "PATTERN_CANDIDATE", "source_gate_eligible": True,
        }

    def test_product_lane_rejects_incidental_target_mention_before_ranking(self):
        client = FakeSearchClient([OFF_TOPIC, TARGET_AD])
        tracker = _DiscoveryEvidenceTracker(client, offer=TARGET)
        result = tracker.fetch_sponsored_search(query="car vacuum")
        # Preserve raw rows for the legacy ranker's detailed rejection diagnostics.
        self.assertEqual([row["ad_id"] for row in result["items"]], ["wrong-blower-1", "target-vacuum-1"])
        self.assertEqual(tracker.product_role_gate["raw_rows"], 2)
        self.assertEqual(tracker.product_role_gate["accepted_rows"], 1)
        self.assertEqual(tracker.product_role_gate["rejected_by_reason"].get("OFF_TOPIC"), 1)
        self.assertEqual(result["items"][1]["source_role_info"]["role"], TARGET_PRODUCT)
        selected, diagnostics = _apply_product_selection_gates(
            [self.candidate(OFF_TOPIC), self.candidate(TARGET_AD)],
            {**TARGET, "requested_performance_order_bys": []},
        )
        self.assertEqual([row["legacy_record"]["ad_id"] for row in selected], ["target-vacuum-1"])
        self.assertEqual(diagnostics["product_selection"]["rejected_by_reason"].get("OFF_TOPIC"), 1)

    def test_requested_playback_board_does_not_fall_back_to_engagement_only(self):
        selected, diagnostics = _apply_product_selection_gates(
            [self.candidate(TARGET_AD)],
            {**TARGET, "requested_performance_order_bys": ["play_6s_rate"]},
        )
        self.assertEqual(selected, [])
        self.assertEqual(diagnostics["requested_performance"]["rejected_candidates"], 1)
        self.assertEqual(
            diagnostics["requested_performance"]["rejected_examples"][0]["reason"],
            "REQUESTED_PERFORMANCE_EVIDENCE_MISSING",
        )

    def test_real_position_on_requested_board_is_sufficient_when_rate_is_not_returned(self):
        record = {**TARGET_AD, "discovery_board_positions": {"play_6s_rate|keyword:car vacuum": 2}}
        selected, diagnostics = _apply_product_selection_gates(
            [self.candidate(record)],
            {**TARGET, "requested_performance_order_bys": ["play_6s_rate"]},
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(diagnostics["requested_performance"]["accepted_candidates"], 1)


class CrossStageContractTests(unittest.TestCase):
    def test_brain_rejects_atom_truncated_at_a_separator(self):
        base = {
            "id": "mech_result_1", "kind": "action", "text": "The changed result is visible,",
            "priority": "MUST", "channel": "image", "shot": 1,
        }
        bad = parse_atoms([base])
        self.assertTrue(any("ATOM_TEXT_INCOMPLETE" in error for error in bad.errors["mech_result_1"]))
        good = parse_atoms([{**base, "text": "The changed result is visible after the tool passes."}])
        self.assertEqual(good.errors, {})

    def analysis_fixture(self):
        return {
            "hook": {"text": "The product enters the scene", "kind": "OBSERVED"},
            "buyer_problem": {"text": "Visible debris is present", "kind": "OBSERVED"},
            "product_intervention": {"text": "The tool contacts debris", "kind": "OBSERVED"},
            "result_demo": {"text": "The area is visibly clean", "kind": "OBSERVED"},
            "cta": {"text": "A call to action is visible", "kind": "OBSERVED"},
            "mechanism_hypothesis": {"text": "The visual contrast may capture attention", "kind": "INFERRED"},
            "transferability": {"text": "The idea may transfer", "kind": "INFERRED"},
            "evidence": [{"t_sec": 3.0, "frame": 1, "observation": "product enters scene", "supports": ["hook"]}],
            "limits": {"text": "Sampled frames only", "missing": []}, "ratings": {},
        }

    def test_analyze_rejects_observed_claim_without_its_own_evidence_link(self):
        with self.assertRaisesRegex(ValueError, "buyer_problem.*no valid frame-linked evidence"):
            validate_analysis(self.analysis_fixture(), [1.0, 3.0])

    def test_analyze_rejects_evidence_frame_timestamp_mismatch(self):
        data = self.analysis_fixture()
        data["evidence"][0]["frame"] = 0
        with self.assertRaisesRegex(ValueError, "hook.*no valid frame-linked evidence"):
            validate_analysis(data, [1.0, 3.0])

    def test_must_visual_judge_checks_are_required_not_optional(self):
        atom = Atom("mech_result_1", "action", "The changed result is clearly visible.", "MUST", "image", 1)
        rows = judge_batch_checks(
            Path("unused-frame.png"), [atom],
            lambda _path, atoms: {atoms[0].id: {"pass": True, "evidence": "visible result state"}},
        )
        self.assertEqual(rows[0].status, "PASS")
        self.assertTrue(rows[0].required)

    def test_brain_and_visual_judge_use_different_free_models(self):
        catalog = json.loads((ROOT / "adapters.json").read_text(encoding="utf-8"))
        rows = catalog["adapters"] if isinstance(catalog, dict) and "adapters" in catalog else catalog
        models = {item["name"]: item["model_id"] for item in rows}
        self.assertNotEqual(models["openrouter_free_brain"], models["openrouter_free_judge"])
        self.assertEqual(next(item for item in rows if item["name"] == "openrouter_free_judge")["tier"], "free")

    def test_negative_transfer_cannot_ignore_non_vacuum_observation(self):
        source = {"evidence": [{
            "t_sec": 4.0,
            "observation": "The cleaner wipes a coffee stain from fabric.",
            "supports": ["product_intervention"],
        }]}
        decision = {
            "transferable": False,
            "reason": "No source evidence shows wiping the coffee stain.",
            "source_elements_dropped": [],
        }
        self.assertIn("SOURCE_EVIDENCE_CONTRADICTION", _source_evidence_contradiction(decision, source))

    def test_visual_judge_prompt_requires_visible_change_and_contact_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "frame.png"
            image_path.write_bytes(png_bytes())
            captured = {}

            def send(_method, _url, _headers, payload):
                captured["payload"] = payload
                user_items = payload["messages"][1]["content"]
                req_text = next(item["text"] for item in user_items if item.get("type") == "text" and item["text"].startswith("["))
                requested = json.loads(req_text)
                response = {
                    "results": [
                        {"id": row["id"], "pass": True, "evidence": "visible product and outcome"}
                        for row in requested
                    ]
                }
                return 200, json.dumps({
                    "model": payload["model"],
                    "choices": [{"message": {"content": json.dumps(response)}}],
                    "usage": {"cost": 0},
                }).encode()

            judge = make_openrouter_judge(
                {"model_id": "test/independent-vision", "response_format_mode": "json_object"},
                api_key="test-key", transport=send,
            )
            atoms = [
                Atom("mech_intervention_1", "action", "The tool visibly contacts the problem area.", "MUST", "image", 1),
                Atom("mech_result_1", "action", "The changed result is clearly distinguishable.", "MUST", "image", 1),
            ]
            judge(image_path, atoms)
            system = captured["payload"]["messages"][0]["content"]
            request_rows = json.loads(next(
                item["text"] for item in captured["payload"]["messages"][1]["content"]
                if item.get("type") == "text" and item["text"].startswith("[")
            ))
            self.assertIn("after-state", system)
            self.assertIn("physical contact or an action", system)
            self.assertIn("RESULT-STATE GATE", request_rows[1]["requirement"])
            self.assertIn("ACTIVE-INTERVENTION GATE", request_rows[0]["requirement"])


if __name__ == "__main__":
    unittest.main()
