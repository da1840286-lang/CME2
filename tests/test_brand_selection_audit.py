"""Selection order, system-owned branding/text checks, collage wording and the frame-form line."""
import unittest
from dataclasses import replace
from pathlib import Path

from cme.brain import atom_budget_errors, build_request, reserved_system_atom_errors
from cme.branding import NO_BRANDING, TARGET_BRAND_ALLOWED
from cme.compiler import FRAME_FORM_INSTRUCTION, compile_image_prompt
from cme.offer import load_offer
from cme.radar.select import rank_transfer_candidates, select_transfer_candidate
from tests.helpers import A

EX = Path(__file__).resolve().parent.parent / "examples" / "offer_example.json"


def cand(cid, score, cls="DIRECT", gate=True, action="DEEP"):
    return {"id": cid, "video_url": "https://v.example/x.mp4", "source_check": {"classification": cls},
            "source_score": {"final_score": score, "action": action,
                             "components": {"money_signal_gate": {"pass": gate}}}}


def atom(i, text, kind="qa_check", ch="qa", prio="MUST", shot=1):
    return {"id": i, "kind": kind, "text": text, "priority": prio, "channel": ch, "shot": shot}


class SelectionOrderTests(unittest.TestCase):
    def test_money_gate_and_scorer_reject_do_not_block_pattern_extraction(self):
        candidates = [
            cand("a", 60.1, "EXPANDED"),
            cand("b", 28.3, gate=False, action="REJECT"),
            cand("c", 51.6, gate=False, action="REJECT"),
        ]
        ordered, skipped = rank_transfer_candidates(candidates)
        self.assertEqual([c["id"] for c in ordered], ["a", "b", "c"])
        self.assertEqual(skipped, [])

    def test_scores_and_relevance_labels_do_not_reorder_find_output(self):
        candidates = [cand("exp", 80.0, "EXPANDED"), cand("dir", 10.0, "DIRECT"),
                      cand("dir2", 99.0, "DIRECT", gate=False, action="REJECT")]
        ordered, skipped = rank_transfer_candidates(candidates)
        self.assertEqual([c["id"] for c in ordered], ["exp", "dir", "dir2"])
        self.assertEqual(skipped, [])

    def test_unusable_sources_are_still_skipped_with_a_reason(self):
        novideo = {**cand("nv", 90.0), "video_url": ""}
        chosen, skipped = select_transfer_candidate([novideo, cand("ok", 10.0)])
        self.assertEqual(chosen["id"], "ok")
        self.assertEqual(len(skipped), 1)
        self.assertIn("video", skipped[0])

    def test_money_gate_failure_alone_does_not_empty_the_pattern_pool(self):
        chosen, skipped = select_transfer_candidate(
            [cand("rejected", 95.0, gate=False, action="REJECT")])
        self.assertEqual(chosen["id"], "rejected")
        self.assertEqual(skipped, [])


class SystemOwnedChecksTests(unittest.TestCase):
    def test_brain_cannot_author_its_own_branding_check(self):
        errors = reserved_system_atom_errors([
            atom("qa_check_1", "no brand names, logos, or text appear on product or scene"),
            atom("qa_check_2", "the vacuum nozzle is in contact with the surface"),
            atom("scene_1", "a clean table with a small logo-shaped sticker", kind="scene", ch="image")])
        self.assertEqual(set(errors), {"qa_check_1"})
        self.assertIn("SYSTEM_OWNED_QA", errors["qa_check_1"][0])


class CollageWordingTests(unittest.TestCase):
    def test_split_frame_wording_is_sent_back(self):
        errors = atom_budget_errors([atom("comp_1", "split frame: left dirty seat crevice, right cleaned strip", kind="scene", ch="image")], 1)
        self.assertIn("comp_1", errors)
        self.assertIn("FRAME_COLLAGE_WORDING", errors["comp_1"][0])

    def test_one_continuous_surface_wording_passes(self):
        text = "one continuous car seat: left part littered with crumbs, right part clean, nozzle at the boundary"
        self.assertEqual(atom_budget_errors([atom("comp_1", text, kind="scene", ch="image")], 1), {})


class FrameFormTests(unittest.TestCase):
    def test_every_branded_image_prompt_carries_the_frame_form_line(self):
        for policy in (NO_BRANDING, TARGET_BRAND_ALLOWED):
            text = compile_image_prompt([A("s", "Black handheld vacuum on a seat", kind="scene")], 1, branding_policy=policy).text
            self.assertIn(FRAME_FORM_INSTRUCTION, text)
        self.assertNotIn("FRAME FORM", compile_image_prompt([A("s", "Black handheld vacuum on a seat", kind="scene")], 1).text)


class BrainMarksTests(unittest.TestCase):
    def test_brain_gets_the_same_marks_as_qa_when_none_are_configured(self):
        offer = replace(load_offer(EX), branding_policy=TARGET_BRAND_ALLOWED, approved_brand_marks=(),
                        name="Fanttik V8 Mate Cordless Car Vacuum", product_type="cordless handheld car vacuum")
        rules = " ".join(build_request(offer, "brief", 1)["rules"])
        self.assertIn("Fanttik V8 Mate", rules)


if __name__ == "__main__":
    unittest.main()
