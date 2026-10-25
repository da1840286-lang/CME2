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
    def test_automatic_transfer_excludes_candidates_rejected_by_commercial_gate(self):
        ordered, skipped = rank_transfer_candidates([cand("a", 60.1, "EXPANDED"), cand("b", 28.3, gate=False, action="REJECT"),
                                                     cand("c", 51.6, gate=False, action="REJECT")])
        self.assertEqual([c["id"] for c in ordered], ["a"])
        self.assertEqual(len(skipped), 2)
        self.assertTrue(any("REJECT" in reason for reason in skipped))

    def test_high_score_cannot_override_a_commercial_rejection(self):
        ordered, skipped = rank_transfer_candidates([cand("strong", 55.6, "EXPANDED"), cand("weak_high", 61.7, gate=False, action="REJECT")])
        self.assertEqual([c["id"] for c in ordered], ["strong"])
        self.assertEqual(len(skipped), 1)
        self.assertIn("REJECT", skipped[0])

    def test_commercial_pass_breaks_a_close_call(self):
        ordered, _ = rank_transfer_candidates([cand("weak", 52.0, gate=False, action="REJECT"), cand("strong", 49.0)])
        self.assertEqual(ordered[0]["id"], "strong")

    def test_relevance_breaks_equal_scores_and_find_order_the_rest(self):
        ordered, _ = rank_transfer_candidates([cand("exp", 50.0, "EXPANDED"), cand("dir", 50.0, "DIRECT"), cand("dir2", 50.0, "DIRECT")])
        self.assertEqual([c["id"] for c in ordered], ["dir", "dir2", "exp"])

    def test_unusable_sources_are_still_skipped_with_a_reason(self):
        novideo = {**cand("nv", 90.0), "video_url": ""}
        chosen, skipped = select_transfer_candidate([novideo, cand("ok", 10.0)])
        self.assertEqual(chosen["id"], "ok")
        self.assertEqual(len(skipped), 1)

    def test_no_candidate_is_selected_when_every_source_fails_commercial_gate(self):
        chosen, skipped = select_transfer_candidate([cand("rejected", 95.0, gate=False, action="REJECT")])
        self.assertIsNone(chosen)
        self.assertEqual(len(skipped), 1)
        self.assertIn("REJECT", skipped[0])


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
