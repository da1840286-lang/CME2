"""Frame retry inside all, word-level OCR, and strict auto-transfer eligibility."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cme.qa import brand_tokens, ocr_text_check
from cme.radar.select import rank_transfer_candidates, select_transfer_candidate
from cme.smoke import main as smoke_main
import tests.test_firstrun as first_run

BRAND = brand_tokens("Fanttik V8 Mate Cordless Car Vacuum", "cordless handheld car vacuum")


class OcrWordRulesTests(unittest.TestCase):
    def check(self, text, allowed=BRAND):
        return ocr_text_check("no_text", "x.png", False, ocr=lambda p: text, allowed_tokens=allowed)

    def test_generic_product_type_words_are_not_allowlisted(self):
        self.assertEqual(set(BRAND), {"fanttik", "mate"})
        self.assertEqual(self.check("Cordless Car Vacuum").status, "FAIL")

    def test_brand_lettering_of_the_target_product_is_not_foreign_text(self):
        self.assertEqual(self.check("FANTTIK Fanttik").status, "PASS")

    def test_misread_brand_still_matches(self):
        self.assertEqual(self.check("Fanttk").status, "PASS")

    def test_added_caption_or_promo_text_still_fails_and_is_named(self):
        r = self.check("FANTTIK 50% OFF today")
        self.assertEqual(r.status, "FAIL")
        self.assertIn("today", r.detail)

    def test_short_fragments_are_noise(self):
        self.assertEqual(self.check("I ll } 8 ab").status, "PASS")

    def test_without_allowed_tokens_brand_text_fails(self):
        self.assertEqual(self.check("FANTTIK", allowed=()).status, "FAIL")

    def test_text_allowed_contract_passes(self):
        self.assertEqual(ocr_text_check("o", "x.png", True, ocr=lambda p: "ANY CAPTION HERE").status, "PASS")

    def test_missing_engine_is_not_run(self):
        def boom(p):
            raise RuntimeError("no engine")
        self.assertEqual(ocr_text_check("o", "x.png", False, ocr=boom).status, "NOT_RUN")

    def test_alternative_layout_finds_text_missed_by_default_layout(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "frame.png"
            Image.new("RGB", (64, 96), (128, 128, 128)).save(path)
            outputs = {
                "--psm 3": {"text": [""], "conf": ["-1"]},
                "--psm 11": {"text": ["Brushless", "motor"], "conf": ["90", "81"]},
                "--psm 12": {"text": ["motor"], "conf": ["80"]},
            }
            def fake_image_to_data(image, *, config, output_type):
                return outputs[config]
            with patch("pytesseract.image_to_data", side_effect=fake_image_to_data) as mocked:
                result = ocr_text_check("no_text", path, False, allowed_tokens=BRAND)
            self.assertEqual(result.status, "FAIL", result.detail)
            self.assertEqual(
                [call.kwargs["config"] for call in mocked.call_args_list],
                ["--psm 3", "--psm 11", "--psm 12"],
            )

    @unittest.skipUnless(shutil.which("tesseract"), "tesseract not installed")
    def test_real_tesseract_ignores_noise_keeps_promo(self):
        import random
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        except OSError:
            self.skipTest("no TrueType font")
        with tempfile.TemporaryDirectory() as d:
            random.seed(4)
            noise = Image.new("RGB", (576, 1024), (120, 110, 100))
            dr = ImageDraw.Draw(noise)
            for _ in range(400):
                x, y = random.randint(0, 576), random.randint(0, 1024)
                dr.rectangle([x, y, x + random.randint(5, 60), y + random.randint(3, 40)],
                             fill=tuple(random.randint(30, 220) for _ in range(3)))
            noise.filter(ImageFilter.GaussianBlur(1.2)).save(Path(d) / "noise.png")
            promo = Image.new("RGB", (576, 1024), (40, 40, 40))
            ImageDraw.Draw(promo).text((40, 400), "FREE SHIPPING", font=font, fill=(240, 240, 0))
            promo.save(Path(d) / "promo.png")
            self.assertEqual(ocr_text_check("t", str(Path(d) / "noise.png"), False, allowed_tokens=BRAND).status, "PASS")
            self.assertEqual(ocr_text_check("t", str(Path(d) / "promo.png"), False, allowed_tokens=BRAND).status, "FAIL")


class FrameRetryTests(unittest.TestCase):
    make_ctx = first_run.SmokeEndToEndTests.make_ctx
    settings = first_run.SmokeEndToEndTests.settings

    def ctx_failing_first_judge(self, tmp, fail_calls):
        ctx = self.make_ctx(tmp=tmp)
        base, state = ctx["transport"], {"judge": 0}

        def transport(method, url, headers, payload):
            content = payload["messages"][1]["content"]
            if isinstance(content, list):
                state["judge"] += 1
                ids = [r["id"] for r in json.loads(content[0]["text"])]
                ok = state["judge"] > fail_calls
                return 200, first_run.completion(
                    json.dumps({"results": [{"id": i, "pass": ok, "evidence": "seen"} for i in ids]}),
                    payload["model"])
            return base(method, url, headers, payload)
        ctx["transport"] = transport
        return ctx

    def run_all(self, tmp, fail_calls, **settings):
        out = Path(tmp) / "out"
        code = smoke_main(
            ["all", "--settings", self.settings(tmp, **settings), "--out", str(out), "--no-ocr-required"],
            self.ctx_failing_first_judge(tmp, fail_calls))
        return code, out

    def test_failed_qa_gets_a_second_sample_and_can_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_all(tmp, fail_calls=1)
            self.assertEqual(code, 0)
            self.assertTrue((out / "shots" / "shot_01_a01.png").is_file())
            self.assertTrue((out / "shots" / "shot_01_a02.png").is_file())
            receipts = [json.loads(p.read_text()) for p in (out / "receipts").glob("drec_*.json")]
            self.assertEqual(sorted(r["overall"] for r in receipts), ["FAILED_QA", "PASS"])
            seeds = {r["provider"].get("seed") for r in receipts}
            self.assertEqual(len(seeds), 2)

    def test_persistent_failure_stops_at_the_attempt_limit_with_code_7(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_all(tmp, fail_calls=99, max_attempts_per_shot=2)
            self.assertEqual(code, 7)
            self.assertEqual(len(list((out / "shots").glob("shot_01_a*.png"))), 2)
            self.assertFalse(list((out / "shots").glob("shot_01_a03.png")))

    def test_limit_of_one_means_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_all(tmp, fail_calls=1, max_attempts_per_shot=1)
            self.assertEqual(code, 7)
            self.assertEqual(len(list((out / "shots").glob("shot_01_a*.png"))), 1)

    def test_configured_retry_limit_above_two_is_still_capped_at_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_all(tmp, fail_calls=99, max_attempts_per_shot=5)
            self.assertEqual(code, 7)
            self.assertEqual(len(list((out / "shots").glob("shot_01_a*.png"))), 2)
            self.assertFalse(list((out / "shots").glob("shot_01_a03.png")))

    def test_a_passing_frame_is_never_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_all(tmp, fail_calls=0)
            self.assertEqual(code, 0)
            self.assertEqual(len(list((out / "shots").glob("shot_01_a*.png"))), 1)


def cand(cid, video=True, cls="DIRECT", money=True, action="DEEP"):
    return {"id": cid, "video_url": "https://v/x.mp4" if video else "", "source_check": {"classification": cls},
            "source_score": {"action": action, "final_score": 50, "components": {"money_signal_gate": {"pass": money}}}}


class AutoTransferEligibilityTests(unittest.TestCase):
    def test_commercially_eligible_source_is_selected_and_weak_sources_are_skipped(self):
        ordered, skipped = rank_transfer_candidates([
            cand("weak", money=False), cand("strong"), cand("novideo", video=False),
            cand("excluded", cls="HARD_EXCLUDE")])
        self.assertEqual([c["id"] for c in ordered], ["strong"])
        self.assertEqual(len(skipped), 3)

    def test_find_order_is_preserved_among_commercially_eligible_sources(self):
        ordered, skipped = rank_transfer_candidates([
            cand("a"), cand("b"), cand("c", money=False), cand("d", money=False)])
        self.assertEqual([c["id"] for c in ordered], ["a", "b"])
        self.assertEqual(len(skipped), 2)

    def test_weak_only_pool_is_rejected(self):
        chosen, skipped = select_transfer_candidate([cand("only", money=False, cls="UNVERIFIED", action="REJECT")])
        self.assertIsNone(chosen)
        self.assertEqual(len(skipped), 1)

    def test_empty_pool_returns_none(self):
        self.assertEqual(select_transfer_candidate([]), (None, []))


if __name__ == "__main__":
    unittest.main()
