"""Regression tests for campaign-scoped branding across Brain, image prompt and QA."""
import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw
from cme.brain import build_request
from cme.branding import (NO_BRANDING, TARGET_BRAND_ALLOWED, allowed_brand_tokens,
                          prepare_unbranded_reference, qa_branding_rule, resolve_branding_contract,
                          _fill_clusters, _merge_polygons)
from cme.compiler import compile_image_prompt
from cme.offer import load_offer
from cme.qa import CheckResult, SYSTEM_NO_TEXT_ID, make_qa_checks
from cme.run import ReferenceCleaningError, produce_shot
from cme.state import new_run
from cme.vertical import GateError
from tests.helpers import A, png_bytes

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "offer_example.json"

class CampaignBrandingPolicyTests(unittest.TestCase):
    def test_strict_mode_never_whitelists_target_name_but_brand_mode_does(self):
        self.assertEqual(allowed_brand_tokens(NO_BRANDING, ("Fanttik",), "Fanttik V8 Mate Vacuum", "cordless vacuum"), ())
        self.assertEqual(set(allowed_brand_tokens(TARGET_BRAND_ALLOWED, (), "Fanttik V8 Mate Cordless Car Vacuum",
                                                  "cordless handheld car vacuum")), {"fanttik", "mate"})
        self.assertEqual(set(allowed_brand_tokens(TARGET_BRAND_ALLOWED, ("ExampleBrand",), "Fanttik Vacuum", "vacuum")),
                         {"examplebrand"})

    def test_unified_branding_contract_has_only_effective_marks(self):
        strict = resolve_branding_contract(NO_BRANDING, ("Grove Co.",),
                                           "Grove Co. Reusable Silicone Food Storage Bag 12 oz",
                                           "reusable silicone food storage bag")
        self.assertEqual(strict, {"branding_policy": NO_BRANDING, "approved_brand_marks": []})
        branded = resolve_branding_contract(TARGET_BRAND_ALLOWED, ("Grove Co.",),
                                            "Grove Co. Reusable Silicone Food Storage Bag 12 oz",
                                            "reusable silicone food storage bag")
        self.assertEqual(branded, {"branding_policy": TARGET_BRAND_ALLOWED, "approved_brand_marks": ["Grove Co."]})

    def test_qa_rules_separate_strict_and_target_brand_modes(self):
        strict = qa_branding_rule(NO_BRANDING)
        branded = qa_branding_rule(TARGET_BRAND_ALLOWED, ("ExampleBrand",))
        self.assertIn("Product-native markings are NOT exempt", strict)
        self.assertIn("ExampleBrand", branded)
        self.assertIn("source-ad", branded)

    def test_brain_and_generation_prompt_follow_campaign_mode(self):
        offer = load_offer(EX)
        strict = replace(offer, branding_policy=NO_BRANDING)
        brand = replace(offer, branding_policy=TARGET_BRAND_ALLOWED, approved_brand_marks=("ExampleBrand",))
        strict_rules = " ".join(build_request(strict, "brief", 1)["rules"])
        brand_rules = " ".join(build_request(brand, "brief", 1)["rules"])
        strict_prompt = compile_image_prompt([A("product_main", "Black handheld vacuum body is visible", kind="product")],
                                             1, branding_policy=NO_BRANDING).text
        brand_prompt = compile_image_prompt([A("product_main", "Black handheld vacuum body is visible", kind="product")],
                                            1, branding_policy=TARGET_BRAND_ALLOWED,
                                            approved_brand_marks=("ExampleBrand",)).text
        self.assertIn("even when present on supplied target-product photos", strict_rules)
        self.assertIn("ExampleBrand", brand_rules)
        self.assertIn("SYSTEM-OWNED BRANDING POLICY", strict_prompt)
        self.assertIn("clean, unmarked material", strict_prompt)
        self.assertIn("ExampleBrand", brand_prompt)

    def test_detected_brand_text_fails_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "frame.png"
            path.write_bytes(png_bytes())
            seen = {}
            def judge(_path, atoms):
                seen["rule"] = {a.id: a.text for a in atoms}[SYSTEM_NO_TEXT_ID]
                return {a.id: {"pass": True, "evidence": "No readable text or branding"} for a in atoms}
            qa = make_qa_checks([A("product_main", "Black handheld vacuum is visible", kind="product")], 1,
                               judge_fn=judge, ocr=lambda _p: "LABEL", branding_policy=NO_BRANDING, allowed_text=())
            rows = {x.check_id:x for x in qa(path)}
            self.assertEqual(rows["no_text"].status, "FAIL")
            self.assertTrue(rows["no_text"].required)
            self.assertIn("Product-native markings are NOT exempt", seen["rule"])

    def test_empty_ocr_does_not_bypass_required_visual_judge(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "frame.png"
            path.write_bytes(png_bytes())
            def judge(_path, atoms):
                return {a.id: {"pass": a.id != SYSTEM_NO_TEXT_ID,
                               "evidence": "branding violation" if a.id == SYSTEM_NO_TEXT_ID else "visible"}
                        for a in atoms}
            qa = make_qa_checks([A("product_main", "Black handheld vacuum is visible", kind="product")], 1,
                               judge_fn=judge, ocr=lambda _p: "", branding_policy=NO_BRANDING)
            rows = {x.check_id:x for x in qa(path)}
            self.assertEqual(rows["no_text"].status, "NOT_RUN")
            self.assertFalse(rows["no_text"].required)
            self.assertTrue(rows[SYSTEM_NO_TEXT_ID].required)
            self.assertEqual(rows[SYSTEM_NO_TEXT_ID].status, "FAIL")

    @unittest.skipUnless(os.environ.get("CME_RUN_REAL_OCR_TESTS") == "1" and shutil.which("tesseract"),
                     "real Tesseract checks run in an isolated CI process")
    def test_real_lettering_is_removed_and_the_original_is_preserved(self):
        font = _font()
        if font is None:
            self.skipTest("no TrueType font")
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "product.png"
            image = Image.new("RGB", (640, 360), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((40, 90, 600, 270), 40, fill=(22, 22, 22))     # product body
            draw.ellipse((470, 150, 530, 210), outline=(0, 190, 200), width=8)     # a coloured control that must survive
            draw.text((90, 135), "LABEL", font=font, fill=(250, 215, 0))
            image.save(source)
            before = image.copy()
            prepared = prepare_unbranded_reference(source, Path(d) / "out" / "clean.png")
            self.assertEqual(prepared["status"], "TEXT_REGIONS_MASKED", prepared)
            self.assertEqual(prepared["residual_regions"], 0)
            with Image.open(source) as original, Image.open(prepared["path"]) as clean:
                self.assertEqual(list(original.getdata()), list(before.getdata()), "the original must stay untouched")
                clean = clean.convert("RGB")
                yellow = sum(1 for px in clean.crop((80, 120, 460, 220)).getdata() if px[0] > 200 and px[1] > 170 and px[2] < 80)
                self.assertLess(yellow, 40, "letters should be gone")
                self.assertEqual(clean.getpixel((20, 20)), (255, 255, 255), "plain background must stay as it was")
                teal = sum(1 for px in clean.crop((460, 140, 540, 220)).getdata() if px[1] > 150 and px[2] > 150 and px[0] < 60)
                self.assertGreater(teal, 150, "product detail outside the lettering must survive")
            again = prepare_unbranded_reference(source, Path(d) / "out" / "clean.png")
            self.assertTrue(again["cached"])


    @unittest.skipUnless(os.environ.get("CME_RUN_REAL_OCR_TESTS") == "1" and shutil.which("tesseract"),
                     "real Tesseract checks run in an isolated CI process")
    def test_neutral_text_is_removed_without_flattening_a_surface_gradient(self):
        from numpy import abs as np_abs, asarray, uint8
        font = _font()
        if font is None:
            self.skipTest("no TrueType font")
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "gradient_product.png"
            height, width = 360, 720
            base = Image.new("RGB", (width, height), (255, 255, 255))
            body_mask = Image.new("L", (width, height), 0)
            ImageDraw.Draw(body_mask).rounded_rectangle((40, 90, 680, 270), radius=40, fill=255)
            pixels = asarray([[[int(22 + 35 * x / width + 5 * y / height)] * 3
                               for x in range(width)] for y in range(height)], dtype=uint8)
            gradient = Image.fromarray(pixels, "RGB")
            base.paste(gradient, (0, 0), body_mask)
            ImageDraw.Draw(base).ellipse((560, 150, 620, 210), outline=(0, 190, 200), width=8)
            marked = base.copy()
            ImageDraw.Draw(marked).text((135, 142), "MODEL 27", font=font, fill=(225, 225, 225))
            marked.save(source)

            before = list(marked.getdata())
            prepared = prepare_unbranded_reference(source, Path(d) / "out" / "clean.png")
            self.assertEqual(prepared["status"], "TEXT_REGIONS_MASKED", prepared)
            with Image.open(source) as original:
                self.assertEqual(list(original.getdata()), before, "the original reference must remain untouched")
            with Image.open(prepared["path"]) as cleaned:
                output = asarray(cleaned.convert("RGB")).astype("int16")
            expected = asarray(base).astype("int16")
            error = np_abs(output - expected)
            self.assertLess(float(error[130:225, 90:535].mean()), 3.0,
                            "restoration must preserve gradual surface shading rather than create a patch")

    @unittest.skipUnless(os.environ.get("CME_RUN_REAL_OCR_TESTS") == "1" and shutil.which("tesseract"),
                     "real Tesseract checks run in an isolated CI process")
    def test_seese_real_reference_fails_closed_without_publishing_original_as_unbranded(self):
        source = ROOT / "tests" / "fixtures" / "seese_product_reference.jpg"
        with tempfile.TemporaryDirectory() as d:
            destination = Path(d) / "out" / "target_product_unbranded_01.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Simulate a stale cache restore from an older version: it must be removed before a failed retry.
            destination.write_bytes(png_bytes())
            destination.with_suffix(".prep.json").write_text(
                json.dumps({"status": "MASK_INCOMPLETE", "algorithm_version": 2}), encoding="utf-8")
            source_before = source.read_bytes()
            report = prepare_unbranded_reference(source, destination)
            self.assertEqual(report["status"], "REFERENCE_CLEANING_FAILED", report)
            self.assertEqual(report["failure_reason"], "AMBIGUOUS_LOW_CONFIDENCE_OCR")
            self.assertGreaterEqual(len(report["uncertain_detections"]), 1)
            self.assertIsNone(report["path"])
            self.assertIsNone(report["prepared_sha256"])
            self.assertFalse(destination.exists(), "failed preparation must not publish an image as unbranded")
            sidecar = destination.with_suffix(".prep.json")
            self.assertTrue(sidecar.exists(), "diagnostic sidecar may remain without an image payload")
            sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(sidecar_data["status"], "REFERENCE_CLEANING_FAILED")
            self.assertIsNone(sidecar_data["path"])
            self.assertIsNone(sidecar_data["prepared_sha256"])
            self.assertEqual(source.read_bytes(), source_before, "the original branded source must remain untouched")

    @unittest.skipUnless(os.environ.get("CME_RUN_REAL_OCR_TESTS") == "1" and shutil.which("tesseract"),
                     "real Tesseract checks run in an isolated CI process")
    def test_seese_real_reference_blocks_frame_before_generator_and_records_distinct_status(self):
        source = ROOT / "tests" / "fixtures" / "seese_product_reference.jpg"
        with tempfile.TemporaryDirectory() as d:
            offer = replace(load_offer(EX), photos=(str(source),), branding_policy=NO_BRANDING)
            provider_calls = []
            state = new_run("seese-reference-test")
            with self.assertRaises(ReferenceCleaningError):
                produce_shot(
                    offer=offer,
                    atoms=[A("product_main", "Cordless pressure washer body is visible", kind="product")],
                    shot=1,
                    contract_id="seese-reference-contract",
                    state=state,
                    out_dir=Path(d) / "out",
                    image_provider=lambda prompt, images: (provider_calls.append((prompt, images)), png_bytes())[1],
                    qa_checks=lambda _p: [CheckResult("file", "PASS", True, "code")],
                    reference_judge=self._judge(True),
                )
            self.assertEqual(provider_calls, [], "the image generator must not receive a damaged or ambiguous reference")
            self.assertEqual(state.attempts[-1]["status"], "REFERENCE_CLEANING_FAILED")
            blocked = list((Path(d) / "out" / "receipts").glob("blocked_*.json"))
            self.assertEqual(len(blocked), 1)
            self.assertEqual(json.loads(blocked[0].read_text())["overall"], "REFERENCE_CLEANING_FAILED")
            self.assertFalse((Path(d) / "out" / "reference_inputs" / "target_product_unbranded_01.png").exists())

    def test_failed_reference_judge_removes_candidate_unbranded_file(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(GateError):
                self._run(d, None, status="NO_CONFIDENT_TEXT_FOUND")
            self.assertFalse((Path(d) / "out" / "reference_inputs" / "target_product_unbranded_01.png").exists())

    def test_polygon_groups_do_not_merge_by_their_expanded_union_box(self):
        def hit(box, word):
            x0, y0, x1, y1 = box
            return {"poly": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                    "word": word, "conf": 90, "angle": 0, "variant": "gray"}
        # A and B overlap and form one text group. C does not overlap either original polygon by 30%,
        # but the old expanded union rectangle of A+B overlapped C and incorrectly merged the groups.
        groups = _merge_polygons([
            hit((0, 0, 100, 100), "A"),
            hit((40, 0, 140, 100), "B"),
            hit((110, 40, 160, 100), "C"),
        ])
        self.assertEqual(len(groups), 2)

    def test_reference_repair_rejects_large_regions_and_any_change_outside_support(self):
        import numpy as np

        image = Image.new("RGB", (100, 100), (40, 70, 100))
        large_polygon = [[(5, 5), (95, 5), (95, 95), (5, 95)]]
        unchanged, accepted, rejected, _support, reports = _fill_clusters(image, [large_polygon])
        self.assertEqual(rejected, 1)
        self.assertEqual(reports[0]["reason"], "OCR_REGION_TOO_LARGE")
        self.assertEqual(list(unchanged.getdata()), list(image.getdata()))

        small_polygon = [[(35, 35), (55, 35), (55, 55), (35, 55)]]
        repair_support = Image.new("L", image.size, 0)
        ImageDraw.Draw(repair_support).rectangle((42, 42, 47, 47), fill=255)
        glyph_mask = repair_support

        def letter_mask(_image, _polygons):
            return glyph_mask, (40, 70, 100)

        def unsafe_repair(source, _polygons, text_mask=None):
            pixels = np.asarray(source.convert("RGB")).copy()
            pixels[0, 0] = (255, 0, 0)  # deliberate drift outside the declared repair support
            return Image.fromarray(pixels), np.asarray(repair_support)

        with patch("cme.branding._letter_mask", side_effect=letter_mask), \
             patch("cme.branding._directional_surface_repair", side_effect=unsafe_repair):
            unchanged, accepted, rejected, _support, reports = _fill_clusters(image, [small_polygon])
        self.assertEqual(rejected, 1)
        self.assertEqual(reports[0]["reason"], "CHANGES_OUTSIDE_REPAIR_MASK")
        self.assertEqual(reports[0]["outside_changed_pixels"], 1)
        self.assertEqual(list(unchanged.getdata()), list(image.getdata()))

    def test_multicolour_regions_are_left_alone(self):
        from cme.branding import _letter_mask
        image = Image.new("RGB", (200, 120), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        for i in range(0, 200, 8):                                               # photo-like product detail
            draw.rectangle((i, 30, i + 6, 90), fill=((i * 5) % 255, (i * 11) % 255, (i * 17) % 255))
        mask, _surface = _letter_mask(image, [[(0, 20), (200, 20), (200, 100), (0, 100)]])
        self.assertIsNone(mask)

    def _prepared_stub(self, d, status="TEXT_REGIONS_MASKED", **extra):
        def prepare(source, destination):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(png_bytes())
            return {"path": str(destination), "status": status, "masked_regions": 1, "residual_regions": 0,
                    "detected_words": ["LABEL"], "source_sha256": "a" * 64, "prepared_sha256": "b" * 64, **extra}
        return prepare

    def _run(self, d, judge, status="TEXT_REGIONS_MASKED", **extra):
        source = Path(d) / "target.png"
        Image.new("RGB", (160, 100), (15, 15, 15)).save(source)
        offer = replace(load_offer(EX), photos=(str(source),), branding_policy=NO_BRANDING)
        calls = []

        def provider(prompt, images):
            calls.append((prompt, list(images)))
            return png_bytes()
        with patch("cme.run.prepare_unbranded_reference", side_effect=self._prepared_stub(d, status, **extra)):
            result = produce_shot(
                offer=offer, atoms=[A("product_main", "Black handheld vacuum body is visible", kind="product")],
                shot=1, contract_id="branding-test-contract", state=new_run("branding-test-run"),
                out_dir=Path(d) / "out", image_provider=provider,
                qa_checks=lambda _p: [CheckResult("file", "PASS", True, "code")], reference_judge=judge)
        return result, calls, source

    @staticmethod
    def _judge(passed, evidence="no lettering visible"):
        def judge(_path, atoms):
            return {a.id: {"pass": passed, "evidence": evidence} for a in atoms}
        return judge

    def test_generator_receives_confirmed_reference_and_receipt_records_it(self):
        with tempfile.TemporaryDirectory() as d:
            (receipt, _, _), calls, source = self._run(d, self._judge(True))
            self.assertEqual(len(calls), 1)
            self.assertNotEqual(Path(calls[0][1][0]).resolve(), source.resolve())
            self.assertIn("no_branding", calls[0][0])
            self.assertIn("FRAME FORM", calls[0][0])
            data = json.loads((Path(d) / "out" / "receipts" / f"{receipt.receipt_id}.json").read_text())
            self.assertEqual(data["branding_policy"], NO_BRANDING)
            row = data["image_inputs"][0]
            self.assertEqual(row["role"], "target_product_photo_unbranded_reference")
            self.assertEqual(row["source_filename"], "target.png")
            self.assertEqual(row["preparation"]["status"], "TEXT_REGIONS_MASKED")
            self.assertEqual(row["preparation"]["verification"]["status"], "PASS")
            self.assertEqual(row["preparation"]["detected_words"], ["LABEL"])
            self.assertEqual(data["reference_preparation"][0]["prepared_sha256"], "b" * 64)

    def test_no_text_found_is_not_proof_the_reference_is_clean(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(GateError) as cm:
                self._run(d, self._judge(False, "yellow wordmark LABEL visible on the body"), status="NO_CONFIDENT_TEXT_FOUND")
            self.assertIn("still visible", str(cm.exception))
            self.assertIn("LABEL", str(cm.exception))
            blocked = list((Path(d) / "out" / "receipts").glob("blocked_*.json"))
            self.assertEqual(len(blocked), 1)
            self.assertEqual(json.loads(blocked[0].read_text())["overall"], "REFERENCE_CLEANING_FAILED")

    def test_generation_is_blocked_when_nothing_can_confirm_the_reference(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(GateError) as cm:
                self._run(d, None)
            self.assertIn("could not be confirmed", str(cm.exception))

    def test_unavailable_or_malformed_judge_blocks(self):
        def broken(_path, _atoms):
            raise TimeoutError("504 upstream idle timeout")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(GateError):
                self._run(d, broken)

    def test_incomplete_mask_blocks_before_any_provider_call(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(GateError) as cm:
                self._run(d, self._judge(True), status="MASK_INCOMPLETE", residual_regions=2)
            self.assertIn("still read as text", str(cm.exception))

    def test_target_brand_allowed_needs_no_reference_cleaning(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "target.png"
            Image.new("RGB", (160, 100), (15, 15, 15)).save(source)
            offer = replace(load_offer(EX), photos=(str(source),), branding_policy=TARGET_BRAND_ALLOWED)
            seen = []
            with patch("cme.run.prepare_unbranded_reference", side_effect=AssertionError("must not run")):
                produce_shot(offer=offer, atoms=[A("product_main", "Black handheld vacuum body is visible", kind="product")],
                             shot=1, contract_id="c", state=new_run("r"), out_dir=Path(d) / "out",
                             image_provider=lambda prompt, images: (seen.append(list(images)), png_bytes())[1],
                             qa_checks=lambda _p: [CheckResult("file", "PASS", True, "code")])
            self.assertEqual(Path(seen[0][0]).resolve(), source.resolve())

    def test_marks_are_resolved_once_for_every_stage(self):
        from cme.branding import resolved_brand_marks
        self.assertEqual(resolved_brand_marks(NO_BRANDING, ("Fanttik",), "Fanttik V8 Mate Cordless Car Vacuum", "cordless handheld car vacuum"), ())
        self.assertEqual(resolved_brand_marks(TARGET_BRAND_ALLOWED, ("Fanttik",), "x", "y"), ("Fanttik",))
        self.assertEqual(resolved_brand_marks(TARGET_BRAND_ALLOWED, (), "Fanttik V8 Mate Cordless Car Vacuum", "cordless handheld car vacuum"),
                         ("Fanttik V8 Mate",))


def _font():
    from PIL import ImageFont
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except OSError:
        return None


if __name__ == "__main__":
    unittest.main()
