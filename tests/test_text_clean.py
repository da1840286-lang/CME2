"""no_branding reference cleaner: detector rules, mask limits, status contract, and (when the engine is installed)
a real end-to-end run that checks the original is untouched and every pixel outside the mask is bit-identical."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from cme import text_clean as T
from cme.branding import (INPAINT_PREPARATION_VERSION, prepare_unbranded_reference, reference_preparation_version,
                          unbranded_engine)


def _poly(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class FakeOCR:
    """rapidocr stand-in: returns scripted boxes per orientation call order (0, 90, 270)."""
    def __init__(self, by_call):
        self.by_call, self.n = by_call, 0

    def __call__(self, _img):
        rows = self.by_call[self.n] if self.n < len(self.by_call) else []
        self.n += 1
        return rows, None


class DetectorRules(unittest.TestCase):
    def _detect(self, upright=(), rot90=(), rot270=()):
        img = np.full((200, 300, 3), 255, np.uint8)
        with patch.object(T, "_get_ocr", return_value=FakeOCR([list(upright), list(rot90), list(rot270)])):
            return T.detect(img)

    def test_single_upright_badge_letter_is_lettering_only_when_confident(self):
        acc, unc = self._detect(upright=[(_poly(10, 10, 40, 40), "S", 0.895)])
        self.assertEqual([d["text"] for d in acc], ["S"])
        acc, unc = self._detect(upright=[(_poly(10, 10, 40, 40), "C", 0.668)])
        self.assertEqual((acc, unc), ([], []), "a lone low-confidence glyph is product geometry")

    def test_rotated_single_character_is_never_lettering(self):
        acc, unc = self._detect(rot90=[(_poly(10, 10, 40, 40), "C", 0.95)])
        self.assertEqual((acc, unc), ([], []))

    def test_vertical_word_is_found_in_rotated_pass_and_mapped_back(self):
        # rotated 90 clockwise: a box at x'=[20,60], y'=[30,90] lives at x=[30,90], y=[200-1-60, 200-1-20] in a 300x200 frame
        acc, _ = self._detect(rot90=[(_poly(20, 30, 60, 90), "ke", 0.73)])
        self.assertEqual([d["text"] for d in acc], ["ke"])
        x0, y0, x1, y1 = T._bbox(acc[0]["poly"])
        self.assertEqual((int(x0), int(x1)), (30, 90))
        self.assertEqual((int(y0), int(y1)), (139, 179))

    def test_doubtful_upright_word_is_reported_not_dropped(self):
        acc, unc = self._detect(upright=[(_poly(10, 10, 80, 40), "VAC", 0.52)])
        self.assertEqual(acc, [])
        self.assertEqual([d["text"] for d in unc], ["VAC"])

    def test_same_word_seen_in_two_orientations_is_kept_once(self):
        acc, _ = self._detect(upright=[(_poly(10, 10, 80, 40), "LOGO", 0.99)],
                              rot90=[(_poly(160, 10, 190, 80), "LOGO", 0.70)])
        self.assertEqual(len(acc), 1)


class MaskRules(unittest.TestCase):
    def test_label_plate_is_included_but_not_the_product_body(self):
        img = np.full((300, 400, 3), 235, np.uint8)
        img[40:260, 40:360] = (150, 200, 20)               # green body
        img[124:156, 128:252] = (10, 10, 10)               # black sticker, a bit larger than its lettering
        for x in range(146, 236, 16):                      # white strokes standing in for letters
            img[133:147, x:x + 7] = (250, 250, 250)
        m = T.build_mask(img, [{"poly": _poly(140, 130, 240, 150)}])
        self.assertTrue(m[126, 130] > 0 and m[154, 250] > 0, "the whole sticker is masked, not just the letters")
        self.assertEqual(int(m[60, 60]), 0, "the body away from the sticker is not")

    def test_plate_that_merges_with_the_body_falls_back_to_the_text_box(self):
        img = np.full((300, 400, 3), 12, np.uint8)         # everything is the same black: no separable plate
        img[130:150, 140:240] = (250, 250, 250)
        m = T.build_mask(img, [{"poly": _poly(140, 130, 240, 150)}])
        ys, xs = np.where(m > 0)
        self.assertLessEqual(xs.max() - xs.min(), 100 + 2 * T.MASK_DILATE_PX + 2)


class StatusContract(unittest.TestCase):
    def _clean(self, accepted, uncertain=(), mask_cov=None):
        img = np.full((300, 400, 3), 200, np.uint8)
        seq = [(list(accepted), list(uncertain)), ([], [])]
        with patch.object(T, "detect", side_effect=seq + [([], [])]), \
             patch.object(T, "_tesseract_words", return_value=[]), \
             patch.object(T, "inpaint", side_effect=lambda im, m: im.copy()), \
             patch.object(T, "seam_ratio", return_value=0.5):
            return T.clean(img)

    def test_nothing_found_is_not_proof(self):
        status, out, rep = self._clean([])
        self.assertEqual(status, "NO_CONFIDENT_TEXT_FOUND")

    def test_only_doubtful_lettering_needs_review_and_has_no_candidate(self):
        unc = [{"poly": _poly(5, 5, 40, 20), "text": "VAC", "conf": 0.52, "orientation": 0}]
        status, out, rep = self._clean([], unc)
        self.assertEqual((status, out), ("NEEDS_REVIEW", None))
        self.assertEqual(rep["uncertain_detections"][0]["word"], "VAC")

    def test_mask_over_eight_percent_is_rejected_never_repaired(self):
        big = [{"poly": _poly(0, 0, 200, 150), "text": "HUGE", "conf": 0.99, "orientation": 0}]   # 25% of the frame
        with patch.object(T, "inpaint", side_effect=AssertionError("must not repair")):
            img = np.full((300, 400, 3), 200, np.uint8)
            with patch.object(T, "detect", return_value=(big, [])):
                status, out, rep = T.clean(img)
        self.assertEqual((status, rep["failure_reason"]), ("REFERENCE_CLEANING_FAILED", "MASK_TOO_LARGE"))
        self.assertEqual(T.MAX_MASK_COVERAGE, 0.08)

    def test_text_that_survives_cleaning_fails(self):
        acc = [{"poly": _poly(20, 20, 40, 30), "text": "LOGO", "conf": 0.99, "orientation": 0}]
        img = np.full((300, 400, 3), 200, np.uint8)
        with patch.object(T, "detect", side_effect=[(acc, []), (acc, [])]), \
             patch.object(T, "_tesseract_words", return_value=[]), \
             patch.object(T, "inpaint", side_effect=lambda im, m: im.copy()):
            status, out, rep = T.clean(img)
        self.assertEqual((status, rep["failure_reason"]), ("REFERENCE_CLEANING_FAILED", "RESIDUAL_TEXT_AFTER_MASKING"))

    def test_a_pixel_changed_outside_the_mask_fails(self):
        acc = [{"poly": _poly(20, 20, 40, 30), "text": "LOGO", "conf": 0.99, "orientation": 0}]
        img = np.full((300, 400, 3), 200, np.uint8)

        def leaky(im, m):
            out = im.copy(); out[90, 90] = 0
            return out
        with patch.object(T, "detect", return_value=(acc, [])), patch.object(T, "inpaint", side_effect=leaky):
            status, out, rep = T.clean(img)
        self.assertEqual((status, rep["failure_reason"]), ("REFERENCE_CLEANING_FAILED", "CHANGES_OUTSIDE_REPAIR_MASK"))

    def test_doubtful_leftover_after_cleaning_blocks_the_pass(self):
        acc = [{"poly": _poly(20, 20, 40, 30), "text": "LOGO", "conf": 0.99, "orientation": 0}]
        doubt = [{"poly": _poly(60, 60, 90, 75), "text": "BAG", "conf": 0.51, "orientation": 0}]
        img = np.full((300, 400, 3), 200, np.uint8)
        with patch.object(T, "detect", side_effect=[(acc, []), ([], doubt)]), \
             patch.object(T, "_tesseract_words", return_value=[]), \
             patch.object(T, "inpaint", side_effect=lambda im, m: im.copy()), \
             patch.object(T, "seam_ratio", return_value=0.4):
            status, out, rep = T.clean(img)
        self.assertEqual(status, "NEEDS_REVIEW")
        self.assertIsNotNone(out)          # the candidate exists for the reviewer, but prepare_* does not publish it

    def test_visible_repair_edge_needs_review(self):
        acc = [{"poly": _poly(20, 20, 40, 30), "text": "LOGO", "conf": 0.99, "orientation": 0}]
        img = np.full((300, 400, 3), 200, np.uint8)
        with patch.object(T, "detect", side_effect=[(acc, []), ([], [])]), \
             patch.object(T, "_tesseract_words", return_value=[]), \
             patch.object(T, "inpaint", side_effect=lambda im, m: im.copy()), \
             patch.object(T, "seam_ratio", return_value=T.SEAM_REVIEW_RATIO + 0.5):
            status, out, rep = T.clean(img)
        self.assertEqual((status, rep["failure_reason"]), ("NEEDS_REVIEW", "REPAIRED_EDGE_VISIBLE"))


class PrepareContract(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict(os.environ, {"CME_UNBRANDED_ENGINE": "inpaint"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _source(self, d):
        p = Path(d) / "product.png"
        Image.new("RGB", (120, 80), (30, 30, 30)).save(p)
        return p

    def test_engine_selection_and_cache_version(self):
        self.assertEqual(unbranded_engine(), "inpaint")
        self.assertEqual(reference_preparation_version(), INPAINT_PREPARATION_VERSION)
        with patch.dict(os.environ, {"CME_UNBRANDED_ENGINE": "legacy"}):
            self.assertNotEqual(reference_preparation_version(), INPAINT_PREPARATION_VERSION)

    def test_missing_engine_is_reported_not_silently_degraded(self):
        with tempfile.TemporaryDirectory() as d, patch.object(T, "engine_status", return_value=(False, "weights missing")):
            rep = prepare_unbranded_reference(self._source(d), Path(d) / "o" / "c.png")
            self.assertEqual(rep["status"], "OCR_UNAVAILABLE")
            self.assertIsNone(rep["path"])
            self.assertFalse((Path(d) / "o" / "c.png").exists())

    def test_needs_review_publishes_no_file_and_no_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "o" / "c.png"
            fake = ("NEEDS_REVIEW", np.zeros((80, 120, 3), np.uint8),
                    {"engine": "ppocr+lama", "failure_reason": "UNCERTAIN_TEXT_REMAINS", "uncertain_detections": [],
                     "detected_words": [], "masked_regions": 1})
            with patch.object(T, "engine_status", return_value=(True, "ok")), patch.object(T, "clean", return_value=fake):
                rep = prepare_unbranded_reference(self._source(d), dest)
            self.assertEqual(rep["status"], "NEEDS_REVIEW")
            self.assertIsNone(rep["path"])
            self.assertFalse(dest.exists() or dest.with_suffix(".prep.json").exists())

    def test_pass_candidate_is_written_with_hashes_and_cached(self):
        with tempfile.TemporaryDirectory() as d:
            src, dest = self._source(d), Path(d) / "o" / "c.png"
            fake = ("TEXT_REGIONS_MASKED", np.full((80, 120, 3), 90, np.uint8),
                    {"engine": "ppocr+lama", "detected_words": ["LOGO"], "masked_regions": 1})
            with patch.object(T, "engine_status", return_value=(True, "ok")), patch.object(T, "clean", return_value=fake) as c:
                first = prepare_unbranded_reference(src, dest)
                second = prepare_unbranded_reference(src, dest)
            self.assertEqual(first["status"], "TEXT_REGIONS_MASKED")
            self.assertTrue(first["prepared_sha256"] and first["source_sha256"])
            self.assertTrue(second["cached"])
            self.assertEqual(c.call_count, 1)
            self.assertEqual(json.loads(dest.with_suffix(".prep.json").read_text())["engine"], "ppocr+lama")


@unittest.skipUnless(os.environ.get("CME_RUN_REAL_CLEANER_TESTS") == "1" and T.engine_status()[0],
                     "real PP-OCR / LaMa checks run in an isolated CI process")
class RealEngine(unittest.TestCase):
    @staticmethod
    def _photo(text=True, grille=False):
        import cv2
        rng = np.random.default_rng(3)
        img = np.full((560, 760, 3), 236, np.uint8)
        cv2.rectangle(img, (90, 120), (670, 470), (26, 26, 29), -1)
        cv2.rectangle(img, (90, 120), (670, 215), (240, 110, 20), -1)
        img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        if grille:
            for x in range(130, 640, 14):
                cv2.line(img, (x, 260), (x, 440), (70, 70, 75), 2)
        if text:
            cv2.putText(img, "SEESE", (150, 190), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
        return img

    def _run(self, img):
        with tempfile.TemporaryDirectory() as d:
            src, dest = Path(d) / "p.png", Path(d) / "o" / "c.png"
            Image.fromarray(img).save(src)
            before = src.read_bytes()
            with patch.dict(os.environ, {"CME_UNBRANDED_ENGINE": "inpaint"}):
                rep = prepare_unbranded_reference(src, dest)
            self.assertEqual(src.read_bytes(), before, "the original photo must stay untouched")
            cleaned = np.array(Image.open(dest).convert("RGB")) if dest.is_file() else None
            return rep, cleaned

    def test_lettering_is_removed_and_nothing_outside_the_mask_changes(self):
        img = self._photo()
        rep, cleaned = self._run(img)
        self.assertEqual(rep["status"], "TEXT_REGIONS_MASKED", rep)
        self.assertEqual(rep["outside_changed_pixels"], 0)
        self.assertEqual(rep["residual_regions"], 0)
        left, _ = T.detect(cleaned)
        self.assertEqual(left, [])
        far = np.zeros(img.shape[:2], bool); far[300:, :] = True
        self.assertTrue(np.array_equal(img[far], cleaned[far]))

    def test_clean_photo_is_not_blocked_and_not_altered(self):
        for grille in (False, True):
            img = self._photo(text=False, grille=grille)
            rep, cleaned = self._run(img)
            self.assertEqual(rep["status"], "NO_CONFIDENT_TEXT_FOUND", rep)
            self.assertTrue(np.array_equal(img, cleaned))


if __name__ == "__main__":
    unittest.main()
