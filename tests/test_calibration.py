"""Offline unit tests for the calibration harness."""
import itertools
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cme.atoms import Atom
from cme.calibration import (
    build_calibration_atoms, make_synthetic_negatives, run_calibration, verdict_hint,
)
from cme.qa import PRODUCT_IDENTITY_CHECK_ID, SYSTEM_NO_TEXT_ID, TARGET_BRANDING_CHECK_ID

ATOMS = [Atom("mech_result_1", "action", "after-state is visible", "MUST", "image", 1),
         Atom("mech_intervention_1", "action", "tool touches the problem", "MUST", "image", 1)]
FRAMES = {"shown.png": "shown.png", "not_shown.png": "not_shown.png"}
LABELS = {"shown.png": {"mech_result_1": True, "mech_intervention_1": True},
          "not_shown.png": {"mech_result_1": False, "mech_intervention_1": False}}


def judge_from(rule):
    def judge(path, atoms, refs=()):
        return {a.id: {"pass": rule(Path(path).name, a.id), "evidence": "visible fact"} for a in atoms}
    return judge


class CalibrationTests(unittest.TestCase):
    def test_sycophantic_judge_is_caught(self):
        report = run_calibration({"observe_verify": judge_from(lambda f, a: True)}, FRAMES, ATOMS, labels=LABELS, repeats=3)
        mode = report["modes"]["observe_verify"]
        self.assertEqual(mode["false_pass_rate"], 1.0)
        self.assertEqual(mode["missed_pass_rate"], 0.0)
        self.assertTrue(verdict_hint(report).startswith("FAIL"))

    def test_correct_and_repeatable_judge_is_ok_only_with_both_classes(self):
        report = run_calibration({"observe_verify": judge_from(lambda f, a: f == "shown.png")}, FRAMES, ATOMS, labels=LABELS, repeats=4)
        mode = report["modes"]["observe_verify"]
        self.assertEqual((mode["false_pass_rate"], mode["missed_pass_rate"], mode["unstable_cells"]), (0.0, 0.0, 0))
        self.assertTrue(verdict_hint(report).startswith("OK on this labeled sample"))

    def test_flaky_judge_is_reported_as_unstable(self):
        flips = {a.id: itertools.cycle([True, False]) for a in ATOMS}
        report = run_calibration({"observe_verify": judge_from(lambda f, a: next(flips[a]))}, {"shown.png": "shown.png"}, ATOMS,
                                 labels=LABELS, repeats=4)
        self.assertGreater(report["modes"]["observe_verify"]["unstable_cells"], 0)

    def test_synthetic_negatives_do_not_count_as_real_negative_labels(self):
        report = run_calibration({"observe_verify": judge_from(lambda f, a: True)}, {"synthetic:blank": "blank.png"}, ATOMS, repeats=2)
        self.assertIsNone(report["modes"]["observe_verify"]["false_pass_rate"])
        self.assertEqual(report["modes"]["observe_verify"]["negative_calls"], 0)
        self.assertTrue(verdict_hint(report).startswith("INCOMPLETE"))

    def test_judge_errors_are_not_run_never_pass(self):
        def broken(path, atoms, refs=()):
            raise RuntimeError("provider down")
        report = run_calibration({"observe_verify": broken}, FRAMES, ATOMS, labels=LABELS, repeats=2)
        mode = report["modes"]["observe_verify"]
        self.assertEqual(mode["false_pass_rate"], 0.0)
        self.assertEqual(mode["missed_pass_rate"], 1.0)
        self.assertEqual(mode["not_run_rate"], 1.0)
        self.assertTrue(verdict_hint(report).startswith("INCOMPLETE"))

    def test_missing_positive_labels_cannot_report_ok(self):
        report = run_calibration({"observe_verify": judge_from(lambda f, a: False)}, {"synthetic:blank": "blank.png"}, ATOMS, repeats=2)
        self.assertTrue(verdict_hint(report).startswith("INCOMPLETE"))
    def test_synthetic_controls_do_not_count_as_real_negative_labels(self):
        report = run_calibration(
            {"observe_verify": judge_from(lambda _f, _a: False)},
            {"synthetic:blank": "blank.png"}, ATOMS, repeats=2,
        )
        mode = report["modes"]["observe_verify"]
        self.assertEqual(mode["negative_cells"], 0)
        self.assertIsNone(mode["false_pass_rate"])
        self.assertTrue(verdict_hint(report).startswith("INCOMPLETE"))

    def test_calibration_atoms_mirror_production_system_pixel_checks(self):
        atoms = build_calibration_atoms(
            ATOMS, shot=1, branding_policy="target_brand_allowed",
            approved_brand_marks=("SEESE",), include_brand_reference=True,
        )
        by_id = {atom.id: atom for atom in atoms}
        self.assertIn(SYSTEM_NO_TEXT_ID, by_id)
        self.assertIn(TARGET_BRANDING_CHECK_ID, by_id)
        self.assertEqual(by_id[SYSTEM_NO_TEXT_ID].channel, "qa")
        self.assertIn("NO-ADDED-TEXT PIXEL CHECK", by_id[SYSTEM_NO_TEXT_ID].text)
        self.assertIn("SEESE", by_id[TARGET_BRANDING_CHECK_ID].text)
        self.assertFalse(any(atom.id == PRODUCT_IDENTITY_CHECK_ID for atom in atoms))

    def test_reference_identity_result_is_included_as_an_extra_check(self):
        seen_refs = []

        def judge(path, atoms, refs=()):
            seen_refs.append(tuple(refs))
            out = {atom.id: {"pass": True, "evidence": "visible check"} for atom in atoms}
            if refs:
                out[PRODUCT_IDENTITY_CHECK_ID] = {"pass": True, "evidence": "matching product shape"}
            return out

        report = run_calibration(
            {"observe_verify": judge}, {"real.png": "real.png"}, ATOMS,
            labels={"real.png": {"mech_result_1": True}}, repeats=1,
            refs=("reference.png",), extra_check_ids=(PRODUCT_IDENTITY_CHECK_ID,),
        )
        cells = {row["atom"] for row in report["modes"]["observe_verify"]["cells"]}
        self.assertIn(PRODUCT_IDENTITY_CHECK_ID, cells)
        self.assertEqual(seen_refs, [("reference.png",)])

    def test_each_system_check_requires_real_positive_and_negative_labels(self):
        ids = (SYSTEM_NO_TEXT_ID, TARGET_BRANDING_CHECK_ID, PRODUCT_IDENTITY_CHECK_ID)
        atoms = build_calibration_atoms(
            ATOMS, branding_policy="target_brand_allowed", approved_brand_marks=("SEESE",),
            include_brand_reference=True,
        )
        labels = {
            "good.png": {**{check_id: True for check_id in ids}, "mech_result_1": True},
            "bad.png": {**{check_id: False for check_id in ids}, "mech_result_1": False},
        }

        def judge(path, judge_atoms, refs=()):
            name = Path(path).name
            out = {atom.id: {"pass": labels[name].get(atom.id, False), "evidence": "manual reference"}
                   for atom in judge_atoms}
            if refs:
                out[PRODUCT_IDENTITY_CHECK_ID] = {
                    "pass": labels[name][PRODUCT_IDENTITY_CHECK_ID], "evidence": "reference comparison"}
            return out

        report = run_calibration(
            {"observe_verify": judge}, {"good.png": "good.png", "bad.png": "bad.png"},
            atoms, labels=labels, repeats=2, refs=("reference.png",),
            extra_check_ids=(PRODUCT_IDENTITY_CHECK_ID,), balanced_real_ids=ids,
        )
        self.assertTrue(verdict_hint(report).startswith("OK on this labeled sample"))

        labels["bad.png"].pop(TARGET_BRANDING_CHECK_ID)
        incomplete = run_calibration(
            {"observe_verify": judge}, {"good.png": "good.png", "bad.png": "bad.png"},
            atoms, labels=labels, repeats=1, refs=("reference.png",),
            extra_check_ids=(PRODUCT_IDENTITY_CHECK_ID,), balanced_real_ids=ids,
        )
        self.assertIn(TARGET_BRANDING_CHECK_ID, verdict_hint(incomplete))

    def test_synthetic_negative_frames_are_created(self):
        with tempfile.TemporaryDirectory() as d:
            ref = Path(d) / "ref.png"
            Image.new("RGB", (200, 300), (10, 120, 200)).save(ref)
            made = make_synthetic_negatives(Path(d) / "neg", ref)
            self.assertEqual(set(made), {"synthetic:blank", "synthetic:product_only"})
            self.assertTrue(all(p.is_file() for p in made.values()))


if __name__ == "__main__":
    unittest.main()


