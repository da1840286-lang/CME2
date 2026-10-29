"""Offline unit tests for the calibration harness."""
import itertools
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cme.atoms import Atom
from cme.calibration import make_synthetic_negatives, run_calibration, verdict_hint

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

    def test_synthetic_negatives_count_as_negatives_without_labels(self):
        report = run_calibration({"m": judge_from(lambda f, a: True)}, {"synthetic:blank": "blank.png"}, ATOMS, repeats=2)
        self.assertEqual(report["modes"]["m"]["false_pass_rate"], 1.0)

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

    def test_synthetic_negative_frames_are_created(self):
        with tempfile.TemporaryDirectory() as d:
            ref = Path(d) / "ref.png"
            Image.new("RGB", (200, 300), (10, 120, 200)).save(ref)
            made = make_synthetic_negatives(Path(d) / "neg", ref)
            self.assertEqual(set(made), {"synthetic:blank", "synthetic:product_only"})
            self.assertTrue(all(p.is_file() for p in made.values()))


if __name__ == "__main__":
    unittest.main()


