"""Regression tests for defects found in an outside review. Each one was reproduced on the previous code first."""
import json, os, tempfile, unittest
from pathlib import Path
from cme.brain import BrainError, request_atoms
from cme.compiler import BudgetError
from cme.offer import load_offer
from cme.providers import CallLog, InfraError, ModelRef, ProviderPolicyError, call_with_policy
from cme.qa import frame_content_check, CheckResult, file_check
from cme.receipt import attach_qa, build_video_receipt, overall
from cme.run import produce_shot
from cme.state import StateMismatch, accept_shot, new_run
from cme.vertical import GateError, load_vertical
from tests.helpers import A, png_bytes

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "offer_example.json"
GOOD = {"id": "a_1", "kind": "scene", "text": "A kitchen counter", "priority": "MUST", "channel": "image", "shot": 1}


def run_shot(atoms, qa, d, state=None, offer=None, provider=None, **kw):
    return produce_shot(offer=offer or load_offer(EX), atoms=atoms, shot=1, contract_id="c", state=state or new_run("r"),
                        out_dir=d, image_provider=provider or (lambda p, r: png_bytes()), qa_checks=qa, **kw)


class B1_EveryMustNeedsVerification(unittest.TestCase):
    def test_file_check_plus_one_qa_atom_is_not_a_pass(self):
        atoms = [A("scene", "A bright kitchen counter"), A("prod", "Product in the foreground"),
                 A("q1", "Product visible", ch="qa", kind="qa_check")]
        qa = lambda p: [file_check("file", p), CheckResult("q1", "PASS", True, "judge")]
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_shot(atoms, qa, d)[2], "UNVERIFIED")

    def test_pass_only_when_each_must_atom_has_a_passing_check(self):
        atoms = [A("scene", "A bright kitchen counter"), A("prod", "Product in the foreground")]
        qa = lambda p: [frame_content_check("frame_content", p), CheckResult("scene", "PASS", True, "judge"),
                        CheckResult("prod", "PASS", True, "judge")]
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_shot(atoms, qa, d)[2], "PASS")

    def test_should_atoms_do_not_block_pass(self):
        atoms = [A("scene", "A bright kitchen counter"), A("nice", "Warm light", "SHOULD")]
        qa = lambda p: [frame_content_check("frame_content", p), CheckResult("scene", "PASS", True, "judge")]
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_shot(atoms, qa, d)[2], "PASS")


class B2_ImageAndVideoReceiptsAreSeparate(unittest.TestCase):
    def test_motion_atom_does_not_block_an_image_shot(self):
        atoms = [A("scene", "A bright kitchen counter"), A("mv", "Slow push in", ch="motion")]
        qa = lambda p: [frame_content_check("frame_content", p), CheckResult("scene", "PASS", True, "judge")]
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_shot(atoms, qa, d)[2], "PASS")

    def test_video_receipt_has_its_own_evidence(self):
        atoms = [A("mv", "Slow push in", ch="motion", shot=1), A("mv2", "Pan left", ch="motion", shot=2)]
        missing = build_video_receipt("c", "r", 1, atoms, "Shot 1: Slow push in")
        self.assertEqual(overall(missing), "NOT_DELIVERED")
        full = build_video_receipt("c", "r", 1, atoms, "Shot 1: Slow push in\nShot 2: Pan left")
        done = attach_qa(full, atoms, [CheckResult("video_file", "PASS", pixels=True), CheckResult("mv", "PASS", True, "judge"),
                                       CheckResult("mv2", "PASS", True, "judge")])
        self.assertEqual(overall(done), "PASS")
        unchecked = attach_qa(full, atoms, [CheckResult("video_file", "PASS")])
        self.assertEqual(overall(unchecked), "UNVERIFIED")


class B3_PhotoPathsAreResolved(unittest.TestCase):
    def test_provider_gets_real_files_from_any_working_directory(self):
        seen = []
        cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            with tempfile.TemporaryDirectory() as d:
                def provider(p, refs):
                    seen.extend(refs)
                    return png_bytes()
                run_shot([A("a", "A kitchen counter")], lambda p: [file_check("file", p)], d, provider=provider)
        finally:
            os.chdir(cwd)
        self.assertTrue(seen and all(Path(x).is_absolute() and Path(x).exists() for x in seen))

    def test_brain_request_has_no_local_paths(self):
        req = load_offer(EX).to_dict()
        self.assertEqual(req["photos"], ["product_01.png"])


class B4_DuplicateIdsAreRejected(unittest.TestCase):
    def test_duplicate_is_rejected_and_retried(self):
        offer, calls = load_offer(EX), []
        def call(req):
            calls.append(req)
            if len(calls) == 1:
                return {"atoms": [GOOD, dict(GOOD, text="Another line here")]}
            return {"atoms": [GOOD]}
        atoms = request_atoms(offer, "b", 1, call)
        self.assertEqual(len(calls), 2)
        self.assertIn("a_1", [f["id"] for f in calls[1]["only_fix"]])
        self.assertEqual(len(atoms), 1)

    def test_unfixed_duplicate_raises(self):
        with self.assertRaises(BrainError):
            request_atoms(load_offer(EX), "b", 1, lambda r: {"atoms": [GOOD, dict(GOOD)]}, max_retries=1)

    def test_valid_earlier_round_does_not_hide_a_later_duplicate(self):
        calls = []
        def call(req):
            calls.append(req)
            return {"atoms": [GOOD]} if len(calls) == 1 else {"atoms": [GOOD, dict(GOOD)]}
        # round 1 is valid, so nothing is pending and the result is returned immediately
        self.assertEqual(len(request_atoms(load_offer(EX), "b", 1, call)), 1)
        self.assertEqual(len(calls), 1)


class DiagnosticsSurviveFailures(unittest.TestCase):
    def test_provider_error_is_recorded_then_raised(self):
        def boom(p, r):
            raise InfraError("quota")
        st = new_run("r")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(InfraError):
                run_shot([A("a", "A kitchen counter")], lambda p: [], d, state=st, provider=boom)
            rep = json.loads(next((Path(d) / "receipts").glob("*.json")).read_text())
        self.assertEqual(st.attempts[0]["status"], "ERROR_InfraError")
        self.assertEqual(rep["overall"], "ERROR_InfraError")
        self.assertIn("quota", rep["error"])

    def test_qa_error_is_recorded_too(self):
        def bad_qa(p):
            raise RuntimeError("qa crashed")
        st = new_run("r")
        with tempfile.TemporaryDirectory() as d, self.assertRaises(RuntimeError):
            run_shot([A("a", "A kitchen counter")], bad_qa, d, state=st)
        self.assertEqual(st.attempts[0]["status"], "ERROR_RuntimeError")

    def test_blocked_attempts_are_recorded(self):
        st = new_run("r")
        with tempfile.TemporaryDirectory() as d, self.assertRaises(BudgetError):
            run_shot([A("huge", "x" * 3700)], lambda p: [], d, state=st)
        self.assertEqual(st.attempts[0]["status"], "BLOCKED_BUDGET")
        vert = load_vertical(ROOT / "verticals" / "physical_goods_us.json")
        st2 = new_run("r2")
        with tempfile.TemporaryDirectory() as d, self.assertRaises(GateError):
            run_shot([A("a", "Proudly made in the USA")], lambda p: [], d, state=st2, vertical=vert)
        self.assertEqual(st2.attempts[0]["status"], "BLOCKED_GATE")


class StateIdentity(unittest.TestCase):
    def test_carry_only_for_same_offer_and_contract(self):
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "s.png"
            img.write_bytes(png_bytes())
            s1 = new_run("r1", offer_id="o1", contract_id="c1")
            accept_shot(s1, 1, img, "x")
            self.assertIn(1, new_run("r2", s1, "o1", "c1").shots)
            other_offer = new_run("r3", s1, "o2", "c1")
            self.assertNotIn(1, other_offer.shots)
            self.assertIn("another offer or contract", other_offer.carry_notes[0])
            self.assertNotIn(1, new_run("r4", s1, "o1", "c2").shots)

    def test_state_cannot_be_rebound_to_another_offer(self):
        st = new_run("r")
        st.bind("o1", "c1")
        with self.assertRaises(StateMismatch):
            st.bind("o2", "c1")
        with tempfile.TemporaryDirectory() as d, self.assertRaises(StateMismatch):
            run_shot([A("a", "A kitchen counter")], lambda p: [], d, state=st)


class RespondedModelIsVerified(unittest.TestCase):
    def test_substituted_model_is_rejected_and_logged(self):
        with tempfile.TemporaryDirectory() as d:
            log = CallLog(Path(d) / "c.jsonl")
            with self.assertRaises(ProviderPolicyError):
                call_with_policy(ModelRef("p", "model-a"), lambda ref: ("x", {"model": "model-b"}), log, "r")
            row = json.loads((Path(d) / "c.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["status"], "model_mismatch")
        self.assertEqual(row["responded_model"], "model-b")


if __name__ == "__main__":
    unittest.main()
