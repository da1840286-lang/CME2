import json, tempfile, unittest
from pathlib import Path
from cme.brain import BrainError, PatchError, apply_patch, request_atoms
from cme.offer import load_offer
from cme.providers import (CallLog, InfraError, ModelRef, ProviderPolicyError, call_with_policy,
                           require_paid)
from cme.qa import CheckResult, anchor_similarity, file_check, frame_content_check, judge_checks, ocr_text_check
from cme.receipt import (PayloadMismatch, ReceiptError, assert_exact_payload, attach_qa, build_receipt,
                         build_video_receipt, overall)
from cme.run import produce_shot
from cme.compiler import BudgetError
from cme.state import accept_shot, load_state, new_run, record_attempt, save_state, this_run_attempts
from tests.helpers import A, png_bytes

EX = Path(__file__).resolve().parent.parent / "examples" / "offer_example.json"


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.atoms = [A("a", "A bright kitchen"), A("q", "Product visible", ch="qa", kind="qa_check"),
                      A("m", "Slow push in", ch="motion")]

    def test_contract_id_required_no_silent_default(self):
        with self.assertRaises(ReceiptError):
            build_receipt("", "r1", 1, 1, self.atoms, "A bright kitchen")

    def test_scopes_keep_image_and_video_apart(self):
        img = build_receipt("c1", "r1", 1, 1, self.atoms, "A bright kitchen")
        vid = build_video_receipt("c1", "r1", 1, self.atoms, "Shot 1: Slow push in")
        self.assertEqual({x["atom_id"] for x in img.rows}, {"a", "q"})
        self.assertEqual({x["atom_id"] for x in vid.rows}, {"m"})
        self.assertNotEqual(img.receipt_id, vid.receipt_id)

    def test_statuses_and_exact_payload(self):
        r = build_receipt("c1", "r1", 1, 1, self.atoms, "A bright kitchen")
        st = {x["atom_id"]: x["status"] for x in r.rows}
        self.assertEqual(st, {"a": "DELIVERED", "q": "UNVERIFIED"})
        assert_exact_payload(r, "A bright kitchen")
        with self.assertRaises(PayloadMismatch):
            assert_exact_payload(r, "A bright kitchen plus appended text")

    def test_missing_must_text_is_not_delivered(self):
        r = build_receipt("c1", "r1", 1, 1, self.atoms, "something else")
        self.assertEqual(overall(r), "NOT_DELIVERED")

    def test_attach_qa_keeps_identity_and_needs_every_must_verified(self):
        r = build_receipt("c1", "r1", 1, 1, self.atoms, "A bright kitchen")
        partial = attach_qa(r, self.atoms, [CheckResult("q", "PASS"), CheckResult("frame_content", "PASS", pixels=True)])
        self.assertEqual(overall(partial), "UNVERIFIED")  # MUST atom 'a' was never checked
        full = attach_qa(r, self.atoms, [CheckResult("q", "PASS"), CheckResult("a", "PASS"), CheckResult("frame_content", "PASS", pixels=True)])
        self.assertEqual(r.receipt_id, full.receipt_id)
        self.assertEqual(r.sent_prompt, full.sent_prompt)
        self.assertEqual(overall(full), "PASS")

    def test_judge_alone_cannot_pass(self):
        r = build_receipt("c1", "r1", 1, 1, self.atoms, "A bright kitchen")
        r2 = attach_qa(r, self.atoms, [CheckResult("q", "PASS", True, "judge"), CheckResult("a", "PASS", True, "judge")])
        self.assertEqual(overall(r2), "UNVERIFIED")

    def test_required_fail_and_not_run(self):
        r = build_receipt("c1", "r1", 1, 1, self.atoms, "A bright kitchen")
        ok = [CheckResult("q", "PASS"), CheckResult("a", "PASS")]
        failed = attach_qa(r, self.atoms, ok + [CheckResult("ocr", "FAIL")])
        self.assertEqual(overall(failed), "FAILED_QA")
        notrun = attach_qa(r, self.atoms, ok + [CheckResult("frame_content", "PASS", pixels=True), CheckResult("ocr", "NOT_RUN")])
        self.assertEqual(overall(notrun), "UNVERIFIED")


class StateTests(unittest.TestCase):
    def test_carry_over_checks_hash_and_tags_runs(self):
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "s1.png"
            img.write_bytes(png_bytes())
            s1 = new_run("run1")
            accept_shot(s1, 1, img, "drec_x")
            record_attempt(s1, 1, "drec_x", "PASS")
            sp = Path(d) / "state.json"
            save_state(s1, sp)
            s2 = new_run("run2", load_state(sp))
            self.assertIn(1, s2.shots)
            self.assertEqual(s2.previous_run_id, "run1")
            self.assertEqual(this_run_attempts(s2), [])      # old diagnostics are not this run's
            img.write_bytes(png_bytes((1, 2, 3)))              # file changed after acceptance
            s3 = new_run("run3", load_state(sp))
            self.assertNotIn(1, s3.shots)
            self.assertIn("hash changed", s3.carry_notes[0])


class BrainTests(unittest.TestCase):
    def test_retry_only_failed_atoms(self):
        offer = load_offer(EX)
        reqs = []
        good = {"id": "a_ok", "kind": "scene", "text": "A kitchen counter", "priority": "MUST",
                "channel": "image", "shot": 1}
        bad = {"id": "a_bad", "kind": "claim", "text": "Lasts forever", "priority": "MUST",
               "channel": "qa", "shot": 1}
        fixed = dict(bad, text="Seals with a zip closure")

        def call(req):
            reqs.append(req)
            return {"atoms": [good, bad]} if len(reqs) == 1 else {"atoms": [fixed]}

        atoms = request_atoms(offer, "brief", 1, call)
        self.assertEqual({a.id for a in atoms}, {"a_ok", "a_bad"})
        self.assertEqual([f["id"] for f in reqs[1]["only_fix"]], ["a_bad"])
        self.assertNotIn("only_fix", reqs[0])

    def test_gives_up_with_errors(self):
        offer = load_offer(EX)
        bad = {"id": "a_bad", "kind": "scene", "text": "x" * 130, "priority": "MUST",
               "channel": "image", "shot": 1}
        with self.assertRaises(BrainError):
            request_atoms(offer, "b", 1, lambda r: {"atoms": [bad]}, max_retries=1)

    def test_patch_delete_must_needs_reason(self):
        atoms = [A("m1", "must text"), A("s1", "should text", "SHOULD")]
        with self.assertRaises(PatchError):
            apply_patch(atoms, [{"id": "m1", "action": "delete"}])
        new, report = apply_patch(atoms, [{"id": "s1", "action": "delete"},
                                          {"id": "m1", "action": "delete", "reason": "merged into m2"}])
        self.assertEqual(new, [])
        self.assertIn("m1 (MUST): merged into m2", report)

    def test_patch_update_and_downgrade_guard(self):
        atoms = [A("m1", "must text")]
        upd = {"id": "m1", "kind": "scene", "text": "new text", "priority": "MUST", "channel": "image", "shot": 1}
        new, report = apply_patch(atoms, [{"id": "m1", "action": "update", "atom": upd}])
        self.assertEqual(new[0].text, "new text")
        self.assertEqual(report, "потеряно: нет")
        down = dict(upd, priority="MAY")
        with self.assertRaises(PatchError):
            apply_patch(atoms, [{"id": "m1", "action": "update", "atom": down}])


class ProviderTests(unittest.TestCase):
    def test_free_models_blocked(self):
        for mid in ("openrouter/free", "google/x:free", ""):
            with self.assertRaises(ProviderPolicyError):
                require_paid(ModelRef("p", mid))

    def test_no_silent_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            log = CallLog(Path(d) / "calls.jsonl")

            def boom(ref):
                raise InfraError("429")

            with self.assertRaises(InfraError):
                call_with_policy(ModelRef("a", "m1"), boom, log, "r1")

    def test_declared_fallback_is_logged(self):
        with tempfile.TemporaryDirectory() as d:
            log = CallLog(Path(d) / "calls.jsonl")

            def fn(ref):
                if ref.model_id == "m1":
                    raise InfraError("timeout")
                return "ok", {"cost": 0.04}

            out = call_with_policy(ModelRef("a", "m1"), fn, log, "r1", fallback=ModelRef("b", "m2"))
            self.assertEqual(out, "ok")
            rows = [json.loads(x) for x in (Path(d) / "calls.jsonl").read_text().splitlines()]
            self.assertEqual([r["status"] for r in rows], ["infra_error", "ok"])
            self.assertTrue(rows[1]["fallback_used"])
            self.assertEqual(rows[1]["requested_model"], "m1")

    def test_quality_error_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as d:
            log = CallLog(Path(d) / "c.jsonl")

            def fn(ref):
                raise ValueError("bad output")

            with self.assertRaises(ValueError):
                call_with_policy(ModelRef("a", "m1"), fn, log, "r1", fallback=ModelRef("b", "m2"))


class QATests(unittest.TestCase):
    def test_file_check_and_anchor(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = (Path(d) / n for n in ("a.png", "b.png", "c.png"))
            a.write_bytes(png_bytes())
            b.write_bytes(png_bytes())
            c.write_bytes(png_bytes(patch=((0, 0, 40, 60), (255, 255, 255))))
            self.assertEqual(file_check("f", a, (64, 96)).status, "PASS")
            self.assertEqual(file_check("f", a, (10, 10)).status, "FAIL")
            self.assertEqual(file_check("f", Path(d) / "none.png").status, "FAIL")
            self.assertEqual(anchor_similarity("an", a, b, (0, 0, 0.6, 0.6)).status, "PASS")
            self.assertEqual(anchor_similarity("an", a, c, (0, 0, 0.6, 0.6)).status, "FAIL")

    def test_ocr_never_silent_pass(self):
        for raw in ("", "   \n\t  "):
            with self.subTest(raw=repr(raw)):
                self.assertEqual(ocr_text_check("o", "x.png", False, ocr=lambda p, value=raw: value).status, "NOT_RUN")
        self.assertEqual(ocr_text_check("o", "x.png", False, ocr=lambda p: "PATCH REVIEW LOTS OF TEXT").status, "FAIL")
        def broken(p):
            raise RuntimeError("no engine")
        self.assertEqual(ocr_text_check("o", "x.png", False, ocr=broken).status, "NOT_RUN")

    def test_judge_is_marked_as_judge(self):
        res = judge_checks("x.png", [A("q", "ok", ch="qa")], lambda p, t: True)
        self.assertEqual(res[0].source, "judge")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.offer = load_offer(EX)
        self.atoms = [A("scene", "A bright kitchen counter"), A("prod", "Product in the foreground"),
                      A("q1", "Product clearly visible", ch="qa", kind="qa_check")]
        self.sent = []

    def provider(self, prompt, refs):
        self.sent.append((prompt, refs))
        return png_bytes()

    def qa(self, path):
        return [file_check("file", path, (64, 96)), frame_content_check("frame_content", path), CheckResult("scene", "PASS", True, "judge"),
                CheckResult("prod", "PASS", True, "judge"), CheckResult("q1", "PASS", True, "judge")]

    def test_pass_path_sends_exact_receipt_text(self):
        with tempfile.TemporaryDirectory() as d:
            st = new_run("run1")
            r, comp, verdict = produce_shot(offer=self.offer, atoms=self.atoms, shot=1, contract_id="c1",
                                            state=st, out_dir=d, image_provider=self.provider, qa_checks=self.qa)
            self.assertEqual(verdict, "PASS")
            self.assertEqual(self.sent[0][0], r.sent_prompt)
            self.assertIn(1, st.shots)
            saved = json.loads(next((Path(d) / "receipts").glob("*.json")).read_text())
            self.assertEqual(saved["sent_prompt"], comp.text)
            self.assertEqual(saved["lost"], "потеряно: нет")

    def test_failed_qa_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            st = new_run("run1")
            qa = lambda p: [CheckResult("frame_content", "PASS", pixels=True), CheckResult("ocr", "FAIL")]
            _, _, verdict = produce_shot(offer=self.offer, atoms=self.atoms, shot=1, contract_id="c1", state=st,
                                         out_dir=d, image_provider=self.provider, qa_checks=qa)
            self.assertEqual(verdict, "FAILED_QA")
            self.assertNotIn(1, st.shots)
            self.assertEqual(this_run_attempts(st)[0]["status"], "FAILED_QA")

    def test_budget_overflow_stops_before_provider_call(self):
        with tempfile.TemporaryDirectory() as d:
            big = [A("huge", "x" * 3700)]
            with self.assertRaises(BudgetError):
                produce_shot(offer=self.offer, atoms=big, shot=1, contract_id="c1", state=new_run("r"),
                             out_dir=d, image_provider=self.provider, qa_checks=self.qa)
            self.assertEqual(self.sent, [])


class NoOfferHardcodeTests(unittest.TestCase):
    def test_engine_source_has_no_example_product_words(self):
        offer = json.loads(EX.read_text())
        words = {w.lower() for w in (offer["name"] + " " + offer["product_type"]).split() if len(w) > 4}
        src = "\n".join(p.read_text().lower() for p in (Path(__file__).resolve().parent.parent / "cme").glob("*.py"))
        for w in words:
            self.assertNotIn(w, src, f"offer word {w!r} leaked into engine code")

    def test_no_unknown_contract_literal(self):
        src = "\n".join(p.read_text() for p in (Path(__file__).resolve().parent.parent / "cme").glob("*.py"))
        self.assertNotIn("unknown-contract", src)


if __name__ == "__main__":
    unittest.main()
