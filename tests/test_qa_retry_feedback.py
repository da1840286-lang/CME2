import json
import tempfile
import unittest
from pathlib import Path

from cme.compiler import compile_image_prompt
from cme.offer import load_offer
from cme.run import produce_shot, qa_retry_feedback
from cme.state import new_run, record_attempt
from tests.helpers import A, png_bytes


class QARetryFeedbackTests(unittest.TestCase):
    def write_failed_receipt(self, receipts_dir, receipt_id, checks):
        receipts_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "receipt_id": receipt_id,
            "overall": "FAILED_QA",
            "checks": checks,
        }
        (receipts_dir / f"{receipt_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def test_required_logo_and_product_identity_failures_become_corrections(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipts_dir = Path(tmp) / "receipts"
            receipt_id = "drec_" + "a" * 24
            self.write_failed_receipt(receipts_dir, receipt_id, [
                {"check_id": "system_no_text_policy", "status": "FAIL", "required": True,
                 "detail": "Readable blue logo visible on vacuum body."},
                {"check_id": "cme_product_identity_ref", "status": "FAIL", "required": True,
                 "detail": "Lacks transparent dust cup and yellow latch."},
                {"check_id": "optional_scene", "status": "FAIL", "required": False,
                 "detail": "Optional angle differs."},
            ])
            state = type("State", (), {"attempts": [
                {"shot": 1, "receipt_id": receipt_id, "status": "FAILED_QA"}
            ]})()
            feedback = qa_retry_feedback(state, receipts_dir, 1)
            self.assertEqual(len(feedback), 2)
            self.assertTrue(any("no logos" in item.lower() for item in feedback))
            self.assertTrue(any("transparent dust cup and yellow latch" in item for item in feedback))
            self.assertFalse(any("Optional angle" in item for item in feedback))

    def test_does_not_apply_stale_failure_after_a_newer_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipts_dir = Path(tmp) / "receipts"
            failed_id = "drec_" + "a" * 24
            self.write_failed_receipt(receipts_dir, failed_id, [
                {"check_id": "system_no_text_policy", "status": "FAIL", "required": True,
                 "detail": "Logo found."}
            ])
            state = type("State", (), {"attempts": [
                {"shot": 1, "receipt_id": failed_id, "status": "FAILED_QA"},
                {"shot": 1, "receipt_id": "drec_" + "b" * 24, "status": "PASS"},
            ]})()
            self.assertEqual(qa_retry_feedback(state, receipts_dir, 1), ())

    def test_missing_or_mismatched_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = type("State", (), {"attempts": [
                {"shot": 1, "receipt_id": "drec_" + "c" * 24, "status": "FAILED_QA"}
            ]})()
            self.assertEqual(qa_retry_feedback(state, Path(tmp) / "receipts", 1), ())

    def test_compiler_preserves_contract_and_adds_required_qa_corrections(self):
        compiled = compile_image_prompt(
            [A("scene_1", "A cordless vacuum cleans car-seat crumbs.")],
            1,
            branding_policy="no_branding",
            retry_feedback=[
                "system_no_text_policy. Remove logos and keep all surfaces unmarked.",
                "cme_product_identity_ref. Match the supplied target reference exactly.",
            ],
        )
        self.assertIn("SYSTEM-OWNED BRANDING POLICY", compiled.text)
        self.assertIn("QA RETRY CORRECTIONS", compiled.text)
        self.assertIn("Remove logos and keep all surfaces unmarked.", compiled.text)
        self.assertIn("A cordless vacuum cleans car-seat crumbs.", compiled.text)

    def test_produce_shot_binds_prior_required_failures_into_actual_sent_prompt(self):
        offer = load_offer("examples/offer_example.json")
        contract_id = "retry-feedback-contract"
        previous = new_run("prior-run", offer_id=offer.offer_id, contract_id=contract_id)
        pass_id = "drec_" + "d" * 24
        fail_id = "drec_" + "e" * 24
        record_attempt(previous, 1, pass_id, "PASS")
        record_attempt(previous, 1, fail_id, "FAILED_QA")

        with tempfile.TemporaryDirectory() as tmp:
            receipts_dir = Path(tmp) / "receipts"
            self.write_failed_receipt(receipts_dir, fail_id, [
                {"check_id": "system_no_text_policy", "status": "FAIL", "required": True,
                 "detail": "Readable logo-like blue symbol visible on the product."},
                {"check_id": "cme_product_identity_ref", "status": "FAIL", "required": True,
                 "detail": "Transparent dust cup and yellow latch are missing."},
            ])
            current = new_run("current-run", previous, offer.offer_id, contract_id)
            sent = []
            def image_provider(prompt, refs):
                sent.append(prompt)
                return png_bytes()
            image_provider.info = {"adapter": "test-free", "tier": "free"}
            produce_shot(
                offer=offer,
                atoms=[A("scene_1", "A cordless vacuum removes crumbs from a car seat.")],
                shot=1, contract_id=contract_id, state=current, out_dir=tmp,
                image_provider=image_provider, qa_checks=lambda path: [],
                attempt=3, max_attempts=3,
            )
            self.assertEqual(len(sent), 1)
            self.assertIn("QA RETRY CORRECTIONS", sent[0])
            self.assertIn("logo-like blue symbol", sent[0])
            self.assertIn("Transparent dust cup and yellow latch are missing", sent[0])
            receipt_files = sorted(receipts_dir.glob("drec_*.json"))
            newest = max(receipt_files, key=lambda path: path.stat().st_mtime)
            receipt_payload = json.loads(newest.read_text(encoding="utf-8"))
            self.assertTrue(receipt_payload.get("qa_retry_feedback"))
            self.assertIn("QA RETRY CORRECTIONS", receipt_payload["sent_prompt"])


if __name__ == "__main__":
    unittest.main()
