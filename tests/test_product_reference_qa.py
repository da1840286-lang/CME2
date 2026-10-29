"""Offline regression tests for target-product references in generated-frame QA."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from cme.adapters.openrouter import make_openrouter_judge
from cme.offer import load_offer
from cme.qa import (
    PRODUCT_IDENTITY_CHECK_ID,
    CheckResult,
    make_qa_checks,
    validate_product_references,
)
from cme.run import produce_shot
from cme.state import new_run
from tests.helpers import A, png_bytes

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "offer_example.json"


def completion(content, model="vendor/vision-test"):
    return json.dumps({
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"cost": 0, "completion_tokens": 40},
    }).encode()


def varied_png(path: Path):
    im = Image.new("RGB", (128, 128), (238, 238, 238))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((26, 24, 102, 110), radius=12, fill=(42, 78, 115), outline=(20, 20, 20), width=4)
    draw.rectangle((44, 43, 82, 84), fill=(230, 170, 70))
    im.save(path)


class ProductReferenceValidationTests(unittest.TestCase):
    def test_known_flat_color_example_is_rejected(self):
        offer = load_offer(EX)
        errors = validate_product_references(offer.photos)
        self.assertTrue(any("flat-color placeholder" in e for e in errors), errors)

    def test_varied_realistic_reference_is_not_rejected_as_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            ref = Path(d) / "product.png"
            varied_png(ref)
            self.assertEqual(validate_product_references([ref]), [])


class ProductIdentityJudgeTests(unittest.TestCase):
    def test_judge_sends_generated_frame_and_separate_original_product_reference(self):
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "generated.png"
            reference = Path(d) / "target.png"
            generated.write_bytes(png_bytes())
            varied_png(reference)
            seen = {}

            def send(method, url, headers, payload):
                seen["payload"] = payload
                req_text = next(p["text"] for p in payload["messages"][1]["content"] if p["type"] == "text" and p["text"].startswith("["))
                requested = json.loads(req_text)
                return 200, completion(json.dumps({"results": [
                    {"id": row["id"], "pass": True, "evidence": "Visible product shape and details correspond"}
                    for row in requested
                ]}))

            judge = make_openrouter_judge({"model_id": "vendor/vision-test", "judge_mode": "single_pass"}, "key", send)
            answers, _meta = judge(generated, [A("product_main", "Target product is clear", kind="product")], [reference])
            self.assertTrue(answers["product_main"]["pass"])
            self.assertTrue(answers[PRODUCT_IDENTITY_CHECK_ID]["pass"])
            content = seen["payload"]["messages"][1]["content"]
            image_parts = [p for p in content if p.get("type") == "image_url"]
            self.assertEqual(len(image_parts), 2)
            labels = [p["text"] for p in content if p.get("type") == "text"]
            self.assertTrue(any("GENERATED FRAME" in t for t in labels))
            self.assertTrue(any("ORIGINAL TARGET-PRODUCT REFERENCE" in t for t in labels))

    def test_missing_identity_result_is_required_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "generated.png"
            reference = Path(d) / "target.png"
            generated.write_bytes(png_bytes())
            varied_png(reference)
            atom = A("product_main", "Target product is clear", kind="product")

            def judge(_path, _atoms, _refs):
                return {"product_main": {"pass": True, "evidence": "Product visible"}}

            qa = make_qa_checks([atom], 1, judge_fn=judge, ocr=lambda _p: "", product_references=[reference])
            identity = next(x for x in qa(generated) if x.check_id == PRODUCT_IDENTITY_CHECK_ID)
            self.assertEqual(identity.status, "NOT_RUN")
            self.assertTrue(identity.required)

    def test_failed_identity_comparison_is_required_fail(self):
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "generated.png"
            reference = Path(d) / "target.png"
            generated.write_bytes(png_bytes())
            varied_png(reference)
            atom = A("product_main", "Target product is clear", kind="product")

            def judge(_path, _atoms, _refs):
                return {
                    "product_main": {"pass": True, "evidence": "Product visible"},
                    PRODUCT_IDENTITY_CHECK_ID: {"pass": False, "evidence": "Different silhouette and construction"},
                }

            qa = make_qa_checks([atom], 1, judge_fn=judge, ocr=lambda _p: "", product_references=[reference])
            identity = next(x for x in qa(generated) if x.check_id == PRODUCT_IDENTITY_CHECK_ID)
            self.assertEqual(identity.status, "FAIL")
            self.assertTrue(identity.required)

    def test_generation_receipt_records_exact_reference_roles_order_and_hashes(self):
        offer = load_offer(EX)
        atoms = [A("product_main", "Target product is clear", kind="product")]
        with tempfile.TemporaryDirectory() as d:
            previous = Path(d) / "previous.png"
            previous.write_bytes(png_bytes((10, 30, 80)))
            provider_refs = []

            def provider(_prompt, refs):
                provider_refs.extend(refs)
                return png_bytes()

            result, _compiled, _verdict = produce_shot(
                offer=offer,
                atoms=atoms,
                shot=1,
                contract_id="ref-test-contract",
                state=new_run("ref-test-run"),
                out_dir=d,
                image_provider=provider,
                qa_checks=lambda p: [CheckResult("file", "PASS", True, "code")],
                refs=[str(previous)],
            )
            receipt_path = Path(d) / "receipts" / f"{result.receipt_id}.json"
            report = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(provider_refs, [str(previous), *offer.photos])
            manifest = report["image_inputs"]
            self.assertEqual([x["role"] for x in manifest], ["previous_generated_frame", "target_product_photo"])
            self.assertEqual([x["index"] for x in manifest], [1, 2])
            self.assertTrue(all(len(x["sha256"]) == 64 for x in manifest))
            self.assertTrue(all(x["available"] for x in manifest))


if __name__ == "__main__":
    unittest.main()
