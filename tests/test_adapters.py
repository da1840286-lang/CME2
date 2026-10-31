import base64, json, tempfile, unittest
from pathlib import Path
from cme.adapters.fal_http import make_fal_image
from cme.adapters.http import check_status
from cme.adapters.hf_space import extract_image_path, make_hf_space_image
from cme.adapters.openrouter import ANALYST_SYSTEM
from cme.providers import (CallLog, InfraError, ModelRef, ProviderCallError, ProviderPolicyError,
                           call_with_policy, check_model)
from cme.qa import file_check
from cme.registry import AdapterSpec, Settings, build_adapter, describe, load_catalog, make_image_provider
from cme.run import produce_shot
from cme.smoke import image_seed_for_attempt
from cme.state import new_run
from tests.helpers import A, png_bytes
from cme.offer import load_offer

ROOT = Path(__file__).resolve().parent.parent
PNG = png_bytes()


class AnalystPromptTests(unittest.TestCase):
    def test_radar_analysis_writes_english_for_downstream_transfer_gate(self):
        self.assertIn("Write all text values in English.", ANALYST_SYSTEM)
        self.assertNotIn("Write all text values in Russian.", ANALYST_SYSTEM)


class HTTPStatusDiagnosticTests(unittest.TestCase):
    def test_rate_limit_error_detail_is_preserved_and_api_key_redacted(self):
        secret = "sk-or-v1-" + "a" * 32
        raw = json.dumps({
            "error": {
                "code": 429,
                "message": f"Rate limit exceeded; token={secret}",
                "metadata": {
                    "provider_name": "ExampleProvider",
                    "raw": {"message": "upstream capacity temporarily unavailable"},
                },
            }
        }).encode("utf-8")
        with self.assertRaises(InfraError) as raised:
            check_status("openrouter", 429, raw)
        message = str(raised.exception)
        self.assertIn("openrouter HTTP 429", message)
        self.assertIn("provider_code=429", message)
        self.assertIn("provider_message=Rate limit exceeded", message)
        self.assertIn("provider=ExampleProvider", message)
        self.assertIn("provider_detail=upstream capacity temporarily unavailable", message)
        self.assertNotIn(secret, message)
        self.assertIn("[REDACTED]", message)


class TierPolicyTests(unittest.TestCase):
    def test_free_needs_declaration_and_explicit_allow(self):
        free = ModelRef("hf_space", "some/space", "free")
        with self.assertRaises(ProviderPolicyError):
            check_model(free)
        self.assertEqual(check_model(free, allow_free=True), free)

    def test_mislabeled_and_ambiguous_rejected(self):
        with self.assertRaises(ProviderPolicyError):
            check_model(ModelRef("p", "vendor/model:free", "paid"), allow_free=True)
        with self.assertRaises(ProviderPolicyError):
            check_model(ModelRef("p", "openrouter/free", "free"), allow_free=True)
        with self.assertRaises(ProviderPolicyError):
            check_model(ModelRef("p", "m", "gold"))

    def test_free_fallback_blocked_unless_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            log = CallLog(Path(d) / "l.jsonl")
            calls = []
            def fn(ref):
                calls.append(ref)
                return "x", {}
            with self.assertRaises(ProviderPolicyError):
                call_with_policy(ModelRef("a", "paid-m"), fn, log, "r", fallback=ModelRef("b", "f", "free"))
            self.assertEqual(calls, [])


class HFAdapterTests(unittest.TestCase):
    PARAMS = {"space_id": "s/x", "api_name": "/gen", "width": 576, "height": 1024, "seed": 7,
              "args": [{"from": "prompt"}, {"from": "ref", "index": 0, "default": None},
                       {"from": "seed"}, {"from": "width"}, {"from": "height"}, {"const": True}]}

    def fake_factory(self, predict):
        class C:
            def predict(self, *a, api_name=None):
                return predict(a, api_name)
        return lambda space, token: (C(), lambda p: f"FILE:{p}")

    def test_args_mapping_and_result_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "out.png"
            img.write_bytes(PNG)
            seen = {}
            def predict(a, api):
                seen["a"], seen["api"] = a, api
                return [{"path": str(img)}, "ignored"]
            fn = make_hf_space_image(self.PARAMS, client_factory=self.fake_factory(predict))
            out = fn("a prompt", ["ref.png"])
            self.assertEqual(out, PNG)
            self.assertEqual(seen["a"], ("a prompt", "FILE:ref.png", 7, 576, 1024, True))
            self.assertEqual(seen["api"], "/gen")

    def test_quota_is_infra_other_errors_are_real(self):
        def quota(a, api):
            raise RuntimeError("You have exceeded your GPU quota (ZeroGPU)")
        with self.assertRaises(InfraError):
            make_hf_space_image(self.PARAMS, client_factory=self.fake_factory(quota))("p", [])
        def boom(a, api):
            raise ValueError("bad input shape")
        with self.assertRaises(ProviderCallError):
            make_hf_space_image(self.PARAMS, client_factory=self.fake_factory(boom))("p", [])

    def test_no_file_in_result(self):
        with self.assertRaises(ProviderCallError):
            extract_image_path({"nothing": 1})

    def test_attempt_seed_varies_by_contract_and_attempt_reproducibly(self):
        first = image_seed_for_attempt(0, 1, "contract-123456789abcdef0")
        retry = image_seed_for_attempt(0, 2, "contract-123456789abcdef0")
        other_contract = image_seed_for_attempt(0, 1, "contract-abcdef0123456789")
        self.assertGreater(first, 0)
        self.assertEqual(retry, first + 1)
        self.assertNotEqual(first, other_contract)
        self.assertEqual(first, image_seed_for_attempt(0, 1, "contract-123456789abcdef0"))
        with self.assertRaises(ValueError):
            image_seed_for_attempt(0, 0, "contract-123456789abcdef0")
        with self.assertRaises(ValueError):
            image_seed_for_attempt(True, 2, "contract-123456789abcdef0")
        with self.assertRaises(ValueError):
            image_seed_for_attempt(0, 1, "not-a-contract")

    def test_seed_override_reaches_hf_and_is_written_to_provider_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "out.png"
            img.write_bytes(PNG)
            seen = {}
            def predict(a, api):
                seen["args"], seen["api"] = a, api
                return str(img)
            spec = AdapterSpec("hf", "image", "hf_space", "s/x", "free", self.PARAMS, verified=True)
            provider = make_image_provider(
                Settings(image_adapter="hf", allow_free=True), {"hf": spec},
                CallLog(Path(d) / "calls.jsonl"), "seed-test",
                client_factory=self.fake_factory(predict), image_seed=18)
            provider("prompt", ["ref.png"])
            self.assertEqual(seen["args"][2], 18)
            self.assertEqual(seen["api"], "/gen")
            self.assertEqual(provider.info["seed"], 18)
            self.assertEqual(provider.info["call_status"], "OK")


class FalAdapterTests(unittest.TestCase):
    PARAMS = {"model_path": "vendor/model", "extra": {"image_size": "portrait_16_9"},
              "ref_param": "image_urls", "ref_style": "list", "max_refs": 2}

    def transport(self, post_status=200, calls=None):
        def send(method, url, headers, body):
            if calls is not None:
                calls.append((method, url, headers, body))
            if method == "POST":
                if post_status != 200:
                    return post_status, b"err"
                return 200, json.dumps({"images": [{"url": "https://cdn.example/x.png"}]}).encode()
            return 200, PNG
        return send

    def test_request_shape_and_download(self):
        with tempfile.TemporaryDirectory() as d:
            ref = Path(d) / "r.png"
            ref.write_bytes(PNG)
            calls = []
            fn = make_fal_image(self.PARAMS, api_key="k", transport=self.transport(calls=calls))
            self.assertEqual(fn("prompt text", [str(ref), str(ref), str(ref)]), PNG)
            method, url, headers, body = calls[0]
            self.assertEqual(url, "https://fal.run/vendor/model")
            self.assertEqual(headers["Authorization"], "Key k")
            self.assertEqual(body["prompt"], "prompt text")
            self.assertEqual(body["image_size"], "portrait_16_9")
            self.assertEqual(len(body["image_urls"]), 2)
            self.assertTrue(body["image_urls"][0].startswith("data:image/png;base64,"))
            self.assertEqual(base64.b64decode(body["image_urls"][0].split(",", 1)[1]), PNG)

    def test_status_classification_and_missing_key(self):
        with self.assertRaises(InfraError):
            make_fal_image(self.PARAMS, "k", self.transport(429))("p", [])
        with self.assertRaises(InfraError):
            make_fal_image(self.PARAMS, "k", self.transport(503))("p", [])
        with self.assertRaises(ProviderCallError):
            make_fal_image(self.PARAMS, "k", self.transport(400))("p", [])
        import os
        os.environ.pop("FAL_KEY", None)
        with self.assertRaises(ProviderCallError):
            make_fal_image(self.PARAMS, None, self.transport())("p", [])


class RegistryTests(unittest.TestCase):
    def test_catalog_has_free_and_paid_and_describes_verification(self):
        cat = load_catalog(ROOT / "adapters.json")
        tiers = {s.tier for s in cat.values()}
        self.assertEqual(tiers, {"free", "paid"})
        self.assertIn("NO (confirm", describe(cat))

    def test_unconfigured_placeholder_refuses_to_build(self):
        cat = load_catalog(ROOT / "adapters.json")
        with self.assertRaises(ProviderPolicyError) as cm:
            build_adapter(cat["fal_flux2_pro"])
        self.assertIn("CONFIRM_ME", str(cm.exception))
        with self.assertRaises(ProviderPolicyError) as cm:
            build_adapter(cat["hf_zerogpu_flux2_klein"], allow_free=True)  # filled in, but never confirmed live
        self.assertIn("never confirmed", str(cm.exception))

    def make_catalog(self):
        hf = AdapterSpec("hf", "image", "hf_space", "s/x", "free",
                         {"space_id": "s/x", "api_name": "/g", "args": [{"from": "prompt"}]}, verified=True)
        fal = AdapterSpec("paid", "image", "fal", "paid-model", "paid", {"model_path": "v/m"}, verified=True)
        return {"hf": hf, "paid": fal}

    def test_free_adapter_blocked_without_allow_free(self):
        with self.assertRaises(ProviderPolicyError):
            make_image_provider(Settings("hf"), self.make_catalog(), CallLog("/tmp/x.jsonl"), "r")

    def test_declared_paid_fallback_after_quota(self):
        cat = self.make_catalog()
        with tempfile.TemporaryDirectory() as d:
            log = CallLog(Path(d) / "calls.jsonl")
            def quota(a, api):
                raise RuntimeError("ZeroGPU quota exceeded")
            class C:
                def predict(self, *a, api_name=None):
                    return quota(a, api_name)
            prov = make_image_provider(Settings("hf", "paid", allow_free=True), cat, log, "run1",
                                       client_factory=lambda s, t: (C(), lambda p: p),
                                       fal_key="k",
                                       transport=FalAdapterTests().transport())
            self.assertEqual(prov("p", []), PNG)
            self.assertTrue(prov.info["fallback_used"])
            self.assertEqual(prov.info["tier"], "paid")
            rows = [json.loads(x) for x in (Path(d) / "calls.jsonl").read_text().splitlines()]
            self.assertEqual([r["status"] for r in rows], ["infra_error", "ok"])
            self.assertEqual(rows[1]["tier"], "paid")

    def test_no_fallback_declared_means_quota_stops_the_run(self):
        cat = self.make_catalog()
        with tempfile.TemporaryDirectory() as d:
            class C:
                def predict(self, *a, api_name=None):
                    raise RuntimeError("ZeroGPU quota exceeded")
            prov = make_image_provider(Settings("hf", allow_free=True), cat, CallLog(Path(d) / "c.jsonl"), "r",
                                       client_factory=lambda s, t: (C(), lambda p: p))
            with self.assertRaises(InfraError):
                prov("p", [])

    def test_receipt_report_records_adapter_tier(self):
        cat = self.make_catalog()
        offer = load_offer(ROOT / "examples" / "offer_example.json")
        with tempfile.TemporaryDirectory() as d:
            prov = make_image_provider(Settings("paid"), cat, CallLog(Path(d) / "c.jsonl"), "r1", fal_key="k",
                                       transport=FalAdapterTests().transport())
            produce_shot(offer=offer, atoms=[A("a", "A kitchen counter")], shot=1, contract_id="c",
                         state=new_run("r1"), out_dir=d, image_provider=prov,
                         qa_checks=lambda p: [file_check("file", p)])
            rep = json.loads(next((Path(d) / "receipts").glob("*.json")).read_text())
            self.assertEqual(rep["provider"]["tier"], "paid")
            self.assertEqual(rep["provider"]["adapter"], "paid")


if __name__ == "__main__":
    unittest.main()
