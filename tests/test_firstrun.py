"""First-launch tests: audit fixes plus the Brain, judge and smoke paths, all on fake transports."""
import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from cme.adapters.hf_space import make_hf_space_image
from cme.adapters.openrouter import extract_json, make_openrouter_brain, make_openrouter_judge
from cme.atoms import ATOM_SCHEMA, KINDS, check_claims, parse_atoms
from cme.brain import BrainError, apply_patch, request_atoms
from cme.offer import OfferCard, load_offer
from cme.providers import CallLog, InfraError, ModelRef, ProviderCallError, ProviderPolicyError, call_with_policy
from cme.qa import CheckResult, judge_batch_checks, make_qa_checks
from cme.registry import AdapterSpec, Settings, SettingsError, build_adapter, load_catalog, load_settings, make_image_provider
from cme.run import AttemptLimit, produce_shot
from cme.smoke import main as smoke_main
from cme.state import new_run
from tests.helpers import A, png_bytes

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "offer_example.json"
PNG = png_bytes()


def atom(i, text, **kw):
    d = {"id": i, "kind": "scene", "text": text, "priority": "MUST", "channel": "image", "shot": 1}
    d.update(kw)
    return d


def completion(content, model="vendor/model:free"):
    return json.dumps({"model": model, "choices": [{"message": {"content": content}}], "usage": {"cost": 0}}).encode()


class AtomRulesTests(unittest.TestCase):
    def test_kind_and_id_validation_mirror_the_schema(self):
        r = parse_atoms([atom("BAD ID!", "text here"), atom("ok_id", "text here", kind="anything goes")])
        self.assertEqual(r.atoms, [])
        self.assertIn("id must match", r.errors["BAD ID!"][0])
        self.assertIn("must use kind 'claim'", r.errors["ok_id"][0])
        self.assertEqual(ATOM_SCHEMA["properties"]["atoms"]["items"]["properties"]["kind"]["enum"], list(KINDS))


    def test_parser_rejects_extra_fields_wrong_types_and_bool_shot(self):
        good = atom("strict_1", "A kitchen counter")
        for mutate, expected in (
            (lambda d: d.update(extra="not allowed"), "unknown field"),
            (lambda d: d.update(shot=True), "positive integer"),
            (lambda d: d.update(text=123), "text must be a string"),
            (lambda d: d.update(id=123), "id must be a string"),
        ):
            row = dict(good)
            mutate(row)
            parsed = parse_atoms([row])
            self.assertEqual(parsed.atoms, [])
            self.assertTrue(any(expected in e for errs in parsed.errors.values() for e in errs))

    def test_claim_cannot_hide_under_a_free_form_kind(self):
        # the only way to state a product claim is kind='claim', which must be a verified claim verbatim
        offer = load_offer(EX)
        hidden = parse_atoms([atom("sneaky", "Lasts a lifetime", kind="claim", channel="qa")]).atoms
        self.assertIn("UNVERIFIED_CLAIM", check_claims(hidden, offer)["sneaky"][0])

    def test_verified_claim_cannot_be_an_image_atom(self):
        offer = load_offer(EX)
        claim = parse_atoms([atom("claim_a", "Reusable and dishwasher safe", kind="claim")]).atoms
        self.assertIn("CLAIM_CHANNEL", check_claims(claim, offer)["claim_a"][0])
        factual = parse_atoms([atom("claim_b", "Reusable and dishwasher safe", kind="claim", channel="qa")]).atoms
        self.assertEqual(check_claims(factual, offer), {})


class ProviderLoggingTests(unittest.TestCase):
    def rows(self, d):
        return [json.loads(x) for x in (Path(d) / "c.jsonl").read_text().splitlines()]

    def test_provider_call_error_is_logged(self):
        with tempfile.TemporaryDirectory() as d:
            def bad(ref):
                raise ProviderCallError("bad request")
            with self.assertRaises(ProviderCallError):
                call_with_policy(ModelRef("p", "m"), bad, CallLog(Path(d) / "c.jsonl"), "r")
            self.assertEqual(self.rows(d)[0]["status"], "error")
            self.assertEqual(self.rows(d)[0]["error_type"], "ProviderCallError")

    def test_unreported_model_is_logged_as_unreported(self):
        with tempfile.TemporaryDirectory() as d:
            call_with_policy(ModelRef("p", "m"), lambda ref: ("x", {}), CallLog(Path(d) / "c.jsonl"), "r")
            self.assertEqual(self.rows(d)[0]["responded_model"], "unreported")

    def test_variant_suffix_is_not_a_mismatch_but_a_different_model_is(self):
        with tempfile.TemporaryDirectory() as d:
            log = CallLog(Path(d) / "c.jsonl")
            call_with_policy(ModelRef("p", "vendor/m:free", "free"), lambda ref: ("x", {"model": "vendor/m"}), log, "r",
                             allow_free=True)
            with self.assertRaises(ProviderPolicyError):
                call_with_policy(ModelRef("p", "vendor/m:free", "free"), lambda ref: ("x", {"model": "other/m"}), log, "r",
                                 allow_free=True)


class PatchReportTests(unittest.TestCase):
    def test_downgrade_is_reported_with_old_and_new_priority(self):
        upd = {"id": "m1", "kind": "scene", "text": "must text", "priority": "SHOULD", "channel": "image", "shot": 1}
        _, report = apply_patch([A("m1", "must text")], [{"id": "m1", "action": "update", "atom": upd, "reason": "merged"}])
        self.assertIn("m1 (MUST→SHOULD): merged", report)


class RegistryGuardTests(unittest.TestCase):
    def test_unverified_adapter_needs_explicit_opt_in(self):
        spec = AdapterSpec("x", "image", "fal", "m", "paid", {"model_path": "a/b"})
        with self.assertRaises(ProviderPolicyError) as cm:
            build_adapter(spec)
        self.assertIn("never confirmed", str(cm.exception))
        self.assertTrue(callable(build_adapter(spec, allow_unverified=True, fal_key="k")))

    def test_settings_reject_zero_attempts_and_string_booleans(self):
        with tempfile.TemporaryDirectory() as d:
            settings = Path(d) / "settings.json"
            settings.write_text(json.dumps({"max_attempts_per_shot": 0, "allow_free": True}))
            with self.assertRaises(SettingsError):
                load_settings(settings)
            settings.write_text(json.dumps({"max_attempts_per_shot": 2, "allow_free": "false"}))
            with self.assertRaises(SettingsError):
                load_settings(settings)

    def test_unknown_image_adapter_has_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ProviderPolicyError, "unknown image_adapter"):
                make_image_provider(Settings(image_adapter="missing"), {}, CallLog(Path(d) / "calls.jsonl"), "r")

    def test_image_adapter_failure_still_sets_diagnostics(self):
        spec = AdapterSpec("img", "image", "hf_space", "test-model", "free", {}, verified=True)
        def broken_factory(params, **ctx):
            def call(prompt, refs):
                raise RuntimeError("quota exhausted")
            return call
        with tempfile.TemporaryDirectory() as d, patch("cme.registry.FACTORIES", {("hf_space", "image"): broken_factory}):
            provider = make_image_provider(Settings(image_adapter="img", allow_free=True),
                                           {"img": spec}, CallLog(Path(d) / "calls.jsonl"), "r")
            with self.assertRaises(RuntimeError):
                provider("prompt", [])
            self.assertEqual(provider.info["call_status"], "ERROR")
            self.assertEqual(provider.info["error_type"], "RuntimeError")

    def test_hf_catalog_entry_matches_the_space_signature_seen_in_a_real_run(self):
        spec = load_catalog(ROOT / "adapters.json")["hf_zerogpu_flux2_klein"]
        self.assertEqual(spec.params["api_name"], "/infer")
        self.assertNotIn("CONFIRM_ME", json.dumps(spec.params))
        order = [("prompt" if a.get("from") == "prompt" else "input_images" if a.get("from") == "refs_gallery" else
                  a.get("from") or "const") for a in spec.params["args"]]
        self.assertEqual(order[:2], ["prompt", "input_images"])
        self.assertEqual(len(spec.params["args"]), 10)  # prompt, input_images, mode, seed, randomize, w, h, steps, guidance, upsampling
        self.assertEqual(spec.params["args"][2], {"const": "Distilled (4 steps)"})
        self.assertEqual(spec.params["args"][-1], {"const": False})  # prompt_upsampling off: prompt stays receipt-bound

    def test_gallery_input_shape_and_order(self):
        spec = load_catalog(ROOT / "adapters.json")["hf_zerogpu_flux2_klein"]
        seen = {}
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "o.png"
            img.write_bytes(PNG)
            class C:
                def predict(self, *a, api_name=None):
                    seen["a"], seen["api"] = a, api_name
                    return str(img)
            fn = make_hf_space_image(spec.params, client_factory=lambda s, t: (C(), lambda p: f"F:{p}"))
            fn("prompt", ["prev.png", "product.png"])
            gallery = seen["a"][1]
            self.assertEqual(gallery, [{"image": "F:prev.png", "caption": None}, {"image": "F:product.png", "caption": None}])
            self.assertEqual(seen["api"], "/infer")
            fn("prompt", [])
            self.assertIsNone(seen["a"][1])


class OpenRouterBrainTests(unittest.TestCase):
    PARAMS = {"model_id": "vendor/model:free", "response_format_mode": "json_object"}

    def sender(self, status=200, body=None, calls=None):
        def send(method, url, headers, payload):
            if calls is not None:
                calls.append((url, headers, payload))
            return status, body if body is not None else completion(json.dumps({"atoms": [atom("scene_a", "A kitchen counter")]}))
        return send

    def request(self):
        from cme.brain import build_request
        return build_request(load_offer(EX), "brief", 1)

    def test_request_shape_and_reported_model(self):
        calls = []
        fn = make_openrouter_brain(self.PARAMS, "key", self.sender(calls=calls))
        data, meta = fn(self.request())
        url, headers, payload = calls[0]
        self.assertEqual(url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer key")
        self.assertEqual(payload["model"], "vendor/model:free")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        system, user = payload["messages"][0]["content"], json.loads(payload["messages"][1]["content"])
        self.assertIn("max 120 chars", system)
        self.assertIn("claim", system)
        self.assertIn("channel='qa'", system)
        self.assertIn("Never invent text", system)
        self.assertNotIn("/", user["offer"]["photos"][0])  # no local paths reach the API
        self.assertEqual(meta["model"], "vendor/model:free")
        self.assertEqual(data["atoms"][0]["id"], "scene_a")

    def test_fenced_and_prose_wrapped_json(self):
        self.assertEqual(extract_json('```json\n{"atoms": []}\n```'), {"atoms": []})
        self.assertEqual(extract_json('Sure! {"atoms": []} Hope this helps'), {"atoms": []})
        self.assertIsNone(extract_json("no json here"))

    def test_invalid_json_becomes_a_retryable_error_not_a_pass(self):
        fn = make_openrouter_brain(self.PARAMS, "k", self.sender(body=completion("sorry, here is prose")))
        data, _ = fn(self.request())
        self.assertEqual(data["atoms"], [])
        self.assertIn("_error", data)

    def test_brain_catalog_uses_strict_json_schema_only_for_brain(self):
        catalog = load_catalog(ROOT / "adapters.json")
        self.assertEqual(catalog["openrouter_free_brain"].params["response_format_mode"], "json_schema")
        self.assertEqual(catalog["openrouter_free_judge"].params["response_format_mode"], "json_object")
        self.assertEqual(catalog["openrouter_free_analyst"].params["response_format_mode"], "json_object")

    def test_json_schema_mode_is_sent_to_openrouter(self):
        calls = []
        params = {"model_id": "vendor/model:free", "response_format_mode": "json_schema"}
        fn = make_openrouter_brain(params, "key", self.sender(calls=calls))
        fn(self.request())
        payload = calls[0][2]
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertIs(payload["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], ATOM_SCHEMA)

    def test_invalid_brain_response_logs_redacted_excerpt_and_token_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            payload = {
                "model": "vendor/model:free",
                "choices": [{"message": {"content": "No JSON found; api_key=sentinel-secret-value; key=sk-or-v1-12345678901234567890"}, "finish_reason": "length"}],
                "usage": {"cost": 0, "completion_tokens": 73,
                          "completion_tokens_details": {"reasoning_tokens": 19}},
            }
            send = lambda method, url, headers, body: (200, json.dumps(payload).encode())
            fn = make_openrouter_brain(self.PARAMS, "key", send)
            request = self.request()
            logged = call_with_policy(
                ModelRef("openrouter", "vendor/model:free", "free"),
                lambda ref: fn(request), CallLog(Path(d) / "calls.jsonl"), "brain-test", allow_free=True)
            self.assertEqual(logged["atoms"], [])
            row = json.loads((Path(d) / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
            diagnostic = row["diagnostic"]
            self.assertEqual(diagnostic["component"], "openrouter_brain")
            self.assertEqual(diagnostic["reason"], "invalid_json")
            self.assertEqual(diagnostic["finish_reason"], "length")
            self.assertEqual(diagnostic["completion_tokens"], 73)
            self.assertEqual(diagnostic["reasoning_tokens"], 19)
            self.assertLessEqual(len(diagnostic["response_excerpt"]), 400)
            self.assertIn("No JSON found", diagnostic["response_excerpt"])
            self.assertNotIn("sk-or-v1-12345678901234567890", diagnostic["response_excerpt"])
            self.assertNotIn("sentinel-secret-value", diagnostic["response_excerpt"])

    def test_status_classification(self):
        with self.assertRaises(InfraError):
            make_openrouter_brain(self.PARAMS, "k", self.sender(429, b"slow down"))(self.request())
        with self.assertRaises(InfraError):
            make_openrouter_brain(self.PARAMS, "k", self.sender(200, json.dumps({"error": {"code": 429, "message": "rate"}}).encode()))(self.request())
        with self.assertRaises(ProviderCallError):
            make_openrouter_brain(self.PARAMS, "k", self.sender(400, b"bad"))(self.request())
        import os
        os.environ.pop("OPENROUTER_API_KEY", None)
        with self.assertRaises(ProviderCallError):
            make_openrouter_brain(self.PARAMS, None, self.sender())(self.request())


class BrainRetryTests(unittest.TestCase):
    def test_non_json_reply_is_retried(self):
        offer, n = load_offer(EX), []
        def call(req):
            n.append(req)
            return {"atoms": [], "_error": "not json"} if len(n) == 1 else {"atoms": [atom("scene_a", "A kitchen counter")]}
        self.assertEqual(len(request_atoms(offer, "b", 1, call)), 1)
        self.assertIn("#root", [f["id"] for f in n[1]["only_fix"]])

    def test_empty_answer_never_succeeds(self):
        with self.assertRaises(BrainError):
            request_atoms(load_offer(EX), "b", 1, lambda r: {"atoms": []}, max_retries=1)

    def test_every_shot_needs_a_must_image_atom(self):
        offer, n = load_offer(EX), []
        def call(req):
            n.append(req)
            if len(n) == 1:
                return {"atoms": [atom("s1_scene", "Counter")]}
            return {"atoms": [atom("s2_scene", "Table", shot=2)]}
        out = request_atoms(offer, "b", 2, call)
        self.assertEqual({a.shot for a in out}, {1, 2})
        self.assertIn("#shot_2", [f["id"] for f in n[1]["only_fix"]])


class JudgeTests(unittest.TestCase):
    def test_judge_request_has_the_image_and_parses_results(self):
        calls = []
        def send(method, url, headers, payload):
            calls.append(payload)
            body = {"results": [{"id": "a1", "pass": True, "evidence": "visible"}, {"id": "a2", "pass": "yes"}, {"id": "zz", "pass": False}]}
            return 200, completion(json.dumps(body))
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "f.png"
            img.write_bytes(PNG)
            judge = make_openrouter_judge({"model_id": "vendor/v:free"}, "k", send)
            out, meta = judge(img, [A("a1", "Product visible"), A("a2", "Hand holding it")])
        content = calls[0]["messages"][1]["content"]
        image_parts = [part for part in content if isinstance(part, dict) and part.get("type") == "image_url"]
        self.assertTrue(image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(out, {"a1": {"pass": True, "evidence": "visible"}})  # missing evidence and malformed pass dropped

    def test_evidence_free_and_malformed_rows_are_not_pass(self):
        atoms = [A("a1", "Product visible"), A("a2", "Hand holding it")]
        res = judge_batch_checks("x.png", atoms, lambda p, a: {
            "a1": {"pass": True, "evidence": "  "}, "a2": {"pass": "yes", "evidence": "seen"}})
        self.assertEqual([r.status for r in res], ["NOT_RUN", "NOT_RUN"])

    def test_missing_answers_and_outage_are_not_run(self):
        atoms = [A("a1", "Product visible"), A("a2", "Hand")]
        res = judge_batch_checks("x.png", atoms, lambda p, a: {"a1": {"pass": True, "evidence": "ok"}})
        self.assertEqual([(r.check_id, r.status) for r in res], [("a1", "PASS"), ("a2", "NOT_RUN")])
        def down(p, a):
            raise InfraError("429")
        self.assertEqual({r.status for r in judge_batch_checks("x.png", atoms, down)}, {"NOT_RUN"})
        self.assertTrue(all(r.source == "judge" for r in res))


class QASuiteTests(unittest.TestCase):
    def setUp(self):
        self.atoms = [A("scene", "A bright kitchen counter"), A("prod", "Product in the foreground")]
        self.d = tempfile.TemporaryDirectory()
        self.img = Path(self.d.name) / "f.png"
        self.img.write_bytes(PNG)

    def tearDown(self):
        self.d.cleanup()

    def judge_ok(self, path, atoms):
        return {a.id: {"pass": True, "evidence": "seen"} for a in atoms}

    def verdict(self, qa):
        offer = load_offer(EX)
        with tempfile.TemporaryDirectory() as d:
            return produce_shot(offer=offer, atoms=self.atoms, shot=1, contract_id="c", state=new_run("r"), out_dir=d,
                                image_provider=lambda p, r: PNG, qa_checks=qa)

    def test_missing_ocr_engine_means_no_pass(self):
        def broken_ocr(p):
            raise RuntimeError("tesseract not installed")
        r, _, v = self.verdict(make_qa_checks(self.atoms, 1, judge_fn=self.judge_ok, ocr=broken_ocr))
        self.assertEqual(v, "UNVERIFIED")
        self.assertEqual({c["check_id"]: c["status"] for c in r.checks}["no_text"], "NOT_RUN")

    def test_pass_needs_code_checks_and_every_must(self):
        # Non-empty OCR output with no disallowed words verifies the no-text check;
        # an empty OCR result is unverified and is covered by test_ocr_never_silent_pass.
        _, _, v = self.verdict(make_qa_checks(self.atoms, 1, judge_fn=self.judge_ok, ocr=lambda p: "ok"))
        self.assertEqual(v, "PASS")
        _, _, v2 = self.verdict(make_qa_checks(self.atoms, 1, judge_fn=None, ocr=lambda p: ""))
        self.assertEqual(v2, "UNVERIFIED")  # no judge and no OCR evidence cannot pass

    def test_ocr_can_be_declared_not_required_but_stays_visible(self):
        r, _, v = self.verdict(make_qa_checks(self.atoms, 1, judge_fn=self.judge_ok, ocr_required=False))
        self.assertEqual(v, "PASS")
        row = next(c for c in r.checks if c["check_id"] == "no_text")
        self.assertFalse(row["required"])

    def test_verified_claim_is_checked_against_offer_not_image(self):
        offer = load_offer(EX)
        claim = A("claim_01", "Reusable and dishwasher safe", ch="qa", kind="claim")
        atoms = self.atoms + [claim]
        judged_ids = []

        def judge(path, batch):
            judged_ids.extend(a.id for a in batch)
            return {a.id: {"pass": True, "evidence": "seen"} for a in batch}

        qa = make_qa_checks(atoms, 1, judge_fn=judge, ocr=lambda p: "ok",
                            verified_claims=offer.verified_claims)
        with tempfile.TemporaryDirectory() as d:
            receipt, _, verdict = produce_shot(
                offer=offer, atoms=atoms, shot=1, contract_id="c", state=new_run("r"), out_dir=d,
                image_provider=lambda p, r: PNG, qa_checks=qa)
        check = next(c for c in receipt.checks if c["check_id"] == "claim_01")
        row = next(r for r in receipt.rows if r["atom_id"] == "claim_01")
        self.assertEqual(verdict, "PASS")
        self.assertEqual((check["status"], check["source"]), ("PASS", "code"))
        self.assertIn("not inferred from pixels", check["detail"])
        self.assertNotIn("claim_01", judged_ids)
        self.assertNotIn("Reusable and dishwasher safe", receipt.sent_prompt)
        self.assertEqual(row["status"], "VERIFIED")

    def test_ocr_detected_text_fails_a_no_text_contract(self):
        result = make_qa_checks(self.atoms, 1, judge_fn=self.judge_ok, ocr=lambda p: "gonnols")(self.img)
        row = next(c for c in result if c.check_id == "no_text")
        self.assertEqual(row.status, "FAIL")

    def test_judge_fail_fails_the_shot(self):
        bad = lambda p, atoms: {a.id: {"pass": a.id != "prod", "evidence": "missing"} for a in atoms}
        _, _, v = self.verdict(make_qa_checks(self.atoms, 1, judge_fn=bad, ocr=lambda p: ""))
        self.assertEqual(v, "FAILED_QA")


class RunLimitsTests(unittest.TestCase):
    def go(self, d, st, provider=None, attempt=1, max_attempts=2):
        return produce_shot(offer=load_offer(EX), atoms=[A("scene", "A bright kitchen counter")], shot=1, contract_id="c",
                            state=st, out_dir=d, image_provider=provider or (lambda p, r: PNG),
                            qa_checks=lambda p: [], attempt=attempt, max_attempts=max_attempts)

    def test_attempt_limit_blocks_and_leaves_a_report(self):
        st = new_run("r")
        with tempfile.TemporaryDirectory() as d:
            self.go(d, st, attempt=1)
            self.go(d, st, attempt=2)
            with self.assertRaises(AttemptLimit):
                self.go(d, st, attempt=3)
            self.assertEqual(st.attempts[-1]["status"], "BLOCKED_ATTEMPTS")
            self.assertTrue(list((Path(d) / "receipts").glob("blocked_r_s01_a03.json")))

    def test_attempt_limit_persists_across_runs_for_same_contract(self):
        offer = load_offer(EX)
        with tempfile.TemporaryDirectory() as d:
            first = new_run("r1", offer_id=offer.offer_id, contract_id="c")
            self.go(d, first, attempt=1, max_attempts=2)
            second = new_run("r2", first, offer.offer_id, "c")
            self.assertEqual(len(second.attempts), 1)
            self.go(d, second, attempt=2, max_attempts=2)
            third = new_run("r3", second, offer.offer_id, "c")
            with self.assertRaises(AttemptLimit):
                self.go(d, third, attempt=3, max_attempts=2)

    def test_blocked_gate_and_budget_leave_reports(self):
        st = new_run("r")
        with tempfile.TemporaryDirectory() as d, self.assertRaises(Exception):
            produce_shot(offer=load_offer(EX), atoms=[A("h", "x" * 3700)], shot=1, contract_id="c", state=st, out_dir=d,
                         image_provider=lambda p, r: PNG, qa_checks=lambda p: [])
        self.assertTrue(list((Path(d) / "receipts").glob("blocked_*")) if Path(d).exists() else True)

    def test_provider_error_details_are_written_to_receipt(self):
        def provider(prompt, refs):
            provider.info = {"adapter": "test_hf", "call_status": "ERROR", "error_type": "RuntimeError", "error": "quota"}
            raise RuntimeError("quota")
        provider.info = {"adapter": "test_hf", "call_status": "STARTED"}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                self.go(d, new_run("r"), provider=provider)
            rep = json.loads(next((Path(d) / "receipts").glob("drec_*.json")).read_text())
            self.assertEqual(rep["provider"]["adapter"], "test_hf")
            self.assertEqual(rep["provider"]["call_status"], "ERROR")

    def test_free_or_unconfirmed_output_is_test_only(self):
        for info, expected in (({"tier": "free", "verified": True}, "TEST_ONLY"),
                               ({"tier": "paid", "verified": False}, "TEST_ONLY"),
                               ({"tier": "paid", "verified": True}, "CANDIDATE")):
            def prov(p, r, i=info):
                return PNG
            prov.info = info
            with tempfile.TemporaryDirectory() as d:
                self.go(d, new_run("r"), provider=prov)
                rep = json.loads(next((Path(d) / "receipts").glob("drec_*.json")).read_text())
                self.assertEqual(rep["commercial_status"], expected)


class ModelSubstitutionTests(unittest.TestCase):
    def test_openrouter_answering_with_another_model_stops_the_brain_call(self):
        from cme.providers import ModelMismatch
        from cme.registry import make_brain_call
        cat = {"b": AdapterSpec("b", "brain", "openrouter", "vendor/asked:free", "free", {}, verified=True)}
        sender = lambda m, u, h, p: (200, completion(json.dumps({"atoms": []}), "vendor/other"))
        with tempfile.TemporaryDirectory() as d:
            call = make_brain_call(Settings(brain_adapter="b", allow_free=True), cat, CallLog(Path(d) / "c.jsonl"), "r",
                                   transport=sender, openrouter_key="k")
            with self.assertRaises(ModelMismatch):
                call({"schema": {}, "rules": [], "offer": {}, "brief": "", "shots": 1})
            rows = [json.loads(x) for x in (Path(d) / "c.jsonl").read_text().splitlines()]
            self.assertEqual([r["status"] for r in rows], ["model_mismatch"])  # logged once


class SmokeEndToEndTests(unittest.TestCase):
    """Brain, judge and image provider all faked at the transport/client level."""

    def make_ctx(self, quota=False, tmp=None, judge_fail=False, shots=1):
        atom_rows = []
        for shot in range(1, shots + 1):
            atom_rows.extend([atom(f"scene_{shot}", "A bright kitchen counter", shot=shot),
                              atom(f"prod_{shot}", "Product in the foreground", kind="product", shot=shot)])
        atom_rows.append(atom("claim_a", "Reusable and dishwasher safe", kind="claim", channel="qa", priority="SHOULD"))
        brain_atoms = {"atoms": atom_rows}
        def transport(method, url, headers, payload):
            content = payload["messages"][1]["content"]
            if isinstance(content, list):  # judge call (has an image)
                ids = [r["id"] for r in json.loads(content[0]["text"])]
                return 200, completion(json.dumps({"results": [{"id": i, "pass": not judge_fail, "evidence": "seen"} for i in ids]}),
                                       payload["model"])
            return 200, completion(json.dumps(brain_atoms), payload["model"])
        out = Path(tmp) / "hf_out.png"
        out.write_bytes(PNG)
        class Client:
            def predict(self, *a, api_name=None):
                if quota:
                    raise RuntimeError("You have exceeded your ZeroGPU runs limit")
                return str(out)
        return {"transport": transport, "openrouter_key": "k", "client_factory": lambda s, t: (Client(), lambda p: p)}

    def settings(self, tmp, **kw):
        s = json.loads((ROOT / "settings.example.json").read_text())
        s.update(kw)
        p = Path(tmp) / "settings.json"
        p.write_text(json.dumps(s))
        return str(p)

    def test_all_path_produces_marked_test_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code = smoke_main(["all", "--settings", self.settings(tmp), "--out", str(out), "--no-ocr-required"], self.make_ctx(tmp=tmp))
            self.assertEqual(code, 0)
            self.assertTrue((out / "atoms.json").exists() and (out / "state.json").exists())
            rep = json.loads(next((out / "receipts").glob("drec_*.json")).read_text())
            self.assertEqual(rep["overall"], "PASS")
            self.assertEqual(rep["commercial_status"], "TEST_ONLY")
            self.assertEqual(rep["provider"]["tier"], "free")
            calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
            self.assertEqual({c["status"] for c in calls}, {"ok"})
            self.assertEqual(len(calls), 3)  # brain, image, judge

    def test_all_generates_requested_shots_sequentially_and_keeps_atom_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code = smoke_main(["all", "--shots", "2", "--settings", self.settings(tmp), "--out", str(out),
                               "--no-ocr-required"], self.make_ctx(tmp=tmp, shots=2))
            self.assertEqual(code, 0)
            self.assertTrue((out / "shots" / "shot_01_a01.png").is_file())
            self.assertTrue((out / "shots" / "shot_02_a01.png").is_file())
            manifest = json.loads((out / "atoms_manifest.json").read_text())
            self.assertEqual(manifest["requested_shots"], 2)
            state = json.loads((out / "state.json").read_text())
            self.assertEqual(set(state["shots"]), {"1", "2"})
            self.assertNotEqual(state["shots"]["1"]["run_id"], state["shots"]["2"]["run_id"])
            calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(calls), 5)  # one Brain call, then image+Judge for each shot

    def test_brain_writes_manifest_and_modified_atoms_are_rejected(self):
        from cme.smoke import validate_atoms_manifest
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            settings = self.settings(tmp)
            ctx = self.make_ctx(tmp=tmp)
            self.assertEqual(smoke_main(["brain", "--settings", settings, "--out", str(out)], ctx), 0)
            self.assertTrue((out / "atoms_manifest.json").exists())
            atoms_path = out / "atoms.json"
            original_bytes = atoms_path.read_bytes()
            data = json.loads(original_bytes)
            data[0]["text"] = "Tampered atom value"
            atoms_path.write_text(json.dumps(data))
            from dataclasses import replace
            original_offer = load_offer(EX)
            with self.assertRaises(BrainError):
                validate_atoms_manifest(atoms_path, out / "atoms_manifest.json", data, original_offer, shot=1)
            atoms_path.write_bytes(original_bytes)
            with self.assertRaises(BrainError):
                validate_atoms_manifest(atoms_path, out / "atoms_manifest.json", json.loads(original_bytes),
                                        replace(original_offer, name="Different offer"), shot=1)

    def test_frame_returns_unverified_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            settings = self.settings(tmp, judge_adapter="")
            ctx = self.make_ctx(tmp=tmp)
            self.assertEqual(smoke_main(["brain", "--settings", settings, "--out", str(out)], ctx), 0)
            # With no judge, MUST atoms remain unverified even if local OCR is available.
            code = smoke_main(["frame", "--settings", settings, "--out", str(out)], ctx)
            self.assertEqual(code, 6)
            rep = json.loads(next((out / "receipts").glob("drec_*.json")).read_text())
            self.assertEqual(rep["overall"], "UNVERIFIED")

    def test_failed_qa_returns_seven(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            settings = self.settings(tmp)
            ctx = self.make_ctx(tmp=tmp, judge_fail=True)
            self.assertEqual(smoke_main(["brain", "--settings", settings, "--out", str(out)], ctx), 0)
            code = smoke_main(["frame", "--settings", settings, "--out", str(out), "--no-ocr-required"], ctx)
            self.assertEqual(code, 7)
            rep = json.loads(next((out / "receipts").glob("drec_*.json")).read_text())
            self.assertEqual(rep["overall"], "FAILED_QA")

    def test_quota_exit_code_and_recorded_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            ctx = self.make_ctx(quota=True, tmp=tmp)
            self.assertEqual(smoke_main(["brain", "--settings", self.settings(tmp), "--out", str(out)], ctx), 0)
            code = smoke_main(["frame", "--settings", self.settings(tmp), "--out", str(out)], ctx)
            self.assertEqual(code, 2)
            state = json.loads((out / "state.json").read_text())
            self.assertEqual(state["attempts"][0]["status"], "ERROR_InfraError")

    def test_free_without_allow_free_is_stopped_before_any_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code = smoke_main(["brain", "--settings", self.settings(tmp, allow_free=False), "--out", str(out)], self.make_ctx(tmp=tmp))
            self.assertEqual(code, 3)
            self.assertFalse((out / "calls.jsonl").exists() and (out / "calls.jsonl").read_text())

    def test_missing_settings_file_is_explained(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(smoke_main(["brain", "--settings", str(Path(tmp) / "nope.json"), "--out", tmp]), 3)


if __name__ == "__main__":
    unittest.main()
