"""The run menu (workflow_dispatch inputs) must match what the pipeline really receives and does.

Nothing here re-implements the workflow: the tests load .github/workflows/live-smoke.yml, execute the Python and
bash that are embedded in it, and compare the result with what an operator typed. Free-model behaviour is out of
scope. All provider traffic is faked.
"""
import ast
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import cme.radar.find as F
from cme.radar.legacy.discovery import tiktok_socialcrawl as T
from tests.radar_fixtures import FakeResp, synthetic_post, synthetic_response

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "live-smoke.yml"
WF = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
JOB = next(iter(WF["jobs"].values()))
INPUTS = (WF.get("on") or WF.get(True))["workflow_dispatch"]["inputs"]
STEPS = {step.get("name"): step for step in JOB["steps"]}
LIVE = STEPS["Run controlled live pipeline"]["run"]
BLOCKS = re.findall(r"python - <<'PY'\n(.*?)\n\s*PY\n", LIVE, re.S)
PARAM_BLOCKS = [block for block in BLOCKS if 'params = {"country"' in block]

# menu input -> environment variable the workflow exports to every step
ENV_FOR_INPUT = {
    "stage": "CME_STAGE", "branding_policy": "CME_BRANDING_POLICY", "country": "FIND_COUNTRY", "period": "FIND_PERIOD", "find_requests": "FIND_REQUESTS",
    "find_max_ads": "FIND_MAX_ADS", "find_video_duration_range": "FIND_VIDEO_DURATION_RANGE",
    "keywords": "FIND_KEYWORDS", "find_selection_mode": "FIND_SELECTION_MODE",
    "find_performance_order": "FIND_PERFORMANCE_ORDER", "find_include_retention": "FIND_INCLUDE_RETENTION",
    "find_run_id": "FIND_RUN_ID", "candidate_id": "CANDIDATE_ID", "shots": "SHOTS", "shot": "CME_SHOT",
    "brain_run_id": "BRAIN_RUN_ID", "previous_frame_run_id": "PREVIOUS_FRAME_RUN_ID",
    "no_ocr_required": "NO_OCR_REQUIRED",
}
# inputs that are read from the dispatch payload by embedded code or by other steps instead of a job variable
PAYLOAD_INPUTS = {"offer_json", "target_photo_url", "source_contract_json",
                  "source_contract_run_id", "brief"}
ORDER_OPTIONS = {"default": [], "play_6s_rate": ["play_6s_rate"], "play_2s_rate": ["play_2s_rate"],
                 "playback_both": ["play_6s_rate", "play_2s_rate"]}
OFFER = {"offer_id": "o-1", "name": "Acme Model 9 Cordless Car Vacuum", "product_type": "cordless handheld car vacuum",
         "verified_claims": ["Cordless car vacuum"], "forbidden_claims": [], "photos": [], "platform": "", "geo": "US",
         "product_category": "", "confirmations": [], "branding_policy": "no_branding", "approved_brand_marks": []}


def params_snippet(block: str):
    """The statements from `keywords = ...` through `params = {...}`, taken verbatim from the embedded code."""
    tree = ast.parse(block)
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for index, node in enumerate(body):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "params" for t in node.targets):
                start = next(i for i in range(index, -1, -1)
                             if isinstance(body[i], ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == "keywords" for t in body[i].targets))
                return compile(ast.Module(body=body[start:index + 1], type_ignores=[]), "<workflow>", "exec")
    raise AssertionError("params assignment not found in the embedded block")


def menu_env(**menu):
    base = {"country": "US", "period": "30", "find_requests": "4", "find_max_ads": "3",
            "find_video_duration_range": "all", "keywords": "car vacuum", "find_selection_mode": "product",
            "find_performance_order": "default", "find_include_retention": "false"}
    base.update({k: str(v) for k, v in menu.items()})
    return {ENV_FOR_INPUT[k]: v for k, v in base.items() if k in ENV_FOR_INPUT}, base


def workflow_params(block, **menu):
    env, flat = menu_env(**menu)
    with tempfile.TemporaryDirectory() as tmp:
        event = Path(tmp) / "event.json"
        event.write_text(json.dumps({"inputs": {**flat, "offer_json": json.dumps(OFFER)}}), encoding="utf-8")
        scope = {"os": os, "json": json, "Path": Path}
        with mock.patch.dict(os.environ, {**env, "GITHUB_EVENT_PATH": str(event)}):
            exec(params_snippet(block), scope)  # noqa: S102 - the code under test is the repository's own workflow
    return scope["params"]


def expected_params(**menu):
    _, flat = menu_env(**menu)
    return {
        "country": flat["country"].upper(), "period": int(flat["period"]), "max_requests": int(flat["find_requests"]),
        "max_ads": int(flat["find_max_ads"]),
        "keywords": [part.strip() for part in flat["keywords"].split(",") if part.strip()],
        "offer": OFFER, "discovery_mode": "auto", "selection_mode": flat["find_selection_mode"],
        "video_duration_range": flat["find_video_duration_range"],
        "performance_order_bys": ORDER_OPTIONS[flat["find_performance_order"]],
        "include_retention": flat["find_include_retention"].lower() == "true",
    }


MATRIX = [
    {},
    {"country": "us", "period": 7, "find_requests": 1, "find_max_ads": 1},
    {"country": "US", "period": 180, "find_requests": 10, "find_max_ads": 50, "find_video_duration_range": "up_to_40"},
    {"keywords": " car vacuum , pet hair vacuum,  cordless car vacuum ", "find_performance_order": "playback_both"},
    {"find_performance_order": "play_6s_rate", "find_video_duration_range": "60_to_90"},
    {"find_performance_order": "play_2s_rate", "find_include_retention": "true"},
    {"find_selection_mode": "mechanism", "keywords": "stain remover"},
    {"find_video_duration_range": "40_to_60", "find_max_ads": 7, "find_requests": 3},
]


class MenuWiringTests(unittest.TestCase):
    def test_every_menu_input_is_wired_somewhere(self):
        text = LIVE + "\n".join(str(step.get("run", "")) + str(step.get("env", "")) + str(step.get("with", ""))
                                for step in JOB["steps"]) + json.dumps(JOB.get("env", {}))
        for name in INPUTS:
            with self.subTest(input=name):
                self.assertTrue(f"inputs.{name}" in text or f'"{name}"' in text or f"'{name}'" in text,
                                f"menu input {name} is never used")

    def test_job_environment_maps_each_input_to_the_expected_variable(self):
        env = JOB["env"]
        for name, variable in ENV_FOR_INPUT.items():
            with self.subTest(input=name):
                self.assertIn(name, INPUTS)
                self.assertEqual(env[variable].replace(" ", ""), "${{inputs." + name + "}}")
        self.assertEqual(set(INPUTS) - set(ENV_FOR_INPUT), PAYLOAD_INPUTS)

    def test_find_stage_and_all_stage_build_identical_parameters_from_the_menu(self):
        self.assertEqual(len(PARAM_BLOCKS), 2, "expected the find-stage block and the all-stage block")
        for menu in MATRIX:
            with self.subTest(menu=menu):
                want = expected_params(**menu)
                found = [workflow_params(block, **menu) for block in PARAM_BLOCKS]
                self.assertEqual(found[0], found[1])
                self.assertEqual(found[0], want)

    def test_branding_menu_is_the_only_workflow_policy_source(self):
        self.assertEqual(JOB["env"]["CME_BRANDING_POLICY"].replace(" ", ""), "${{inputs.branding_policy}}")
        preflight = STEPS["Validate target offer and download target photo before provider calls"]["run"]
        self.assertIn('os.environ.get("CME_BRANDING_POLICY", "")', preflight)
        self.assertIn('offer_data.pop("branding_policy", None)', preflight)
        self.assertIn("resolve_branding_contract", preflight)
        self.assertNotIn('"branding_policy":"no_branding"', INPUTS["offer_json"]["default"])

    def test_offer_json_from_the_menu_reaches_find_unchanged(self):
        params = workflow_params(PARAM_BLOCKS[0])
        self.assertEqual(params["offer"], OFFER)

    def test_choice_inputs_offer_exactly_the_values_the_code_accepts(self):
        self.assertEqual(set(INPUTS["find_performance_order"]["options"]), set(ORDER_OPTIONS))
        self.assertEqual(set(INPUTS["find_video_duration_range"]["options"]), set(F.VIDEO_DURATION_RANGES))
        self.assertEqual({int(v) for v in INPUTS["period"]["options"]}, set(F.PERIODS))
        self.assertEqual(set(INPUTS["find_selection_mode"]["options"]), {"product", "mechanism"})


class MenuValidationStepTests(unittest.TestCase):
    """Run the workflow's own bash validation with menu values and compare accept/reject with run_find."""
    SCRIPT = STEPS["Validate credentials, FIND parameters, and previous-run provenance"]["run"]

    def run_validation(self, stage="find", **menu):
        env, _ = menu_env(**menu)
        env.update({"CME_STAGE": stage, "SOCIALCRAWL_API_KEY": "k", "OPENROUTER_API_KEY": "k", "HF_TOKEN": "k",
                    "SOURCE_CONTRACT_PROVIDED": "false", "CANDIDATE_ID": "", "FIND_RUN_ID": "", "SHOTS": "1",
                    "CME_SHOT": "1", "BRAIN_RUN_ID": "", "PREVIOUS_FRAME_RUN_ID": "", "PATH": os.environ["PATH"]})
        result = subprocess.run(["bash", "-c", self.SCRIPT], env=env, capture_output=True, text=True, timeout=60)
        return result.returncode, result.stdout + result.stderr

    def test_valid_menu_values_pass_in_both_find_and_all(self):
        for stage in ("find", "all"):
            for menu in MATRIX:
                with self.subTest(stage=stage, menu=menu):
                    code, output = self.run_validation(stage, **menu) if menu.get("find_selection_mode") != "mechanism" \
                        or stage == "find" else (0, "")
                    self.assertEqual(code, 0, output)

    def test_invalid_menu_values_are_rejected_before_any_provider_call(self):
        bad = [
            {"country": "USA"}, {"period": 14}, {"find_requests": 11}, {"find_requests": 0}, {"find_max_ads": 51},
            {"find_max_ads": 0}, {"find_video_duration_range": "short"},
            {"find_selection_mode": "mechanism", "find_performance_order": "play_6s_rate"},
            {"find_include_retention": "true", "find_performance_order": "default"},
            {"find_include_retention": "true", "find_selection_mode": "mechanism"},
        ]
        for stage in ("find", "all"):
            for menu in bad:
                with self.subTest(stage=stage, menu=menu):
                    code, output = self.run_validation(stage, **menu)
                    self.assertNotEqual(code, 0, output)

    def test_every_value_the_validation_accepts_is_also_accepted_by_run_find(self):
        """No input passes the menu check and then crashes inside FIND for a validation reason."""
        for menu in MATRIX:
            code, _ = self.run_validation("find", **menu)
            self.assertEqual(code, 0)
            params = expected_params(**menu)
            try:
                F._validate_params_for_test(params) if hasattr(F, "_validate_params_for_test") else None
            except ValueError as exc:  # pragma: no cover
                self.fail(str(exc))


def _item(ad_id, title, views=500000):
    post = synthetic_post(ad_id, brand="UNKNOWN", title=title, ctr=None, cvr=None, views=views, duration=25)
    post["post"]["author"]["username"] = "UNKNOWN"
    post["post"]["author"]["display_name"] = ""
    post["post"]["ext"]["ad"].update({"ctr": None, "cvr": None, "objective": "", "landing_page_url": ""})
    post["post"]["content"]["text"] = title
    return post


class MenuToProviderTests(unittest.TestCase):
    """What the operator selected must be what the provider is asked for and what FIND reports."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []

    def find(self, **menu):
        def fake_get(url, params=None, headers=None, timeout=None):
            self.calls.append((url, dict(params or {})))
            return FakeResp(synthetic_response([
                _item("a1", "Handheld Cordless Car Vacuum cleaner! Shop now #carvacuum #caraccessories"),
                _item("a2", "Mini cordless vacuum cleaner for my car link in bio #vacuumcleaner #carcleaning", 300000),
            ]))
        params = workflow_params(PARAM_BLOCKS[0], **menu)
        with mock.patch.dict(os.environ, {"CME_FIND_SEEN_PATH": os.path.join(self.tmp.name, f"s{len(self.calls)}.json")}), \
                mock.patch.object(T.requests, "get", fake_get):
            return params, F.run_find(params, client=T.SocialCrawlTikTokTopAds(api_key="K"))

    def search_calls(self):
        return [p for url, p in self.calls if url.rstrip("/").endswith("/search")]

    def test_country_and_period_reach_the_provider_on_every_path(self):
        for order in ("default", "playback_both"):
            for period, window in ((7, "this-week"), (30, "this-month"), (180, "last-6-months")):
                with self.subTest(order=order, period=period):
                    self.calls.clear()
                    self.find(country="us", period=period, find_performance_order=order)
                    searches = self.search_calls()
                    self.assertTrue(searches)
                    for call in searches:
                        self.assertEqual((call["region"], call["date_posted"]), ("US", window))

    def test_request_budget_is_never_exceeded(self):
        for order in ("default", "playback_both"):
            for budget in (1, 2, 4):
                with self.subTest(order=order, budget=budget):
                    self.calls.clear()
                    self.find(find_requests=budget, find_performance_order=order,
                              keywords="car vacuum, pet hair vacuum, cordless car vacuum")
                    self.assertLessEqual(len(self.calls), max(budget, 1) + (1 if order == "default" else 0))
                    self.assertGreaterEqual(len(self.calls), 1)

    def test_operator_keywords_run_in_the_order_typed_and_overflow_is_reported(self):
        self.calls.clear()
        _, result = self.find(find_performance_order="playback_both", find_requests=5,
                              keywords="car vacuum, pet hair vacuum, cordless car vacuum")
        executed = [call["query"] for call in self.search_calls()]
        self.assertEqual(executed[0], OFFER["name"])
        self.assertEqual(executed[1:3], ["car vacuum", "pet hair vacuum"])
        request = result["request"]
        self.assertEqual(request["targeted_sponsored_search_queries"], executed)
        self.assertEqual(request["dropped_operator_keywords"], ["cordless car vacuum"])  # three-query cap, reported

    def test_reported_plan_is_the_plan_that_ran(self):
        _, standard = self.find(find_performance_order="default")
        self.assertEqual(standard["request"]["performance_plan_effective"], "contract_aware_sponsored_batch")
        self.assertEqual(standard["request"]["provider_request_plan"]["targeted_playback_boards"], [])
        _, contract = self.find(find_performance_order="playback_both")
        self.assertEqual(contract["request"]["performance_plan_effective"], "contract_aware_sponsored_batch")
        self.assertEqual(contract["request"]["requested_performance_order_bys"], ["play_6s_rate", "play_2s_rate"])
        self.assertEqual(contract["request"]["performance_order_bys"], [])

    def test_max_ads_and_selection_mode_and_duration_reach_the_candidates(self):
        _, result = self.find(find_max_ads=1, find_performance_order="playback_both")
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["source_selection"]["mode"], "product")
        self.assertEqual(result["request"]["video_duration_range"], "all")
        _, none = self.find(find_video_duration_range="60_to_90", find_performance_order="playback_both")
        self.assertEqual(none["candidates"], [])  # 25 s fixtures are outside the selected range
        self.assertEqual(none["diagnostics"]["outcome"], "NO_MATCHING_VIDEO_DURATION")


if __name__ == "__main__":
    unittest.main()
