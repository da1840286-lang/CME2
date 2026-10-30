"""Creative Radar acceptance tests. All data here is SYNTHETIC and lives only in temporary folders."""
import hashlib, http.client, json, os, subprocess, tempfile, threading, unittest
from pathlib import Path
from unittest import mock

import cme.radar.find as F
from cme.radar import analyze as A
from cme.radar.legacy import discover as legacy_discover
from cme.radar.legacy.discovery import tiktok_socialcrawl as T
from cme.radar.server import RadarApp, make_server
from cme.radar.select import select_candidate
from cme.radar.store import Store
from tests.helpers import png_bytes
from tests.radar_fixtures import FakeResp, synthetic_post, synthetic_response

ROOT = Path(__file__).resolve().parent.parent

def configured_analyst_model():
    """Keep expectations aligned with the explicitly configured Creative Radar adapter."""
    from cme.registry import load_catalog
    settings = json.loads((ROOT / "settings.example.json").read_text(encoding="utf-8"))
    return load_catalog(ROOT / "adapters.json")[settings["analyst_adapter"]].model_id


HAS_FFMPEG = A.tools_status()["state"] == "READY"


class Env(unittest.TestCase):
    """No external keys, no home-directory writes: offline tests need nothing from the outside."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"HOME": self.tmp.name}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in ("SOCIALCRAWL_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"):
            os.environ.pop(k, None)
        self.data = Path(self.tmp.name) / "radar_data"

    def fake_find(self, items, calls=None):
        def fake_get(url, params=None, headers=None, timeout=None):
            if calls is not None:
                calls.append((url, dict(params or {})))
            return FakeResp(synthetic_response(items))
        return mock.patch.object(T.requests, "get", fake_get)


class VendoredCodeIsTheRealOne(Env):
    def test_vendored_files_match_manifest_hashes(self):
        man = json.loads((ROOT / "cme/radar/legacy/MANIFEST.json").read_text())
        self.assertTrue(man["files"])
        for rel, digest in man["files"].items():
            self.assertEqual(hashlib.sha256((ROOT / "cme/radar" / rel).read_bytes()).hexdigest(), digest, f"{rel} was edited")

    def test_find_uses_the_old_functions_not_a_stub(self):
        self.assertIs(F.discover_and_rank, legacy_discover.discover_and_rank)
        self.assertIs(F.SocialCrawlTikTokTopAds, T.SocialCrawlTikTokTopAds)
        self.assertEqual(F.LEGACY_ERROR, "")

    def test_socialcrawl_preserves_zero_playback_rate(self):
        item = synthetic_post("zero-playback", ctr=0, cvr=0, views=1000)
        item["post"]["ext"]["ad"]["play_2s_rate"] = 0
        item["computed"]["play_6s_rate"] = "0.42"
        record = T.SocialCrawlTikTokTopAds._unwrap_item(item)
        self.assertEqual(record["play_2s_rate"], 0)
        self.assertEqual(record["play_6s_rate"], "0.42")

    def test_merge_item_preserves_playback_metrics_when_duplicate_arrives_from_ads_top(self):
        combined = {
            "same-ad": {
                "ad_id": "same-ad",
                "play_2s_rate": None,
                "play_6s_rate": None,
                "discovery_boards": ["sponsored_search"],
                "discovery_keywords": ["car vacuum"],
                "discovery_hits": 1,
            }
        }
        incoming = {"ad_id": "same-ad", "play_2s_rate": 0, "play_6s_rate": "0.42"}

        merged = legacy_discover._merge_item(combined, incoming, "play_6s_rate", "car vacuum")

        self.assertFalse(merged)  # duplicate updated in-place, not added as a new ad
        self.assertEqual(combined["same-ad"]["play_2s_rate"], 0)  # measured zero is data
        self.assertEqual(combined["same-ad"]["play_6s_rate"], "0.42")
        self.assertIn("play_6s_rate", combined["same-ad"]["discovery_boards"])

    def test_query_offer_conflict_stops_before_provider_call(self):
        params = {"country": "US", "max_requests": 2, "keywords": ["portable tire inflator"],
                  "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                            "product_type": "cordless handheld car vacuum"}}
        calls = []
        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append((url, params))
            return FakeResp(synthetic_response([]))
        with mock.patch.object(T.requests, "get", fake_get):
            result = F.run_find(params, client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"))
        self.assertEqual(result["diagnostics"]["outcome"], "INPUT_CONFLICT")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["diagnostics"]["requests_executed"], 0)
        self.assertEqual(calls, [])

    def test_query_matching_offer_and_generic_keyword_do_not_conflict(self):
        self.assertIsNone(F._query_offer_conflict(["car vacuum"], {
            "product_type": "cordless handheld car vacuum", "product_category": ""}))
        self.assertIsNone(F._query_offer_conflict(["kitchen gadget"], {
            "product_type": "portable blender", "product_category": ""}))

    def test_playback_rates_score_against_a_comparison_population(self):
        from cme.radar.legacy.arbitrage import _performance_signal
        population = [
            {"play_2s_rate": 0.67, "play_6s_rate": 0.32},
            {"play_2s_rate": 0.28, "play_6s_rate": 0.14},
            {"play_2s_rate": 0.45, "play_6s_rate": 0.21},
        ]
        _, components, _ = _performance_signal(population[0], population)
        self.assertIn("play_2s_rate", components)
        self.assertIn("play_6s_rate", components)

    def test_run_find_calls_the_real_client_and_ranker(self):
        calls = []
        item = synthetic_post("1001", title="Shop now: cordless car vacuum removes crumbs")
        item["post"]["ext"]["ad"]["play_2s_rate"] = 0.67
        item["computed"]["play_6s_rate"] = 0.32
        comparison = synthetic_post("1002", title="Shop now: handheld car vacuum cleans seat crevices")
        comparison["post"]["ext"]["ad"]["play_2s_rate"] = 0.31
        comparison["computed"]["play_6s_rate"] = 0.18
        with self.fake_find([item, comparison], calls):
            res = F.run_find({"country": "us", "max_requests": 2, "keywords": ["car vacuum"], "discovery_mode": "strict",
                              "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                                        "product_type": "cordless handheld car vacuum"}},
                             client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"))
        self.assertTrue(calls and "socialcrawl" in calls[0][0] and len(calls) <= 2)
        self.assertEqual(calls[0][0], T.SocialCrawlTikTokTopAds.SEARCH_URL)
        self.assertEqual(calls[0][1].get("region"), "US")
        self.assertEqual(calls[0][1].get("query"), "car vacuum")
        sc = res["candidates"][0]["source_score"]
        self.assertIsNotNone(sc["final_score"])  # produced by the old arbitrage.score_tiktok_candidate
        self.assertEqual(res["request"]["vertical"], "Physical Products")
        self.assertEqual(res["request"]["search_terms"], ["car vacuum", "Fanttik V8 Mate Cordless Car Vacuum", "cordless handheld car vacuum"])
        self.assertTrue(res["stats"]["discovery_strategy"]["supply_search_enabled"])
        candidate = res["candidates"][0]
        self.assertEqual(candidate["metrics"]["views"]["source"], "socialcrawl:tiktok_sponsored_search")
        self.assertEqual(candidate["metrics"]["play_2s_rate"]["value"], 0.67)
        self.assertEqual(candidate["metrics"]["play_6s_rate"]["value"], 0.32)
        self.assertIn("play_2s_rate", candidate["commercial_evidence"]["available_metrics"])
        self.assertIn("play_6s_rate", candidate["commercial_evidence"]["available_metrics"])
        self.assertEqual(candidate["commercial_evidence"]["level"], "COMMERCIAL_PROXY")

    def test_find_can_target_playback_boards_and_report_provider_metric_presence(self):
        calls = []
        item = synthetic_post("playback-board-probe", title="Shop now: cordless car vacuum removes crumbs", ctr=None, cvr=None, views=1000)
        item["post"]["ext"]["ad"]["play_2s_rate"] = 0.67
        item["computed"]["play_6s_rate"] = "0.32"
        offer = {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                 "product_type": "cordless handheld car vacuum"}

        with self.fake_find([item], calls):
            result = F.run_find(
                {"country": "US", "max_requests": 4, "max_ads": 20, "keywords": ["car vacuum"],
                 "performance_order_bys": ["play_6s_rate", "play_2s_rate"], "offer": offer,
                 "performance_plan": "legacy_ads_top"},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )

        top_calls = [params for url, params in calls if url == T.SocialCrawlTikTokTopAds.BASE_URL]
        self.assertEqual([params.get("order_by") for params in top_calls],
                         ["play_6s_rate", "play_2s_rate"])
        self.assertEqual(result["request"]["performance_order_bys"], ["play_6s_rate", "play_2s_rate"])
        top_stats = [call for call in result["stats"]["calls"] if call["source"] == "ads_top"]
        self.assertEqual([call["order_by"] for call in top_stats], ["play_6s_rate", "play_2s_rate"])
        for call in top_stats:
            self.assertEqual(call["playback_metrics_returned"],
                             {"play_2s_rate": 1, "play_6s_rate": 1})
            self.assertTrue(call["playback_metric_examples"])
            raw_diag = call["raw_playback_field_diagnostics"]
            self.assertGreater(raw_diag["matched_field_occurrences"], 0)
            matched_paths = " ".join(raw_diag["matched_field_paths"])
            self.assertIn("play_2s_rate", matched_paths)
            self.assertIn("play_6s_rate", matched_paths)
            self.assertTrue(call["provider_params"].get("order_by") in {"play_6s_rate", "play_2s_rate"})

    def test_dual_playback_retention_runs_both_target_searches_and_only_requested_boards(self):
        calls = []
        item = synthetic_post(
            "dual-board-target", brand="UNKNOWN",
            title="Shop now: cordless car vacuum removes crumbs",
            ctr=None, cvr=None, views=311000, duration=28,
        )
        item["post"]["author"]["username"] = "UNKNOWN"
        item["post"]["author"]["display_name"] = ""
        item["post"]["ext"]["ad"].update({
            "ctr": None, "cvr": None, "objective": "", "landing_page_url": "",
        })
        item["post"]["content"]["text"] = "Shop now: cordless car vacuum removes crumbs"

        with self.fake_find([item], calls):
            result = F.run_find(
                {"country": "US", "period": 30, "max_requests": 5, "max_ads": 10,
                 "keywords": ["car vacuum"],
                 "performance_order_bys": ["play_6s_rate", "play_2s_rate"],
                 "include_retention": True, "performance_plan": "legacy_ads_top",
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )

        search_calls = [(url, params) for url, params in calls if url == T.SocialCrawlTikTokTopAds.SEARCH_URL]
        top_calls = [(url, params) for url, params in calls if url == T.SocialCrawlTikTokTopAds.BASE_URL]
        self.assertEqual(len(search_calls), 2)
        self.assertEqual(len({params.get("query") for _, params in search_calls}), 2)
        self.assertEqual(
            [params.get("order_by") for _, params in top_calls],
            ["play_6s_rate", "play_2s_rate", "play_6s_rate"],
        )
        self.assertNotIn("ctr", [params.get("order_by") for _, params in top_calls])
        self.assertNotIn("cvr", [params.get("order_by") for _, params in top_calls])
        self.assertTrue(top_calls[0][1].get("keyword"))
        self.assertTrue(top_calls[1][1].get("keyword"))
        self.assertEqual(top_calls[2][1].get("include"), "retention")
        self.assertNotIn("keyword", top_calls[2][1])
        self.assertNotIn("industry", top_calls[2][1])

        self.assertEqual(result["stats"]["executed_requests"], 5)
        self.assertEqual(
            [call["source"] for call in result["stats"]["calls"]],
            ["sponsored_search", "sponsored_search", "ads_top", "ads_top", "ads_top"],
        )
        self.assertEqual(
            [call["order_by"] for call in result["stats"]["calls"] if call["source"] == "ads_top"],
            ["play_6s_rate", "play_2s_rate", "play_6s_rate"],
        )
        strategy = result["stats"]["discovery_strategy"]
        self.assertEqual(strategy["supply_search_budget"], 2)
        self.assertEqual(strategy["performance_budget"], 2)
        self.assertEqual(
            strategy["supply_queries_executed"],
            [params["query"] for _, params in search_calls],
        )
        self.assertFalse(result["request"]["provider_request_plan"]["unrequested_performance_fallbacks"])


    def test_find_uses_product_terms_and_reports_raw_rows_rejected_by_source_gate(self):
        calls = []
        def fake_get(url, params=None, headers=None, timeout=None):
            params = dict(params or {})
            calls.append((url, params))
            # Intentionally neutral synthetic source: usable media but no shop/product evidence.
            item = synthetic_post(str(7100 + len(calls)), title="[SYNTHETIC TEST] neutral lifestyle clip")
            item["post"]["ext"]["ad"]["landing_page_url"] = "https://example.invalid/page"
            item["post"]["ext"]["ad"]["landing_page"] = ""
            item["post"]["ext"]["ad"]["industry"] = "Entertainment"
            return FakeResp(synthetic_response([item]))
        with mock.patch.object(T.requests, "get", fake_get):
            res = F.run_find({"performance_plan": "legacy_ads_top", "country": "US", "max_requests": 3, "keywords": ["car vacuum"],
                              "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                                        "product_type": "cordless handheld car vacuum"}},
                             client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"))
        queries = [p["query"] for url, p in calls if url == T.SocialCrawlTikTokTopAds.SEARCH_URL]
        self.assertEqual(queries, ["car vacuum", "Fanttik V8 Mate Cordless Car Vacuum"])
        self.assertEqual(len(calls), 3)
        self.assertEqual([url for url, _ in calls], [
            T.SocialCrawlTikTokTopAds.SEARCH_URL,
            T.SocialCrawlTikTokTopAds.SEARCH_URL,
            T.SocialCrawlTikTokTopAds.BASE_URL,
        ])
        self.assertEqual(calls[2][1].get("order_by"), "ctr")
        self.assertEqual(res["candidates"], [])  # no weakening of the target-relevance gate
        self.assertEqual(res["diagnostics"]["outcome"], "NO_TARGET_RELEVANCE")
        self.assertEqual(res["diagnostics"]["raw_ads_returned"], 3)
        self.assertEqual(res["diagnostics"]["requests_executed"], 3)

    def test_target_relevance_without_money_gate_still_uses_reserved_ads_top(self):
        calls = []
        def fake_get(url, params=None, headers=None, timeout=None):
            params = dict(params or {})
            calls.append((url, params))
            item = synthetic_post(
                str(9200 + len(calls)), brand="UNKNOWN",
                title="Shop now free shipping: cordless car vacuum removes crumbs",
                ctr=None, cvr=None, views=3110000, duration=28,
            )
            item["post"]["author"]["username"] = "UNKNOWN"
            item["post"]["author"]["display_name"] = ""
            ad = item["post"]["ext"]["ad"]
            ad["ctr"] = None
            ad["cvr"] = None
            ad["objective"] = ""
            ad["landing_page_url"] = ""
            item["post"]["content"]["text"] = "Shop now free shipping: cordless car vacuum removes crumbs"
            return FakeResp(synthetic_response([item]))
        with mock.patch.object(T.requests, "get", fake_get):
            result = F.run_find(
                {"performance_plan": "legacy_ads_top", "country": "US", "max_requests": 2, "max_ads": 1, "keywords": ["car vacuum"],
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], T.SocialCrawlTikTokTopAds.SEARCH_URL)
        self.assertEqual(calls[1][0], T.SocialCrawlTikTokTopAds.BASE_URL)
        self.assertEqual(calls[1][1].get("order_by"), "ctr")
        self.assertEqual(calls[1][1].get("keyword"), "car vacuum")
        self.assertEqual([params.get("query") for url, params in calls
                          if url == T.SocialCrawlTikTokTopAds.SEARCH_URL], ["car vacuum"])
        self.assertEqual(result["stats"]["money_ready_source_ads"], 0)
        self.assertEqual(result["stats"]["mechanism_ready_source_ads"], 0)
        self.assertGreaterEqual(result["stats"]["target_relevant_ads"], 1)
        self.assertEqual([call.get("source") for call in result["stats"]["calls"]],
                         ["sponsored_search", "ads_top"])
        strategy = result["stats"]["discovery_strategy"]
        self.assertEqual(strategy["supply_search_budget"], 1)
        self.assertEqual(strategy["performance_budget"], 1)

    def test_find_shortlists_product_relevant_performance_sources_without_claiming_money(self):
        calls = []
        target = synthetic_post(
            "vacuum-target", brand="UNKNOWN",
            title="Shop now free shipping: cordless car vacuum removes crumbs",
            ctr=None, cvr=None, views=3110000, duration=28,
        )
        target["post"]["author"]["username"] = "UNKNOWN"
        target["post"]["author"]["display_name"] = ""
        target["post"]["ext"]["ad"].update({
            "ctr": None, "cvr": None, "objective": "", "landing_page_url": "",
        })
        target["post"]["content"]["text"] = "Shop now free shipping: cordless car vacuum removes crumbs"

        distractor = synthetic_post(
            "unrelated-gift", brand="UNKNOWN", title="Gifts they will actually use",
            ctr=None, cvr=None, views=5000, duration=28,
        )
        distractor["post"]["author"]["username"] = "UNKNOWN"
        distractor["post"]["author"]["display_name"] = ""
        distractor["post"]["ext"]["ad"].update({
            "ctr": None, "cvr": None, "objective": "", "landing_page_url": "",
        })

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append((url, dict(params or {})))
            return FakeResp(synthetic_response([target, distractor]))

        with mock.patch.object(T.requests, "get", fake_get):
            result = F.run_find(
                {"performance_plan": "legacy_ads_top", "country": "US", "max_requests": 10, "max_ads": 10, "keywords": ["car vacuum"],
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )

        sources = [call.get("source") for call in result["stats"]["calls"]]
        self.assertEqual(sources.count("sponsored_search"), 7)
        self.assertEqual(sources.count("ads_top"), 3)
        top_calls = [params for url, params in calls if url == T.SocialCrawlTikTokTopAds.BASE_URL]
        self.assertEqual([params.get("order_by") for params in top_calls],
                         ["ctr", "cvr", "play_6s_rate"])
        top_keywords = [params.get("keyword") for params in top_calls]
        self.assertTrue(all(top_keywords))
        self.assertEqual(len(set(top_keywords)), 3)
        self.assertTrue(set(top_keywords).issubset(set(result["request"]["search_terms"])))
        self.assertEqual(result["stats"]["money_ready_source_ads"], 0)
        self.assertEqual(result["stats"]["mechanism_ready_source_ads"], 0)
        self.assertEqual(result["stats"]["discovery_strategy"]["supply_search_budget"], 7)
        self.assertEqual(result["stats"]["discovery_strategy"]["performance_budget"], 3)
        self.assertEqual([c["ad_id"] for c in result["candidates"]], ["vacuum-target"])
        self.assertEqual(result["diagnostics"]["outcome"], "CANDIDATES_FOUND")
        selection_diag = result["diagnostics"]["source_selection_diagnostics"]
        self.assertGreater(selection_diag["not_selected_count"], 0)
        self.assertGreater(selection_diag["selected_by_basis"].get("ADS_TOP_PERFORMANCE_BOARD", 0), 0)
        self.assertIn("PRODUCT_TYPE_MISMATCH", selection_diag["by_reason"])
        candidate = result["candidates"][0]
        self.assertFalse(candidate["source_score"]["components"]["money_signal_gate"]["pass"])
        self.assertEqual(candidate["performance_evidence"]["basis"], "ADS_TOP_PERFORMANCE_BOARD")
        self.assertFalse(candidate["performance_evidence"]["commercial_result_confirmed"])


    def test_ads_top_money_ready_candidate_is_kept_without_weak_backfill(self):
        calls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            params = dict(params or {})
            calls.append((url, params))
            if url == T.SocialCrawlTikTokTopAds.SEARCH_URL:
                item = synthetic_post(
                    "weak-sponsored", brand="UNKNOWN",
                    title="Shop now free shipping: cordless car vacuum removes crumbs",
                    ctr=None, cvr=None, views=50000, duration=28,
                )
                item["post"]["author"]["username"] = "UNKNOWN"
                item["post"]["author"]["display_name"] = ""
                item["post"]["ext"]["ad"].update({
                    "ctr": None, "cvr": None, "objective": "", "landing_page_url": "",
                })
            else:
                item = synthetic_post(
                    "strong-top", brand="Fanttik",
                    title="Shop now: cordless car vacuum removes crumbs",
                    ctr=2.4, cvr=1.1, views=3110000, duration=28,
                )
                item["post"]["ext"]["ad"].update({
                    "ctr": 2.4, "cvr": 1.1, "rank": 1, "objective": "Conversions",
                    "landing_page_url": "https://example.invalid/products/car-vacuum",
                    "industry": "Home & Garden",
                })
            return FakeResp(synthetic_response([item]))

        with mock.patch.object(T.requests, "get", fake_get):
            result = F.run_find(
                {"performance_plan": "legacy_ads_top", "country": "US", "max_requests": 2, "max_ads": 10, "keywords": ["car vacuum"],
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual([call.get("source") for call in result["stats"]["calls"]],
                         ["sponsored_search", "ads_top"])
        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["ad_id"], "strong-top")
        self.assertTrue(candidate["source_score"]["components"]["money_signal_gate"]["pass"])
        self.assertEqual(candidate["legacy_record"]["discovery_source"], "ads_top")
        self.assertEqual(result["stats"]["final_ranked_ads"], 1)

    def test_unknown_source_relevance_is_held_for_review_not_auto_selected(self):
        calls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            params = dict(params or {})
            calls.append((url, params))
            item = synthetic_post(
                "unknown-" + str(len(calls)), brand="UNKNOWN",
                title="A surprising video with a reveal",
                ctr=2.4, cvr=1.1, views=3110000, duration=28,
            )
            item["post"]["author"]["username"] = "UNKNOWN"
            item["post"]["author"]["display_name"] = ""
            item["post"]["content"]["text"] = "A surprising video with a reveal"
            item["post"]["ext"]["ad"].update({
                "ctr": 2.4, "cvr": 1.1, "rank": 1, "objective": "Conversions",
                "landing_page_url": "", "industry": "Entertainment",
            })
            return FakeResp(synthetic_response([item]))

        with mock.patch.object(T.requests, "get", fake_get):
            result = F.run_find(
                {"country": "US", "max_requests": 2, "max_ads": 10, "keywords": ["car vacuum"],
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["diagnostics"]["outcome"], "NO_TARGET_RELEVANCE")
        by_reason = result["diagnostics"]["source_selection_diagnostics"]["by_reason"]
        self.assertIn("OFF_TOPIC", by_reason)



    def test_auto_shortlists_high_engagement_exact_product_as_proxy_not_sale(self):
        target = synthetic_post(
            "car-vacuum-hit", brand="UNKNOWN",
            title="Use the blower mode to push dust out of AC vents, then vacuum it away #carvacuumcleaner #carcleaning",
            ctr=None, cvr=None, views=8000000, duration=28,
        )
        target["post"]["author"]["username"] = "UNKNOWN"
        target["post"]["author"]["display_name"] = ""
        target["post"]["ext"]["ad"].update({"ctr": None, "cvr": None, "objective": "", "landing_page_url": ""})
        target["post"]["content"]["text"] = "Use the blower mode to push dust out of AC vents, then vacuum it away #carvacuumcleaner #carcleaning"
        distractors = [
            synthetic_post("unrelated-f150", brand="UNKNOWN", title="Twin Turbo F150 for sale", ctr=None, cvr=None,
                           views=4000, duration=28),
            synthetic_post("unrelated-gifts", brand="UNKNOWN", title="Gifts they will actually use", ctr=None, cvr=None,
                           views=7000, duration=28),
        ]
        for item in distractors:
            item["post"]["author"]["username"] = "UNKNOWN"
            item["post"]["author"]["display_name"] = ""
            item["post"]["ext"]["ad"].update({"ctr": None, "cvr": None, "objective": "", "landing_page_url": ""})

        def fake_get(url, params=None, headers=None, timeout=None):
            return FakeResp(synthetic_response([target] + distractors))

        with mock.patch.object(T.requests, "get", fake_get):
            result = F.run_find(
                {"country": "US", "max_requests": 1, "max_ads": 3, "keywords": ["car vacuum"],
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )

        # High engagement and a product-related hashtag cannot turn use-context into an advertised target product.
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["diagnostics"]["outcome"], "NO_TARGET_RELEVANCE")
        gate = result["diagnostics"]["product_selection_gate"]
        self.assertGreater(gate["rejected_candidates"], 0)
        self.assertIn("OFF_TOPIC", gate["rejected_by_reason"])


    def test_auto_rejects_ambient_car_wash_vacuum_mention(self):
        ambient = synthetic_post(
            "ambient-car-wash", brand="UNKNOWN",
            title="Going to the car wash to vacuum 0 crumbs up! I can't touch the vacuums #carvacuum #carcleaning",
            ctr=None, cvr=None, views=8000000, duration=28,
        )
        ambient["post"]["author"]["username"] = "UNKNOWN"
        ambient["post"]["author"]["display_name"] = ""
        ambient["post"]["ext"]["ad"].update({"ctr": None, "cvr": None, "objective": "", "landing_page_url": ""})

        def fake_get(url, params=None, headers=None, timeout=None):
            return FakeResp(synthetic_response([ambient]))

        with mock.patch.object(T.requests, "get", fake_get):
            result = F.run_find(
                {"country": "US", "max_requests": 1, "max_ads": 10, "keywords": ["car vacuum"],
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )

        self.assertEqual(result["candidates"], [])
        by_reason = result["diagnostics"]["source_selection_diagnostics"]["by_reason"]
        self.assertIn("OFF_TOPIC", by_reason)
        row = result["stats"]["source_selection_diagnostics"]["examples"][0]
        self.assertEqual(row["product_match_reason"], "QUERY_PRODUCT_NOT_PROMOTED")

    def test_auto_rejects_product_with_explicit_non_target_market(self):
        uganda = synthetic_post(
            "uganda-market-vacuum", brand="UNKNOWN",
            title="Cordless car vacuum with powerful suction. Delivery available in Kampala and upcountry. "
                  "Call/WhatsApp #CarAccessoriesUganda #Kampala — jumia.ug",
            ctr=None, cvr=None, views=8000000, duration=28,
        )
        uganda["post"]["author"]["username"] = "UNKNOWN"
        uganda["post"]["author"]["display_name"] = ""
        uganda["post"]["ext"]["ad"].update({"ctr": None, "cvr": None, "objective": "", "landing_page_url": ""})

        def fake_get(url, params=None, headers=None, timeout=None):
            return FakeResp(synthetic_response([uganda]))

        with mock.patch.object(T.requests, "get", fake_get):
            result = F.run_find(
                {"country": "US", "max_requests": 1, "max_ads": 10, "keywords": ["car vacuum"],
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )

        self.assertEqual(result["candidates"], [])
        by_reason = result["diagnostics"]["source_selection_diagnostics"]["by_reason"]
        self.assertIn("SOURCE_MARKET_MISMATCH", by_reason)
        self.assertEqual(result["stats"]["source_market_mismatch_ads"], 1)
        row = next(
            row for row in result["stats"]["source_selection_diagnostics"]["examples"]
            if row["ad_id"] == "uganda-market-vacuum"
        )
        self.assertEqual(row["source_market_status"], "MISMATCH")
        self.assertIn("UG", json.dumps(row["source_market_evidence"]))

class CandidateRecord(Env):
    def run_candidates(self, *items):
        with self.fake_find(list(items)):
            return F.run_find({"country": "US", "max_requests": 1, "discovery_mode": "strict"},
                              client=T.SocialCrawlTikTokTopAds(api_key="K"), now="2026-10-21T00:00:00Z")["candidates"]

    def test_provenance_url_and_real_metrics_are_kept(self):
        c = self.run_candidates(synthetic_post("1001", views=120000, ctr=2.4))[0]
        self.assertEqual(c["metrics"]["views"]["value"], 120000)
        self.assertEqual(c["metrics"]["views"]["source"], "socialcrawl:tiktok_sponsored_search")
        self.assertEqual(c["metrics"]["views"]["collected_at"], "2026-10-21T00:00:00Z")
        self.assertEqual(c["metrics"]["ctr"]["value"], 2.4)
        self.assertTrue(c["source_url"].startswith("https://example.invalid/ad/1001"))
        self.assertEqual(c["country"], "US")
        self.assertEqual(c["origin"], "found")
        self.assertIn("legacy_record", c)

    def test_unknown_stays_unknown_not_zero(self):
        c = self.run_candidates(synthetic_post("1002", views=None, ctr=None, cvr=None))[0]
        for name in ("views", "ctr", "cvr"):
            self.assertNotIn(name, c["metrics"])

    def test_missing_video_is_none_not_invented(self):
        post = synthetic_post("1003")
        post["post"]["content"]["media_urls"] = []
        record = T.SocialCrawlTikTokTopAds._unwrap_item(post)
        c = F.to_candidate(
            {"record": record},
            {"country": "US", "period": 30, "vertical": "Physical Products"},
            "2026-10-23T00:00:00Z",
            "found",
        )
        self.assertIsNone(c["video_url"])


class DedupeAndFilters(Env):
    def cand(self, ad, **kw):
        country = kw.pop("country", "US")
        self._fixture_seen_counter = getattr(self, "_fixture_seen_counter", 0) + 1
        fixture_seen_path = Path(self.tmp.name) / f"find-seen-fixture-{self._fixture_seen_counter}.json"
        # These tests exercise Store deduplication, not the separate cross-run FIND history.
        # Give every synthetic discovery fixture fresh FIND state; persistence is tested explicitly.
        with mock.patch.dict(os.environ, {"CME_FIND_SEEN_PATH": str(fixture_seen_path)}):
            with self.fake_find([synthetic_post(ad, **kw)]):
                return F.run_find({"country": country, "max_requests": 1, "discovery_mode": "strict"},
                                  client=T.SocialCrawlTikTokTopAds(api_key="K"),
                                  now="2026-10-21T00:00:00Z")["candidates"][0]

    def test_same_ad_twice_is_one_card_with_history_and_latest_metrics(self):
        s = Store(self.data / "r.db")
        s.ingest([self.cand("2001", views=1000)], "run1", "found")
        out = s.ingest([self.cand("2001", views=5000)], "run2", "found")
        self.assertEqual(out, {"new": 0, "updated": 1})
        cards = s.list(state="all")
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["metrics"]["views"]["value"], 5000)
        hist = s.history(cards[0]["id"])
        self.assertEqual([h["metrics"]["views"]["value"] for h in hist], [1000, 5000])

    def test_duplicates_inside_one_batch_and_across_identity_keys(self):
        s = Store(self.data / "r.db")
        a, b = self.cand("2002"), self.cand("2002")
        self.assertEqual(s.ingest([a, b], "run1", "found"), {"new": 1, "updated": 1})
        c = dict(self.cand("2002"), id="tiktok:ad:other", keys=["tiktok:ad:other"] + self.cand("2002")["keys"][:1])
        s.ingest([c], "run2", "found")
        self.assertEqual(len(s.list(state="all")), 1)

    def test_filters(self):
        s = Store(self.data / "r.db")
        us = self.cand("3001", brand="alphabrand")
        de = dict(self.cand("3002", brand="betabrand"), country="DE", period_days=7, origin="imported")
        s.ingest([us], "r", "found")
        s.ingest([de], "r", "imported")
        self.assertEqual(len(s.list(state="all")), 2)
        self.assertEqual([c["ad_id"] for c in s.list(state="all", country="de")], ["3002"])
        self.assertEqual([c["ad_id"] for c in s.list(state="all", origin="imported")], ["3002"])
        self.assertEqual([c["ad_id"] for c in s.list(state="all", period="7")], ["3002"])
        self.assertEqual([c["ad_id"] for c in s.list(state="all", q="ALPHA")], ["3001"])
        self.assertEqual(s.list(state="all", platform="youtube"), [])


class OperatorDecisions(Env):
    def setUp(self):
        super().setUp()
        with self.fake_find([synthetic_post("4001"), synthetic_post("4002", brand="otherbrand")]):
            self.cands = F.run_find({"country": "US", "max_requests": 1, "discovery_mode": "strict"}, client=T.SocialCrawlTikTokTopAds(api_key="K"))["candidates"]

    def blob(self, s, cid):
        return s.db.execute("SELECT data FROM candidates WHERE id=?", (cid,)).fetchone()["data"]

    def test_decisions_persist_across_restart_and_leave_the_original_untouched(self):
        s = Store(self.data / "r.db")
        s.ingest(self.cands, "run", "found")
        a, b = [c["id"] for c in self.cands]
        before = self.blob(s, a)
        s.decide(a, "shortlist", "strong demo")
        s.decide(b, "exclude", "not a physical product")
        self.assertEqual(self.blob(s, a), before)  # operator choice never edits the ad record
        s.close()
        s2 = Store(self.data / "r.db")
        self.assertEqual([c["id"] for c in s2.list(state="shortlist")], [a])
        self.assertEqual(s2.get(b)["operator"]["reason"], "not a physical product")
        self.assertEqual(s2.list(state="feed"), [])
        self.assertEqual(s2.counts()["saved"], 1)

    def test_reasons_are_required_and_skip_leaves_the_feed(self):
        s = Store(self.data / "r.db")
        s.ingest(self.cands, "run", "found")
        a = self.cands[0]["id"]
        with self.assertRaises(ValueError):
            s.decide(a, "exclude", "")
        with self.assertRaises(ValueError):
            s.decide(a, "shortlist", "  ")
        s.decide(a, "skip")
        self.assertNotIn(a, [c["id"] for c in s.list(state="feed")])
        s.decide(a, "restore")
        self.assertIn(a, [c["id"] for c in s.list(state="feed")])
        self.assertEqual(len(s.decision_log(a)), 2)

    def test_viewed_counter(self):
        s = Store(self.data / "r.db")
        s.ingest(self.cands, "run", "found")
        s.mark_viewed(self.cands[0]["id"])
        self.assertEqual(s.counts()["viewed"], 1)


class ImportIsLabelled(Env):
    def test_import_marks_origin_and_rejects_other_formats(self):
        with self.fake_find([synthetic_post("5001")]):
            found = F.run_find({"country": "US", "max_requests": 1, "discovery_mode": "strict"}, client=T.SocialCrawlTikTokTopAds(api_key="K"))["candidates"][0]
        saved = Path(self.tmp.name) / "tiktok_discovery.json"
        saved.write_text(json.dumps({"request": {"country": "US", "period": 30}, "stats": {},
                                     "results": [{"record": found["legacy_record"], "decision": found["source_score"]}]}))
        res = F.import_discovery_file(saved)
        self.assertEqual(res["candidates"][0]["origin"], "imported")
        self.assertTrue(res["candidates"][0]["metrics"]["views"]["source"].startswith("import:"))
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text(json.dumps([1, 2]))
        with self.assertRaises(ValueError):
            F.import_discovery_file(bad)


class FindStates(Env):
    def test_not_configured_without_key(self):
        self.assertEqual(F.find_status()["state"], "NOT_CONFIGURED")
        app = RadarApp(self.data, settings_path=ROOT / "settings.example.json", catalog_path=ROOT / "adapters.json")
        res = app.do_find({"country": "US"})
        self.assertEqual(res["status"], "NOT_CONFIGURED")
        self.assertIn("SOCIALCRAWL_API_KEY", res["detail"])
        self.assertEqual(app.store.counts()["found"], 0)

    def test_local_find_uses_configured_offer_and_returns_source_diagnostics(self):
        offer_path = Path(self.tmp.name) / "offer.json"
        offer_path.write_text(json.dumps({"name": "Fanttik V8 Mate Cordless Car Vacuum",
                                          "product_type": "cordless handheld car vacuum",
                                          "product_category": "car cleaning"}), encoding="utf-8")
        calls = []
        app = RadarApp(self.data, settings_path=ROOT / "settings.example.json", catalog_path=ROOT / "adapters.json",
                       offer_path=offer_path, find_client=T.SocialCrawlTikTokTopAds(api_key="K"))
        with self.fake_find([synthetic_post("8101", title="Shop now: cordless car vacuum removes crumbs")], calls):
            result = app.do_find({"country": "US", "max_requests": 1, "max_ads": 3, "discovery_mode": "strict"})
        self.addCleanup(app.store.close)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(calls[0][1].get("query"), "Fanttik V8 Mate Cordless Car Vacuum")
        self.assertIn("diagnostics", result)
        self.assertEqual(result["diagnostics"]["outcome"], "CANDIDATES_FOUND")
        self.assertEqual(result["discovery_strategy"]["supply_search_calls_used"], 1)

    def test_api_failure_is_reported_not_replaced(self):
        def boom(*a, **k):
            raise OSError("network down")
        with mock.patch.object(T.requests, "get", boom):
            app = RadarApp(self.data, find_client=T.SocialCrawlTikTokTopAds(api_key="K"))
            res = app.do_find({"country": "US", "max_requests": 1})
        self.assertIn(res["status"], ("UNAVAILABLE",))
        self.assertEqual(app.store.counts()["found"], 0)

    def test_bad_parameters_are_rejected(self):
        app = RadarApp(self.data, find_client=T.SocialCrawlTikTokTopAds(api_key="K"))
        self.assertEqual(app.do_find({"period": 14})["status"], "BAD_REQUEST")
        self.assertEqual(app.do_find({"max_requests": 99})["status"], "BAD_REQUEST")



class CommercialEvidenceContractTests(unittest.TestCase):
    def test_native_ad_metrics_are_proxy_not_proof_of_profit(self):
        from cme.radar.evidence import classify_commercial_evidence
        evidence = classify_commercial_evidence(
            {"ctr": 0, "cvr": 0.8, "objective": "Conversions",
             "landing_page_url": "https://example.invalid"},
            {"final_score": 95, "components": {"money_signal_gate": {"pass": True}}},
            {"basis": "NATIVE_PERFORMANCE_METRICS", "available_fields": ["ctr", "cvr"]},
        )
        self.assertEqual(evidence["level"], "COMMERCIAL_PROXY")
        self.assertFalse(evidence["commercial_result_confirmed"])
        self.assertFalse(evidence["money_gate_is_profit_proof"])
        self.assertFalse(evidence["source_score_is_profit_probability"])

    def test_views_and_likes_remain_attention_only_even_with_high_score(self):
        from cme.radar.evidence import classify_commercial_evidence
        evidence = classify_commercial_evidence(
            {"views": 8000000, "likes": 200000},
            {"final_score": 99, "components": {"money_signal_gate": {"pass": True}}},
            {"basis": "ENGAGEMENT_PROXY", "available_fields": ["views", "likes"]},
        )
        self.assertEqual(evidence["level"], "ATTENTION_ONLY")
        self.assertFalse(evidence["commercial_result_confirmed"])

    def test_economic_validation_requires_verified_outcome_and_criterion(self):
        from cme.radar.evidence import classify_commercial_evidence
        candidate = {"economic_validation": {
            "status": "CONFIRMED", "target_met": True,
            "metrics": {"roas": 2.4, "net_profit": 135},
            "success_criteria": "ROAS >= 2.0 and net profit > 0",
            "verification_source": "TikTok Ads Manager export",
            "verified_at": "2026-10-26", "verified_by": "operator",
        }}
        evidence = classify_commercial_evidence(candidate, {}, {})
        self.assertEqual(evidence["level"], "ECONOMICALLY_VALIDATED")
        self.assertTrue(evidence["commercial_result_confirmed"])
        candidate["economic_validation"].pop("verified_by")
        self.assertNotEqual(
            classify_commercial_evidence(candidate, {}, {})["level"], "ECONOMICALLY_VALIDATED"
        )


class AutomaticSourceReadinessGate(unittest.TestCase):
    @staticmethod
    def candidate(cid, *, action, final_score, tags, money_pass=False,
                  mechanism_score=0, source_value_type="NONE", duration_block=False):
        return {
            "id": cid,
            "video_url": "https://example.invalid/ad.mp4",
            "source_check": {"classification": "DIRECT", "excluded": False},
            "source_score": {
                "action": action,
                "final_score": final_score,
                "tags": tags,
                "mechanism_source_score": mechanism_score,
                "source_value_type": source_value_type,
                "components": {
                    "money_signal_gate": {"pass": money_pass, "hard_block": not money_pass},
                    "duration": {"hard_block": duration_block},
                },
            },
        }

    def test_scorer_reject_money_gate_and_zero_score_do_not_block_pattern_analysis(self):
        rejected = self.candidate(
            "rejected", action="REJECT", final_score=0,
            tags=["money_signal_hard_block", "duration_over_60_hard_block"],
        )
        money_blocked = self.candidate(
            "money-blocked", action="SHALLOW", final_score=64,
            tags=["money_signal_hard_block"],
        )
        zero_score = self.candidate(
            "zero-score", action="DEEP", final_score=0,
            tags=["money_signal_pass"], money_pass=True, source_value_type="COMMERCIAL_WINNER",
        )
        selected, skipped = select_candidate([rejected, money_blocked, zero_score])
        self.assertEqual(selected["id"], "rejected")
        self.assertEqual(skipped, [])

    def test_money_gate_or_explicit_mechanism_exception_can_pass(self):
        commercial = self.candidate(
            "commercial", action="DEEP", final_score=70,
            tags=["money_signal_pass"], money_pass=True, source_value_type="COMMERCIAL_WINNER",
        )
        mechanism = self.candidate(
            "mechanism", action="DEEP", final_score=0,
            tags=["money_signal_hard_block", "duration_over_60_hard_block", "long_source_allowed_for_mechanism"],
            mechanism_score=72, source_value_type="MECHANISM_SOURCE", duration_block=True,
        )
        selected, skipped = select_candidate([mechanism, commercial])
        self.assertEqual(selected["id"], "mechanism")
        self.assertEqual(skipped, [])

def fake_runner(frames_dir_times=None):
    """Stands in for FFmpeg metadata probing/extraction: duration 12s."""
    class R:
        returncode, stdout, stderr = 0, "", ""
    def run(cmd):
        r = R()
        if "-frames:v" not in cmd:
            r.returncode = 1
            r.stderr = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'source.mp4':\n  Duration: 00:00:12.00, start: 0.000000"
        else:
            Path(cmd[-1]).write_bytes(b"\xff\xd8fakejpeg" + cmd[cmd.index("-ss") + 1].encode())
        return r
    return run


class FakeVideo:
    status_code = 200
    def __init__(self, data=b"fakevideo" * 50):
        self.data = data
    def iter_content(self, chunk_size=0):
        yield self.data


def analyst_transport(calls, content=None, status=200):
    good = {k: {"text": f"текст {k}", "kind": "OBSERVED"} for k in ("hook", "buyer_problem", "product_intervention", "result_demo", "cta")}
    good["mechanism_hypothesis"] = {"text": "гипотеза", "kind": "OBSERVED"}  # model overclaims: code must force INFERRED
    good["transferability"] = {"text": "перенос", "kind": "INFERRED"}
    good["evidence"] = [
        {"t_sec": 2.72, "frame": 1, "observation": "visible hook", "supports": ["hook", "product_intervention"]},
        {"t_sec": 4.94, "frame": 2, "observation": "visible buyer problem", "supports": ["buyer_problem"]},
        {"t_sec": 7.16, "frame": 3, "observation": "visible result state", "supports": ["result_demo"]},
        {"t_sec": 11.6, "frame": 5, "observation": "visible CTA", "supports": ["cta"]},
        {"t_sec": 8.0, "frame": 3, "observation": "fabricated timestamp", "supports": ["buyer_problem"]},
    ]
    good["limits"] = {"text": "кадры, не всё видео", "missing": ["звук"]}
    good["ratings"] = {"problem_clarity": 4, "product_demo": 9, "proof_visibility": None, "sequence_quality": 3}
    def send(method, url, headers, payload):
        calls.append(payload)
        text = content if content is not None else json.dumps(good)
        return status, json.dumps({"model": payload["model"], "choices": [{"message": {"content": text}}]}).encode()
    return send


class RadarAppFixture(Env):
    """Shared setup only; keep fixture tests from being inherited and run twice."""
    def make_app(self, calls, **kw):
        s = json.loads((ROOT / "settings.example.json").read_text())
        sp = Path(self.tmp.name) / "settings.json"
        sp.write_text(json.dumps(s))
        ctx = {"transport": analyst_transport(calls, **kw), "openrouter_key": "K"}
        app = RadarApp(self.data, settings_path=sp, catalog_path=ROOT / "adapters.json", ctx=ctx, getter=lambda *a, **k: FakeVideo(),
                       runner=fake_runner(), tools={"state": "READY", "detail": "", "how_to_fix": ""})
        with self.fake_find([synthetic_post("6001")]):
            res = F.run_find({"country": "US", "max_requests": 1, "discovery_mode": "strict"}, client=T.SocialCrawlTikTokTopAds(api_key="K"))
        app.store.ingest(res["candidates"], "run", "found")
        return app, res["candidates"][0]["id"]


class ManualAnalysis(RadarAppFixture):
    def test_nothing_calls_the_model_until_the_operator_asks(self):
        calls = []
        app, cid = self.make_app(calls)
        app.state(), app.feed({}), app.candidate(cid), app.store.mark_viewed(cid), app.store.decide(cid, "shortlist", "x")
        self.assertEqual(calls, [])
        app.do_analyze(cid)
        self.assertEqual(len(calls), 1)

    def test_analysis_keeps_source_link_frames_and_is_honest(self):
        calls = []
        app, cid = self.make_app(calls)
        res = app.do_analyze(cid)
        self.assertEqual(res["status"], "OK")
        a = app.store.analyses(cid)[0]
        self.assertEqual(a["status"], "OK")
        expected_model = configured_analyst_model()
        self.assertEqual(a["model_requested"], expected_model)
        self.assertEqual(a["model_responded"], expected_model)
        m = a["material"]
        # 12 s video: adaptive plan = 6 frames from just after the start to just before the end
        self.assertEqual([f["t_sec"] for f in m["frames"]], [0.5, 2.72, 4.94, 7.16, 9.38, 11.6])
        self.assertEqual(m["frame_sampling"]["mode"], "adaptive_anchored")
        self.assertEqual(m["frame_sampling"]["end_anchor_sec"], 11.6)
        self.assertFalse(m["video_fully_viewed"])
        self.assertTrue(m["source_url"].startswith("https://example.invalid/ad/6001"))
        self.assertEqual(m["ad_id"], "6001")
        r = a["result"]
        self.assertEqual(r["mechanism_hypothesis"]["kind"], "INFERRED")           # hypothesis never becomes an observation
        self.assertEqual([e["t_sec"] for e in r["evidence"]], [2.72, 4.94, 7.16, 11.6])  # invented timestamp dropped
        self.assertTrue(any("8.0" in w for w in r["warnings"]))
        self.assertIsNone(r["ratings"]["product_demo"])                           # out-of-range rating is not trusted
        self.assertEqual(len([c for c in calls[0]["messages"][1]["content"] if c["type"] == "image_url"]), 6)

    def test_missing_key_is_not_configured_and_makes_no_call(self):
        calls = []
        app, cid = self.make_app(calls)
        app.ctx = {}
        res = app.do_analyze(cid)
        self.assertEqual(res["status"], "NOT_CONFIGURED")
        self.assertIn("OPENROUTER_API_KEY", res["detail"])
        self.assertEqual(calls, [])
        self.assertEqual(app.store.analyses(cid), [])

    def test_api_error_is_recorded_without_inventing_an_analysis(self):
        calls = []
        app, cid = self.make_app(calls, status=500)
        res = app.do_analyze(cid)
        self.assertEqual(res["status"], "API_ERROR")
        a = app.store.analyses(cid)[0]
        self.assertEqual((a["status"], a["result"]), ("API_ERROR", None))
        self.assertTrue(a["material"]["frames"])  # what was sent is still on record

    def test_garbage_output_is_invalid_not_accepted(self):
        calls = []
        app, cid = self.make_app(calls, content="I think this ad is great!")
        self.assertEqual(app.do_analyze(cid)["status"], "INVALID_OUTPUT")
        a = app.store.analyses(cid)[0]
        self.assertIsNone(a["result"].get("hook"))
        self.assertIn("raw_excerpt", a["result"])

    def test_video_unavailable_is_reported(self):
        calls = []
        app, cid = self.make_app(calls)
        class Gone(FakeVideo):
            status_code = 403
        app.getter = lambda *a, **k: Gone()
        res = app.do_analyze(cid)
        self.assertEqual(res["status"], "VIDEO_UNAVAILABLE")
        self.assertEqual(calls, [])

    def test_no_video_url_and_missing_ffmpeg(self):
        calls = []
        app, cid = self.make_app(calls)
        app.tools = {"state": "NOT_CONFIGURED", "detail": "FFmpeg binary not found", "how_to_fix": "install imageio-ffmpeg"}
        self.assertEqual(app.do_analyze(cid)["status"], "NOT_CONFIGURED")
        self.assertEqual(calls, [])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
    def test_real_ffmpeg_extracts_frames_with_timestamps(self):
        video = Path(self.tmp.name) / "t.mp4"
        subprocess.run([A.ffmpeg_executable(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=6:size=320x568:rate=10",
                        "-pix_fmt", "yuv420p", str(video)], check=True)
        frames = A.extract_frames(video, Path(self.tmp.name) / "fr", count=3)
        self.assertEqual([f["t_sec"] for f in frames], [1.0, 3.0, 5.0])
        self.assertTrue(all(Path(f["path"]).stat().st_size > 500 and len(f["sha256"]) == 64 for f in frames))


class ProductionHandoff(RadarAppFixture):
    def test_handoff_requires_shortlist_and_real_target_product_photo_before_provider_calls(self):
        calls = []
        app, cid = self.make_app(calls)
        self.assertEqual(app.do_produce(cid)["status"], "NOT_SHORTLISTED")
        self.assertEqual(app.do_analyze(cid)["status"], "OK")
        app.store.decide(cid, "shortlist", "commercial mechanism worth testing")
        # The default example photo is deliberately a flat placeholder; do not waste a Brain/provider call.
        with mock.patch("cme.radar.server.subprocess.run") as run:
            result = app.do_produce(cid)
        self.assertEqual(result["status"], "TARGET_PHOTO_BLOCKED")
        self.assertEqual(result["provider_calls_made"], 0)
        run.assert_not_called()

    def test_transfer_rejection_keeps_its_own_status_not_atom_contract_failed(self):
        calls = []
        app, cid = self.make_app(calls)
        self.assertEqual(app.do_analyze(cid)["status"], "OK")
        app.store.decide(cid, "shortlist", "audit transfer rejection classification")
        photo = Path(self.tmp.name) / "target-product.png"
        photo.write_bytes(png_bytes(patch=((0, 0, 32, 48), (240, 15, 70))))
        offer_path = Path(self.tmp.name) / "offer.json"
        offer_path.write_text(json.dumps({
            "offer_id": "target-vacuum-test", "name": "Cordless car vacuum",
            "product_type": "car-cleaning vacuum", "verified_claims": ["Cordless"],
            "forbidden_claims": ["guaranteed"], "photos": [str(photo)], "geo": "US"
        }), encoding="utf-8")
        app.offer_path = offer_path

        def fake_run(cmd, **kwargs):
            out = Path(cmd[cmd.index("--out") + 1])
            (out / "run_result.json").write_text(json.dumps({
                "exit_code": 4, "failure_class": "TRANSFER_REJECTED",
                "message": "The source mechanism cannot be honestly transferred."
            }), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 4, stdout="TRANSFER_REJECTED", stderr="")

        with mock.patch("cme.radar.server.subprocess.run", side_effect=fake_run):
            report = app.do_produce(cid)
        self.assertEqual(report["exit_code"], 4)
        self.assertEqual(report["failure_class"], "TRANSFER_REJECTED")
        self.assertEqual(report["status"], "TRANSFER_REJECTED")

    def test_shortlisted_analyzed_source_hands_off_to_existing_atom_cli_and_keeps_artifact_preview(self):
        calls = []
        app, cid = self.make_app(calls)
        self.assertEqual(app.do_analyze(cid)["status"], "OK")
        app.store.decide(cid, "shortlist", "test the transfer hypothesis")
        photo = Path(self.tmp.name) / "target-product.png"
        photo.write_bytes(png_bytes(patch=((0, 0, 32, 48), (240, 15, 70))))
        offer_path = Path(self.tmp.name) / "offer.json"
        offer_path.write_text(json.dumps({"offer_id": "target-vacuum-test", "name": "Cordless car vacuum",
                                          "product_type": "car-cleaning vacuum", "verified_claims": ["Cordless"],
                                          "forbidden_claims": ["guaranteed"], "photos": [str(photo)], "geo": "US"}), encoding="utf-8")
        app.offer_path = offer_path
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            out = Path(cmd[cmd.index("--out") + 1])
            source_path = Path(cmd[cmd.index("--source-contract") + 1])
            seen["handoff"] = json.loads(source_path.read_text(encoding="utf-8"))
            (out / "shots").mkdir(parents=True)
            (out / "shots" / "shot_01_a01.png").write_bytes(png_bytes(patch=((0, 0, 20, 20), (15, 220, 60))))
            (out / "receipts").mkdir()
            (out / "receipts" / "drec_synthetic.json").write_text(json.dumps({"receipt_id": "drec_synthetic", "shot": 1,
                  "overall": "PASS", "commercial_status": "TEST_ONLY", "image_path": str(out / "shots" / "shot_01_a01.png")}), encoding="utf-8")
            (out / "calls.jsonl").write_text('{"status":"ok"}\n{"status":"ok"}\n{"status":"ok"}\n', encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="Brain atoms delivered; frame QA PASS", stderr="")

        with mock.patch("cme.radar.server.subprocess.run", side_effect=fake_run) as run:
            report = app.do_produce(cid)
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["provider_calls_made"], 3)
        self.assertEqual(report["images"], ["shot_01_a01.png"])
        self.assertEqual(report["verdicts"][0]["overall"], "PASS")
        self.assertEqual(report["verdicts"][0]["commercial_status"], "TEST_ONLY")
        self.assertEqual(seen["cmd"][seen["cmd"].index("all")], "all")
        self.assertEqual(seen["cmd"][seen["cmd"].index("--offer") + 1], str(offer_path))
        self.assertEqual(seen["cmd"][seen["cmd"].index("--source-contract") + 1], str(Path(app.data_dir) / "production" / report["run_id"] / "source_contract.json"))
        handoff_text = json.dumps(seen["handoff"], ensure_ascii=False)
        self.assertIn("mechanism_hypothesis", handoff_text)
        self.assertNotIn("source_url", handoff_text)
        self.assertNotIn("video_url", handoff_text)
        self.assertNotIn(app.candidate(cid)["analyses"][0]["material"]["frames"][0]["path"], handoff_text)
        self.assertEqual(app.productions(cid)[0]["run_id"], report["run_id"])
        run.assert_called_once()


class Server(ManualAnalysis):
    def start(self, calls):
        app, cid = self.make_app(calls)
        srv = make_server(app, port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return app, cid, srv.server_address[1]

    def req(self, port, method, path, body=None, token=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        h = {"Content-Type": "application/json", **(headers or {})}
        if token:
            h["X-Radar-Token"] = token
        conn.request(method, path, json.dumps(body) if body is not None else None, h)
        r = conn.getresponse()
        raw = r.read()
        conn.close()
        return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode())

    def test_get_endpoints_never_trigger_analysis_and_page_is_served(self):
        calls = []
        app, cid, port = self.start(calls)
        code, page = self.req(port, "GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("Physical Products", page)
        self.assertIn(app.token, page)
        self.assertIn("Видео недоступно", page)           # honest placeholder for missing video
        self.assertEqual(self.req(port, "GET", "/api/state")[0], 200)
        code, feed = self.req(port, "GET", "/api/feed?state=feed")
        self.assertEqual([c["id"] for c in feed["items"]], [cid])
        self.assertNotIn("legacy_record", feed["items"][0])
        self.assertEqual(self.req(port, "GET", f"/api/candidate?id={cid}")[0], 200)
        self.assertEqual(calls, [])

    def test_post_needs_token_origin_and_host(self):
        calls = []
        app, cid, port = self.start(calls)
        self.assertEqual(self.req(port, "POST", "/api/analyze", {"id": cid})[0], 403)
        self.assertEqual(self.req(port, "POST", "/api/analyze", {"id": cid}, token="wrong")[0], 403)
        self.assertEqual(self.req(port, "POST", "/api/analyze", {"id": cid}, token=app.token,
                                  headers={"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.req(port, "GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(calls, [])
        code, res = self.req(port, "POST", "/api/analyze", {"id": cid}, token=app.token)
        self.assertEqual((code, res["status"]), (200, "OK"))
        self.assertEqual(len(calls), 1)

    def test_decisions_over_http(self):
        app, cid, port = self.start([])
        code, res = self.req(port, "POST", "/api/decide", {"id": cid, "action": "exclude", "reason": ""}, token=app.token)
        self.assertEqual(code, 400)
        self.assertEqual(self.req(port, "POST", "/api/decide", {"id": cid, "action": "shortlist", "reason": "good"}, token=app.token)[0], 200)
        _, sl = self.req(port, "GET", "/api/feed?state=shortlist")
        self.assertEqual(sl["items"][0]["operator"]["reason"], "good")
        _, state = self.req(port, "GET", "/api/state")
        self.assertEqual(state["counts"]["saved"], 1)

    def test_production_preview_route_only_serves_known_png_artifacts(self):
        app, cid, port = self.start([])
        run_id = "prod-previewtest"
        image = Path(app.data_dir) / "production" / run_id / "shots" / "shot_01_a01.png"
        image.parent.mkdir(parents=True)
        payload = png_bytes(patch=((0, 0, 24, 30), (240, 20, 80)))
        image.write_bytes(payload)
        report = image.parents[1] / "production.json"
        report.write_text(json.dumps({"run_id": run_id, "candidate_id": cid, "started_at": "2026-10-23T00:00:00Z",
                                      "images": [image.name], "status": "OK"}), encoding="utf-8")
        code, data = self.req(port, "GET", f"/api/productions?id={cid}")
        self.assertEqual(code, 200)
        self.assertEqual(data["items"][0]["run_id"], run_id)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", f"/api/production-file?run={run_id}&name={image.name}")
        response = conn.getresponse()
        body = response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "image/png")
        self.assertEqual(body, payload)
        conn.close()
        self.assertEqual(self.req(port, "GET", f"/api/production-file?run={run_id}&name=..%2F..%2Fradar.db")[0], 400)

    def test_server_refuses_non_local_binding(self):
        app, _cid = self.make_app([])
        with self.assertRaises(ValueError):
            make_server(app, host="0.0.0.0")


class CoreStaysCompatible(unittest.TestCase):
    def test_settings_positional_order_and_new_field_are_backward_compatible(self):
        from cme.registry import Settings, load_settings
        s = Settings("hf", "paid", True)
        self.assertEqual((s.image_adapter, s.fallback_adapter, s.allow_free, s.analyst_adapter), ("hf", "paid", True, ""))
        self.assertEqual(load_settings(ROOT / "settings.example.json").analyst_adapter, "openrouter_free_analyst")

    def test_radar_does_not_touch_core_contract_modules(self):
        import cme.atoms, cme.receipt, cme.compiler
        src = "\n".join(Path(m.__file__).read_text() for m in (cme.atoms, cme.receipt, cme.compiler))
        self.assertNotIn("radar", src)


if __name__ == "__main__":
    unittest.main()


class ProductHeadContextAuditRegressionTests(unittest.TestCase):
    def test_combo_vacuum_blower_is_retained_when_target_product_is_actually_promoted(self):
        text = "This 3-in-1 car vacuum with blower mode blows dust from vents and vacuums it away. Order now."
        result = legacy_discover._product_type_relevance(
            {"title": text, "raw": {"post": {"content": {"text": text}}}},
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(result["product_anchor"], "vacuum")

    def test_trailing_use_context_is_not_mistaken_for_product_head(self):
        text = "Tire inflator inflates a flat tire in seconds. Get yours today."
        result = legacy_discover._product_type_relevance(
            {"title": text, "raw": {"post": {"content": {"text": text}}}},
            "tire inflator for cars",
        )
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(result["product_anchor"], "inflator")
        self.assertFalse(result["context_verified"])
        self.assertEqual(result["confidence"], "LOW")

    def test_product_noun_without_target_context_is_low_confidence_not_silently_misclassified(self):
        text = "This vacuum removes crumbs in seconds. Get yours today."
        result = legacy_discover._product_type_relevance(
            {"title": text, "raw": {"post": {"content": {"text": text}}}},
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(result["product_anchor"], "vacuum")
        self.assertFalse(result["context_verified"])
        self.assertEqual(result["confidence"], "LOW")

    def test_verb_only_hashtag_is_not_a_product_anchor(self):
        text = "Cleaning dusty seats. Get yours today. #vacuuming"
        result = legacy_discover._product_type_relevance(
            {"title": text, "raw": {"post": {"content": {"text": text}}}},
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "NO_MATCH")
        self.assertNotEqual(result["product_anchor"], "vacuuming")

    def test_compound_product_hashtag_needs_independent_pitch(self):
        text = "A quiet morning. #carvacuum"
        result = legacy_discover._product_type_relevance(
            {"title": text, "raw": {"post": {"content": {"text": text}}}},
            "cordless handheld car vacuum",
        )
        self.assertEqual(result["status"], "NO_MATCH")
        self.assertIn("product hashtag lacks independent source-body product evidence", result["reason"])
