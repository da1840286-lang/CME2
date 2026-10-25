"""Creative Radar acceptance tests. All data here is SYNTHETIC and lives only in temporary folders."""
import hashlib, http.client, json, os, shutil, subprocess, tempfile, threading, unittest
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


HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


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

    def test_run_find_calls_the_real_client_and_ranker(self):
        calls = []
        with self.fake_find([synthetic_post("1001")], calls):
            res = F.run_find({"country": "us", "max_requests": 2, "keywords": ["car vacuum"],
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
        self.assertEqual(res["candidates"][0]["metrics"]["views"]["source"], "socialcrawl:tiktok_sponsored_search")

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
            res = F.run_find({"country": "US", "max_requests": 3, "keywords": ["car vacuum"],
                              "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                                        "product_type": "cordless handheld car vacuum"}},
                             client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"))
        queries = [p["query"] for url, p in calls if url == T.SocialCrawlTikTokTopAds.SEARCH_URL]
        self.assertEqual(queries, ["car vacuum", "Fanttik V8 Mate Cordless Car Vacuum", "cordless handheld car vacuum"])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(url == T.SocialCrawlTikTokTopAds.SEARCH_URL for url, _ in calls))
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
                {"country": "US", "max_requests": 2, "max_ads": 1, "keywords": ["car vacuum"],
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], T.SocialCrawlTikTokTopAds.SEARCH_URL)
        self.assertEqual(calls[1][0], T.SocialCrawlTikTokTopAds.BASE_URL)
        self.assertEqual(calls[1][1].get("order_by"), "ctr")
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

    def test_find_reserves_ads_top_requests_without_weakening_money_gate(self):
        calls = []
        def fake_get(url, params=None, headers=None, timeout=None):
            params = dict(params or {})
            calls.append((url, params))
            item = synthetic_post(
                str(9300 + len(calls)), brand="UNKNOWN",
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
                {"country": "US", "max_requests": 10, "max_ads": 10, "keywords": ["car vacuum"],
                 "offer": {"name": "Fanttik V8 Mate Cordless Car Vacuum",
                           "product_type": "cordless handheld car vacuum"}},
                client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"),
            )
        sources = [call.get("source") for call in result["stats"]["calls"]]
        self.assertEqual(sources.count("sponsored_search"), 7)
        self.assertEqual(sources.count("ads_top"), 3)
        self.assertEqual([params.get("order_by") for url, params in calls
                          if url == T.SocialCrawlTikTokTopAds.BASE_URL],
                         ["ctr", "cvr", "play_6s_rate"])
        self.assertEqual(result["stats"]["money_ready_source_ads"], 0)
        self.assertEqual(result["stats"]["mechanism_ready_source_ads"], 0)
        self.assertEqual(result["stats"]["discovery_strategy"]["supply_search_budget"], 7)
        self.assertEqual(result["stats"]["discovery_strategy"]["performance_budget"], 3)
        for candidate in result["candidates"]:
            score = candidate.get("source_score") or {}
            components = score.get("components") or {}
            gate = components.get("money_signal_gate") or {}
            self.assertFalse(gate.get("pass"))


class CandidateRecord(Env):
    def run_candidates(self, *items):
        with self.fake_find(list(items)):
            return F.run_find({"country": "US", "max_requests": 1}, client=T.SocialCrawlTikTokTopAds(api_key="K"),
                              now="2026-10-21T00:00:00Z")["candidates"]

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
        with self.fake_find([synthetic_post(ad, **kw)]):
            return F.run_find({"country": kw.pop("country", "US"), "max_requests": 1},
                              client=T.SocialCrawlTikTokTopAds(api_key="K"), now="2026-10-21T00:00:00Z")["candidates"][0]

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
            self.cands = F.run_find({"country": "US", "max_requests": 1}, client=T.SocialCrawlTikTokTopAds(api_key="K"))["candidates"]

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
            found = F.run_find({"country": "US", "max_requests": 1}, client=T.SocialCrawlTikTokTopAds(api_key="K"))["candidates"][0]
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
        with self.fake_find([synthetic_post("8101")], calls):
            result = app.do_find({"country": "US", "max_requests": 1, "max_ads": 3})
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

    def test_reject_money_block_and_zero_score_are_skipped(self):
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
        self.assertIsNone(selected)
        self.assertEqual(len(skipped), 3)
        self.assertIn("action is REJECT", skipped[0])
        self.assertIn("money-signal hard block", skipped[1])
        self.assertIn("zero or below", skipped[2])

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
    """Stands in for ffprobe/ffmpeg: duration 12s; every ffmpeg call writes a tiny file."""
    class R:
        returncode, stdout, stderr = 0, "", ""
    def run(cmd):
        r = R()
        if cmd[0] == "ffprobe":
            r.stdout = "12.0\n"
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
    good["evidence"] = [{"t_sec": 3.0, "frame": 1, "observation": "виден товар"}, {"t_sec": 8.0, "frame": 3, "observation": "выдумано"}]
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
            res = F.run_find({"country": "US", "max_requests": 1}, client=T.SocialCrawlTikTokTopAds(api_key="K"))
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
        self.assertEqual([f["t_sec"] for f in m["frames"]], [1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        self.assertFalse(m["video_fully_viewed"])
        self.assertTrue(m["source_url"].startswith("https://example.invalid/ad/6001"))
        self.assertEqual(m["ad_id"], "6001")
        r = a["result"]
        self.assertEqual(r["mechanism_hypothesis"]["kind"], "INFERRED")           # hypothesis never becomes an observation
        self.assertEqual([e["t_sec"] for e in r["evidence"]], [3.0])             # invented timestamp dropped
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
        app.tools = {"state": "NOT_CONFIGURED", "detail": "ffmpeg/ffprobe not found", "how_to_fix": "install ffmpeg"}
        self.assertEqual(app.do_analyze(cid)["status"], "NOT_CONFIGURED")
        self.assertEqual(calls, [])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
    def test_real_ffmpeg_extracts_frames_with_timestamps(self):
        video = Path(self.tmp.name) / "t.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=6:size=320x568:rate=10",
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
