"""Contract-aware FIND: regressions reproduced from audited runs 40647469407 .. 40681442750.

Titles below are real Sponsored Search rows from those runs (run 40681442750 rejected the relevant ones
as OFF_TOPIC / ADJACENT_MECHANISM and kept spending 42 credits on Ads Top calls that returned unrelated ads).
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cme.radar.find import _retention_execution_plan, run_find
from cme.radar.fit import TARGET_PRODUCT, classify_source_role, match_source_to_product_query
from cme.radar.provider_contract import ordered_target_queries

OFFER = {"name": "Fanttik V8 Mate Cordless Car Vacuum", "product_type": "cordless handheld car vacuum"}
BRAND = OFFER["name"]
PRODUCT_TYPE = OFFER["product_type"]

RELEVANT = [
    ("7695019243103898902", "my car been so messy ngl new vacuum here we go #car #carvacuumcleaner #vacuumcleaner #caraccessories", 82628, 474),
    ("7700228812834983198", "It Blows AND Vacuums… And It's on Black Friday Sale?! #blackfriday #carvacuumcleaner #carcleaning", 137627, 316),
    ("7698115546994314510", "Give your car a little cleanup with the BiLOLLY cordless handheld vacuum. Two suction modes, a washable HEPA filter. #HandheldVacuum #CarCleaning", 38000, 330),
    ("7664704984188996895", "The compact vacuum that keeps your car looking its best every day#CarVacuum #CarCare #CleanCarDaily", 1757622, 4400),
    ("7000000000000000001", "Handheld Cordless Car Vacuum cleaner! #tiktokmademebuyit #foryoudays #vacuumcleaner #handheldvacuum", 900000, 9000),
]
IRRELEVANT = [
    ("7646732321416088854", "This is why you should always lightly mist your carpets before vacuuming #cleaninghacks", 1951090, 26020),
    ("7000000000000000002", "Vacuum sealed meal prep containers keep food fresh #kitchen #mealprep", 600000, 6000),
    ("7000000000000000003", "Use the blower mode to push dust out of AC vents, then vacuum it away #carvacuumcleaner #carcleaning", 8000000, 900),
    ("7000000000000000004", "Bird feeder that squirrels cannot raid #garden", 700000, 7000),
]


def row(ad_id, title, views, likes):
    return {
        "ad_id": ad_id, "video_id": "v-" + ad_id,
        "source_url": "https://www.tiktok.com/@seller/video/" + ad_id,
        "video_url": "https://cdn.example.test/" + ad_id + ".mp4",
        "title": title, "views": views, "likes": likes, "comments": 30, "shares": 20,
        "published_at": "2026-10-10T00:00:00Z",
        "raw": {"post": {"content": {"text": title}, "ext": {"ad": {"title": title}}}},
    }


class RecordingClient:
    """Sponsored Search only. Any Ads Top / retention call fails the test."""
    enabled = True

    def __init__(self, rows):
        self.rows, self.calls, self.last_discovery_stats = rows, [], {}

    def fetch_sponsored_search(self, **kwargs):
        self.calls.append(("sponsored_search", kwargs))
        return {"items": [dict(r) for r in self.rows], "credits_used": 1, "request_id": "fixture", "params": kwargs}

    def fetch_board(self, **kwargs):
        self.calls.append(("ads_top", kwargs))
        raise AssertionError("contract-aware product FIND must not call Ads Top: " + repr(kwargs))


class ContractAwarePlanTests(unittest.TestCase):
    def test_playback_boards_with_retention_are_not_executed_by_default(self):
        boards, max_requests, strategy = _retention_execution_plan(["play_6s_rate", "play_2s_rate"], True, 5)
        self.assertEqual(boards, [])
        self.assertEqual(max_requests, 5)
        self.assertEqual(strategy["mode"], "contract_aware_sponsored_batch")
        self.assertEqual(strategy["retention_calls"], 0)
        self.assertEqual(strategy["targeted_board_calls"], 0)
        self.assertTrue(strategy["retention_requested_but_not_executed"])
        self.assertEqual(strategy["requested_metric_status"], "NOT_OBTAINABLE_FOR_PRODUCT_QUERY")
        self.assertEqual(strategy["requested_performance_order_bys"], ["play_6s_rate", "play_2s_rate"])

    def test_legacy_plan_stays_available_for_explicit_probes(self):
        boards, _, strategy = _retention_execution_plan(
            ["play_6s_rate", "play_2s_rate"], True, 5, performance_plan="legacy_ads_top")
        self.assertEqual(boards, ["play_6s_rate", "play_2s_rate"])
        self.assertEqual(strategy["mode"], "targeted_playback_boards_plus_retention")

    def test_exact_product_name_always_leads_the_query_plan(self):
        queries = ordered_target_queries(BRAND, PRODUCT_TYPE, ["car vacuum"], "", limit=3)
        self.assertEqual(queries, [BRAND, PRODUCT_TYPE, "car vacuum"])
        rotated = ordered_target_queries(BRAND, PRODUCT_TYPE, ["car vacuum"], "", limit=3,
                                         rotate=lambda terms: list(reversed(terms)))
        self.assertEqual(rotated[0], BRAND)  # only the tail rotates between runs


class RoleClassifierRegressionTests(unittest.TestCase):
    def role(self, title, source="tiktok_sponsored_search"):
        record = {"title": title, "ad_id": "x"}
        if source:
            record["discovery_source"] = source
        return classify_source_role(record, product_type=PRODUCT_TYPE, target_offer=BRAND)

    def test_relevant_car_vacuum_ads_from_run_40681442750_are_target_product(self):
        for _, title, _, _ in RELEVANT:
            with self.subTest(title=title[:50]):
                role = self.role(title)
                self.assertEqual(role["role"], TARGET_PRODUCT)
                self.assertTrue(role["promotion_evidence"] and role["target_promotion_evidence"])

    def test_passing_mentions_and_verb_use_stay_rejected(self):
        for _, title, _, _ in IRRELEVANT:
            with self.subTest(title=title[:50]):
                role = self.role(title)
                self.assertFalse(role["role"] == TARGET_PRODUCT and role["target_promotion_evidence"])

    def test_paid_inventory_marker_is_required_for_title_only_product_names(self):
        title = "Handheld Cordless Car Vacuum cleaner! #foryoudays"
        self.assertNotEqual(self.role(title, source=None)["role"], TARGET_PRODUCT)
        self.assertEqual(self.role(title)["role"], TARGET_PRODUCT)

    def test_query_match_follows_the_same_rule(self):
        record = {"title": RELEVANT[4][1], "discovery_source": "tiktok_sponsored_search"}
        self.assertTrue(match_source_to_product_query(record, BRAND)["matches"])
        self.assertFalse(match_source_to_product_query(
            {"title": IRRELEVANT[3][1], "discovery_source": "tiktok_sponsored_search"}, "car vacuum")["matches"])


class ContractAwareFindTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = Path(self.temp.name) / "seen.json"

    def run_find(self, rows, **extra):
        client = RecordingClient([row(*item) for item in rows])
        params = {"country": "US", "period": 30, "max_requests": 5, "max_ads": 10, "keywords": ["car vacuum"],
                  "performance_order_bys": ["play_6s_rate", "play_2s_rate"], "include_retention": True,
                  "offer": OFFER, **extra}
        with mock.patch.dict("os.environ", {"CME_FIND_SEEN_PATH": str(self.ledger)}):
            return run_find(params, client=client, now="2026-10-29T00:00:00Z"), client

    def test_run_40681442750_replay_finds_relevant_ads_without_ads_top_spend(self):
        result, client = self.run_find(RELEVANT + IRRELEVANT)
        self.assertEqual([call[0] for call in client.calls], ["sponsored_search"] * 3)
        queries = [call[1]["query"] for call in client.calls]
        self.assertEqual(queries[0], BRAND)  # brand query is in the executed plan
        ids = {c["ad_id"] for c in result["candidates"]}
        self.assertEqual(ids, {r[0] for r in RELEVANT})
        self.assertEqual(result["diagnostics"]["outcome"], "CANDIDATES_FOUND")
        self.assertEqual(result["diagnostics"]["performance_calls_used"], 0)
        self.assertEqual(result["stats"]["total_credits_reported"], 3)

    def test_requested_metric_is_labelled_not_faked_and_not_a_gate(self):
        result, _ = self.run_find(RELEVANT + IRRELEVANT)
        availability = result["diagnostics"]["performance_metric_availability"]
        self.assertEqual(availability["requested_metrics"], ["play_6s_rate", "play_2s_rate"])
        self.assertEqual(availability["status"], "NOT_OBTAINABLE_FOR_PRODUCT_QUERY")
        self.assertEqual(availability["numeric_values_returned"], {})
        for candidate in result["candidates"]:
            evidence = candidate["performance_evidence"]
            self.assertEqual(evidence["basis"], "ATTENTION_PROXY_ONLY")
            self.assertFalse(evidence["is_commercial_proof"])
            self.assertFalse(evidence["requested_metric_confirmed"])
            self.assertEqual(evidence["requested_metric_status"], "NOT_OBTAINABLE_FOR_PRODUCT_QUERY")
            self.assertEqual(candidate["source_selection"]["role"], TARGET_PRODUCT)

    def test_relevant_ads_are_not_cut_by_a_pool_relative_floor(self):
        # 5 relevant ads with very different reach: all pass an absolute attention floor; none is cut
        # because irrelevant rows with more views sit above them.
        result, _ = self.run_find(RELEVANT + IRRELEVANT)
        self.assertEqual(len(result["candidates"]), len(RELEVANT))
        reasons = result["diagnostics"]["source_selection_diagnostics"]["rejected_by_reason"]
        self.assertNotIn("BELOW_PERFORMANCE_FLOOR", reasons)

    def test_attention_floor_remains_an_explicit_operator_setting(self):
        result, _ = self.run_find(RELEVANT, attention_floor=1000)
        self.assertEqual(result["candidates"], [])
        reasons = result["diagnostics"]["source_selection_diagnostics"]["rejected_by_reason"]
        self.assertEqual(reasons.get("BELOW_ATTENTION_PROXY_FLOOR"), len(RELEVANT))

    def test_only_irrelevant_rows_yield_an_honest_no_target_relevance(self):
        result, client = self.run_find(IRRELEVANT)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["diagnostics"]["outcome"], "NO_TARGET_RELEVANCE")
        self.assertTrue(all(call[0] == "sponsored_search" for call in client.calls))

    def test_seen_ledger_still_applies(self):
        first, _ = self.run_find(RELEVANT)
        second, _ = self.run_find(RELEVANT)
        self.assertEqual(len(first["candidates"]), len(RELEVANT))
        self.assertEqual(second["candidates"], [])
        self.assertEqual(second["stats"]["seen_before_ads"], len(RELEVANT))


if __name__ == "__main__":
    unittest.main()
