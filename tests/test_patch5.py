"""Patch 5: remove false rejections without lowering quality control.

Each change is covered twice: the audited false rejection is gone, and the nearby genuine rejection stays.
Fixtures are generic; the audited run ids are only named in comments.
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cme.brain import request_transfer_fit
from cme.branding import NO_BRANDING, SWEEP_ANGLES, reference_lettering_rule
from cme.offer import load_offer
from cme.qa import CheckResult
from cme.radar.find import _query_offer_conflict
from cme.radar.fit import TARGET_PRODUCT, classify_source_role, match_source_to_product_query
from cme.reference_cache import ReferenceVerificationCache
from cme.run import produce_shot
from cme.state import new_run
from tests.helpers import A, gate_account, png_bytes
from tests.test_target_fit_gate import source_context, target_offer

EX = Path(__file__).resolve().parents[1] / "examples" / "offer_example.json"
PT, OFFER = "cordless handheld car vacuum", "Fanttik V8 Mate Cordless Car Vacuum"


def dropped_everything(why):
    account = gate_account(preserved=("hook",), dropped=("problem", "mechanism", "intervention", "result"))
    for row in account["source_elements_dropped"]:
        row["why"] = why
    return account


def negative(why):
    return {"transferable": False, "abstract_mechanism": "Visible dry debris is removed by a handheld action",
            "target_native_mapping": "Show debris and one cleaning pass", "target_evidence": [],
            "reason": "see element account", **dropped_everything(why)}


def positive():
    return {"transferable": True, "abstract_mechanism": "Visible dry debris is removed by a handheld action",
            "target_native_mapping": "Show crumbs on a car seat and one cleaning pass",
            "target_evidence": ["cordless handheld car vacuum"],
            "reason": "The target class performs the same visible action", **gate_account()}


class EvidenceGapGateTests(unittest.TestCase):
    GAP = "The smart display reading cannot be verified without target evidence for it"
    REAL = "The target has no suction mechanism at all, so it cannot perform the source's pulling-in action"

    def run_gate(self, decisions):
        calls = []

        def call(request):
            calls.append(request)
            return {"atoms": [], "transfer_decision": decisions[min(len(calls), len(decisions)) - 1]}
        return request_transfer_fit(target_offer(), "brief", source_context(), call), calls

    def test_evidence_gap_rejection_gets_one_extra_round_and_can_be_reconsidered(self):
        result, calls = self.run_gate([negative(self.GAP), positive()])
        self.assertTrue(result["transferable"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("NEGATIVE_TRANSFER_EVIDENCE_GAP" in e for e in calls[1]["previous_errors"]))

    def test_persistent_evidence_gap_stays_a_rejection_never_an_approval(self):
        result, calls = self.run_gate([negative(self.GAP)])
        self.assertFalse(result["transferable"])
        self.assertEqual(result["rejection_basis"], "EVIDENCE_GAP")
        self.assertEqual(len(calls), 3)  # the default two rounds plus one granted round

    def test_naming_a_real_incompatibility_after_the_hint_is_final_without_a_gap_label(self):
        result, calls = self.run_gate([negative(self.GAP), negative(self.REAL)])
        self.assertFalse(result["transferable"])
        self.assertNotIn("rejection_basis", result)
        self.assertEqual(len(calls), 2)

    def test_genuine_incompatibility_is_not_retried(self):
        result, calls = self.run_gate([negative(self.REAL)])
        self.assertFalse(result["transferable"])
        self.assertEqual(len(calls), 1)

    def test_mixed_reasons_are_not_treated_as_a_data_gap(self):
        account = dropped_everything(self.REAL)
        account["source_elements_dropped"][0]["why"] = self.GAP
        decision = {**negative(self.REAL), **account}
        result, calls = self.run_gate([decision])
        self.assertFalse(result["transferable"])
        self.assertEqual(len(calls), 1)

    def test_prompt_defines_essential_versus_optional_without_dropping_the_mechanism_rule(self):
        _, calls = self.run_gate([positive()])
        rules = " ".join(calls[0]["rules"])
        self.assertIn("ESSENTIAL VERSUS OPTIONAL", rules)
        self.assertIn("MECHANISM IS NON-NEGOTIABLE", rules)
        self.assertIn("must be rejected", rules)  # the unverified-unique-feature rule is untouched


class OwnedToolClassifierTests(unittest.TestCase):
    TIP = ("Before you vacuum, lightly mist the carpet so your vacuum can pull out more embedded debris instead of "
           "just grabbing surface dirt. It also helps reduce dust flying around inside the car. Then vacuum for a "
           "deeper clean. Save this for your next detail. #cardetailing #autodetailing #interiordetailing")

    def role(self, title, source="tiktok_sponsored_search", **extra):
        return classify_source_role({"title": title, "discovery_source": source, **extra},
                                    product_type=PT, target_offer=OFFER)

    def test_advice_about_a_tool_the_viewer_owns_is_not_the_promoted_product(self):
        role = self.role(self.TIP)  # audited false positive: selected as the only candidate in run 40924972088
        self.assertNotEqual(role["role"], TARGET_PRODUCT)
        record = {"title": self.TIP, "discovery_source": "tiktok_sponsored_search"}
        self.assertFalse(match_source_to_product_query(record, "car vacuum")["matches"])

    def test_real_ads_with_possessive_or_comparison_wording_stay_targets(self):
        for title in (
            "Upgrade your vacuum with this cordless one. Shop now #carvacuum",
            "Your car deserves a better vacuum #carvacuum #caraccessories",
            "Stop using your old vacuum for the car! This compact cordless tool fits in your boot. Link in bio",
            "Handheld Cordless Car Vacuum cleaner! #tiktokmademebuyit",
        ):
            with self.subTest(title=title[:50]):
                self.assertEqual(self.role(title)["role"], TARGET_PRODUCT)

    def test_strong_commercial_signals_keep_a_post_a_target_even_with_advice_wording(self):
        self.assertEqual(self.role(self.TIP, landing_page_url="https://shop.example.test/p")["role"], TARGET_PRODUCT)
        self.assertEqual(self.role(self.TIP + " Use code CLEAN10 at checkout")["role"], TARGET_PRODUCT)

    def test_ordinary_noun_use_anywhere_keeps_the_target(self):
        mixed = self.TIP + " This vacuum is the one I use every day."
        self.assertEqual(self.role(mixed)["role"], TARGET_PRODUCT)


class BuiltInFeatureTests(unittest.TestCase):
    """Run 41026869082: a car's built-in vacuum was admitted as the promoted product (attention 79)."""

    def role(self, title):
        return classify_source_role({"title": title, "discovery_source": "tiktok_sponsored_search"},
                                    product_type=PT, target_offer=OFFER)["role"]

    def test_a_feature_of_another_object_is_not_the_promoted_product(self):
        for title in ("The Pacifica has a built in vacuum and gives you so much storage space",
                      "This SUV has a built-in vacuum and a fridge!"):
            with self.subTest(title=title[:40]):
                self.assertNotEqual(self.role(title), TARGET_PRODUCT)

    def test_the_product_itself_may_have_built_in_parts_and_keeps_its_role(self):
        for title in ("Cordless handheld vacuum with built-in LED light and HEPA filter. Shop now #carvacuum",
                      "Handheld car vacuum with a built in battery #carvacuum"):
            with self.subTest(title=title[:40]):
                self.assertEqual(self.role(title), TARGET_PRODUCT)

    def test_built_in_wording_with_a_purchase_signal_or_hashtag_stays_a_target(self):
        self.assertEqual(self.role("Our van has a built in vacuum, get yours today, link in bio"), TARGET_PRODUCT)
        self.assertEqual(self.role("Built in vacuum for the car #carvacuum"), TARGET_PRODUCT)


class ConflictGuardTests(unittest.TestCase):
    OFFER = {"product_type": PT, "product_category": "car cleaning"}

    def test_offer_product_with_a_use_case_phrase_is_not_a_conflict(self):  # run 40930055638
        self.assertIsNone(_query_offer_conflict(["cordless car vacuum pet hair cleaning demonstration"], self.OFFER))

    def test_different_products_and_scattered_second_products_still_conflict(self):
        for query in ("blender recipes smoothie", "cordless car vacuum pet hair blender juicer",
                      "blender car vacuum pet hair", "blender car vacuum", "desk lamp"):
            with self.subTest(query=query):
                self.assertIsNotNone(_query_offer_conflict([query], self.OFFER), query)


class ReferenceVerdictCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        Image.new("RGB", (80, 80), (30, 60, 90)).save(self.source)
        self.clock = [1000.0]
        self.cache = ReferenceVerificationCache(self.root / "cache", "rule v1", 33, 30,
                                                algorithm_version="prep-v1", now=lambda: self.clock[0])

    def prepared(self, status="PASS", name="out/prep.png"):
        import hashlib, json
        destination = self.root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 80), (31, 61, 91)).save(destination)
        meta = {"status": "TEXT_REGIONS_MASKED", "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
                "prepared_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "sweep": 33,
                "algorithm_version": self.cache.algorithm_version,
                "verification": {"status": status, "evidence": "no lettering visible"}}
        destination.with_suffix(".prep.json").write_text(json.dumps(meta), encoding="utf-8")
        return destination

    def test_pass_roundtrips_into_a_fresh_output_directory(self):
        self.assertTrue(self.cache.store(self.source, self.prepared()))
        fresh = self.root / "newrun" / "prep.png"
        self.assertTrue(self.cache.restore(self.source, fresh))
        self.assertTrue(fresh.is_file() and fresh.with_suffix(".prep.json").is_file())

    def test_only_pass_is_ever_stored(self):
        for status in ("FAIL", "NOT_RUN"):
            self.assertFalse(self.cache.store(self.source, self.prepared(status)), status)
        self.assertFalse(self.cache.restore(self.source, self.root / "x.png"))

    def test_changed_source_changed_rule_tampered_png_and_age_are_misses(self):
        self.cache.store(self.source, self.prepared())
        stricter = ReferenceVerificationCache(self.root / "cache", "rule v2", 33, 30,
                                              algorithm_version="prep-v1", now=lambda: self.clock[0])
        self.assertFalse(stricter.restore(self.source, self.root / "a.png"))
        other = self.root / "other.png"
        Image.new("RGB", (80, 80), (1, 2, 3)).save(other)
        self.assertFalse(self.cache.restore(other, self.root / "b.png"))
        self.clock[0] += 31 * 86400
        self.assertFalse(self.cache.restore(self.source, self.root / "c.png"))
        self.clock[0] = 1000.0
        stored = next((self.root / "cache").glob("*.png"))
        stored.write_bytes(b"tampered")
        self.assertFalse(self.cache.restore(self.source, self.root / "d.png"))

    def test_changed_preparation_algorithm_invalidates_the_cached_pass(self):
        self.cache.store(self.source, self.prepared())
        changed = ReferenceVerificationCache(self.root / "cache", "rule v1", 33, 30,
                                             algorithm_version="prep-v2", now=lambda: self.clock[0])
        self.assertFalse(changed.restore(self.source, self.root / "algorithm-change" / "prep.png"))

    def test_cache_is_off_without_the_environment_variable(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("CME_REFERENCE_CACHE_DIR", None)
            self.assertIsNone(ReferenceVerificationCache.from_env("r", 1))

    def test_cached_pass_skips_the_judge_but_a_miss_still_requires_it(self):
        offer = replace(load_offer(EX), photos=(str(self.source),), branding_policy=NO_BRANDING)
        cache = ReferenceVerificationCache(self.root / "cache", reference_lettering_rule(), len(SWEEP_ANGLES))
        judge_calls = []

        def judge(_path, atoms):
            judge_calls.append(1)
            return {a.id: {"pass": True, "evidence": "no lettering visible"} for a in atoms}

        def run(tag):
            with patch.dict("os.environ", {"CME_REFERENCE_CACHE_DIR": str(self.root / "cache")}):
                return produce_shot(
                    offer=offer, atoms=[A("product_main", "Black handheld vacuum body is visible", kind="product")],
                    shot=1, contract_id="c-" + tag, state=new_run("r-" + tag), out_dir=self.root / ("out-" + tag),
                    image_provider=lambda prompt, images: png_bytes(),
                    qa_checks=lambda _p: [CheckResult("file", "PASS", True, "code")], reference_judge=judge)

        stub = self._stub()
        with patch("cme.run.prepare_unbranded_reference", side_effect=stub):
            run("one")
            self.assertEqual(len(judge_calls), 1)  # first run: judged, then stored
            run("two")
            self.assertEqual(len(judge_calls), 1)  # second run: served from cache, no judge call
        self.assertTrue(any((self.root / "cache").glob("*.prep.json")))

    def _stub(self):
        import hashlib, json
        source = self.source

        def prepare(src, destination):
            destination = Path(destination)
            sidecar = destination.with_suffix(".prep.json")
            if sidecar.is_file() and destination.is_file():
                cached = json.loads(sidecar.read_text())
                return {**cached, "path": str(destination), "cached": True}
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), (40, 70, 100)).save(destination)
            report = {"path": str(destination), "status": "TEXT_REGIONS_MASKED", "masked_regions": 1,
                      "residual_regions": 0, "detected_words": ["x"], "sweep": len(SWEEP_ANGLES), "cached": False,
                      "algorithm_version": 3,
                      "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                      "prepared_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
            sidecar.write_text(json.dumps(report), encoding="utf-8")
            return report
        return prepare


if __name__ == "__main__":
    unittest.main()
