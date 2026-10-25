import json, tempfile, unittest
from pathlib import Path
from cme.offer import OfferCard, load_offer
from cme.run import produce_shot
from cme.state import new_run
from cme.vertical import GateError, check_vertical, load_vertical, offer_gate
from cme.brain import request_atoms
from cme.qa import CheckResult, file_check
from tests.helpers import A, png_bytes

ROOT = Path(__file__).resolve().parent.parent
VERT = load_vertical(ROOT / "verticals" / "physical_goods_us.json")


CONFIRMED = ("audience_18_plus", "verify_market_eligibility_on_platform")


def offer(**kw):
    base = dict(offer_id="o", name="n", product_type="t", verified_claims=("Reusable and dishwasher safe",),
                forbidden_claims=(), photos=("p.png",), platform="", geo="US", product_category="")
    base.update(kw)
    return OfferCard(**base)


class VerticalDataTests(unittest.TestCase):
    def test_example_offer_defaults_to_us_and_passes_gate(self):
        o = load_offer(ROOT / "examples" / "offer_example.json")
        self.assertEqual(o.geo, "US")
        self.assertEqual(offer_gate(o, VERT).blocked, ())

    def test_geo_outside_vertical_is_blocked(self):
        g = offer_gate(offer(geo="DE"), VERT)
        self.assertIn("not covered", g.blocked[0])

    def test_non_culinary_knife_blocked_on_meta_only(self):
        self.assertIn("not allowed on meta", offer_gate(offer(product_category="knives_non_culinary", platform="meta"), VERT).blocked[0])
        ok = offer_gate(offer(product_category="knives_non_culinary", platform="tiktok", confirmations=CONFIRMED), VERT)
        self.assertEqual(ok.blocked, ())
        self.assertIn("audience_18_plus", ok.requirements)

    def test_restricted_category_without_platform_fails_closed(self):
        g = offer_gate(offer(product_category="knives_non_culinary"), VERT)
        self.assertIn("explicit platform", g.blocked[0])

    def test_culinary_knife_requires_adult_audience(self):
        g = offer_gate(offer(product_category="knives_culinary", platform="meta", confirmations=CONFIRMED), VERT)
        self.assertEqual(g.blocked, ())
        self.assertIn("audience_18_plus", g.requirements)

    def test_unconfirmed_requirements_block(self):
        g = offer_gate(offer(product_category="knives_culinary", platform="meta"), VERT)
        self.assertEqual(len(g.blocked), 2)
        self.assertIn("audience_18_plus", g.blocked[0])
        self.assertIn("not confirmed", g.blocked[0])


class AtomRuleTests(unittest.TestCase):
    def test_substantiation_patterns_need_verified_claim(self):
        bad = [A("a", "Proudly made in the USA"), A("b", "Five-star customer reviews"), A("c", "The #1 best-selling pan"),
               A("d", "American flag in the background")]
        errs = check_vertical(bad, offer(), VERT)
        self.assertEqual(set(errs), {"a", "b", "c", "d"})
        self.assertIn("made_in_usa", errs["a"][0])

    def test_same_claim_passes_when_verified_verbatim(self):
        o = offer(verified_claims=("Made in USA",))
        self.assertEqual(check_vertical([A("a", "Made in USA")], o, VERT), {})

    def test_weapon_context_blocked_for_knives(self):
        o = offer(product_category="knives_culinary", platform="tiktok")
        errs = check_vertical([A("a", "A tactical blade for self-defense"), A("b", "Slicing tomatoes")], o, VERT)
        self.assertEqual(set(errs), {"a"})
        self.assertIn("RESTRICTED_CONTEXT", errs["a"][0])

    def test_plain_atoms_pass(self):
        self.assertEqual(check_vertical([A("a", "A bright kitchen counter")], offer(), VERT), {})

    def test_brain_retry_fixes_vertical_violation(self):
        o = load_offer(ROOT / "examples" / "offer_example.json")
        bad = {"id": "a_1", "kind": "scene", "text": "Made in the USA", "priority": "MUST", "channel": "image", "shot": 1}
        good = dict(bad, text="A clean kitchen counter")
        calls = []
        def call(req):
            calls.append(req)
            return {"atoms": [bad]} if len(calls) == 1 else {"atoms": [good]}
        atoms = request_atoms(o, "b", 1, call, vertical=VERT)
        self.assertEqual(atoms[0].text, "A clean kitchen counter")
        self.assertIn("NEEDS_SUBSTANTIATION", calls[1]["only_fix"][0]["errors"][0])


class PipelineGateTests(unittest.TestCase):
    def test_gate_stops_before_provider_call(self):
        sent = []
        def provider(p, r):
            sent.append(p)
            return png_bytes()
        o = offer(product_category="knives_non_culinary", platform="meta")
        with tempfile.TemporaryDirectory() as d, self.assertRaises(GateError):
            produce_shot(offer=o, atoms=[A("a", "Slicing bread")], shot=1, contract_id="c", state=new_run("r"),
                         out_dir=d, image_provider=provider, qa_checks=lambda p: [], vertical=VERT)
        self.assertEqual(sent, [])

    def test_requirements_land_in_receipt_report(self):
        o = offer(product_category="knives_culinary", platform="tiktok", confirmations=CONFIRMED)
        with tempfile.TemporaryDirectory() as d:
            produce_shot(offer=o, atoms=[A("a", "Slicing bread")], shot=1, contract_id="c", state=new_run("r"),
                         out_dir=d, image_provider=lambda p, r: png_bytes(),
                         qa_checks=lambda p: [file_check("file", p)], vertical=VERT)
            rep = json.loads(next((Path(d) / "receipts").glob("*.json")).read_text())
            self.assertIn("audience_18_plus", rep["ad_requirements"])


class RulesLiveInDataTests(unittest.TestCase):
    def test_engine_code_has_no_vertical_words(self):
        src = "\n".join(p.read_text().lower() for p in (ROOT / "cme").glob("*.py"))
        import re
        for rx in (r"\bknife\b", r"\bknives\b", r"made in usa", r"\btiktok\b", r"\bmeta ads\b", r"\bfacebook\b", r"self-defense"):
            self.assertIsNone(re.search(rx, src), f"vertical word {rx!r} leaked into engine code")

    def test_vertical_file_declares_status_and_sources(self):
        d = json.loads((ROOT / "verticals" / "physical_goods_us.json").read_text())
        self.assertIn("Not legal advice", d["status"])
        self.assertTrue(d["sources"])


if __name__ == "__main__":
    unittest.main()
