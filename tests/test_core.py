import json, tempfile, unittest
from pathlib import Path
from cme.atoms import ATOM_SCHEMA, MAX_TEXT, check_claims, parse_atoms
from cme.brain import build_request
from cme.compiler import BudgetError, compile_image_prompt, compile_motion_prompt
from cme.offer import OfferError, load_offer
from tests.helpers import A

EX = Path(__file__).resolve().parent.parent / "examples" / "offer_example.json"


class OfferTests(unittest.TestCase):
    def test_example_loads(self):
        o = load_offer(EX)
        self.assertEqual(o.offer_id, "example-001")

    def test_missing_photo_and_claims_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.json"
            p.write_text(json.dumps({"offer_id": "x", "name": "n", "product_type": "t",
                                     "verified_claims": [], "photos": []}))
            with self.assertRaises(OfferError) as cm:
                load_offer(p)
            self.assertIn("verified_claims", str(cm.exception))
            self.assertIn("photo", str(cm.exception))

    def test_long_claim_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.json"
            p.write_text(json.dumps({"offer_id": "x", "name": "n", "product_type": "t",
                                     "verified_claims": ["c" * 121], "photos": ["a.png"]}))
            with self.assertRaises(OfferError):
                load_offer(p, check_files=False)


class AtomTests(unittest.TestCase):
    def good(self, **kw):
        d = {"id": "a_1", "kind": "scene", "text": "A kitchen counter", "priority": "MUST",
             "channel": "image", "shot": 1}
        d.update(kw)
        return d

    def test_schema_carries_length_limit(self):
        item = ATOM_SCHEMA["properties"]["atoms"]["items"]["properties"]["text"]
        self.assertEqual(item["maxLength"], MAX_TEXT)

    def test_parse_collects_per_atom_errors(self):
        raw = [self.good(), self.good(id="b_2", text="x" * 121), self.good(id="c_3", priority="HIGH"),
               self.good(id="a_1")]
        r = parse_atoms(raw)
        self.assertEqual(r.atoms, [])  # a_1 is used twice: every occurrence is rejected
        self.assertEqual(set(r.errors), {"b_2", "c_3", "a_1"})
        self.assertIn("2 times", r.errors["a_1"][0])
        self.assertIn("split or shorten", r.errors["b_2"][0])

    def test_claims_verified_verbatim_only(self):
        o = load_offer(EX)
        ok = parse_atoms([self.good(kind="claim", text="Reusable and dishwasher safe", channel="qa")]).atoms
        bad = parse_atoms([self.good(id="b_2", kind="claim", text="Lasts a lifetime", channel="qa")]).atoms
        forb = parse_atoms([self.good(id="c_3", text="Guaranteed fresh food")]).atoms
        self.assertEqual(check_claims(ok, o), {})
        self.assertIn("UNVERIFIED_CLAIM", check_claims(bad, o)["b_2"][0])
        self.assertIn("FORBIDDEN_CLAIM", check_claims(forb, o)["c_3"][0])

    def test_verified_claim_cannot_be_compiled_as_an_image_requirement(self):
        o = load_offer(EX)
        claim = parse_atoms([self.good(kind="claim", text="Reusable and dishwasher safe")]).atoms
        errors = check_claims(claim, o)
        self.assertIn("CLAIM_CHANNEL", errors["a_1"][0])

    def test_qa_check_must_live_in_qa_channel_not_image_prompt(self):
        bad = parse_atoms([self.good(kind="qa_check", text="No added text", channel="image")])
        self.assertEqual(bad.atoms, [])
        self.assertIn("QA_CHECK_CHANNEL", bad.errors["a_1"][0])

        qa_atom = self.good(id="qa_no_text", kind="qa_check", text="No added overlays", channel="qa")
        parsed = parse_atoms([self.good(), qa_atom])
        self.assertEqual(parsed.errors, {})
        self.assertEqual(compile_image_prompt(parsed.atoms, 1).text, "A kitchen counter")

    def test_brain_does_not_author_system_owned_no_text_check(self):
        request = build_request(load_offer(EX), "brief", 1)
        rules = " ".join(request["rules"])
        self.assertIn("kind='qa_check'", rules)
        self.assertIn("channel='qa'", rules)
        self.assertIn("qa_no_text", rules)
        self.assertIn("injected and checked by code", rules)
        self.assertNotIn("allow native logos", rules)

    def test_target_product_form_factor_is_preserved(self):
        request = build_request(load_offer(EX), "brief", 1)
        rules = " ".join(request["rules"])
        self.assertIn("form factor, silhouette, proportions", rules)
        self.assertIn("Keep the target product body recognizable", rules)

    def test_source_only_hardware_does_not_transfer_to_target_product(self):
        request = build_request(load_offer(EX), "brief", 1, source_transfer={
            "format": "radar_analysis_hypothesis",
            "buyer_problem": "Dust is hard to see in dark corners",
            "product_intervention": "Source vacuum has a motorized brush, hose, and green LED",
            "result_demo": "Green LED reveals dirt beside a wall",
        })
        rules = " ".join(request["rules"])
        self.assertIn("Source-transfer context is authority only for the causal mechanism", rules)
        self.assertIn("Never copy source-only attachments, tools, features", rules)
        self.assertIn("confirmed by OfferCard/brief", rules)
        self.assertIn("omit accessories not visible", rules)

    def test_source_transfer_result_atoms_require_a_visible_before_after_boundary(self):
        request = build_request(load_offer(EX), "brief", 1,
                                source_transfer={"mechanism_hypothesis": "show removal of the problem"})
        rules = " ".join(request["rules"])
        self.assertIn("untreated/dirty side", rules)
        self.assertIn("treated/clean side", rules)
        self.assertIn("at their visible boundary", rules)
        self.assertIn("substantial, unobstructed debris-free strip", rules)


class CompilerTests(unittest.TestCase):
    def test_text_is_only_atoms_no_engine_template(self):
        atoms = [A("a", "first line"), A("b", "second line", "SHOULD")]
        c = compile_image_prompt(atoms, 1)
        self.assertEqual(c.text, "first line\nsecond line")
        self.assertEqual(c.lost_line(), "потеряно: нет")

    def test_drops_may_then_should_never_must(self):
        atoms = [A("m1", "M" * 40), A("s1", "S" * 40, "SHOULD"), A("y1", "Y" * 40, "MAY"), A("m2", "N" * 40)]
        c = compile_image_prompt(atoms, 1, limit=85)
        self.assertEqual(c.included, ("m1", "m2"))
        self.assertEqual(c.dropped, (("y1", "MAY"), ("s1", "SHOULD")))
        self.assertIn("y1 (MAY)", c.lost_line())

    def test_may_dropped_before_should(self):
        atoms = [A("s1", "S" * 40, "SHOULD"), A("y1", "Y" * 40, "MAY")]
        c = compile_image_prompt(atoms, 1, limit=45)
        self.assertEqual(c.included, ("s1",))

    def test_must_overflow_stops_with_longest_first(self):
        atoms = [A("short", "a" * 10), A("long", "b" * 60), A("mid", "c" * 30)]
        with self.assertRaises(BudgetError) as cm:
            compile_image_prompt(atoms, 1, limit=50)
        self.assertEqual(cm.exception.must_ids, ["long", "mid", "short"])

    def test_reserve_counts_against_limit(self):
        atoms = [A("a", "x" * 50)]
        with self.assertRaises(BudgetError):
            compile_image_prompt(atoms, 1, limit=60, reserve=20)

    def test_only_this_shot_and_channel(self):
        atoms = [A("a", "one", shot=1), A("b", "two", shot=2), A("c", "move", ch="motion")]
        self.assertEqual(compile_image_prompt(atoms, 1).text, "one")
        self.assertEqual(compile_motion_prompt(atoms).text, "Shot 1: move")


if __name__ == "__main__":
    unittest.main()
