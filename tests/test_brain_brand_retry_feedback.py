"""Regression: a branded visual atom must be repaired, not silently discarded."""
from dataclasses import replace
import unittest
import json
from pathlib import Path

from cme.brain import request_atoms
from cme.branding import TARGET_BRAND_ALLOWED
from cme.offer import load_offer


EX = Path(__file__).resolve().parents[1] / "examples" / "offer_example.json"


class BrainBrandRetryFeedbackTests(unittest.TestCase):
    def test_brand_in_visual_atom_is_reported_and_repaired_within_four_calls(self):
        base = load_offer(EX)
        offer = replace(
            base,
            name="SEESE Cordless Pressure Washer",
            product_type="cordless pressure washer",
            branding_policy=TARGET_BRAND_ALLOWED,
            approved_brand_marks=("SEESE",),
        )
        calls = []
        rounds = []

        def model(request):
            calls.append(request)
            # Reproduce the observed four-call staircase: the model needs several
            # bounded retries before it returns the same atom without the brand.
            text = (
                "The SEESE cordless pressure washer is held above bathroom tiles."
                if len(calls) < 4
                else "A cordless pressure washer is held above bathroom tiles."
            )
            return {
                "atoms": [{
                    "id": "scene_01",
                    "kind": "scene",
                    "text": text,
                    "priority": "MUST",
                    "channel": "image",
                    "shot": 1,
                }]
            }

        atoms = request_atoms(
            offer,
            "Show the cordless pressure washer in a bathroom scene.",
            1,
            model,
            max_retries=3,
            rounds_out=rounds,
        )

        assert len(calls) == 4
        assert len(rounds) == 4
        for retry in calls[1:]:
            only_fix = json.dumps(retry.get("only_fix", []), ensure_ascii=False)
            assert "BRAND_IN_VISUAL_ATOM" in only_fix
            assert "scene_01" in only_fix
        assert "BRAND_IN_VISUAL_ATOM" in json.dumps(rounds[0]["pending"], ensure_ascii=False)
        assert [atom.id for atom in atoms] == ["scene_01"]
        assert "SEESE" not in atoms[0].text
        assert rounds[-1]["pending"] == {}
