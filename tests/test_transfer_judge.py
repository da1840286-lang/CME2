import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cme.atoms import ATOM_SCHEMA, TRANSFER_ATOM_SCHEMA, Atom
from cme.qa import make_qa_checks


class StrictTransferSchemaTests(unittest.TestCase):
    def assert_strict_objects(self, schema):
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                self.assertEqual(set(schema.get("required", [])), set(schema.get("properties", {})))
                self.assertIs(schema.get("additionalProperties"), False)
            for value in schema.values():
                self.assert_strict_objects(value)
        elif isinstance(schema, list):
            for value in schema:
                self.assert_strict_objects(value)

    def test_normal_schema_does_not_require_transfer_decision(self):
        self.assert_strict_objects(ATOM_SCHEMA)
        self.assertEqual(ATOM_SCHEMA["required"], ["atoms"])
        self.assertNotIn("transfer_decision", ATOM_SCHEMA["properties"])

    def test_transfer_schema_requires_all_root_fields(self):
        self.assert_strict_objects(TRANSFER_ATOM_SCHEMA)
        self.assertEqual(set(TRANSFER_ATOM_SCHEMA["required"]), {"atoms", "transfer_decision"})


class TransferPreservationJudgeTests(unittest.TestCase):
    def test_preserved_property_is_an_independent_required_visual_check(self):
        carrier = Atom("mech_result_clean", "action", "A visible clean strip after vacuuming", "MUST", "image", 1)
        decision = {"transferable": True, "preserved": [{
            "property": "the same spot visibly changes from dirty to clean",
            "atom_id": "mech_result_clean",
        }]}
        seen = {}

        def judge(_path, atoms, *_refs):
            seen["atoms"] = atoms
            return {a.id: {"pass": True, "evidence": "The property is visible"} for a in atoms}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            Image.new("RGB", (128, 128), (120, 130, 140)).save(path)
            checks = make_qa_checks([carrier], 1, judge_fn=judge, ocr_required=False,
                                    transfer_decision=decision)(path)
        self.assertIn("transfer_preserved_1", {a.id for a in seen["atoms"]})
        check = next(c for c in checks if c.check_id == "transfer_preserved_1")
        self.assertTrue(check.required)
        self.assertEqual(check.status, "PASS")

    def test_missing_judge_answer_for_preserved_property_is_not_run(self):
        carrier = Atom("mech_result_clean", "action", "A visible clean strip after vacuuming", "MUST", "image", 1)
        decision = {"transferable": True, "preserved": [{
            "property": "the same spot visibly changes from dirty to clean",
            "atom_id": "mech_result_clean",
        }]}

        def judge(_path, atoms, *_refs):
            return {a.id: {"pass": True, "evidence": "Visible"} for a in atoms if a.id != "transfer_preserved_1"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            Image.new("RGB", (128, 128), (120, 130, 140)).save(path)
            checks = make_qa_checks([carrier], 1, judge_fn=judge, ocr_required=False,
                                    transfer_decision=decision)(path)
        check = next(c for c in checks if c.check_id == "transfer_preserved_1")
        self.assertTrue(check.required)
        self.assertEqual(check.status, "NOT_RUN")
