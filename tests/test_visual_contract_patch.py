"""Regression checks for visual composition, system-owned no-text QA, and run diagnostics."""
import json
import tempfile
import unittest
from pathlib import Path
from cme.atoms import Atom
from cme.brain import reserved_system_atom_errors, visual_structure_errors
from cme.smoke import write_run_result
class VisualContractPatchTests(unittest.TestCase):
    def test_one_frame_requires_explicit_composition(self):
        atoms=[Atom("mech_problem_1","scene","problem","MUST","image",1),Atom("mech_result_1","action","result","MUST","image",1)]
        self.assertIn("#composition",visual_structure_errors(atoms,1,["problem","result"]))
    def test_valid_composition_passes(self):
        atoms=[{"id":"mech_problem_1","kind":"scene","priority":"MUST","channel":"image","shot":1},{"id":"mech_result_1","kind":"action","priority":"MUST","channel":"image","shot":1},{"id":"comp_split_1","kind":"scene","priority":"MUST","channel":"image","shot":1}]
        self.assertNotIn("#composition",visual_structure_errors(atoms,1,["problem","result"]))
    def test_result_is_routed_to_second_shot(self):
        atoms=[{"id":"mech_problem_1","kind":"scene","priority":"MUST","channel":"image","shot":1},{"id":"mech_intervention_1","kind":"action","priority":"MUST","channel":"image","shot":1},{"id":"mech_result_1","kind":"action","priority":"MUST","channel":"image","shot":1}]
        self.assertIn("mech_result_1",visual_structure_errors(atoms,2,["problem","intervention","result"]))
    def test_no_text_rule_is_system_owned(self):
        self.assertIn("qa_no_text_1",reserved_system_atom_errors([Atom("qa_no_text_1","qa_check","No text","MUST","qa",1)]))
    def test_run_result_is_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run_result(Path(tmp),"frame",7,"VISUAL_QA_FAILED","QA fail","FAILED_QA")
            got=json.loads((Path(tmp)/"run_result.json").read_text(encoding="utf-8"))
            self.assertEqual((got["exit_code"],got["failure_class"]),(7,"VISUAL_QA_FAILED"))
if __name__=="__main__": unittest.main()
