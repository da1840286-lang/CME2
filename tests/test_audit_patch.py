"""Regression tests for the audit patch: English atoms, honest PASS, mechanism coverage, strict auto-selection."""
import json
import tempfile
import unittest
from pathlib import Path

from cme.adapters.openrouter import make_openrouter_brain
from cme.atoms import Atom, check_language
from cme.brain import (BrainError, build_request, build_source_transfer_context, derive_mechanism_links,
                       missing_mechanism_links, request_atoms)
from cme.offer import load_offer
from tests.helpers import gate_account
from cme.qa import CheckResult, file_check, frame_content_check
from cme.radar.select import (analysis_review_reason, block_reason, pattern_source_reason,
                              select_analysis_candidate, select_candidate, select_pattern_candidate,
                              select_transfer_candidate)
from cme.receipt import attach_qa, build_receipt, overall
from cme.smoke import contract_id_for, main as smoke_main, offer_fingerprint, validate_atoms_manifest
from tests.helpers import png_bytes

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "offer_example.json"


def atom(i, text="A clean kitchen counter", shot=1, kind="scene", prio="MUST", ch="image"):
    return {"id": i, "kind": kind, "text": text, "priority": prio, "channel": ch, "shot": shot}


def transfer_decision(atom_id="mech_problem_seat"):
    return {"transferable": True, "abstract_mechanism": "Show a clear problem and visible improvement",
            "preserved": [{"property": "the source problem becomes a visibly improved result", "atom_id": atom_id}],
            "reason": "The target product can honestly demonstrate the same causal mechanism."}


RADAR = {"result": {
    "hook": {"text": "Visible mess", "kind": "OBSERVED"},
    "buyer_problem": {"text": "Crumbs trapped between car seats", "kind": "OBSERVED"},
    "product_intervention": {"text": "Handheld vacuum lifts crumbs from the seat gap", "kind": "OBSERVED"},
    "result_demo": {"text": "Seat gap visibly clean afterwards", "kind": "OBSERVED"},
    "mechanism_hypothesis": {"text": "Show mess, remove it, show clean", "kind": "INFERRED"},
    "transferability": {"text": "Works for other cleaning tools", "kind": "INFERRED"}}}
FULL = [atom("mech_problem_seat", "Crumbs packed into a seat gap"),
        atom("mech_intervention_nozzle", "Compact vacuum nozzle pulling crumbs from the gap"),
        atom("mech_result_clean", "The same seat gap now visibly clean"),
        atom("comp_split_1", "One-frame comparison layout with separate problem and result regions", kind="scene"),
        atom("keep_mechanism_visible_change", "Show the problem, target intervention, and resulting improvement as one visible sequence"),
        atom("keep_hook_visible_mess", "Open on the distinctive attention hook before the product action appears")]


class LanguageTests(unittest.TestCase):
    def test_non_latin_letters_are_rejected_accents_are_fine(self):
        bad = check_language([Atom("ru", "scene", "Чистый салон автомобиля", "MUST", "image", 1)])
        self.assertIn("NON_ENGLISH_ATOM", bad["ru"][0])
        self.assertEqual(check_language([Atom("ok", "scene", "A café-style counter, 3 m wide", "MUST", "image", 1)]), {})

    def test_brain_retries_a_russian_atom_and_rules_demand_english(self):
        offer = load_offer(EX)
        self.assertTrue(any("English" in r for r in build_request(offer, "b", 1)["rules"]))
        calls = []

        def call(req):
            calls.append(req)
            return {"atoms": [atom("s1_scene", "Чистый салон")]} if len(calls) == 1 else {"atoms": [atom("s1_scene")]}
        self.assertEqual(len(request_atoms(offer, "b", 1, call)), 1)
        self.assertIn("NON_ENGLISH_ATOM", json.dumps(calls[1]["only_fix"], ensure_ascii=False))
        with self.assertRaises(BrainError):
            request_atoms(offer, "b", 1, lambda r: {"atoms": [atom("s1_scene", "Чистый салон")]}, max_retries=0)


class HonestPassTests(unittest.TestCase):
    def setUp(self):
        self.atoms = [Atom("a", "scene", "A bright kitchen", "MUST", "image", 1)]
        self.r = build_receipt("c1", "r1", 1, 1, self.atoms, "A bright kitchen")

    def test_blank_frame_fails_textured_frame_passes(self):
        with tempfile.TemporaryDirectory() as d:
            blank, ok = Path(d) / "b.png", Path(d) / "o.png"
            blank.write_bytes(png_bytes(blank=True))
            ok.write_bytes(png_bytes())
            self.assertEqual(frame_content_check("frame_content", blank).status, "FAIL")
            self.assertEqual(frame_content_check("frame_content", ok).status, "PASS")
            self.assertTrue(frame_content_check("frame_content", ok).pixels)

    def test_file_integrity_plus_judge_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"
            p.write_bytes(png_bytes())
            res = [file_check("file", p), CheckResult("a", "PASS", True, "judge", "ok")]
            self.assertEqual(overall(attach_qa(self.r, self.atoms, res)), "UNVERIFIED")

    def test_pixel_check_plus_judge_passes_and_blank_frame_cannot(self):
        res = [CheckResult("frame_content", "PASS", True, "code", "", True), CheckResult("a", "PASS", True, "judge", "ok")]
        self.assertEqual(overall(attach_qa(self.r, self.atoms, res)), "PASS")
        res[0] = CheckResult("frame_content", "FAIL", True, "code", "blank", True)
        self.assertEqual(overall(attach_qa(self.r, self.atoms, res)), "FAILED_QA")

    def test_claim_match_is_not_pixel_evidence(self):
        atoms = [Atom("a", "scene", "A bright kitchen", "MUST", "image", 1),
                 Atom("c", "claim", "Cordless car vacuum", "MUST", "qa", 1)]
        r = build_receipt("c1", "r1", 1, 1, atoms, "A bright kitchen")
        res = [CheckResult("c", "PASS", True, "code", "matches verified offer claim"),
               CheckResult("a", "PASS", True, "judge", "ok")]
        self.assertEqual(overall(attach_qa(r, atoms, res)), "UNVERIFIED")


class MechanismCoverageTests(unittest.TestCase):
    def ctx(self, drop=()):
        data = json.loads(json.dumps(RADAR))
        for k in drop:
            del data["result"][k]
        return build_source_transfer_context(data)

    def test_links_are_derived_by_code_and_missing_ones_stay_unrequired(self):
        self.assertEqual([l["link"] for l in derive_mechanism_links(self.ctx())], ["problem", "intervention", "result"])
        self.assertEqual([l["link"] for l in derive_mechanism_links(self.ctx(drop=("result_demo",)))], ["problem", "intervention"])

    def test_legacy_contract_links(self):
        ctx = build_source_transfer_context({"mechanism_representation": {
            "user_problem": "Dust on the shelf", "intervention": "Wipe with the tool", "proof": ["Shelf visibly clean"]}})
        self.assertEqual([l["link"] for l in derive_mechanism_links(ctx)], ["problem", "intervention", "result"])

    def test_request_carries_links_and_openrouter_forwards_them(self):
        req = build_request(load_offer(EX), "b", 1, source_transfer=self.ctx())
        self.assertEqual(len(req["mechanism_links"]), 3)
        self.assertTrue(any("atom_id_prefix" in r for r in req["rules"]))
        seen = []

        def send(method, url, headers, payload):
            seen.append(payload)
            return 200, json.dumps({"model": payload["model"], "choices": [{"message": {"content": json.dumps({"atoms": FULL})}}]}).encode()
        make_openrouter_brain({"model_id": "v/t", "response_format_mode": "json_object"}, "k", send)(req)
        self.assertEqual(len(json.loads(seen[0]["messages"][1]["content"])["mechanism_links"]), 3)

    def test_brain_must_cover_every_link_and_only_the_missing_ones_are_retried(self):
        offer, ctx, calls = load_offer(EX), self.ctx(), []

        def call(req):
            calls.append(req)
            return ({"atoms": FULL[:1] + [atom("s1_scene")], "transfer_decision": transfer_decision()}
                    if len(calls) == 1 else {"atoms": FULL[1:], "transfer_decision": transfer_decision()})
        atoms = request_atoms(offer, "b", 1, call, source_transfer=ctx)
        self.assertEqual(missing_mechanism_links(atoms, ["problem", "intervention", "result"]), [])
        fix_ids = {f["id"] for f in calls[1]["only_fix"]}
        self.assertEqual(fix_ids, {"#mechanism_intervention", "#mechanism_result", "#composition"})

    def test_brain_fails_closed_when_a_link_is_never_carried(self):
        with self.assertRaises(BrainError) as cm:
            request_atoms(load_offer(EX), "b", 1, lambda r: {"atoms": [atom("s1_scene")]}, max_retries=1,
                          source_transfer=self.ctx())
        self.assertIn("#mechanism_problem", cm.exception.errors)

    def test_claim_or_motion_atoms_do_not_count_as_a_carrier(self):
        fake = [atom("mech_problem_x", "Cordless car vacuum", kind="claim", ch="qa"),
                atom("mech_intervention_x", "slow push in", ch="motion"),
                atom("mech_result_x", "clean seat", prio="SHOULD")]
        self.assertEqual(missing_mechanism_links(fake, ["problem", "intervention", "result"]),
                         ["problem", "intervention", "result"])

    def test_manifest_requires_target_fit_gate_and_rejects_contract_that_lost_a_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "settings.json").write_text((ROOT / "settings.example.json").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "src.json").write_text(json.dumps(RADAR), encoding="utf-8")

            fit_decision = {"transferable": True,
                            "abstract_mechanism": transfer_decision()["abstract_mechanism"],
                            "target_native_mapping": "Show a target product fixing a visible problem and prove the result",
                            "target_evidence": ["kitchen storage"],
                            "reason": "The target can demonstrate a familiar task and visible improvement",
                            **gate_account(preserved=("mechanism", "problem", "intervention", "result"), dropped=("hook",))}
            def transport(method, url, headers, payload):
                user_payload = json.loads(payload["messages"][1]["content"])
                answer = ({"atoms": [], "transfer_decision": fit_decision}
                          if user_payload.get("task") == "target_fit_gate"
                          else {"atoms": FULL, "transfer_decision": transfer_decision()})
                body = {"model": payload["model"], "choices": [{"message": {"content": json.dumps(answer)}}]}
                return 200, json.dumps(body).encode()
            out = root / "out"
            code = smoke_main(["brain", "--settings", str(root / "settings.json"), "--out", str(out),
                               "--source-contract", str(root / "src.json")],
                              {"transport": transport, "openrouter_key": "test-key"})
            self.assertEqual(code, 0)
            manifest = json.loads((out / "atoms_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mechanism_links"], ["problem", "intervention", "result"])
            offer = load_offer(EX)
            data = json.loads((out / "atoms.json").read_text(encoding="utf-8"))
            validate_atoms_manifest(out / "atoms.json", out / "atoms_manifest.json", data, offer, shot=1)

            # A saved source-bound Brain contract may not bypass the gate at frame time.
            # Exercise missing gate, explicit negative verdict and a gate/atom-mechanism mismatch.
            valid_gate = manifest["target_fit_gate"]
            invalid_gate_manifests = [
                {**manifest, "target_fit_gate": None},
                {**manifest, "target_fit_gate": {**valid_gate, "transferable": False}},
                {**manifest, "target_fit_gate": {
                    **valid_gate, "abstract_mechanism": "A different causal mechanism with no matching transfer"
                }},
            ]
            for invalid_manifest in invalid_gate_manifests:
                (out / "atoms_manifest.json").write_text(
                    json.dumps(invalid_manifest, ensure_ascii=False, indent=1), encoding="utf-8")
                with self.assertRaises(BrainError) as gate_cm:
                    validate_atoms_manifest(out / "atoms.json", out / "atoms_manifest.json", data, offer, shot=1)
                self.assertIn("#target_fit_gate", gate_cm.exception.errors)
            # Restore the valid manifest before exercising the separate missing-link regression.
            (out / "atoms_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

            # Re-bind a contract that silently dropped the result link: hashes are consistent, coverage is not.
            data = [a for a in data if a["id"] != "mech_result_clean"]
            (out / "atoms.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
            fingerprint, photos = offer_fingerprint(offer)
            import hashlib
            manifest.update({"atoms_sha256": hashlib.sha256((out / "atoms.json").read_bytes()).hexdigest(),
                             "contract_id": contract_id_for(data, offer)})
            (out / "atoms_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(BrainError) as cm:
                validate_atoms_manifest(out / "atoms.json", out / "atoms_manifest.json", data, offer, shot=1)
            self.assertIn("#mechanism", cm.exception.errors)


class AutoSelectionTests(unittest.TestCase):
    def cand(self, cid, cls="DIRECT", video="https://v", excluded=None, score=None):
        return {"id": cid, "video_url": video,
                "source_check": {"classification": cls, "excluded": excluded or []},
                "source_score": {"final_score": score, "action": "DEEP",
                                 "tags": ["money_signal_pass"],
                                 "components": {"money_signal_gate": {"pass": True, "hard_block": False}}}}

    def test_legacy_order_is_kept_and_unconfirmed_sources_are_skipped(self):
        items = [self.cand("a", "UNVERIFIED"), self.cand("b", "LOW"), self.cand("c", video=None),
                 self.cand("d", "HARD_EXCLUDE"), self.cand("e", "EXPANDED"), self.cand("f", "DIRECT")]
        chosen, skipped = select_candidate(items)
        self.assertEqual(chosen["id"], "e")   # first valid in legacy order, not re-ranked
        self.assertEqual(len(skipped), 4)

    def test_nothing_eligible_returns_none_with_reasons(self):
        chosen, skipped = select_candidate([self.cand("a", "UNVERIFIED")])
        self.assertIsNone(chosen)
        self.assertIn("UNVERIFIED", skipped[0])

    def test_excluded_hits_and_optional_score_floor(self):
        self.assertIn("hard-excluded", block_reason(self.cand("a", excluded=["weapon"])))
        self.assertEqual(block_reason(self.cand("a", score=0.8), min_final_score=0.5), "")
        self.assertIn("below", block_reason(self.cand("a", score=0.2), min_final_score=0.5))
        self.assertIn("below", block_reason(self.cand("a", score=None), min_final_score=0.5))




class PatternTransferSelectionTests(unittest.TestCase):
    def candidate(self, cid, *, relevance="EXPANDED", final=51.7, performance=56.0,
                  commercial=44.0, duration=35.934, money_pass=False, action="REJECT",
                  video="https://video.example/source.mp4", excluded=None):
        tags = ["money_signal_hard_block"] if not money_pass else ["money_signal_pass"]
        duration_block = duration is not None and duration > 60
        if duration_block:
            tags.append("duration_over_60_hard_block")
        return {
            "id": cid,
            "video_url": video,
            "source_check": {"classification": relevance, "excluded": excluded or []},
            "source_score": {
                "final_score": final,
                "performance_score": performance,
                "commercial_score": commercial,
                "action": action,
                "source_value_type": "NONE",
                "mechanism_source_score": 0,
                "tags": tags,
                "components": {
                    "money_signal_gate": {"pass": money_pass, "hard_block": not money_pass},
                    "duration": {"seconds": duration, "hard_block": duration_block},
                    "mechanism_source": {"score": 0, "eligible": False},
                },
            },
        }

    def test_failed_money_gate_does_not_block_pattern_extraction(self):
        candidate = self.candidate("tess", final=18.0, performance=6.0, commercial=5.0)
        self.assertIn("REJECT", block_reason(candidate))
        self.assertEqual(pattern_source_reason(candidate), "")
        picked, skipped = select_pattern_candidate([candidate])
        self.assertEqual(picked["id"], "tess")
        self.assertEqual(skipped, [])
        chosen, _ = select_candidate([candidate])
        self.assertIsNone(chosen)

    def test_commercially_rejected_long_source_is_not_auto_selected(self):
        candidate = self.candidate("long", relevance="DIRECT", final=0.0,
                                   performance=86.0, commercial=0.0, duration=83.5)
        self.assertEqual(pattern_source_reason(candidate), "")
        picked, skipped = select_transfer_candidate([candidate])
        self.assertIsNone(picked)
        self.assertTrue(any("REJECT" in reason or "money-signal" in reason or "duration" in reason
                            for reason in skipped))

    def test_low_or_unverified_relevance_is_not_auto_eligible(self):
        low = self.candidate("low", relevance="LOW", final=0.0, performance=0.0, commercial=0.0)
        unverified = self.candidate("unverified", relevance="UNVERIFIED", final=None,
                                    performance=None, commercial=None, action="UNKNOWN")
        for candidate in (low, unverified):
            with self.subTest(candidate=candidate["id"]):
                self.assertEqual(pattern_source_reason(candidate), "")
        picked, skipped = select_transfer_candidate([low, unverified])
        self.assertIsNone(picked)
        self.assertTrue(any("LOW" in reason for reason in skipped))
        self.assertEqual(analysis_review_reason(unverified), "")

    def test_missing_video_and_hard_exclusions_remain_blocked(self):
        no_video = self.candidate("no-video", video=None)
        excluded = self.candidate("excluded", relevance="HARD_EXCLUDE")
        explicit_exclusion = self.candidate("excluded-list", relevance="DIRECT", excluded=["unsafe"])
        for candidate in (no_video, excluded, explicit_exclusion):
            with self.subTest(candidate=candidate["id"]):
                self.assertNotEqual(pattern_source_reason(candidate), "")
        picked, skipped = select_pattern_candidate([no_video, excluded, explicit_exclusion])
        self.assertIsNone(picked)
        self.assertEqual(len(skipped), 3)

    def test_selector_skips_unusable_records_but_keeps_existing_order(self):
        blocked = self.candidate("blocked", relevance="HARD_EXCLUDE")
        missing = self.candidate("missing", video=None)
        usable = self.candidate("usable", relevance="EXPANDED", final=35, performance=25,
                                commercial=15, money_pass=True, action="DEEP")
        picked, skipped = select_transfer_candidate([blocked, missing, usable])
        self.assertEqual(picked["id"], "usable")
        self.assertEqual(len(skipped), 2)

    def test_explicit_money_pass_keeps_existing_strict_path(self):
        candidate = self.candidate("money-pass", relevance="DIRECT", final=70, performance=70,
                                   commercial=60, money_pass=True, action="DEEP")
        self.assertEqual(pattern_source_reason(candidate), "")
        chosen, _ = select_candidate([candidate])
        self.assertEqual(chosen["id"], "money-pass")


class PatternTransferWorkflowWiringTests(unittest.TestCase):
    def test_manual_analysis_and_full_pipeline_use_pattern_policy(self):
        workflow = (ROOT / ".github" / "workflows" / "live-smoke.yml").read_text(encoding="utf-8")
        self.assertIn("from cme.radar.select import pattern_source_reason", workflow)
        self.assertIn("from cme.radar.select import pattern_source_reason, select_transfer_candidate", workflow)
        self.assertIn("candidate, skipped = select_transfer_candidate(candidates, threshold)", workflow)
        self.assertNotIn("candidate, skipped = select_candidate(candidates)", workflow)
        self.assertIn("PATTERN_EXTRACTION_NOT_COMMERCIAL_APPROVAL", workflow)
        self.assertIn("Pattern extraction is not commercial approval", workflow)
        self.assertNotIn("Analysis-only source contract is not commercial approval and cannot be used by stage=all", workflow)
        self.assertIn("Automatic stage=all applies the strict source-commercial/mechanism eligibility gate before analysis", workflow)
        self.assertIn("independent Target-Fit gate before transfer", workflow)

    def test_all_can_reuse_prior_find_artifact_from_same_branch(self):
        workflow = (ROOT / ".github" / "workflows" / "live-smoke.yml").read_text(encoding="utf-8")
        self.assertIn(
            """else:
              # FIND candidate lists are source data, not compiled contracts. Reuse a
              # successful FIND artifact from an earlier commit on this same branch.
              check_run(os.environ["FIND_RUN_ID"], "cme-find-out", True, allow_prior_commit=True)""",
            workflow,
        )
        # The previous SHA may differ, but branch, success, and unexpired artifact checks remain.
        self.assertIn('if run.get("head_branch") != branch:', workflow)
        self.assertIn('if require_success and run.get("conclusion") != "success":', workflow)
        self.assertIn('if not any(a.get("name") == artifact_name and not a.get("expired") for a in artifacts):', workflow)

    def test_find_diagnostics_no_longer_describe_money_gate_as_transfer_prerequisite(self):
        find_source = (ROOT / "cme" / "radar" / "find.py").read_text(encoding="utf-8")
        self.assertNotIn("automatic use still requires the DIRECT/EXPANDED source gate", find_source)
        self.assertIn("commercial success is not required to extract a pattern", find_source)
if __name__ == "__main__":
    unittest.main()
