"""Brain over API: structured output with a schema; retry only the atoms that failed.

`call` is injected (provider-specific code lives elsewhere): call(request: dict) -> dict.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
from .atoms import (ATOM_SCHEMA, TRANSFER_ATOM_SCHEMA, MODEL_GATE_REASON_MAX, GATE_REASON_MAX_TEXT, GATE_REASON_REWRITE_MAX,
                    atom_text_incomplete_error, check_claims, check_language, parse_atoms, text_boundary_rewrite_error)
from .branding import NO_BRANDING, TARGET_BRAND_ALLOWED, brain_branding_rule, resolved_brand_marks
from .vertical import check_vertical


class BrainError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"Brain output still invalid after retries: {errors}")


class PatchError(Exception):
    pass


# Narrow bridge from the older CME transfer contract into the atom authoring request.
# This is a payload adapter, not a new pipeline or atom schema.
def _pick_fields(value, keys):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in keys if key in value and value[key] not in (None, "", [], {})}


def _shot_fields(shots, keys, id_keys=("shot_id", "id")):
    if not isinstance(shots, list):
        return []
    out = []
    for index, shot in enumerate(shots[:8], start=1):
        if not isinstance(shot, dict):
            continue
        row = _pick_fields(shot, keys)
        for key in id_keys:
            if key in shot:
                row["shot_id"] = shot[key]
                break
        row.setdefault("shot_id", index)
        if row:
            out.append(row)
    return out


def build_source_transfer_context(contract: dict) -> dict:
    """Reduce an old CME contract or a saved Radar analysis to useful transfer context.

    Source URLs, raw media paths, and the original source shot prompts are deliberately
    omitted. The current OfferCard remains authoritative for target-product facts.
    """
    if not isinstance(contract, dict):
        raise ValueError("source contract must be a JSON object")

    analysis = contract.get("result") or contract.get("data") or contract
    if isinstance(analysis, dict) and ("mechanism_hypothesis" in analysis or "transferability" in analysis) and not contract.get("mechanism"):
        def statement(key):
            val = analysis.get(key)
            if isinstance(val, dict):
                return val.get("text") or val.get("value")
            return val
        def section_kind(key):
            val = analysis.get(key)
            kind = str(val.get("kind") or "").strip().upper() if isinstance(val, dict) else ""
            return kind if kind in ("OBSERVED", "INFERRED") else "UNKNOWN"
        sections = ("hook", "buyer_problem", "product_intervention", "result_demo", "cta",
                    "mechanism_hypothesis", "transferability")
        return {
            "format": "radar_analysis_hypothesis",
            "source_evidence_scope": "manual analysis of sampled frames only; not full-video verification or commercial proof",
            # OBSERVED = seen in a cited frame; INFERRED = the analyst's reading. Both used to be flattened to plain text.
            "section_kinds": {key: section_kind(key) for key in sections},
            "hook": statement("hook"),
            "buyer_problem": statement("buyer_problem"),
            "product_intervention": statement("product_intervention"),
            "result_demo": statement("result_demo"),
            "cta": statement("cta"),
            "mechanism_hypothesis": statement("mechanism_hypothesis"),
            "transferability_hypothesis": statement("transferability"),
            "limits": _pick_fields(analysis.get("limits"), ("text", "missing")),
            "evidence": analysis.get("evidence", [])[:8] if isinstance(analysis.get("evidence"), list) else [],
        }

    mechanism = contract.get("mechanism")
    representation = contract.get("mechanism_representation")
    transfer = contract.get("transfer")
    plan = contract.get("rebuild_plan")
    creative = contract.get("image_creative_contract")
    if not isinstance(creative, dict) and isinstance(plan, dict):
        nested = plan.get("image_creative_contract")
        creative = nested if isinstance(nested, dict) else {}
    creative = creative if isinstance(creative, dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    mechanism = mechanism if isinstance(mechanism, dict) else {}
    representation = representation if isinstance(representation, dict) else {}
    transfer = transfer if isinstance(transfer, dict) else {}
    if not any((mechanism, representation, transfer, plan, creative)):
        raise ValueError(
            "unsupported source contract: expected old CME mechanism/transfer/rebuild_plan/image_creative_contract "
            "or a saved Radar analysis with mechanism_hypothesis and transferability"
        )

    review = contract.get("source_review") if isinstance(contract.get("source_review"), dict) else {}
    verification = review.get("source_verification") if isinstance(review.get("source_verification"), dict) else {}
    verify_status = str(verification.get("status") or "").strip().upper()
    if verification and verification.get("required", True) and verify_status != "PASS":
        raise ValueError(
            "source contract is not accepted: required source_verification.status must be PASS "
            f"(received {verify_status or 'MISSING'})"
        )
    source_review = _pick_fields(verification, ("status", "scope", "required", "evidence"))
    # Source-context and mechanism fields explain the commercial idea; raw source URLs and
    # source video/image prompts are not sent to the Brain.
    mechanism_fields = (
        "name", "canonical_slug", "mechanism_type", "human_desire", "problem", "barrier", "tension",
        "psychological_trigger", "belief_shift", "attention_mechanism", "conversion_mechanism", "proof",
        "transferability", "transfer_notes", "source_context", "contraindications", "required_conditions",
        "mechanism_signature", "mechanism_family", "family_confidence",
    )
    representation_fields = (
        "trigger", "user_problem", "hidden_constraint", "intervention", "belief_shift", "proof",
        "desired_action", "abstraction_statement", "causal_chain", "invariant_elements",
        "source_surface_to_discard", "transfer_conditions", "failure_conditions", "abstraction_confidence",
        "purity_score", "required_conditions", "contraindications", "causal_invariants",
        "abstraction_status", "completeness_score",
    )
    transfer_fields = (
        "transfer_distance", "rationale", "preserve_rules", "change_rules", "contraindications",
        "target_context", "assessment", "reasoning_confidence", "confidence",
    )
    plan_fields = ("target_offer", "total_duration_seconds", "aspect_ratio", "strategy", "mechanism_contract",
                   "visual_strategy", "creative_direction", "funnel_bridge", "voiceover", "cta", "surface_variants")
    shot_keys = (
        "start", "end", "purpose", "scene_description", "action", "subject_motion", "camera_motion",
        "movement_direction", "motion_style", "framing", "camera_angle", "subject_placement", "visual_action",
        "proof", "state_before", "state_after", "mechanism_slots", "result_artifact", "visual_delta_from_previous",
    )
    creative_shot_keys = (
        "purpose", "scene_description", "action", "proof", "state_before", "state_after",
        "mechanism_slots", "result_artifact", "visual_delta_from_previous",
    )
    creative_fields = ("schema_version", "selected_angle", "visual_concept", "hook_frame_purpose",
                       "transfer_explanation", "negative_prompt")
    result = {
        "format": "legacy_cme_transfer_contract",
        "source_review": source_review,
        "mechanism": _pick_fields(mechanism, mechanism_fields),
        "mechanism_representation": _pick_fields(representation, representation_fields),
        "transfer": _pick_fields(transfer, transfer_fields),
        "rebuild_plan": _pick_fields(plan, plan_fields),
        "target_shots": _shot_fields(plan.get("shots"), shot_keys, id_keys=("id", "shot_id")),
        "image_creative_contract": _pick_fields(creative, creative_fields),
        "image_creative_shots": _shot_fields(creative.get("shots"), creative_shot_keys),
    }
    # Keep transfer evidence useful but bounded; do not ship unselected alternative ad scripts.
    assess = result.get("transfer", {}).get("assessment")
    if isinstance(assess, dict):
        keep_assessment = (
            "verdict", "compatibility_score", "source_vertical", "target_vertical", "shared_psychological_core",
            "source_specific_components", "target_native_reframing", "mismatch_reasons", "evidence_for_transfer",
            "required_target_conditions", "counterfactual_failures", "replacement_requirements", "causal_mapping",
            "confidence", "rationale",
        )
        result["transfer"]["assessment"] = _pick_fields(assess, keep_assessment)
    if isinstance(transfer.get("angles"), list):
        result["transfer"]["available_angle_count"] = len(transfer["angles"])
        # The selected angle is represented by rebuild_plan/image_creative_contract, so unused
        # angle scripts are intentionally not copied into the atom authoring prompt.
    return result


# ---- mechanism coverage -------------------------------------------------------------------------
# The atom schema is unchanged. A mechanism link is carried by an ordinary MUST atom whose id starts with
# `mech_<link>_`; code can then verify that every required link reached the contract, and the Judge checks
# those atoms on the frame like any other MUST atom.
MECHANISM_LINKS = ("problem", "intervention", "result")
LINK_TEXT_LIMIT = 220


def _link_text(value) -> str:
    if isinstance(value, dict):
        value = value.get("text") or value.get("value") or " ".join(str(v) for v in value.values() if isinstance(v, str))
    elif isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value if isinstance(v, (str, int, float)))
    return " ".join(str(value or "").split())[:LINK_TEXT_LIMIT]


def derive_mechanism_links(source_transfer) -> list:
    """Pick the causal links the target frames must keep: problem -> intervention -> visible result.

    Derived by code from the already-reduced source_transfer context, never by the Brain. Missing links are
    simply not required (unknown stays unknown).
    """
    if not isinstance(source_transfer, dict):
        return []
    if source_transfer.get("format") == "radar_analysis_hypothesis":
        sources = {"problem": source_transfer.get("buyer_problem"),
                   "intervention": source_transfer.get("product_intervention"),
                   "result": source_transfer.get("result_demo")}
    else:
        rep = source_transfer.get("mechanism_representation") or {}
        mech = source_transfer.get("mechanism") or {}
        sources = {"problem": rep.get("user_problem") or rep.get("trigger") or mech.get("problem"),
                   "intervention": rep.get("intervention"),
                   "result": rep.get("proof") or mech.get("proof")}
    kinds = source_transfer.get("section_kinds") if isinstance(source_transfer.get("section_kinds"), dict) else {}
    kind_key = {"problem": "buyer_problem", "intervention": "product_intervention", "result": "result_demo"}
    links = []
    for name in MECHANISM_LINKS:
        text = _link_text(sources.get(name))
        if text:
            link = {"link": name, "atom_id_prefix": f"mech_{name}_", "source_statement": text}
            if kinds:
                link["source_kind"] = kinds.get(kind_key[name], "UNKNOWN")
            links.append(link)
    return links


def source_evidence_timestamps(source_transfer) -> list:
    """Return only timestamps backed by a non-empty observation in the supplied source evidence."""
    if not isinstance(source_transfer, dict) or source_transfer.get("format") != "radar_analysis_hypothesis":
        return []
    rows = source_transfer.get("evidence")
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows[:8]:
        if not isinstance(row, dict) or not str(row.get("observation") or "").strip():
            continue
        try:
            t_sec = float(row.get("t_sec"))
        except (TypeError, ValueError):
            continue
        if t_sec >= 0 and t_sec not in out:
            out.append(t_sec)
    return out


def _atom_view(atom):
    get = atom.get if isinstance(atom, dict) else (lambda k, d=None: getattr(atom, k, d))
    return get("id") or "", get("priority"), get("channel"), get("kind")


def atom_role(atom_id):
    """Return the logical atom role for bare or underscore-suffixed IDs."""
    match = re.match(r"^(keep_[a-z]+|mech_[a-z]+|comp)(?:_|$)", str(atom_id or ""))
    return match.group(1) if match else None


def atom_id_matches_role(atom_id, role):
    """Match an exact role id or a suffix family without accepting partial words."""
    return atom_role(atom_id) == str(role or "").rstrip("_")


def atom_text_quality_errors(atoms) -> dict:
    """Reject unfinished atom text using the same message as parse_atoms()."""
    errors = {}
    for atom in atoms:
        aid, _priority, _channel, _kind = _atom_view(atom)
        text = atom.get("text", "") if isinstance(atom, dict) else getattr(atom, "text", "")
        message = atom_text_incomplete_error(text)
        if message:
            errors[str(aid)] = [message]
    return errors


def atom_target_text_policy_errors(atoms, branding_policy=TARGET_BRAND_ALLOWED) -> dict:
    """Reject model-authored image atoms that reintroduce text prohibited by campaign policy."""
    errors = {}
    for atom in atoms:
        aid, _priority, channel, kind = _atom_view(atom)
        if channel != "image" or kind == "claim":
            continue
        text = atom.get("text", "") if isinstance(atom, dict) else getattr(atom, "text", "")
        if _positive_text_display_directive(
            text, forbid_product_marks=(branding_policy == NO_BRANDING)
        ):
            errors[str(aid)] = [
                "NO_ADDED_TEXT_POLICY: image atom requests a text overlay, caption, subtitle, title card, UI, "
                "watermark or prohibited product lettering. Rewrite it as a visual-only scene."
            ]
    return errors


def missing_mechanism_links(atoms, links) -> list:
    """Link names that no MUST image atom (not a claim) visibly carries."""
    missing = []
    for link in links or []:
        prefix = link["atom_id_prefix"] if isinstance(link, dict) else f"mech_{link}_"
        name = link["link"] if isinstance(link, dict) else link
        ok = False
        for atom in atoms:
            aid, prio, ch, kind = _atom_view(atom)
            if atom_id_matches_role(aid, prefix) and prio == "MUST" and ch == "image" and kind != "claim":
                ok = True
                break
        if not ok:
            missing.append(name)
    return missing


_BRAND_TEXT_CHECK = re.compile(r"\b(logos?|brand(ing|s)?|wordmarks?|lettering|watermarks?|captions?|subtitles?|"
                               r"(no|any|without|free of)\s+(visible\s+)?(text|labels?|writing))\b", re.I)


def reserved_system_atom_errors(atoms, approved_brand_marks=()) -> dict:
    """Reject model-authored policy checks and brand names in visual atoms.

    Brand presence is checked separately against the target reference; embedding a brand in an
    atom makes the Judge treat identity as an unprovable text requirement.
    """
    errors = {}
    marks = [str(x).strip() for x in approved_brand_marks if isinstance(x, str) and str(x).strip()]
    patterns = [re.compile(r"(?<![\w])" + r"\s+".join(re.escape(part) for part in mark.split()) + r"(?![\w])", re.I)
                for mark in marks]
    for atom in atoms:
        aid, _priority, channel, kind = _atom_view(atom)
        text = atom.get("text") if isinstance(atom, dict) else getattr(atom, "text", "")
        if str(aid) == "qa_no_text" or str(aid).startswith("qa_no_text_"):
            errors[str(aid)] = ["SYSTEM_OWNED_QA: do not author qa_no_text atoms; the canonical no-text check is injected by code"]
        elif kind == "qa_check" and _BRAND_TEXT_CHECK.search(str(text or "")):
            errors[str(aid)] = ["SYSTEM_OWNED_QA: branding/text rules are owned by the campaign branding policy; "
                                "remove this atom (or re-send it as a check about something else)"]
        elif channel == "image" and any(pattern.search(str(text or "")) for pattern in patterns):
            errors[str(aid)] = ["BRAND_IN_VISUAL_ATOM: remove target/competitor brand names from visual atoms; "
                                "brand consistency is checked separately against the target-product reference"]
    return errors


def visual_structure_errors(atoms, shots: int, links, fit_gate=None) -> dict:
    """Deterministic visual-contract checks; this is structural validation, not semantic vision."""
    rows = []
    for atom in atoms:
        get = atom.get if isinstance(atom, dict) else (lambda key, default=None: getattr(atom, key, default))
        rows.append({"id": str(get("id") or ""), "kind": get("kind"), "priority": get("priority"), "channel": get("channel"), "shot": get("shot")})
    names = {str(link.get("link") if isinstance(link, dict) else link) for link in (links or [])}
    errors = {}
    for row in rows:
        if row["id"].startswith("comp_") and not (
            row["kind"] == "scene" and row["channel"] == "image" and row["priority"] == "MUST"
            and type(row["shot"]) is int and 1 <= row["shot"] <= shots
        ):
            errors[row["id"]] = ["COMPOSITION_ATOM_SHAPE: atoms with id prefix comp_ must be kind=scene, channel=image, priority=MUST, and assigned to a requested shot"]
    if shots == 1 and {"problem", "result"}.issubset(names):
        comp = [r for r in rows if r["id"].startswith("comp_") and r["kind"] == "scene" and r["channel"] == "image" and r["priority"] == "MUST" and r["shot"] == 1]
        if not comp:
            target_form = str((fit_gate or {}).get("demo_form_target") or "demonstration").strip().lower()
            if target_form in CONTRAST_FORMS:
                direction = "describe the approved in-scene contrast and place the active intervention at its visible boundary"
            else:
                direction = "describe one uninterrupted target-native demonstration and its approved visible result cue; do not invent before/after or untreated-versus-treated framing"
            errors["#composition"] = [
                "COMPOSITION_REQUIRED: shots=1 with problem+result mechanism links requires a MUST image scene atom "
                "for shot 1 whose id starts with comp_; " + direction
            ]
    if shots >= 2:
        expected_shot = {"problem": 1, "intervention": 1, "result": 2}
        for name in sorted(names.intersection(expected_shot)):
            expected = expected_shot[name]
            prefix = f"mech_{name}_"
            matching = [r for r in rows if r["id"].startswith(prefix) and r["kind"] != "claim" and r["priority"] == "MUST" and r["channel"] == "image"]
            if not any(r["shot"] == expected for r in matching):
                errors[f"#mechanism_shot_{name}"] = [f"MECHANISM_SHOT_ORDER: required MUST image link '{name}' must be assigned to shot {expected}"]
            for row in matching:
                if row["shot"] != expected:
                    errors[row["id"]] = [f"MECHANISM_SHOT_ORDER: atom '{row['id']}' carries link '{name}' but is in shot {row['shot']}; assign it to shot {expected}"]
    return errors



# Frame budget protects the image prompt from contradictory/redundant MUST requirements.
# One additional slot is reserved for the independently gated selling-idea atom. Without it,
# a one-shot contract can need product + three causal links + composition + hook + mechanism.
MAX_MUST_IMAGE_PER_SHOT = 7
REDUNDANCY_OVERLAP = 0.75
_STOPWORDS = frozenset("a an the of on in at to and or with without from into onto by for as is are be its it "
                       "this that then than same visible clearly showing show shown image frame scene product target "
                       "source shot must should may and".split())


def _content_tokens(text) -> frozenset:
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(token)
    return frozenset(tokens)


def _budget_row(atom) -> dict:
    get = atom.get if isinstance(atom, dict) else (lambda key, default=None: getattr(atom, key, default))
    return {"id": str(get("id") or ""), "kind": get("kind"), "priority": get("priority"),
            "channel": get("channel"), "shot": get("shot"), "text": str(get("text") or "")}


_COLLAGE_WORDS = re.compile(r"\b(split[- ]?(frame|screen)|collage|grids?|panels?|side[- ]by[- ]side|diptych|triptych|"
                            r"before[/ -]and[/ -]after (layout|collage))\b", re.I)


def atom_budget_errors(atoms, shots: int, fit_gate=None) -> dict:
    """Return per-atom fixes and frame-level conflicts for redundant or over-budget MUST atoms."""
    rows = [_budget_row(a) for a in atoms]
    errors = {}
    contrast_allowed = (
        isinstance(fit_gate, dict)
        and str(fit_gate.get("demo_form_target") or "").strip().lower() in CONTRAST_FORMS
    )
    for row in rows:
        if row["channel"] == "image" and _COLLAGE_WORDS.search(row["text"]):
            if contrast_allowed:
                direction = (
                    "Keep it one continuous in-scene contrast, with the active intervention at the visible boundary; "
                    "do not use panels, grids or a multi-panel collage."
                )
            else:
                direction = (
                    "Use one uninterrupted target-native demonstration and its approved visible result cue; do not "
                    "introduce left/right zones, before/after states or an untreated-versus-treated comparison."
                )
            errors[row["id"]] = [
                "FRAME_COLLAGE_WORDING: do not write split frame / split-screen / panel / collage / grid / side-by-side; "
                "describe one continuous scene instead. " + direction
            ]
    for shot in range(1, int(shots) + 1):
        must = [r for r in rows if r["shot"] == shot and r["channel"] == "image" and r["priority"] == "MUST"
                and r["id"] not in errors]

        def protected(row):
            return row["id"].startswith(("mech_", "comp_", "keep_"))

        layout = [r for r in must if r["kind"] == "composition" or r["id"].startswith(("comp_", "composition"))]
        layout.sort(key=lambda r: (not r["id"].startswith("comp_"), not r["id"].startswith("mech_")))
        for extra in layout[1:]:
            if not protected(extra):
                errors[extra["id"]] = [
                    f"FRAME_LAYOUT_CONFLICT: shot {shot} already has layout atom '{layout[0]['id']}'; one frame gets ONE "
                    f"layout. Merge this into it or re-send '{extra['id']}' with priority='SHOULD'."
                ]

        kept = []
        ordered = sorted((r for r in must if r["id"] not in errors), key=lambda r: not r["id"].startswith("mech_"))
        for row in ordered:
            tokens = _content_tokens(row["text"])
            duplicate, best = None, 0.0
            if len(tokens) >= 4:
                for old, old_tokens in kept:
                    if len(old_tokens) < 4:
                        continue
                    score = len(tokens & old_tokens) / min(len(tokens), len(old_tokens))
                    if score >= REDUNDANCY_OVERLAP and score > best:
                        duplicate, best = old["id"], score
            if duplicate is not None and not protected(row):
                errors[row["id"]] = [
                    f"FRAME_REDUNDANT: '{row['id']}' repeats requirement '{duplicate}' in shot {shot}. "
                    "Re-send it with priority='SHOULD' or describe a different visible requirement."
                ]
            else:
                kept.append((row, tokens))

        remaining = [r for r in must if r["id"] not in errors]
        if len(remaining) > MAX_MUST_IMAGE_PER_SHOT:
            errors[f"#must_budget_{shot}"] = [
                f"FRAME_BUDGET: shot {shot} has {len(remaining)} MUST image atoms "
                f"(limit {MAX_MUST_IMAGE_PER_SHOT}). Re-send the least essential ones with priority='SHOULD'; "
                "keep the mechanism and target-product requirements."
            ]
    return errors


class TransferRejected(BrainError):
    """The source mechanism cannot be shown honestly with the selected target product."""

    def __init__(self, decision: dict):
        self.decision = dict(decision or {})
        super().__init__({"#transfer": [f"TRANSFER_REJECTED: {self.decision.get('reason', '')}"]})


_CYRILLIC = re.compile("[\\u0400-\\u04ff]")

# The transfer gate runs before atom authoring. It intentionally shares the Brain
# provider, but uses a separate strict contract so an invalid atom budget cannot hide
# whether the source mechanism is transferable. The gate never returns production atoms.
GATE_ELEMENTS = ("hook", "problem", "mechanism", "intervention", "result")
# A hook can be adapted or dropped, but the causal mechanism itself cannot be dropped on a passing transfer.
GATE_CORE_ELEMENTS = ("mechanism",)
DEMO_FORMS = ("comparison", "before_after", "demonstration", "testimonial", "tutorial", "other")
# Forms that carry their persuasion through contrast. They must stay contrast-shaped in the target.
CONTRAST_FORMS = ("comparison", "before_after")

_CONTRAST_LAYOUT_CUE = re.compile(
    r"\b(?:before\s*(?:and|/|-)\s*after|side[- ]by[- ]side|comparison|compare|"
    r"versus|vs\.?|split[- ]?(?:screen|frame)|diptych|triptych)\b",
    re.I,
)
_UNTREATED_STATE = re.compile(r"\b(?:untreated|dirty|grimy|soiled|unclean)\b", re.I)
_TREATED_STATE = re.compile(r"\b(?:treated|clean|cleaned|foam[- ]coated)\b", re.I)


def composition_form_errors(atoms, fit_gate) -> dict:
    """Reject contrast layouts unless the independent Target-Fit Gate approved them."""
    if not isinstance(fit_gate, dict):
        return {}
    target_form = str(fit_gate.get("demo_form_target") or "").strip().lower()
    if target_form in CONTRAST_FORMS:
        return {}

    errors = {}
    for atom in atoms:
        aid, priority, channel, _kind = _atom_view(atom)
        if priority != "MUST" or channel != "image":
            continue
        text = atom.get("text", "") if isinstance(atom, dict) else getattr(atom, "text", "")
        value = str(text or "")
        sides = {side for side in ("left", "right") if re.search(rf"\b{side}\b", value, re.I)}
        is_composition_atom = str(aid).startswith("comp_")
        state_contrast = (
            sides == {"left", "right"}
            and (
                is_composition_atom
                or (bool(_UNTREATED_STATE.search(value)) and bool(_TREATED_STATE.search(value)))
            )
        )
        if _CONTRAST_LAYOUT_CUE.search(value) or state_contrast:
            errors[str(aid)] = [
                f"DEMO_FORM_CONFLICT: Target-Fit Gate approved demo_form_target='{target_form or 'unspecified'}', "
                "not a contrast. Remove before/after, side-by-side, or untreated-left/treated-right framing; "
                "show one uninterrupted target-native demonstration and only its approved visible result cue."
            ]
    return errors


TRANSFER_FIT_SCHEMA = {
    "type": "object",
    "required": ["atoms", "transfer_decision"],
    "additionalProperties": False,
    "properties": {
        "atoms": {"type": "array", "maxItems": 0},
        "transfer_decision": {
            "type": "object",
            "required": ["transferable", "abstract_mechanism", "target_native_mapping", "target_evidence", "reason",
                         "demo_form_source", "demo_form_target",
                         "source_elements_preserved", "source_elements_dropped"],
            "additionalProperties": False,
            "properties": {
                "transferable": {"type": "boolean"},
                "abstract_mechanism": {"type": "string", "minLength": 10, "maxLength": 200},
                "target_native_mapping": {"type": "string", "minLength": 10, "maxLength": 240},
                "target_evidence": {"type": "array", "maxItems": 3, "items": {"type": "string", "minLength": 1, "maxLength": 120}},
                "reason": {"type": "string", "minLength": 10, "maxLength": MODEL_GATE_REASON_MAX},
                "demo_form_source": {"type": "string", "enum": list(DEMO_FORMS)},
                "demo_form_target": {"type": "string", "enum": list(DEMO_FORMS)},
                "source_elements_preserved": {
                    "type": "array", "maxItems": 5,
                    "items": {
                        "type": "object",
                        "required": ["element", "source_form", "target_form", "evidence_t_sec"],
                        "additionalProperties": False,
                        "properties": {
                            "element": {"type": "string", "enum": list(GATE_ELEMENTS)},
                            "source_form": {"type": "string", "minLength": 8, "maxLength": 160},
                            "target_form": {"type": "string", "minLength": 8, "maxLength": 160},
                            "evidence_t_sec": {"type": "array", "maxItems": 3,
                                               "items": {"type": "number", "minimum": 0}},
                        },
                    },
                },
                "source_elements_dropped": {
                    "type": "array", "maxItems": 5,
                    "items": {
                        "type": "object",
                        "required": ["element", "why"],
                        "additionalProperties": False,
                        "properties": {
                            "element": {"type": "string", "enum": list(GATE_ELEMENTS)},
                            "why": {"type": "string", "minLength": 8, "maxLength": 160},
                        },
                    },
                },
            },
        },
    },
}

GATE_DECISION_KEYS = ("transferable", "abstract_mechanism", "target_native_mapping", "target_evidence", "reason",
                      "demo_form_source", "demo_form_target", "source_elements_preserved", "source_elements_dropped")


def _source_text(value) -> str:
    if isinstance(value, dict):
        value = value.get("text") or value.get("value")
    return value.strip() if isinstance(value, str) else ""


def required_gate_elements(source_transfer) -> tuple:
    """Source elements the gate must account for (preserved or dropped). The mechanism is always required;
    the others only when the source analysis actually states them (what is absent is not demanded)."""
    st = source_transfer if isinstance(source_transfer, dict) else {}
    rep = st.get("mechanism_representation") if isinstance(st.get("mechanism_representation"), dict) else {}
    present = {"mechanism": True}
    present["hook"] = bool(_source_text(st.get("hook")) or _source_text(rep.get("trigger")))
    present["problem"] = bool(_source_text(st.get("buyer_problem")) or _source_text(rep.get("user_problem")))
    present["intervention"] = bool(_source_text(st.get("product_intervention")) or _source_text(rep.get("intervention")))
    present["result"] = bool(_source_text(st.get("result_demo")) or _source_text(rep.get("proof")))
    return tuple(name for name in GATE_ELEMENTS if present.get(name))


_TARGET_TEXT_DISPLAY_DIRECTIVE = re.compile(
    r"\b(?:text\s+overlays?|overlay\s+text|on[- ]screen\s+text|captions?|subtitles?|"
    r"title\s+cards?|speech\s+bubbles?|watermarks?|promotional\s+(?:text|wording|copy)|"
    r"written\s+(?:words?|text)|(?:text|words?)\s+(?:reading|saying)|"
    r"readable\s+(?:text|lettering)|screen\s+text|ui\s+text)\b",
    re.I,
)
_TARGET_MARK_DISPLAY_DIRECTIVE = re.compile(
    r"\b(?:logos?|brand\s+names?|wordmarks?|product\s+labels?|readable\s+lettering|"
    r"printed\s+(?:words?|text)|engraved\s+(?:words?|text))\b",
    re.I,
)
_NEGATED_TEXT_DISPLAY_DIRECTIVE = re.compile(
    r"\b(?:no|without|not|never|avoid(?:s|ed|ing)?|omit(?:s|ted|ting)?|remove(?:s|d)?|"
    r"exclude(?:s|d)?|prohibit(?:s|ed)?|forbid(?:s|den)?|do\s+not|must\s+not|should\s+not)\s+"
    r"(?:(?:any|added|visible|readable|on[- ]screen|promotional)\s+)*"
    r"(?:text\s+overlays?|overlay\s+text|on[- ]screen\s+text|captions?|subtitles?|"
    r"title\s+cards?|speech\s+bubbles?|watermarks?|promotional\s+(?:text|wording|copy)|"
    r"written\s+(?:words?|text)|screen\s+text|ui\s+text|logos?|brand\s+names?|"
    r"wordmarks?|product\s+labels?|lettering)\b",
    re.I,
)
_NEGATED_TEXT_DISPLAY_AFTER = re.compile(
    r"^\s*(?:(?:must|should|will|is|are|be|being)\s+)?not\s+"
    r"(?:be\s+)?(?:shown|used|added|included|present|visible|rendered|displayed)\b",
    re.I,
)


def _positive_text_display_directive(value: str, *, forbid_product_marks: bool = False) -> bool:
    """True only for an affirmative target request to render text or prohibited product marks."""
    text = str(value or "")
    patterns = [_TARGET_TEXT_DISPLAY_DIRECTIVE]
    if forbid_product_marks:
        patterns.append(_TARGET_MARK_DISPLAY_DIRECTIVE)
    for pattern in patterns:
        for match in pattern.finditer(text):
            clause_start = max(text.rfind(".", 0, match.start()),
                               text.rfind("!", 0, match.start()),
                               text.rfind("?", 0, match.start()),
                               text.rfind(";", 0, match.start()))
            prefix_through_match = text[clause_start + 1:match.end()]
            if _NEGATED_TEXT_DISPLAY_DIRECTIVE.search(prefix_through_match):
                continue
            suffix = text[match.end():match.end() + 55]
            if _NEGATED_TEXT_DISPLAY_AFTER.search(suffix):
                continue
            return True
    return False


def validate_gate_elements(decision, source_transfer=None, branding_policy=TARGET_BRAND_ALLOWED) -> list:
    """Structure of the preserved/dropped account. Shape and bookkeeping only; meaning stays with the model."""
    errors = []
    for key in ("demo_form_source", "demo_form_target"):
        if decision.get(key) not in DEMO_FORMS:
            errors.append(f"transfer_decision.{key} must be one of {', '.join(DEMO_FORMS)}")
    seen = {}
    available_evidence = source_evidence_timestamps(source_transfer)
    for list_key, text_keys in (("source_elements_preserved", ("source_form", "target_form")),
                                ("source_elements_dropped", ("why",))):
        items = decision.get(list_key)
        if not isinstance(items, list) or len(items) > len(GATE_ELEMENTS):
            errors.append(f"transfer_decision.{list_key} must be a list of at most {len(GATE_ELEMENTS)} entries")
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"transfer_decision.{list_key}[{index}] must be an object")
                continue
            element = item.get("element")
            if element not in GATE_ELEMENTS:
                errors.append(f"transfer_decision.{list_key}[{index}].element must be one of {', '.join(GATE_ELEMENTS)}")
                continue
            if element in seen:
                errors.append(f"transfer_decision: element '{element}' appears more than once across preserved/dropped")
            seen[element] = list_key
            for text_key in text_keys:
                value = item.get(text_key)
                if not isinstance(value, str) or not 8 <= len(value.strip()) <= 160:
                    errors.append(f"transfer_decision.{list_key}[{index}].{text_key} must contain 8-160 characters")
                elif _CYRILLIC.search(value):
                    errors.append(f"transfer_decision.{list_key}[{index}].{text_key} must be in English")
                elif list_key == "source_elements_preserved" and text_key == "target_form":
                    if len(_keep_content_words(value)) < 2:
                        errors.append(f"transfer_decision.{list_key}[{index}].target_form must specify at least two meaningful visual or causal details")
                    source_form = item.get("source_form")
                    if isinstance(source_form, str):
                        source_words = _target_form_content_words(source_form)
                        target_words = _target_form_content_words(value)
                        if source_words and not (target_words - source_words):
                            errors.append(
                                f"transfer_decision.{list_key}[{index}].target_form must record a target-specific "
                                "adaptation instead of repeating the source form; name the changed target context or action"
                            )
                    if _positive_text_display_directive(
                        value, forbid_product_marks=(branding_policy == NO_BRANDING)
                    ):
                        errors.append(
                            f"NO_ADDED_TEXT_POLICY: transfer_decision.{list_key}[{index}].target_form requests "
                            "visible text, captions, overlays, UI, watermarks or prohibited product lettering; "
                            "adapt the source hook to a visual-only target form or account for it as dropped"
                        )
            if list_key == "source_elements_preserved":
                if "evidence_t_sec" not in item:
                    errors.append(f"transfer_decision.{list_key}[{index}].evidence_t_sec is required (use [] only when source frame observations are unavailable)")
                refs = item.get("evidence_t_sec", [])
                if not isinstance(refs, list) or len(refs) > 3 or any(type(t) not in (int, float) or t < 0 for t in (refs or [])):
                    errors.append(f"transfer_decision.{list_key}[{index}].evidence_t_sec must be a list of 0-3 non-negative source timestamps")
                elif available_evidence:
                    if not refs:
                        errors.append(f"transfer_decision.{list_key}[{index}].evidence_t_sec must cite at least one supplied source frame timestamp")
                    for t in refs:
                        if not any(abs(float(t) - known) <= 0.05 for known in available_evidence):
                            errors.append(f"transfer_decision.{list_key}[{index}].evidence_t_sec contains {t}, which is not a timestamp with a supplied source observation")
    required = set(required_gate_elements(source_transfer))
    for element in sorted(required - set(seen)):
        errors.append(f"transfer_decision must account for source element '{element}' as preserved or dropped")
    if isinstance(source_transfer, dict):
        for element in sorted(set(seen) - required):
            errors.append(
                f"transfer_decision accounts for source element '{element}' that is not present in the supplied source contract"
            )
    return errors


def core_preservation_verdict(decision) -> str:
    """Empty string only when a transferable decision keeps the source's causal mechanism.

    A non-empty string is the code-enforced reason to reject the transfer. The model's own 'transferable: true'
    cannot override it: dropping the causal mechanism or turning a contrast demo into an ordinary demo means the
    result is a different ad with the target product in it.
    """
    preserved = {item.get("element") for item in decision.get("source_elements_preserved") or [] if isinstance(item, dict)}
    if "mechanism" not in preserved:
        return ("TRANSFER_MECHANISM_DROPPED: the source's causal mechanism is not preserved; "
                "preserving only a hook or visual style is not a mechanism transfer")
    source_form, target_form = decision.get("demo_form_source"), decision.get("demo_form_target")
    if source_form in CONTRAST_FORMS and target_form not in CONTRAST_FORMS:
        return (f"TRANSFER_FORM_DOWNGRADED: the source persuades through {source_form}, but the target form is "
                f"{target_form}; a comparison must stay a comparison (for example this product against the old way)")
    return ""


def required_keep_elements(fit_gate) -> list:
    """Core source elements explicitly approved by the gate that must reach an image atom."""
    if not isinstance(fit_gate, dict) or fit_gate.get("transferable") is not True:
        return []
    preserved = {
        item.get("element") for item in (fit_gate.get("source_elements_preserved") or [])
        if isinstance(item, dict)
    }
    # A pass requires the mechanism by code. A hook needs its own visible carrier only when the gate says it is kept.
    required = ["mechanism"] if "mechanism" in preserved else []
    if "hook" in preserved:
        required.append("hook")
    return required


_KEEP_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "then", "into", "onto", "over", "under",
    "using", "use", "show", "shows", "shown", "visible", "visibly", "clear", "same", "target", "source",
    "product", "element", "mechanism", "hook", "form", "frame", "scene", "make", "makes", "made",
    "before", "after", "there", "their", "they", "them", "where", "while", "when", "what", "which",
    "from", "onto", "into", "onto", "can", "will", "must", "should", "would", "does", "done", "over",
}


def _keep_content_words(value: str) -> set[str]:
    """Extract specific visual/causal terms, excluding generic instruction words."""
    words = re.findall(r"[a-z][a-z0-9'-]{2,}", str(value or "").casefold())
    return {word.strip("'-") for word in words if len(word.strip("'-")) >= 3 and word not in _KEEP_STOPWORDS}


def _target_form_content_words(value: str) -> set[str]:
    """Keep specific words for source-vs-target adaptation checks; less aggressive than atom matching."""
    stop = {"the", "and", "for", "with", "from", "into", "onto", "that", "this", "then", "when",
            "while", "before", "after", "as", "at", "by", "in", "on", "of", "to", "a", "an",
            "is", "are", "be", "was", "were", "or", "but"}
    return {word for word in re.findall(r"[a-z][a-z0-9'-]{2,}", str(value or "").casefold()) if word not in stop}


def _keep_atom_matches_target_form(atom_text: str, target_form: str) -> bool:
    """Require the MUST atom to carry meaningful details from the Gate's approved target form.

    IDs alone are not evidence: a generic product beauty-shot atom must not satisfy the mechanism contract.
    This lexical check is deliberately conservative and deterministic; semantic correctness still needs live QA.
    """
    expected = _keep_content_words(target_form)
    actual = _keep_content_words(atom_text)
    if not expected:
        return False
    needed = min(3, max(2, int((len(expected) * 0.30) + 0.999))) if len(expected) > 1 else 1
    return len(expected & actual) >= needed


def missing_keep_atoms(atoms, fit_gate) -> list:
    """Elements that lack a MUST image carrier matching the Gate-approved target form."""
    missing = []
    rows = []
    for atom in atoms:
        aid, priority, channel, kind = _atom_view(atom)
        text = atom.get("text", "") if isinstance(atom, dict) else getattr(atom, "text", "")
        rows.append((aid, priority, channel, kind, text))
    preserved = {
        item.get("element"): str(item.get("target_form") or "")
        for item in (fit_gate.get("source_elements_preserved") or [])
        if isinstance(item, dict)
    } if isinstance(fit_gate, dict) else {}
    for element in required_keep_elements(fit_gate):
        role = f"keep_{element}"
        target_form = preserved.get(element, "")
        if not any(atom_id_matches_role(aid, role) and priority == "MUST" and channel == "image" and kind != "claim"
                   and _keep_atom_matches_target_form(text, target_form)
                   for aid, priority, channel, kind, text in rows):
            missing.append(element)
    return missing


def _normalized_evidence(value: str) -> str:
    """Normalize punctuation/case for exact, ordered evidence matching against the OfferCard."""
    return re.sub(r"[^\w]+", " ", str(value or "").casefold(), flags=re.UNICODE).strip()


# Only capability markers that add a feature beyond ordinary product-class use.
# This intentionally stays conservative: scene nouns (crumbs, tiles, pass, result) and generic
# adjectives must never turn an ordinary demo into an unsupported special-capability claim.
_GATE_SPECIAL_CAPABILITY_TERMS = frozenset((
    "foam", "detergent", "soap", "steam", "heated", "heating", "cooling", "chilled",
    "ultraviolet", "ozone", "sterilize", "sterilization", "disinfect", "disinfection",
    "sanitize", "sanitization", "aroma", "scent", "inflate", "inflation", "deflate",
    "blower", "blow", "polish", "polishing", "magnetic", "wireless", "waterproof",
    "self-clean", "self-cleaning", "automatic", "autonomous", "rotating", "oscillating",
))

_GATE_CAPABILITY_STOPWORDS = frozenset((
    "a an the and or with without to onto into on in at of for from by as is are be was were "
    "show shows showing visibly visible observable target product class type same ordinary normal "
    "creator person user hand held holding use uses using used perform performs performing action "
    "demonstrate demonstrates demonstration apply applies applying applied spray sprays spraying "
    "clean cleans cleaning cleaned wash washes washing scrub scrubs scrubbing manual effort result proof "
    "surface surfaces area areas bathroom outdoor indoor tile tiles wall walls floor floors "
    "fixture fixtures shower caddie caddies tub sink countertop counter patio furniture scene frame "
    "without manual work hard hard-surface exact specific actual"
).split())


def _gate_capability_terms(value: str, excluded: set[str] | None = None) -> set[str]:
    """Extract concrete target-function terms; ignore scene wording and product identity."""
    excluded = excluded or set()
    out = set()
    for raw in re.findall(r"[a-z][a-z0-9'-]{2,}", str(value or "").casefold()):
        token = raw.strip("'-")
        if not token or token in _GATE_CAPABILITY_STOPWORDS or token in excluded:
            continue
        # Small inflection normalizer for capability words (foam/foaming, sprays/spraying).
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            token = token[:-1]
        if token and token not in _GATE_CAPABILITY_STOPWORDS and token not in excluded:
            out.add(token)
    return out


def _target_capability_evidence_errors(offer, decision: dict, evidence: list[str]) -> list[str]:
    """A product name/category cannot prove an added function such as foam spraying.

    Capability evidence must be a verbatim excerpt from a verified claim or confirmation and
    share a concrete function term with the Gate's approved target-native mechanism. The model
    never receives the photo pixels at this gate; a visible accessory alone is not functional proof.
    """
    preserved = decision.get("source_elements_preserved")
    preserved = preserved if isinstance(preserved, list) else []
    if not any(isinstance(row, dict) and row.get("element") in {"mechanism", "intervention"}
               for row in preserved):
        return []

    identifiers = set()
    for field in (getattr(offer, "name", ""), getattr(offer, "product_type", ""),
                  getattr(offer, "product_category", "")):
        identifiers.update(_gate_capability_terms(field))

    target_forms = [str(row.get("target_form") or "") for row in preserved
                    if isinstance(row, dict) and row.get("element") in {"mechanism", "intervention"}]
    target_function_terms = _gate_capability_terms(
        str(decision.get("target_native_mapping") or "") + " " + " ".join(target_forms), identifiers
    ) & _GATE_SPECIAL_CAPABILITY_TERMS
    # If the target form adds a concrete feature beyond the product identity, that feature must
    # be grounded in an explicit OfferCard claim/confirmation. Generic scene nouns are filtered above.
    if not target_function_terms:
        return []

    functional_fields = [str(x) for x in (
        *getattr(offer, "verified_claims", ()), *getattr(offer, "confirmations", ())
    ) if str(x or "").strip()]
    matching_quotes = [item for item in evidence if any(
        " ".join(item.split()).casefold() in " ".join(field.split()).casefold()
        for field in functional_fields
    )]
    evidence_terms = (set().union(*(_gate_capability_terms(item, identifiers) for item in matching_quotes))
                      if matching_quotes else set()) & _GATE_SPECIAL_CAPABILITY_TERMS
    covered = target_function_terms & evidence_terms
    if covered:
        return []

    missing = ", ".join(sorted(target_function_terms)[:8])
    return [
        "TARGET_CAPABILITY_UNVERIFIED: transferable=true preserves a target function that is not "
        "supported by a relevant verified_claim or confirmation. target_evidence must quote a "
        "capability-specific OfferCard claim/confirmation sharing the function term(s) "
        f"[{missing}]. Product name/type/category and a visible accessory alone do not establish "
        "that function; otherwise narrow the target mapping to a separately confirmed function "
        "or set transferable=false."
    ]


_GATE_UNCERTAINTY = re.compile(
    r"\b(?:plausibl\w*|probably|presumably|likely|may\s+be\s+able\s+to|"
    r"should\s+be\s+able\s+to|could\s+possibly)\b", re.IGNORECASE
)


def validate_transfer_fit_decision(offer, decision, source_transfer=None) -> list:
    """Fail closed unless the transfer verdict is well-formed and its target evidence is real."""
    if not isinstance(decision, dict):
        return ["transfer-fit decision is missing or is not an object"]
    errors = []
    if type(decision.get("transferable")) is not bool:
        errors.append("transfer_decision.transferable must be a boolean")
    for key, minimum, maximum in (("abstract_mechanism", 10, 200),
                                  ("target_native_mapping", 10, 240)):
        value = decision.get(key)
        if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
            errors.append(f"transfer_decision.{key} must contain {minimum}-{maximum} characters")
        elif _CYRILLIC.search(value):
            errors.append(f"transfer_decision.{key} must be in English")
    reason = decision.get("reason")
    if not isinstance(reason, str) or len(reason.strip()) < 10:
        errors.append("transfer_decision.reason must contain at least 10 characters")
    else:
        reason_error = text_boundary_rewrite_error(
            reason,
            field="target_fit_gate.reason",
            hard_limit=GATE_REASON_MAX_TEXT,
            rewrite_limit=GATE_REASON_REWRITE_MAX,
            allow_exact_limit=False,
            require_terminal_punctuation=True,
        )
        if reason_error:
            errors.append(reason_error)
        if _CYRILLIC.search(reason):
            errors.append("transfer_decision.reason must be in English")
        if decision.get("transferable") is True:
            preserved_rows = decision.get("source_elements_preserved")
            preserved_rows = preserved_rows if isinstance(preserved_rows, list) else []
            gate_text = " ".join([
                reason,
                str(decision.get("abstract_mechanism") or ""),
                str(decision.get("target_native_mapping") or ""),
                " ".join(str(row.get("target_form") or "") for row in preserved_rows if isinstance(row, dict)),
            ])
            if _GATE_UNCERTAINTY.search(gate_text):
                errors.append(
                    "TARGET_CAPABILITY_UNCERTAIN: a transferable decision cannot rely on plausibility or "
                    "probability language; cite a verified target capability or reject/narrow the transfer"
                )
    if _positive_text_display_directive(
        str(decision.get("target_native_mapping") or ""),
        forbid_product_marks=(getattr(offer, "branding_policy", TARGET_BRAND_ALLOWED) == NO_BRANDING),
    ):
        errors.append(
            "NO_ADDED_TEXT_POLICY: transfer_decision.target_native_mapping requests visible text, captions, "
            "overlays, UI, watermarks or prohibited product lettering; describe a visual-only target hook"
        )
    evidence = decision.get("target_evidence")
    if not isinstance(evidence, list) or len(evidence) > 3 or any(
        not isinstance(item, str) or not 1 <= len(item.strip()) <= 120 for item in (evidence or [])
    ):
        errors.append(
            "transfer_decision.target_evidence must be a list of 0-3 evidence excerpts, each 1-120 characters; "
            "excerpts shorter than 8 characters must exactly match a complete authoritative OfferCard field"
        )
        evidence = []
    if decision.get("transferable") is True and not evidence:
        errors.append("a transferable decision requires target_evidence copied from the OfferCard")

    allowed = [getattr(offer, "name", ""), getattr(offer, "product_type", ""),
               getattr(offer, "product_category", ""), *getattr(offer, "verified_claims", ()),
               *getattr(offer, "confirmations", ())]
    # Whitespace may differ at the edges, but punctuation and wording must remain verbatim.
    def verbatim_normalize(value):
        return " ".join(str(value or "").split()).casefold()
    allowed_normalized = [verbatim_normalize(item) for item in allowed if str(item or "").strip()]
    allowed_exact = set(allowed_normalized)
    for index, item in enumerate(evidence):
        if not isinstance(item, str):
            continue
        needle = verbatim_normalize(item)
        if not needle or not any(needle in haystack for haystack in allowed_normalized):
            errors.append(
                f"transfer_decision.target_evidence[{index}] is not present in authoritative OfferCard fields "
                "as a verbatim contiguous excerpt from one authoritative OfferCard field"
            )
        elif len(item.strip()) < 8 and needle not in allowed_exact:
            errors.append(
                f"transfer_decision.target_evidence[{index}] is shorter than 8 characters and must exactly match "
                "a complete authoritative OfferCard field"
            )
    if decision.get("transferable") is True:
        errors.extend(_target_capability_evidence_errors(offer, decision, evidence))
    errors.extend(validate_gate_elements(
        decision,
        source_transfer,
        branding_policy=getattr(offer, "branding_policy", TARGET_BRAND_ALLOWED),
    ))
    return errors


def _canonical_sha256(value) -> str:
    """Hash a JSON-compatible request component with stable key ordering."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _request_provenance(request: dict) -> dict:
    """Fingerprint the exact CME schema and structured prompt payload for one model round."""
    schema = request.get("schema")
    prompt_payload = {key: value for key, value in request.items() if key != "schema"}
    return {
        "schema_sha256": _canonical_sha256(schema),
        "prompt_payload_sha256": _canonical_sha256(prompt_payload),
        "request_sha256": _canonical_sha256(request),
    }


def _call_provenance(call) -> dict:
    """Capture primary/selected adapter identity and the provider-reported responding model."""
    info = getattr(call, "info", {}) or {}
    return {
        "primary_adapter": info.get("primary_adapter"),
        "primary_model_id": info.get("primary_model_id"),
        "adapter": info.get("adapter"),
        "provider": info.get("provider"),
        "model_id": info.get("model_id"),
        "responded_model": info.get("responded_model"),
        "fallback_used": bool(info.get("fallback_used", False)),
    }


def _trace_excerpt(value, limit: int = 3500) -> str:
    """Bounded JSON text of a model response for the audit trace (no secrets pass through Brain responses)."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + f"...[+{len(text) - limit} chars]"


def _source_evidence_content_tokens(text) -> set[str]:
    """Category-agnostic terms from a source-frame observation or a gate's denial claim."""
    stop = {
        "the", "and", "for", "from", "with", "without", "that", "this", "there", "their",
        "source", "evidence", "observation", "observations", "frame", "frames", "show", "shows",
        "shown", "display", "displays", "visible", "visibly", "clearly", "actually", "were",
        "was", "are", "is", "not", "no", "never", "cannot", "does", "doesnt", "did", "didnt",
        "wasnt", "werent", "into", "onto", "then", "than", "when", "where", "which", "what",
        "have", "has", "had", "been", "being", "only", "also", "just", "product",
        "tool", "item", "items", "object", "objects", "action", "actions", "result", "results",
        "mechanism", "intervention", "demo", "demonstration", "happened", "happens", "happen",
        "про", "это", "что", "как", "для", "или", "без", "при", "был", "была", "были",
        "есть", "нет", "не", "видно", "показано", "кадр", "кадры", "источник", "действие",
        "результат", "товар", "продукт", "инструмент",
    }
    tokens = re.findall(r"[a-z][a-z0-9'-]{2,}|[а-яё][а-яё0-9'-]{2,}", str(text or "").casefold())
    out = set()
    for token in tokens:
        token = token.strip("'-")
        if token in stop:
            continue
        roots = {token}
        for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                roots.add(token[:-len(suffix)])
                if suffix == "ing" and token[:-3].endswith(("p", "v", "t", "k", "g", "l", "r")):
                    roots.add(token[:-3] + "e")
        out.update(root for root in roots if len(root) >= 3 and root not in stop)
    # Normalize common source-language variants before cross-language overlap checks.
    translated = {
        "прибор": "device", "прибора": "device", "прибором": "device",
        "устройство": "device", "устройства": "device", "устройством": "device",
        "пылесос": "vacuum", "пылесоса": "vacuum", "пылесосом": "vacuum",
        "всасывает": "suction", "всасывал": "suction", "всасывать": "suction",
        "крошки": "crumbs", "крошек": "crumbs", "крошку": "crumbs",
        "пыли": "dust", "пыль": "dust", "грязи": "dirt",
        "ковра": "carpet", "ковре": "carpet", "коврик": "mat",
    }
    out.update(translated[token] for token in list(out) if token in translated)
    return out


_PRODUCT_ACTION_VERB = re.compile(
    r"\b(?:vacuum\w*|suction|sucks?\s+up|picks?\s+up|clean(?:s|ed|ing)?|wip(?:e|es|ed|ing)|remov\w*|lift\w*|scrub\w*|"
    r"cut\w*|slic(?:ing|ed)|slices(?=\s+(?:the|a|an|it|them)\b)|grind\w*|sand(?:s|ed|ing)?|polish\w*|inflate\w*|pump\w*|spray\w*|dispens\w*|pour\w*|"
    r"mix\w*|stir\w*|heat\w*|cool\w*|charg\w*|connect\w*|disconnect\w*|lock\w*|unlock\w*|"
    r"open(?:s|ing)\b|opened(?=\s+(?:the|a|an|it|them)\b)|"
    r"clos(?:es|ing)\b|closed(?=\s+(?:the|a|an|it|them)\b)|close(?=\s+(?:the|a|an|it|them)\b)|close(?=\s*[.!?,;]|$)|"
    r"fold\w*|unfold\w*|press\w*|push\w*|pull\w*|tighten\w*|loosen\w*|wash\w*|dry\w*|"
    r"filter\w*|appl\w*|repair\w*|fix\w*|measur\w*|detect\w*|organiz\w*|store\w*|fill\w*|drain\w*|"
    r"clear(?:s|ed|ing)?\s+(?:away|out|off|crumbs?|dust|dirt|debris|clog)\b|flush\w*|"
    r"sealing\b|seals?(?=\s+(?:the|a|an|it|them)\b)|sealed(?=\s+(?:the|a|an|it|them)\b)|"
    r"protect\w*|block\w*|reduc\w*|всасыва\w*|пылесос\w*|очища\w*|чист\w*|"
    r"протира\w*|удаля\w*|вытира\w*|реж\w*|нареза\w*|шлифу\w*|полиру\w*|накачива\w*|распыля\w*|"
    r"налива\w*|смешива\w*|нагрева\w*|охлажда\w*|заряжа\w*|подключа\w*|отключа\w*|закрыва\w*|"
    r"открыва\w*|складыва\w*|толка\w*|тян\w*|закручива\w*|откручива\w*|фильтру\w*|нанос\w*|"
    r"измеря\w*|обнаружива\w*|фиксиру\w*|заполня\w*|слива\w*|устраня\w*|убира\w*)\b", re.I
)


def _explicit_product_action_evidence(source_transfer) -> list:
    """Return timestamps for physical-action observations, independent of product category."""
    if not isinstance(source_transfer, dict):
        return []
    rows = source_transfer.get("evidence")
    if not isinstance(rows, list):
        return []
    negation = re.compile(
        r"\b(?:does\s+not|doesn't|did\s+not|didn't|cannot|can't|failed\s+to|unable\s+to|without|not)\b",
        re.I,
    )
    out = []
    for row in rows[:12]:
        if not isinstance(row, dict):
            continue
        observation = str(row.get("observation") or "").strip()
        try:
            timestamp = float(row.get("t_sec"))
        except (TypeError, ValueError, OverflowError):
            continue
        if timestamp < 0 or not _PRODUCT_ACTION_VERB.search(observation):
            continue
        if len(_source_evidence_content_tokens(observation)) < 2:
            continue
        supports = row.get("supports")
        if isinstance(supports, list) and supports and not {
            "product_intervention", "result_demo"
        }.intersection(str(item) for item in supports):
            continue
        if negation.search(observation):
            continue
        if timestamp not in out:
            out.append(timestamp)
    return out


def _source_action_tokens(text) -> set[str]:
    """Return normalized action words only; nouns shared by a denial and an observation are not proof."""
    tokens = set()
    for match in _PRODUCT_ACTION_VERB.finditer(str(text or "")):
        # Keep the verb/head only; an object such as "dust" in "clear dust" is not an action.
        head = re.match(r"[a-z][a-z'-]*|[а-яё][а-яё'-]*", match.group(0).casefold())
        if head:
            token = head.group(0).strip("'-")
            if token.endswith("ing") and len(token) > 5:
                token = token[:-3]
                if token.endswith(("p", "v", "t", "k", "g", "l", "r")):
                    token += "e"
            elif token.endswith("ed") and len(token) > 4:
                token = token[:-2]
                if token.endswith(("p", "v", "t", "k", "g", "l", "r")):
                    token += "e"
            elif token.endswith("es") and len(token) > 4:
                token = token[:-2]
                if token.endswith(("p", "v", "t", "k", "g", "l", "r")):
                    token += "e"
            elif token.endswith("s") and len(token) > 3:
                token = token[:-1]
            tokens.update(_source_evidence_content_tokens(token))
    return tokens


def _source_evidence_contradiction(decision, source_transfer) -> str:
    """Catch a negative gate verdict that contradicts specific source-frame evidence."""
    timestamps = _explicit_product_action_evidence(source_transfer)
    if not timestamps or not isinstance(decision, dict) or decision.get("transferable") is not False:
        return ""
    denial_patterns = (
        re.compile(
            r"\b(?:no|lack(?:s|ing)?|without|never|not)\b.{0,100}\b"
            r"(?:source\s+)?(?:evidence|observation|frame|show|demonstrat|depict|record)\w*\b",
            re.I,
        ),
        re.compile(
            r"\b(?:does\s+not|doesn't|did\s+not|didn't|cannot|can't|was\s+not|wasn't)\b.{0,100}\b"
            r"(?:shown|visible|observed|demonstrated|recorded|captured|depicted|present)\b",
            re.I,
        ),
    )
    claims = [str(decision.get("reason") or "")]
    dropped = decision.get("source_elements_dropped")
    if isinstance(dropped, list):
        claims.extend(
            str(item.get("why") or "")
            for item in dropped
            if isinstance(item, dict) and item.get("element") in {"mechanism", "intervention", "result"}
        )
    for claim in claims:
        if not any(pattern.search(claim) for pattern in denial_patterns):
            continue
        claim_tokens = _source_evidence_content_tokens(claim)
        for row in (source_transfer.get("evidence") or [])[:12]:
            if not isinstance(row, dict):
                continue
            try:
                timestamp = float(row.get("t_sec"))
            except (TypeError, ValueError, OverflowError):
                continue
            if timestamp not in timestamps:
                continue
            observation = str(row.get("observation") or "")
            # Require a shared action, not merely two shared nouns such as "bag" and "dish".
            action_overlap = _source_action_tokens(claim) & _source_action_tokens(observation)
            overlap = claim_tokens & _source_evidence_content_tokens(observation)
            if action_overlap and len(overlap) >= 2:
                marks = ", ".join(f"{t:g}s" for t in timestamps)
                return (
                    "SOURCE_EVIDENCE_CONTRADICTION: the negative transfer rationale overlaps with concrete "
                    "source-frame observation terms at " + marks + ". Reconcile the specific observation "
                    "before rejecting the transfer; a distinct, evidenced target incompatibility may still "
                    "justify transferable=false."
                )
    return ""


def _negative_gate_mechanism_contradiction(decision) -> str:
    """A rejected transfer cannot simultaneously claim its core mechanism is preserved."""
    if not isinstance(decision, dict) or decision.get("transferable") is not False:
        return ""
    preserved = decision.get("source_elements_preserved")
    if not isinstance(preserved, list):
        return ""
    if not any(isinstance(item, dict) and item.get("element") == "mechanism" for item in preserved):
        return ""
    return (
        "NEGATIVE_TRANSFER_MECHANISM_CONTRADICTION: transferable=false conflicts with "
        "source_elements_preserved, which still marks the core mechanism as preserved. "
        "If a separate essential incompatibility prevents transfer, drop the mechanism with "
        "a specific explanation; otherwise reconsider the negative verdict. Do not reject a "
        "compatible target merely because optional source features are unsupported."
    )

_EVIDENCE_GAP = re.compile(
    r"cannot\s+be\s+(?:verified|confirmed)|can(?:not|'t)\s+(?:verify|confirm)|not\s+(?:verified|confirmed|evidenced|supported)|"
    r"unverified|without\s+(?:any\s+)?(?:target|offer)\s+(?:evidence|verification|support)|"
    r"no\s+(?:target|offer)\s+evidence|lacks?\s+(?:target\s+|offer\s+)?evidence", re.IGNORECASE)
_EVIDENCE_GAP_ERROR = "NEGATIVE_TRANSFER_EVIDENCE_GAP"


def _negative_gate_evidence_gap(decision) -> str:
    """A rejection whose every dropped element is justified only by missing target evidence for a source feature.

    Missing evidence for a feature (a readout, a rating, a spec number) is a data gap, not proof that the target
    product class cannot perform the source's causal action. This correction is requested only within the
    fixed Gate round budget; exhausting that budget never turns a rejection into an approval.
    """
    if not isinstance(decision, dict) or decision.get("transferable") is not False:
        return ""
    dropped = decision.get("source_elements_dropped")
    if not isinstance(dropped, list) or not dropped:
        return ""
    reasons = [str(item.get("why") or "") for item in dropped if isinstance(item, dict)]
    if not reasons or not all(_EVIDENCE_GAP.search(reason) for reason in reasons):
        return ""
    return (
        f"{_EVIDENCE_GAP_ERROR}: every dropped element is justified only by missing or unverified target evidence "
        "for a source feature. A missing target claim for a feature that does not change the causal action or its "
        "observable result (display readouts, power ratings, spec numbers, accessories) is not an incompatibility: "
        "omit that feature from the creative and keep the elements it does not define as preserved. Set "
        "transferable=false only if the target product class cannot perform the source's causal action at all, and "
        "then name that essential missing capability in each dropped element's why instead of citing unverified evidence."
    )


def request_transfer_fit(offer, brief: str, source_transfer: dict, call, trace_out=None, max_rounds: int = 3) -> dict:
    """Run the pre-generation fit gate with a fixed, auditable total-round budget.

    Only malformed/invalid responses are retried. A valid transferable=false verdict is final,
    and provider exceptions are left to the registry's infrastructure retry policy.
    """
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")

    base_request = {
        "task": "target_fit_gate",
        "schema": TRANSFER_FIT_SCHEMA,
        "offer": offer.to_dict(),
        "brief": brief,
        "source_transfer": source_transfer,
        "required_source_elements": list(required_gate_elements(source_transfer)),
        "source_evidence_refs": [
            {"t_sec": float(row.get("t_sec")), "observation": str(row.get("observation") or "").strip()}
            for row in (source_transfer.get("evidence") or [])
            if isinstance(row, dict)
            and str(row.get("observation") or "").strip()
            and isinstance(row.get("t_sec"), (int, float))
            and float(row.get("t_sec")) in source_evidence_timestamps(source_transfer)
        ][:8] if isinstance(source_transfer, dict) else [],
        "rules": [
            "This is a transferability gate only. Return atoms=[]; do not write image atoms or an ad prompt.",
            "Judge the source's causal commercial mechanism, not superficial product similarity, source score, generic ease/speed language, or CTA.",
            "Cross-category transfer is allowed only when you can explain how the same causal persuasion mechanism maps to a real target-native problem, intervention and observable proof.",
            "The current OfferCard is authoritative for target facts. The sampled source analysis is a hypothesis, not full-video or performance proof. Never transfer source-only hardware, special features, quantitative claims, or capabilities without target evidence.",
            "SOURCE LANGUAGE AND EVIDENCE CONSISTENCY: source fields and observations may be Russian or another language. Interpret them semantically and write this decision in English; language alone is never a reason to drop an element. Treat source_evidence_refs as the observation ledger. If an observation explicitly records an action (for example, a device vacuuming crumbs), do not claim that action was never observed. Acknowledge the exact evidence and reject only for a separate, specific target incompatibility if one remains.",
            "Separate ordinary product-class actions from added functions. The product name/type/category may support only the ordinary function inherent in that class; it does not prove a special mode, material, accessory function, or result such as foam/detergent spraying. For every preserved mechanism/intervention whose target form adds a concrete function beyond product identity, target_evidence MUST quote a relevant verified_claim or confirmation that names that function. A product photo filename or visible accessory is not functional proof here; this gate receives no photo pixels, so an owner must record any visually confirmed capability in OfferCard.confirmations. If evidence is absent, narrow the target form to a function explicitly confirmed in OfferCard or set transferable=false. Never write 'plausible', 'likely', 'probably' or equivalent as a substitute for evidence.",
            "ORDINARY DEMO IS NOT A DEFAULT ANSWER: a plain demonstration of the target product is a valid transfer only when the source mechanism itself is a plain demonstration. If the source hook or mechanism is a comparison (old vs new, A vs B, rival method, before vs after), replacing it with a single-product demo is a substitution, not a transfer. Map the comparison onto a target-native comparison (for example this product against the old way of doing the same task) or choose transferable=false.",
            "Do not invent or imply unverified specifications, runtime, accessories, container design, special features, or performance outcomes. An ordinary visible action may transfer without those claims; a mechanism that depends on an unverified unique feature must be rejected.",
            "For transferable=true, EACH target_evidence array item must be one contiguous verbatim quote from ONE OfferCard field only (name, product_type, product_category, one verified_claim, or one confirmation). For a preserved mechanism/intervention with a concrete added function, at least one quote must come from verified_claims or confirmations and directly support that function; quoting only the product name/type/category is insufficient. Never join multiple claims, paraphrase, normalize, or upgrade a generic specification into a use-specific claim. No source-ad text and no invented target facts.",
            "A target_evidence excerpt shorter than 8 characters is allowed only when it exactly matches a complete OfferCard field; do not use a short fragment from a longer field.",
            "A shared generic desire (easy, fast, fewer cables, convenient) is not sufficient. Map the core causal action and observable result only to functions explicitly supported by OfferCard evidence. Ordinary dry-debris vacuuming does not require verified suction numbers; do not reject an ordinary product-class action merely because optional specs or supporting shots are unverified. Choose transferable=false when an essential special target capability is unsupported, the core intervention cannot be demonstrated from confirmed target facts, or material uncertainty remains about that essential capability.",
            "ESSENTIAL VERSUS OPTIONAL FEATURES: a source feature is essential only if removing it changes the causal action or its observable result (a device without suction cannot demonstrate suction). A display readout, a power rating, a spec number or an accessory that does not change what visibly happens is optional: omit it from the creative, keep every element it does not define as preserved, and do not cite its missing target evidence as the reason for rejection.",
            "ELEMENT ACCOUNT (required): account for EVERY name in required_source_elements exactly once, either in source_elements_preserved (element, source_form = what it is in the source, target_form = how it appears with the target product, evidence_t_sec = timestamps selected from source_evidence_refs) or in source_elements_dropped (element, why). Do not invent unlisted elements. For every preserved element, target_form must describe the target-native adaptation, not repeat source_form. An empty source_elements_dropped is valid when every source element is preserved. If source_evidence_refs are supplied, every preserved element must cite 1-3 exact supplied timestamps that support the source observation; do not invent timestamps.",
            "NO-ADDED-TEXT POLICY (non-negotiable): Never request text overlays, on-screen text, captions, subtitles, title cards, speech bubbles, UI, watermarks or promotional copy in target_form or target_native_mapping. If the source hook is a text overlay, describe it factually only in source_form; convert its attention idea into a visual-only target_form and target_native_mapping, or list hook as dropped. The source may contain text; the target creative must not add it.",
            "MECHANISM IS NON-NEGOTIABLE: transferable=true is impossible unless the source element 'mechanism' is in source_elements_preserved. Preserving only the hook, visual style, result, or product action is not a mechanism transfer. If the causal mechanism cannot be carried to the target honestly, set transferable=false.",
            "demo_form_source and demo_form_target name the demonstration shape of the source and of your target plan (comparison, before_after, demonstration, testimonial, tutorial, other). A comparison or before_after source must keep a comparison or before_after target. Code rejects a missing source mechanism and rejects a comparison downgraded to an ordinary demonstration, whatever transferable says.",
            "Keep transfer_decision.reason below 220 characters and end it with a complete sentence. The response schema permits 320 characters only so code can detect over-limit output; never use that extra schema allowance.",
            "Return transfer_decision with transferable, abstract_mechanism, target_native_mapping, target_evidence, reason, demo_form_source, demo_form_target, source_elements_preserved and source_elements_dropped. Explain the source-specific causal mechanism and how the target can or cannot demonstrate it.",
        ],
    }

    previous_errors = []
    last_errors = []
    gap_decision = None
    round_number = 0
    while round_number < max_rounds:
        round_number += 1
        request = dict(base_request)
        request["rules"] = list(base_request["rules"])
        if previous_errors:
            request["previous_errors"] = list(previous_errors)
            request["rules"].append(
                "Your previous Target-Fit response failed validation. Correct EVERY issue listed in previous_errors, "
                "return the complete response, and do not weaken any transferability or target-evidence requirement."
            )

        record = {"phase": "target_fit_gate", "round": round_number, "round_budget": max_rounds}
        if previous_errors:
            record["previous_errors"] = list(previous_errors)
        record.update(_request_provenance(request))
        if trace_out is not None:
            trace_out.append(record)
        try:
            response = call(request)
        except Exception as exc:
            record.update(_call_provenance(call))
            record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            raise

        record.update(_call_provenance(call))
        record["response_excerpt"] = _trace_excerpt(response)
        decision = None
        if not isinstance(response, dict) or response.get("_error"):
            detail = response.get("_error") if isinstance(response, dict) else "response is not a JSON object"
            errors = [f"target-fit gate returned invalid output: {detail or 'missing response'}"]
        elif response.get("atoms") != []:
            errors = ["target-fit gate must return atoms=[]; production atoms are forbidden at this stage"]
        else:
            decision = response.get("transfer_decision")
            errors = validate_transfer_fit_decision(offer, decision, source_transfer)
            for contradiction in (
                _source_evidence_contradiction(decision, source_transfer),
                _negative_gate_mechanism_contradiction(decision),
                _negative_gate_evidence_gap(decision),
            ):
                if contradiction:
                    errors.append(contradiction)

        if errors:
            record["validation_errors"] = errors
            previous_errors = list(errors)
            last_errors = errors
            gap_only = len(errors) == 1 and errors[0].startswith(_EVIDENCE_GAP_ERROR)
            gap_decision = decision if gap_only else None
            continue

        result = {key: decision[key] for key in GATE_DECISION_KEYS}
        if result["transferable"] is True:
            override = core_preservation_verdict(result)
            if override:
                # The model approved the transfer, but its own account shows the source was replaced. Code rejects.
                result = {**result, "transferable": False, "model_transferable": True,
                          "reason": override[:240], "rejected_by_code": True}
        record["verdict"] = {"transferable": result["transferable"], "reason": result["reason"],
                             "rejected_by_code": bool(result.get("rejected_by_code"))}
        return result

    if gap_decision is not None and len(last_errors) == 1 and last_errors[0].startswith(_EVIDENCE_GAP_ERROR):
        # The model kept rejecting on missing target evidence after being told that is not an incompatibility.
        # The rejection stands (nothing is approved) and is reported as a data gap the operator can close by
        # adding verified claims to the OfferCard, so the source loop may still try the next candidate.
        final = {key: gap_decision[key] for key in GATE_DECISION_KEYS}
        final["rejection_basis"] = "EVIDENCE_GAP"
        if trace_out is not None:
            trace_out.append({"phase": "target_fit_gate", "verdict": {
                "transferable": False, "rejection_basis": "EVIDENCE_GAP", "reason": final["reason"]}})
        return final
    raise BrainError({"#transfer_fit": last_errors or ["target-fit gate failed without a validated response"]})


def validate_transfer_decision(decision, atoms, fit_gate=None) -> list:
    """Validate the decision and require every preserved property to have a MUST image carrier."""
    if not isinstance(decision, dict):
        return ["transfer_decision is required when source_transfer is supplied"]
    errors = []
    if type(decision.get("transferable")) is not bool:
        errors.append("transfer_decision.transferable must be true or false")
    reason = decision.get("reason")
    if not isinstance(reason, str) or len(reason.strip()) < 10:
        errors.append("transfer_decision.reason must explain the decision (at least 10 characters)")
    for key in ("abstract_mechanism", "reason"):
        value = decision.get(key)
        if isinstance(value, str) and _CYRILLIC.search(value):
            errors.append(f"transfer_decision.{key} must be in English")
    if isinstance(reason, str) and len(reason.strip()) >= 10:
        reason_error = text_boundary_rewrite_error(
            reason,
            field="transfer_decision.reason",
            hard_limit=GATE_REASON_MAX_TEXT,
            rewrite_limit=GATE_REASON_REWRITE_MAX,
            allow_exact_limit=False,
        )
        if reason_error:
            errors.append(reason_error)
    if decision.get("transferable") is True:
        mechanism = decision.get("abstract_mechanism")
        if not isinstance(mechanism, str) or len(mechanism.strip()) < 10:
            errors.append("transfer_decision.abstract_mechanism must state the brand-free causal idea (at least 10 characters)")
        preserved = decision.get("preserved")
        if not isinstance(preserved, list) or not 1 <= len(preserved) <= 5:
            errors.append("transfer_decision.preserved must list 1-5 source properties that stay visible")
        else:
            by_id = {}
            for atom in atoms or []:
                aid, priority, channel, kind = _atom_view(atom)
                by_id[aid] = (priority, channel, kind)
            seen_preserved = set()
            for index, item in enumerate(preserved):
                if not isinstance(item, dict):
                    errors.append(f"transfer_decision.preserved[{index}] must be an object")
                    continue
                prop = item.get("property")
                prop_key = prop.strip().lower() if isinstance(prop, str) else ""
                # The gate's canonical element IDs (hook/problem/mechanism/intervention/result)
                # are meaningful property identifiers, not malformed short descriptions.
                if not isinstance(prop, str) or (len(prop_key) < 8 and prop_key not in GATE_ELEMENTS):
                    errors.append(f"transfer_decision.preserved[{index}].property must describe the preserved property")
                    continue
                has_gate_element_account = (
                    isinstance(fit_gate, dict)
                    and isinstance(fit_gate.get("source_elements_preserved"), list)
                    and isinstance(fit_gate.get("source_elements_dropped"), list)
                )
                if has_gate_element_account and prop_key not in GATE_ELEMENTS:
                    errors.append(
                        f"transfer_decision.preserved[{index}].property must use a canonical Target-Fit Gate element "
                        f"({', '.join(GATE_ELEMENTS)})"
                    )
                    continue
                if prop_key in seen_preserved:
                    errors.append(f"transfer_decision.preserved contains duplicate Gate element '{prop_key}'")
                    continue
                seen_preserved.add(prop_key)
                atom_id = str(item.get("atom_id") or "")
                carrier = by_id.get(atom_id)
                # The hook/mechanism Gate elements have dedicated keep_* atoms.
                # The causal problem/intervention/result chain is already carried by mech_* atoms;
                # duplicating it as keep_* would consume the frame budget without adding evidence.
                required_role = {
                    "hook": "keep_hook",
                    "mechanism": "keep_mechanism",
                    "problem": "mech_problem",
                    "intervention": "mech_intervention",
                    "result": "mech_result",
                }.get(prop_key)
                if required_role and not atom_id_matches_role(atom_id, required_role):
                    expected_prefix = required_role + "_"
                    errors.append(
                        f"transfer_decision.preserved[{index}].atom_id must use the {expected_prefix} prefix "
                        f"(or exact base id) for the preserved source element")
                    continue
                if carrier is None or carrier[0] != "MUST" or carrier[1] != "image" or carrier[2] == "claim":
                    errors.append(
                        f"transfer_decision.preserved[{index}].atom_id must name an existing MUST image atom "
                        f"that carries '{prop[:60]}'")
        has_gate_element_account = (
            isinstance(fit_gate, dict)
            and isinstance(fit_gate.get("source_elements_preserved"), list)
            and isinstance(fit_gate.get("source_elements_dropped"), list)
        )
        if has_gate_element_account:
            gate_preserved = {
                str(item.get("element"))
                for item in (fit_gate.get("source_elements_preserved") or [])
                if isinstance(item, dict)
            }
            gate_dropped = {
                str(item.get("element"))
                for item in (fit_gate.get("source_elements_dropped") or [])
                if isinstance(item, dict)
            }
            decision_preserved = {
                str(item.get("property") or "").strip().lower()
                for item in (preserved or []) if isinstance(item, dict)
            }
            for element in sorted(gate_preserved - decision_preserved):
                errors.append(
                    f"transfer_decision.preserved must include Gate-preserved source element '{element}'"
                )
            for element in sorted(gate_dropped.intersection(decision_preserved)):
                errors.append(
                    f"transfer_decision.preserved must not reintroduce Gate-dropped source element '{element}'"
                )
    return errors


def build_request(offer, brief: str, shots: int, failed=None, source_transfer=None) -> dict:
    policy = getattr(offer, "branding_policy", TARGET_BRAND_ALLOWED)
    branding_rule = brain_branding_rule(policy, resolved_brand_marks(
        policy, getattr(offer, "approved_brand_marks", ()), getattr(offer, "name", ""), getattr(offer, "product_type", "")))
    req = {"schema": TRANSFER_ATOM_SCHEMA if source_transfer else ATOM_SCHEMA, "offer": offer.to_dict(), "brief": brief, "shots": shots,
           "rules": ["Return atoms only, no finished prompts.",
                     "Every kind='claim' atom must copy one verified claim verbatim and use channel='qa'; it records offer truth, not pixels.",
                     "Never put kind='claim' atoms in channel='image'. If visible proof is needed, create a separate image atom of kind product/action/scene that describes an observable cue, such as a close-up of the zip closure visibly shut.",
                     "Do not treat abstract care or durability claims as visually verifiable from a generic still. Ask for a visible label only when the brief explicitly requires on-image text.",
                     "Every kind='qa_check' atom must use channel='qa'; it is a verification criterion and must not be included in the image-generation prompt.",
                     "Do not emit an atom whose id is 'qa_no_text' or starts with 'qa_no_text_'. The canonical no-added-text requirement is injected and checked by code, so Brain must not author a duplicate QA criterion.",
                      branding_rule,
                     "BRAND-FREE VISUAL ATOMS: never put the target brand name, model name, approved brand mark or competitor name in any image atom. Describe the product by its generic product type and visible physical form; target branding is governed separately by the campaign policy and product-reference QA, not by a scene/action atom.",
                     "NO-ADDED-TEXT POLICY: do not add text overlays, captions, subtitles, title cards, speech bubbles, UI, watermarks or promotional wording to any image atom. If the source hook is text, translate its attention mechanism into a visual-only action; never reproduce the source wording.",
                     "Never use a forbidden claim.",
                     "When target-product reference photos are supplied, preserve the actual target product's form factor, silhouette, proportions, and distinctive visible parts. Keep the target product body recognizable in the frame; do not substitute a generic lookalike or change a compact handheld product into an upright/stick device.",
                     "Write every atom text in English, even when the input context is in another language. Translate the source statements; never copy non-English words into atoms.",
                     "Aim to keep each atom text at or below 100 characters. Complete atoms up to the 120-character code limit remain valid; do not pad or add secondary details just to use the available length.",
                     "Frame budget: at most %d MUST image atoms per shot; use only one layout atom per shot; do not repeat the same visible requirement in a scene, action and mechanism atom. The mech_* atom carries the idea; other atoms add only distinct requirements." % MAX_MUST_IMAGE_PER_SHOT]}
    if source_transfer:
        req["source_transfer"] = source_transfer
        fit_gate = source_transfer.get("target_fit_gate") if isinstance(source_transfer, dict) else None
        gate_form = (fit_gate or {}).get("demo_form_target") if isinstance(fit_gate, dict) else None
        has_gate_form = isinstance(gate_form, str) and gate_form.strip().lower() in DEMO_FORMS
        demo_form_target = str(gate_form or "demonstration").strip().lower()
        contrast_allowed = has_gate_form and demo_form_target in CONTRAST_FORMS
        if isinstance(fit_gate, dict):
            req["rules"].append(
                "MANDATORY PRE-APPROVED TARGET-FIT GATE: the independent gate already approved this exact abstract mechanism: "
                + str(fit_gate.get("abstract_mechanism") or "") + ". Do not broaden it into generic convenience, speed, or ease; "
                "build the target-native creative only around that approved causal mechanism and its target_native_mapping: "
                + str(fit_gate.get("target_native_mapping") or "") + ". Keep target_evidence grounded in the OfferCard. If this exact mechanism "
                "cannot be visibly preserved, return transfer_decision.transferable=false rather than silently substituting a different idea.")
            kept = [f"{i.get('element')}: {i.get('target_form')}" for i in fit_gate.get("source_elements_preserved") or [] if isinstance(i, dict)]
            dropped = [f"{i.get('element')}" for i in fit_gate.get("source_elements_dropped") or [] if isinstance(i, dict)]
            if kept:
                req["rules"].append(
                    "The gate recorded which source elements must stay visible, and in what target form: " + "; ".join(kept)
                    + (". Elements the gate dropped (do not reintroduce them): " + ", ".join(dropped) if dropped else "")
                    + ". Keep the approved demonstration shape (" + str(fit_gate.get("demo_form_target") or "as approved")
                    + "); do not replace it with a generic product demo.")
            keep_elements = required_keep_elements(fit_gate)
            if keep_elements:
                prefixes = ", ".join(f"{name}=keep_{name}_" for name in keep_elements)
                req["rules"].append(
                    "CORE PRESERVATION MUST ATOMS: the Gate approved these source elements for preservation: "
                    + ", ".join(keep_elements) + ". For EACH, create a separate MUST image atom with these id prefixes: "
                    + prefixes + ". The atom must express the specific target_form "
                    "recorded by the Gate, not a generic product shot. Do not merge it into a mech_* atom."
                )
        links = derive_mechanism_links(source_transfer)
        if links:
            req["mechanism_links"] = links
            req["rules"].append(
                "mechanism_links lists the causal links that must survive the transfer. For EACH link create at least one "
                "MUST atom with channel='image' (kind product/action/scene/composition, never claim) whose id starts with its "
                "atom_id_prefix (for example mech_intervention_clamp) and whose text shows that link in the TARGET product's own "
                "scene, with visible, checkable content. Do not copy the source product, brand, creator or setting. "
                "A link whose source_kind is INFERRED is the analyst's hypothesis, not something the source was seen to "
                "show (its source_statement may even say the source does not show it): write the target's version from the "
                "target product's own honest demonstration and never present the unproven source statement as an observed result.")
            link_names = {link["link"] for link in links}
            if shots == 1 and {"problem", "result"}.issubset(link_names):
                if contrast_allowed:
                    composition_rule = (
                        "ONE-SHOT COMPOSITION CONTRACT: the Gate explicitly approved the contrast form "
                        + demo_form_target + ". Create an additional MUST image scene atom (id prefix 'comp_') "
                        "that describes one continuous, legible in-scene contrast and locates the relevant states. "
                        "Place the active intervention at the visible boundary. Do not turn it into panels, a "
                        "split-screen, collage or grid; do not copy a mech_* atom into comp_."
                    )
                else:
                    if has_gate_form:
                        form_authority = (
                            "The Gate approved target form '" + demo_form_target
                            + "', not a before/after comparison. "
                        )
                    else:
                        form_authority = (
                            "No contrast form was approved; default to one demonstration, not a before/after comparison. "
                        )
                    composition_rule = (
                        "ONE-SHOT DEMONSTRATION CONTRACT: " + form_authority
                        + "Create an additional MUST image scene atom (id prefix 'comp_') describing one uninterrupted "
                        "target-native demonstration: show the product, active contact/action and the approved or "
                        "observable in-progress result cue in one coherent scene. Do NOT create separate untreated/treated "
                        "or dirty/clean left/right zones, a completed-cleaning comparison, before/after framing, panels or a "
                        "collage. Do not copy a mech_* atom into comp_."
                    )
                req["rules"].append(composition_rule)
            elif shots >= 2:
                req["rules"].append(
                    "MULTI-SHOT ROUTING CONTRACT: mech_problem_* and mech_intervention_* MUST image atoms go to shot 1; mech_result_* MUST image atoms go to shot 2.")
        req["rules"].extend([
            "When source_transfer is supplied, preserve its causal mechanism and target-native rebuild sequence as structured atom requirements.",
            "The current OfferCard and supplied target-product photos are the ONLY authority for target-product facts. Source-transfer context is authority only for the causal mechanism, not target hardware or specifications. Never copy source-only attachments, tools, features, colors, construction, or product form into target atoms unless visible in target-product references or explicitly confirmed by OfferCard/brief. When no target-native feature is verified, retain the generic target product shown in the reference and transfer the demo logic only.",
            "Transfer the problem/intervention/proof logic, not the source ad's exact scene, creator, brand, UI, props, wording, or composition. Respect source_specific_components, replacement_requirements, must_change, contraindications, and limits.",
            "For each requested shot, keep the target-native action, proof, state_before/state_after, and result/delta where present. Express required visual directions as stable MUST atoms; do not silently omit them.",
            "ACTIVE INTERVENTION EVIDENCE: every mech_intervention_* MUST image atom must describe the target product's active working end physically engaging the exact problem material at a visible contact point. Show a concrete visual cue that material is being acted on or removed in that same frame. Do not describe the product as merely resting beside, hovering over, or pointing toward the problem. For a still image, prove the action through contact and visible state change, not invisible motion.",
            "Do not pack several unverified accessories into one MUST atom. Keep the main target product body visible and show only the attachment performing the action; omit accessories not visible in the target reference or confirmed by OfferCard/brief.",
            (
                "APPROVED CONTRAST FORM: use a visibly checkable in-scene contrast, with the active working end "
                "physically engaging the material at the transition and a visible contact point linking the action "
                "to the change. Keep it one continuous scene, not a multi-panel collage."
                if contrast_allowed else
                "APPROVED DEMONSTRATION FORM: show only the target-native action and the "
                + ("Gate-approved" if has_gate_form else "observable target-native")
                + " visible result cue in one uninterrupted scene. Do not invent a before/after, side-by-side, or "
                "untreated-versus-treated layout; do not imply a completed result that the Gate did not approve. "
                "Keep a visible contact point linking the action to the change."
            ),
            "The source's commercial metrics or reasoning-confidence values are not proof that the mechanism is profitable for the target offer.",
        ])
        req["rules"].append(
            "TRANSFER DECISION (required): decide whether the source's causal mechanism can be demonstrated honestly "
            "by the TARGET product. Return transfer_decision {transferable, abstract_mechanism, preserved, reason} "
            "beside atoms. The abstract_mechanism is owned by the independently approved Target-Fit Gate; do not broaden "
            "or substitute its causal idea (code carries the exact approved wording into the final decision). If the target "
            "cannot demonstrate it without inventing features, or the mechanism becomes an unrelated ordinary target-product "
            "demo, set transferable=false, explain why, and return atoms=[]. If true, preserved must list EVERY element in "
            "source_elements_preserved exactly once (1-5 entries), and must not list any element in source_elements_dropped. "
            "Keep transfer_decision.reason below 220 characters; the response schema allows extra headroom only so code can detect an over-limit answer. "
            "Each property must use its canonical Gate element name and point to an existing MUST image carrier: hook uses "
            "keep_hook[_suffix], mechanism uses keep_mechanism[_suffix], and problem/intervention/result use their matching "
            "mech_* atoms. Do not create duplicate keep_intervention or keep_result atoms; their mech_* atoms are the causal-link "
            "carriers. English only."
        )
    if failed:
        req["only_fix"] = failed  # [{id, errors, previous_text}]
    return req


def _retry_target(pending: dict, by_id: dict) -> dict:
    """Choose one repair unit and explicitly describe which atom IDs that unit may change."""
    direct_ids = sorted(key for key in pending if not str(key).startswith("#"))
    if direct_ids:
        atom_id = direct_ids[0]
        return {"key": atom_id, "mode": "exact", "id": atom_id, "role": atom_role(atom_id)}

    structural = sorted(str(key) for key in pending)
    for key in structural:
        match = re.fullmatch(r"#keep_([a-z]+)", key)
        if match:
            return {"key": key, "mode": "role", "role": f"keep_{match.group(1)}"}
        match = re.fullmatch(r"#mechanism_shot_([a-z]+)", key)
        if match:
            expected_shot = {"problem": 1, "intervention": 1, "result": 2}.get(match.group(1))
            if expected_shot is not None:
                return {"key": key, "mode": "role_shot", "role": f"mech_{match.group(1)}",
                        "shot": expected_shot}
        match = re.fullmatch(r"#mechanism_([a-z]+)", key)
        if match:
            return {"key": key, "mode": "role", "role": f"mech_{match.group(1)}"}
        if key == "#composition":
            return {"key": key, "mode": "role", "role": "comp"}
        match = re.fullmatch(r"#shot_(\d+)", key)
        if match:
            return {"key": key, "mode": "shot", "shot": int(match.group(1))}
        if key == "#transfer_decision":
            return {"key": key, "mode": "decision"}

    # A malformed first response can be safely re-requested in full only when no
    # accepted atoms exist yet. Once atoms are accepted, unknown structural errors
    # must not reopen the whole contract and allow the model to rewrite them.
    if not by_id:
        return {"key": structural[0] if structural else "#root", "mode": "full"}
    return {"key": structural[0] if structural else "#retry", "mode": "blocked"}


def _retry_request(offer, brief, shots, source_transfer, target, errors, previous_text, retry_number):
    """Build a one-atom repair request with a narrower schema and a distinct strategy per retry."""
    strategy = {
        1: {"name": "single_atom_primary", "route": "primary", "text_max_chars": 100,
            "instruction": "Rewrite only the failed atom. Remove introductory scene-setting and secondary details; state the visible action directly."},
        2: {"name": "single_atom_reserve", "route": "fallback", "text_max_chars": 100,
            "instruction": "Use a different sentence construction. Keep only the indispensable visible subject, action and result cue; do not reuse the earlier sentence structure."},
        3: {"name": "single_atom_reserve_compact", "route": "fallback", "text_max_chars": 80,
            "instruction": "Final compact repair. Write the shortest complete sentence that preserves the required visual meaning. Remove all optional qualifiers."},
    }.get(retry_number, {"name": "single_atom_primary_compact", "route": "primary", "text_max_chars": 80,
                       "instruction": "Write the shortest complete sentence preserving the required visual meaning."})
    failed = [{"id": target["key"], "errors": list(errors), "previous_text": previous_text or ""}]
    request = build_request(offer, brief, shots, failed, source_transfer=source_transfer)
    schema = copy.deepcopy(request["schema"])
    atoms_schema = schema["properties"]["atoms"]
    atoms_schema["minItems"] = 0 if target["mode"] in ("decision", "blocked") else 1
    atoms_schema["maxItems"] = 0 if target["mode"] in ("decision", "blocked") else (None if target["mode"] == "full" else 1)
    if atoms_schema["maxItems"] is None:
        atoms_schema.pop("maxItems", None)
    item_schema = atoms_schema.get("items", {}).get("properties", {})
    if target["mode"] not in ("decision", "blocked", "full"):
        # Keep provider-side schema headroom above the hard code limit (160 vs 120).
        # The shorter 100/80-character target is prompt guidance and remains enforced
        # as a quality preference; it must not become a provider truncation boundary.
        # In particular, structured-output providers may cut a value at maxLength.
        id_schema = item_schema.get("id")
        if isinstance(id_schema, dict):
            if target["mode"] == "exact":
                id_schema["const"] = target["id"]
            elif target["mode"] in ("role", "role_shot"):
                id_schema["pattern"] = rf"^{re.escape(target['role'])}(?:_[a-z0-9_]+)?$"
        if target["mode"] in ("shot", "role_shot"):
            shot_schema = item_schema.get("shot")
            if isinstance(shot_schema, dict):
                shot_schema["const"] = target["shot"]
    request["schema"] = schema
    request["retry_strategy"] = {
        "attempt": retry_number,
        "name": strategy["name"],
        "route": strategy["route"],
        "text_max_chars": strategy["text_max_chars"],
    }
    request["retry_contract"] = {
        "mode": target["mode"],
        "target_key": target["key"],
        "expected_id": target.get("id"),
        "allowed_role": target.get("role"),
        "required_shot": target.get("shot"),
        "max_atoms": 0 if target["mode"] in ("decision", "blocked") else (None if target["mode"] == "full" else 1),
    }
    request["rules"].append(
        "REPAIR RESPONSE CONTRACT: return only the single requested repair atom and no other atoms. "
        "Keep every previously accepted atom unchanged; do not regenerate, rename, summarize or improve it. "
        "The code will ignore every atom outside retry_contract. The full code-side validators still apply. "
        f"{strategy['instruction']} The repaired text must be a complete English sentence no longer than "
        f"{strategy['text_max_chars']} characters."
    )
    if target["mode"] == "exact":
        request["rules"].append(f"The only allowed atom id is exactly '{target['id']}'. Return that exact id.")
    elif target["mode"] == "role":
        request["rules"].append(
            f"Return exactly one atom whose id belongs to role '{target['role']}' (same role prefix, optional suffix)."
        )
    elif target["mode"] == "role_shot":
        request["rules"].append(
            f"Return exactly one MUST image atom for role '{target['role']}' in shot {target['shot']}."
        )
    elif target["mode"] == "shot":
        request["rules"].append(
            f"Return exactly one MUST image atom for shot {target['shot']}; do not return atoms for another shot."
        )
    elif target["mode"] == "decision":
        request["rules"].append(
            "This is a transfer_decision-only repair. Return atoms as an empty list; repair only transfer_decision."
        )
    elif target["mode"] == "blocked":
        request["rules"].append(
            "No safe atom ID is authorized for this unresolved structural error. Return atoms as an empty list; do not alter existing atoms."
        )
    return request, strategy


def _retry_atom_allowed(atom: dict, target: dict) -> bool:
    """Check the explicit retry allow-list before parsing or validating returned atoms."""
    if not isinstance(atom, dict):
        return False
    atom_id = atom.get("id")
    if not isinstance(atom_id, str):
        return False
    mode = target.get("mode")
    if mode == "exact":
        return atom_id == target.get("id")
    if mode == "role":
        return atom_role(atom_id) == target.get("role")
    if mode == "role_shot":
        return (atom_role(atom_id) == target.get("role")
                and type(atom.get("shot")) is int and atom.get("shot") == target.get("shot")
                and atom.get("priority") == "MUST" and atom.get("channel") == "image")
    if mode == "shot":
        return (type(atom.get("shot")) is int and atom.get("shot") == target.get("shot")
                and atom.get("priority") == "MUST" and atom.get("channel") == "image")
    if mode == "full":
        return True
    return False


def _ignored_rewrite(atom) -> dict:
    """Record the identity and text digest of an unrequested rewrite without retaining its text."""
    atom = atom if isinstance(atom, dict) else {}
    value = atom.get("text")
    digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest() if value is not None else None
    return {"id": atom.get("id") if isinstance(atom.get("id"), str) else None,
            "text_sha256": digest, "reason": "outside_retry_contract"}


def _is_recomputed_structural_key(key: str) -> bool:
    return (key in ("#root", "#composition", "#transfer_decision")
            or key.startswith("#shot_") or key.startswith("#mechanism_") or key.startswith("#keep_"))


def request_atoms(offer, brief: str, shots: int, call, max_retries: int = 3, vertical=None, source_transfer=None, decision_out=None, rounds_out=None, atom_provenance_out=None) -> list:
    """Build atoms with bounded, one-target repair rounds; max_retries counts retries after the initial call."""
    if type(max_retries) is not int or not 0 <= max_retries <= 3:
        raise ValueError("max_retries must be an integer from 0 to 3; the BRAIN repair budget is fixed")
    total_rounds = max_retries + 1
    by_id: dict = {}
    pending: dict = {}
    failed = None
    required_links = derive_mechanism_links(source_transfer)
    fit_gate = source_transfer.get("target_fit_gate") if isinstance(source_transfer, dict) else None
    held_decision = None
    decision_draft = None
    accepted_atom_provenance: dict = {}
    for round_number in range(1, total_rounds + 1):
        previous_pending = {key: list(value) for key, value in pending.items()}
        retry_target = _retry_target(pending, by_id) if round_number > 1 and pending else None
        record = {"phase": "brain_atoms", "round": round_number, "round_budget": total_rounds,
                  "retry_budget": max_retries,
                  "only_fix_ids": [retry_target["key"]] if retry_target else []}
        if rounds_out is not None:
            rounds_out.append(record)
        if retry_target:
            target_errors = pending.get(retry_target["key"], [])
            previous_text = ""
            failed_previous = next((item.get("previous_text", "") for item in (failed or [])
                                    if item.get("id") == retry_target["key"]), "")
            if failed_previous:
                previous_text = failed_previous
            elif retry_target.get("id") and retry_target["id"] in by_id:
                previous_text = by_id[retry_target["id"]].text
            elif retry_target.get("role"):
                previous = next((a for a in by_id.values() if atom_role(a.id) == retry_target["role"]), None)
                previous_text = previous.text if previous else ""
            request, strategy = _retry_request(
                offer, brief, shots, source_transfer, retry_target,
                target_errors, previous_text, round_number - 1,
            )
            record["retry_strategy"] = strategy["name"]
            record["requested_route"] = strategy["route"]
            record["retry_target"] = retry_target["key"]
            if retry_target["mode"] == "exact":
                record["allowed_atom_ids"] = [retry_target["id"]]
            elif retry_target["mode"] in ("role", "role_shot"):
                record["allowed_atom_role"] = retry_target["role"]
                if retry_target["mode"] == "role_shot":
                    record["allowed_shot"] = retry_target["shot"]
            elif retry_target["mode"] == "shot":
                record["allowed_shot"] = retry_target["shot"]
            else:
                record["allowed_atom_ids"] = []
        else:
            request = build_request(offer, brief, shots, failed, source_transfer=source_transfer)
            strategy = {"name": "initial_full_contract", "route": "primary", "text_max_chars": None}
            record["retry_strategy"] = strategy["name"]
            record["requested_route"] = "primary"
        record.update(_request_provenance(request))
        route = strategy.get("route", "primary")
        if retry_target and retry_target.get("mode") == "blocked":
            # Never spend a provider call when the structural failure has no explicit safe
            # atom allow-list (for example, an unresolved frame-budget conflict).
            record["retry_strategy"] = "blocked_no_safe_allowlist"
            record["requested_route"] = "none"
            record["actual_route"] = "not_called"
            record["accepted_ids"] = sorted(by_id)
            record["accepted_atom_provenance"] = {key: accepted_atom_provenance[key] for key in sorted(accepted_atom_provenance)}
            record["pending"] = {key: list(value) for key, value in pending.items()}
            if isinstance(atom_provenance_out, dict):
                atom_provenance_out.clear()
                atom_provenance_out.update(accepted_atom_provenance)
            raise BrainError(pending)
        try:
            has_fallback = getattr(call, "has_fallback", None)
            fallback_callable = callable(getattr(call, "call_fallback", None))
            if route == "fallback" and fallback_callable and has_fallback is not False:
                record["actual_route"] = "fallback"
                resp = call.call_fallback(request)
            else:
                if route == "fallback":
                    record["actual_route"] = "primary_fallback_unconfigured"
                    record["route_note"] = "brain_fallback_adapter is not configured; this repair used the primary model"
                else:
                    record["actual_route"] = "primary"
                resp = call(request)
        except Exception as exc:
            record.update(_call_provenance(call))
            record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            record["accepted_ids_before_failure"] = sorted(by_id)
            record["pending_before_failure"] = {k: list(v) for k, v in pending.items()}
            raise
        record.update(_call_provenance(call))
        offered = resp.get("transfer_decision") if isinstance(resp, dict) else None
        if source_transfer and isinstance(offered, dict) and offered.get("transferable") is False:
            rejection_errors = validate_transfer_decision(offered, [])
            if not rejection_errors:
                raise TransferRejected(offered)
        raw = resp.get("atoms", []) if isinstance(resp, dict) else []
        raw_for_parse = raw
        ignored_rewrites = []
        retry_contract_error = None
        if retry_target and retry_target["mode"] != "full":
            raw_for_parse = []
            if isinstance(raw, list):
                permitted = [item for item in raw if _retry_atom_allowed(item, retry_target)]
                ignored_rewrites = [_ignored_rewrite(item) for item in raw if not _retry_atom_allowed(item, retry_target)]
                if len(permitted) > 1:
                    retry_contract_error = "RETRY_SINGLE_ATOM_REQUIRED: one repair round may return at most one authorized atom"
                    ignored_rewrites.extend({**_ignored_rewrite(item),
                                             "reason": "single_atom_contract_violation"}
                                            for item in permitted)
                    permitted = []
                raw_for_parse = permitted
                if retry_target["mode"] not in ("decision", "blocked") and not permitted:
                    retry_contract_error = (
                        "RETRY_TARGET_MISSING: response did not return the requested atom or a structurally required atom"
                    )
            elif retry_target["mode"] not in ("decision", "blocked"):
                retry_contract_error = "RETRY_TARGET_MISSING: response atoms must be a list containing the requested atom"
            record["ignored_rewrites"] = ignored_rewrites
            if isinstance(resp, dict):
                audited_response = dict(resp)
                audited_response["atoms"] = raw_for_parse
            else:
                audited_response = resp
            record["response_excerpt"] = _trace_excerpt(audited_response)
        else:
            record["response_excerpt"] = _trace_excerpt(resp)
        parsed = parse_atoms(raw_for_parse)
        round_errs = {k: list(v) for k, v in parsed.errors.items()}
        if retry_contract_error:
            round_errs[retry_target["key"]] = [retry_contract_error]
        # Parser-rejected atoms are absent from parsed.atoms. Remove only those IDs that
        # this round was authorized to repair; ignored rewrites never touch accepted state.
        for rejected_id in parsed.errors:
            if not rejected_id.startswith("#"):
                by_id.pop(rejected_id, None)
                accepted_atom_provenance.pop(rejected_id, None)
        for atom in parsed.atoms:
            if atom.shot > shots:
                round_errs.setdefault(atom.id, []).append(f"shot {atom.shot} is outside requested range 1..{shots}")
        if not isinstance(resp, dict):
            round_errs["#root"] = ["Brain response must be a JSON object"]
        elif resp.get("_error"):
            round_errs["#root"] = [str(resp["_error"])]
        for aid, errs in check_claims(parsed.atoms, offer).items():
            round_errs.setdefault(aid, []).extend(errs)
        for aid, errs in check_language(parsed.atoms).items():
            round_errs.setdefault(aid, []).extend(errs)
        for aid, errs in atom_text_quality_errors(parsed.atoms).items():
            round_errs.setdefault(aid, []).extend(errs)
        for aid, errs in atom_target_text_policy_errors(
            parsed.atoms, getattr(offer, "branding_policy", TARGET_BRAND_ALLOWED)
        ).items():
            round_errs.setdefault(aid, []).extend(errs)
        # Branding/no-added-text QA is injected by campaign policy. Ignore duplicate Brain-authored
        # checks instead of spending correction rounds asking the model to remove them repeatedly.
        reserved_system_errors = reserved_system_atom_errors(
            parsed.atoms,
            resolved_brand_marks(getattr(offer, "branding_policy", TARGET_BRAND_ALLOWED),
                                 getattr(offer, "approved_brand_marks", ()),
                                 getattr(offer, "name", ""), getattr(offer, "product_type", "")),
        )
        # System-owned policy checks are discarded: code injects and verifies them.
        # A branded visual atom is different: it is a repairable model-authored atom
        # and its exact reason must enter the normal bounded Brain retry loop.
        reserved_system_atom_ids = {
            aid for aid, errors in reserved_system_errors.items()
            if any(str(error).startswith("SYSTEM_OWNED_QA:") for error in errors)
        }
        for aid, errors in reserved_system_errors.items():
            if any(str(error).startswith("BRAND_IN_VISUAL_ATOM:") for error in errors):
                round_errs.setdefault(aid, []).extend(errors)
        if vertical is not None:
            for aid, errs in check_vertical(parsed.atoms, offer, vertical).items():
                round_errs.setdefault(aid, []).extend(errs)

        def retry_family(atom_id):
            return atom_role(atom_id)

        valid_incoming = [
            a for a in parsed.atoms
            if a.id not in round_errs and a.id not in reserved_system_atom_ids
        ]
        incoming_families = {family for family in (retry_family(a.id) for a in valid_incoming) if family}
        # A new suffix is permitted only when an active structural key explicitly
        # requires a role, never when a direct atom ID was requested for repair.
        if not retry_target or retry_target.get("mode") in ("role", "role_shot"):
            replacement_families = incoming_families
        else:
            replacement_families = set()
        for old_id in list(by_id):
            if retry_family(old_id) in replacement_families:
                by_id.pop(old_id, None)
                accepted_atom_provenance.pop(old_id, None)
        for old_id in list(pending):
            if not old_id.startswith("#") and retry_family(old_id) in replacement_families:
                pending.pop(old_id, None)

        for atom in parsed.atoms:
            if atom.id in reserved_system_atom_ids:
                by_id.pop(atom.id, None)
                accepted_atom_provenance.pop(atom.id, None)
                pending.pop(atom.id, None)
                continue
            if atom.id in round_errs:
                by_id.pop(atom.id, None)
                accepted_atom_provenance.pop(atom.id, None)
            else:
                by_id[atom.id] = atom
                pending.pop(atom.id, None)
                accepted_atom_provenance[atom.id] = {
                    **_call_provenance(call),
                    "round": round_number,
                    "retry_strategy": strategy["name"],
                }

        # Narrow retries only update their selected issue. Other atom errors remain
        # pending until their own repair round; structural errors are recomputed below.
        if retry_target:
            pending = {
                key: value for key, value in previous_pending.items()
                if not _is_recomputed_structural_key(key) and key != retry_target["key"]
            }
            pending.update(round_errs)
            if retry_target["mode"] == "blocked":
                pending[retry_target["key"]] = previous_pending.get(
                    retry_target["key"], ["RETRY_TARGET_UNMAPPED: no safe atom ID is authorized for this structural error"]
                )
            target_satisfied = False
            if retry_target["mode"] == "exact":
                target_satisfied = any(a.id == retry_target["id"] for a in valid_incoming)
            elif retry_target["mode"] == "role":
                target_satisfied = any(atom_role(a.id) == retry_target["role"] for a in valid_incoming)
            elif retry_target["mode"] == "role_shot":
                target_satisfied = any(
                    atom_role(a.id) == retry_target["role"] and a.shot == retry_target["shot"]
                    and a.channel == "image" and a.priority == "MUST" for a in valid_incoming
                )
            elif retry_target["mode"] == "shot":
                target_satisfied = any(a.shot == retry_target["shot"] and a.channel == "image" and a.priority == "MUST" for a in valid_incoming)
            elif retry_target["mode"] == "decision":
                target_satisfied = not any(key == retry_target["key"] for key in round_errs)
            elif retry_target["mode"] == "blocked":
                target_satisfied = False
            elif retry_target["mode"] == "full":
                target_satisfied = not any(key.startswith("#") for key in round_errs)
            if target_satisfied:
                pending.pop(retry_target["key"], None)
        else:
            pending = {k: v for k, v in pending.items() if not k.startswith("#")}
            pending.update(round_errs)

        for k in range(1, shots + 1):
            if not any(a.shot == k and a.channel == "image" and a.priority == "MUST" for a in by_id.values()):
                pending[f"#shot_{k}"] = [f"shot {k} has no MUST image atom"]
        for name in missing_mechanism_links(by_id.values(), required_links):
            pending[f"#mechanism_{name}"] = [f"mechanism link '{name}' has no MUST image atom with id prefix mech_{name}_"]
        for name in missing_keep_atoms(by_id.values(), fit_gate):
            pending[f"#keep_{name}"] = [f"the Gate kept source element '{name}': add a separate MUST image atom with id prefix keep_{name}_ that visibly carries the approved target_form"]
        structure_errors = {
            **visual_structure_errors(by_id.values(), shots, required_links, fit_gate=fit_gate),
            **composition_form_errors(by_id.values(), fit_gate),
            **atom_budget_errors(by_id.values(), shots, fit_gate=fit_gate),
        }
        pending.update(structure_errors)
        for aid in list(structure_errors):
            if not aid.startswith("#"):
                by_id.pop(aid, None)
                accepted_atom_provenance.pop(aid, None)
        for name in missing_mechanism_links(by_id.values(), required_links):
            pending[f"#mechanism_{name}"] = [f"mechanism link '{name}' has no MUST image atom with id prefix mech_{name}_"]
        for name in missing_keep_atoms(by_id.values(), fit_gate):
            pending[f"#keep_{name}"] = [f"the Gate kept source element '{name}': add a separate MUST image atom with id prefix keep_{name}_ that visibly carries the approved target_form"]
        structure_errors = {
            **visual_structure_errors(by_id.values(), shots, required_links, fit_gate=fit_gate),
            **composition_form_errors(by_id.values(), fit_gate),
            **atom_budget_errors(by_id.values(), shots, fit_gate=fit_gate),
        }
        pending.update(structure_errors)
        # A composition atom rejected for unfinished wording still exists as a logical
        # requirement. Do not turn that one atom-level repair into a second, contradictory
        # request to create a new composition atom. The structural requirement remains
        # active and is re-evaluated as soon as a valid comp_* replacement arrives.
        incomplete_comp_pending = any(
            key.startswith("comp_")
            and any(str(error).startswith("TEXT_REWRITE_REQUIRED") for error in errors)
            for key, errors in pending.items()
        )
        composition_pending = pending.get("#composition") or []
        if incomplete_comp_pending and composition_pending and all(
            str(error).startswith("COMPOSITION_REQUIRED") for error in composition_pending
        ):
            pending.pop("#composition", None)
        if source_transfer:
            # Atom repair must not let the model silently rewrite unrelated preserved-ID
            # references. Keep a draft of the original decision and merge only the entry
            # corresponding to a structurally requested atom role. A decision-only retry
            # may replace the full decision because that is its explicit repair target.
            if isinstance(offered, dict):
                if not retry_target or retry_target.get("mode") in ("full", "decision") or decision_draft is None:
                    decision_draft = copy.deepcopy(offered)
                else:
                    target_role = retry_target.get("role")
                    target_property = None
                    if target_role and target_role.startswith("keep_"):
                        target_property = target_role[len("keep_"):]
                    elif target_role and target_role.startswith("mech_"):
                        target_property = target_role[len("mech_"):]
                    if target_property:
                        merged = copy.deepcopy(decision_draft)
                        old_preserved = merged.get("preserved") if isinstance(merged.get("preserved"), list) else []
                        new_preserved = offered.get("preserved") if isinstance(offered.get("preserved"), list) else []
                        replacement = next((item for item in new_preserved
                                            if isinstance(item, dict) and item.get("property") == target_property), None)
                        if replacement:
                            old_preserved = [item for item in old_preserved
                                             if not (isinstance(item, dict) and item.get("property") == target_property)]
                            old_preserved.append(copy.deepcopy(replacement))
                            merged["preserved"] = old_preserved
                        decision_draft = merged
            candidate_decision = copy.deepcopy(decision_draft) if isinstance(decision_draft, dict) else held_decision
            fit_gate = source_transfer.get("target_fit_gate") if isinstance(source_transfer, dict) else None
            if isinstance(fit_gate, dict) and isinstance(candidate_decision, dict) and candidate_decision.get("transferable") is True:
                candidate_decision = dict(candidate_decision)
                approved_mechanism = fit_gate.get("abstract_mechanism")
                if isinstance(approved_mechanism, str) and approved_mechanism.strip():
                    if _normalized_evidence(candidate_decision.get("abstract_mechanism", "")) != _normalized_evidence(approved_mechanism):
                        record["abstract_mechanism_canonicalized"] = True
                    candidate_decision["abstract_mechanism"] = approved_mechanism
                    record["abstract_mechanism_source"] = "target_fit_gate"
            decision_errors = validate_transfer_decision(candidate_decision, by_id.values(), fit_gate=fit_gate)
            if decision_errors:
                pending["#transfer_decision"] = decision_errors
                held_decision = None
            else:
                held_decision = candidate_decision
                record["transfer_decision_source"] = _call_provenance(call) if isinstance(offered, dict) else "previous_valid_round"
        record["accepted_ids"] = sorted(by_id)
        record["accepted_atom_provenance"] = {key: accepted_atom_provenance[key] for key in sorted(accepted_atom_provenance)}
        if isinstance(atom_provenance_out, dict):
            atom_provenance_out.clear()
            atom_provenance_out.update(accepted_atom_provenance)
        record["pending"] = {k: list(v) for k, v in pending.items()}
        if not pending:
            if isinstance(decision_out, dict) and held_decision is not None:
                decision_out.clear()
                decision_out.update(held_decision)
            if isinstance(atom_provenance_out, dict):
                atom_provenance_out.clear()
                atom_provenance_out.update(accepted_atom_provenance)
            return list(by_id.values())
        prev_text = {i["id"]: i.get("text", "") for i in raw if isinstance(i, dict) and i.get("id")}
        failed = [{"id": k, "errors": v, "previous_text": prev_text.get(k, "")} for k, v in pending.items()]
    raise BrainError(pending)


def apply_patch(atoms, changes):
    """Patch by id. Brain returns only changed atoms; deleting a MUST needs a stated reason."""
    cur = {a.id: a for a in atoms}
    lost = []
    for ch in changes:
        aid, action = ch.get("id"), ch.get("action")
        if action == "delete":
            old = cur.get(aid)
            if old is None:
                raise PatchError(f"cannot delete unknown atom {aid}")
            if old.priority == "MUST" and not str(ch.get("reason", "")).strip():
                raise PatchError(f"deleting MUST atom {aid} requires a reason")
            lost.append(f"{aid} ({old.priority}): {ch.get('reason', 'no reason given')}")
            del cur[aid]
        elif action == "update":
            parsed = parse_atoms([ch.get("atom", {})])
            if parsed.errors or not parsed.atoms or parsed.atoms[0].id != aid:
                raise PatchError(f"invalid update for {aid}: {parsed.errors}")
            new = parsed.atoms[0]
            old = cur.get(aid)
            if old and old.priority == "MUST" and new.priority != "MUST":
                if not str(ch.get("reason", "")).strip():
                    raise PatchError(f"downgrading MUST atom {aid} requires a reason")
                lost.append(f"{aid} (MUST→{new.priority}): {ch['reason']}")
            cur[aid] = new
        else:
            raise PatchError(f"unknown action {action}")
    report = "потеряно: " + ("; ".join(lost) if lost else "нет")
    return list(cur.values()), report
