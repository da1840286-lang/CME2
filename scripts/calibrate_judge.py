#!/usr/bin/env python3
"""Calibrate the QA judge on existing frames; no image generation is performed.

Example:
  OPENROUTER_API_KEY=... python scripts/calibrate_judge.py --settings settings.json --catalog adapters.json \
      --atoms atoms.json --frames shot_01_a01.png shot_01_a02.png --labels labels.json \
      --reference-image target_product_reference.jpg --branding-policy target_brand_allowed \
      --approved-brand-marks SEESE --repeats 5 --out calib_out

labels.json example:
  {"shot_01_a01.png": {"mech_result_1": false, "system_no_text_policy": true,
                        "target_branding_reference": true, "cme_product_identity_ref": true},
   "shot_01_a02.png": {"mech_result_1": true, "system_no_text_policy": false,
                        "target_branding_reference": false, "cme_product_identity_ref": false}}
Every included system check needs at least one real positive and one real negative label.
Synthetic controls are diagnostic only; they do not count as labeled ground truth.
For a reserve model, pass a separate settings file with judge_adapter set to that model and
judge_fallback_adapter empty, so the report measures the reserve itself.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cme.atoms import Atom  # noqa: E402
from cme.branding import NO_BRANDING, TARGET_BRAND_ALLOWED  # noqa: E402
from cme.calibration import (  # noqa: E402
    build_calibration_atoms, make_synthetic_negatives, render_markdown, run_calibration, verdict_hint,
)
from cme.qa import PRODUCT_IDENTITY_CHECK_ID, SYSTEM_NO_TEXT_ID, TARGET_BRANDING_CHECK_ID  # noqa: E402
from cme.providers import CallLog  # noqa: E402
from cme.registry import load_catalog, load_settings, make_judge  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--atoms", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--branding-policy", choices=(NO_BRANDING, TARGET_BRAND_ALLOWED),
                    default=TARGET_BRAND_ALLOWED)
    ap.add_argument("--approved-brand-marks", nargs="*", default=[])
    ap.add_argument("--reference-image")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--modes", default="single_pass,observe_verify")
    ap.add_argument("--priorities", default="MUST")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--out", default="calibration_out")
    args = ap.parse_args(argv)
    if args.repeats < 1:
        ap.error("--repeats must be at least 1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {p.strip().upper() for p in args.priorities.split(",") if p.strip()}
    rows = json.loads(Path(args.atoms).read_text(encoding="utf-8"))
    atoms = [Atom(**{k: row[k] for k in ("id", "kind", "text", "priority", "channel", "shot")}) for row in rows]
    atoms = [a for a in atoms if a.channel == "image" and a.priority in wanted and a.shot == args.shot]
    if not atoms:
        ap.error("no image atoms with the requested priority/shot in atoms file")

    for path in frames.values():
        if not path.is_file():
            ap.error(f"frame does not exist: {path}")
    frames = {Path(p).name: Path(p) for p in args.frames}
    frames.update(make_synthetic_negatives(out / "synthetic", args.reference_image))
    if not isinstance(labels, dict):
        ap.error("--labels must contain a JSON object mapping frame filenames to check-id booleans")
    real_frame_names = {Path(p).name for p in args.frames}
    unknown_frames = sorted(set(labels) - real_frame_names)
    if unknown_frames:
        ap.error("labels reference frames not supplied via --frames: " + ", ".join(unknown_frames))
    for frame_name, frame_labels in labels.items():
        if not isinstance(frame_labels, dict) or any(type(value) is not bool for value in frame_labels.values()):
            ap.error(f"labels for {frame_name} must map check ids to JSON booleans")

    if args.reference_image and not Path(args.reference_image).is_file():
        ap.error(f"reference image does not exist: {args.reference_image}")
    system_atoms = build_calibration_atoms(
        atoms, shot=args.shot, branding_policy=args.branding_policy,
        approved_brand_marks=args.approved_brand_marks,
        include_brand_reference=bool(args.reference_image),
    )
    system_ids = [SYSTEM_NO_TEXT_ID]
    extra_check_ids = []
    if args.reference_image:
        extra_check_ids.append(PRODUCT_IDENTITY_CHECK_ID)
        if args.branding_policy == TARGET_BRAND_ALLOWED:
            system_ids.append(TARGET_BRANDING_CHECK_ID)
    balanced_ids = [*system_ids, *extra_check_ids]
    valid_check_ids = {atom.id for atom in system_atoms} | set(extra_check_ids)
    unknown_ids = sorted({check_id for frame_labels in labels.values()
                          for check_id in frame_labels if check_id not in valid_check_ids})
    if unknown_ids:
        ap.error("labels contain unknown check ids: " + ", ".join(unknown_ids))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8")) if args.labels else {}
    refs = [args.reference_image] if args.reference_image else []
    settings, catalog = load_settings(args.settings), load_catalog(args.catalog)
    log = CallLog(out / "calls.jsonl")
    judges = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        patched = dict(catalog)
        if settings.judge_adapter not in patched:
            ap.error(f"judge adapter {settings.judge_adapter!r} not found in catalog")
        spec = patched[settings.judge_adapter]
        patched[settings.judge_adapter] = dataclasses.replace(spec, params={**spec.params, "judge_mode": mode})
        judges[mode] = make_judge(settings, patched, log, f"calibration_{mode}")

    report = run_calibration(
        judges, frames, system_atoms, labels=labels, repeats=args.repeats, refs=refs,
        extra_check_ids=extra_check_ids, balanced_real_ids=balanced_ids,
    )
    report["branding_policy"] = args.branding_policy
    report["approved_brand_marks"] = list(args.approved_brand_marks)
    report["reference_image"] = Path(args.reference_image).name if args.reference_image else None
    report["synthetic_controls_are_scored_as_ground_truth"] = False
    report["hint"] = verdict_hint(report)
    (out / "calibration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "calibration_report.md").write_text(render_markdown(report) + "\n" + report["hint"] + "\n", encoding="utf-8")
    print(render_markdown(report))
    print(report["hint"])
    return 0 if report["hint"].startswith("OK on this labeled sample") else 2


if __name__ == "__main__":
    raise SystemExit(main())


