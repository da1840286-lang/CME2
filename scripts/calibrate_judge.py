#!/usr/bin/env python3
"""Calibrate the QA judge on existing frames; no image generation is performed.

Example:
  OPENROUTER_API_KEY=... python scripts/calibrate_judge.py --settings settings.json --catalog adapters.json \
      --atoms atoms.json --frames shot_01_a01.png shot_01_a02.png --labels labels.json \
      --reference-image target_product_reference.jpg --repeats 5 --out calib_out

labels.json example: {"shot_01_a02.png": {"mech_result_1": false, "mech_intervention_1": true}}.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cme.atoms import Atom  # noqa: E402
from cme.calibration import make_synthetic_negatives, render_markdown, run_calibration, verdict_hint  # noqa: E402
from cme.providers import CallLog  # noqa: E402
from cme.registry import load_catalog, load_settings, make_judge  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--atoms", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--labels")
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

    frames = {Path(p).name: Path(p) for p in args.frames}
    frames.update(make_synthetic_negatives(out / "synthetic", args.reference_image))
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

    report = run_calibration(judges, frames, atoms, labels=labels, repeats=args.repeats, refs=refs)
    report["hint"] = verdict_hint(report)
    (out / "calibration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "calibration_report.md").write_text(render_markdown(report) + "\n" + report["hint"] + "\n", encoding="utf-8")
    print(render_markdown(report))
    print(report["hint"])
    return 0 if report["hint"].startswith("OK on this labeled sample") else 2


if __name__ == "__main__":
    raise SystemExit(main())


