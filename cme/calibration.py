"""Offline/live calibration harness for a visual QA judge.

The harness measures false passes, missed passes, invalid answers, and repeatability. It never generates
images. A calibration cannot claim OK without both labeled positive and negative examples and valid calls.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

NEGATIVE_PREFIX = "synthetic:"


def make_synthetic_negatives(out_dir, reference_image=None, size=(576, 1024)):
    """Create negative controls: a blank canvas and optionally the bare target product reference."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    blank = out_dir / "synthetic_blank.png"
    Image.new("RGB", size, (128, 128, 128)).save(blank)
    paths[NEGATIVE_PREFIX + "blank"] = blank
    if reference_image:
        ref = Image.open(reference_image).convert("RGB")
        ref.thumbnail(size)
        canvas = Image.new("RGB", size, (255, 255, 255))
        canvas.paste(ref, ((size[0] - ref.width) // 2, (size[1] - ref.height) // 2))
        product_only = out_dir / "synthetic_product_only.png"
        canvas.save(product_only)
        paths[NEGATIVE_PREFIX + "product_only"] = product_only
    return paths


def _one_call(judge, frame, atoms, refs):
    try:
        answers = judge(str(frame), atoms, refs) if refs else judge(str(frame), atoms)
        if isinstance(answers, tuple):
            answers = answers[0]
    except Exception as exc:  # noqa: BLE001 - a failing call is calibration data, not a crash
        return {a.id: ("NOT_RUN", f"{type(exc).__name__}: {exc}"[:120]) for a in atoms}
    out = {}
    for atom in atoms:
        row = answers.get(atom.id) if isinstance(answers, dict) else None
        if isinstance(row, dict) and type(row.get("pass")) is bool and str(row.get("evidence") or "").strip():
            out[atom.id] = ("PASS" if row["pass"] else "FAIL", str(row["evidence"])[:120])
        else:
            out[atom.id] = ("NOT_RUN", "no valid answer")
    return out


def run_calibration(judge_factories: dict, frames: dict, atoms, *, labels=None, repeats=5, refs=()):
    """Run repeated checks. labels maps frame name -> atom id -> bool (True means shown).

    Frames named synthetic:* are negative for every atom. Unlabeled real frames do not affect accuracy rates,
    but still contribute to invalid-call and repeatability measurements.
    """
    labels = labels or {}
    report = {"repeats": repeats, "atoms": [a.id for a in atoms], "modes": {}}
    for mode, judge in judge_factories.items():
        cells = defaultdict(list)
        evidence = {}
        for name, path in frames.items():
            for _ in range(repeats):
                for atom_id, (status, ev) in _one_call(judge, path, atoms, refs).items():
                    cells[(name, atom_id)].append(status)
                    evidence.setdefault((name, atom_id), ev)
        rows, fp, neg_total, missed, pos_total, unstable = [], 0, 0, 0, 0, 0
        not_run_calls, total_calls = 0, 0
        positive_cells, negative_cells = 0, 0
        for (name, atom_id), statuses in cells.items():
            passes = statuses.count("PASS")
            rate = passes / len(statuses)
            expected = False if name.startswith(NEGATIVE_PREFIX) else (labels.get(name) or {}).get(atom_id)
            total_calls += len(statuses)
            not_run_calls += statuses.count("NOT_RUN")
            if len(set(statuses)) > 1:
                unstable += 1
            if expected is False:
                negative_cells += 1
                neg_total += len(statuses)
                fp += passes
            elif expected is True:
                positive_cells += 1
                pos_total += len(statuses)
                missed += len(statuses) - passes
            rows.append({"frame": name, "atom": atom_id, "pass_rate": round(rate, 2),
                         "statuses": {s: statuses.count(s) for s in sorted(set(statuses))},
                         "expected_shown": expected, "sample_evidence": evidence.get((name, atom_id), "")})
        report["modes"][mode] = {
            "false_pass_rate": round(fp / neg_total, 3) if neg_total else None,
            "negative_calls": neg_total,
            "missed_pass_rate": round(missed / pos_total, 3) if pos_total else None,
            "positive_calls": pos_total,
            "not_run_calls": not_run_calls,
            "total_calls": total_calls,
            "not_run_rate": round(not_run_calls / total_calls, 3) if total_calls else None,
            "positive_cells": positive_cells,
            "negative_cells": negative_cells,
            "unstable_cells": unstable, "total_cells": len(cells),
            "cells": sorted(rows, key=lambda r: (r["frame"], r["atom"])),
        }
    return report


def render_markdown(report) -> str:
    lines = ["# Judge calibration", "", f"repeats per frame: {report['repeats']}; atoms: {', '.join(report['atoms'])}", ""]
    lines += ["| mode | false-pass rate (target 0) | missed-pass rate | NOT_RUN rate | unstable cells |",
              "|---|---|---|---|---|"]
    for mode, m in report["modes"].items():
        lines.append(f"| {mode} | {m['false_pass_rate']} ({m['negative_calls']} calls) | "
                     f"{m['missed_pass_rate']} ({m['positive_calls']} calls) | "
                     f"{m['not_run_rate']} ({m['not_run_calls']}/{m['total_calls']}) | "
                     f"{m['unstable_cells']}/{m['total_cells']} |")
    for mode, m in report["modes"].items():
        lines += ["", f"## {mode}", "", "| frame | atom | pass rate | expected shown | statuses |", "|---|---|---|---|---|"]
        for r in m["cells"]:
            lines.append(f"| {r['frame']} | {r['atom']} | {r['pass_rate']} | {r['expected_shown']} | {r['statuses']} |")
    return "\n".join(lines) + "\n"


def verdict_hint(report) -> str:
    obs = report["modes"].get("observe_verify")
    if not obs:
        return "observe_verify was not run"
    if obs["false_pass_rate"] not in (None, 0.0):
        return "FAIL: observe_verify passes frames that do not show the requirement - do not ship."
    if obs["total_calls"] == 0 or not obs["positive_cells"] or not obs["negative_cells"]:
        return "INCOMPLETE: label at least one positive and one negative real example; synthetic negatives alone are insufficient."
    if obs["not_run_rate"] not in (None, 0.0):
        return "INCOMPLETE: one or more judge calls were NOT_RUN; zero false passes is not conclusive."
    if obs["unstable_cells"]:
        return "UNSTABLE: identical calls disagree - use a stronger judge or a second independent vote."
    if obs["missed_pass_rate"] is not None and obs["missed_pass_rate"] > 0.5:
        return "STRICT: no false passes but most true examples are rejected - inspect atom wording before loosening the judge."
    return "OK on this labeled sample: no false passes and repeatable decisions. Confirm with a live generation before shipping."


