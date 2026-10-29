"""Manual analysis of one selected ad: fetch the video, take frames with timestamps, ask the analyst once.

Never runs automatically. The model sees only the extracted frames, and the record says so.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

SECTIONS = ("hook", "buyer_problem", "product_intervention", "result_demo", "cta", "mechanism_hypothesis", "transferability")
ALWAYS_INFERRED = ("mechanism_hypothesis", "transferability")
RATINGS = ("problem_clarity", "product_demo", "proof_visibility", "sequence_quality")
TOLERANCE_SEC = 0.6


def ffmpeg_executable(which=shutil.which) -> str:
    """Prefer the FFmpeg binary bundled by imageio-ffmpeg; fall back to system FFmpeg."""
    try:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        if executable:
            return str(executable)
    except (ImportError, OSError, RuntimeError):
        pass
    executable = which("ffmpeg")
    if executable:
        return str(executable)
    raise RuntimeError("FFmpeg binary not found; install imageio-ffmpeg or system ffmpeg")


def tools_status(which=shutil.which) -> dict:
    try:
        ffmpeg_executable(which)
    except RuntimeError:
        return {"state": "NOT_CONFIGURED", "detail": "FFmpeg binary not found",
                "how_to_fix": "install the imageio-ffmpeg Python dependency or system FFmpeg"}
    return {"state": "READY", "detail": "", "how_to_fix": ""}


def fetch_video(url: str, dest, max_mb: int = 60, timeout: int = 60, getter=None) -> Path:
    if getter is None:
        import requests
        getter = requests.get
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = getter(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    if getattr(r, "status_code", 200) >= 400:
        raise RuntimeError(f"video HTTP {r.status_code} (the preview link may have expired)")
    total = 0
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            total += len(chunk)
            if total > max_mb * 1024 * 1024:
                raise RuntimeError(f"video larger than {max_mb} MB, not downloaded")
            f.write(chunk)
    if total == 0:
        raise RuntimeError("empty video response")
    return dest


def _run(cmd, runner=None):
    runner = runner or (lambda c: subprocess.run(c, capture_output=True, text=True, timeout=90))
    res = runner(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {(res.stderr or '')[-200:]}")
    return res


ADAPTIVE_SPACING_SEC = 4.0   # target gap between frames
ADAPTIVE_MIN_FRAMES = 6
ADAPTIVE_MAX_FRAMES = 10      # keeps the vision request within free-model limits
START_ANCHOR_SEC = 0.5        # the hook lives in the first second
END_MARGIN_SEC = 0.4          # the final shot (result, CTA) often carries the proof


def plan_frame_times(duration: float, count: int | None = None) -> tuple[list[float], dict]:
    """Timestamps to sample, plus a description of the plan.

    count=None (default): adaptive. About one frame per ADAPTIVE_SPACING_SEC, clamped to
    [ADAPTIVE_MIN_FRAMES, ADAPTIVE_MAX_FRAMES], evenly spread from just after the start to just before the
    end, so the opening hook and the final shot are always sampled. The earlier midpoint scheme never sampled
    the first or last ~half interval, which for a 21 s ad meant the closing result shot (the last ~1.8 s) was
    never seen.
    count=N: the previous behaviour (N evenly spaced midpoints), kept for explicit callers.
    """
    if count is not None:
        times = [round(duration * (i + 0.5) / count, 2) for i in range(count)]
        return times, {"mode": "fixed_midpoints", "count": count}
    last = max(duration - END_MARGIN_SEC, duration / 2)
    first = min(START_ANCHOR_SEC, last / 2)
    if duration <= 1.0:
        return [round(duration / 2, 2)], {"mode": "adaptive_anchored", "count": 1}
    import math
    n = max(ADAPTIVE_MIN_FRAMES, min(ADAPTIVE_MAX_FRAMES, math.ceil(duration / ADAPTIVE_SPACING_SEC)))
    step = (last - first) / (n - 1)
    times = [round(first + i * step, 2) for i in range(n)]
    times[-1] = round(last, 2)
    return times, {"mode": "adaptive_anchored", "count": n, "target_spacing_sec": ADAPTIVE_SPACING_SEC,
                   "actual_spacing_sec": round(step, 2), "start_anchor_sec": times[0], "end_anchor_sec": times[-1],
                   "capped": math.ceil(duration / ADAPTIVE_SPACING_SEC) > ADAPTIVE_MAX_FRAMES}


def extract_frames(video_path, out_dir, count: int | None = None, runner=None, info: dict | None = None) -> list:
    """Frames with their timestamps (seconds), using one FFmpeg binary. See plan_frame_times for the sampling."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_executable()
    probe_runner = runner or (lambda c: subprocess.run(c, capture_output=True, text=True, timeout=30))
    # FFmpeg prints container duration while inspecting the input, so ffprobe is not
    # required. A non-zero status is expected because no output file is requested.
    probe = probe_runner([ffmpeg, "-hide_banner", "-i", str(video_path)])
    probe_text = "\n".join(
        value for value in (getattr(probe, "stderr", ""), getattr(probe, "stdout", "")) if value
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe_text)
    if not match:
        raise RuntimeError("cannot read the video duration from FFmpeg input metadata")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if duration <= 0:
        raise RuntimeError("video duration must be positive")
    times, plan = plan_frame_times(duration, count)
    if info is not None:
        info.update({"duration_sec": round(duration, 3), **plan})
    frames = []
    for i, t in enumerate(times):
        path = out / f"frame_{i:02d}_{t:.2f}s.jpg"
        _run([ffmpeg, "-y", "-v", "error", "-ss", str(t), "-i", str(video_path), "-frames:v", "1",
              "-vf", "scale=640:-2", "-q:v", "4", str(path)], runner)
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"no frame produced at {t}s")
        frames.append({"t_sec": t, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return frames

def validate_analysis(data, frame_times):
    """Return (clean_result, warnings). Raises ValueError when the structure cannot be used."""
    if not isinstance(data, dict):
        raise ValueError("the model did not return a JSON object")
    warnings, out = [], {}
    for k in SECTIONS:
        sec = data.get(k)
        if not isinstance(sec, dict) or not str(sec.get("text", "")).strip():
            raise ValueError(f"section {k!r} is missing or empty")
        kind = sec.get("kind")
        if k in ALWAYS_INFERRED and kind != "INFERRED":
            warnings.append(f"{k}: kind forced to INFERRED (it is a hypothesis by definition)")
            kind = "INFERRED"
        if kind not in ("OBSERVED", "INFERRED"):
            warnings.append(f"{k}: unknown kind {kind!r}, treated as INFERRED")
            kind = "INFERRED"
        out[k] = {"text": str(sec["text"]).strip(), "kind": kind}
    evidence = []
    valid_sections = set(SECTIONS)
    for row_index, e in enumerate(data.get("evidence") or []):
        if not isinstance(e, dict):
            warnings.append(f"evidence row {row_index} was dropped: expected an object")
            continue
        try:
            t = float(e["t_sec"])
        except (KeyError, TypeError, ValueError, OverflowError):
            warnings.append("evidence row without a valid t_sec was dropped")
            continue
        frame = e.get("frame")
        if type(frame) is not int or not 0 <= frame < len(frame_times):
            warnings.append(f"evidence at {t}s dropped: frame index is invalid")
            continue
        if abs(t - float(frame_times[frame])) > TOLERANCE_SEC:
            warnings.append(
                f"evidence at {t}s dropped: timestamp does not match indexed frame {frame} "
                f"({frame_times[frame]}s)"
            )
            continue
        observation = str(e.get("observation") or "").strip()
        if not observation:
            warnings.append(f"evidence at {t}s dropped: observation is empty")
            continue
        raw_supports = e.get("supports")
        if not isinstance(raw_supports, list):
            warnings.append(f"evidence at {t}s has no section supports; it cannot verify OBSERVED claims")
            raw_supports = []
        supports = []
        for support in raw_supports:
            key = str(support or "").strip()
            if key in valid_sections and key not in supports:
                supports.append(key)
            elif key:
                warnings.append(f"evidence at {t}s dropped unsupported section key {key!r}")
        evidence.append({"t_sec": t, "observation": observation, "frame": frame, "supports": supports})
    out["evidence"] = evidence
    for section_name, section in out.items():
        if section_name in valid_sections and isinstance(section, dict) and section.get("kind") == "OBSERVED":
            supported = any(section_name in row.get("supports", []) for row in evidence)
            if not supported:
                raise ValueError(
                    f"OBSERVED section {section_name!r} has no valid frame-linked evidence that supports it"
                )
    lim = data.get("limits") if isinstance(data.get("limits"), dict) else {}
    out["limits"] = {"text": str(lim.get("text", "")).strip(), "missing": [str(x) for x in (lim.get("missing") or [])]}
    ratings = data.get("ratings") if isinstance(data.get("ratings"), dict) else {}
    out["ratings"] = {r: (ratings.get(r) if isinstance(ratings.get(r), int) and 1 <= ratings.get(r) <= 5 else None) for r in RATINGS}
    out["warnings"] = warnings
    return out, warnings


def analyze_candidate(store, cid: str, analyst, data_dir, analyst_info=lambda: {}, frames_count: int | None = None,
                      getter=None, runner=None, tools=None) -> dict:
    """One explicit analysis. Every attempt that reaches a provider or the video is recorded."""
    c = store.get(cid)
    if c is None:
        raise KeyError(cid)
    tools = tools or tools_status()
    if tools["state"] != "READY":
        return {"status": "NOT_CONFIGURED", **tools}
    if not c.get("video_url"):
        n = store.save_analysis(cid, "NO_VIDEO", error="the source record has no video URL")
        return {"status": "NO_VIDEO", "detail": "the source record has no video URL", "analysis_id": n}
    work = Path(data_dir) / "media" / hashlib.sha256(cid.encode()).hexdigest()[:16]
    try:
        video = fetch_video(c["video_url"], work / "source.mp4", getter=getter)
        sampling = {}
        frames = extract_frames(video, work / "frames", frames_count, runner, info=sampling)
    except Exception as e:
        n = store.save_analysis(cid, "VIDEO_UNAVAILABLE", error=str(e))
        return {"status": "VIDEO_UNAVAILABLE", "detail": str(e), "analysis_id": n}
    material = {"candidate_id": cid, "ad_id": c.get("ad_id"), "source_url": c.get("source_url"), "video_url": c.get("video_url"),
                "video_sha256": hashlib.sha256(Path(video).read_bytes()).hexdigest(), "frames": frames,
                "frames_sent": len(frames), "frame_sampling": sampling, "video_fully_viewed": False,
                "note": "The model received only these frames, not the whole video."}
    context = {k: c.get(k) for k in ("brand", "brand_name", "title", "industry", "landing_page_url", "duration_seconds")}
    try:
        out = analyst(frames, context)  # the registry wrapper returns the adapter result; meta goes to the call log
    except Exception as e:
        n = store.save_analysis(cid, "API_ERROR", material=material, error=f"{type(e).__name__}: {e}")
        return {"status": "API_ERROR", "detail": f"{type(e).__name__}: {e}", "analysis_id": n}
    info = analyst_info() or {}
    try:
        result, warnings = validate_analysis(out.get("data"), [f["t_sec"] for f in frames])
    except ValueError as e:
        n = store.save_analysis(cid, "INVALID_OUTPUT", info.get("model_id", ""), info.get("responded_model") or "", material,
                                {"raw_excerpt": out.get("raw_excerpt", "")}, str(e))
        return {"status": "INVALID_OUTPUT", "detail": str(e), "analysis_id": n}
    n = store.save_analysis(cid, "OK", info.get("model_id", ""), info.get("responded_model") or "", material, result)
    return {"status": "OK", "analysis_id": n, "warnings": warnings}
