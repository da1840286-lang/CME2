"""Manual analysis of one selected ad: fetch the video, take frames with timestamps, ask the analyst once.

Never runs automatically. The model sees only the extracted frames, and the record says so.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SECTIONS = ("hook", "buyer_problem", "product_intervention", "result_demo", "cta", "mechanism_hypothesis", "transferability")
ALWAYS_INFERRED = ("mechanism_hypothesis", "transferability")
RATINGS = ("problem_clarity", "product_demo", "proof_visibility", "sequence_quality")
TOLERANCE_SEC = 0.6


def tools_status(which=shutil.which) -> dict:
    if not (which("ffmpeg") and which("ffprobe")):
        return {"state": "NOT_CONFIGURED", "detail": "ffmpeg/ffprobe not found",
                "how_to_fix": "install ffmpeg (it provides ffprobe) and make sure both are on PATH"}
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


def extract_frames(video_path, out_dir, count: int = 6, runner=None) -> list:
    """Evenly spaced frames with their timestamps (seconds)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)], runner)
    try:
        duration = float(str(res.stdout).strip().splitlines()[0])
    except (ValueError, IndexError) as e:
        raise RuntimeError("cannot read the video duration") from e
    frames = []
    for i in range(count):
        t = round(duration * (i + 0.5) / count, 2)
        path = out / f"frame_{i:02d}_{t:.2f}s.jpg"
        _run(["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", str(video_path), "-frames:v", "1",
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
    for e in data.get("evidence") or []:
        try:
            t = float(e["t_sec"])
        except (KeyError, TypeError, ValueError):
            warnings.append("evidence row without a valid t_sec was dropped")
            continue
        if not any(abs(t - ft) <= TOLERANCE_SEC for ft in frame_times):
            warnings.append(f"evidence at {t}s dropped: no frame was sent at that time")
            continue
        evidence.append({"t_sec": t, "observation": str(e.get("observation", "")).strip(), "frame": e.get("frame")})
    out["evidence"] = evidence
    lim = data.get("limits") if isinstance(data.get("limits"), dict) else {}
    out["limits"] = {"text": str(lim.get("text", "")).strip(), "missing": [str(x) for x in (lim.get("missing") or [])]}
    ratings = data.get("ratings") if isinstance(data.get("ratings"), dict) else {}
    out["ratings"] = {r: (ratings.get(r) if isinstance(ratings.get(r), int) and 1 <= ratings.get(r) <= 5 else None) for r in RATINGS}
    out["warnings"] = warnings
    return out, warnings


def analyze_candidate(store, cid: str, analyst, data_dir, analyst_info=lambda: {}, frames_count: int = 6,
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
        frames = extract_frames(video, work / "frames", frames_count, runner)
    except Exception as e:
        n = store.save_analysis(cid, "VIDEO_UNAVAILABLE", error=str(e))
        return {"status": "VIDEO_UNAVAILABLE", "detail": str(e), "analysis_id": n}
    material = {"candidate_id": cid, "ad_id": c.get("ad_id"), "source_url": c.get("source_url"), "video_url": c.get("video_url"),
                "video_sha256": hashlib.sha256(Path(video).read_bytes()).hexdigest(), "frames": frames,
                "frames_sent": len(frames), "video_fully_viewed": False,
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
