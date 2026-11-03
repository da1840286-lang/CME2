"""no_branding reference cleaning: text-region detector + LaMa inpainting restricted to the mask.

Contract (see cme.branding.prepare_unbranded_reference):
  * the original photo is never modified; a cleaned PNG is written to a separate destination;
  * LaMa only sees a local crop; its output is copied back ONLY where the mask is set, so every
    pixel outside the mask is bit-identical to the original (verified, not assumed);
  * a mask above MAX_MASK_COVERAGE (8% of the frame) is rejected, never repaired;
  * doubtful lettering never produces an automatic PASS: it ends as NEEDS_REVIEW or a failure.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

# ---- thresholds ---------------------------------------------------------------------------------
ACCEPT_CONF = 0.55          # recognised text with >=2 letters/digits and at least this confidence is lettering
ACCEPT_ROTATED_CONF = 0.65  # vertical/rotated reads are noisier, so they need a higher confidence
ACCEPT_SINGLE_CONF = 0.85   # a single upright character needs this confidence (e.g. the "S" size badge)
UNCERTAIN_CONF = 0.50       # anything below ACCEPT but at/above this is "doubtful", never silently ignored
MAX_MASK_COVERAGE = 0.08    # same ceiling as branding.MAX_REFERENCE_REPAIR_COVERAGE; must not be relaxed here
MASK_DILATE_PX = 7
GROW_COLOR_TOL = 38
GROW_MAX_BBOX_X = 1.6
GROW_MAX_AREA_X = 7.0
GROW_MIN_RECT_FILL = 0.88
CONTEXT_PX = 96
LAMA_MIN_SIDE = 512
LAMA_MAX_SIDE = 640
DETECT_ORIENTATIONS = (0, 90, 270)   # upright, plus the two vertical reading directions
SEAM_REVIEW_RATIO = 1.5              # calibrated on 11 images (clean repairs 0.3-1.2, SEESE battery 1.77); recalibrate on a wider set
TESS_MIN_CONF = 60
_WORD3 = re.compile(r"[A-Za-z0-9]{3,}")
_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")
_ANY_ALNUM = re.compile(r"[^\W_]", re.UNICODE)

LAMA_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


def lama_weights_path() -> Path:
    env = os.environ.get("CME_LAMA_WEIGHTS")
    return Path(env) if env else Path.home() / ".cache" / "cme" / "big-lama.pt"


def engine_status():
    """(ok, reason). The engine needs rapidocr-onnxruntime, torch (CPU is enough), and the LaMa weights."""
    try:
        import cv2  # noqa: F401
        import rapidocr_onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"detector unavailable: {type(exc).__name__}: {exc}"
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"torch unavailable: {type(exc).__name__}: {exc}"
    weights = lama_weights_path()
    if not weights.is_file():
        return False, f"LaMa weights missing at {weights} (run scripts/fetch_lama.py)"
    return True, "ok"


def verify_weights() -> bool:
    h = hashlib.sha256()
    with open(lama_weights_path(), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == LAMA_SHA256


# ---- detection -----------------------------------------------------------------------------------
_ocr = None
_lama = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def _bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b):
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / max(1e-6, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def _back_map(poly, orientation, w, h):
    """Map a polygon found in an image rotated by `orientation` back to the original frame (w x h)."""
    if orientation == 0:
        return [[float(x), float(y)] for x, y in poly]
    if orientation == 90:      # image was rotated clockwise: x' = h-1-y, y' = x
        return [[float(y), float(h - 1 - x)] for x, y in poly]
    # 270: rotated counter-clockwise: x' = y, y' = w-1-x
    return [[float(w - 1 - y), float(x)] for x, y in poly]


def detect(image_rgb: np.ndarray, orientations=DETECT_ORIENTATIONS):
    """Return (accepted, uncertain): lists of {poly, text, conf, orientation}."""
    import cv2
    h, w = image_rgb.shape[:2]
    found = []
    for ori in orientations:
        view = image_rgb
        if ori == 90:
            view = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
        elif ori == 270:
            view = cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        res, _ = _get_ocr()(cv2.cvtColor(np.ascontiguousarray(view), cv2.COLOR_RGB2BGR))
        for box, text, conf in (res or []):
            found.append({"poly": _back_map(box, ori, w, h), "text": str(text), "conf": float(conf), "orientation": ori})
    found.sort(key=lambda d: -d["conf"])
    unique = []
    for d in found:                       # same text seen in several orientations: keep the most confident
        if all(_iou(d["poly"], u["poly"]) < 0.4 for u in unique):
            unique.append(d)
    accepted, uncertain = [], []
    for d in unique:
        any_n = len(_ANY_ALNUM.findall(d["text"]))
        c = d["conf"]
        upright = d["orientation"] == 0
        accept_min = ACCEPT_CONF if upright else ACCEPT_ROTATED_CONF
        item = {**d, "conf": round(c, 3)}
        if any_n >= 2 and c >= accept_min:
            accepted.append(item)
        elif any_n == 1 and upright and c >= ACCEPT_SINGLE_CONF:
            accepted.append(item)           # e.g. the "S" size badge; only when read upright
        elif any_n >= 2 and upright and c >= UNCERTAIN_CONF:
            uncertain.append(item)          # doubtful upright lettering: never silently ignored
        # rotated reads below ACCEPT_ROTATED_CONF are dropped (a ring-shaped button read as "GX"), and
        # a lone character below ACCEPT_SINGLE_CONF, or any lone character in a rotated pass, is treated as
        # product geometry (a clip shaped like a "C", an icon), not as evidence of lettering
    return accepted, uncertain


# ---- mask ------------------------------------------------------------------------------------------
def grow_label(image_rgb, poly):
    """Extend a text box over the plate it sits on (sticker / printed panel) so no plate edge is left behind."""
    import cv2
    h, w = image_rgb.shape[:2]
    pts = np.array(poly, np.int32)
    box = np.zeros((h, w), np.uint8)
    cv2.fillPoly(box, [pts], 255)
    x, y, bw, bh = cv2.boundingRect(pts)
    pad = int(max(bw, bh) * 0.6) + 8
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
    roi = image_rgb[y0:y1, x0:x1].astype(np.int16)
    rb = box[y0:y1, x0:x1] > 0
    px = roi[rb]
    if len(px) < 8:
        return box
    lum = px.mean(axis=1)
    med = np.median(lum)
    dev = np.abs(lum - med)
    bg = px[dev <= np.percentile(dev, 60)]
    if len(bg) < 8:
        return box
    colour = np.median(bg, axis=0)
    close = (np.abs(roi - colour).max(axis=2) <= GROW_COLOR_TOL).astype(np.uint8)
    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, lab = cv2.connectedComponents(np.maximum(close, rb.astype(np.uint8)))
    keep = np.unique(lab[rb])
    keep = keep[keep > 0]
    region = np.isin(lab, keep).astype(np.uint8)
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return box
    filled = np.zeros_like(region)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    if filled[0, :].any() or filled[-1, :].any() or filled[:, 0].any() or filled[:, -1].any():
        return box
    gx, gy, gw, gh = cv2.boundingRect(filled)
    if gw > GROW_MAX_BBOX_X * bw or gh > GROW_MAX_BBOX_X * bh:
        return box
    cnt = max(cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    rect_fill = cv2.contourArea(cnt) / max(1.0, cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(cnt))))
    if rect_fill < GROW_MIN_RECT_FILL or filled.sum() > GROW_MAX_AREA_X * max(1, int(rb.sum())):
        return box
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = filled * 255
    return np.maximum(out, box)


def build_mask(image_rgb, items):
    import cv2
    h, w = image_rgb.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for it in items:
        m = np.maximum(m, grow_label(image_rgb, it["poly"]))
    if MASK_DILATE_PX and m.any():
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_DILATE_PX + 1,) * 2))
    return m


# ---- LaMa ------------------------------------------------------------------------------------------
def _get_lama():
    global _lama
    if _lama is None:
        import torch
        _lama = torch.jit.load(str(lama_weights_path()), map_location="cpu").eval()
    return _lama


def _lama_crop(crop_rgb, crop_mask):
    import torch
    h, w = crop_rgb.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    img = np.pad(crop_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    msk = np.pad(crop_mask, ((0, ph), (0, pw)), mode="constant")
    t_img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255).unsqueeze(0)
    t_msk = torch.from_numpy((msk > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = _get_lama()(t_img, t_msk)[0].permute(1, 2, 0).numpy()
    return np.clip(out * 255, 0, 255).astype(np.uint8)[:h, :w]


def inpaint(image_rgb, mask):
    """LaMa per connected region on a local crop. Output is copied back only where mask > 0."""
    import cv2
    result = image_rgb.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    height, width = mask.shape
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        x0, y0 = max(0, x - CONTEXT_PX), max(0, y - CONTEXT_PX)
        x1, y1 = min(width, x + w + CONTEXT_PX), min(height, y + h + CONTEXT_PX)
        crop = result[y0:y1, x0:x1]
        cm = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        longest = max(crop.shape[:2])
        if longest > LAMA_MAX_SIDE or longest < LAMA_MIN_SIDE:
            scale = (LAMA_MAX_SIDE if longest > LAMA_MAX_SIDE else LAMA_MIN_SIDE) / float(longest)
            interp_in = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            interp_out = cv2.INTER_CUBIC if scale < 1 else cv2.INTER_AREA
            small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp_in)
            small_m = cv2.resize(cm, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            out = cv2.resize(_lama_crop(small, small_m), (crop.shape[1], crop.shape[0]), interpolation=interp_out)
        else:
            out = _lama_crop(crop, cm)
        sel = cm > 0
        result[y0:y1, x0:x1][sel] = out[sel]
    return result


# ---- checks ----------------------------------------------------------------------------------------
def seam_ratio(original_rgb, result_rgb, mask):
    """How visibly the repaired area differs from its surroundings along the mask boundary.

    Ratio of the colour step across the boundary to the natural pixel-to-pixel variation just outside it.
    A clean repair is near 1; a smeared or deformed edge is much higher.
    """
    import cv2
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return 0.0
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner = (m - cv2.erode(m, k3)) > 0
    outer = (cv2.dilate(m, k3) - m) > 0
    f = result_rgb.astype(np.float32)
    # mean colour of the inner ring vs the outer ring, compared locally in blocks around the boundary
    if not inner.any():
        return 0.0
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    d_in = cv2.blur(f * inner[..., None], (9, 9))
    w_in = cv2.blur(inner.astype(np.float32), (9, 9))
    d_out = cv2.blur(f * outer[..., None], (9, 9))
    w_out = cv2.blur(outer.astype(np.float32), (9, 9))
    ok = (w_in > 1e-3) & (w_out > 1e-3) & (near > 0)
    if not ok.any():
        return 0.0
    mean_in = d_in[ok] / w_in[ok][:, None]
    mean_out = d_out[ok] / w_out[ok][:, None]
    step = np.abs(mean_in - mean_out).mean()
    band = (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0) & (near == 0)
    if not band.any():
        return 0.0
    gx = np.abs(np.diff(f, axis=1))[:, :, :].mean(axis=2)
    nat = float(gx[band[:, :-1]].mean()) if band[:, :-1].any() else 1.0
    return float(step / max(nat, 1.0))


def _tesseract_words(image_rgb):
    try:
        import pytesseract
    except Exception:  # noqa: BLE001
        return None
    from PIL import Image
    im = Image.fromarray(image_rgb)
    words = []
    for psm in (3, 11, 12):
        data = pytesseract.image_to_data(im, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
        for i, word in enumerate(data.get("text", [])):
            if not _WORD3.search(str(word or "")):
                continue
            try:
                conf, x, y, w, h = (float(data["conf"][i]), int(data["left"][i]), int(data["top"][i]),
                                    int(data["width"][i]), int(data["height"][i]))
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            if conf >= TESS_MIN_CONF and w >= 1 and h >= 1:
                words.append({"word": str(word).strip(), "conf": conf, "box": (x, y, x + w, y + h)})
    return words


def _hits_mask(box, mask, min_frac=0.3):
    x0, y0, x1, y1 = box
    sub = mask[max(0, y0):y1, max(0, x0):x1]
    return sub.size > 0 and float(np.count_nonzero(sub)) / sub.size >= min_frac


# ---- main entry --------------------------------------------------------------------------------------
def clean(image_rgb: np.ndarray):
    """Return (status, cleaned_rgb_or_None, report). Status is one of
    NO_CONFIDENT_TEXT_FOUND, TEXT_REGIONS_MASKED, NEEDS_REVIEW, REFERENCE_CLEANING_FAILED."""
    import cv2
    h, w = image_rgb.shape[:2]
    accepted, uncertain = detect(image_rgb)
    report = {
        "engine": "ppocr+lama",
        "detected_words": sorted({d["text"] for d in accepted})[:20],
        "uncertain_detections": [{"word": d["text"], "confidence": d["conf"], "orientation": d["orientation"],
                                  "box": [int(v) for v in _bbox(d["poly"])]} for d in uncertain[:20]],
        "masked_regions": 0, "rejected_regions": 0, "residual_regions": 0,
    }
    if not accepted:
        if uncertain:
            report.update(status="NEEDS_REVIEW", failure_reason="UNCERTAIN_TEXT_NOT_CLEANED")
            return "NEEDS_REVIEW", None, report
        report["status"] = "NO_CONFIDENT_TEXT_FOUND"
        return "NO_CONFIDENT_TEXT_FOUND", image_rgb.copy(), report

    mask = build_mask(image_rgb, accepted)
    coverage = float(np.count_nonzero(mask)) / float(h * w)
    report["candidate_mask_coverage_ratio"] = coverage
    if coverage > MAX_MASK_COVERAGE:
        report.update(status="REFERENCE_CLEANING_FAILED", failure_reason="MASK_TOO_LARGE", rejected_regions=len(accepted))
        return "REFERENCE_CLEANING_FAILED", None, report

    cleaned = inpaint(image_rgb, mask)
    changed = np.any(cleaned != image_rgb, axis=2)
    outside_changed = int(np.count_nonzero(changed & (mask == 0)))
    report.update(candidate_changed_area_ratio=float(np.count_nonzero(changed)) / float(h * w),
                  outside_changed_pixels=outside_changed, outside_mask_similarity=1.0 if not outside_changed else 0.0,
                  masked_regions=len(accepted))
    if outside_changed:
        report.update(status="REFERENCE_CLEANING_FAILED", failure_reason="CHANGES_OUTSIDE_REPAIR_MASK")
        return "REFERENCE_CLEANING_FAILED", None, report

    # --- independent residual checks on the repaired image -----------------------------------------
    left, left_uncertain = detect(cleaned)
    tess = _tesseract_words(cleaned)
    tess_in_mask = [t for t in (tess or []) if _hits_mask(t["box"], mask)]
    residual = len(left) + len(tess_in_mask)
    report["residual_regions"] = residual
    report["residual_words"] = sorted({d["text"] for d in left} | {t["word"] for t in tess_in_mask})[:12]
    if residual:
        report.update(status="REFERENCE_CLEANING_FAILED", failure_reason="RESIDUAL_TEXT_AFTER_MASKING")
        return "REFERENCE_CLEANING_FAILED", None, report

    # Anything doubtful, before or after cleaning, blocks the automatic PASS.
    doubt = [{"word": d["text"], "confidence": d["conf"], "orientation": d["orientation"],
              "box": [int(v) for v in _bbox(d["poly"])]} for d in left_uncertain]
    tess_outside = [t for t in (tess or []) if not _hits_mask(t["box"], mask)]
    # Words Tesseract still reads outside the mask are only a concern when the original already had them
    # unmasked, i.e. the detector missed lettering. Report them; they route to review, not to PASS.
    doubt += [{"word": t["word"], "confidence": t["conf"], "orientation": 0, "box": list(t["box"]), "source": "tesseract"}
              for t in tess_outside]
    seam = seam_ratio(image_rgb, cleaned, mask)
    report["seam_ratio"] = round(seam, 3)
    report["uncertain_detections"] = (report["uncertain_detections"] + doubt)[:30]
    if seam > SEAM_REVIEW_RATIO:
        report["review_reasons"] = ["REPAIRED_EDGE_VISIBLE"]
    if report["uncertain_detections"] or seam > SEAM_REVIEW_RATIO:
        report.update(status="NEEDS_REVIEW",
                      failure_reason="UNCERTAIN_TEXT_REMAINS" if report["uncertain_detections"] else "REPAIRED_EDGE_VISIBLE")
        return "NEEDS_REVIEW", cleaned, report
    report["status"] = "TEXT_REGIONS_MASKED"
    return "TEXT_REGIONS_MASKED", cleaned, report
