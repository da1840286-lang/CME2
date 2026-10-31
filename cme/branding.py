"""Campaign-scoped branding rules shared by Brain, generation and QA."""
from __future__ import annotations
import re
from pathlib import Path
from PIL import Image

NO_BRANDING = "no_branding"
TARGET_BRAND_ALLOWED = "target_brand_allowed"
BRANDING_POLICIES = (NO_BRANDING, TARGET_BRAND_ALLOWED)

def _marks(values):
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(v).strip() for v in values if isinstance(v, str) and v.strip())

def _tokens(text):
    return {t.lower() for t in re.findall(r"[^\W_]+", str(text or "")) if len(t) >= 3}

def allowed_brand_tokens(policy, approved_brand_marks=(), offer_name="", product_type=""):
    """OCR allowlist for the campaign mode; strict mode never whitelists the target name."""
    if policy == NO_BRANDING:
        return ()
    if policy != TARGET_BRAND_ALLOWED:
        raise ValueError(f"unsupported branding_policy {policy!r}")
    explicit = _marks(approved_brand_marks)
    tokens = set().union(*(_tokens(m) for m in explicit)) if explicit else (_tokens(offer_name) - _tokens(product_type))
    return tuple(sorted(tokens))

def branding_prompt_instruction(policy, approved_brand_marks=()):
    """System-owned instruction included in the exact receipt-bound provider prompt."""
    if policy == NO_BRANDING:
        return (
            "SYSTEM-OWNED BRANDING POLICY (no_branding; non-negotiable): Show NO logos, brand names, "
            "wordmarks, model names, labels, branding emblems, readable lettering or pseudo-text anywhere, "
            "including marks printed, painted, embossed or engraved on the product itself. This overrides "
            "every source/target reference and conflicting atom. Use product references only for silhouette, "
            "proportions, colors, materials, controls and non-text construction. Render marked surfaces as "
            "clean, unmarked material; do not reproduce or replace markings with symbols."
        )
    if policy == TARGET_BRAND_ALLOWED:
        marks = _marks(approved_brand_marks)
        allow = ("Explicitly approved target-brand marks: " + ", ".join(marks) + ". ") if marks else (
            "Target-product branding may appear only if already visible on the original target reference and "
            "consistent with the OfferCard product name/model. "
        )
        return ("SYSTEM-OWNED BRANDING POLICY (target_brand_allowed; non-negotiable): " + allow +
                "Never invent text, pseudo-lettering, logos, brand names or labels beyond the explicitly approved target marks. "
                "Never copy source-ad, competitor or unrelated branding. No captions, subtitles, UI, watermarks or promotional text.")
    raise ValueError(f"unsupported branding_policy {policy!r}")

def brain_branding_rule(policy, approved_brand_marks=()):
    marks = _marks(approved_brand_marks)
    if policy == NO_BRANDING:
        return (
            "CAMPAIGN BRANDING POLICY: no_branding. Never invent text, pseudo-lettering, logos, brand names or labels. "
            "Never reproduce those marks, even when present on supplied target-product photos. Preserve the target "
            "product's non-text shape, proportions, colors, materials and construction; require clean, unmarked "
            "product surfaces. This overrides reference-photo markings."
        )
    if policy == TARGET_BRAND_ALLOWED:
        allow = ("Only these explicitly approved target-brand marks may be shown: " + ", ".join(marks) + ". ") if marks else (
            "Only target-product markings already visible in the target reference and consistent with the "
            "OfferCard name/model may be shown. "
        )
        return ("CAMPAIGN BRANDING POLICY: target_brand_allowed. " + allow +
                "Never invent text, pseudo-lettering, logos, brand names or labels beyond the approved target marks. "
                "Never copy source-ad/competitor branding. Do not add text overlays, captions, subtitles, UI, "
                "watermarks or promotional wording.")
    raise ValueError(f"unsupported branding_policy {policy!r}")

def qa_branding_rule(policy, approved_brand_marks=()):
    marks = _marks(approved_brand_marks)
    if policy == NO_BRANDING:
        return (
            "STRICT NO-BRANDING CHECK: FAIL if any logo, brand name, wordmark, model name, product label, "
            "branding emblem, readable letters/numbers or pseudo-text is visible anywhere, including lettering "
            "physically printed, painted, embossed or engraved on the product. Product-native markings are NOT "
            "exempt. The original reference may contain branding; its presence is not permission. Judge actual "
            "pixels. Product identity is judged from non-text shape, proportions, colors, materials, controls and construction."
        )
    if policy == TARGET_BRAND_ALLOWED:
        allow = ("The only approved target-brand text/marks are: " + ", ".join(marks) + ". ") if marks else (
            "Only branding already visible on the original target-product reference and consistent with the "
            "OfferCard name/model may be retained. "
        )
        return ("TARGET-BRAND-ONLY CHECK: No added captions, subtitles, UI, watermarks or promotional text. " +
                allow + "Target markings must appear on the actual product and match its reference. FAIL on "
                "source-ad, competitor, unrelated or invented branding, and unapproved readable text.")
    raise ValueError(f"unsupported branding_policy {policy!r}")

# ---- one resolver for the brand marks every stage must agree on -------------------------------------
def default_target_marks(offer_name="", product_type=""):
    """Brand/model words of the OfferCard name: the name without the product-type description."""
    type_words = _tokens(product_type)
    words = [w for w in re.findall(r"[^\W_]+", str(offer_name or ""))
             if w.lower() not in type_words]
    return (" ".join(words),) if words else ()


def resolved_brand_marks(policy, approved_brand_marks=(), offer_name="", product_type=""):
    """The marks Brain, compiler, OCR allowlist and Judge all use. No_branding has none; target_brand_allowed
    uses the approved marks or, when none are configured, the brand/model words of the OfferCard name."""
    if policy != TARGET_BRAND_ALLOWED:
        return ()
    explicit = _marks(approved_brand_marks)
    return explicit if explicit else default_target_marks(offer_name, product_type)


def resolve_branding_contract(policy, approved_brand_marks=(), offer_name="", product_type=""):
    """Return the single normalized branding contract consumed by every pipeline stage."""
    normalized_policy = str(policy or "").strip().lower()
    if normalized_policy not in BRANDING_POLICIES:
        raise ValueError(f"unsupported branding_policy {policy!r}")
    marks = resolved_brand_marks(normalized_policy, approved_brand_marks, offer_name, product_type)
    return {
        "branding_policy": normalized_policy,
        "approved_brand_marks": list(marks),
    }


def reference_lettering_rule():
    """Visual acceptance rule for the prepared image-provider reference."""
    return (
        "REFERENCE IMAGE CHECK: PASS only if no logo, brand name, wordmark, model name, product label, branding "
        "emblem, readable letters/numbers or pseudo-text remains anywhere, including on the product itself. "
        "Also FAIL if text removal leaves a flat-colour rectangle, polygon, hard-edged fill, blur smear, halo, "
        "broken material texture, interrupted lighting gradient, lost highlight, or visible discontinuity. "
        "The repaired region must follow the surrounding surface's shape, texture, shading and perspective. "
        "Preserve adjacent real product details such as buttons, controls, seams, ports and material boundaries. "
        "Judge the actual pixels, not the status label or OCR result."
    )


# ---- text localisation ------------------------------------------------------------------------------
SWEEP_ANGLES = tuple(range(-80, 81, 5))
SWEEP_MAX_SIDE = 1400
SOLO_CONF, AGREE_CONF = 60, 40
_WORD3 = re.compile(r"[A-Za-z0-9]{3,}")


def _to_original(point, angle, rotated_size, original_size):
    """Map a point of PIL's rotate(angle, expand=True) result back to the original image."""
    import math
    rad = math.radians(angle)
    dx, dy = point[0] - rotated_size[0] / 2.0, point[1] - rotated_size[1] / 2.0
    return (dx * math.cos(rad) - dy * math.sin(rad) + original_size[0] / 2.0,
            dx * math.sin(rad) + dy * math.cos(rad) + original_size[1] / 2.0)


def _bbox(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(a, b):
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    w, h = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / max(1.0, min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)))


def _sweep_one(job):
    import pytesseract
    name, variant, angle, scale, work_size = job
    rotated = variant.rotate(angle, expand=True, fillcolor=255) if angle else variant
    data = pytesseract.image_to_data(rotated, config="--psm 11", output_type=pytesseract.Output.DICT)
    hits = []
    for i, word in enumerate(data.get("text", [])):
        word = str(word or "").strip()
        if not _WORD3.search(word):
            continue
        try:
            conf, x, y, w, h = (float(data["conf"][i]), int(data["left"][i]), int(data["top"][i]),
                                int(data["width"][i]), int(data["height"][i]))
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if conf < AGREE_CONF or w < 2 or h < 2:
            continue
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        poly = [tuple(c / scale for c in _to_original(c, angle, rotated.size, work_size)) for c in corners]
        hits.append({"poly": poly, "word": word, "conf": conf, "angle": angle, "variant": name})
    return hits


def locate_text(image, angles=SWEEP_ANGLES):
    """Return [{'poly': 4 corners in image coordinates, 'word', 'conf', 'angle'}] for lettering at any angle.

    Standard page segmentation finds horizontal text; a rotation sweep over grey and inverted copies finds stylised
    or diagonal wordmarks that it misses. A lone weak hit is ignored; weak hits that agree with each other are kept.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from PIL import ImageOps
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # one tesseract thread per process; we parallelise the sweep
    W, H = image.size
    scale = min(1.0, SWEEP_MAX_SIDE / float(max(W, H)))
    work = image.resize((max(1, int(W * scale)), max(1, int(H * scale)))) if scale < 1.0 else image
    gray = ImageOps.grayscale(work)
    variants = {"gray": gray, "invert": ImageOps.invert(gray)}
    jobs = [(name, variant, angle, scale, work.size) for name, variant in variants.items() for angle in angles]
    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as pool:
        hits = [h for part in pool.map(_sweep_one, jobs) for h in part]
    kept = []
    for i, hit in enumerate(hits):
        if hit["conf"] >= SOLO_CONF:
            kept.append(hit)
        elif any(j != i and _overlap(hit["poly"], other["poly"]) >= 0.3
                 and (other["angle"], other["variant"]) != (hit["angle"], hit["variant"])
                 for j, other in enumerate(hits)):
            kept.append(hit)
    return kept


LETTER_MIN_PIXELS, LETTER_EXPLAINED_SHARE, LETTER_LINE_TOLERANCE, LETTER_DIFF = 30, 0.6, 40, 40
LETTER_STRONG_DIFF = 90
REFERENCE_PREPARATION_VERSION = 3


def _dominant_colour(pixels):
    """Most common colour (32-level bins) of a pixel list; robust against a minority of other pixels."""
    from collections import Counter
    bins, sums = Counter(), {}
    for px in pixels:
        key = (px[0] // 32, px[1] // 32, px[2] // 32)
        bins[key] += 1
        acc = sums.setdefault(key, [0, 0, 0])
        acc[0] += px[0]; acc[1] += px[1]; acc[2] += px[2]
    key, count = bins.most_common(1)[0]
    return tuple(v / count for v in sums[key])


def _letter_mask(image, polygons):
    """Mask of lettering pixels in and around the detected regions, or (None, surface) when it is not lettering.

    surface = majority colour of the region (without plain border background); ink = the commonest colour that
    differs from it. Lettering means the differing pixels are explained by ONE ink (they lie on the surface-ink
    colour line, anti-aliased edges included). Product detail, photographs and noise are not, and are left alone.
    The mask then grows over connected ink-coloured pixels, so a word is removed whole, not only where OCR boxed it.
    """
    from PIL import ImageChops, ImageDraw, ImageFilter, ImageStat
    W, H = image.size
    probe = image.copy()
    marker = (255, 0, 255)
    for corner in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        if probe.getpixel(corner) != marker:
            ImageDraw.floodfill(probe, corner, marker, thresh=22)   # plain background connected to the border
    border_bg = Image.eval(ImageChops.difference(probe, Image.new("RGB", (W, H), marker)).convert("L"),
                           lambda v: 255 if v == 0 else 0)
    region = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(region)
    for poly in polygons:
        cx, cy = sum(p[0] for p in poly) / 4.0, sum(p[1] for p in poly) / 4.0
        draw.polygon([(cx + (x - cx) * 1.15, cy + (y - cy) * 1.15) for x, y in poly], fill=255)
    usable = ImageChops.multiply(region, ImageChops.invert(border_bg))
    box = usable.getbbox()
    if box is None:
        return None, None
    pad = 70
    work_box = (max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad))
    crop, crop_usable = image.crop(work_box), usable.crop(work_box)
    crop_bg = border_bg.crop(work_box)
    inside = [px for px, m in zip(crop.getdata(), crop_usable.getdata()) if m]
    if len(inside) < LETTER_MIN_PIXELS:
        return None, None
    surface = tuple(int(v) for v in _dominant_colour(inside))
    differing = [px for px in inside if max(abs(px[i] - surface[i]) for i in range(3)) > LETTER_DIFF]
    if len(differing) < LETTER_MIN_PIXELS:
        return None, surface
    strong = [px for px in differing if max(abs(px[i] - surface[i]) for i in range(3)) >= LETTER_STRONG_DIFF]
    ink = _dominant_colour(strong or differing)      # lettering contrasts strongly with its surface
    axis = tuple(i - s_ for i, s_ in zip(ink, surface))
    norm2 = sum(v * v for v in axis) or 1.0

    def inked(px):
        if max(abs(px[i] - surface[i]) for i in range(3)) <= LETTER_DIFF:
            return False
        rel = tuple(q - s_ for q, s_ in zip(px, surface))
        t = sum(r * a_ for r, a_ in zip(rel, axis)) / norm2
        if t < 0.15 or t > 1.25:
            return False
        return sum((r - t * a_) ** 2 for r, a_ in zip(rel, axis)) ** 0.5 <= LETTER_LINE_TOLERANCE

    def select(allowed):
        flags = [255 if (m and inked(px)) else 0 for px, m in zip(crop.getdata(), allowed.getdata())]
        out = Image.new("L", crop.size, 0)
        out.putdata(flags)
        return out

    def count(mask):
        return ImageStat.Stat(mask).sum[0] / 255.0

    letters = select(crop_usable)
    if count(letters) < LETTER_MIN_PIXELS or count(letters) / float(len(differing)) < LETTER_EXPLAINED_SHARE:
        return None, surface
    for _ in range(3):                                   # grow over connected ink so the whole word goes
        allowed = ImageChops.multiply(letters.filter(ImageFilter.MaxFilter(15)), ImageChops.invert(crop_bg))
        grown = ImageChops.lighter(letters, select(allowed))
        if count(grown) <= count(letters) * 1.03:
            break
        letters = grown
    mask = Image.new("L", (W, H), 0)
    mask.paste(letters.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1)), work_box[:2])
    return ImageChops.multiply(mask, ImageChops.invert(border_bg)), surface




def _tight_text_region(polygons, width, height):
    """Build a conservative OCR region from geometry, without brand or product-specific rules."""
    import cv2
    import numpy as np

    region = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 4 or not np.isfinite(points).all():
            continue
        center = points.mean(axis=0)
        edge_a, edge_b = points[1] - points[0], points[2] - points[1]
        norm_a, norm_b = float(np.linalg.norm(edge_a)), float(np.linalg.norm(edge_b))
        long_edge = edge_a if norm_a >= norm_b else edge_b
        edge_len = float(np.linalg.norm(long_edge))
        if edge_len < 2:
            continue
        axis_u = long_edge / edge_len
        axis_v = np.asarray((-axis_u[1], axis_u[0]), dtype=np.float32)
        projected = np.column_stack(((points - center) @ axis_u, (points - center) @ axis_v))
        u0, u1 = float(projected[:, 0].min()), float(projected[:, 0].max())
        v0, v1 = float(projected[:, 1].min()), float(projected[:, 1].max())
        axis_angle = abs(float(np.degrees(np.arctan2(axis_u[1], axis_u[0])))) % 90.0
        axis_deviation = min(axis_angle, 90.0 - axis_angle)
        if axis_deviation <= 12.0:
            trim_u = trim_v = 0.0
        else:
            trim_u = (u1 - u0) * 0.005
            trim_v = (v1 - v0) * 0.125
        if (u1 - u0) <= 2 * trim_u or (v1 - v0) <= 2 * trim_v:
            continue
        corners_uv = np.asarray(((u0 + trim_u, v0 + trim_v), (u1 - trim_u, v0 + trim_v),
                                 (u1 - trim_u, v1 - trim_v), (u0 + trim_u, v1 - trim_v)), dtype=np.float32)
        corners = center + corners_uv[:, 0, None] * axis_u + corners_uv[:, 1, None] * axis_v
        cv2.fillPoly(region, [np.round(corners).astype(np.int32)], 255)
    return region


def _directional_surface_repair(image, polygons):
    """Remove text by rebuilding its surface from neighbouring pixels; return None if unsafe.

    All location, orientation and contrast thresholds derive from the current image and OCR geometry.
    There are no brand names, product-specific colours or fixed image coordinates in this algorithm.
    """
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    region = _tight_text_region(polygons, width, height)
    if cv2.countNonZero(region) < 30:
        return None

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    sigma = max(1.5, min(width, height) / 360.0)
    local_surface = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    contrast = np.linalg.norm(lab - local_surface, axis=2)
    values = contrast[region > 0]
    if values.size < 30:
        return None
    threshold = max(10.0, float(np.percentile(values, 60)))
    ink = ((contrast >= threshold) & (region > 0)).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    ys, xs = np.where(ink > 0)
    if len(xs) < 30:
        return None

    coords = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(coords, mean=None, maxComponents=2)
    if eigenvectors is None or len(eigenvectors) == 0:
        return None
    direction = eigenvectors[0]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    covariance = np.cov(coords.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[-1] <= 1.0 or eigenvalues[-1] / max(1.0, eigenvalues[0]) < 1.15:
        return None

    center = (float(mean[0, 0]), float(mean[0, 1]))
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    inverse = cv2.invertAffineTransform(matrix)
    rotated = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT_101)
    rotated_ink = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rotated_region = cv2.warpAffine(region, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    margin = max(2, int(round(max(width, height) / 1400.0 * 6.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    expanded = cv2.dilate(rotated_ink, kernel, iterations=1)
    use_word_band = min(abs(angle), abs(abs(angle) - 90.0)) <= 15.0
    fill_shape = cv2.dilate(rotated_region, kernel, iterations=1) if use_word_band else expanded
    rows = np.flatnonzero(np.any(fill_shape > 0, axis=1))
    if rows.size < 2:
        return None

    detail_source = rotated[:, :, 0].astype(np.float32)
    detail = np.abs(detail_source - cv2.GaussianBlur(detail_source, (0, 0), sigmaX=1.5, sigmaY=1.5))
    pad = max(2, int(round(max(width, height) / 1400.0 * 3.0)))
    sample_width = max(3, pad + 1)
    side_detail = []
    for y in rows:
        columns = np.flatnonzero(fill_shape[y] > 0)
        if columns.size == 0:
            continue
        left, right = int(columns.min()), int(columns.max())
        if left > sample_width and right + sample_width < width:
            side_detail.extend(detail[y, left - sample_width:left].tolist())
            side_detail.extend(detail[y, right + 1:right + 1 + sample_width].tolist())

    if side_detail and float(np.percentile(side_detail, 75)) > 8.0:
        # For textured surfaces, use a narrow glyph mask and inpainting, never a flat rectangle.
        inpaint_mask = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        repaired = cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), inpaint_mask, 3.0, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB), mode="RGB")

    changed = rotated.copy()
    band = np.zeros((height, width), dtype=np.uint8)
    for y in rows:
        columns = np.flatnonzero(fill_shape[y] > 0)
        if columns.size == 0:
            continue
        x0 = max(sample_width, int(columns.min()) - pad)
        x1 = min(width - sample_width - 1, int(columns.max()) + pad)
        if x1 <= x0:
            continue
        left_pixels = rotated[y, x0 - sample_width:x0].astype(np.float32)
        right_pixels = rotated[y, x1 + 1:x1 + 1 + sample_width].astype(np.float32)
        left_color, right_color = np.median(left_pixels, axis=0), np.median(right_pixels, axis=0)
        weights = np.linspace(0.0, 1.0, x1 - x0 + 1, dtype=np.float32)[:, None]
        changed[y, x0:x1 + 1] = np.clip(
            left_color[None, :] * (1.0 - weights) + right_color[None, :] * weights, 0, 255
        ).astype(np.uint8)
        cv2.line(band, (x0, int(y)), (x1, int(y)), 255, 1)

    restored = cv2.warpAffine(changed, inverse, (width, height), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT_101)
    blend = cv2.warpAffine(cv2.GaussianBlur(band, (0, 0), sigmaX=1.5, sigmaY=1.5), inverse,
                           (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) / 255.0
    result = (rgb.astype(np.float32) * (1.0 - blend[:, :, None]) +
              restored.astype(np.float32) * blend[:, :, None]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def _fill_clusters(image, clusters):
    """Remove text by reconstructing surface texture and lighting, not by painting a solid colour."""
    cleaned, accepted, rejected = image.copy(), [], 0
    for cluster in clusters:
        repaired = _directional_surface_repair(cleaned, cluster)
        if repaired is None:
            rejected += 1
            continue
        cleaned = repaired
        accepted.extend(cluster)
    return cleaned, accepted, rejected


def _merge_polygons(found):
    """Merge overlapping OCR detections of the same wordmark using image-space geometry only."""
    groups = []
    for hit in sorted(found, key=lambda h: -h["conf"]):
        for group in groups:
            if any(_overlap(hit["poly"], existing["poly"]) >= 0.3 for existing in group):
                group.append(hit)
                break
        else:
            groups.append([hit])
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            box_i = _bbox([point for hit in groups[i] for point in hit["poly"]])
            poly_i = [(box_i[0], box_i[1]), (box_i[2], box_i[1]), (box_i[2], box_i[3]), (box_i[0], box_i[3])]
            for j in range(i + 1, len(groups)):
                box_j = _bbox([point for hit in groups[j] for point in hit["poly"]])
                poly_j = [(box_j[0], box_j[1]), (box_j[2], box_j[1]), (box_j[2], box_j[3]), (box_j[0], box_j[3])]
                if _overlap(poly_i, poly_j) >= 0.3:
                    groups[i].extend(groups[j])
                    del groups[j]
                    changed = True
                    break
    return [[h["poly"] for h in group] for group in groups]


def _sha(path):
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def prepare_unbranded_reference(source: str | Path, destination: str | Path):
    """Mask lettering in the image-provider input; the original stays the identity reference for QA.

    Returns a report. 'NO_CONFIDENT_TEXT_FOUND' is NOT proof that the image is clean: the caller must confirm the
    prepared image independently (visual Judge) before it is sent to a generator.
    """
    import json
    source, destination = Path(source), Path(destination)
    sidecar = destination.with_suffix(".prep.json")
    source_sha = _sha(source)
    try:
        if sidecar.is_file() and destination.is_file():
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            if (cached.get("source_sha256") == source_sha
                    and cached.get("prepared_sha256") == _sha(destination)
                    and cached.get("sweep") == len(SWEEP_ANGLES)
                    and cached.get("algorithm_version") == REFERENCE_PREPARATION_VERSION
                    and cached.get("status") in {
                        "TEXT_REGIONS_MASKED", "NO_CONFIDENT_TEXT_FOUND", "MASK_INCOMPLETE"}):
                return {**cached, "path": str(destination), "cached": True}
    except (OSError, ValueError):
        pass
    try:
        import pytesseract  # noqa: F401
        from PIL import Image as _Image  # noqa: F401
    except Exception as exc:
        return {"path": str(source), "status": "OCR_UNAVAILABLE", "masked_regions": 0, "error_type": type(exc).__name__}
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        found = list(locate_text(image))
        # Horizontal pass keeps the previous behaviour for ordinary captions.
        for psm in (3, 11, 12):
            data = pytesseract.image_to_data(image, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
            for i, word in enumerate(data.get("text", [])):
                if not re.search(r"[A-Za-z0-9]{2,}", str(word or "")):
                    continue
                try:
                    conf, x, y, w, h = (float(data["conf"][i]), int(data["left"][i]), int(data["top"][i]),
                                        int(data["width"][i]), int(data["height"][i]))
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
                if conf >= 55 and w >= 1 and h >= 1:
                    found.append({"poly": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], "word": str(word).strip(),
                                  "conf": conf, "angle": 0, "variant": f"psm{psm}"})
        destination.parent.mkdir(parents=True, exist_ok=True)
        words = sorted({str(f["word"]) for f in found})[:12]
        if not found:
            image.save(destination, format="PNG")
            report = {"status": "NO_CONFIDENT_TEXT_FOUND", "masked_regions": 0, "residual_regions": 0, "detected_words": []}
        else:
            clusters = _merge_polygons(found)
            cleaned, polys, rejected = _fill_clusters(image, clusters)
            cleaned.save(destination, format="PNG")
            seen_angles = {h["angle"] for h in found}
            recheck = sorted({0, *seen_angles, *(a + d for a in seen_angles for d in (-5, 5))})
            leftovers = locate_text(cleaned, angles=[a for a in recheck if -80 <= a <= 80])
            # Only lettering that survives where we masked counts as incomplete; stray hits elsewhere are noise
            # that the independent visual check of the prepared image still gets to judge.
            residual = sum(1 for h in leftovers if any(_overlap(h["poly"], poly) >= 0.3 for poly in polys))
            masked = len(clusters) - rejected
            report = {"status": ("MASK_INCOMPLETE" if residual or rejected else
                                 "TEXT_REGIONS_MASKED" if masked else "NO_CONFIDENT_TEXT_FOUND"),
                      "masked_regions": masked, "rejected_regions": rejected, "residual_regions": residual,
                      "unrelated_hits": len(leftovers) - residual,
                      "detected_words": words}
        report.update({"path": str(destination), "source_sha256": source_sha, "prepared_sha256": _sha(destination),
                       "sweep": len(SWEEP_ANGLES), "algorithm_version": REFERENCE_PREPARATION_VERSION, "cached": False})
        sidecar.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return report
    except Exception as exc:
        kind = "OCR_UNAVAILABLE" if "tesseract" in str(exc).lower() else "OCR_FAILED"
        return {"path": str(source), "status": kind, "masked_regions": 0, "error_type": type(exc).__name__}
