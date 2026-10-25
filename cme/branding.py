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


def reference_lettering_rule():
    """Judge criterion for the image that will be sent to the generator as the product reference."""
    return ("REFERENCE IMAGE CHECK: PASS only if NO logo, brand name, wordmark, model name, product label, branding "
            "emblem, readable letters/numbers or pseudo-text is visible anywhere in this image, including on the "
            "product itself. Judge the actual pixels. A smooth or smudged patch where lettering used to be is "
            "acceptable only if no letter shapes remain recognisable.")


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


def _fill_clusters(image, clusters):
    """Fill each lettering cluster separately; clusters that do not look like lettering are left untouched.
    Returns (cleaned_image, accepted_polygons, rejected_cluster_count)."""
    cleaned, accepted, rejected = image.copy(), [], 0
    for cluster in clusters:
        mask, surface = _letter_mask(cleaned, cluster)
        if mask is None:
            rejected += 1
            continue
        cleaned = Image.composite(Image.new("RGB", cleaned.size, surface), cleaned, mask)
        accepted.extend(cluster)
    return cleaned, accepted, rejected


def _merge_polygons(found):
    """One polygon per lettering cluster: the best-supported hit plus hits at a similar angle that overlap it."""
    groups = []
    for hit in sorted(found, key=lambda h: -h["conf"]):
        for g in groups:
            if _overlap(hit["poly"], g[0]["poly"]) >= 0.3 and abs(hit["angle"] - g[0]["angle"]) <= 20:
                g.append(hit)
                break
        else:
            groups.append([hit])
    return [[h["poly"] for h in g] for g in groups]


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
            if cached.get("source_sha256") == source_sha and cached.get("prepared_sha256") == _sha(destination) \
                    and cached.get("sweep") == len(SWEEP_ANGLES) and cached.get("status") in {
                        "TEXT_REGIONS_MASKED", "NO_CONFIDENT_TEXT_FOUND", "MASK_INCOMPLETE"}:
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
            report = {"status": ("MASK_INCOMPLETE" if residual else
                                 "TEXT_REGIONS_MASKED" if masked else "NO_CONFIDENT_TEXT_FOUND"),
                      "masked_regions": masked, "rejected_regions": rejected, "residual_regions": residual,
                      "unrelated_hits": len(leftovers) - residual,
                      "detected_words": words}
        report.update({"path": str(destination), "source_sha256": source_sha, "prepared_sha256": _sha(destination),
                       "sweep": len(SWEEP_ANGLES), "cached": False})
        sidecar.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return report
    except Exception as exc:
        kind = "OCR_UNAVAILABLE" if "tesseract" in str(exc).lower() else "OCR_FAILED"
        return {"path": str(source), "status": kind, "masked_regions": 0, "error_type": type(exc).__name__}
