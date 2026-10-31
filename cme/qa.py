"""Tiered QA. Code checks first.

PASS needs (a) a code check that looked at the pixels (OCR, blank-frame, anchor similarity), and (b) every MUST
atom verified. For image atoms (b) can only come from the Judge, so in practice the Judge decides MUST atoms and
code guards the frame itself (decodes, not blank, no stray text). The Judge alone cannot produce PASS.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Sequence
from PIL import Image, ImageChops, ImageStat
from .atoms import Atom, norm
from .branding import NO_BRANDING, TARGET_BRAND_ALLOWED, qa_branding_rule


PRODUCT_IDENTITY_CHECK_ID = "cme_product_identity_ref"
SYSTEM_NO_TEXT_ID = "system_no_text_policy"
SYSTEM_NO_TEXT_RULE = qa_branding_rule(TARGET_BRAND_ALLOWED)
MAX_PRODUCT_REFERENCES_FOR_JUDGE = 3


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str  # PASS / FAIL / NOT_RUN
    required: bool = True
    source: str = "code"  # code / judge
    detail: str = ""
    pixels: bool = False  # True only when the check actually looked at the frame content


def file_check(check_id, path, expected_size=None, required=True) -> CheckResult:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return CheckResult(check_id, "FAIL", required, "code", "file missing or empty")
    try:
        with Image.open(p) as im:
            size = im.size
    except Exception as e:  # unreadable image is a real failure, reported with the reason
        return CheckResult(check_id, "FAIL", required, "code", f"unreadable image: {e}")
    if expected_size and tuple(size) != tuple(expected_size):
        return CheckResult(check_id, "FAIL", required, "code", f"size {size} != {expected_size}")
    return CheckResult(check_id, "PASS", required, "code", f"size {size}")


def frame_content_check(check_id, path, min_std=4.0, min_range=32, required=True) -> CheckResult:
    """Code check on pixels: a blank or near-uniform frame is a failed generation, whatever the Judge says.

    The thresholds are stubs; calibrate them on real frames.
    """
    try:
        with Image.open(path) as im:
            gray = im.convert("L")
            std = ImageStat.Stat(gray).stddev[0]
            lo, hi = gray.getextrema()
    except Exception as e:
        return CheckResult(check_id, "FAIL", required, "code", f"cannot read frame content: {e}", True)
    ok = std >= min_std and (hi - lo) >= min_range
    return CheckResult(check_id, "PASS" if ok else "FAIL", required, "code",
                       f"contrast std {std:.1f} range {hi - lo}" + ("" if ok else " (blank or near-uniform frame)"), True)


def validate_product_references(paths: Sequence[str | Path]) -> list[str]:
    """Return blocking issues for missing, unreadable, or obvious flat-color placeholder references.

    This intentionally catches only near-uniform placeholders; it does not claim to judge whether a
    real product photo is good. The real visual identity comparison runs in the Judge after generation.
    """
    issues = []
    if not paths:
        return ["no target-product reference photo supplied"]
    for raw in paths:
        path = Path(raw)
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"target-product reference is missing or empty: {path.name}")
            continue
        try:
            with Image.open(path) as opened:
                im = opened.convert("RGB")
                if im.width < 64 or im.height < 64:
                    issues.append(f"target-product reference is too small ({im.width}x{im.height}): {path.name}")
                    continue
                im.thumbnail((64, 64))
                pixel_data = im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata()
                pixels = list(pixel_data)
        except Exception as e:
            issues.append(f"target-product reference cannot be decoded ({type(e).__name__}): {path.name}")
            continue
        unique = len(set(pixels))
        if unique <= 2:
            issues.append(f"target-product reference is a flat-color placeholder ({unique} unique colours): {path.name}; replace it with a real product photo")
    return issues


def anchor_similarity(check_id, prev_path, cur_path, region, min_similarity=0.85, required=True):
    """Compare a normalized region (x0, y0, x1, y1) of two frames; guards continuity anchors."""
    try:
        with Image.open(prev_path) as a, Image.open(cur_path) as b:
            b = b.convert("L")
            a = a.convert("L").resize(b.size)
            w, h = b.size
            box = (int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h))
            ca, cb = a.crop(box).resize((32, 32)), b.crop(box).resize((32, 32))
            diff = ImageStat.Stat(ImageChops.difference(ca, cb)).mean[0] / 255.0
    except Exception as e:
        return CheckResult(check_id, "FAIL", required, "code", f"cannot compare: {e}", True)
    sim = 1.0 - diff
    status = "PASS" if sim >= min_similarity else "FAIL"
    return CheckResult(check_id, status, required, "code", f"similarity {sim:.3f} (min {min_similarity})", True)


OCR_MIN_CONF = 60       # tesseract word confidence below this is texture noise, not text
OCR_MIN_WORD = 3        # one- and two-letter fragments are never counted as visible text
OCR_BRAND_RATIO = 0.8   # tolerance for OCR misreads of the product's own lettering


def brand_tokens(name: str, product_type: str = "") -> tuple:
    """Allow printed target name/model terms, excluding generic words already in product_type."""
    import re
    name_words = {t.lower() for t in re.findall(r"[^\W_]+", str(name or "")) if len(t) >= OCR_MIN_WORD}
    category_words = {t.lower() for t in re.findall(r"[^\W_]+", str(product_type or "")) if len(t) >= OCR_MIN_WORD}
    return tuple(sorted(name_words - category_words))


def _tesseract_confident_words(path) -> str:
    """Read confident text using several page-segmentation modes.

    Short promotional overlays can be missed by Tesseract's default layout mode. Try
    both block and sparse-text layouts so an empty first pass cannot hide visible text.
    """
    import pytesseract  # optional dependency
    words = []
    with Image.open(path) as im:
        for psm in (3, 11, 12):
            data = pytesseract.image_to_data(
                im, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
            )
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    continue
                word = str(word).strip()
                if word and c >= OCR_MIN_CONF:
                    words.append(word)
    # The same visible word is often returned by more than one layout pass.
    return " ".join(dict.fromkeys(words))


def ocr_text_check(check_id, path, text_allowed, ocr=None, max_chars=3, required=True, allowed_tokens=()):
    """No-text rule. Without an OCR engine the check is NOT_RUN, never a silent PASS.

    Counts confident words of 3+ characters which are not allowed product-brand/model terms.
    Added captions, prices and promo text still fail; texture noise and reference-native product
    lettering may pass. The detail lists the found words to make failures diagnosable.
    """
    import re
    from difflib import SequenceMatcher
    if ocr is None:
        try:
            import pytesseract  # noqa: F401
            ocr = _tesseract_confident_words
        except Exception:
            return CheckResult(check_id, "NOT_RUN", required, "code", "no OCR engine available")
    try:
        raw = ocr(path)
    except Exception as e:
        return CheckResult(check_id, "NOT_RUN", required, "code", f"OCR failed: {e}")
    if not text_allowed and not str(raw or "").strip():
        return CheckResult(
            check_id, "NOT_RUN", required, "code",
            "OCR returned no confident words; absence of text is unverified",
        )
    allowed = {str(t).lower() for t in allowed_tokens if len(str(t)) >= OCR_MIN_WORD}

    def own_lettering(tok):
        t = tok.lower()
        return any(t == a or SequenceMatcher(None, t, a).ratio() >= OCR_BRAND_RATIO for a in allowed)

    tokens = [t for t in re.findall(r"[^\W_]+", str(raw or "")) if len(t) >= OCR_MIN_WORD]
    foreign = [t for t in tokens if not own_lettering(t)]
    found = "".join(foreign)
    if text_allowed or len(found) <= max_chars:
        return CheckResult(check_id, "PASS", required, "code", f"{len(found)} text chars", True)
    sample = ", ".join(foreign[:8])
    return CheckResult(check_id, "FAIL", required, "code",
                       f"{len(found)} text chars found, none allowed: {sample}", True)

def judge_checks(path, qa_atoms, judge_fn, required=False):
    """judge_fn(path, atom_text) -> bool. Evidence only: source='judge'."""
    out = []
    for a in qa_atoms:
        try:
            ok = bool(judge_fn(path, a.text))
            out.append(CheckResult(a.id, "PASS" if ok else "FAIL", required, "judge", ""))
        except Exception as e:
            out.append(CheckResult(a.id, "NOT_RUN", required, "judge", f"judge error: {e}"))
    return out


def judge_batch_checks(path, atoms, judge_fn, required=False, reference_images=()):
    """One Judge request per frame; references trigger a separate, required product-identity verdict."""
    refs = [str(p) for p in list(reference_images or ())[:MAX_PRODUCT_REFERENCES_FOR_JUDGE]]
    try:
        answers = judge_fn(path, atoms, refs) if refs else judge_fn(path, atoms)
    except Exception as e:
        out = [CheckResult(a.id, "NOT_RUN", bool(required or a.priority == "MUST"), "judge", f"judge unavailable: {e}") for a in atoms]
        if refs:
            out.append(CheckResult(PRODUCT_IDENTITY_CHECK_ID, "NOT_RUN", True, "judge",
                                   f"product-reference comparison unavailable: {type(e).__name__}: {e}"))
        return out
    if not isinstance(answers, dict):
        out = [CheckResult(a.id, "NOT_RUN", bool(required or a.priority == "MUST"), "judge", "malformed judge response") for a in atoms]
        if refs:
            out.append(CheckResult(PRODUCT_IDENTITY_CHECK_ID, "NOT_RUN", True, "judge", "malformed judge response"))
        return out
    out = []
    for a in atoms:
        r = answers.get(a.id)
        if not isinstance(r, dict):
            out.append(CheckResult(a.id, "NOT_RUN", bool(required or a.priority == "MUST"), "judge", "no valid answer for this requirement"))
            continue
        evidence = r.get("evidence")
        if type(r.get("pass")) is not bool or not isinstance(evidence, str) or not evidence.strip():
            out.append(CheckResult(a.id, "NOT_RUN", bool(required or a.priority == "MUST"), "judge", "malformed answer or missing evidence"))
            continue
        out.append(CheckResult(a.id, "PASS" if r["pass"] else "FAIL", bool(required or a.priority == "MUST"), "judge", evidence.strip()[:120]))
    if refs:
        r = answers.get(PRODUCT_IDENTITY_CHECK_ID)
        if not isinstance(r, dict):
            out.append(CheckResult(PRODUCT_IDENTITY_CHECK_ID, "NOT_RUN", True, "judge",
                                   "Judge did not return a product-identity result for the supplied target-photo reference"))
        else:
            evidence = r.get("evidence")
            if type(r.get("pass")) is not bool or not isinstance(evidence, str) or not evidence.strip():
                out.append(CheckResult(PRODUCT_IDENTITY_CHECK_ID, "NOT_RUN", True, "judge",
                                       "malformed product-identity answer or missing visual evidence"))
            else:
                out.append(CheckResult(PRODUCT_IDENTITY_CHECK_ID, "PASS" if r["pass"] else "FAIL", True,
                                       "judge", evidence.strip()[:120]))
    return out


REFERENCE_CHECK_ID = "reference_no_lettering"


def check_reference_unbranded(path, judge_fn, shot=1):
    """Independent pixel check of the image that will be sent to the generator as the product reference.
    An empty OCR result never confirms a clean reference; only a Judge PASS does."""
    from .branding import reference_lettering_rule
    if judge_fn is None:
        return CheckResult(REFERENCE_CHECK_ID, "NOT_RUN", True, "judge", "no Judge available to confirm the prepared reference")
    atom = Atom(REFERENCE_CHECK_ID, "qa_check", reference_lettering_rule(), "MUST", "qa", shot)
    for row in judge_batch_checks(path, [atom], judge_fn, required=True):
        if row.check_id == REFERENCE_CHECK_ID:
            return row
    return CheckResult(REFERENCE_CHECK_ID, "NOT_RUN", True, "judge", "Judge returned no verdict for the reference")


def make_qa_checks(atoms, shot, judge_fn=None, ocr=None, expected_size=None, ocr_required=True,
                   text_allowed=False, anchors=(), verified_claims=(), product_references=(), allowed_text=(),
                   transfer_decision=None, branding_policy=TARGET_BRAND_ALLOWED, approved_brand_marks=()):
    """Build QA for one shot, with independent visual checks for preserved source properties."""
    claims = [a for a in atoms if a.shot == shot and a.channel == "qa" and a.kind == "claim"]
    claim_set = {norm(c) for c in verified_claims}
    pool = [a for a in atoms if a.shot == shot and a.channel in ("image", "qa")
            and not (a.channel == "qa" and a.kind == "claim")
            and a.id != "qa_no_text" and not a.id.startswith("qa_no_text_")]
    product_refs = tuple(str(p) for p in list(product_references or ())[:MAX_PRODUCT_REFERENCES_FOR_JUDGE])

    def run(path):
        ocr_result = ocr_text_check(
            "no_text", path, text_allowed, ocr=ocr, required=ocr_required, allowed_tokens=allowed_text
        )
        # Empty OCR stays NOT_RUN. A required visual Judge is the independent fallback in strict mode.
        if (branding_policy == NO_BRANDING and judge_fn is not None
                and ocr_result.status == "NOT_RUN"
                and ocr_result.detail.startswith("OCR returned no confident words")):
            ocr_result = replace(
                ocr_result, required=False,
                detail=ocr_result.detail + "; required system_no_text_policy visual Judge is the independent fallback"
            )
        res = [file_check("file", path, expected_size),
               frame_content_check("frame_content", path),
               ocr_result]
        for cid, prev, region, minimum in anchors:
            res.append(anchor_similarity(cid, prev, path, region, minimum))
        for a in claims:
            valid = norm(a.text) in claim_set
            res.append(CheckResult(
                a.id, "PASS" if valid else "FAIL", a.priority == "MUST", "code",
                "matches verified offer claim; not inferred from pixels" if valid
                else "claim is absent from offer.verified_claims"))
        if judge_fn is not None:
            system_no_text = Atom(SYSTEM_NO_TEXT_ID, "qa_check", qa_branding_rule(branding_policy, approved_brand_marks), "MUST", "qa", shot)
            judge_atoms = [*pool, system_no_text]
            transfer_check_ids = set()
            rows_by_id = {a.id: a for a in atoms}
            preserved = (transfer_decision or {}).get("preserved", []) if isinstance(transfer_decision, dict) else []
            if isinstance(preserved, list):
                for index, item in enumerate(preserved, 1):
                    if not isinstance(item, dict):
                        continue
                    carrier = rows_by_id.get(str(item.get("atom_id") or ""))
                    prop = str(item.get("property") or "").strip()
                    if carrier is None or carrier.shot != shot or not prop:
                        continue
                    check_id = f"transfer_preserved_{index}"
                    transfer_check_ids.add(check_id)
                    judge_atoms.append(Atom(
                        check_id, "qa_check",
                        "Independent transfer-preservation visual check. Source-mechanism property that must be "
                        f"visible in this generated target-product frame: {prop}. Intended carrier atom text: "
                        f"{carrier.text}. Judge the actual pixels, not the wording. PASS only if the stated "
                        "property is visibly demonstrated; FAIL if the frame shows an unrelated demo or the "
                        "property is not visually evidenced.",
                        "MUST", "qa", shot))
            judge_rows = judge_batch_checks(path, judge_atoms, judge_fn, reference_images=product_refs)
            required_judge_ids = {SYSTEM_NO_TEXT_ID, *transfer_check_ids}
            res.extend(replace(row, required=True) if row.check_id in required_judge_ids else row for row in judge_rows)
        elif product_refs:
            res.append(CheckResult(PRODUCT_IDENTITY_CHECK_ID, "NOT_RUN", True, "judge",
                                   "no Judge adapter configured; target-product identity was not compared"))
        return res

    return run
