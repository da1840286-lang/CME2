"""Budget is code's job: drop MAY, then SHOULD; if MUST does not fit, stop with a list."""
from __future__ import annotations
from dataclasses import dataclass
from .branding import branding_prompt_instruction

IMAGE_LIMIT = 3600
MOTION_LIMIT = 1700


class BudgetError(Exception):
    def __init__(self, must_ids, chars, limit):
        self.must_ids, self.chars, self.limit = must_ids, chars, limit
        super().__init__(
            f"MUST atoms need {chars} chars > limit {limit}. Split or merge these atoms "
            f"(longest first): {', '.join(must_ids)}")


@dataclass(frozen=True)
class Compiled:
    text: str
    included: tuple
    dropped: tuple  # (atom id, priority)
    chars: int
    limit: int

    def lost_line(self) -> str:
        if not self.dropped:
            return "потеряно: нет"
        return "потеряно: " + ", ".join(f"{i} ({p})" for i, p in self.dropped)


def _fit(pool, limit, reserve, render):
    keep, dropped = list(pool), []
    for prio in ("MAY", "SHOULD"):
        while len(render(keep)) + reserve > limit:
            cands = [a for a in reversed(keep) if a.priority == prio]
            if not cands:
                break
            keep.remove(cands[0])
            dropped.append((cands[0].id, prio))
    text = render(keep)
    if len(text) + reserve > limit:
        must = sorted((a for a in keep if a.priority == "MUST"), key=lambda a: -len(a.text))
        raise BudgetError([a.id for a in must], len(text) + reserve, limit)
    return Compiled(text, tuple(a.id for a in keep), tuple(dropped), len(text), limit)


FRAME_FORM_INSTRUCTION = (
    "FRAME FORM: one single continuous photograph from one camera view; never a collage, grid, multi-panel layout, "
    "split-screen, dividers or borders. Any before/after contrast lies on one continuous surface inside that single photo.")


def compile_image_prompt(atoms, shot: int, limit: int = IMAGE_LIMIT, reserve: int = 0,
                        branding_policy=None, approved_brand_marks=(), retry_feedback=()) -> Compiled:
    pool = [a for a in atoms if a.shot == shot and a.channel == "image"]
    instruction = branding_prompt_instruction(branding_policy, approved_brand_marks) if branding_policy is not None else ""
    if instruction:
        instruction = instruction + " " + FRAME_FORM_INSTRUCTION
    corrections = [str(item).strip() for item in retry_feedback if str(item).strip()]
    if corrections:
        retry_instruction = (
            "QA RETRY CORRECTIONS — mandatory corrections from the previous rejected frame. "
            "Fix every listed defect while preserving all original MUST elements and system-owned policies; "
            "do not weaken, omit, or reinterpret any contract requirement. "
            + " ".join(f"[{index}] {item}" for index, item in enumerate(corrections, 1))
        )
        instruction = (instruction + "\n" if instruction else "") + retry_instruction
    def render(selected):
        rows = ([instruction] if instruction else []) + [a.text for a in selected]
        return "\n".join(rows)
    return _fit(pool, limit, reserve, render)


def compile_motion_prompt(atoms, limit: int = MOTION_LIMIT, reserve: int = 0) -> Compiled:
    pool = [a for a in atoms if a.channel == "motion"]
    return _fit(pool, limit, reserve, lambda sel: "\n".join(f"Shot {a.shot}: {a.text}" for a in sel))
