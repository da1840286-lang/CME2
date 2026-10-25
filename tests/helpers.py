import io
from PIL import Image
from cme.atoms import Atom


def png_bytes(color=(120, 120, 120), size=(64, 96), patch=None, blank=False):
    """Synthetic frame. By default it carries a thin high-contrast stripe band so it is not a blank frame."""
    im = Image.new("RGB", size, color)
    if not blank:
        for x in range(0, size[0], 4):
            im.paste((20, 20, 20), (x, size[1] - 6, min(x + 2, size[0]), size[1]))
            im.paste((235, 235, 235), (min(x + 2, size[0]), size[1] - 6, min(x + 4, size[0]), size[1]))
    if patch:
        box, c = patch
        im.paste(c, box)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def A(i, text, prio="MUST", ch="image", shot=1, kind="scene"):
    return Atom(i, kind, text, prio, ch, shot)


def gate_account(preserved=("hook", "problem", "mechanism", "intervention", "result"), dropped=(),
                 form_source="demonstration", form_target="demonstration") -> dict:
    """Valid Target-Fit Gate element account for offline fixtures (shape only; meaning is the model's job)."""
    target_forms = {
        "hook": "Open on the distinctive attention hook before the product action appears",
        "problem": "Show the concrete user problem before intervention",
        "mechanism": ("Show the problem, target intervention, and resulting improvement in causal order"
                      if form_target == "comparison" else
                      "Show the problem, target intervention, and resulting improvement as one visible sequence"),
        "intervention": "Show the target intervention applied directly to the problem",
        "result": "Show an observable result after the intervention",
    }
    return {
        "demo_form_source": form_source, "demo_form_target": form_target,
        "source_elements_preserved": [
            {"element": name, "source_form": f"source {name} as shown in the ad",
             "target_form": target_forms.get(name, f"show the target {name} in a specific visible action"), "evidence_t_sec": []} for name in preserved],
        "source_elements_dropped": [
            {"element": name, "why": f"the target cannot honestly show the source {name}"} for name in dropped],
    }
