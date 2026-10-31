"""Fake judge responses for transport-level tests of observe -> verify and legacy mode."""
import json


def judge_reply_body(payload, ok=True):
    messages = payload["messages"]
    system, content = str(messages[0]["content"]), messages[1]["content"]
    if "careful visual inspector" in system:
        return {"observations": [
            {"id": "o1", "kind": "object", "text": "a small product stands in the centre of the frame"},
            {"id": "o2", "kind": "region", "text": "scattered crumbs cover the left portion of a cloth surface"},
            {"id": "o3", "kind": "region", "text": "a clear strip lies where the tool passed beside untouched crumbs on the cloth"},
            {"id": "o4", "kind": "relation", "text": "the tool tip touches the cloth at the border of the clear strip and crumbs"}]}
    if "verify requirements against a written inventory" in system:
        asked = json.loads(content)["requirements"]
        results = []
        for req in asked:
            rid = req["id"]
            if rid.startswith("mech_result_"):
                aspects = [("problem_state", "crumbs cover the left portion"), ("changed_state", "clear strip lies where the tool passed beside untouched crumbs")]
            elif rid.startswith("mech_intervention_"):
                aspects = [("problem_state", "crumbs cover the left portion"), ("contact_action", "tool tip touches the cloth at the border")]
            else:
                aspects = [("other", "small product stands in")]
            results.append({"id": rid, "conditions": [
                {"aspect": aspect, "claim": "visible in inventory", "observation_ids": ["o2" if aspect == "problem_state" else "o4" if aspect == "contact_action" else "o3" if aspect == "changed_state" else "o1"],
                 "quote": quote, "holds": bool(ok)} for aspect, quote in aspects]})
        return {"results": results}
    if isinstance(content, list):
        ids = [r["id"] for r in json.loads(content[0]["text"])]
        return {"results": [{"id": i, "pass": bool(ok), "evidence": "seen"} for i in ids]}
    return None


