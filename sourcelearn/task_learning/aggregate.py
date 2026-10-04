"""Consolidation of representation lessons into the policy.

    Pi_{t+1} = Aggregate(Pi_t, {l_1 .. l_n})

Single experiences refine knowledge (refine.refine_region); only recurring
ones change how knowledge is represented: a preference enters the policy
when it consolidates lessons of at least LESSON_MIN_SUPPORT distinct tasks.
One consolidation call per epoch; the policy stays small (POLICY_MAX_ITEMS).
"""
from __future__ import annotations

from sourcelearn.core.llm import LLMClient
from sourcelearn.core.schema import coerce_str_list
from sourcelearn.reconstruction.defaults import LESSON_MIN_SUPPORT, POLICY_MAX_ITEMS
from sourcelearn.task_learning.lesson import too_specific
from sourcelearn.task_learning.policy import KINDS, Policy

AGG_SCHEMA = {"type": "object", "properties": {"preferences": {"type": "array", "items": {"type": "object", "properties": {
    "text": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)},
    "from_lessons": {"type": "array", "items": {"type": "integer"}, "description": "numbers of the lessons this preference consolidates"},
    "keeps_item": {"type": ["integer", "null"], "description": "number of the existing policy item this preference restates or refines, if any"}},
    "required": ["text", "kind", "from_lessons"]}}}, "required": ["preferences"]}

AGG_PROMPT = (
    "You maintain the REPRESENTATION POLICY of a source model: a short numbered list of preferences saying what "
    "kinds of information, at what granularity, and which distinctions, relations and conditions deserve explicit "
    "representation when this knowledge source is modelled for use. You are given the current task-derived items "
    "(if any) and new lessons, one per task, each marked solved or not. Consolidate the lessons into general "
    "preferences: merge lessons that express the same preference, keep each preference actionable for a writer "
    "(what to preserve / distinguish / relate, at what level of detail, under which condition), and cite for each "
    "the lesson numbers it consolidates. A preference may restate or refine an existing item (give keeps_item). "
    "No source-specific identifiers, and no single feature of the source: when several lessons concern one "
    "feature, state the preference for the CLASS of content that feature belongs to (e.g. 'the syntax rules of "
    "any addressing or naming scheme', 'the options of any configuration surface'), so that it applies to other "
    "instances in this source. At most {n} preferences, the most broadly supported first. Do not invent "
    "preferences no lesson supports.")


def aggregate(policy: Policy, lessons: list[dict], llm: LLMClient, min_support: int = LESSON_MIN_SUPPORT,
              max_items: int = POLICY_MAX_ITEMS, idents: set[str] | None = None) -> tuple[Policy, dict]:
    """Returns (Pi_{t+1}, telemetry). Existing items persist; a new
    preference is admitted when its consolidated lessons come from at least
    `min_support` distinct tasks and it repeats no evidence identifier."""
    tele = {"lessons": len(lessons), "proposed": 0, "admitted": [], "dropped_support": [], "dropped_specific": []}
    new = Policy(generic=policy.generic, items=[dict(it) for it in policy.items], round=policy.round + 1)
    if not lessons:
        return new, tele
    items_txt = "\n".join(f"({i + 1}) [{it['kind']}, support {len(it['support'])}] {it['text']}" for i, it in enumerate(policy.items)) or "(none)"
    les_txt = "\n".join(f"({i + 1}) [{'solved' if l['correct'] else 'not solved'}; {l['kind']}] {l['text']}" for i, l in enumerate(lessons))
    out = llm.complete_json(system=AGG_PROMPT.format(n=max_items),
                            user=f"## Current task-derived items\n{items_txt}\n\n## New lessons\n{les_txt}",
                            schema=AGG_SCHEMA, purpose="policy_aggregate")
    for p in out.get("preferences") or []:
        if not isinstance(p, dict) or not str(p.get("text") or "").strip():
            continue
        tele["proposed"] += 1
        text = str(p["text"]).strip()[:500]
        cited = [int(i) for i in coerce_str_list(p.get("from_lessons")) if str(i).isdigit() and 1 <= int(i) <= len(lessons)]
        support = {lessons[i - 1]["qid"] for i in cited}
        keeps = p.get("keeps_item")
        kept = new.items[int(keeps) - 1] if isinstance(keeps, int) and 1 <= keeps <= len(policy.items) else None
        if kept is not None:
            support |= set(kept["support"])
        if idents and too_specific(text, idents):
            tele["dropped_specific"].append(text); continue
        if len(support) < min_support:
            tele["dropped_support"].append({"text": text, "support": sorted(support)}); continue
        item = {"text": text, "kind": p.get("kind") if p.get("kind") in KINDS else "preserve", "support": sorted(support),
                "round": new.round, "from_lessons": [lessons[i - 1]["qid"] for i in cited]}
        if kept is not None:
            kept.update(item)
        else:
            new.items.append(item)
        tele["admitted"].append(item)
    new.items.sort(key=lambda it: -len(it["support"]))
    new.items = new.items[:max_items]
    tele["items"] = len(new.items)
    return new, tele
