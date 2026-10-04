"""What a task required from the source (E*) versus what the model offered (E^M).

    task_evidence(fb, ...) -> {evidence, needed, tier, sufficient}
        candidates retrieved around (question + reference); the elements the
        reference actually rests on are selected with what each establishes
        (needed claims — the ONLY step that reads the reference); optionally
        checked: the needed excerpts alone let the answerer solve the task.
    claim_status(needed, evidence, model, loaded_ids, llm) -> per claim
        represented | in_model | missing.  The local refinement uses only the
        guard "does M already say it?" (missing vs the rest); the finer split
        is telemetry — the method does not diagnose why a task failed.

E^M itself is what Retrieve_M loaded for the question (answering.answer.MRetriever).
"""
from __future__ import annotations

from contextlib import nullcontext

from sourcelearn.answering.evaluate import answer_free, answer_mc
from sourcelearn.core.llm import LLMClient
from sourcelearn.core.schema import SourceElement
from sourcelearn.reconstruction.blocks import entity_key, entity_key_of_file
from sourcelearn.reconstruction.defaults import NEEDED_MAX, REFERENCE_QUERY_CHARS, TASK_EVIDENCE_K
from sourcelearn.reconstruction.inspect import inspect_source, render_evidence
from sourcelearn.reconstruction.update import EXCERPT_CHARS, _content, ground_batch, nearest_units
from sourcelearn.answering.feedback import Feedback
from sourcelearn.answering.answer import compare
from sourcelearn.source_model import SourceModelUnit

CANDIDATE_UNITS = 40      # M units shown to the entailment check per task

NEEDED_SCHEMA = {"type": "object", "properties": {"needed": {"type": "array", "maxItems": NEEDED_MAX, "items": {
    "type": "object", "properties": {"element_id": {"type": "string"}, "claim": {"type": "string"}},
    "required": ["element_id", "claim"]}}}, "required": ["needed"]}
NEEDED_PROMPT = (
    "A real task about a knowledge source is given with its reference outcome and candidate source excerpts "
    "retrieved for it. Select the excerpts the reference actually rests on — the ones a reader of the source "
    "needs in order to produce it — and for each state, as ONE declarative sentence about the SOURCE (not about "
    "the task or its answer), what that excerpt establishes that the task needed: a field, an option, a rule, a "
    "condition, a step, a relation, an exact name or value. Cite excerpt ids exactly as shown. Omit excerpts that "
    "are merely related. Return an empty list when the reference does not rest on the shown excerpts.")


def answer_context(fb: Feedback, context: str, answer_llm: LLMClient, judge_llm: LLMClient) -> tuple[bool, str, dict]:
    """Answer the task from a given context and judge it (same answering
    and judging path as evaluation)."""
    info: dict = {}
    if fb.is_mc:
        info["choice"] = answer_mc(answer_llm, fb.question, fb.options, context)
        pred = f"({info['choice']}) {fb.options[info['choice']] if 0 <= info['choice'] < len(fb.options) else '?'}"
    else:
        pred = answer_free(answer_llm, fb.question, context)
    return compare(fb, pred, info, judge_llm), pred, info


def task_evidence(fb: Feedback, retriever, all_files: list[str], llm: LLMClient, k: int = TASK_EVIDENCE_K,
                  ws=None, elements: dict | None = None, answer_llm: LLMClient | None = None,
                  judge_llm: LLMClient | None = None, retrieve_lock=None) -> dict:
    """E*: {evidence: {id: element}, needed: [{element_id, claim}], tier,
    sufficient}. `sufficient` (None when not checked) = the needed excerpts
    alone let the answerer solve the task. `retrieve_lock`: retrieval keeps
    per-call state; hold it when several tasks are processed concurrently."""
    query = f"{fb.question}\n{fb.reference_text()[:REFERENCE_QUERY_CHARS]}"
    with (retrieve_lock or nullcontext()):
        ev, tier = inspect_source(query, retriever, all_files, k, hint=fb.evidence_hint, ws=ws, elements=elements)
    out = llm.complete_json(
        system=NEEDED_PROMPT,
        user=(f"## Task\n{fb.question}\n\n## Reference outcome\n{fb.reference_text()[:2500]}\n\n"
              f"## Candidate source excerpts\n{render_evidence(ev, EXCERPT_CHARS)}"),
        schema=NEEDED_SCHEMA, purpose="task_needed")
    needed = []
    for n in (out.get("needed") or [])[:NEEDED_MAX]:
        if not isinstance(n, dict):
            continue
        eid, claim = str(n.get("element_id") or "").strip().strip("[]"), str(n.get("claim") or "").strip()
        if eid in ev and claim and eid not in {x["element_id"] for x in needed}:
            needed.append({"element_id": eid, "claim": claim[:600]})
    sufficient = None
    if needed and answer_llm is not None and judge_llm is not None:
        ctx = "## Raw source excerpts\n" + render_evidence({n["element_id"]: ev[n["element_id"]] for n in needed}, EXCERPT_CHARS)
        sufficient = answer_context(fb, ctx, answer_llm, judge_llm)[0]
    return {"evidence": ev, "needed": needed, "tier": tier, "sufficient": sufficient}


def pseudo_elements(units: list[SourceModelUnit]) -> dict[str, SourceElement]:
    """M units as excerpts, so the grounder can decide whether M ENTAILS a
    claim (the pattern of reconstruct._may_supersede)."""
    return {u.unit_id: SourceElement(element_id=u.unit_id, node_id="m", kind="section", name="",
                                     file=(u.support_anchors[0].split("#")[0] if u.support_anchors else "(model)"),
                                     fingerprint="", excerpt=u.statement + (f" Conditions: {u.conditions}" if u.conditions else ""))
            for u in units}


def candidate_units(needed: list[dict], evidence: dict[str, SourceElement], model: list[SourceModelUnit],
                    cap: int = CANDIDATE_UNITS) -> list[SourceModelUnit]:
    """Units that could hold a needed claim: anchored in the claim's file or
    entity, plus the nearest by content; ordered by overlap with the claims."""
    files = {(evidence[n["element_id"]].file if n["element_id"] in evidence else n["element_id"].split("#")[0]) for n in needed}
    keys = {entity_key_of_file(f) for f in files}
    cands: dict[str, SourceModelUnit] = {}
    for u in model:
        if u.level == "scaffold" or u.role == "map":
            continue
        if any(a.split("#")[0] in files for a in u.support_anchors) or entity_key(u) in keys:
            cands[u.unit_id] = u
    text = " ".join(n["claim"] for n in needed)
    for u in nearest_units(text, model, 8):
        cands.setdefault(u.unit_id, u)
    want = _content(text)
    return sorted(cands.values(), key=lambda u: -len(want & _content(u.statement)))[:cap]


def claim_status(needed: list[dict], evidence: dict[str, SourceElement], model: list[SourceModelUnit],
                 loaded_ids: set[str], llm: LLMClient) -> list[dict]:
    """Per needed claim: {element_id, claim, status, units} — represented (a
    loaded unit entails it), in_model (some unit does, none loaded), missing."""
    if not needed:
        return []
    cands = candidate_units(needed, evidence, model)
    if not cands:
        return [dict(n, status="missing", units=[]) for n in needed]
    res = ground_batch([n["claim"] for n in needed], pseudo_elements(cands), llm)
    out = []
    for n, (ok, anchors) in zip(needed, res):
        units = sorted(anchors)
        status = "missing" if not ok else ("represented" if set(units) & loaded_ids else "in_model")
        out.append(dict(n, status=status, units=units))
    return out


def evidence_to_json(ev: dict[str, SourceElement]) -> dict:
    return {eid: {"file": e.file, "excerpt": e.excerpt} for eid, e in ev.items()}


def evidence_from_json(d: dict, elements: dict[str, SourceElement]) -> dict[str, SourceElement]:
    """Cached E* back to elements: workspace elements with the cached excerpt
    (code windows carry text the compile-time excerpt does not)."""
    out = {}
    for eid, x in d.items():
        base = elements.get(eid)
        out[eid] = (base.model_copy(update={"excerpt": x["excerpt"]}) if base is not None else
                    SourceElement(element_id=eid, node_id="cache", kind="section", name=eid, file=x["file"], fingerprint="", excerpt=x["excerpt"]))
    return out
