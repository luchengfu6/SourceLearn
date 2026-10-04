"""Failure-Guided Local Refinement: a task that fails exposes a source region
the current model did not support well enough.

    M_{t+1} = R(M_t, S, q_t, E*_t)

The region's units are re-read against the task's evidence and refined so
that what the task required from the source is represented explicitly,
preserving existing supported knowledge. No taxonomy of failure causes: the
only guard is mechanical — claims the model already states (claim_status
in_model / represented) are not written again. The writer sees the question
and the needed claims (source statements tied to excerpts), never the
reference outcome; everything it proposes passes the usual gates
(reconstruction.reconstruct.ground_and_apply).
"""
from __future__ import annotations

from sourcelearn.core.llm import LLMClient
from sourcelearn.reconstruction.blocks import bounded_evidence
from sourcelearn.reconstruction.reconstruct import ground_and_apply, reconstruct_region, relevant_units
from sourcelearn.answering.feedback import Feedback
from sourcelearn.source_model import SourceModelUnit

SHOWN_UNITS = 12


def region_units(model: list[SourceModelUnit], statuses: list[dict], needed: list[dict], k: int = SHOWN_UNITS) -> list[SourceModelUnit]:
    """Units the refinement may touch: the ones that already state a needed
    claim (context, kept unless superseded) plus the nearest by content."""
    must = [u for s in statuses for u in s.get("units", [])]
    return relevant_units(model, " ".join(n["claim"] for n in needed), must=must, k=k)[:k + len(must)]


def refine_region(model: list[SourceModelUnit], fb: Feedback, needed: list[dict], evidence: dict, statuses: list[dict],
                  llm: LLMClient, brief: str = "", policy: str = "") -> dict:
    """Mutates `model`. Returns the ground_and_apply telemetry plus the
    proposal; {"skipped": ...} when nothing was missing."""
    missing = [s for s in statuses if s["status"] == "missing"]
    if not needed:
        return {"skipped": "no_needed_evidence", "applied": False}
    if not missing:
        return {"skipped": "already_in_model", "applied": False, "statuses": [s["status"] for s in statuses]}
    units = region_units(model, statuses, needed)
    ordered = [(n["element_id"], evidence[n["element_id"]]) for n in needed if n["element_id"] in evidence]
    ordered += [(eid, e) for eid, e in evidence.items() if eid not in {x for x, _ in ordered}]
    ev = bounded_evidence(ordered)
    req = "\n".join(f"- [{s['element_id']}] {s['claim']}" for s in missing)
    target = {"kind": "task", "text": f"Task: {fb.question}\n\nThe task required from the source (not explicit in the model):\n{req}",
              "why": "a real task depended on these claims and was not solved with the current model"}
    proposal = reconstruct_region(target, units, ev, llm, brief, policy=policy)
    tele = ground_and_apply(model, proposal, ev, llm, "task_refine", fb.qid, shown=units, verify_known=True)
    by_id = {u.unit_id: u for u in model}
    tele.update({"proposal": {"why": proposal["why"], "replace_ids": proposal["replace_ids"],
                              "new_units": [n["statement"][:200] for n in proposal["new_units"]]},
                 "shown_units": [u.unit_id for u in units], "missing": [s["element_id"] for s in missing],
                 "new_units_text": {i: by_id[i].statement for i in tele["new_ids"] if i in by_id}})
    return tele
