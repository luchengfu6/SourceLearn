"""Task-Induced Representation Calibration, applied: the whole model re-read
through the policy.

    M_final = Recalibrate(S, M; Pi)

Every entity the model holds units about is read in full (learning.inspect
.full_text_evidence), observed under the policy ONLY (observe kind
"calibrate": what the policy says deserves representation and the current
units do not make explicit — not what a general re-reading would add), and
rewritten through the self-study rewrite path (rewrite_region ->
prepare_rewrite -> apply_rewrite: grounding, preservation, structure gate).
Task-learned units (origin task_refine) are shown as context and never
rewritten, so the policy cannot compress away what real tasks just taught.
Capacity is (nearly) fixed: a per-entity growth cap and a whole-model
budget of CALIBRATE_GROWTH x |M| — the policy decides what fills it, not
how much is written. Entities without units are self-study's job
(coverage), not calibration's.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sourcelearn.core.llm import LLMClient
from sourcelearn.core.workspace import Workspace
from sourcelearn.reconstruction.blocks import auto_partition, bounded_evidence, entity_key, entity_key_of_file
from sourcelearn.reconstruction.defaults import CALIBRATE_ALLOWANCE, CALIBRATE_GROWTH, TASK_WORKERS, workers
from sourcelearn.reconstruction.inspect import full_text_evidence
from sourcelearn.reconstruction.reconstruct import apply_rewrite, observe, prepare_rewrite, rewrite_region
from sourcelearn.reconstruction.update import m_tokens
from sourcelearn.source_model import SourceModelUnit
from sourcelearn.task_learning.policy import Policy

PROTECTED_ORIGIN = "task_refine"


def entity_jobs(model: list[SourceModelUnit], elements: dict, partition: dict | None = None) -> list[dict]:
    """One job per entity the model holds units about: {block, key, files, units}."""
    files_of_units = {a.split("#")[0] for u in model for a in u.support_anchors} & {e.file for e in elements.values()}
    part = partition or auto_partition(model) or {"blocks": [{"name": "all", "files": sorted(files_of_units)}]}
    file_block = {f: b["name"] for b in part["blocks"] for f in b["files"]}
    by_key: dict[str, dict] = {}
    for f in sorted(files_of_units):
        key = entity_key_of_file(f)
        by_key.setdefault(key, {"block": file_block.get(f, "all"), "key": key, "files": [], "units": []})["files"].append(f)
    for u in model:
        if u.level in ("scaffold", "global") or u.role == "map":
            continue
        k = entity_key(u)
        if k in by_key:
            by_key[k]["units"].append(u)
    return [j for j in by_key.values() if j["units"]]


def recalibrate(model: list[SourceModelUnit], policy: Policy | str, ws: Workspace, elements: dict, llm: LLMClient,
                brief: str = "", partition: dict | None = None, growth: float = CALIBRATE_GROWTH,
                log_path: str | Path | None = None, step_prefix: str = "c") -> dict:
    """Mutates `model`; returns the pass summary. Phase A (observe) and the
    LLM half of phase B (prepare) run in parallel over entities; the
    mutation is sequential under the whole-model budget."""
    text = policy.render() if isinstance(policy, Policy) else str(policy)
    n_items = len(policy.items) if isinstance(policy, Policy) else 0
    n_workers = max(1, workers(TASK_WORKERS))
    start = m_tokens(model)
    budget = int(start * growth) if growth else 0
    jobs = entity_jobs(model, elements, partition)
    for j in jobs:
        j["chunks"] = full_text_evidence(j["files"], ws, elements)

    def observe_job(j: dict) -> list[dict]:
        notes = []
        target = {"kind": "calibrate", "text": f"{j['key']} (files: {', '.join(j['files'])})"}
        for ev in j["chunks"]:
            try:
                notes += observe(target, j["units"], ev, llm, brief, policy=text)
            except Exception as e:  # noqa: BLE001 — one bad reply must not lose the pass
                print(f"  [observe {j['key'][-50:]}] ERROR {str(e)[:120]}", file=sys.stderr)
        return notes

    print(f"  [recalibrate] {len(jobs)} entities, {sum(len(j['chunks']) for j in jobs)} full-text chunks; observing ...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for j, notes in zip(jobs, pool.map(observe_job, jobs)):
            j["notes"] = notes
    print(f"  [recalibrate] observed: {sum(1 for j in jobs if j['notes'])} entities with notes; rewriting ...", file=sys.stderr)

    def prepare_job(item: tuple[int, dict]) -> dict | None:
        i, j = item
        if not j["notes"]:
            return None
        protected = [u for u in j["units"] if u.origin == PROTECTED_ORIGIN]
        old = [u for u in j["units"] if u.origin != PROTECTED_ORIGIN]
        block = {"name": j["key"], "description": f"entity {j['key'].split('/')[-1]} (files: {', '.join(j['files'])})",
                 "files": j["files"], "units": old}
        # what the notes rest on first (full text), then what the old units rest on, then the rest of the entity
        full = {eid: e for ch in j["chunks"] for eid, e in ch.items()}
        ev = bounded_evidence([(eid, full[eid]) for n in j["notes"] for eid in n.get("support", []) if eid in full]
                              + [(a, full.get(a) or elements[a]) for u in j["units"] for a in u.support_anchors if a in full or a in elements]
                              + list(full.items()))
        try:
            proposal = rewrite_region(block, old, j["notes"], ev, llm, brief, policy=text, context_units=protected, policy_n_items=n_items)
            tele = prepare_rewrite(model, block, proposal, ev, llm, f"{step_prefix}{i}", growth=growth, allowance=CALIBRATE_ALLOWANCE)
        except Exception as e:  # noqa: BLE001
            return {"block": j["key"], "applied": False, "reasons": [f"ERROR {str(e)[:120]}"], "old_ids": [u.unit_id for u in old],
                    "subsumed": [], "kept": [], "new_ids": [], "_grounded": [], "_old": old, "_lost": set(), "_growth": growth,
                    "_allowance": CALIBRATE_ALLOWANCE}
        tele["proposed_units"] = [u["statement"][:200] for u in proposal["units"]]
        tele["summary"] = proposal["summary"]
        return tele

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        prepared = list(pool.map(prepare_job, enumerate(jobs)))
    fh = open(log_path, "a") if log_path else None
    rows, reasons, items = [], Counter(), Counter()
    for j, tele in zip(jobs, prepared):
        row = {"entity": j["key"], "block": j["block"], "files": j["files"], "units": len(j["units"]),
               "protected": sum(1 for u in j["units"] if u.origin == PROTECTED_ORIGIN), "chunks": len(j["chunks"]),
               "notes": [n["statement"][:200] for n in j.get("notes", [])], "applied": False, "reasons": [], "tokens_before": m_tokens(j["units"])}
        if tele is not None:
            if not tele["reasons"] and budget:      # whole-model budget: a growing rewrite is refused once |M| would exceed it
                sub = [u for u in tele["_old"] if u.unit_id not in tele["_lost"]]
                projected = m_tokens(model) - m_tokens(sub) + m_tokens(tele["_grounded"])
                if projected > budget and projected > m_tokens(model):
                    tele["reasons"].append("OVER_BUDGET_MODEL")
            tele = apply_rewrite(model, tele)
            row.update({k: tele.get(k) for k in ("applied", "reasons", "subsumed", "kept", "new_ids", "new_units_text", "proposed_units",
                                                  "summary", "ungrounded", "before", "after")})
            for u in (tele.get("new_units_text") or {}).values():
                items.update(u.get("policy_items") or [])
        reasons.update(row["reasons"] if not row["applied"] else ["APPLIED"])
        if tele is not None:
            print(f"  [recalibrate {j['key'][-50:]}] {'APPLIED' if row['applied'] else 'no-op'} {row['reasons']} "
                  f"old {len(tele.get('old_ids') or [])} -> subsumed {len(row.get('subsumed') or [])}, kept {len(row.get('kept') or [])}, new {len(row.get('new_ids') or [])}", file=sys.stderr)
        row["tokens_after"] = m_tokens([u for u in model if entity_key(u) == j["key"] and u.level not in ("scaffold", "global") and u.role != "map"])
        rows.append(row)
        if fh:
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
    if fh:
        fh.close()
    return {"entities": len(jobs), "with_notes": sum(1 for j in jobs if j.get("notes")), "applied": sum(1 for r in rows if r["applied"]),
            "outcomes": dict(reasons), "tokens_before": start, "tokens_after": m_tokens(model), "budget": budget,
            "subsumed": sum(len(r.get("subsumed") or []) for r in rows), "new": sum(len(r.get("new_ids") or []) for r in rows),
            "policy_items_cited": dict(items)}
