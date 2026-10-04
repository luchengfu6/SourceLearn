"""Task-Guided Source Learning, per source: one training pass -> one policy ->
M_T -> evaluation.

    E* for every question (shared cache per source and strong model)
    train pass (ordered train questions, sequential):
        E^M = Retrieve_M(q, M);  answer with E^M + raw;  E* from the cache
        every task  -> representation lesson
        failed task -> local refinement of the task's region  (M_local)
    Pi_1 = Aggregate(Pi_0, lessons)
    M_T  = Recalibrate(M_local; Pi_1)               (m_task.json)
    evaluation: M_T answers the test split EVAL_REPEATS times.

Every step is skipped when its artifact exists, so an interrupted run resumes in the same --out
(one process per --out at a time: results.jsonl is rewritten whole).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sourcelearn.answering.answer import MRetriever, attempt, compare
from sourcelearn.answering.evaluate import run as eval_run
from sourcelearn.answering.feedback import load_feedback
from sourcelearn.core.llm import LLMClient, load_project_env
from sourcelearn.core.trace import Trace
from sourcelearn.core.workspace import Workspace
from sourcelearn.draft.artifacts import CompileBundle
from sourcelearn.reconstruction.defaults import CALIBRATE_GROWTH, EVAL_REPEATS, M_BUDGET, RAW_TOP_K, TASK_EVIDENCE_K, TASK_WORKERS, workers
from sourcelearn.reconstruction.update import m_tokens
from sourcelearn.retrieval import build_retriever
from sourcelearn.source_model import SourceModelSession
from sourcelearn.task_learning.aggregate import aggregate
from sourcelearn.task_learning.evidence import claim_status, evidence_from_json, evidence_to_json, task_evidence
from sourcelearn.task_learning.lesson import domain_vocabulary, identifiers, policy_lesson
from sourcelearn.task_learning.policy import Policy
from sourcelearn.task_learning.recalibrate import recalibrate
from sourcelearn.task_learning.refine import refine_region, region_units
from sourcelearn.task_learning.splits import make_split

TRAIN_FRAC, MIN_TRAIN = 0.3, 4  # random split: share of the questions trained on; fewer training questions = nothing to learn from
JUDGE_EFFORT = "medium"         # reasoning effort of the judge (the backbone's comes from SOURCELEARN_REASONING_EFFORT)


def model_tag(model: str) -> str:
    """File-name tag of a model id (provider prefix and variant suffix dropped)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model.split("/")[-1].split(":")[-1])


def _load_rows(path: Path) -> list[dict]:
    """Rows written so far. A run killed mid-write can leave a truncated final line; it is dropped
    rather than failing the resume (the question it belonged to is simply answered again)."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break                      # a truncated tail: everything after it is incomplete too
    return rows


def _save_model(model, path: Path) -> None:
    path.write_text(json.dumps([u.model_dump() for u in model], indent=1, default=str))


def summary(out: Path, m_task_path: Path) -> str:
    """Accuracy of M_T over the eval repeats (VALID rows only), majority accuracy,
    model size, and the training / policy / recalibration telemetry."""
    by: dict = defaultdict(lambda: {"correct": [], "errors": 0})
    for r in _load_rows(out / "eval" / "results.jsonl"):
        cell = by[r["qid"]]
        if r.get("status", "VALID") != "VALID":      # N/A and failures are reported, never counted as wrong
            cell["errors"] += 1
        else:
            cell["correct"].append(bool(r.get("correct")))
    lines = []
    cells = [c for c in by.values() if c["correct"]]
    if cells:
        reps = max(len(c["correct"]) for c in cells)
        per_rep = [sum(c["correct"][i] for c in cells if len(c["correct"]) > i) / max(1, sum(1 for c in cells if len(c["correct"]) > i)) for i in range(reps)]
        maj = sum(sum(c["correct"]) / len(c["correct"]) >= 0.5 for c in cells) / len(cells)
        m = CompileBundle.load(m_task_path).model if m_task_path.exists() else None
        size = f"{m_tokens(m):,} | {len(m)}" if m is not None else "- | -"
        lines += ["| model | acc mean ± std over repeats | majority acc | M tokens | units |", "|---|---|---|---|---|",
                  f"| m_task (M_T) | {statistics.mean(per_rep):.3f} ± {statistics.pstdev(per_rep) if len(per_rep) > 1 else 0:.3f} ({reps} rep) | {maj:.3f} | {size} |"]
    elif not m_task_path.exists():
        lines.append("m_task: not built (the policy has no task-derived item), nothing evaluated")
    errs = sum(c["errors"] for c in by.values())
    if errs:
        lines.append(f"\nRows not VALID (excluded from the accuracy above, see eval/results.jsonl status): {errs}")
    split = json.loads((out / "split.json").read_text()) if (out / "split.json").exists() else {}
    if split:
        lines.append(f"\nSplit: {split.get('mode')} seed {split.get('seed')}, train {len(split.get('train', []))}, test {len(split.get('test', []))}")
    train = _load_rows(out / "train_log.jsonl")
    if train:
        wrong = [r for r in train if r.get("correct") is False]
        ref = Counter((r.get("refine") or {}).get("skipped") or ("applied" if (r.get("refine") or {}).get("applied") else "refused") for r in wrong)
        lines.append(f"Training pass: {len(train)} tasks, {len(wrong)} failed; local refinement on failures: {dict(ref)}; "
                     f"lessons {sum(1 for r in train if (r.get('lesson') or {}).get('text'))}; errors {sum(1 for r in train if r.get('error'))}")
    if (out / "policy_1.json").exists():
        lines.append(f"Policy: {len(json.loads((out / 'policy_1.json').read_text()).get('items', []))} task-derived items")
    if (out / "recal_summary.json").exists():
        s = json.loads((out / "recal_summary.json").read_text())
        lines.append(f"Recalibration: entities {s['entities']}, with notes {s['with_notes']}, applied {s['applied']}, "
                     f"tokens {s['tokens_before']:,} -> {s['tokens_after']:,} (budget {s['budget']:,})")
    return "\n".join(lines)


def run_stage(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--m-self", required=True, help="the self-studied model M_S")
    p.add_argument("--partition", default="", help="partition.json for block routing when M exceeds M_BUDGET")
    p.add_argument("--brief", default="", help="reading instruction shown to the writers")
    p.add_argument("--out", required=True)
    p.add_argument("--split-file", default="", help="split json with train/test qid lists, used verbatim (otherwise a random split by --seed)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random split")
    p.add_argument("--file-cap", type=int, default=3, help="hybrid retrieval: excerpts per file (0 = off; 3 for code repos, 0 for documents)")
    p.add_argument("--evidence-only", action="store_true", help="compute the shared E* cache of this source and exit "
                                                               "(so the per-seed runs can then start together)")
    p.add_argument("--answer-model", required=True)
    p.add_argument("--strong-model", required=True, help="necessity / lesson / writer / grounder / aggregation")
    p.add_argument("--judge-model", required=True)
    p.add_argument("--cache-dir", default="", help="directory of the shared E* cache of this source; default <workspace>/../cache")
    args = p.parse_args(argv)
    load_project_env()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ws = Path(args.workspace)
    feedbacks = load_feedback(args.questions)
    fb_by = {fb.qid: fb for fb in feedbacks}
    questions = json.loads(Path(args.questions).read_text())
    questions = questions if isinstance(questions, list) else (questions.get("questions") or next(iter(questions.values())))
    by_qid = {str(q.get("qid") or q.get("id")): q for q in questions}
    bundle = CompileBundle.load(args.m_self, args.partition or None, args.brief or None)
    m_self, partition, brief = bundle.model, bundle.partition, bundle.brief
    trace = Trace(out, "trace")
    answer = LLMClient(model=args.answer_model, trace=trace, seed=0)
    strong = LLMClient(model=args.strong_model, trace=trace, seed=0)
    judge = LLMClient(model=args.judge_model, trace=trace, seed=0, reasoning_effort=JUDGE_EFFORT)
    router = LLMClient(model=args.answer_model, trace=trace, seed=0)     # block routing: the answer model
    sm = SourceModelSession(ws, router)
    cache_dir = Path(args.cache_dir) if args.cache_dir else ws.parent / "cache"
    retriever = build_retriever("rag_hybrid", sm._retrievable, answer, file_cap=args.file_cap)
    all_files = sorted({e.file for e in sm._retrievable})
    workspace = Workspace(ws)
    n_workers = max(1, workers(TASK_WORKERS))
    (out / "config.json").write_text(json.dumps({**vars(args), "n_questions": len(feedbacks),
                                                 "m_self_units": len(m_self), "m_self_tokens": m_tokens(m_self)}, indent=1))
    t_start = time.time()

    # 1. E* for every question (the supervision of the training tasks).
    # E* depends only on (question, source, strong model), so the cache is SHARED by every run of one
    # source (each seed's task learning) instead of being recomputed per seed.
    ev_path = out / "evidence.json"
    shared_ev = cache_dir / f"evidence_{model_tag(args.strong_model)}.json"
    cache = json.loads(shared_ev.read_text()) if shared_ev.exists() else {}
    if ev_path.exists():                                  # a run dir written before the shared cache existed
        cache = {**json.loads(ev_path.read_text()), **cache}
    todo = [fb for fb in feedbacks if fb.qid not in cache]
    if todo:
        lock = threading.Lock()

        def one(fb):
            try:
                r = task_evidence(fb, retriever, all_files, strong, TASK_EVIDENCE_K, ws=workspace, elements=sm.elements,
                                  answer_llm=answer, judge_llm=judge, retrieve_lock=lock)
                return fb.qid, {"evidence": evidence_to_json(r["evidence"]), "needed": r["needed"], "tier": r["tier"], "sufficient": r["sufficient"]}
            except Exception as e:  # noqa: BLE001
                return fb.qid, {"evidence": {}, "needed": [], "tier": "error", "sufficient": None, "error": str(e)[:200]}
        print(f"[evidence] {len(todo)} questions ({shared_ev}) ...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for qid, r in pool.map(one, todo):
                cache[qid] = r
        shared_ev.parent.mkdir(parents=True, exist_ok=True)
        if shared_ev.exists():                            # another run of this source may have added questions meanwhile
            cache = {**json.loads(shared_ev.read_text()), **cache}
        tmp = shared_ev.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=1)); tmp.replace(shared_ev)      # atomic: parallel runs share this file
    ev_path.write_text(json.dumps(cache, indent=1))       # the run dir keeps its own copy

    if args.evidence_only:
        print(f"[evidence] {len(cache)} questions cached -> {shared_ev}", file=sys.stderr)
        trace.close()
        return 0

    # 2. split
    if args.split_file:                          # a given split: train/test verbatim, nothing re-derived
        fz = json.loads(Path(args.split_file).read_text())
        known = {fb.qid for fb in feedbacks}
        split = {"mode": "file", "seed": fz.get("seed", args.seed), "file": args.split_file,
                 "train": [q for q in fz["train"] if q in known], "test": [q for q in fz["test"] if q in known]}
        split["trainable"] = len(split["train"]) >= MIN_TRAIN
    else:
        split = {**make_split([fb.qid for fb in feedbacks], args.seed, TRAIN_FRAC, MIN_TRAIN), "mode": "random"}
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print(f"[split] {split['mode']}: train {len(split['train'])} test {len(split['test'])}", file=sys.stderr)
    if not split["trainable"]:
        print(f"[split] only {len(split['train'])} training questions (< MIN_TRAIN {MIN_TRAIN}): nothing to learn from; "
              "use a larger question set", file=sys.stderr)
        return 2

    # 3. one training pass: lessons from every task, local refinement on failures
    policy0 = Policy(); policy0.save(out / "policy_0.json")
    m_local_path, lessons_path, log_path = out / "m_local.json", out / "lessons.jsonl", out / "train_log.jsonl"
    if not m_local_path.exists():
        partial, idents_path = out / "m_local_partial.json", out / "identifiers_partial.json"
        done_steps = [r for r in _load_rows(log_path)] if (log_path.exists() and partial.exists()) else []
        if done_steps:                                                    # resume: the model as it was after the last logged step
            model = CompileBundle.load(partial).model
            idents = set(json.loads(idents_path.read_text())) if idents_path.exists() else set()
            print(f"[train] resuming after step {len(done_steps) - 1}", file=sys.stderr)
        else:
            model = [u.model_copy(deep=True) for u in m_self]
            idents = set()
        mret = MRetriever(sm, M_BUDGET, partition, router)
        common = domain_vocabulary([fb.question for fb in feedbacks])     # questions only: no reference text, no test answers
        with open(log_path, "a" if done_steps else "w") as fh, open(lessons_path, "a" if done_steps else "w") as lh:
            for step, qid in enumerate(split["train"]):
                if step < len(done_steps):
                    continue
                fb, E = fb_by[qid], cache[qid]
                evidence, needed = evidence_from_json(E["evidence"], sm.elements), E["needed"]
                row = {"qid": qid, "step": step, "needed": len(needed), "sufficient": E.get("sufficient"), "m_tokens_before": m_tokens(model)}
                try:
                    ctx = mret.context(fb.question, model)                       # E^M, routed once
                    pred, info = attempt(fb, ctx, retriever, answer, RAW_TOP_K, mret)
                    ok = compare(fb, pred, info, judge)
                    statuses = claim_status(needed, evidence, model, {u.unit_id for u in ctx[1]}, strong)
                    shown = region_units(model, statuses, needed) if needed else []
                    lesson = policy_lesson(fb, needed, evidence, statuses, shown, ok, strong, common=common)
                    row.update({"correct": ok, "pred": pred[:600], "attempt": info, "statuses": [s["status"] for s in statuses],
                                "lesson": lesson})
                    if lesson.get("text"):
                        lh.write(json.dumps(lesson) + "\n"); lh.flush()
                        idents |= identifiers([fb.question] + [n["claim"] for n in needed] + [evidence[n["element_id"]].excerpt for n in needed if n["element_id"] in evidence])
                    if not ok:
                        tele = refine_region(model, fb, needed, evidence, statuses, strong, brief, "")
                        row["refine"] = {k: tele.get(k) for k in ("skipped", "verdict", "applied", "new_ids", "replaced_ids", "rejected",
                                                                     "kept_old", "proposal", "shown_units", "missing", "new_units_text")}
                except Exception as e:  # noqa: BLE001 — one bad step must not lose the pass
                    row["error"] = str(e)[:300]
                row["m_tokens_after"] = m_tokens(model)
                _save_model(model, partial); idents_path.write_text(json.dumps(sorted(idents)))   # checkpoint before the log row
                fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
                rf = row.get("refine") or {}
                print(f"[train {step}] {qid} correct={row.get('correct')} refine={rf.get('skipped') or ('applied' if rf.get('applied') else ('refused' if rf else '-'))} "
                      f"lesson={'yes' if (row.get('lesson') or {}).get('text') else (row.get('lesson') or {}).get('dropped', '-')}"
                      f"{' ERROR ' + row['error'] if row.get('error') else ''}", file=sys.stderr)
        _save_model(model, m_local_path)
        (out / "identifiers.json").write_text(json.dumps(sorted(idents)))
        for pth in (partial, idents_path):
            pth.unlink(missing_ok=True)

    # 4. Pi_1
    if (out / "policy_1.json").exists():
        policy1 = Policy.load(out / "policy_1.json")
    else:
        lessons = _load_rows(lessons_path)
        idents = set(json.loads((out / "identifiers.json").read_text())) if (out / "identifiers.json").exists() else set()
        policy1, tele = aggregate(policy0, lessons, strong, idents=idents)
        policy1.save(out / "policy_1.json"); (out / "aggregate.json").write_text(json.dumps(tele, indent=1))
        print(f"[policy] {len(lessons)} lessons -> {len(policy1.items)} items", file=sys.stderr)

    # 5. M_T = Recalibrate(M_local; Pi_1)
    m_task_path = out / "m_task.json"
    if not m_task_path.exists():
        if not policy1.items:       # nothing reached the support threshold: Pi_1 = Pi_0, there is no task-derived policy to recalibrate under
            print("[recalibrate] m_task not built: the policy has no task-derived item (too few recurring lessons)", file=sys.stderr)
        else:
            model = [u.model_copy(deep=True) for u in CompileBundle.load(m_local_path).model]
            print("[recalibrate] M_T from M_local under Pi_1 ...", file=sys.stderr)
            tele = recalibrate(model, policy1, workspace, sm.elements, strong, brief, partition, CALIBRATE_GROWTH,
                               log_path=out / "recal_m_task.jsonl", step_prefix="m_task")
            _save_model(model, m_task_path)
            (out / "recal_summary.json").write_text(json.dumps(tele, indent=1))

    # 6. evaluation: M_T answers the test split EVAL_REPEATS times
    ev_dir = out / "eval"; ev_dir.mkdir(exist_ok=True)
    rows = _load_rows(ev_dir / "results.jsonl")
    # question-level: a killed run resumes mid-way. A row counts as done only when it records a
    # decision a rerun cannot change -- a scored answer (VALID) or a question that does not apply (N/A).
    # BUILD_FAILURE / RUNTIME_FAILURE / EVAL_FAILURE are transient (full disk, a 403 from the
    # gateway) and must be retried: skipping them made a relaunch exit 0 while leaving the run as
    # broken as before.
    KEEP = ("VALID", "N/A")
    have = {(r.get("repeat", 0), r["qid"]) for r in rows if r.get("status", "VALID") in KEEP}
    EVAL_BATCH = 8                                                          # rows are written after every batch
    test_qs = [by_qid[q] for q in split["test"]]
    if m_task_path.exists():
        m_units = CompileBundle.load(m_task_path).model
        for rep in range(EVAL_REPEATS):
            todo = [q for q in test_qs if (rep, str(q.get("qid") or q.get("id"))) not in have]
            if not todo:
                continue
            answer_r = LLMClient(model=args.answer_model, trace=trace, seed=rep)
            for i in range(0, len(todo), EVAL_BATCH):
                batch = todo[i:i + EVAL_BATCH]
                r = eval_run(ws, batch, answer_r, router, judge, k=RAW_TOP_K,
                             m_units=m_units, m_budget=M_BUDGET, m_partition=partition, file_cap=args.file_cap)
                for x in r:
                    x["arm"], x["repeat"] = "m_task", rep
                rows += r
                (ev_dir / "results.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    trace.close()
    manifest = json.loads((out / "config.json").read_text())
    manifest.update({"runtime_seconds": round(time.time() - t_start, 1),
                     "usage": {n: c.usage_summary() for n, c in (("answer", answer), ("strong", strong), ("judge", judge), ("router", router))}})
    (out / "config.json").write_text(json.dumps(manifest, indent=1))
    text = summary(out, m_task_path)
    (out / "summary.md").write_text(text + "\n")
    print(text)
    return 0
