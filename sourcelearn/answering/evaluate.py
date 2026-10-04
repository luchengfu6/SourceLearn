"""Evaluation harness: one source D, one source model M, many questions.

The answerer sees the hybrid raw excerpts (BM25 + dense, RRF; per-file cap
`file_cap`) and Retrieve_M(q, M, budget): the whole M when it fits, routed
blocks otherwise.

Every row carries: status (VALID, BUILD_FAILURE, RUNTIME_FAILURE, EVAL_FAILURE,
N/A), error (str | None), prompt_chars, evidence_tokens (cl100k tokens of the
retrieved context), prompt_tokens_est / answer_tokens_est (cl100k tokens of
every answer-side LLM prompt / reply made for the row; the judge is not
counted), llm_calls, retrieval_calls. A row whose status is not VALID has
correct=None and is never scored: a failure is never a wrong answer.

Scoring: multiple choice (exact option) or free text judged against a
reference answer by a judge model.
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sourcelearn.core.llm import LLMClient
from sourcelearn.reconstruction.defaults import EVAL_WORKERS, workers
from sourcelearn.retrieval import PER_FILE_CAP, build_retriever
from sourcelearn.source_model import SourceModelSession

STATUSES = ("VALID", "BUILD_FAILURE", "RUNTIME_FAILURE", "EVAL_FAILURE", "N/A")

MC_SCHEMA = {"type": "object", "properties": {"choice": {"type": "integer"}, "why": {"type": "string"}},
             "required": ["choice"]}
FREE_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
JUDGE_SCHEMA = {"type": "object", "properties": {"correct": {"type": "boolean"}, "missing": {"type": "string"}},
                "required": ["correct"]}


ANSWER_MC_SYSTEM = ("Answer the multiple-choice question about the source using the material shown. "
                    "Pick exactly one option index. If the material is insufficient, pick the most likely option.")


def answer_mc(llm: LLMClient, question: str, options: list[str], context: str) -> int:
    opts = "\n".join(f"({i}) {o}" for i, o in enumerate(options))
    out = llm.complete_json(system=ANSWER_MC_SYSTEM, user=f"{context}\n\nQuestion: {question}\nOptions:\n{opts}",
                            schema=MC_SCHEMA, purpose="fsqa_mc")
    try:
        return int(out.get("choice", -1))
    except (TypeError, ValueError):
        return -1


ANSWER_FREE_SYSTEM = ("Answer the question about this code repository / knowledge source from the material shown. "
                      "Be specific: name the classes, functions, files, rules or values involved and how they relate. "
                      "Ground every claim in the material; say what it cannot establish.")


def answer_free(llm: LLMClient, question: str, context: str) -> str:
    try:
        out = llm.complete_json(system=ANSWER_FREE_SYSTEM, user=f"{context}\n\nQuestion: {question}",
                                schema=FREE_SCHEMA, purpose="fsqa_free")
        return str(out.get("answer", ""))
    except ValueError:      # long free-text replies sometimes never become valid JSON: the prose itself is the answer
        res = llm.chat([{"role": "system", "content": ANSWER_FREE_SYSTEM},
                        {"role": "user", "content": f"{context}\n\nQuestion: {question}"}], purpose="fsqa_free_plain")
        return str(res.get("content") or "")


def judge_free(judge: LLMClient, question: str, answer: str, reference: str) -> bool:
    out = judge.complete_json(
        system=("Judge whether the candidate answer is correct against the reference answer. Correct means the "
                "candidate makes the same key claims (entities, relations, behaviours) without contradicting the "
                "reference; extra correct detail is fine, missing a decisive part or asserting something the "
                "reference contradicts is not."),
        user=f"Question: {question}\n\nReference answer:\n{reference}\n\nCandidate answer:\n{answer}",
        schema=JUDGE_SCHEMA, purpose="fsqa_judge")
    return bool(out.get("correct"))


_ENC = None


def cl100k_len(text: str) -> int:
    """Token count under cl100k_base (the tokenizer every budget is stated in)."""
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text or "", disallowed_special=()))


_ROW = threading.local()      # per-worker-thread counters for the row being answered


def _count(key: str, n: int = 1) -> None:
    c = getattr(_ROW, "n", None)
    if c is not None:
        c[key] = c.get(key, 0) + n


class _Counting:
    """A transparent proxy over an LLMClient that charges every complete_json /
    chat call (count, cl100k prompt and reply tokens) to the row the calling
    thread is answering; everything else (embed, model, provider, ...) is the
    client's own."""

    def __init__(self, llm):
        object.__setattr__(self, "_llm", llm)

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def __setattr__(self, name, value):
        setattr(self._llm, name, value)

    def _charge(self, prompt: str, reply: str) -> None:
        if getattr(_ROW, "n", None) is not None:
            _count("llm"); _count("in", cl100k_len(prompt)); _count("out", cl100k_len(reply))

    def complete_json(self, system, user, *a, **kw):
        out = self._llm.complete_json(system, user, *a, **kw)
        self._charge(f"{system}\n{user}", json.dumps(out, default=str))
        return out

    def chat(self, messages, *a, **kw):
        out = self._llm.chat(messages, *a, **kw)
        self._charge("\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict)), str(out.get("content") or ""))
        return out


def unwrap_llm(llm):
    return llm._llm if isinstance(llm, _Counting) else llm


def _new_row(q: dict, arm: str) -> dict:
    return {"qid": q["qid"], "arm": arm, "question": q["question"], "status": "VALID", "error": None, "correct": None,
            "prompt_chars": 0, "evidence_tokens": 0, "prompt_tokens_est": 0, "answer_tokens_est": 0,
            "llm_calls": 0, "retrieval_calls": 0}


def _fail(row: dict, status: str, error: str) -> dict:
    """A row that is not a scored answer: no `correct` (never counted as wrong)."""
    assert status in STATUSES and status != "VALID", status
    row.update({"status": status, "error": error[:300], "correct": None})
    return row


def is_valid(row: dict) -> bool:
    return row.get("status", "VALID") == "VALID"     # rows written before the status field count as scored


def run(ws: Path, questions: list[dict], answer: LLMClient, router: LLMClient, judge: LLMClient | None,
        m_units: list, k: int = 8, m_budget: int = 24000, m_partition: dict | None = None,
        file_cap: int | None = PER_FILE_CAP, load_budget: int = 24000) -> list[dict]:
    """`m_units` + `m_budget` (+ `m_partition` for routing) feed Retrieve_M through
    MRetriever. `file_cap`: per-file cap of the hybrid retriever
    (0 = off; 3 on code repositories, 0 on document collections)."""
    from sourcelearn.answering.answer import MRetriever
    answer, router = _Counting(unwrap_llm(answer)), _Counting(unwrap_llm(router))   # per-row call / token accounting
    judge_llm = judge or unwrap_llm(answer)                                            # the judge is not charged to the row
    arm = "m_task"
    s = SourceModelSession(ws, router, raw_top_k=k, load_budget=load_budget)
    mret = MRetriever(s, m_budget, m_partition, router)
    try:
        s.retriever = build_retriever("rag_hybrid", s._retrievable, answer, file_cap=file_cap)
    except Exception as e:  # noqa: BLE001 — an index failure costs these rows, not the run
        print(f"  [{arm}] index build failed: {str(e)[:200]}", file=sys.stderr)
        return [_fail(_new_row(q, arm), "BUILD_FAILURE", f"index: {str(e)[:200]}") for q in questions]
    ctx_lock = threading.Lock()      # retrieval + routing keep per-call state on the session: build contexts one at a time

    def one(q: dict) -> dict:
        try:
            return answer_one(q)
        except Exception as e:  # noqa: BLE001 — last resort: this row, not the run
            row = _fail(_new_row(q, arm), "RUNTIME_FAILURE", f"row: {str(e)[:200]}")
            print(f"  [{arm}] {q['qid']} ERROR {row['error']}", file=sys.stderr)
            return row

    def build_context(q: dict, row: dict) -> str:
        with ctx_lock:
            _count("retrieval")
            raw_text, _raw_els = s._raw_block(q["question"])
            parts = ["## Raw source excerpts\n" + (raw_text or "(nothing retrieved)")]
            _count("retrieval")
            m_text, _mu = mret.context(q["question"], m_units)
            parts.append("## Source model\n" + m_text)
            row["m_mode"], row["m_blocks"] = mret.last["mode"], list(mret.last["blocks"])
            row["m_unit_ids"] = [u.unit_id for u in _mu]
            return "\n\n".join(parts)

    def answer_one(q: dict) -> dict:
        _ROW.n = {"llm": 0, "retrieval": 0, "in": 0, "out": 0}
        row = _new_row(q, arm)
        try:
            context = build_context(q, row)
            row["prompt_chars"] = len(context)
            row["evidence_tokens"] = cl100k_len(context)
        except Exception as e:  # noqa: BLE001 — retrieval / routing failed before the answer: this row, not the run
            return _finish(_fail(row, "RUNTIME_FAILURE", f"context: {str(e)[:200]}"))
        try:
            if "options" in q:
                row["choice"] = answer_mc(answer, q["question"], q["options"], context)
            else:
                row["answer"] = answer_free(answer, q["question"], context)
        except Exception as e:  # noqa: BLE001
            return _finish(_fail(row, "RUNTIME_FAILURE", f"answer: {str(e)[:200]}"))
        try:
            if "options" in q:
                row["correct"] = row["choice"] == int(q["answer"])
            else:
                row["correct"] = judge_free(judge_llm, q["question"], row["answer"], q["answer"])
        except Exception as e:  # noqa: BLE001 — the answer exists, the judgement does not
            return _finish(_fail(row, "EVAL_FAILURE", f"judge: {str(e)[:200]}"))
        return _finish(row)

    def _finish(row: dict) -> dict:
        n = getattr(_ROW, "n", None) or {}
        row.update({"llm_calls": n.get("llm", 0), "retrieval_calls": n.get("retrieval", 0),
                    "prompt_tokens_est": n.get("in", 0), "answer_tokens_est": n.get("out", 0)})
        _ROW.n = None
        print(f"  [{arm}] {row['qid']} " + (f"correct={row['correct']}" if is_valid(row) else f"{row['status']} {row['error']}"), file=sys.stderr)
        return row

    with ThreadPoolExecutor(max_workers=max(1, min(workers(EVAL_WORKERS), len(questions) or 1))) as pool:
        return list(pool.map(one, questions))          # map keeps question order
