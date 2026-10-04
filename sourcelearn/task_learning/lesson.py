"""Representation lesson: what one task experience says about how this source
should be represented — for EVERY task, solved or not.

    A solved task shows which source distinctions were actually used and
    deserve to stay (importance signal); a failed task additionally shows
    where the current representation was not enough (refinement signal).
    One extractor serves both; the outcome is part of its input.

A lesson names a kind of information, a granularity, a distinction, a
relation or a condition — never a source-specific identifier, and never the
task's own topic restated: identifiers of the task's evidence and a high
content overlap with the question / needed claims are filtered mechanically
(one retry, then dropped).
"""
from __future__ import annotations

import re

from sourcelearn.core.llm import LLMClient
from sourcelearn.reconstruction.defaults import LESSON_MAX_OVERLAP
from sourcelearn.reconstruction.inspect import render_evidence
from sourcelearn.reconstruction.reconstruct import render_units
from sourcelearn.reconstruction.update import _content
from sourcelearn.answering.feedback import Feedback
from sourcelearn.source_model import SourceModelUnit
from sourcelearn.task_learning.policy import KINDS

LESSON_SCHEMA = {"type": "object", "properties": {
    "lesson": {"type": ["object", "null"], "properties": {
        "kind": {"type": "string", "enum": list(KINDS)}, "text": {"type": "string"}}, "required": ["kind", "text"]},
    "rationale": {"type": "string", "description": "what representational choice made the needed knowledge easy or hard to find (audit only)"}},
    "required": ["lesson"]}

LESSON_PROMPT = (
    "A real task was answered with a SOURCE MODEL (a compact, persistent representation of a knowledge source) "
    "next to raw retrieval. You are shown what the task required from the source (the excerpts the reference "
    "rests on and what each establishes), the model units the system had about it, and the outcome. State the ONE "
    "general representational preference this experience suggests for building a reusable model of this source: "
    "what kind of information, at what granularity, which distinction, relation or condition deserves explicit "
    "representation — and when. A solved task shows what was used and deserves to stay represented; a failed task "
    "additionally shows what the current representation lacked or compressed away. Kinds: preserve (a type of "
    "information to keep), granularity (level of detail), distinguish (things not to conflate), relate "
    "(cross-entity relations), condition (conditions, defaults, exceptions). The lesson must contain NO "
    "source-specific identifier (class, function, field, option, file, product or value names) and must NOT "
    "restate this task's topic: name the CLASS of source content the needed knowledge belongs to (e.g. 'the "
    "options of a configuration file', 'public fields of configurable classes', 'the steps of a lifecycle "
    "procedure', 'exceptions to a shared rule') and what about it to represent and when, so that the same "
    "preference applies to every other instance of that class elsewhere in this source. "
    "Return lesson null when the task required nothing beyond generic reading of the source.")

_TOKEN = re.compile(r"`[^`]+`|\"[^\"]+\"|'[^']+'|[\w./-]+\.(?:py|md|rst|txt|json|yaml|yml|toml|cfg|ini)|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_CAMEL = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")


def _is_identifier(tok: str) -> bool:
    t = tok.strip("`\"'")
    if len(t) <= 2:
        return False
    if "." in t and "/" not in t and any(len(part) < 2 for part in t.split(".")):
        return False        # "e.g", "i.e": abbreviations, not dotted names
    if "_" in t or "." in t or "/" in t or any(c.isdigit() for c in t) or bool(_CAMEL.search(t)):
        return True
    # a quoted / backticked span is an identifier when it is a verbatim multi-word label; a plain word the
    # source merely typesets as code (`state`, "artifact") is ordinary vocabulary a class-level lesson may use
    return tok.startswith(("`", '"', "'")) and len(t.split()) >= 2


def identifiers(texts: list[str]) -> set[str]:
    """Source-specific tokens of the texts: snake_case, CamelCase, dotted
    paths, file names, quoted / backticked spans, tokens with digits."""
    out = set()
    for t in texts:
        for tok in _TOKEN.findall(t or ""):
            if _is_identifier(tok):
                out.add(tok.strip("`\"'").lower())
    return out


def domain_vocabulary(questions: list[str], min_questions: int = 3) -> set[str]:
    """Content words that recur across the source's questions (in at least
    `min_questions` of them): the domain's own vocabulary — 'agent', 'client',
    'benefit' — which any class-level lesson about this source has to use.
    Only words outside it can show that a lesson restates ONE task."""
    df: dict[str, int] = {}
    for q in questions:
        for w in _content(q):
            df[w] = df.get(w, 0) + 1
    return {w for w, n in df.items() if n >= min_questions}


def restates_task(text: str, task_texts: list[str], max_overlap: float = LESSON_MAX_OVERLAP,
                  common: set[str] | None = None) -> list[str]:
    """Task-distinctive content words the lesson repeats (question / needed
    claims, minus the domain vocabulary `common`) when they make up more than
    `max_overlap` of the lesson's own distinctive words."""
    common = common or set()
    lt = _content(text) - common
    tt = (set().union(*(_content(t) for t in task_texts)) if task_texts else set()) - common
    shared = lt & tt
    return sorted(shared) if lt and len(shared) / len(lt) > max_overlap else []


def too_specific(text: str, idents: set[str]) -> list[str]:
    """Identifiers of the evidence that a lesson repeats (case-insensitive,
    whole tokens; also the words inside snake/dotted identifiers)."""
    hits = []
    for tok in _TOKEN.findall(text or ""):
        t = tok.strip("`\"'").lower()
        if t in idents or (_is_identifier(tok) and t in idents):
            hits.append(t)
    return sorted(set(hits))


def policy_lesson(fb: Feedback, needed: list[dict], evidence: dict, statuses: list[dict],
                  shown_units: list[SourceModelUnit], correct: bool, llm: LLMClient, common: set[str] | None = None) -> dict:
    """One lesson for one task: {qid, correct, kind, text, rationale, statuses},
    or {dropped: reason, candidates} when none survives. Reads no reference
    text: only the needed claims (already source statements), their excerpts and M."""
    if not needed:
        return {"dropped": "no_needed_evidence"}
    ev = {n["element_id"]: evidence[n["element_id"]] for n in needed if n["element_id"] in evidence}
    idents = identifiers([fb.question] + [n["claim"] for n in needed] + [e.excerpt for e in ev.values()])
    status_of = {s["element_id"]: s["status"] for s in statuses}
    req = "\n".join(f"- [{n['element_id']}] {n['claim']} (in the model: {status_of.get(n['element_id'], '?')})" for n in needed)
    user = (f"## Task\n{fb.question}\n\n## Outcome\n{'solved' if correct else 'NOT solved'}\n\n"
            f"## What the task required from the source\n{req}\n\n"
            f"## Model units the system had about it\n{render_units(shown_units)}\n\n"
            f"## The source excerpts themselves\n{render_evidence(ev, 1500)}")
    extra, tried = "", []
    for _ in range(2):
        out = llm.complete_json(system=LESSON_PROMPT + extra, user=user, schema=LESSON_SCHEMA, purpose="task_lesson")
        les = out.get("lesson")
        if not isinstance(les, dict) or not str(les.get("text") or "").strip():
            return {"dropped": "null", "candidates": tried}
        text = str(les["text"]).strip()[:500]
        hits = too_specific(text, idents)
        echo = restates_task(text, [fb.question] + [n["claim"] for n in needed], common=common)
        tried.append({"text": text, "identifiers": hits, "task_words": echo})
        if not hits and not echo:
            return {"qid": fb.qid, "correct": bool(correct), "kind": les.get("kind") if les.get("kind") in KINDS else "preserve",
                    "text": text, "rationale": str(out.get("rationale") or "")[:400],
                    "statuses": [s["status"] for s in statuses]}
        extra = ("\n\nYour previous lesson was too specific to this task: it repeated "
                 + (f"source-specific identifiers ({', '.join(hits[:6])})" if hits else "")
                 + (" and " if hits and echo else "")
                 + (f"the task's own words ({', '.join(echo[:8])})" if echo else "")
                 + ". Restate it at the level of the CLASS of source content (no identifier, not this task's topic) "
                 "so that it applies to other instances of that class in this source.")
    return {"dropped": "too_specific", "candidates": tried}
