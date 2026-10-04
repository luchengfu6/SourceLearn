"""Task-feedback packet F = (q, y, E_hint) and the questions.json adapter.

Every field except the question may be empty. The adapter only reshapes the
annotations; it never changes the learning algorithm.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Feedback:
    qid: str
    question: str
    answer: str | None = None            # y: reference outcome (free text, or the correct option's text)
    options: list[str] | None = None     # multiple choice
    answer_index: int | None = None
    evidence_hint: list[str] = field(default_factory=list)   # E: files / doc ids / anchors (optional)

    @property
    def is_mc(self) -> bool:
        return bool(self.options) and self.answer_index is not None

    def reference_text(self) -> str:
        """The reference outcome as the evidence step reads it."""
        if self.is_mc:
            return f"Correct option: ({self.answer_index}) {self.options[self.answer_index]}"
        return self.answer or ""


_PATH_TOK = re.compile(r"[\w./-]+\.(?:py|md|rst|txt)(?::\d+(?:-\d+)?)?")


def from_question(q: dict) -> Feedback:
    """Adapter for a `questions.json` row: qid, question, answer
    [, options (answer = the correct option's index)] [, gold_docs]."""
    fb = Feedback(qid=str(q.get("qid") or q.get("id")), question=str(q["question"]))
    if "options" in q:
        fb.options = [str(o) for o in q["options"]]
        fb.answer_index = int(q["answer"])
        fb.answer = fb.options[fb.answer_index] if 0 <= fb.answer_index < len(fb.options) else None
    else:
        fb.answer = str(q.get("answer") or q.get("gold") or "") or None
    hints = list(q.get("gold_docs") or q.get("required_documents") or q.get("anchors") or [])
    if not hints and fb.answer and not fb.is_mc:
        # references that mention files/lines: a hint to where to look, not evidence
        hints = sorted({m.split(":")[0] for m in _PATH_TOK.findall(fb.answer)})[:8]
    fb.evidence_hint = [str(h) for h in hints]
    return fb


def load_feedback(path: str | Path) -> list[Feedback]:
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else (data.get("questions") or next(iter(data.values())))
    return [from_question(r) for r in rows]
