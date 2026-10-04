"""Answering with a source model: Retrieve_M and one task attempt.

    Retrieve_M(q, M, B)   MRetriever.context: the whole model when it fits the budget B,
                          otherwise question-routed blocks, otherwise the nearest units
    attempt(q, M, D)      answer from the hybrid raw excerpts + Retrieve_M (the evaluation path)
    compare(pred, F)      correct? exact option, or the judge against the reference
"""
from __future__ import annotations

from sourcelearn.core.llm import LLMClient
from sourcelearn.reconstruction.blocks import assign_units_to_blocks, auto_partition
from sourcelearn.reconstruction.inspect import render_evidence
from sourcelearn.reconstruction.update import EXCERPT_CHARS, m_tokens, nearest_units
from sourcelearn.answering.evaluate import answer_free, answer_mc, judge_free
from sourcelearn.answering.feedback import Feedback
from sourcelearn.source_model import SourceModelSession, SourceModelUnit


def render_m(units: list[SourceModelUnit]) -> str:
    scaffold = [u for u in units if u.level == "scaffold"]
    reason = [u for u in units if u.level != "scaffold"]
    def lines(us):
        return "\n".join(
            f"- [{u.unit_id}] {u.statement}"
            + (f" | {u.conditions[0]}" if u.conditions else "")
            for u in us)
    out = ""
    if scaffold:
        out += "### Structural map of the source\n" + lines(scaffold) + "\n"
    return out + "### Source model (complete)\n" + lines(reason)


# ----------------------------------------------------------------- Retrieve_M
class MRetriever:
    """Retrieve_M(q, M, B): the whole model when it fits the budget; otherwise
    question-routed blocks (a partition's, or mechanical per-region blocks),
    and when the model has a single region, the lexically nearest units within B.
    Full-M is the in-budget special case, partition or not."""

    def __init__(self, sm: SourceModelSession, budget: int, partition: dict | None = None,
                 router: LLMClient | None = None):
        self.sm, self.budget, self.partition = sm, budget, partition
        if router is not None:
            sm.router = router
        sm.min_blocks = 1
        self.last = {"mode": "", "blocks": []}
        self._auto: dict | None = None
        self._auto_files: frozenset[str] = frozenset()

    def _blocks(self, model: list[SourceModelUnit]) -> list[dict]:
        part = self.partition
        if part is None:
            # mechanical blocks follow the model's anchor files: a candidate model
            # whose rewrite added units for a file outside the old blocks (a root
            # script, docs/) would otherwise be routed without those units
            files = frozenset(a.split("#")[0] for u in model for a in u.support_anchors)
            if self._auto is None or files != self._auto_files:
                self._auto, self._auto_files = auto_partition(model) or {}, files
            part = self._auto
        if not part:
            return []
        byb = assign_units_to_blocks(model, part)
        return [{**b, "units": byb[b["name"]]} for b in part["blocks"]]

    def context(self, question: str, model: list[SourceModelUnit]) -> tuple[str, list[SourceModelUnit]]:
        fresh = [u for u in model if self.sm._fresh(u)]
        if self.budget <= 0 or m_tokens(fresh) <= self.budget:
            self.last = {"mode": "full", "blocks": []}
            return render_m(fresh), fresh
        blocks = self._blocks(fresh)
        if blocks:
            self.sm.blocks = blocks
            self.sm.load_budget = self.budget
            self.sm._cards_cache = None
            units = self.sm._load_blocks(question)
            self.last = {"mode": "blocks", "blocks": list(self.sm._last_blocks)}
            return ("### Source model (selected blocks: " + ", ".join(self.sm._last_blocks) + ")\n"
                    + self.sm._unit_lines(units)), units
        # single-region model over budget: units by content overlap with the
        # question (BM25 idf degenerates on tiny corpora), filled to the budget
        scaffold = [u for u in fresh if u.level == "scaffold"]
        reason = [u for u in fresh if u.level != "scaffold"]
        ranked = nearest_units(question, reason, len(reason))
        ranked += [u for u in reason if u not in ranked]
        units, spent = [], m_tokens(scaffold)
        for u in ranked:
            cost = m_tokens([u])
            if spent + cost > self.budget:
                continue
            units.append(u); spent += cost
        self.last = {"mode": "lexical", "blocks": []}
        return ("### Source model (units selected for this question)\n"
                + self.sm._unit_lines(scaffold + units)), scaffold + units


# ------------------------------------------------------------------ attempt
def attempt(fb: Feedback, ctx: tuple[str, list[SourceModelUnit]], retriever, answer_llm: LLMClient,
            k: int, mret: MRetriever | None = None) -> tuple[str, dict]:
    """Answer one task from the hybrid raw excerpts and ctx = mret.context(q, M),
    computed ONCE per question so the training step sees exactly what the
    answer saw."""
    m_text, m_units = ctx
    els = retriever.retrieve(fb.question, k) if retriever is not None else []   # None: source model only
    context = (f"## Raw source excerpts\n{render_evidence(els_dict(els), EXCERPT_CHARS) or '(nothing retrieved)'}\n\n" if retriever is not None else "") \
        + f"## Source model\n{m_text}"
    info = {"m_mode": mret.last["mode"] if mret else "", "m_blocks": mret.last["blocks"] if mret else [],
            "m_units": len(m_units), "raw_ids": [e.element_id for e in els], "prompt_chars": len(context)}
    if fb.is_mc:
        choice = answer_mc(answer_llm, fb.question, fb.options, context)
        info["choice"] = choice
        return f"({choice}) {fb.options[choice] if 0 <= choice < len(fb.options) else '?'}", info
    return answer_free(answer_llm, fb.question, context), info


def els_dict(els) -> dict:
    return {e.element_id: e for e in els}


def compare(fb: Feedback, pred: str, info: dict, judge_llm: LLMClient) -> bool:
    """Correct? The chosen option for multiple choice, the judge against the
    reference answer otherwise (the same scoring as evaluation)."""
    if fb.is_mc:
        return info.get("choice") == fb.answer_index
    return judge_free(judge_llm, fb.question, pred, fb.answer or "")
