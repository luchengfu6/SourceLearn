"""The source model M and the task-time view of the source.

M is the only persistent state an agent keeps about a source: a set of source-grounded knowledge
units (statement, role, scope, applicability conditions, anchors to the supporting source
elements). Past tasks influence future ones only by changing M -- there is no case memory and no
answer cache on the inference path.

- Units carry multi-anchor provenance (`support_anchors` + `support_fingerprints`): understanding
  synthesized from several source pieces is first-class. Freshness = every supporting fingerprint
  still holds against the current source.
- `origin` separates what the draft read from the source ("compiled") from what later learning
  wrote ("self_study", "task_refine", "learned"); `support_mode` separates explicit from synthesized.
- `SourceModelSession` holds the source elements, the raw-evidence retriever and the block router
  that Retrieve_M (answering.answer.MRetriever) uses when M exceeds the budget.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sourcelearn.core.elements import source_elements, tokens as _tokens
from sourcelearn.core.llm import LLMClient
from sourcelearn.core.workspace import Workspace
from sourcelearn.reconstruction.defaults import RAW_TOP_K


def _family_of(file: str) -> str:
    """Entity family of a source file: docs strip the numeric suffix
    (doc_x_001.md -> dir/doc_x); code = the file itself. Matches
    induce.sibling_key so M scopes map back onto files."""
    d = file.rsplit("/", 1)[0] if "/" in file else "."
    stem = Path(file).stem
    if file.endswith((".md", ".rst", ".txt")):
        prefix = re.sub(r"_\d+$", "", stem)
        return f"{d}/{prefix}"
    return file


class SourceModelUnit(BaseModel):
    """One piece of the source model. Trust invariants unchanged: anchors
    and fingerprints are stamped mechanically, never model-written."""

    unit_id: str
    statement: str
    kind: Literal["fact", "model"] = "fact"
    scope: str | None = None
    conditions: list[str] = Field(default_factory=list)
    support_anchors: list[str] = Field(default_factory=list)
    support_fingerprints: dict[str, str] = Field(default_factory=dict)
    origin: Literal["compiled", "learned", "self_study", "task_refine"] = "compiled"
    # compiled: the initial reading; learned: task-supervised revisions; self_study /
    # task_refine: written by reconstruction.reconstruct (lineage in derived_from)
    support_mode: Literal["explicit", "synthesized"] = "explicit"
    learned_from_task: str | None = None  # audit only, never retrieved
    # what kind of understanding this unit is
    role: str | None = None   # map|pattern|mechanism|distinction|procedure|exception|detail|attribute
    level: str | None = None  # global|regional
    attr_name: str | None = None   # attribute units: normalized name (entity x attribute table)
    attr_value: str | None = None
    derived_from: list[str] = Field(default_factory=list)  # lineage: the unit ids this unit was abstracted from or replaces
    policy_items: list[int] = Field(default_factory=list)  # tasklearn: representation-policy items a rewrite cited (audit only)


class SourceModelSession:
    """The source as the agent sees it at task time: its fingerprinted
    elements, the raw-evidence retriever (sourcelearn.retrieval), and the
    block router. `blocks` = [{name, description, families, files, units}];
    the router picks blocks for the question and whole block models are
    loaded in rank order until `load_budget`."""

    def __init__(self, workspace: str | Path, llm: LLMClient, raw_top_k: int = RAW_TOP_K,
                 retriever=None, blocks: list[dict] | None = None, load_budget: int = 12000):
        self.ws = Workspace(workspace)
        self.llm = llm
        self.raw_top_k = raw_top_k
        self.retriever = retriever
        self.blocks = blocks or []
        self.load_budget = load_budget
        self._last_blocks: list[str] = []
        self.elements = source_elements(self.ws)
        self._retrievable = [e for e in self.elements.values()
                             if e.kind in ("section", "symbol")]

    # -- freshness: EVERY supporting fingerprint must still hold ---------------
    def _fresh(self, u: SourceModelUnit) -> bool:
        return bool(u.support_anchors) and all(
            a in self.elements
            and self.elements[a].fingerprint == u.support_fingerprints.get(a)
            for a in u.support_anchors)

    # -- retrieval --------------------------------------------------------------
    def _raw_block(self, question: str) -> tuple[str, list]:
        els = self.retriever.retrieve(question, self.raw_top_k)
        def loc(e):  # index summaries carry no line span
            return (f"{e.file}:{e.start_line}-{e.end_line}"
                    if e.start_line is not None else e.file)
        return "\n\n".join(
            f"[{e.element_id}] ({loc(e)})\n{e.excerpt}" for e in els), els

    @staticmethod
    def _unit_lines(units: list[SourceModelUnit]) -> str:
        lines = []
        for u in units:
            cond = f" | conditions: {u.conditions}" if u.conditions else ""
            scope = f" | scope: {u.scope}" if u.scope else ""
            lines.append(f"- [{u.unit_id}] {u.statement}{scope}{cond}")
        return "\n".join(lines)

    # -- block routing ------------------------------------------------------------
    RICH_ROUTE_SCHEMA = {"type": "object", "properties": {
        "entities": {"type": "array", "items": {"type": "string"}},
        "aspects": {"type": "array", "items": {"type": "string"}},
        "blocks": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "required": {"type": "boolean"}, "why": {"type": "string"}},
            "required": ["name", "required"]}}},
        "required": ["blocks"]}
    min_blocks = 1           # load at least this many blocks when the router offers them
    router = None            # separate LLM for routing (defaults to self.llm)

    # -- routing cards: a derived inventory of what each block CONTAINS ------------
    def _display(self, fam: str, region: str) -> str:
        e = fam.rsplit("/", 1)[-1]
        e = re.sub(r"_\d{3,4}(\.md)?$", "", e)
        for p in (f"doc_{region}_", "doc_"):
            if e.startswith(p):
                e = e[len(p):]; break
        return e.replace("_", " ").strip()

    def _block_cards(self) -> list[dict]:
        if getattr(self, "_cards_cache", None) is not None:
            return self._cards_cache
        cards = []
        for b in self.blocks:
            region = (b.get("files") or [""])[0].split("/")[0]
            files = set(b.get("files") or [])
            entities = sorted({self._display(f, region) for f in b.get("families", [])})
            for u in b.get("units", []):
                if u.role == "card" and u.scope and "/" in u.scope:
                    entities.append(self._display(u.scope, region))
            entities = sorted(set(e for e in entities if e))
            # headings = the source's own structure; drop doc titles (first heading) and numbering
            heads: Counter = Counter()
            for e in self._retrievable:
                if e.file in files and e.kind == "section" and e.name:
                    h = re.sub(r"^[\d.\s]+", "", e.name).strip()
                    h = re.sub(r"^internal:\s*", "", h, flags=re.I)   # keep the operation, drop the tag
                    if h and len(h) < 60 and not re.fullmatch(r"faqs?(-\d+)?", h.lower()):
                        heads[h.lower()] += 1
            topics = [h for h, _ in heads.most_common(40)]
            attrs: Counter = Counter()
            for u in b.get("units", []):
                if u.role == "card":
                    for row in u.statement.split(";"):
                        k = row.partition(":")[0].strip().replace("_", " ")
                        if 3 < len(k) < 40:
                            attrs[k] += 1
            cards.append({"name": b["name"], "description": b.get("description", ""),
                          "entities": entities, "topics": topics,
                          "attributes": [a for a, _ in attrs.most_common(20)]})
        self._cards_cache = cards
        return cards

    def _cards_text(self) -> str:
        out = []
        for c in self._block_cards():
            out.append(f"BLOCK: {c['name']}\n  about: {c['description'][:200]}\n"
                       f"  entities: {', '.join(c['entities'][:30]) or '-'}\n"
                       f"  topics/operations (section headings): {'; '.join(c['topics'][:40]) or '-'}\n"
                       f"  attributes recorded: {', '.join(c['attributes'][:20]) or '-'}")
        return "\n\n".join(out)

    def _mech_hints(self, question: str) -> tuple[str, list[str]]:
        """Exact entity-name matches and heading-word overlaps: the cheap,
        certain part of routing that the LLM should not have to guess."""
        q = question.lower().replace("_", " ")   # entity display names are underscore-free (station_42.py -> "station 42.py")
        qt = set(_tokens(question))
        ent_hits, topic_hits = [], []
        for c in self._block_cards():
            for e in c["entities"]:
                el = e.lower()
                if len(el) >= 6 and el in q and not re.fullmatch(r"(general|credit cards|bank accounts)", el):
                    ent_hits.append((e, c["name"]))
            score = sum(1 for t in c["topics"] if len(set(_tokens(t)) & qt) >= 2)
            if score:
                topic_hits.append((c["name"], score))
        lines = []
        if ent_hits:
            lines.append("Exact entity matches: " + "; ".join(f"{e} -> {n}" for e, n in ent_hits[:8]))
        if topic_hits:
            topic_hits.sort(key=lambda x: -x[1])
            lines.append("Heading-word overlaps: " + ", ".join(f"{n} ({k})" for n, k in topic_hits[:4]))
        return ("\n".join(lines) or "(none)"), sorted({n for _, n in ent_hits})

    def _route_blocks(self, question: str) -> list[tuple[str, bool]]:
        """Returns ranked (block, required): the routing cards, mechanical hints
        and an explicit interpretation step; the router is asked to verify
        that no second block is needed before returning one."""
        names = {b["name"] for b in self.blocks}
        llm = self.router or self.llm
        hints, ent_blocks = self._mech_hints(question)
        out = llm.complete_json(
            system=("The knowledge source is split into content blocks. Each block's routing "
                    "card lists, mechanically, what it CONTAINS: its entities (products/tools), "
                    "its section headings (operations and topics) and the attributes recorded. "
                    "First interpret the question: the entities it names or implies, and the "
                    "aspects/operations it asks about. Then rank the blocks a reader must load "
                    "(up to 4), marking each required or optional. Use the mechanical hints: an "
                    "exact entity match settles which block holds that entity (personal and "
                    "business products are different blocks). Before returning ONE block, "
                    "explicitly check that no second block is needed for another entity, a "
                    "procedure, an eligibility rule or a shared policy in the question. Copy "
                    "block names exactly. Do not answer the question."),
            user=f"Question: {question}\n\n## Mechanical hints\n{hints}\n\n## Routing cards\n{self._cards_text()}",
            schema=self.RICH_ROUTE_SCHEMA, purpose="sm_block_route_rich")
        ranked: list[tuple[str, bool]] = []
        for item in (out.get("blocks") or []):
            if isinstance(item, dict) and item.get("name") in names and item["name"] not in [r[0] for r in ranked]:
                ranked.append((item["name"], bool(item.get("required", True))))
        for n in ent_blocks:                 # exact entity matches are always candidates
            if n not in [r[0] for r in ranked]:
                ranked.append((n, True))
        return ranked[:4]

    def _load_blocks(self, question: str) -> list[SourceModelUnit]:
        ranked = self._route_blocks(question)
        # load order: required first (in rank order), then optional; budget-bound,
        # but the first block always loads and min_blocks are loaded when offered
        order = [n for n, req in ranked if req] + [n for n, req in ranked if not req]
        by_name = {b["name"]: b for b in self.blocks}
        units, used, spent = [], [], 0
        qt = set(_tokens(question))
        for n in order:
            bu = [u for u in by_name[n]["units"] if self._fresh(u)]
            cost = sum(len(u.statement) // 4 for u in bu)
            if used and spent + cost > self.load_budget and len(used) >= self.min_blocks:
                continue
            if self.load_budget and spent + cost > self.load_budget:
                # a single block larger than the budget: keep the question-nearest units (scaffold first)
                # so the prompt stays bounded (a 51k-token block once overflowed the answerer's context)
                room = max(0, self.load_budget - spent)
                ranked = sorted(bu, key=lambda u: (u.level != "scaffold", -len(qt & set(_tokens(u.statement)))))
                bu, cost = [], 0
                for u in ranked:
                    c = len(u.statement) // 4
                    if cost + c > room:
                        continue
                    bu.append(u); cost += c
            units += bu; used.append(n); spent += cost
        self._last_blocks = used
        return units
