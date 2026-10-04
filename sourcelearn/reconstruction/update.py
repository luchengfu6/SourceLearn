"""The update layer: every change to a source model M passes through here.

Trust gates (mechanical unless stated):
- grounding: a statement enters M only when a grounder finds excerpts that
  ENTAIL it (`ground`, `ground_or_narrow`);
- scope: evidence must lie inside the target unit's declared scope
  (`scope_allows`); L3 (level "global") is never edited directly;
- novelty: `m_has_it` answers "does M already express this?" before anything
  is written.
"""
from __future__ import annotations

import re

from sourcelearn.draft.induce import sibling_key
from sourcelearn.core.llm import LLMClient
from sourcelearn.core.schema import SourceElement, coerce_str_list
from sourcelearn.source_model import SourceModelUnit

EXCERPT_CHARS = 2000   # code symbols run to 1200 chars; a 900 cut hid the tail from judge/grounder
from sourcelearn.reconstruction.defaults import GROUND_CHARS  # noqa: E402  (6000: evidence shown to the grounder)

# ----------------------------------------------------------- lexical helpers
_WORD = re.compile(r"[a-z0-9_$%.]+")
_SUFFIX = re.compile(r"(ing|ed|es|s)$")
STOP = {"the", "and", "for", "with", "that", "this", "are", "is", "of", "to", "a", "in",
        "on", "by", "or", "not", "its", "their", "be", "as", "an", "when", "only", "card",
        "cards", "credit", "rewards", "must", "can", "which", "from", "also", "before",
        "after", "if", "account", "customer", "users", "user"}
KNOWN_OVERLAP = 0.6      # candidate lesson vs an existing M unit -> M already has it


def _toks(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _stem(w: str) -> str:
    if len(w) > 4:
        return _SUFFIX.sub("", w)
    return w[:-1] if len(w) == 4 and w.endswith("s") else w     # fees -> fee, days -> day


def _content(text: str) -> set[str]:
    # card row keys are snake_case (subscription_required): split them
    return {_stem(w) for w in _toks(text.replace("_", " ")) if w not in STOP and len(w) > 2}


def family_key(entity: str) -> str:
    """Cross-region counterpart key: doc_business_credit_cards_business_gold_rewards_card
    and doc_credit_cards_gold_rewards_card both -> gold_rewards_card."""
    e = entity.rsplit("/", 1)[-1]
    e = re.sub(r"_\d{3,4}(\.md)?$", "", e)
    e = re.sub(r"^doc_", "", e)
    for _ in range(3):   # strip region words progressively: region names are dir names
        e2 = re.sub(r"^(business_|personal_|checking_accounts_|savings_accounts_|credit_cards_|"
                    r"bank_accounts_|business_checking_accounts_|business_savings_accounts_|"
                    r"business_credit_cards_|buy_now_pay_later_|customer_support_|everyone_pay_)", "", e)
        if e2 == e:
            break
        e = e2
    return e


def m_has_it(statement: str, units: list[SourceModelUnit], exclude: str = "",
             entities: list[str] | None = None) -> str | None:
    """Mechanical 'M already has this'. Two rules, both biased toward KEEPING a
    candidate (a false 'absent' is caught later by the trainer's reflect; a
    false 'known' loses a real target):
      1. an M statement covers >= KNOWN_OVERLAP of the candidate's content
         tokens (normalised by the CANDIDATE, so a 4-token card row cannot
         'cover' a 15-token lesson);
      2. a card of one of the candidate's entities has a row whose key
         (>= 2 content tokens) is contained in the candidate AND whose full
         text (key + value) covers >= half of the candidate — a two-word
         cell cannot stand for a multi-step procedure."""
    st = _content(statement)
    if not st:
        return None
    ent_keys = {family_key(e) for e in (entities or [])}
    best, best_id = 0.0, None
    for u in units:
        if u.unit_id == exclude or u.level == "scaffold":
            continue
        pt = _content(u.statement)
        if len(pt) >= 3:
            o = len(st & pt) / len(st)
            if o > best:
                best, best_id = o, u.unit_id
        if u.role == "card" and u.scope and family_key(u.scope) in ent_keys:
            for row in u.statement.split(";"):
                key = _content(row.partition(":")[0])
                if len(key) >= 2 and key <= st and len(st & _content(row)) / len(st) >= 0.5:
                    return u.unit_id
    return best_id if best >= KNOWN_OVERLAP else None


def m_tokens(units: list[SourceModelUnit]) -> int:
    return sum(len(u.statement) // 4 + sum(len(c) for c in u.conditions) // 4
               for u in units if u.level != "scaffold")


def nearest_units(stmt: str, model: list[SourceModelUnit], k: int = 6) -> list[SourceModelUnit]:
    st = _content(stmt)
    scored = []
    for u in model:
        if u.level == "scaffold":
            continue
        pt = _content(u.statement)
        if pt:
            scored.append((len(st & pt) / max(1, len(st)), u))
    return [u for o, u in sorted(scored, key=lambda x: -x[0])[:k] if o > 0]


# ----------------------------------------------------------------- grounding
GROUND_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"},
                   "supporting_excerpt_ids": {"type": "array", "items": {"type": "string"}},
                   "unsupported_claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["supported", "supporting_excerpt_ids"],
}

GROUND_RULES = (
    "Decide whether the candidate statement is SUPPORTED by the source "
    "excerpts shown (entailed by them, not merely consistent). If supported, "
    "return the ids of the excerpts that establish it; if any part is not "
    "established, supported=false and list the unsupported parts. "
    "Source-bound rule for NEGATIVE claims: a statement saying that something does NOT exist, is NOT supported, or CANNOT be done is supported only if an excerpt explicitly states that denial; a feature merely absent from the excerpts is NOT evidence that it is absent from the source. ")


def ground(statement: str, gold: dict[str, SourceElement], llm_strong: LLMClient
           ) -> tuple[bool, dict[str, str]]:
    """Grounder: the writer decides WHAT to change; the grounder decides
    WHERE the evidence is. Returns (supported, {anchor: fingerprint})."""
    text = "\n\n".join(f"[{eid}]\n{e.excerpt[:GROUND_CHARS]}" for eid, e in gold.items())
    out = llm_strong.complete_json(
        system=GROUND_RULES,
        user=f"Candidate statement:\n{statement}\n\n## Source excerpts\n{text}",
        schema=GROUND_SCHEMA, purpose="hm_ground")
    ids = [r for r in (resolve_excerpt_id(a, gold)
                       for a in coerce_str_list(out.get("supporting_excerpt_ids"))) if r]
    if not out.get("supported") or not ids:
        return False, {}
    return True, {a: gold[a].fingerprint for a in ids}


GROUND_BATCH_SCHEMA = {"type": "object", "properties": {"claims": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "supported": {"type": "boolean"},
    "supporting_excerpt_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["n", "supported"]}}}, "required": ["claims"]}

GROUND_BATCH_RULES = (
    "For EACH numbered candidate statement decide whether it is SUPPORTED by the source excerpts shown "
    "(entailed by them, not merely consistent); if supported, give the ids of the excerpts that establish it. "
    "A statement with any part not established is unsupported. "
    "Source-bound rule for NEGATIVE claims: a statement saying that something does NOT exist, is NOT supported, or CANNOT be done is supported only if an excerpt explicitly states that denial; a feature merely absent from the excerpts is NOT evidence that it is absent from the source. "
    "Judge every statement independently; return one entry per number.")


def ground_batch(statements: list[str], gold: dict[str, SourceElement], llm_strong: LLMClient,
                 batch: int = 20) -> list[tuple[bool, dict[str, str]]]:
    """`ground` for many statements against the same evidence in a few calls
    (same judgement, one prompt per `batch` statements). Returns per
    statement (supported, {anchor: fingerprint})."""
    from sourcelearn.reconstruction.inspect import paged   # local import: inspect imports this module
    text = "\n\n".join(paged(eid, e, GROUND_CHARS) for eid, e in gold.items())
    out: list[tuple[bool, dict[str, str]]] = [(False, {})] * len(statements)
    for i in range(0, len(statements), batch):
        chunk = statements[i:i + batch]
        res = llm_strong.complete_json(
            system=GROUND_BATCH_RULES,
            user="## Candidate statements\n" + "\n".join(f"({j + 1}) {s}" for j, s in enumerate(chunk))
                 + f"\n\n## Source excerpts\n{text}",
            schema=GROUND_BATCH_SCHEMA, purpose="hm_ground_batch")
        for c in res.get("claims") or []:
            if not isinstance(c, dict) or not str(c.get("n", "")).isdigit():
                continue
            n = int(c["n"])
            if not 1 <= n <= len(chunk) or not c.get("supported"):
                continue
            ids = [r for r in (resolve_excerpt_id(a, gold) for a in coerce_str_list(c.get("supporting_excerpt_ids"))) if r]
            if ids:
                out[i + n - 1] = (True, {a: gold[a].fingerprint for a in ids})
    return out


def narrow_to_evidence(statement: str, gold: dict[str, SourceElement], llm_strong: LLMClient) -> str:
    """Scope correction: a writer often generalises over every entity in its
    window while only some excerpts say it. Returns the narrowed statement
    ('' when nothing survives)."""
    try:
        nar = llm_strong.complete_json(
            system=("Rewrite the statement so that it claims ONLY what the excerpts "
                    "establish: drop or restrict the entities/claims the excerpts do "
                    "not support, keep the wording otherwise. Return an empty string "
                    "if nothing survives."),
            user=f"Statement:\n{statement}\n\n## Excerpts\n"
                 + "\n\n".join(f"[{eid}]\n{e.excerpt[:GROUND_CHARS]}" for eid, e in gold.items()),
            schema={"type": "object", "properties": {"statement": {"type": "string"}},
                    "required": ["statement"]}, purpose="assim_narrow")
    except ValueError:      # no valid JSON from an open model: nothing survives, the run goes on
        return ""
    return str(nar.get("statement") or "").strip()


def ground_or_narrow(statement: str, gold: dict[str, SourceElement], llm_strong: LLMClient
                     ) -> tuple[str, dict[str, str]]:
    """ground; on failure narrow once and re-ground. Returns (statement,
    anchors) — statement '' means ungrounded."""
    ok, cited = ground(statement, gold, llm_strong)
    if ok:
        return statement, cited
    stmt2 = narrow_to_evidence(statement, gold, llm_strong)
    if not stmt2 or stmt2 == statement:
        return "", {}
    ok, cited = ground(stmt2, gold, llm_strong)
    return (stmt2, cited) if ok else ("", {})


def family_of_element(e: SourceElement) -> str:
    return sibling_key(e)


def scope_allows(unit: SourceModelUnit, gold: dict[str, SourceElement]) -> bool:
    """Evidence must lie inside the unit's declared scope. Judged by the
    files the unit is ANCHORED in (works for docs and code alike), with
    the region as the outer bound."""
    if unit.level == "global":
        return False  # L3 is revised only through L2 + recompile
    files = {e.file for e in gold.values()}
    region = (unit.scope or "").split("/")[0]
    if unit.level == "regional":
        # a cross-region relation is scoped "regA/regB": evidence from either area
        regions = [seg for seg in (unit.scope or "").split("/") if seg and not seg.startswith("doc_")] or [region]
        return any(f.startswith(r + "/") for r in regions for f in files)
    ufiles = {a.split("#")[0] for a in unit.support_anchors}
    if files & ufiles:
        return True
    # same family (docs: name prefix; code: same file) counts as in scope
    fams = {family_of_element(e) for e in gold.values()}
    ufams = {sibling_key(SourceElement(element_id=a, node_id="x", kind="section",
                                       name="", file=a.split("#")[0], fingerprint=""))
             for a in unit.support_anchors}
    if fams & ufams:
        return True
    # docs units whose scope IS the family name (no anchors needed)
    scope = unit.scope or ""
    return any(scope == f or scope.endswith("/" + f.split("/")[-1])
               or f.endswith("/" + scope.split("/")[-1]) for f in fams)


def resolve_excerpt_id(a: str, gold: dict[str, SourceElement]) -> str | None:
    """LLM-echoed evidence ids come back imperfect (truncated path, bare
    basename, wrapped in [brackets]). Coerce leniently; ambiguity fails,
    never guesses."""
    a = a.strip().strip("[]").strip()
    if a in gold:
        return a
    if len(a) < 8:
        return None
    hits = [k for k in gold if k.endswith(a) or a.endswith(k)
            or (len(a) >= 16 and a in k)]
    return hits[0] if len(hits) == 1 else None


def resolve_unit(tid: str, by_id: dict[str, SourceModelUnit]) -> SourceModelUnit | None:
    if tid in by_id:
        return by_id[tid]
    if len(tid) < 6:
        return None
    hits = [u for k, u in by_id.items()
            if k.endswith(tid) or tid.endswith(k)]
    return hits[0] if len(hits) == 1 else None


