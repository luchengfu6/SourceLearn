"""Adaptive study planning (paper: the adaptive-study stage of Self-Directed
Source Learning). The planner sees the observation digest of the inspection
(every entity with its notes so far) and chooses DEEPEN(entity) or
CONNECT(entity_a, entity_b) actions, each with ONE focused question that
resolves one identified gap.
"""
from __future__ import annotations

from sourcelearn.core.llm import LLMClient
from sourcelearn.core.schema import coerce_str_list
from sourcelearn.reconstruction.defaults import STUDY_DIGEST_CHARS
from sourcelearn.source_model import SourceModelUnit


def entity_index(blocks: list[dict], entity_key, n_units: int = 3) -> tuple[dict[str, dict], str]:
    """{entity key: {block, units}} and its rendering (one line per entity)."""
    ents: dict[str, dict] = {}
    for b in blocks:
        for u in b["units"]:
            if u.level == "scaffold" or u.role == "card":
                continue
            ents.setdefault(entity_key(u), {"block": b["name"], "units": []})["units"].append(u)
    lines = []
    for key, e in sorted(ents.items()):
        heads = "; ".join(u.statement[:90] for u in e["units"][:n_units])
        lines.append(f"- {key}  [{e['block']}; {len(e['units'])} units]: {heads}")
    return ents, "\n".join(lines)


# ------------------------------------------------ adaptive study over inspection observations
STUDY_SCHEMA = {"type": "object", "properties": {"actions": {"type": "array", "items": {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["DEEPEN", "CONNECT"]},
    "entities": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
    "observation": {"type": "string", "description": "the ONE unresolved observation (copied from the digest) this action resolves"},
    "question": {"type": "string"}, "why": {"type": "string"}},
    "required": ["kind", "entities", "observation", "question"]}}}, "required": ["actions"]}

STUDY_PLAN_PROMPT = (
    "You direct the self-study of an agent that maintains a compact SOURCE MODEL of a knowledge source. A full "
    "inspection of the source against the model has produced the OBSERVATIONS below: temporary notes on what each "
    "entity's source establishes that the model does not yet explain well. Before the model is consolidated, choose "
    "the {n} most consequential observations to resolve, ONE observation per action: DEEPEN one entity when its "
    "observation leaves a mechanism, governing condition, step order or exception unresolved; CONNECT two entities "
    "when an observation names a relation between them the model does not express (one constraining, triggering or "
    "feeding the other; a shared rule with different instances; a procedure or dependency spanning both). For each "
    "action copy the observation it resolves and write ONE focused question — the single thing that has to be "
    "understood to resolve that observation (for example: under what condition does Y switch to Z, and what follows?). "
    "Never ask for everything about an entity and never enumerate several aspects in one question; a question is an "
    "instrument for resolving one identified gap. Prefer entities and pairs not yet studied; never repeat an action "
    "already taken. Copy entity keys exactly as listed.")

STUDY_QUESTION_CHARS = 220   # a focused question; longer ones enumerate aspects and are dropped

STUDY_DIGEST_OBS = 3        # observations shown per entity in the planner digest


def observation_digest(ents: dict[str, dict], notes_by_entity: dict[str, list[dict]], chars_by_entity: dict[str, int],
                       cap: int = STUDY_DIGEST_CHARS, per_entity: int = STUDY_DIGEST_OBS) -> str:
    """One line per entity (block, units held, source size, observations),
    entities with the most observations first, within `cap` chars; the
    entities that did not fit are named so the planner knows they exist."""
    keys = sorted(set(ents) | set(notes_by_entity), key=lambda e: (-len(notes_by_entity.get(e, [])), e))
    lines, spent, rest = [], 0, []
    for e in keys:
        ns = notes_by_entity.get(e, [])
        info = ents.get(e, {})
        line = (f"- {e}  [{info.get('block', '?')}; {len(info.get('units', []))} units; "
                f"{chars_by_entity.get(e, 0):,} source chars; {len(ns)} observations]")
        line += "".join(f"\n    * {n['statement'][:220]}" + (f" (=> {n['implication'][:120]})" if n.get("implication") else "")
                        + (" [connect]" if n.get("pair") else "") for n in ns[:per_entity])
        if spent + len(line) > cap:
            rest.append(e); continue
        lines.append(line); spent += len(line)
    if rest:
        lines.append(f"- ({len(rest)} more entities not shown: " + ", ".join(rest[:80]) + (", ..." if len(rest) > 80 else "") + ")")
    return "\n".join(lines)


def study_actions(ents: dict[str, dict], digest: str, relations: list[SourceModelUnit], taken: list[tuple],
                  n: int, brief: str, llm: LLMClient) -> list[dict]:
    """Up to `n` DEEPEN / CONNECT actions the planner chooses on the
    observations; unknown entities, empty questions and repeats of `taken`
    ((kind, entities) already executed) are dropped."""
    rel_txt = "\n".join(f"- ({u.scope}) {u.statement[:160]}" for u in relations[:60]) or "(none yet)"
    taken_txt = "\n".join(f"- {k} {' <-> '.join(es)}" for k, es in taken) or "(none)"
    out = llm.complete_json(
        system=STUDY_PLAN_PROMPT.format(n=n) + (f"\n\nReading instruction for this source:\n{brief[:2000]}" if brief else ""),
        user=(f"## Entities with their inspection observations\n{digest}\n\n## Existing cross-entity units\n{rel_txt}\n\n"
              f"## Actions already taken\n{taken_txt}"),
        schema=STUDY_SCHEMA, purpose="study_plan")
    acts, seen = [], {(k, tuple(es)) for k, es in taken}
    for a in (out.get("actions") or []):
        if not isinstance(a, dict):
            continue
        kind, q = a.get("kind"), str(a.get("question") or "").strip()
        es = [e for e in dict.fromkeys(coerce_str_list(a.get("entities"))) if e in ents]
        es = es[:1] if kind == "DEEPEN" else sorted(es[:2]) if kind == "CONNECT" else []
        if not q or (kind == "DEEPEN" and len(es) != 1) or (kind == "CONNECT" and len(es) != 2):
            continue
        if len(q) > STUDY_QUESTION_CHARS:      # enumerating aspects again: not one gap, one question
            continue
        key = (kind, tuple(es))
        if key in seen:
            continue
        seen.add(key)
        acts.append({"kind": kind, "entities": es, "question": q, "why": str(a.get("why") or "")[:200],
                     "observation": str(a.get("observation") or "")[:400]})
        if len(acts) >= n:
            break
    return acts
