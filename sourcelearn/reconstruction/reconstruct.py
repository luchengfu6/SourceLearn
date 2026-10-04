"""Model reconstruction: the writers shared by self-directed and task-guided
source learning. Every change to M passes a grounding gate first.

Local reconstruction (failure-guided local refinement):

    reconstruct_region(target, units, evidence, llm) -> proposal
        KNOWN          the shown units already express what the evidence says
        ADD            genuinely new grounded units, nothing replaced
        REPLACE_GROUP  these old units are superseded by this group of new
                       units (covers revise 1->1, merge n->1, split 1->n,
                       restructure n->m without naming the operation)

    ground_and_apply(model, proposal, evidence, ...) -> telemetry
        ground first, replace second: every new unit of a REPLACE_GROUP must
        ground (one narrowing attempt allowed) or the whole group is refused —
        a correct old unit is never deleted for a half-grounded replacement.
        Old units disappear only by being superseded here (or by a stale
        fingerprint); nothing is ever deleted on judgement alone.
        Preservation gate: an old unit is superseded only if the new group
        ENTAILS it, or it was not supported by the evidence anyway (wrong);
        an old unit the evidence supports but the new group dropped stays.
        Entity cards (the compiler's attribute grid) are never replaced whole.

Observe, then rewrite (self-study consolidation, representation calibration):

    observe(target, units, evidence, llm) -> study notes (no edit)
    rewrite_region(region, old units, notes, evidence, llm) -> proposal
    prepare_rewrite -> apply_rewrite
        set-level gates: every new unit grounds, old meanings the rewrite
        does not represent are KEPT, the diff (subsumed / kept / new) is
        mechanical; a rewrite that neither improves structure nor
        incorporates notes is a NOOP. Atomic.
"""
from __future__ import annotations

from sourcelearn.core.llm import LLMClient
from sourcelearn.core.schema import SourceElement, coerce_str_list
from sourcelearn.reconstruction.inspect import render_evidence
from sourcelearn.reconstruction.update import (GROUND_CHARS, _content, ground, ground_batch, ground_or_narrow, m_has_it, m_tokens,
                                 narrow_to_evidence, nearest_units, resolve_unit, scope_allows)
from sourcelearn.source_model import SourceModelUnit, _family_of
from sourcelearn.reconstruction.grounding import supported_by_excerpts

VERDICTS = ("KNOWN", "ADD", "REPLACE_GROUP")
ROLES = ("pattern", "mechanism", "distinction", "exception", "procedure", "relation", "detail", "card")
from sourcelearn.reconstruction.defaults import DUP_OVERLAP, GROWTH_ALLOWANCE, MAX_REWRITE_UNITS, OBSERVATIONS_PER_QUESTION, REWRITE_STATEMENT_CHARS, STATEMENT_CHARS  # noqa: E402

MAX_NEW_UNITS = 4
MAX_REPLACE = 4       # a reconstruction is LOCAL: at most this many old units superseded at once

RECON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "why": {"type": "string"},
        "replace_ids": {"type": "array", "items": {"type": "string"},
                        "description": "REPLACE_GROUP only: ids of the shown units the new units supersede"},
        "new_units": {"type": "array", "maxItems": MAX_NEW_UNITS, "items": {"type": "object", "properties": {
            "statement": {"type": "string"},
            "role": {"type": "string", "enum": list(ROLES)},
            "scope": {"type": "string", "description": "scope string copied from a shown unit, or the entity/region it concerns"},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "entities": {"type": "array", "items": {"type": "string"}}},
            "required": ["statement", "role"]}},
    },
    "required": ["verdict"],
}

RECON_PROMPT = (
    "You maintain a long-lived SOURCE MODEL: a compact set of grounded units (rules, mechanisms, "
    "procedures, distinctions, exceptions, entity cards) that an agent keeps in context while using a "
    "knowledge source. A LEARNING TARGET was investigated and the raw source evidence is shown, together "
    "with the units of the current model that concern it. Decide how this LOCAL portion of the model "
    "should read after the evidence:\n"
    "- KNOWN: the shown units already express what the evidence establishes about the target (nothing "
    "to write; mere rewording is not a reason to change).\n"
    "- ADD: the evidence establishes reusable understanding the model lacks and no shown unit is wrong "
    "or incomplete about it: give the new units, replace nothing.\n"
    "- REPLACE_GROUP: some shown units are incomplete, too coarse, wrongly scoped, fragmented, or "
    "contradicted by the evidence: list their ids in replace_ids and give the group of new units that "
    "supersedes them. The new group must carry everything the old units said that is still true, so a "
    "reader loses nothing. Use it to merge fragments into one rule, to split an over-broad unit, to add "
    "the missing condition or dependency into the unit that should hold it, or to connect two areas.\n"
    "Keep every change LOCAL: replace at most 4 units and write at most 4; if more seems needed, pick "
    "the fragment that matters most for the target and leave the rest untouched.\n"
    "Rules for new units: declarative statements about how the SOURCE works, established by the shown "
    "evidence (never by general knowledge, never the answer to one question); copy names, identifiers, "
    "numbers and conditions verbatim; scope each unit to the entity or region the evidence actually "
    "concerns; prefer few, structural units over many facts the raw source can be asked for. "
    "Statements at most 600 characters.")

TARGET_HEADER = {
    "question": "Study question",
    "residual": "Task residual (what the model lacked when a real task failed)",
    "review": "Full read of one entity's source (every section shown; nothing was retrieved or omitted)",
    "connect": "Connective question about two entities (evidence retrieved from both)",
    "task": "Task-conditioned refinement (what a real task required from this region of the source)",
    "calibrate": "Recalibration of an already studied region under the representation policy (full source text shown)",
    "deepen": "Deepening study of one entity (a probing question on its inspection observations; evidence retrieved from the entity)",
}
DEEPEN_NOTE = (
    "\nThis is a DEEPENING study: an inspection of this entity already produced the observations listed with the "
    "question. Note only what the evidence establishes that resolves or sharpens them — the governing condition, the "
    "mechanism, the step order, the exception, the distinction — not what they already say and not what the model "
    "already explains.")
CALIBRATE_NOTE = (
    "\nThis is a RECALIBRATION: the region was already studied, so do not note what a general reading would add. "
    "Note ONLY what the representation policy given in the instructions calls for and the current units do not make "
    "explicit (or represent at the wrong granularity), citing the policy item in the implication. Return an empty "
    "observations list when the region already satisfies the policy.")


def _policy_note(policy: str) -> str:
    """Representation policy (sourcelearn.task_learning.policy): what deserves representation
    in this source model. Appended to every writer prompt after the brief,
    never truncated with it."""
    return (f"\n\nRepresentation policy for this source model (what kinds of information, at what granularity and "
            f"which relations deserve explicit representation; apply it when deciding what to write, keep, merge or "
            f"leave to the raw source):\n{policy}" if policy else "")
CONNECT_NOTE = (
    "\nThis is a CONNECTIVE study: note only what the evidence establishes about how the two entities relate "
    "(one constraining, triggering or feeding the other; a shared rule and where they differ; a procedure or "
    "eligibility dependency spanning both). Facts about one entity alone are not notes here.")
REVIEW_NOTE = (
    "\nThis is a REVIEW: the entity's complete source text is shown, so judge the model's portion about it "
    "as a whole — is every rule, condition, procedure step, exception and cross-reference the text establishes "
    "represented, correctly scoped and not fragmented? Reconstruct only what a reader would otherwise get wrong; "
    "look-up values stay in the source.")
TASK_NOTE = (
    "\nThis is a TASK-CONDITIONED refinement: a real task depended on the cited claims and the current model did "
    "not make them explicit. Refine this local portion so that the source-grounded information needed to understand "
    "and resolve such tasks is represented explicitly — the field, option, condition, relation or step itself with "
    "the context that governs it — while preserving existing supported knowledge. Write statements about the source, "
    "never the task's answer.")


def _known_scopes(model: list[SourceModelUnit]) -> tuple[set[str], set[str]]:
    scopes = {u.scope for u in model if u.scope}
    regions = {s.split("/")[0] for s in scopes}
    return scopes, regions


def clamp_scope(raw: str, evidence: dict[str, SourceElement], scopes: set[str], regions: set[str]) -> str:
    """LLM-written scope -> a scope the model knows (verbatim, or by tail),
    else the family/region the evidence lies in."""
    r = (raw or "").strip()
    if r in scopes or r in regions:
        return r
    tail = r.split("/")[-1]
    hits = [s for s in scopes if s.split("/")[-1] == tail] if tail else []
    if len(hits) == 1:
        return hits[0]
    ev_regions = sorted({e.file.split("/")[0] for e in evidence.values() if "/" in e.file})
    if len(ev_regions) > 1:
        return "/".join(ev_regions)
    files = sorted({e.file for e in evidence.values()})
    return _family_of(files[0]) if files else (ev_regions[0] if ev_regions else "")


def scope_from_anchors(raw: str, anchors: dict[str, str], scopes: set[str], regions: set[str]) -> str:
    """Scope of a rewritten unit: an entity scope the writer copied verbatim
    wins; otherwise the family the anchors lie in, or the region when they
    span several families (the writer tends to scope everything to the region)."""
    r = (raw or "").strip()
    if r in scopes and r not in regions:
        return r
    fams = sorted({_family_of(a.split("#")[0]) for a in anchors})
    if len(fams) == 1:
        return fams[0]
    regs = sorted({f.split("/")[0] for f in fams})
    return "/".join(regs) if regs else r


def unit_level(scope: str, regions: set[str]) -> str:
    parts = scope.split("/")
    # a region name, or "regA/regB" (cross-region relation) -> regional; "region/entity" -> local
    region_level = len(parts) == 1 or all(p in regions for p in parts)
    return "regional" if region_level else "local"


def _may_supersede(old: SourceModelUnit, new: list[SourceModelUnit], evidence: dict[str, SourceElement],
                   llm: LLMClient, tele: dict) -> bool:
    """Preservation gate for one old unit (see module docstring)."""
    if old.role == "card":
        tele["rejected"].append(f"CARD_KEPT {old.unit_id}"); tele["kept_old"].append(old.unit_id)
        return False
    pseudo = {u.unit_id: SourceElement(element_id=u.unit_id, node_id="m", kind="section", name="", file="(new units)",
                                       fingerprint="", excerpt=u.statement + (f" Conditions: {u.conditions}" if u.conditions else ""))
              for u in new}
    ok, _ = supported_by_excerpts(old.statement, pseudo, llm)
    if ok:
        return True
    # not carried over: it may go only if the evidence actually covers what it
    # claims (its anchors are in the evidence) and contradicts / fails to support it
    covered = {a.split("#")[0] for a in old.support_anchors} & {e.file for e in evidence.values()}
    still_true = True
    if covered:
        still_true, _ = supported_by_excerpts(old.statement, evidence, llm)
    if still_true:
        tele["rejected"].append(f"NOT_PRESERVED {old.unit_id}"); tele["kept_old"].append(old.unit_id)
        return False
    return True      # its own source no longer supports it: a wrong unit may go


def relevant_units(model: list[SourceModelUnit], text: str, must: list[str] | None = None, k: int = 8
                   ) -> list[SourceModelUnit]:
    """Units a reconstruction may touch: explicitly named ones plus the
    nearest by content overlap; scaffold and L3 never (L3 is re-derived)."""
    by_id = {u.unit_id: u for u in model}
    out: list[SourceModelUnit] = []
    for uid in must or []:
        u = resolve_unit(str(uid), by_id)
        if u is not None and u.level not in ("scaffold", "global") and u not in out:
            out.append(u)
    for u in nearest_units(text, model, k):
        if u.level not in ("scaffold", "global") and u not in out:
            out.append(u)
    return out[:k + len(must or [])]


def render_units(units: list[SourceModelUnit]) -> str:
    return "\n".join(f"- [{u.unit_id}] ({u.role}, scope {u.scope}) {u.statement}"
                     + (f" | conditions: {u.conditions}" if u.conditions else "")
                     for u in units) or "(no unit of the current model concerns this target)"


def reconstruct_region(target: dict, units: list[SourceModelUnit], evidence: dict[str, SourceElement],
                       llm: LLMClient, brief: str = "", policy: str = "") -> dict:
    """target = {kind: question|residual, text, [type], [entities], [why]}.
    Returns the normalised proposal: verdict, why, replace_ids, new_units."""
    kind = target.get("kind", "question")
    head = f"## {TARGET_HEADER.get(kind, 'Learning target')}\n{target.get('text', '')}"
    if target.get("type"):
        head += f"\n(residual type: {target['type']}; kind of knowledge wanted: {target.get('kind_hint', '')})"
    if target.get("entities"):
        head += f"\nEntities involved: {', '.join(target['entities'])}"
    if target.get("why"):
        head += f"\nWhy it was raised: {target['why']}"
    if kind == "review":
        head += REVIEW_NOTE
    if kind == "task":
        head += TASK_NOTE
    out = llm.complete_json(
        system=RECON_PROMPT + (f"\n\nReading instruction for this source:\n{brief[:3000]}" if brief else "") + _policy_note(policy),
        user=(f"{head}\n\n## Current model units concerning the target\n{render_units(units)}\n\n"
              f"## Source evidence\n{render_evidence(evidence, GROUND_CHARS)}"),
        schema=RECON_SCHEMA, purpose="reconstruct")
    verdict = str(out.get("verdict") or "KNOWN")
    if verdict not in VERDICTS:
        verdict = "KNOWN"
    new_units = []
    for nu in (out.get("new_units") or [])[:MAX_NEW_UNITS]:
        if not isinstance(nu, dict):
            continue
        stmt = str(nu.get("statement") or "").strip()[:STATEMENT_CHARS]
        if stmt:
            new_units.append({"statement": stmt,
                              "role": nu.get("role") if nu.get("role") in ROLES else "mechanism",
                              "scope": str(nu.get("scope") or ""),
                              "conditions": coerce_str_list(nu.get("conditions"))[:4],
                              "entities": coerce_str_list(nu.get("entities"))})
    return {"verdict": verdict, "why": str(out.get("why") or "")[:300],
            "replace_ids": coerce_str_list(out.get("replace_ids")) if verdict == "REPLACE_GROUP" else [],
            "new_units": new_units if verdict != "KNOWN" else []}


def ground_and_apply(model: list[SourceModelUnit], proposal: dict, evidence: dict[str, SourceElement],
                     llm: LLMClient, origin: str, step_id: str,
                     shown: list[SourceModelUnit] | None = None, verify_known: bool = False) -> dict:
    """Apply a proposal under the gates.
    `shown` = the units the writer saw: replace_ids resolve against these
    only (a union of block compiles repeats ids like l2:<region>:abs:5, and a
    lenient whole-model lookup once superseded an unrelated block's unit).
    `verify_known`: the lexical "M already has it" hit is confirmed by
    entailment (the matched unit must entail the new statement) — task
    refinement writes distinctions next to same-topic units that overlap in
    words but not in meaning. Returns {verdict, applied, new_ids,
    replaced_ids, rejected, tokens_before/after}."""
    tele = {"verdict": proposal["verdict"], "applied": False, "new_ids": [], "replaced_ids": [],
            "rejected": [], "tokens_before": m_tokens(model),
            "n_old": len(proposal["replace_ids"]), "n_new": len(proposal["new_units"]),
            "kept_old": [], "dup_of": {}}       # kept_old: named for replacement but left in place
    if proposal["verdict"] == "KNOWN" or not proposal["new_units"]:
        tele["tokens_after"] = tele["tokens_before"]
        return tele
    by_id = {u.unit_id: u for u in (shown if shown is not None else model)}
    replace = proposal["verdict"] == "REPLACE_GROUP"
    old: list[SourceModelUnit] = []
    if replace:
        # an old unit may be superseded only if it resolves, is editable and the
        # evidence covers its scope; the others simply STAY (never deleted on
        # judgement alone) and the group still replaces what it may
        for rid in proposal["replace_ids"]:
            u = resolve_unit(str(rid), by_id)
            if u is None:
                tele["rejected"].append(f"NO_TARGET {rid}")
            elif u.level in ("global", "scaffold"):
                tele["rejected"].append(f"NOT_EDITABLE {u.unit_id}"); tele["kept_old"].append(u.unit_id)
            elif not scope_allows(u, evidence):
                tele["rejected"].append(f"OFF_SCOPE {u.unit_id}"); tele["kept_old"].append(u.unit_id)
            elif len(old) >= MAX_REPLACE:
                tele["rejected"].append(f"BEYOND_MAX_REPLACE {u.unit_id}"); tele["kept_old"].append(u.unit_id)
            elif u not in old:
                old.append(u)
        if not old:
            replace = False          # nothing replaceable: the new units are an ADD
    scopes, regions = _known_scopes(model)
    # ground every candidate before touching the model
    others = [u for u in model if u not in old]
    grounded: list[SourceModelUnit] = []
    for i, nu in enumerate(proposal["new_units"]):
        stmt, anchors = ground_or_narrow(nu["statement"], evidence, llm)
        if not stmt:
            tele["rejected"].append(f"UNGROUNDED {i + 1}")
            continue
        known = m_has_it(stmt, others, entities=nu["entities"]) if not replace else None
        if known and verify_known:
            by_id_all = {u.unit_id: u for u in others}
            ku = by_id_all.get(known)
            pseudo = {known: SourceElement(element_id=known, node_id="m", kind="section", name="", file="(model)", fingerprint="",
                                           excerpt=ku.statement + (f" Conditions: {ku.conditions}" if ku.conditions else ""))} if ku else {}
            if not pseudo or not supported_by_excerpts(stmt, pseudo, llm)[0]:
                known = None                    # same words, different meaning: not known
        if known:
            tele["rejected"].append(f"KNOWN {i + 1}")
            continue
        scope = clamp_scope(nu["scope"], {a: evidence[a] for a in anchors}, scopes, regions)
        role = nu["role"]
        grounded.append(SourceModelUnit(
            unit_id=f"{origin}:{step_id}:{i + 1}", statement=stmt,
            kind="fact" if role in ("card", "detail") else "model",
            scope=scope, conditions=nu["conditions"], role=role,
            level=unit_level(scope, regions),
            support_anchors=sorted(anchors), support_fingerprints=dict(anchors),
            origin=origin, support_mode="synthesized" if len(anchors) > 1 else "explicit",
            learned_from_task=step_id, derived_from=[u.unit_id for u in old]))
    if replace and (not grounded or len(grounded) < len(proposal["new_units"])):
        tele["rejected"].append("REPLACE_REFUSED")     # ground first, replace second: all new units or nothing
        tele["tokens_after"] = tele["tokens_before"]
        return tele
    if replace:
        old = [u for u in old if _may_supersede(u, grounded, evidence, llm, tele)]
    if not grounded:
        tele["tokens_after"] = tele["tokens_before"]
        return tele
    for u in old:
        model.remove(u)
    model.extend(grounded)
    if replace and not old:
        tele["verdict"] = "ADD"        # everything named was kept: what remains is an addition
    # redundancy telemetry: does a new unit largely restate a unit that stays
    # (a kept-out old one, or anything else)? logged, never enforced
    for nu in grounded:
        st = _content(nu.statement)
        best, best_id = 0.0, None
        for u in model:
            if u is nu or u.level == "scaffold":
                continue
            ov = len(st & _content(u.statement)) / max(1, len(st))
            if ov > best:
                best, best_id = ov, u.unit_id
        if best_id and best >= DUP_OVERLAP:
            tele["dup_of"][nu.unit_id] = {"unit": best_id, "overlap": round(best, 2)}
    tele.update({"applied": True, "new_ids": [u.unit_id for u in grounded],
                 "replaced_ids": [u.unit_id for u in old], "tokens_after": m_tokens(model)})
    return tele


# ------------------------------------------------------------ two-phase study
# Phase A (explore): a question / full read yields temporary OBSERVATIONS,
# never a persistent edit. Phase B (reorganize): a region's old units + its
# observations + their evidence -> the region rewritten as one coherent
# model; the diff (subsumed / kept / new) is mechanical.
OBSERVE_SCHEMA = {"type": "object", "properties": {
    "learned": {"type": "boolean"},
    "observations": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
        "statement": {"type": "string"}, "support_ids": {"type": "array", "items": {"type": "string"}},
        "implication": {"type": "string", "description": "what this means for how the current model organises this region"},
        "concerns_unit_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["statement", "support_ids"]}}},
    "required": ["learned", "observations"]}

OBSERVE_PROMPT = (
    "You are studying a knowledge source to improve a compact SOURCE MODEL of it. The relevant part of the "
    "current model and the raw evidence for a learning target are shown. Record what the evidence teaches that "
    "the model does not yet explain well, as STUDY NOTES (temporary; they will be consolidated later into a "
    "rewrite of this region, so do not write model units now): each note = one observation the evidence "
    "establishes (a rule, condition, dependency, procedure step order, exception, distinction, or how two "
    "things connect), the evidence ids it rests on, its implication for the model (e.g. 'these three facts "
    "are stages of one procedure', 'unit X is missing the governing condition'), and the model units it "
    "concerns. Do not note look-up values the source answers on its own. Return an empty observations list "
    "(learned=false) when the model already explains the target.")


def observe(target: dict, units: list[SourceModelUnit], evidence: dict[str, SourceElement], llm: LLMClient,
            brief: str = "", policy: str = "") -> list[dict]:
    kind = target.get("kind", "question")
    head = f"## {TARGET_HEADER.get(kind, 'Learning target')}\n{target.get('text', '')}"
    if kind == "review":
        head += REVIEW_NOTE.replace("Reconstruct only what", "Note only what")
    if kind == "connect":
        head += CONNECT_NOTE
    if kind == "calibrate":
        head += CALIBRATE_NOTE
    if kind == "deepen":
        head += DEEPEN_NOTE
    out = llm.complete_json(
        system=OBSERVE_PROMPT + (f"\n\nReading instruction for this source:\n{brief[:3000]}" if brief else "") + _policy_note(policy),
        user=(f"{head}\n\n## Current model units concerning the target\n{render_units(units)}\n\n"
              f"## Source evidence\n{render_evidence(evidence, GROUND_CHARS)}"),
        schema=OBSERVE_SCHEMA, purpose="study_observe")
    shown = {u.unit_id for u in units}
    notes = []
    # `learned` is advisory only: models set it false while still listing observations
    if out.get("observations"):
        for o in (out.get("observations") or [])[:OBSERVATIONS_PER_QUESTION]:
            if not isinstance(o, dict) or not str(o.get("statement") or "").strip():
                continue
            support = [s for s in (resolve_unit(str(x), {k: k for k in evidence}) for x in coerce_str_list(o.get("support_ids"))) if s]
            notes.append({"statement": str(o["statement"]).strip()[:STATEMENT_CHARS], "support": support,
                          "implication": str(o.get("implication") or "")[:300],
                          "concerns": [u.strip().strip("[]") for u in coerce_str_list(o.get("concerns_unit_ids"))
                                       if u.strip().strip("[]") in shown]})
    return notes


SENTENCE_ROLES = ("pattern", "mechanism", "distinction", "exception", "procedure", "relation", "detail", "attribute")
RELATIONAL_ROLES = ("pattern", "mechanism", "distinction", "exception", "procedure", "relation")

REWRITE_SCHEMA = {"type": "object", "properties": {
    "summary": {"type": "string"},
    "units": {"type": "array", "maxItems": MAX_REWRITE_UNITS, "items": {"type": "object", "properties": {
        "statement": {"type": "string"}, "role": {"type": "string", "enum": list(ROLES)},
        "scope": {"type": "string"}, "conditions": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "absorbs": {"type": "array", "items": {"type": "string"}, "description": "old unit ids this unit subsumes"},
        "from_notes": {"type": "array", "items": {"type": "integer"}},
        "addresses_gap": {"type": "boolean", "description": "true if this unit answers the Direction (the diagnosed gap)"},
        "policy_items": {"type": "array", "items": {"type": "integer"},
                         "description": "numbers of the representation-policy items (if a policy is given) this unit follows"}},
        "required": ["statement", "role"]}}},
    "required": ["units"]}

REWRITE_PROMPT = (
    "Revise the SOURCE MODEL of one region of a knowledge source. You are given the region's current units, the "
    "study notes gathered by inspecting and questioning this region against the raw source, and the source evidence "
    "those notes rest on. The objective: the same compact model, revised at the SAME level of abstraction so that it "
    "also carries what the notes established — not a fuller account of the source. The source evidence is there to "
    "verify and ground what you write; it is NOT material to summarise: never restate the source passage by passage, "
    "never copy or closely paraphrase its sentences, never take over its voice (no 'you', 'your', 'we'). Write "
    "statements ABOUT the source in the model's own register, each unit ONE reusable semantic commitment (a rule with "
    "its governing condition, a mechanism with its consequence, a procedure with its step order, a distinction, an "
    "exception to a shared rule, a dependency between entities) — not one source excerpt, and not a replay of "
    "implementation steps unless the sequence itself is what matters. Preserve every source-supported meaning of the "
    "old units unless a stronger statement subsumes it; merge fragments that are stages of one procedure or instances "
    "of one rule; split a unit that mixes unrelated things; drop nothing the evidence supports. Never write a negative "
    "claim (something does not exist, is not supported, cannot be done) unless the source states it explicitly. Do not "
    "add look-up values the source answers on its own and do not keep isolated look-up details (a phone number, an "
    "office hour, a form number, a single value) solely for completeness. Entity cards (attribute grids) are kept as "
    "they are and are shown only as context: do not rewrite them. For each unit give its role, its scope (an entity "
    "scope copied from the old units, or the region), the old unit ids it absorbs, and the note numbers it draws on. "
    "Names, identifiers, numbers and conditions exactly as in the source; statements at most {chars} characters; "
    "declarative.")

PRESERVE_SCHEMA = {"type": "object", "properties": {"not_represented": {"type": "array", "items": {"type": "integer"}}},
                   "required": ["not_represented"]}
PRESERVE_PROMPT = (
    "A region of a source model was rewritten. Below are the OLD units' meanings (numbered) and the NEW region. "
    "Return the numbers of old meanings that the new region no longer represents — a meaning counts as "
    "represented when the new region states it, entails it, or expresses it through a broader rule, procedure, "
    "or abstraction that covers it (merging, factoring and abstracting are fine; a lost condition, exception, "
    "value, entity or step is not).")


def _dup_pairs(units: list[SourceModelUnit]) -> int:
    toks = [(_content(u.statement)) for u in units]
    n = 0
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            if toks[i] and toks[j] and len(toks[i] & toks[j]) / max(1, min(len(toks[i]), len(toks[j]))) >= DUP_OVERLAP:
                n += 1
    return n


def structure_stats(units: list[SourceModelUnit]) -> dict:
    sent = [u for u in units if u.role in SENTENCE_ROLES]
    return {"units": len(units), "sentence_units": len(sent), "tokens": m_tokens(units),
            "relational": sum(1 for u in sent if u.role in RELATIONAL_ROLES),
            "isolated": sum(1 for u in sent if u.role in ("detail", "attribute")),
            "dup_pairs": _dup_pairs(sent)}


RELATION_PASS_NOTE = (
    "\nThis is a CROSS-ENTITY pass: the notes connect several entities of the region (one area constraining "
    "another, a shared rule with its instances and exceptions, a dependency or data flow between entities). "
    "Write ONLY such cross-entity units (mechanism / relation / pattern / distinction / exception); do not "
    "restate any single entity's own facts, which its own units already hold.")


def rewrite_region(block: dict, old: list[SourceModelUnit], notes: list[dict], evidence: dict[str, SourceElement],
                   llm: LLMClient, brief: str = "", relation_pass: bool = False, direction: list[str] | None = None,
                   policy: str = "", context_units: list[SourceModelUnit] | None = None, policy_n_items: int = 0) -> dict:
    """Phase B proposal for one region (a block, or one entity of it): old
    sentence-form units + cards (context) + notes + evidence -> new unit list
    (normalised). relation_pass=True: cross-entity units only. `policy`:
    representation policy appended to the prompt; `context_units`: units
    shown but not rewritable (task-learned units under recalibration);
    `policy_n_items`: numbered items the policy holds — a cited item outside
    1..n is dropped (writers fill the field even when there is nothing to
    cite: 62% of units under an item-less policy carried a citation).
    Returns {summary, units}."""
    cards = [u for u in old if u.role == "card"]
    sent = [u for u in old if u.role != "card" and u.level not in ("scaffold", "global")]
    notes_txt = "\n".join(f"({i + 1}) {n['statement']}" + (f" [implication: {n['implication']}]" if n.get("implication") else "")
                          + (f" <- {', '.join(n['support'][:3])}" if n.get("support") else "") for i, n in enumerate(notes)) or "(none)"
    out = llm.complete_json(
        system=REWRITE_PROMPT.format(chars=REWRITE_STATEMENT_CHARS) + (RELATION_PASS_NOTE if relation_pass else "")
        + (f"\n\nReading instruction for this source:\n{brief[:3000]}" if brief else "") + _policy_note(policy),
        user=(f"## Region: {block['name']} — {block.get('description', '')}\n\n## Current units of the region (rewrite these)\n"
              f"{render_units(sent)}\n\n## Entity cards of the region (context only, kept as they are)\n"
              f"{render_units(cards) if cards else '(none)'}\n\n"
              + ((f"## Task-learned units of the region (context only, kept as they are; do not restate them)\n"
                  f"{render_units(context_units)}\n\n") if context_units else "")
              + (("## Direction (a real task exposed that this region must make the following clear; this is a question to answer "
                  "from the evidence, NOT a source statement — never copy it into a unit)\n" + "\n".join(f"- {d}" for d in direction) + "\n\n")
                 if direction else "")
              + f"## Study notes\n{notes_txt}\n\n## Source evidence\n{render_evidence(evidence, 1500)}"),
        schema=REWRITE_SCHEMA, purpose="study_rewrite")
    old_ids = {u.unit_id for u in sent}
    units = []
    for nu in (out.get("units") or [])[:MAX_REWRITE_UNITS]:
        if not isinstance(nu, dict):
            continue
        stmt = str(nu.get("statement") or "").strip()[:REWRITE_STATEMENT_CHARS]
        if not stmt:
            continue
        role = nu.get("role") if nu.get("role") in ROLES and nu.get("role") != "card" else "mechanism"
        units.append({"statement": stmt, "role": role, "scope": str(nu.get("scope") or ""),
                      "conditions": coerce_str_list(nu.get("conditions"))[:4], "entities": coerce_str_list(nu.get("entities")),
                      "absorbs": [a.strip().strip("[]") for a in coerce_str_list(nu.get("absorbs")) if a.strip().strip("[]") in old_ids],
                      "from_notes": [int(i) for i in coerce_str_list(nu.get("from_notes")) if str(i).isdigit() and 1 <= int(i) <= len(notes)],
                      "addresses_gap": bool(nu.get("addresses_gap")),
                      "policy_items": [int(i) for i in coerce_str_list(nu.get("policy_items"))
                                       if str(i).isdigit() and 1 <= int(i) <= policy_n_items]})
    return {"summary": str(out.get("summary") or "")[:400], "units": units}


def preservation_check(old: list[SourceModelUnit], new_text: str, llm: LLMClient, batch: int = 25) -> set[str]:
    """Set-level semantic preservation: ids of old units whose meaning the new
    region no longer represents (those are carried over unchanged)."""
    lost: set[str] = set()
    for i in range(0, len(old), batch):
        chunk = old[i:i + batch]
        out = llm.complete_json(
            system=PRESERVE_PROMPT,
            user=("## Old meanings\n" + "\n".join(f"({j + 1}) {u.statement}" + (f" | conditions: {u.conditions}" if u.conditions else "")
                                                   for j, u in enumerate(chunk)) + f"\n\n## New region\n{new_text}"),
            schema=PRESERVE_SCHEMA, purpose="study_preserve")
        for k in coerce_str_list(out.get("not_represented")):
            if str(k).isdigit() and 1 <= int(k) <= len(chunk):
                lost.add(chunk[int(k) - 1].unit_id)
    return lost


def is_required(nu: dict) -> bool:
    """A proposed unit carries this rewrite's learning when it subsumes old
    units, rests on a study note / observation, or answers the diagnosed gap
    (provenance the writer reports; the class is decided here, not by it).
    Everything else is auxiliary."""
    return bool(nu.get("absorbs")) or bool(nu.get("from_notes")) or bool(nu.get("addresses_gap"))


def prepare_rewrite(model: list[SourceModelUnit], block: dict, proposal: dict, evidence: dict[str, SourceElement],
                    llm: LLMClient, step_id: str, growth: float = 0.0,
                    license: int | None = None, allowance: int | None = None) -> dict:
    """The LLM half of a rewrite commit (safe to run for several regions in
    parallel; the model is only read): batched grounding of the new units,
    narrowing retries for the failures, the claim-level gate, the persistence
    licence, and the set-level preservation check. Returns the telemetry
    dict plus `_grounded`, `_old`, `_lost` for `apply_rewrite`.

    Claim-level grounding gate: a REQUIRED unit (see `is_required`) must
    ground (one narrowing retry) or it is not written; an AUXILIARY unit that
    does not ground is simply dropped. The rewrite is refused only when no
    grounded unit is left, or when every required unit failed (the rewrite
    then does not carry its learning). No share-of-failures cutoff.

    Observation-bounded persistence (`license`): among grounded units that
    subsume nothing (pure additions), at most `license` are written —
    required ones first — so M grows in proportion to grounded new
    understanding, not to how much text was read. Reconstruction of old
    units (absorbing) is never limited by it."""
    old = [u for u in block["units"] if u.role != "card" and u.level not in ("scaffold", "global")]
    tele = {"block": block["name"], "applied": False, "reasons": [], "old_ids": [u.unit_id for u in old],
            "subsumed": [], "kept": [], "new_ids": [], "ungrounded": 0, "proposed": len(proposal["units"]),
            "_grounded": [], "_old": old, "_lost": set(), "_growth": growth, "_allowance": allowance}
    if not proposal["units"]:
        tele["reasons"].append("EMPTY_PROPOSAL"); return tele
    scopes, regions = _known_scopes(model)
    grounded: list[SourceModelUnit] = []
    flags: dict[str, dict] = {}     # unit id -> {required, pure_new}
    req_total = req_grounded = aux_dropped = 0
    first = ground_batch([nu["statement"] for nu in proposal["units"]], evidence, llm)
    for i, nu in enumerate(proposal["units"]):
        required = is_required(nu); req_total += required
        ok, anchors = first[i]
        stmt = nu["statement"]
        if not ok:      # one narrowing retry, judged alone
            stmt2 = narrow_to_evidence(stmt, evidence, llm)
            ok, anchors = ground(stmt2, evidence, llm) if stmt2 and stmt2 != stmt else (False, {})
            stmt = stmt2 if ok else ""
        if not stmt:
            tele["ungrounded"] += 1
            aux_dropped += not required
            continue
        req_grounded += required
        scope = scope_from_anchors(nu["scope"], anchors, scopes, regions)
        grounded.append(SourceModelUnit(
            unit_id=f"self_study:{block['name']}:{step_id}:{i + 1}", statement=stmt,
            kind="fact" if nu["role"] == "detail" else "model", scope=scope, conditions=nu["conditions"],
            role=nu["role"], level=unit_level(scope, regions), support_anchors=sorted(anchors),
            support_fingerprints=dict(anchors), origin="self_study", learned_from_task=step_id,
            support_mode="synthesized" if len(anchors) > 1 else "explicit", derived_from=nu["absorbs"],
            policy_items=nu.get("policy_items", [])))
        flags[grounded[-1].unit_id] = {"required": required, "pure_new": not nu["absorbs"]}
    tele.update({"required": req_total, "required_grounded": req_grounded, "auxiliary_dropped": aux_dropped})
    if not grounded or (req_total and not req_grounded):
        tele["reasons"].append("REWRITE_REFUSED_UNGROUNDED" if not grounded else "REQUIRED_UNGROUNDED"); return tele
    if license is not None:      # persistence licence: pure additions <= license, required ones first
        pure = [u for u in grounded if flags[u.unit_id]["pure_new"]]
        keep = sorted(pure, key=lambda u: not flags[u.unit_id]["required"])[:max(0, license)]
        over = [u for u in pure if u not in keep]
        tele["over_license"] = len(over)
        grounded = [u for u in grounded if u not in over]
        if not grounded:
            tele["reasons"].append("NOTHING_LICENSED"); return tele
    tele["_grounded"] = grounded
    tele["_notes_used"] = sum(1 for nu in proposal["units"] if nu["from_notes"])
    tele["_lost"] = preservation_check(old, render_units(grounded), llm) if old else set()
    return tele


def apply_rewrite(model: list[SourceModelUnit], tele: dict) -> dict:
    """The mutation half: structure-improvement gate, then atomic diff.
    Strips the private fields."""
    old, grounded, lost = tele.pop("_old"), tele.pop("_grounded"), tele.pop("_lost")
    growth, allowance = tele.pop("_growth", 0.0), tele.pop("_allowance", None)
    if allowance is None:
        allowance = GROWTH_ALLOWANCE      # resolved at call time (self-study default)
    if tele["reasons"]:
        return tele
    kept = [u for u in old if u.unit_id in lost]
    subsumed = [u for u in old if u.unit_id not in lost]
    before, after = structure_stats(old), structure_stats(grounded + kept)
    by_role: dict[str, dict[str, int]] = {}
    for u in old:      # writer coverage per role: subsumed = its meaning is represented by the rewrite
        r = by_role.setdefault(u.role or "?", {"subsumed": 0, "kept": 0})
        r["kept" if u.unit_id in lost else "subsumed"] += 1
    tele.update({"before": before, "after": after, "notes_used": tele.pop("_notes_used", 0),
                 "preserve_by_role": by_role})
    improved = [k for k, ok in (("dup_pairs_down", after["dup_pairs"] < before["dup_pairs"]),
                                ("isolated_down", after["isolated"] < before["isolated"]),
                                ("relational_up", after["relational"] > before["relational"]),
                                ("tokens_down", after["tokens"] < before["tokens"]),
                                ("notes_incorporated", tele["notes_used"] > 0)) if ok]
    if not improved:
        tele["reasons"].append("NOOP_NO_IMPROVEMENT"); return tele
    # growth cap per region: a rewrite may grow its region by at most `growth`x
    # (plus a small allowance for tiny regions), so growth is bounded uniformly
    # instead of first-come against one global budget
    if growth and after["tokens"] > max(before["tokens"] * growth, before["tokens"] + allowance):
        tele["reasons"].append("OVER_BUDGET"); return tele
    for u in subsumed:
        model.remove(u)
    model.extend(grounded)
    tele.update({"applied": True, "reasons": improved, "subsumed": [u.unit_id for u in subsumed],
                 "kept": [u.unit_id for u in kept], "new_ids": [u.unit_id for u in grounded],
                 "new_units_text": {u.unit_id: {"statement": u.statement, "role": u.role, "level": u.level, "scope": u.scope,
                                                "conditions": u.conditions, "anchors": u.support_anchors, "absorbs": u.derived_from,
                                                "policy_items": u.policy_items}
                                    for u in grounded}})
    return tele


