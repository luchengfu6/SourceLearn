"""Draft source modeling: a single-pass, lean-at-write-time compiler from source to source model.

Three leanness principles (no post-hoc compression):
- WRITE-ONCE: the LLM produces content only when first reading source
  text; every later operation (folding, promotion, dedup) only counts,
  moves or deletes — it never rewrites words.
- STRUCTURE carries L1: an entity's attributes live as table rows on
  ONE entity card, not as sentences. Admission is judged once, at
  extraction: does this change an answer/boundary/exception/decision?
- COUNTERS carry L2: values consistent across entities fold into a
  region prototype mechanically (majority counting); deviating cells
  stay on the entity's card — they ARE the exceptions. Support and
  violation counts are counted, never claimed.

Hierarchy: L1 entity cards / exceptions / details; L2 region
prototypes (mechanical) + mechanisms (gated) + the source-conditioned
abstraction pass; the scaffold = the structural map, which joins the
reasoning context as a map.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sourcelearn.core.elements import source_elements
from sourcelearn.core.llm import LLMClient, load_project_env
from sourcelearn.core.schema import SourceElement, coerce_str_list
from sourcelearn.draft.ids import unique_ids
from sourcelearn.core.workspace import Workspace
from sourcelearn.source_model import SourceModelUnit

GROUP_CHARS = 60000
CATALOG_DOCS = 12
EXCERPT_CHARS = 900
FOLD_MIN = 3        # a column folds into the prototype at >= this many agreeing entities
FOLD_SHARE = 0.6    # ... and >= this share of entities that have the column
WINDOW_CHARS = 16000  # entity-centred read: full text of the entity + its closure
READ_BUDGET = {"n": 6,           # max mechanisms+distinctions+details per entity (0 = off)
               "block_tokens": 0,  # SOFT block budget: told to the reader, never enforced
               "entities": 0}      # entities sharing that block budget


# ---------------------------------------------------------------- adapter
# Everything downstream operates on (entity, attribute, value, anchor) rows
# and (sentence, anchors) — source-agnostic. The adapter is the only place
# that knows what a region / family / entity IS for a given source kind.
#   docs: region = top dir, family = doc-name prefix (one product = one
#         entity), junk = internal tool ids like open_bank_account_4821
#   code: region = top dir (re-root the workspace so it is the app/package),
#         family = module file, entities = the module's top-level symbols
ADAPTER = {"name": "docs"}
# An optional one-page reading instruction
# (sourcelearn.draft.brief) appended to the per-family READ prompt only.
# Nothing else in the compiler changes.
BRIEF = {"text": ""}


def set_adapter(name: str) -> None:
    ADAPTER["name"] = name


def set_brief(text: str) -> None:
    BRIEF["text"] = text.strip()


# ------------------------------------------- entity-centred reading (code)
# The brief's MAIN OBJECTS name the knowledge entities; the compiler reads
# each entity's own code plus the same-module closure it reaches (and one
# hop into region modules whose classes it constructs) — "knowledge entity
# -> collect evidence -> read", not "file -> read".
import ast as _ast


def _module_index(text: str) -> dict:
    tree = _ast.parse(text)
    funcs, classes, consts, imports = {}, {}, {}, {}
    for n in tree.body:
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            funcs[n.name] = n
        elif isinstance(n, _ast.ClassDef):
            classes[n.name] = n
        elif isinstance(n, (_ast.Assign, _ast.AnnAssign)):
            for t in (n.targets if isinstance(n, _ast.Assign) else [n.target]):
                if isinstance(t, _ast.Name):
                    consts[t.id] = n
        elif isinstance(n, _ast.ImportFrom) and n.module:
            for a in n.names:
                imports[a.asname or a.name] = n.module
    return {"funcs": funcs, "classes": classes, "consts": consts, "imports": imports}


def _names_used(node) -> list[str]:
    seen, out = set(), []
    for n in _ast.walk(node):
        if isinstance(n, _ast.Name) and n.id not in seen:
            seen.add(n.id); out.append(n.id)
    return out


def build_window(ws: Workspace, file: str, name: str,
                 elements: dict[str, SourceElement], region: str = ""
                 ) -> list[tuple[str, str, str]]:
    """[(element_id, label, full_text)]: the entity (function or class),
    the same-module functions/classes/constants it reaches transitively,
    then one hop into other modules of the region that define classes it
    imports. Full text, never excerpt-truncated; capped at WINDOW_CHARS."""
    text = ws.read_text(file)
    try:
        idx = _module_index(text)
    except SyntaxError:
        return []
    if name not in idx["funcs"] and name not in idx["classes"]:
        return []
    pieces: list[tuple[str, str, str]] = []
    size = 0

    def eid(f: str, n: str | None) -> str:
        cand = f"{f}#{n}" if n else f
        return cand if cand in elements else f

    def add(f: str, n: str | None, src: str) -> bool:
        nonlocal size
        if not src or size + len(src) > WINDOW_CHARS:
            return False
        pieces.append((eid(f, n), f"{f}:{n or '(module)'}", src)); size += len(src)
        return True

    queue, done = [name], set()
    while queue:
        cur = queue.pop(0)
        if cur in done:
            continue
        done.add(cur)
        node = idx["funcs"].get(cur) or idx["classes"].get(cur)
        if node is None:
            continue
        if not add(file, cur, _ast.get_source_segment(text, node) or ""):
            break
        for used in _names_used(node):
            if used in idx["funcs"] or used in idx["classes"]:
                queue.append(used)
            elif used in idx["consts"] and used not in done:
                done.add(used)
                add(file, None, _ast.get_source_segment(text, idx["consts"][used]) or "")
    wanted = {u for n in done for u in _names_used(
        idx["funcs"].get(n) or idx["classes"].get(n) or _ast.Module(body=[], type_ignores=[]))
        if u in idx["imports"]}
    if wanted:
        for f2 in ws.list_files(suffixes=(".py",)):
            if f2 == file or (region and not f2.startswith(region.rstrip("/") + "/")):
                continue
            try:
                idx2 = _module_index(ws.read_text(f2))
            except SyntaxError:
                continue
            for w in sorted(wanted):
                if w in idx2["classes"]:
                    add(f2, w, _ast.get_source_segment(ws.read_text(f2), idx2["classes"][w]) or "")
    return pieces


def all_entities(elements: dict[str, SourceElement], region: str,
                 files: set[str] | None = None) -> list[tuple[str, str]]:
    """Every public top-level symbol of a code region as a knowledge entity
    (no brief needed); test modules are skipped. [(file, name)], sorted."""
    hits: set[tuple[str, str]] = set()
    for e in elements.values():
        if e.kind != "symbol" or "." in e.name or e.name.startswith("_"):
            continue
        if region and not e.file.startswith(region.rstrip("/") + "/"):
            continue
        if "test" in e.file.lower() or (files is not None and e.file not in files):
            continue
        hits.add((e.file, e.name))
    return sorted(hits)


def family_entities(key: str, group: list[SourceElement]) -> list[str]:
    """Candidate entity names a family's read may attribute rows to.
    docs: the family itself. code: top-level class/function names."""
    if ADAPTER["name"] != "code":
        return [key.split("/")[-1]]
    names: list[str] = []
    for e in group:
        top = (e.name or "").split(".")[0].strip()
        if top and top not in names and not top.startswith("_"):
            names.append(top)
    return names[:40] or [Path(key).stem]


def clamp_entity(raw: str, candidates: list[str], default: str) -> str:
    """LLM-written entity labels fragment stores (one entity -> five
    spellings); clamp onto the mechanical candidate list."""
    r = (raw or "").strip()
    if r in candidates:
        return r
    low = {c.lower(): c for c in candidates}
    if r.lower() in low:
        return low[r.lower()]
    head = r.split(".")[-1].split("(")[0].strip().lower()
    hits = [c for c in candidates if c.lower() == head
            or c.lower().startswith(head) or head.startswith(c.lower())]
    return hits[0] if len(hits) == 1 and head else default


def unit_ent(region: str, ent: str) -> str:
    """Entity tag used in unit ids: docs keep bare doc stems (unique per
    source); code qualifies by region (Case exists in many apps)."""
    return ent if ADAPTER["name"] != "code" else f"{region}.{ent}"


def sibling_key(e: SourceElement) -> str:
    d = e.file.rsplit("/", 1)[0] if "/" in e.file else "."
    stem = Path(e.file).stem
    if e.file.endswith((".md", ".rst", ".txt")):
        prefix = re.sub(r"_\d+$", "", stem)
        return f"{d}/{prefix}"
    return e.file


def sibling_groups(elements: dict[str, SourceElement],
                   region: str = "", files: set[str] | None = None
                   ) -> dict[str, list[SourceElement]]:
    groups: dict[str, list[SourceElement]] = defaultdict(list)
    # a family's size is judged over the WHOLE source, not over the block being
    # compiled: a 47-document procedure family is still 47 distinct documents
    # when a block holds 10 of them (otherwise a block compile collapses them
    # into one entity and reads almost nothing)
    fam_files: dict[str, set[str]] = defaultdict(set)
    for e in elements.values():
        if e.kind in ("section", "symbol"):
            fam_files[sibling_key(e)].add(e.file)
    for e in elements.values():
        if e.kind not in ("section", "symbol"):
            continue
        if region and not e.file.startswith(region.rstrip("/") + "/"):
            continue
        if files is not None and e.file not in files:
            continue
        groups[sibling_key(e)].append(e)
    split: dict[str, list[SourceElement]] = {}
    for k, v in groups.items():
        if len(fam_files[k]) > CATALOG_DOCS:
            for e in v:
                split.setdefault(e.file, []).append(e)
        else:
            split[k] = v
    return {k: sorted(v, key=lambda x: (x.file, x.start_line or 0))
            for k, v in sorted(split.items())}


def region_of(key: str) -> str:
    return key.split("/", 1)[0]


def norm_attr(name: str) -> str:
    n = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return re.sub(r"_+", "_", n)


def norm_val(v: str) -> str:
    return re.sub(r"[\s,$]", "", str(v).lower())


def resolve_in(anchor: str, ids: dict[str, SourceElement]) -> str | None:
    a = str(anchor or "").strip()
    if not a:
        return None
    if a in ids:
        return a
    cands = [i for i in ids if i.startswith(a) or i.endswith(a) or a in i]
    if len(cands) == 1:
        return cands[0]
    if "#" in a:
        f, _, frag = a.partition("#")
        cands = [i for i in ids if i.startswith(f) and frag[:25] in i]
        if len(cands) == 1:
            return cands[0]
    return None


# ------------------------------------------------- stage 2: scaffold (map)
MAP_PROMPT = (
    "You are forming the structural map of a knowledge source. For the "
    "region below say what it is about, its main entities/areas, and "
    "which families relate to or depend on each other. Structural only; "
    "no rules, no values.")


def source_map(groups: dict[str, list[SourceElement]], region: str,
               llm: LLMClient) -> list[SourceModelUnit]:
    digest = "\n".join(
        f"- {k.split('/')[-1]} ({len(v)} sections, "
        f"{len({e.file for e in v})} files)" for k, v in groups.items())
    out = llm.complete_json(
        system=MAP_PROMPT,
        user=f"Region: {region}\n\n{digest}",
        schema={"type": "object", "properties": {
            "about": {"type": "string"},
            "relations": {"type": "array", "items": {"type": "string"}}},
            "required": ["about"]},
        purpose="induce_map")
    ids, fps = [], {}
    for v in groups.values():
        for e in v:
            ids.append(e.element_id); fps[e.element_id] = e.fingerprint
    units = [SourceModelUnit(
        unit_id=f"map:{region}", statement=str(out.get("about", "")).strip(),
        kind="model", scope=region, role="map", level="scaffold",
        support_anchors=ids, support_fingerprints=fps,
        support_mode="synthesized")]
    for i, r in enumerate(coerce_str_list(out.get("relations"))):
        if len(r.split()) < 4:
            continue  # identifier echo, not a structural sentence
        units.append(SourceModelUnit(
            unit_id=f"map:{region}:rel{i + 1}", statement=r, kind="model",
            scope=region, role="map", level="scaffold",
            support_anchors=ids, support_fingerprints=fps,
            support_mode="synthesized"))
    return units


# ------------------------------------- stage 3: per-family read (write once)
INDUCE_PROMPT = (
    "You are reading everything the source says about ONE entity "
    "(product / module / topic) side by side, to build its profile — "
    "not to summarize it. Fill the slots:\n"
    "- attributes: decision-relevant properties as name/value pairs "
    "(short snake_case names another entity of the same kind would also "
    "have); one element id each. Skip properties that decide nothing.\n"
    "- mechanisms: how something works or a procedure's logic, when "
    "worth understanding; cite >=1 element ids.\n"
    "- distinctions: things a reader must not confuse.\n"
    "- decisive_details: the few remaining facts that change a decision "
    "or behavior — one element id each. Rarity alone is NOT a reason; "
    "it must plausibly change an answer, boundary, exception, or "
    "decision. Skip anything merely retrievable.\n"
    "Prefer fewer, structural entries.\n"
    "element_id / element_ids must be copied VERBATIM from the [bracketed] "
    "ids in the material; never invent labels.\n"
    "The raw source stays available at use time: do NOT store facts that a "
    "reader can simply look up. Keep constraints, mechanisms, shared rules, "
    "exceptions and distinctions.")

BUDGET_NOTE = ("\nBudget: at most {n} entries in total across mechanisms, "
               "distinctions and decisive_details for this entity; choose "
               "the ones a user would otherwise get wrong.")
SOFT_BUDGET_NOTE = ("\nCompactness (soft target, you decide what earns its place): the "
                    "whole model of this block should stay around {block:,} tokens "
                    "across {n} entities, i.e. about {per} tokens for this entity — its "
                    "card plus roughly {k} short statements. Spend them on reusable "
                    "structure (rules, conditions, exceptions, distinctions, procedures); "
                    "leave look-up values to the raw source.")


def budget_note() -> str:
    """The per-entity budget line for the read prompt: a soft token target
    derived from the block budget when one is set, else the entry cap."""
    bt, n = READ_BUDGET.get("block_tokens", 0), READ_BUDGET.get("entities", 0)
    hard = BUDGET_NOTE.format(n=READ_BUDGET["n"]) if READ_BUDGET["n"] else ""
    if bt and n:
        per = max(120, bt // n)
        k = max(2, min(READ_BUDGET["n"] or 8, per // 45))
        # the hard per-entity cap stays (it is the only control the reader
        # obeys); the soft block target is added as context, not a substitute
        return hard + SOFT_BUDGET_NOTE.format(block=bt, n=n, per=per, k=k)
    return hard

INDUCE_SCHEMA = {
    "type": "object",
    "properties": {
        "attributes": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "value": {"type": "string"},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "element_id": {"type": "string"}},
            "required": ["name", "value", "element_id"]}},
        "mechanisms": {"type": "array", "items": {"type": "object", "properties": {
            "statement": {"type": "string"},
            "element_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["statement", "element_ids"]}},
        "distinctions": {"type": "array", "items": {"type": "object", "properties": {
            "statement": {"type": "string"},
            "element_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["statement", "element_ids"]}},
        "decisive_details": {"type": "array", "items": {"type": "object", "properties": {
            "statement": {"type": "string"}, "element_id": {"type": "string"}},
            "required": ["statement", "element_id"]}},
    },
    "required": ["attributes"],
}

GATE_PROMPT = (
    "Verify each numbered claim against the source excerpts it cites. "
    "Return the numbers of claims NOT supported: asserting behavior, "
    "order, conditions or values the excerpts do not establish; or "
    "derived from knowledge OUTSIDE the source (language or library "
    "semantics, platform conventions) rather than from the excerpts — a "
    "definite rule needs definite source support. Claims that merely "
    "restate a visible signature or docstring are fine to keep here.")


def _batches(group: list[SourceElement]) -> list[list[SourceElement]]:
    batches, cur, size = [], [], 0
    by_file: dict[str, list[SourceElement]] = defaultdict(list)
    for e in group:
        by_file[e.file].append(e)
    for els in by_file.values():
        n = sum(min(len(e.excerpt), EXCERPT_CHARS) + 80 for e in els)
        if cur and size + n > GROUP_CHARS:
            batches.append(cur); cur, size = [], 0
        cur.extend(els); size += n
    if cur:
        batches.append(cur)
    return batches


INDUCE_MULTI_NOTE = (
    "\nThis family is one MODULE defining several entities (listed). "
    "Every attribute / mechanism / distinction / detail must name the "
    "entity it belongs to in its 'entity' field, using a listed name; "
    "module-wide facts use the module name.")


def _blank(key: str) -> dict:
    return {"key": key, "cells": {}, "mechanisms": [], "distinctions": [],
            "details": [], "stats": {"cells": 0, "cell_conflicts": 0,
                                     "mech_dropped": 0, "claims_dropped": 0,
                                     "rejected": 0}}


def _read(key: str, group: list[SourceElement], llm: LLMClient,
          gate: LLMClient, entities: list[str],
          window: list[tuple[str, str, str]] | None = None,
          elements: dict[str, SourceElement] | None = None) -> dict[str, dict]:
    """One LLM read -> per-entity scratch: cells (deduped by counter),
    mechanisms, distinctions, details. Nothing is rewritten later.
    Two evidence modes: a FAMILY (list of elements, excerpt-truncated) or
    an entity WINDOW ([(element_id, label, full_text)], entity-centred).
    Every semantic claim of every kind passes ONE batched grounding gate."""
    region = key.split("/")[0]
    multi = len(entities) > 1
    default = Path(key).stem if multi else entities[0]
    fams: dict[str, dict] = {}

    def fam(raw_ent) -> dict:
        ent = clamp_entity(str(raw_ent or ""), entities, default) if multi \
            else entities[0]
        if ent not in fams:
            fams[ent] = _blank(f"{region}/{ent}")
        return fams[ent]

    def anchor_for(raw_id, raw_ent, bids: dict, text: str = "") -> str | None:
        """Resolve a cited id; when the model invented a label (e.g.
        'del_op', 'sq.error_shape'):
        1. code rows name their entity -> that entity's own element;
        2. otherwise -> the batch element whose text overlaps the row's
           own text most (values are usually quoted verbatim)."""
        a = resolve_in(raw_id, bids)
        if a is not None:
            return a
        if multi:
            ent = clamp_entity(str(raw_ent or ""), entities, "")
            hits = [i for i in bids if i.endswith(f"#{ent}") or f"#{ent}." in i] if ent else []
            if hits:
                stats_fallback[0] += 1
                return sorted(hits, key=len)[0]
        if text:
            t = set(re.findall(r"[a-z0-9$%.]{2,}", text.lower()))
            best = max(bids, key=lambda i: len(t & set(re.findall(
                r"[a-z0-9$%.]{2,}", btext[i].lower()))), default=None)
            if best is not None and t & set(re.findall(r"[a-z0-9$%.]{2,}", btext[best].lower())):
                stats_fallback[1] += 1
                return best
        return None

    stats_fallback = [0, 0]
    schema = json.loads(json.dumps(INDUCE_SCHEMA))
    if multi:
        for k in ("attributes", "mechanisms", "distinctions", "decisive_details"):
            schema["properties"][k]["items"]["properties"]["entity"] = {"type": "string"}
    # evidence batches: [(bids, btext, prompt_text)]
    batches: list[tuple[dict, dict, str]] = []
    if window is not None:
        els = elements or {}
        bids = {eid: els[eid] for eid, _l, _t in window if eid in els}
        btext = {eid: t for eid, _l, t in window}
        text = "\n\n".join(f"[{eid}]  # {label}\n{t}" for eid, label, t in window
                           if eid in bids)
        batches.append((bids, btext, text))
    else:
        for batch in _batches(group):
            bids = {e.element_id: e for e in batch}
            btext = {e.element_id: e.excerpt[:EXCERPT_CHARS] for e in batch}
            batches.append((bids, btext, "\n\n".join(
                f"[{e.element_id}]\n{btext[e.element_id]}" for e in batch)))
    for bids, btext, text in batches:
        head = (f"Module: {key.split('/')[-1]}\nEntities: {', '.join(entities)}"
                if multi else f"Entity: {entities[0]}")
        out = llm.complete_json(
            system=INDUCE_PROMPT + (INDUCE_MULTI_NOTE if multi else "")
            + budget_note()
            + (f"\n\nReading instruction for this source (follow it when "
               f"deciding what to extract and what to skip):\n{BRIEF['text']}"
               if BRIEF["text"] else ""),
            user=f"{head}\n\n{text}", schema=schema, purpose="induce_family")
        # collect every claim first; gate them together; then admit
        claims: list[dict] = []   # {kind, text, anchors{}, ent, raw}
        for at in out.get("attributes") or []:
            if not isinstance(at, dict):
                continue
            name = norm_attr(at.get("name") or "")
            val = str(at.get("value") or "").strip()
            a = anchor_for(at.get("element_id"), at.get("entity"), bids,
                           f"{at.get('name', '')} {val}")
            if not name or not val or a is None:
                fam(at.get("entity"))["stats"]["rejected"] += 1
                continue
            claims.append({"kind": "attribute", "text": f"{name}: {val}",
                           "anchors": {a: bids[a].fingerprint}, "ent": at.get("entity"),
                           "name": name, "val": val,
                           "conditions": coerce_str_list(at.get("conditions"))})
        for kind, key_ in (("mechanism", "mechanisms"), ("distinction", "distinctions")):
            for m in out.get(key_) or []:
                if not isinstance(m, dict):
                    continue
                stmt = str(m.get("statement") or "").strip()
                anchors = [x for x in (anchor_for(a, m.get("entity"), bids, stmt)
                                       for a in coerce_str_list(m.get("element_ids"))) if x]
                if stmt and anchors:
                    claims.append({"kind": kind, "text": stmt,
                                   "anchors": {a: bids[a].fingerprint for a in anchors},
                                   "ent": m.get("entity")})
        for x in out.get("decisive_details") or []:
            if not isinstance(x, dict):
                continue
            stmt = str(x.get("statement") or "").strip()
            a = anchor_for(x.get("element_id"), x.get("entity"), bids, stmt)
            if stmt and a:
                claims.append({"kind": "detail", "text": stmt,
                               "anchors": {a: bids[a].fingerprint}, "ent": x.get("entity")})
        bad: set[int] = set()
        if claims:  # ONE grounding gate for all claim kinds
            # the gate sees the FULL evidence once (a 500-char head per
            # anchor showed only signatures/docstrings and made the gate
            # reject code-supported contracts), then the numbered claims
            cited = {a for c in claims for a in c["anchors"] if a in btext}
            ev = ("## Source evidence\n" + "\n\n".join(
                f"[{a}]\n{btext[a][:6000]}" for a in sorted(cited))
                + "\n\n## Claims (each lists the ids it rests on)\n" + "\n".join(
                f"({i + 1}) [{c['kind']}] {c['text']}  <- "
                + ", ".join(a for a in c["anchors"] if a in btext)
                for i, c in enumerate(claims)))
            g = gate.complete_json(
                system=GATE_PROMPT, user=ev,
                schema={"type": "object", "properties": {
                    "unsupported": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["unsupported"]},
                purpose="induce_gate")
            bad = {int(i) for i in g.get("unsupported") or []
                   if isinstance(i, (int, float, str)) and str(i).isdigit()}
        for i, c in enumerate(claims):
            f = fam(c["ent"])
            stats = f["stats"]
            if i + 1 in bad:
                stats["mech_dropped" if c["kind"] == "mechanism" else "claims_dropped"] += 1
                continue
            if c["kind"] == "attribute":
                cells, name, val = f["cells"], c["name"], c["val"]
                cur = cells.get(name)
                if cur is None:
                    cells[name] = {"value": val, "anchors": dict(c["anchors"]),
                                   "support": 1, "conditions": c["conditions"]}
                    stats["cells"] += 1
                elif norm_val(cur["value"]) == norm_val(val):
                    cur["anchors"].update(c["anchors"])  # confirmation: count, don't write
                    cur["support"] += 1
                else:
                    cells[f"{name}#{stats['cell_conflicts'] + 2}"] = {
                        "value": val, "anchors": dict(c["anchors"]),
                        "support": 1, "conditions": ["CONFLICTS with " + name]}
                    stats["cell_conflicts"] += 1
            elif c["kind"] == "mechanism":
                prev = next((x for x in f["mechanisms"]
                             if _overlap(x["statement"], c["text"]) >= 0.7), None)
                if prev:
                    prev["support"] += 1
                    prev["anchors"].update(c["anchors"])
                else:
                    f["mechanisms"].append({"statement": c["text"],
                                            "anchors": dict(c["anchors"]), "support": 1})
            elif c["kind"] == "distinction":
                f["distinctions"].append({"statement": c["text"], "anchors": dict(c["anchors"])})
            else:
                f["details"].append({"statement": c["text"], "anchors": dict(c["anchors"])})
    if not fams:
        fams[default] = _blank(f"{region}/{default}")
    for f in fams.values():
        f["stats"]["anchor_from_entity"] = stats_fallback[0]
        f["stats"]["anchor_from_content"] = stats_fallback[1]
    return fams


def read_entity(region: str, name: str, file: str, ws: Workspace,
                elements: dict[str, SourceElement], llm: LLMClient,
                gate: LLMClient) -> dict:
    """Entity-centred read: one knowledge entity (a public callable or
    class), evidence = its window. Returns one scratch dict."""
    window = build_window(ws, file, name, elements, region)
    if not window:
        f = _blank(f"{region}/{name}"); f["key"] = f"{region}/{name}"; return f
    fams = _read(f"{region}/{name}", [], llm, gate, [name], window=window,
                 elements=elements)
    f = next(iter(fams.values()))
    f["key"] = f"{region}/{name}"
    return f


def read_family_multi(key: str, group: list[SourceElement], llm: LLMClient,
                      gate: LLMClient) -> list[dict]:
    """Adapter-aware read: a code module yields one scratch per entity."""
    return list(_read(key, group, llm, gate, family_entities(key, group)).values())


def _overlap(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z0-9$%.]+", a.lower())), \
        set(re.findall(r"[a-z0-9$%.]+", b.lower()))
    return len(ta & tb) / max(1, len(ta | tb))


# ------------------- stage 4a: column-name normalization (mechanical)
GENERIC_TOKENS = {"frequency", "amount", "value", "rate_type", "type"}


def _col_sig(name: str) -> frozenset:
    toks = {t[:-1] if t.endswith("s") and len(t) > 3 else t
            for t in name.split("_")}
    return frozenset(t for t in toks if t not in GENERIC_TOKENS)


def canonical_cols(families: list[dict]) -> dict[str, str]:
    """Merge synonym column names (plural drift, generic suffixes):
    same token signature -> the most frequent (then shortest) name wins.
    'overdraft_fee(s)' merge; 'fee' vs 'fee_waiver' do NOT."""
    count: Counter = Counter()
    for f in families:
        for n in f["cells"]:
            if "#" not in n:
                count[n] += 1
    by_sig: dict[frozenset, list[str]] = defaultdict(list)
    for n in count:
        by_sig[_col_sig(n)].append(n)
    rename: dict[str, str] = {}
    for names in by_sig.values():
        if len(names) > 1:
            keep = sorted(names, key=lambda n: (-count[n], len(n)))[0]
            for n in names:
                if n != keep:
                    rename[n] = keep
    return rename


def apply_renames(families: list[dict], rename: dict[str, str]) -> int:
    merged = 0
    for f in families:
        for old, new in rename.items():
            if old in f["cells"]:
                c = f["cells"].pop(old)
                cur = f["cells"].get(new)
                if cur is None:
                    f["cells"][new] = c
                elif norm_val(cur["value"]) == norm_val(c["value"]):
                    cur["anchors"].update(c["anchors"])
                    cur["support"] += c["support"]
                else:
                    f["cells"][f"{new}#x{merged + 2}"] = c
                merged += 1
    return merged


# code adapter: attribute names that only echo what a caller already sees
# in the tool signature / description (mechanical junk rule, like the docs
# identifier rule)
CODE_JUNK_ATTR = re.compile(
    r"^(function_name|tool_name|name|description|one_liner|tool_description|"
    r"docstring|summary|purpose|module(_path|_name)?|file|accepted_args|"
    r"accepted_params|accepted_arguments|parameters|params|arguments|"
    r"optional_(params|args|arguments|fields|parameters)|"
    r"required_(params|args|arguments|fields|parameters|inputs)|"
    r"input_param_style|signature|return_type|returns)$")


# --------------------------- stage 4b: region close (mechanical folding)
def region_close(families: list[dict], region: str
                 ) -> tuple[list[SourceModelUnit], dict]:
    """Constant columns fold into region prototype units (counted, not
    written by an LLM); deviating cells STAY on their entity cards —
    they are the exceptions. Returns prototype units + per-entity cards
    + mechanisms/distinctions/details as units."""
    units: list[SourceModelUnit] = []
    stats = {"folded_cols": 0, "proto_absorbed_cells": 0,
             "mech_merged": 0, "junk_cards": 0, "junk_cells": 0}
    # 4c: mechanisms shared across families collapse into ONE region
    # mechanism (support summed, anchors unioned) — counting, no rewrite
    pool: list[dict] = []
    for f in families:
        for m in f["mechanisms"]:
            m["_ent"] = f["key"].split("/")[-1]
            pool.append(m)
    clusters: list[list[dict]] = []
    for m in pool:
        home = next((c for c in clusters
                     if _overlap(c[0]["statement"], m["statement"]) >= 0.6), None)
        (home.append(m) if home else clusters.append([m]))
    region_mechs, kept_per_family = [], defaultdict(list)
    for c in clusters:
        ents = {m["_ent"] for m in c}
        if len(ents) >= 2:
            rep = max(c, key=lambda m: m["support"])
            anchors = {}
            for m in c:
                anchors.update(m["anchors"])
            region_mechs.append({"statement": rep["statement"],
                                 "anchors": anchors,
                                 "support": sum(m["support"] for m in c),
                                 "ents": sorted(ents)})
            stats["mech_merged"] += len(c) - 1
        else:
            kept_per_family[c[0]["_ent"]].extend(c)
    for i, m in enumerate(region_mechs):
        units.append(SourceModelUnit(
            unit_id=f"mech:{region}:{i + 1}", statement=m["statement"],
            kind="model", scope=region, role="mechanism", level="regional",
            conditions=[f"support: {m['support']}",
                        f"shared by: {', '.join(m['ents'])[:200]}"],
            support_anchors=sorted(m["anchors"]),
            support_fingerprints=m["anchors"], support_mode="synthesized"))
    col: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for f in families:
        for name, c in f["cells"].items():
            if "#" not in name:
                col[name].append((f["key"], c))
    folded: dict[str, str] = {}
    for name, owners in col.items():
        vals = Counter(norm_val(c["value"]) for _, c in owners)
        top_val, top_n = vals.most_common(1)[0]
        if top_n >= FOLD_MIN and top_n / len(owners) >= FOLD_SHARE:
            keep = [(k, c) for k, c in owners if norm_val(c["value"]) == top_val]
            exceptions = [k.split("/")[-1] for k, c in owners
                          if norm_val(c["value"]) != top_val]
            anchors, fps = [], {}
            for _, c in keep:
                for a, fp in c["anchors"].items():
                    anchors.append(a); fps[a] = fp
            val = keep[0][1]["value"]
            exc = f" (exceptions: {', '.join(exceptions)})" if exceptions else ""
            units.append(SourceModelUnit(
                unit_id=f"l2:{region}:proto:{name}",
                statement=f"{name} = {val} for {top_n}/{len(owners)} "
                          f"entities in {region}{exc}",
                kind="fact", scope=region, role="pattern", level="regional",
                conditions=[f"support: {top_n}", f"of: {len(owners)}"],
                support_anchors=sorted(set(anchors)), support_fingerprints=fps,
                support_mode="synthesized"))
            folded[name] = top_val
            stats["folded_cols"] += 1
            stats["proto_absorbed_cells"] += top_n
    for f in families:
        ent = f["key"].split("/")[-1]
        scoped = f"{region}/{ent}"
        uent = unit_ent(region, ent)
        rows, anchors, fps = [], [], {}
        for name, c in sorted(f["cells"].items()):
            base = name.split("#")[0]
            if folded.get(base) == norm_val(c["value"]):
                continue  # absorbed by the prototype
            if ADAPTER["name"] == "docs" and re.fullmatch(
                    r"[a-z0-9_]+_\d{3,4}", c["value"].strip()):
                stats["junk_cells"] += 1   # internal identifier, not an attribute
                continue
            if ADAPTER["name"] == "code" and CODE_JUNK_ATTR.match(base):
                stats["junk_cells"] += 1   # signature/docstring echo the caller already sees
                continue
            cond = f" [{'; '.join(c['conditions'])}]" if c.get("conditions") else ""
            rows.append(f"{name}: {c['value']}{cond}")
            for a, fp in c["anchors"].items():
                anchors.append(a); fps[a] = fp
        if ADAPTER["name"] == "docs" and len(rows) == 1 \
                and not re.search(r"[\d$%]", rows[0]):
            stats["junk_cards"] += 1       # one digit-less cell says nothing
            rows = []
        if rows:
            units.append(SourceModelUnit(
                unit_id=f"card:{uent}", statement="; ".join(rows),
                kind="fact", scope=scoped, role="card", level="local",
                support_anchors=sorted(set(anchors)), support_fingerprints=fps,
                support_mode="synthesized" if len(anchors) > 1 else "explicit"))
        for i, m in enumerate(kept_per_family.get(ent, [])):
            units.append(SourceModelUnit(
                unit_id=f"mech:{uent}:{i + 1}", statement=m["statement"],
                kind="model", scope=scoped, role="mechanism", level="regional",
                conditions=[f"support: {m['support']}"],
                support_anchors=sorted(m["anchors"]),
                support_fingerprints=m["anchors"],
                support_mode="synthesized" if len(m["anchors"]) > 1 else "explicit"))
        for i, d in enumerate(f["distinctions"]):
            units.append(SourceModelUnit(
                unit_id=f"dist:{uent}:{i + 1}", statement=d["statement"],
                kind="model", scope=scoped, role="distinction", level="local",
                support_anchors=sorted(d["anchors"]),
                support_fingerprints=d["anchors"]))
        for i, d in enumerate(f["details"]):
            units.append(SourceModelUnit(
                unit_id=f"det:{uent}:{i + 1}", statement=d["statement"],
                kind="fact", scope=scoped, role="detail", level="local",
                support_anchors=sorted(d["anchors"]),
                support_fingerprints=d["anchors"], support_mode="explicit"))
    return units, stats


# a vague contrast without direction ('X varies') is not a rule
VAGUE = re.compile(r"\b(vary|varies|variation|differ|differs|different)\b", re.I)
DIRECTION = re.compile(
    r"\b(higher|lower|rise|rises|increase|increases|decrease|decreases|"
    r"more|less|larger|smaller|only|if|when|unless|except|while|whereas|"
    r"above|below|free|waive[sd]?|premium|0%)\b", re.I)


# ------------------ stage 5: source-conditioned abstraction pass (L2)
ABSTRACT_PROMPT = (
    "Below are the L1 units of ONE source region (each with an id and its "
    "entity). Induce the region's SHARED RULES (a contract, rule or "
    "behavior that several entities share), DISTINCTIONS (two entities or "
    "operations that behave differently where a user would expect the "
    "same) and EXCEPTIONS (one entity deviating from a shared rule). "
    "Every item must cite the L1 unit ids it rests on: a shared rule needs "
    "units from >= 2 entities; a distinction cites both sides; an "
    "exception cites the rule side and the deviating unit. If a shared "
    "rule fully subsumes some cited L1 units, list them under 'replaces' "
    "so the model stays compact. State each item so a unit could "
    "contradict it; no 'varies/differs' without direction.")

ABSTRACT_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "kind": {"type": "string",
                     "enum": ["shared_rule", "distinction", "exception"]},
            "statement": {"type": "string"},
            "from_unit_ids": {"type": "array", "items": {"type": "string"}},
            "replaces": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["kind", "statement", "from_unit_ids"]}}},
    "required": ["items"],
}


def abstraction_pass(units: list[SourceModelUnit], region: str,
                     llm: LLMClient) -> tuple[list[SourceModelUnit], list[str], dict]:
    """L1 (+ brief, via the read prompt's perspective) -> shared rules /
    distinctions / exceptions. Mechanical admission: cited ids must exist;
    shared rules need >= 2 entities; 'replaces' is honoured only for ids
    the item itself cites. Returns (new units, replaced unit ids, stats)."""
    l1 = [u for u in units if u.level == "local"
          and (u.scope or "").split("/")[0] == region]
    if len(l1) < 4:
        return [], [], {"abstract": 0, "rejected": 0, "replaced": 0}
    by_id = {u.unit_id: u for u in l1}
    ent = lambda u: (u.scope or "").split("/")[-1]  # noqa: E731
    text = "\n".join(f"[{u.unit_id}] ({ent(u)}) {u.statement[:300]}" for u in l1)
    out = llm.complete_json(
        system=ABSTRACT_PROMPT
        + (f"\n\nReading instruction for this source:\n{BRIEF['text'][:3000]}"
           if BRIEF["text"] else ""),
        user=f"Region: {region}\n\n{text}"[:120000],
        schema=ABSTRACT_SCHEMA, purpose="induce_abstract")
    new, replaced, st = [], [], {"abstract": 0, "rejected": 0, "replaced": 0}
    n = 0
    for it in out.get("items") or []:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind") or "")
        stmt = str(it.get("statement") or "").strip()
        srcs = [by_id[i] for i in coerce_str_list(it.get("from_unit_ids")) if i in by_id]
        ents = {ent(u) for u in srcs}
        ok = bool(stmt) and srcs and (
            (kind == "shared_rule" and len(ents) >= 2)
            or (kind == "distinction" and len(srcs) >= 2)
            or (kind == "exception" and len(srcs) >= 1))
        if not ok or (VAGUE.search(stmt) and not DIRECTION.search(stmt)):
            st["rejected"] += 1
            continue
        n += 1
        fps = {a: u.support_fingerprints[a] for u in srcs for a in u.support_anchors
               if a in u.support_fingerprints}
        role = {"shared_rule": "pattern", "distinction": "distinction",
                "exception": "exception"}[kind]
        new.append(SourceModelUnit(
            unit_id=f"l2:{region}:abs:{n}", statement=stmt, kind="model",
            scope=region, role=role, level="regional",
            conditions=[f"applies to: {', '.join(sorted(ents))[:200]}"],
            derived_from=[u.unit_id for u in srcs],
            support_anchors=sorted(fps), support_fingerprints=fps,
            support_mode="synthesized"))
        st["abstract"] += 1
        if kind == "shared_rule":
            cited = {u.unit_id for u in srcs}
            for r in coerce_str_list(it.get("replaces")):
                if r in cited and r not in replaced:
                    replaced.append(r); st["replaced"] += 1
    return new, replaced, st


# ------------------------------------------------- stage 6 + reporting
def dedup(units: list[SourceModelUnit]) -> tuple[list[SourceModelUnit], int]:
    keep, seen, dropped = [], [], 0
    for u in sorted(units, key=lambda x: -len(x.support_anchors)):
        if any(u.role == r and _overlap(u.statement, s) >= 0.8
               for s, r in seen):
            dropped += 1
            continue
        seen.append((u.statement, u.role)); keep.append(u)
    order = {id(u): i for i, u in enumerate(units)}
    keep.sort(key=lambda u: order.get(id(u), 0))
    return keep, dropped


def report(units: list[SourceModelUnit], elements: dict, region: str) -> str:
    in_region = [e for e in elements.values()
                 if e.kind in ("section", "symbol")
                 and (not region or e.file.startswith(region.rstrip("/") + "/"))]
    reason = [u for u in units if u.level != "scaffold"]
    covered = {a for u in reason for a in u.support_anchors}
    m_chars = sum(len(u.statement) for u in reason)
    src_chars = sum(len(e.excerpt) for e in in_region)
    by = defaultdict(list)
    for u in units:
        by[u.role].append(u)
    lines = [f"# Induced M0 — {region or 'all'}",
             f"M_reason: {len(reason)} units, {m_chars:,} chars "
             f"(~{m_chars // 4:,} tokens) | roles: "
             f"{ {r: len(v) for r, v in by.items()} } | covered "
             f"{len(covered & {e.element_id for e in in_region})}/{len(in_region)}"
             + (f" | M/source {m_chars / max(1, src_chars):.0%}" if in_region else ""),
             ""]
    for role in ("map", "pattern", "mechanism", "distinction", "card",
                 "exception", "detail"):
        if by.get(role):
            lines.append(f"## {role} ({len(by[role])})")
            for u in by[role]:
                lv = {"scaffold": "S", "global": "G", "regional": "r",
                      "local": "l"}.get(u.level or "", "?")
                cond = f"  [{u.conditions[0]}]" if u.conditions else ""
                lines.append(f"- {lv} ({len(u.support_anchors)}a; {u.scope}) "
                             f"{u.statement[:400]}{cond}")
            lines.append("")
    return "\n".join(lines)


def run_stage(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--region", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default="docs", choices=["docs", "code"],
                   help="what a family/entity is: docs (one doc family = "
                        "one entity) or code (module = family, top-level "
                        "symbols = entities)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel family reads (output order is unchanged)")
    p.add_argument("--brief", default="",
                   help="path to a Source Compilation Instruction (brief.md), appended to the read prompt")
    p.add_argument("--files", default="",
                   help="JSON list of source files to compile (a partition block)")
    p.add_argument("--token-budget", type=int, default=0,
                   help="SOFT token budget for this block: given to the reader as a "
                        "compactness target (0 = off)")
    args = p.parse_args(argv)
    load_project_env()
    set_adapter(args.adapter)
    if args.brief:
        set_brief(Path(args.brief).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    elements = source_elements(Workspace(args.workspace))
    from sourcelearn.core.trace import Trace
    trace = Trace(out, "trace")
    llm = LLMClient(model=args.model, trace=trace)
    gate = LLMClient(model=args.model, trace=trace)     # admission gate: the same model
    file_set = set(json.loads(Path(args.files).read_text())) if args.files else None
    groups = sibling_groups(elements, args.region, file_set)
    READ_BUDGET["block_tokens"], READ_BUDGET["entities"] = args.token_budget, len(groups)
    by_region: dict[str, dict[str, list[SourceElement]]] = defaultdict(dict)
    for k, v in groups.items():
        by_region[region_of(k)][k] = v
    print(f"{len(groups)} families in {len(by_region)} region(s)",
          file=sys.stderr)
    units: list[SourceModelUnit] = []
    all_stats: dict = {}
    ws = Workspace(args.workspace)
    from concurrent.futures import ThreadPoolExecutor
    for reg, rgroups in by_region.items():
        units += source_map(rgroups, reg, llm)
        fams = []
        # entity-centred mode (code): every public top-level symbol outside tests is a
        # knowledge entity -> read each entity's window; families are not read
        ents = all_entities(elements, reg, file_set) if ADAPTER["name"] == "code" else []
        if ents:
            print(f"  [entities {reg}] {len(ents)} public symbols (families skipped)",
                  file=sys.stderr)
            def _safe_read(fn):   # one bad model reply must not lose the whole compile
                try:
                    return read_entity(reg, fn[1], fn[0], ws, elements, llm, gate)
                except Exception as e:  # noqa: BLE001
                    f = _blank(f"{reg}/{fn[1]}"); f["key"] = f"{reg}/{fn[1]}"
                    f["stats"]["error"] = str(e)[:100]
                    return f
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(_safe_read, ents))
            for (f_, n_), fr in zip(ents, results):
                fams.append(fr)
                print(f"  [read {n_[:40]}] {fr['stats']}", file=sys.stderr)
                all_stats[f"{reg}/{n_}"] = fr["stats"]
        else:
            keys = list(rgroups)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(
                    lambda k: read_family_multi(k, rgroups[k], llm, gate), keys))
            for k, fs in zip(keys, results):  # family order preserved
                fams.extend(fs)
                st = {s_: sum(f["stats"][s_] for f in fs) for s_ in fs[0]["stats"]}
                print(f"  [read {k.split('/')[-1][:40]}] {len(fs)} entit"
                      f"{'y' if len(fs) == 1 else 'ies'} {st}", file=sys.stderr)
                all_stats[k] = st
        rename = canonical_cols(fams)
        merged = apply_renames(fams, rename)
        cunits, cst = region_close(fams, reg)   # counted prototypes + cards
        cst["cols_renamed"] = merged
        aunits, replaced, ast_ = [], [], {}
        try:
            aunits, replaced, ast_ = abstraction_pass(cunits, reg, llm)
        except Exception as e:  # noqa: BLE001 — degrade, don't lose the run
            print(f"  [abstract {reg}] ERROR {e}", file=sys.stderr)
            ast_ = {"error": str(e)[:120]}
        cunits = [u for u in cunits if u.unit_id not in set(replaced)]
        units += cunits + aunits
        all_stats[f"_close:{reg}"] = cst
        all_stats[f"_abstract:{reg}"] = ast_
        print(f"  [close {reg}] {cst} | abstract {ast_}", file=sys.stderr)
    units, dropped = dedup(units)
    all_stats["_dedup"] = dropped
    compiled_tok = sum(len(u.statement) // 4 for u in units)
    all_stats["_budget"] = {"soft_tokens": args.token_budget, "entities": len(groups),
                            "compiled_tokens": compiled_tok}
    print(f"  [budget] soft {args.token_budget} -> compiled ~{compiled_tok} tokens "
          f"over {len(groups)} entities", file=sys.stderr)
    print(f"  [dedup] dropped {dropped}", file=sys.stderr)
    trace.close()
    rows = [u.model_dump() for u in units]
    ren = unique_ids(rows)                 # ids compose (role, entity, local index) and can repeat; relabel only
    if ren:
        print(f"  [ids] {sum(ren.values())} duplicate unit ids suffixed (#2, #3 ...)", file=sys.stderr)
    (out / "m0_induced.json").write_text(json.dumps(rows, indent=1, default=str))
    rep = report(units, elements, args.region)
    (out / "report.md").write_text(
        rep + "\n\n## stats\n" + json.dumps(all_stats, indent=1))
    print(rep.split("\n\n")[0])
    return 0
