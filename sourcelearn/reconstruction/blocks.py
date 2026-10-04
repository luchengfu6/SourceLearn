"""Blocks: the unit of routing (Retrieve_M) and of study-target selection.

A partition (sourcelearn.draft.partition) names blocks by files; units join the
block that holds most of their anchor files. Without a partition, blocks
are the top-level directories of the anchor files (mechanical), so every
consumer sees the same interface whether the source was partitioned or not.
"""
from __future__ import annotations

import re

from sourcelearn.core.workspace import Workspace
from sourcelearn.reconstruction.defaults import AUTO_BLOCK_MAX_FILES, AUTO_BLOCK_MAX_TOKENS, EVIDENCE_PER_ELEMENT, EVIDENCE_TOTAL
from sourcelearn.reconstruction.update import m_tokens
from sourcelearn.source_model import SourceModelUnit, _family_of


def assign_units_to_blocks(model: list[SourceModelUnit], partition: dict) -> dict[str, list[SourceModelUnit]]:
    """Units belong to the block holding most of their anchor files; map
    units follow their region's first block."""
    file_of = {f: b["name"] for b in partition["blocks"] for f in b["files"]}
    byb: dict[str, list[SourceModelUnit]] = {b["name"]: [] for b in partition["blocks"]}
    for u in model:
        names = [n for n in (file_of.get(a.split("#")[0]) for a in u.support_anchors) if n]
        if names:
            byb[max(set(names), key=names.count)].append(u)
        elif u.role == "map":
            reg = (u.scope or "").split("/")[0]
            for b in partition["blocks"]:
                if b["files"] and b["files"][0].split("/")[0] == reg:
                    byb[b["name"]].append(u)
                    break
    return byb




def _split_dirs(files: set[str], depth: int) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for f in files:
        parts = f.split("/")
        groups.setdefault("/".join(parts[:depth]) if len(parts) > depth else "/".join(parts[:-1]) or ".", set()).add(f)
    return groups




def auto_partition(model: list[SourceModelUnit]) -> dict | None:
    """No partition given: mechanical blocks = directories of the anchor files
    (top level; a directory with more than AUTO_BLOCK_MAX_FILES files or more
    than AUTO_BLOCK_MAX_TOKENS of units splits into its sub-directories, and a
    flat directory that is still too large splits into consecutive file
    chunks), so a large code package yields blocks a 24k budget can route.
    None when everything is one block."""
    tok_of_file: dict[str, int] = {}
    for u in model:
        files = [a.split("#")[0] for a in u.support_anchors]
        if files:
            tok_of_file[files[0]] = tok_of_file.get(files[0], 0) + len(u.statement) // 4
    all_files = {a.split("#")[0] for u in model for a in u.support_anchors}
    tokens = lambda fs: sum(tok_of_file.get(f, 0) for f in fs)  # noqa: E731
    too_big = lambda fs: len(fs) > AUTO_BLOCK_MAX_FILES or tokens(fs) > AUTO_BLOCK_MAX_TOKENS  # noqa: E731
    groups, depth = _split_dirs(all_files, 1), 1
    while depth < 4:
        big = {k: v for k, v in groups.items() if too_big(v)}
        if not big:
            break
        depth += 1
        for k, v in big.items():
            sub = _split_dirs(v, depth)
            if len(sub) > 1:
                groups.pop(k); groups.update(sub)
    for k, v in list(groups.items()):          # flat directories still over the token cap: consecutive file chunks
        if tokens(v) > AUTO_BLOCK_MAX_TOKENS and len(v) > 1:
            groups.pop(k)
            chunk, size, i = set(), 0, 1
            for f in sorted(v):
                if chunk and size + tok_of_file.get(f, 0) > AUTO_BLOCK_MAX_TOKENS:
                    groups[f"{k}/part{i}"] = chunk; chunk, size, i = set(), 0, i + 1
                chunk.add(f); size += tok_of_file.get(f, 0)
            groups[f"{k}/part{i}"] = chunk
    if len(groups) < 2:
        return None
    blocks = []
    for reg, fs in sorted(groups.items()):
        blocks.append({"name": re.sub(r"[^a-z0-9]+", "_", reg.lower()).strip("_") or "root",
                       "description": f"Everything under {reg}/ ({len(fs)} files).",
                       "files": sorted(fs), "families": sorted({_family_of(f) for f in fs})})
    return {"blocks": blocks, "load_budget": 0}


_VERSION = re.compile(r"_v\d+$")


def entity_key_of_file(file: str) -> str:
    """Entity of a source file: its family with a version suffix (doc_v2)
    stripped, so versions of one document are one entity."""
    return _VERSION.sub("", _family_of(file))


def entity_key(u: SourceModelUnit) -> str:
    files = [a.split("#")[0] for a in u.support_anchors]
    if files:
        keys = [entity_key_of_file(f) for f in files]
        return max(set(keys), key=keys.count)
    return _VERSION.sub("", u.scope or "")


def bounded_evidence(items, per_element: int = EVIDENCE_PER_ELEMENT, total: int = EVIDENCE_TOTAL) -> dict:
    """{element_id: element} from an ordered (id, element) iterable, each
    excerpt cut to `per_element` chars, stopping at `total` chars — so a
    rewrite and its grounding fit one context."""
    out, spent = {}, 0
    for eid, e in items:
        if eid in out:
            continue
        text = e.excerpt[:per_element]
        if spent + len(text) > total:
            break
        out[eid] = e.model_copy(update={"excerpt": text}); spent += len(text)
    return out


def block_view(partition: dict, model: list[SourceModelUnit], ws: Workspace | None = None) -> list[dict]:
    """Blocks with their units and size statistics: source chars, entity
    (family) count, M tokens — what target selection weighs."""
    byb = assign_units_to_blocks(model, partition)
    out = []
    for b in partition["blocks"]:
        units = byb[b["name"]]
        chars = b.get("chars") or (sum(len(ws.read_text(f)) for f in b["files"]) if ws is not None else 0)
        fams = b.get("families") or sorted({_family_of(f) for f in b["files"]})
        out.append({**b, "units": units, "chars": chars, "families": fams,
                    "entities": len(fams), "m_tokens": m_tokens(units)})
    return out
