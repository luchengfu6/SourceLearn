"""Source inspection: a learning target (self-study question or task
residual) -> raw evidence E.

    inspect_source(query, ...) -> ({element_id: SourceElement}, tier)

Tiers, first with >= MIN_HITS hits wins: hint-restricted files (a task's
evidence hint) -> M-scoped files (the entities the target names) -> global.
Retrieval is whatever retriever the caller uses at answer time, so learning
looks at the source through the same lens as use.

Code adapter: a retrieved symbol is expanded into its entity-centred window
(`sourcelearn.draft.induce.build_window`: the symbol, the same-module helpers and
constants it reaches, one hop into region classes it constructs) — a study
of a code repository must read the mechanism, not a 2k-char excerpt.
"""
from __future__ import annotations

from pathlib import Path

from sourcelearn.draft.induce import build_window
from sourcelearn.core.schema import SourceElement
from sourcelearn.core.workspace import Workspace

from sourcelearn.reconstruction.defaults import CODE_EXPAND_HITS, CODE_WINDOW_CHARS, INSPECT_CAND_MULT as CAND_MULT, INSPECT_MIN_HITS as MIN_HITS, STUDY_READ_CHARS  # noqa: E402


def hint_files(hints: list[str], all_files: list[str]) -> set[str]:
    out = set()
    for h in hints:
        h0 = h.split("#")[0]
        for f in all_files:
            if f == h0 or f.endswith("/" + h0) or Path(f).stem == Path(h0).stem or h0 in f:
                out.add(f)
    return out


def expand_code_windows(evidence: dict[str, SourceElement], ws: Workspace,
                        elements: dict[str, SourceElement]) -> dict[str, SourceElement]:
    """Replace the excerpt of top-level symbol hits by their full window and
    add the reached pieces as evidence (element ids of the workspace)."""
    out = dict(evidence)
    spent, expanded = 0, 0
    for eid, e in list(evidence.items()):
        if e.kind != "symbol" or "." in e.name or expanded >= CODE_EXPAND_HITS:
            continue
        region = e.file.split("/")[0] if "/" in e.file else ""
        window = build_window(ws, e.file, e.name, elements, region)
        if not window:
            continue
        expanded += 1
        for weid, _label, text in window:
            if spent + len(text) > CODE_WINDOW_CHARS:
                break
            base = elements.get(weid) or e
            out[weid] = base.model_copy(update={"excerpt": text})
            spent += len(text)
    return out


def inspect_source(query: str, retriever, all_files: list[str], k: int,
                   hint: list[str] | None = None, scoped: set[str] | None = None,
                   ws: Workspace | None = None, elements: dict[str, SourceElement] | None = None
                   ) -> tuple[dict[str, SourceElement], str]:
    """Hint-restricted -> M-scoped -> global; the first tier with >= MIN_HITS
    hits wins. `hint` = file/doc/anchor hints (task feedback), `scoped` =
    files the target's entities live in (from M)."""
    cands = retriever.retrieve(query, CAND_MULT * k)
    tiers = [("hint", hint_files(hint or [], all_files)), ("scoped", scoped or set())]
    hits, tier = None, "global"
    for name, files in tiers:
        if files:
            hit = [e for e in cands if e.file in files][:k]
            if len(hit) >= MIN_HITS:
                hits, tier = hit, name
                break
    if hits is None:
        hits = cands[:k]
    evidence = {e.element_id: e for e in hits}
    if ws is not None and elements:
        evidence = expand_code_windows(evidence, ws, elements)
    return evidence, tier


def full_text_evidence(files: list[str], ws: Workspace, elements: dict[str, SourceElement],
                       cap_chars: int = STUDY_READ_CHARS) -> list[dict[str, SourceElement]]:
    """Every section / symbol of these files with its FULL text (compile-time
    excerpts are bounded), split into chunks of at most cap_chars so a chunk
    fits one reading. Returns [{element_id: element}] in file order."""
    chunks: list[dict[str, SourceElement]] = [{}]
    size = 0
    for f in files:
        lines = ws.read_text(f).splitlines()
        els = sorted((e for e in elements.values() if e.file == f and e.kind in ("section", "symbol")),
                     key=lambda e: e.start_line or 0)
        for e in els:
            text = "\n".join(lines[(e.start_line or 1) - 1:e.end_line]) if e.end_line else e.excerpt
            text = text or e.excerpt
            if size + len(text) > cap_chars and chunks[-1]:
                chunks.append({}); size = 0
            chunks[-1][e.element_id] = e.model_copy(update={"excerpt": text[:cap_chars]})
            size += len(text)
    return [c for c in chunks if c]


def render_evidence(evidence: dict[str, SourceElement], cap: int) -> str:
    """Every excerpt in full: an excerpt longer than `cap` is shown in
    consecutive parts under the same id (truncation once hid the tail of a
    long function from the observer and the grounder, which then denied or
    refused what the tail established). Total size is the caller's budget."""
    return "\n\n".join(paged(eid, e, cap) for eid, e in evidence.items())


def paged(eid: str, e: SourceElement, cap: int) -> str:
    text = e.excerpt
    if len(text) <= cap:
        return f"[{eid}] ({e.file})\n{text}"
    parts = [text[i:i + cap] for i in range(0, len(text), cap)]
    return "\n\n".join(f"[{eid}] ({e.file}) part {k + 1}/{len(parts)}\n{p}" for k, p in enumerate(parts))
