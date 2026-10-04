"""Source elements: the stable, fingerprinted units of a workspace (files,
document sections, code symbols). Every unit of M is anchored to element ids,
retrieval indexes the section/symbol elements, and the grounding gate reads
their text. Enumeration is deterministic: the same workspace always yields the
same ids in the same order.
"""
from __future__ import annotations

import ast
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from sourcelearn.core.schema import SourceElement
from sourcelearn.core.workspace import Workspace

MAX_FILES_PER_NODE = 60
DOC_SUFFIXES = (".md", ".rst", ".txt")
EXCERPT_CHARS = 1200


def tokens(text: str) -> list[str]:
    """Lowercase content tokens; shared by all lexical recall/overlap gates."""
    return re.findall(r"[a-z0-9]{2,}", text.lower())


def fingerprint(text: str) -> str:
    """Whitespace-insensitive content hash: formatting-only edits are free."""
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "untitled"


def unique_id(base: str, taken: set[str]) -> str:
    eid, n = base, 2
    while eid in taken:
        eid = f"{base}-{n}"
        n += 1
    taken.add(eid)
    return eid


def parse_symbols(text: str) -> list[dict]:
    """Top-level and class-nested defs: [{kind, name, line, end_line}]."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []

    def visit(body, prefix=""):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(n, ast.ClassDef) else "function"
                name = prefix + n.name
                out.append({"kind": kind, "name": name, "line": n.lineno,
                            "end_line": getattr(n, "end_lineno", n.lineno)})
                if isinstance(n, ast.ClassDef):
                    visit(n.body, prefix=name + ".")

    visit(tree.body)
    return out


def split_sections(text: str, file: str) -> list[dict]:
    """Split a markdown/rst/plain file into heading-delimited sections."""
    sections, current = [], {"heading": "(top)", "file": file, "start": 1, "lines": []}
    for i, line in enumerate(text.splitlines(), 1):
        if re.match(r"^#{1,6} +\S", line):
            if current["lines"]:
                sections.append(current)
            current = {"heading": line.lstrip("# ").strip(), "file": file,
                       "start": i, "lines": []}
        current["lines"].append(line)
    sections.append(current)
    for s in sections:
        s["end"] = s["start"] + len(s["lines"]) - 1
        s["text"] = "\n".join(s.pop("lines"))
    return sections


def _is_test_file(f: str) -> bool:
    name = Path(f).name
    return (name.startswith("test_") or name.endswith("_test.py")
            or name == "conftest.py" or "tests/" in f or f.startswith("tests"))


def _classify_files(ws: Workspace) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"code": [], "test": [], "doc": []}
    for f in ws.list_files():
        if f.endswith(".py"):
            groups["test" if _is_test_file(f) else "code"].append(f)
        elif f.endswith(DOC_SUFFIXES):
            groups["doc"].append(f)
    return groups


def _group_by_top_dir(files: list[str]) -> dict[str, list[str]]:
    """Group by top-level directory; oversized groups split one level deeper."""
    by_dir: dict[str, list[str]] = defaultdict(list)
    for f in files:
        parts = f.split("/")
        by_dir[parts[0] if len(parts) > 1 else "(root)"].append(f)
    out: dict[str, list[str]] = {}
    for d, fs in by_dir.items():
        if len(fs) <= MAX_FILES_PER_NODE or d == "(root)":
            out[d] = fs
            continue
        deeper: dict[str, list[str]] = defaultdict(list)
        for f in fs:
            parts = f.split("/")
            deeper["/".join(parts[:2]) if len(parts) > 2 else d].append(f)
        out.update(deeper)
    return out


def _file_elements(ws: Workspace, node_id: str,
                   files: list[str]) -> dict[str, SourceElement]:
    """Elements of one file group; ids are made unique within the group."""
    out: dict[str, SourceElement] = {}
    taken: set[str] = set()

    def add(kind: str, base_id: str, name: str, file: str, text: str,
            start: int | None = None, end: int | None = None) -> None:
        eid = unique_id(base_id, taken)
        out[eid] = SourceElement(
            element_id=eid, node_id=node_id, kind=kind,  # type: ignore[arg-type]
            name=name, file=file, start_line=start, end_line=end,
            fingerprint=fingerprint(text), excerpt=text[:EXCERPT_CHARS])

    if node_id == "history:git":
        head = ws.version.commit_sha or "unversioned"
        add("log", "(git)#log", "git history", "(git)", head)
        return out

    for f in files:
        text = ws.read_text(f)
        lines = text.count("\n") + 1
        add("file", f, f, f, text, 1, lines)
        if f.endswith(DOC_SUFFIXES):
            for s in split_sections(text, f):
                add("section", f"{f}#{slug(s['heading'])}", s["heading"], f,
                    s["text"], s["start"], s["end"])
        elif f.endswith(".py"):
            file_lines = text.splitlines()
            for sym in parse_symbols(text):
                body = "\n".join(file_lines[sym["line"] - 1:sym["end_line"]])
                add("symbol", f"{f}#{sym['name']}", sym["name"], f, body,
                    sym["line"], sym["end_line"])
    return out


def source_elements(ws: Workspace) -> dict[str, SourceElement]:
    """All elements of a workspace, grouped as code packages (by top-level
    directory), the test suite, documentation (one group per top-level
    directory) and, for git checkouts, the history."""
    groups = _classify_files(ws)
    node_files: list[tuple[str, list[str]]] = [
        (f"code:{d}", fs) for d, fs in _group_by_top_dir(groups["code"]).items()]
    if groups["test"]:
        node_files.append(("test:all", groups["test"]))
    for d, fs in _group_by_top_dir(groups["doc"]).items():
        node_files.append(("doc:all" if d == "(root)" else f"doc:{d}", fs))
    if ws.version.commit_sha:
        node_files.append(("history:git", []))

    elements: dict[str, SourceElement] = {}
    for node_id, files in node_files:
        elements.update(_file_elements(ws, node_id, files))
    return elements
