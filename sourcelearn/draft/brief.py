"""Stage 1: a one-page Source Compilation Instruction ("brief") from a skim.

Not a summary of the source and not a compilation plan: it answers
"how should this source be READ when compiling it into a compact,
reusable source model — what to focus on, what to usually ignore".
The skim is mechanical: directory tree, public interfaces / main
objects, a few representative excerpts. The brief is saved for human
audit; nothing is compiled here.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

from sourcelearn.core.llm import LLMClient
from sourcelearn.core.workspace import Workspace

SKIM_CHARS = 12000
SAMPLE_FILES = 3          # representative files shown with content
SAMPLE_CHARS = 1500       # per sample


def skim(ws: Workspace, root: str) -> str:
    files = sorted(f for f in ws.list_files() if f.startswith(root.rstrip("/") + "/")
                   or root in ("", "."))
    py = [f for f in files if f.endswith(".py")]
    docs = [f for f in files if f.endswith((".md", ".rst", ".txt"))]
    parts = [f"# Source root: {root or '.'}  ({len(files)} files: {len(py)} .py, {len(docs)} docs)"]
    # 1. tree (dirs with counts)
    dirs = Counter(f.rsplit("/", 1)[0] if "/" in f else "." for f in files)
    parts.append("## Directory tree\n" + "\n".join(f"- {d}/ ({n} files)" for d, n in sorted(dirs.items())))
    # 2. public interfaces / main objects
    if py:
        lines = ["## Modules and their top-level definitions (signature — docstring first line)"]
        for f in py:
            try:
                tree = ast.parse(ws.read_text(f))
            except SyntaxError:
                continue
            defs = []
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = (ast.get_docstring(n) or "").strip().split("\n")[0][:90]
                    if isinstance(n, ast.ClassDef):
                        sig = f"class {n.name}"
                    else:
                        args = [a.arg for a in n.args.args][:8]
                        sig = f"def {n.name}({', '.join(args)}{', ...' if len(n.args.args) > 8 else ''})"
                    deco = [getattr(d, "id", getattr(getattr(d, "func", None), "id", "")) for d in n.decorator_list]
                    defs.append(f"    {sig}" + (f"  @{','.join(x for x in deco if x)}" if any(deco) else "")
                                + (f" — {doc}" if doc else ""))
                elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets):
                    defs.append(f"    {', '.join(t.id for t in n.targets)} = <module constant>")
            lines.append(f"- {f}\n" + "\n".join(defs[:25]) + ("\n    ..." if len(defs) > 25 else ""))
        parts.append("\n".join(lines))
    if docs:
        fams = Counter(re.sub(r"_\d+\.md$", "", f.rsplit("/", 1)[-1]) for f in docs)
        lines = ["## Document families (name prefix -> number of docs)"]
        lines += [f"- {k} ({v})" for k, v in sorted(fams.items())[:60]]
        if len(fams) > 60:
            lines.append(f"- ... {len(fams) - 60} more families")
        parts.append("\n".join(lines))
    # 3. representative content: spread across the source
    pick = files[:: max(1, len(files) // SAMPLE_FILES)][:SAMPLE_FILES]
    parts.append("## Representative excerpts\n" + "\n\n".join(
        f"### {f}\n{ws.read_text(f)[:SAMPLE_CHARS]}" for f in pick))
    text = "\n\n".join(parts)
    return text[:SKIM_CHARS]


BRIEF_PROMPT = (
    "You are about to compile a knowledge source into a compact, reusable "
    "SOURCE MODEL that an assistant or agent will keep in context while "
    "using the source. Before reading it in detail, skim the material below "
    "and write a one-page READING INSTRUCTION for the compiler — not a "
    "summary of the content, not a plan: guidance on how to read.\n"
    "Write five sections, about 500-1000 tokens total:\n"
    "0. USER SCENARIOS — first put yourself in the position of whoever will "
    "use this source (infer who that is from the material itself). List 5-8 "
    "typical requests they would bring to it, as they would phrase them. "
    "Requests only, no answers. Everything below must follow from these "
    "scenarios.\n"
    "1. SOURCE PURPOSE — what this source is and what a user or agent "
    "typically uses it for (be concrete).\n"
    "2. MAIN OBJECTS / INTERFACES — what the knowledge should be organised "
    "around (the things a user refers to or calls), and how they are "
    "grouped.\n"
    "3. WHAT TO FOCUS ON — 5 to 8 concrete reading questions the compiler "
    "can answer directly while reading each part (e.g. 'what input does "
    "X reject', 'which products share rule Y, which are exceptions'); tag "
    "each with the scenario(s) it serves.\n"
    "4. USUALLY IGNORE — 3 to 5 kinds of content that normally should not "
    "enter the model, named concretely.\n"
    "Avoid abstract phrasing such as 'decision-relevant information' or "
    "'important details'; every line must be specific enough to act on. "
    "Do not aim for complete coverage; capture the main usage logic.")


def run_stage(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ws = Workspace(args.workspace)
    sk = skim(ws, "")
    (out / "skim.md").write_text(sk)
    from sourcelearn.core.trace import Trace
    trace = Trace(out, "trace")
    llm = LLMClient(model=args.model, trace=trace)
    res = llm.chat([{"role": "system", "content": BRIEF_PROMPT},
                    {"role": "user", "content": sk}], purpose="source_brief")
    brief = str(res.get("content") or "").strip()
    trace.close()
    (out / "brief.md").write_text(brief + "\n")
    (out / "meta.json").write_text(json.dumps({
        "workspace": args.workspace, "model": args.model,
        "skim_chars": len(sk), "brief_chars": len(brief),
        "brief_tokens_approx": len(brief) // 4}, indent=1))
    print(f"skim {len(sk):,} chars -> brief {len(brief):,} chars (~{len(brief) // 4} tok)",
          file=sys.stderr)
    print(brief)
    return 0
