"""Partition a source into content blocks BEFORE compiling.

One source -> several source models. A skim of the source (directories,
document families with sizes, and for large single-family directories the
per-document titles) is read once; an LLM proposes content-coherent blocks,
each with a one-sentence description that later serves as the ROUTING
index (which block(s) to load for a question). Every file is assigned to
exactly one block (mechanically enforced; leftovers form a `<dir>:rest`
block). Each block then goes through the unchanged compile pipeline
(`sourcelearn.draft.induce --files ... --token-budget ...`).

Budgets: TOTAL_BUDGET bounds the sum of all block models; each block's budget
is its raw-size share of the total, capped at BLOCK_CAP; the per-question load
budget (M_BUDGET) bounds how much M a router may put in context.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from sourcelearn.core.llm import LLMClient
from sourcelearn.core.schema import coerce_str_list
from sourcelearn.reconstruction.defaults import M_BUDGET
from sourcelearn.core.trace import Trace
from sourcelearn.core.workspace import Workspace

DOC_EXT = (".md", ".rst", ".txt")
SINGLE_FAMILY_TITLES = 60     # list per-doc titles when a dir is one big family
FAMILY_SPLIT_DOCS = (5, 15)   # such a family is split into operational areas of this many docs
MIN_BLOCK_BUDGET = 1200

PARTITION_SCHEMA = {
    "type": "object",
    "properties": {"blocks": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "members": {"type": "array", "items": {"type": "string"}},
            "why": {"type": "string"},
        },
        "required": ["name", "description", "members"],
    }}},
    "required": ["blocks"],
}

PARTITION_PROMPT = (
    "You are splitting a knowledge source into CONTENT BLOCKS, each of which "
    "will be compiled into its own compact source model and loaded only when "
    "a question concerns it. Read the digest and propose blocks such that:\n"
    "- a block is coherent by CONTENT (one product line, one operational "
    "area, one set of shared rules), not by size alone;\n"
    "- a question typically needs ONE block, sometimes two; things that are "
    "always consulted together belong in the same block;\n"
    "- a directory that is one large family of many documents (its per-document "
    "titles are listed) MUST be split by those titles into operational areas — "
    "e.g. opening, closing, card servicing, disputes, transfers, promotions — "
    "of roughly {fam_lo}-{fam_hi} documents each (give the file names as "
    "members); other families are assigned whole (give the family name as a "
    "member);\n"
    "- keep blocks between roughly {lo:,} and {hi:,} source characters when "
    "the material allows; do not merge unrelated areas just to fill a block;\n"
    "- every family / listed file must appear in exactly one block.\n"
    "For each block give: name (short, snake_case), description (ONE "
    "sentence saying what questions it answers — this is the routing index; "
    "name the products/operations concretely), members (family names or file "
    "names exactly as listed), why (one phrase).")


SOURCE_EXT = DOC_EXT + (".py",)


def source_files(ws: Workspace) -> list[str]:
    """Only compilable files take part (manifest.json and the like are not
    source and must not eat budget share)."""
    return sorted(f for f in ws.list_files() if f.endswith(SOURCE_EXT))


def family_of(file: str) -> str:
    return re.sub(r"_\d+$", "", Path(file).stem) if file.endswith(DOC_EXT) else file


def title_of(ws: Workspace, f: str) -> str:
    for line in ws.read_text(f).splitlines():
        if line.startswith("#"):
            return line.strip("# ").strip()[:80]
    return Path(f).stem


def digest(ws: Workspace, brief: str = "") -> tuple[str, dict[str, list[str]]]:
    """Directory -> family -> files digest with sizes; per-doc titles for
    single-family directories. Returns (text, member -> files)."""
    files = source_files(ws)
    by_dir: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        d = f.rsplit("/", 1)[0] if "/" in f else "."
        by_dir[d][family_of(f)].append(f)
    members: dict[str, list[str]] = {}
    lines = ["# Source digest (directory / family: docs, characters)"]
    for d, fams in sorted(by_dir.items()):
        n = sum(len(v) for v in fams.values())
        chars = sum(len(ws.read_text(f)) for v in fams.values() for f in v)
        lines.append(f"\n## {d}/  ({n} files, {chars:,} chars)")
        for fam, fs in sorted(fams.items()):
            members[fam] = fs
            c = sum(len(ws.read_text(f)) for f in fs)
            lines.append(f"- {fam}  ({len(fs)} docs, {c:,} chars)")
            if len(fams) == 1 and len(fs) > 6:     # one big family: show titles
                for f in fs[:SINGLE_FAMILY_TITLES]:
                    members[f.rsplit("/", 1)[-1]] = [f]
                    lines.append(f"    - {f.rsplit('/', 1)[-1]}: {title_of(ws, f)}")
    if brief:
        m = re.search(r"(2[.)]?\s*MAIN OBJECTS.*?)(?=\n\s*3[.)]|\n\s*WHAT TO FOCUS|\Z)", brief, re.S | re.I)
        if m:
            lines.append("\n## How the source's author groups things (reading brief)\n" + m.group(1)[:2500])
    return "\n".join(lines), members


def validate(blocks: list[dict], members: dict[str, list[str]],
             all_files: list[str]) -> list[dict]:
    """Every file in exactly one block; unknown members dropped; leftovers
    become `<dir>:rest`. Deterministic: first block to claim a file wins."""
    claimed: set[str] = set()
    out = []
    for b in blocks:
        fs: list[str] = []
        for mname in coerce_str_list(b.get("members")):
            key = mname.strip()
            cand = members.get(key) or members.get(key.rsplit("/", 1)[-1]) or []
            for f in cand:
                if f not in claimed:
                    claimed.add(f); fs.append(f)
        if fs:
            out.append({"name": re.sub(r"[^a-z0-9]+", "_", str(b.get("name", "block")).lower()).strip("_"),
                        "description": str(b.get("description", "")).strip(),
                        "why": str(b.get("why", "")).strip(), "files": sorted(fs)})
    rest: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        if f not in claimed:
            rest[f.rsplit("/", 1)[0] if "/" in f else "."].append(f)
    for d, fs in sorted(rest.items()):
        out.append({"name": f"{d.replace('/', '_')}_rest", "description": f"Remaining documents of {d}/.",
                    "why": "unassigned by the partitioner", "files": sorted(fs)})
    names = Counter(b["name"] for b in out)
    seen: Counter = Counter()
    for b in out:
        if names[b["name"]] > 1:
            seen[b["name"]] += 1
            b["name"] = f"{b['name']}_{seen[b['name']]}"
    return out


def assign_budgets(blocks: list[dict], ws: Workspace, total: int, cap: int) -> None:
    for b in blocks:
        b["chars"] = sum(len(ws.read_text(f)) for f in b["files"])
        b["families"] = sorted({family_of(f) for f in b["files"]})
    tot = max(1, sum(b["chars"] for b in blocks))
    for b in blocks:
        b["budget_tokens"] = int(max(MIN_BLOCK_BUDGET, min(cap, round(total * b["chars"] / tot))))


def partition(ws: Workspace, llm: LLMClient, brief: str = "", total: int = 40000,
              cap: int = 6000, load: int = 12000, lo: int = 40000, hi: int = 200000) -> dict:
    text, members = digest(ws, brief)
    out = llm.complete_json(system=PARTITION_PROMPT.format(lo=lo, hi=hi, fam_lo=FAMILY_SPLIT_DOCS[0], fam_hi=FAMILY_SPLIT_DOCS[1]),
                            user=text, schema=PARTITION_SCHEMA, purpose="partition")
    blocks = validate([b for b in (out.get("blocks") or []) if isinstance(b, dict)],
                      members, source_files(ws))
    assign_budgets(blocks, ws, total, cap)
    return {"total_budget": total, "block_cap": cap, "load_budget": load,
            "digest_chars": len(text), "blocks": blocks}


def render(part: dict) -> str:
    lines = [f"# Partition: {len(part['blocks'])} blocks | total budget {part['total_budget']:,} tok, "
             f"block cap {part['block_cap']:,}, load budget {part['load_budget']:,}", "",
             "| block | files | chars | budget tok | description |", "|---|---|---|---|---|"]
    for b in part["blocks"]:
        lines.append(f"| {b['name']} | {len(b['files'])} | {b['chars']:,} | {b['budget_tokens']:,} | {b['description'][:160]} |")
    return "\n".join(lines)


TOTAL_BUDGET = 40000      # tokens: the sum of all block models
BLOCK_CAP = 6000          # tokens: one block model


def run_stage(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--out", required=True, help="partition.json path")
    p.add_argument("--brief", default="")
    p.add_argument("--model", required=True)
    args = p.parse_args(argv)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    trace = Trace(out.parent, "trace_partition")
    llm = LLMClient(model=args.model, trace=trace)
    ws = Workspace(args.workspace)
    brief = Path(args.brief).read_text() if args.brief else ""
    part = partition(ws, llm, brief, TOTAL_BUDGET, BLOCK_CAP, M_BUDGET)
    trace.close()
    out.write_text(json.dumps(part, indent=1))
    (out.parent / "partition.md").write_text(render(part) + "\n")
    print(render(part))
    return 0
