"""Shared helpers of the dataset adapters: markdown splitting, the prepared
layout, and the frozen manifests of the paper."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

PART_CHARS = 800      # <= the 900-char section excerpt of the draft stage, so no section is truncated
PARTS_PER_FILE = 4    # one file = one knowledge entity of the docs adapter (~3k chars)
FROZEN = Path(__file__).with_name("frozen")


# ---------------------------------------------------------------- markdown splitting

def split_long(para: str) -> list[str]:
    """A paragraph longer than PART_CHARS (book prose, script blocks) is cut at
    sentence ends, then at line breaks, so no section exceeds the limit."""
    if len(para) <= PART_CHARS:
        return [para]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’)])\s+|\n", para):
        if cur and len(cur) + 1 + len(sent) > PART_CHARS:
            out.append(cur); cur = sent
        else:
            cur = (cur + " " + sent) if cur else sent
    if cur:
        out.append(cur)
    return out


def parts(text: str) -> list[str]:
    """Plain text -> parts of at most PART_CHARS, paragraph boundaries kept."""
    paras = [q for p in re.split(r"\n\s*\n", text) if p.strip() for q in split_long(p.strip())]
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > PART_CHARS:
            out.append(cur); cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        out.append(cur)
    return out


def sections(heading: str, lines: list[str]) -> list[str]:
    """One markdown section per <= PART_CHARS chunk of `lines`; continuations
    repeat the heading with '(cont. N)'."""
    chunks, cur = [], ""
    for ln in lines:
        if cur and len(cur) + len(ln) + 1 > PART_CHARS:
            chunks.append(cur); cur = ln
        else:
            cur = (cur + "\n" + ln) if cur else ln
    if cur:
        chunks.append(cur)
    return [f"## {heading}{'' if i == 0 else f' (cont. {i + 1})'}\n\n{c}" for i, c in enumerate(chunks)]


def slug(text: str, maxlen: int = 70) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:maxlen]


# ---------------------------------------------------------------- the prepared layout

def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    return path


def frozen(name: str) -> dict:
    """A frozen manifest shipped with the package (`frozen/<name>`)."""
    return json.loads((FROZEN / name).read_text())


def copy_splits(bench: str, source: str, dest: Path) -> Path | None:
    """Copy the paper's frozen splits of one source into `dest`; None when the
    source is not one of the paper's (the pipeline then draws a random 30/70
    split per seed)."""
    src = FROZEN / "splits" / bench / source
    if not src.is_dir():
        return None
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")):
        shutil.copy(f, dest / f.name)
    return dest


def check_split_coverage(split_dir: Path | None, qids: list[str], label: str) -> None:
    """A frozen split must name exactly the prepared questions; a mismatch means
    the benchmark release differs from the paper's and is reported, not hidden."""
    if split_dir is None:
        return
    have = set(qids)
    for f in sorted(Path(split_dir).glob("*.json")):
        d = json.loads(f.read_text())
        named = set(d.get("train", [])) | set(d.get("test", [])) | set(d.get("dev", []))
        missing, extra = sorted(named - have), sorted(have - named)
        if missing or extra:
            print(f"[warn] {label}: {f.name} names {len(missing)} question(s) not prepared and omits {len(extra)} "
                  f"prepared one(s); the release you prepared from differs from the paper's "
                  f"(e.g. {missing[:2] or extra[:2]})")


def index_row(sid: str, workspace: Path, questions: Path, adapter: str, split_dir: Path | None = None,
              region: str = "", **extra) -> dict:
    row = {"id": sid, "workspace": str(workspace), "questions": str(questions), "adapter": adapter,
           "region": region, "split_dir": str(split_dir) if split_dir else ""}
    row.update(extra)
    return row


def write_index(out: Path, rows: list[dict]) -> Path:
    return write_json(Path(out) / "index.json", rows)
