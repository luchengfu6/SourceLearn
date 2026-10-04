"""NarrativeQA (Kocisky et al., 2018) -> one persistent source per story, read in full.

Input: the Hugging Face release `deepmind/narrativeqa` (official test split; the parquet rows carry
the full text):

    huggingface-cli download deepmind/narrativeqa --repo-type dataset --include "data/test-*" --local-dir <raw>

Source  the cleaned full text of a story (Project Gutenberg boilerplate and IMSDb page markup
        removed; never the summary), written as `story/partNN.md` files of PARTS_PER_FILE
        sections of at most PART_CHARS characters each.
Tasks   the official questions of the story; `answer` is the first official reference,
        `references` both.

By default the 12 stories of the paper are prepared (`frozen/narrativeqa_stories.json`): six movie
scripts and six books spanning 18k to 301k tokens. `--story-ids` / `--titles` name other test
documents instead; each must resolve to exactly one document.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from sourcelearn.datasets.common import (PARTS_PER_FILE, check_split_coverage, copy_splits, frozen, index_row, parts,
                                         write_index, write_json)

MIN_CHARS = 1000       # a cleaned "full text" shorter than this is a fetch-failure stub, not a story
_GUT_START = re.compile(r"^.*(\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG|\*END\*THE SMALL PRINT).*$", re.I | re.M)
_GUT_END = re.compile(r"^.*(\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG|End of (the |this )?Project Gutenberg).*$", re.I | re.M)


def clean(text: str, kind: str) -> str:
    """Boilerplate rules, no per-story edits. gutenberg: the text between the
    '*** START OF ...' line and the first '*** END OF ...' line after it.
    movie (HTML page): IMSDb's <td class="scrtext"> cell, else the largest
    <pre> block, else the whole page, tags stripped and entities unescaped."""
    if kind == "gutenberg":
        m = _GUT_START.search(text)
        if m:
            text = text[m.end():]
        m = _GUT_END.search(text)
        if m:
            text = text[:m.start()]
    else:
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.S | re.I)
        m = re.search(r'<td class="scrtext">(.*?)</td>', text, flags=re.S)
        pres = re.findall(r"<pre[^>]*>(.*?)</pre>", text, flags=re.S | re.I)
        if m:
            text = m.group(1)
        elif pres:
            text = max(pres, key=len)
        text = re.sub(r"<\s*(br|/p|/pre|/div)\s*/?>", "\n", text, flags=re.I)
        text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def is_story(raw: str, cleaned: str) -> bool:
    """False for fetch-failure stubs: an HTTP error page or a (nearly) empty text."""
    title = re.search(r"<title>(.*?)</title>", raw[:5000], flags=re.S | re.I)
    if title and re.search(r"\b40[0-9]\b|not found|error", title.group(1), flags=re.I):
        return False
    return len(cleaned) >= MIN_CHARS


def read_test_split(raw_dir: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """The official test-split parquets -> {document id: document}, {document id: question rows}."""
    import glob
    try:
        import pyarrow.parquet as pq
    except ImportError as e:                       # pragma: no cover
        raise SystemExit("narrativeqa: pyarrow is required (pip install 'sourcelearn[datasets]')") from e
    files = sorted(glob.glob(str(Path(raw_dir) / "data" / "test-*.parquet"))) or sorted(glob.glob(str(Path(raw_dir) / "test-*.parquet")))
    if not files:
        raise SystemExit(f"narrativeqa: no test-*.parquet under {raw_dir} (download deepmind/narrativeqa data/test-*)")
    docs: dict[str, dict] = {}
    qs_of: dict[str, list[dict]] = {}
    for f in files:
        for r in pq.read_table(f, columns=["document", "question", "answers"]).to_pylist():
            d = r["document"]
            docs.setdefault(d["id"], d)
            qs_of.setdefault(d["id"], []).append({"question": r["question"]["text"],
                                                  "references": [a["text"] for a in r["answers"]]})
    return docs, qs_of


def resolve(docs: dict[str, dict], story_ids=(), titles=()) -> list[str]:
    """Each id (full or hex prefix) and each title (case-insensitive substring of the title or url)
    must match exactly one test document; nothing is guessed."""
    out: list[str] = []
    for spec in [str(s).strip() for s in list(story_ids) + list(titles) if str(s).strip()]:
        hits = sorted(i for i in docs if i.startswith(spec)) if re.fullmatch(r"[0-9a-f]{6,40}", spec) else []
        if not hits:
            k = spec.lower()
            hits = sorted(i for i, d in docs.items()
                          if k in str(d["summary"]["title"] or "").lower() or k in str(d["url"] or "").lower())
        if len(hits) != 1:
            raise SystemExit(f"narrativeqa: {spec!r} matches {len(hits)} test documents {[h[:8] for h in hits[:6]]}")
        if hits[0] not in out:
            out.append(hits[0])
    return out


def write_story(out: Path, did: str, title: str, text: str, q_rows: list[dict]) -> tuple[Path, Path, int]:
    ws = out / did / "workspace" / "story"
    ws.mkdir(parents=True, exist_ok=True)
    for old in ws.glob("part*.md"):
        old.unlink()
    ps = parts(text)
    files = [ps[k:k + PARTS_PER_FILE] for k in range(0, len(ps), PARTS_PER_FILE)]
    width = max(2, len(str(len(files))))
    for fi, chunk in enumerate(files):
        body = f"# {title} (part {fi + 1} of {len(files)})\n\n" + "\n\n".join(
            f"## Part {fi * PARTS_PER_FILE + j + 1}\n\n{pt}" for j, pt in enumerate(chunk))
        (ws / f"part{fi + 1:0{width}d}.md").write_text(body)
    qs = [{"qid": f"{did[:8]}_{j + 1}", "question": q["question"], "answer": q["references"][0],
           "references": q["references"]} for j, q in enumerate(q_rows)]
    return out / did / "workspace", write_json(out / did / "questions.json", qs), len(files)


def prepare(out: Path, raw_dir: Path, story_ids=(), titles=()) -> list[dict]:
    out = Path(out)
    docs, qs_of = read_test_split(Path(raw_dir))
    if not story_ids and not titles:
        story_ids = [s["id"] for s in frozen("narrativeqa_stories.json")["stories"]]
    index = []
    for did in resolve(docs, story_ids, titles):
        d = docs[did]
        text = clean(d["text"], d["kind"])
        if not is_story(d["text"], text):
            raise SystemExit(f"narrativeqa: {did[:8]} is a fetch-failure stub after cleaning ({len(text)} chars), not a story")
        title = (d["summary"]["title"] or did).strip()[:120]
        ws, qpath, n_files = write_story(out, did, title, text, qs_of[did])
        split_dir = copy_splits("narrativeqa", did, out / did / "splits")
        check_split_coverage(split_dir, [f"{did[:8]}_{j + 1}" for j in range(len(qs_of[did]))], f"narrativeqa/{did[:8]}")
        index.append(index_row(did, ws, qpath, "docs", split_dir, title=title,
                               kind="book" if d["kind"] == "gutenberg" else "movie script",
                               chars=len(text), files=n_files, n_questions=len(qs_of[did])))
        print(f"narrativeqa/{did[:8]}: {title[:50]:52s} {len(text):>9,} chars {n_files:>4} files {len(qs_of[did]):>3} questions")
    write_index(out, index)
    return index


def add_arguments(p) -> None:
    p.add_argument("--raw", required=True, help="directory holding the deepmind/narrativeqa test parquets (data/test-*.parquet)")
    p.add_argument("--story-ids", default="", help="comma-separated document ids (full or hex prefix); default: the paper's 12 stories")
    p.add_argument("--titles", default="", help="comma-separated title substrings, each matching exactly one test document")


def run(args) -> int:
    csv = lambda x: tuple(v.strip() for v in x.split(",") if v.strip())     # noqa: E731
    prepare(Path(args.out), Path(args.raw), csv(args.story_ids), csv(args.titles))
    return 0
