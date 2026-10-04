"""MultiDoc2Dial (Feng et al., 2021) -> four persistent sources, one per government domain.

Input: the official release directory (https://github.com/IBM/multidoc2dial, `multidoc2dial.zip`)
holding `multidoc2dial_doc.json` and `multidoc2dial_dial_validation.json`.

Source  each domain's documents (92-149 of them), rebuilt as markdown from the official spans:
        h-tags become headings and the sub-sentence spans of a section are joined into a paragraph.
Tasks   agent turns of the validation dialogues whose preceding user turn is a query and whose
        reply is a grounded solution; at most one question per dialogue, `per_domain` (60) drawn
        with `seed` (0). The question carries up to `history` (4) earlier turns; the reference is
        the agent's reply plus the grounding section(s); `gold_docs` names the cited document(s).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from sourcelearn.datasets.common import check_split_coverage, copy_splits, index_row, write_index, write_json

DOMAINS = ("dmv", "ssa", "studentaid", "va")
PER_DOMAIN, SEED, HISTORY, SPLIT = 60, 0, 4, "validation"


def _slug(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", text.split("#")[0].lower()).strip("_")[:60] or "doc"
    base = re.sub(r"_\d+$", "", base) or "doc"      # no trailing _N: each document is its own entity
    out, k = base, 1
    while out in taken:
        k += 1; out = f"{base}_v{k}"
    taken.add(out)
    return out


def doc_markdown(doc: dict) -> str:
    """One MultiDoc2Dial document as markdown, rebuilt from its spans."""
    lines = [f"# {doc['title'].split('#')[0].strip()}", ""]
    para: list[str] = []
    sec = None

    def flush():
        if para:
            lines.append(re.sub(r"\s+([,.;:!?])", r"\1", " ".join(para)).strip()); lines.append("")
        para.clear()

    for sp in sorted(doc["spans"].values(), key=lambda x: int(x["id_sp"])):
        t = re.sub(r"\s+", " ", sp["text_sp"]).strip()
        if not t:
            continue
        tag = str(sp.get("tag", ""))
        if re.fullmatch(r"h[1-6]", tag):
            flush(); lines += ["#" * min(6, int(tag[1]) + 1) + " " + t, ""]; sec = sp.get("id_sec")
            continue
        if sp.get("id_sec") != sec:
            flush(); sec = sp.get("id_sec")
        para.append(t)
    flush()
    return "\n".join(lines).strip() + "\n"


def prepare(out: Path, raw_dir: Path, per_domain: int = PER_DOMAIN, seed: int = SEED, history: int = HISTORY,
            split: str = SPLIT, domains: tuple[str, ...] = DOMAINS) -> list[dict]:
    out, raw_dir = Path(out), Path(raw_dir)
    docs = json.load(open(raw_dir / "multidoc2dial_doc.json"))["doc_data"]
    dials = json.load(open(raw_dir / f"multidoc2dial_dial_{split}.json"))["dial_data"]
    rng = random.Random(seed)
    index = []
    for dom in sorted(docs):                     # the one RNG runs over every domain in sorted order
        ws = out / dom / "workspace" / dom
        ws.mkdir(parents=True, exist_ok=True)
        taken: set[str] = set()
        slug_of: dict[str, str] = {}
        for doc_id, doc in docs[dom].items():
            slug_of[doc_id] = _slug(doc["title"], taken)
            (ws / f"{slug_of[doc_id]}.md").write_text(doc_markdown(doc))
        cands = []
        for dl in dials.get(dom, []):
            turns = dl["turns"]
            opts = []
            for i in range(1, len(turns)):
                cur, prev = turns[i], turns[i - 1]
                if cur["role"] != "agent" or prev["role"] != "user" or not str(prev["da"]).startswith("query"):
                    continue
                if not str(cur["da"]).startswith("respond_solution"):
                    continue
                if len(cur["utterance"]) < 30 or not cur.get("references"):
                    continue
                opts.append(i)
            if opts:
                cands.append((dl, rng.choice(opts)))       # at most one question per dialogue
        rng.shuffle(cands)
        qs = []
        for dl, i in cands[:per_domain]:
            turns = dl["turns"]
            hist = turns[max(0, i - 1 - history):i - 1]
            lines = [f"{'User' if t['role'] == 'user' else 'Agent'}: {t['utterance']}" for t in hist]
            q = ("Conversation so far:\n" + "\n".join(lines) + "\n\n" if lines else "") + \
                f"User's current question: {turns[i - 1]['utterance']}\n" \
                "Answer the user's current question as the agent would, using the source."
            spans: list[str] = []     # the whole section around each cited span (spans are sub-sentence)
            for r in turns[i]["references"]:
                d = docs[dom].get(r["doc_id"])
                sp = (d or {}).get("spans", {}).get(str(r["id_sp"]))
                if sp:
                    sec_t = re.sub(r"\s+", " ", sp.get("text_sec") or sp["text_sp"]).strip()[:1500]
                    if sec_t not in spans:
                        spans.append(sec_t)
            ref = turns[i]["utterance"] + "\n\nGrounding passage(s) from the source:\n" + "\n".join(f"- {x}" for x in spans)
            gold = sorted({f"{dom}/{slug_of[r['doc_id']]}.md" for r in turns[i]["references"] if r["doc_id"] in slug_of})
            qs.append({"qid": f"{dom}_{dl['dial_id'][:8]}_{turns[i]['turn_id']}", "question": q, "answer": ref,
                       "gold_docs": gold, "da": turns[i]["da"], "user_da": turns[i - 1]["da"]})
        if dom not in domains:
            continue
        qpath = write_json(out / dom / "questions.json", qs)
        split_dir = copy_splits("multidoc2dial", dom, out / dom / "splits")
        check_split_coverage(split_dir, [q["qid"] for q in qs], f"multidoc2dial/{dom}")
        chars = sum(len(d["doc_text"]) for d in docs[dom].values())
        index.append(index_row(dom, out / dom / "workspace", qpath, "docs", split_dir,
                               docs=len(docs[dom]), chars=chars, n_questions=len(qs)))
        print(f"multidoc2dial/{dom}: {len(docs[dom])} documents, {chars:,} chars, {len(qs)} questions "
              f"from {len(cands)} eligible dialogues")
    write_index(out, index)
    return index


def add_arguments(p) -> None:
    p.add_argument("--raw", required=True, help="the unzipped official release (multidoc2dial_doc.json, multidoc2dial_dial_validation.json)")
    p.add_argument("--domains", default=",".join(DOMAINS), help="comma-separated domains to prepare")
    p.add_argument("--per-domain", type=int, default=PER_DOMAIN)
    p.add_argument("--seed", type=int, default=SEED)


def run(args) -> int:
    prepare(Path(args.out), Path(args.raw), args.per_domain, args.seed,
            domains=tuple(x.strip() for x in args.domains.split(",") if x.strip()))
    return 0
