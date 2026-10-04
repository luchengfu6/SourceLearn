"""APIBench (Patil et al., 2024) -> one persistent source per API hub of the Gorilla release.

Input: the pinned Gorilla checkout (https://github.com/ShishirPatil/gorilla, commit 6ea5797).

Source  the COMPLETE API documentation corpus of a hub (`data/api/<hub>_api.jsonl`: 94 TorchHub,
        936 HuggingFace, 696 TensorFlow Hub records), one markdown document per record under
        `<domain>/<corpus index>_<api name>.md`; nothing is sampled or abbreviated.
Tasks   instructions of the official EVAL pairs (`data/apibench/<hub>_eval.json` with the 0-shot
        question file); the reference is the gold call in the official response format and
        `gold_docs` names the API document the instruction was written from. The train pairs are
        never read. By default the paper's frozen pools are rebuilt (`frozen/apibench_tasks.json`:
        TorchHub 186, HuggingFace 100, TensorFlow Hub 100); `--all` takes every eval task that maps
        to exactly one API document instead.

The paper scores APIBench with Gorilla's official AST evaluator (functionality accuracy); the
pipeline's own judge-based accuracy is a proxy, see README.
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

from sourcelearn.datasets.common import check_split_coverage, copy_splits, frozen, index_row, parts, sections, slug, write_index, write_json

HUBS: dict[str, dict] = {
    "huggingface": {"tag": "hf", "api": "data/api/huggingface_api.jsonl", "eval": "data/apibench/huggingface_eval.json",
                    "questions": "eval/eval-data/questions/huggingface/questions_huggingface_0_shot.jsonl"},
    "tensorflowhub": {"tag": "tf", "api": "data/api/tensorflowhub_api.jsonl", "eval": "data/apibench/tensorflow_eval.json",
                      "questions": "eval/eval-data/questions/tensorflowhub/questions_tensorflowhub_0_shot.jsonl"},
    "torchhub": {"tag": "th", "api": "data/api/torchhub_api.jsonl", "eval": "data/apibench/torchhub_eval.json",
                 "questions": "eval/eval-data/questions/torchhub/questions_torchhub_0_shot.jsonl"},
}
DOC_FIELDS = ("domain", "framework", "functionality", "api_name", "api_call", "api_arguments",
              "python_environment_requirements", "example_code", "performance", "description")
FIELD_HEADINGS = (("Arguments", "api_arguments"), ("Environment requirements", "python_environment_requirements"),
                  ("Example", "example_code"), ("Performance", "performance"))
# the shape of the official response format, so a generic answerer emits what the AST scorer reads
FORMAT_LINE = ("Answer with exactly one API call, in this format: <<<domain>>>: $DOMAIN, "
               "<<<api_call>>>: $API_CALL, <<<api_provider>>>: $API_PROVIDER. The $API_CALL must be "
               "one line of code that calls the API.")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _questions_path(repo: Path, hub: str) -> Path:
    rel = HUBS[hub]["questions"]
    for cand in (repo / rel, repo / "gorilla" / rel):       # the checkout has moved this directory once
        if cand.exists():
            return cand
    raise SystemExit(f"apibench/{hub}: official question file not found under {repo} ({rel})")


def load_apis(repo: Path, hub: str) -> list[dict]:
    return _jsonl(repo / HUBS[hub]["api"])


def load_eval(repo: Path, hub: str) -> list[dict]:
    return _jsonl(repo / HUBS[hub]["eval"])


def load_questions(repo: Path, hub: str) -> dict[int, str]:
    """{question_id (1-based index into the eval file) -> instruction}."""
    return {int(r["question_id"]): str(r["text"]).replace("\\n", "\n").strip() for r in _jsonl(_questions_path(repo, hub))}


def _norm(v):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        s = v.strip()
        return "N/A" if s in ("", "N/A", "n/a", "None") else s
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def doc_key(doc: dict) -> str:
    return json.dumps({f: _norm(doc.get(f)) for f in DOC_FIELDS}, sort_keys=True)


def map_gold(apis: list[dict], evals: list[dict]) -> list[dict]:
    """The mechanical task -> gold API document map, one entry per eval row, tried in order of
    strictness: every documented field agrees (content); (domain, api_name, api_call) agrees
    (call); (domain, api_name) agrees (name). `n_contents` > 1 means a genuinely ambiguous task."""
    tiers = [("content", defaultdict(list)), ("call", defaultdict(list)), ("name", defaultdict(list))]
    for i, a in enumerate(apis):
        tiers[0][1][doc_key(a)].append(i)
        tiers[1][1][(a["domain"], a["api_name"], a["api_call"])].append(i)
        tiers[2][1][(a["domain"], a["api_name"])].append(i)
    out = []
    for qi, row in enumerate(evals):
        d = row["api_data"]
        keys = {"content": doc_key(d), "call": (d["domain"], d["api_name"], d["api_call"]), "name": (d["domain"], d["api_name"])}
        rec = {"question_id": qi + 1, "gold_index": None, "tier": "none", "n_contents": 0, "domain": d["domain"],
               "functionality": d.get("functionality"), "api_name": d["api_name"], "api_call": row["api_call"],
               "provider": row["provider"]}
        for name, table in tiers:
            hits = table.get(keys[name]) or []
            if hits:
                rec.update(tier=name, n_contents=len({doc_key(apis[i]) for i in hits}), gold_index=min(hits))
                break
        out.append(rec)
    return out


def gold_answer(rec: dict) -> str:
    return f"<<<domain>>>: {rec['domain']}\n<<<api_call>>>: {rec['api_call']}\n<<<api_provider>>>: {rec['provider']}"


def _lines(value, prefix: str = "") -> list[str]:
    if value is None or value in ("N/A", "", [], {}):
        return []
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out += _lines(v, key) if isinstance(v, (dict, list)) else [f"- `{key}`: {v}"]
        return out
    if isinstance(value, list):
        return [f"- {x}" if not prefix else f"- `{prefix}`: {x}" for x in value if str(x).strip()]
    return ([f"- {ln}" for ln in str(value).splitlines() if ln.strip()] if not prefix else [f"- `{prefix}`: {value}"])


def doc_path(i: int, doc: dict) -> str:
    """`<domain>/<corpus index>_<api name>.md`; the index disambiguates records sharing a name."""
    return f"{slug(doc.get('domain') or 'other')}/{i:04d}_{slug(doc['api_name'])}.md"


def api_markdown(doc: dict) -> str:
    body = [f"# {doc['api_name']}"]
    head = [f"- API name: `{doc['api_name']}`", f"- Canonical API call: `{doc['api_call']}`",
            f"- Provider / framework: {doc.get('framework') or 'N/A'}", f"- Domain: {doc.get('domain') or 'N/A'}",
            f"- Functionality: {doc.get('functionality') or 'N/A'}"]
    body += sections("API", head)
    if str(doc.get("description") or "").strip():
        body += sections("Description", parts(str(doc["description"])))
    for heading, field in FIELD_HEADINGS:
        lines = _lines(doc.get(field))
        if lines:
            body += sections(heading, lines)
    return "\n\n".join(body).strip() + "\n"


def task_row(hub: str, rec: dict, gold_file: str) -> dict:
    tag = HUBS[hub]["tag"]
    return {"qid": f"{tag}_{rec['question_id']:04d}", "question": f"{rec['instruction']}\n\n{FORMAT_LINE}",
            "answer": gold_answer(rec), "gold_docs": [gold_file], "api_id": f"{tag}:{rec['gold_index']}",
            "hub": hub, "question_id": rec["question_id"], "instruction": rec["instruction"], "domain": rec["domain"],
            "functionality": rec["functionality"], "api_name": rec["api_name"], "gold_api_call": rec["api_call"],
            "provider": rec["provider"]}


def prepare(out: Path, gorilla_repo: Path, hubs: tuple[str, ...] | None = None, all_tasks: bool = False) -> list[dict]:
    out, repo = Path(out), Path(gorilla_repo)
    pools = frozen("apibench_tasks.json")
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if commit and commit != pools["gorilla_commit"]:
        print(f"[warn] apibench: the checkout is at {commit[:7]}, the paper used {pools['gorilla_commit'][:7]}")
    index = []
    for hub in hubs or sorted(HUBS):
        apis = load_apis(repo, hub)
        ws = out / hub / "workspace"
        files, n_chars = [], 0
        for i, doc in enumerate(apis):
            rel = doc_path(i, doc)
            p = ws / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            md = api_markdown(doc)
            p.write_text(md)
            files.append(rel)
            n_chars += len(md)
        if len(set(files)) != len(files):
            raise SystemExit(f"apibench/{hub}: document paths collide")
        qs_text = load_questions(repo, hub)
        mapping = {rec["question_id"]: rec for rec in map_gold(apis, load_eval(repo, hub))}
        if all_tasks:
            wanted = [q for q in sorted(qs_text) if q in mapping and mapping[q]["gold_index"] is not None and mapping[q]["n_contents"] == 1]
        else:
            wanted = pools["question_ids"][hub]
        rows = []
        for qid in wanted:
            rec = mapping.get(qid)
            if rec is None or rec["gold_index"] is None or qid not in qs_text:
                print(f"[warn] apibench/{hub}: task {qid} cannot be rebuilt from this checkout; skipped")
                continue
            rows.append(task_row(hub, {**rec, "instruction": qs_text[qid]}, files[rec["gold_index"]]))
        qpath = write_json(out / hub / "questions.json", rows)
        split_dir = copy_splits("apibench", hub, out / hub / "splits") if not all_tasks else None
        check_split_coverage(split_dir, [r["qid"] for r in rows], f"apibench/{hub}")
        index.append(index_row(hub, ws, qpath, "docs", split_dir, apis=len(files), chars=n_chars, n_questions=len(rows),
                               domains=len({r["domain"] for r in rows}), commit=commit[:7]))
        print(f"apibench/{hub}: {len(files)} API documents, {n_chars:,} chars; {len(rows)} tasks over "
              f"{len({r['domain'] for r in rows})} domains")
    write_index(out, index)
    return index


def add_arguments(p) -> None:
    p.add_argument("--gorilla", required=True, help="checkout of ShishirPatil/gorilla (commit 6ea5797 for the paper's pools)")
    p.add_argument("--hub", default="", help="one hub (huggingface, tensorflowhub, torchhub); default all three")
    p.add_argument("--all", action="store_true", help="every scorable eval task instead of the paper's frozen pools")


def run(args) -> int:
    prepare(Path(args.out), Path(args.gorilla), (args.hub,) if args.hub else None, args.all)
    return 0
