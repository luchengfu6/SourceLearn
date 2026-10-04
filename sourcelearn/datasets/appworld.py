"""AppWorld (Trivedi et al., 2024) -> one persistent source: the static documentation of its 457 APIs.

Input: the AppWorld data release (`pip install appworld && appworld install && appworld download data`,
https://github.com/StonyBrookNLP/appworld), i.e. a root holding `data/api_docs/standard/*.json`,
`data/datasets/{train,dev,test_normal}.txt` and `data/tasks/<task>/`.

Source  one markdown specification per API (`<app>/<api>.md`: capability, call, parameters,
        authentication, response shapes) and one catalogue per app (`apps/<app>.md`), built ONLY
        from `api_docs/standard/*.json` and the official app descriptions. Databases, task
        instructions, gold solutions and the live documentation service never enter the source.
Tasks   the official train / dev / test_normal tasks, verbatim: `question` is the task instruction,
        `required_documents` names the API documents of the reference solution (the supervision E*
        task-guided learning reads), and `answer` lists those APIs. The paper trains on the 90
        train tasks and reports the 168 test_normal tasks; `splits/official.json` carries these
        partitions, so the 30/70 seed protocol does not apply.

Scoring an AppWorld task means running the official interactive environment and its evaluator
(task goal completion); this adapter prepares the source and the tasks, see README.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sourcelearn.datasets.common import FROZEN, index_row, write_index, write_json

SID = "apis"
STATIC_DOCS = "api_docs/standard"
OFFICIAL_SPLITS = ("train", "dev", "test_normal")
ALLOWED_DOC_KEYS = ("app_name", "api_name", "path", "method", "description", "parameters", "response_schemas")
ALLOWED_PARAM_KEYS = ("name", "type", "required", "description", "default", "constraints")


def static_api_doc(record: dict) -> dict:
    """One API record projected onto its documentation fields (the dataset's canary string and any
    state-bearing field a later release might add are dropped)."""
    out = {k: record[k] for k in ALLOWED_DOC_KEYS if k in record}
    out["parameters"] = [{k: p[k] for k in ALLOWED_PARAM_KEYS if k in p} for p in record.get("parameters") or []]
    return out


def read_static_docs(data_dir: Path) -> dict[str, dict[str, dict]]:
    """{app: {api_name: filtered record}}; the whole read surface of the source."""
    docs: dict[str, dict[str, dict]] = {}
    files = sorted((Path(data_dir) / STATIC_DOCS).glob("*.json"))
    if not files:
        raise SystemExit(f"appworld: no API documentation under {Path(data_dir) / STATIC_DOCS} (run `appworld download data`)")
    for p in files:
        docs[p.stem] = {name: static_api_doc(rec) for name, rec in json.loads(p.read_text()).items()}
    return docs


def app_descriptions(appworld_package: str = "") -> dict[str, str]:
    """The official one-line app descriptions (`appworld.apps.APP_TO_DESCRIPTION`), from the
    installed package or from a checkout's `apps/__init__.py`."""
    if not appworld_package:
        try:
            from appworld.apps import APP_TO_DESCRIPTION
            return dict(APP_TO_DESCRIPTION)
        except ImportError:
            print("[warn] appworld: package not importable, app descriptions left empty (pass --appworld-package)", file=sys.stderr)
            return {}
    text = (Path(appworld_package) / "apps" / "__init__.py").read_text()
    body = text.split("APP_TO_DESCRIPTION = {", 1)[1].split("\n}", 1)[0]
    return dict(eval("{" + body + "}"))          # noqa: S307 - a literal dict of the official source


def _params_md(params: list[dict]) -> str:
    if not params:
        return "This API takes no parameters.\n"
    lines = []
    for p in params:
        req = "required" if p.get("required") else "optional"
        head = f"- `{p['name']}` ({p.get('type', 'unknown')}, {req})"
        if not p.get("required") and p.get("default") is not None:
            head += f", default `{json.dumps(p['default'])}`"
        lines.append(head + f": {(p.get('description') or '').strip()}")
        for c in p.get("constraints") or []:
            lines.append(f"  - constraint: {c}")
    return "\n".join(lines) + "\n"


def api_markdown(doc: dict, app_description: str) -> str:
    app, name = doc["app_name"], doc["api_name"]
    params = doc.get("parameters") or []
    auth = next((p for p in params if p["name"] == "access_token"), None)
    schemas = doc.get("response_schemas") or {}
    out = [f"# {app}.{name}\n", f"App: {app} - {app_description}\n",
           f"## Capability\n\n{(doc.get('description') or '').strip()}\n",
           f"## Call\n\n`{doc.get('method', '')} {doc.get('path', '')}`, called from the agent's python environment as "
           f"`apis.{app}.{name}(" + ", ".join(f"{p['name']}=..." for p in params if p.get("required")) + ")`.\n",
           f"## Parameters\n\n{_params_md(params)}",
           "## Authentication\n\n" + (f"Requires an access token: {auth.get('description', '').strip()} "
                                      f"Log in to the {app} app first to obtain it.\n" if auth
                                      else f"No access token parameter: this API does not require a {app} login.\n")]
    if "success" in schemas:
        out.append("## Successful response\n\n```json\n" + json.dumps(schemas["success"], indent=1) + "\n```\n")
    if "failure" in schemas:
        out.append("## Failure response\n\nA failed call returns this shape instead of the successful one:\n\n```json\n"
                   + json.dumps(schemas["failure"], indent=1) + "\n```\n")
    return "\n".join(out)


def app_markdown(app: str, description: str, docs: dict[str, dict]) -> str:
    lines = [f"# App: {app}\n", f"{description.strip()}\n",
             f"## API catalogue\n\nThe {app} app exposes {len(docs)} APIs, callable as `apis.{app}.<api_name>(...)`:\n"]
    for name in sorted(docs):
        lines.append(f"- `{name}` ({docs[name].get('method', '')} {docs[name].get('path', '')}): "
                     f"{(docs[name].get('description') or '').strip()}")
    needs_login = sorted(n for n, d in docs.items() if any(p["name"] == "access_token" for p in d.get("parameters") or []))
    lines.append(f"\n## Login\n\nOf these, {len(needs_login)} require an `access_token` obtained from the {app} app's login API: "
                 + (", ".join(f"`{n}`" for n in needs_login) if needs_login else "none.") + "\n")
    return "\n".join(lines)


def _path_matches(template: str, url: str) -> bool:
    a, b = template.strip("/").split("/"), url.strip("/").split("/")
    return len(a) == len(b) and all(x.startswith("{") or x == y for x, y in zip(a, b))


def required_apis(api_calls: list[dict], docs: dict[str, dict[str, dict]]) -> list[dict]:
    """The APIs the reference solution called, as (app, api_name) resolved from the method and url
    of each gold call; the arguments of the calls (per-task state) are never read."""
    want, out = [], []
    for call in api_calls:
        url = str(call.get("url") or "").split("?")[0]
        method = str(call.get("method") or "").upper()
        if (method, url) not in want:
            want.append((method, url))
    for method, url in want:
        app = url.strip("/").split("/")[0]
        for name, doc in sorted((docs.get(app) or {}).items()):
            if doc.get("method", "").upper() == method and _path_matches(doc.get("path", ""), url):
                out.append({"app": app, "api_name": name, "method": method, "path": doc["path"]})
                break
    seen, uniq = set(), []
    for r in out:
        if (r["app"], r["api_name"]) not in seen:
            seen.add((r["app"], r["api_name"])); uniq.append(r)
    return uniq


def task_rows(data_dir: Path, docs: dict[str, dict[str, dict]]) -> list[dict]:
    root = Path(data_dir)
    rows = []
    for split in OFFICIAL_SPLITS:
        for tid in [x for x in (root / "datasets" / f"{split}.txt").read_text().split() if x.strip()]:
            d = root / "tasks" / tid
            spec = json.loads((d / "specs.json").read_text())
            gt = d / "ground_truth"
            calls = json.loads((gt / "api_calls.json").read_text()) if (gt / "api_calls.json").exists() else []
            meta = json.loads((gt / "metadata.json").read_text()) if (gt / "metadata.json").exists() else {}
            apps = json.loads((gt / "required_apps.json").read_text()) if (gt / "required_apps.json").exists() else []
            apis = required_apis(calls, docs)
            rows.append({
                "qid": tid, "task_id": tid, "split": split, "question": spec["instruction"],
                "answer": "APIs used by the reference solution, in order of first use: "
                          + ", ".join(f"{r['app']}.{r['api_name']}" for r in apis),
                "required_apis": apis,
                "required_documents": [f"{r['app']}/{r['api_name']}.md" for r in apis]
                                      + [f"apps/{a}.md" for a in sorted({r["app"] for r in apis})],
                "scenario_id": tid.rsplit("_", 1)[0], "difficulty": meta.get("difficulty"),
                "num_apps": meta.get("num_apps"), "required_apps": sorted(apps),
            })
    return rows


def prepare(out: Path, data_root: Path, appworld_package: str = "") -> list[dict]:
    out, data_dir = Path(out), Path(data_root) / "data"
    ws = out / SID / "workspace"
    docs = read_static_docs(data_dir)
    desc = app_descriptions(appworld_package)
    written = []
    (ws / "apps").mkdir(parents=True, exist_ok=True)
    for app in sorted(docs):
        (ws / app).mkdir(parents=True, exist_ok=True)
        (ws / "apps" / f"{app}.md").write_text(app_markdown(app, desc.get(app, ""), docs[app]))
        written.append(f"apps/{app}.md")
        for name, doc in sorted(docs[app].items()):
            (ws / app / f"{name}.md").write_text(api_markdown(doc, desc.get(app, "")))
            written.append(f"{app}/{name}.md")
    rows = task_rows(data_dir, docs)
    qpath = write_json(out / SID / "questions.json", rows)
    # the official partitions, frozen; rebuilt from the release when the shipped file does not match it
    split = {"bench": "appworld", "source": SID, "seed": None, "mode": "official", "n_questions": len(rows),
             "train": [r["qid"] for r in rows if r["split"] == "train"],
             "dev": [r["qid"] for r in rows if r["split"] == "dev"],
             "test": [r["qid"] for r in rows if r["split"] == "test_normal"]}
    shipped = json.loads((FROZEN / "splits/appworld/apis/official.json").read_text())
    if any(sorted(shipped[k]) != sorted(split[k]) for k in ("train", "dev", "test")):
        print("[warn] appworld: the official partitions of this release differ from the paper's; using the release's", file=sys.stderr)
    split_dir = out / SID / "splits"
    write_json(split_dir / "official.json", split)
    n_chars = sum(len((ws / f).read_text()) for f in written)
    index = [index_row(SID, ws, qpath, "docs", split_dir, apps=len(docs), apis=sum(len(v) for v in docs.values()),
                       files=len(written), chars=n_chars, n_questions=len(rows),
                       questions_by_split={s: sum(1 for r in rows if r["split"] == s) for s in OFFICIAL_SPLITS})]
    write_index(out, index)
    print(f"appworld/{SID}: {len(docs)} apps, {sum(len(v) for v in docs.values())} APIs -> {len(written)} documents, "
          f"{n_chars:,} chars; {len(rows)} tasks {index[0]['questions_by_split']}")
    return index


def add_arguments(p) -> None:
    p.add_argument("--data-root", required=True, help="the AppWorld root holding data/api_docs, data/datasets, data/tasks")
    p.add_argument("--appworld-package", default="", help="path of the appworld package / checkout (for APP_TO_DESCRIPTION) when it is not installed")


def run(args) -> int:
    prepare(Path(args.out), Path(args.data_root), args.appworld_package)
    return 0
