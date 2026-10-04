"""SWE-QA (Peng et al., 2026) -> one persistent source per Python repository at its pinned commit.

Input: a checkout of the benchmark (https://github.com/peng-weihan/SWE-QA-Bench): `repo_commit.txt`
(`<url> <commit>` per repository) and `Benchmark/<repo>.jsonl` (question, answer rows).

Source  the repository cloned at the pinned commit (source code, documentation and configuration
        files); the draft stage reads the package directory `region`.
Tasks   the 48 official questions of the repository; qid `<repo>-<i>`.

By default the six repositories of the paper are prepared (`frozen/sweqa_repos.json`: streamlink,
sphinx, xarray, pytest, flask, conan). Clones go under `--repos-dir` (default `<out>/repos`) as
`<owner>__<repo>@<commit>` and are reused when present.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sourcelearn.datasets.common import check_split_coverage, copy_splits, frozen, index_row, write_index, write_json

# the package directory the draft stage reads, per benchmark repository
REGIONS = {
    "astropy": "astropy", "django": "django", "flask": "src", "matplotlib": "lib",
    "pylint": "pylint", "pytest": "src", "requests": "src", "scikit-learn": "sklearn",
    "sphinx": "sphinx", "sqlfluff": "src", "sympy": "sympy", "xarray": "xarray",
    "conan": "conan", "reflex": "reflex", "streamlink": "src",
}


def load_repo_commits(bench_dir: Path) -> dict[str, tuple[str, str]]:
    """{repo name: (url, commit)} from the benchmark's repo_commit.txt."""
    out = {}
    for line in (Path(bench_dir) / "repo_commit.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 2:
            out[p[0].rstrip("/").rsplit("/", 1)[-1]] = (p[0], p[1])
    return out


def clone_pinned(dest: Path, url: str, sha: str) -> Path:
    if not (dest / ".git").exists():
        print(f"[clone] {url} -> {dest}", file=sys.stderr)
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True, timeout=1800)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", sha], check=True)
    return dest


def prepare(out: Path, bench_dir: Path, repos_dir: Path | None = None, repos: tuple[str, ...] | None = None,
            clone: bool = True) -> list[dict]:
    out, bench_dir = Path(out), Path(bench_dir)
    repos_dir = Path(repos_dir) if repos_dir else out / "repos"
    commits = load_repo_commits(bench_dir)
    paper = {r["repo"]: r for r in frozen("sweqa_repos.json")["repos"]}
    names = list(repos) if repos else list(paper)
    index = []
    for name in names:
        if name not in commits:
            raise SystemExit(f"sweqa: {name!r} is not in {bench_dir / 'repo_commit.txt'}; known: {sorted(commits)}")
        url, sha = commits[name]
        if name in paper and paper[name]["commit"] != sha:
            print(f"[warn] sweqa/{name}: the checkout pins {sha}, the paper used {paper[name]['commit']}", file=sys.stderr)
        owner = url.rstrip("/").rsplit("/", 2)[-2]
        rid = f"{owner}__{name}"
        ws = repos_dir / f"{rid}@{sha}"
        if clone:
            clone_pinned(ws, url, sha)
        elif not ws.is_dir():
            print(f"[warn] sweqa/{rid}: clone missing at {ws}", file=sys.stderr)
        rows = [json.loads(l) for l in (bench_dir / "Benchmark" / f"{name}.jsonl").read_text().splitlines() if l.strip()]
        qs = [{"qid": f"{name}-{i}", "question": r["question"], "answer": r["answer"]} for i, r in enumerate(rows)]
        qpath = write_json(out / rid / "questions.json", qs)
        split_dir = copy_splits("sweqa", rid, out / rid / "splits")
        check_split_coverage(split_dir, [q["qid"] for q in qs], f"sweqa/{rid}")
        region = paper.get(name, {}).get("region") or REGIONS.get(name, "")
        n_py = sum(1 for _ in ws.rglob("*.py")) if ws.is_dir() else 0
        index.append(index_row(rid, ws, qpath, "code", split_dir, region=region, repo=name, url=url, commit=sha,
                               py_files=n_py, n_questions=len(qs)))
        print(f"sweqa/{rid}@{sha}: {n_py} .py files, region={region or '(whole repo)'}, {len(qs)} questions")
    write_index(out, index)
    return index


def add_arguments(p) -> None:
    p.add_argument("--bench", required=True, help="checkout of SWE-QA-Bench (repo_commit.txt, Benchmark/*.jsonl)")
    p.add_argument("--repos-dir", default="", help="where the pinned clones live or are created (default <out>/repos)")
    p.add_argument("--repos", default="", help="comma-separated repository names (default: the paper's six)")
    p.add_argument("--no-clone", action="store_true", help="do not clone; expect the pinned checkouts under --repos-dir")


def run(args) -> int:
    prepare(Path(args.out), Path(args.bench), Path(args.repos_dir) if args.repos_dir else None,
            tuple(x.strip() for x in args.repos.split(",") if x.strip()) or None, clone=not args.no_clone)
    return 0
