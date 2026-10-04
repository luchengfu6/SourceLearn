"""Run SourceLearn on every source of a prepared benchmark and aggregate the test accuracy.

    python -m sourcelearn.datasets multidoc2dial --raw <dir> --out data/multidoc2dial
    python examples/run_benchmark.py --index data/multidoc2dial/index.json --model <backbone> \
        --judge-model <judge> --out runs/multidoc2dial [--seeds 0,1,2] [--sources dmv,ssa]

For each source the full pipeline (construction -> self-directed -> task-guided learning) runs
with the paper's frozen guidance/test split of every seed when the prepared source carries one,
and a random 30/70 split per seed otherwise. A source whose run exists is resumed, so the script
can be re-run after an interruption. The accuracy is read from `task_s<seed>/eval/results.jsonl`
(VALID rows only), macro-averaged over the sources of the benchmark and then over the seeds, as
in the paper.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def accuracy(results: Path) -> tuple[float | None, int]:
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()] if results.exists() else []
    valid = [bool(r.get("correct")) for r in rows if r.get("status", "VALID") == "VALID"]
    return (sum(valid) / len(valid) if valid else None), len(valid)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", required=True, help="index.json of a prepared benchmark")
    p.add_argument("--model", required=True, help="the backbone of every stage")
    p.add_argument("--judge-model", default="", help="answer judge (default: --model)")
    p.add_argument("--out", required=True, help="run directory; one sub-directory per source")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--sources", default="", help="comma-separated source ids (default: every source of the index)")
    p.add_argument("--workers", type=int, default=6, help="parallel LLM calls")
    a = p.parse_args(argv)

    index = json.loads(Path(a.index).read_text())
    want = {s.strip() for s in a.sources.split(",") if s.strip()}
    sources = [r for r in index if not want or r["id"] in want]
    if not sources:
        raise SystemExit(f"no source of {a.index} matches {sorted(want)}")
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out = Path(a.out)
    table: dict[str, dict[int, float | None]] = {}
    for src in sources:
        run_dir = out / src["id"]
        cmd = [sys.executable, "-m", "sourcelearn.pipeline", "--workspace", src["workspace"], "--questions", src["questions"],
               "--adapter", src["adapter"], "--model", a.model, "--out", str(run_dir), "--seeds", a.seeds,
               "--workers", str(a.workers)]
        if a.judge_model:
            cmd += ["--judge-model", a.judge_model]
        if src.get("region"):
            cmd += ["--region", src["region"]]
        if src.get("split_dir") and Path(src["split_dir"]).is_dir():
            cmd += ["--split-dir", src["split_dir"]]
        print(f"\n=== {src['id']} ===\n$ {' '.join(cmd)}", file=sys.stderr)
        rc = subprocess.run(cmd).returncode
        table[src["id"]] = {}
        for seed in seeds:
            acc, n = accuracy(run_dir / f"task_s{seed}" / "eval" / "results.jsonl")
            table[src["id"]][seed] = acc
            print(f"{src['id']} seed {seed}: {'%.3f' % acc if acc is not None else 'n/a'} over {n} test tasks"
                  + ("" if rc == 0 else f"  (pipeline exit {rc})"), file=sys.stderr)

    lines = ["| source | " + " | ".join(f"seed {s}" for s in seeds) + " | mean |", "|---|" + "---|" * (len(seeds) + 1)]
    per_seed_means = []
    for seed in seeds:
        vals = [table[s][seed] for s in table if table[s][seed] is not None]
        per_seed_means.append(statistics.mean(vals) if vals else None)
    for sid, cells in table.items():
        vals = [v for v in cells.values() if v is not None]
        lines.append(f"| {sid} | " + " | ".join("%.3f" % cells[s] if cells[s] is not None else "n/a" for s in seeds)
                     + f" | {'%.3f' % statistics.mean(vals) if vals else 'n/a'} |")
    done = [m for m in per_seed_means if m is not None]
    lines.append("| **benchmark (macro over sources)** | " + " | ".join("%.3f" % m if m is not None else "n/a" for m in per_seed_means)
                 + f" | **{'%.3f' % statistics.mean(done) if done else 'n/a'}** |")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
