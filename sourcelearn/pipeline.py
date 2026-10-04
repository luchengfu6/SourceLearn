"""SourceLearn end to end on one source, one model for every stage.

  draft   Draft Source Modeling. docs: brief -> partition (sources of at least
          PARTITION_MIN_CHARS) -> one induce per block -> union (m0_union.json);
          smaller docs sources: one induce. code: one induce, every public symbol an entity.
  self    Self-Directed Source Learning: inspection -> adaptive study ->
          consolidation (self_learning/m_self.json).
  task    Task-Guided Source Learning, one run per seed (task_s<seed>/); the
          shared E* cache of the source is computed once before the seeds.

This is the only entry point: the stage modules have no command line of their own
and always run in this order, so M0 and M_S exist only on the way to M_T. Each
stage runs as a subprocess and is skipped when its output exists, so a rerun
resumes; its arguments and output go to <out>/logs/<stage>.log.

Usage:
  python -m sourcelearn.pipeline --workspace <source> --questions <questions.json> \\
      --adapter docs --model <model> --judge-model <judge> --out <run> [--seeds 0,1,2]
  (--adapter code [--region <package>] for a code repository; --split-dir <dir>
   with seed<N>.json train/test files instead of a random split per seed)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sourcelearn.core.workspace import Workspace
from sourcelearn.draft.ids import unique_ids

PARTITION_MIN_CHARS = 100_000     # docs sources below this are drafted in one induce
BLOCK_PARALLEL = 4                # block induces run at a time
FILE_CAP = {"code": 3, "docs": 0}  # hybrid retrieval: excerpts per file
# the stage modules are not runnable themselves; each is entered here, in a subprocess
_STAGE = "import importlib, sys; sys.exit(importlib.import_module(sys.argv[1]).run_stage(sys.argv[2:]))"


def union_blocks(by_block: dict[str, list[dict]]) -> list[dict]:
    """Concatenate the per-block drafts into one model with unique unit ids.

    Each block's induce numbers its units within its own region, so two blocks
    can produce the same id for different units. Ids are namespaced with the
    block name on collision only (a single-block source keeps its ids), and
    `derived_from` references are remapped with the ids of their own block."""
    seen: set[str] = set()
    out: list[dict] = []
    for name, units in by_block.items():
        ren = {i: f"{name}/{i}" for i in {u["unit_id"] for u in units} & seen}
        for u in units:
            u = dict(u)
            u["unit_id"] = ren.get(u["unit_id"], u["unit_id"])
            if u.get("derived_from"):
                u["derived_from"] = [ren.get(d, d) for d in u["derived_from"]]
            out.append(u)
        seen |= {u["unit_id"] for u in out[-len(units):]}
    unique_ids(out)          # a block may also repeat an id inside itself
    return out


def source_chars(workspace: str) -> int:
    ws = Workspace(workspace)
    return sum(len(ws.read_text(f)) for f in ws.list_files())


class Pipeline:
    def __init__(self, args):
        self.a = args
        self.out = Path(args.out)
        self.logs = self.out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.py = sys.executable

    def stage(self, name: str, module: str, flags: list, marker: Path) -> bool:
        """Run the stage `module` with `flags` in a subprocess unless `marker` exists; False on failure."""
        if marker.exists():
            print(f"[skip] {name}: {marker} exists", file=sys.stderr)
            return True
        cmd = [self.py, "-c", _STAGE, module, *map(str, flags)]
        print(f"[run] {name} -> {self.logs / name}.log", file=sys.stderr)
        with open(self.logs / f"{name}.log", "w") as fh:
            fh.write(f"$ {module} {' '.join(map(str, flags))}\n\n")
            fh.flush()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"[{'ok' if rc == 0 else f'rc={rc}'}] {name}", file=sys.stderr)
        return rc == 0 and marker.exists()

    # -- Draft Source Modeling ---------------------------------------------------------
    def draft(self) -> Path | None:
        """The draft model M0 (m0_union.json or m0_induced.json), or None."""
        a, C = self.a, self.out / "draft"
        for name in ("m0_union.json", "m0_induced.json"):
            if (C / name).exists():
                return C / name
        common = ["--workspace", a.workspace, "--model", a.model, "--workers", a.workers]
        if a.adapter == "code":
            flags = common + ["--adapter", "code", "--out", C] + (["--region", a.region] if a.region else [])
            return C / "m0_induced.json" if self.stage("draft_induce", "sourcelearn.draft.induce", flags, C / "m0_induced.json") else None
        brief = C / "brief" / "brief.md"
        if not self.stage("draft_brief", "sourcelearn.draft.brief",
                          ["--workspace", a.workspace, "--out", C / "brief", "--model", a.model], brief):
            return None
        induce = common + ["--adapter", "docs", "--brief", brief]
        if source_chars(a.workspace) < PARTITION_MIN_CHARS:
            return C / "m0_induced.json" if self.stage("draft_induce", "sourcelearn.draft.induce", induce + ["--out", C],
                                                       C / "m0_induced.json") else None
        part = C / "partition.json"
        if not self.stage("draft_partition", "sourcelearn.draft.partition",
                          ["--workspace", a.workspace, "--brief", brief, "--model", a.model, "--out", part], part):
            return None
        blocks = json.loads(part.read_text())["blocks"]

        def one(b: dict) -> bool:
            files = C / f"files_{b['name']}.json"
            files.write_text(json.dumps(b["files"], indent=1))
            return self.stage(f"draft_induce_{b['name']}", "sourcelearn.draft.induce",
                              induce + ["--files", files, "--token-budget", b.get("budget_tokens", 0),
                                        "--out", C / "blocks" / b["name"]],
                              C / "blocks" / b["name"] / "m0_induced.json")
        with ThreadPoolExecutor(max_workers=max(1, min(BLOCK_PARALLEL, len(blocks)))) as pool:
            if not all(pool.map(one, blocks)):
                return None
        union = union_blocks({b["name"]: json.loads((C / "blocks" / b["name"] / "m0_induced.json").read_text())
                              for b in blocks})
        (C / "m0_union.json").write_text(json.dumps(union, indent=1))
        return C / "m0_union.json"

    def bundle_flags(self) -> list:
        C = self.out / "draft"
        return ((["--partition", C / "partition.json"] if (C / "partition.json").exists() else [])
                + (["--brief", C / "brief" / "brief.md"] if (C / "brief" / "brief.md").exists() else []))

    # -- Self-Directed Source Learning -------------------------------------------------
    def self_learning(self, m0: Path) -> Path | None:
        a, S = self.a, self.out / "self_learning"
        ok = self.stage("self_learning", "sourcelearn.self_learning.run",
                        ["--workspace", a.workspace, "--m0", m0, "--out", S, "--model", a.model, "--workers", a.workers]
                        + self.bundle_flags(),
                        S / "FINISHED")
        return S / "m_self.json" if ok else None

    # -- Task-Guided Source Learning ---------------------------------------------------
    def task_learning(self, m_self: Path) -> list[str]:
        a = self.a
        common = ["--workspace", a.workspace, "--questions", a.questions, "--m-self", m_self,
                  "--answer-model", a.model, "--strong-model", a.model, "--judge-model", a.judge_model or a.model,
                  "--file-cap", FILE_CAP[a.adapter], "--cache-dir", self.out / "cache"] + self.bundle_flags()
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        runs = []
        for seed in seeds:
            split = Path(a.split_dir) / f"seed{seed}.json" if a.split_dir else None
            if split is not None and not split.exists():
                return [f"seed{seed}: {split} not found"]
            runs.append((seed, ["--split-file", split] if split else ["--seed", seed]))
        # E* depends only on (question, source, model): computed once, shared by every seed.
        E = self.out / "evidence"
        if not self.stage("evidence", "sourcelearn.task_learning.run", common + ["--out", E, "--evidence-only"],
                          E / "evidence.json"):
            return ["evidence: FAILED"]

        def one(run) -> str:
            seed, split = run
            T = self.out / f"task_s{seed}"
            ok = self.stage(f"task_s{seed}", "sourcelearn.task_learning.run", common + split + ["--out", T], T / "summary.md")
            return f"task_s{seed}: {'ok -> ' + str(T / 'summary.md') if ok else 'FAILED'}"
        with ThreadPoolExecutor(max_workers=max(1, len(runs))) as pool:
            return list(pool.map(one, runs))

    def run(self) -> int:
        m0 = self.draft()
        if m0 is None:
            print("draft: failed (a rerun resumes it)", file=sys.stderr)
            return 1
        m_self = self.self_learning(m0)
        if m_self is None:
            print("self-learning: failed (a rerun resumes it)", file=sys.stderr)
            return 1
        msgs = self.task_learning(m_self)
        print("\n".join(msgs), file=sys.stderr)
        return 0 if all(": ok" in m for m in msgs) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", required=True, help="the source: a directory of documents or a code repository")
    p.add_argument("--questions", required=True, help="questions.json of the source")
    p.add_argument("--adapter", choices=("docs", "code"), required=True)
    p.add_argument("--region", default="", help="code: restrict the draft to one package directory")
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True, help="the backbone of every stage (reasoning effort: SOURCELEARN_REASONING_EFFORT)")
    p.add_argument("--judge-model", default="", help="answer judge (default: --model)")
    p.add_argument("--workers", type=int, default=6, help="parallel LLM calls")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--split-dir", default="", help="directory of seed<N>.json train/test files (default: random 30/70 split per seed)")
    return Pipeline(p.parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
