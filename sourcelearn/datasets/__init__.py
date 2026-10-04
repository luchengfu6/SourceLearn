"""Dataset adapters: the five benchmarks of the paper as SourceLearn inputs.

Every adapter turns one benchmark release into the generic input of
`sourcelearn.pipeline` -- one directory per persistent source holding

    workspace/        the source D (markdown documents, or a pinned code checkout)
    questions.json    the tasks: [{"qid", "question", "answer", ...}]
    splits/           the paper's frozen guidance/test splits (seed0..2.json, or official.json)

and an `index.json` that `examples/run_benchmark.py` iterates over. Only the
documentation of a benchmark ever enters a workspace; task text, reference
answers and any per-task state stay in questions.json.

    python -m sourcelearn.datasets multidoc2dial --raw <multidoc2dial dir>  --out data/multidoc2dial
    python -m sourcelearn.datasets narrativeqa   --raw <deepmind/narrativeqa dir> --out data/narrativeqa
    python -m sourcelearn.datasets sweqa         --bench <SWE-QA-Bench checkout> --out data/sweqa
    python -m sourcelearn.datasets apibench      --gorilla <gorilla checkout> --out data/apibench
    python -m sourcelearn.datasets appworld      --data-root <appworld root>  --out data/appworld

The frozen manifests under `frozen/` (story ids, task pools, repository
commits, seed splits) reproduce the sources and splits of the paper.
"""
from __future__ import annotations

BENCHMARKS = {
    "multidoc2dial": {"module": "sourcelearn.datasets.multidoc2dial", "adapter": "docs",
                      "paper": "MultiDoc2Dial (Feng et al., 2021): 4 government document collections, 60 questions each"},
    "narrativeqa": {"module": "sourcelearn.datasets.narrativeqa", "adapter": "docs",
                    "paper": "NarrativeQA (Kocisky et al., 2018): 12 full stories of the test split, 27-30 questions each"},
    "sweqa": {"module": "sourcelearn.datasets.sweqa", "adapter": "code",
              "paper": "SWE-QA (Peng et al., 2026): 6 Python repositories at pinned commits, 48 questions each"},
    "apibench": {"module": "sourcelearn.datasets.apibench", "adapter": "docs",
                 "paper": "APIBench (Patil et al., 2024): the complete API documentation of 3 model hubs, 186/100/100 tasks"},
    "appworld": {"module": "sourcelearn.datasets.appworld", "adapter": "docs",
                 "paper": "AppWorld (Trivedi et al., 2024): the static documentation of 457 APIs, official train/test-normal tasks"},
}

__all__ = ["BENCHMARKS"]
