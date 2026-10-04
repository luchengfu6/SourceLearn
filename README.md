<div align="center">

# 📚 SourceLearn

### *From Knowledge Access to Source Learning: Developing Source-Specific Competence*

[![arXiv](https://img.shields.io/badge/arXiv-2610.02150-b31b1b.svg)](https://arxiv.org/abs/2610.02150)
[![Project Page](https://img.shields.io/badge/Project_Page-sourcelearn.github.io-4f46e5.svg?logo=githubpages&logoColor=white)](https://sourcelearn.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-luchengfu6%2FSourceLearn-181717.svg?logo=github)](https://github.com/luchengfu6/SourceLearn)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API_compatible-412991.svg?logo=openai&logoColor=white)](https://platform.openai.com/)

[Lucheng Fu](https://luchengfu6.github.io/)¹, Kejing Xia¹, [Yiyang Wang](https://hello-diana.github.io/)¹, [Yiqiao Jin](https://ahren09.github.io)¹, Jinjin He¹,
Xiyuan Yang², Haoxin Liu¹, Ye Yu², [Haibo Jin](https://haibojin001.github.io/)², Yijia Xiao³,
Wenke Lee¹†, [B. Aditya Prakash](https://faculty.cc.gatech.edu/~badityap/)¹†, [Haohan Wang](https://haohanwang.github.io/)²†

¹ Georgia Institute of Technology · ² University of Illinois at Urbana-Champaign · ³ University of California, Los Angeles

<sub>† Corresponding authors</sub>

</div>

---

## 📖 Abstract

Large language model (LLM) agents increasingly rely on persistent external sources to solve sequences of knowledge-intensive tasks. Existing methods improve how source content is accessed and organized, while agent-memory systems preserve reusable knowledge from prior interactions, but repeated use of the same source is still largely treated as repeated access rather than an opportunity to progressively improve understanding of that source. We study *source learning*: developing reusable *source-specific competence* over a persistent authoritative source. We represent this competence with a persistent *source model* that captures reusable understanding of the source, including how its knowledge is structured, interpreted, and applied. To construct and progressively refine such models, we propose **SourceLearn**, which combines two complementary learning mechanisms. *Self-Directed Source Learning* identifies what remains incompletely understood and adaptively revisits the source, while *Task-Guided Source Learning* uses downstream experience to reveal local representational gaps and recurring needs in how source knowledge should be organized. In both cases, learning signals determine what should be reconsidered, while persistent updates are reconstructed from the authoritative source. Across five benchmarks and three LLM backends, **SourceLearn** achieves the best performance in **13 of 15 settings**, with gains of up to **22.6 points over Hybrid RAG** and substantial overall improvements over static source representations and experience-based memory baselines.

## ✨ Highlights

- 🎯 **Source learning as a problem**: repeated use of a persistent, authoritative source becomes the cumulative development of *source-specific competence*, represented by an explicit, persistent and revisable **source model** that complements direct source access
- 🧩 **Two complementary learning mechanisms**: Self-Directed Source Learning (Inspect → Study → Consolidate) and Task-Guided Source Learning (failure-guided local refinement + cross-task representation learning), both writing only source-grounded reconstructions
- 📈 **Strong empirical performance**: best in **13 of 15** benchmark × backbone settings over document QA, code QA, tool use and interactive environments, up to **+22.6 points** over Hybrid RAG

## 🧩 The SourceLearn Framework

SourceLearn maintains a source model `M` over an authoritative source `D` and develops it in three steps. Every stage follows one principle: learning signals determine *what should be reconsidered*, while persistent knowledge is *reconstructed from the source*.

```
M_0 = Construct(D; Π_0)        M_S = SelfLearn(D, M_0)        M_T = TaskLearn(D, M_S; 𝒢)
```

| Stage | What it does | Code |
| :--- | :--- | :--- |
| 🏗️ **1. Source Model Construction** | Source-aware, entity-centred reading under a generic modeling instruction `Π_0` yields a provisional model `M_0`: entities with their properties, mechanisms, governing conditions, procedures, exceptions, distinctions and relations. Large sources are partitioned into coherent regions first. | [`sourcelearn/draft/`](sourcelearn/draft/) |
| 🔍 **2. Self-Directed Source Learning** | One *Inspect–Study–Consolidate* cycle: reread every entity against its current representation, plan `Deepen` / `Connect` study actions with focused study questions, then consolidate the observations into the model (read-many, write-once). | [`sourcelearn/self_learning/`](sourcelearn/self_learning/) |
| 🎯 **3. Task-Guided Source Learning** | Diagnose what each guidance task requires from the source; *failure-guided local refinement* rereads the implicated region when a required claim is missing; *cross-task representation learning* aggregates recurring lessons into a representation policy `Π_1` and recalibrates the whole model under it. | [`sourcelearn/task_learning/`](sourcelearn/task_learning/) |
| 🧱 **Grounded reconstruction** | Shared by every stage: `Inspect → Reconstruct → GroundApply`. A rewrite is committed only when its content is entailed by source evidence and previously supported meaning is preserved. | [`sourcelearn/reconstruction/`](sourcelearn/reconstruction/) |
| ⚡ **Activation & answering** | At task time the solver sees hybrid retrieval `R(D, q)` next to the activated source model `A_B(M, q)` (the whole model when it fits the budget `B`, routed regions otherwise). | [`sourcelearn/source_model.py`](sourcelearn/source_model.py) · [`sourcelearn/answering/`](sourcelearn/answering/) |

> 📖 For the full formulation (the source model, the learning objective, the two mechanisms and the grounded reconstruction rule), see the [paper](https://arxiv.org/abs/2610.02150).

## 🚀 Installation

SourceLearn targets **Python 3.11+**; every LLM call goes through an OpenAI-compatible chat API.

```bash
# 1) Create and activate the env
conda create -n sourcelearn python=3.11 -y
conda activate sourcelearn

# 2) Install sourcelearn (pydantic, openai, rank-bm25, numpy, tiktoken)
pip install -e .

# optional: pyarrow, needed only to prepare NarrativeQA from its parquet release
pip install -e ".[datasets]"
```

> Full dependency list: [`requirements.txt`](requirements.txt) / [`pyproject.toml`](pyproject.toml).

### Set up API keys

```bash
cp .env.example .env
# then edit .env:
#   OPENAI_API_KEY=sk-...                  # backbone, judge and embeddings
#   (optional) OPENROUTER_API_KEY=...      # for vendor-prefixed model ids such as deepseek/deepseek-v4.1-flash
#   (optional) SOURCELEARN_REASONING_EFFORT=medium
```

Every run loads `./.env` at startup (override the location with `SOURCELEARN_ENV_FILE`), so nothing has to be exported in your shell.

## ⚡ Quick Start

### Option 1 — Learn your own source

A source is a directory of documents (`.md`, `.rst`, `.txt`; `--adapter docs`) or a Python repository (`--adapter code`, optionally `--region <package dir>`). Tasks are a JSON list:

```json
[
  {"qid": "q1", "question": "...", "answer": "reference answer"},
  {"qid": "q2", "question": "...", "options": ["A", "B", "C"], "answer": 1},
  {"qid": "q3", "question": "...", "answer": "...", "gold_docs": ["guide/billing.md"]}
]
```

`options` + an integer `answer` makes a multiple-choice task; `gold_docs` (optional) hints where the supporting evidence lives.

```bash
python -m sourcelearn.pipeline \
    --workspace <source dir> --questions <questions.json> --adapter docs \
    --model gpt-5.6-luna --judge-model gpt-5.6-luna \
    --out runs/my_source --seeds 0,1,2
```

The pipeline always runs the whole method in order — construction → self-directed learning → one task-guided learning run per seed, each on a random 30 % / 70 % guidance/test split of the questions (pass `--split-dir <dir>` holding `seed<N>.json` files `{"train": [...], "test": [...]}` to use given splits). A stage whose output exists is skipped, so an interrupted run resumes with the same command. `--model` is the backbone of every stage; `--judge-model` judges answers (default: the backbone). Run `python -m sourcelearn.pipeline --help` for the full flag list.

### Option 2 — Reproduce a benchmark of the paper

Prepare a benchmark once (see [Datasets](#-datasets)), then run every source of it:

```bash
python -m sourcelearn.datasets multidoc2dial --raw <multidoc2dial dir> --out data/multidoc2dial

python examples/run_benchmark.py --index data/multidoc2dial/index.json \
    --model gpt-5.6-luna --judge-model gpt-5.6-luna --out runs/multidoc2dial --seeds 0,1,2
```

`examples/run_benchmark.py` runs the pipeline on each source with the paper's frozen guidance/test split of every seed, then reports the test accuracy macro-averaged over sources and seeds in `runs/multidoc2dial/summary.md`. `--sources dmv,ssa` restricts the run to some sources.

## 📊 Datasets

The paper evaluates SourceLearn on **5 benchmarks** spanning document, code and API sources. Each document collection, story, repository or API library is one persistent source; guidance and test tasks are disjoint (three fixed 30 % / 70 % splits, except AppWorld, which keeps its official partitions). `python -m sourcelearn.datasets <benchmark>` turns the official release into pipeline inputs and copies the frozen splits of the paper next to them; the exact sources, task pools and splits of the paper are recorded in [`sourcelearn/datasets/frozen/`](sourcelearn/datasets/frozen/).

| Benchmark | Source type | Sources · tasks | Get the raw data | Prepare |
| :--- | :--- | :--- | :--- | :--- |
| **MultiDoc2Dial** | government document collections | 4 domains (92–149 docs) · 60 q each | official zip from [IBM/multidoc2dial](https://github.com/IBM/multidoc2dial), unzipped | `python -m sourcelearn.datasets multidoc2dial --raw <dir> --out data/multidoc2dial` |
| **NarrativeQA** | full stories (6 movie scripts, 6 books) | 12 stories · 27–30 q each | `huggingface-cli download deepmind/narrativeqa --repo-type dataset --include "data/test-*" --local-dir <dir>` | `python -m sourcelearn.datasets narrativeqa --raw <dir> --out data/narrativeqa` |
| **SWE-QA** | Python repositories at pinned commits | 6 repos (streamlink, sphinx, xarray, pytest, flask, conan) · 48 q each | `git clone https://github.com/peng-weihan/SWE-QA-Bench` (repositories are cloned at their pinned commits by the adapter) | `python -m sourcelearn.datasets sweqa --bench <SWE-QA-Bench> --out data/sweqa` |
| **APIBench** | complete API documentation of a model hub | TorchHub 94 docs · 186 q, HuggingFace 936 docs · 100 q, TensorFlow Hub 696 docs · 100 q | `git clone https://github.com/ShishirPatil/gorilla && git -C gorilla checkout 6ea5797` | `python -m sourcelearn.datasets apibench --gorilla <gorilla> --out data/apibench` |
| **AppWorld** | static documentation of 457 APIs (11 apps) | 1 source · 90 train / 168 test-normal tasks | `pip install appworld && appworld install && appworld download data` | `python -m sourcelearn.datasets appworld --data-root <root> --out data/appworld` |

Only documentation enters a workspace: task text, reference answers and any per-task state stay in `questions.json`. The prepared layout is the same for every benchmark:

```
data/<benchmark>/
    index.json                    one row per source: workspace, questions, adapter, region, split_dir
    <source>/workspace/           the source D (markdown documents, or the pinned code checkout)
    <source>/questions.json       the tasks
    <source>/splits/seed{0,1,2}.json   the paper's frozen guidance/test splits (official.json for AppWorld)
```

> ⚠️ **Official evaluators.** The QA benchmarks are scored by an LLM judge against the reference answer, exactly as the pipeline does. The paper scores **APIBench** with Gorilla's official AST evaluator (functionality accuracy) and **AppWorld** by running the official interactive environment and its evaluator (task goal completion). This repository prepares the sources and tasks of both and learns the source model from them, but does not bundle those evaluators; `summary.md` on these two benchmarks reports the judge-based proxy, not the paper's metric.

## 🔧 Hyperparameters

All numeric knobs live in one place, [`sourcelearn/reconstruction/defaults.py`](sourcelearn/reconstruction/defaults.py), each tagged with the principle it stands in for.

| Hyperparameter | Value | Notes |
| :--- | :--- | :--- |
| `M_BUDGET` (`B`) | `24000` tokens | Activation budget: the whole source model when it fits, routed regions otherwise. |
| `STUDY_K` (`K`) | `20` | Adaptive study actions (`Deepen` / `Connect`) per self-directed cycle; the paper runs one cycle. |
| `RAW_TOP_K` | `8` | Retrieved source elements shown next to the model (BM25 + dense, reciprocal rank fusion). |
| train fraction / seeds | `0.3` / `0,1,2` | Guidance/test split per seed; results are the mean over seeds. |
| `LESSON_MIN_SUPPORT` / `POLICY_MAX_ITEMS` | `2` / `8` | A representation preference needs two supporting tasks to enter `Π_1`, which holds at most eight. |
| `SOURCELEARN_REASONING_EFFORT` | `medium` | Reasoning effort of the backbone and judge (DeepSeek-V4.1-Flash: `high` on the QA benchmarks). |

> 📝 **Paper setup**: backbones = `gpt-5.6-luna`, `gpt-oss-120b` (served locally with vLLM) and `deepseek-v4.1-flash`; within a configuration one backbone performs every LLM operation of every method. Embeddings = `text-embedding-3-large` for every method. QA judge = `gpt-5.6-luna` for every method and backbone. SourceLearn itself needs no GPU.

## 🧪 Output

```
runs/<source>/
    draft/m0_induced.json | m0_union.json   M_0, the constructed source model
    self_learning/m_self.json               M_S, after Self-Directed Source Learning
    task_s<seed>/m_task.json                M_T, the learned source model: a JSON list of knowledge units
                                            (statement, role, scope, conditions, support anchors into D)
    task_s<seed>/eval/results.jsonl         one row per test task (status, correct, tokens, calls)
    task_s<seed>/summary.md                 accuracy of M_T on the test tasks, training and policy telemetry
    logs/<stage>.log                        the arguments and output of every stage
```

A row whose `status` is not `VALID` (a harness failure) is reported and never counted as a wrong answer.

## 📁 Project layout

```
sourcelearn/
    pipeline.py          the only method entry point: construction -> self-directed -> task-guided learning
    draft/               M_0: reading brief, partition of large sources into regions, entity-centred reading
    self_learning/       M_S: Inspect -> Study (Deepen / Connect) -> Consolidate
    task_learning/       M_T: diagnose E*_t and C_t, failure-guided local refinement, lessons -> Π_1, recalibration
    reconstruction/      grounded reconstruction shared by every stage (Inspect / Reconstruct / GroundApply); defaults.py
    answering/           the task solver F(q, R(D, q), A_B(M, q)) and the judge
    source_model.py      knowledge units and source-model activation A_B(M, q)
    retrieval.py         hybrid retrieval R(D, q): BM25 + dense with reciprocal rank fusion
    core/                LLM client (OpenAI-compatible), source elements, tracing
    datasets/            the five benchmarks of the paper as pipeline inputs; frozen/ holds the paper's sources, task pools and splits
examples/
    run_benchmark.py     run every source of a prepared benchmark and aggregate the test accuracy
```

## 📝 Citation

If you find SourceLearn useful in your research, please cite:

```bibtex
@article{fu2026knowledge,
  title={From Knowledge Access to Source Learning: Developing Source-Specific Competence},
  author={Fu, Lucheng and Xia, Kejing and Wang, Yiyang and Jin, Yiqiao and He, Jinjin and Yang, Xiyuan and Liu, Haoxin and Yu, Ye and Jin, Haibo and Xiao, Yijia and others},
  journal={arXiv preprint arXiv:2610.02150},
  year={2026}
}
```

## 🙏 Acknowledgements

SourceLearn is evaluated on the following benchmarks and builds on the following open-source projects:
[MultiDoc2Dial](https://github.com/IBM/multidoc2dial) ·
[NarrativeQA](https://github.com/google-deepmind/narrativeqa) ·
[SWE-QA](https://github.com/peng-weihan/SWE-QA-Bench) ·
[Gorilla APIBench](https://github.com/ShishirPatil/gorilla) ·
[AppWorld](https://github.com/StonyBrookNLP/appworld) ·
[OpenAI Python SDK](https://github.com/openai/openai-python) ·
[rank-bm25](https://github.com/dorianbrown/rank_bm25) ·
[tiktoken](https://github.com/openai/tiktoken) ·
[pydantic](https://github.com/pydantic/pydantic).

## ⚖️ License

This project is released under the [MIT License](LICENSE).
