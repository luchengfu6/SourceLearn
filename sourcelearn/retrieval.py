"""Retrieval over the fingerprinted source elements. Each retriever exposes
`retrieve(question, k) -> list[SourceElement]` and plugs into
SourceModelSession(retriever=...).

- HybridRetriever   : BM25 top-30 + dense top-30 (text-embedding-3-large),
                      RRF c=60, k=8, no query rewriting. The raw excerpts the
                      answerer sees next to the source model come from it.

Per-file cap: `file_cap` on the retriever and `build_retriever(..., file_cap=)`;
0 disables it (3 on code repositories, 0 on document collections).
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from sourcelearn.core.llm import LLMClient
from sourcelearn.core.schema import SourceElement
from sourcelearn.core.elements import tokens as _tokens

CAND_K = 30          # candidates per ranker before fusion
RRF_C = 60           # reciprocal-rank-fusion constant (standard)
PER_FILE_CAP = 3     # default per-file cap of the hybrid retriever; 0 = disabled
EMBED_MODEL = "text-embedding-3-large"


def _cosine_top(qv: np.ndarray, mat: np.ndarray, k: int) -> list[int]:
    sims = mat @ qv / (np.linalg.norm(mat, axis=1) * np.linalg.norm(qv) + 1e-9)
    return [int(i) for i in np.argsort(-sims)[:k]]


def _cap_per_file(els: list[SourceElement], k: int, cap: int | None = PER_FILE_CAP) -> list[SourceElement]:
    """The first k elements, at most `cap` per file; cap 0 / None = no cap."""
    out, seen = [], {}
    for e in els:
        if cap and seen.get(e.file, 0) >= cap:
            continue
        seen[e.file] = seen.get(e.file, 0) + 1
        out.append(e)
        if len(out) >= k:
            break
    return out


def _bm25_top(items: list, texts: list[str], query: str, k: int,
              per_file_cap: dict | None = None) -> list:
    corpus = [_tokens(t) for t in texts]
    q = _tokens(query)
    if not q or not any(corpus):
        return []
    scores = BM25Okapi(corpus).get_scores(q)
    picked, seen_files = [], {}
    for s, item in sorted(zip(scores, items), key=lambda x: -x[0]):
        if s <= 0 and picked:
            break
        if per_file_cap is not None:
            f = per_file_cap["key"](item)
            if seen_files.get(f, 0) >= per_file_cap["cap"]:
                continue
            seen_files[f] = seen_files.get(f, 0) + 1
        picked.append(item)
        if len(picked) >= k:
            break
    return picked


class HybridRetriever:
    def __init__(self, elements: list[SourceElement], embed, file_cap: int | None = PER_FILE_CAP):
        self.elements = elements
        self.texts = [f"{e.name} {e.excerpt}" for e in elements]
        self.embed = embed
        self.file_cap = file_cap              # per-file cap on the fused ranking; 0 = disabled
        self.mat = embed(self.texts)          # N x D, cached on disk

    def retrieve(self, question: str, k: int) -> list[SourceElement]:
        lex = _bm25_top(self.elements, self.texts, question, CAND_K)
        den = [self.elements[i] for i in
               _cosine_top(self.embed([question])[0], self.mat, CAND_K)]
        score: dict[str, float] = {}
        by_id = {}
        for ranked in (lex, den):
            for r, e in enumerate(ranked):
                score[e.element_id] = score.get(e.element_id, 0) + 1 / (RRF_C + r + 1)
                by_id[e.element_id] = e
        fused = [by_id[i] for i in sorted(score, key=lambda i: -score[i])]
        return _cap_per_file(fused, k, self.file_cap)


def build_retriever(name: str, elements: list[SourceElement], llm: LLMClient,
                    file_cap: int | None = PER_FILE_CAP):
    """`name`: "rag_hybrid" (embeddings through `llm`, cached by the client).
    `file_cap`: per-file cap (0 = disabled)."""
    if name == "rag_hybrid":
        return HybridRetriever(elements, lambda texts: llm.embed(texts, model=EMBED_MODEL), file_cap)
    raise ValueError(name)
