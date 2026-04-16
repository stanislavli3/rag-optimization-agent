"""Retrieval with vector / hybrid modes and optional cross-encoder reranking."""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _vector_search(index: dict, query: str, top_k: int) -> list[tuple[int, float]]:
    col = index["chroma"]
    # Both chroma and the fallback expose .query returning ids/distances
    try:
        res = col.query(query_texts=[query], n_results=top_k)
    except TypeError:
        res = col.query(query_embeddings=None, n_results=top_k)
    ids = res["ids"][0]
    dists = res.get("distances", [[0.0] * len(ids)])[0]
    scores = [1.0 - float(d) for d in dists]
    return list(zip([int(i) for i in ids], scores))


def _bm25_search(index: dict, query: str, top_k: int) -> list[tuple[int, float]]:
    bm = index["bm25"]
    scores = bm.get_scores(_tokenise(query))
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
    return ranked


def _rrf_fuse(
    rankings: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


def _cross_encoder_rerank(
    query: str, candidates: list[tuple[Any, float]], top_k: int
):
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, getattr(doc, "page_content", str(doc))) for doc, _ in candidates]
        scores = ce.predict(pairs).tolist()
    except Exception as e:
        logger.info("cross-encoder unavailable (%s); keeping input order", e)
        scores = [s for _, s in candidates]
    rescored = sorted(zip([c for c, _ in candidates], scores), key=lambda x: -x[1])
    return rescored[:top_k]


def retrieve(
    query: str,
    index: dict,
    top_k: int = 5,
    search_mode: str = "vector",
    reranker: str | None = None,
) -> list[tuple[Any, float]]:
    """Return a ranked list of (chunk, score) tuples."""
    chunks = index["chunks"]
    overfetch = top_k * 3 if reranker else top_k

    if search_mode == "hybrid":
        v = _vector_search(index, query, overfetch)
        b = _bm25_search(index, query, overfetch)
        merged = _rrf_fuse([v, b])[:overfetch]
    else:
        merged = _vector_search(index, query, overfetch)

    candidates = [(chunks[i], float(s)) for i, s in merged if 0 <= i < len(chunks)]

    if reranker == "cross-encoder":
        return _cross_encoder_rerank(query, candidates, top_k)
    return candidates[:top_k]
