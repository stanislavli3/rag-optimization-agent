"""Build ChromaDB + BM25 indices from chunks, with a per-(size,overlap,model) cache.

Index objects are returned as a dict so retriever + runner can stay decoupled
from specific vector-store APIs.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INDEX_CACHE: dict[str, dict] = {}


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _cache_key(chunks: list[Any], embedding_model_name: str) -> str:
    sizes = {getattr(c, "metadata", {}).get("chunk_size_used") for c in chunks}
    overlap = {getattr(c, "metadata", {}).get("chunk_overlap") for c in chunks}
    sample = "".join(getattr(c, "page_content", "")[:200] for c in chunks[:5])
    h = hashlib.md5(sample.encode()).hexdigest()[:10]
    return f"{embedding_model_name}|sz{sizes}|ov{overlap}|n{len(chunks)}|{h}"


def _build_bm25(chunks: list[Any]) -> Any:
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except Exception:
        return _FallbackBM25([_tokenise(getattr(c, "page_content", "")) for c in chunks])
    tokenised = [_tokenise(getattr(c, "page_content", "")) for c in chunks]
    return BM25Okapi(tokenised)


class _FallbackBM25:
    """Tiny TF-based ranker used when rank_bm25 isn't installed."""

    def __init__(self, docs: list[list[str]]) -> None:
        self.docs = docs

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        q = set(query)
        for d in self.docs:
            s = sum(1 for t in d if t in q)
            scores.append(float(s) / (len(d) or 1))
        return scores


def _build_chroma(chunks: list[Any], embedding_model_name: str, persist_dir: str | Path):
    try:
        import chromadb  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return _FallbackVectorIndex(chunks, embedding_model_name)

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    name = f"col_{hashlib.md5((embedding_model_name + str(len(chunks))).encode()).hexdigest()[:10]}"
    collection = client.get_or_create_collection(name=name)
    # Only (re)embed if collection empty
    if collection.count() < len(chunks):
        model = SentenceTransformer(embedding_model_name)
        texts = [getattr(c, "page_content", "") for c in chunks]
        embs = model.encode(texts, show_progress_bar=False).tolist()
        ids = [str(i) for i in range(len(chunks))]
        metas = [dict(getattr(c, "metadata", {}) or {}) for c in chunks]
        collection.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
    return collection


class _FallbackVectorIndex:
    """Deterministic BoW vector index used when chroma/sentence-transformers unavailable."""

    def __init__(self, chunks: list[Any], _model_name: str) -> None:
        import numpy as np  # type: ignore

        self.chunks = chunks
        self.texts = [getattr(c, "page_content", "") for c in chunks]
        vocab = sorted({t for tx in self.texts for t in _tokenise(tx)})
        self.vocab = vocab
        self.index = {v: i for i, v in enumerate(vocab)}
        vecs = np.zeros((len(self.texts), len(vocab)), dtype=float)
        for i, tx in enumerate(self.texts):
            for tok in _tokenise(tx):
                vecs[i, self.index[tok]] += 1
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vecs = vecs / norms

    def query(self, query_texts=None, n_results: int = 5, **_):
        import numpy as np  # type: ignore

        q = query_texts[0] if isinstance(query_texts, list) else query_texts
        qv = np.zeros(len(self.vocab), dtype=float)
        for tok in _tokenise(q):
            if tok in self.index:
                qv[self.index[tok]] += 1
        nq = np.linalg.norm(qv) or 1
        qv = qv / nq
        sims = self.vecs @ qv
        order = list(np.argsort(-sims)[:n_results])
        return {
            "ids": [[str(i) for i in order]],
            "documents": [[self.texts[i] for i in order]],
            "distances": [[float(1 - sims[i]) for i in order]],
        }

    def count(self) -> int:
        return len(self.texts)


def build_index(
    chunks: list[Any],
    embedding_model_name: str,
    persist_dir: str | Path,
    use_cache: bool = True,
) -> dict:
    key = _cache_key(chunks, embedding_model_name)
    if use_cache and key in _INDEX_CACHE:
        logger.info("Reusing cached index for %s", key)
        return _INDEX_CACHE[key]

    vector_index = _build_chroma(chunks, embedding_model_name, persist_dir)
    bm25 = _build_bm25(chunks)
    bundle = {"chroma": vector_index, "bm25": bm25, "chunks": chunks, "cache_key": key}
    _INDEX_CACHE[key] = bundle
    return bundle


def clear_index_cache() -> None:
    _INDEX_CACHE.clear()
