"""Execute one RAG config end-to-end over a set of queries."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingest.chunker import chunk_documents
from src.pipeline.generator import generate_answer
from src.pipeline.indexer import build_index
from src.pipeline.retriever import retrieve
from src.testgen.llm import LLMLike, get_llm

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: dict = {
    "chunk_size": 512,
    "chunk_overlap": 0.15,
    "top_k": 5,
    "search_mode": "vector",
    "reranker": None,
    "prompt_style": "zero-shot",
    "embedding_model": "all-MiniLM-L6-v2",
}


def run_pipeline(
    config: dict,
    documents: list[Any],
    queries: list[dict],
    llm: LLMLike | None = None,
    persist_dir: str | Path = "data/chroma",
) -> list[dict]:
    """Run the RAG pipeline.

    queries items may be plain strings or dicts with ``question`` + optional
    ``ground_truth_answer`` / ``ground_truth_context`` / ``source_chunk_id`` fields.
    """
    cfg = {**DEFAULT_CONFIG, **config}
    llm = llm or get_llm()

    chunks = chunk_documents(documents, chunk_size=cfg["chunk_size"], overlap_ratio=cfg["chunk_overlap"])
    for c in chunks:
        c.metadata["chunk_overlap"] = cfg["chunk_overlap"]
    index = build_index(chunks, embedding_model_name=cfg["embedding_model"], persist_dir=persist_dir)
    logger.info("Indexed %d chunks (cache_key=%s)", len(chunks), index["cache_key"])

    results: list[dict] = []
    for q in queries:
        if isinstance(q, str):
            q = {"question": q}
        question = q["question"]
        retrieved = retrieve(
            question,
            index,
            top_k=cfg["top_k"],
            search_mode=cfg["search_mode"],
            reranker=cfg["reranker"],
        )
        contexts = [doc for doc, _ in retrieved]
        answer = generate_answer(question, contexts, llm, prompt_style=cfg["prompt_style"])
        results.append(
            {
                "question": question,
                "answer": answer,
                "retrieved_contexts": [getattr(c, "page_content", str(c)) for c in contexts],
                "retrieved_chunk_ids": [
                    getattr(c, "metadata", {}).get("chunk_id") for c in contexts
                ],
                "retrieval_scores": [float(s) for _, s in retrieved],
                "ground_truth_answer": q.get("ground_truth_answer"),
                "ground_truth_context": q.get("ground_truth_context"),
                "source_chunk_id": q.get("source_chunk_id"),
            }
        )
    return results
