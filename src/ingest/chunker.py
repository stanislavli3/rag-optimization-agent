"""Chunk documents with configurable size and overlap."""
from __future__ import annotations

import uuid
from typing import Any


def _splitter(chunk_size: int, overlap: int):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _naive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(1, chunk_size - overlap)
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + chunk_size])
        if i + chunk_size >= len(text):
            break
        i += step
    return out


def _doc_cls():
    try:
        from langchain_core.documents import Document  # type: ignore
        return Document
    except Exception:
        from src.ingest.loader import _Document  # type: ignore
        return _Document


def chunk_documents(
    docs: list[Any],
    chunk_size: int = 512,
    overlap_ratio: float = 0.15,
) -> list[Any]:
    """Split LangChain-style Documents into overlapping chunks with metadata."""
    overlap = max(0, int(chunk_size * overlap_ratio))
    Doc = _doc_cls()

    try:
        splitter = _splitter(chunk_size, overlap)
        use_lc = True
    except Exception:
        splitter = None
        use_lc = False

    chunks: list[Any] = []
    for parent in docs:
        text = getattr(parent, "page_content", "") or ""
        base_meta = dict(getattr(parent, "metadata", {}) or {})
        parent_id = base_meta.get("source_file", "doc") + f":{base_meta.get('page_number', 0)}"

        if use_lc:
            parts = splitter.split_text(text)
        else:
            parts = _naive_split(text, chunk_size, overlap)

        for idx, piece in enumerate(parts):
            meta = {
                **base_meta,
                "chunk_index": idx,
                "chunk_size_used": chunk_size,
                "parent_doc": parent_id,
                "chunk_id": uuid.uuid4().hex[:12],
            }
            chunks.append(Doc(page_content=piece, metadata=meta))
    return chunks
