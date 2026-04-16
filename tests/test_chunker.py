"""Tests for the chunker: chunk counts shrink as chunk_size grows; metadata intact."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.chunker import chunk_documents  # noqa: E402
from src.ingest.loader import _Document  # noqa: E402


def _fake_docs(n: int = 3, size: int = 2000) -> list:
    return [
        _Document(
            page_content=("The quick brown fox jumps over the lazy dog. " * 60)[:size],
            metadata={"source_file": f"doc_{i}.txt", "page_number": 0, "file_type": "txt"},
        )
        for i in range(n)
    ]


def test_chunk_count_monotone_decreasing():
    docs = _fake_docs()
    c256 = chunk_documents(docs, chunk_size=256, overlap_ratio=0.10)
    c512 = chunk_documents(docs, chunk_size=512, overlap_ratio=0.10)
    c1024 = chunk_documents(docs, chunk_size=1024, overlap_ratio=0.10)
    assert len(c256) > len(c512) > len(c1024), (len(c256), len(c512), len(c1024))


def test_metadata_fields_present():
    docs = _fake_docs()
    chunks = chunk_documents(docs, chunk_size=512, overlap_ratio=0.10)
    required = {"source_file", "chunk_index", "chunk_size_used", "parent_doc", "chunk_id"}
    for c in chunks:
        missing = required - set(c.metadata)
        assert not missing, f"Missing metadata keys: {missing}"


if __name__ == "__main__":
    test_chunk_count_monotone_decreasing()
    test_metadata_fields_present()
    print("chunker tests passed")
