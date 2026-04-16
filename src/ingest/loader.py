"""Load PDF / MD / TXT documents from a directory into LangChain Documents."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class _Document:
    """Minimal stand-in when langchain isn't installed. Mirrors the LC interface."""

    def __init__(self, page_content: str, metadata: dict | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


def _lc_document_cls():
    try:
        from langchain_core.documents import Document  # type: ignore
        return Document
    except Exception:
        return _Document


def _load_pdf(path: Path, Doc) -> list:
    # Preferred: langchain-community PyPDFLoader (keeps per-page metadata).
    try:
        from langchain_community.document_loaders import PyPDFLoader  # type: ignore
        pages = PyPDFLoader(str(path)).load()
        for i, d in enumerate(pages):
            d.metadata.update(
                {"source_file": path.name, "page_number": i, "file_type": "pdf"}
            )
        return [d for d in pages if (d.page_content or "").strip()]
    except Exception as e:
        logger.warning("PyPDFLoader failed for %s: %s; trying pypdf direct", path, e)

    # Fallback: raw pypdf — works without langchain-community installed.
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        out = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            out.append(
                Doc(
                    page_content=text,
                    metadata={
                        "source_file": path.name,
                        "page_number": i,
                        "file_type": "pdf",
                    },
                )
            )
        if out:
            return out
        logger.warning("PDF %s yielded 0 pages with extractable text", path)
    except Exception as e:
        logger.warning("pypdf direct load failed for %s: %s", path, e)
    return []


def _load_md(path: Path, Doc) -> list:
    try:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader  # type: ignore
        docs = UnstructuredMarkdownLoader(str(path)).load()
    except Exception:
        docs = [Doc(page_content=path.read_text(encoding="utf-8", errors="replace"))]
    for d in docs:
        d.metadata = {**(d.metadata or {}), "source_file": path.name, "page_number": 0, "file_type": "md"}
    return docs


def _load_txt(path: Path, Doc) -> list:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return []
    return [Doc(page_content=text, metadata={"source_file": path.name, "page_number": 0, "file_type": "txt"})]


def load_documents(dir_path: str | Path) -> list:
    """Load all supported docs under dir_path. Skips unreadable files with a warning."""
    Doc = _lc_document_cls()
    root = Path(dir_path)
    if not root.exists():
        logger.warning("Docs dir does not exist: %s", root)
        return []

    out: list = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".pdf":
            out.extend(_load_pdf(p, Doc))
        elif suf in {".md", ".markdown"}:
            out.extend(_load_md(p, Doc))
        elif suf == ".txt":
            out.extend(_load_txt(p, Doc))
        else:
            logger.info("Skipping unsupported file: %s", p.name)
    logger.info("Loaded %d documents from %s", len(out), root)
    return out


def iter_documents(dir_path: str | Path) -> Iterable:
    for d in load_documents(dir_path):
        yield d
