"""Project-wide configuration: paths, models, search space, BFTS settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


class Stage(IntEnum):
    PRELIMINARY = 1
    BASELINE = 2
    EXPLORATION = 3
    ABLATION = 4


RAG_SEARCH_SPACE: dict = {
    "chunk_size": [256, 512, 1024],
    "chunk_overlap": [0.10, 0.20],
    "top_k": [3, 5, 10],
    "reranker": [None, "cross-encoder"],
    "search_mode": ["vector", "hybrid"],
    "prompt_style": ["zero-shot", "few-shot", "chain-of-thought"],
    "embedding_model": ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
}


@dataclass
class Config:
    docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"
    testset_dir: Path = PROJECT_ROOT / "data" / "testsets"
    results_dir: Path = PROJECT_ROOT / "data" / "results"
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "chroma"

    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "claude-sonnet-4-5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 512
    chunk_overlap: float = 0.15
    top_k: int = 5
    reranker: str | None = None
    search_mode: str = "vector"
    prompt_style: str = "zero-shot"

    search_space: dict = field(default_factory=lambda: RAG_SEARCH_SPACE)

    def __post_init__(self) -> None:
        for p in (self.docs_dir, self.testset_dir, self.results_dir, self.chroma_persist_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


@dataclass
class BFTSConfig:
    num_seeds: int = 3
    max_steps: int = 20
    max_debug_depth: int = 3
    debug_prob: float = 0.5
    convergence_window: int = 3
    convergence_eps: float = 0.01
    stage_budgets: dict = field(
        default_factory=lambda: {
            Stage.PRELIMINARY: 3,
            Stage.BASELINE: 4,
            Stage.EXPLORATION: 10,
            Stage.ABLATION: 3,
        }
    )


@dataclass
class TestGenConfig:
    num_seeds: int = 30
    target_size: int = 50
    distribution: dict = field(
        default_factory=lambda: {
            "simple": 0.30,
            "multi_context": 0.30,
            "reasoning": 0.25,
            "conditional": 0.15,
        }
    )
    groundedness_threshold: float = 0.7
