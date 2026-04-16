"""End-to-end synthetic test-set generator.

    docs → chunks → KG → seeds → evolved → grounded filter → difficulty-scored CSV
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from src.ingest.chunker import chunk_documents
from src.testgen.difficulty_matrix import (
    compute_difficulty_matrix,
    difficulty_breakdown_report,
)
from src.testgen.evol_instruct import DEFAULT_DISTRIBUTION, evolve_batch
from src.testgen.groundtruth import verify_groundedness
from src.testgen.knowledge_graph import extract_knowledge_graph
from src.testgen.llm import LLMLike, get_llm
from src.testgen.seed_generator import generate_seeds

logger = logging.getLogger(__name__)


OUTPUT_COLUMNS = [
    "question",
    "ground_truth_answer",
    "ground_truth_context",
    "question_type",
    "evolution",
    "reasoning_depth",
    "semantic_distance",
    "difficulty",
]


@dataclass
class TestGenPipeline:
    llm: LLMLike = field(default_factory=get_llm)
    embedding_model: Any = None
    chunk_size: int = 512
    overlap_ratio: float = 0.15
    target_size: int = 50
    groundedness_threshold: float = 0.7
    distribution: dict = field(default_factory=lambda: dict(DEFAULT_DISTRIBUTION))
    out_dir: Path = Path("data/testsets")

    def generate(self, documents: list[Any]):
        rows_df = None
        for event in self.generate_with_progress(documents):
            if event.get("step") == "done" and event.get("result") is not None:
                rows_df = event["result"]
        return rows_df

    def generate_with_progress(self, documents: list[Any]) -> Iterator[dict]:
        yield {"step": "chunk", "status": "running"}
        chunks = chunk_documents(documents, chunk_size=self.chunk_size, overlap_ratio=self.overlap_ratio)
        yield {"step": "chunk", "status": "done", "stats": {"n_chunks": len(chunks)}}

        yield {"step": "knowledge_graph", "status": "running"}
        kg = extract_knowledge_graph(chunks, self.llm)
        yield {
            "step": "knowledge_graph",
            "status": "done",
            "stats": {"n_nodes": len(kg.nodes), "n_edges": len(kg.edges), "n_facts": len(kg.get_facts())},
        }

        yield {"step": "seeds", "status": "running"}
        seeds = generate_seeds(kg, self.llm, num_seeds=self.target_size * 2, chunks=chunks)
        yield {"step": "seeds", "status": "done", "stats": {"n_seeds": len(seeds)}}

        yield {"step": "evolve", "status": "running"}
        evolved = evolve_batch(seeds, kg, self.llm, distribution=self.distribution)
        by_type: dict = {}
        for q in evolved:
            by_type[q["question_type"]] = by_type.get(q["question_type"], 0) + 1
        yield {"step": "evolve", "status": "done", "stats": {"by_type": by_type, "n": len(evolved)}}

        yield {"step": "groundedness", "status": "running"}
        filtered: list[dict] = []
        rejected = 0
        for q in evolved:
            ok, conf = verify_groundedness(
                q["question"], q.get("ground_truth_answer", ""), q.get("ground_truth_context", ""), self.llm
            )
            if ok and conf >= self.groundedness_threshold:
                filtered.append({**q, "groundedness_conf": conf})
            else:
                rejected += 1
        yield {
            "step": "groundedness",
            "status": "done",
            "stats": {"kept": len(filtered), "rejected": rejected},
        }

        yield {"step": "difficulty", "status": "running"}
        df = compute_difficulty_matrix(filtered, embedding_model=self.embedding_model)
        stats = difficulty_breakdown_report(df)
        yield {"step": "difficulty", "status": "done", "stats": stats}

        if len(df) > self.target_size:
            df = df.sample(n=self.target_size, random_state=42).reset_index(drop=True)

        # Ensure required columns exist
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[OUTPUT_COLUMNS + [c for c in df.columns if c not in OUTPUT_COLUMNS]]

        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = self.out_dir / f"testset_{ts}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Testset saved to %s (%d rows)", out_path, len(df))

        yield {"step": "done", "status": "done", "stats": {"path": str(out_path), "rows": len(df)}, "result": df}
