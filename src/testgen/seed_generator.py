"""Generate seed Q&A pairs from KG facts via the LLM.

Each seed needs two LLM calls:
  P(question | fact, context)
  P(answer | question, context)

Ungrounded answers (INSUFFICIENT) are dropped before sampling for diversity.
"""
from __future__ import annotations

import logging
import random
from typing import Any

from src.testgen.knowledge_graph import KnowledgeGraph
from src.testgen.llm import LLMLike

logger = logging.getLogger(__name__)


GENERATE_SEED_QUESTION = """Given the FACT and the CONTEXT it came from, write a single natural
user question whose correct answer is the FACT (or its core information). The question
must be answerable solely from the CONTEXT. Do not include the answer. Return only the
question text, no preamble.

FACT: {fact}

CONTEXT:
\"\"\"{context}\"\"\"
"""


GENERATE_GROUND_TRUTH = """Answer the QUESTION using ONLY the CONTEXT. If the context does
not contain enough information to answer, respond with exactly the single word
INSUFFICIENT. Otherwise respond with a concise, self-contained answer.

QUESTION: {question}

CONTEXT:
\"\"\"{context}\"\"\"
"""


def _context_for_chunk(chunks_by_id: dict, chunk_id: str) -> str:
    c = chunks_by_id.get(chunk_id)
    if c is None:
        return ""
    return getattr(c, "page_content", str(c))


def generate_seeds(
    kg: KnowledgeGraph,
    llm: LLMLike,
    num_seeds: int = 30,
    chunks: list[Any] | None = None,
) -> list[dict]:
    """Produce up to ``num_seeds`` valid seeds, prioritising source-chunk diversity."""
    chunks_by_id: dict = {}
    if chunks:
        for c in chunks:
            meta = getattr(c, "metadata", {}) or {}
            cid = meta.get("chunk_id") or meta.get("source_file")
            if cid:
                chunks_by_id[cid] = c

    facts = kg.get_facts()
    if not facts:
        logger.warning("No facts in KG — cannot generate seeds")
        return []

    random.shuffle(facts)
    seeds: list[dict] = []
    chunk_use_count: dict[str, int] = {}

    # Sort by least-used chunk first to encourage diversity
    def diversity_key(f):
        return chunk_use_count.get(f["source_chunk_id"], 0)

    while facts and len(seeds) < num_seeds * 2:  # oversample; caller truncates
        facts.sort(key=diversity_key)
        f = facts.pop(0)
        ctx = _context_for_chunk(chunks_by_id, f["source_chunk_id"]) or f["fact"]

        try:
            question = llm.invoke(
                GENERATE_SEED_QUESTION.format(fact=f["fact"], context=ctx[:3000])
            ).strip()
            answer = llm.invoke(
                GENERATE_GROUND_TRUTH.format(question=question, context=ctx[:3000])
            ).strip()
        except Exception as e:
            logger.warning("LLM seed generation failed: %s", e)
            continue

        if not question or answer.upper().startswith("INSUFFICIENT"):
            continue

        seeds.append(
            {
                "question": question,
                "ground_truth_answer": answer,
                "ground_truth_context": ctx,
                "source_fact": f["fact"],
                "source_chunk_id": f["source_chunk_id"],
                "question_type": "simple",
                "evolution": "seed",
            }
        )
        chunk_use_count[f["source_chunk_id"]] = chunk_use_count.get(f["source_chunk_id"], 0) + 1

    # Final sample down to num_seeds, keeping the diversified ordering
    return seeds[:num_seeds]
