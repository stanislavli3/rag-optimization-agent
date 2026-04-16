"""Evol-Instruct evolution of seed questions into 4 types.

References: WizardLM (Evol-Instruct), RAGAS (EACL 2024), GRADE (EMNLP 2025 Findings).
Generative aspect: `P(hard_question | seed_question, target_type, context)`.
"""
from __future__ import annotations

import logging
import random
from enum import Enum

from src.testgen.knowledge_graph import KnowledgeGraph
from src.testgen.llm import LLMLike

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    SIMPLE = "simple"
    MULTI_CONTEXT = "multi_context"
    REASONING = "reasoning"
    CONDITIONAL = "conditional"


EVOLVE_MULTI_CONTEXT = """You are rewriting a question so that it requires BOTH passages
below to be answered. The new question must not be answerable from a single passage
alone. Return only the new question.

SEED QUESTION: {seed_q}

PASSAGE A:
\"\"\"{ctx_a}\"\"\"

PASSAGE B:
\"\"\"{ctx_b}\"\"\"
"""


EVOLVE_REASONING = """Rewrite the question so that answering it requires reasoning —
causal chains, implications, or "why/how" analysis — rather than direct fact lookup.
The answer must still be derivable from the CONTEXT. Return only the new question.

SEED QUESTION: {seed_q}

CONTEXT:
\"\"\"{context}\"\"\"
"""


EVOLVE_CONDITIONAL = """Rewrite the question by adding a plausible constraint or
condition ("assuming X...", "under what circumstances...", "if Y changes...")
that changes or narrows the answer. The answer must still be derivable from the
CONTEXT. Return only the new question.

SEED QUESTION: {seed_q}

CONTEXT:
\"\"\"{context}\"\"\"
"""


def _find_related_context(seed: dict, kg: KnowledgeGraph) -> str | None:
    """Pick a second context from a chunk the KG connects to the seed's chunk."""
    src = seed.get("source_chunk_id")
    if not src:
        return None
    neighbours: set[str] = set()
    for e in kg.edges:
        # Walk through fact->entity edges
        src_fact = next((n for n in kg.nodes if n.id == e.source_id), None)
        tgt_fact = next((n for n in kg.nodes if n.id == e.target_id), None)
        if src_fact and src_fact.source_chunk_id == src and tgt_fact:
            neighbours.add(tgt_fact.source_chunk_id)
        if tgt_fact and tgt_fact.source_chunk_id == src and src_fact:
            neighbours.add(src_fact.source_chunk_id)
    neighbours.discard(src)
    if not neighbours:
        return None
    other_chunk_id = sorted(neighbours)[0]
    # Compose a synthetic context from facts in that chunk
    facts_there = [n.text for n in kg.nodes if n.node_type == "fact" and n.source_chunk_id == other_chunk_id]
    return "\n".join(facts_there[:5]) if facts_there else None


def evolve_to_multi_context(seed: dict, kg: KnowledgeGraph, llm: LLMLike) -> dict:
    ctx_b = _find_related_context(seed, kg)
    if not ctx_b:
        return {**seed, "question_type": QuestionType.SIMPLE.value, "evolution": "fallback:simple"}
    prompt = EVOLVE_MULTI_CONTEXT.format(
        seed_q=seed["question"], ctx_a=seed["ground_truth_context"][:1500], ctx_b=ctx_b[:1500]
    )
    try:
        new_q = llm.invoke(prompt).strip()
    except Exception as e:
        logger.warning("multi_context evolution failed: %s", e)
        return {**seed, "question_type": QuestionType.SIMPLE.value, "evolution": "fallback:simple"}
    return {
        **seed,
        "question": new_q or seed["question"],
        "question_type": QuestionType.MULTI_CONTEXT.value,
        "evolution": "seed->multi_context",
        "ground_truth_context": seed["ground_truth_context"] + "\n---\n" + ctx_b,
    }


def evolve_to_reasoning(seed: dict, llm: LLMLike) -> dict:
    try:
        new_q = llm.invoke(
            EVOLVE_REASONING.format(seed_q=seed["question"], context=seed["ground_truth_context"][:3000])
        ).strip()
    except Exception as e:
        logger.warning("reasoning evolution failed: %s", e)
        return {**seed, "question_type": QuestionType.SIMPLE.value, "evolution": "fallback:simple"}
    return {
        **seed,
        "question": new_q or seed["question"],
        "question_type": QuestionType.REASONING.value,
        "evolution": "seed->reasoning",
    }


def evolve_to_conditional(seed: dict, llm: LLMLike) -> dict:
    try:
        new_q = llm.invoke(
            EVOLVE_CONDITIONAL.format(seed_q=seed["question"], context=seed["ground_truth_context"][:3000])
        ).strip()
    except Exception as e:
        logger.warning("conditional evolution failed: %s", e)
        return {**seed, "question_type": QuestionType.SIMPLE.value, "evolution": "fallback:simple"}
    return {
        **seed,
        "question": new_q or seed["question"],
        "question_type": QuestionType.CONDITIONAL.value,
        "evolution": "seed->conditional",
    }


def evolve_question(seed: dict, target_type: QuestionType | str, kg: KnowledgeGraph, llm: LLMLike) -> dict:
    t = QuestionType(target_type) if not isinstance(target_type, QuestionType) else target_type
    if t == QuestionType.MULTI_CONTEXT:
        return evolve_to_multi_context(seed, kg, llm)
    if t == QuestionType.REASONING:
        return evolve_to_reasoning(seed, llm)
    if t == QuestionType.CONDITIONAL:
        return evolve_to_conditional(seed, llm)
    return {**seed, "question_type": QuestionType.SIMPLE.value, "evolution": "seed"}


DEFAULT_DISTRIBUTION = {
    QuestionType.SIMPLE: 0.30,
    QuestionType.MULTI_CONTEXT: 0.30,
    QuestionType.REASONING: 0.25,
    QuestionType.CONDITIONAL: 0.15,
}


def evolve_batch(
    seeds: list[dict],
    kg: KnowledgeGraph,
    llm: LLMLike,
    distribution: dict | None = None,
) -> list[dict]:
    dist = {QuestionType(k) if isinstance(k, str) else k: v for k, v in (distribution or DEFAULT_DISTRIBUTION).items()}
    total = sum(dist.values()) or 1.0
    weights = {k: v / total for k, v in dist.items()}

    n = len(seeds)
    targets: list[QuestionType] = []
    for qt, w in weights.items():
        targets.extend([qt] * int(round(w * n)))
    # Pad/truncate to len(seeds)
    while len(targets) < n:
        targets.append(QuestionType.SIMPLE)
    targets = targets[:n]
    random.shuffle(targets)

    out: list[dict] = []
    for seed, t in zip(seeds, targets):
        out.append(evolve_question(seed, t, kg, llm))
    return out
