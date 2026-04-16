"""LLM answer generation with configurable prompt style."""
from __future__ import annotations

import logging
from typing import Any

from src.testgen.llm import LLMLike

logger = logging.getLogger(__name__)


ZERO_SHOT = """Answer the QUESTION using only the CONTEXT. Be concise and specific. If
the context does not contain the answer, say "I don't know."

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


FEW_SHOT = """Answer the QUESTION using only the CONTEXT. Be concise and specific.

EXAMPLE 1
CONTEXT: Paris is the capital of France. France is in Europe.
QUESTION: What country is Paris the capital of?
ANSWER: France.

EXAMPLE 2
CONTEXT: The mitochondrion is the powerhouse of the cell and produces ATP.
QUESTION: What does the mitochondrion produce?
ANSWER: ATP.

NOW YOUR TURN
CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


CHAIN_OF_THOUGHT = """Answer the QUESTION using only the CONTEXT.

Think step by step. First identify which parts of the CONTEXT are relevant to the
QUESTION. Then reason over those parts. Finally, give a concise final answer on the
last line prefixed with "Answer:".

CONTEXT:
{context}

QUESTION: {question}
"""


PROMPTS = {
    "zero-shot": ZERO_SHOT,
    "few-shot": FEW_SHOT,
    "chain-of-thought": CHAIN_OF_THOUGHT,
}


def _format_context(contexts: list[Any]) -> str:
    parts = []
    for i, c in enumerate(contexts):
        text = getattr(c, "page_content", str(c))
        parts.append(f"[{i + 1}] {text}")
    return "\n\n".join(parts)


def _postprocess_cot(raw: str) -> str:
    """Extract the final Answer: line from a chain-of-thought response."""
    for line in reversed(raw.splitlines()):
        low = line.lower().strip()
        if low.startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return raw.strip()


def generate_answer(
    query: str,
    contexts: list[Any],
    llm: LLMLike,
    prompt_style: str = "zero-shot",
) -> str:
    template = PROMPTS.get(prompt_style, ZERO_SHOT)
    prompt = template.format(context=_format_context(contexts), question=query)
    try:
        raw = llm.invoke(prompt)
    except Exception as e:
        logger.warning("generation failed: %s", e)
        return ""
    if prompt_style == "chain-of-thought":
        return _postprocess_cot(raw)
    return raw.strip()
