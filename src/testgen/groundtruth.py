"""LLM-as-judge groundedness filter (RAGEval, ACL 2025)."""
from __future__ import annotations

import logging
import re

from src.testgen.llm import LLMLike

logger = logging.getLogger(__name__)


VERIFY_GROUNDEDNESS = """You are a strict fact-checker. Given the QUESTION, the proposed
ANSWER, and the CONTEXT, decide whether EVERY claim in the ANSWER is directly supported
by the CONTEXT.

Return your verdict in exactly this format:

GROUNDED: YES or NO
CONFIDENCE: a float between 0.0 and 1.0

QUESTION: {question}

ANSWER: {answer}

CONTEXT:
\"\"\"{context}\"\"\"
"""


def _parse(raw: str) -> tuple[bool, float]:
    grounded = bool(re.search(r"GROUNDED\s*:\s*YES", raw, re.IGNORECASE))
    m = re.search(r"CONFIDENCE\s*:\s*([01](?:\.\d+)?)", raw, re.IGNORECASE)
    conf = float(m.group(1)) if m else (0.9 if grounded else 0.1)
    return grounded, max(0.0, min(1.0, conf))


def verify_groundedness(
    question: str, answer: str, context: str, llm: LLMLike
) -> tuple[bool, float]:
    """LLM judge: `is_grounded, confidence`. Defaults to ``(False, 0.0)`` on error."""
    try:
        raw = llm.invoke(
            VERIFY_GROUNDEDNESS.format(
                question=question, answer=answer, context=context[:4000]
            )
        )
    except Exception as e:
        logger.warning("groundedness check failed: %s", e)
        return False, 0.0
    return _parse(raw)
