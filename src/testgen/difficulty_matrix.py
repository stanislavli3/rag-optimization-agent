"""GRADE-style 2-dimensional difficulty scoring for synthetic test questions.

Axes:
  * reasoning_depth (int 1–3): from question type + fact count.
  * semantic_distance (float 0–1): cosine distance between question and context
    embeddings. High distance → vocabulary mismatch → harder for retrieval.

The 2-D matrix lets downstream evaluation stratify pipeline failures (retriever
vs generator).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DifficultyScore:
    reasoning_depth: int
    semantic_distance: float
    overall: str  # "easy" | "medium" | "hard"


_DEPTH_BY_TYPE = {
    "simple": 1,
    "multi_context": 2,
    "reasoning": 3,
    "conditional": 3,
}


def score_reasoning_depth(question_data: dict) -> int:
    base = _DEPTH_BY_TYPE.get(question_data.get("question_type", "simple"), 1)
    # Bump depth by 1 if the context appears to combine multiple chunks
    ctx = question_data.get("ground_truth_context", "")
    if "\n---\n" in ctx:
        base = min(3, base + 1)
    return max(1, min(3, base))


def _cosine_distance(a, b) -> float:
    import numpy as np  # type: ignore

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    return max(0.0, min(1.0, 1.0 - (sim + 1.0) / 2.0))  # normalised to 0..1


def _bow_vector(text: str, vocab: list[str]):
    import numpy as np  # type: ignore

    toks = [t.lower() for t in text.split() if t.strip()]
    counts = {v: 0 for v in vocab}
    for t in toks:
        if t in counts:
            counts[t] += 1
    return np.asarray([counts[v] for v in vocab], dtype=float)


def _encode(model, texts: list[str]):
    if model is None:
        # Deterministic bag-of-words fallback
        vocab = sorted({t.lower() for text in texts for t in text.split() if t.strip()})
        return [_bow_vector(t, vocab) for t in texts]
    try:
        return list(model.encode(texts, show_progress_bar=False))
    except Exception as e:
        logger.warning("embedding failed (%s) — falling back to BoW", e)
        vocab = sorted({t.lower() for text in texts for t in text.split() if t.strip()})
        return [_bow_vector(t, vocab) for t in texts]


def score_semantic_distance(question: str, context: str, embedding_model=None) -> float:
    vecs = _encode(embedding_model, [question, context])
    return _cosine_distance(vecs[0], vecs[1])


def _classify(depth: int, dist: float) -> str:
    if depth == 1 and dist < 0.3:
        return "easy"
    if depth >= 3 or dist > 0.6:
        return "hard"
    return "medium"


def compute_difficulty_matrix(testset: list[dict], embedding_model=None) -> Any:
    """Return a pandas DataFrame with difficulty columns attached."""
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required for compute_difficulty_matrix") from e

    rows: list[dict] = []
    # Encode all in one batch where possible
    questions = [q.get("question", "") for q in testset]
    contexts = [q.get("ground_truth_context", "") for q in testset]
    q_vecs = _encode(embedding_model, questions) if embedding_model is not None else None
    c_vecs = _encode(embedding_model, contexts) if embedding_model is not None else None

    for i, q in enumerate(testset):
        depth = score_reasoning_depth(q)
        if q_vecs is not None and c_vecs is not None:
            dist = _cosine_distance(q_vecs[i], c_vecs[i])
        else:
            dist = score_semantic_distance(questions[i], contexts[i], embedding_model=None)
        bucket = _classify(depth, dist)
        rows.append(
            {
                **q,
                "reasoning_depth": depth,
                "semantic_distance": round(float(dist), 4),
                "difficulty": bucket,
            }
        )
    return pd.DataFrame(rows)


def difficulty_breakdown_report(df) -> dict:
    import pandas as pd  # type: ignore

    out: dict = {"total": int(len(df))}
    if "question_type" in df.columns:
        out["by_type"] = df["question_type"].value_counts().to_dict()
    if "difficulty" in df.columns:
        out["by_difficulty"] = df["difficulty"].value_counts().to_dict()

    matrix: dict = {}
    if {"reasoning_depth", "semantic_distance"}.issubset(df.columns):
        df = df.copy()
        df["_dist_bucket"] = pd.cut(
            df["semantic_distance"], bins=[-0.01, 0.3, 0.6, 1.01], labels=["low", "mid", "high"]
        )
        grouped = df.groupby(["reasoning_depth", "_dist_bucket"], observed=True).size()
        matrix = {f"depth={k[0]},dist={k[1]}": int(v) for k, v in grouped.items()}
    out["matrix"] = matrix
    return out
