"""Classic information-retrieval metrics: MRR, NDCG@k, P@k, R@k."""
from __future__ import annotations

import math
from typing import Sequence


def _reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def mrr(retrieved: list[list[str]], relevant: list[list[str]]) -> float:
    if not retrieved:
        return 0.0
    return sum(_reciprocal_rank(r, g) for r, g in zip(retrieved, relevant)) / len(retrieved)


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: list[list[str]], relevant: list[list[str]], k: int) -> float:
    if not retrieved:
        return 0.0
    scores = []
    for ret, rel in zip(retrieved, relevant):
        rel_set = set(rel)
        gains = [1.0 if d in rel_set else 0.0 for d in ret[:k]]
        ideal = sorted(gains, reverse=True)
        dcg = _dcg(gains)
        idcg = _dcg(ideal) or 1.0
        scores.append(dcg / idcg)
    return sum(scores) / len(scores)


def precision_at_k(retrieved: list[list[str]], relevant: list[list[str]], k: int) -> float:
    if not retrieved:
        return 0.0
    scores = []
    for ret, rel in zip(retrieved, relevant):
        rel_set = set(rel)
        hits = sum(1 for d in ret[:k] if d in rel_set)
        scores.append(hits / k if k > 0 else 0.0)
    return sum(scores) / len(scores)


def recall_at_k(retrieved: list[list[str]], relevant: list[list[str]], k: int) -> float:
    if not retrieved:
        return 0.0
    scores = []
    for ret, rel in zip(retrieved, relevant):
        rel_set = set(rel)
        if not rel_set:
            continue
        hits = sum(1 for d in ret[:k] if d in rel_set)
        scores.append(hits / len(rel_set))
    return sum(scores) / max(1, len(scores))


def evaluate_ir(
    retrieved: list[list[str]], relevant: list[list[str]], k: int = 5
) -> dict:
    return {
        "mrr": mrr(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
    }
