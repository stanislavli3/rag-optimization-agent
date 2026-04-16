"""Paired bootstrap significance, Cohen's d, experiment comparison utilities."""
from __future__ import annotations

import random
from statistics import mean, pstdev
from typing import Sequence


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    if len(scores_a) != len(scores_b) or not scores_a:
        return {"p_value": 1.0, "ci_lower": 0.0, "ci_upper": 0.0, "significant": False}
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    observed = mean(diffs)
    rng = random.Random(seed)

    boot = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        boot.append(mean(sample))
    boot.sort()

    lo = boot[int(n_bootstrap * (alpha / 2))]
    hi = boot[int(n_bootstrap * (1 - alpha / 2)) - 1]

    # Two-sided p-value: fraction of centred bootstrap samples at least as extreme
    centred = [b - observed for b in boot]
    extreme = sum(1 for c in centred if abs(c) >= abs(observed))
    p = extreme / n_bootstrap

    return {
        "p_value": p,
        "ci_lower": lo,
        "ci_upper": hi,
        "observed_diff": observed,
        "significant": not (lo <= 0 <= hi),
    }


def cohens_d(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    if not scores_a or not scores_b:
        return 0.0
    ma, mb = mean(scores_a), mean(scores_b)
    sa, sb = pstdev(scores_a), pstdev(scores_b)
    pooled = ((sa**2 + sb**2) / 2) ** 0.5
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def compare_experiments(
    results_a: list[dict],
    results_b: list[dict],
    metrics: list[str] | None = None,
) -> dict:
    """Per-metric deltas, paired bootstrap p-values, Cohen's d."""
    metrics = metrics or ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]

    def _scores(results, metric):
        out = []
        for r in results:
            pq = r.get("per_question") or []
            for q in pq:
                out.append(float(q.get("scores", {}).get(metric, 0.0)))
        return out

    out: dict[str, dict] = {}
    for m in metrics:
        a = _scores(results_a, m)
        b = _scores(results_b, m)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        boot = paired_bootstrap_test(a, b)
        out[m] = {
            "mean_a": mean(a) if a else 0.0,
            "mean_b": mean(b) if b else 0.0,
            "delta": (mean(a) - mean(b)) if a and b else 0.0,
            "p_value": boot["p_value"],
            "ci": [boot["ci_lower"], boot["ci_upper"]],
            "significant": boot["significant"],
            "cohens_d": cohens_d(a, b),
        }
    return out
