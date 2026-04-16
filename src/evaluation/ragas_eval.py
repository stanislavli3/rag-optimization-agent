"""RAGAS LLM-as-judge metrics with stratified breakdowns.

When ``ragas`` is installed and configured, we delegate to it. Otherwise we fall back
to a lightweight heuristic scorer so the pipeline stays runnable in CI/offline.
"""
from __future__ import annotations

import logging
import re
from statistics import harmonic_mean
from typing import Any

logger = logging.getLogger(__name__)

CORE_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
ALL_METRICS = CORE_METRICS + ["answer_correctness"]


def _tokset(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _lexical_score(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    p, g = _tokset(pred), _tokset(gold)
    if not p or not g:
        return 0.0
    inter = len(p & g)
    prec = inter / len(p)
    rec = inter / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _heuristic_per_question(row: dict) -> dict:
    pred = row.get("answer", "")
    gold = row.get("ground_truth_answer") or row.get("ground_truth") or ""
    ctx_list = row.get("retrieved_contexts") or []
    ctx_text = " ".join(ctx_list)
    gold_ctx = row.get("ground_truth_context", "")

    faith = _lexical_score(pred, ctx_text)
    rel = _lexical_score(pred, row.get("question", ""))
    c_prec = _lexical_score(ctx_text, gold_ctx) if gold_ctx else faith
    c_rec = _lexical_score(gold_ctx, ctx_text) if gold_ctx else faith
    ans_correct = _lexical_score(pred, gold) if gold else 0.0
    return {
        "faithfulness": faith,
        "answer_relevancy": rel,
        "context_precision": c_prec,
        "context_recall": c_rec,
        "answer_correctness": ans_correct,
    }


def _ragas_evaluate(results: list[dict], llm, embeddings) -> dict | None:
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as e:
        logger.info("ragas not available (%s) — using heuristic eval", e)
        return None

    ds = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r.get("retrieved_contexts", []),
                "ground_truth": r.get("ground_truth_answer") or "",
            }
            for r in results
        ]
    )
    try:
        report = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
            llm=llm,
            embeddings=embeddings,
        )
        df = report.to_pandas()
    except Exception as e:
        logger.warning("ragas.evaluate failed: %s — falling back", e)
        return None

    agg = {m: float(df[m].mean()) for m in ALL_METRICS if m in df.columns}
    per_q = [
        {"question": r["question"], "scores": {m: float(df.iloc[i].get(m, 0.0)) for m in ALL_METRICS if m in df.columns}}
        for i, r in enumerate(results)
    ]
    return {**agg, "per_question": per_q}


def evaluate_ragas(results: list[dict], llm=None, embeddings=None) -> dict:
    """Evaluate with real RAGAS if available, else lexical heuristics."""
    ragas_out = _ragas_evaluate(results, llm, embeddings) if llm is not None else None
    if ragas_out is None:
        per_q = []
        agg: dict[str, float] = {m: 0.0 for m in ALL_METRICS}
        for r in results:
            scores = _heuristic_per_question(r)
            per_q.append({"question": r["question"], "scores": scores})
            for m, v in scores.items():
                agg[m] += v
        n = max(1, len(results))
        agg = {m: v / n for m, v in agg.items()}
        out = {**agg, "per_question": per_q}
    else:
        out = ragas_out

    core = [out[m] for m in CORE_METRICS if m in out]
    out["ragas_score"] = float(harmonic_mean(core)) if core and all(c > 0 for c in core) else 0.0
    return out


def evaluate_by_difficulty(results: list[dict], testset_df) -> dict:
    """Group per-question scores by the testset's difficulty column."""
    by_q = {r["question"]: r for r in results}
    out: dict[str, dict] = {"easy": [], "medium": [], "hard": []}
    for _, row in testset_df.iterrows():
        q = row.get("question")
        diff = row.get("difficulty", "medium")
        r = by_q.get(q)
        if not r:
            continue
        r = {**r, "ground_truth_answer": row.get("ground_truth_answer"), "ground_truth_context": row.get("ground_truth_context")}
        out.setdefault(diff, []).append(_heuristic_per_question(r))
    return {
        d: ({m: sum(s[m] for s in rows) / len(rows) for m in ALL_METRICS} if rows else {})
        for d, rows in out.items()
    }


def evaluate_by_question_type(results: list[dict], testset_df) -> dict:
    by_q = {r["question"]: r for r in results}
    grouped: dict[str, list[dict]] = {}
    for _, row in testset_df.iterrows():
        q = row.get("question")
        qt = row.get("question_type", "simple")
        r = by_q.get(q)
        if not r:
            continue
        r = {**r, "ground_truth_answer": row.get("ground_truth_answer"), "ground_truth_context": row.get("ground_truth_context")}
        grouped.setdefault(qt, []).append(_heuristic_per_question(r))
    return {
        qt: {m: sum(s[m] for s in rows) / len(rows) for m in ALL_METRICS}
        for qt, rows in grouped.items()
        if rows
    }
