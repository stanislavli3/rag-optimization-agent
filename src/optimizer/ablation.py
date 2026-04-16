"""Stage-4 ablation runner: measure each non-default parameter's contribution."""
from __future__ import annotations

from typing import Callable

from src.optimizer.search_space import (
    generate_ablation_configs as _generate_ablation_configs,
    get_default_config,
)


class AblationRunner:
    """Generates ablated configs and ranks components by score drop when removed."""

    @staticmethod
    def generate_ablation_configs(best_config: dict) -> list[dict]:
        return _generate_ablation_configs(best_config)

    @staticmethod
    def compute_ablation_report(
        best_score: float,
        ablation_results: list[dict],
    ) -> list[dict]:
        """Given per-ablation results, produce a ranked report.

        Each input item must have: ``ablated_param``, ``original_value``,
        ``default_value``, ``config``, ``score_without`` (score with that param
        reverted to default).
        """
        report: list[dict] = []
        for r in ablation_results:
            score_without = float(r.get("score_without", 0.0))
            delta = float(best_score) - score_without
            contribution_pct = (delta / best_score * 100.0) if best_score else 0.0
            report.append(
                {
                    "param": r["ablated_param"],
                    "with_value": r["original_value"],
                    "without_value": r["default_value"],
                    "score_with": float(best_score),
                    "score_without": score_without,
                    "delta": delta,
                    "contribution_pct": contribution_pct,
                }
            )
        report.sort(key=lambda x: x["delta"], reverse=True)
        return report

    @staticmethod
    def run(
        best_config: dict,
        best_score: float,
        evaluate_config: Callable[[dict], float],
    ) -> list[dict]:
        """Helper that runs the full ablation loop using an ``evaluate_config`` callable.

        ``evaluate_config(config) -> score`` should execute the RAG pipeline on the
        ablated config and return a single composite score (e.g. ``ragas_score``).
        """
        ablations = AblationRunner.generate_ablation_configs(best_config)
        results: list[dict] = []
        for a in ablations:
            score = float(evaluate_config(a["config"]))
            results.append({**a, "score_without": score})
        return AblationRunner.compute_ablation_report(best_score, results)


def baseline_vs_best_improvement(
    best_config: dict,
    best_score: float,
    baseline_score: float,
) -> dict:
    """Summary of how much the optimizer improved over the default baseline."""
    default = get_default_config()
    changed = {k: (default.get(k), best_config.get(k)) for k in default if default.get(k) != best_config.get(k)}
    return {
        "baseline_score": baseline_score,
        "best_score": best_score,
        "absolute_improvement": best_score - baseline_score,
        "relative_improvement_pct": ((best_score - baseline_score) / baseline_score * 100.0) if baseline_score else 0.0,
        "n_changed_params": len(changed),
        "changed_params": {k: {"default": d, "best": b} for k, (d, b) in changed.items()},
    }
