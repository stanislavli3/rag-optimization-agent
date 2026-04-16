"""Main BFTS orchestrator — 4-stage progressive experimentation.

Stages:
  1. PRELIMINARY — seed `num_seeds` configs to confirm feasibility.
  2. BASELINE    — expand from the best seed until convergence.
  3. EXPLORATION — best-first tree expansion through the config space.
  4. ABLATION    — strip each non-default parameter from the best config to confirm
                   its marginal contribution.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

from config import BFTSConfig
from src.optimizer.ablation import AblationRunner
from src.optimizer.experiment_manager import ExperimentManager
from src.optimizer.tree_node import NodeStatus, SearchNode, Stage

logger = logging.getLogger(__name__)


class BFTSLoop:
    """Wires run_fn + eval_fn + ExperimentManager + AblationRunner together."""

    def __init__(
        self,
        documents: list[Any],
        testset: list[dict] | Any,
        run_fn: Callable[[dict, list[Any], list[dict]], list[dict]],
        eval_fn: Callable[[list[dict]], dict],
        bfts_config: BFTSConfig | None = None,
        score_key: str = "ragas_score",
    ) -> None:
        self.documents = documents
        self.testset = testset
        self.run_fn = run_fn
        self.eval_fn = eval_fn
        self.cfg = bfts_config or BFTSConfig()
        self.score_key = score_key
        self.manager = ExperimentManager(self.cfg)
        self.ablation_report: list[dict] = []

    # ---- queries / testset conversion -----------------------------------------
    def _queries(self) -> list[dict]:
        ts = self.testset
        if hasattr(ts, "to_dict"):
            return ts.to_dict(orient="records")  # pandas
        return list(ts)

    # ---- node execution --------------------------------------------------------
    def _execute_node(self, node: SearchNode) -> dict:
        node.status = NodeStatus.RUNNING
        try:
            results = self.run_fn(node.config, self.documents, self._queries())
            metrics = self.eval_fn(results)
            score = float(metrics.get(self.score_key, 0.0))
            self.manager.report_success(node.id, metrics, score)
            return {
                "type": "node_complete",
                "node": node.to_dict(),
                "status": "success",
                "score": score,
                "metrics": metrics,
            }
        except Exception as e:  # noqa: BLE001 — we report the error to the manager
            decision = self.manager.report_failure(node.id, str(e))
            return {
                "type": "node_complete",
                "node": node.to_dict(),
                "status": "failed",
                "decision": decision,
                "error": str(e),
            }

    # ---- ablation --------------------------------------------------------------
    def _run_ablations(self) -> list[dict]:
        best = self.manager.get_best_node()
        if best is None:
            return []

        ablations = AblationRunner.generate_ablation_configs(best.config)
        executed: list[dict] = []
        events: list[dict] = []
        for a in ablations:
            ab_node = SearchNode(config=a["config"], stage=Stage.ABLATION)
            self.manager.nodes[ab_node.id] = ab_node
            try:
                results = self.run_fn(a["config"], self.documents, self._queries())
                metrics = self.eval_fn(results)
                score = float(metrics.get(self.score_key, 0.0))
                ab_node.mark_success(metrics, score)
                executed.append({**a, "score_without": score, "node_id": ab_node.id})
                events.append({"type": "ablation_node", "param": a["ablated_param"], "score": score, "node": ab_node.to_dict()})
            except Exception as e:  # noqa: BLE001
                ab_node.mark_pruned(str(e))
                events.append({"type": "ablation_node", "param": a["ablated_param"], "error": str(e), "node": ab_node.to_dict()})

        report = AblationRunner.compute_ablation_report(best.score, executed)
        self.ablation_report = report
        # Attach events so callers streaming events can see them
        self._last_ablation_events = events
        return report

    # ---- main loops ------------------------------------------------------------
    def run(self) -> dict:
        for _ in self.run_iter():
            pass
        return self._final_summary()

    def run_iter(self) -> Iterator[dict]:
        seeds = self.manager.seed_roots()
        prev_stage = self.manager.current_stage
        for s in seeds:
            yield self._execute_node(s)

        while True:
            node = self.manager.select_next()
            if self.manager.current_stage != prev_stage:
                yield {
                    "type": "stage_transition",
                    "transitions": self.manager.get_stage_transitions()[-1:],
                    "current_stage": self.manager.current_stage.name,
                }
                prev_stage = self.manager.current_stage

            if node is None:
                if self.manager.current_stage == Stage.ABLATION and not self.ablation_report:
                    for ev in self._run_ablations_streaming():
                        yield ev
                    yield {"type": "ablation_complete", "report": self.ablation_report}
                break

            yield self._execute_node(node)

        yield {"type": "search_complete", "summary": self._final_summary()}

    def _run_ablations_streaming(self) -> Iterator[dict]:
        best = self.manager.get_best_node()
        if best is None:
            return
        ablations = AblationRunner.generate_ablation_configs(best.config)
        executed: list[dict] = []
        for a in ablations:
            ab_node = SearchNode(config=a["config"], stage=Stage.ABLATION)
            self.manager.nodes[ab_node.id] = ab_node
            try:
                results = self.run_fn(a["config"], self.documents, self._queries())
                metrics = self.eval_fn(results)
                score = float(metrics.get(self.score_key, 0.0))
                ab_node.mark_success(metrics, score)
                executed.append({**a, "score_without": score, "node_id": ab_node.id})
                yield {
                    "type": "ablation_node",
                    "param": a["ablated_param"],
                    "score": score,
                    "node": ab_node.to_dict(),
                }
            except Exception as e:  # noqa: BLE001
                ab_node.mark_pruned(str(e))
                yield {
                    "type": "ablation_node",
                    "param": a["ablated_param"],
                    "error": str(e),
                    "node": ab_node.to_dict(),
                }
        self.ablation_report = AblationRunner.compute_ablation_report(best.score, executed)

    # ---- summary ---------------------------------------------------------------
    def _final_summary(self) -> dict:
        best = self.manager.get_best_node()
        trajectory = self.manager.get_trajectory()
        per_stage: dict[str, dict[str, int]] = {}
        for n in self.manager.nodes.values():
            bucket = per_stage.setdefault(n.stage.name, {"success": 0, "failed": 0, "pruned": 0, "pending": 0, "running": 0})
            key = n.status.name.lower()
            bucket[key] = bucket.get(key, 0) + 1
        return {
            "best_config": best.config if best else None,
            "best_score": best.score if best else 0.0,
            "best_metrics": best.metrics if best else {},
            "trajectory": trajectory,
            "ablation_report": self.ablation_report,
            "tree_summary": per_stage,
            "stage_transitions": self.manager.get_stage_transitions(),
        }

    def get_tree_visualization_data(self) -> dict:
        """Shape designed for the AgentTree React component."""
        nodes = []
        edges = []
        best = self.manager.get_best_node()
        best_path: list[str] = []
        if best is not None:
            cur: SearchNode | None = best
            while cur is not None:
                best_path.append(cur.id)
                cur = self.manager.nodes.get(cur.parent_id) if cur.parent_id else None
            best_path.reverse()

        stages: dict[str, list[str]] = {s.name: [] for s in Stage}
        for n in self.manager.nodes.values():
            nodes.append(
                {
                    "id": n.id,
                    "parent_id": n.parent_id,
                    "stage": n.stage.name.lower(),
                    "status": n.status.name.lower(),
                    "config": n.config,
                    "score": n.score if n.status == NodeStatus.SUCCESS else None,
                    "depth": n.depth,
                }
            )
            if n.parent_id:
                edges.append({"source": n.parent_id, "target": n.id})
            stages[n.stage.name].append(n.id)

        return {
            "nodes": nodes,
            "edges": edges,
            "best_path": best_path,
            "stages": stages,
        }
