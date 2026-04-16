"""BFTS brain: node selection, debug-or-abandon, stage transitions.

Adapted from Sakana AI Scientist v2 Experiment Progress Manager (Yamada et al., 2025).
"""
from __future__ import annotations

import logging
import random
from typing import Iterable

from config import BFTSConfig
from src.optimizer.search_space import (
    get_default_config,
    mutate_config,
    sample_random_config,
)
from src.optimizer.tree_node import NodeStatus, SearchNode, Stage

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Decides what to run next, how to respond to failures, when to change stage."""

    def __init__(self, bfts_config: BFTSConfig | None = None, rng: random.Random | None = None) -> None:
        self.cfg = bfts_config or BFTSConfig()
        self.rng = rng or random.Random(42)
        self.nodes: dict[str, SearchNode] = {}
        self.current_stage: Stage = Stage.PRELIMINARY
        self._stage_node_counts: dict[Stage, int] = {s: 0 for s in Stage}
        self._best_score_history: list[float] = []
        self._stage_transitions: list[dict] = []
        self._steps_taken: int = 0

    # ---- seeding ---------------------------------------------------------------
    def seed_roots(self) -> list[SearchNode]:
        roots: list[SearchNode] = []
        default = get_default_config()
        roots.append(SearchNode(config=default, stage=Stage.PRELIMINARY))
        for _ in range(max(0, self.cfg.num_seeds - 1)):
            # Mutate default so each root probes a different axis
            roots.append(SearchNode(config=mutate_config(default, self.rng), stage=Stage.PRELIMINARY))
        for n in roots:
            self.nodes[n.id] = n
            self._stage_node_counts[Stage.PRELIMINARY] += 1
        return roots

    # ---- selection -------------------------------------------------------------
    def select_next(self) -> SearchNode | None:
        if self._steps_taken >= self.cfg.max_steps:
            return None

        self._maybe_advance_stage()

        if self.current_stage == Stage.ABLATION:
            # Ablation is driven by AblationRunner, not select_next
            return None

        leaves = self._success_leaves()
        if not leaves:
            # Fall back: sample a fresh random config to bootstrap exploration
            child = SearchNode(
                config=sample_random_config(self.rng),
                stage=self.current_stage,
            )
            self.nodes[child.id] = child
            self._stage_node_counts[self.current_stage] += 1
            self._steps_taken += 1
            return child

        parent = max(leaves, key=lambda n: n.score)
        child_config = mutate_config(parent.config, self.rng)
        child = SearchNode(config=child_config, stage=self.current_stage)
        parent.add_child(child)
        self.nodes[child.id] = child
        self._stage_node_counts[self.current_stage] += 1
        self._steps_taken += 1
        return child

    def _success_leaves(self) -> list[SearchNode]:
        return [
            n
            for n in self.nodes.values()
            if n.status == NodeStatus.SUCCESS and not n.children_ids
        ]

    # ---- reporting -------------------------------------------------------------
    def report_success(self, node_id: str, metrics: dict, score: float) -> None:
        node = self.nodes[node_id]
        node.mark_success(metrics, score)
        best = self.get_best_node()
        self._best_score_history.append(best.score if best else 0.0)

    def report_failure(self, node_id: str, error_msg: str) -> str:
        node = self.nodes[node_id]
        node.debug_attempts += 1
        should_debug = (
            node.debug_attempts < self.cfg.max_debug_depth
            and self.rng.random() < self.cfg.debug_prob
        )
        if should_debug:
            node.status = NodeStatus.PENDING
            node.error_msg = error_msg
            return "debug"
        node.mark_pruned(f"abandoned after {node.debug_attempts} debug attempts: {error_msg}")
        return "abandon"

    # ---- stage transitions -----------------------------------------------------
    def _maybe_advance_stage(self) -> None:
        prev = self.current_stage
        if self.current_stage == Stage.PRELIMINARY:
            if self._any_success_in(Stage.PRELIMINARY):
                self._transition(Stage.BASELINE, "a preliminary node succeeded")
        elif self.current_stage == Stage.BASELINE:
            if self._score_converged() or self._stage_node_counts[Stage.BASELINE] >= self.cfg.stage_budgets.get(Stage.BASELINE, 4):
                trigger = "converged" if self._score_converged() else "budget exhausted"
                self._transition(Stage.EXPLORATION, trigger)
        elif self.current_stage == Stage.EXPLORATION:
            budget = self.cfg.stage_budgets.get(Stage.EXPLORATION, 10)
            if self._stage_node_counts[Stage.EXPLORATION] >= budget:
                self._transition(Stage.ABLATION, "exploration budget exhausted")
        if prev != self.current_stage:
            logger.info("Stage transition: %s → %s", prev.name, self.current_stage.name)

    def _transition(self, to: Stage, trigger: str) -> None:
        self._stage_transitions.append(
            {
                "from": self.current_stage.name,
                "to": to.name,
                "trigger": trigger,
                "at_step": self._steps_taken,
                "best_score": self._best_score_history[-1] if self._best_score_history else 0.0,
            }
        )
        self.current_stage = to

    def _any_success_in(self, stage: Stage) -> bool:
        return any(n.stage == stage and n.status == NodeStatus.SUCCESS for n in self.nodes.values())

    def _score_converged(self) -> bool:
        window = self.cfg.convergence_window
        hist = self._best_score_history
        if len(hist) < window + 1:
            return False
        recent = hist[-window:]
        return (max(recent) - min(recent)) < self.cfg.convergence_eps

    # ---- queries ---------------------------------------------------------------
    def get_best_node(self) -> SearchNode | None:
        succeeded = [n for n in self.nodes.values() if n.status == NodeStatus.SUCCESS]
        return max(succeeded, key=lambda n: n.score) if succeeded else None

    def get_trajectory(self) -> list[dict]:
        return sorted(
            [n.to_dict() for n in self.nodes.values()],
            key=lambda n: (n["stage"], n["depth"], n["id"]),
        )

    def get_stage_transitions(self) -> list[dict]:
        return list(self._stage_transitions)

    def iter_nodes(self) -> Iterable[SearchNode]:
        return iter(self.nodes.values())
