"""SearchNode and related enums for the BFTS optimizer."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import IntEnum

from config import Stage  # re-exported for convenience


class NodeStatus(IntEnum):
    PENDING = 0
    RUNNING = 1
    SUCCESS = 2
    FAILED = 3
    PRUNED = 4


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class SearchNode:
    config: dict
    parent_id: str | None = None
    stage: Stage = Stage.PRELIMINARY
    metrics: dict = field(default_factory=dict)
    score: float = 0.0
    status: NodeStatus = NodeStatus.PENDING
    debug_attempts: int = 0
    error_msg: str = ""
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    id: str = field(default_factory=_new_id)

    def mark_success(self, metrics: dict, score: float) -> None:
        self.status = NodeStatus.SUCCESS
        self.metrics = metrics
        self.score = score

    def mark_failed(self, error: str) -> None:
        self.status = NodeStatus.FAILED
        self.error_msg = error

    def mark_pruned(self, reason: str = "") -> None:
        self.status = NodeStatus.PRUNED
        if reason:
            self.error_msg = reason

    def add_child(self, child: "SearchNode") -> None:
        child.parent_id = self.id
        child.depth = self.depth + 1
        self.children_ids.append(child.id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "stage": int(self.stage),
            "status": int(self.status),
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "score": self.score,
            "debug_attempts": self.debug_attempts,
            "error_msg": self.error_msg,
            "children_ids": list(self.children_ids),
            "depth": self.depth,
        }


__all__ = ["SearchNode", "NodeStatus", "Stage"]
