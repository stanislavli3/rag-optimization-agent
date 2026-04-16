"""Config-space operations: defaults, random sampling, mutations, ablations."""
from __future__ import annotations

import random
from typing import Any

from config import RAG_SEARCH_SPACE


def get_default_config() -> dict:
    """A sensible starting config drawn from the centre of the search space."""
    return {
        "chunk_size": 512,
        "chunk_overlap": 0.10,
        "top_k": 5,
        "reranker": None,
        "search_mode": "vector",
        "prompt_style": "zero-shot",
        "embedding_model": "all-MiniLM-L6-v2",
    }


def sample_random_config(rng: random.Random | None = None) -> dict:
    r = rng or random
    return {key: r.choice(values) for key, values in RAG_SEARCH_SPACE.items()}


def mutate_config(config: dict, rng: random.Random | None = None) -> dict:
    """Change exactly one parameter to a different valid value."""
    r = rng or random
    mutable = [k for k in RAG_SEARCH_SPACE if len(RAG_SEARCH_SPACE[k]) > 1]
    key = r.choice(mutable)
    current = config.get(key)
    alternatives = [v for v in RAG_SEARCH_SPACE[key] if v != current]
    if not alternatives:
        return dict(config)
    return {**config, key: r.choice(alternatives)}


def get_neighbors(config: dict) -> list[dict]:
    """All configs differing from ``config`` by exactly one parameter value."""
    out: list[dict] = []
    for key, values in RAG_SEARCH_SPACE.items():
        for v in values:
            if v != config.get(key):
                out.append({**config, key: v})
    return out


def generate_ablation_configs(best_config: dict) -> list[dict]:
    """For each param where ``best_config`` diverges from the default, revert it."""
    default = get_default_config()
    out: list[dict] = []
    for key, values in RAG_SEARCH_SPACE.items():
        best_val = best_config.get(key)
        def_val = default.get(key)
        if best_val == def_val:
            continue
        if def_val not in values:
            def_val = values[0]
        ablated = {**best_config, key: def_val}
        out.append(
            {
                "ablated_param": key,
                "original_value": best_val,
                "default_value": def_val,
                "config": ablated,
            }
        )
    return out


def config_distance(a: dict, b: dict) -> int:
    """Number of parameters that differ — useful for tree proximity heuristics."""
    return sum(1 for k in RAG_SEARCH_SPACE if a.get(k) != b.get(k))


def config_is_valid(config: dict) -> bool:
    for key, values in RAG_SEARCH_SPACE.items():
        if key in config and config[key] not in values:
            return False
    return True


__all__: list[Any] = [
    "get_default_config",
    "sample_random_config",
    "mutate_config",
    "get_neighbors",
    "generate_ablation_configs",
    "config_distance",
    "config_is_valid",
]
