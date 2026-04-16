"""CLI entry point. Expanded in Phase 6 to orchestrate end-to-end runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pprint import pprint

from config import BFTSConfig, Config, RAG_SEARCH_SPACE, TestGenConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-optimizer")
    parser.add_argument("--strategy", choices=["bfts", "random", "greedy"], default="bfts")
    parser.add_argument("--show-config", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    bfts = BFTSConfig()
    tg = TestGenConfig()

    print("=== Config ===")
    pprint({k: (str(v) if hasattr(v, "as_posix") else v) for k, v in asdict(cfg).items()})
    print("\n=== BFTSConfig ===")
    pprint(asdict(bfts))
    print("\n=== TestGenConfig ===")
    pprint(asdict(tg))
    print("\n=== Search space ===")
    pprint(RAG_SEARCH_SPACE)
    print(f"\nStrategy: {args.strategy}")


if __name__ == "__main__":
    main()
