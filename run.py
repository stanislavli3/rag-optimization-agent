"""CLI entry point — end-to-end pipeline runner.

Usage
-----
    python run.py --docs data/sample_docs --experiments 15 --strategy bfts
    python run.py --docs data/sample_docs --strategy random --testset data/testsets/my.csv
    python run.py --show-config
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

from config import BFTSConfig, Config, RAG_SEARCH_SPACE, TestGenConfig

logger = logging.getLogger("rag-optimizer")


def _log_configure(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _show_config_only() -> None:
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


def _hardcoded_testset() -> list[dict]:
    return [
        {
            "question": "What is the overall topic of the provided documents?",
            "ground_truth_answer": "",
            "ground_truth_context": "",
            "difficulty": "easy",
            "question_type": "simple",
        },
        {
            "question": "Summarise the main argument of the documents in one sentence.",
            "ground_truth_answer": "",
            "ground_truth_context": "",
            "difficulty": "medium",
            "question_type": "reasoning",
        },
    ]


def _load_testset(path: str | Path) -> list[dict]:
    import pandas as pd  # type: ignore

    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _generate_testset(documents: list[Any], cfg: Config, tg: TestGenConfig) -> list[dict]:
    from src.testgen.llm import get_llm
    from src.testgen.pipeline import TestGenPipeline

    pipe = TestGenPipeline(
        llm=get_llm(),
        chunk_size=cfg.chunk_size,
        overlap_ratio=cfg.chunk_overlap,
        target_size=tg.target_size,
        groundedness_threshold=tg.groundedness_threshold,
        distribution=tg.distribution,
        out_dir=cfg.testset_dir,
    )
    df = pipe.generate(documents)
    return df.to_dict(orient="records")


def _random_search(docs, queries, run_fn, eval_fn, n_steps: int, rng_seed: int = 42) -> dict:
    from src.optimizer.search_space import sample_random_config

    rng = random.Random(rng_seed)
    best_score = -1.0
    best_config: dict | None = None
    best_metrics: dict = {}
    trajectory: list[dict] = []
    for i in range(n_steps):
        cfg_i = sample_random_config(rng)
        try:
            res = run_fn(cfg_i, docs, queries)
            m = eval_fn(res)
            score = float(m.get("ragas_score", 0.0))
        except Exception as e:
            trajectory.append({"iter": i, "config": cfg_i, "error": str(e), "score": 0.0})
            continue
        trajectory.append({"iter": i, "config": cfg_i, "metrics": m, "score": score})
        if score > best_score:
            best_score, best_config, best_metrics = score, cfg_i, m
        print(f"[random {i + 1}/{n_steps}] score={score:.3f} (best={best_score:.3f})")
    return {
        "best_config": best_config,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "trajectory": trajectory,
        "ablation_report": [],
        "stage_transitions": [],
        "tree_summary": {"RANDOM": {"count": len(trajectory)}},
    }


def _greedy_search(docs, queries, run_fn, eval_fn, n_steps: int) -> dict:
    from src.optimizer.search_space import get_default_config, get_neighbors

    current = get_default_config()
    res = run_fn(current, docs, queries)
    m = eval_fn(res)
    best_score = float(m.get("ragas_score", 0.0))
    best_config = current
    best_metrics = m
    trajectory = [{"iter": 0, "config": current, "metrics": m, "score": best_score}]
    steps = 1
    while steps < n_steps:
        improved = False
        for nb in get_neighbors(best_config):
            if steps >= n_steps:
                break
            try:
                res = run_fn(nb, docs, queries)
                m = eval_fn(res)
                score = float(m.get("ragas_score", 0.0))
            except Exception as e:
                trajectory.append({"iter": steps, "config": nb, "error": str(e), "score": 0.0})
                steps += 1
                continue
            trajectory.append({"iter": steps, "config": nb, "metrics": m, "score": score})
            print(f"[greedy {steps + 1}/{n_steps}] score={score:.3f} (best={best_score:.3f})")
            steps += 1
            if score > best_score:
                best_score, best_config, best_metrics = score, nb, m
                improved = True
                break
        if not improved:
            break
    return {
        "best_config": best_config,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "trajectory": trajectory,
        "ablation_report": [],
        "stage_transitions": [],
        "tree_summary": {"GREEDY": {"count": len(trajectory)}},
    }


def _bfts(docs, queries, run_fn, eval_fn, n_steps: int) -> dict:
    from src.optimizer.bfts_loop import BFTSLoop

    bfts_cfg = BFTSConfig(max_steps=n_steps)
    loop = BFTSLoop(documents=docs, testset=queries, run_fn=run_fn, eval_fn=eval_fn, bfts_config=bfts_cfg)
    for ev in loop.run_iter():
        t = ev.get("type")
        if t == "node_complete":
            status = ev["status"]
            score = ev.get("score")
            stage = ev["node"]["stage"]
            stage_name = _stage_label(stage)
            if status == "success":
                print(f"[{stage_name}] Node {ev['node']['id']} → score={score:.3f}")
            else:
                print(f"[{stage_name}] Node {ev['node']['id']} FAILED ({ev.get('decision', '')})")
        elif t == "stage_transition":
            print(f"*** Stage → {ev['current_stage']} ({ev['transitions'][0].get('trigger')})")
        elif t == "ablation_node":
            if "score" in ev:
                print(f"[ABLATION] {ev['param']} reverted → score={ev['score']:.3f}")
    return loop._final_summary()


_STAGE_LABELS = {
    1: "Stage 1/4 PRELIMINARY",
    2: "Stage 2/4 BASELINE",
    3: "Stage 3/4 EXPLORATION",
    4: "Stage 4/4 ABLATION",
}


def _stage_label(stage) -> str:
    if isinstance(stage, int):
        return _STAGE_LABELS.get(stage, str(stage))
    if isinstance(stage, str):
        return {
            "preliminary": "Stage 1/4 PRELIMINARY",
            "baseline": "Stage 2/4 BASELINE",
            "exploration": "Stage 3/4 EXPLORATION",
            "ablation": "Stage 4/4 ABLATION",
        }.get(stage.lower(), stage)
    return str(stage)


def _save_artifacts(out_dir: Path, summary: dict, testset: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd  # type: ignore

    (out_dir / "best_config.json").write_text(json.dumps(summary["best_config"], indent=2, default=str))
    try:
        import yaml  # type: ignore
        (out_dir / "best_config.yaml").write_text(yaml.safe_dump(summary["best_config"], sort_keys=False))
    except Exception:
        pass

    (out_dir / "ablation_report.json").write_text(json.dumps(summary["ablation_report"], indent=2, default=str))
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "best_score": summary["best_score"],
                "best_metrics": summary["best_metrics"],
                "tree_summary": summary["tree_summary"],
                "stage_transitions": summary["stage_transitions"],
            },
            indent=2,
            default=str,
        )
    )

    trajectory = summary.get("trajectory", [])
    if trajectory:
        pd.DataFrame(trajectory).to_csv(out_dir / "trajectory.csv", index=False)
    if testset:
        pd.DataFrame(testset).to_csv(out_dir / "testset_used.csv", index=False)


def _print_summary(summary: dict) -> None:
    print("\n=============================================")
    print(" Best config")
    print("=============================================")
    pprint(summary["best_config"])
    print(f"\nBest score: {summary['best_score']:.4f}")
    if summary.get("best_metrics"):
        print("\nMetrics:")
        for k, v in summary["best_metrics"].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
    if summary.get("ablation_report"):
        print("\nAblation (most important component first):")
        print(f"  {'param':<18} {'with':<20} {'without':<15} {'Δ':>8} {'contrib %':>10}")
        for r in summary["ablation_report"]:
            print(
                f"  {r['param']:<18} {str(r['with_value']):<20} {str(r['without_value']):<15} "
                f"{r['delta']:>8.3f} {r['contribution_pct']:>9.1f}%"
            )
    if summary.get("stage_transitions"):
        print("\nStage transitions:")
        for t in summary["stage_transitions"]:
            print(f"  {t['from']:>12} → {t['to']:<12} @step {t['at_step']} ({t['trigger']})")


def main() -> int:
    ap = argparse.ArgumentParser(prog="rag-optimizer")
    ap.add_argument("--docs", type=str, default=None, help="Directory of PDF/MD/TXT documents")
    ap.add_argument("--experiments", type=int, default=20, help="Max optimizer steps")
    ap.add_argument("--strategy", choices=["bfts", "random", "greedy"], default="bfts")
    ap.add_argument("--testset", type=str, default=None, help="Existing testset CSV (skips testgen)")
    ap.add_argument("--skip-testgen", action="store_true", help="Use minimal hardcoded testset")
    ap.add_argument("--show-config", action="store_true", help="Print config and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _log_configure(args.verbose)

    if args.show_config or args.docs is None:
        _show_config_only()
        return 0

    cfg = Config()
    tg = TestGenConfig()

    from src.ingest.loader import load_documents
    from src.pipeline.runner import run_pipeline
    from src.evaluation.ragas_eval import evaluate_ragas
    from src.testgen.llm import get_llm

    documents = load_documents(args.docs)
    if not documents:
        print(f"No documents found under {args.docs}", file=sys.stderr)
        return 2

    if args.testset:
        testset = _load_testset(args.testset)
    elif args.skip_testgen:
        testset = _hardcoded_testset()
    else:
        testset = _generate_testset(documents, cfg, tg)

    print(f"Loaded {len(documents)} documents, {len(testset)} test questions")

    llm = get_llm()

    def run_fn(config_dict, docs, queries):
        return run_pipeline(config_dict, docs, queries, llm=llm, persist_dir=cfg.chroma_persist_dir)

    def eval_fn(results):
        return evaluate_ragas(results, llm=None)

    if args.strategy == "random":
        summary = _random_search(documents, testset, run_fn, eval_fn, args.experiments)
    elif args.strategy == "greedy":
        summary = _greedy_search(documents, testset, run_fn, eval_fn, args.experiments)
    else:
        summary = _bfts(documents, testset, run_fn, eval_fn, args.experiments)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.results_dir / f"run_{args.strategy}_{ts}"
    _save_artifacts(out_dir, summary, testset)
    _print_summary(summary)
    print(f"\nArtifacts saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
