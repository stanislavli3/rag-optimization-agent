"""Generate the 6 demo notebooks from Python source.

Running this file (``python notebooks/_build_notebooks.py``) writes 01–06
``.ipynb`` files so they can be executed cell-by-cell or via ``jupyter nbconvert``.
"""
from __future__ import annotations

import json
from pathlib import Path


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "source": src.splitlines(keepends=True), "outputs": []}


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


_BOOTSTRAP = """
import sys, pathlib
ROOT = pathlib.Path().resolve().parent if pathlib.Path('notebooks').exists() else pathlib.Path().resolve()
sys.path.insert(0, str(ROOT))
"""


def nb_01_ingest() -> dict:
    cells = [
        _md("# 01 · Ingestion — loader + chunker\nLoad local documents, chunk at 256/512/1024 and compare chunk counts."),
        _code(_BOOTSTRAP),
        _code(
            "from src.ingest.loader import load_documents, _Document\n"
            "from src.ingest.chunker import chunk_documents\n"
            "\n"
            "# Fake 3 docs if no sample_docs present\n"
            "docs = load_documents('data/sample_docs') or [\n"
            "    _Document(('The project tracks multiple agents collaborating. ' * 80),\n"
            "              {'source_file': f'd{i}.txt', 'page_number': 0, 'file_type': 'txt'}) for i in range(3)\n"
            "]\n"
            "print(f'{len(docs)} documents loaded')\n"
        ),
        _code(
            "for size in (256, 512, 1024):\n"
            "    chunks = chunk_documents(docs, chunk_size=size, overlap_ratio=0.10)\n"
            "    print(f'chunk_size={size:>5} -> {len(chunks)} chunks | first meta keys: {list(chunks[0].metadata)}')\n"
        ),
    ]
    return _notebook(cells)


def nb_02_testgen() -> dict:
    cells = [
        _md("# 02 · TestGen — KG → seeds → evolve → difficulty\nThe generative centrepiece of the project."),
        _code(_BOOTSTRAP),
        _code(
            "from src.ingest.loader import _Document\n"
            "from src.ingest.chunker import chunk_documents\n"
            "from src.testgen.llm import MockLLM  # swap for get_llm() when ANTHROPIC_API_KEY is set\n"
            "from src.testgen.knowledge_graph import extract_knowledge_graph\n"
            "from src.testgen.seed_generator import generate_seeds\n"
            "from src.testgen.evol_instruct import evolve_batch\n"
            "from src.testgen.difficulty_matrix import compute_difficulty_matrix, difficulty_breakdown_report\n"
            "from src.testgen.groundtruth import verify_groundedness\n"
            "\n"
            "llm = MockLLM()\n"
            "docs = [_Document('Alpha uses Rust. Bob leads Alpha. Beta uses Go. Alice leads Beta.' * 20,\n"
            "        {'source_file': f'd{i}.txt', 'page_number':0,'file_type':'txt'}) for i in range(5)]\n"
            "chunks = chunk_documents(docs, chunk_size=256, overlap_ratio=0.1)\n"
        ),
        _md("## Knowledge graph"),
        _code(
            "kg = extract_knowledge_graph(chunks, llm)\n"
            "print('nodes:', len(kg.nodes), 'edges:', len(kg.edges))\n"
            "for f in kg.get_facts()[:5]:\n"
            "    print('FACT:', f['fact'])\n"
        ),
        _md("## Seeds"),
        _code(
            "seeds = generate_seeds(kg, llm, num_seeds=20, chunks=chunks)\n"
            "for s in seeds[:5]:\n"
            "    print('-', s['question'])\n"
        ),
        _md("## Evolve (4 types)"),
        _code(
            "evolved = evolve_batch(seeds, kg, llm)\n"
            "from collections import Counter\n"
            "print(Counter(q['question_type'] for q in evolved))\n"
            "for q in evolved[:4]:\n"
            "    print(q['question_type'], '|', q['question'][:140])\n"
        ),
        _md("## Groundedness filter + difficulty matrix"),
        _code(
            "kept = []\n"
            "for q in evolved:\n"
            "    ok, conf = verify_groundedness(q['question'], q['ground_truth_answer'], q['ground_truth_context'], llm)\n"
            "    if ok and conf >= 0.5:\n"
            "        kept.append(q)\n"
            "print('kept:', len(kept), 'of', len(evolved))\n"
            "df = compute_difficulty_matrix(kept)\n"
            "print(difficulty_breakdown_report(df))\n"
            "df[['question','question_type','reasoning_depth','semantic_distance','difficulty']].head()\n"
        ),
    ]
    return _notebook(cells)


def nb_03_pipeline() -> dict:
    cells = [
        _md("# 03 · RAG pipeline — one config, one testset"),
        _code(_BOOTSTRAP),
        _code(
            "from src.ingest.loader import _Document\n"
            "from src.pipeline.runner import run_pipeline\n"
            "from src.testgen.llm import MockLLM\n"
            "\n"
            "docs = [_Document('Alpha uses Rust. Bob leads Alpha. Beta uses Go. Alice leads Beta.' * 20,\n"
            "        {'source_file': 'd.txt', 'page_number':0,'file_type':'txt'})]\n"
            "queries = [{'question': 'Which language does Alpha use?'}, {'question': 'Who leads Beta?'}]\n"
            "results = run_pipeline({'chunk_size':256,'top_k':3,'search_mode':'hybrid','prompt_style':'chain-of-thought'}, docs, queries, llm=MockLLM())\n"
            "for r in results:\n"
            "    print('Q:', r['question'])\n"
            "    print('A:', r['answer'][:120])\n"
            "    print('ctx hits:', len(r['retrieved_contexts']))\n"
        ),
    ]
    return _notebook(cells)


def nb_04_eval() -> dict:
    cells = [
        _md("# 04 · Evaluation — RAGAS + IR + stratified + stats"),
        _code(_BOOTSTRAP),
        _code(
            "from src.evaluation.ragas_eval import evaluate_ragas, evaluate_by_difficulty, evaluate_by_question_type\n"
            "from src.evaluation.ir_metrics import evaluate_ir\n"
            "from src.evaluation.statistics import paired_bootstrap_test, cohens_d\n"
            "import pandas as pd\n"
            "\n"
            "results = [\n"
            "    {'question':'q1','answer':'paris','retrieved_contexts':['paris is capital of france'],\n"
            "     'ground_truth_answer':'paris','ground_truth_context':'paris is capital of france'},\n"
            "    {'question':'q2','answer':'atp','retrieved_contexts':['mitochondria produce atp'],\n"
            "     'ground_truth_answer':'atp','ground_truth_context':'mitochondria produce atp'},\n"
            "]\n"
            "print('RAGAS:', evaluate_ragas(results))\n"
            "print('IR:', evaluate_ir([['a','b','c'],['b','a']], [['a'],['a','c']], k=3))\n"
            "print('bootstrap:', paired_bootstrap_test([0.8,0.7,0.9,0.85], [0.6,0.55,0.7,0.65], n_bootstrap=500))\n"
            "print('cohens_d:', cohens_d([0.8,0.7,0.9,0.85],[0.6,0.55,0.7,0.65]))\n"
            "\n"
            "df = pd.DataFrame([\n"
            "    {'question':'q1','difficulty':'easy','question_type':'simple','ground_truth_answer':'paris','ground_truth_context':'paris is capital of france'},\n"
            "    {'question':'q2','difficulty':'hard','question_type':'reasoning','ground_truth_answer':'atp','ground_truth_context':'mitochondria produce atp'},\n"
            "])\n"
            "print('by difficulty:', evaluate_by_difficulty(results, df))\n"
            "print('by question_type:', evaluate_by_question_type(results, df))\n"
        ),
    ]
    return _notebook(cells)


def nb_05_bfts() -> dict:
    cells = [
        _md("# 05 · BFTS demo — tree grows across 4 stages"),
        _code(_BOOTSTRAP),
        _code(
            "import random\n"
            "from config import BFTSConfig\n"
            "from src.optimizer.bfts_loop import BFTSLoop\n"
            "from src.optimizer.tree_node import Stage\n"
            "from src.visualization import plot_tree, plot_trajectory, plot_ablation_bar\n"
            "\n"
            "rng = random.Random(0)\n"
            "def run_fn(cfg, docs, queries):\n"
            "    return [{'question':q['question'],'answer':'ok','retrieved_contexts':[]} for q in queries]\n"
            "def eval_fn(results):\n"
            "    return {'ragas_score': 0.3 + rng.random()*0.6}\n"
            "\n"
            "loop = BFTSLoop(documents=['d'], testset=[{'question':'q'}], run_fn=run_fn, eval_fn=eval_fn,\n"
            "                bfts_config=BFTSConfig(num_seeds=3, max_steps=15,\n"
            "                    stage_budgets={Stage.PRELIMINARY:3, Stage.BASELINE:4, Stage.EXPLORATION:6, Stage.ABLATION:3}))\n"
            "\n"
            "events = []\n"
            "for ev in loop.run_iter():\n"
            "    events.append(ev)\n"
            "    if ev['type'] in ('stage_transition', 'ablation_complete', 'search_complete'):\n"
            "        print(ev['type'], '->', ev.get('current_stage') or ev.get('summary', {}).get('best_score'))\n"
        ),
        _code(
            "summary = loop._final_summary()\n"
            "print('best score:', summary['best_score'])\n"
            "print('tree:', summary['tree_summary'])\n"
            "for t in summary['stage_transitions']:\n"
            "    print(' ', t)\n"
        ),
        _code(
            "plot_tree(loop.get_tree_visualization_data())\n"
        ),
        _code("plot_trajectory(summary['trajectory'], summary['stage_transitions'])\n"),
        _code("plot_ablation_bar(summary['ablation_report'])\n"),
    ]
    return _notebook(cells)


def nb_06_end_to_end() -> dict:
    cells = [
        _md("# 06 · End-to-end — upload → testgen → optimize → results"),
        _code(_BOOTSTRAP),
        _code(
            "from src.ingest.loader import _Document\n"
            "from src.ingest.chunker import chunk_documents\n"
            "from src.testgen.llm import MockLLM\n"
            "from src.testgen.pipeline import TestGenPipeline\n"
            "from src.pipeline.runner import run_pipeline\n"
            "from src.evaluation.ragas_eval import evaluate_ragas\n"
            "from src.optimizer.bfts_loop import BFTSLoop\n"
            "from src.optimizer.tree_node import Stage\n"
            "from config import BFTSConfig\n"
            "\n"
            "docs = [_Document('Alpha uses Rust. Bob leads Alpha. Beta uses Go. Alice leads Beta.' * 20,\n"
            "        {'source_file':f'd{i}.txt','page_number':0,'file_type':'txt'}) for i in range(3)]\n"
            "pipe = TestGenPipeline(llm=MockLLM(), target_size=10, groundedness_threshold=0.5)\n"
            "df = pipe.generate(docs)\n"
            "print('testset rows:', len(df))\n"
        ),
        _code(
            "llm = MockLLM()\n"
            "def run_fn(cfg, documents, queries):\n"
            "    return run_pipeline(cfg, documents, queries, llm=llm)\n"
            "def eval_fn(results):\n"
            "    return evaluate_ragas(results)\n"
            "\n"
            "loop = BFTSLoop(documents=docs, testset=df, run_fn=run_fn, eval_fn=eval_fn,\n"
            "                bfts_config=BFTSConfig(num_seeds=2, max_steps=6,\n"
            "                    stage_budgets={Stage.PRELIMINARY:2, Stage.BASELINE:2, Stage.EXPLORATION:2, Stage.ABLATION:2}))\n"
            "summary = loop.run()\n"
            "print('best:', summary['best_config'])\n"
            "print('score:', summary['best_score'])\n"
            "print('ablations:', len(summary['ablation_report']))\n"
        ),
    ]
    return _notebook(cells)


BUILDERS = {
    "01_ingest_test.ipynb": nb_01_ingest,
    "02_testgen_demo.ipynb": nb_02_testgen,
    "03_pipeline_test.ipynb": nb_03_pipeline,
    "04_eval_test.ipynb": nb_04_eval,
    "05_bfts_demo.ipynb": nb_05_bfts,
    "06_end_to_end.ipynb": nb_06_end_to_end,
}


def main() -> None:
    out_dir = Path(__file__).parent
    for name, builder in BUILDERS.items():
        (out_dir / name).write_text(json.dumps(builder(), indent=1))
        print("wrote", name)


if __name__ == "__main__":
    main()
