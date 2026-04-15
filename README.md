# RAG Optimizer

### Upload your data. Find the best RAG config. Automatically.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)

> **RAG Optimizer** is a full-stack platform that finds the optimal Retrieval-Augmented Generation configuration for your specific data. Upload a sample of your documents, and an AI agent systematically tests chunk sizes, retrieval strategies, reranking methods, and prompting techniques to find what works best for your use case.

---

## The Problem

Every RAG pipeline needs tuning, and every dataset needs different settings. Legal documents need large chunks and exact keyword matching. FAQs need small chunks and semantic search. Medical papers need aggressive reranking. There is no universal best config.

Today, engineers spend days manually tweaking parameters, eyeballing results, with no systematic comparison and no reproducibility. The search space spans **46,000+ possible configurations** (RAGSmith, 2025), making manual optimization impractical.

**RAG Optimizer automates this entire process.**

---

## Two Novel Contributions

### 1. Generative Test-Data Pipeline
Instead of requiring users to hand-label evaluation questions, the system synthesizes a complete test set from uploaded documents using a 5-step generative pipeline:

```
Documents → Knowledge Graph → Seed Q&A → Evol-Instruct Evolution → Groundedness Filter → GRADE Difficulty Matrix
```

This is conditional generation: `P(question | context, type, difficulty)`. The pipeline produces 4 question types (simple, multi-context, reasoning, conditional) and scores each on a 2D difficulty matrix (reasoning depth × semantic distance).

### 2. BFTS Optimization Loop (adapted from Sakana AI Scientist v2)
Progressive 4-stage experimentation with best-first tree search:

| Stage | What happens |
|---|---|
| **1 — Preliminary** | Prove feasibility: `num_seeds` root configs run end-to-end |
| **2 — Baseline** | Expand from best seed, tune basic params until convergence |
| **3 — Exploration** | Best-first tree expansion — always expand the highest-scoring leaf |
| **4 — Ablation** | Disable each non-default component to measure marginal contribution |

The agent expands the highest-scoring leaf, prunes failing branches, and runs ablations to confirm component contributions. Achieves near-optimal results in ~20% of the iterations grid search would require.

---

## How It Works

```
1. Upload          2. Generate Testset      3. Auto-Optimize         4. Get Results
───────────        ──────────────────       ──────────────────       ─────────────────
Upload 10-50       KG extraction →          BFTS agent runs          Dashboard shows
sample docs        seed Q&A →               15-20 experiments        best config +
                   Evol-Instruct →          on YOUR data,            metrics + charts +
                   difficulty matrix        learning from each        exportable config
```

### User Flow

1. **Upload** sample documents (PDF, text, markdown)
2. **Generate** test questions (automatic) — KG → seed → evolve → filter → score difficulty
3. **Click** "Find Best Config" — BFTS agent explores, prunes, and ablates
4. **Dashboard shows** the winning configuration with full metrics, trajectory, and ablation report
5. **Export** the config as YAML/JSON or a ready-to-use LangChain / LlamaIndex snippet

---

## Generative AI Components

The project has four explicit LLM integration points:

| # | Component | What it does |
|---|---|---|
| **1** | Knowledge Graph Extraction | LLM extracts entities, facts, and relations from each document chunk to seed question generation |
| **2** | Evol-Instruct Question Evolution | LLM evolves seed questions into harder types: multi-context (requires 2 passages), reasoning (logical inference), conditional (constraint-dependent answer) |
| **3** | RAGAS LLM-as-Judge | LLM scores each RAG run on faithfulness, answer relevancy, context precision, and context recall |
| **4** | Groundedness Filter | LLM verifies every generated Q&A pair is fully grounded in source text before it enters the evaluation set |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Upload  │ │ Optimize │ │ Results  │ │  Export  │           │
│  │ +TestGen │ │  (BFTS)  │ │Dashboard │ │ Config   │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
└───────┼─────────────┼────────────┼─────────────┼────────────────┘
        │             │            │             │
        ▼             ▼            ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Python src/ package                                             │
│                                                                  │
│  src/testgen/          src/pipeline/       src/optimizer/        │
│  ┌────────────┐        ┌──────────────┐    ┌─────────────────┐  │
│  │ KG extract │        │ indexer      │    │ ExperimentMgr   │  │
│  │ seed gen   │        │ retriever    │    │ BFTSLoop        │  │
│  │ evol inst  │        │ generator    │    │ AblationRunner  │  │
│  │ difficulty │        │ runner       │    └─────────────────┘  │
│  └────────────┘        └──────────────┘                         │
│                                                                  │
│  src/evaluation/       src/ingest/                               │
│  ┌────────────┐        ┌──────────────┐                         │
│  │ RAGAS eval │        │ loader       │                         │
│  │ IR metrics │        │ chunker      │                         │
│  │ statistics │        └──────────────┘                         │
│  └────────────┘                                                  │
└─────────────────────────────────────────────────────────────────┘
        │                       │
        ▼                       ▼
┌──────────────┐       ┌──────────────────┐
│   ChromaDB   │       │  Local LLM       │
│   + BM25     │       │  (Mistral-7B /   │
│  (retrieval) │       │  Llama-3.1-8B)   │
└──────────────┘       └──────────────────┘
```

### The BFTS Loop (Core Engine)

```python
# Stage 1: Seed
seeds = manager.seed_roots()          # num_seeds root configs
for seed in seeds: execute_node(seed)

# Stages 2–4: Main loop
while True:
    node = manager.select_next()      # best-first selection
    if node is None: break
    execute_node(node)                # run pipeline + evaluate
    # Debug-or-abandon on failure: retry with tweaked config, or prune
    # Stage transitions: PRELIMINARY → BASELINE → EXPLORATION → ABLATION
```

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit, Plotly |
| **LLM Generator** | Mistral-7B / Llama-3.1-8B via HuggingFace Transformers |
| **Embeddings** | all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5 via sentence-transformers |
| **Vector Store** | ChromaDB + BM25 hybrid search (rank-bm25) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Knowledge Graph** | networkx + spaCy |
| **Evaluation** | RAGAS + custom IR metrics (MRR, NDCG@k, P@k, R@k) |
| **Optimizer** | BFTS (Best-First Tree Search), adapted from Sakana AI Scientist v2 |
| **Statistics** | scipy — paired bootstrap tests, BCa confidence intervals, Cohen's d |

---

## Synthetic Test Generation Pipeline

The test-generation module is the primary generative AI contribution. It runs in 5 steps:

### Step 1 — Knowledge Graph Extraction
For each document chunk, the LLM extracts entities, factual statements, and relations between entities. A `networkx` graph is built and cross-chunk connections are found by matching entity names. This mirrors RAGAS's knowledge graph construction phase (Es et al., 2024).

### Step 2 — Seed Question Generation
For each KG fact, the LLM generates a natural question and a grounded answer (conditioned only on the source chunk). Seeds where the LLM cannot answer from context alone are discarded.

### Step 3 — Evol-Instruct Evolution
Seeds are evolved into 4 harder question types using the Evol-Instruct paradigm (WizardLM, 2023; adapted by RAGAS):

| Type | Description | LLM operation |
|---|---|---|
| **Simple** | Keep seed as-is | — |
| **Multi-context** | Requires combining 2 KG-connected passages | `P(question | context_1, context_2, relation)` |
| **Reasoning** | Requires logical inference beyond stated facts | `P(harder_question | seed_question, context)` |
| **Conditional** | Answer changes under a specified constraint | `P(conditional_question | seed_question, context)` |

Default distribution: simple 30%, multi-context 30%, reasoning 25%, conditional 15%.

### Step 4 — Groundedness Filter
An LLM judge verifies every evolved Q&A pair: "Is every claim in this answer supported by the context?" Pairs with confidence < 0.7 are rejected (quality gate from RAGEval, ACL 2025).

### Step 5 — GRADE Difficulty Matrix
Each question is scored on two orthogonal axes:
- **Reasoning depth** — 1-hop (simple), 2-hop (multi-context), 3-hop (reasoning/conditional)
- **Semantic distance** — cosine distance between question and context embeddings (high = vocabulary mismatch → harder retrieval)

| Difficulty | Criteria |
|---|---|
| Easy | depth=1 AND distance < 0.3 |
| Medium | depth=2 OR distance 0.3–0.6 |
| Hard | depth ≥ 3 OR distance > 0.6 |

---

## Optimization Search Space

| Parameter | Options | Why It Matters |
|-----------|---------|---------------|
| **Chunk Size** | 256, 512, 1024 tokens | Legal docs need big chunks; FAQs need small ones |
| **Chunk Overlap** | 10%, 20% | Prevents splitting key info across boundaries |
| **Top-k Depth** | 3, 5, 10 documents | More docs = more noise; fewer = coverage gaps |
| **Reranking** | None, cross-encoder | Up to 67% retrieval failure reduction |
| **Search Mode** | vector-only, hybrid (BM25+vector) | 15-30% precision improvement for keyword-heavy domains |
| **Prompt Style** | zero-shot, few-shot, chain-of-thought | CoT improves across 9 reasoning datasets |
| **Embedding Model** | all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5 | Domain fine-tuning yields +10-30% gains |

---

## Evaluation Framework

### Retrieval Metrics

| Metric | What It Measures |
|--------|-----------------|
| **NDCG@k** | Rank-aware graded relevance (MTEB default) |
| **Context Precision** | Are relevant chunks ranked higher? (RAGAS) |
| **Context Recall** | Does the context contain all needed info? (RAGAS) |
| **MRR** | Speed to first relevant document |

### Generation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Every claim supported by context? (most important) |
| **Answer Relevance** | Does the answer address the query? |
| **Answer Correctness** | F1-like factual overlap with ground truth |
| **RAGAS Score** | Composite across all dimensions |

### Stratified Evaluation
Results are broken down by difficulty level (easy / medium / hard) and question type (simple / multi-context / reasoning / conditional). This reveals WHERE the pipeline fails: retriever vocabulary mismatch vs. generator reasoning limitation.

### Statistical Rigor
All comparisons use paired bootstrap tests (10K samples), BCa confidence intervals, Cohen's d effect sizes, and significance stars (*, **, ***) in comparison tables.

---

## Dashboard Pages

| Page | What Users See |
|------|---------------|
| **Upload** | Drag-and-drop documents + configure testgen (type distribution, size) |
| **Optimization** | BFTS strategy controls, live score trajectory, stage progress |
| **Results** | Radar chart, comparison table with significance stars, ablation bar chart, stratified breakdown |
| **Export** | Best config as YAML/JSON + LangChain/LlamaIndex code snippet |

---

## Project Structure

```
rag-optimizer/
├── config.py                  # Config + BFTSConfig dataclasses + RAG_SEARCH_SPACE
├── run.py                     # CLI entry point
├── app.py                     # Streamlit UI
├── requirements.txt
│
├── src/
│   ├── ingest/
│   │   ├── loader.py          # PDF/MD/TXT → LangChain Documents
│   │   └── chunker.py         # Configurable chunking strategies
│   │
│   ├── testgen/               # ← GENERATIVE AI CORE
│   │   ├── knowledge_graph.py # Entity/relation extraction (LLM + networkx)
│   │   ├── seed_generator.py  # Initial Q&A from KG facts (LLM)
│   │   ├── evol_instruct.py   # Evol-Instruct: 4 question types (LLM)
│   │   ├── difficulty_matrix.py  # GRADE-style 2D difficulty scoring
│   │   ├── groundtruth.py     # LLM-as-judge groundedness filter
│   │   └── pipeline.py        # Orchestrates full testgen flow
│   │
│   ├── pipeline/
│   │   ├── indexer.py         # ChromaDB + BM25 with cache-key reuse
│   │   ├── retriever.py       # Vector / hybrid / reranked retrieval
│   │   ├── generator.py       # LLM answer generation (3 prompt styles)
│   │   └── runner.py          # run_pipeline(config, docs, queries)
│   │
│   ├── evaluation/
│   │   ├── ragas_eval.py      # RAGAS + stratified breakdown
│   │   ├── ir_metrics.py      # MRR, NDCG@k, P@k, R@k
│   │   └── statistics.py      # Bootstrap tests, Cohen's d
│   │
│   ├── optimizer/             # ← BFTS (Sakana AI Scientist v2)
│   │   ├── tree_node.py       # SearchNode + Stage + NodeStatus
│   │   ├── search_space.py    # Mutations, neighbors, ablation configs
│   │   ├── experiment_manager.py  # Agent: select, expand, prune, stage transitions
│   │   ├── ablation.py        # Stage 4 ablation runner
│   │   └── bfts_loop.py       # Main orchestrator (4 stages)
│   │
│   └── visualization.py       # Plotly figure builders
│
├── notebooks/
│   ├── 01_ingest_test.ipynb
│   ├── 02_testgen_demo.ipynb  # KG → evolve → difficulty pipeline demo
│   ├── 03_pipeline_test.ipynb
│   ├── 04_eval_test.ipynb
│   ├── 05_bfts_demo.ipynb     # Tree growing across 4 stages
│   └── 06_end_to_end.ipynb
│
└── data/
    ├── sample_docs/
    └── testsets/
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# CLI — full pipeline
python run.py --docs data/sample_docs --experiments 15 --strategy bfts

# CLI — skip testgen (load existing testset)
python run.py --docs data/sample_docs --testset data/testsets/my.csv

# Streamlit UI
streamlit run app.py
```

---

## Deep Learning Connection

This project is built for a **Deep & Generative Learning** course.

| Course Topic | Connection |
|-------------|------------|
| **Transformers & Attention** | Retrieved context becomes external key-value memory for transformer self-attention |
| **Autoregressive Models** | Conditional generation `P(y|x,D)` where D is dynamically determined at inference |
| **Latent Variable Models** | RAG as discrete latent variable model: `P(y|x) = Σ_z P(y|x,z)·P(z|x)`, paralleling VAE with ELBO optimization |
| **Evaluation** | RAGAS LLM-as-judge + GRADE 2D difficulty matrix |
| **Emerging Trends** | BFTS agentic tree search (Sakana AI Scientist v2); Evol-Instruct data synthesis (WizardLM) |

---

## Related Work

| System | What It Does | Gap We Fill |
|--------|-------------|------------|
| **AI Scientist v2** (Sakana AI) | Agentic tree search for automated research | We adapt BFTS specifically to RAG-config optimization with structured evaluation |
| **RAGAS** (EACL 2024) | RAG evaluation metrics + KG-based testgen | No optimization, no UI, no agent loop |
| **AutoRAG-HP** (EMNLP 2024) | MAB-based RAG tuning | No testgen, no UI, no ablation stage |
| **GRADE** (EMNLP 2025) | 2D difficulty matrix for RAG eval | No optimization, no testgen pipeline |
| **WizardLM / Evol-Instruct** (2023) | LLM-driven question evolution | General instruction tuning, not RAG evaluation |
| **RAGSmith** (2025) | NAS over 46K RAG configs | Research framework, not a usable platform |
| **LangSmith** | LLM tracing and debugging | No automated optimization |

**No existing tool offers: upload docs → synthetic testgen → BFTS agent finds best config → visual dashboard → export config.** That's the gap.

---

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
2. Es, S., et al. (2024). "RAGAs: Automated Evaluation of Retrieval Augmented Generation." *EACL 2024*.
3. Yamada, Y., et al. (2025). "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search." *arXiv:2504.08066*.
4. Xu, C., et al. (2023). "WizardLM: Empowering Large Language Models to Follow Complex Instructions." *ICLR 2024*. *(Evol-Instruct)*
5. Asai, A., et al. (2024). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR 2024*.
6. Liu, N., et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." *TACL*.
7. NVIDIA (2025). "Finding the Best Chunking Strategy for Accurate AI Responses." *NVIDIA Technical Blog*.
8. Anthropic (2024). "Introducing Contextual Retrieval." *anthropic.com*.
9. Khattab, O., et al. (2024). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." *ICLR 2024*.
10. *(GRADE paper — EMNLP 2025 Findings, citation TBD)*

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

*Stanislav Li — Deep & Generative Learning — Spring 2026*

---
