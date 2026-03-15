# Evaluation-Driven Agentic Optimization of Transformer-Based RAG Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Course](https://img.shields.io/badge/Course-Deep%20%26%20Generative%20Learning-teal.svg)](#alignment-with-course-topics)

> An agent-driven experimental framework that replaces manual RAG tuning with systematic, evaluation-driven optimization — treating RAG configuration as a scientific process, not an engineering heuristic.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Project Goals](#project-goals)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Evaluation Framework](#evaluation-framework)
- [Optimization Search Space](#optimization-search-space)
- [Data & Experimental Setup](#data--experimental-setup)
- [Technical Stack](#technical-stack)
- [Deliverables](#deliverables)
- [Alignment with Course Topics](#alignment-with-course-topics)
- [Related Work](#related-work)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [References](#references)
- [License](#license)

---

## Overview

This project develops an evaluation-driven agentic framework for analyzing and improving transformer-based generative models, with a specific focus on **Retrieval-Augmented Generation (RAG)** systems. RAG serves as a controlled experimental setting for studying how conditioning, attention, and retrieval quality influence autoregressive generation.

Inspired by the scientific workflow modeled in Sakana AI's AI Scientist (hypothesis → experiment → evaluation → decision → iteration), this project applies the same principles in a constrained, reproducible, and measurable manner. The core insight: **RAG optimization is best framed as a latent variable model optimization problem** where evaluation metrics serve as non-differentiable reward signals.

### Deep Learning Connection

RAG is a latent variable model: **P(y|x) = Σ_z P(y|x,z) · P(z|x)**, where retrieved documents *z* act as discrete latent variables. This parallels VAE structure (retriever = encoder, generator = decoder) with ELBO optimization (Lewis et al., NeurIPS 2020). When retrieved passages are concatenated with the input, transformer self-attention treats them as additional key-value memory, extending parametric memory with non-parametric external knowledge.

---

## Motivation

Transformer-based generative models are highly sensitive to how external context is retrieved, structured, and presented during generation. In practice:

- **RAG adoption hit 51% among enterprises in 2024** (Menlo Ventures), yet optimization remains ad-hoc
- The configuration search space spans **46,000+ possible combinations** (RAGSmith, 2025)
- Chunking strategy alone causes up to **9% variance in recall** between best and worst approaches
- The "lost in the middle" effect degrades performance by **>30%** when relevant information shifts to middle positions

This project replaces manual trial-and-error with an **agent-driven experimental loop** that systematically explores design choices and evaluates their impact using quantitative metrics.

---

## Project Goals

| # | Goal | Description |
|---|------|-------------|
| 1 | **Analyze** | How retrieval configuration affects autoregressive generation quality, studying attention over retrieved context as external key-value memory |
| 2 | **Automate** | Build an agent that systematically explores the RAG design space via controlled ablation studies |
| 3 | **Evaluate** | Apply RAGAS metrics (faithfulness, groundedness, answer relevance) + IR metrics (NDCG@k, MRR) with bootstrap CIs and effect sizes |
| 4 | **Deliver** | Interactive Streamlit web dashboard with leaderboard → comparison → trace drill-down — not a notebook or standalone model |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Controller                         │
│  ┌───────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Hypothesize│→│  Execute   │→│ Evaluate  │→│  Decide   │  │
│  │ RAG config │  │ pipeline   │  │ RAGAS +   │  │ compare,  │  │
│  │            │  │ on dataset │  │ IR metrics│  │ iterate   │  │
│  └───────────┘  └───────────┘  └──────────┘  └──────────┘  │
│       ↑                                            │         │
│       └────────────── feedback loop ───────────────┘         │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
    │ Vector  │         │  LLM    │         │Experiment│
    │ Index   │         │Generator│         │  Logs    │
    │(Chroma) │         │(Mistral)│         │ (JSON)   │
    └─────────┘         └─────────┘         └─────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Transformer-based Generator** | Autoregressive LLM conditioned on retrieved context (Mistral-7B / Llama-3.1-8B) |
| **Retriever & Index** | ChromaDB vector store + BM25 hybrid search with configurable embedding models |
| **Agent Controller** | Manages hypothesis generation, experiment scheduling, and MAB-style decision-making |
| **Evaluation Harness** | Computes RAGAS + IR metrics with paired bootstrap significance testing |
| **Experiment Logger** | Stores configurations, metrics, and agent decision trajectories for reproducibility |
| **Web Dashboard** | Streamlit app for interactive exploration of results |

---

## Methodology

### Agentic Experimentation Loop

The system follows a closed-loop experimental process inspired by the scientific method:

**Hypothesize → Execute → Evaluate → Decide → Iterate**

#### 1. Hypothesis Generation
The agent proposes testable RAG configurations by varying:
- Chunk size (128, 256, 512 tokens)
- Top-k retrieval depth (3, 5, 10 documents)
- Reranking strategy (none, ColBERT, cross-encoder)
- Prompting technique (zero-shot, few-shot, chain-of-thought)
- Embedding model (all-MiniLM-L6-v2, BGE-M3, E5)

#### 2. Experiment Execution
Each configuration runs under a **fixed computational budget** using:
- A consistent evaluation dataset with ground-truth answers
- A standardized transformer-based generator
- Controlled retrieval and indexing parameters
- Seeded runs for reproducibility

#### 3. Evaluation
Outputs are assessed using RAG-specific quantitative metrics (see [Evaluation Framework](#evaluation-framework)).

#### 4. Decision & Iteration
The agent:
- Compares results against baseline and previous configurations
- Selects best-performing configuration based on composite scores
- Proposes subsequent experiments informed by observed patterns
- Uses **multi-armed bandit (MAB) exploration** to balance exploration vs. exploitation (inspired by AutoRAG-HP, EMNLP 2024)

---

## Evaluation Framework

### Retrieval Metrics

| Metric | What It Measures |
|--------|-----------------|
| **NDCG@k** | Rank-aware graded relevance (MTEB default) |
| **Context Precision** | Are relevant chunks ranked higher than irrelevant ones? (RAGAS) |
| **Context Recall** | Does the context contain all needed information? (RAGAS) |
| **MRR** | How quickly does the first relevant document surface? |
| **Recall@k / Precision@k** | Coverage vs. noise tradeoff |

### Generation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Is every claim in the output supported by retrieved context? (#1 metric) |
| **Groundedness** | Are there no hallucinated claims in the output? |
| **Answer Relevance** | Does the generated answer address the user query? |
| **Answer Correctness** | F1-like factual overlap with ground truth |
| **RAGAS Score** | Composite score across all dimensions |

### Efficiency Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Latency (ms)** | End-to-end query response time |
| **Token Usage** | Total tokens consumed per query |
| **Cost per Query** | Estimated compute cost |

### Statistical Rigor

All comparisons use:
- **Paired bootstrap tests** (10,000+ samples) for significance
- **BCa confidence intervals** (bias-corrected and accelerated)
- **Cohen's d** effect sizes alongside p-values
- **Benjamini-Hochberg correction** for multiple comparisons across configurations

### Why RAGAS over BLEU/ROUGE?

BLEU and ROUGE measure surface-level n-gram overlap. They cannot detect hallucination, assess groundedness, or evaluate whether answers are faithful to retrieved context. RAGAS uses LLM-as-judge for semantic evaluation of each dimension independently, which is essential for RAG-specific quality assessment.

---

## Optimization Search Space

The agent explores these parameters, yielding ~46,000+ possible configurations:

| Parameter | Options | Key Research Finding |
|-----------|---------|---------------------|
| **Chunk Size** | 128, 256, 512, 1024 tokens | Factoid queries → small chunks; reasoning → large (NVIDIA, 2025) |
| **Chunk Overlap** | 10%, 15%, 20% | 15% optimal on FinanceBench (NVIDIA) |
| **Chunking Strategy** | Fixed, recursive, semantic | Recursive at 512 tokens is strongest default |
| **Top-k Depth** | 3, 5, 10, 20 | Fewer, more relevant docs often beat more docs |
| **Reranking** | None, ColBERT, cross-encoder | Up to 67% retrieval failure reduction (Anthropic) |
| **Hybrid Search** | Vector-only, BM25+vector (α=0.3–0.7) | 15–30% precision improvements |
| **Prompting** | Zero-shot, few-shot, CoT | CoT-RAG improves across 9 reasoning datasets |
| **Embedding Model** | all-MiniLM-L6-v2, BGE-M3, E5 | Domain fine-tuning yields +10–30% gains |
| **Query Transform** | None, HyDE, decomposition | HyDE adds 25–60% latency but bridges query-doc gap |

---

## Data & Experimental Setup

### Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| **Natural Questions** | 323K queries | Real Google searches against Wikipedia — primary benchmark |
| **HotpotQA** | 113K queries | Multi-hop reasoning; tests iterative retrieval |
| **RAGAS Synthetic** | 200–500 custom | Auto-generated via knowledge graph transforms |

### Evaluation Dataset Design

- 50 hand-crafted golden questions with expert-verified answers
- 200–300 synthetic questions from RAGAS (50% simple, 25% reasoning, 25% multi-context)
- 50+ adversarial edge cases
- Version-controlled — never modified in-place

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| **Generator** | Mistral-7B / Llama-3.1-8B via HuggingFace Transformers |
| **Retriever** | ChromaDB (vector) + BM25 (keyword) hybrid search |
| **Embeddings** | all-MiniLM-L6-v2, BGE-M3 via sentence-transformers |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Evaluation** | RAGAS + DeepEval + custom IR metric implementations |
| **Orchestration** | LangChain + custom Python agent controller |
| **Dashboard** | Streamlit + Plotly |
| **Experiment Tracking** | JSON logs + SQLite |

---

## Deliverables

### Primary: Interactive Web Dashboard (Streamlit)

| Page | Description |
|------|-------------|
| **Configuration Lab** | Select RAG parameters, run experiments via UI |
| **Results Dashboard** | Heatmaps, radar charts, KPI cards with metric deltas |
| **Comparison View** | Side-by-side configs with significance indicators (*, **, ***) |
| **Agent Trajectory** | Decision path visualization across iterations |
| **Export** | CSV/JSON results + publication-ready SVG/PDF charts |

### Supporting Deliverables

- **Reproducible Pipeline** — Documented code, configs, README, version-controlled datasets
- **Quantitative Report** — Ablation tables, bootstrap CIs, effect sizes, per-question-type analysis
- **Agent Decision Logs** — Full trajectory: what was tried, scores achieved, rationale for next config
- **Final Poster & Presentation** — Summarizing methodology, findings, and insights

---

## Alignment with Course Topics

This project directly addresses core topics in **Deep and Generative Learning**:

| Course Topic | Connection |
|-------------|------------|
| **Transformers & Attention** | Attention over retrieved context as external key-value memory; "lost-in-the-middle" effect from positional encoding decay |
| **Autoregressive Models** | Conditional generation P(y\|x,D) where conditioning set D is dynamically determined at inference |
| **Latent Variable Models / VAEs** | RAG as discrete latent variable model with ELBO optimization; retriever = encoder, generator = decoder |
| **Evaluation of Generative Systems** | RAGAS framework: 30+ quantitative metrics for retrieval and generation quality |
| **Real-World Implementation** | Production-style Streamlit web application for interactive RAG analysis |
| **Emerging Trends** | Agentic AI for automated pipeline optimization; MAB-based hyperparameter search |

---

## Related Work

| System | What It Does | How We Differ |
|--------|-------------|--------------|
| **AI Scientist v2** (Sakana AI, 2025) | Agentic tree search for automated experiments; first AI paper accepted at ICLR workshop | We adapt the loop for constrained, reproducible RAG-specific optimization with robust RAGAS evaluation |
| **RAGAS** (Es et al., EACL 2024) | Reference-free RAG evaluation framework with 30+ metrics | We integrate RAGAS as our primary evaluation backend |
| **DeepEval** (Confident AI) | "Pytest for LLMs" with debuggable scores and CI/CD integration | We use DeepEval for JSON-confineable metric computation |
| **AutoRAG-HP** (EMNLP 2024) | Multi-armed bandit for RAG hyperparameter tuning; 80% fewer API calls than grid search | We adopt MAB exploration strategy within our agentic loop |
| **RAGSmith** (2025) | NAS-inspired genetic algorithm over 46,080 RAG configurations | We frame our search space similarly but use agent-driven sequential exploration |
| **DSPy** (Stanford NLP) | Programmatic LLM pipeline optimization via Bayesian optimization | Complementary approach; we focus on retrieval-side optimization |
| **Self-RAG** (ICLR 2024) | LM generates reflection tokens controlling retrieval and self-critique | Informs our agent's decide-to-retrieve logic |

---

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
2. Es, S., et al. (2024). "RAGAs: Automated Evaluation of Retrieval Augmented Generation." *EACL 2024*.
3. Yamada, Y., et al. (2025). "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search." *arXiv:2504.08066*.
4. Beel, J., et al. (2025). "Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results." *arXiv:2502.14297*.
5. Asai, A., et al. (2024). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR 2024 (Oral)*.
6. NVIDIA (2025). "Finding the Best Chunking Strategy for Accurate AI Responses." *NVIDIA Technical Blog*.
7. Anthropic (2024). "Introducing Contextual Retrieval." *anthropic.com*.
8. Khattab, O., et al. (2024). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." *ICLR 2024*.
9. Liu, N., et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." *TACL*.
10. Firecrawl (2025). "Best Chunking Strategies for RAG in 2025."

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

*Stanislav Li — Deep & Generative Learning — Spring 2026*
