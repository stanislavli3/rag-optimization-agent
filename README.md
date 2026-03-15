# RAG Optimizer

### Upload your data. Find the best RAG config. Automatically.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-React-blue.svg)](https://typescriptlang.org)
[![Django](https://img.shields.io/badge/Backend-Django-green.svg)](https://djangoproject.com)

> **RAG Optimizer** is a full-stack platform that finds the optimal Retrieval-Augmented Generation configuration for your specific data. Upload a sample of your documents, and an AI agent systematically tests chunk sizes, retrieval strategies, reranking methods, and prompting techniques to find what works best for your use case.

---

## The Problem

Every RAG pipeline needs tuning, and every dataset needs different settings. Legal documents need large chunks and exact keyword matching. FAQs need small chunks and semantic search. Medical papers need aggressive reranking. There is no universal best config.

Today, engineers spend days manually tweaking parameters, eyeballing results, with no systematic comparison and no reproducibility. The search space spans **46,000+ possible configurations** (RAGSmith, 2025), making manual optimization impractical.

**RAG Optimizer automates this entire process.**

---

## How It Works

```
1. Upload          2. Auto-Optimize         3. Get Results
───────────        ──────────────────       ─────────────────
Upload 10-50       Agent runs 15-20         Dashboard shows
sample docs +      experiments on YOUR      best config +
test questions     data, learning from      metrics + charts +
(or auto-generate) each round               exportable config
```

### User Flow

1. **Upload** sample documents (PDF, text, markdown)
2. **Add** test questions with expected answers (or let the system auto-generate them)
3. **Click** "Find Best Config"
4. **Agent runs** experiments on your data, using multi-armed bandit exploration to focus on promising configs
5. **Dashboard shows** the winning configuration with full metrics and comparison charts
6. **Export** the config as YAML/JSON to plug directly into your LangChain or LlamaIndex pipeline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  React / TypeScript Frontend                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Upload   │ │  Config  │ │ Dashboard│ │  Agent   │           │
│  │  Docs     │ │  Lab     │ │ Results  │ │Trajectory│           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
└───────┼─────────────┼────────────┼─────────────┼────────────────┘
        │             │            │             │
        ▼             ▼            ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Django REST API                                                 │
│  /api/documents/   /api/experiments/   /api/results/             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Celery  │ │PostgreSQL│ │  Redis   │
        │  Worker  │ │  (data)  │ │ (queue)  │
        └────┬─────┘ └──────────┘ └──────────┘
             │
     ┌───────┼───────────────┐
     ▼       ▼               ▼
┌────────┐ ┌─────────┐ ┌──────────┐
│ChromaDB│ │   LLM   │ │  RAGAS   │
│+ BM25  │ │(Mistral)│ │Evaluator │
└────────┘ └─────────┘ └──────────┘
```

### The Agentic Loop (Core Engine)

```python
while budget_remaining:
    past_results = db.get_all_experiments()
    next_config = agent.propose(past_results)   # MAB-style exploration
    results = run_pipeline(next_config)          # Execute on user's data
    metrics = compute_ragas_metrics(results)     # Evaluate
    agent.update_beliefs(next_config, metrics)   # Learn and iterate
```

The agent is not random grid search. It learns from each experiment and focuses on promising parameter regions, achieving near-optimal results in ~20% of the runs that grid search would require (AutoRAG-HP, EMNLP 2024).

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | TypeScript, React, Plotly.js / Recharts |
| **Backend** | Django, Django REST Framework |
| **Task Queue** | Celery + Redis (async experiment execution) |
| **Database** | PostgreSQL (experiments, results, configs) |
| **Vector Store** | ChromaDB + BM25 hybrid search |
| **LLM Generator** | Mistral-7B / Llama-3.1-8B via HuggingFace |
| **Embeddings** | all-MiniLM-L6-v2, BGE-M3 via sentence-transformers |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Evaluation** | RAGAS + DeepEval + custom IR metrics |
| **Agent Logic** | Custom Python controller with MAB strategy |
| **Deployment** | Docker Compose |

---

## Optimization Search Space

The agent explores these parameters to find the best combination for your data:

| Parameter | Options | Why It Matters |
|-----------|---------|---------------|
| **Chunk Size** | 128, 256, 512, 1024 tokens | Legal docs need big chunks; FAQs need small ones |
| **Chunk Overlap** | 10%, 15%, 20% | Prevents splitting key info across boundaries |
| **Top-k Depth** | 3, 5, 10, 20 documents | More docs = more noise; fewer = coverage gaps |
| **Reranking** | None, ColBERT, cross-encoder | Up to 67% retrieval failure reduction |
| **Hybrid Search** | Vector-only, BM25+vector | 15-30% precision improvement for keyword-heavy domains |
| **Prompting** | Zero-shot, few-shot, chain-of-thought | CoT improves across 9 reasoning datasets |
| **Embedding Model** | MiniLM, BGE-M3, E5 | Domain fine-tuning yields +10-30% gains |

Different data types need completely different settings. That's why you need to optimize on YOUR data.

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
| **Faithfulness** | Every claim supported by context? (most important metric) |
| **Groundedness** | No hallucinated claims in output? |
| **Answer Relevance** | Does the answer address the query? |
| **Answer Correctness** | F1-like factual overlap with ground truth |
| **RAGAS Score** | Composite across all dimensions |

### Statistical Rigor

All comparisons use paired bootstrap tests (10K+ samples), BCa confidence intervals, Cohen's d effect sizes, and Benjamini-Hochberg correction for multiple comparisons.

---

## Dashboard Pages

| Page | What Users See |
|------|---------------|
| **Upload** | Drag-and-drop documents, add test questions or auto-generate them |
| **Configuration Lab** | Manual mode: pick params from dropdowns, click Run |
| **Auto-Optimize** | Agent mode: click one button, agent finds best config |
| **Results** | Heatmaps, radar charts, KPI cards with metric deltas |
| **Comparison** | Side-by-side configs with significance indicators (*, **, ***) |
| **Agent Trajectory** | Visual timeline of what the agent tried and why |
| **Export** | Download best config as YAML/JSON for production use |

---

## Deep Learning Connection

This project is built for a **Deep & Generative Learning** course. The theoretical foundations:

| Course Topic | Connection |
|-------------|------------|
| **Transformers & Attention** | Retrieved context becomes external key-value memory for transformer self-attention |
| **Autoregressive Models** | Conditional generation P(y\|x,D) where D is dynamically determined at inference |
| **Latent Variable Models** | RAG as discrete latent variable model: P(y\|x) = Σ_z P(y\|x,z)·P(z\|x), paralleling VAE with ELBO optimization |
| **Evaluation** | RAGAS framework: 30+ metrics for generation quality assessment |
| **Emerging Trends** | Agentic AI for automated optimization; MAB-based hyperparameter search |

---

## Related Work

| System | What It Does | Gap We Fill |
|--------|-------------|------------|
| **AI Scientist v2** (Sakana AI) | Agentic tree search for automated research | We constrain to RAG-specific optimization with robust evaluation |
| **RAGAS** (EACL 2024) | RAG evaluation metrics | No optimization, no UI, no agent |
| **AutoRAG-HP** (EMNLP 2024) | MAB-based RAG tuning | No web interface, no document upload, no product |
| **RAGSmith** (2025) | NAS over 46K RAG configs | Research framework, not a usable platform |
| **LangSmith** | LLM tracing and debugging | No automated optimization |
| **Arize Phoenix** | LLM observability | No experiment runner |

**No existing tool offers: upload docs → agent finds best config → visual dashboard → export config.** That's the gap.

---

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
2. Es, S., et al. (2024). "RAGAs: Automated Evaluation of Retrieval Augmented Generation." *EACL 2024*.
3. Yamada, Y., et al. (2025). "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search." *arXiv:2504.08066*.
4. Asai, A., et al. (2024). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR 2024*.
5. NVIDIA (2025). "Finding the Best Chunking Strategy for Accurate AI Responses." *NVIDIA Technical Blog*.
6. Anthropic (2024). "Introducing Contextual Retrieval." *anthropic.com*.
7. Khattab, O., et al. (2024). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." *ICLR 2024*.
8. Liu, N., et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." *TACL*.

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

*Stanislav Li — Deep & Generative Learning — Spring 2026*

---
