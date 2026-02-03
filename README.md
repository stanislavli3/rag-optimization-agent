# Evaluation-Driven Agentic Optimization of Transformer-Based Generative Models

## Overview

This project develops an evaluation-driven agentic framework for analyzing and improving transformer-based generative models, with a specific focus on Retrieval-Augmented Generation (RAG) systems. The project treats RAG as a controlled experimental setting for studying how conditioning, attention, and retrieval quality influence autoregressive generation.

Inspired by the scientific workflow modeled in Sakana AI's AI Scientist (hypothesis → experiment → evaluation → decision → iteration), this project applies the same principles in a constrained, reproducible, and measurable manner without reproducing or extending the original system.

## Motivation

Transformer-based generative models are highly sensitive to how external context is retrieved, structured, and presented during generation. In practice, RAG systems are often tuned manually through heuristics and trial-and-error, which is time-consuming, non-systematic, and difficult to reproduce.

This project replaces that manual process with an agent-driven experimental loop that systematically explores design choices and evaluates their impact using quantitative metrics. The goal is to study and improve generative behavior through principled experimentation, treating RAG optimization as a scientific process rather than an engineering heuristic.

## Project Goals

- **Analyze** the behavior of autoregressive transformer models under different retrieval and conditioning strategies
- **Automate** controlled ablation studies over key RAG components (chunk size, retrieval depth, reranking, prompting)
- **Evaluate** generative quality using retrieval-aware metrics such as recall, precision, and groundedness
- **Demonstrate** how agentic systems can improve generative pipelines through measurable, iterative feedback

## Methodology

### Agentic Experimentation Loop

The system follows a closed-loop experimental process inspired by the scientific method:

**Hypothesis → Experiment → Evaluation → Decision → Iteration**

#### 1. Hypothesis Generation
The agent proposes testable hypotheses in the form of RAG configurations, such as:
- Chunk size variations (128, 256, 512 tokens)
- Top-k retrieval parameters (1, 3, 5, 10 documents)
- Reranking strategies (semantic similarity, cross-encoder)
- Prompting techniques (few-shot, chain-of-thought, structured templates)

#### 2. Experiment Execution
Each configuration is executed under a fixed computational budget using:
- A consistent evaluation dataset with ground-truth answers
- A standardized transformer-based generator
- Controlled retrieval and indexing parameters

#### 3. Evaluation
Outputs are assessed using RAG-specific quantitative metrics:
- **Retrieval performance**: Recall@k, MRR (Mean Reciprocal Rank)
- **Generative quality**: Answer groundedness, factual consistency, faithfulness to retrieved context
- **Efficiency**: Latency, token usage, computational cost

#### 4. Decision & Iteration
The agent:
- Compares experimental results against baseline and previous configurations
- Selects the best-performing configuration based on composite scores
- Proposes subsequent experiments informed by observed patterns and performance gaps

## Key Components

| Component | Description |
|-----------|-------------|
| **Transformer-based Generator** | Autoregressive language model conditioned on retrieved context (e.g., GPT-based or open-source LLM) |
| **Retriever & Index** | Vector-based document retrieval with configurable parameters (embedding model, similarity metric, top-k) |
| **Agent Controller** | Manages hypothesis generation, experiment scheduling, and decision-making logic |
| **Evaluation Harness** | Computes retrieval and generative quality metrics with statistical significance testing |
| **Experiment Logging** | Stores configurations, metrics, and summaries for reproducibility and analysis |

## Alignment with Course Topics

This project directly addresses core topics in Deep and Generative Learning:

- **Transformers and attention mechanisms**: Analysis of how attention over retrieved context affects generation
- **Autoregressive generative models**: Study of conditional generation quality under varying retrieval strategies
- **Evaluation of generative systems**: Development and application of quantitative metrics for RAG
- **Real-world implementations**: Practical application of modern generative AI in production-like scenarios
- **Emerging trends**: Exploration of agentic and adaptive learning systems for automated optimization

## Expected Deliverables

1. **Reproducible experimental pipeline** for RAG evaluation with documented code and configuration files
2. **Quantitative comparison** of multiple RAG configurations with statistical analysis
3. **Agent-driven optimization results** with detailed logs showing the experimental trajectory
4. **Final poster and presentation** summarizing methodology, findings, and insights
5. **Website**: Interactive web interface for exploring results or academic-style technical report

## License

MIT License - See [LICENSE](LICENSE) for details
