# Evaluation-Driven Agentic Optimization of Transformer-Based Generative Models

## Overview
This project develops an **evaluation-driven agentic framework** for analyzing and improving **transformer-based generative models**, with a focus on **Retrieval-Augmented Generation (RAG)** systems. The project treats RAG as a controlled experimental setting for studying how conditioning, attention, and retrieval quality influence autoregressive generation.

Inspired by the scientific workflow modeled in Sakana AI’s *AI Scientist* (hypothesis → experiment → evaluation → decision → iteration), this project applies the same principles in a **constrained, reproducible, and measurable** manner without reproducing or extending the original system.

---

## Motivation
Transformer-based generative models are highly sensitive to how external context is retrieved, structured, and presented during generation. In practice, RAG systems are often tuned manually through heuristics and trial-and-error. This project replaces that process with an **agent-driven experimental loop** that systematically explores design choices and evaluates their impact using quantitative metrics.

The goal is not to **study and improve generative behavior** through principled experimentation.

---

## Project Goals
- Analyze the behavior of autoregressive transformer models under different retrieval and conditioning strategies  
- Automate controlled ablation studies over key RAG components  
- Evaluate generative quality using **retrieval-aware metrics** such as recall and groundedness  
- Demonstrate how agentic systems can improve generative pipelines through measurable feedback  

---

## Methodology

### Agentic Experimentation Loop
The system follows a closed-loop experimental process:

**Hypothesis → Experiment → Evaluation → Decision → Iteration**

1. **Hypothesis Generation**  
   The agent proposes testable hypotheses in the form of RAG configurations (e.g., chunk size, top-k retrieval, reranking, prompting strategy).

2. **Experiment Execution**  
   Each configuration is executed under a fixed computational budget using a consistent dataset and transformer-based generator.

3. **Evaluation**  
   Outputs are evaluated using RAG-specific quantitative metrics:
   - Retrieval performance (e.g., Recall@k)
   - Answer groundedness and faithfulness to retrieved context

4. **Decision & Iteration**  
   The agent compares results, selects the best-performing configuration, and proposes subsequent experiments based on observed improvements.

---

## Key Components
- **Transformer-based Generator**: Autoregressive language model conditioned on retrieved context  
- **Retriever & Index**: Vector-based document retrieval with configurable parameters  
- **Agent Controller**: Manages hypothesis generation, experiment scheduling, and decision logic  
- **Evaluation Harness**: Computes retrieval and generative quality metrics  
- **Experiment Logging**: Stores configurations, metrics, and summaries for reproducibility  

---

## Alignment with Course Topics
This project directly aligns with the core topics of *Deep and Generative Learning*:
- Transformers and attention mechanisms  
- Autoregressive generative models  
- Evaluation of generative systems  
- Real-world implementations of modern generative AI  
- Emerging trends in agentic and adaptive learning systems  

---

## Expected Deliverables
- Reproducible experimental pipeline for RAG evaluation  
- Quantitative comparison of multiple RAG configurations  
- Agent-driven optimization results with logged metrics  
- Final poster and presentation summarizing findings  
- Optional web interface or academic-style report  

---
