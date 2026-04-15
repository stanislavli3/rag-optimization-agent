# RAG Optimizer — Full Implementation Plan (Claude Code)

## Everything we discussed, incorporated

**Two novel contributions this project makes:**

1. **Generative test data pipeline** — Knowledge graph extraction → Evol-Instruct question synthesis → controllable type distribution (simple / multi-hop / reasoning) with GRADE-style 2D difficulty matrix. This is conditional generation: P(question | context, type, difficulty).

2. **BFTS optimization loop adapted from Sakana AI Scientist v2** — Progressive 4-stage experimentation (preliminary → baseline → exploration → ablation) with best-first tree search, an experiment manager agent, and debug-or-abandon logic. Each tree node is a RAG config + eval score. The agent expands the highest-scoring leaf, prunes failing branches, and runs ablations to confirm component contributions.

---

## Project structure

```
rag-optimizer/
├── config.py
├── run.py                     # CLI entry point
├── app.py                     # Streamlit UI (Phase 7)
│
├── data/
│   ├── sample_docs/
│   └── testsets/
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── loader.py          # PDF/MD/TXT → LangChain Documents
│   │   └── chunker.py         # Configurable chunking
│   │
│   ├── testgen/                    # ← GENERATIVE AI CORE
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py      # Entity/relation extraction from docs
│   │   ├── seed_generator.py       # Initial Q-A pairs from KG facts
│   │   ├── evol_instruct.py        # Evolve seeds → harder question types
│   │   ├── difficulty_matrix.py    # GRADE-style 2D difficulty scoring
│   │   ├── groundtruth.py          # Answer generation + groundedness filter
│   │   └── pipeline.py            # Orchestrates full testgen flow
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── indexer.py         # ChromaDB + BM25
│   │   ├── retriever.py       # Vector / hybrid / reranked
│   │   ├── generator.py       # LLM answer generation
│   │   └── runner.py          # run_pipeline(config, docs, queries)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ragas_eval.py      # RAGAS metrics
│   │   ├── ir_metrics.py      # MRR, NDCG, P@K
│   │   └── statistics.py      # Bootstrap tests, Cohen's d
│   │
│   └── optimizer/                  # ← SAKANA AI SCIENTIST v2 METHOD
│       ├── __init__.py
│       ├── search_space.py         # Config space + mutations
│       ├── tree_node.py            # SearchNode dataclass
│       ├── experiment_manager.py   # Agent: select, expand, prune, stage transitions
│       ├── ablation.py             # Stage 4 ablation runner
│       └── bfts_loop.py            # Main BFTS orchestrator with 4 stages
│
├── notebooks/
│   ├── 01_ingest_test.ipynb
│   ├── 02_testgen_demo.ipynb       # Show full KG → evolve → difficulty pipeline
│   ├── 03_pipeline_test.ipynb
│   ├── 04_eval_test.ipynb
│   ├── 05_bfts_demo.ipynb          # Show tree growing across 4 stages
│   └── 06_end_to_end.ipynb
│
└── requirements.txt
```

---

## Phase 0 — Project setup

**Claude Code prompt:**
```
Create the rag-optimizer project skeleton. Set up the full directory
structure shown above with empty __init__.py files. Create:

1. requirements.txt with:
   langchain>=0.3.0, langchain-community>=0.3.0, langchain-huggingface>=0.1.0
   sentence-transformers>=3.0.0, chromadb>=0.5.0, rank-bm25>=0.2.2
   transformers>=4.40.0, torch>=2.0.0, huggingface-hub>=0.23.0
   ragas>=0.2.0, datasets>=2.20.0
   pypdf>=4.0.0
   scikit-learn>=1.5.0, scipy>=1.13.0, numpy>=1.26.0
   streamlit>=1.36.0, plotly>=5.22.0, pandas>=2.2.0, matplotlib>=3.9.0
   networkx>=3.3.0  # for knowledge graph
   spacy>=3.7.0     # for entity extraction

2. config.py with a Config dataclass containing all paths, model names,
   search space definition, and testgen parameters. Include the full
   RAG_SEARCH_SPACE dict:
   chunk_size: [256, 512, 1024]
   chunk_overlap: [0.10, 0.20]
   top_k: [3, 5, 10]
   reranker: [None, "cross-encoder"]
   search_mode: ["vector", "hybrid"]
   prompt_style: ["zero-shot", "few-shot", "chain-of-thought"]
   embedding_model: ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]

   Also include BFTSConfig dataclass:
   num_seeds: 3
   max_steps: 20
   max_debug_depth: 3
   debug_prob: 0.5
   convergence_window: 3
   convergence_eps: 0.01
   stage_budgets: {PRELIMINARY: 3, BASELINE: 4, EXPLORATION: 10, ABLATION: 3}

3. run.py that prints "RAG Optimizer ready" and the config.

Do NOT install packages yet.
```

**Validate:** `python run.py` prints config. All directories exist.

---

## Phase 1 — Document ingestion & chunking

**Claude Code prompt:**
```
Implement src/ingest/loader.py and src/ingest/chunker.py.

loader.py:
- load_documents(dir_path: str) -> list[Document]
- Support .pdf (PyPDFLoader), .md (UnstructuredMarkdownLoader), .txt (TextLoader)
- Add metadata: source_file, page_number, file_type
- Skip unreadable files with warning, don't crash

chunker.py:
- chunk_documents(docs: list[Document], chunk_size: int, overlap_ratio: float) -> list[Document]
- Use RecursiveCharacterTextSplitter from LangChain
- overlap = int(chunk_size * overlap_ratio)
- Add metadata: chunk_index, chunk_size_used, parent_doc
- Return chunked documents

Write tests/test_chunker.py:
- Create 3 fake documents of ~2000 chars
- Assert chunk counts decrease as chunk_size increases (256 > 512 > 1024)
- Assert every chunk has required metadata fields
```

**Validate:** Load 3-5 sample PDFs, chunk at 512, print counts.

---

## Phase 2 — Synthetic test generation (GENERATIVE AI CORE)

This is the most important phase. It has 5 sub-steps that mirror the research we discussed: RAGAS knowledge graph approach, Evol-Instruct evolution from the RAGAS paper, and GRADE difficulty matrix.

### Phase 2a — Knowledge graph extraction

**Claude Code prompt:**
```
Implement src/testgen/knowledge_graph.py.

This module extracts a knowledge graph from chunked documents — entities,
factual statements, and relationships — which will seed the question
generation. This mirrors RAGAS's first phase: "knowledge graph construction
with transformation enrichment" (RAGAS docs, 2024).

Classes and functions:

@dataclass
class KGNode:
    id: str
    text: str              # the entity or concept name
    node_type: str         # "entity", "concept", "fact"
    source_chunk_id: str   # which chunk this came from
    source_doc: str        # original document filename

@dataclass
class KGEdge:
    source_id: str
    target_id: str
    relation: str          # e.g. "causes", "is_part_of", "contrasts_with"
    evidence: str          # the sentence that supports this relation

@dataclass
class KnowledgeGraph:
    nodes: list[KGNode]
    edges: list[KGEdge]

    def get_facts(self) -> list[dict]:
        """Return all factual statements as {subject, relation, object, evidence}."""

    def get_connected_facts(self, n_hops: int = 2) -> list[tuple[dict, dict]]:
        """Return pairs of facts connected within n_hops — for multi-hop questions."""

    def get_clusters(self) -> list[list[KGNode]]:
        """Group semantically related nodes — for multi-context questions."""

def extract_knowledge_graph(
    chunks: list[Document],
    llm,
) -> KnowledgeGraph:
    """
    For each chunk:
    1. Prompt the LLM to extract entities and factual statements
    2. Prompt the LLM to identify relationships between entities
    3. Build a networkx graph, detect cross-chunk connections
       by matching entity names across chunks
    Return the KnowledgeGraph.
    """

LLM prompts to use:

EXTRACT_ENTITIES_AND_FACTS = '''Given this text passage, extract:
1. Key entities (people, organizations, concepts, technical terms)
2. Factual statements — each should be a self-contained claim that
   could be verified from this passage alone.
3. Relationships between entities (A relates-to B).

Passage:
{chunk_text}

Respond in this exact format:
ENTITIES:
- entity_name | entity_type
...
FACTS:
- factual statement
...
RELATIONS:
- entity_A | relation | entity_B | supporting sentence
...'''

Use networkx to build the graph. Find cross-chunk connections by
matching entity names (case-insensitive) across different chunks.
```

**Validate:** Extract KG from 5 chunks. Print node count, edge count, and 3 sample facts with their evidence sentences.

### Phase 2b — Seed question generation

**Claude Code prompt:**
```
Implement src/testgen/seed_generator.py.

This generates initial "seed" questions from the knowledge graph facts.
Each seed is a simple, direct question that can be answered from one chunk.
These will be EVOLVED into harder types in the next step.

def generate_seeds(
    kg: KnowledgeGraph,
    llm,
    num_seeds: int = 30,
) -> list[dict]:
    """
    For each fact in the KG:
    1. Generate a natural question where this fact is the answer
    2. Record the source chunk as ground_truth_context
    3. Generate the ground_truth_answer using the LLM conditioned
       on the context (NOT from memory)

    Return list of:
    {
        "question": str,
        "ground_truth_answer": str,
        "ground_truth_context": str,      # the chunk text
        "source_fact": str,               # the KG fact used
        "source_chunk_id": str,
        "question_type": "simple",        # all seeds start as simple
        "evolution": "seed",              # tracking the evolution chain
    }
    """

LLM prompts:

GENERATE_SEED_QUESTION = '''Given this factual statement extracted from
a document, write a natural question that a user would ask, where
knowing this fact is necessary to answer correctly.

The question should:
- Sound like something a real user would type
- NOT contain the answer
- Be answerable ONLY from the given context

Fact: {fact}
Source context: {context}

Question:'''

GENERATE_GROUND_TRUTH = '''Answer this question using ONLY the provided
context. Do not use any outside knowledge. If the context doesn't
contain enough information, say "INSUFFICIENT".

Context: {context}
Question: {question}

Answer:'''

Filter out any seeds where the ground truth answer is "INSUFFICIENT".
Sample num_seeds from the valid seeds (prioritize diversity of source chunks).
```

**Validate:** Generate 20 seeds from the KG. Print 5 examples showing question, answer, source fact, and context snippet.

### Phase 2c — Evol-Instruct question evolution

**Claude Code prompt:**
```
Implement src/testgen/evol_instruct.py.

This is the core generative novelty. Following the Evol-Instruct paradigm
adapted by RAGAS (Es et al., 2024), we evolve simple seed questions into
harder variants. RAGAS adapts three evolution types: simple (keep as-is),
multi_context (combine two contexts), and reasoning (require inference).

We add a fourth from GRADE (2025): conditional (add a constraint that
changes the answer depending on context).

from enum import Enum

class QuestionType(Enum):
    SIMPLE = "simple"
    MULTI_CONTEXT = "multi_context"
    REASONING = "reasoning"
    CONDITIONAL = "conditional"

def evolve_question(
    seed: dict,
    target_type: QuestionType,
    kg: KnowledgeGraph,
    llm,
) -> dict:
    """
    Evolve a seed question into a harder variant based on target_type.
    Returns the evolved question dict with updated fields.
    """

def evolve_to_multi_context(seed: dict, kg: KnowledgeGraph, llm) -> dict:
    """
    Multi-context evolution:
    1. Find a DIFFERENT chunk that is connected to the seed's chunk
       in the knowledge graph (via shared entities or KG edges)
    2. Prompt the LLM to generate a question that requires info
       from BOTH chunks to answer
    3. The ground_truth_context now contains BOTH chunks
    4. Generate a new ground_truth_answer from both contexts
    """

def evolve_to_reasoning(seed: dict, llm) -> dict:
    """
    Reasoning evolution:
    1. Take the seed fact and its context
    2. Prompt the LLM to generate a question that requires
       INFERENCE beyond what is directly stated
       (e.g., "What would happen if...", "Why does...",
        "What is the implication of...")
    3. The answer requires logical deduction from the context
    """

def evolve_to_conditional(seed: dict, llm) -> dict:
    """
    Conditional evolution (from GRADE):
    1. Take the seed question
    2. Add a constraint or condition that narrows or changes the answer
       (e.g., "Under what circumstances..." or "Assuming X, what...")
    3. The answer depends on correctly applying the condition to the context
    """

def evolve_batch(
    seeds: list[dict],
    kg: KnowledgeGraph,
    llm,
    distribution: dict = None,
) -> list[dict]:
    """
    Evolve all seeds according to the target distribution.

    Default distribution:
    {"simple": 0.3, "multi_context": 0.3, "reasoning": 0.25, "conditional": 0.15}

    For each seed:
    1. Sample a target type from the distribution
    2. If "simple" — keep the seed as-is
    3. Otherwise — call the appropriate evolve function
    4. If evolution fails (LLM returns garbage), keep as simple + log warning
    """

LLM prompts:

EVOLVE_MULTI_CONTEXT = '''Given these two related passages from a document,
generate a question that can ONLY be answered by combining information
from BOTH passages. The question should be natural and specific.

Passage 1: {context_1}
Key fact from passage 1: {fact_1}

Passage 2: {context_2}
Key fact from passage 2: {fact_2}

Relationship between passages: {relation}

Multi-hop question (must require both passages):'''

EVOLVE_REASONING = '''Given this question and its context, create a HARDER
version that requires logical reasoning or inference — not just finding
a fact in the text.

Original question: {question}
Context: {context}
Original answer: {answer}

Techniques to make it harder:
- Ask "why" or "how" instead of "what"
- Ask about implications or consequences
- Ask for comparison or contrast
- Ask what would change if a condition were different

Reasoning question:'''

EVOLVE_CONDITIONAL = '''Given this question, add a specific condition or
constraint that changes what the correct answer is. The condition should
be realistic and answerable from the context.

Original question: {question}
Context: {context}

Add a condition like "Under what circumstances...", "If X were different...",
"In the case where...", "Given that..."

Conditional question:'''
```

**Validate:** Start with 20 seeds, evolve with default distribution. Print the count per type and 2 examples of each type showing the evolution chain (seed → evolved).

### Phase 2d — GRADE difficulty matrix

**Claude Code prompt:**
```
Implement src/testgen/difficulty_matrix.py.

This implements the GRADE-style 2D difficulty scoring from
"GRADE: Generating multi-hop QA and fine-grained Difficulty matrix
for RAG Evaluation" (EMNLP 2025 Findings).

Two orthogonal difficulty dimensions:
1. Reasoning depth — how many inference steps needed (1-hop, 2-hop, 3-hop)
2. Semantic distance — how far the query terms are from the evidence terms
   (measured by embedding cosine distance between question and ground_truth_context)

@dataclass
class DifficultyScore:
    reasoning_depth: int     # 1, 2, or 3
    semantic_distance: float # 0.0 (easy) to 1.0 (hard)
    overall: str             # "easy", "medium", "hard"

def score_reasoning_depth(question_data: dict) -> int:
    """
    Estimate reasoning depth:
    - simple questions → 1
    - multi_context → 2 (requires connecting 2 sources)
    - reasoning → 2-3 (requires inference steps)
    - conditional → 2-3 (requires applying constraint)
    For multi_context and reasoning, count the number of distinct
    facts from the KG needed to answer.
    """

def score_semantic_distance(
    question: str,
    context: str,
    embedding_model,
) -> float:
    """
    Compute cosine distance between question embedding and context embedding.
    High distance = question uses very different words than the context
    (harder for retrieval). Low distance = high lexical overlap (easier).
    Normalize to 0-1 range.
    """

def compute_difficulty_matrix(
    testset: list[dict],
    embedding_model,
) -> pd.DataFrame:
    """
    For each question in the testset:
    1. Score reasoning depth
    2. Score semantic distance
    3. Classify into overall difficulty: easy/medium/hard
       - easy: depth=1, distance < 0.3
       - medium: depth=2 OR distance 0.3-0.6
       - hard: depth>=3 OR distance > 0.6
    4. Return DataFrame with columns added: reasoning_depth,
       semantic_distance, difficulty

    This matrix is used for:
    - Showing users what types of questions their RAG struggles with
    - Stratified evaluation (score per difficulty bucket)
    - Identifying if retriever fails on high-distance queries
      vs generator fails on high-depth queries
    """

def difficulty_breakdown_report(df: pd.DataFrame) -> dict:
    """
    Return a report dict:
    {
        "total": int,
        "by_type": {"simple": n, "multi_context": n, ...},
        "by_difficulty": {"easy": n, "medium": n, "hard": n},
        "matrix": {
            (depth=1, dist="low"): n,
            (depth=1, dist="mid"): n,
            ...
        }
    }
    """
```

**Validate:** Score the evolved testset. Print the 2D matrix as a heatmap (reasoning_depth × semantic_distance buckets). Verify distribution across difficulty levels.

### Phase 2e — Testgen orchestrator + groundedness filter

**Claude Code prompt:**
```
Implement src/testgen/groundtruth.py and src/testgen/pipeline.py.

groundtruth.py:
def verify_groundedness(
    question: str,
    answer: str,
    context: str,
    llm,
) -> tuple[bool, float]:
    """
    Use the LLM as a judge to verify that the generated answer
    is fully grounded in the provided context.
    Returns (is_grounded: bool, confidence: float 0-1).

    Prompt the LLM:
    "Given this context and answer, is every claim in the answer
     supported by the context? Answer YES or NO with confidence 0-1."

    Filter out any Q-A pairs where is_grounded=False or confidence < 0.7.
    This is the quality gate from RAGEval (ACL 2025).
    """

pipeline.py — the full orchestrator:
class TestGenPipeline:
    """
    Orchestrates the full testgen flow:
    Documents → KG → Seeds → Evolution → Difficulty scoring → Filtering

    This is the generative AI centerpiece of the project.
    """

    def __init__(self, llm, embedding_model, config):
        self.llm = llm
        self.embedding_model = embedding_model
        self.config = config

    def generate(self, documents: list[Document]) -> pd.DataFrame:
        """
        Full pipeline:
        1. extract_knowledge_graph(chunks, self.llm)
           → KnowledgeGraph with entities, facts, relations
        2. generate_seeds(kg, self.llm, num_seeds=config.testset_size * 2)
           → oversample because some will be filtered out
        3. evolve_batch(seeds, kg, self.llm, config.question_distribution)
           → evolved questions with types assigned
        4. For each evolved question, verify_groundedness()
           → filter out ungrounded Q-A pairs
        5. compute_difficulty_matrix(filtered, self.embedding_model)
           → add difficulty scores
        6. Sample to target testset_size if we have excess
        7. Save to CSV at config.testset_dir/testset_{timestamp}.csv
        8. Return the DataFrame

        Log statistics at each step:
        - KG: {n} nodes, {m} edges, {k} cross-chunk connections
        - Seeds: {n} generated, {m} passed groundedness filter
        - Evolution: {n} simple, {m} multi_context, {p} reasoning, {q} conditional
        - Difficulty: {n} easy, {m} medium, {p} hard
        """

    def generate_with_progress(self, documents):
        """Generator version that yields progress updates for Streamlit."""
        yield {"step": "knowledge_graph", "status": "running"}
        kg = extract_knowledge_graph(...)
        yield {"step": "knowledge_graph", "status": "done",
               "stats": {"nodes": len(kg.nodes), "edges": len(kg.edges)}}
        # ... etc for each step
```

**Validate:** Full pipeline on 5 sample docs → CSV with 20 rows. Each row has: question, ground_truth_answer, ground_truth_context, question_type, evolution, reasoning_depth, semantic_distance, difficulty.

---

## Phase 3 — Parameterized RAG pipeline

**Claude Code prompt:**
```
Implement the four files in src/pipeline/.

indexer.py:
- build_index(chunks, embedding_model_name, persist_dir) -> dict
- Creates a ChromaDB collection with embeddings
- Also builds a BM25 index from the same chunks using rank_bm25
- Returns {"chroma": collection, "bm25": bm25_index, "chunks": chunks,
           "cache_key": f"{chunk_size}_{overlap}_{embedding_model}"}
- Support cache_key check: if the index already exists with same key, reuse it

retriever.py:
- retrieve(query, index, top_k, search_mode, reranker) -> list[tuple[Document, float]]
- search_mode="vector": ChromaDB similarity search, return top_k
- search_mode="hybrid": query both ChromaDB and BM25, combine with
  reciprocal rank fusion (RRF): score = Σ 1/(k + rank_i) for each system
  Take union of results, sort by RRF score, return top_k
- reranker=None: return as-is
- reranker="cross-encoder": load cross-encoder/ms-marco-MiniLM-L-6-v2,
  re-score all candidates with the cross-encoder, re-sort, return top_k
- Return list of (document, score) tuples

generator.py:
- generate_answer(query, contexts, llm, prompt_style) -> str
- prompt_style="zero-shot":
    "Answer the question based on the context.\n\nContext: {ctx}\n\nQuestion: {q}\n\nAnswer:"
- prompt_style="few-shot":
    Include 2 hardcoded examples before the actual query
- prompt_style="chain-of-thought":
    "Answer step by step. First identify relevant info in the context,
     then reason through to the answer.\n\nContext: {ctx}\n\nQuestion: {q}\n\nThinking:"
- Return the generated answer string

runner.py:
- run_pipeline(config: dict, documents: list[Document], queries: list[dict]) -> list[dict]
- config is a flat dict like:
    {"chunk_size": 512, "chunk_overlap": 0.15, "top_k": 5,
     "search_mode": "hybrid", "reranker": "cross-encoder",
     "prompt_style": "chain-of-thought", "embedding_model": "all-MiniLM-L6-v2"}
- Steps:
    1. Chunk documents with config chunk_size and chunk_overlap
    2. Build/reuse index (check cache_key)
    3. For each query in queries:
       a. Retrieve with config top_k, search_mode, reranker
       b. Generate answer with config prompt_style
       c. Collect: {query, answer, retrieved_contexts, ground_truth}
    4. Return all results
- INDEX CACHING IS CRITICAL:
    Cache key = f"{chunk_size}_{chunk_overlap}_{embedding_model}"
    If the key matches the existing index, skip re-indexing.
    This makes the optimizer 5-10x faster since most experiments
    only change retrieval/generation params, not chunking.
```

**Validate:** Run one config against the testset from Phase 2. Print 3 Q-A pairs with retrieved contexts.

---

## Phase 4 — Evaluation framework

**Claude Code prompt:**
```
Implement the three files in src/evaluation/.

ragas_eval.py:
- evaluate_ragas(results: list[dict], llm, embeddings) -> dict
- Convert results to RAGAS Dataset format:
    question, answer, contexts, ground_truth
- Compute metrics: faithfulness, answer_relevancy, context_precision,
  context_recall, answer_correctness
- Compute composite ragas_score = harmonic_mean of the 4 core metrics
- Also return per-question scores for stratified analysis
- Return:
    {"faithfulness": 0.82, "answer_relevancy": 0.75, ...,
     "ragas_score": 0.74,
     "per_question": [{"question": ..., "scores": {...}}, ...]}

ir_metrics.py:
- evaluate_ir(retrieved: list[list[str]], relevant: list[list[str]], k: int) -> dict
- MRR: mean reciprocal rank of first relevant document
- NDCG@k: using sklearn.metrics.ndcg_score
- Precision@k and Recall@k
- Return: {"mrr": 0.65, "ndcg@5": 0.58, "precision@5": 0.42, "recall@5": 0.71}

statistics.py:
- paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000) -> dict
  Returns {"p_value": float, "ci_lower": float, "ci_upper": float,
           "significant": bool}  # significant if p < 0.05
- cohens_d(scores_a, scores_b) -> float
- compare_experiments(results_a, results_b) -> dict
  Full comparison: significance, effect size, per-metric deltas

Also add stratified evaluation:
- evaluate_by_difficulty(results, testset_df) -> dict
  Group results by difficulty level (easy/medium/hard) from the
  difficulty_matrix, compute metrics for each group separately.
  This reveals WHERE the pipeline fails: retrieval on hard queries?
  Generation on reasoning questions?
  Return: {"easy": {metrics}, "medium": {metrics}, "hard": {metrics}}

- evaluate_by_question_type(results, testset_df) -> dict
  Same but grouped by question_type (simple/multi_context/reasoning/conditional)
```

**Validate:** Evaluate one pipeline run. Print the metrics table and the stratified breakdown by difficulty.

---

## Phase 5 — BFTS optimization loop (Sakana AI Scientist v2 method)

This is where we adapt the AI Scientist v2's progressive staged tree search. The implementation has 4 sub-steps matching the 4 modules.

### Phase 5a — Tree node + search space

**Claude Code prompt:**
```
Implement src/optimizer/tree_node.py and src/optimizer/search_space.py.

tree_node.py:
from enum import IntEnum
from dataclasses import dataclass, field

class Stage(IntEnum):
    PRELIMINARY = 1   # Can a config even run end-to-end?
    BASELINE = 2      # Tune basic params to stable baseline
    EXPLORATION = 3   # Best-first search through config space
    ABLATION = 4      # Confirm each component's contribution

class NodeStatus(IntEnum):
    PENDING = 0
    RUNNING = 1
    SUCCESS = 2
    FAILED = 3
    PRUNED = 4

@dataclass
class SearchNode:
    """One node in the BFTS tree. Each node = a RAG config + its eval results."""
    id: str                              # uuid hex[:12]
    parent_id: str | None = None
    stage: Stage = Stage.PRELIMINARY
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    score: float = 0.0                   # composite RAGAS score
    status: NodeStatus = NodeStatus.PENDING
    debug_attempts: int = 0
    error_msg: str = ""
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0

search_space.py:
- RAG_SEARCH_SPACE: dict of param → list of values (from config.py)
- get_default_config() -> dict: sensible default baseline
- sample_random_config() -> dict: random valid config
- mutate_config(config: dict) -> dict:
    Pick ONE random parameter, change it to a different value.
    Return the new config.
- get_neighbors(config: dict) -> list[dict]:
    For each param, generate all configs that differ by one value.
    Return the list (for more systematic exploration).
- generate_ablation_configs(best_config: dict) -> list[dict]:
    For each param where best_config differs from default,
    create a version reverting that ONE param to default.
    Return list of {"ablated_param": str, "original_value": Any,
                    "default_value": Any, "config": dict}
```

### Phase 5b — Experiment manager agent

**Claude Code prompt:**
```
Implement src/optimizer/experiment_manager.py.

This is the brain of the optimization — adapted from AI Scientist v2's
"Experiment Progress Manager" that decides which node to expand next,
whether to debug or abandon failing nodes, and when to advance stages.

class ExperimentManager:
    """
    The agent that manages the BFTS tree.
    Inspired by AI Scientist v2 (Yamada et al., 2025):
    'the Experiment Manager agent decides which node to expand next,
     whether to debug or abandon a failing path.'
    """

    def __init__(self, bfts_config: BFTSConfig):
        self.cfg = bfts_config
        self.nodes: dict[str, SearchNode] = {}
        self.current_stage: Stage = Stage.PRELIMINARY
        self._stage_node_counts: dict[Stage, int] = {s: 0 for s in Stage}
        self._best_score_history: list[float] = []

    def seed_roots(self) -> list[SearchNode]:
        """
        Stage 1: Create cfg.num_seeds independent root nodes.
        First root = default config. Others each vary one random param.
        This mirrors AI Scientist v2's Stage 1 seed phase.
        """

    def select_next(self) -> SearchNode | None:
        """
        BEST-FIRST SELECTION — the core BFTS step.
        1. Check if global step budget exhausted → return None
        2. Check/trigger stage transitions
        3. Find the highest-scoring SUCCESS leaf node
        4. Propose a child by mutating one param of that leaf's config
        5. Register the child, return it

        'The tree-search approach allows the system to allocate resources
         efficiently by expanding promising branches while pruning less
         successful paths.' (Yamada et al., 2025)
        """

    def report_success(self, node_id: str, metrics: dict, score: float):
        """Called after successful pipeline run + evaluation."""

    def report_failure(self, node_id: str, error_msg: str) -> str:
        """
        Called when a pipeline run fails. Returns "debug" or "abandon".

        Debug-or-abandon logic (from AI Scientist v2 bfts_config.yaml):
        - If debug_attempts < max_debug_depth AND random() < debug_prob:
            → tweak one param (mutate_config), return "debug"
        - Otherwise:
            → mark as PRUNED, return "abandon"
        """

    def get_best_node(self) -> SearchNode | None:
        """Return globally best SUCCESS node."""

    def get_trajectory(self) -> list[dict]:
        """All nodes ordered by (stage, depth) for visualization."""

    # --- Stage transition logic ---

    def _maybe_advance_stage(self, force: bool = False):
        """
        Stage transition criteria (adapted from AI Scientist v2 §3.2):

        PRELIMINARY → BASELINE:
            When ANY seed node succeeds.
            'Stage 1 concludes when a basic working prototype is
             successfully executed.'

        BASELINE → EXPLORATION:
            When score converges: the best score hasn't improved by
            more than convergence_eps over the last convergence_window
            nodes. 'Stage 2 ends when experiments stabilize, as
            indicated by convergence.'

        EXPLORATION → ABLATION:
            When stage budget exhausted.
            'Stages 3 and 4 conclude when the allocated computational
             budget is exhausted.'

        Or force=True when a stage budget runs out.
        """

    def _has_converged(self) -> bool:
        """
        Check last convergence_window scores.
        If max - min < convergence_eps → converged.
        """

    def _best_leaf(self) -> SearchNode | None:
        """Find highest-scoring SUCCESS node with no children (a leaf)."""

    def _propose_child_config(self, parent: SearchNode) -> dict:
        """Mutate one param of parent's config."""
```

### Phase 5c — Ablation runner

**Claude Code prompt:**
```
Implement src/optimizer/ablation.py.

class AblationRunner:
    """
    Stage 4: Systematically disable each component of the best config
    to measure its marginal contribution.

    'Each stage builds upon the findings from previous stages, creating
     a progressive research narrative.' (Yamada et al., 2025)
    """

    @staticmethod
    def generate_ablation_configs(best_config: dict) -> list[dict]:
        """
        For each param where best_config != default_config:
        Create a version that reverts ONLY that param to default.
        Return list of:
        {
            "ablated_param": "reranker",
            "original_value": "cross-encoder",
            "default_value": None,
            "config": {... with reranker=None, rest unchanged ...}
        }
        """

    @staticmethod
    def compute_ablation_report(
        best_score: float,
        ablation_results: list[dict],  # each has "ablated_param" + "score"
    ) -> list[dict]:
        """
        For each ablation:
        {
            "param": "reranker",
            "with_value": "cross-encoder",
            "without_value": None,
            "score_with": 0.78,
            "score_without": 0.61,
            "delta": 0.17,
            "contribution_pct": 21.8,   # delta / best_score * 100
        }
        Sort by delta descending — most important component first.
        """
```

### Phase 5d — Main BFTS loop

**Claude Code prompt:**
```
Implement src/optimizer/bfts_loop.py.

This is the main orchestrator that ties everything together — equivalent
to AI Scientist v2's perform_experiments_bfts() function.

class BFTSLoop:
    """
    The full BFTS optimization loop with progressive 4-stage experimentation.

    Adapted from Sakana AI Scientist v2's agentic tree search:
    - Stage 1: Seed num_seeds root configs, prove feasibility
    - Stage 2: Expand from best seed, tune basic params until convergence
    - Stage 3: Best-first tree expansion — always expand highest-scoring leaf
    - Stage 4: Ablate best config to confirm component contributions

    Each iteration: propose config → run pipeline → evaluate → update tree
    Failed nodes: debug (tweak + retry) or prune (abandon branch)
    """

    def __init__(
        self,
        documents: list,
        testset: pd.DataFrame,
        run_fn,           # run_pipeline function
        eval_fn,          # evaluate function
        bfts_config = None,
    ):
        self.documents = documents
        self.testset = testset
        self.run_fn = run_fn
        self.eval_fn = eval_fn
        self.cfg = bfts_config or BFTSConfig()
        self.manager = ExperimentManager(self.cfg)
        self._ablations_done = False

    def run(self) -> dict:
        """
        Blocking execution. Returns full results.

        Algorithm:
        1. seeds = manager.seed_roots()
        2. For each seed: execute_node(seed)
        3. Loop:
             node = manager.select_next()
             if None: break
             execute_node(node)
             if stage == ABLATION and not done: run_ablations()
        4. Return {
             best_config, best_score, best_metrics,
             trajectory, ablation_report,
             tree_summary: {total_nodes, successful, pruned, per_stage_counts},
             stage_transitions: [{from, to, trigger, at_node}],
           }
        """

    def run_iter(self):
        """
        Generator version for Streamlit real-time updates.
        Yields after each node:
        {
            "event": "node_complete" | "stage_transition" |
                     "ablation_complete" | "search_complete",
            "node": SearchNode,
            "current_stage": str,
            "best_score": float,
            "progress": float,  # nodes_explored / max_steps
            ...
        }
        """

    def _execute_node(self, node: SearchNode):
        """
        Run pipeline + evaluate for one node.
        On success: manager.report_success(node_id, metrics, score)
        On failure: action = manager.report_failure(node_id, error_msg)
                    if action == "debug": re-execute (config was tweaked)
                    if action == "abandon": do nothing, move on
        """

    def _run_ablations(self) -> list[dict]:
        """
        Stage 4: For best config, generate ablation configs,
        run each, compute ablation report.
        """

    def get_tree_visualization_data(self) -> dict:
        """
        Return tree structure for frontend visualization:
        {
            "nodes": [{id, parent_id, config, score, status, stage, depth}],
            "edges": [{source, target}],
            "best_path": [node_ids from root to best node],
            "stages": {stage_name: [node_ids]},
        }
        """
```

**Validate:** Run BFTS with max_steps=15 on sample docs + testset. Print:
- Stage transition log (when each stage started/ended and why)
- Tree summary (nodes per stage, success/fail/pruned counts)
- Best config and score
- Ablation table showing which component matters most
- Verify the tree structure: best node's ancestors should show progressive improvement

---

## Phase 6 — CLI entry point

**Claude Code prompt:**
```
Implement run.py as the full end-to-end CLI:

python run.py --docs data/sample_docs --experiments 15 --strategy bfts

It should:
1. Load documents from --docs
2. Generate synthetic test data (or load from --testset if provided)
3. Run the BFTS optimization loop with --experiments max steps
4. Print real-time progress: stage, node id, score, best-so-far
5. At the end print:
   - Best config (formatted table)
   - RAGAS metrics (formatted table)
   - Ablation report (formatted table)
   - Stage transition log
6. Save to results/ directory:
   - best_config.json and best_config.yaml
   - trajectory.csv
   - ablation_report.json
   - metrics_summary.json
   - testset_used.csv

Also support:
  --strategy random   (use RandomStrategy instead of BFTS)
  --strategy greedy   (use GreedyMutationStrategy)
  --testset path.csv  (skip testgen, load existing testset)
  --skip-testgen      (use a minimal hardcoded testset for quick testing)
```

---

## Phase 7 — Streamlit UI

**Claude Code prompt:**
```
Create app.py as a Streamlit multi-page app with 4 pages.

PAGE 1 — Upload & generate test data
- File uploader (PDF, MD, TXT — multiple)
- Sliders: testset_size (5-50), question type distribution (4 sliders
  for simple/multi_context/reasoning/conditional, must sum to 1.0)
- Button: "Generate test questions"
- Progress: show each testgen step (KG extraction → seeding → evolution → filtering)
- Results: display table of questions with type and difficulty columns
- Stats: bar chart of question types, heatmap of difficulty matrix
- Download CSV button

PAGE 2 — Run optimization
- Strategy selector: BFTS (default), Random, Greedy
- Slider: max experiments (5-30)
- BFTS config panel: num_seeds, max_debug_depth, debug_prob
- Button: "Find best config"
- Real-time: progress bar, current stage label, live score trajectory chart
- When done: best config card, stage transition timeline

PAGE 3 — Results dashboard
- Best config display with parameter values
- Radar chart: 5 RAGAS metrics
- Comparison table: best vs default baseline with deltas and significance stars
- Ablation bar chart: contribution of each component
- Stratified results: metrics broken down by difficulty level
- Trajectory line chart with stage boundaries marked

PAGE 4 — Export
- Download best config as JSON / YAML
- Download trajectory as CSV
- Code snippet: "Use this config with LangChain" (auto-generated Python code)
- Full experiment report as downloadable markdown
```

---

## Phase 8 — Polish

**Claude Code prompt:**
```
Add finishing touches:

1. src/visualization.py with plotly figure builders:
   - plot_trajectory(trajectory, stage_transitions) — line chart with
     vertical lines at stage boundaries, color-coded by stage
   - plot_radar(metrics_dict) — radar/spider chart of RAGAS metrics
   - plot_difficulty_heatmap(testset_df) — 2D heatmap of reasoning_depth
     × semantic_distance with question counts
   - plot_ablation_bar(ablation_report) — horizontal bar chart of deltas
   - plot_tree(tree_data) — tree visualization using plotly treemap
     with nodes colored by status (green=success, red=failed, gray=pruned)

2. Comprehensive README.md with:
   - One-command install + run
   - Architecture diagram (can reference the diagrams from our conversation)
   - How testgen works (KG → Evol-Instruct → difficulty matrix)
   - How BFTS works (4 stages, from AI Scientist v2)
   - References to all papers we discussed

3. Clean up: remove any leftover files, ensure all imports work,
   run the full pipeline end-to-end one more time to verify.
```

---

## Execution order

| Phase | What | Hours | Depends on |
|-------|------|-------|------------|
| 0 | Skeleton + config | 1 | — |
| 1 | Ingest + chunk | 2 | 0 |
| 2a | Knowledge graph | 3 | 1 |
| 2b | Seed questions | 2 | 2a |
| 2c | Evol-Instruct evolution | 3-4 | 2a, 2b |
| 2d | Difficulty matrix | 2 | 2c |
| 2e | Testgen orchestrator | 2 | 2a-2d |
| 3 | RAG pipeline | 4-5 | 1 |
| 4 | Evaluation + stratified | 3 | 3, 2e |
| 5a | Tree node + search space | 1 | — |
| 5b | Experiment manager | 3-4 | 5a |
| 5c | Ablation runner | 1 | 5a |
| 5d | BFTS loop | 2-3 | 5a-5c, 3, 4 |
| 6 | CLI run.py | 2 | all above |
| 7 | Streamlit UI | 3-4 | all above |
| 8 | Visualization + polish | 2-3 | all above |
| **Total** | | **~35-42 hours** | |

**If short on time, cut in this order (last to first):**
1. Phase 8 polish — demo from notebooks
2. Phase 7 Streamlit — use CLI only
3. Phase 2d difficulty matrix — still novel without it
4. Phase 2c conditional evolution — keep simple/multi-context/reasoning only
5. Simplify Phase 5b — use greedy mutation instead of full BFTS

**Minimum to show ALL novelty (~20 hours):**
Phases 0, 1, 2a-2c, 2e, 3, 4, 5a-5d, 6 — run from CLI, demo from notebook