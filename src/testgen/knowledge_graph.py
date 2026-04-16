"""LLM-driven knowledge-graph extraction from chunked documents.

Generative AI aspect: each chunk is passed through a structured extraction prompt
(`P(entities, facts, relations | chunk_text)`). Outputs are merged into a networkx
graph with cross-chunk entity matching.
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.testgen.llm import LLMLike

logger = logging.getLogger(__name__)


EXTRACT_ENTITIES_AND_FACTS = """You are extracting a knowledge graph from a passage.

Given the PASSAGE below, output three sections verbatim in this exact format:

ENTITIES:
- <short canonical name of entity or concept>
- ...

FACTS:
- <one self-contained factual statement grounded in the passage>
- ...

RELATIONS:
- <source_entity> | <relation_type> | <target_entity> | <evidence sentence from passage>
- ...

Rules:
- Entities must be short noun phrases, canonical (no pronouns).
- Facts must be stand-alone claims verifiable from the passage.
- Relations must connect entities that both appear in ENTITIES.
- Use only information from the passage. No outside knowledge.

PASSAGE:
\"\"\"{chunk}\"\"\"
"""


@dataclass
class KGNode:
    id: str
    text: str
    node_type: str  # "entity" | "concept" | "fact"
    source_chunk_id: str
    source_doc: str


@dataclass
class KGEdge:
    source_id: str
    target_id: str
    relation: str
    evidence: str


@dataclass
class KnowledgeGraph:
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)

    def get_facts(self) -> list[dict]:
        return [
            {
                "fact": n.text,
                "source_chunk_id": n.source_chunk_id,
                "source_doc": n.source_doc,
                "node_id": n.id,
            }
            for n in self.nodes
            if n.node_type == "fact"
        ]

    def get_connected_facts(self, n_hops: int = 2) -> list[tuple]:
        try:
            import networkx as nx  # type: ignore
        except Exception:
            return []
        g = nx.Graph()
        for n in self.nodes:
            g.add_node(n.id)
        for e in self.edges:
            g.add_edge(e.source_id, e.target_id, relation=e.relation)

        facts = [n for n in self.nodes if n.node_type == "fact"]
        out: list[tuple] = []
        for a in facts:
            for b in facts:
                if a.id >= b.id:
                    continue
                try:
                    path = nx.shortest_path(g, a.id, b.id)
                except Exception:
                    continue
                if 2 <= len(path) - 1 <= n_hops:
                    out.append((a, b, path))
        return out

    def get_clusters(self) -> list[list[KGNode]]:
        try:
            import networkx as nx  # type: ignore
        except Exception:
            return [self.nodes]
        g = nx.Graph()
        for n in self.nodes:
            g.add_node(n.id, obj=n)
        for e in self.edges:
            g.add_edge(e.source_id, e.target_id)
        clusters: list[list[KGNode]] = []
        for comp in nx.connected_components(g):
            clusters.append([g.nodes[n]["obj"] for n in comp])
        return clusters


def _parse_extraction(raw: str) -> dict:
    """Parse the three-section text response."""
    sections = {"ENTITIES": [], "FACTS": [], "RELATIONS": []}
    current: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        up = stripped.upper().rstrip(":")
        if up in sections:
            current = up
            continue
        if current is None:
            continue
        if stripped.startswith(("-", "*", "•")):
            stripped = stripped[1:].strip()
        if stripped:
            sections[current].append(stripped)
    return sections


def _mk_id() -> str:
    return uuid.uuid4().hex[:10]


def extract_knowledge_graph(chunks: list[Any], llm: LLMLike) -> KnowledgeGraph:
    """Run the extraction prompt on each chunk and merge into a single KG."""
    kg = KnowledgeGraph()
    entity_index: dict[str, str] = {}  # normalized name -> node id

    for ch in chunks:
        text = getattr(ch, "page_content", str(ch))
        meta = getattr(ch, "metadata", {}) or {}
        chunk_id = meta.get("chunk_id") or meta.get("source_file", "chunk")
        source_doc = meta.get("source_file", "unknown")

        try:
            raw = llm.invoke(EXTRACT_ENTITIES_AND_FACTS.format(chunk=text[:4000]))
        except Exception as e:
            logger.warning("LLM extraction failed for %s: %s", chunk_id, e)
            continue

        parsed = _parse_extraction(raw)

        # Entities
        local_entities: dict[str, str] = {}
        for ent in parsed.get("ENTITIES", []):
            key = ent.lower().strip()
            if not key:
                continue
            if key in entity_index:
                local_entities[key] = entity_index[key]
                continue
            nid = _mk_id()
            entity_index[key] = nid
            local_entities[key] = nid
            kg.nodes.append(
                KGNode(
                    id=nid,
                    text=ent,
                    node_type="entity",
                    source_chunk_id=chunk_id,
                    source_doc=source_doc,
                )
            )

        # Facts
        fact_ids: list[str] = []
        for fact in parsed.get("FACTS", []):
            nid = _mk_id()
            fact_ids.append(nid)
            kg.nodes.append(
                KGNode(
                    id=nid,
                    text=fact,
                    node_type="fact",
                    source_chunk_id=chunk_id,
                    source_doc=source_doc,
                )
            )
            # Link fact to entities it mentions
            for ent_key, ent_id in local_entities.items():
                if ent_key in fact.lower():
                    kg.edges.append(KGEdge(nid, ent_id, "mentions", fact))

        # Relations (entity | relation | entity | evidence)
        for rel in parsed.get("RELATIONS", []):
            parts = [p.strip() for p in rel.split("|")]
            if len(parts) < 3:
                continue
            src_key = parts[0].lower()
            tgt_key = parts[2].lower()
            relation = parts[1]
            evidence = parts[3] if len(parts) > 3 else ""
            if src_key in local_entities and tgt_key in local_entities:
                kg.edges.append(
                    KGEdge(
                        local_entities[src_key],
                        local_entities[tgt_key],
                        relation,
                        evidence,
                    )
                )

    # Cross-chunk bridging: fact nodes sharing an entity get a "co_mentions" edge
    entity_to_facts: dict[str, list[str]] = defaultdict(list)
    for e in kg.edges:
        if e.relation == "mentions":
            entity_to_facts[e.target_id].append(e.source_id)
    seen: set[tuple[str, str]] = set()
    for fact_ids_ in entity_to_facts.values():
        for i, a in enumerate(fact_ids_):
            for b in fact_ids_[i + 1 :]:
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                kg.edges.append(KGEdge(a, b, "co_mentions", ""))

    logger.info("KG: %d nodes, %d edges", len(kg.nodes), len(kg.edges))
    return kg
