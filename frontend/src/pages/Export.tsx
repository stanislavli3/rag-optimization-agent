/**
 * Export — ship the winning config out.
 *
 * Offers downloads (JSON / YAML / CSV / Markdown) plus an inline LangChain
 * code snippet the user can copy-paste. All serialisation is client-side so
 * no extra backend endpoint is needed.
 */
import { CSSProperties, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { useRun } from "../context/RunContext";
import {
  callout,
  card,
  chip,
  colors,
  font,
  ghostButton,
  metricLabel,
  pageStyle,
  pageSubtitle,
  pageTitle,
  primaryButton,
  radius,
  sectionTitle,
  space,
  tableStyles,
} from "../theme";
import { AblationEntry } from "../components/AblationWaterfall";
import { TrajectoryPoint } from "../components/TrajectoryGraph";

export default function ExportPage() {
  const { current } = useRun();
  const [copied, setCopied] = useState<"langchain" | "llamaindex" | null>(null);

  const artefacts = useMemo(() => {
    if (!current) return null;
    return {
      json: JSON.stringify(current.bestConfig, null, 2),
      yaml: toYaml(current.bestConfig),
      csv: trajectoryCsv(current.trajectory),
      markdown: markdownReport(
        current.bestConfig,
        current.bestMetrics,
        current.ablation,
        current.bestScore,
        current.baselineScore,
      ),
      langchain: langchainSnippet(current.bestConfig),
      llamaindex: llamaIndexSnippet(current.bestConfig),
    };
  }, [current]);

  if (!current || !artefacts) {
    return (
      <div style={pageStyle}>
        <h1 style={pageTitle}>Export</h1>
        <p style={pageSubtitle}>Run the optimizer first, then return here.</p>
        <div style={callout("neutral")}>
          <Link to="/optimize" style={{ color: colors.accent }}>
            → Go to Auto-Optimize
          </Link>
        </div>
      </div>
    );
  }

  const copySnippet = async (kind: "langchain" | "llamaindex") => {
    try {
      const payload =
        kind === "langchain" ? artefacts.langchain : artefacts.llamaindex;
      await navigator.clipboard.writeText(payload);
      setCopied(kind);
      setTimeout(() => setCopied(null), 1200);
    } catch {
      setCopied(null);
    }
  };

  return (
    <div style={pageStyle}>
      <h1 style={pageTitle}>Export</h1>
      <p style={pageSubtitle}>
        Take the winning configuration and ship it. All downloads are produced
        client-side from the current run: {current.label}.
      </p>

      <section style={{ ...card, marginBottom: space.lg }}>
        <h2 style={sectionTitle}>Downloads</h2>
        <div style={downloadsGrid}>
          <DownloadCard
            title="best_config.json"
            description="Winning configuration in JSON."
            blob={artefacts.json}
            mime="application/json"
            filename="best_config.json"
          />
          <DownloadCard
            title="best_config.yaml"
            description="Same, YAML for human-friendly diff-ability."
            blob={artefacts.yaml}
            mime="text/yaml"
            filename="best_config.yaml"
          />
          <DownloadCard
            title="trajectory.csv"
            description="Score per iteration, with stage + insight."
            blob={artefacts.csv}
            mime="text/csv"
            filename="trajectory.csv"
          />
          <DownloadCard
            title="experiment_report.md"
            description="Markdown report with metrics and ablation table."
            blob={artefacts.markdown}
            mime="text/markdown"
            filename="experiment_report.md"
          />
          <DownloadCard
            title="rag_pipeline_langchain.py"
            description="LangChain runtime wired to the winning config."
            blob={artefacts.langchain}
            mime="text/x-python"
            filename="rag_pipeline_langchain.py"
          />
          <DownloadCard
            title="rag_pipeline_llamaindex.py"
            description="Same pipeline, rebuilt on LlamaIndex."
            blob={artefacts.llamaindex}
            mime="text/x-python"
            filename="rag_pipeline_llamaindex.py"
          />
        </div>
      </section>

      <section style={{ ...card, marginBottom: space.lg }}>
        <div style={snippetHead}>
          <h2 style={{ ...sectionTitle, margin: 0 }}>LangChain snippet</h2>
          <button style={ghostButton} onClick={() => copySnippet("langchain")}>
            {copied === "langchain" ? "✓ Copied" : "Copy"}
          </button>
        </div>
        <pre style={codeBlock}>{artefacts.langchain}</pre>
      </section>

      <section style={{ ...card, marginBottom: space.lg }}>
        <div style={snippetHead}>
          <h2 style={{ ...sectionTitle, margin: 0 }}>LlamaIndex snippet</h2>
          <button style={ghostButton} onClick={() => copySnippet("llamaindex")}>
            {copied === "llamaindex" ? "✓ Copied" : "Copy"}
          </button>
        </div>
        <pre style={codeBlock}>{artefacts.llamaindex}</pre>
      </section>

      <section style={{ ...card, marginBottom: space.lg }}>
        <h2 style={sectionTitle}>Summary</h2>
        <table style={tableStyles.table}>
          <tbody>
            <SummaryRow
              label="Best score"
              value={
                <span style={chip("success")}>
                  {current.bestScore?.toFixed(3) ?? "—"}
                </span>
              }
            />
            <SummaryRow
              label="Baseline"
              value={
                current.baselineScore !== null
                  ? current.baselineScore.toFixed(3)
                  : "—"
              }
            />
            <SummaryRow
              label="Iterations"
              value={String(current.trajectory.length)}
            />
            <SummaryRow
              label="Ablation entries"
              value={String(current.ablation.length)}
            />
            <SummaryRow
              label="Completed"
              value={new Date(current.completedAt).toLocaleString()}
            />
          </tbody>
        </table>
      </section>

      <div style={{ display: "flex", gap: space.sm }}>
        <Link to="/results" style={primaryButton}>
          Back to results
        </Link>
        <Link to="/comparison" style={ghostButton}>
          Compare with past runs
        </Link>
      </div>
    </div>
  );
}

function SummaryRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <tr>
      <td
        style={{
          ...tableStyles.td,
          width: 200,
          color: colors.textMuted,
        }}
      >
        {label}
      </td>
      <td style={{ ...tableStyles.td, fontFamily: font.mono }}>{value}</td>
    </tr>
  );
}

function DownloadCard({
  title,
  description,
  blob,
  mime,
  filename,
}: {
  title: string;
  description: string;
  blob: string;
  mime: string;
  filename: string;
}) {
  const href = useMemo(() => {
    const b = new Blob([blob], { type: mime });
    return URL.createObjectURL(b);
  }, [blob, mime]);

  return (
    <a href={href} download={filename} style={downloadCard}>
      <div style={{ display: "flex", alignItems: "center", gap: space.sm }}>
        <span style={downloadIcon}>↓</span>
        <span style={{ fontFamily: font.mono, fontSize: 13, fontWeight: 500 }}>
          {title}
        </span>
      </div>
      <div style={{ fontSize: 12, color: colors.textMuted, marginTop: 4 }}>
        {description}
      </div>
    </a>
  );
}

// ---- Serialisation helpers -------------------------------------------------

function toYaml(value: unknown, indent = 0): string {
  const pad = " ".repeat(indent);
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    return value.map((v) => `\n${pad}- ${toYaml(v, indent + 2)}`).join("");
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return "{}";
    return entries
      .map(([k, v]) => {
        const nested = toYaml(v, indent + 2);
        const isScalar = !nested.startsWith("\n") && !nested.includes("\n");
        return `\n${pad}${k}: ${isScalar ? nested : nested}`;
      })
      .join("");
  }
  return String(value);
}

function trajectoryCsv(points: TrajectoryPoint[]): string {
  const header = ["iteration", "stage", "status", "score", "insight"];
  const rows = points.map((p) =>
    [
      p.iteration,
      p.stage,
      p.status ?? "",
      p.score.toFixed(6),
      p.insight ? `"${p.insight.replace(/"/g, '""')}"` : "",
    ].join(","),
  );
  return [header.join(","), ...rows].join("\n");
}

function markdownReport(
  config: Record<string, unknown>,
  metrics: Record<string, any>,
  ablation: AblationEntry[],
  bestScore: number | null,
  baselineScore: number | null,
): string {
  const lines: string[] = ["# RAG Optimizer — Experiment Report", ""];
  if (bestScore !== null) {
    lines.push(`**Best RAGAS score**: ${bestScore.toFixed(3)}`);
  }
  if (baselineScore !== null) {
    lines.push(`**Baseline**: ${baselineScore.toFixed(3)}`);
    if (bestScore !== null) {
      lines.push(
        `**Lift**: +${(bestScore - baselineScore).toFixed(3)} (${(
          ((bestScore - baselineScore) / baselineScore) *
          100
        ).toFixed(1)}%)`,
      );
    }
  }
  lines.push("", "## Best configuration", "```json", JSON.stringify(config, null, 2), "```", "");

  if (Object.keys(metrics).length > 0) {
    lines.push("## Metrics", "");
    for (const [k, v] of Object.entries(metrics)) {
      if (typeof v === "number") {
        lines.push(`- **${metricLabel[k] ?? k}**: ${v.toFixed(4)}`);
      }
    }
    lines.push("");
  }

  if (ablation.length > 0) {
    lines.push(
      "## Ablation",
      "",
      "| Parameter | Optimized | Default | Δ | Contribution |",
      "|---|---|---|---|---|",
    );
    for (const r of ablation) {
      lines.push(
        `| ${r.param} | ${fmtAblationValue(r.optimized_value)} | ${fmtAblationValue(
          r.default_value,
        )} | ${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(3)} | ${r.contribution_pct.toFixed(1)}% |`,
      );
    }
    lines.push("");
  }

  return lines.join("\n");
}

function fmtAblationValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? `${v}` : v.toFixed(2);
  return String(v);
}

function langchainSnippet(config: Record<string, unknown>): string {
  const topK = config.top_k ?? 5;
  const chunkSize = config.chunk_size ?? 512;
  const overlap = config.chunk_overlap ?? 50;
  const embeddingModel =
    config.embedding_model ?? "sentence-transformers/all-MiniLM-L6-v2";
  const reranker = config.reranker ?? null;
  const promptStyle = config.prompt_style ?? "default";
  return `# Drop-in LangChain RAG pipeline with the optimizer's best config.
# Install: pip install langchain langchain-community langchain-huggingface chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CONFIG = ${JSON.stringify(config, null, 2)}

def build_chain(raw_documents, llm):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=${chunkSize},
        chunk_overlap=${overlap},
    )
    docs = splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(model_name=${JSON.stringify(embeddingModel)})
    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": ${topK}})

    reranker = ${JSON.stringify(reranker)}
    if reranker:
        # Plug in cross-encoder rerank here, e.g. BAAI/bge-reranker-base.
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        ce = HuggingFaceCrossEncoder(model_name=reranker)
        retriever = ContextualCompressionRetriever(
            base_compressor=CrossEncoderReranker(model=ce, top_n=${topK}),
            base_retriever=retriever,
        )

    prompt_style = ${JSON.stringify(promptStyle)}
    tmpl = {
        "default": "Use the context to answer the question.\\n\\n{context}\\n\\nQ: {question}\\nA:",
        "cot": "Think step by step using the context, then answer.\\n\\n{context}\\n\\nQ: {question}\\nA:",
        "structured": "You answer using ONLY the context. If missing, say so.\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\\nAnswer:",
    }.get(prompt_style, "{context}\\n\\n{question}")
    prompt = ChatPromptTemplate.from_template(tmpl)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
`;
}

function llamaIndexSnippet(config: Record<string, unknown>): string {
  const topK = config.top_k ?? 5;
  const chunkSize = config.chunk_size ?? 512;
  const overlap = config.chunk_overlap ?? 50;
  const embeddingModel =
    config.embedding_model ?? "sentence-transformers/all-MiniLM-L6-v2";
  const reranker = config.reranker ?? null;
  const promptStyle = config.prompt_style ?? "default";
  return `# Drop-in LlamaIndex RAG pipeline with the optimizer's best config.
# Install: pip install llama-index llama-index-embeddings-huggingface
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

CONFIG = ${JSON.stringify(config, null, 2)}

def build_query_engine(documents, llm):
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=${JSON.stringify(embeddingModel)})
    Settings.node_parser = SentenceSplitter(
        chunk_size=${chunkSize},
        chunk_overlap=${overlap},
    )

    index = VectorStoreIndex.from_documents(documents)

    postprocessors = []
    reranker = ${JSON.stringify(reranker)}
    if reranker:
        from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
        postprocessors.append(
            SentenceTransformerRerank(model=reranker, top_n=${topK})
        )

    prompt_style = ${JSON.stringify(promptStyle)}
    tmpl_text = {
        "default": "Context:\\n{context_str}\\n\\nQuery: {query_str}\\nAnswer:",
        "cot": "Think step by step using the context below.\\n\\nContext:\\n{context_str}\\n\\nQuery: {query_str}\\nAnswer:",
        "structured": "Answer only from the context. If unknown, say so.\\n\\nContext:\\n{context_str}\\n\\nQuery: {query_str}\\nAnswer:",
    }.get(prompt_style)
    text_qa_template = PromptTemplate(tmpl_text) if tmpl_text else None

    return index.as_query_engine(
        similarity_top_k=${topK},
        node_postprocessors=postprocessors,
        text_qa_template=text_qa_template,
    )
`;
}

const downloadsGrid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(2, 1fr)",
  gap: space.sm,
};

const downloadCard: CSSProperties = {
  display: "block",
  padding: space.md,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.md,
  background: colors.bgSubtle,
  color: colors.text,
  textDecoration: "none",
  transition: "background 80ms ease",
  cursor: "pointer",
};

const downloadIcon: CSSProperties = {
  display: "inline-grid",
  placeItems: "center",
  width: 22,
  height: 22,
  borderRadius: 11,
  background: colors.accentSoft,
  color: colors.accent,
  fontWeight: 700,
  fontSize: 13,
};

const snippetHead: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: space.sm,
};

const codeBlock: CSSProperties = {
  margin: 0,
  background: colors.bgSunken,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.md,
  padding: space.md,
  fontFamily: font.mono,
  fontSize: 12,
  lineHeight: 1.5,
  color: colors.text,
  overflow: "auto",
  whiteSpace: "pre",
};
