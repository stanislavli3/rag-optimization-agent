/**
 * Upload — TestGen pipeline page.
 *
 * Visualises the 5-step synthetic testset generation flow: KG extraction →
 * seeds → Evol-Instruct evolution → groundedness filter → difficulty scoring.
 * Each stage is a card that transitions pending → running (spinner) → done
 * (stats) as SSE events arrive from /api/testgen/{jobId}/stream/. On
 * completion the page shows the generated-question table, the 2D difficulty
 * heatmap, a CSV download, and a shortcut to kick off optimisation.
 */
import { CSSProperties, useEffect, useMemo, useRef, useState } from "react";

import Heatmap from "../components/Heatmap";


type StepKey =
  | "knowledge_graph"
  | "seeds"
  | "evolution"
  | "filter"
  | "difficulty";

type StepStatus = "pending" | "running" | "done" | "failed";

interface StepState {
  key: StepKey;
  title: string;
  icon: string;
  status: StepStatus;
  stats: Record<string, number | string>;
}

interface UploadedDoc {
  name: string;
  size: number;
  file: File;
}

interface TypeDistribution {
  simple: number;
  multi_context: number;
  reasoning: number;
  conditional: number;
}

interface GeneratedQuestion {
  question: string;
  question_type: string;
  difficulty: "easy" | "medium" | "hard";
  reasoning_depth?: number;
  semantic_distance?: number;
}


const INITIAL_STEPS: StepState[] = [
  { key: "knowledge_graph", title: "Knowledge Graph", icon: "🕸", status: "pending", stats: {} },
  { key: "seeds", title: "Seeds", icon: "🌱", status: "pending", stats: {} },
  { key: "evolution", title: "Evol-Instruct", icon: "🧬", status: "pending", stats: {} },
  { key: "filter", title: "Groundedness Filter", icon: "🛡", status: "pending", stats: {} },
  { key: "difficulty", title: "Difficulty Matrix", icon: "📊", status: "pending", stats: {} },
];


function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}


export default function Upload() {
  const [docs, setDocs] = useState<UploadedDoc[]>([]);
  const [numQuestions, setNumQuestions] = useState(20);
  const [dist, setDist] = useState<TypeDistribution>({
    simple: 30,
    multi_context: 30,
    reasoning: 25,
    conditional: 15,
  });
  const [steps, setSteps] = useState<StepState[]>(INITIAL_STEPS);
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [questions, setQuestions] = useState<GeneratedQuestion[]>([]);
  const [csvUrl, setCsvUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  const distSum = dist.simple + dist.multi_context + dist.reasoning + dist.conditional;
  const distValid = Math.abs(distSum - 100) < 0.01;

  const addFiles = (files: FileList | File[]) => {
    const arr = Array.from(files).map((f) => ({
      name: f.name,
      size: f.size,
      file: f,
    }));
    setDocs((prev) => [...prev, ...arr]);
  };

  const removeDoc = (name: string) =>
    setDocs((prev) => prev.filter((d) => d.name !== name));

  const updateStep = (key: StepKey, patch: Partial<StepState>) =>
    setSteps((prev) =>
      prev.map((s) => (s.key === key ? { ...s, ...patch } : s)),
    );

  const resetRun = () => {
    esRef.current?.close();
    esRef.current = null;
    setSteps(INITIAL_STEPS.map((s) => ({ ...s, stats: {} })));
    setQuestions([]);
    setCsvUrl(null);
    setError(null);
  };

  const onGenerate = async () => {
    if (docs.length === 0) {
      setError("Upload at least one document first.");
      return;
    }
    if (!distValid) {
      setError(`Distribution must sum to 100 (currently ${distSum}).`);
      return;
    }
    resetRun();
    setRunning(true);

    try {
      const form = new FormData();
      docs.forEach((d) => form.append("documents", d.file, d.name));
      form.append("num_questions", String(numQuestions));
      form.append("distribution", JSON.stringify(dist));

      const res = await fetch("/api/testgen/", { method: "POST", body: form });
      if (!res.ok) throw new Error(`failed to start testgen (${res.status})`);
      const { job_id } = await res.json();
      setJobId(job_id);
    } catch (err) {
      setRunning(false);
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(`/api/testgen/${jobId}/stream/`);
    esRef.current = es;

    es.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data);
        const { event, data } = ev;
        switch (event) {
          case "step_start":
            updateStep(data.step as StepKey, { status: "running" });
            break;
          case "step_progress":
            updateStep(data.step as StepKey, {
              status: "running",
              stats: { ...(data.stats ?? {}) },
            });
            break;
          case "step_done":
            updateStep(data.step as StepKey, {
              status: "done",
              stats: { ...(data.stats ?? {}) },
            });
            break;
          case "step_failed":
            updateStep(data.step as StepKey, { status: "failed" });
            setError(data.error || "step failed");
            break;
          case "questions":
            setQuestions(data.items as GeneratedQuestion[]);
            break;
          case "complete":
            if (data.csv_url) setCsvUrl(data.csv_url);
            if (data.questions) setQuestions(data.questions);
            setRunning(false);
            es.close();
            break;
          case "stream_end":
            setRunning(false);
            es.close();
            break;
          default:
            break;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    };

    es.onerror = () => {
      setError("SSE connection lost");
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [jobId]);

  // Build the 3×3 difficulty matrix from generated questions for the heatmap.
  const difficultyMatrix = useMemo(() => {
    const rows = ["low", "mid", "high"];
    const m = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const bucket = (v: number | undefined) => {
      if (v === undefined) return -1;
      if (v < 1.5) return 0;
      if (v < 2.5) return 1;
      return 2;
    };
    for (const q of questions) {
      const r = bucket(q.reasoning_depth);
      const c = bucket(q.semantic_distance);
      if (r >= 0 && c >= 0) m[r][c] += 1;
    }
    return { matrix: m, rowLabels: rows, colLabels: rows };
  }, [questions]);

  return (
    <div style={page}>
      <h2 style={pageTitle}>TestGen pipeline</h2>

      <section style={card}>
        <h3 style={cardTitle}>
          <span style={{ marginRight: 8 }}>📄</span>Upload documents
        </h3>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            if (e.dataTransfer.files) addFiles(e.dataTransfer.files);
          }}
          style={{
            ...dropZone,
            borderColor: dragOver ? "#3b82f6" : "#cbd5e1",
            background: dragOver ? "#eff6ff" : "#f8fafc",
          }}
        >
          <div style={{ fontSize: 14, color: "#475569" }}>
            Drag &amp; drop .pdf / .md / .txt files here
          </div>
          <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4 }}>or</div>
          <label style={browseBtn}>
            Browse
            <input
              type="file"
              multiple
              accept=".pdf,.md,.txt"
              style={{ display: "none" }}
              onChange={(e) => e.target.files && addFiles(e.target.files)}
            />
          </label>
        </div>
        {docs.length > 0 && (
          <ul style={docList}>
            {docs.map((d) => (
              <li key={d.name} style={docRow}>
                <span>📎 {d.name}</span>
                <span style={{ color: "#64748b", fontSize: 12 }}>
                  {fmtBytes(d.size)}
                </span>
                <button style={removeBtn} onClick={() => removeDoc(d.name)}>
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section style={card}>
        <h3 style={cardTitle}>
          <span style={{ marginRight: 8 }}>⚙️</span>Configure testset
        </h3>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
          <div>
            <label style={label}>
              Size: <strong>{numQuestions}</strong> questions
            </label>
            <input
              type="range"
              min={5}
              max={100}
              value={numQuestions}
              onChange={(e) => setNumQuestions(parseInt(e.target.value, 10))}
              style={{ width: 260 }}
            />
          </div>
          <div>
            <label style={label}>
              Distribution (must sum to 100, currently {distSum})
            </label>
            {(
              [
                ["simple", "Simple"],
                ["multi_context", "Multi-context"],
                ["reasoning", "Reasoning"],
                ["conditional", "Conditional"],
              ] as Array<[keyof TypeDistribution, string]>
            ).map(([k, lbl]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 110, fontSize: 12, color: "#475569" }}>
                  {lbl}
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={dist[k]}
                  onChange={(e) =>
                    setDist({ ...dist, [k]: parseInt(e.target.value, 10) })
                  }
                  style={{ flex: 1, width: 200 }}
                />
                <span style={{ width: 40, textAlign: "right", fontSize: 12 }}>
                  {dist[k]}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <button
        onClick={onGenerate}
        disabled={running || docs.length === 0 || !distValid}
        style={generateBtn}
      >
        {running ? "Generating…" : "Generate test questions"}
      </button>

      {error && <div style={errorBox}>{error}</div>}

      <section style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 20 }}>
        {steps.map((s) => (
          <StepCard key={s.key} step={s} />
        ))}
      </section>

      {questions.length > 0 && (
        <section style={card}>
          <h3 style={cardTitle}>Generated questions ({questions.length})</h3>
          <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
            <div style={{ flex: 2, minWidth: 320 }}>
              <table style={table}>
                <thead>
                  <tr>
                    <th style={th}>Question</th>
                    <th style={th}>Type</th>
                    <th style={th}>Difficulty</th>
                  </tr>
                </thead>
                <tbody>
                  {questions.slice(0, 50).map((q, i) => (
                    <tr key={i}>
                      <td style={td}>{q.question}</td>
                      <td style={td}>{q.question_type}</td>
                      <td style={td}>
                        <span style={difficultyBadge(q.difficulty)}>
                          {q.difficulty}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {questions.length > 50 && (
                <div style={{ fontSize: 11, color: "#64748b", marginTop: 6 }}>
                  Showing first 50 of {questions.length}.
                </div>
              )}
            </div>
            <div>
              <div style={{ fontSize: 12, color: "#475569", marginBottom: 4 }}>
                Difficulty matrix
              </div>
              <Heatmap
                matrix={difficultyMatrix.matrix}
                rowLabels={difficultyMatrix.rowLabels}
                colLabels={difficultyMatrix.colLabels}
                yAxisLabel="reasoning depth"
                xAxisLabel="semantic distance"
                formatValue={(v) => `${v}`}
              />
            </div>
          </div>

          <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
            {csvUrl && (
              <a href={csvUrl} download style={downloadBtn}>
                ⬇ Download CSV
              </a>
            )}
            <a href="/optimize" style={runOptBtn}>
              Run Optimization →
            </a>
          </div>
        </section>
      )}
    </div>
  );
}


function StepCard({ step }: { step: StepState }) {
  return (
    <div style={{ ...card, padding: 14, minHeight: 120 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontWeight: 600, fontSize: 13 }}>
          <span style={{ marginRight: 6 }}>{step.icon}</span>
          {step.title}
        </div>
        <StatusPill status={step.status} />
      </div>
      <div style={{ marginTop: 10 }}>
        {Object.keys(step.stats).length === 0 ? (
          <div style={{ fontSize: 11, color: "#94a3b8", fontStyle: "italic" }}>
            {step.status === "running" ? "working…" : "awaiting data"}
          </div>
        ) : (
          <ul style={statsList}>
            {Object.entries(step.stats).map(([k, v]) => (
              <li key={k}>
                <span style={{ color: "#64748b" }}>{k}:</span>{" "}
                <strong>{typeof v === "number" ? v : String(v)}</strong>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}


function StatusPill({ status }: { status: StepStatus }) {
  const meta = {
    pending: { color: "#94a3b8", bg: "#f1f5f9", label: "pending", icon: "○" },
    running: { color: "#1d4ed8", bg: "#dbeafe", label: "running", icon: "⟳" },
    done: { color: "#047857", bg: "#d1fae5", label: "done", icon: "✓" },
    failed: { color: "#b91c1c", bg: "#fee2e2", label: "failed", icon: "✗" },
  }[status];
  return (
    <span
      style={{
        padding: "2px 8px",
        borderRadius: 10,
        fontSize: 11,
        fontWeight: 600,
        color: meta.color,
        background: meta.bg,
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
      }}
    >
      <span
        style={{
          display: "inline-block",
          animation: status === "running" ? "spin 1.1s linear infinite" : undefined,
        }}
      >
        {meta.icon}
      </span>
      {meta.label}
      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
    </span>
  );
}


const page: CSSProperties = {
  padding: 24,
  background: "#f8fafc",
  minHeight: "100vh",
  fontFamily: "system-ui, -apple-system, sans-serif",
  maxWidth: 1100,
  margin: "0 auto",
};

const pageTitle: CSSProperties = {
  fontSize: 20,
  color: "#0f172a",
  marginTop: 0,
  marginBottom: 20,
};

const card: CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 16,
  marginBottom: 16,
};

const cardTitle: CSSProperties = {
  fontSize: 14,
  marginTop: 0,
  marginBottom: 12,
  color: "#0f172a",
};

const dropZone: CSSProperties = {
  border: "2px dashed",
  borderRadius: 8,
  padding: 24,
  textAlign: "center",
  transition: "background 120ms",
};

const browseBtn: CSSProperties = {
  display: "inline-block",
  marginTop: 8,
  padding: "6px 14px",
  background: "#1e293b",
  color: "#f8fafc",
  borderRadius: 4,
  fontSize: 12,
  fontWeight: 600,
  cursor: "pointer",
};

const docList: CSSProperties = {
  listStyle: "none",
  padding: 0,
  marginTop: 12,
  marginBottom: 0,
};

const docRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "6px 8px",
  background: "#f8fafc",
  borderRadius: 4,
  fontSize: 13,
  marginTop: 4,
};

const removeBtn: CSSProperties = {
  marginLeft: "auto",
  background: "none",
  border: "none",
  fontSize: 16,
  cursor: "pointer",
  color: "#94a3b8",
};

const label: CSSProperties = {
  display: "block",
  fontSize: 12,
  color: "#475569",
  marginBottom: 6,
  fontWeight: 600,
};

const generateBtn: CSSProperties = {
  padding: "10px 20px",
  background: "#3b82f6",
  color: "#fff",
  border: "none",
  borderRadius: 4,
  fontWeight: 600,
  cursor: "pointer",
  fontSize: 14,
};

const errorBox: CSSProperties = {
  marginTop: 12,
  padding: "8px 12px",
  background: "#fef2f2",
  color: "#b91c1c",
  border: "1px solid #fecaca",
  borderRadius: 4,
  fontSize: 13,
};

const statsList: CSSProperties = {
  listStyle: "none",
  padding: 0,
  margin: 0,
  fontSize: 12,
  lineHeight: 1.6,
};

const table: CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: 12,
};

const th: CSSProperties = {
  textAlign: "left",
  padding: "6px 8px",
  borderBottom: "1px solid #e2e8f0",
  color: "#64748b",
  fontWeight: 600,
};

const td: CSSProperties = {
  padding: "6px 8px",
  borderBottom: "1px solid #f1f5f9",
  color: "#0f172a",
  verticalAlign: "top",
};

function difficultyBadge(d: "easy" | "medium" | "hard"): CSSProperties {
  const map = {
    easy: { bg: "#d1fae5", color: "#047857" },
    medium: { bg: "#fef3c7", color: "#b45309" },
    hard: { bg: "#fee2e2", color: "#b91c1c" },
  }[d];
  return {
    display: "inline-block",
    padding: "1px 8px",
    borderRadius: 8,
    background: map.bg,
    color: map.color,
    fontSize: 11,
    fontWeight: 600,
  };
}

const downloadBtn: CSSProperties = {
  padding: "8px 14px",
  background: "#0f172a",
  color: "#f8fafc",
  textDecoration: "none",
  borderRadius: 4,
  fontSize: 13,
  fontWeight: 600,
};

const runOptBtn: CSSProperties = {
  padding: "8px 14px",
  background: "#10b981",
  color: "#fff",
  textDecoration: "none",
  borderRadius: 4,
  fontSize: 13,
  fontWeight: 600,
};
