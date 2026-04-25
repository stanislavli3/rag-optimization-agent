/**
 * Upload — TestGen pipeline page.
 *
 * Visualises the 5-step synthetic testset generation flow: KG extraction →
 * seeds → Evol-Instruct evolution → groundedness filter → difficulty scoring.
 * Each stage is a Notion-style card that transitions pending → running →
 * done as SSE events arrive from /api/testgen/{jobId}/stream/. On completion
 * the page shows the generated-question table, the 2D difficulty heatmap, a
 * CSV download, and a shortcut to kick off optimisation.
 */
import { CSSProperties, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import Heatmap from "../components/Heatmap";
import {
  GeneratedQuestion,
  StepKey,
  StepState,
  StepStatus,
  TypeDistribution,
  useTestGen,
} from "../context/TestGenContext";
import {
  callout,
  card,
  chip,
  colors,
  font,
  ghostButton,
  pageStyle,
  pageSubtitle,
  pageTitle,
  primaryButton,
  radius,
  sectionTitle,
  space,
  tableStyles,
} from "../theme";

interface UploadedDoc {
  name: string;
  size: number;
  file: File;
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Upload() {
  const {
    snapshot,
    setDocs: persistDocs,
    setNumQuestions,
    setDist,
    setSteps,
    setJobId,
    setQuestions,
    setCsvUrl,
    markComplete,
    resetRun: resetPersistedRun,
  } = useTestGen();

  const { numQuestions, dist, steps, jobId, questions, csvUrl } = snapshot;

  // File blobs live only in memory — localStorage can't hold them. Doc
  // metadata (name/size) is persisted via the context so the list survives
  // tab navigation even when the actual File objects don't.
  const [fileBlobs, setFileBlobs] = useState<Record<string, File>>({});
  const docs: UploadedDoc[] = useMemo(
    () =>
      snapshot.docs.map((d) => ({
        name: d.name,
        size: d.size,
        file: fileBlobs[d.name] ?? new File([], d.name),
      })),
    [snapshot.docs, fileBlobs],
  );

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  const distSum = dist.simple + dist.multi_context + dist.reasoning + dist.conditional;
  const distValid = Math.abs(distSum - 100) < 0.01;

  const addFiles = (files: FileList | File[]) => {
    const arr = Array.from(files);
    setFileBlobs((prev) => {
      const next = { ...prev };
      for (const f of arr) next[f.name] = f;
      return next;
    });
    const metas = arr.map((f) => ({ name: f.name, size: f.size }));
    const existing = snapshot.docs.filter(
      (d) => !metas.some((m) => m.name === d.name),
    );
    persistDocs([...existing, ...metas]);
  };

  const removeDoc = (name: string) => {
    persistDocs(snapshot.docs.filter((d) => d.name !== name));
    setFileBlobs((prev) => {
      if (!(name in prev)) return prev;
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };

  const updateStep = (key: StepKey, patch: Partial<StepState>) =>
    setSteps((prev) =>
      prev.map((s) => (s.key === key ? { ...s, ...patch } : s)),
    );

  const resetRun = () => {
    esRef.current?.close();
    esRef.current = null;
    resetPersistedRun();
    setError(null);
  };

  const missingBlobs = snapshot.docs.filter((d) => !(d.name in fileBlobs));

  const onGenerate = async () => {
    if (snapshot.docs.length === 0) {
      setError("Upload at least one document first.");
      return;
    }
    if (missingBlobs.length > 0) {
      setError(
        `Re-upload to re-run: ${missingBlobs.map((d) => d.name).join(", ")}`,
      );
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
      for (const meta of snapshot.docs) {
        const blob = fileBlobs[meta.name];
        form.append("documents", blob, meta.name);
      }
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
    // If this job already finished in a previous session, don't reopen the
    // stream — the persisted snapshot has the final state.
    if (snapshot.completedAt) return;
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
            markComplete();
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

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
    <div style={pageStyle}>
      <h1 style={pageTitle}>Upload &amp; TestGen</h1>
      <p style={pageSubtitle}>
        Drop in a handful of documents. The pipeline builds a knowledge graph,
        evolves seed questions into harder variants, and scores each on a 2D
        difficulty matrix. The resulting testset drives the optimizer on the
        next step.
      </p>

      <section style={{ ...card, marginBottom: space.lg }}>
        <h2 style={sectionTitle}>Documents</h2>
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
            borderColor: dragOver ? colors.accent : colors.border,
            background: dragOver ? colors.accentSoft : colors.bgSubtle,
          }}
        >
          <div style={{ fontSize: 14, color: colors.text, fontWeight: 500 }}>
            Drop .pdf / .md / .txt files
          </div>
          <div style={{ fontSize: 12, color: colors.textFaint, marginTop: 2 }}>
            or click to browse
          </div>
          <label style={{ ...ghostButton, marginTop: space.md, cursor: "pointer" }}>
            Choose files
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
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                  <span style={{ color: colors.textMuted, marginRight: 6 }}>
                    ∎
                  </span>
                  {d.name}
                </span>
                <span style={{ color: colors.textFaint, fontSize: 12, fontFamily: font.mono }}>
                  {fmtBytes(d.size)}
                </span>
                <button style={removeBtn} onClick={() => removeDoc(d.name)} aria-label="remove">
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section style={{ ...card, marginBottom: space.lg }}>
        <h2 style={sectionTitle}>Testset configuration</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: space.xl }}>
          <div>
            <div style={fieldLabel}>
              Size <span style={valueHint}>{numQuestions} questions</span>
            </div>
            <input
              type="range"
              min={5}
              max={100}
              value={numQuestions}
              onChange={(e) => setNumQuestions(parseInt(e.target.value, 10))}
              style={{ width: "100%" }}
            />
          </div>

          <div>
            <div style={fieldLabel}>
              Question type distribution
              <span
                style={{
                  ...valueHint,
                  color: distValid ? colors.textFaint : colors.danger,
                }}
              >
                {distSum}% / 100%
              </span>
            </div>
            {(
              [
                ["simple", "Simple"],
                ["multi_context", "Multi-context"],
                ["reasoning", "Reasoning"],
                ["conditional", "Conditional"],
              ] as Array<[keyof TypeDistribution, string]>
            ).map(([k, lbl]) => (
              <div key={k} style={distRow}>
                <span style={{ width: 120, fontSize: 13, color: colors.textMuted }}>
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
                  style={{ flex: 1 }}
                />
                <span style={{ width: 44, textAlign: "right", fontFamily: font.mono, fontSize: 12, color: colors.text }}>
                  {dist[k]}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div style={{ display: "flex", alignItems: "center", gap: space.md, marginBottom: space.lg }}>
        <button
          onClick={onGenerate}
          disabled={running || docs.length === 0 || !distValid}
          style={{
            ...primaryButton,
            opacity: running || docs.length === 0 || !distValid ? 0.5 : 1,
            cursor: running || docs.length === 0 || !distValid ? "not-allowed" : "pointer",
          }}
        >
          {running ? "Generating…" : "Generate testset"}
        </button>
        <span style={{ color: colors.textFaint, fontSize: 12 }}>
          {docs.length} document{docs.length === 1 ? "" : "s"} ready
        </span>
      </div>

      {error && (
        <div style={{ ...callout("danger"), marginBottom: space.lg }}>{error}</div>
      )}

      <section style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: space.sm, marginBottom: space.xl }}>
        {steps.map((s, i) => (
          <StepCard key={s.key} index={i} step={s} />
        ))}
      </section>

      {questions.length > 0 && (
        <section style={{ ...card, marginBottom: space.lg }}>
          <h2 style={sectionTitle}>
            Generated questions · {questions.length}
          </h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1fr) auto",
              gap: space.xl,
              alignItems: "start",
            }}
          >
            <div style={{ minWidth: 0 }}>
              <div style={tableShell}>
                <table style={tableStyles.table}>
                  <thead>
                    <tr>
                      <th style={tableStyles.th}>Question</th>
                      <th style={{ ...tableStyles.th, width: 120 }}>Type</th>
                      <th style={{ ...tableStyles.th, width: 88 }}>Difficulty</th>
                    </tr>
                  </thead>
                  <tbody>
                    {questions.slice(0, 50).map((q, i) => (
                      <tr key={i}>
                        <td style={tableStyles.td}>{q.question}</td>
                        <td style={{ ...tableStyles.td, color: colors.textMuted }}>
                          {q.question_type}
                        </td>
                        <td style={tableStyles.td}>
                          <span style={difficultyChip(q.difficulty)}>
                            {q.difficulty}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {questions.length > 50 && (
                <div style={{ fontSize: 11, color: colors.textFaint, marginTop: space.xs }}>
                  Showing first 50 of {questions.length}.
                </div>
              )}
            </div>

            <Heatmap
              matrix={difficultyMatrix.matrix}
              rowLabels={difficultyMatrix.rowLabels}
              colLabels={difficultyMatrix.colLabels}
              yAxisLabel="reasoning depth"
              xAxisLabel="semantic distance"
              formatValue={(v) => `${v}`}
              title="Difficulty matrix"
            />
          </div>

          <div style={{ display: "flex", gap: space.sm, marginTop: space.lg }}>
            {csvUrl && (
              <a href={csvUrl} download style={ghostButton}>
                ↓ Download CSV
              </a>
            )}
            <Link to="/optimize" style={primaryButton}>
              Run optimization →
            </Link>
          </div>
        </section>
      )}
    </div>
  );
}

function StepCard({ step, index }: { step: StepState; index: number }) {
  const tone: Parameters<typeof chip>[0] =
    step.status === "done"
      ? "success"
      : step.status === "running"
      ? "accent"
      : step.status === "failed"
      ? "danger"
      : "neutral";

  return (
    <div style={stepCardStyle(step.status)}>
      <div style={stepHead}>
        <span style={stepIndex}>{String(index + 1).padStart(2, "0")}</span>
        <StatusDot status={step.status} />
      </div>
      <div style={stepTitle}>{step.title}</div>
      <div style={stepSubtitle}>{step.subtitle}</div>
      <div style={{ marginTop: space.sm, fontSize: 11, color: colors.textMuted }}>
        <span style={chip(tone)}>
          {step.status === "running" ? "running" : step.status}
        </span>
      </div>
      {Object.keys(step.stats).length > 0 && (
        <ul style={statsList}>
          {Object.entries(step.stats).map(([k, v]) => (
            <li key={k} style={statRow}>
              <span style={{ color: colors.textFaint }}>{k}</span>
              <span style={{ fontFamily: font.mono, color: colors.text }}>
                {typeof v === "number" ? v : String(v)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function StatusDot({ status }: { status: StepStatus }) {
  const color =
    status === "done"
      ? colors.success
      : status === "running"
      ? colors.accent
      : status === "failed"
      ? colors.danger
      : colors.bgHover;
  return (
    <span
      style={{
        width: 8,
        height: 8,
        borderRadius: 4,
        background: color,
        animation:
          status === "running" ? "rag-pulse 1.4s ease-in-out infinite" : undefined,
      }}
    />
  );
}

const dropZone: CSSProperties = {
  border: "1px dashed",
  borderRadius: radius.md,
  padding: `${space.xl}px ${space.lg}px`,
  textAlign: "center",
  transition: "background 120ms, border-color 120ms",
};

const docList: CSSProperties = {
  listStyle: "none",
  padding: 0,
  marginTop: space.md,
  marginBottom: 0,
};

const docRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.md,
  padding: `${space.xs + 2}px ${space.sm}px`,
  borderRadius: radius.sm,
  fontSize: 13,
};

const removeBtn: CSSProperties = {
  background: "transparent",
  border: "none",
  fontSize: 16,
  cursor: "pointer",
  color: colors.textFaint,
  padding: "0 6px",
  lineHeight: 1,
};

const fieldLabel: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  fontSize: 12,
  color: colors.textMuted,
  fontWeight: 500,
  marginBottom: space.xs,
};

const valueHint: CSSProperties = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.textFaint,
};

const distRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.sm,
  padding: "2px 0",
};

const tableShell: CSSProperties = {
  border: `1px solid ${colors.border}`,
  borderRadius: radius.md,
  overflow: "hidden",
};

function difficultyChip(d: "easy" | "medium" | "hard"): CSSProperties {
  const tone = d === "easy" ? "success" : d === "medium" ? "warn" : "danger";
  return chip(tone);
}

const stepCardStyle = (status: StepStatus): CSSProperties => ({
  ...card,
  padding: space.md,
  minHeight: 120,
  display: "flex",
  flexDirection: "column",
  borderColor:
    status === "running"
      ? colors.accent
      : status === "failed"
      ? colors.danger
      : colors.border,
  background: status === "done" ? colors.bgSubtle : colors.bg,
  transition: "border-color 200ms ease, background 200ms ease",
});

const stepHead: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
};

const stepIndex: CSSProperties = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.textFaint,
};

const stepTitle: CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: colors.text,
  marginTop: space.xs,
};

const stepSubtitle: CSSProperties = {
  fontSize: 12,
  color: colors.textMuted,
  marginTop: 2,
};

const statsList: CSSProperties = {
  listStyle: "none",
  padding: 0,
  marginTop: space.sm,
  marginBottom: 0,
  fontSize: 12,
};

const statRow: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  padding: "1px 0",
};
