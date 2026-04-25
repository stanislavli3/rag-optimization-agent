/**
 * ConfigLab — single-config evaluator.
 *
 * Lets the user hand-pick one RAG configuration and run it through the same
 * evaluation path the optimizer uses. Under the hood this is just an
 * experiment with strategy="grid", max_iterations=1 and a search space where
 * every dimension is pinned to a single value — so the agent produces exactly
 * one IterationResult plus the usual baseline.
 *
 * Useful for: (a) sanity-checking a handcrafted config, (b) reproducing the
 * winner from a past run, (c) A/B-ing two configs without running the full
 * optimizer.
 */
import { CSSProperties, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import TrajectoryGraph from "../components/TrajectoryGraph";
import { useRun } from "../context/RunContext";
import useAgentStream from "../hooks/useAgentStream";
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

interface DimensionSpec {
  key: string;
  label: string;
  kind: "select" | "number";
  options?: (string | number)[];
  min?: number;
  max?: number;
  step?: number;
  default: string | number;
}

const DIMENSIONS: DimensionSpec[] = [
  {
    key: "chunk_size",
    label: "Chunk size (tokens)",
    kind: "select",
    options: [128, 256, 512, 1024],
    default: 512,
  },
  {
    key: "chunk_overlap",
    label: "Chunk overlap (ratio)",
    kind: "select",
    options: [0, 0.1, 0.2, 0.3],
    default: 0.1,
  },
  {
    key: "top_k",
    label: "Top-k retrieval",
    kind: "select",
    options: [3, 5, 10, 20],
    default: 5,
  },
  {
    key: "search_mode",
    label: "Search mode",
    kind: "select",
    options: ["dense", "sparse", "hybrid"],
    default: "dense",
  },
  {
    key: "reranker",
    label: "Reranker",
    kind: "select",
    options: ["none", "cross_encoder", "cohere"],
    default: "none",
  },
  {
    key: "prompt_style",
    label: "Prompt style",
    kind: "select",
    options: ["basic", "cot", "few_shot"],
    default: "basic",
  },
  {
    key: "embedding_model",
    label: "Embedding model",
    kind: "select",
    options: ["all-MiniLM-L6-v2", "bge-small-en", "bge-large-en-v1.5"],
    default: "all-MiniLM-L6-v2",
  },
];

function makeDefaults(): Record<string, string | number> {
  const out: Record<string, string | number> = {};
  DIMENSIONS.forEach((d) => (out[d.key] = d.default));
  return out;
}

async function runSingleConfig(
  config: Record<string, string | number>,
): Promise<string> {
  const searchSpace: Record<string, (string | number)[]> = {};
  Object.entries(config).forEach(([k, v]) => (searchSpace[k] = [v]));

  const body = {
    name: `ConfigLab · ${new Date().toLocaleTimeString()}`,
    strategy: "grid",
    search_space: searchSpace,
    max_iterations: 1,
    stopping_mode: "fixed",
    baseline_run_enabled: true,
  };
  const res = await fetch("/api/experiments/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`failed to create experiment (${res.status})`);
  const exp = await res.json();
  await fetch(`/api/experiments/${exp.id}/start/`, { method: "POST" });
  return exp.id as string;
}

export default function ConfigLab() {
  const { current } = useRun();
  const [config, setConfig] = useState<Record<string, string | number>>(
    makeDefaults,
  );
  const [experimentId, setExperimentId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { events, status } = useAgentStream(experimentId);

  const result = useMemo(() => deriveResult(events), [events]);

  const update = (k: string, v: string | number) =>
    setConfig((c) => ({ ...c, [k]: v }));

  const clone = () => {
    if (!current) return;
    const next: Record<string, string | number> = { ...config };
    DIMENSIONS.forEach((d) => {
      const v = current.bestConfig[d.key];
      if (typeof v === "string" || typeof v === "number") next[d.key] = v;
    });
    setConfig(next);
  };

  const reset = () => {
    setConfig(makeDefaults());
    setExperimentId(null);
    setErr(null);
  };

  const onRun = async () => {
    setErr(null);
    setExperimentId(null);
    setSubmitting(true);
    try {
      const id = await runSingleConfig(config);
      setExperimentId(id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={pageStyle}>
      <h1 style={pageTitle}>Config Lab</h1>
      <p style={pageSubtitle}>
        Hand-pick one RAG configuration and score it. Skips the search entirely
        — you get the metrics for exactly the config below, plus a naive
        baseline for comparison.
      </p>

      <section style={twoCol}>
        <div style={card}>
          <div style={rowBetween}>
            <h2 style={{ ...sectionTitle, margin: 0 }}>Configuration</h2>
            <div style={{ display: "flex", gap: space.xs }}>
              <button
                style={ghostButton}
                onClick={clone}
                disabled={!current}
                title={
                  current
                    ? "Copy the best config from the current run"
                    : "No current run to clone from"
                }
              >
                Clone latest
              </button>
              <button style={ghostButton} onClick={reset}>
                Reset
              </button>
            </div>
          </div>

          <div style={{ marginTop: space.md, display: "grid", gap: space.sm }}>
            {DIMENSIONS.map((d) => (
              <DimRow
                key={d.key}
                dim={d}
                value={config[d.key]}
                onChange={(v) => update(d.key, v)}
              />
            ))}
          </div>

          <div
            style={{
              marginTop: space.lg,
              display: "flex",
              gap: space.sm,
              alignItems: "center",
            }}
          >
            <button
              style={{
                ...primaryButton,
                opacity: submitting ? 0.6 : 1,
                cursor: submitting ? "wait" : "pointer",
              }}
              onClick={onRun}
              disabled={submitting}
            >
              {submitting ? "Starting…" : "Evaluate this config"}
            </button>
            {experimentId && (
              <span style={chip(status === "open" ? "accent" : "neutral")}>
                {status === "open" ? "streaming" : status}
              </span>
            )}
          </div>

          {err && (
            <div style={{ marginTop: space.md, ...callout("danger") }}>
              {err}
            </div>
          )}
          {!current && !experimentId && (
            <div style={{ marginTop: space.md, ...callout("neutral") }}>
              Tip: run the full optimizer first (
              <Link to="/optimize" style={{ color: colors.accent }}>
                Auto-Optimize
              </Link>
              ) — then "Clone latest" brings the winner here for tweaking.
            </div>
          )}
        </div>

        <div style={card}>
          <h2 style={sectionTitle}>Live result</h2>
          {!experimentId ? (
            <div style={{ color: colors.textFaint, fontSize: 13 }}>
              Results will appear here after you press Evaluate.
            </div>
          ) : (
            <ResultPanel
              score={result.score}
              baseline={result.baseline}
              metrics={result.metrics}
              done={result.done}
              streamOpen={status === "open"}
            />
          )}
          {experimentId && result.trajectory.length > 0 && (
            <div style={{ marginTop: space.md }}>
              <TrajectoryGraph
                points={result.trajectory}
                baselineScore={result.baseline}
                width={420}
                height={200}
                title={false}
              />
            </div>
          )}
        </div>
      </section>

      {current && (
        <section style={{ ...card, marginTop: space.lg }}>
          <h2 style={sectionTitle}>Latest optimizer winner (reference)</h2>
          <table style={tableStyles.table}>
            <tbody>
              {DIMENSIONS.map((d) => {
                const theirs = current.bestConfig[d.key];
                const mine = config[d.key];
                const same = JSON.stringify(theirs) === JSON.stringify(mine);
                return (
                  <tr key={d.key}>
                    <td
                      style={{
                        ...tableStyles.td,
                        color: colors.textMuted,
                        width: "35%",
                      }}
                    >
                      {d.label}
                    </td>
                    <td
                      style={{
                        ...tableStyles.td,
                        fontFamily: font.mono,
                        fontSize: 12,
                      }}
                    >
                      {fmt(theirs)}
                    </td>
                    <td
                      style={{
                        ...tableStyles.td,
                        fontFamily: font.mono,
                        fontSize: 12,
                        color: same ? colors.textMuted : colors.warn,
                      }}
                    >
                      {same ? "=" : "→"} {fmt(mine)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

interface DerivedResult {
  score: number | null;
  baseline: number | null;
  metrics: Record<string, number>;
  trajectory: import("../components/TrajectoryGraph").TrajectoryPoint[];
  done: boolean;
}

function deriveResult(events: ReturnType<typeof useAgentStream>["events"]): DerivedResult {
  const traj: import("../components/TrajectoryGraph").TrajectoryPoint[] = [];
  let score: number | null = null;
  let baseline: number | null = null;
  let metrics: Record<string, number> = {};
  let done = false;

  for (const e of events) {
    const d = (e.data ?? {}) as Record<string, unknown>;
    if (e.event === "node_success") {
      const s = typeof d.score === "number" ? (d.score as number) : null;
      if (s !== null) {
        if (d.is_baseline === true && baseline === null) baseline = s;
        else score = s;
        traj.push({
          iteration: traj.length,
          score: s,
          stage: (d.stage as never) ?? "baseline",
          status: "success",
        });
        if (d.metrics && typeof d.metrics === "object") {
          metrics = { ...(d.metrics as Record<string, number>) };
        }
      }
    } else if (e.event === "complete") {
      done = true;
      if (typeof d.best_score === "number") score = d.best_score as number;
      if (d.best_metrics && typeof d.best_metrics === "object") {
        metrics = { ...(d.best_metrics as Record<string, number>) };
      }
    }
  }

  return { score, baseline, metrics, trajectory: traj, done };
}

function ResultPanel({
  score,
  baseline,
  metrics,
  done,
  streamOpen,
}: {
  score: number | null;
  baseline: number | null;
  metrics: Record<string, number>;
  done: boolean;
  streamOpen: boolean;
}) {
  const lift =
    score !== null && baseline !== null ? score - baseline : null;
  return (
    <div>
      <div style={statGrid}>
        <Stat
          label="Your config"
          value={score !== null ? score.toFixed(3) : "—"}
          tone="success"
        />
        <Stat
          label="Baseline"
          value={baseline !== null ? baseline.toFixed(3) : "—"}
        />
        <Stat
          label="Lift"
          value={
            lift === null
              ? "—"
              : `${lift >= 0 ? "+" : ""}${lift.toFixed(3)}`
          }
          tone={lift === null ? undefined : lift >= 0 ? "success" : "danger"}
        />
        <Stat label="Status" value={done ? "Done" : streamOpen ? "Running" : "…"} />
      </div>

      {Object.keys(metrics).length > 0 && (
        <table style={{ ...tableStyles.table, marginTop: space.md }}>
          <tbody>
            {Object.entries(metrics).map(([k, v]) => (
              <tr key={k}>
                <td
                  style={{
                    ...tableStyles.td,
                    color: colors.textMuted,
                    width: "60%",
                  }}
                >
                  {metricLabel[k] ?? k}
                </td>
                <td style={{ ...tableStyles.td, fontFamily: font.mono }}>
                  {typeof v === "number" ? v.toFixed(3) : String(v)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "success" | "danger";
}) {
  const color =
    tone === "success" ? colors.success : tone === "danger" ? colors.danger : colors.text;
  return (
    <div style={statCell}>
      <div style={statLabel}>{label}</div>
      <div style={{ ...statValue, color }}>{value}</div>
    </div>
  );
}

function DimRow({
  dim,
  value,
  onChange,
}: {
  dim: DimensionSpec;
  value: string | number;
  onChange: (v: string | number) => void;
}) {
  return (
    <label style={dimRow}>
      <span style={dimLabel}>{dim.label}</span>
      {dim.kind === "select" ? (
        <select
          value={String(value)}
          onChange={(e) => {
            const raw = e.target.value;
            const asNum = Number(raw);
            onChange(Number.isFinite(asNum) && raw !== "" && !isNaN(asNum) ? asNum : raw);
          }}
          style={selectStyle}
        >
          {(dim.options ?? []).map((o) => (
            <option key={String(o)} value={String(o)}>
              {String(o)}
            </option>
          ))}
        </select>
      ) : (
        <input
          type="number"
          value={value as number}
          min={dim.min}
          max={dim.max}
          step={dim.step}
          onChange={(e) => onChange(Number(e.target.value))}
          style={inputStyle}
        />
      )}
    </label>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (typeof v === "number") return Number.isInteger(v) ? `${v}` : v.toFixed(3);
  if (typeof v === "boolean") return v ? "true" : "false";
  return JSON.stringify(v);
}

const twoCol: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: space.lg,
};

const rowBetween: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
};

const dimRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.sm,
  justifyContent: "space-between",
};

const dimLabel: CSSProperties = {
  fontSize: 13,
  color: colors.textMuted,
};

const selectStyle: CSSProperties = {
  minWidth: 160,
  padding: `${space.xs}px ${space.sm}px`,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.sm,
  background: colors.bg,
  color: colors.text,
  fontSize: 13,
  fontFamily: font.mono,
};

const inputStyle: CSSProperties = {
  width: 120,
  padding: `${space.xs}px ${space.sm}px`,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.sm,
  fontSize: 13,
  fontFamily: font.mono,
};

const statGrid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(4, 1fr)",
  gap: space.sm,
};

const statCell: CSSProperties = {
  padding: `${space.sm}px ${space.md}px`,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.md,
  background: colors.bgSubtle,
};

const statLabel: CSSProperties = {
  fontSize: 10,
  fontWeight: 600,
  letterSpacing: 0.3,
  textTransform: "uppercase",
  color: colors.textFaint,
  marginBottom: 2,
};

const statValue: CSSProperties = {
  fontSize: 16,
  fontWeight: 600,
  fontFamily: font.mono,
};
