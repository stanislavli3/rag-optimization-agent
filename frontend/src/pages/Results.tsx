/**
 * Results — post-run dashboard.
 *
 * Reads from RunContext (no API calls): the best config the agent landed on,
 * its RAGAS metrics, a radar, the full score trajectory, ablation waterfall,
 * and the search tree. Falls back to an empty state prompting the user to
 * run the optimizer if no run is present.
 */
import { CSSProperties } from "react";
import { Link } from "react-router-dom";

import AblationWaterfall from "../components/AblationWaterfall";
import AgentTree from "../components/AgentTree";
import ClaudeCodePrompt from "../components/ClaudeCodePrompt";
import RadarChart from "../components/RadarChart";
import TrajectoryGraph from "../components/TrajectoryGraph";
import { useRun } from "../context/RunContext";
import {
  callout,
  card,
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

const METRIC_ORDER = [
  "faithfulness",
  "answer_relevancy",
  "context_precision",
  "context_recall",
  "answer_correctness",
];

const DIFFICULTY_ORDER = ["easy", "medium", "hard"];
const QTYPE_ORDER = ["simple", "multi_context", "reasoning", "conditional"];

export default function Results() {
  const { current } = useRun();

  if (!current) {
    return (
      <div style={pageStyle}>
        <h1 style={pageTitle}>Results</h1>
        <p style={pageSubtitle}>
          No run yet. Start the optimizer and a dashboard will appear here.
        </p>
        <div style={callout("neutral")}>
          <strong>Nothing to show.</strong>
          <div style={{ marginTop: space.xs, color: colors.textMuted }}>
            Visit{" "}
            <Link to="/optimize" style={{ color: colors.accent }}>
              Auto-Optimize
            </Link>{" "}
            to launch a run. Results are cached locally so you can come back
            here after it completes.
          </div>
        </div>
      </div>
    );
  }

  const {
    bestConfig,
    bestMetrics,
    trajectory,
    ablation,
    tree,
    bestNodeId,
    bestScore,
    baselineScore,
  } = current;

  const metricsToShow = METRIC_ORDER.filter(
    (k) => typeof bestMetrics[k] === "number",
  );
  const radarMetrics =
    metricsToShow.length > 0 ? metricsToShow : Object.keys(bestMetrics);
  const liftPct =
    baselineScore && bestScore
      ? ((bestScore - baselineScore) / baselineScore) * 100
      : null;

  return (
    <div style={pageStyle}>
      <h1 style={pageTitle}>Results</h1>
      <p style={pageSubtitle}>{current.label}</p>

      <section style={heroGrid}>
        <HeroStat
          label="Best RAGAS score"
          value={bestScore !== null ? bestScore.toFixed(3) : "—"}
          detail={
            liftPct !== null
              ? `${liftPct >= 0 ? "+" : ""}${liftPct.toFixed(1)}% vs baseline`
              : undefined
          }
          tone="success"
        />
        <HeroStat
          label="Baseline"
          value={baselineScore !== null ? baselineScore.toFixed(3) : "—"}
          detail="Unoptimized reference"
        />
        <HeroStat
          label="Best node"
          value={bestNodeId ? bestNodeId.slice(0, 8) : "—"}
          detail={`${tree.length} nodes explored`}
          mono
        />
        <HeroStat
          label="Ablation signals"
          value={String(ablation.length)}
          detail={
            ablation.length > 0 ? "tuned parameters" : "matched defaults"
          }
        />
      </section>

      <section style={twoCol}>
        <div style={card}>
          <h2 style={sectionTitle}>Best configuration</h2>
          <table style={tableStyles.table}>
            <tbody>
              {Object.entries(bestConfig).map(([k, v]) => (
                <tr key={k}>
                  <td
                    style={{
                      ...tableStyles.td,
                      width: "40%",
                      color: colors.textMuted,
                    }}
                  >
                    {k}
                  </td>
                  <td
                    style={{
                      ...tableStyles.td,
                      fontFamily: font.mono,
                      fontSize: 12,
                      color: colors.text,
                    }}
                  >
                    {formatConfigValue(v)}
                  </td>
                </tr>
              ))}
              {Object.keys(bestConfig).length === 0 && (
                <tr>
                  <td
                    style={{
                      ...tableStyles.td,
                      color: colors.textFaint,
                    }}
                  >
                    (no config captured)
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div style={card}>
          <h2 style={sectionTitle}>Metrics</h2>
          <table style={tableStyles.table}>
            <tbody>
              {METRIC_ORDER.filter(
                (k) => typeof bestMetrics[k] === "number",
              ).map((k) => (
                <tr key={k}>
                  <td
                    style={{
                      ...tableStyles.td,
                      width: "55%",
                      color: colors.textMuted,
                    }}
                  >
                    {metricLabel[k] ?? k}
                  </td>
                  <td
                    style={{
                      ...tableStyles.td,
                      fontFamily: font.mono,
                      color: colors.text,
                    }}
                  >
                    <MetricBar value={bestMetrics[k]} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {Object.keys(bestMetrics).length === 0 && (
            <div
              style={{
                color: colors.textFaint,
                fontSize: 13,
                padding: space.sm,
              }}
            >
              No metric breakdown captured for this run.
            </div>
          )}
        </div>
      </section>

      {radarMetrics.length >= 3 && (
        <section
          style={{
            display: "flex",
            gap: space.lg,
            alignItems: "start",
            marginBottom: space.lg,
          }}
        >
          <RadarChart
            title="Metric profile"
            metrics={radarMetrics}
            labelMap={metricLabel}
            series={[
              {
                label: "Best",
                values: bestMetrics,
                color: colors.accent,
              },
            ]}
            size={300}
          />
          <div style={{ flex: 1, minWidth: 0 }}>
            <TrajectoryGraph
              points={trajectory}
              baselineScore={baselineScore}
              width={560}
              height={260}
            />
          </div>
        </section>
      )}

      {(() => {
        const byDiff = (bestMetrics as any)?.by_difficulty as
          | Record<string, Record<string, number>>
          | undefined;
        const byQt = (bestMetrics as any)?.by_question_type as
          | Record<string, Record<string, number>>
          | undefined;
        const hasDiff = byDiff && Object.keys(byDiff).some(
          (k) => byDiff[k] && Object.keys(byDiff[k]).length > 0,
        );
        const hasQt = byQt && Object.keys(byQt).length > 0;
        if (!hasDiff && !hasQt) return null;
        return (
          <section style={twoCol}>
            {hasDiff && (
              <div style={card}>
                <h2 style={sectionTitle}>By difficulty</h2>
                <StratifiedTable
                  rows={DIFFICULTY_ORDER.filter(
                    (d) => byDiff![d] && Object.keys(byDiff![d]).length > 0,
                  )}
                  data={byDiff!}
                />
              </div>
            )}
            {hasQt && (
              <div style={card}>
                <h2 style={sectionTitle}>By question type</h2>
                <StratifiedTable
                  rows={QTYPE_ORDER.filter(
                    (q) => byQt![q] && Object.keys(byQt![q]).length > 0,
                  ).concat(
                    Object.keys(byQt!).filter(
                      (q) => !QTYPE_ORDER.includes(q),
                    ),
                  )}
                  data={byQt!}
                />
              </div>
            )}
          </section>
        );
      })()}

      {ablation.length > 0 && bestScore !== null && (
        <section style={{ marginBottom: space.lg }}>
          <AblationWaterfall
            entries={ablation}
            bestScore={bestScore}
            baselineScore={baselineScore ?? 0}
          />
        </section>
      )}

      {tree.length > 0 && (
        <section style={{ ...card, marginBottom: space.lg }}>
          <h2 style={sectionTitle}>Search tree</h2>
          <AgentTree
            nodes={tree}
            bestPath={bestNodeId ? [bestNodeId] : []}
            width={900}
            height={420}
          />
        </section>
      )}

      <ClaudeCodePrompt run={current} />

      <div style={{ display: "flex", gap: space.sm, marginTop: space.lg }}>
        <Link to="/export" style={primaryButton}>
          Export configuration →
        </Link>
        <Link to="/optimize" style={ghostButton}>
          Start another run
        </Link>
      </div>
    </div>
  );
}

function HeroStat({
  label,
  value,
  detail,
  tone,
  mono,
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "success" | "neutral";
  mono?: boolean;
}) {
  return (
    <div style={heroCard}>
      <div style={heroLabel}>{label}</div>
      <div
        style={{
          ...heroValue,
          color: tone === "success" ? colors.success : colors.text,
          fontFamily: mono ? font.mono : font.sans,
        }}
      >
        {value}
      </div>
      {detail && <div style={heroDetail}>{detail}</div>}
    </div>
  );
}

function StratifiedTable({
  rows,
  data,
}: {
  rows: string[];
  data: Record<string, Record<string, number>>;
}) {
  const cols = METRIC_ORDER;
  return (
    <table style={tableStyles.table}>
      <thead>
        <tr>
          <th style={{ ...tableStyles.th, textAlign: "left" }}></th>
          {cols.map((m) => (
            <th
              key={m}
              style={{
                ...tableStyles.th,
                textAlign: "right",
                color: colors.textMuted,
                fontSize: 11,
                padding: `${space.xs}px ${space.xs}px`,
              }}
            >
              {(metricLabel[m] ?? m).split(" ")[0]}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r}>
            <td
              style={{
                ...tableStyles.td,
                color: colors.text,
                textTransform: "capitalize",
                fontWeight: 500,
              }}
            >
              {r.replace(/_/g, " ")}
            </td>
            {cols.map((m) => {
              const v = data[r]?.[m];
              return (
                <td
                  key={m}
                  style={{
                    ...tableStyles.td,
                    fontFamily: font.mono,
                    textAlign: "right",
                    color:
                      typeof v === "number" ? colors.text : colors.textFaint,
                  }}
                >
                  {typeof v === "number" ? v.toFixed(3) : "—"}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MetricBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: space.sm }}>
      <div
        style={{
          flex: 1,
          height: 4,
          background: colors.bgHover,
          borderRadius: radius.sm,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct * 100}%`,
            background: colors.accent,
            transition: "width 300ms ease",
          }}
        />
      </div>
      <span style={{ width: 44, textAlign: "right" }}>{value.toFixed(3)}</span>
    </div>
  );
}

function formatConfigValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (typeof v === "number")
    return Number.isInteger(v) ? `${v}` : v.toFixed(3);
  if (typeof v === "boolean") return v ? "true" : "false";
  return JSON.stringify(v);
}

const heroGrid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(4, 1fr)",
  gap: space.md,
  marginBottom: space.lg,
};

const heroCard: CSSProperties = {
  ...card,
  padding: space.md,
};

const heroLabel: CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: 0.3,
  textTransform: "uppercase",
  color: colors.textFaint,
  marginBottom: space.xs,
};

const heroValue: CSSProperties = {
  fontSize: 24,
  fontWeight: 600,
  lineHeight: 1.2,
};

const heroDetail: CSSProperties = {
  fontSize: 12,
  color: colors.textMuted,
  marginTop: 4,
};

const twoCol: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: space.lg,
  marginBottom: space.lg,
};
