/**
 * Trajectory — full-width view of the optimizer's score-over-iteration curve.
 *
 * Reuses the SVG TrajectoryGraph component but at a wider size, and adds a
 * stage-level breakdown underneath so the user can see how each of the 4
 * BFTS phases contributed. Reads from RunContext, so it works offline after
 * a run completes.
 */
import { CSSProperties, useMemo } from "react";
import { Link } from "react-router-dom";

import TrajectoryGraph, { TrajectoryPoint, TrajectoryStage } from "../components/TrajectoryGraph";
import { useRun } from "../context/RunContext";
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
  sectionTitle,
  space,
  stageChipTone,
  tableStyles,
} from "../theme";

const STAGES: TrajectoryStage[] = [
  "preliminary",
  "baseline",
  "exploration",
  "ablation",
];

interface StageRow {
  stage: TrajectoryStage;
  count: number;
  best: number | null;
  mean: number | null;
  first: number | null;
  last: number | null;
  delta: number | null;
}

function summariseByStage(points: TrajectoryPoint[]): StageRow[] {
  return STAGES.map((stage) => {
    const slice = points.filter((p) => p.stage === stage);
    if (slice.length === 0) {
      return { stage, count: 0, best: null, mean: null, first: null, last: null, delta: null };
    }
    const scores = slice.map((p) => p.score);
    const best = Math.max(...scores);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const first = scores[0];
    const last = scores[scores.length - 1];
    return {
      stage,
      count: slice.length,
      best,
      mean,
      first,
      last,
      delta: last - first,
    };
  });
}

export default function Trajectory() {
  const { current, history } = useRun();

  const stageRows = useMemo(
    () => (current ? summariseByStage(current.trajectory) : []),
    [current],
  );

  if (!current) {
    return (
      <div style={pageStyle}>
        <h1 style={pageTitle}>Trajectory</h1>
        <p style={pageSubtitle}>
          No run yet. Once the optimizer finishes, the full score trajectory
          lands here.
        </p>
        <div style={callout("neutral")}>
          <Link to="/optimize" style={{ color: colors.accent }}>
            → Go to Auto-Optimize
          </Link>
        </div>
      </div>
    );
  }

  const { trajectory, baselineScore, bestScore } = current;
  const otherRuns = history.filter((r) => r.experimentId !== current.experimentId);

  return (
    <div style={pageStyle}>
      <h1 style={pageTitle}>Trajectory</h1>
      <p style={pageSubtitle}>
        Every iteration, plotted. The amber envelope is the best-so-far score;
        the charcoal line is the raw score. Background bands mark BFTS stages.
      </p>

      <section style={{ ...card, marginBottom: space.lg }}>
        <TrajectoryGraph
          points={trajectory}
          baselineScore={baselineScore}
          width={960}
          height={360}
          title={current.label}
        />
      </section>

      <section style={{ ...card, marginBottom: space.lg }}>
        <h2 style={sectionTitle}>Stage breakdown</h2>
        <table style={tableStyles.table}>
          <thead>
            <tr>
              <th style={tableStyles.th}>Stage</th>
              <th style={tableStyles.th}>Iterations</th>
              <th style={tableStyles.th}>Best</th>
              <th style={tableStyles.th}>Mean</th>
              <th style={tableStyles.th}>Δ within</th>
            </tr>
          </thead>
          <tbody>
            {stageRows.map((row) => (
              <tr key={row.stage}>
                <td style={tableStyles.td}>
                  <span style={chip(stageChipTone(row.stage))}>{row.stage}</span>
                </td>
                <td style={{ ...tableStyles.td, fontFamily: font.mono }}>
                  {row.count}
                </td>
                <td style={{ ...tableStyles.td, fontFamily: font.mono }}>
                  {row.best !== null ? row.best.toFixed(3) : "—"}
                </td>
                <td style={{ ...tableStyles.td, fontFamily: font.mono }}>
                  {row.mean !== null ? row.mean.toFixed(3) : "—"}
                </td>
                <td
                  style={{
                    ...tableStyles.td,
                    fontFamily: font.mono,
                    color:
                      row.delta === null
                        ? colors.textFaint
                        : row.delta >= 0
                        ? colors.success
                        : colors.danger,
                  }}
                >
                  {row.delta === null
                    ? "—"
                    : `${row.delta >= 0 ? "+" : ""}${row.delta.toFixed(3)}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section style={{ ...card, marginBottom: space.lg }}>
        <h2 style={sectionTitle}>Summary</h2>
        <div style={summaryGrid}>
          <SummaryStat label="Total iterations" value={String(trajectory.length)} />
          <SummaryStat
            label="Best score"
            value={bestScore !== null ? bestScore.toFixed(3) : "—"}
            accent="success"
          />
          <SummaryStat
            label="Baseline"
            value={baselineScore !== null ? baselineScore.toFixed(3) : "—"}
          />
          <SummaryStat
            label="Lift"
            value={
              bestScore !== null && baselineScore !== null
                ? `${((bestScore - baselineScore) * 1).toFixed(3)}`
                : "—"
            }
          />
        </div>
      </section>

      {otherRuns.length > 0 && (
        <section style={{ ...card, marginBottom: space.lg }}>
          <h2 style={sectionTitle}>Past runs</h2>
          <div style={{ fontSize: 12, color: colors.textMuted, marginBottom: space.sm }}>
            Click a past run on the Comparison page to inspect it side-by-side.
          </div>
          <Link to="/comparison" style={ghostButton}>
            Compare {otherRuns.length + 1} runs →
          </Link>
        </section>
      )}
    </div>
  );
}

function SummaryStat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "success";
}) {
  return (
    <div style={summaryCell}>
      <div style={summaryLabel}>{label}</div>
      <div
        style={{
          ...summaryValue,
          color: accent === "success" ? colors.success : colors.text,
        }}
      >
        {value}
      </div>
    </div>
  );
}

const summaryGrid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(4, 1fr)",
  gap: space.md,
};

const summaryCell: CSSProperties = {
  padding: `${space.sm}px ${space.md}px`,
  background: colors.bgSubtle,
  border: `1px solid ${colors.border}`,
  borderRadius: 4,
};

const summaryLabel: CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: 0.3,
  textTransform: "uppercase",
  color: colors.textFaint,
  marginBottom: space.xxs,
};

const summaryValue: CSSProperties = {
  fontSize: 18,
  fontWeight: 600,
  fontFamily: font.mono,
};
