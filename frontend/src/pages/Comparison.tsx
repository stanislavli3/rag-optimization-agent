/**
 * Comparison — side-by-side view of past runs stored in RunContext.history.
 *
 * Lets the user pick 2+ runs, then shows:
 *   · a RadarChart with one polygon per selected run
 *   · a score + lift table
 *   · a config-diff table that highlights which params differ across runs
 *
 * All data comes from localStorage-backed RunContext — no backend calls.
 */
import { CSSProperties, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import RadarChart, { RadarSeries } from "../components/RadarChart";
import { RunSnapshot, useRun } from "../context/RunContext";
import {
  callout,
  card,
  chip,
  colors,
  font,
  metricLabel,
  pageStyle,
  pageSubtitle,
  pageTitle,
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

const SERIES_COLORS = [
  colors.accent,
  colors.stageExploration,
  colors.warn,
  colors.success,
  colors.danger,
];

function perQuestionScores(
  snapshot: RunSnapshot,
  metric: string,
): { q: string; s: number }[] {
  const per = (snapshot.bestMetrics as any)?.per_question;
  if (!Array.isArray(per)) return [];
  const out: { q: string; s: number }[] = [];
  for (const row of per) {
    const q = typeof row?.question === "string" ? row.question : "";
    const s = row?.scores?.[metric];
    if (q && typeof s === "number") out.push({ q, s });
  }
  return out;
}

function pairedBootstrapStars(
  a: RunSnapshot,
  b: RunSnapshot,
  metric: string,
  nBoot = 2000,
): { stars: string; delta: number; p: number } | null {
  const aRows = perQuestionScores(a, metric);
  const bRows = perQuestionScores(b, metric);
  if (aRows.length === 0 || bRows.length === 0) return null;
  const bMap = new Map(bRows.map((r) => [r.q, r.s]));
  const diffs: number[] = [];
  for (const { q, s } of aRows) {
    const bs = bMap.get(q);
    if (typeof bs === "number") diffs.push(s - bs);
  }
  if (diffs.length < 3) return null;
  const observed = diffs.reduce((x, y) => x + y, 0) / diffs.length;

  // Deterministic LCG so results are stable across renders.
  let seed = 42;
  const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 0xffffffff);

  let extreme = 0;
  for (let i = 0; i < nBoot; i++) {
    let sum = 0;
    for (let j = 0; j < diffs.length; j++) {
      sum += diffs[Math.floor(rand() * diffs.length)];
    }
    const m = sum / diffs.length;
    if (Math.abs(m - observed) >= Math.abs(observed)) extreme += 1;
  }
  const p = extreme / nBoot;
  const stars = p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "";
  return { stars, delta: observed, p };
}

export default function Comparison() {
  const { history, current } = useRun();

  const allRuns = useMemo(() => {
    const list: RunSnapshot[] = [];
    if (current) list.push(current);
    for (const r of history) {
      if (!current || r.experimentId !== current.experimentId) list.push(r);
    }
    return list;
  }, [history, current]);

  const [selected, setSelected] = useState<string[]>(() =>
    allRuns.slice(0, 2).map((r) => r.experimentId),
  );

  const picked = useMemo(
    () =>
      selected
        .map((id) => allRuns.find((r) => r.experimentId === id))
        .filter((r): r is RunSnapshot => Boolean(r)),
    [selected, allRuns],
  );

  if (allRuns.length === 0) {
    return (
      <div style={pageStyle}>
        <h1 style={pageTitle}>Comparison</h1>
        <p style={pageSubtitle}>
          No runs on file yet. Complete at least two runs to compare them here.
        </p>
        <div style={callout("neutral")}>
          <Link to="/optimize" style={{ color: colors.accent }}>
            → Go to Auto-Optimize
          </Link>
        </div>
      </div>
    );
  }

  const toggle = (id: string) =>
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );

  const metricsPresent = METRIC_ORDER.filter((m) =>
    picked.some((r) => typeof r.bestMetrics[m] === "number"),
  );

  const radarSeries: RadarSeries[] = picked.map((r, i) => ({
    label: shortLabel(r),
    values: r.bestMetrics,
    color: SERIES_COLORS[i % SERIES_COLORS.length],
  }));

  const configKeys = useMemo(() => {
    const set = new Set<string>();
    for (const r of picked) Object.keys(r.bestConfig).forEach((k) => set.add(k));
    return [...set].sort();
  }, [picked]);

  return (
    <div style={pageStyle}>
      <h1 style={pageTitle}>Comparison</h1>
      <p style={pageSubtitle}>
        Stack up to {Math.min(allRuns.length, SERIES_COLORS.length)} runs
        side-by-side. Radar overlays metric profiles; the config table
        highlights rows where at least one run differs.
      </p>

      <section style={{ ...card, marginBottom: space.lg }}>
        <h2 style={sectionTitle}>Pick runs</h2>
        <div style={pickerGrid}>
          {allRuns.map((r, i) => {
            const isOn = selected.includes(r.experimentId);
            const colorIdx = selected.indexOf(r.experimentId);
            const swatch =
              colorIdx >= 0 ? SERIES_COLORS[colorIdx % SERIES_COLORS.length] : "transparent";
            return (
              <button
                key={r.experimentId}
                onClick={() => toggle(r.experimentId)}
                style={runChip(isOn)}
              >
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: 5,
                    background: swatch,
                    border: `1px solid ${isOn ? swatch : colors.border}`,
                    display: "inline-block",
                  }}
                />
                <span style={{ flex: 1, textAlign: "left" }}>
                  {shortLabel(r)}
                  {i === 0 && current?.experimentId === r.experimentId && (
                    <span style={{ marginLeft: space.xs, ...chip("accent") }}>
                      latest
                    </span>
                  )}
                </span>
                <span style={{ fontFamily: font.mono, color: colors.textMuted }}>
                  {r.bestScore !== null ? r.bestScore.toFixed(3) : "—"}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      {picked.length === 0 ? (
        <div style={callout("neutral")}>
          Select at least one run above to view a comparison.
        </div>
      ) : (
        <>
          <section style={{ ...card, marginBottom: space.lg }}>
            <h2 style={sectionTitle}>Score & lift</h2>
            <table style={tableStyles.table}>
              <thead>
                <tr>
                  <th style={tableStyles.th}>Run</th>
                  <th style={tableStyles.th}>Best</th>
                  <th style={tableStyles.th}>Baseline</th>
                  <th style={tableStyles.th}>Lift</th>
                  <th style={tableStyles.th}>Iterations</th>
                  <th style={tableStyles.th}>Completed</th>
                </tr>
              </thead>
              <tbody>
                {picked.map((r, i) => {
                  const lift =
                    r.bestScore !== null && r.baselineScore !== null
                      ? r.bestScore - r.baselineScore
                      : null;
                  return (
                    <tr key={r.experimentId}>
                      <td style={tableStyles.td}>
                        <span
                          style={{
                            display: "inline-block",
                            width: 8,
                            height: 8,
                            borderRadius: 4,
                            background: SERIES_COLORS[i % SERIES_COLORS.length],
                            marginRight: space.xs,
                          }}
                        />
                        {shortLabel(r)}
                      </td>
                      <td style={{ ...tableStyles.td, fontFamily: font.mono }}>
                        {r.bestScore !== null ? r.bestScore.toFixed(3) : "—"}
                      </td>
                      <td style={{ ...tableStyles.td, fontFamily: font.mono }}>
                        {r.baselineScore !== null ? r.baselineScore.toFixed(3) : "—"}
                      </td>
                      <td
                        style={{
                          ...tableStyles.td,
                          fontFamily: font.mono,
                          color:
                            lift === null
                              ? colors.textFaint
                              : lift >= 0
                              ? colors.success
                              : colors.danger,
                        }}
                      >
                        {lift === null ? "—" : `${lift >= 0 ? "+" : ""}${lift.toFixed(3)}`}
                      </td>
                      <td style={{ ...tableStyles.td, fontFamily: font.mono }}>
                        {r.trajectory.length}
                      </td>
                      <td style={{ ...tableStyles.td, color: colors.textMuted }}>
                        {new Date(r.completedAt).toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>

          {metricsPresent.length >= 3 && (
            <section style={{ ...card, marginBottom: space.lg }}>
              <h2 style={sectionTitle}>Metric overlay</h2>
              <div style={{ display: "flex", justifyContent: "center" }}>
                <RadarChart
                  metrics={metricsPresent}
                  labelMap={metricLabel}
                  series={radarSeries}
                  size={340}
                />
              </div>
            </section>
          )}

          {metricsPresent.length > 0 && (
            <section style={{ ...card, marginBottom: space.lg }}>
              <h2 style={sectionTitle}>Per-metric</h2>
              <table style={tableStyles.table}>
                <thead>
                  <tr>
                    <th style={tableStyles.th}>Metric</th>
                    {picked.map((r, i) => (
                      <th key={r.experimentId} style={tableStyles.th}>
                        <span
                          style={{
                            display: "inline-block",
                            width: 8,
                            height: 8,
                            borderRadius: 4,
                            background: SERIES_COLORS[i % SERIES_COLORS.length],
                            marginRight: space.xs,
                          }}
                        />
                        {shortLabel(r)}
                      </th>
                    ))}
                    {picked.length === 2 && (
                      <th style={tableStyles.th}>Δ · sig</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {metricsPresent.map((m) => {
                    const values = picked.map((r) => r.bestMetrics[m]);
                    const valid = values.filter(
                      (v): v is number => typeof v === "number",
                    );
                    const best = valid.length > 0 ? Math.max(...valid) : null;
                    const stat =
                      picked.length === 2
                        ? pairedBootstrapStars(picked[0], picked[1], m)
                        : null;
                    return (
                      <tr key={m}>
                        <td style={{ ...tableStyles.td, color: colors.textMuted }}>
                          {metricLabel[m] ?? m}
                        </td>
                        {values.map((v, i) => (
                          <td
                            key={i}
                            style={{
                              ...tableStyles.td,
                              fontFamily: font.mono,
                              fontWeight: v === best ? 600 : 400,
                              color:
                                typeof v !== "number"
                                  ? colors.textFaint
                                  : v === best
                                  ? colors.success
                                  : colors.text,
                            }}
                          >
                            {typeof v === "number" ? v.toFixed(3) : "—"}
                          </td>
                        ))}
                        {picked.length === 2 && (
                          <td
                            style={{
                              ...tableStyles.td,
                              fontFamily: font.mono,
                              color: stat
                                ? stat.stars
                                  ? colors.success
                                  : colors.textMuted
                                : colors.textFaint,
                            }}
                            title={
                              stat
                                ? `paired bootstrap p=${stat.p.toFixed(3)}`
                                : "per-question scores missing"
                            }
                          >
                            {stat
                              ? `${stat.delta >= 0 ? "+" : ""}${stat.delta.toFixed(3)} ${stat.stars || "ns"}`
                              : "—"}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {picked.length === 2 && (
                <div
                  style={{
                    fontSize: 12,
                    color: colors.textMuted,
                    marginTop: space.sm,
                  }}
                >
                  Δ is (run A − run B) paired bootstrap over matching questions
                  · *** p&lt;.001 · ** p&lt;.01 · * p&lt;.05 · ns not significant.
                </div>
              )}
            </section>
          )}

          {configKeys.length > 0 && (
            <section style={{ ...card, marginBottom: space.lg }}>
              <h2 style={sectionTitle}>Config diff</h2>
              <table style={tableStyles.table}>
                <thead>
                  <tr>
                    <th style={tableStyles.th}>Parameter</th>
                    {picked.map((r, i) => (
                      <th key={r.experimentId} style={tableStyles.th}>
                        <span
                          style={{
                            display: "inline-block",
                            width: 8,
                            height: 8,
                            borderRadius: 4,
                            background: SERIES_COLORS[i % SERIES_COLORS.length],
                            marginRight: space.xs,
                          }}
                        />
                        {shortLabel(r)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {configKeys.map((k) => {
                    const values = picked.map((r) => r.bestConfig[k]);
                    const allSame = values.every(
                      (v) => JSON.stringify(v) === JSON.stringify(values[0]),
                    );
                    return (
                      <tr key={k}>
                        <td
                          style={{
                            ...tableStyles.td,
                            color: colors.textMuted,
                            width: "22%",
                          }}
                        >
                          {k}
                        </td>
                        {values.map((v, i) => (
                          <td
                            key={i}
                            style={{
                              ...tableStyles.td,
                              fontFamily: font.mono,
                              fontSize: 12,
                              background: allSame
                                ? "transparent"
                                : colors.warnSoft,
                              color: allSame ? colors.text : colors.warn,
                            }}
                          >
                            {formatVal(v)}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div
                style={{
                  fontSize: 12,
                  color: colors.textMuted,
                  marginTop: space.sm,
                }}
              >
                Amber rows mark parameters where at least one run differs.
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}

function shortLabel(r: RunSnapshot): string {
  if (r.label.length < 30) return r.label;
  return r.label.slice(0, 28) + "…";
}

function formatVal(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (typeof v === "number") return Number.isInteger(v) ? `${v}` : v.toFixed(3);
  if (typeof v === "boolean") return v ? "true" : "false";
  return JSON.stringify(v);
}

const pickerGrid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
  gap: space.sm,
};

const runChip = (on: boolean): CSSProperties => ({
  display: "flex",
  alignItems: "center",
  gap: space.sm,
  padding: `${space.sm}px ${space.md}px`,
  border: `1px solid ${on ? colors.accent : colors.border}`,
  background: on ? colors.accentSoft : colors.bgSubtle,
  color: colors.text,
  borderRadius: radius.md,
  fontSize: 13,
  cursor: "pointer",
  textAlign: "left",
});
