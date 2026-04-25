/**
 * TrajectoryGraph — score-per-iteration line chart with stage bands.
 *
 * Lightweight SVG (no D3/Plotly) so it can update on every SSE tick without
 * jank. Notion-style: faint background bands, a thin charcoal line for raw
 * scores, and a dotted accent "best-so-far" envelope.
 */
import { CSSProperties, useMemo } from "react";

import { card, colors, font, space, stageColor } from "../theme";

export type TrajectoryStage = "preliminary" | "baseline" | "exploration" | "ablation";

export interface TrajectoryPoint {
  iteration: number;
  score: number;
  stage: TrajectoryStage;
  status?: "success" | "failed" | "pruned";
  insight?: string;
}

export interface TrajectoryGraphProps {
  points: TrajectoryPoint[];
  width?: number;
  height?: number;
  baselineScore?: number | null;
  /** Optional heading — set to `false` to hide the card header. */
  title?: string | false;
}

// Translucent versions of the stage palette for the background bands.
const STAGE_FILL: Record<TrajectoryStage, string> = {
  preliminary: "rgba(107, 114, 128, 0.06)",
  baseline: "rgba(35, 131, 226, 0.07)",
  exploration: "rgba(105, 64, 165, 0.07)",
  ablation: "rgba(203, 145, 47, 0.08)",
};

export default function TrajectoryGraph({
  points,
  width = 540,
  height = 260,
  baselineScore = null,
  title = "Score trajectory",
}: TrajectoryGraphProps) {
  const padL = 40;
  const padR = 16;
  const padT = 18;
  const padB = 28;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const { xs, ys, bestCurve, maxX, scoreMin, scoreMax, stageSpans } = useMemo(() => {
    if (points.length === 0) {
      return {
        xs: [] as number[],
        ys: [] as number[],
        bestCurve: [] as number[],
        maxX: 1,
        scoreMin: 0,
        scoreMax: 1,
        stageSpans: [] as Array<{ stage: TrajectoryStage; x0: number; x1: number }>,
      };
    }
    const scores = points.map((p) => p.score);
    const iters = points.map((p) => p.iteration);
    const lo = Math.min(...scores, baselineScore ?? Infinity);
    const hi = Math.max(...scores, baselineScore ?? -Infinity);
    const pad = Math.max(0.02, (hi - lo) * 0.1);
    let best = -Infinity;
    const bc = scores.map((s) => (best = Math.max(best, s)));
    const spans: Array<{ stage: TrajectoryStage; x0: number; x1: number }> = [];
    let cur = points[0].stage;
    let x0 = points[0].iteration;
    for (let i = 1; i < points.length; i++) {
      if (points[i].stage !== cur) {
        spans.push({ stage: cur, x0, x1: points[i].iteration });
        cur = points[i].stage;
        x0 = points[i].iteration;
      }
    }
    spans.push({ stage: cur, x0, x1: points[points.length - 1].iteration + 1 });

    return {
      xs: iters,
      ys: scores,
      bestCurve: bc,
      maxX: Math.max(...iters, 1),
      scoreMin: Math.max(0, lo - pad),
      scoreMax: Math.min(1, hi + pad),
      stageSpans: spans,
    };
  }, [points, baselineScore]);

  const sx = (i: number) => padL + (maxX === 0 ? 0 : (i / maxX) * plotW);
  const sy = (s: number) =>
    padT + plotH - ((s - scoreMin) / (scoreMax - scoreMin || 1)) * plotH;

  const linePath = xs
    .map((x, i) => `${i === 0 ? "M" : "L"}${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`)
    .join(" ");
  const bestPath = xs
    .map(
      (x, i) =>
        `${i === 0 ? "M" : "L"}${sx(x).toFixed(1)},${sy(bestCurve[i]).toFixed(1)}`,
    )
    .join(" ");

  const yTicks = [scoreMin, (scoreMin + scoreMax) / 2, scoreMax];

  return (
    <div style={wrap}>
      {title !== false && (
        <div style={header}>
          <span>{title}</span>
          <span style={meta}>
            {points.length} iteration{points.length === 1 ? "" : "s"}
          </span>
        </div>
      )}
      <svg width={width} height={height} style={{ display: "block" }}>
        {stageSpans.map((band, i) => (
          <rect
            key={i}
            x={sx(band.x0)}
            y={padT}
            width={Math.max(1, sx(band.x1) - sx(band.x0))}
            height={plotH}
            fill={STAGE_FILL[band.stage]}
          />
        ))}

        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={padL}
              x2={padL + plotW}
              y1={sy(t)}
              y2={sy(t)}
              stroke={colors.border}
            />
            <text
              x={padL - 8}
              y={sy(t) + 3}
              fontSize={10}
              fontFamily={font.mono}
              textAnchor="end"
              fill={colors.textFaint}
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        {baselineScore !== null && baselineScore !== undefined && (
          <>
            <line
              x1={padL}
              x2={padL + plotW}
              y1={sy(baselineScore)}
              y2={sy(baselineScore)}
              stroke={colors.textFaint}
              strokeDasharray="3 3"
            />
            <text
              x={padL + 4}
              y={sy(baselineScore) - 4}
              fontSize={10}
              fill={colors.textMuted}
            >
              baseline {baselineScore.toFixed(2)}
            </text>
          </>
        )}

        {bestPath && (
          <path
            d={bestPath}
            fill="none"
            stroke={colors.warn}
            strokeWidth={1.5}
            strokeDasharray="4 3"
            opacity={0.85}
          />
        )}

        {linePath && (
          <path d={linePath} fill="none" stroke={colors.text} strokeWidth={1.25} />
        )}

        {points.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.iteration)}
            cy={sy(p.score)}
            r={p.insight ? 4.5 : 3}
            fill={
              p.status === "failed"
                ? colors.danger
                : p.status === "pruned"
                ? colors.textMuted
                : stageColor(p.stage)
            }
            stroke={colors.bg}
            strokeWidth={1}
          >
            <title>
              {`iter ${p.iteration} · ${p.stage} · ${p.score.toFixed(3)}${
                p.insight ? "\n" + p.insight : ""
              }`}
            </title>
          </circle>
        ))}

        <line
          x1={padL}
          x2={padL + plotW}
          y1={padT + plotH}
          y2={padT + plotH}
          stroke={colors.borderStrong}
        />
        <text
          x={padL + plotW}
          y={padT + plotH + 18}
          fontSize={10}
          fontFamily={font.mono}
          textAnchor="end"
          fill={colors.textFaint}
        >
          iter {maxX}
        </text>
      </svg>
    </div>
  );
}

const wrap: CSSProperties = {
  ...card,
  padding: space.md,
};

const header: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  fontWeight: 600,
  fontSize: 13,
  color: colors.text,
  marginBottom: space.xs,
};

const meta: CSSProperties = {
  fontWeight: 400,
  fontSize: 12,
  color: colors.textFaint,
  fontFamily: font.mono,
};
