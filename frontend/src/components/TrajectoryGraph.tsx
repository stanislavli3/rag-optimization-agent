/**
 * TrajectoryGraph — score-per-iteration line chart with stage bands.
 *
 * Renders the RAGAS score of each completed BFTS node in the order the agent
 * explored them, shading horizontal bands to reflect stage boundaries and
 * overlaying the running best-so-far envelope. Lightweight SVG — no d3/plotly
 * dependency — so it can stream-update on every SSE event without jank.
 */
import { CSSProperties, useMemo } from "react";

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
}

const STAGE_FILL: Record<TrajectoryStage, string> = {
  preliminary: "rgba(125,211,252,0.08)",
  baseline: "rgba(96,165,250,0.08)",
  exploration: "rgba(167,139,250,0.08)",
  ablation: "rgba(245,158,11,0.08)",
};

const STAGE_STROKE: Record<TrajectoryStage, string> = {
  preliminary: "#7dd3fc",
  baseline: "#60a5fa",
  exploration: "#a78bfa",
  ablation: "#f59e0b",
};


export default function TrajectoryGraph({
  points,
  width = 540,
  height = 260,
  baselineScore = null,
}: TrajectoryGraphProps) {
  const padL = 36;
  const padR = 16;
  const padT = 18;
  const padB = 28;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const { xs, ys, bestCurve, maxX, scoreMin, scoreMax, stageSpans } = useMemo(() => {
    if (points.length === 0) {
      return {
        xs: [],
        ys: [],
        bestCurve: [],
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
    // Detect stage transitions to draw background bands.
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
      <div style={header}>
        Score trajectory
        <span style={{ fontWeight: 400, color: "#64748b", fontSize: 12 }}>
          {points.length} iterations
        </span>
      </div>
      <svg width={width} height={height} style={{ display: "block" }}>
        {/* stage bands */}
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

        {/* y-axis ticks */}
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={padL}
              x2={padL + plotW}
              y1={sy(t)}
              y2={sy(t)}
              stroke="#f1f5f9"
            />
            <text x={padL - 6} y={sy(t) + 3} fontSize={10} textAnchor="end" fill="#94a3b8">
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        {/* baseline */}
        {baselineScore !== null && baselineScore !== undefined && (
          <>
            <line
              x1={padL}
              x2={padL + plotW}
              y1={sy(baselineScore)}
              y2={sy(baselineScore)}
              stroke="#94a3b8"
              strokeDasharray="3 3"
            />
            <text
              x={padL + 4}
              y={sy(baselineScore) - 4}
              fontSize={10}
              fill="#64748b"
            >
              baseline {baselineScore.toFixed(2)}
            </text>
          </>
        )}

        {/* best-so-far */}
        {bestPath && (
          <path
            d={bestPath}
            fill="none"
            stroke="#eab308"
            strokeWidth={2}
            strokeDasharray="4 2"
            opacity={0.8}
          />
        )}

        {/* raw line */}
        {linePath && (
          <path d={linePath} fill="none" stroke="#1e293b" strokeWidth={1.5} />
        )}

        {/* points */}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.iteration)}
            cy={sy(p.score)}
            r={p.insight ? 5 : 3.5}
            fill={
              p.status === "failed"
                ? "#ef4444"
                : p.status === "pruned"
                ? "#64748b"
                : STAGE_STROKE[p.stage]
            }
            stroke="#fff"
            strokeWidth={1}
          >
            <title>
              {`iter ${p.iteration} · ${p.stage} · ${p.score.toFixed(3)}${
                p.insight ? "\n" + p.insight : ""
              }`}
            </title>
          </circle>
        ))}

        {/* x-axis */}
        <line
          x1={padL}
          x2={padL + plotW}
          y1={padT + plotH}
          y2={padT + plotH}
          stroke="#cbd5e1"
        />
        <text
          x={padL + plotW}
          y={padT + plotH + 18}
          fontSize={10}
          textAnchor="end"
          fill="#94a3b8"
        >
          iter {maxX}
        </text>
      </svg>
    </div>
  );
}

const wrap: CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 12,
};

const header: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  fontWeight: 600,
  fontSize: 13,
  color: "#0f172a",
  marginBottom: 4,
};
