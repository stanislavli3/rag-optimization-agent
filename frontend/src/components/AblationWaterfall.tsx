/**
 * AblationWaterfall — horizontal bar chart of per-parameter score contribution.
 *
 * Rendered after Stage 4 (Ablation) completes. Each bar represents one
 * non-default parameter in the winning config and shows how much score is lost
 * when that parameter is reverted to its default — i.e. how load-bearing the
 * optimizer's choice is. Bars sort by delta descending; the bar colour follows
 * a red→yellow→green gradient on contribution percentage.
 */
import { CSSProperties, useMemo } from "react";

export interface AblationEntry {
  param: string;
  optimized_value: unknown;
  default_value: unknown;
  score_with: number;
  score_without: number;
  delta: number;
  contribution_pct: number;
}

export interface AblationWaterfallProps {
  entries: AblationEntry[];
  bestScore: number;
  baselineScore: number;
  width?: number;
  height?: number;
}


function deltaColor(pct: number): string {
  // 0% → red, 15% → yellow, 30%+ → green
  const t = Math.max(0, Math.min(1, pct / 30));
  if (t < 0.5) {
    const k = t / 0.5;
    const r = 239 + (234 - 239) * k;
    const g = 68 + (179 - 68) * k;
    const b = 68 + (8 - 68) * k;
    return `rgb(${r | 0},${g | 0},${b | 0})`;
  }
  const k = (t - 0.5) / 0.5;
  const r = 234 + (34 - 234) * k;
  const g = 179 + (197 - 179) * k;
  const b = 8 + (94 - 8) * k;
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

function fmtVal(v: unknown): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? `${v}` : v.toFixed(2);
  return String(v);
}

export default function AblationWaterfall({
  entries,
  bestScore,
  baselineScore,
  width = 720,
  height,
}: AblationWaterfallProps) {
  const sorted = useMemo(
    () => [...entries].sort((a, b) => b.delta - a.delta),
    [entries],
  );
  const maxDelta = useMemo(
    () => Math.max(0.01, ...sorted.map((e) => Math.abs(e.delta))),
    [sorted],
  );

  const rowH = 36;
  const padTop = 34;
  const padBot = 30;
  const chartH = height ?? padTop + padBot + Math.max(1, sorted.length) * rowH;
  const barLeft = 180;
  const barRight = 120;
  const barWidth = width - barLeft - barRight;

  const bestX = barLeft + barWidth;
  const baselineX =
    bestScore === 0
      ? barLeft
      : barLeft + (baselineScore / bestScore) * barWidth;

  if (sorted.length === 0) {
    return (
      <div style={wrap}>
        <div style={header}>Ablation contribution</div>
        <div style={empty}>
          The winning config matches the default on every parameter — nothing to
          ablate.
        </div>
      </div>
    );
  }

  return (
    <div style={wrap}>
      <div style={header}>
        Ablation contribution
        <span style={subheader}>
          baseline {baselineScore.toFixed(3)} → best {bestScore.toFixed(3)}{" "}
          (Δ {(bestScore - baselineScore).toFixed(3)})
        </span>
      </div>
      <svg width={width} height={chartH} style={{ display: "block" }}>
        {/* baseline reference line */}
        <line
          x1={baselineX}
          x2={baselineX}
          y1={padTop - 8}
          y2={chartH - padBot + 4}
          stroke="#94a3b8"
          strokeDasharray="4 3"
          strokeWidth={1}
        />
        <text
          x={baselineX}
          y={padTop - 12}
          fill="#64748b"
          fontSize={11}
          textAnchor="middle"
        >
          baseline
        </text>

        {/* best reference line */}
        <line
          x1={bestX}
          x2={bestX}
          y1={padTop - 8}
          y2={chartH - padBot + 4}
          stroke="#16a34a"
          strokeWidth={1.5}
        />
        <text
          x={bestX}
          y={padTop - 12}
          fill="#16a34a"
          fontSize={11}
          fontWeight={600}
          textAnchor="middle"
        >
          best
        </text>

        {sorted.map((e, i) => {
          const y = padTop + i * rowH;
          const barW = Math.max(2, (Math.abs(e.delta) / maxDelta) * barWidth);
          const color = deltaColor(e.contribution_pct);
          return (
            <g key={e.param}>
              <text
                x={barLeft - 10}
                y={y + rowH / 2 + 4}
                fill="#0f172a"
                fontSize={12}
                fontWeight={600}
                textAnchor="end"
              >
                {e.param}
              </text>
              <text
                x={barLeft - 10}
                y={y + rowH / 2 + 18}
                fill="#64748b"
                fontSize={10}
                textAnchor="end"
              >
                {fmtVal(e.optimized_value)} vs {fmtVal(e.default_value)}
              </text>

              <rect
                x={barLeft}
                y={y + 6}
                width={barW}
                height={rowH - 16}
                rx={3}
                fill={color}
                opacity={0.92}
              />

              <text
                x={barLeft + barW + 8}
                y={y + rowH / 2 + 4}
                fill="#0f172a"
                fontSize={12}
                fontWeight={600}
              >
                +{e.delta.toFixed(3)}
              </text>
              <text
                x={barLeft + barW + 8}
                y={y + rowH / 2 + 18}
                fill="#64748b"
                fontSize={10}
              >
                {e.contribution_pct.toFixed(1)}%
              </text>
            </g>
          );
        })}

        {/* x-axis scale hint */}
        <line
          x1={barLeft}
          x2={barLeft + barWidth}
          y1={chartH - padBot + 6}
          y2={chartH - padBot + 6}
          stroke="#e2e8f0"
        />
        <text
          x={barLeft}
          y={chartH - padBot + 20}
          fill="#94a3b8"
          fontSize={10}
        >
          0
        </text>
        <text
          x={barLeft + barWidth}
          y={chartH - padBot + 20}
          fill="#94a3b8"
          fontSize={10}
          textAnchor="end"
        >
          Δ {maxDelta.toFixed(2)}
        </text>
      </svg>
    </div>
  );
}

const wrap: CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 16,
};

const header: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  fontWeight: 600,
  fontSize: 14,
  color: "#0f172a",
  marginBottom: 8,
};

const subheader: CSSProperties = {
  fontWeight: 400,
  fontSize: 12,
  color: "#64748b",
};

const empty: CSSProperties = {
  padding: 24,
  color: "#94a3b8",
  fontStyle: "italic",
  textAlign: "center",
};
