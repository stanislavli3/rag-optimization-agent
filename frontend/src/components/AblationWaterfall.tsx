/**
 * AblationWaterfall — horizontal bar chart of per-parameter score contribution.
 *
 * Rendered after Stage 4 (Ablation) completes. Each bar represents one
 * non-default parameter in the winning config and shows how much score is lost
 * when that parameter is reverted to its default — i.e. how load-bearing the
 * optimizer's choice is. Bars sort by delta descending; colour follows a red →
 * amber → green ramp tied to contribution percentage.
 */
import { CSSProperties, useMemo } from "react";

import { card, colors, font, radius, space } from "../theme";

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
  // 0% red → 15% amber → 30%+ green
  const t = Math.max(0, Math.min(1, pct / 30));
  if (t < 0.5) {
    const k = t / 0.5;
    const r = 224 + (203 - 224) * k;
    const g = 62 + (145 - 62) * k;
    const b = 62 + (47 - 62) * k;
    return `rgb(${r | 0},${g | 0},${b | 0})`;
  }
  const k = (t - 0.5) / 0.5;
  const r = 203 + (15 - 203) * k;
  const g = 145 + (123 - 145) * k;
  const b = 47 + (108 - 47) * k;
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

  const rowH = 40;
  const padTop = 42;
  const padBot = 34;
  const chartH = height ?? padTop + padBot + Math.max(1, sorted.length) * rowH;
  const barLeft = 200;
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
          The winning config matches the default on every parameter — nothing
          to ablate.
        </div>
      </div>
    );
  }

  return (
    <div style={wrap}>
      <div style={header}>
        <span>Ablation contribution</span>
        <span style={subheader}>
          baseline {baselineScore.toFixed(3)} → best {bestScore.toFixed(3)}{" "}
          <span style={{ color: colors.success, fontWeight: 500 }}>
            Δ +{(bestScore - baselineScore).toFixed(3)}
          </span>
        </span>
      </div>
      <svg width={width} height={chartH} style={{ display: "block" }}>
        <line
          x1={baselineX}
          x2={baselineX}
          y1={padTop - 10}
          y2={chartH - padBot + 6}
          stroke={colors.textFaint}
          strokeDasharray="3 3"
          strokeWidth={1}
        />
        <text
          x={baselineX}
          y={padTop - 14}
          fill={colors.textMuted}
          fontSize={11}
          textAnchor="middle"
        >
          baseline
        </text>

        <line
          x1={bestX}
          x2={bestX}
          y1={padTop - 10}
          y2={chartH - padBot + 6}
          stroke={colors.success}
          strokeWidth={1.5}
        />
        <text
          x={bestX}
          y={padTop - 14}
          fill={colors.success}
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
                x={barLeft - 12}
                y={y + rowH / 2 + 4}
                fill={colors.text}
                fontSize={13}
                fontWeight={500}
                textAnchor="end"
              >
                {e.param}
              </text>
              <text
                x={barLeft - 12}
                y={y + rowH / 2 + 20}
                fill={colors.textFaint}
                fontSize={11}
                fontFamily={font.mono}
                textAnchor="end"
              >
                {fmtVal(e.optimized_value)} vs {fmtVal(e.default_value)}
              </text>

              <rect
                x={barLeft}
                y={y + 8}
                width={barW}
                height={rowH - 20}
                rx={radius.sm}
                fill={color}
                opacity={0.9}
              />

              <text
                x={barLeft + barW + 8}
                y={y + rowH / 2 + 4}
                fill={colors.text}
                fontSize={13}
                fontWeight={600}
              >
                +{e.delta.toFixed(3)}
              </text>
              <text
                x={barLeft + barW + 8}
                y={y + rowH / 2 + 20}
                fill={colors.textFaint}
                fontSize={11}
              >
                {e.contribution_pct.toFixed(1)}% of lift
              </text>
            </g>
          );
        })}

        <line
          x1={barLeft}
          x2={barLeft + barWidth}
          y1={chartH - padBot + 8}
          y2={chartH - padBot + 8}
          stroke={colors.border}
        />
        <text
          x={barLeft}
          y={chartH - padBot + 22}
          fill={colors.textFaint}
          fontSize={10}
          fontFamily={font.mono}
        >
          0
        </text>
        <text
          x={barLeft + barWidth}
          y={chartH - padBot + 22}
          fill={colors.textFaint}
          fontSize={10}
          fontFamily={font.mono}
          textAnchor="end"
        >
          Δ {maxDelta.toFixed(2)}
        </text>
      </svg>
    </div>
  );
}

const wrap: CSSProperties = {
  ...card,
};

const header: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  fontWeight: 600,
  fontSize: 14,
  color: colors.text,
  marginBottom: space.sm,
};

const subheader: CSSProperties = {
  fontWeight: 400,
  fontSize: 12,
  color: colors.textMuted,
};

const empty: CSSProperties = {
  padding: space.xl,
  color: colors.textFaint,
  fontStyle: "italic",
  textAlign: "center",
  fontSize: 13,
};
