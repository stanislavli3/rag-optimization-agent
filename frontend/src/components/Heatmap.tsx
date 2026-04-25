/**
 * Heatmap — 2D matrix visualiser.
 *
 * Used by the TestGen page for the GRADE-style reasoning_depth × semantic_distance
 * difficulty matrix, and by Results for score-per-parameter-pair grids. Colour
 * scale is a linear interpolation between `lowColor` and `highColor`. The
 * default palette is tuned to Notion's soft blue callout tone.
 */
import { CSSProperties, useMemo } from "react";

import { card, colors, font, radius, space } from "../theme";

export interface HeatmapProps {
  matrix: number[][];
  rowLabels: string[];
  colLabels: string[];
  xAxisLabel?: string;
  yAxisLabel?: string;
  lowColor?: [number, number, number];
  highColor?: [number, number, number];
  cellSize?: number;
  formatValue?: (v: number) => string;
  title?: string;
}

function interp(
  lo: [number, number, number],
  hi: [number, number, number],
  t: number,
): string {
  const c = (i: number) => Math.round(lo[i] + (hi[i] - lo[i]) * t);
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}

export default function Heatmap({
  matrix,
  rowLabels,
  colLabels,
  xAxisLabel,
  yAxisLabel,
  // Soft Notion blue callout → deep accent
  lowColor = [231, 243, 251],
  highColor = [35, 131, 226],
  cellSize = 44,
  formatValue = (v) => (Number.isInteger(v) ? `${v}` : v.toFixed(2)),
  title,
}: HeatmapProps) {
  const { min, max } = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of matrix) {
      for (const v of row) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!Number.isFinite(lo)) return { min: 0, max: 1 };
    if (hi === lo) return { min: lo, max: lo + 1 };
    return { min: lo, max: hi };
  }, [matrix]);

  const rowLabelW = 84;
  const colLabelH = 28;
  const axisPad = xAxisLabel || yAxisLabel ? 16 : 0;
  const svgW = rowLabelW + colLabels.length * cellSize + axisPad;
  const svgH = colLabelH + rowLabels.length * cellSize + 18 + axisPad;

  return (
    <div style={wrap}>
      {title && <div style={header}>{title}</div>}
      <svg width={svgW} height={svgH}>
        {xAxisLabel && (
          <text
            x={rowLabelW + (colLabels.length * cellSize) / 2}
            y={12}
            textAnchor="middle"
            fill={colors.textMuted}
            fontSize={11}
            fontWeight={500}
            letterSpacing="0.3"
          >
            {xAxisLabel}
          </text>
        )}
        {yAxisLabel && (
          <text
            x={12}
            y={colLabelH + (rowLabels.length * cellSize) / 2}
            textAnchor="middle"
            transform={`rotate(-90 12 ${
              colLabelH + (rowLabels.length * cellSize) / 2
            })`}
            fill={colors.textMuted}
            fontSize={11}
            fontWeight={500}
            letterSpacing="0.3"
          >
            {yAxisLabel}
          </text>
        )}
        {colLabels.map((c, i) => (
          <text
            key={c}
            x={rowLabelW + i * cellSize + cellSize / 2}
            y={colLabelH - 6}
            textAnchor="middle"
            fontSize={11}
            fill={colors.textFaint}
            fontFamily={font.mono}
          >
            {c}
          </text>
        ))}
        {rowLabels.map((r, i) => (
          <text
            key={r}
            x={rowLabelW - 8}
            y={colLabelH + i * cellSize + cellSize / 2 + 4}
            textAnchor="end"
            fontSize={11}
            fill={colors.textFaint}
            fontFamily={font.mono}
          >
            {r}
          </text>
        ))}
        {matrix.map((row, ri) =>
          row.map((v, ci) => {
            const t = (v - min) / (max - min || 1);
            const fill = interp(lowColor, highColor, t);
            return (
              <g
                key={`${ri}-${ci}`}
                transform={`translate(${rowLabelW + ci * cellSize}, ${
                  colLabelH + ri * cellSize
                })`}
              >
                <rect
                  width={cellSize - 3}
                  height={cellSize - 3}
                  rx={radius.sm}
                  fill={fill}
                  stroke={colors.bg}
                />
                <text
                  x={(cellSize - 3) / 2}
                  y={(cellSize - 3) / 2 + 3}
                  textAnchor="middle"
                  fontSize={11}
                  fontWeight={600}
                  fill={t > 0.55 ? "#fff" : colors.text}
                >
                  {formatValue(v)}
                </text>
              </g>
            );
          }),
        )}
      </svg>
    </div>
  );
}

const wrap: CSSProperties = {
  ...card,
  padding: space.md,
  display: "inline-block",
};

const header: CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: colors.text,
  marginBottom: space.sm,
};
