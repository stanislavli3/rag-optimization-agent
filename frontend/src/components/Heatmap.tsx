/**
 * Heatmap — 2D matrix visualiser.
 *
 * Used by the TestGen page for the GRADE-style reasoning_depth × semantic_distance
 * difficulty matrix, and by Results for score-per-parameter-pair grids. Colour
 * scale is a linear interpolation from `lowColor` to `highColor`; each cell
 * shows the numeric value and (optionally) a count on hover.
 */
import { CSSProperties, useMemo } from "react";

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
  lowColor = [224, 242, 254],
  highColor = [30, 64, 175],
  cellSize = 40,
  formatValue = (v) => (Number.isInteger(v) ? `${v}` : v.toFixed(2)),
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

  const rowLabelW = 80;
  const colLabelH = 28;
  const width = rowLabelW + colLabels.length * cellSize + 20;
  const height = colLabelH + rowLabels.length * cellSize + 30;

  return (
    <div style={wrap}>
      <svg width={width} height={height}>
        {xAxisLabel && (
          <text
            x={rowLabelW + (colLabels.length * cellSize) / 2}
            y={12}
            textAnchor="middle"
            fill="#475569"
            fontSize={11}
            fontWeight={600}
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
            fill="#475569"
            fontSize={11}
            fontWeight={600}
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
            fontSize={10}
            fill="#64748b"
          >
            {c}
          </text>
        ))}
        {rowLabels.map((r, i) => (
          <text
            key={r}
            x={rowLabelW - 6}
            y={colLabelH + i * cellSize + cellSize / 2 + 4}
            textAnchor="end"
            fontSize={10}
            fill="#64748b"
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
                  width={cellSize - 2}
                  height={cellSize - 2}
                  rx={3}
                  fill={fill}
                  stroke="#fff"
                />
                <text
                  x={(cellSize - 2) / 2}
                  y={(cellSize - 2) / 2 + 3}
                  textAnchor="middle"
                  fontSize={10}
                  fontWeight={600}
                  fill={t > 0.55 ? "#fff" : "#0f172a"}
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
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 12,
  display: "inline-block",
};
