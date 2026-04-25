/**
 * RadarChart — polygon chart of RAGAS metric scores.
 *
 * Consumed by the Results page to show best-vs-baseline side-by-side. All
 * metric values must be in [0, 1]; missing keys are shown as 0. Pure SVG —
 * no D3 — so it's cheap to rerender.
 */
import { CSSProperties, useMemo } from "react";

import { card, colors, font, space } from "../theme";

export interface RadarSeries {
  label: string;
  values: Record<string, number>;
  color: string;
}

export interface RadarChartProps {
  metrics: string[];
  series: RadarSeries[];
  size?: number;
  title?: string;
  /** Optional pretty-label mapping keyed by metric id. */
  labelMap?: Record<string, string>;
}

export default function RadarChart({
  metrics,
  series,
  size = 320,
  title,
  labelMap = {},
}: RadarChartProps) {
  const cx = size / 2;
  const cy = size / 2 + 10;
  const r0 = size / 2 - 48;

  const angle = (i: number) => (i / metrics.length) * Math.PI * 2 - Math.PI / 2;
  const point = (i: number, v: number) => {
    const a = angle(i);
    const r = r0 * Math.max(0, Math.min(1, v));
    return [cx + Math.cos(a) * r, cy + Math.sin(a) * r] as const;
  };

  const rings = [0.25, 0.5, 0.75, 1];

  const polygons = useMemo(
    () =>
      series.map((s) => ({
        ...s,
        points: metrics
          .map((m, i) => {
            const [x, y] = point(i, s.values[m] ?? 0);
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .join(" "),
      })),
    // point() depends on size + metrics.length; size is stable across renders
    // and metrics.length drives both; safe to include only metrics + series.
    [series, metrics],
  );

  return (
    <div style={wrap}>
      {title && <div style={header}>{title}</div>}
      <svg width={size} height={size + 40} style={{ display: "block" }}>
        {rings.map((r) => (
          <polygon
            key={r}
            points={metrics
              .map((_m, i) => {
                const [x, y] = point(i, r);
                return `${x.toFixed(1)},${y.toFixed(1)}`;
              })
              .join(" ")}
            fill="none"
            stroke={colors.border}
            strokeWidth={1}
          />
        ))}

        {metrics.map((m, i) => {
          const [x, y] = point(i, 1.14);
          const anchor =
            Math.abs(x - cx) < 4 ? "middle" : x > cx ? "start" : "end";
          return (
            <g key={m}>
              <line
                x1={cx}
                y1={cy}
                x2={point(i, 1)[0]}
                y2={point(i, 1)[1]}
                stroke={colors.border}
              />
              <text
                x={x}
                y={y}
                textAnchor={anchor}
                dominantBaseline="middle"
                fontSize={11}
                fill={colors.textMuted}
              >
                {labelMap[m] ?? m}
              </text>
            </g>
          );
        })}

        {polygons.map((s) => (
          <g key={s.label}>
            <polygon
              points={s.points}
              fill={s.color}
              fillOpacity={0.14}
              stroke={s.color}
              strokeWidth={1.5}
            />
            {metrics.map((m, i) => {
              const [x, y] = point(i, s.values[m] ?? 0);
              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r={3}
                  fill={s.color}
                  stroke={colors.bg}
                  strokeWidth={1}
                />
              );
            })}
          </g>
        ))}
      </svg>

      <div style={legend}>
        {series.map((s) => (
          <div key={s.label} style={legendRow}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 5,
                background: s.color,
                display: "inline-block",
              }}
            />
            <span style={{ fontWeight: 500 }}>{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const wrap: CSSProperties = {
  ...card,
  padding: space.lg,
  display: "inline-block",
};

const header: CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: colors.text,
  marginBottom: space.xs,
};

const legend: CSSProperties = {
  display: "flex",
  gap: space.lg,
  justifyContent: "center",
  marginTop: space.xs,
  color: colors.textMuted,
  fontSize: 12,
  fontFamily: font.sans,
};

const legendRow: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: space.xs,
};
