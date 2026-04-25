/**
 * StageRail — horizontal 4-stage progress timeline.
 *
 * Visualises the progressive-research narrative of BFTS: PRELIMINARY →
 * BASELINE → EXPLORATION → ABLATION. Each stage shows its node count, active
 * stage pulses, completed stages show their transition trigger.
 *
 * Notion-style: no gradient fill, no heavy shadows, hairline connectors, and
 * small low-contrast dots. The active stage uses a thin accent ring.
 */
import { CSSProperties } from "react";

import { card, chip, colors, font, radius, space, stageChipTone } from "../theme";

export type RailStage = "preliminary" | "baseline" | "exploration" | "ablation";

export interface StageSummary {
  stage: RailStage;
  nodeCount: number;
  bestScore: number | null;
  status: "done" | "active" | "pending";
  transitionTrigger?: string;
}

export interface StageRailProps {
  currentStage: RailStage;
  stageSummaries: StageSummary[];
  totalSteps: number;
  completedSteps: number;
}

const STAGE_LABEL: Record<RailStage, string> = {
  preliminary: "Preliminary",
  baseline: "Baseline",
  exploration: "Exploration",
  ablation: "Ablation",
};

const STAGE_ORDER: RailStage[] = ["preliminary", "baseline", "exploration", "ablation"];

export default function StageRail({
  currentStage,
  stageSummaries,
  totalSteps,
  completedSteps,
}: StageRailProps) {
  const byStage = new Map(stageSummaries.map((s) => [s.stage, s]));
  const pct =
    totalSteps > 0 ? Math.min(100, (completedSteps / totalSteps) * 100) : 0;

  return (
    <div style={wrap}>
      <div style={progressHeader}>
        <span style={{ color: colors.textMuted }}>Progress</span>
        <span style={{ fontFamily: font.mono, color: colors.textFaint }}>
          {completedSteps} / {totalSteps} · {pct.toFixed(0)}%
        </span>
      </div>
      <div style={progressBar}>
        <div style={{ ...progressFill, width: `${pct}%` }} />
      </div>

      <div style={rail}>
        {STAGE_ORDER.map((stage, idx) => {
          const s = byStage.get(stage);
          const isActive = stage === currentStage;
          const isDone = s?.status === "done";

          return (
            <div key={stage} style={stageCell(idx === STAGE_ORDER.length - 1)}>
              <div style={stageRow}>
                <div style={dotWrap}>
                  <div
                    style={{
                      ...dot,
                      background: isDone
                        ? colors.success
                        : isActive
                        ? colors.accent
                        : colors.bgHover,
                      border: isActive
                        ? `2px solid ${colors.accent}`
                        : `1px solid ${colors.border}`,
                      color: isDone || isActive ? "#fff" : colors.textFaint,
                    }}
                  >
                    {isDone ? "✓" : idx + 1}
                  </div>
                  {idx < STAGE_ORDER.length - 1 && (
                    <div
                      style={{
                        ...connector,
                        background: isDone ? colors.success : colors.border,
                      }}
                    />
                  )}
                </div>

                <div style={stageBody}>
                  <div style={stageLabelRow}>
                    <span
                      style={{
                        fontWeight: isActive ? 600 : 500,
                        color: isActive || isDone ? colors.text : colors.textMuted,
                        fontSize: 13,
                      }}
                    >
                      {STAGE_LABEL[stage]}
                    </span>
                    {isActive && (
                      <span style={chip(stageChipTone(stage))}>running</span>
                    )}
                  </div>
                  <div style={stageMeta}>
                    {isDone && s?.transitionTrigger
                      ? s.transitionTrigger
                      : s
                      ? `${s.nodeCount} node${s.nodeCount === 1 ? "" : "s"}${
                          typeof s.bestScore === "number"
                            ? ` · best ${s.bestScore.toFixed(2)}`
                            : ""
                        }`
                      : "pending"}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const wrap: CSSProperties = {
  ...card,
  padding: `${space.md}px ${space.lg}px`,
};

const progressHeader: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  fontSize: 12,
  marginBottom: space.xs,
};

const progressBar: CSSProperties = {
  height: 3,
  background: colors.bgHover,
  borderRadius: radius.sm,
  overflow: "hidden",
  marginBottom: space.lg,
};

const progressFill: CSSProperties = {
  height: "100%",
  background: colors.accent,
  transition: "width 300ms ease",
};

const rail: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(4, 1fr)",
  gap: 0,
};

const stageCell = (_isLast: boolean): CSSProperties => ({
  display: "flex",
});

const stageRow: CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  gap: space.sm,
  flex: 1,
  minWidth: 0,
};

const dotWrap: CSSProperties = {
  display: "flex",
  alignItems: "center",
  flexShrink: 0,
};

const dot: CSSProperties = {
  width: 22,
  height: 22,
  borderRadius: 11,
  display: "grid",
  placeItems: "center",
  fontSize: 11,
  fontWeight: 600,
  transition: "background 160ms ease, border-color 160ms ease",
};

const connector: CSSProperties = {
  width: space.md,
  height: 1,
  marginLeft: space.xs,
  transition: "background 160ms ease",
};

const stageBody: CSSProperties = {
  minWidth: 0,
  paddingTop: 2,
};

const stageLabelRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.xs,
  marginBottom: 2,
};

const stageMeta: CSSProperties = {
  fontSize: 11,
  color: colors.textFaint,
  lineHeight: 1.4,
  whiteSpace: "nowrap" as const,
  overflow: "hidden",
  textOverflow: "ellipsis",
};
