/**
 * StageRail — horizontal 4-stage progress timeline.
 *
 * Visualises the progressive-research narrative of BFTS: PRELIMINARY →
 * BASELINE → EXPLORATION → ABLATION. Each stage shows its node count, active
 * stage pulses, completed stages show their transition trigger.
 */
import { CSSProperties } from "react";

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
  preliminary: "PRELIMINARY",
  baseline: "BASELINE",
  exploration: "EXPLORATION",
  ablation: "ABLATION",
};

const STAGE_COLOR: Record<RailStage, string> = {
  preliminary: "#7dd3fc",
  baseline: "#60a5fa",
  exploration: "#a78bfa",
  ablation: "#f59e0b",
};

const STAGE_ORDER: RailStage[] = ["preliminary", "baseline", "exploration", "ablation"];


export default function StageRail({
  currentStage,
  stageSummaries,
  totalSteps,
  completedSteps,
}: StageRailProps) {
  const summaryByStage = new Map(stageSummaries.map((s) => [s.stage, s]));

  const pct = totalSteps > 0 ? Math.min(100, (completedSteps / totalSteps) * 100) : 0;

  return (
    <div style={wrap}>
      <style>{`
        @keyframes stage-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246, 0.8); }
          50% { box-shadow: 0 0 0 10px rgba(59,130,246, 0); }
        }
      `}</style>
      <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>
        Progress: {completedSteps} / {totalSteps} steps ({pct.toFixed(0)}%)
      </div>
      <div style={progressBar}>
        <div style={{ ...progressFill, width: `${pct}%` }} />
      </div>

      <div style={rail}>
        {STAGE_ORDER.map((stage, idx) => {
          const s = summaryByStage.get(stage);
          const isActive = stage === currentStage;
          const isDone = s?.status === "done";
          const color = STAGE_COLOR[stage];
          const dotBg = isDone || isActive ? color : "#e2e8f0";

          return (
            <div key={stage} style={{ display: "flex", alignItems: "center", flex: idx === 3 ? 0 : 1 }}>
              <div style={stageContainer}>
                <div
                  style={{
                    ...dot,
                    background: dotBg,
                    border: isActive ? `2px solid ${color}` : "2px solid transparent",
                    animation: isActive ? "stage-pulse 1.6s ease-in-out infinite" : undefined,
                  }}
                />
                <div style={{ marginTop: 6 }}>
                  <div style={{ ...label, opacity: isDone || isActive ? 1 : 0.5, fontWeight: isActive ? 700 : 500 }}>
                    {STAGE_LABEL[stage]}
                  </div>
                  <div style={sublabel}>
                    {isDone && s?.transitionTrigger
                      ? s.transitionTrigger
                      : s
                      ? `${s.nodeCount} nodes${typeof s.bestScore === "number" ? ` · best ${s.bestScore.toFixed(2)}` : ""}`
                      : "pending"}
                  </div>
                </div>
              </div>
              {idx < STAGE_ORDER.length - 1 && (
                <div
                  style={{
                    ...connector,
                    borderTopStyle: isDone ? "solid" : "dashed",
                    borderTopColor: isDone ? color : "#cbd5e1",
                  }}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const wrap: CSSProperties = {
  padding: "12px 16px",
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
};

const progressBar: CSSProperties = {
  height: 4,
  background: "#e2e8f0",
  borderRadius: 2,
  overflow: "hidden",
  marginBottom: 20,
};

const progressFill: CSSProperties = {
  height: "100%",
  background: "linear-gradient(90deg,#7dd3fc,#60a5fa,#a78bfa,#f59e0b)",
  transition: "width 300ms ease",
};

const rail: CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "flex-start",
};

const stageContainer: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  textAlign: "center",
  minWidth: 140,
};

const dot: CSSProperties = {
  width: 18,
  height: 18,
  borderRadius: 10,
};

const label: CSSProperties = {
  fontSize: 12,
  color: "#0f172a",
  letterSpacing: 0.5,
};

const sublabel: CSSProperties = {
  fontSize: 11,
  color: "#64748b",
  marginTop: 2,
};

const connector: CSSProperties = {
  flex: 1,
  borderTopWidth: 2,
  borderTopStyle: "dashed",
  borderTopColor: "#cbd5e1",
  marginTop: 9,
  marginLeft: 4,
  marginRight: 4,
  minWidth: 40,
};
