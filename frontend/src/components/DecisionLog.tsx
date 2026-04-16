/**
 * DecisionLog — scrolling live feed of BFTS agent decisions.
 *
 * Entries stream in from SSE. The pane auto-scrolls to the bottom unless the
 * user has scrolled up; clicking an entry emits the associated nodeId so the
 * parent component can highlight the matching node in <AgentTree />.
 */
import { CSSProperties, useEffect, useRef, useState } from "react";

export type DecisionType =
  | "expand"
  | "success"
  | "failed"
  | "debug"
  | "pruned"
  | "stage_transition"
  | "insight";

export interface DecisionEntry {
  timestamp: string;
  iteration: number;
  type: DecisionType;
  message: string;
  nodeId?: string;
  score?: number;
  insight?: string;
}

export interface DecisionLogProps {
  entries: DecisionEntry[];
  autoScroll?: boolean;
  onEntryClick?: (entry: DecisionEntry) => void;
  maxHeight?: number;
}

const TYPE_META: Record<DecisionType, { color: string; icon: string; label: string }> = {
  expand: { color: "#3b82f6", icon: "→", label: "expand" },
  success: { color: "#22c55e", icon: "✓", label: "success" },
  failed: { color: "#ef4444", icon: "✗", label: "failed" },
  debug: { color: "#f97316", icon: "↻", label: "debug" },
  pruned: { color: "#64748b", icon: "✂", label: "pruned" },
  stage_transition: { color: "#a855f7", icon: "★", label: "stage" },
  insight: { color: "#6366f1", icon: "💡", label: "insight" },
};


export default function DecisionLog({
  entries,
  autoScroll = true,
  onEntryClick,
  maxHeight = 360,
}: DecisionLogProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [userScrolled, setUserScrolled] = useState(false);

  // Detect manual scroll-up — if the user is within 24px of the bottom, assume
  // they want to keep following the tail.
  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    setUserScrolled(!atBottom);
  };

  useEffect(() => {
    const el = ref.current;
    if (!el || !autoScroll || userScrolled) return;
    el.scrollTop = el.scrollHeight;
  }, [entries, autoScroll, userScrolled]);

  return (
    <div style={wrap}>
      <div style={header}>
        <span>Decision log</span>
        <span style={{ color: "#64748b", fontWeight: 400 }}>{entries.length} entries</span>
      </div>
      <div
        ref={ref}
        onScroll={onScroll}
        style={{ ...body, maxHeight }}
      >
        {entries.length === 0 && (
          <div style={empty}>Agent hasn't made any decisions yet.</div>
        )}
        {entries.map((e, i) => {
          const meta = TYPE_META[e.type] ?? TYPE_META.expand;
          return (
            <div
              key={i}
              onClick={() => onEntryClick?.(e)}
              style={{
                ...entryStyle,
                cursor: onEntryClick ? "pointer" : "default",
                borderLeft: `3px solid ${meta.color}`,
              }}
            >
              <div style={entryHead}>
                <span style={{ color: meta.color, fontWeight: 700, marginRight: 8 }}>
                  {meta.icon} {meta.label.toUpperCase()}
                </span>
                <span style={{ color: "#94a3b8" }}>[iter {e.iteration}]</span>
                {e.nodeId && <span style={{ color: "#475569", marginLeft: 6 }}>{e.nodeId.slice(0, 8)}</span>}
                {typeof e.score === "number" && (
                  <span style={{ color: "#0f172a", marginLeft: 8, fontWeight: 600 }}>
                    score {e.score.toFixed(3)}
                  </span>
                )}
                <span style={{ color: "#cbd5e1", marginLeft: "auto", fontSize: 11 }}>
                  {e.timestamp}
                </span>
              </div>
              <div style={entryBody}>{e.message}</div>
              {e.insight && (
                <div style={insightStyle}>{e.insight}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const wrap: CSSProperties = {
  background: "#0f172a",
  color: "#e2e8f0",
  borderRadius: 8,
  overflow: "hidden",
  border: "1px solid #1e293b",
  fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
};

const header: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  padding: "8px 12px",
  background: "#1e293b",
  fontSize: 13,
  fontWeight: 600,
  letterSpacing: 0.4,
  borderBottom: "1px solid #334155",
};

const body: CSSProperties = {
  overflowY: "auto",
  padding: "4px 0",
};

const empty: CSSProperties = {
  padding: 20,
  color: "#64748b",
  textAlign: "center",
  fontStyle: "italic",
};

const entryStyle: CSSProperties = {
  padding: "6px 12px",
  margin: "2px 8px",
  background: "#111827",
  borderRadius: 4,
};

const entryHead: CSSProperties = {
  display: "flex",
  alignItems: "center",
  fontSize: 11,
  marginBottom: 2,
};

const entryBody: CSSProperties = {
  fontSize: 12,
  color: "#cbd5e1",
  whiteSpace: "pre-wrap",
};

const insightStyle: CSSProperties = {
  marginTop: 4,
  padding: "4px 8px",
  background: "#1e1b4b",
  borderRadius: 3,
  fontSize: 11,
  color: "#c7d2fe",
  fontStyle: "italic",
};
