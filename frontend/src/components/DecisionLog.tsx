/**
 * DecisionLog — scrolling live feed of BFTS agent decisions.
 *
 * Notion-style: light surface, a hairline rule between entries, no dark code
 * panel. Each entry is a compact row with a tiny coloured dot (type),
 * iteration pill, node id, timestamp, and an optional italic insight quote.
 * Auto-scrolls to the tail unless the user has scrolled up.
 */
import { CSSProperties, useEffect, useRef, useState } from "react";

import { card, chip, colors, font, radius, space } from "../theme";

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

const TYPE_META: Record<
  DecisionType,
  { dot: string; label: string; chipTone: Parameters<typeof chip>[0] }
> = {
  expand: { dot: colors.accent, label: "expand", chipTone: "accent" },
  success: { dot: colors.success, label: "success", chipTone: "success" },
  failed: { dot: colors.danger, label: "failed", chipTone: "danger" },
  debug: { dot: colors.warn, label: "debug", chipTone: "warn" },
  pruned: { dot: colors.textMuted, label: "pruned", chipTone: "neutral" },
  stage_transition: { dot: colors.purple, label: "stage", chipTone: "purple" },
  insight: { dot: colors.purple, label: "insight", chipTone: "purple" },
};

export default function DecisionLog({
  entries,
  autoScroll = true,
  onEntryClick,
  maxHeight = 360,
}: DecisionLogProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [userScrolled, setUserScrolled] = useState(false);

  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    // Treat "within 24px of the bottom" as following the tail.
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
        <span style={{ color: colors.textFaint, fontWeight: 400, fontSize: 12 }}>
          {entries.length} {entries.length === 1 ? "entry" : "entries"}
        </span>
      </div>
      <div ref={ref} onScroll={onScroll} style={{ ...body, maxHeight }}>
        {entries.length === 0 && (
          <div style={empty}>Agent hasn't made any decisions yet.</div>
        )}
        {entries.map((e, i) => {
          const meta = TYPE_META[e.type] ?? TYPE_META.expand;
          return (
            <div
              key={i}
              className="rag-fade-in"
              onClick={() => onEntryClick?.(e)}
              style={{ ...entryStyle, cursor: onEntryClick ? "pointer" : "default" }}
            >
              <div style={entryHead}>
                <span style={{ ...dot, background: meta.dot }} />
                <span style={chip(meta.chipTone)}>{meta.label}</span>
                <span style={iterPill}>iter {e.iteration}</span>
                {e.nodeId && <span style={nodeIdStyle}>{e.nodeId.slice(0, 8)}</span>}
                {typeof e.score === "number" && (
                  <span style={scoreStyle}>score {e.score.toFixed(3)}</span>
                )}
                <span style={{ flex: 1 }} />
                <span style={timestamp}>{e.timestamp}</span>
              </div>
              <div style={entryBody}>{e.message}</div>
              {e.insight && <div style={insightStyle}>“{e.insight}”</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const wrap: CSSProperties = {
  ...card,
  padding: 0,
  overflow: "hidden",
};

const header: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  padding: `${space.sm}px ${space.lg}px`,
  fontSize: 13,
  fontWeight: 600,
  color: colors.text,
  borderBottom: `1px solid ${colors.border}`,
  background: colors.bgSubtle,
};

const body: CSSProperties = {
  overflowY: "auto",
  padding: `${space.xs}px 0`,
};

const empty: CSSProperties = {
  padding: `${space.xl}px`,
  color: colors.textFaint,
  textAlign: "center",
  fontStyle: "italic",
  fontSize: 13,
};

const entryStyle: CSSProperties = {
  padding: `${space.sm}px ${space.lg}px`,
  borderBottom: `1px solid ${colors.border}`,
  transition: "background 80ms ease",
};

const entryHead: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.xs,
  fontSize: 12,
  marginBottom: 2,
};

const dot: CSSProperties = {
  width: 6,
  height: 6,
  borderRadius: 3,
  display: "inline-block",
  marginRight: space.xs,
};

const iterPill: CSSProperties = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.textFaint,
};

const nodeIdStyle: CSSProperties = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.textMuted,
  background: colors.bgHover,
  borderRadius: radius.sm,
  padding: "1px 4px",
};

const scoreStyle: CSSProperties = {
  fontSize: 12,
  fontWeight: 600,
  color: colors.text,
};

const timestamp: CSSProperties = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.textFaint,
};

const entryBody: CSSProperties = {
  fontSize: 13,
  color: colors.text,
  whiteSpace: "pre-wrap",
};

const insightStyle: CSSProperties = {
  marginTop: space.xs,
  padding: `${space.xs}px ${space.sm}px`,
  background: colors.purpleSoft,
  color: colors.purple,
  borderRadius: radius.sm,
  fontSize: 12,
  fontStyle: "italic",
};
