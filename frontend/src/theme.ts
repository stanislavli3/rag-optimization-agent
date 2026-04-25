/**
 * Notion-inspired design tokens.
 *
 * Shared palette, spacing, and reusable style objects used across every page.
 * Kept intentionally tight — if a token isn't here, don't invent a new one,
 * pick the closest existing value. The visual language is:
 *   - near-white canvas, hairline borders, near-black text
 *   - accent colour used sparingly (links, the single "primary" button)
 *   - radii 3–6, never larger; shadows avoided entirely
 *   - typography does the work of hierarchy, not colour or rules
 */
import { CSSProperties } from "react";

export const colors = {
  bg: "#ffffff",
  bgSubtle: "#fbfbfa",
  bgHover: "#f1f1ef",
  bgSunken: "#f7f6f3",

  text: "#37352f",
  textMuted: "#787774",
  textFaint: "#9b9a97",

  border: "rgba(55, 53, 47, 0.09)",
  borderStrong: "rgba(55, 53, 47, 0.16)",

  accent: "#2383e2",
  accentSoft: "#e7f3fb",

  success: "#0f7b6c",
  successSoft: "#ddedea",
  warn: "#cb912f",
  warnSoft: "#fbf3db",
  danger: "#e03e3e",
  dangerSoft: "#fbe4e4",
  purple: "#6940a5",
  purpleSoft: "#eae4f2",

  // Stage palette — low-chroma earth tones that Notion's callouts use.
  stagePreliminary: "#6b7280",
  stageBaseline: "#2383e2",
  stageExploration: "#6940a5",
  stageAblation: "#cb912f",
} as const;

export const space = {
  xxs: 2,
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
} as const;

export const radius = {
  sm: 3,
  md: 4,
  lg: 6,
} as const;

export const font = {
  sans:
    'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
  mono: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
} as const;

// Top-level page layout (used by everything routed inside the app shell).
export const pageStyle: CSSProperties = {
  padding: `${space.xl}px ${space.xxl}px`,
  maxWidth: 1080,
  margin: "0 auto",
  color: colors.text,
};

export const pageTitle: CSSProperties = {
  fontSize: 28,
  fontWeight: 700,
  letterSpacing: -0.2,
  margin: 0,
  marginBottom: space.xs,
  color: colors.text,
};

export const pageSubtitle: CSSProperties = {
  fontSize: 14,
  color: colors.textMuted,
  margin: 0,
  marginBottom: space.xl,
};

export const sectionTitle: CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  letterSpacing: 0.3,
  textTransform: "uppercase",
  color: colors.textMuted,
  margin: 0,
  marginBottom: space.sm,
};

// Cards: we use a single card style everywhere for visual consistency.
// Notion cards are essentially "a block with a hairline border and some padding".
export const card: CSSProperties = {
  background: colors.bg,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.lg,
  padding: space.lg,
};

export const cardFlush: CSSProperties = {
  ...card,
  padding: 0,
  overflow: "hidden",
};

export const callout = (tone: "neutral" | "accent" | "success" | "warn" | "danger" = "neutral"): CSSProperties => {
  const bg = {
    neutral: colors.bgSubtle,
    accent: colors.accentSoft,
    success: colors.successSoft,
    warn: colors.warnSoft,
    danger: colors.dangerSoft,
  }[tone];
  const fg = {
    neutral: colors.text,
    accent: colors.accent,
    success: colors.success,
    warn: colors.warn,
    danger: colors.danger,
  }[tone];
  return {
    background: bg,
    color: fg,
    border: `1px solid ${colors.border}`,
    borderRadius: radius.md,
    padding: `${space.sm}px ${space.md}px`,
    fontSize: 13,
    lineHeight: 1.5,
  };
};

// Buttons. Notion's buttons are low-contrast — we use a single accent button
// for the single primary action on each page, everything else is ghost-style.
export const primaryButton: CSSProperties = {
  background: colors.accent,
  color: "#ffffff",
  border: "none",
  borderRadius: radius.md,
  padding: `${space.sm}px ${space.lg}px`,
  fontSize: 14,
  fontWeight: 500,
  cursor: "pointer",
  transition: "opacity 120ms ease, transform 80ms ease",
};

export const ghostButton: CSSProperties = {
  background: "transparent",
  color: colors.text,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.md,
  padding: `${space.xs + 2}px ${space.md}px`,
  fontSize: 13,
  fontWeight: 500,
  cursor: "pointer",
  transition: "background 120ms ease",
};

export const subtleButton: CSSProperties = {
  background: "transparent",
  color: colors.textMuted,
  border: "none",
  padding: `${space.xs}px ${space.sm}px`,
  borderRadius: radius.sm,
  fontSize: 13,
  cursor: "pointer",
  transition: "background 120ms ease",
};

// Chip / pill with coloured soft background, used for stage labels,
// difficulty badges, status pills, etc.
export const chip = (
  tone: "neutral" | "accent" | "success" | "warn" | "danger" | "purple" = "neutral",
): CSSProperties => {
  const map = {
    neutral: { bg: colors.bgHover, fg: colors.textMuted },
    accent: { bg: colors.accentSoft, fg: colors.accent },
    success: { bg: colors.successSoft, fg: colors.success },
    warn: { bg: colors.warnSoft, fg: colors.warn },
    danger: { bg: colors.dangerSoft, fg: colors.danger },
    purple: { bg: colors.purpleSoft, fg: colors.purple },
  }[tone];
  return {
    display: "inline-flex",
    alignItems: "center",
    gap: space.xs,
    background: map.bg,
    color: map.fg,
    borderRadius: radius.md,
    padding: "1px 6px",
    fontSize: 12,
    fontWeight: 500,
    lineHeight: 1.5,
  };
};

export const divider: CSSProperties = {
  height: 1,
  background: colors.border,
  border: "none",
  margin: `${space.lg}px 0`,
};

export const inlineKbd: CSSProperties = {
  fontFamily: font.mono,
  fontSize: 12,
  color: colors.textMuted,
  background: colors.bgHover,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.sm,
  padding: "1px 6px",
};

export const tableStyles = {
  table: {
    width: "100%",
    borderCollapse: "collapse" as const,
    fontSize: 13,
  },
  th: {
    textAlign: "left" as const,
    padding: `${space.sm}px ${space.md}px`,
    borderBottom: `1px solid ${colors.border}`,
    color: colors.textMuted,
    fontWeight: 500,
    fontSize: 12,
    textTransform: "uppercase" as const,
    letterSpacing: 0.3,
  },
  td: {
    padding: `${space.sm}px ${space.md}px`,
    borderBottom: `1px solid ${colors.border}`,
    color: colors.text,
    verticalAlign: "top" as const,
  },
} as const;

// Map RAGAS metric name → nice label + tone, used on Results / Radar / etc.
export const metricLabel: Record<string, string> = {
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer Relevancy",
  context_precision: "Context Precision",
  context_recall: "Context Recall",
  answer_correctness: "Answer Correctness",
  ragas_score: "RAGAS Score",
};

export const stageColor = (
  stage: "preliminary" | "baseline" | "exploration" | "ablation",
): string =>
  ({
    preliminary: colors.stagePreliminary,
    baseline: colors.stageBaseline,
    exploration: colors.stageExploration,
    ablation: colors.stageAblation,
  }[stage]);

export const stageChipTone = (
  stage: "preliminary" | "baseline" | "exploration" | "ablation",
): Parameters<typeof chip>[0] =>
  ({
    preliminary: "neutral" as const,
    baseline: "accent" as const,
    exploration: "purple" as const,
    ablation: "warn" as const,
  }[stage]);
