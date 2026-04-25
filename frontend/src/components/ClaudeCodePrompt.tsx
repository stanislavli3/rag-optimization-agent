/**
 * ClaudeCodePrompt — renders a copy-pasteable brief from a completed run.
 *
 * Takes the RunSnapshot (best config, metrics, ablation, baseline) and
 * formats it as a markdown prompt the user can paste into Claude Code (or
 * any LLM) to apply the winning config to their own repo. Pure
 * presentational — no API calls, no state beyond "copied?".
 */
import { CSSProperties, useMemo, useState } from "react";

import { AblationEntry } from "./AblationWaterfall";
import { RunSnapshot } from "../context/RunContext";
import {
  card,
  colors,
  font,
  ghostButton,
  primaryButton,
  radius,
  sectionTitle,
  space,
} from "../theme";

interface Props {
  run: RunSnapshot;
}

export default function ClaudeCodePrompt({ run }: Props) {
  const prompt = useMemo(() => buildPrompt(run), [run]);
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(prompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Fallback for browsers that block clipboard without a user gesture
      // (shouldn't happen here since this fires from onClick, but cheap to hedge).
      const ta = document.createElement("textarea");
      ta.value = prompt;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    }
  };

  const download = () => {
    const blob = new Blob([prompt], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rag-optimizer-prompt-${run.experimentId.slice(0, 8)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section style={{ ...card, marginTop: space.lg }}>
      <div style={headerRow}>
        <div>
          <h2 style={sectionTitle}>Claude Code prompt</h2>
          <p style={subtitle}>
            Paste this into your own Claude Code session to apply the winning
            config to your repo's RAG pipeline.
          </p>
        </div>
        <div style={{ display: "flex", gap: space.xs }}>
          <button onClick={download} style={ghostButton}>
            ↓ Download .md
          </button>
          <button onClick={copy} style={primaryButton}>
            {copied ? "✓ Copied" : "Copy prompt"}
          </button>
        </div>
      </div>
      <pre style={preStyle}>{prompt}</pre>
    </section>
  );
}

function buildPrompt(run: RunSnapshot): string {
  const {
    bestConfig,
    bestMetrics,
    bestScore,
    baselineScore,
    ablation,
    trajectory,
  } = run;

  const liftLine = (() => {
    if (bestScore === null || baselineScore === null) return "";
    const delta = bestScore - baselineScore;
    const pct = baselineScore !== 0 ? (delta / baselineScore) * 100 : 0;
    const sign = delta >= 0 ? "+" : "";
    return ` (${sign}${delta.toFixed(3)}, ${sign}${pct.toFixed(1)}% vs baseline ${baselineScore.toFixed(3)})`;
  })();

  const configLines = Object.entries(bestConfig)
    .map(([k, v]) => `- ${k}: ${formatValue(v)}`)
    .join("\n");

  const metricLines = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
  ]
    .filter((k) => typeof bestMetrics[k] === "number")
    .map((k) => `- ${prettyMetricName(k)}: ${Number(bestMetrics[k]).toFixed(3)}`)
    .join("\n");

  const ablationSection = buildAblationSection(ablation);
  const trajectorySection = buildTrajectorySection(trajectory, bestScore);

  return `# Apply optimized RAG configuration

I ran a BFTS optimizer over a RAG pipeline and it converged on the config below. Please apply it to this repo's RAG pipeline code.

## Winning config
RAGAS score: ${bestScore !== null ? bestScore.toFixed(3) : "?"}${liftLine}

${configLines || "(no config captured)"}

## Per-metric scores
${metricLines || "(no metrics captured)"}
${ablationSection}${trajectorySection}

## What to do
1. Locate this repo's RAG pipeline — look for where chunk_size, top_k, embedding model, and retriever are set. Typical locations: \`src/**/pipeline*\`, \`src/**/retriever*\`, a config file, or env vars.
2. Update the defaults to match the winning config above. Prefer the existing configuration mechanism (dataclass, dict, env var) — don't introduce a new one.
3. Keep anything not listed above unchanged.
4. Run the test suite if one exists; confirm no regressions.
5. Report back: which file(s) you changed, which tests you ran, and whether anything unexpected surfaced.

If the winning config disagrees with a value the code currently relies on (e.g. a chunk_size that breaks a fixture), stop and flag it rather than silently diverging.
`;
}

function buildAblationSection(ablation: AblationEntry[]): string {
  if (!ablation || ablation.length === 0) return "";
  // Biggest positive deltas = params that mattered most. Sort descending.
  const ranked = [...ablation].sort((a, b) => (b.delta ?? 0) - (a.delta ?? 0));
  const lines = ranked
    .map((a) => {
      const delta = typeof a.delta === "number" ? a.delta : 0;
      const sign = delta >= 0 ? "+" : "";
      const verdict =
        delta > 0.02
          ? "essential — keep as tuned"
          : delta < -0.02
          ? "negative contribution — consider reverting"
          : "marginal — safe to default";
      return `- \`${a.param}\` (${sign}${delta.toFixed(3)}): ${verdict}`;
    })
    .join("\n");
  return `

## Ablation — which params matter most
${lines}
`;
}

function buildTrajectorySection(
  trajectory: RunSnapshot["trajectory"],
  bestScore: number | null,
): string {
  if (!trajectory || trajectory.length < 3 || bestScore === null) return "";
  const scores = trajectory
    .map((p) => (typeof p.score === "number" ? p.score : null))
    .filter((s): s is number => s !== null);
  if (scores.length < 3) return "";
  const first = scores[0];
  const peakIter = scores.indexOf(bestScore);
  return `

## Search trajectory
- Iterations explored: ${trajectory.length}
- First score: ${first.toFixed(3)}
- Peak at iteration: ${peakIter >= 0 ? peakIter : "unknown"}
`;
}

function prettyMetricName(k: string): string {
  return k
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return JSON.stringify(v);
}

const headerRow: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  gap: space.md,
  marginBottom: space.md,
};

const subtitle: CSSProperties = {
  marginTop: 2,
  marginBottom: 0,
  color: colors.textMuted,
  fontSize: 13,
  maxWidth: 520,
};

const preStyle: CSSProperties = {
  background: colors.bgSubtle,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.md,
  padding: space.md,
  fontFamily: font.mono,
  fontSize: 12,
  lineHeight: 1.5,
  color: colors.text,
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  margin: 0,
  maxHeight: 420,
  overflow: "auto",
};
