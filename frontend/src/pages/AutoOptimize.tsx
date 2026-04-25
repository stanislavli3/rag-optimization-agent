/**
 * AutoOptimize — the Agent Console.
 *
 * One unified dashboard for watching the BFTS optimizer run in real time.
 * Top rail shows stage progression; left pane grows the search tree as nodes
 * finish; right pane stacks a score trajectory over the live decision log.
 * When Stage 4 (Ablation) completes, an AblationWaterfall pops in below to
 * show which parameters actually carried the winning config.
 *
 * Data plumbing: a single SSE connection (useAgentStream) feeds all five
 * sub-components. Reducer-style derivation here keeps sub-components pure.
 * On completion the derived snapshot is saved to RunContext so subsequent
 * pages (Results / Export / Trajectory / Comparison) can read from it.
 */
import { CSSProperties, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import AblationWaterfall, { AblationEntry } from "../components/AblationWaterfall";
import AgentTree, { TreeNode, TreeStage } from "../components/AgentTree";
import DecisionLog, { DecisionEntry, DecisionType } from "../components/DecisionLog";
import StageRail, { RailStage, StageSummary } from "../components/StageRail";
import TrajectoryGraph, { TrajectoryPoint } from "../components/TrajectoryGraph";
import { RunSnapshot, useRun } from "../context/RunContext";
import useAgentStream, { AgentStreamEvent } from "../hooks/useAgentStream";
import {
  callout,
  card,
  chip,
  colors,
  font,
  ghostButton,
  pageStyle,
  pageSubtitle,
  pageTitle,
  primaryButton,
  radius,
  sectionTitle,
  space,
} from "../theme";

interface ConfigForm {
  maxSteps: number;
  numSeeds: number;
  budgetUsd: string;
  strategy: "bfts" | "random" | "greedy";
  searchSpace: Record<string, boolean>;
}

const SEARCH_SPACE_KEYS = [
  "chunk_size",
  "top_k",
  "reranker",
  "search_mode",
  "prompt_style",
  "embedding_model",
];

const DEFAULT_FORM: ConfigForm = {
  maxSteps: 15,
  numSeeds: 2,
  budgetUsd: "",
  strategy: "bfts",
  searchSpace: Object.fromEntries(SEARCH_SPACE_KEYS.map((k) => [k, true])),
};

interface DerivedState {
  nodes: TreeNode[];
  decisions: DecisionEntry[];
  trajectory: TrajectoryPoint[];
  stageSummaries: StageSummary[];
  currentStage: RailStage;
  completedSteps: number;
  totalSteps: number;
  ablationEntries: AblationEntry[] | null;
  baselineScore: number | null;
  bestScore: number | null;
  bestNodeId: string | null;
  bestConfig: Record<string, unknown>;
  bestMetrics: Record<string, any>;
  isComplete: boolean;
}

const EMPTY_STATE: DerivedState = {
  nodes: [],
  decisions: [],
  trajectory: [],
  stageSummaries: [
    { stage: "preliminary", nodeCount: 0, bestScore: null, status: "pending" },
    { stage: "baseline", nodeCount: 0, bestScore: null, status: "pending" },
    { stage: "exploration", nodeCount: 0, bestScore: null, status: "pending" },
    { stage: "ablation", nodeCount: 0, bestScore: null, status: "pending" },
  ],
  currentStage: "preliminary",
  completedSteps: 0,
  totalSteps: 1,
  ablationEntries: null,
  baselineScore: null,
  bestScore: null,
  bestNodeId: null,
  bestConfig: {},
  bestMetrics: {},
  isComplete: false,
};

function deriveState(
  events: AgentStreamEvent[],
  maxSteps: number,
): DerivedState {
  const nodesById = new Map<string, TreeNode>();
  const decisions: DecisionEntry[] = [];
  const trajectory: TrajectoryPoint[] = [];
  const stageCounts: Record<RailStage, number> = {
    preliminary: 0,
    baseline: 0,
    exploration: 0,
    ablation: 0,
  };
  const stageBest: Record<RailStage, number | null> = {
    preliminary: null,
    baseline: null,
    exploration: null,
    ablation: null,
  };
  const stageTransitions: Record<RailStage, string | undefined> = {
    preliminary: undefined,
    baseline: undefined,
    exploration: undefined,
    ablation: undefined,
  };

  let currentStage: RailStage = "preliminary";
  let iter = 0;
  let ablationEntries: AblationEntry[] | null = null;
  let baselineScore: number | null = null;
  let bestScore: number | null = null;
  let bestNodeId: string | null = null;
  let bestConfig: Record<string, unknown> = {};
  let bestMetrics: Record<string, any> = {};
  let isComplete = false;

  const pushDecision = (
    type: DecisionType,
    message: string,
    extra: Partial<DecisionEntry> = {},
  ) => {
    decisions.push({
      timestamp: new Date().toLocaleTimeString(),
      iteration: iter,
      type,
      message,
      ...extra,
    });
  };

  for (const ev of events) {
    const d = (ev.data ?? {}) as Record<string, any>;
    switch (ev.event) {
      case "node_start": {
        iter = d.iteration ?? iter + 1;
        const node: TreeNode = {
          id: d.node_id,
          parent_id: d.parent_id ?? null,
          stage: (d.stage as TreeStage) ?? currentStage,
          status: "running",
          config: d.config ?? {},
          score: null,
          depth: d.depth ?? 0,
          iteration_number: iter,
        };
        nodesById.set(node.id, node);
        pushDecision(
          "expand",
          `expanding node (${Object.keys(node.config).length} params)`,
          { nodeId: node.id },
        );
        break;
      }
      case "node_success": {
        const prev = nodesById.get(d.node_id);
        if (prev) {
          prev.status = "success";
          prev.score = typeof d.score === "number" ? d.score : prev.score;
          nodesById.set(d.node_id, { ...prev });
        }
        const s = typeof d.score === "number" ? d.score : null;
        if (s !== null) {
          stageCounts[currentStage] += 1;
          const curBest = stageBest[currentStage];
          if (curBest === null || s > curBest) stageBest[currentStage] = s;
          if (bestScore === null || s > bestScore) {
            bestScore = s;
            bestNodeId = d.node_id;
            if (d.config) bestConfig = d.config;
            if (d.metrics) bestMetrics = d.metrics;
          }
          if (baselineScore === null && currentStage === "baseline") {
            baselineScore = s;
          }
          trajectory.push({
            iteration: iter,
            score: s,
            stage: currentStage,
            status: "success",
            insight: d.insight,
          });
        }
        pushDecision("success", d.insight || `score ${s?.toFixed(3) ?? "—"}`, {
          nodeId: d.node_id,
          score: s ?? undefined,
          insight: d.insight,
        });
        break;
      }
      case "node_failed": {
        const prev = nodesById.get(d.node_id);
        if (prev) {
          prev.status = "failed";
          nodesById.set(d.node_id, { ...prev });
        }
        trajectory.push({
          iteration: iter,
          score: 0,
          stage: currentStage,
          status: "failed",
        });
        pushDecision(
          d.debug_attempt ? "debug" : "failed",
          d.error || "node failed",
          { nodeId: d.node_id },
        );
        break;
      }
      case "node_pruned": {
        const prev = nodesById.get(d.node_id);
        if (prev) {
          prev.status = "pruned";
          nodesById.set(d.node_id, { ...prev });
        }
        pushDecision("pruned", "node pruned by policy", { nodeId: d.node_id });
        break;
      }
      case "stage_change": {
        const from = d.from as RailStage;
        const to = d.to as RailStage;
        if (from) stageTransitions[from] = d.trigger || "";
        if (to) currentStage = to;
        pushDecision(
          "stage_transition",
          `${from ?? "?"} → ${to ?? "?"}${d.trigger ? " · " + d.trigger : ""}`,
        );
        break;
      }
      case "insight": {
        pushDecision("insight", d.message || d.insight || "insight", {
          insight: d.insight,
        });
        break;
      }
      case "ablation": {
        ablationEntries = (d.entries as AblationEntry[]) || [];
        if (typeof d.baseline_score === "number") baselineScore = d.baseline_score;
        if (typeof d.best_score === "number") bestScore = d.best_score;
        pushDecision(
          "stage_transition",
          `ablation complete · ${ablationEntries.length} params`,
        );
        break;
      }
      case "complete": {
        isComplete = true;
        if (typeof d.best_score === "number") bestScore = d.best_score;
        if (d.best_node_id) bestNodeId = d.best_node_id;
        if (d.best_config) bestConfig = d.best_config;
        if (d.best_metrics) bestMetrics = d.best_metrics;
        pushDecision(
          "success",
          `search complete · best ${bestScore?.toFixed(3) ?? "—"}`,
          { nodeId: bestNodeId ?? undefined, score: bestScore ?? undefined },
        );
        break;
      }
      default:
        break;
    }
  }

  const order: RailStage[] = [
    "preliminary",
    "baseline",
    "exploration",
    "ablation",
  ];
  const curIdx = order.indexOf(currentStage);
  const stageSummaries: StageSummary[] = order.map((stage, idx) => ({
    stage,
    nodeCount: stageCounts[stage],
    bestScore: stageBest[stage],
    status:
      isComplete || idx < curIdx
        ? "done"
        : idx === curIdx
        ? "active"
        : "pending",
    transitionTrigger: stageTransitions[stage],
  }));

  return {
    nodes: Array.from(nodesById.values()),
    decisions,
    trajectory,
    stageSummaries,
    currentStage,
    completedSteps: trajectory.length,
    totalSteps: Math.max(maxSteps, trajectory.length, 1),
    ablationEntries,
    baselineScore,
    bestScore,
    bestNodeId,
    bestConfig,
    bestMetrics,
    isComplete,
  };
}

async function createAndStartExperiment(form: ConfigForm): Promise<string> {
  const searchSpace: Record<string, boolean> = {};
  SEARCH_SPACE_KEYS.forEach((k) => (searchSpace[k] = !!form.searchSpace[k]));

  const body = {
    name: `Run · ${new Date().toLocaleTimeString()}`,
    strategy: form.strategy === "bfts" ? "bayesian" : form.strategy,
    search_space: searchSpace,
    max_iterations: form.maxSteps,
    num_seeds: form.numSeeds,
    cost_budget_usd: form.budgetUsd ? parseFloat(form.budgetUsd) : null,
  };

  const res = await fetch("/api/experiments/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`failed to create experiment (${res.status})`);
  const exp = await res.json();
  await fetch(`/api/experiments/${exp.id}/start/`, { method: "POST" });
  return exp.id as string;
}

export default function AutoOptimize() {
  const [form, setForm] = useState<ConfigForm>(DEFAULT_FORM);
  const [experimentId, setExperimentId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { events, status } = useAgentStream(experimentId);
  const { saveRun } = useRun();

  const derived = useMemo(
    () => (events.length === 0 ? EMPTY_STATE : deriveState(events, form.maxSteps)),
    [events, form.maxSteps],
  );

  // Persist snapshot as the run progresses, not only on completion.
  //
  // Previously this effect only fired when `isComplete` flipped true — so if
  // the backend errored, the user navigated away, or the tab was closed
  // mid-run, nothing was ever written to RunContext and the Results page
  // showed "no run yet". Now we save whenever the best score improves (and
  // once more on completion, which flushes ablation + final metrics).
  useEffect(() => {
    if (!experimentId) return;
    if (derived.bestScore === null && !derived.isComplete) return;
    const snapshot: RunSnapshot = {
      experimentId,
      label: `Run · ${new Date(Date.now()).toLocaleString()}`,
      completedAt: new Date().toISOString(),
      bestConfig: derived.bestConfig,
      bestMetrics: derived.bestMetrics,
      trajectory: derived.trajectory,
      ablation: derived.ablationEntries ?? [],
      tree: derived.nodes,
      bestNodeId: derived.bestNodeId,
      bestScore: derived.bestScore,
      baselineScore: derived.baselineScore,
    };
    saveRun(snapshot);
  }, [derived.bestScore, derived.isComplete, experimentId]); // eslint-disable-line react-hooks/exhaustive-deps

  const onRun = async () => {
    setSubmitError(null);
    setSubmitting(true);
    try {
      const id = await createAndStartExperiment(form);
      setExperimentId(id);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const toggleParam = (k: string) =>
    setForm((f) => ({
      ...f,
      searchSpace: { ...f.searchSpace, [k]: !f.searchSpace[k] },
    }));

  const running =
    submitting || (experimentId !== null && !derived.isComplete);

  return (
    <div style={pageStyle}>
      <h1 style={pageTitle}>Auto-Optimize</h1>
      <p style={pageSubtitle}>
        The BFTS agent explores the RAG search space progressively —
        preliminary → baseline → exploration → ablation — and surfaces the
        winning config once every stage converges. Watch the tree grow live.
      </p>

      <div style={layout}>
        <aside style={sidebarCard}>
          <h2 style={sectionTitle}>Run configuration</h2>

          <div style={fieldGroup}>
            <label style={fieldLabel}>Strategy</label>
            <select
              value={form.strategy}
              onChange={(e) =>
                setForm({ ...form, strategy: e.target.value as any })
              }
              style={{ width: "100%" }}
            >
              <option value="bfts">BFTS (tree search)</option>
              <option value="random">Random</option>
              <option value="greedy">Greedy</option>
            </select>
          </div>

          <div style={fieldGroup}>
            <label style={fieldLabel}>
              Max steps <span style={valueHint}>{form.maxSteps}</span>
            </label>
            <input
              type="range"
              min={5}
              max={30}
              value={form.maxSteps}
              onChange={(e) =>
                setForm({ ...form, maxSteps: parseInt(e.target.value, 10) })
              }
              style={{ width: "100%" }}
            />
          </div>

          <div style={fieldGroup}>
            <label style={fieldLabel}>
              Seed nodes <span style={valueHint}>{form.numSeeds}</span>
            </label>
            <input
              type="range"
              min={1}
              max={5}
              value={form.numSeeds}
              onChange={(e) =>
                setForm({ ...form, numSeeds: parseInt(e.target.value, 10) })
              }
              style={{ width: "100%" }}
            />
          </div>

          <div style={fieldGroup}>
            <label style={fieldLabel}>Budget (USD, optional)</label>
            <input
              type="number"
              step="0.01"
              min="0"
              placeholder="e.g. 2.50"
              value={form.budgetUsd}
              onChange={(e) => setForm({ ...form, budgetUsd: e.target.value })}
              style={{ width: "100%" }}
            />
          </div>

          <div style={fieldGroup}>
            <label style={fieldLabel}>Search space</label>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {SEARCH_SPACE_KEYS.map((k) => (
                <label key={k} style={checkRow}>
                  <input
                    type="checkbox"
                    checked={!!form.searchSpace[k]}
                    onChange={() => toggleParam(k)}
                  />
                  <span style={{ fontFamily: font.mono, fontSize: 12 }}>{k}</span>
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={onRun}
            disabled={running}
            style={{
              ...primaryButton,
              width: "100%",
              marginTop: space.md,
              opacity: running ? 0.6 : 1,
              cursor: running ? "not-allowed" : "pointer",
            }}
          >
            {submitting ? "Starting…" : running ? "Running…" : "Start run"}
          </button>

          {submitError && (
            <div style={{ ...callout("danger"), marginTop: space.sm }}>
              {submitError}
            </div>
          )}

          {experimentId && (
            <div style={statusLine}>
              <span style={{ color: colors.textFaint }}>experiment</span>
              <span style={{ fontFamily: font.mono, color: colors.text }}>
                {experimentId.slice(0, 8)}
              </span>
              <span style={chip(status === "open" ? "accent" : "neutral")}>
                {status}
              </span>
            </div>
          )}
        </aside>

        <div style={mainCol}>
          <StageRail
            currentStage={derived.currentStage}
            stageSummaries={derived.stageSummaries}
            totalSteps={derived.totalSteps}
            completedSteps={derived.completedSteps}
          />

          <div style={consoleGrid}>
            <div style={{ ...card, padding: space.md, minWidth: 0 }}>
              <div style={consoleHeader}>
                <span>Search tree</span>
                <span style={{ color: colors.textFaint, fontSize: 12, fontFamily: font.mono }}>
                  {derived.nodes.length} nodes
                </span>
              </div>
              <AgentTree
                nodes={derived.nodes}
                bestPath={derived.bestNodeId ? [derived.bestNodeId] : []}
                width={520}
                height={440}
              />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: space.lg, minWidth: 0 }}>
              <TrajectoryGraph
                points={derived.trajectory}
                baselineScore={derived.baselineScore}
                width={460}
                height={220}
              />
              <DecisionLog entries={derived.decisions} maxHeight={260} />
            </div>
          </div>

          {derived.ablationEntries && derived.bestScore !== null && (
            <div style={{ marginTop: space.lg }}>
              <AblationWaterfall
                entries={derived.ablationEntries}
                bestScore={derived.bestScore}
                baselineScore={derived.baselineScore ?? 0}
              />
            </div>
          )}

          {derived.isComplete && derived.bestScore !== null && (
            <div style={{ ...callout("success"), marginTop: space.lg }}>
              <div style={{ display: "flex", alignItems: "center", gap: space.sm, marginBottom: 4 }}>
                <span style={chip("success")}>done</span>
                <strong>
                  Best config · score {derived.bestScore.toFixed(3)}
                </strong>
              </div>
              <div style={{ fontSize: 12, color: colors.textMuted }}>
                Node {derived.bestNodeId?.slice(0, 8) ?? "—"} —{" "}
                {derived.baselineScore !== null && (
                  <>
                    +
                    {(derived.bestScore - derived.baselineScore).toFixed(3)} over
                    baseline
                  </>
                )}
              </div>
              <div style={{ display: "flex", gap: space.sm, marginTop: space.sm }}>
                <Link to="/results" style={primaryButton}>
                  View results →
                </Link>
                <Link to="/export" style={ghostButton}>
                  Export config
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const layout: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "260px 1fr",
  gap: space.lg,
  alignItems: "start",
};

const sidebarCard: CSSProperties = {
  ...card,
  position: "sticky",
  top: space.lg,
};

const mainCol: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: space.lg,
  minWidth: 0,
};

const consoleGrid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "minmax(0, 1.1fr) minmax(0, 1fr)",
  gap: space.lg,
};

const consoleHeader: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  fontSize: 13,
  fontWeight: 600,
  color: colors.text,
  marginBottom: space.xs,
};

const fieldGroup: CSSProperties = {
  marginTop: space.md,
};

const fieldLabel: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  fontSize: 12,
  color: colors.textMuted,
  fontWeight: 500,
  marginBottom: space.xs,
};

const valueHint: CSSProperties = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.textFaint,
};

const checkRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.xs,
  padding: `${space.xs - 2}px 0`,
  color: colors.text,
  cursor: "pointer",
};

const statusLine: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.xs,
  marginTop: space.md,
  padding: `${space.xs}px ${space.sm}px`,
  background: colors.bgSunken,
  borderRadius: radius.sm,
  fontSize: 12,
};
