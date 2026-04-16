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
 * sub-components. Reducer-style derivation here keeps sub-components pure and
 * avoids fan-out of network logic.
 */
import { CSSProperties, useMemo, useState } from "react";

import AblationWaterfall, {
  AblationEntry,
} from "../components/AblationWaterfall";
import AgentTree, { TreeNode, TreeStage, TreeStatus } from "../components/AgentTree";
import DecisionLog, { DecisionEntry, DecisionType } from "../components/DecisionLog";
import StageRail, { RailStage, StageSummary } from "../components/StageRail";
import TrajectoryGraph, {
  TrajectoryPoint,
} from "../components/TrajectoryGraph";
import useAgentStream, { AgentStreamEvent } from "../hooks/useAgentStream";


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
  isComplete: false,
};


function deriveState(events: AgentStreamEvent[], maxSteps: number): DerivedState {
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
    const d: any = ev.data ?? {};
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
        pushDecision("expand", `expanding node (${Object.keys(node.config).length} params)`, {
          nodeId: node.id,
        });
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

  // Finalise stage statuses based on current stage ordering.
  const order: RailStage[] = ["preliminary", "baseline", "exploration", "ablation"];
  const curIdx = order.indexOf(currentStage);
  const stageSummaries: StageSummary[] = order.map((stage, idx) => ({
    stage,
    nodeCount: stageCounts[stage],
    bestScore: stageBest[stage],
    status:
      isComplete || idx < curIdx ? "done" : idx === curIdx ? "active" : "pending",
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

  const derived = useMemo(
    () => (events.length === 0 ? EMPTY_STATE : deriveState(events, form.maxSteps)),
    [events, form.maxSteps],
  );

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

  return (
    <div style={page}>
      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <aside style={sidebar}>
          <h3 style={sidebarTitle}>Run configuration</h3>

          <label style={label}>Strategy</label>
          <select
            value={form.strategy}
            onChange={(e) => setForm({ ...form, strategy: e.target.value as any })}
            style={input}
          >
            <option value="bfts">BFTS (tree search)</option>
            <option value="random">Random</option>
            <option value="greedy">Greedy</option>
          </select>

          <label style={label}>Max steps: {form.maxSteps}</label>
          <input
            type="range"
            min={5}
            max={30}
            value={form.maxSteps}
            onChange={(e) =>
              setForm({ ...form, maxSteps: parseInt(e.target.value, 10) })
            }
          />

          <label style={label}>Seed nodes: {form.numSeeds}</label>
          <input
            type="range"
            min={1}
            max={5}
            value={form.numSeeds}
            onChange={(e) =>
              setForm({ ...form, numSeeds: parseInt(e.target.value, 10) })
            }
          />

          <label style={label}>Budget (USD, optional)</label>
          <input
            type="number"
            step="0.01"
            min="0"
            placeholder="e.g. 2.50"
            value={form.budgetUsd}
            onChange={(e) => setForm({ ...form, budgetUsd: e.target.value })}
            style={input}
          />

          <label style={label}>Search space</label>
          <div>
            {SEARCH_SPACE_KEYS.map((k) => (
              <label key={k} style={checkRow}>
                <input
                  type="checkbox"
                  checked={!!form.searchSpace[k]}
                  onChange={() => toggleParam(k)}
                />{" "}
                <span>{k}</span>
              </label>
            ))}
          </div>

          <button
            onClick={onRun}
            disabled={submitting || (experimentId !== null && !derived.isComplete)}
            style={runBtn}
          >
            {submitting
              ? "Starting…"
              : experimentId && !derived.isComplete
              ? "Running"
              : "Run"}
          </button>
          {submitError && <div style={errorBox}>{submitError}</div>}
          {experimentId && (
            <div style={statusLine}>
              experiment {experimentId.slice(0, 8)} · stream {status}
            </div>
          )}
        </aside>

        <main style={{ flex: 1, minWidth: 0 }}>
          <StageRail
            currentStage={derived.currentStage}
            stageSummaries={derived.stageSummaries}
            totalSteps={derived.totalSteps}
            completedSteps={derived.completedSteps}
          />

          <div style={{ display: "flex", gap: 16, marginTop: 16 }}>
            <div style={{ flex: 1.2, minWidth: 0 }}>
              <div style={panel}>
                <div style={panelHeader}>Search tree</div>
                <AgentTree
                  nodes={derived.nodes}
                  bestPath={derived.bestNodeId ? [derived.bestNodeId] : []}
                  width={520}
                  height={460}
                />
              </div>
            </div>

            <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
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
            <div style={{ marginTop: 16 }}>
              <AblationWaterfall
                entries={derived.ablationEntries}
                bestScore={derived.bestScore}
                baselineScore={derived.baselineScore ?? 0}
              />
            </div>
          )}

          {derived.isComplete && derived.bestScore !== null && (
            <div style={bestCard}>
              <div style={{ fontWeight: 700, fontSize: 15 }}>
                Best config · score {derived.bestScore.toFixed(3)}
              </div>
              <div style={{ fontSize: 12, color: "#475569", marginTop: 4 }}>
                node {derived.bestNodeId?.slice(0, 8) ?? "—"}
              </div>
              <a href="/results" style={resultsLink}>
                View Results →
              </a>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

const page: CSSProperties = {
  padding: 20,
  background: "#f8fafc",
  minHeight: "100vh",
  fontFamily: "system-ui, -apple-system, sans-serif",
};

const sidebar: CSSProperties = {
  width: 260,
  flexShrink: 0,
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 16,
  position: "sticky",
  top: 20,
};

const sidebarTitle: CSSProperties = {
  fontSize: 14,
  margin: 0,
  marginBottom: 12,
  color: "#0f172a",
};

const label: CSSProperties = {
  display: "block",
  fontSize: 12,
  color: "#475569",
  marginTop: 12,
  marginBottom: 4,
  fontWeight: 600,
};

const input: CSSProperties = {
  width: "100%",
  padding: "6px 8px",
  border: "1px solid #cbd5e1",
  borderRadius: 4,
  fontSize: 13,
  boxSizing: "border-box",
};

const checkRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  fontSize: 12,
  color: "#334155",
  padding: "2px 0",
};

const runBtn: CSSProperties = {
  marginTop: 16,
  width: "100%",
  padding: "8px 12px",
  background: "#1e293b",
  color: "#f8fafc",
  border: "none",
  borderRadius: 4,
  fontWeight: 600,
  cursor: "pointer",
};

const errorBox: CSSProperties = {
  marginTop: 8,
  padding: "6px 8px",
  background: "#fef2f2",
  color: "#b91c1c",
  borderRadius: 4,
  fontSize: 12,
};

const statusLine: CSSProperties = {
  marginTop: 8,
  fontSize: 11,
  color: "#64748b",
  fontFamily: "monospace",
};

const panel: CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 12,
};

const panelHeader: CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: "#0f172a",
  marginBottom: 8,
};

const bestCard: CSSProperties = {
  marginTop: 16,
  padding: 16,
  background: "#ecfdf5",
  border: "1px solid #a7f3d0",
  borderRadius: 8,
};

const resultsLink: CSSProperties = {
  display: "inline-block",
  marginTop: 8,
  color: "#047857",
  fontWeight: 600,
  fontSize: 13,
  textDecoration: "none",
};
