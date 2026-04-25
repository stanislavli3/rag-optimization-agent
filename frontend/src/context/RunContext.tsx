/**
 * RunContext — in-memory + localStorage cache of the last completed optimizer
 * run plus a short history of previous runs.
 *
 * Every page that needs post-run data (Results, Export, Trajectory, Comparison)
 * reads from here instead of refetching. AutoOptimize writes when a run is
 * complete. Persistence is localStorage so a refresh doesn't destroy results.
 */
import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import { AblationEntry } from "../components/AblationWaterfall";
import { TreeNode } from "../components/AgentTree";
import { TrajectoryPoint } from "../components/TrajectoryGraph";

export interface RunSnapshot {
  experimentId: string;
  label: string;
  completedAt: string;
  bestConfig: Record<string, unknown>;
  bestMetrics: Record<string, any>;
  trajectory: TrajectoryPoint[];
  ablation: AblationEntry[];
  tree: TreeNode[];
  bestNodeId: string | null;
  bestScore: number | null;
  baselineScore: number | null;
}

interface RunContextValue {
  current: RunSnapshot | null;
  history: RunSnapshot[];
  saveRun: (run: RunSnapshot) => void;
  clear: () => void;
}

const LS_KEY = "rag-optimizer:runs:v1";
const MAX_HISTORY = 10;

const Ctx = createContext<RunContextValue | null>(null);

function loadFromStorage(): { current: RunSnapshot | null; history: RunSnapshot[] } {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { current: null, history: [] };
    const parsed = JSON.parse(raw);
    return {
      current: parsed.current ?? null,
      history: Array.isArray(parsed.history) ? parsed.history : [],
    };
  } catch {
    return { current: null, history: [] };
  }
}

export function RunProvider({ children }: { children: ReactNode }) {
  const initial = useMemo(loadFromStorage, []);
  const [current, setCurrent] = useState<RunSnapshot | null>(initial.current);
  const [history, setHistory] = useState<RunSnapshot[]>(initial.history);

  useEffect(() => {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify({ current, history }));
    } catch {
      // Quota exceeded — drop oldest history entries and retry once.
      try {
        localStorage.setItem(
          LS_KEY,
          JSON.stringify({ current, history: history.slice(0, 3) }),
        );
      } catch {
        // Give up silently rather than crash the app.
      }
    }
  }, [current, history]);

  const saveRun = useCallback((run: RunSnapshot) => {
    setCurrent(run);
    setHistory((prev) => {
      const without = prev.filter((r) => r.experimentId !== run.experimentId);
      return [run, ...without].slice(0, MAX_HISTORY);
    });
  }, []);

  const clear = useCallback(() => {
    setCurrent(null);
    setHistory([]);
  }, []);

  const value = useMemo(
    () => ({ current, history, saveRun, clear }),
    [current, history, saveRun, clear],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useRun(): RunContextValue {
  const v = useContext(Ctx);
  if (!v) throw new Error("useRun must be called inside <RunProvider>");
  return v;
}
