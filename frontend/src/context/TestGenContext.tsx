/**
 * TestGenContext — localStorage-backed state for the Upload & TestGen page.
 *
 * Keeps uploaded-doc metadata (name/size), user configuration (size +
 * distribution), step-card progress, generated questions, and the CSV link
 * around when the user navigates to another tab. File blobs themselves are
 * kept in memory only — localStorage can't hold them and base64-encoding a
 * PDF would blow the quota. If the user wants to re-run generation after a
 * reload, they re-drop the files; existing results still display.
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

export type StepKey =
  | "knowledge_graph"
  | "seeds"
  | "evolution"
  | "filter"
  | "difficulty";

export type StepStatus = "pending" | "running" | "done" | "failed";

export interface StepState {
  key: StepKey;
  title: string;
  subtitle: string;
  status: StepStatus;
  stats: Record<string, number | string>;
}

export interface DocMeta {
  name: string;
  size: number;
}

export interface TypeDistribution {
  simple: number;
  multi_context: number;
  reasoning: number;
  conditional: number;
}

export interface GeneratedQuestion {
  question: string;
  question_type: string;
  difficulty: "easy" | "medium" | "hard";
  reasoning_depth?: number;
  semantic_distance?: number;
}

export interface TestGenSnapshot {
  docs: DocMeta[];
  numQuestions: number;
  dist: TypeDistribution;
  steps: StepState[];
  jobId: string | null;
  questions: GeneratedQuestion[];
  csvUrl: string | null;
  completedAt: string | null;
}

export const INITIAL_STEPS: StepState[] = [
  {
    key: "knowledge_graph",
    title: "Knowledge Graph",
    subtitle: "Entity & relation extraction",
    status: "pending",
    stats: {},
  },
  {
    key: "seeds",
    title: "Seed Questions",
    subtitle: "Grounded Q&A from KG facts",
    status: "pending",
    stats: {},
  },
  {
    key: "evolution",
    title: "Evol-Instruct",
    subtitle: "Multi-context, reasoning, conditional",
    status: "pending",
    stats: {},
  },
  {
    key: "filter",
    title: "Groundedness Filter",
    subtitle: "LLM-as-judge quality gate",
    status: "pending",
    stats: {},
  },
  {
    key: "difficulty",
    title: "Difficulty Matrix",
    subtitle: "Depth × semantic distance",
    status: "pending",
    stats: {},
  },
];

const DEFAULT_SNAPSHOT: TestGenSnapshot = {
  docs: [],
  numQuestions: 20,
  dist: { simple: 30, multi_context: 30, reasoning: 25, conditional: 15 },
  steps: INITIAL_STEPS,
  jobId: null,
  questions: [],
  csvUrl: null,
  completedAt: null,
};

interface TestGenContextValue {
  snapshot: TestGenSnapshot;
  setDocs: (docs: DocMeta[]) => void;
  setNumQuestions: (n: number) => void;
  setDist: (d: TypeDistribution) => void;
  setSteps: (updater: StepState[] | ((prev: StepState[]) => StepState[])) => void;
  setJobId: (id: string | null) => void;
  setQuestions: (qs: GeneratedQuestion[]) => void;
  setCsvUrl: (url: string | null) => void;
  markComplete: () => void;
  resetRun: () => void;
  clearAll: () => void;
}

const LS_KEY = "rag-optimizer:testgen:v1";

const Ctx = createContext<TestGenContextValue | null>(null);

function loadFromStorage(): TestGenSnapshot {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return DEFAULT_SNAPSHOT;
    const parsed = JSON.parse(raw);
    return {
      ...DEFAULT_SNAPSHOT,
      ...parsed,
      // Guard against shape drift
      steps: Array.isArray(parsed.steps) && parsed.steps.length === INITIAL_STEPS.length
        ? parsed.steps
        : INITIAL_STEPS,
      docs: Array.isArray(parsed.docs) ? parsed.docs : [],
      questions: Array.isArray(parsed.questions) ? parsed.questions : [],
    };
  } catch {
    return DEFAULT_SNAPSHOT;
  }
}

export function TestGenProvider({ children }: { children: ReactNode }) {
  const initial = useMemo(loadFromStorage, []);
  const [snapshot, setSnapshot] = useState<TestGenSnapshot>(initial);

  useEffect(() => {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(snapshot));
    } catch {
      // Quota exceeded — drop the heavy parts and retry once.
      try {
        localStorage.setItem(
          LS_KEY,
          JSON.stringify({ ...snapshot, questions: snapshot.questions.slice(0, 50) }),
        );
      } catch {
        // Give up silently.
      }
    }
  }, [snapshot]);

  const setDocs = useCallback(
    (docs: DocMeta[]) => setSnapshot((p) => ({ ...p, docs })),
    [],
  );
  const setNumQuestions = useCallback(
    (numQuestions: number) => setSnapshot((p) => ({ ...p, numQuestions })),
    [],
  );
  const setDist = useCallback(
    (dist: TypeDistribution) => setSnapshot((p) => ({ ...p, dist })),
    [],
  );
  const setSteps = useCallback(
    (updater: StepState[] | ((prev: StepState[]) => StepState[])) =>
      setSnapshot((p) => ({
        ...p,
        steps: typeof updater === "function" ? updater(p.steps) : updater,
      })),
    [],
  );
  const setJobId = useCallback(
    (jobId: string | null) => setSnapshot((p) => ({ ...p, jobId })),
    [],
  );
  const setQuestions = useCallback(
    (questions: GeneratedQuestion[]) => setSnapshot((p) => ({ ...p, questions })),
    [],
  );
  const setCsvUrl = useCallback(
    (csvUrl: string | null) => setSnapshot((p) => ({ ...p, csvUrl })),
    [],
  );
  const markComplete = useCallback(
    () => setSnapshot((p) => ({ ...p, completedAt: new Date().toISOString() })),
    [],
  );
  const resetRun = useCallback(
    () =>
      setSnapshot((p) => ({
        ...p,
        steps: INITIAL_STEPS,
        jobId: null,
        questions: [],
        csvUrl: null,
        completedAt: null,
      })),
    [],
  );
  const clearAll = useCallback(() => setSnapshot(DEFAULT_SNAPSHOT), []);

  const value = useMemo(
    () => ({
      snapshot,
      setDocs,
      setNumQuestions,
      setDist,
      setSteps,
      setJobId,
      setQuestions,
      setCsvUrl,
      markComplete,
      resetRun,
      clearAll,
    }),
    [
      snapshot,
      setDocs,
      setNumQuestions,
      setDist,
      setSteps,
      setJobId,
      setQuestions,
      setCsvUrl,
      markComplete,
      resetRun,
      clearAll,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTestGen(): TestGenContextValue {
  const v = useContext(Ctx);
  if (!v) throw new Error("useTestGen must be called inside <TestGenProvider>");
  return v;
}
