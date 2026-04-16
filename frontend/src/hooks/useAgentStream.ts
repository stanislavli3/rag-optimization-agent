/**
 * useAgentStream — subscribe to a BFTS experiment's SSE event feed.
 *
 * Opens an EventSource against `/api/experiments/{id}/stream/`, accumulates
 * decoded events, and returns both the full event list and a small derived
 * snapshot (`status`, `lastError`) that the Agent Console uses to drive its UI
 * state machine. Closes the underlying connection on unmount or when the
 * backend emits a `stream_end` frame.
 */
import { useEffect, useRef, useState } from "react";

export type AgentEventType =
  | "node_start"
  | "node_success"
  | "node_failed"
  | "node_pruned"
  | "stage_change"
  | "ablation"
  | "insight"
  | "complete"
  | "stream_end"
  | "error";

export interface AgentStreamEvent<T = unknown> {
  id?: number;
  event: AgentEventType;
  data: T;
}

export type StreamStatus = "idle" | "open" | "closed" | "error";

export interface UseAgentStreamResult {
  events: AgentStreamEvent[];
  status: StreamStatus;
  lastError: string | null;
  reset: () => void;
}

export function useAgentStream(
  experimentId: string | null,
  baseUrl = "",
): UseAgentStreamResult {
  const [events, setEvents] = useState<AgentStreamEvent[]>([]);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [lastError, setLastError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!experimentId) return;
    const url = `${baseUrl}/api/experiments/${experimentId}/stream/`;
    const es = new EventSource(url);
    esRef.current = es;
    setStatus("open");
    setLastError(null);

    es.onmessage = (msg) => {
      try {
        const parsed: AgentStreamEvent = JSON.parse(msg.data);
        setEvents((prev) => [...prev, parsed]);
        if (parsed.event === "stream_end") {
          es.close();
          setStatus("closed");
        }
      } catch (err) {
        setLastError(err instanceof Error ? err.message : String(err));
      }
    };

    es.onerror = () => {
      setStatus("error");
      setLastError("EventSource connection error");
      // Let the browser auto-reconnect for transient errors; close only on
      // explicit stream_end.
    };

    return () => {
      es.close();
      esRef.current = null;
      setStatus("closed");
    };
  }, [experimentId, baseUrl]);

  const reset = () => {
    esRef.current?.close();
    esRef.current = null;
    setEvents([]);
    setStatus("idle");
    setLastError(null);
  };

  return { events, status, lastError, reset };
}

export default useAgentStream;
