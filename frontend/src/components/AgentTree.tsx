/**
 * AgentTree — live D3 force-directed tree of BFTS search nodes.
 *
 * Feed it nodes + edges as they stream in from the SSE backend; the simulation
 * re-settles on each append so the tree visibly grows in real time. Best path
 * (root → best leaf) is highlighted in gold; node colour encodes status, ring
 * colour encodes stage.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

export type TreeStage = "preliminary" | "baseline" | "exploration" | "ablation";
export type TreeStatus = "pending" | "running" | "success" | "failed" | "pruned";

export interface TreeNode {
  id: string;
  parent_id: string | null;
  stage: TreeStage;
  status: TreeStatus;
  config: Record<string, unknown>;
  score: number | null;
  depth: number;
  iteration_number?: number;
}

export interface AgentTreeProps {
  nodes: TreeNode[];
  bestPath?: string[];
  width?: number;
  height?: number;
  onNodeClick?: (node: TreeNode) => void;
}

const STATUS_FILL: Record<TreeStatus, string> = {
  pending: "#94a3b8",
  running: "#3b82f6",
  success: "#22c55e",
  failed: "#ef4444",
  pruned: "#64748b",
};

const STAGE_RING: Record<TreeStage, string> = {
  preliminary: "#7dd3fc",
  baseline: "#60a5fa",
  exploration: "#a78bfa",
  ablation: "#f59e0b",
};

type Sim = d3.Simulation<d3.SimulationNodeDatum & TreeNode, undefined>;

export default function AgentTree({
  nodes,
  bestPath = [],
  width = 720,
  height = 500,
  onNodeClick,
}: AgentTreeProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<TreeNode | null>(null);
  const bestSet = useMemo(() => new Set(bestPath), [bestPath]);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .append("g")
      .attr("transform", `translate(${width / 2},40)`);

    const simNodes = nodes.map((n) => ({ ...n }));
    const byId = new Map(simNodes.map((n) => [n.id, n]));
    const links = simNodes
      .filter((n) => n.parent_id && byId.has(n.parent_id))
      .map((n) => ({ source: n.parent_id!, target: n.id }));

    const sim: Sim = d3
      .forceSimulation(simNodes as any)
      .force("charge", d3.forceManyBody().strength(-160))
      .force(
        "link",
        d3
          .forceLink(links as any)
          .id((d: any) => d.id)
          .distance(60)
          .strength(0.7),
      )
      .force("y", d3.forceY((d: any) => d.depth * 70).strength(0.9))
      .force("x", d3.forceX(0).strength(0.05))
      .force("collision", d3.forceCollide(18));

    const link = g
      .append("g")
      .attr("stroke-linecap", "round")
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", (d: any) =>
        bestSet.has(d.source) && bestSet.has(d.target) ? "#eab308" : "#cbd5e1",
      )
      .attr("stroke-width", (d: any) =>
        bestSet.has(d.source) && bestSet.has(d.target) ? 3 : 1,
      );

    const node = g
      .append("g")
      .selectAll("g.node")
      .data(simNodes as any)
      .enter()
      .append("g")
      .attr("class", "node")
      .style("cursor", "pointer")
      .on("mouseenter", (_, d: any) => setHover(d))
      .on("mouseleave", () => setHover(null))
      .on("click", (_, d: any) => onNodeClick?.(d));

    node
      .append("circle")
      .attr("r", (d: any) => (bestSet.has(d.id) ? 14 : 11))
      .attr("fill", (d: any) => STATUS_FILL[d.status as TreeStatus] ?? "#cbd5e1")
      .attr("stroke", (d: any) =>
        bestSet.has(d.id) ? "#eab308" : STAGE_RING[d.stage as TreeStage] ?? "#94a3b8",
      )
      .attr("stroke-width", (d: any) => (bestSet.has(d.id) ? 3 : 2));

    // Running-node pulse
    node
      .filter((d: any) => d.status === "running")
      .append("circle")
      .attr("r", 14)
      .attr("fill", "none")
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 2)
      .attr("opacity", 0.7)
      .style("animation", "pulse 1.2s ease-in-out infinite");

    node
      .append("text")
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .style("font-size", "9px")
      .style("fill", "#f8fafc")
      .style("pointer-events", "none")
      .text((d: any) =>
        typeof d.score === "number" ? d.score.toFixed(2) : "·",
      );

    sim.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);
      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => {
      sim.stop();
    };
  }, [nodes, bestSet, width, height, onNodeClick]);

  return (
    <div style={{ position: "relative" }}>
      <style>{`@keyframes pulse { 0%,100%{ transform: scale(1); opacity:0.7 } 50% { transform: scale(1.35); opacity: 0 } }`}</style>
      <svg ref={svgRef} width={width} height={height} />
      {hover && (
        <div
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            background: "#0f172a",
            color: "#e2e8f0",
            padding: "8px 12px",
            borderRadius: 6,
            fontSize: 12,
            fontFamily: "monospace",
            pointerEvents: "none",
            maxWidth: 280,
          }}
        >
          <div>
            <strong>{hover.id.slice(0, 8)}</strong>
            {hover.iteration_number !== undefined && ` · iter ${hover.iteration_number}`}
          </div>
          <div>stage: {hover.stage}</div>
          <div>status: {hover.status}</div>
          <div>score: {hover.score?.toFixed(3) ?? "–"}</div>
          <div style={{ marginTop: 4, opacity: 0.8 }}>
            {Object.entries(hover.config)
              .map(([k, v]) => `${k}=${String(v)}`)
              .join(", ")}
          </div>
        </div>
      )}
      <Legend />
    </div>
  );
}

function Legend() {
  const chip = (bg: string, label: string) => (
    <span
      key={label}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        marginRight: 12,
        fontSize: 12,
      }}
    >
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: 5,
          background: bg,
          display: "inline-block",
        }}
      />
      {label}
    </span>
  );
  return (
    <div style={{ marginTop: 8, color: "#475569" }}>
      {chip("#22c55e", "success")}
      {chip("#ef4444", "failed")}
      {chip("#3b82f6", "running")}
      {chip("#64748b", "pruned")}
      <span style={{ color: "#eab308", fontSize: 12 }}>● best path</span>
    </div>
  );
}
