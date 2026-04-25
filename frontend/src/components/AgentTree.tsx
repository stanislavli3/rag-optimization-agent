/**
 * AgentTree — live D3 force-directed tree of BFTS search nodes.
 *
 * Nodes + edges stream in from the SSE backend; the simulation re-settles on
 * every append so the tree visibly grows. Best path (root → best leaf) is
 * highlighted in amber; node fill encodes status, node ring encodes stage.
 * Notion-style: low-chroma palette, thin strokes, hover tooltip sits on a
 * near-white surface rather than a dark chip.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

import { colors, font, radius, space, stageColor } from "../theme";

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
  pending: colors.textFaint,
  running: colors.accent,
  success: colors.success,
  failed: colors.danger,
  pruned: colors.textMuted,
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

    const sim: Sim = (d3.forceSimulation(simNodes as any) as unknown as Sim)
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
        bestSet.has(d.source) && bestSet.has(d.target) ? colors.warn : colors.border,
      )
      .attr("stroke-width", (d: any) =>
        bestSet.has(d.source) && bestSet.has(d.target) ? 2.5 : 1,
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
      .attr("r", (d: any) => (bestSet.has(d.id) ? 13 : 10))
      .attr("fill", (d: any) => STATUS_FILL[d.status as TreeStatus] ?? colors.textFaint)
      .attr("stroke", (d: any) =>
        bestSet.has(d.id) ? colors.warn : stageColor(d.stage as TreeStage),
      )
      .attr("stroke-width", (d: any) => (bestSet.has(d.id) ? 2.5 : 1.5));

    // Running-node pulse ring
    node
      .filter((d: any) => d.status === "running")
      .append("circle")
      .attr("r", 13)
      .attr("fill", "none")
      .attr("stroke", colors.accent)
      .attr("stroke-width", 1.5)
      .attr("opacity", 0.7)
      .style("animation", "rag-tree-pulse 1.2s ease-in-out infinite");

    node
      .append("text")
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .style("font-size", "9px")
      .style("font-family", font.mono)
      .style("fill", "#ffffff")
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
      <style>{`@keyframes rag-tree-pulse { 0%,100%{ transform: scale(1); opacity:0.7 } 50% { transform: scale(1.4); opacity: 0 } }`}</style>
      <svg ref={svgRef} width={width} height={height} />
      {hover && (
        <div style={tooltip}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            <span style={{ fontFamily: font.mono }}>{hover.id.slice(0, 8)}</span>
            {hover.iteration_number !== undefined && (
              <span style={{ color: colors.textFaint, marginLeft: 8, fontWeight: 400 }}>
                iter {hover.iteration_number}
              </span>
            )}
          </div>
          <div style={tooltipRow}>
            <span style={tooltipKey}>stage</span>
            <span>{hover.stage}</span>
          </div>
          <div style={tooltipRow}>
            <span style={tooltipKey}>status</span>
            <span>{hover.status}</span>
          </div>
          <div style={tooltipRow}>
            <span style={tooltipKey}>score</span>
            <span style={{ fontFamily: font.mono }}>
              {hover.score?.toFixed(3) ?? "–"}
            </span>
          </div>
          <div style={{ marginTop: space.xs, color: colors.textMuted, fontSize: 11 }}>
            {Object.entries(hover.config)
              .map(([k, v]) => `${k}=${String(v)}`)
              .join(" · ")}
          </div>
        </div>
      )}
      <Legend />
    </div>
  );
}

function Legend() {
  const chipStyle: import("react").CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    marginRight: space.md,
    fontSize: 11,
    color: colors.textMuted,
  };
  const dot = (bg: string): import("react").CSSProperties => ({
    width: 8,
    height: 8,
    borderRadius: 4,
    background: bg,
    display: "inline-block",
  });
  return (
    <div style={legendWrap}>
      <span style={chipStyle}>
        <span style={dot(colors.success)} /> success
      </span>
      <span style={chipStyle}>
        <span style={dot(colors.danger)} /> failed
      </span>
      <span style={chipStyle}>
        <span style={dot(colors.accent)} /> running
      </span>
      <span style={chipStyle}>
        <span style={dot(colors.textMuted)} /> pruned
      </span>
      <span style={{ color: colors.warn, fontSize: 11 }}>● best path</span>
    </div>
  );
}

const tooltip: import("react").CSSProperties = {
  position: "absolute",
  top: 8,
  right: 8,
  background: colors.bg,
  color: colors.text,
  padding: `${space.sm}px ${space.md}px`,
  borderRadius: radius.md,
  fontSize: 12,
  border: `1px solid ${colors.border}`,
  boxShadow: "0 4px 12px rgba(15, 15, 15, 0.06)",
  pointerEvents: "none",
  maxWidth: 300,
  lineHeight: 1.4,
};

const tooltipRow: import("react").CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  gap: space.md,
};

const tooltipKey: import("react").CSSProperties = {
  color: colors.textFaint,
  fontFamily: font.mono,
  fontSize: 11,
};

const legendWrap: import("react").CSSProperties = {
  marginTop: space.sm,
  color: colors.textMuted,
  fontSize: 11,
};
