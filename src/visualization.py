"""Centralised Plotly figure builders. All use plotly.graph_objects."""
from __future__ import annotations

from typing import Any

_STAGE_COLORS = {
    "PRELIMINARY": "#7dd3fc",
    "BASELINE": "#60a5fa",
    "EXPLORATION": "#a78bfa",
    "ABLATION": "#f59e0b",
    "preliminary": "#7dd3fc",
    "baseline": "#60a5fa",
    "exploration": "#a78bfa",
    "ablation": "#f59e0b",
}

_STATUS_COLORS = {
    "pending": "#94a3b8",
    "running": "#3b82f6",
    "success": "#22c55e",
    "failed": "#ef4444",
    "pruned": "#6b7280",
}


def _go():
    import plotly.graph_objects as go  # type: ignore
    return go


_STAGE_INT_TO_NAME = {0: "PRELIMINARY", 1: "BASELINE", 2: "EXPLORATION", 3: "ABLATION"}


def _stage_name(raw: Any) -> str:
    """Normalise a stage value (IntEnum, int, or str) to an upper-case label."""
    if raw is None:
        return "BFTS"
    name = getattr(raw, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    if isinstance(raw, int):
        return _STAGE_INT_TO_NAME.get(int(raw), f"STAGE_{raw}")
    return str(raw).upper() or "BFTS"


def plot_trajectory(trajectory: list[dict], stage_transitions: list[dict] | None = None):
    go = _go()
    xs = list(range(len(trajectory)))
    ys = [float(n.get("score") or 0.0) for n in trajectory]
    stages = [_stage_name(n.get("stage")) for n in trajectory]
    colors = [_STAGE_COLORS.get(s, "#3b82f6") for s in stages]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#111"), marker=dict(color=colors, size=10), name="score"))

    if stage_transitions:
        for t in stage_transitions:
            fig.add_vline(x=t.get("at_step", 0), line_dash="dash", line_color="#9ca3af", annotation_text=t.get("to", ""))

    fig.update_layout(
        title="Score trajectory",
        xaxis_title="iteration",
        yaxis_title="score",
        template="plotly_white",
        height=380,
    )
    return fig


def plot_radar(metrics_dict: dict, baseline: dict | None = None):
    go = _go()
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
    best_values = [float(metrics_dict.get(m, 0.0)) for m in metrics]
    traces = [
        go.Scatterpolar(r=best_values + [best_values[0]], theta=metrics + [metrics[0]], fill="toself", name="best"),
    ]
    if baseline:
        base_values = [float(baseline.get(m, 0.0)) for m in metrics]
        traces.append(
            go.Scatterpolar(
                r=base_values + [base_values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name="baseline",
                opacity=0.45,
            )
        )
    fig = go.Figure(traces)
    fig.update_layout(
        title="RAGAS metrics",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=420,
    )
    return fig


def plot_difficulty_heatmap(testset_df):
    import pandas as pd  # type: ignore

    go = _go()
    df = testset_df.copy()
    df["dist_bucket"] = pd.cut(
        df["semantic_distance"].astype(float),
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["low (<0.3)", "mid (0.3-0.6)", "high (>0.6)"],
    )
    pivot = (
        df.groupby(["reasoning_depth", "dist_bucket"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[f"depth={i}" for i in pivot.index],
            colorscale="Viridis",
            colorbar_title="count",
        )
    )
    fig.update_layout(title="Difficulty matrix (reasoning × vocabulary distance)", height=360)
    return fig


def plot_ablation_bar(ablation_report: list[dict]):
    go = _go()
    report = sorted(ablation_report, key=lambda r: r.get("delta", 0.0))
    names = [r["param"] for r in report]
    deltas = [r.get("delta", 0.0) for r in report]
    colors = ["#dc2626" if d > 0.15 else ("#f59e0b" if d > 0.05 else "#64748b") for d in deltas]

    fig = go.Figure(
        go.Bar(x=deltas, y=names, orientation="h", marker_color=colors, text=[f"{d:+.3f}" for d in deltas], textposition="outside")
    )
    fig.update_layout(title="Ablation — score drop when component reverted to default", height=320, template="plotly_white", xaxis_title="Δ score")
    return fig


def plot_tree(tree_data: dict):
    """Tree layout using a simple BFS level layout on plotly scatter."""
    go = _go()
    nodes = tree_data.get("nodes", [])
    edges = tree_data.get("edges", [])
    best_path = set(tree_data.get("best_path", []))

    # Level-based layout: group by depth, spread horizontally
    by_depth: dict[int, list[dict]] = {}
    for n in nodes:
        by_depth.setdefault(int(n.get("depth", 0)), []).append(n)
    positions: dict[str, tuple[float, float]] = {}
    for depth, level_nodes in by_depth.items():
        for i, n in enumerate(level_nodes):
            x = (i - (len(level_nodes) - 1) / 2) * 1.5
            positions[n["id"]] = (x, -float(depth))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for e in edges:
        if e["source"] in positions and e["target"] in positions:
            x0, y0 = positions[e["source"]]
            x1, y1 = positions[e["target"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    xs, ys, colors, texts, borders = [], [], [], [], []
    for n in nodes:
        if n["id"] not in positions:
            continue
        x, y = positions[n["id"]]
        xs.append(x)
        ys.append(y)
        colors.append(_STATUS_COLORS.get(n.get("status", "pending"), "#94a3b8"))
        score = n.get("score")
        score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
        texts.append(f"{n['id'][:6]}<br>{n.get('stage', '')}<br>score={score_text}")
        borders.append("gold" if n["id"] in best_path else "rgba(0,0,0,0.3)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#cbd5e1", width=1), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(color=colors, size=22, line=dict(color=borders, width=2)),
            text=texts,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Search tree",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        height=500,
    )
    return fig


__all__ = [
    "plot_trajectory",
    "plot_radar",
    "plot_difficulty_heatmap",
    "plot_ablation_bar",
    "plot_tree",
]
