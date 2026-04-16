"""Streamlit 4-page UI: upload & testgen → optimize → results → export."""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import BFTSConfig, Config, TestGenConfig
from src.evaluation.ragas_eval import evaluate_by_difficulty, evaluate_ragas
from src.ingest.loader import load_documents
from src.optimizer.bfts_loop import BFTSLoop
from src.pipeline.runner import run_pipeline
from src.testgen.llm import get_llm
from src.testgen.pipeline import TestGenPipeline
from src.visualization import (
    plot_ablation_bar,
    plot_difficulty_heatmap,
    plot_radar,
    plot_trajectory,
    plot_tree,
)

st.set_page_config(page_title="RAG Optimizer", layout="wide")


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("testset", None)
    ss.setdefault("documents", None)
    ss.setdefault("best_config", None)
    ss.setdefault("best_metrics", None)
    ss.setdefault("trajectory", None)
    ss.setdefault("stage_transitions", None)
    ss.setdefault("ablation_report", None)
    ss.setdefault("tree_data", None)


_init_state()


PAGE = st.sidebar.radio("Page", ["1 · Upload & TestGen", "2 · Run Optimization", "3 · Results", "4 · Export"])


# ---- Page 1 --------------------------------------------------------------------
def page_upload() -> None:
    st.title("1 · Upload & Generate Test Data")
    uploads = st.file_uploader("Documents (PDF / MD / TXT)", accept_multiple_files=True)
    st.markdown("**Question type distribution** (should sum to 1.0)")
    c1, c2, c3, c4 = st.columns(4)
    d_simple = c1.slider("simple", 0.0, 1.0, 0.30, 0.05)
    d_multi = c2.slider("multi_context", 0.0, 1.0, 0.30, 0.05)
    d_reason = c3.slider("reasoning", 0.0, 1.0, 0.25, 0.05)
    d_cond = c4.slider("conditional", 0.0, 1.0, 0.15, 0.05)
    st.caption(f"Sum = {d_simple + d_multi + d_reason + d_cond:.2f}")

    size = st.slider("Testset size", 5, 50, 20)
    run_btn = st.button("Generate test questions", type="primary", disabled=not uploads)

    if run_btn and uploads:
        tmp = Path(tempfile.mkdtemp(prefix="ragopt_"))
        for f in uploads:
            (tmp / f.name).write_bytes(f.getbuffer())
        docs = load_documents(tmp)
        st.session_state["documents"] = docs

        tg_cfg = TestGenConfig(target_size=size, distribution={"simple": d_simple, "multi_context": d_multi, "reasoning": d_reason, "conditional": d_cond})
        pipe = TestGenPipeline(
            llm=get_llm(),
            target_size=size,
            distribution=tg_cfg.distribution,
            groundedness_threshold=tg_cfg.groundedness_threshold,
            out_dir=Config().testset_dir,
        )
        progress = st.progress(0, text="Starting…")
        log = st.empty()
        lines: list[str] = []
        steps_total = 6
        steps_done = 0
        final_df = None
        for ev in pipe.generate_with_progress(docs):
            step = ev.get("step")
            status = ev.get("status")
            stats = ev.get("stats") or {}
            if status == "done":
                steps_done += 1
                lines.append(f"✓ **{step}** — {stats}")
                log.markdown("\n".join(lines))
                progress.progress(min(1.0, steps_done / steps_total), text=f"{step}: {status}")
            if ev.get("result") is not None:
                final_df = ev["result"]

        if final_df is not None:
            st.session_state["testset"] = final_df
            st.success(f"Generated {len(final_df)} questions")

    if st.session_state["testset"] is not None:
        df = st.session_state["testset"]
        st.subheader("Testset preview")
        st.dataframe(df[["question", "question_type", "difficulty", "reasoning_depth", "semantic_distance"]].head(30))

        col1, col2 = st.columns(2)
        with col1:
            by_type = df["question_type"].value_counts().reset_index()
            by_type.columns = ["type", "count"]
            st.bar_chart(by_type, x="type", y="count")
        with col2:
            st.plotly_chart(plot_difficulty_heatmap(df), use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download testset.csv", csv, file_name="testset.csv", mime="text/csv")


# ---- Page 2 --------------------------------------------------------------------
def page_optimize() -> None:
    st.title("2 · Run Optimization")
    if st.session_state["testset"] is None or st.session_state["documents"] is None:
        st.warning("Upload documents and generate a testset first (page 1).")
        return

    strategy = st.selectbox("Strategy", ["bfts", "random", "greedy"])
    n_steps = st.slider("Max experiments", 5, 30, 15)
    with st.expander("BFTS advanced"):
        num_seeds = st.slider("num_seeds", 1, 5, 3)
        max_debug_depth = st.slider("max_debug_depth", 1, 5, 3)
        debug_prob = st.slider("debug_prob", 0.0, 1.0, 0.5, 0.05)

    run_btn = st.button("Find best config", type="primary")
    if not run_btn:
        return

    docs = st.session_state["documents"]
    testset = st.session_state["testset"].to_dict(orient="records")
    llm = get_llm()

    def run_fn(cfg_dict, documents, queries):
        return run_pipeline(cfg_dict, documents, queries, llm=llm, persist_dir=Config().chroma_persist_dir)

    def eval_fn(results):
        return evaluate_ragas(results, llm=None)

    if strategy == "bfts":
        loop = BFTSLoop(
            documents=docs,
            testset=testset,
            run_fn=run_fn,
            eval_fn=eval_fn,
            bfts_config=BFTSConfig(num_seeds=num_seeds, max_steps=n_steps, max_debug_depth=max_debug_depth, debug_prob=debug_prob),
        )
        log_area = st.empty()
        score_chart = st.empty()
        stage_label = st.empty()
        scores: list[float] = []
        messages: list[str] = []
        for ev in loop.run_iter():
            t = ev.get("type")
            if t == "node_complete":
                if ev["status"] == "success":
                    scores.append(float(ev.get("score", 0.0)))
                    messages.append(f"✓ {ev['node']['id']} · {ev['node']['stage']} · score={ev['score']:.3f}")
                else:
                    messages.append(f"✗ {ev['node']['id']} failed ({ev.get('decision', '')})")
            elif t == "stage_transition":
                messages.append(f"★ Stage → {ev['current_stage']} ({ev['transitions'][0].get('trigger')})")
                stage_label.info(f"Current stage: **{ev['current_stage']}**")
            elif t == "ablation_node":
                messages.append(f"↻ ablation [{ev['param']}] → {ev.get('score', 0.0):.3f}")

            log_area.code("\n".join(messages[-15:]))
            if scores:
                score_chart.line_chart(pd.DataFrame({"best_so_far": [max(scores[: i + 1]) for i in range(len(scores))], "score": scores}))

        summary = loop._final_summary()
    elif strategy == "random":
        from run import _random_search  # reuse CLI helper
        summary = _random_search(docs, testset, run_fn, eval_fn, n_steps)
    else:
        from run import _greedy_search
        summary = _greedy_search(docs, testset, run_fn, eval_fn, n_steps)

    st.session_state["best_config"] = summary["best_config"]
    st.session_state["best_metrics"] = summary["best_metrics"]
    st.session_state["trajectory"] = summary["trajectory"]
    st.session_state["stage_transitions"] = summary["stage_transitions"]
    st.session_state["ablation_report"] = summary["ablation_report"]
    if strategy == "bfts":
        st.session_state["tree_data"] = loop.get_tree_visualization_data()
    st.success(f"Done. Best score: {summary['best_score']:.3f}")
    st.json(summary["best_config"])


# ---- Page 3 --------------------------------------------------------------------
def page_results() -> None:
    st.title("3 · Results Dashboard")
    if st.session_state["best_config"] is None:
        st.warning("Run the optimizer first (page 2).")
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Best config")
        st.json(st.session_state["best_config"])
        st.subheader("Metrics")
        st.json({k: round(float(v), 4) for k, v in (st.session_state["best_metrics"] or {}).items() if isinstance(v, (int, float))})

    with col2:
        metrics = st.session_state["best_metrics"] or {}
        st.plotly_chart(plot_radar(metrics), use_container_width=True)

    traj = st.session_state["trajectory"] or []
    st.plotly_chart(plot_trajectory(traj, st.session_state["stage_transitions"]), use_container_width=True)

    ab = st.session_state["ablation_report"] or []
    if ab:
        st.plotly_chart(plot_ablation_bar(ab), use_container_width=True)

    if st.session_state["tree_data"]:
        st.plotly_chart(plot_tree(st.session_state["tree_data"]), use_container_width=True)

    if st.session_state["testset"] is not None:
        st.subheader("Stratified breakdown — best config")
        # We'd need per-question results to stratify; show difficulty distribution instead as proxy
        st.dataframe(st.session_state["testset"]["difficulty"].value_counts().rename("count"))


# ---- Page 4 --------------------------------------------------------------------
def page_export() -> None:
    st.title("4 · Export")
    if st.session_state["best_config"] is None:
        st.warning("No optimization run yet.")
        return

    best = st.session_state["best_config"]
    st.download_button("best_config.json", json.dumps(best, indent=2).encode(), file_name="best_config.json")
    try:
        import yaml  # type: ignore
        st.download_button("best_config.yaml", yaml.safe_dump(best, sort_keys=False).encode(), file_name="best_config.yaml")
    except Exception:
        pass

    if st.session_state["trajectory"]:
        buf = io.StringIO()
        pd.DataFrame(st.session_state["trajectory"]).to_csv(buf, index=False)
        st.download_button("trajectory.csv", buf.getvalue().encode(), file_name="trajectory.csv")

    st.subheader("LangChain snippet")
    snippet = _langchain_snippet(best)
    st.code(snippet, language="python")

    st.subheader("Markdown report")
    md = _markdown_report(best, st.session_state["best_metrics"] or {}, st.session_state["ablation_report"] or [])
    st.download_button("experiment_report.md", md.encode(), file_name="experiment_report.md")


def _langchain_snippet(cfg: dict) -> str:
    return f"""# Drop-in LangChain RAG pipeline with the optimizer's best config
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

CONFIG = {json.dumps(cfg, indent=2)}

embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
# ... chunk with chunk_size={cfg.get('chunk_size')}, overlap_ratio={cfg.get('chunk_overlap')}
# ... retrieve top_k={cfg.get('top_k')} with search_mode={cfg.get('search_mode')!r}
# ... rerank={cfg.get('reranker')!r}
# ... generate with prompt_style={cfg.get('prompt_style')!r}
"""


def _markdown_report(cfg: dict, metrics: dict, ablation: list[dict]) -> str:
    lines = ["# RAG Optimizer — Experiment Report", "", "## Best config", "```json", json.dumps(cfg, indent=2), "```", ""]
    lines += ["## Metrics", ""]
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            lines.append(f"- **{k}**: {v:.4f}")
    if ablation:
        lines += ["", "## Ablation", "", "| Component | With | Without | Δ | Contribution % |", "|---|---|---|---|---|"]
        for r in ablation:
            lines.append(f"| {r['param']} | {r['with_value']} | {r['without_value']} | {r['delta']:+.3f} | {r['contribution_pct']:.1f}% |")
    return "\n".join(lines)


# ---- Router -------------------------------------------------------------------
if PAGE.startswith("1"):
    page_upload()
elif PAGE.startswith("2"):
    page_optimize()
elif PAGE.startswith("3"):
    page_results()
else:
    page_export()
