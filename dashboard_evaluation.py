"""
Dashboard de visualisation des resultats d'evaluation comparative (JOUR 8).

Ce module Streamlit charge exclusivement trois artefacts produits par le
pipeline d'evaluation (`generate_evaluation_artifacts.py`) :

    files/corpus/evaluation_report.json   -> rapport detaille par requete x config
    files/corpus/evaluation_results.csv   -> table plate par requete x config
    files/corpus/comparison_table.json    -> tableau agrege + gains par module

Aucune metrique n'est recalculee : le dashboard se contente de lire,
mettre en forme et visualiser les resultats pre-calculs.

Sections du dashboard :
    1. Tableau comparatif   : Dense / Hybrid / Hybrid + Rerank /
                              Hybrid + Rerank + Verification
    2. KPI Cards            : meilleur Recall@5, Recall@10, MRR,
                              Faithfulness, Citation Accuracy, Latency
    3. Graphiques           : Recall@3/5/10, MRR, Faithfulness,
                              Citation Accuracy, Average & P95 Latency
    4. Radar chart          : comparaison multi-metriques des 4 configs
    5. Analyse automatique  : gain BM25, gain Reranker, gain NLI
    6. Export CSV           : telechargement de la table complete

Usage :
    streamlit run dashboard_evaluation.py

Environnement :
    Python 3.11+ · Streamlit >= 1.41 · pandas · numpy · matplotlib
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR: Path = Path(__file__).resolve().parent
CORPUS_DIR: Path = SCRIPT_DIR / "files" / "corpus"

REPORT_PATH: Path = CORPUS_DIR / "evaluation_report.json"
RESULTS_CSV_PATH: Path = CORPUS_DIR / "evaluation_results.csv"
COMPARISON_PATH: Path = CORPUS_DIR / "comparison_table.json"

# Palette consistante : une couleur par configuration.
CONFIG_COLORS: dict[str, str] = {
    "Dense": "#4C72B0",
    "Hybrid": "#55A868",
    "Hybrid + Rerank": "#DD8452",
    "Hybrid + Rerank + Verification": "#C44E52",
}

# Metriques ou une valeur plus faible est meilleure.
LOWER_IS_BETTER: frozenset[str] = frozenset({"Average Latency", "P95 Latency"})

st.set_page_config(
    page_title="RAG Citation Validator - Evaluation Dashboard",
    page_icon="\U0001F50D",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Chargement des artefacts (cache : lecture disque uniquement)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Charging evaluation_report.json ...")
def load_report(path: Path) -> dict[str, Any]:
    """Load the detailed evaluation report (JSON).

    Args:
        path: path of ``evaluation_report.json``.

    Returns:
        Parsed report dict, or an empty dict when the file is missing.
    """
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading evaluation_results.csv ...")
def load_results_csv(path: Path) -> pd.DataFrame:
    """Load the flat per-query-per-config results table (CSV).

    Args:
        path: path of ``evaluation_results.csv``.

    Returns:
        DataFrame of results (possibly empty when the file is missing).
    """
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading comparison_table.json ...")
def load_comparison(path: Path) -> dict[str, Any]:
    """Load the aggregated comparison table (JSON).

    Args:
        path: path of ``comparison_table.json``.

    Returns:
        Parsed comparison dict, or an empty dict when the file is missing.
    """
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers de mise en forme
# ---------------------------------------------------------------------------

def format_metric(metric: str, value: float | None) -> str:
    """Format a metric value for display.

    Latency metrics are shown in milliseconds, quality metrics as
    percentages, and missing values as ``N/A``.

    Args:
        metric: human-readable metric name (e.g. ``"Recall@5"``).
        value: raw value from the comparison table.

    Returns:
        Formatted string.
    """
    if value is None:
        return "N/A"
    if metric in LOWER_IS_BETTER:
        return f"{value:,.1f} ms"
    return f"{value:.1f}%"


def highlight_best(df: pd.DataFrame) -> Any:
    """Return a styler that highlights the best value per column.

    For quality metrics the best is the maximum; for latency metrics it
    is the minimum.  Missing cells are ignored.

    Args:
        df: comparison table indexed by config, columns = metrics.

    Returns:
        A pandas Styler with best values highlighted in bold green.
    """
    def _style(col: pd.Series) -> list[str]:
        vals = col.dropna()
        if vals.empty:
            return [""] * len(col)
        best = vals.min() if col.name in LOWER_IS_BETTER else vals.max()
        return [
            "color: #1a7f37; font-weight: bold;" if v == best else ""
            for v in col
        ]

    return df.style.apply(_style, axis=0)


def build_radar_figure(
    table: dict[str, dict[str, float | None]],
    configs: list[str],
    radar_metrics: list[str],
) -> matplotlib.figure.Figure:
    """Build a radar chart comparing configurations across quality metrics.

    Each axis is normalised to [0, 1] using the column max so all axes
    share a comparable scale.

    Args:
        table: mapping config -> {metric -> value}.
        configs: ordered list of configuration names.
        radar_metrics: metric names to plot (quality only).

    Returns:
        A matplotlib Figure containing the radar chart.
    """
    labels = radar_metrics
    n_axes = len(labels)
    angles = np.linspace(0.0, 2.0 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})

    maxima = {
        m: (max((table[c][m] or 0.0) for c in configs) or 1.0)
        for m in labels
    }

    for config in configs:
        values = [(table[config].get(m) or 0.0) / maxima[m] for m in labels]
        values += values[:1]
        color = CONFIG_COLORS.get(config, "#888888")
        ax.plot(angles, values, linewidth=2, label=config, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticklabels([])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)
    fig.tight_layout()
    return fig


def build_bar_figure(
    series: pd.Series,
    title: str,
    ylabel: str,
    higher_is_better: bool = True,
) -> matplotlib.figure.Figure:
    """Build a vertical bar chart of one metric across configurations.

    The best bar label is emphasised in bold while every bar uses its
    config colour.

    Args:
        series: index = config name, values = metric values.
        title: chart title.
        ylabel: y-axis label.
        higher_is_better: whether to highlight max (True) or min (False).

    Returns:
        A matplotlib Figure containing the bar chart.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = [CONFIG_COLORS.get(c, "#888888") for c in series.index]
    bars = ax.bar(series.index, series.values, color=colors, width=0.62)

    best_val = series.max() if higher_is_better else series.min()
    for bar, val in zip(bars, series.values):
        emphasis = "bold" if val == best_val else "normal"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,.1f}",
            ha="center", va="bottom", fontsize=9, fontweight=emphasis,
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, rotation=12)
    ax.margins(y=0.18)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sections du dashboard
# ---------------------------------------------------------------------------

def render_header(
    report: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    """Render the page header with project metadata."""
    st.title("RAG Citation Validator — Evaluation Dashboard")
    metadata = report.get("metadata", {})
    corpus_info = metadata.get("corpus_info", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Queries", metadata.get("num_queries", "N/A"))
    col2.metric(
        "Configurations",
        metadata.get("num_configs", len(comparison.get("configs", []))),
    )
    col3.metric("Corpus (chunks)", corpus_info.get("total_chunks", "N/A"))
    col4.metric("Documents", corpus_info.get("total_documents", "N/A"))

    generated_at = str(metadata.get("generated_at", "")).replace("T", " · ")
    st.caption(
        f"Artefacts generated on {generated_at} — metrics loaded as-is, "
        "no recalculation."
    )


def render_comparison_table(comparison: dict[str, Any]) -> pd.DataFrame:
    """Render the comparative table of the 4 pipeline configurations.

    Args:
        comparison: parsed comparison table.

    Returns:
        The DataFrame displayed (config x metric) for reuse.
    """
    st.subheader("Comparative table")
    configs: list[str] = comparison.get("configs", [])
    table: dict[str, dict[str, float | None]] = comparison.get("table", {})
    best: dict[str, str] = comparison.get("best_config_per_metric", {})

    metrics: list[str] = [
        m for m in comparison.get("metrics", [])
        if any(table.get(c, {}).get(m) is not None for c in configs)
    ]

    df = pd.DataFrame.from_dict(
        {c: {m: table.get(c, {}).get(m) for m in metrics} for c in configs},
        orient="index",
    )[metrics]

    st.dataframe(highlight_best(df), use_container_width=True)

    winners = [f"**{m}**: {best[m]}" for m in metrics if best.get(m)]
    if winners:
        st.caption("Best per metric — " + " · ".join(winners))
    return df


def render_kpi_cards(comparison: dict[str, Any]) -> None:
    """Render one KPI card per key metric showing the best configuration.

    Args:
        comparison: parsed comparison table.
    """
    st.subheader("KPI Cards — best per metric")
    table = comparison.get("table", {})
    configs = comparison.get("configs", [])

    kpi_metrics = [
        ("Recall@5", "\U0001F3AF"),
        ("Recall@10", "\U0001F3AF"),
        ("MRR", "\U0001F3C5"),
        ("Faithfulness", "\U0001F9E9"),
        ("Citation Accuracy", "\U0001F4DA"),
        ("Average Latency", "\u26A1"),
    ]
    columns = st.columns(len(kpi_metrics))
    for col, (metric, icon) in zip(columns, kpi_metrics):
        values = [
            (table.get(c, {}).get(metric), c)
            for c in configs
            if table.get(c, {}).get(metric) is not None
        ]
        if not values:
            col.metric(f"{icon} {metric}", "N/A")
            continue
        value, config = (
            min(values) if metric in LOWER_IS_BETTER else max(values)
        )
        short = (
            config.replace("Hybrid + Rerank + Verification", "H+R+V")
            .replace("Hybrid + Rerank", "H+R")
        )
        col.metric(
            f"{icon} {metric}",
            format_metric(metric, value),
            delta=f"best: {short}",
            border=True,
        )


def render_metric_charts(df: pd.DataFrame) -> None:
    """Render bar charts for every metric in the comparison DataFrame.

    Args:
        df: comparison DataFrame (config index, metric columns).
    """
    st.subheader("Charts")
    quality_metrics = [m for m in df.columns if m not in LOWER_IS_BETTER]
    latency_metrics = [m for m in df.columns if m in LOWER_IS_BETTER]

    rows = [quality_metrics[i:i + 2] for i in range(0, len(quality_metrics), 2)]
    for pair in rows:
        cols = st.columns(len(pair))
        for col, metric in zip(cols, pair):
            series = df[metric].dropna()
            if series.empty:
                col.info(f"{metric}: not available for this run.")
                continue
            fig = build_bar_figure(series, metric, "% / score")
            col.pyplot(fig)

    for metric in latency_metrics:
        series = df[metric].dropna()
        if series.empty:
            continue
        left, right = st.columns([2, 1])
        with left:
            fig = build_bar_figure(
                series, f"{metric} (ms)", "milliseconds",
                higher_is_better=False,
            )
            st.pyplot(fig)
        with right:
            st.markdown(
                "**Reading note** — the reranker and the NLI verification add "
                "cross-encoder inference on CPU: an order-of-magnitude "
                "latency cost for a measurable gain in faithfulness."
            )


def render_radar_chart(comparison: dict[str, Any]) -> None:
    """Render the comparative radar chart across quality metrics.

    Args:
        comparison: parsed comparison table.
    """
    st.subheader("Radar chart")
    table = comparison.get("table", {})
    configs: list[str] = comparison.get("configs", [])
    radar_metrics = [
        "Recall@3", "Recall@5", "Recall@10", "MRR",
        "Faithfulness", "Citation Accuracy",
    ]
    available = [
        m for m in radar_metrics
        if any(table.get(c, {}).get(m) is not None for c in configs)
    ]
    if len(available) < 3:
        st.info("Not enough metrics available to draw a radar chart.")
        return

    left, right = st.columns([3, 2])
    with left:
        fig = build_radar_figure(table, configs, available)
        st.pyplot(fig)
    with right:
        st.markdown("**How to read it**")
        st.markdown(
            "- Each axis is a quality metric normalised by the best "
            "configuration (max = outer edge).\n"
            "- **Dense** covers semantic retrieval only.\n"
            "- **Hybrid** adds BM25 fusion (RRF): broader recall.\n"
            "- **Hybrid + Rerank** sharpens ranking precision.\n"
            "- **H + R + Verification** adds NLI-checked citations: "
            "faithfulness and citation accuracy become measurable."
        )


def render_gains_analysis(comparison: dict[str, Any]) -> None:
    """Render the automatic analysis of module gains.

    Three cards summarise the marginal contribution of BM25, the BGE
    reranker and the NLI verification module.

    Args:
        comparison: parsed comparison table.
    """
    st.subheader("Automatic analysis — marginal gain per module")
    gains = comparison.get("gains", {})

    descriptions = {
        "bm25_gain": (
            "Gain BM25 (Dense \u2192 Hybrid)",
            "Lexical channel added to dense retrieval via Reciprocal Rank Fusion.",
        ),
        "reranker_gain": (
            "Gain Reranker (Hybrid \u2192 Hybrid + Rerank)",
            "BGE cross-encoder re-scores the top-20 hybrid candidates.",
        ),
        "nli_gain": (
            "Gain NLI (\u2192 + Verification)",
            "roberta-large-mnli verifies each citation against its source passage.",
        ),
    }

    cols = st.columns(3)
    for col, (key, (title, subtitle)) in zip(cols, descriptions.items()):
        metrics = gains.get(key, {}).get("metrics", {})
        lines: list[str] = []
        for name in ("Recall@5", "MRR", "Faithfulness", "Citation Accuracy"):
            entry = metrics.get(name)
            if not entry:
                continue
            target = entry.get("to")
            if target is None:
                continue
            pct = entry.get("gain_pct")
            arrow = "\u2197" if (pct or 0) > 0 or entry.get("direction") == "new" else "\u2198"
            change = f" ({pct:+.1f}%)" if pct is not None else " (new)"
            lines.append(f"- {name}: **{target:.1f}%** {arrow}{change}")

        lat = metrics.get("Average Latency") or {}
        if lat.get("from") is not None and lat.get("to") is not None:
            lines.append(
                f"- Latency: {lat['from']:,.0f} ms \u2192 "
                f"**{lat['to']:,.0f} ms** ({lat.get('gain_pct'):+.1f}%)"
            )

        with col:
            st.markdown(f"#### {title}")
            st.caption(subtitle)
            if lines:
                st.markdown("\n".join(lines))
            else:
                st.info("No data.")


def render_per_query_explorer(report: dict[str, Any]) -> None:
    """Render a per-query drill-down explorer from the detailed report.

    A theme filter plus a query selector display the four configurations
    side by side for a single question.

    Args:
        report: parsed evaluation report.
    """
    st.subheader("Per-query explorer")
    queries = report.get("queries", [])
    if not queries:
        st.info("evaluation_report.json unavailable or empty.")
        return

    themes = ["All"] + sorted({q["theme"] for q in queries})
    selected_theme = st.selectbox("Theme", themes, index=0)
    pool = [
        q for q in queries
        if selected_theme == "All" or q["theme"] == selected_theme
    ]

    options = {f"{q['query_id']} — {q['query']}": q for q in pool}
    choice = st.selectbox("Query", list(options.keys()), index=0)
    query_result = options[choice]

    rows = []
    for config_name, cfg in query_result.get("configs", {}).items():
        rows.append({
            "Config": config_name,
            "Recall@3": cfg.get("recall_at_3"),
            "Recall@5": cfg.get("recall_at_5"),
            "Recall@10": cfg.get("recall_at_10"),
            "MRR": cfg.get("mrr"),
            "Faithfulness": cfg.get("faithfulness"),
            "Citation Acc.": cfg.get("citation_accuracy"),
            "Latency (ms)": cfg.get("avg_latency_ms"),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Config"), use_container_width=True)

    verif = query_result.get("configs", {}).get(
        "Hybrid + Rerank + Verification", {}
    )
    details = verif.get("verification_details")
    if details:
        counts = details.get("verdict_counts", {})
        st.markdown(
            f"NLI verification: **{details.get('citations_verified', 0)}** "
            "citations checked — "
            f"\U0001F7E2 Supported **{counts.get('Supported', 0)}** · "
            f"\U0001F7E1 Weak **{counts.get('Weak Support', 0)}** · "
            f"\U0001F534 Unsupported **{counts.get('Unsupported', 0)}**"
        )


def render_csv_export(results_df: pd.DataFrame) -> None:
    """Render the CSV export block.

    Args:
        results_df: flat per-query-per-config results table.
    """
    st.subheader("Export CSV")
    if results_df.empty:
        st.info("No CSV data to export.")
        return
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download evaluation_results.csv",
        data=csv_bytes,
        file_name="rag_evaluation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        f"{len(results_df)} rows (query x config), "
        f"{len(results_df.columns)} columns."
    )


# ---------------------------------------------------------------------------
# Point d'entree Streamlit
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: load the three artifacts and render the dashboard."""
    report = load_report(REPORT_PATH)
    results_df = load_results_csv(RESULTS_CSV_PATH)
    comparison = load_comparison(COMPARISON_PATH)

    if not report and not comparison:
        st.error(
            "Artefacts d'evaluation introuvables.\n\n"
            "Lancez d'abord `python files/generate_evaluation_artifacts.py` "
            "puis rechargez cette page."
        )
        return

    with st.sidebar:
        st.header("Artifacts loaded")
        for path, loaded in (
            (REPORT_PATH.name, bool(report)),
            (RESULTS_CSV_PATH.name, not results_df.empty),
            (COMPARISON_PATH.name, bool(comparison)),
        ):
            status = "\u2705 loaded" if loaded else "\u274C missing"
            st.markdown(f"- `{path}` — {status}")
        st.divider()
        st.markdown(
            "**Pipeline evaluated**\n"
            "1. Dense Retrieval (BGE / Qdrant)\n"
            "2. BM25 (rank-bm25)\n"
            "3. Hybrid Search (RRF)\n"
            "4. BGE Reranker\n"
            "5. Generation\n"
            "6. Citation Verification (NLI)"
        )

    render_header(report, comparison)
    df = render_comparison_table(comparison)
    render_kpi_cards(comparison)
    render_metric_charts(df)
    render_radar_chart(comparison)
    render_gains_analysis(comparison)
    render_per_query_explorer(report)
    render_csv_export(results_df)


if __name__ == "__main__":
    main()
