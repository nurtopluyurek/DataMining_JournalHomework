from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from journal_recommender import DatasetConfig, JournalRecommendationPipeline, RecommenderConfig


MODEL_OPTIONS = ["TF-IDF", "BERT", "Hybrid", "Ensemble"]
DISPLAY_ARTICLES = 7711
DISPLAY_JOURNALS = 175
DISPLAY_YEAR_RANGE = (2000, 2018)
EXAMPLE_ABSTRACT = (
    "This paper proposes a machine learning method for network anomaly detection using deep neural networks "
    "and feature selection, improving detection accuracy over traditional traffic classification baselines."
)


st.set_page_config(
    page_title="CS Journal Recommender",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg-1: #fcf7ff;
        --bg-2: #fff8fc;
        --panel: rgba(255, 255, 255, 0.90);
        --panel-strong: rgba(255, 255, 255, 0.96);
        --border: rgba(124, 58, 237, 0.16);
        --border-strong: rgba(236, 72, 153, 0.20);
        --text: #1f1637;
        --muted: #6b5d83;
        --purple: #7c3aed;
        --pink: #ec4899;
        --lavender: #f3e8ff;
        --rose: #fce7f3;
        --shadow: 0 6px 18px rgba(76, 29, 149, 0.06);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(236, 72, 153, 0.08), transparent 34%),
            radial-gradient(circle at top right, rgba(124, 58, 237, 0.08), transparent 30%),
            linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 0.65rem;
        padding-bottom: 0.55rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
    }

    h1, h2, h3, h4, p {
        margin: 0;
    }

    .app-header {
        margin-bottom: 0.55rem;
        padding: 0.5rem 0.1rem 0.55rem 0.1rem;
        border-bottom: 1px solid rgba(124, 58, 237, 0.12);
    }

    .app-title {
        font-size: 1.1rem;
        line-height: 1.1;
        font-weight: 700;
        color: var(--text);
    }

    .app-subtitle {
        margin-top: 0.12rem;
        font-size: 0.82rem;
        color: var(--muted);
    }

    .app-stats {
        margin-top: 0.28rem;
        font-size: 0.76rem;
        color: #5b4f71;
        letter-spacing: 0.01em;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.7rem 0.75rem;
        box-shadow: var(--shadow);
    }

    .section-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #7048b4;
        margin-bottom: 0.35rem;
    }

    .example-box {
        border: 1px solid var(--border-strong);
        border-radius: 10px;
        background: linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(252,231,243,0.45) 100%);
        padding: 0.5rem 0.58rem;
        font-size: 0.79rem;
        line-height: 1.35;
        color: #3d2d57;
        margin-bottom: 0.45rem;
    }

    .compact-note {
        font-size: 0.76rem;
        color: var(--muted);
        margin-bottom: 0.32rem;
    }

    .result-card {
        background: var(--panel-strong);
        border: 1px solid rgba(124, 58, 237, 0.14);
        border-radius: 12px;
        padding: 0.45rem 0.55rem;
        margin-bottom: 0.32rem;
        box-shadow: 0 4px 14px rgba(76, 29, 149, 0.04);
    }

    .result-row {
        display: flex;
        align-items: center;
        gap: 0.32rem;
        flex-wrap: wrap;
    }

    .rank-text {
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--purple);
    }

    .journal-text {
        font-size: 0.86rem;
        font-weight: 700;
        color: var(--text);
        margin-right: 0.2rem;
    }

    .metric-chip {
        display: inline-flex;
        align-items: center;
        font-size: 0.72rem;
        line-height: 1;
        padding: 0.22rem 0.42rem;
        border-radius: 999px;
        border: 1px solid transparent;
        white-space: nowrap;
    }

    .chip-score {
        background: rgba(124, 58, 237, 0.10);
        color: #5b21b6;
        border-color: rgba(124, 58, 237, 0.12);
    }

    .chip-confidence {
        background: rgba(236, 72, 153, 0.10);
        color: #be185d;
        border-color: rgba(236, 72, 153, 0.14);
    }

    .chip-cluster {
        background: rgba(99, 102, 241, 0.10);
        color: #4338ca;
        border-color: rgba(99, 102, 241, 0.14);
    }

    .placeholder-box {
        border: 1px dashed rgba(124, 58, 237, 0.20);
        border-radius: 12px;
        background: rgba(255,255,255,0.72);
        padding: 0.8rem 0.75rem;
        font-size: 0.8rem;
        color: var(--muted);
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(124, 58, 237, 0.12);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.75);
        margin-bottom: 0.22rem;
    }

    div[data-testid="stExpander"] details summary {
        padding-top: 0.03rem;
        padding-bottom: 0.03rem;
    }

    div[data-testid="stExpander"] details summary p {
        font-size: 0.8rem;
        font-weight: 600;
        color: #513476;
    }

    div[data-testid="stDataFrame"] {
        font-size: 0.78rem;
    }

    div[data-baseweb="select"] > div {
        min-height: 2rem;
        border-radius: 10px;
        border-color: rgba(124, 58, 237, 0.18);
        background: rgba(255,255,255,0.9);
    }

    textarea {
        font-size: 0.84rem !important;
        line-height: 1.35 !important;
    }

    div.stButton > button {
        min-height: 2rem;
        padding: 0.2rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: none;
    }

    div.stButton > button[kind="primary"] {
        color: white;
        border: 0;
        background: linear-gradient(90deg, #7c3aed 0%, #ec4899 100%);
    }

    div.stButton > button[kind="secondary"] {
        color: #7c3aed;
        border: 1px solid rgba(124, 58, 237, 0.24);
        background: rgba(255,255,255,0.82);
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.55rem;
    }

    .stMarkdown p,
    .stCaption,
    label {
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Building recommendation assets. First run may take a few minutes.")
def load_pipeline() -> JournalRecommendationPipeline:
    config = RecommenderConfig(
        dataset_config=DatasetConfig(profile="assignment_aligned"),
    )
    return JournalRecommendationPipeline(config).build(use_cache=True, include_evaluation=False)


def run_recommender(
    pipeline: JournalRecommendationPipeline,
    model_name: str,
    abstract_text: str,
) -> pd.DataFrame:
    if model_name == "TF-IDF":
        return pipeline.recommend_journals_tfidf(abstract_text)
    if model_name == "BERT":
        return pipeline.recommend_journals_bert(abstract_text)
    if model_name == "Hybrid":
        return pipeline.recommend_journals_hybrid(abstract_text)
    return pipeline.recommend_journals_ensemble(abstract_text)


def render_result_card(rank: int, row: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-row">
                <span class="rank-text">#{rank}</span>
                <span class="journal-text">{row['journal_name']}</span>
                <span class="metric-chip chip-score">Score: {row['score_percent']:.0f}%</span>
                <span class="metric-chip chip-confidence">Confidence: {row['confidence_label']}</span>
                <span class="metric-chip chip-cluster">Cluster: {row['cluster_label']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Explanation #{rank}", expanded=False):
        st.caption(row["explanation"])
        st.markdown(
            f"**Keywords:** {', '.join(row['overlapping_keywords'][:6]) if row['overlapping_keywords'] else 'No direct keyword overlap extracted'}"
        )
        st.markdown(
            f"**Subjects:** {', '.join(row['overlapping_subjects'][:6]) if row['overlapping_subjects'] else row['cluster_label']}"
        )

        with st.expander("Supporting articles", expanded=False):
            support = pd.DataFrame(row["supporting_articles"])
            if not support.empty:
                support = support.rename(
                    columns={
                        "article_id": "ID",
                        "title": "Title",
                        "year": "Year",
                        "score": "Score",
                    }
                )
                support["Score"] = support["Score"].map(lambda value: f"{value:.4f}")
                st.dataframe(
                    support,
                    use_container_width=True,
                    hide_index=True,
                    height=min(150, 38 * (len(support) + 1)),
                )


pipeline = load_pipeline()
summary = pipeline.dataset_summary()

if "abstract_input" not in st.session_state:
    st.session_state["abstract_input"] = ""
if "last_results" not in st.session_state:
    st.session_state["last_results"] = pd.DataFrame()
if "last_model_name" not in st.session_state:
    st.session_state["last_model_name"] = "Hybrid"


st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">CS Journal Recommender</div>
        <div class="app-subtitle">Top-5 journal recommendations from academic abstracts.</div>
        <div class="app-stats">
            Articles: {DISPLAY_ARTICLES} | Journals: {DISPLAY_JOURNALS} | Years: {DISPLAY_YEAR_RANGE[0]}–{DISPLAY_YEAR_RANGE[1]}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([0.44, 0.56], gap="small")

with left:
    st.markdown("<div class='panel'><div class='section-label'>Input</div></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="example-box">
            <strong>Example abstract</strong><br>
            {EXAMPLE_ABSTRACT}
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_left, control_right = st.columns([0.62, 0.38], gap="small")
    with control_left:
        model_name = st.selectbox(
            "Model",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state["last_model_name"]),
            label_visibility="collapsed",
        )
    with control_right:
        example_clicked = st.button("Use Example Abstract", type="secondary", use_container_width=True)

    abstract_input = st.text_area(
        "Abstract",
        key="abstract_input",
        height=120,
        label_visibility="collapsed",
        placeholder="Paste an academic abstract here...",
    )

    action_left, action_right = st.columns([0.54, 0.46], gap="small")
    run_clicked = action_left.button("Recommend Journals", type="primary", use_container_width=True)
    clear_clicked = action_right.button("Clear", type="secondary", use_container_width=True)

    if example_clicked:
        st.session_state["abstract_input"] = EXAMPLE_ABSTRACT
        st.rerun()

    if clear_clicked:
        st.session_state["abstract_input"] = ""
        st.session_state["last_results"] = pd.DataFrame()
        st.rerun()

    st.markdown(
        f"""
        <div class="compact-note">
            Dataset profile loaded: {summary['records']} records, {summary['journals']} journals.
        </div>
        """,
        unsafe_allow_html=True,
    )


if run_clicked:
    if not abstract_input.strip():
        st.warning("Enter an abstract before running the recommender.")
    else:
        with st.spinner("Scoring journals..."):
            st.session_state["last_results"] = run_recommender(pipeline, model_name, abstract_input)
            st.session_state["last_model_name"] = model_name


with right:
    st.markdown("<div class='section-label'>Top-5 Recommendations</div>", unsafe_allow_html=True)
    results = st.session_state["last_results"]

    if results.empty:
        st.markdown(
            """
            <div class="placeholder-box">
                Run the recommender to display compact Top-5 journal cards here.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for idx, (_, row) in enumerate(results.iterrows(), start=1):
            render_result_card(idx, row)
