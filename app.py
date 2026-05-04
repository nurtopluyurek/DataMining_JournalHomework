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


DEFAULT_ABSTRACT = (
    "This paper proposes a machine learning based method for detecting anomalies in network traffic using "
    "deep neural networks and feature selection. Experimental results show improved detection performance "
    "compared with traditional classification baselines in dynamic communication environments."
)
MODEL_OPTIONS = ["TF-IDF", "BERT", "Hybrid", "Ensemble"]


st.set_page_config(
    page_title="CS Journal Recommender",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
        color: #111827;
    }
    .block-container {
        max-width: 1360px;
        padding-top: 0.9rem;
        padding-bottom: 0.75rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }
    h1, h2, h3, h4, p {
        margin-top: 0;
        margin-bottom: 0;
    }
    .app-header {
        border-bottom: 1px solid #d1d5db;
        padding-bottom: 0.45rem;
        margin-bottom: 0.75rem;
    }
    .app-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 0.87rem;
        color: #4b5563;
        margin-top: 0.2rem;
    }
    .meta-line {
        font-size: 0.8rem;
        color: #374151;
        margin-top: 0.35rem;
    }
    .panel-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        color: #374151;
        margin-bottom: 0.35rem;
    }
    .compact-note {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.5rem 0.65rem;
        background: #fafafa;
        font-size: 0.84rem;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    .result-summary {
        font-size: 0.8rem;
        color: #4b5563;
        margin-bottom: 0.35rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background: #fcfcfc;
        margin-bottom: 0.35rem;
    }
    div[data-testid="stExpander"] details summary {
        padding-top: 0.1rem;
        padding-bottom: 0.1rem;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 0.9rem;
    }
    div[data-testid="stDataFrame"] {
        font-size: 0.84rem;
    }
    div.stButton > button {
        padding-top: 0.28rem;
        padding-bottom: 0.28rem;
        font-size: 0.9rem;
    }
    label, .stCaption, .stMarkdown, .stTextArea, .stSelectbox {
        font-size: 0.9rem;
    }
    textarea {
        font-size: 0.9rem !important;
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


def build_results_table(results: pd.DataFrame) -> pd.DataFrame:
    compact = results.copy().reset_index(drop=True)
    compact.insert(0, "Rank", range(1, len(compact) + 1))
    compact["Score"] = compact["score_percent"].map(lambda value: f"{value:.2f}%")
    compact["Confidence"] = compact["confidence_label"]
    compact["Cluster"] = compact["cluster_label"]
    compact["Journal"] = compact["journal_name"]
    return compact[["Rank", "Journal", "Score", "Confidence", "Cluster"]]


def format_terms(values: list[str], fallback: str) -> str:
    if not values:
        return fallback
    return ", ".join(values[:6])


def render_result_details(results: pd.DataFrame) -> None:
    for rank, row in results.reset_index(drop=True).iterrows():
        label = (
            f"#{rank + 1} {row['journal_name']} | "
            f"Score {row['score_percent']:.2f}% | "
            f"{row['confidence_label']} | "
            f"{row['cluster_label']}"
        )
        with st.expander(label, expanded=False):
            st.caption(row["explanation"])
            meta_left, meta_right = st.columns(2, gap="small")
            with meta_left:
                st.markdown(
                    f"**Keywords:** {format_terms(row['overlapping_keywords'], 'No direct keyword overlap extracted')}"
                )
            with meta_right:
                st.markdown(
                    f"**Subjects:** {format_terms(row['overlapping_subjects'], 'No direct subject overlap extracted')}"
                )

            support = pd.DataFrame(row["supporting_articles"])
            if not support.empty:
                support = support.rename(
                    columns={
                        "article_id": "Article ID",
                        "title": "Title",
                        "year": "Year",
                        "score": "Score",
                    }
                )
                support["Score"] = support["Score"].map(lambda value: f"{value:.4f}")
                st.caption("Supporting articles")
                st.dataframe(
                    support,
                    use_container_width=True,
                    hide_index=True,
                    height=min(150, 38 * (len(support) + 1)),
                )


pipeline = load_pipeline()
summary = pipeline.dataset_summary()

if "abstract_input" not in st.session_state:
    st.session_state["abstract_input"] = DEFAULT_ABSTRACT
if "last_results" not in st.session_state:
    st.session_state["last_results"] = pd.DataFrame()
if "last_model_name" not in st.session_state:
    st.session_state["last_model_name"] = "Hybrid"


st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">Computer Science Journal Recommendation System</div>
        <div class="app-subtitle">
            Compact report-friendly interface for abstract-driven Top-5 journal recommendation.
        </div>
        <div class="meta-line">
            Profile: Assignment-aligned |
            Articles: {summary['records']} |
            Journals: {summary['journals']} |
            Years: {summary['year_range'][0]}-{summary['year_range'][1]}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([0.44, 0.56], gap="small")

with left:
    st.markdown("<div class='panel-label'>Inputs</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="compact-note">
            Paste an abstract and select a recommendation model. The interface is intentionally compact so the
            full input/output view fits into a single academic report screenshot.
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_name = st.selectbox(
        "Recommendation model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state["last_model_name"]),
    )
    abstract_input = st.text_area(
        "Article abstract",
        key="abstract_input",
        height=210,
    )

    action_left, action_right = st.columns(2, gap="small")
    run_clicked = action_left.button("Run Recommendation", type="primary", use_container_width=True)
    demo_clicked = action_right.button("Load Demo Abstract", use_container_width=True)

    if demo_clicked:
        st.session_state["abstract_input"] = DEFAULT_ABSTRACT
        st.rerun()

    with st.expander("Dataset notes", expanded=False):
        st.write(
            "The app uses the assignment-aligned subset generated by the pipeline. "
            "This subset is derived from the accessible SQLite file rather than the smaller PDF description."
        )
        st.dataframe(
            pd.DataFrame(summary["top_journals"].items(), columns=["Journal", "Article Count"]),
            use_container_width=True,
            hide_index=True,
            height=235,
        )


if run_clicked:
    if not abstract_input.strip():
        st.warning("Enter an abstract before running the recommender.")
    else:
        with st.spinner("Scoring journals against the article corpus..."):
            st.session_state["last_results"] = run_recommender(pipeline, model_name, abstract_input)
            st.session_state["last_model_name"] = model_name


with right:
    st.markdown("<div class='panel-label'>Top-5 Recommendations</div>", unsafe_allow_html=True)
    results = st.session_state["last_results"]

    if results.empty:
        st.info("Run the recommender to display the Top-5 journals here.")
        st.dataframe(
            pd.DataFrame(summary["top_journals"].items(), columns=["Journal", "Article Count"]).head(5),
            use_container_width=True,
            hide_index=True,
            height=210,
        )
    else:
        st.markdown(
            f"""
            <div class="result-summary">
                Model: {st.session_state['last_model_name']} |
                Results: {len(results)} journals |
                Query length: {len(abstract_input.split())} words
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            build_results_table(results),
            use_container_width=True,
            hide_index=True,
            height=240,
        )
        render_result_details(results)
