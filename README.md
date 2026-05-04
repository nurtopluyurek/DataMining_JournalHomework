# Computer Science Journal Recommendation and Topic Discovery from Scholarly Metadata

A completed data mining final project that recommends suitable computer science journals from a user-provided academic abstract and complements recommendation with topic discovery over the underlying publication corpus.

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b?logo=streamlit&logoColor=white)](https://dataminingjournalhomework-83wx6pppvyjxjxdnq3fvyb.streamlit.app/)

Live demo: [https://dataminingjournalhomework-83wx6pppvyjxjxdnq3fvyb.streamlit.app/](https://dataminingjournalhomework-83wx6pppvyjxjxdnq3fvyb.streamlit.app/)

## Project Description

This project builds a journal recommendation system for computer science publications using a scholarly SQLite database of article metadata. Given an abstract, the system returns the Top-5 most relevant journals together with supporting explanations such as keywords, subject evidence, confidence, and cluster context. In parallel, the project performs topic discovery using clustering and topic modeling to reveal the thematic structure of the corpus.

The repository includes the full experimental notebook, reusable Python backend, generated analysis outputs, a compact Streamlit interface, and a full IEEE-style report suitable for academic submission.

## Key Highlights

- End-to-end pipeline from relational SQLite extraction to deployable journal recommendation
- Top-5 explainable recommendations from a single abstract query
- Direct comparison of TF-IDF, Sentence-BERT, and ensemble ranking
- Strong evaluation package including ablation, confusion, per-journal, imbalance, and confidence analysis
- Topic discovery with KMeans clustering and LDA topic modeling
- Compact Streamlit application for demonstration and report screenshots
- Full IEEE-style report included in the repository

## Dataset and Reproducibility Note

The project is grounded in the provided SQLite database of computer science publication metadata.

| Scope | Records | Journals | Notes |
|---|---:|---:|---|
| Assignment PDF description | 7,711 | 175 | Reported in the assignment brief |
| Raw accessible SQLite corpus | 23,801 | 466 | Actual accessible source used for reproducibility |
| Final supervised recommendation dataset | 21,907 | 404 | Filtered to article records with non-empty abstracts and journals with at least 5 examples |

Publication years in the accessible corpus span **2000-2018**.

Reproducibility note: the assignment PDF described a smaller `7,711 / 175` dataset, but the accessible SQLite database contained a larger corpus. This repository follows the accessible database as the reproducible source of truth and documents the difference explicitly rather than silently forcing the report to match the PDF description.

## Methodology Overview

The pipeline follows these stages:

1. Inspect the SQLite schema and extract articles, abstracts, journals, keywords, keyword-plus fields, and subjects.
2. Clean and normalize textual metadata.
3. Build multiple text representations, including `abstract_only_text`, `title_abstract_text`, and `combined_text`.
4. Train and evaluate lexical, semantic, and ensemble recommenders.
5. Generate explainable recommendation outputs.
6. Perform topic discovery with clustering and LDA.
7. Export figures, tables, and notebook-ready artifacts.

## Models

### TF-IDF Recommender

The primary deployable model is a tuned TF-IDF journal recommender using lexical features that are highly effective for journal-specific terminology.

### Sentence-BERT Recommender

A semantic recommender built with `all-MiniLM-L6-v2`, used to capture contextual similarity beyond exact word overlap.

### Ensemble Recommender

A score-level fusion of the lexical and semantic models:

\[
S_{ensemble} = 0.5 \, S_{tfidf} + 0.5 \, S_{bert}
\]

This model is retained as an important comparison baseline, although the best final result in this repository is achieved by TF-IDF.

## Evaluation Results

Evaluation uses a stratified `80/20` split with abstract-only query input.

| Model | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| **TF-IDF Recommender** | **0.3389** | **0.5244** | **0.6027** | **0.2833** |
| Sentence-BERT Recommender | 0.2752 | 0.4411 | 0.5358 | 0.2317 |
| Ensemble Recommender | 0.2809 | 0.4507 | 0.5418 | 0.2365 |

Best result:

- **Model:** TF-IDF Recommender
- **Top-1 Accuracy:** `0.3389`
- **Top-3 Accuracy:** `0.5244`
- **Top-5 Accuracy:** `0.6027`
- **Macro-F1:** `0.2833`

The main takeaway is that a tuned lexical model remains strongest in this setting, indicating that journal recommendation is highly sensitive to domain-specific terminology.

## Topic Discovery

Topic discovery is used both for corpus analysis and for recommendation explainability.

- TF-IDF + TruncatedSVD + KMeans clustering
- `k` scanned from `10` to `60`
- Final `k = 60`, selected by maximum silhouette score
- LDA used as a complementary soft topic-modeling view

KMeans provides a hard partition of documents into interpretable thematic groups, while LDA provides softer document-topic and topic-word structure that is useful for higher-level corpus interpretation.

## Streamlit Application

The Streamlit app provides a compact interface for testing the final system:

- paste an academic abstract
- choose a model
- receive Top-5 journal recommendations
- inspect score, confidence, cluster label, keywords, and supporting explanation

Live demo: [https://dataminingjournalhomework-83wx6pppvyjxjxdnq3fvyb.streamlit.app/](https://dataminingjournalhomework-83wx6pppvyjxjxdnq3fvyb.streamlit.app/)

Local entry point:

```text
app.py
```

## Project Structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `notebooks/data_mining_final_project.ipynb` | Main executed notebook and final experimental source |
| `src/project_lib.py` | Reusable backend library for dataset loading, recommendation, evaluation, and clustering |
| `scripts/build_project_assets.py` | Generates analysis outputs and figures |
| `scripts/generate_notebook.py` | Notebook generation utility |
| `outputs/` | Exported tables, figures, JSON outputs, and caches |
| `report/ieee_final_project.tex` | Full IEEE-style report source |
| `tests/test_project_lib.py` | Smoke tests |

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to Run the Notebook

The main notebook is:

```text
notebooks/data_mining_final_project.ipynb
```

Open it with Jupyter:

```bash
jupyter notebook notebooks/data_mining_final_project.ipynb
```

Or execute it in place:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/data_mining_final_project.ipynb
```

## How to Run the Streamlit App Locally

```bash
streamlit run app.py
```

The app loads the same backend logic used by the final notebook and returns compact Top-5 recommendations with explanation details.

## Expected Outputs

The repository generates and stores analysis outputs under `outputs/`.

Representative artifacts include:

| Output | Description |
|---|---|
| `outputs/tables/model_metrics.csv` | Main recommendation metrics |
| `outputs/tables/ablation_results.csv` | Input representation ablation study |
| `outputs/tables/per_journal_top5.csv` | Per-journal Top-5 performance |
| `outputs/tables/top_confused_journal_pairs.csv` | Confusion analysis |
| `outputs/tables/class_imbalance_summary.csv` | Journal imbalance summary |
| `outputs/tables/cluster_summary.csv` | Topic cluster summaries |
| `outputs/figures/abstract_model_comparison.png` | Model comparison chart |
| `outputs/figures/ablation_comparison.png` | Ablation comparison chart |
| `outputs/figures/journal_imbalance.png` | Journal imbalance figure |
| `outputs/figures/cluster_projection.png` | Topic cluster projection |
| `outputs/system_pipeline_diagram.png` | End-to-end system diagram |

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- sentence-transformers
- PyTorch
- Jupyter Notebook
- Streamlit
- SQLite
- Matplotlib

## Report

The repository includes the full IEEE-style academic report source:

- `report/ieee_final_project.tex`

This report is aligned with the final notebook and Streamlit application and is intended to serve as the formal written submission for the project.

## Limitations and Future Work

Current limitations include:

- journal imbalance affects recommendation quality for sparse venues
- some journals overlap heavily in topical scope
- Sentence-BERT does not outperform TF-IDF on this corpus
- abstract-only querying limits contextual richness

Natural next steps include:

- domain-adapted transformer models
- richer journal-profile modeling
- citation-aware recommendation
- improved confidence calibration
- stronger semantic reranking on top of lexical retrieval

## Author / Course Note

Prepared as a **Data Mining final project** in computer science. The repository is structured to be suitable both for academic submission and for technical review as a complete portfolio-style project.
