# Computer Science Journal Recommendation System

This project builds a complete data mining pipeline for recommending the **top 5 most relevant computer science journals** for a new academic abstract.

The system includes:

- relational database inspection from SQLite
- dataset construction with dynamic joins across abstracts, journals, keywords, subjects, and keyword-plus terms
- TF-IDF, Sentence-BERT, hybrid, and ensemble recommenders
- explainable recommendations with supporting articles, keyword overlap, subject overlap, cluster label, and confidence
- KMeans clustering with `K=2..80`
- LDA topic modeling
- leave-one-article-out style evaluation with Top-1, Top-3, and Top-5 metrics
- a Streamlit interface for interactive journal recommendation

## Project Files

- `final_project.ipynb`: main analysis notebook
- `app.py`: Streamlit user interface
- `requirements.txt`: Python dependencies
- `report_outline.md`: IEEE-style report structure
- `presentation_outline.md`: slide plan with speaker notes
- `src/journal_recommender/`: reusable pipeline code shared by the notebook and app
- `scripts/generate_notebook.py`: generates the notebook structure programmatically

## Dataset Inputs

Place the provided files in `data/raw/`:

- `CompSciencePub (1).sqlite`
- `CS_JournalAbstracts.zip`
- `Data_mining_journal_homework 2.pdf`

The current workspace already uses that layout.

## Assignment Alignment

The assignment PDF states that the task is based on **7711 articles from 175 journals**. The SQLite file is larger than that specification, so the project uses a reproducible `assignment_aligned` profile:

- filters to English journal articles/reviews with abstracts
- selects a stable 175-journal subset
- builds a balanced 7,711-record corpus for modeling

The raw database inspection still operates on the full SQLite schema.

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the SQLite file exists in `data/raw/`.

## Run the Notebook

Generate the notebook if needed:

```bash
python scripts/generate_notebook.py
```

Open `final_project.ipynb` in Jupyter or VS Code and run the cells in order.

Notes:

- the default pipeline build is optimized for faster startup
- the notebook includes a dedicated full clustering analysis step for the `K=2..80` sweep
- the first Sentence-BERT run may download `sentence-transformers/all-MiniLM-L6-v2`

## Run the Streamlit App

```bash
streamlit run app.py
```

The app is **not** a chatbot. It provides:

- abstract input area
- model selection: TF-IDF, BERT, Hybrid, Ensemble
- top 5 journal recommendations
- explanation text
- cluster label
- confidence
- supporting article examples

## Methodology Summary

### 1. Database Inspection

The project inspects:

- all table names from `sqlite_master`
- all columns via `PRAGMA table_info`
- primary keys
- declared foreign keys
- sample rows from each table

### 2. Dataset Construction

The final modeling table includes:

- `article_id`
- `title`
- `abstract`
- `journal_name`
- `keywords`
- `subjects`
- `keyword_plus`

One-to-many relations are aggregated into text strings and normalized term lists.

### 3. Text Preprocessing

The preprocessing layer applies:

- lowercase normalization
- HTML stripping
- punctuation removal
- stopword removal
- missing-value handling

The main feature field is:

```text
combined_text = abstract + keywords + subjects + keyword_plus
```

### 4. Recommendation Models

- **TF-IDF**: lexical similarity baseline
- **BERT**: semantic similarity using Sentence-BERT
- **Hybrid**: `0.6 * similarity + 0.2 * keyword_overlap + 0.2 * subject_overlap`
- **Ensemble**: `0.5 * tfidf + 0.5 * bert`

### 5. Explainability

Each recommendation includes:

- overlapping keywords
- overlapping subjects
- similar article examples
- cluster label
- confidence score

### 6. Clustering and Topics

- KMeans clustering over TF-IDF semantic space
- `K=2..80` exploration with elbow and silhouette
- cluster-size balance analysis
- LDA topic modeling for latent topic interpretation

### 7. Evaluation

For each article:

- use its abstract as the query
- exclude the article itself from retrieval
- measure Top-1 accuracy, Top-3 hit rate, Top-5 hit rate

## Expected Outputs

Running the notebook/app will produce:

- journal recommendation results
- clustering diagnostics
- topic summaries
- evaluation tables
- figures saved under `figures/`

## Reproducibility Notes

- the pipeline caches intermediate assets in `artifacts/cache/`
- the app uses the same shared code as the notebook
- if cached BERT files already exist, the project loads them offline first

## Suggested Demo Flow

1. Open the notebook and show schema inspection.
2. Show the assignment-aligned dataset summary.
3. Demonstrate TF-IDF vs BERT vs Hybrid recommendations.
4. Present clustering diagnostics and final `K`.
5. Show evaluation metrics and a few failed cases.
6. End with the Streamlit demo.
