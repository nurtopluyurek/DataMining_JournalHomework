# Computer Science Journal Recommendation Project

This project builds a journal recommendation system for computer science articles and an interpretable topic clustering pipeline using the provided scholarly metadata database.

## Project Scope

The system recommends the top 5 journals for a user-provided abstract and explains the result with keywords, similar articles, confidence scores, and cluster information.

The project includes:

- TF-IDF recommender
- Sentence-BERT recommender
- TF-IDF + BERT ensemble
- explainable outputs
- ablation study
- per-journal analysis
- confusion analysis
- class imbalance analysis
- confidence analysis
- topic clustering
- notebook, report draft, and generated outputs

## Dataset

The assignment PDF mentions `7711` articles from `175` journals, but the accessible SQLite database contains:

- `23,801` total records
- `466` journals

After filtering to `Article` records with non-empty abstracts and journals with at least `5` samples, the final recommendation dataset contains about:

- `21,907` articles
- `404` journals

## Main Results

Evaluation was done with a stratified `80/20` split using abstract-only query input.

- TF-IDF: `Top-1 = 0.3389`, `Top-3 = 0.5244`, `Top-5 = 0.6027`
- BERT: `Top-1 = 0.2752`, `Top-3 = 0.4411`, `Top-5 = 0.5358`
- Ensemble: `Top-1 = 0.2809`, `Top-3 = 0.4507`, `Top-5 = 0.5418`

TF-IDF is the final deployable model because it performs best on all main metrics. BERT and Ensemble are kept as semantic comparison baselines.

## Clustering

Topic clustering is built with TF-IDF, TruncatedSVD, and KMeans.

The cluster count `k` was rescanned from `10` to `60` and selected using the highest silhouette score.

- Final `k = 60`
- Best silhouette score = `0.1105`

## Important Files

- `notebooks/data_mining_final_project.ipynb`
- `src/project_lib.py`
- `scripts/build_project_assets.py`
- `scripts/generate_notebook.py`
- `report/ieee_final_project.tex`
- `outputs/tables/model_metrics.csv`
- `outputs/tables/ablation_results.csv`
- `outputs/tables/cluster_summary.csv`
- `outputs/tables/silhouette_scores.csv`
- `outputs/system_pipeline_diagram.png`
- `outputs/confidence_distribution.png`
- `outputs/case_study_examples.json`

## How to Run

Generate the notebook:

```powershell
& 'D:\jupyter_env\Scripts\python.exe' 'scripts\generate_notebook.py'
