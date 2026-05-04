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
```

Build figures and tables:

```powershell
$env:LOKY_MAX_CPU_COUNT='8'
& 'D:\jupyter_env\Scripts\python.exe' 'scripts\build_project_assets.py'
```

Execute the notebook in place:

```powershell
$env:LOKY_MAX_CPU_COUNT='8'
$env:JUPYTER_ALLOW_INSECURE_WRITES='true'
& 'D:\jupyter_env\Scripts\python.exe' -m jupyter nbconvert --to notebook --execute --inplace 'notebooks\data_mining_final_project.ipynb' --ExecutePreprocessor.timeout=1800
```

Run smoke tests:

```powershell
& 'D:\jupyter_env\Scripts\python.exe' -m unittest 'tests\test_project_lib.py'
```

Sentence-BERT model weights are cached under:

```text
outputs/cache/models
```

Article embedding caches are stored under:

```text
outputs/cache
```

## Important files

- [Metrics table](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/model_metrics.csv)
- [Ablation results](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/ablation_results.csv)
- [Per-journal performance](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/per_journal_top5.csv)
- [Confused journal pairs](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/top_confused_journal_pairs.csv)
- [Class imbalance summary](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/class_imbalance_summary.csv)
- [Cluster summary](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/cluster_summary.csv)
- [Demo recommendations](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/demo_recommendations.json)
- [Model comparison figure](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/figures/abstract_model_comparison.png)
- [Ablation comparison figure](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/figures/ablation_comparison.png)
- [Journal imbalance figure](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/figures/journal_imbalance.png)
- [Cluster projection figure](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/figures/cluster_projection.png)

## Notes

- The notebook and the report are written in English because the assignment expects an academic submission format.
- Notebook execution on Windows required `JUPYTER_ALLOW_INSECURE_WRITES=true` because the local Jupyter runtime attempted to apply unsupported secure-write permissions.
- The Sentence-BERT model used in this project is `all-MiniLM-L6-v2`.
- The clustering section now also includes an explicit `K=30 vs K=60` comparison to justify the final choice of `k=60`.
