# Data Mining Final Project

This project implements a computer science journal recommendation system and an interpretable topic clustering pipeline over the provided SQLite database.

The recommendation layer now includes:

- TF-IDF recommender
- Sentence-BERT semantic recommender
- TF-IDF + BERT ensemble recommender
- Explainable recommendation payloads with keywords, evidence articles, cluster labels, and confidence scores

The academic analysis layer now also includes:

- ablation study across multiple input representations
- per-journal Top-5 performance analysis
- confusion analysis for rank-1 journal mistakes
- class imbalance analysis and visualization
- explicit final model selection discussion

## What is included

- [Notebook](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/notebooks/data_mining_final_project.ipynb)
- [Project library](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/src/project_lib.py)
- [Asset builder](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/scripts/build_project_assets.py)
- [IEEE report draft](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/report/ieee_final_project.tex)
- [Generated figures and tables](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs)

## Dataset assumptions

- Primary data source: `C:\Users\toplu\Downloads\CompSciencePub (1).sqlite`
- The assignment PDF says `7711` articles from `175` journals.
- The accessible SQLite file actually contains `23801` records from `466` journals.
- The supervised modeling subset filters to `21907` article records with abstracts and journals having at least 5 examples.

## Main results

- TF-IDF recommender: `Top-1 = 0.3389`, `Top-3 = 0.5244`, `Top-5 = 0.6027`
- BERT recommender: `Top-1 = 0.2752`, `Top-3 = 0.4411`, `Top-5 = 0.5358`
- Ensemble recommender: `Top-1 = 0.2809`, `Top-3 = 0.4507`, `Top-5 = 0.5418`

On this dataset, the tuned TF-IDF model remains the strongest deployment model.  
The BERT and ensemble layers still add semantic retrieval and more explainable evidence for similar-article support.

## Stronger Academic Analysis

### Ablation Study

The project compares these input representations with the same TF-IDF recommendation pipeline:

- `abstract only`
- `title + abstract`
- `abstract + keywords`
- `abstract + keywords + subjects`
- `full combined_text`

Generated files:

- [Ablation table](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/ablation_results.csv)
- [Ablation chart](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/figures/ablation_comparison.png)

### Per-Journal Performance

For journals with at least `5` test samples, the project computes Top-5 hit rate per journal.

Generated files:

- [Per-journal Top-5 table](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/per_journal_top5.csv)
- [Best-performing journals](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/best_performing_journals.csv)
- [Worst-performing journals](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/worst_performing_journals.csv)

Interpretation:

- Journals with more samples and clearer topical scope perform better.
- Broad or interdisciplinary journals tend to produce more ambiguous recommendation boundaries.

### Confusion Analysis

Incorrect predictions are analyzed as true journal vs. rank-1 predicted journal pairs.

Generated file:

- [Top confused journal pairs](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/top_confused_journal_pairs.csv)

Interpretation:

- Many confusions are caused by overlapping journal scope.
- Interdisciplinary abstracts often share vocabulary with several candidate journals.

### Class Imbalance Analysis

The project analyzes article counts per journal and explicitly counts journals with fewer than `5`, `10`, and `20` articles.

Generated files:

- [Journal article counts](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/journal_article_counts.csv)
- [Class imbalance summary](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/tables/class_imbalance_summary.csv)
- [Journal imbalance plot](/C:/Users/toplu/Documents/Codex/2026-04-26/files-mentioned-by-the-user-compsciencepub/outputs/figures/journal_imbalance.png)

Interpretation:

- Imbalance reduces recommendation quality for sparse journals.
- Frequent journals provide stronger lexical patterns and better classifier calibration.

### Final Model Selection

The final deployable model is **TF-IDF** because:

- it performs best on all metrics
- journal recommendation is keyword-sensitive
- domain-specific terminology matters
- BERT is kept as a semantic comparison baseline

## How to run

Use the existing local Jupyter environment at `D:\jupyter_env\Scripts\python.exe`.

Generate notebook:

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
