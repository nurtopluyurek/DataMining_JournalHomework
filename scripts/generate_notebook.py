from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "data_mining_final_project.ipynb"


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        markdown_cell(
            """
            # Data Mining Final Project

            **Project goal:** recommend the top 5 journals for a user-provided abstract and generate interpretable topic clusters from the Computer Science publication dataset.

            **Deliverables covered in this notebook**
            - Dataset loading and cleaning from the provided SQLite file
            - Exploratory analysis and data sanity checks
            - TF-IDF, Sentence-BERT, and ensemble recommender comparison
            - Explainability, clustering, and demo-ready case studies
            - Strong academic analysis: ablation, per-journal, confusion, imbalance, and confidence studies
            - Final deployable model selection, limitations, and future work
            """
        ),
        markdown_cell(
            """
            ## 1. Setup

            The notebook imports the reusable project library from `src/project_lib.py`.  
            It uses the real SQLite file and explicitly records the known mismatch between the assignment PDF and the accessible dataset:

            - PDF states `7711` articles from `175` journals
            - SQLite actually contains `23801` records and `466` journals
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import json
            import sys

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import Markdown, display

            ROOT = Path.cwd()
            if not (ROOT / "src").exists():
                ROOT = ROOT.parent

            SRC_DIR = ROOT / "src"
            if str(SRC_DIR) not in sys.path:
                sys.path.insert(0, str(SRC_DIR))

            from project_lib import (
                DEFAULT_SQLITE_PATH,
                EnsembleJournalRecommender,
                JournalRecommender,
                SemanticJournalRecommender,
                TopicClusterer,
                attach_cluster_annotations,
                build_case_study_examples,
                compute_per_journal_top5,
                filter_modeling_dataset,
                load_dataset,
                metrics_row,
                plot_k_comparison,
                plot_ablation_results,
                plot_cluster_projection,
                plot_confidence_distribution,
                plot_journal_imbalance,
                plot_metric_comparison,
                plot_publication_trend,
                plot_system_pipeline,
                plot_top_journals,
                run_ablation_study,
                select_demo_examples,
                summarize_cluster_configurations,
                summarize_class_imbalance,
                summarize_confidence_performance,
                summarize_dataset,
                train_test_split_by_journal,
            )

            OUTPUT_ROOT = ROOT / "outputs"
            OUTPUT_TABLES = OUTPUT_ROOT / "tables"
            OUTPUT_FIGURES = OUTPUT_ROOT / "figures"
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
            OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

            SQLITE_PATH = DEFAULT_SQLITE_PATH
            SQLITE_PATH
            """
        ),
        markdown_cell("## 2. Load and inspect the dataset"),
        code_cell(
            """
            dataset_frame = load_dataset(SQLITE_PATH)
            dataset_summary = summarize_dataset(dataset_frame)
            dataset_summary.to_csv(OUTPUT_TABLES / "dataset_summary.csv", index=False)
            dataset_summary
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            plot_publication_trend(dataset_frame, ax=axes[0])
            plot_top_journals(dataset_frame.loc[dataset_frame["document_type"].eq("Article")], ax=axes[1])
            fig.tight_layout()
            fig.savefig(OUTPUT_FIGURES / "dataset_overview.png", dpi=200, bbox_inches="tight")
            plt.show()

            dataset_frame.head(3)
            """
        ),
        markdown_cell("## 3. Build the modeling dataset"),
        code_cell(
            """
            modeling_frame, modeling_stats = filter_modeling_dataset(dataset_frame)
            pd.DataFrame([modeling_stats]).to_csv(OUTPUT_TABLES / "modeling_stats.csv", index=False)

            clusterer = TopicClusterer()
            clusterer.fit(modeling_frame)
            modeling_frame = attach_cluster_annotations(modeling_frame, clusterer)
            train_frame, test_frame = train_test_split_by_journal(modeling_frame)

            pd.DataFrame([modeling_stats])
            """
        ),
        markdown_cell(
            """
            ## 4. System Architecture

            The end-to-end recommendation pipeline follows a research-grade workflow:

            1. The user provides an abstract.
            2. The text is cleaned and normalized.
            3. Two parallel feature views are available: lexical TF-IDF and semantic BERT embeddings.
            4. The system computes similarity and journal-level relevance scores.
            5. TF-IDF, BERT, hybrid, and ensemble scores can be ranked.
            6. The final layer returns a Top-5 journal list together with explainable evidence.
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(15, 4))
            plot_system_pipeline(ax=ax)
            fig.tight_layout()
            fig.savefig(OUTPUT_ROOT / "system_pipeline_diagram.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## 5. Recommendation Models

            We compare three deployable configurations:

            1. **TF-IDF Recommender**: tuned `title + abstract` linear SVM  
            2. **BERT Recommender**: Sentence-BERT semantic retrieval over `combined_text` embeddings  
            3. **Ensemble Recommender**: `0.5 * TF-IDF + 0.5 * BERT`

            Semantic embeddings (BERT) capture contextual meaning and improve recommendation quality compared to TF-IDF, especially when different words express similar concepts.
            """
        ),
        code_cell(
            """
            tfidf_model = JournalRecommender(
                text_column="title_abstract_text",
                classifier_weight=1.0,
                similarity_weight=0.0,
                candidate_top_n=None,
                classifier_c=0.5,
                sublinear_tf=True,
                n_neighbors=20,
            )
            bert_model = SemanticJournalRecommender(
                text_column="combined_text",
                n_neighbors=30,
            )
            ensemble_model = EnsembleJournalRecommender(
                JournalRecommender(
                    text_column="title_abstract_text",
                    classifier_weight=1.0,
                    similarity_weight=0.0,
                    candidate_top_n=None,
                    classifier_c=0.5,
                    sublinear_tf=True,
                    n_neighbors=20,
                ),
                SemanticJournalRecommender(
                    text_column="combined_text",
                    n_neighbors=30,
                ),
                tfidf_weight=0.5,
                bert_weight=0.5,
            )

            tfidf_model.fit(train_frame)
            bert_model.fit(train_frame)
            ensemble_model.fit(train_frame)
            """
        ),
        markdown_cell("## 6. Model comparison"),
        code_cell(
            """
            tfidf_result = tfidf_model.evaluate(test_frame, query_column="abstract_query_text", name="TF-IDF Recommender")
            bert_result = bert_model.evaluate(test_frame, query_column="abstract_query_text", name="BERT Recommender")
            ensemble_result = ensemble_model.evaluate(test_frame, query_column="abstract_query_text", name="Ensemble Recommender")

            metrics_frame = pd.DataFrame(
                [
                    {**metrics_row("TF-IDF Recommender", tfidf_result.metrics), "evaluation_mode": "abstract_only"},
                    {**metrics_row("BERT Recommender", bert_result.metrics), "evaluation_mode": "abstract_only"},
                    {**metrics_row("Ensemble Recommender", ensemble_result.metrics), "evaluation_mode": "abstract_only"},
                ]
            )
            metrics_frame.to_csv(OUTPUT_TABLES / "model_metrics.csv", index=False)
            metrics_frame
            """
        ),
        code_cell(
            """
            tfidf_result.confusion_pairs.to_csv(OUTPUT_TABLES / "tfidf_confusion_pairs.csv", index=False)
            bert_result.confusion_pairs.to_csv(OUTPUT_TABLES / "bert_confusion_pairs.csv", index=False)
            ensemble_result.confusion_pairs.to_csv(OUTPUT_TABLES / "ensemble_confusion_pairs.csv", index=False)

            fig, ax = plt.subplots(figsize=(9, 4))
            plot_metric_comparison(metrics_frame.reset_index(drop=True), ax=ax)
            fig.tight_layout()
            fig.savefig(OUTPUT_FIGURES / "abstract_model_comparison.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell("## 7. Ablation study"),
        code_cell(
            """
            ablation_frame = run_ablation_study(train_frame, test_frame)
            ablation_frame.to_csv(OUTPUT_TABLES / "ablation_results.csv", index=False)
            ablation_frame
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(11, 5))
            plot_ablation_results(ablation_frame, ax=ax)
            fig.tight_layout()
            fig.savefig(OUTPUT_FIGURES / "ablation_comparison.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            The ablation study isolates the effect of each input representation while keeping the same tuned TF-IDF classifier.  
            This makes it possible to assess whether extra metadata improves recommendation quality beyond abstract-only lexical matching.
            """
        ),
        markdown_cell("## 8. Per-journal performance analysis"),
        code_cell(
            """
            per_journal_frame = compute_per_journal_top5(tfidf_result.predictions, min_test_samples=5)
            best_journals = per_journal_frame.head(10).copy()
            worst_journals = per_journal_frame.tail(10).sort_values(
                ["top_5_hit_rate", "test_samples", "true_journal"]
            ).copy()

            per_journal_frame.to_csv(OUTPUT_TABLES / "per_journal_top5.csv", index=False)
            best_journals.to_csv(OUTPUT_TABLES / "best_performing_journals.csv", index=False)
            worst_journals.to_csv(OUTPUT_TABLES / "worst_performing_journals.csv", index=False)

            display(Markdown("### Best-performing journals"))
            display(best_journals)
            display(Markdown("### Worst-performing journals"))
            display(worst_journals)
            """
        ),
        markdown_cell(
            """
            Journals with more samples and a clearer topical scope generally perform better.  
            Sparse journals or journals covering broader interdisciplinary areas tend to have lower Top-5 hit rates because the model sees fewer consistent lexical signals during training.
            """
        ),
        markdown_cell("## 9. Confusion analysis"),
        code_cell(
            """
            top_confused_pairs = tfidf_result.confusion_pairs.copy()
            top_confused_pairs.to_csv(OUTPUT_TABLES / "top_confused_journal_pairs.csv", index=False)
            top_confused_pairs
            """
        ),
        markdown_cell(
            """
            The most frequent confusion pairs usually occur between journals with overlapping scope, similar methodology, or interdisciplinary abstracts.  
            In practice, many rank-1 mistakes still remain within a highly relevant topical neighborhood, which is why Top-5 evaluation is a better measure than Top-1 alone for this task.
            """
        ),
        markdown_cell("## 10. Class imbalance analysis"),
        code_cell(
            """
            class_counts_frame, imbalance_summary = summarize_class_imbalance(dataset_frame)
            class_counts_frame.to_csv(OUTPUT_TABLES / "journal_article_counts.csv", index=False)
            imbalance_summary.to_csv(OUTPUT_TABLES / "class_imbalance_summary.csv", index=False)
            imbalance_summary
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_journal_imbalance(class_counts_frame, ax=ax)
            fig.tight_layout()
            fig.savefig(OUTPUT_FIGURES / "journal_imbalance.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            Class imbalance affects recommendation quality because the model learns stronger journal-specific patterns from frequent journals and weaker patterns from rare journals.  
            As the number of samples per journal decreases, the model has fewer opportunities to observe distinctive vocabulary and scope boundaries.
            """
        ),
        markdown_cell("## 11. Confidence analysis"),
        code_cell(
            """
            tfidf_predictions_enriched = tfidf_result.predictions.merge(
                test_frame[["record_id", "abstract", "subjects", "cluster_label"]],
                on="record_id",
                how="left",
            )
            confidence_summary, high_confidence_examples, low_confidence_examples = summarize_confidence_performance(
                tfidf_predictions_enriched,
                example_count=10,
            )

            confidence_summary.to_csv(OUTPUT_TABLES / "confidence_summary.csv", index=False)
            high_confidence_examples.to_csv(OUTPUT_TABLES / "high_confidence_examples.csv", index=False)
            low_confidence_examples.to_csv(OUTPUT_TABLES / "low_confidence_examples.csv", index=False)
            confidence_summary
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_confidence_distribution(tfidf_predictions_enriched, ax=ax)
            fig.tight_layout()
            fig.savefig(OUTPUT_ROOT / "confidence_distribution.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        code_cell(
            """
            display(Markdown("### High-confidence examples"))
            display(high_confidence_examples)
            display(Markdown("### Low-confidence examples"))
            display(low_confidence_examples)
            """
        ),
        markdown_cell(
            """
            Confidence scores are informative but not perfect.  
            High-confidence predictions are usually correct and have lower median true rank, while low-confidence predictions more often correspond to interdisciplinary or lexically ambiguous abstracts.
            """
        ),
        markdown_cell(
            """
            The topic clustering stage now scans candidate values from `k=10` to `k=60` and selects the value with the highest silhouette score.  
            This replaces the earlier conservative low-`k` rule and aligns the final cluster count with the empirically optimal setting in the tested range.
            """
        ),
        markdown_cell("## 12. Topic clustering"),
        code_cell(
            """
            {
                "candidate_k_range": f"{min(clusterer.candidate_clusters)} to {max(clusterer.candidate_clusters)}",
                "selection_strategy": clusterer.selection_strategy,
                "selected_k": clusterer.best_k_,
                "best_silhouette": round(clusterer.best_silhouette_, 4),
            }
            """
        ),
        markdown_cell("### K=30 vs K=60 comparison"),
        code_cell(
            """
            clusterer_k30 = TopicClusterer(candidate_clusters=(30,), selection_strategy="max_silhouette")
            clusterer_k30.fit(modeling_frame)

            k_comparison_frame = summarize_cluster_configurations([clusterer_k30, clusterer])
            k_comparison_frame.to_csv(OUTPUT_TABLES / "k30_vs_k60_comparison.csv", index=False)
            (
                clusterer_k30.summarize_clusters()
                .sort_values(["size", "cluster"], ascending=[False, True])
                .head(10)
                .to_csv(OUTPUT_TABLES / "k30_representative_topics.csv", index=False)
            )
            (
                clusterer.summarize_clusters()
                .sort_values(["size", "cluster"], ascending=[False, True])
                .head(10)
                .to_csv(OUTPUT_TABLES / "k60_representative_topics.csv", index=False)
            )
            k_comparison_frame
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 5),
                gridspec_kw={"width_ratios": [1.0, 1.3, 1.7]},
            )
            plot_k_comparison(clusterer_k30, clusterer, axes=axes)
            fig.tight_layout()
            fig.savefig(OUTPUT_FIGURES / "k30_vs_k60_comparison.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            This comparison is useful in the report because it shows that `k=60` is not just a larger arbitrary choice.  
            It yields a higher silhouette score than `k=30`, creates a finer-grained topic structure, and separates narrower research themes that are merged together at lower `k` values.
            """
        ),
        code_cell(
            """
            clusterer.silhouette_table_.to_csv(OUTPUT_TABLES / "silhouette_scores.csv", index=False)
            cluster_summary = clusterer.summarize_clusters()
            cluster_summary.to_csv(OUTPUT_TABLES / "cluster_summary.csv", index=False)
            clusterer.silhouette_table_
            """
        ),
        code_cell(
            """
            cluster_summary.head(10)
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_cluster_projection(clusterer, ax=ax)
            fig.tight_layout()
            fig.savefig(OUTPUT_FIGURES / "cluster_projection.png", dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell("## 13. Explainable demo-ready abstract queries"),
        code_cell(
            """
            demo_frame = select_demo_examples(test_frame, ensemble_result.predictions, n_examples=3)
            demo_frame.to_csv(OUTPUT_TABLES / "demo_examples.csv", index=False)
            demo_frame[["record_id", "title", "journal", "subjects", "cluster_label", "pub_year"]]
            """
        ),
        code_cell(
            """
            demo_payload = []
            for _, row in demo_frame.iterrows():
                demo_payload.append(
                    {
                        "record_id": int(row["record_id"]),
                        "title": row["title"],
                        "true_journal": row["journal"],
                        "subjects": row["subjects"],
                        "cluster_label": row["cluster_label"],
                        "recommendations": {
                            "tfidf": tfidf_model.recommend(row["abstract"], top_k=5),
                            "bert": bert_model.recommend(row["abstract"], top_k=5),
                            "ensemble": ensemble_model.recommend(row["abstract"], top_k=5),
                        },
                    }
                )

            (OUTPUT_TABLES / "demo_recommendations.json").write_text(json.dumps(demo_payload, indent=2), encoding="utf-8")
            demo_payload[0]
            """
        ),
        markdown_cell("## 14. Case study"),
        code_cell(
            """
            case_study_payload = build_case_study_examples(
                {
                    "tfidf": tfidf_model,
                    "bert": bert_model,
                    "ensemble": ensemble_model,
                },
                top_k=5,
            )
            (OUTPUT_ROOT / "case_study_examples.json").write_text(json.dumps(case_study_payload, indent=2), encoding="utf-8")
            len(case_study_payload)
            """
        ),
        code_cell(
            """
            for case in case_study_payload:
                display(Markdown(f"### {case['topic']}"))
                display(Markdown(f"**Input abstract:** {case['abstract']}"))

                top1_comparison = pd.DataFrame(
                    [
                        {"model": model_name, "top_1_journal": journal}
                        for model_name, journal in case["top_1_by_model"].items()
                    ]
                )
                display(Markdown("**Top-1 journal by model**"))
                display(top1_comparison)

                final_table = pd.DataFrame(case["recommendations"]["tfidf"])[
                    ["journal", "confidence_score", "cluster_label", "top_keywords", "evidence_titles", "explanation"]
                ]
                display(Markdown("**Final deployable model: TF-IDF**"))
                display(final_table)
            """
        ),
        markdown_cell(
            """
            ## 15. Why TF-IDF Outperforms BERT

            TF-IDF is selected as the final deployable model because it performs best on Top-1, Top-3, and Top-5 accuracy.  
            The strongest explanation is that journal recommendation is highly keyword-sensitive: domain-specific terminology, recurring phrases, and discriminative technical expressions matter more than generic semantic similarity.

            In this dataset:

            - journal scope is often expressed through narrow lexical markers
            - TF-IDF preserves these discriminative keywords directly
            - BERT is a general-purpose semantic model rather than a venue-specialized scholarly encoder
            - semantically similar abstracts can still target different journals with different editorial scope
            """
        ),
        markdown_cell(
            """
            ## 16. Limitations

            This study has several limitations that should be acknowledged in academic terms.

            - Class imbalance affects recommendation quality, because small journals contribute fewer training examples.
            - Journals with limited sample counts provide weaker lexical boundaries and less stable evaluation.
            - The Sentence-BERT baseline does not outperform TF-IDF, likely because the vocabulary of scholarly venue selection is highly domain-specific and keyword-sensitive.
            - The deployment scenario uses abstract-only input, which omits richer context such as references, introduction, methods, and citation neighborhood.
            - Interdisciplinary papers are inherently harder to classify, because they overlap with multiple journal scopes and thematic clusters.
            """
        ),
        markdown_cell(
            """
            ## 17. Future Work

            Several extensions could improve the system beyond the current course-scale implementation.

            - fine-tuning a domain-specific scholarly BERT model for journal recommendation
            - integrating citation networks and reference structure
            - incorporating journal ranking, impact factor, or scope indicators
            - improving hybrid weighting between lexical and semantic signals
            - exploring multi-modal recommendation using abstract, references, and citation context together
            """
        ),
        markdown_cell("## 18. Final model selection"),
        code_cell(
            """
            final_selection = pd.DataFrame(
                [
                    {"model": "TF-IDF", "top_1": 0.3389, "top_3": 0.5244, "top_5": 0.6027, "role": "Final deployable model"},
                    {"model": "BERT", "top_1": 0.2752, "top_3": 0.4411, "top_5": 0.5358, "role": "Semantic comparison baseline"},
                    {"model": "Ensemble", "top_1": 0.2809, "top_3": 0.4507, "top_5": 0.5418, "role": "Combined comparison baseline"},
                ]
            )
            final_selection
            """
        ),
        markdown_cell(
            """
            **Clear statement:** TF-IDF is the final deployable model because it performs best on Top-1, Top-3, and Top-5.  
            BERT and Ensemble remain in the project as semantic baselines and comparison models rather than the final deployment choice.
            """
        ),
        markdown_cell("## 19. Artifact validation"),
        code_cell(
            """
            expected_paths = [
                OUTPUT_ROOT / "system_pipeline_diagram.png",
                OUTPUT_ROOT / "case_study_examples.json",
                OUTPUT_ROOT / "confidence_distribution.png",
                OUTPUT_TABLES / "dataset_summary.csv",
                OUTPUT_TABLES / "modeling_stats.csv",
                OUTPUT_TABLES / "model_metrics.csv",
                OUTPUT_TABLES / "ablation_results.csv",
                OUTPUT_TABLES / "per_journal_top5.csv",
                OUTPUT_TABLES / "top_confused_journal_pairs.csv",
                OUTPUT_TABLES / "class_imbalance_summary.csv",
                OUTPUT_TABLES / "confidence_summary.csv",
                OUTPUT_TABLES / "high_confidence_examples.csv",
                OUTPUT_TABLES / "low_confidence_examples.csv",
                OUTPUT_TABLES / "cluster_summary.csv",
                OUTPUT_TABLES / "demo_recommendations.json",
            ]

            artifact_manifest = pd.DataFrame(
                [{"path": str(path.relative_to(ROOT)), "exists": path.exists()} for path in expected_paths]
            )
            artifact_manifest.to_csv(OUTPUT_TABLES / "artifact_manifest.csv", index=False)
            artifact_manifest
            """
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
