from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "final_project.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
            # Data Mining Final Project
            ## Computer Science Journal Recommendation System

            This notebook builds a journal recommendation system that predicts the **top 5 most relevant journals**
            for a new computer science article abstract. The implementation includes:

            - database inspection from the provided SQLite file
            - dataset construction with dynamic joins
            - text preprocessing and feature engineering
            - TF-IDF, BERT, hybrid, and ensemble recommendation models
            - clustering, topic modeling, evaluation, error analysis, and journal insights
            """
        ),
        md(
            """
            ## Assignment Alignment Note

            The PDF specifies **7711 articles from 175 journals**, but the provided SQLite database contains a larger
            corpus. To stay faithful to the assignment while remaining reproducible, the notebook uses the
            `assignment_aligned` profile:

            - it filters to English journal articles/reviews with abstracts
            - it selects a stable 175-journal subset
            - it builds a balanced 7,711-record corpus for modeling

            The raw schema inspection still reflects the full SQLite file.
            """
        ),
        code(
            """
            import json
            import sys
            from pathlib import Path

            import pandas as pd

            PROJECT_ROOT = Path.cwd()
            SRC_PATH = PROJECT_ROOT / "src"
            if str(SRC_PATH) not in sys.path:
                sys.path.insert(0, str(SRC_PATH))

            from journal_recommender import (
                DatasetConfig,
                JournalRecommendationPipeline,
                RecommenderConfig,
                inspect_sqlite_schema,
                schema_summary_frame,
            )
            from journal_recommender.visualization import (
                plot_cluster_projection,
                plot_clustering_diagnostics,
                plot_evaluation_comparison,
                plot_journal_distribution,
                plot_term_frequency,
            )

            pd.set_option("display.max_colwidth", 150)
            """
        ),
        md("## 1. Database Inspection"),
        code(
            """
            schema = inspect_sqlite_schema()
            print("All tables:")
            for table_name in schema["tables"]:
                print("-", table_name)
            """
        ),
        code(
            """
            schema_frame = schema_summary_frame(schema)
            schema_frame.head(30)
            """
        ),
        code(
            """
            relevant_tables = [
                "AcademicRecord",
                "AcademicRecordAbstract",
                "Publication",
                "AcademicRecordKeyword",
                "AcademicKeyword",
                "AcademicRecordSubject",
                "AcademicSubject",
                "AcademicRecordKeywordPlus",
                "AcademicKeywordPlus",
            ]

            for table_name in relevant_tables:
                details = schema["table_details"][table_name]
                print(f"\\n[{table_name}]")
                print("Primary keys:", details["primary_keys"])
                print("Foreign keys:", details["foreign_keys"] or "None declared")
                print("Sample rows:")
                for row in details["sample_rows"][:2]:
                    print(row)
            """
        ),
        md("## 2. Dataset Construction"),
        code(
            """
            dataset_config = DatasetConfig(profile="assignment_aligned")
            pipeline = JournalRecommendationPipeline(
                RecommenderConfig(dataset_config=dataset_config)
            ).build(use_cache=True, include_evaluation=False, full_cluster_search=False)

            dataset = pipeline.dataset
            profile_metadata = pipeline.profile_metadata
            dataset.head()
            """
        ),
        code(
            """
            profile_metadata
            """
        ),
        code(
            """
            dataset[[
                "article_id",
                "title",
                "abstract",
                "journal_name",
                "keywords",
                "subjects",
                "keyword_plus",
                "combined_text",
            ]].head(3)
            """
        ),
        md(
            """
            ## 3. Text Preprocessing

            The preprocessing pipeline performs:

            - lowercase normalization
            - punctuation removal
            - stopword removal
            - HTML stripping from abstracts
            - missing-value handling

            The final feature field is:

            `combined_text = abstract + keywords + subjects + keyword_plus`
            """
        ),
        code(
            """
            dataset[[
                "abstract_clean",
                "keywords_clean",
                "subjects_clean",
                "keyword_plus_clean",
                "combined_text",
            ]].head(2)
            """
        ),
        md("## 4. Baseline Model: TF-IDF"),
        code(
            """
            demo_abstract = (
                "Distributed data processing is becoming a reality. Modern distributed systems must integrate "
                "heterogeneous data sources, reduce communication costs, exploit replication and caching, and "
                "support parallel query processing."
            )
            pipeline.recommend_journals_tfidf(demo_abstract)
            """
        ),
        md("## 5. Semantic Model: Sentence-BERT"),
        code(
            """
            pipeline.recommend_journals_bert(demo_abstract)
            """
        ),
        md("## 6. Hybrid Ranking and 14. Ensemble Model"),
        code(
            """
            hybrid_results = pipeline.recommend_journals_hybrid(demo_abstract)
            ensemble_results = pipeline.recommend_journals_ensemble(demo_abstract)
            hybrid_results, ensemble_results
            """
        ),
        md(
            """
            ## 7. Explainable AI

            Each recommendation includes:

            - overlapping keywords
            - overlapping subjects
            - supporting similar articles
            - cluster label
            - confidence label and score
            """
        ),
        code(
            """
            hybrid_results[[
                "journal_name",
                "score_percent",
                "confidence_label",
                "cluster_label",
                "overlapping_keywords",
                "overlapping_subjects",
                "explanation",
            ]]
            """
        ),
        md("## 8. KMeans Clustering with K = 2 to 80"),
        code(
            """
            cluster_diagnostics = pipeline.run_full_cluster_analysis(use_cache=True)
            cluster_diagnostics.head()
            """
        ),
        code(
            """
            plot_clustering_diagnostics(
                cluster_diagnostics,
                output_path=PROJECT_ROOT / "figures" / "cluster_diagnostics.png",
            )
            """
        ),
        code(
            """
            cluster_diagnostics.sort_values("selection_score", ascending=False).head(10)
            """
        ),
        md(
            """
            The final K is selected by balancing:

            - silhouette score
            - elbow curvature
            - cluster-size balance
            - interpretability

            If the winning K lands around **46–52**, that is reasonable because the assignment-aligned corpus contains
            175 journals spanning many computer science subfields, so a medium-high cluster count is often needed to
            separate topics without collapsing distinct areas into a few broad groups.
            """
        ),
        code(
            """
            projection = pipeline.projection_frame()
            plot_cluster_projection(
                projection,
                output_path=PROJECT_ROOT / "figures" / "cluster_projection.png",
            )
            """
        ),
        md("## 9. Topic Modeling with LDA"),
        code(
            """
            pipeline.topic_summary
            """
        ),
        md("## 10. Cluster Interpretation"),
        code(
            """
            pipeline.cluster_interpretation
            """
        ),
        md("## 11. Evaluation"),
        code(
            """
            evaluation_summary = pipeline.evaluate_all_models(use_cache=True)
            evaluation_summary
            """
        ),
        code(
            """
            plot_evaluation_comparison(
                evaluation_summary,
                output_path=PROJECT_ROOT / "figures" / "evaluation_comparison.png",
            )
            """
        ),
        md("## 12. Error Analysis"),
        code(
            """
            pipeline.error_analysis.head(20)
            """
        ),
        md("## 13. Journal Insights"),
        code(
            """
            insights = pipeline.journal_insights()
            insights["journal_distribution"].head()
            """
        ),
        code(
            """
            plot_journal_distribution(
                insights["journal_distribution"],
                output_path=PROJECT_ROOT / "figures" / "journal_distribution.png",
            )
            """
        ),
        code(
            """
            plot_term_frequency(
                insights["keyword_distribution"],
                label="Keywords",
                output_path=PROJECT_ROOT / "figures" / "keyword_frequency.png",
            )
            """
        ),
        code(
            """
            plot_term_frequency(
                insights["subject_distribution"],
                label="Subjects",
                output_path=PROJECT_ROOT / "figures" / "subject_frequency.png",
            )
            """
        ),
        md(
            """
            ## 15. Confidence Score

            Confidence is derived from:

            - the predicted journal score
            - the gap to the next-ranked journal

            Labels:

            - **High**: strong score and clear margin
            - **Medium**: moderate score or moderate margin
            - **Low**: close competition between journals
            """
        ),
        md(
            """
            ## 16. Visualizations Included

            The notebook and project generate:

            - journal distribution
            - keyword frequency
            - subject frequency
            - elbow curve
            - silhouette curve
            - cluster projection
            - evaluation comparison chart
            """
        ),
        md(
            """
            ## Conclusion

            This notebook implements a full data-mining workflow for journal recommendation:

            - structured relational data is transformed into a text-rich corpus
            - lexical and semantic recommenders are compared
            - explainability is surfaced for each prediction
            - clusters and latent topics support interpretability
            - evaluation is performed with hidden-journal testing
            """
        ),
    ]
    return nb


if __name__ == "__main__":
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook written to {NOTEBOOK_PATH}")
