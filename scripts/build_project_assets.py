from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_lib import (  # noqa: E402
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
    plot_ablation_results,
    plot_cluster_projection,
    plot_confidence_distribution,
    plot_journal_imbalance,
    plot_metric_comparison,
    plot_publication_trend,
    plot_system_pipeline,
    plot_top_journals,
    select_demo_examples,
    run_ablation_study,
    summarize_confidence_performance,
    summarize_class_imbalance,
    summarize_dataset,
    train_test_split_by_journal,
)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_demo_payload(
    models: dict[str, object],
    demo_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    payload = []
    for _, row in demo_frame.iterrows():
        payload.append(
            {
                "record_id": int(row["record_id"]),
                "title": row["title"],
                "true_journal": row["journal"],
                "subjects": row["subjects"],
                "abstract": row["abstract"],
                "recommendations": {
                    model_name: model.recommend(row["abstract"], top_k=5) for model_name, model in models.items()
                },
            }
        )

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_assets(sqlite_path: str | Path = DEFAULT_SQLITE_PATH) -> None:
    outputs_dir = ROOT / "outputs"
    figures_dir = outputs_dir / "figures"
    tables_dir = outputs_dir / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_frame = load_dataset(sqlite_path)
    modeling_frame, modeling_stats = filter_modeling_dataset(dataset_frame)
    clusterer = TopicClusterer()
    clusterer.fit(modeling_frame)
    modeling_frame = attach_cluster_annotations(modeling_frame, clusterer)
    train_frame, test_frame = train_test_split_by_journal(modeling_frame)

    summary_frame = summarize_dataset(dataset_frame)
    summary_frame.to_csv(tables_dir / "dataset_summary.csv", index=False)
    pd.DataFrame([modeling_stats]).to_csv(tables_dir / "modeling_stats.csv", index=False)
    class_counts_frame, imbalance_summary = summarize_class_imbalance(dataset_frame)
    class_counts_frame.to_csv(tables_dir / "journal_article_counts.csv", index=False)
    imbalance_summary.to_csv(tables_dir / "class_imbalance_summary.csv", index=False)

    ablation_frame = run_ablation_study(train_frame, test_frame)
    ablation_frame.to_csv(tables_dir / "ablation_results.csv", index=False)

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

    evaluation_rows: list[dict[str, object]] = []

    tfidf_result = tfidf_model.evaluate(test_frame, query_column="abstract_query_text", name="TF-IDF Recommender")
    evaluation_rows.append(
        {
            **metrics_row("TF-IDF Recommender", tfidf_result.metrics),
            "evaluation_mode": "abstract_only",
        }
    )
    tfidf_result.confusion_pairs.to_csv(tables_dir / "tfidf_confusion_pairs.csv", index=False)

    bert_result = bert_model.evaluate(test_frame, query_column="abstract_query_text", name="BERT Recommender")
    evaluation_rows.append(
        {
            **metrics_row("BERT Recommender", bert_result.metrics),
            "evaluation_mode": "abstract_only",
        }
    )
    bert_result.confusion_pairs.to_csv(tables_dir / "bert_confusion_pairs.csv", index=False)

    ensemble_result = ensemble_model.evaluate(test_frame, query_column="abstract_query_text", name="Ensemble Recommender")
    evaluation_rows.append(
        {
            **metrics_row("Ensemble Recommender", ensemble_result.metrics),
            "evaluation_mode": "abstract_only",
        }
    )
    ensemble_result.confusion_pairs.to_csv(tables_dir / "ensemble_confusion_pairs.csv", index=False)
    tfidf_result.confusion_pairs.to_csv(tables_dir / "top_confused_journal_pairs.csv", index=False)

    metrics_frame = pd.DataFrame(evaluation_rows)
    metrics_frame.to_csv(tables_dir / "model_metrics.csv", index=False)
    clusterer.summarize_clusters().to_csv(tables_dir / "cluster_summary.csv", index=False)
    clusterer.silhouette_table_.to_csv(tables_dir / "silhouette_scores.csv", index=False)

    per_journal_frame = compute_per_journal_top5(tfidf_result.predictions, min_test_samples=5)
    per_journal_frame.to_csv(tables_dir / "per_journal_top5.csv", index=False)
    per_journal_frame.head(10).to_csv(tables_dir / "best_performing_journals.csv", index=False)
    per_journal_frame.tail(10).sort_values(["top_5_hit_rate", "test_samples", "true_journal"]).to_csv(
        tables_dir / "worst_performing_journals.csv",
        index=False,
    )

    tfidf_predictions_enriched = tfidf_result.predictions.merge(
        test_frame[["record_id", "abstract", "subjects", "cluster_label"]],
        on="record_id",
        how="left",
    )
    confidence_summary, high_confidence_examples, low_confidence_examples = summarize_confidence_performance(
        tfidf_predictions_enriched,
        example_count=10,
    )
    confidence_summary.to_csv(tables_dir / "confidence_summary.csv", index=False)
    high_confidence_examples.to_csv(tables_dir / "high_confidence_examples.csv", index=False)
    low_confidence_examples.to_csv(tables_dir / "low_confidence_examples.csv", index=False)

    demo_frame = select_demo_examples(test_frame, ensemble_result.predictions, n_examples=3)
    demo_frame.to_csv(tables_dir / "demo_examples.csv", index=False)
    _write_demo_payload(
        {
            "tfidf": tfidf_model,
            "bert": bert_model,
            "ensemble": ensemble_model,
        },
        demo_frame,
        tables_dir / "demo_recommendations.json",
    )

    case_study_payload = build_case_study_examples(
        {
            "tfidf": tfidf_model,
            "bert": bert_model,
            "ensemble": ensemble_model,
        },
        top_k=5,
    )
    (outputs_dir / "case_study_examples.json").write_text(json.dumps(case_study_payload, indent=2), encoding="utf-8")

    publication_figure, publication_axis = plt.subplots(figsize=(10, 4))
    plot_publication_trend(dataset_frame, ax=publication_axis)
    _save_figure(publication_figure, figures_dir / "publication_trend.png")

    journal_figure, journal_axis = plt.subplots(figsize=(10, 6))
    plot_top_journals(modeling_frame, ax=journal_axis)
    _save_figure(journal_figure, figures_dir / "top_journals.png")

    abstract_metric_frame = metrics_frame.loc[metrics_frame["evaluation_mode"] == "abstract_only"].reset_index(drop=True)
    metric_figure, metric_axis = plt.subplots(figsize=(9, 4))
    plot_metric_comparison(abstract_metric_frame, ax=metric_axis)
    _save_figure(metric_figure, figures_dir / "abstract_model_comparison.png")

    ablation_figure, ablation_axis = plt.subplots(figsize=(11, 5))
    plot_ablation_results(ablation_frame, ax=ablation_axis)
    _save_figure(ablation_figure, figures_dir / "ablation_comparison.png")

    cluster_figure, cluster_axis = plt.subplots(figsize=(10, 6))
    plot_cluster_projection(clusterer, ax=cluster_axis)
    _save_figure(cluster_figure, figures_dir / "cluster_projection.png")

    imbalance_figure, imbalance_axis = plt.subplots(figsize=(10, 5))
    plot_journal_imbalance(class_counts_frame, ax=imbalance_axis)
    _save_figure(imbalance_figure, figures_dir / "journal_imbalance.png")

    pipeline_figure, pipeline_axis = plt.subplots(figsize=(15, 4))
    plot_system_pipeline(ax=pipeline_axis)
    _save_figure(pipeline_figure, outputs_dir / "system_pipeline_diagram.png")

    confidence_figure, confidence_axis = plt.subplots(figsize=(10, 5))
    plot_confidence_distribution(tfidf_predictions_enriched, ax=confidence_axis)
    _save_figure(confidence_figure, outputs_dir / "confidence_distribution.png")

    expected_paths = [
        tables_dir / "dataset_summary.csv",
        tables_dir / "modeling_stats.csv",
        tables_dir / "model_metrics.csv",
        tables_dir / "ablation_results.csv",
        tables_dir / "per_journal_top5.csv",
        tables_dir / "top_confused_journal_pairs.csv",
        tables_dir / "class_imbalance_summary.csv",
        tables_dir / "confidence_summary.csv",
        tables_dir / "high_confidence_examples.csv",
        tables_dir / "low_confidence_examples.csv",
        tables_dir / "cluster_summary.csv",
        outputs_dir / "system_pipeline_diagram.png",
        outputs_dir / "case_study_examples.json",
        outputs_dir / "confidence_distribution.png",
    ]
    manifest_frame = pd.DataFrame(
        [{"path": str(path.relative_to(ROOT)), "exists": path.exists()} for path in expected_paths]
    )
    manifest_frame.to_csv(tables_dir / "artifact_manifest.csv", index=False)


if __name__ == "__main__":
    build_assets()
