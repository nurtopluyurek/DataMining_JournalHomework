from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="crest")


def _finalize(fig: plt.Figure, output_path: Path | None = None) -> plt.Figure:
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_journal_distribution(journal_distribution: pd.DataFrame, top_n: int = 20, output_path: Path | None = None) -> plt.Figure:
    frame = journal_distribution.head(top_n).sort_values("article_count")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(frame["journal_name"], frame["article_count"], color="#0f766e")
    ax.set_title(f"Top {top_n} Journals by Article Count")
    ax.set_xlabel("Article Count")
    ax.set_ylabel("Journal")
    return _finalize(fig, output_path)


def plot_term_frequency(term_distribution: pd.DataFrame, label: str, top_n: int = 20, output_path: Path | None = None) -> plt.Figure:
    frame = term_distribution.head(top_n).sort_values("frequency")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(frame.iloc[:, 0], frame["frequency"], color="#1d4ed8")
    ax.set_title(f"Top {top_n} {label}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel(label)
    return _finalize(fig, output_path)


def plot_clustering_diagnostics(diagnostics: pd.DataFrame, output_path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(diagnostics["k"], diagnostics["inertia"], color="#7c3aed", linewidth=2)
    axes[0].set_title("Elbow Curve")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(diagnostics["k"], diagnostics["silhouette"], color="#ea580c", linewidth=2)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    return _finalize(fig, output_path)


def plot_cluster_projection(projection: pd.DataFrame, output_path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=projection,
        x="x",
        y="y",
        hue="cluster_label",
        s=25,
        alpha=0.75,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    ax.set_title("Cluster Projection (PCA on TF-IDF Semantic Space)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    return _finalize(fig, output_path)


def plot_evaluation_comparison(summary: pd.DataFrame, output_path: Path | None = None) -> plt.Figure:
    frame = summary.melt(id_vars="model", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=frame, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Evaluation Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    return _finalize(fig, output_path)
