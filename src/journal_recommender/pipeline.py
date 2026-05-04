from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors

from .config import ProjectPaths
from .data import DatasetConfig, build_dataset, normalize_text


def _safe_min_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        return np.ones_like(values) if maximum > 0 else np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def _as_percent(value: float) -> float:
    return round(float(value) * 100, 2)


def _match_terms(query_text: str, candidate_terms: list[str]) -> list[str]:
    if not query_text or not candidate_terms:
        return []
    padded = f" {query_text} "
    matches = [term for term in candidate_terms if f" {term} " in padded]
    return matches


def _jaccard_overlap(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _top_words_from_weights(weights: np.ndarray, feature_names: np.ndarray, top_n: int = 8) -> list[str]:
    indices = np.argsort(weights)[::-1][:top_n]
    return [str(feature_names[index]) for index in indices if weights[index] > 0]


@dataclass
class RecommenderConfig:
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    bert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    tfidf_max_features: int = 6000
    tfidf_min_df: int = 3
    tfidf_max_df: float = 0.85
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    neighbor_pool: int = 60
    random_state: int = 42
    svd_components: int = 100
    silhouette_sample_size: int = 2000
    k_min: int = 2
    k_max: int = 80
    lda_topic_count: int = 12


class JournalRecommendationPipeline:
    def __init__(self, config: RecommenderConfig | None = None) -> None:
        self.config = config or RecommenderConfig()
        self.paths = ProjectPaths.discover()
        self.cache_prefix = (
            f"{self.config.dataset_config.profile}"
            f"_{self.config.dataset_config.target_journals}"
            f"_{self.config.dataset_config.target_records}"
        )

        self.dataset: pd.DataFrame | None = None
        self.profile_metadata: dict[str, Any] | None = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.abstract_query_tfidf = None
        self.bert_model: SentenceTransformer | None = None
        self.bert_embeddings: np.ndarray | None = None
        self.abstract_query_embeddings: np.ndarray | None = None
        self.svd_model: TruncatedSVD | None = None
        self.reduced_matrix: np.ndarray | None = None
        self.cluster_diagnostics: pd.DataFrame | None = None
        self.final_kmeans: KMeans | None = None
        self.final_k: int | None = None
        self.cluster_interpretation: pd.DataFrame | None = None
        self.count_vectorizer: CountVectorizer | None = None
        self.lda_model: LatentDirichletAllocation | None = None
        self.topic_summary: pd.DataFrame | None = None
        self.evaluation_summary: pd.DataFrame | None = None
        self.error_analysis: pd.DataFrame | None = None
        self.journal_profiles: pd.DataFrame | None = None

    @property
    def feature_names(self) -> np.ndarray:
        if self.tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer is not fitted yet.")
        return self.tfidf_vectorizer.get_feature_names_out()

    def _cache_path(self, stem: str, suffix: str) -> Path:
        return self.paths.cache_dir / f"{self.cache_prefix}_{stem}{suffix}"

    def build(
        self,
        use_cache: bool = True,
        include_evaluation: bool = False,
        full_cluster_search: bool = False,
    ) -> "JournalRecommendationPipeline":
        self._load_or_build_dataset(use_cache=use_cache)
        self._load_or_build_tfidf(use_cache=use_cache)
        self._load_or_build_bert(use_cache=use_cache)
        self._load_or_build_clustering(use_cache=use_cache, full_search=full_cluster_search)
        self._load_or_build_topic_model(use_cache=use_cache)
        self._build_journal_profiles()
        if include_evaluation:
            self.evaluate_all_models(use_cache=use_cache)
        return self

    def _load_or_build_dataset(self, use_cache: bool = True) -> None:
        dataset_path = self._cache_path("dataset", ".pkl")
        metadata_path = self._cache_path("profile_metadata", ".json")
        if use_cache and dataset_path.exists() and metadata_path.exists():
            self.dataset = pd.read_pickle(dataset_path)
            self.profile_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return

        dataset, metadata = build_dataset(self.config.dataset_config)
        dataset.to_pickle(dataset_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.dataset = dataset
        self.profile_metadata = metadata

    def _load_or_build_tfidf(self, use_cache: bool = True) -> None:
        vectorizer_path = self._cache_path("tfidf_vectorizer", ".joblib")
        matrix_path = self._cache_path("tfidf_matrix", ".npz")
        abstract_matrix_path = self._cache_path("tfidf_query_matrix", ".npz")
        if use_cache and vectorizer_path.exists() and matrix_path.exists() and abstract_matrix_path.exists():
            self.tfidf_vectorizer = joblib.load(vectorizer_path)
            self.tfidf_matrix = load_npz(matrix_path)
            self.abstract_query_tfidf = load_npz(abstract_matrix_path)
            return

        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before TF-IDF fitting.")
        vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            min_df=self.config.tfidf_min_df,
            max_df=self.config.tfidf_max_df,
            ngram_range=self.config.tfidf_ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(self.dataset["combined_text"])
        abstract_query_tfidf = vectorizer.transform(self.dataset["abstract_clean"])
        joblib.dump(vectorizer, vectorizer_path)
        save_npz(matrix_path, tfidf_matrix)
        save_npz(abstract_matrix_path, abstract_query_tfidf)
        self.tfidf_vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.abstract_query_tfidf = abstract_query_tfidf

    def _load_or_build_bert(self, use_cache: bool = True) -> None:
        embeddings_path = self._cache_path("bert_embeddings", ".npy")
        abstract_embeddings_path = self._cache_path("bert_query_embeddings", ".npy")
        if use_cache and embeddings_path.exists() and abstract_embeddings_path.exists():
            self.bert_embeddings = np.load(embeddings_path)
            self.abstract_query_embeddings = np.load(abstract_embeddings_path)
            self.bert_model = self._load_sentence_transformer()
            return

        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before BERT fitting.")
        model = self._load_sentence_transformer()
        bert_embeddings = model.encode(
            self.dataset["combined_text"].tolist(),
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        abstract_embeddings = model.encode(
            self.dataset["abstract_clean"].tolist(),
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        np.save(embeddings_path, bert_embeddings)
        np.save(abstract_embeddings_path, abstract_embeddings)
        self.bert_model = model
        self.bert_embeddings = bert_embeddings
        self.abstract_query_embeddings = abstract_embeddings

    def _load_sentence_transformer(self) -> SentenceTransformer:
        try:
            return SentenceTransformer(self.config.bert_model_name, local_files_only=True)
        except Exception:
            return SentenceTransformer(self.config.bert_model_name)

    def _load_or_build_clustering(self, use_cache: bool = True, full_search: bool = False) -> None:
        diagnostics_path = self._cache_path("cluster_diagnostics", ".csv")
        final_kmeans_path = self._cache_path("final_kmeans", ".joblib")
        reduced_matrix_path = self._cache_path("reduced_matrix", ".npy")
        svd_path = self._cache_path("svd_model", ".joblib")
        interpretation_path = self._cache_path("cluster_interpretation", ".csv")
        if (
            use_cache
            and diagnostics_path.exists()
            and final_kmeans_path.exists()
            and reduced_matrix_path.exists()
            and svd_path.exists()
            and interpretation_path.exists()
        ):
            self.cluster_diagnostics = pd.read_csv(diagnostics_path)
            self.final_kmeans = joblib.load(final_kmeans_path)
            self.final_k = int(self.final_kmeans.n_clusters)
            self.reduced_matrix = np.load(reduced_matrix_path)
            self.svd_model = joblib.load(svd_path)
            self.cluster_interpretation = pd.read_csv(interpretation_path)
            if self.dataset is not None:
                self.dataset["cluster_id"] = self.final_kmeans.labels_
                cluster_label_map = dict(
                    zip(self.cluster_interpretation["cluster_id"], self.cluster_interpretation["topic_label"])
                )
                self.dataset["cluster_label"] = self.dataset["cluster_id"].map(cluster_label_map)
            return

        if self.dataset is None or self.tfidf_matrix is None:
            raise RuntimeError("Dataset and TF-IDF matrix must be available before clustering.")

        components = min(self.config.svd_components, self.tfidf_matrix.shape[1] - 1)
        svd_model = TruncatedSVD(n_components=max(2, components), random_state=self.config.random_state)
        reduced_matrix = svd_model.fit_transform(self.tfidf_matrix)

        if full_search:
            diagnostics_frame, final_k = self._run_full_cluster_search(reduced_matrix)
        else:
            final_k = self._default_fast_k()
            diagnostics_frame = pd.DataFrame(
                [
                    {
                        "k": final_k,
                        "inertia": np.nan,
                        "silhouette": np.nan,
                        "avg_cluster_size": float(len(reduced_matrix) / final_k),
                        "small_clusters": np.nan,
                        "small_cluster_ratio": np.nan,
                        "min_cluster_size": np.nan,
                        "max_cluster_size": np.nan,
                        "selection_score": np.nan,
                    }
                ]
            )

        final_kmeans = KMeans(n_clusters=final_k, random_state=self.config.random_state, n_init=20)
        final_labels = final_kmeans.fit_predict(self.tfidf_matrix)
        self.dataset["cluster_id"] = final_labels

        cluster_interpretation = self._build_cluster_interpretation_from_labels(final_kmeans, final_labels)
        cluster_label_map = dict(zip(cluster_interpretation["cluster_id"], cluster_interpretation["topic_label"]))
        self.dataset["cluster_label"] = self.dataset["cluster_id"].map(cluster_label_map)

        diagnostics_frame.to_csv(diagnostics_path, index=False)
        cluster_interpretation.to_csv(interpretation_path, index=False)
        joblib.dump(final_kmeans, final_kmeans_path)
        np.save(reduced_matrix_path, reduced_matrix)
        joblib.dump(svd_model, svd_path)

        self.cluster_diagnostics = diagnostics_frame
        self.final_kmeans = final_kmeans
        self.final_k = final_k
        self.reduced_matrix = reduced_matrix
        self.svd_model = svd_model
        self.cluster_interpretation = cluster_interpretation

    def _default_fast_k(self) -> int:
        if self.dataset is None:
            return 48
        estimated = round(len(self.dataset) / 160)
        return int(min(52, max(46, estimated)))

    def _run_full_cluster_search(self, reduced_matrix: np.ndarray) -> tuple[pd.DataFrame, int]:
        diagnostics: list[dict[str, Any]] = []
        sample_size = min(self.config.silhouette_sample_size, len(reduced_matrix))
        for k in range(self.config.k_min, self.config.k_max + 1):
            model = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
            labels = model.fit_predict(reduced_matrix)
            counts = np.bincount(labels)
            avg_size = len(reduced_matrix) / k
            small_clusters = int(np.sum(counts < max(5, avg_size * 0.25)))
            diagnostics.append(
                {
                    "k": k,
                    "inertia": float(model.inertia_),
                    "silhouette": float(
                        silhouette_score(
                            reduced_matrix,
                            labels,
                            sample_size=sample_size,
                            random_state=self.config.random_state,
                        )
                    ),
                    "avg_cluster_size": float(avg_size),
                    "small_clusters": small_clusters,
                    "small_cluster_ratio": float(small_clusters / k),
                    "min_cluster_size": int(counts.min()),
                    "max_cluster_size": int(counts.max()),
                }
            )

        diagnostics_frame = pd.DataFrame(diagnostics)
        diagnostics_frame["inertia_scaled"] = 1 - _safe_min_max(diagnostics_frame["inertia"].to_numpy())
        diagnostics_frame["silhouette_scaled"] = _safe_min_max(diagnostics_frame["silhouette"].to_numpy())
        diagnostics_frame["balance_score"] = 1 - diagnostics_frame["small_cluster_ratio"]
        diagnostics_frame["interpretability_score"] = 1 - (
            (diagnostics_frame["avg_cluster_size"] - 150).abs() / 150
        ).clip(lower=0, upper=1)
        curvature = np.zeros(len(diagnostics_frame))
        inertia_scaled = diagnostics_frame["inertia_scaled"].to_numpy()
        for index in range(1, len(diagnostics_frame) - 1):
            curvature[index] = abs(
                inertia_scaled[index - 1] - 2 * inertia_scaled[index] + inertia_scaled[index + 1]
            )
        diagnostics_frame["elbow_curvature"] = curvature
        diagnostics_frame["elbow_scaled"] = _safe_min_max(curvature)
        diagnostics_frame["selection_score"] = (
            0.4 * diagnostics_frame["silhouette_scaled"]
            + 0.25 * diagnostics_frame["elbow_scaled"]
            + 0.2 * diagnostics_frame["balance_score"]
            + 0.15 * diagnostics_frame["interpretability_score"]
        )
        candidate_frame = diagnostics_frame[diagnostics_frame["small_cluster_ratio"] <= 0.2].copy()
        if candidate_frame.empty:
            candidate_frame = diagnostics_frame.copy()
        final_k = int(
            candidate_frame.sort_values(
                ["selection_score", "silhouette", "elbow_curvature", "k"],
                ascending=[False, False, False, True],
            ).iloc[0]["k"]
        )
        return diagnostics_frame, final_k

    def run_full_cluster_analysis(self, use_cache: bool = True) -> pd.DataFrame:
        self._load_or_build_dataset(use_cache=use_cache)
        self._load_or_build_tfidf(use_cache=use_cache)
        self._load_or_build_clustering(use_cache=False, full_search=True)
        self._build_journal_profiles()
        if self.cluster_diagnostics is None:
            raise RuntimeError("Cluster diagnostics were not generated.")
        return self.cluster_diagnostics

    def _build_cluster_interpretation_from_labels(
        self,
        kmeans_model: KMeans,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        if self.dataset is None:
            raise RuntimeError("Dataset is required to build cluster interpretation.")
        feature_names = self.feature_names
        interpretation_records: list[dict[str, Any]] = []
        dataset = self.dataset.copy()
        dataset["cluster_id"] = labels
        for cluster_id in range(kmeans_model.n_clusters):
            cluster_rows = dataset[dataset["cluster_id"] == cluster_id]
            centroid = kmeans_model.cluster_centers_[cluster_id]
            top_words = _top_words_from_weights(centroid, feature_names, top_n=8)
            common_journals = (
                cluster_rows["journal_name"].value_counts().head(5).index.to_list()
                if not cluster_rows.empty
                else []
            )
            label = " / ".join(word.replace("_", " ") for word in top_words[:3]).title()
            interpretation_records.append(
                {
                    "cluster_id": cluster_id,
                    "topic_label": label or f"Cluster {cluster_id}",
                    "top_words": ", ".join(top_words),
                    "common_journals": ", ".join(common_journals),
                    "size": int(len(cluster_rows)),
                }
            )
        return pd.DataFrame(interpretation_records).sort_values("cluster_id").reset_index(drop=True)

    def _load_or_build_topic_model(self, use_cache: bool = True) -> None:
        vectorizer_path = self._cache_path("count_vectorizer", ".joblib")
        lda_path = self._cache_path("lda_model", ".joblib")
        topic_summary_path = self._cache_path("topic_summary", ".csv")
        if use_cache and vectorizer_path.exists() and lda_path.exists() and topic_summary_path.exists():
            self.count_vectorizer = joblib.load(vectorizer_path)
            self.lda_model = joblib.load(lda_path)
            self.topic_summary = pd.read_csv(topic_summary_path)
            return

        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before topic modeling.")
        vectorizer = CountVectorizer(
            max_features=4000,
            min_df=3,
            max_df=0.9,
            stop_words="english",
        )
        count_matrix = vectorizer.fit_transform(self.dataset["combined_text"])
        topic_count = min(max(6, self.config.lda_topic_count), max(6, len(self.dataset) // 300))
        lda = LatentDirichletAllocation(
            n_components=topic_count,
            learning_method="batch",
            random_state=self.config.random_state,
        )
        lda.fit(count_matrix)
        feature_names = vectorizer.get_feature_names_out()
        topic_records = []
        for topic_index, weights in enumerate(lda.components_):
            top_words = _top_words_from_weights(weights, feature_names, top_n=10)
            topic_records.append(
                {
                    "topic_id": topic_index,
                    "topic_label": " / ".join(top_words[:3]).title(),
                    "top_words": ", ".join(top_words),
                }
            )
        topic_summary = pd.DataFrame(topic_records)
        topic_summary.to_csv(topic_summary_path, index=False)
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(lda, lda_path)
        self.count_vectorizer = vectorizer
        self.lda_model = lda
        self.topic_summary = topic_summary

    def _build_journal_profiles(self) -> None:
        if self.dataset is None or self.cluster_interpretation is None:
            raise RuntimeError("Dataset and clustering must be built before journal profiling.")
        cluster_label_map = dict(zip(self.cluster_interpretation["cluster_id"], self.cluster_interpretation["topic_label"]))

        def union_lists(values: pd.Series) -> list[str]:
            items: list[str] = []
            seen: set[str] = set()
            for value in values:
                for item in value:
                    if item not in seen:
                        items.append(item)
                        seen.add(item)
            return items

        profiles = (
            self.dataset.groupby("journal_name")
            .agg(
                article_count=("article_id", "count"),
                keywords=("keyword_terms", union_lists),
                subjects=("subject_terms", union_lists),
                keyword_plus=("keyword_plus_terms", union_lists),
                dominant_cluster=("cluster_id", lambda values: Counter(values).most_common(1)[0][0]),
            )
            .reset_index()
        )
        profiles["cluster_label"] = profiles["dominant_cluster"].map(cluster_label_map)
        self.journal_profiles = profiles

    def _normalize_query(self, abstract_text: str) -> str:
        if self.dataset is None:
            raise RuntimeError("Build the pipeline before querying it.")
        return normalize_text(abstract_text)

    def _tfidf_scores(self, abstract_text: str) -> np.ndarray:
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF model is not ready.")
        cleaned = self._normalize_query(abstract_text)
        query_vector = self.tfidf_vectorizer.transform([cleaned])
        return linear_kernel(query_vector, self.tfidf_matrix).ravel()

    def _bert_scores(self, abstract_text: str) -> np.ndarray:
        if self.bert_model is None or self.bert_embeddings is None:
            raise RuntimeError("BERT model is not ready.")
        cleaned = self._normalize_query(abstract_text)
        query_embedding = self.bert_model.encode(
            [cleaned],
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        return np.dot(self.bert_embeddings, query_embedding)

    def _journal_rankings_from_article_scores(
        self,
        abstract_text: str,
        article_scores: np.ndarray,
        score_label: str,
        top_n: int = 5,
        exclude_article_id: int | None = None,
    ) -> pd.DataFrame:
        if self.dataset is None:
            raise RuntimeError("Dataset is not loaded.")
        query_text = self._normalize_query(abstract_text)
        candidate_frame = self.dataset.copy()
        candidate_frame[score_label] = article_scores
        if exclude_article_id is not None:
            candidate_frame = candidate_frame[candidate_frame["article_id"] != exclude_article_id]
        candidate_frame = candidate_frame.sort_values(score_label, ascending=False).head(self.config.neighbor_pool).copy()

        if score_label in {"bert_similarity", "tfidf_similarity"}:
            candidate_frame[f"{score_label}_normalized"] = _safe_min_max(candidate_frame[score_label].to_numpy())
        else:
            candidate_frame[f"{score_label}_normalized"] = candidate_frame[score_label].to_numpy()

        ranking_records: list[dict[str, Any]] = []
        for journal_name, group in candidate_frame.groupby("journal_name"):
            top_articles = group.sort_values(score_label, ascending=False).head(3).copy()
            keyword_matches = sorted(
                {
                    term
                    for terms in top_articles["keyword_terms"]
                    for term in _match_terms(query_text, terms)
                }
            )
            subject_matches = sorted(
                {
                    term
                    for terms in top_articles["subject_terms"]
                    for term in _match_terms(query_text, terms)
                }
            )
            cluster_label = top_articles["cluster_label"].mode().iloc[0]
            score = 0.7 * float(top_articles[score_label].iloc[0]) + 0.3 * float(top_articles[score_label].mean())
            ranking_records.append(
                {
                    "journal_name": journal_name,
                    "score": score,
                    "supporting_articles": [
                        {
                            "article_id": int(row.article_id),
                            "title": row.title,
                            "year": int(row.pub_year) if pd.notna(row.pub_year) else None,
                            "score": round(float(getattr(row, score_label)), 4),
                        }
                        for row in top_articles.itertuples()
                    ],
                    "overlapping_keywords": keyword_matches[:8],
                    "overlapping_subjects": subject_matches[:8],
                    "cluster_label": cluster_label,
                }
            )
        ranking = pd.DataFrame(ranking_records).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        if ranking.empty:
            return ranking
        gaps = ranking["score"] - ranking["score"].shift(-1).fillna(0)
        confidences = []
        for score, gap in zip(ranking["score"], gaps):
            if score >= 0.65 and gap >= 0.08:
                label = "High"
            elif score >= 0.5 or gap >= 0.04:
                label = "Medium"
            else:
                label = "Low"
            confidences.append({"confidence_label": label, "confidence_score": round(float(score), 4)})
        ranking = pd.concat([ranking, pd.DataFrame(confidences)], axis=1)
        ranking["explanation"] = ranking.apply(
            lambda row: self._build_explanation(
                row["overlapping_keywords"],
                row["overlapping_subjects"],
                row["cluster_label"],
            ),
            axis=1,
        )
        ranking["score_percent"] = ranking["score"].map(_as_percent)
        return ranking

    @staticmethod
    def _build_explanation(
        overlapping_keywords: list[str],
        overlapping_subjects: list[str],
        cluster_label: str,
    ) -> str:
        keyword_text = ", ".join(overlapping_keywords[:4]) if overlapping_keywords else "content-level semantic similarity"
        subject_text = ", ".join(overlapping_subjects[:3]) if overlapping_subjects else cluster_label
        return (
            "Recommended because the abstract aligns with articles containing "
            f"{keyword_text}, and the strongest supporting evidence falls into {subject_text}."
        )

    def recommend_journals_tfidf(self, input_abstract: str, top_n: int = 5) -> pd.DataFrame:
        scores = self._tfidf_scores(input_abstract)
        return self._journal_rankings_from_article_scores(input_abstract, scores, "tfidf_similarity", top_n=top_n)

    def recommend_journals_bert(self, input_abstract: str, top_n: int = 5) -> pd.DataFrame:
        scores = self._bert_scores(input_abstract)
        return self._journal_rankings_from_article_scores(input_abstract, scores, "bert_similarity", top_n=top_n)

    def recommend_journals_hybrid(self, input_abstract: str, top_n: int = 5) -> pd.DataFrame:
        if self.dataset is None:
            raise RuntimeError("Dataset is not loaded.")
        query_text = self._normalize_query(input_abstract)
        base_scores = self._bert_scores(input_abstract)
        candidate_frame = self.dataset.copy()
        candidate_frame["bert_similarity"] = base_scores
        candidate_frame = candidate_frame.sort_values("bert_similarity", ascending=False).head(self.config.neighbor_pool).copy()
        candidate_frame["similarity"] = _safe_min_max(candidate_frame["bert_similarity"].to_numpy())
        candidate_frame["keyword_overlap"] = candidate_frame["keyword_terms"].map(
            lambda terms: _jaccard_overlap(_match_terms(query_text, terms), terms)
        )
        candidate_frame["subject_overlap"] = candidate_frame["subject_terms"].map(
            lambda terms: _jaccard_overlap(_match_terms(query_text, terms), terms)
        )
        candidate_frame["hybrid_score"] = (
            0.6 * candidate_frame["similarity"]
            + 0.2 * candidate_frame["keyword_overlap"]
            + 0.2 * candidate_frame["subject_overlap"]
        )
        article_scores = np.zeros(len(self.dataset))
        article_scores[candidate_frame.index.to_numpy()] = candidate_frame["hybrid_score"].to_numpy()
        return self._journal_rankings_from_article_scores(input_abstract, article_scores, "hybrid_score", top_n=top_n)

    def recommend_journals_ensemble(self, input_abstract: str, top_n: int = 5) -> pd.DataFrame:
        tfidf_scores = self._tfidf_scores(input_abstract)
        bert_scores = self._bert_scores(input_abstract)
        ensemble_scores = 0.5 * _safe_min_max(tfidf_scores) + 0.5 * _safe_min_max(bert_scores)
        return self._journal_rankings_from_article_scores(input_abstract, ensemble_scores, "ensemble_score", top_n=top_n)

    def dataset_summary(self) -> dict[str, Any]:
        if self.dataset is None:
            raise RuntimeError("Dataset is not loaded.")
        summary = {
            "records": int(len(self.dataset)),
            "journals": int(self.dataset["journal_name"].nunique()),
            "year_range": (
                int(self.dataset["pub_year"].min()),
                int(self.dataset["pub_year"].max()),
            ),
            "avg_articles_per_journal": round(float(len(self.dataset) / self.dataset["journal_name"].nunique()), 2),
            "top_journals": self.dataset["journal_name"].value_counts().head(10).to_dict(),
        }
        return summary

    def evaluate_all_models(self, use_cache: bool = True) -> pd.DataFrame:
        summary_path = self._cache_path("evaluation_summary", ".csv")
        error_path = self._cache_path("error_analysis", ".csv")
        if use_cache and summary_path.exists() and error_path.exists():
            self.evaluation_summary = pd.read_csv(summary_path)
            self.error_analysis = pd.read_csv(error_path)
            return self.evaluation_summary

        model_results = []
        error_frames = []
        for model_name in ("tfidf", "bert", "hybrid", "ensemble"):
            detail_frame = self._evaluate_single_model(model_name)
            model_results.append(
                {
                    "model": model_name.upper(),
                    "top_1_accuracy": round(detail_frame["hit_at_1"].mean(), 4),
                    "top_3_hit_rate": round(detail_frame["hit_at_3"].mean(), 4),
                    "top_5_hit_rate": round(detail_frame["hit_at_5"].mean(), 4),
                }
            )
            failures = detail_frame[detail_frame["hit_at_1"] == 0].copy()
            failures["model"] = model_name.upper()
            error_frames.append(failures.head(15))
        summary = pd.DataFrame(model_results)
        error_analysis = pd.concat(error_frames, ignore_index=True)
        summary.to_csv(summary_path, index=False)
        error_analysis.to_csv(error_path, index=False)
        self.evaluation_summary = summary
        self.error_analysis = error_analysis
        return summary

    def _evaluate_single_model(self, model_name: str) -> pd.DataFrame:
        if self.dataset is None or self.abstract_query_tfidf is None or self.abstract_query_embeddings is None:
            raise RuntimeError("Build the pipeline before evaluation.")
        tfidf_nn = NearestNeighbors(metric="cosine", algorithm="brute")
        tfidf_nn.fit(self.tfidf_matrix)
        tfidf_distances, tfidf_indices = tfidf_nn.kneighbors(
            self.abstract_query_tfidf,
            n_neighbors=self.config.neighbor_pool + 1,
            return_distance=True,
        )

        bert_nn = NearestNeighbors(metric="cosine", algorithm="brute")
        bert_nn.fit(self.bert_embeddings)
        bert_distances, bert_indices = bert_nn.kneighbors(
            self.abstract_query_embeddings,
            n_neighbors=self.config.neighbor_pool + 1,
            return_distance=True,
        )

        detail_records: list[dict[str, Any]] = []
        query_abstracts = self.dataset["abstract"].tolist()
        actual_journals = self.dataset["journal_name"].tolist()
        article_ids = self.dataset["article_id"].tolist()

        for index, (article_id, actual_journal, abstract) in enumerate(zip(article_ids, actual_journals, query_abstracts)):
            tfidf_candidates = [
                (candidate_idx, 1 - float(distance))
                for candidate_idx, distance in zip(tfidf_indices[index], tfidf_distances[index])
                if candidate_idx != index
            ]
            bert_candidates = [
                (candidate_idx, 1 - float(distance))
                for candidate_idx, distance in zip(bert_indices[index], bert_distances[index])
                if candidate_idx != index
            ]
            candidate_indices = sorted({idx for idx, _ in tfidf_candidates} | {idx for idx, _ in bert_candidates})
            candidate_frame = self.dataset.iloc[candidate_indices].copy()
            candidate_frame["tfidf_similarity"] = (
                candidate_frame.index.to_series().map(dict(tfidf_candidates)).fillna(0.0).to_numpy()
            )
            candidate_frame["bert_similarity"] = (
                candidate_frame.index.to_series().map(dict(bert_candidates)).fillna(0.0).to_numpy()
            )

            if model_name == "tfidf":
                score_column = "tfidf_similarity"
                candidate_frame[score_column] = _safe_min_max(candidate_frame[score_column].to_numpy())
            elif model_name == "bert":
                score_column = "bert_similarity"
                candidate_frame[score_column] = _safe_min_max(candidate_frame[score_column].to_numpy())
            elif model_name == "hybrid":
                query_text = self._normalize_query(abstract)
                candidate_frame["similarity"] = _safe_min_max(candidate_frame["bert_similarity"].to_numpy())
                candidate_frame["keyword_overlap"] = candidate_frame["keyword_terms"].map(
                    lambda terms: _jaccard_overlap(_match_terms(query_text, terms), terms)
                )
                candidate_frame["subject_overlap"] = candidate_frame["subject_terms"].map(
                    lambda terms: _jaccard_overlap(_match_terms(query_text, terms), terms)
                )
                score_column = "hybrid_score"
                candidate_frame[score_column] = (
                    0.6 * candidate_frame["similarity"]
                    + 0.2 * candidate_frame["keyword_overlap"]
                    + 0.2 * candidate_frame["subject_overlap"]
                )
            elif model_name == "ensemble":
                score_column = "ensemble_score"
                candidate_frame[score_column] = (
                    0.5 * _safe_min_max(candidate_frame["tfidf_similarity"].to_numpy())
                    + 0.5 * _safe_min_max(candidate_frame["bert_similarity"].to_numpy())
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            grouped = (
                candidate_frame.sort_values(score_column, ascending=False)
                .groupby("journal_name")[score_column]
                .max()
                .sort_values(ascending=False)
            )
            predictions = grouped.head(5).index.tolist()
            detail_records.append(
                {
                    "article_id": article_id,
                    "actual_journal": actual_journal,
                    "predicted_top_1": predictions[0] if predictions else None,
                    "predicted_top_5": " | ".join(predictions),
                    "hit_at_1": int(actual_journal in predictions[:1]),
                    "hit_at_3": int(actual_journal in predictions[:3]),
                    "hit_at_5": int(actual_journal in predictions[:5]),
                }
            )
        return pd.DataFrame(detail_records)

    def journal_insights(self) -> dict[str, pd.DataFrame]:
        if self.dataset is None:
            raise RuntimeError("Dataset is not loaded.")
        journal_distribution = (
            self.dataset["journal_name"].value_counts().rename_axis("journal_name").reset_index(name="article_count")
        )
        keyword_distribution = (
            pd.Series([term for terms in self.dataset["keyword_terms"] for term in terms])
            .value_counts()
            .rename_axis("keyword")
            .reset_index(name="frequency")
        )
        subject_distribution = (
            pd.Series([term for terms in self.dataset["subject_terms"] for term in terms])
            .value_counts()
            .rename_axis("subject")
            .reset_index(name="frequency")
        )
        cluster_journal = (
            self.dataset.groupby(["cluster_label", "journal_name"])["article_id"]
            .count()
            .rename("article_count")
            .reset_index()
            .sort_values(["cluster_label", "article_count"], ascending=[True, False])
        )
        return {
            "journal_distribution": journal_distribution,
            "keyword_distribution": keyword_distribution,
            "subject_distribution": subject_distribution,
            "cluster_journal_relationships": cluster_journal,
        }

    def projection_frame(self, sample_size: int = 2500) -> pd.DataFrame:
        if self.dataset is None or self.reduced_matrix is None:
            raise RuntimeError("Clustering must be built before creating the projection frame.")
        sample = self.dataset.copy()
        sample_indices = sample.index.to_numpy()
        if len(sample) > sample_size:
            rng = np.random.default_rng(self.config.random_state)
            sample_indices = np.sort(rng.choice(sample_indices, size=sample_size, replace=False))
            sample = sample.loc[sample_indices].copy()
        pca = PCA(n_components=2, random_state=self.config.random_state)
        coords = pca.fit_transform(self.reduced_matrix[sample_indices])
        sample["x"] = coords[:, 0]
        sample["y"] = coords[:, 1]
        return sample[["article_id", "journal_name", "cluster_label", "x", "y"]].reset_index(drop=True)
